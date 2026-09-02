"""Blind analyst predictions — immutable, versioned, one row per horizon.

The V2.1 flow: BEFORE the debate, each analyst commits direction /
probability / expected return for every horizon on the frozen ladder via the
``submit_prediction`` tool; those rows are written here and never changed.
After the debate, corrections are NEW rows linked to the row they revise
(``original_prediction_id`` + 1-based ``debate_revision``). Accuracy
attribution then scores exactly what the analyst said in advance — never a
post-hoc edit of it.

Immutability contract (enforced by check-then-insert inside ONE
``BEGIN IMMEDIATE`` transaction, so the check and the insert cannot
interleave):

* a row already exists with this ``prediction_id`` and IDENTICAL content
  (every stored column except ``created_at``) — idempotent no-op, the id is
  returned again;
* a row exists with DIFFERENT content — ``ValueError``; the caller must
  revise instead. There is no UPDATE path anywhere.

Identity: ``prediction_id`` is deterministic from
``(run_id, analyst, prediction_scope, horizon_days)`` — one run analyses one
instrument, so that tuple names exactly one blind call per horizon; a second
submission of the same identity with different content is an immutability
violation, not a second prediction. ``evidence_ids`` is stored as a sorted
JSON array (set semantics), so resubmitting the same evidence chain in a
different order is still identical content.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

from yialpha.ledger.models import (
    AnalystPrediction,
    PredictionEntry,
    coerce_prediction_entry,
    encode_json_list,
    new_prediction_id,
    row_to_prediction,
    validate_scope,
)
from yialpha.ledger.sqlite import (
    get_connection,
    ledger_exists,
    ledger_transaction,
    utc_now_iso,
)
from yialpha.versions import FEATURE_VERSION, SCHEMA_VERSION

#: Every ``predictions`` column, in INSERT/SELECT order (v1 DDL order + the
#: migration-v2 ``regime_id`` tail column).
_PREDICTION_COLUMNS = (
    "prediction_id",
    "run_id",
    "analyst",
    "instrument_id",
    "prediction_scope",
    "horizon_days",
    "direction",
    "prob_up",
    "expected_return",
    "target_price",
    "target_currency",
    "price_basis",
    "confidence",
    "evidence_ids",
    "analysis_as_of",
    "created_at",
    "schema_version",
    "feature_version",
    "original_prediction_id",
    "debate_revision",
    "revision_reason",
    "regime_id",
)

#: Columns that define content identity — everything except ``created_at``.
_PREDICTION_CONTENT_COLUMNS = tuple(
    column for column in _PREDICTION_COLUMNS if column != "created_at"
)


def _validated_entries(
    entries: Sequence[Mapping[str, Any] | PredictionEntry],
) -> list[PredictionEntry]:
    """Coerce and validate every entry up front (nothing is written on failure)."""
    coerced = [coerce_prediction_entry(entry) for entry in entries]
    seen: set[int] = set()
    for entry in coerced:
        if entry.horizon_days in seen:
            raise ValueError(
                f"duplicate horizon_days {entry.horizon_days} in one submission; "
                "submit exactly one entry per horizon"
            )
        seen.add(entry.horizon_days)
    return coerced


def _insert_entries(
    cur: sqlite3.Cursor,
    *,
    run_id: str,
    analyst: str,
    instrument_id: str,
    prediction_scope: str,
    entries: Sequence[PredictionEntry],
    analysis_as_of: str,
    evidence_ids_json: str,
    original_prediction_id: str | None,
    debate_revision: int | None,
    revision_reason: str | None,
    regime_id: str | None = None,
) -> list[str]:
    """Check-then-insert a batch of rows on an open transaction cursor.

    Runs inside the caller's ``BEGIN IMMEDIATE`` transaction, so the
    existence check and the insert are atomic against other writers and the
    table's UNIQUE primary key can never be hit by this code path.
    """
    prediction_ids: list[str] = []
    for entry in entries:
        revision = debate_revision or 0
        prediction_id = new_prediction_id(
            run_id, analyst, prediction_scope, entry.horizon_days, revision=revision
        )
        values: dict[str, Any] = {
            "prediction_id": prediction_id,
            "run_id": run_id,
            "analyst": analyst,
            "instrument_id": instrument_id,
            "prediction_scope": prediction_scope,
            "horizon_days": entry.horizon_days,
            "direction": entry.direction,
            "prob_up": entry.prob_up,
            "expected_return": entry.expected_return,
            "target_price": entry.target_price,
            "target_currency": entry.target_currency,
            "price_basis": entry.price_basis,
            "confidence": entry.confidence,
            "evidence_ids": evidence_ids_json,
            "analysis_as_of": analysis_as_of,
            "created_at": utc_now_iso(),
            "schema_version": SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "original_prediction_id": original_prediction_id,
            "debate_revision": debate_revision,
            "revision_reason": revision_reason,
            "regime_id": regime_id,
        }
        existing = cur.execute(
            "SELECT prediction_id, run_id, analyst, instrument_id, prediction_scope, "
            "horizon_days, direction, prob_up, expected_return, target_price, "
            "target_currency, price_basis, confidence, evidence_ids, analysis_as_of, "
            "created_at, schema_version, feature_version, original_prediction_id, "
            "debate_revision, revision_reason, regime_id FROM predictions "
            "WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        if existing is not None:
            if all(
                existing[column] == values[column]
                for column in _PREDICTION_CONTENT_COLUMNS
            ):
                prediction_ids.append(prediction_id)  # byte-identical replay
                continue
            raise ValueError(
                f"prediction {prediction_id} is immutable; submit a revision "
                "referencing original_prediction_id"
            )
        cur.execute(
            "INSERT INTO predictions "
            "(prediction_id, run_id, analyst, instrument_id, prediction_scope, "
            "horizon_days, direction, prob_up, expected_return, target_price, "
            "target_currency, price_basis, confidence, evidence_ids, "
            "analysis_as_of, created_at, schema_version, feature_version, "
            "original_prediction_id, debate_revision, revision_reason, regime_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(values[column] for column in _PREDICTION_COLUMNS),
        )
        prediction_ids.append(prediction_id)
    return prediction_ids


def submit_predictions(
    run_id: str,
    analyst: str,
    instrument_id: str,
    prediction_scope: str,
    entries: Sequence[Mapping[str, Any] | PredictionEntry],
    analysis_as_of: str,
    evidence_ids: Sequence[str] | None = None,
    regime_id: str | None = None,
) -> list[str]:
    """Write one analyst's blind predictions for the run; returns the new ids.

    Every entry is validated BEFORE any write (horizon on the ladder,
    direction enum, ``prob_up``/``confidence`` in [0, 1]); all rows are then
    inserted in ONE :func:`ledger_transaction` so a crash can never strand
    half a submission. Rows are stamped with the current ``SCHEMA_VERSION``
    and ``FEATURE_VERSION``. ``evidence_ids`` (optional) is stored as a
    sorted JSON array. ``regime_id`` (V2.2, optional) names the run's
    ``RegimeState`` row; it is part of content identity, so resubmitting
    the same prediction id under a DIFFERENT regime is an immutability
    conflict — a prediction row names exactly the context it was decided
    under. Raises ``sqlite3.IntegrityError`` for an unknown ``run_id``
    (foreign key) and ``ValueError`` on any invalid entry or conflicting
    resubmission of an existing prediction id.
    """
    validate_scope(prediction_scope)
    validated = _validated_entries(entries)
    with ledger_transaction() as cur:
        return _insert_entries(
            cur,
            run_id=run_id,
            analyst=analyst,
            instrument_id=instrument_id,
            prediction_scope=prediction_scope,
            entries=validated,
            analysis_as_of=analysis_as_of,
            evidence_ids_json=encode_json_list(evidence_ids),
            original_prediction_id=None,
            debate_revision=None,
            revision_reason=None,
            regime_id=regime_id,
        )


def revise_prediction(
    original_prediction_id: str,
    entries: Sequence[Mapping[str, Any] | PredictionEntry],
    revision_reason: str,
    analysis_as_of: str,
) -> list[str]:
    """Append a post-debate revision chain link; returns the new row ids.

    The original must exist (``ValueError`` otherwise). The revision reuses
    the original's run, analyst, instrument, scope and evidence chain, and is
    linked to the CHAIN ROOT — revising a revision keeps numbering and
    linkage anchored at the original (root = the referenced row's
    ``original_prediction_id``, falling back to itself). ``debate_revision``
    is ``1 + max(existing revisions of the root)``, computed inside the same
    transaction as the inserts so concurrent revisers serialise instead of
    racing the primary key.
    """
    validated = _validated_entries(entries)
    with ledger_transaction() as cur:
        original = cur.execute(
            "SELECT prediction_id, run_id, analyst, instrument_id, prediction_scope, "
            "horizon_days, direction, prob_up, expected_return, target_price, "
            "target_currency, price_basis, confidence, evidence_ids, analysis_as_of, "
            "created_at, schema_version, feature_version, original_prediction_id, "
            "debate_revision, revision_reason, regime_id FROM predictions "
            "WHERE prediction_id = ?",
            (original_prediction_id,),
        ).fetchone()
        if original is None:
            raise ValueError(
                f"original prediction {original_prediction_id} not found in ledger"
            )
        root_id = original["original_prediction_id"] or original["prediction_id"]
        max_row = cur.execute(
            "SELECT MAX(debate_revision) FROM predictions "
            "WHERE original_prediction_id = ?",
            (root_id,),
        ).fetchone()
        current_max = max_row[0] if max_row is not None else None
        next_revision = 1 + (current_max if current_max is not None else 0)
        return _insert_entries(
            cur,
            run_id=str(original["run_id"]),
            analyst=str(original["analyst"]),
            instrument_id=str(original["instrument_id"]),
            prediction_scope=str(original["prediction_scope"]),
            entries=validated,
            analysis_as_of=analysis_as_of,
            evidence_ids_json=str(original["evidence_ids"]),
            original_prediction_id=str(root_id),
            debate_revision=next_revision,
            revision_reason=revision_reason,
            regime_id=original["regime_id"],
        )


def predictions_for_run(run_id: str) -> list[AnalystPrediction]:
    """Every prediction row for ``run_id`` (originals and revisions, horizon order).

    Ordered by ``horizon_days`` then ``debate_revision`` (NULLs first, i.e.
    the original before its revisions) — the order a scorer walks a chain.
    Returns an empty list when no ledger DB exists yet (optional read).
    """
    if not ledger_exists():
        return []
    rows = (
        get_connection(readonly=True)
        .execute(
            "SELECT prediction_id, run_id, analyst, instrument_id, prediction_scope, "
            "horizon_days, direction, prob_up, expected_return, target_price, "
            "target_currency, price_basis, confidence, evidence_ids, analysis_as_of, "
            "created_at, schema_version, feature_version, original_prediction_id, "
            "debate_revision, revision_reason, regime_id FROM predictions "
            "WHERE run_id = ? "
            "ORDER BY horizon_days, debate_revision, prediction_id",
            (run_id,),
        )
        .fetchall()
    )
    return [row_to_prediction(row) for row in rows]


def prediction_by_id(prediction_id: str) -> AnalystPrediction | None:
    """One prediction row by id, or ``None`` when absent."""
    if not ledger_exists():
        return None
    row = (
        get_connection(readonly=True)
        .execute(
            "SELECT prediction_id, run_id, analyst, instrument_id, prediction_scope, "
            "horizon_days, direction, prob_up, expected_return, target_price, "
            "target_currency, price_basis, confidence, evidence_ids, analysis_as_of, "
            "created_at, schema_version, feature_version, original_prediction_id, "
            "debate_revision, revision_reason, regime_id FROM predictions "
            "WHERE prediction_id = ?",
            (prediction_id,),
        )
        .fetchone()
    )
    return row_to_prediction(row) if row is not None else None


def revisions_of(original_prediction_id: str) -> list[AnalystPrediction]:
    """All revision rows linked to a chain root, in ``debate_revision`` order."""
    if not ledger_exists():
        return []
    rows = (
        get_connection(readonly=True)
        .execute(
            "SELECT prediction_id, run_id, analyst, instrument_id, prediction_scope, "
            "horizon_days, direction, prob_up, expected_return, target_price, "
            "target_currency, price_basis, confidence, evidence_ids, analysis_as_of, "
            "created_at, schema_version, feature_version, original_prediction_id, "
            "debate_revision, revision_reason, regime_id FROM predictions "
            "WHERE original_prediction_id = ? "
            "ORDER BY debate_revision, horizon_days",
            (original_prediction_id,),
        )
        .fetchall()
    )
    return [row_to_prediction(row) for row in rows]
