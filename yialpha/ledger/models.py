"""Record models, deterministic IDs, and validators for the V2.1 ledgers.

One module owns the vocabulary every row-level ledger module shares:

* **Enum vocabulary** — scope / replayability / direction are plain ``str``
  constants so rows stay SQLite-portable and human-greppable; validators turn
  violations into precise ``ValueError`` messages (LLM tool output surfaces
  them verbatim, so "invalid" is not good enough).
* **Deterministic IDs** — ``E``/``P``/``O`` plus the first 12 hex chars of a
  SHA-256 over a documented canonical preimage string. Determinism IS the
  dedupe mechanism: re-injecting the same evidence block into the same run
  recomputes the same ``evidence_id`` so ``INSERT OR IGNORE`` collapses to one
  row, and re-submitting an identical prediction is a provable no-op. The ids
  are stable across calls and processes (no clocks, no counters).
* **Frozen dataclasses** — one per ledger row plus :class:`PredictionEntry`
  (one horizon of a blind prediction), with ``row_to_*`` converters from
  :class:`sqlite3.Row`.
* **Timestamp normalization** — the V2 time discipline admits both pure dates
  (``YYYY-MM-DD``) and full instants (``YYYY-MM-DDTHH:MM:SS+00:00``); the
  helpers here define the ONE canonical interpretation of each so every
  ledger module answers "is X available by Y?" identically.

Canonical ID preimages (components joined with ``|``; internal identifiers
— run/analyst/scope names — never contain it):

* evidence: ``evidence|{run_id}|{payload_hash}``
* prediction:
  ``prediction|{run_id}|{analyst}|{prediction_scope}|{horizon_days}|r{revision}``
  — ``r0`` for the original row;
  :func:`yialpha.ledger.predictions.revise_prediction` passes the chain's next
  ``debate_revision`` so every revision is a distinct deterministic row.
* outcome: ``outcome|{prediction_id}|{horizon_days}``

``HORIZON_LADDER_DAYS`` mirrors the FEATURE_VERSION v2 contract in
:mod:`yialpha.versions`. :mod:`yialpha.accuracy` defines no ladder of its own
(its scorers take a single configurable ``holding_days``, default 5 — inside
this ladder), so the frozen plan value ``(1, 5, 21)`` stands.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

# --------------------------------------------------------------------------- #
# Enum vocabulary (stored as TEXT columns; plain strings stay greppable)
# --------------------------------------------------------------------------- #
SCOPE_UNDERLYING = "UNDERLYING"
SCOPE_CONTRACT = "CONTRACT"
SCOPE_MACRO = "MACRO"
#: Every admissible ``scope`` / ``prediction_scope`` value.
SCOPES = frozenset({SCOPE_UNDERLYING, SCOPE_CONTRACT, SCOPE_MACRO})

REPLAYABILITY_PIT_REPLAYABLE = "PIT_REPLAYABLE"
REPLAYABILITY_LIVE_ONLY = "LIVE_ONLY"
#: Every admissible ``replayability`` tag.
REPLAYABILITY_TAGS = frozenset({REPLAYABILITY_PIT_REPLAYABLE, REPLAYABILITY_LIVE_ONLY})

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_FLAT = "flat"
#: Every admissible ``direction`` value.
DIRECTIONS = frozenset({DIRECTION_UP, DIRECTION_DOWN, DIRECTION_FLAT})

#: Frozen forecast horizon ladder — days on the instrument's OWN session
#: calendar (V2.1 Measurability; FEATURE_VERSION v2). Predictions exist only
#: on these rungs and outcomes join back through them, so accuracy
#: attribution can never mix horizons.
HORIZON_LADDER_DAYS: tuple[int, ...] = (1, 5, 21)


# --------------------------------------------------------------------------- #
# Deterministic IDs
# --------------------------------------------------------------------------- #
def _digest(preimage: str) -> str:
    """First 12 hex chars of the SHA-256 of a canonical preimage string."""
    return sha256(preimage.encode("utf-8")).hexdigest()[:12]


def new_evidence_id(run_id: str, payload_hash: str) -> str:
    """Deterministic evidence id: ``E`` + 12 hex over ``evidence|run|hash``.

    Preimage: ``evidence|{run_id}|{payload_hash}`` — evidence identity is the
    (run, payload content) pair, so the same block re-injected into a run
    dedupes to one row while the same block in a different run stays
    separate (per-run evidence chains must not collapse).
    """
    return "E" + _digest(f"evidence|{run_id}|{payload_hash}")


def new_prediction_id(
    run_id: str,
    analyst: str,
    prediction_scope: str,
    horizon_days: int,
    *,
    revision: int = 0,
) -> str:
    """Deterministic prediction id: ``P`` + 12 hex.

    Preimage: ``prediction|{run_id}|{analyst}|{prediction_scope}|{horizon_days}|r{revision}``.
    ``revision`` defaults to 0 (the original row);
    :func:`yialpha.ledger.predictions.revise_prediction` stamps the chain's
    next ``debate_revision`` number so each revision is its own deterministic
    row instead of colliding with the immutable original.
    """
    return "P" + _digest(
        f"prediction|{run_id}|{analyst}|{prediction_scope}|{horizon_days}|r{revision}"
    )


def new_outcome_id(prediction_id: str, horizon_days: int) -> str:
    """Deterministic outcome id: ``O`` + 12 hex over ``outcome|prediction|horizon``.

    Preimage: ``outcome|{prediction_id}|{horizon_days}`` — exactly the
    table's ``UNIQUE (prediction_id, horizon_days)`` constraint, so a due
    outcome rewritten with identical content lands on the same id and is an
    idempotent no-op.
    """
    return "O" + _digest(f"outcome|{prediction_id}|{horizon_days}")


# --------------------------------------------------------------------------- #
# Timestamp normalization (the one canonical date-vs-datetime reading)
# --------------------------------------------------------------------------- #
def parse_timestamp(ts: str) -> tuple[datetime, bool]:
    """Parse one ISO-8601 ledger timestamp into ``(UTC instant, date_only)``.

    A pure date (``YYYY-MM-DD``) becomes midnight UTC flagged ``date_only``;
    a full instant is converted to UTC (a naive datetime is assumed UTC).
    Anything unparseable raises ``ValueError`` — a malformed timestamp must
    never silently enter a point-in-time comparison.
    """
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp {ts!r}") from exc
    instant = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return instant.astimezone(UTC), "t" not in ts.lower()


def timestamp_as_utc(ts: str) -> datetime:
    """Timestamp normalized for elapsed-time arithmetic.

    Date-only strings count as midnight UTC; full instants are exact. Used
    for horizon arithmetic (``analysis_as_of + horizon_days <= now``).
    """
    return parse_timestamp(ts)[0]


def pit_latest_instant(ts: str) -> datetime:
    """Latest instant consistent with ``ts`` (date-only = end of that UTC day).

    A pure date admits any time on that day: the daily pipeline executes
    *during* its ``analysis_as_of`` date, so evidence that became available at
    10:00 on the run date was visible to the run. Applying end-of-day to BOTH
    sides of a PIT comparison keeps date-vs-datetime comparisons honest — a
    date-only ``available_at`` claims nothing finer than "sometime that day",
    so it is treated as the latest instant it could be.
    """
    instant, date_only = parse_timestamp(ts)
    if date_only:
        return instant.replace(hour=23, minute=59, second=59, microsecond=0)
    return instant


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
def validate_direction(direction: str) -> str:
    """Validate ``direction in {up, down, flat}``; returns the canonical value."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r} not in {sorted(DIRECTIONS)}")
    return direction


def validate_probability(name: str, value: float | None) -> float | None:
    """Validate a probability-style field (``prob_up`` / ``confidence``) in [0, 1]."""
    if value is None:
        return None
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value!r}")
    return number


def validate_horizon_days(horizon_days: int) -> int:
    """Validate ``horizon_days`` against the frozen ladder; returns it as int."""
    try:
        horizon = int(horizon_days)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"horizon_days must be an integer, got {horizon_days!r}"
        ) from exc
    if horizon not in HORIZON_LADDER_DAYS:
        raise ValueError(
            f"horizon_days {horizon!r} not in horizon ladder {HORIZON_LADDER_DAYS}"
        )
    return horizon


def validate_scope(scope: str) -> str:
    """Validate ``scope`` / ``prediction_scope`` against the scope enum."""
    if scope not in SCOPES:
        raise ValueError(f"scope {scope!r} not in {sorted(SCOPES)}")
    return scope


def validate_replayability(replayability: str) -> str:
    """Validate ``replayability`` against the replayability enum."""
    if replayability not in REPLAYABILITY_TAGS:
        raise ValueError(
            f"replayability {replayability!r} not in {sorted(REPLAYABILITY_TAGS)}"
        )
    return replayability


# --------------------------------------------------------------------------- #
# JSON list columns (evidence_ids on predictions, legs_missing on outcomes)
# --------------------------------------------------------------------------- #
def encode_json_list(values: Sequence[str] | None) -> str:
    """Canonical JSON array text: sorted, ``[]`` for empty/``None``.

    Sorting makes the stored bytes a function of the SET of ids/legs, so the
    immutability comparison treats a resubmission listing the same evidence
    in a different order as identical content, not a conflict.
    """
    return json.dumps(sorted(values or []), sort_keys=True)


def decode_json_list(raw: str | None) -> tuple[str, ...]:
    """Parse stored JSON array text back to a tuple (empty tuple for NULL/empty)."""
    if not raw:
        return ()
    return tuple(str(item) for item in json.loads(raw))


# --------------------------------------------------------------------------- #
# PredictionEntry (one horizon of a blind prediction)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PredictionEntry:
    """One horizon of a blind prediction (pre-debate, immutable once written).

    ``horizon_days`` and ``direction`` are required; every quantitative field
    is optional because an analyst may commit to direction and probability
    without a price target. Construction validates
    (``__post_init__``), so a row can never be written from an invalid entry.
    """

    horizon_days: int
    direction: str
    prob_up: float | None = None
    expected_return: float | None = None
    target_price: float | None = None
    target_currency: str | None = None
    price_basis: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "horizon_days", validate_horizon_days(self.horizon_days)
        )
        object.__setattr__(self, "direction", validate_direction(self.direction))
        object.__setattr__(
            self, "prob_up", validate_probability("prob_up", self.prob_up)
        )
        object.__setattr__(
            self, "confidence", validate_probability("confidence", self.confidence)
        )


_ENTRY_FIELDS = frozenset({field.name for field in fields(PredictionEntry)})


def coerce_prediction_entry(
    entry: Mapping[str, Any] | PredictionEntry,
) -> PredictionEntry:
    """Coerce one submitted entry (mapping or dataclass) into a validated entry.

    Mappings (the ``submit_prediction`` tool emits JSON objects) may carry
    only :class:`PredictionEntry` field names — a typo'd key must fail loudly
    instead of silently dropping a horizon from the ledger.
    """
    if isinstance(entry, PredictionEntry):
        return entry
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"prediction entry must be a mapping or PredictionEntry, "
            f"got {type(entry).__name__}"
        )
    extra = set(entry) - _ENTRY_FIELDS
    if extra:
        raise ValueError(
            f"unknown prediction entry keys {sorted(extra)}; "
            f"allowed keys: {sorted(_ENTRY_FIELDS)}"
        )
    missing = {"horizon_days", "direction"} - set(entry)
    if missing:
        raise ValueError(
            f"prediction entry requires 'horizon_days' and 'direction'; "
            f"missing {sorted(missing)}"
        )
    return PredictionEntry(
        horizon_days=entry["horizon_days"],
        direction=entry["direction"],
        prob_up=entry.get("prob_up"),
        expected_return=entry.get("expected_return"),
        target_price=entry.get("target_price"),
        target_currency=entry.get("target_currency"),
        price_basis=entry.get("price_basis"),
        confidence=entry.get("confidence"),
    )


# --------------------------------------------------------------------------- #
# Row records + converters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunRecord:
    """Row of ``runs``: one analysis run anchoring its evidence and predictions."""

    run_id: str
    ticker: str
    asset_type: str
    instrument_class: str | None
    analysis_as_of: str
    created_at: str
    config_digest: str | None
    schema_version: str


def row_to_run(row: sqlite3.Row) -> RunRecord:
    """Convert one ``runs`` row into a :class:`RunRecord`."""
    return RunRecord(
        run_id=row["run_id"],
        ticker=row["ticker"],
        asset_type=row["asset_type"],
        instrument_class=row["instrument_class"],
        analysis_as_of=row["analysis_as_of"],
        created_at=row["created_at"],
        config_digest=row["config_digest"],
        schema_version=row["schema_version"],
    )


@dataclass(frozen=True)
class EvidenceRecord:
    """Row of ``evidence``: one auditable, replayability-tagged evidence block."""

    evidence_id: str
    run_id: str
    source: str
    category: str
    symbol: str | None
    scope: str
    payload_hash: str
    source_url: str | None
    event_time: str | None
    available_at: str
    created_at: str
    replayability: str
    quality_status: str | None


def row_to_evidence(row: sqlite3.Row) -> EvidenceRecord:
    """Convert one ``evidence`` row into an :class:`EvidenceRecord`."""
    return EvidenceRecord(
        evidence_id=row["evidence_id"],
        run_id=row["run_id"],
        source=row["source"],
        category=row["category"],
        symbol=row["symbol"],
        scope=row["scope"],
        payload_hash=row["payload_hash"],
        source_url=row["source_url"],
        event_time=row["event_time"],
        available_at=row["available_at"],
        created_at=row["created_at"],
        replayability=row["replayability"],
        quality_status=row["quality_status"],
    )


@dataclass(frozen=True)
class AnalystPrediction:
    """Row of ``predictions``: one analyst's blind call on one horizon.

    One instance per horizon; a revision row carries the root id in
    ``original_prediction_id`` and its 1-based position in the chain in
    ``debate_revision`` (``None`` on original rows).
    """

    prediction_id: str
    run_id: str
    analyst: str
    instrument_id: str
    prediction_scope: str
    horizon_days: int
    direction: str
    prob_up: float | None
    expected_return: float | None
    target_price: float | None
    target_currency: str | None
    price_basis: str | None
    confidence: float | None
    evidence_ids: tuple[str, ...]
    analysis_as_of: str
    created_at: str
    schema_version: str
    feature_version: str
    original_prediction_id: str | None
    debate_revision: int | None
    revision_reason: str | None


def row_to_prediction(row: sqlite3.Row) -> AnalystPrediction:
    """Convert one ``predictions`` row into an :class:`AnalystPrediction`."""
    return AnalystPrediction(
        prediction_id=row["prediction_id"],
        run_id=row["run_id"],
        analyst=row["analyst"],
        instrument_id=row["instrument_id"],
        prediction_scope=row["prediction_scope"],
        horizon_days=int(row["horizon_days"]),
        direction=row["direction"],
        prob_up=row["prob_up"],
        expected_return=row["expected_return"],
        target_price=row["target_price"],
        target_currency=row["target_currency"],
        price_basis=row["price_basis"],
        confidence=row["confidence"],
        evidence_ids=decode_json_list(row["evidence_ids"]),
        analysis_as_of=row["analysis_as_of"],
        created_at=row["created_at"],
        schema_version=row["schema_version"],
        feature_version=row["feature_version"],
        original_prediction_id=row["original_prediction_id"],
        debate_revision=(
            int(row["debate_revision"])
            if row["debate_revision"] is not None
            else None
        ),
        revision_reason=row["revision_reason"],
    )


@dataclass(frozen=True)
class OutcomeRecord:
    """Row of ``outcomes``: the realized forward result of one prediction horizon.

    All return legs are signed fractions of notional (FEATURE_VERSION v2).
    ``legs_missing`` lists data legs the writer could not source; a row with
    ``status='pending'``/``'incomplete'`` stays on the
    :func:`yialpha.ledger.outcomes.pending_predictions` worklist.
    """

    outcome_id: str
    prediction_id: str
    run_id: str
    ticket_id: str | None
    horizon_days: int
    status: str
    contract_price_return: float | None
    underlying_return: float | None
    basis_return: float | None
    funding_pnl: float | None
    fees: float | None
    slippage: float | None
    liquidation_loss: float | None
    net_return: float | None
    legs_missing: tuple[str, ...]
    outcome_available_at: str | None
    computed_at: str


def row_to_outcome(row: sqlite3.Row) -> OutcomeRecord:
    """Convert one ``outcomes`` row into an :class:`OutcomeRecord`."""
    return OutcomeRecord(
        outcome_id=row["outcome_id"],
        prediction_id=row["prediction_id"],
        run_id=row["run_id"],
        ticket_id=row["ticket_id"],
        horizon_days=int(row["horizon_days"]),
        status=row["status"],
        contract_price_return=row["contract_price_return"],
        underlying_return=row["underlying_return"],
        basis_return=row["basis_return"],
        funding_pnl=row["funding_pnl"],
        fees=row["fees"],
        slippage=row["slippage"],
        liquidation_loss=row["liquidation_loss"],
        net_return=row["net_return"],
        legs_missing=decode_json_list(row["legs_missing"]),
        outcome_available_at=row["outcome_available_at"],
        computed_at=row["computed_at"],
    )


@dataclass(frozen=True)
class TicketMirror:
    """Row of ``tickets``: the frozen execution-ticket mirror (audit copy)."""

    ticket_id: str
    decision_id: str | None
    run_id: str | None
    payload: str
    ticket_version: str
    written_at: str


def row_to_ticket_mirror(row: sqlite3.Row) -> TicketMirror:
    """Convert one ``tickets`` row into a :class:`TicketMirror`."""
    return TicketMirror(
        ticket_id=row["ticket_id"],
        decision_id=row["decision_id"],
        run_id=row["run_id"],
        payload=row["payload"],
        ticket_version=row["ticket_version"],
        written_at=row["written_at"],
    )
