"""Run registration + evidence rows (the audit trail behind predictions).

Every ``[EXTERNAL EVIDENCE]`` block an analyst saw becomes one immutable row
here, tagged with when the data became available (``available_at``) and
whether the same view can be reconstructed later (``replayability``). That is
what turns "the analyst said X" into "the analyst could only see Y at the
time" — the measurability core of V2.1.

Rules enforced by this module:

* **Idempotent registration** — :func:`register_run` is ``INSERT OR IGNORE``
  on ``run_id``; a retried/replayed pipeline re-registers the same run
  harmlessly (the original row, including its ``analysis_as_of``, is never
  touched — ledgers are append-only).
* **Content dedupe** — the evidence id is deterministic from
  ``(run_id, sha256(payload))``, so the same block re-injected into one run
  (tool-loop re-entries, prompt replays) collapses to a single row.
* **Point-in-time guard** — recording evidence whose ``available_at`` is
  later than the run's ``analysis_as_of`` raises
  :class:`PITEvidenceViolation` (a ``ValueError`` subclass): a
  prediction labelled "as of T" must never rest on data that did not exist
  at T. Date-vs-datetime normalization follows
  :func:`yialpha.ledger.models.pit_latest_instant` (a date-only
  ``analysis_as_of`` admits evidence available any time that UTC day).
"""

from __future__ import annotations

from hashlib import sha256

from yialpha.ledger.models import (
    EvidenceRecord,
    new_evidence_id,
    pit_latest_instant,
    row_to_evidence,
    validate_replayability,
    validate_scope,
)
from yialpha.ledger.sqlite import (
    get_connection,
    ledger_exists,
    ledger_transaction,
    utc_now_iso,
)
from yialpha.versions import SCHEMA_VERSION

#: Every ``evidence`` column, in INSERT/SELECT order (matches the v1 DDL).
_EVIDENCE_COLUMNS = (
    "evidence_id",
    "run_id",
    "source",
    "category",
    "symbol",
    "scope",
    "payload_hash",
    "source_url",
    "event_time",
    "available_at",
    "created_at",
    "replayability",
    "quality_status",
)


class PITEvidenceViolation(ValueError):
    """The point-in-time guard's typed rejection.

    Raised when evidence ``available_at`` postdates the run's
    ``analysis_as_of``: a prediction labelled "as of T" must never rest on
    data that did not exist at T. Subclasses :class:`ValueError` so existing
    ``except ValueError`` handling (and the fail-soft record stage in
    :mod:`yialpha.ledger.run_context`, which inspects this type to disclose
    the refusal explicitly) keeps working unchanged.
    """


def register_run(
    run_id: str,
    ticker: str,
    asset_type: str,
    instrument_class: str | None,
    analysis_as_of: str,
    config_digest: str | None = None,
) -> None:
    """Insert one ``runs`` row — the FK anchor for evidence and predictions.

    Idempotent on ``run_id`` (``INSERT OR IGNORE``): re-registering an
    existing run keeps the original row byte-for-byte, so a retried pipeline
    never rewrites history. The row is stamped with the current
    ``SCHEMA_VERSION`` so every run's provenance stays interpretable.
    """
    with ledger_transaction() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO runs "
            "(run_id, ticker, asset_type, instrument_class, analysis_as_of, "
            "created_at, config_digest, schema_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                ticker,
                asset_type,
                instrument_class,
                analysis_as_of,
                utc_now_iso(),
                config_digest,
                SCHEMA_VERSION,
            ),
        )


def record_evidence(
    run_id: str,
    source: str,
    category: str,
    symbol: str | None,
    scope: str,
    payload: str,
    *,
    source_url: str | None = None,
    event_time: str | None = None,
    available_at: str | None = None,
    replayability: str,
    quality_status: str | None = None,
    analysis_as_of: str | None = None,
) -> str:
    """Append one evidence row and return its deterministic ``evidence_id``.

    ``payload_hash`` is the SHA-256 hexdigest of the UTF-8 payload; the
    evidence id is derived from ``(run_id, payload_hash)``
    (:func:`yialpha.ledger.models.new_evidence_id`), so the write is
    ``INSERT OR IGNORE`` — re-injecting the same block into the same run
    dedupes to one row and returns the same id. A *different* payload in the
    same run is a different row by construction.

    ``available_at`` defaults to now (``utc_now_iso``). The run-context
    wrapper (:func:`yialpha.ledger.run_context.record_evidence_block`)
    anchors an omitted ``available_at`` to the run's ``analysis_as_of``
    before delegating here, so run-derived evidence is PIT-consistent by
    construction; this raw fallback only covers direct callers. When
    ``analysis_as_of`` is given, a :class:`PITEvidenceViolation` (a
    ``ValueError`` subclass) is raised if the evidence became available
    after that instant (PIT violation); a date-only ``analysis_as_of``
    admits any time on that UTC day. Foreign keys are enforced: an unknown
    ``run_id`` raises ``sqlite3.IntegrityError``.
    """
    validate_scope(scope)
    validate_replayability(replayability)
    payload_hash = sha256(payload.encode("utf-8")).hexdigest()
    available = available_at or utc_now_iso()
    if analysis_as_of is not None and pit_latest_instant(available) > (
        pit_latest_instant(analysis_as_of)
    ):
        raise PITEvidenceViolation(
            "evidence available_at exceeds analysis_as_of (PIT violation)"
        )
    evidence_id = new_evidence_id(run_id, payload_hash)
    values = {
        "evidence_id": evidence_id,
        "run_id": run_id,
        "source": source,
        "category": category,
        "symbol": symbol,
        "scope": scope,
        "payload_hash": payload_hash,
        "source_url": source_url,
        "event_time": event_time,
        "available_at": available,
        "created_at": utc_now_iso(),
        "replayability": replayability,
        "quality_status": quality_status,
    }
    with ledger_transaction() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO evidence "
            "(evidence_id, run_id, source, category, symbol, scope, payload_hash, "
            "source_url, event_time, available_at, created_at, replayability, "
            "quality_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(values[column] for column in _EVIDENCE_COLUMNS),
        )
    return evidence_id


def evidence_for_run(run_id: str) -> list[EvidenceRecord]:
    """All evidence rows linked to ``run_id`` (ordered by ``created_at``, id).

    Returns an empty list when no ledger DB exists yet (optional read).
    """
    if not ledger_exists():
        return []
    rows = (
        get_connection(readonly=True)
        .execute(
            "SELECT evidence_id, run_id, source, category, symbol, scope, "
            "payload_hash, source_url, event_time, available_at, created_at, "
            "replayability, quality_status "
            "FROM evidence WHERE run_id = ? ORDER BY created_at, evidence_id",
            (run_id,),
        )
        .fetchall()
    )
    return [row_to_evidence(row) for row in rows]


def evidence_ids_for_run(run_id: str) -> list[str]:
    """Just the deterministic evidence ids for ``run_id`` (same order as above)."""
    if not ledger_exists():
        return []
    rows = (
        get_connection(readonly=True)
        .execute(
            "SELECT evidence_id FROM evidence "
            "WHERE run_id = ? ORDER BY created_at, evidence_id",
            (run_id,),
        )
        .fetchall()
    )
    return [str(row["evidence_id"]) for row in rows]
