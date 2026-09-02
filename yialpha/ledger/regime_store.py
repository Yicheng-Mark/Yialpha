"""The ``regimes`` table — full versioned ``RegimeState`` rows, keyed by id.

One row per distinct regime id (migration v2). ``regime_id`` is the
content hash of the classifier inputs plus ``REGIME_VERSION``
(:func:`yialpha.regime.state.compute_regime_id`), so the write is
``INSERT OR IGNORE``: recomputing the same regime for a replayed run — or
two runs on the same instrument/date landing in the same context — dedupes
to the ONE row that already names it. The full state (every classifier
field, coverage weights, missing inputs) rides in ``payload`` as canonical
JSON; the scalar columns exist for direct SQL filtering without parsing.

Fail-soft by the record-stage contract: a ledger error is logged as a
WARNING and swallowed — the regime is advisory context, and losing the row
must never abort the run that computed it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from yialpha.ledger.sqlite import (
    get_connection,
    ledger_exists,
    ledger_transaction,
)

logger = logging.getLogger(__name__)


def upsert_regime(regime_state_payload: dict[str, Any]) -> None:
    """Persist one full RegimeState payload (idempotent on ``regime_id``).

    ``regime_state_payload`` is the full state mapping (as produced by
    ``dataclasses.asdict(RegimeState)``) plus the ledger-side context keys
    ``ticker`` / ``instrument_class`` that ride alongside it. Unknown keys
    are simply stored inside the payload blob. A ``sqlite3.Error`` is
    logged and swallowed (record stage must never break a run).
    """
    try:
        regime_id = str(regime_state_payload.get("regime_id") or "")
        if not regime_id:
            raise ValueError("regime_state_payload carries no regime_id")
        with ledger_transaction() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO regimes "
                "(regime_id, payload, regime_version, analysis_as_of, ticker, "
                "instrument_class, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    regime_id,
                    json.dumps(regime_state_payload, sort_keys=True, default=str),
                    str(regime_state_payload.get("regime_version") or ""),
                    regime_state_payload.get("analysis_as_of"),
                    regime_state_payload.get("ticker"),
                    regime_state_payload.get("instrument_class"),
                    str(regime_state_payload.get("computed_at") or ""),
                ),
            )
    except (sqlite3.Error, ValueError) as exc:
        logger.warning("upsert_regime failed; regime row not recorded: %s", exc)


def regime_by_id(regime_id: str) -> dict[str, Any] | None:
    """The stored regime row as a dict, or ``None`` when absent.

    Shape: the scalar columns plus the parsed ``payload`` dict under the
    ``"payload"`` key. Degrades to ``None`` when the ledger DB does not
    exist yet or the read fails (optional read, never raises).
    """
    if not regime_id or not ledger_exists():
        return None
    try:
        row = (
            get_connection(readonly=True)
            .execute(
                "SELECT regime_id, payload, regime_version, analysis_as_of, ticker, "
                "instrument_class, computed_at FROM regimes WHERE regime_id = ?",
                (regime_id,),
            )
            .fetchone()
        )
    except sqlite3.Error as exc:
        logger.warning("regime_by_id(%s) read failed: %s", regime_id, exc)
        return None
    if row is None:
        return None
    try:
        payload = json.loads(str(row["payload"]))
    except ValueError:
        payload = {}
    return {
        "regime_id": row["regime_id"],
        "regime_version": row["regime_version"],
        "analysis_as_of": row["analysis_as_of"],
        "ticker": row["ticker"],
        "instrument_class": row["instrument_class"],
        "computed_at": row["computed_at"],
        "payload": payload,
    }
