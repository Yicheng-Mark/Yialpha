"""Tickets mirror — the ticket table of the central ledger DB.

The full-state JSON log stays the human/report source of truth; the ledger
mirror exists so the outcome computer (:mod:`yialpha.ledger.outcome_compute`)
can join decision-time cost legs to a prediction's run without re-reading
log files, and so the web API can serve tickets read-only.

Append-first: one row per ticket_id (``INSERT OR IGNORE`` — a checkpoint
retry re-rendering the same ticket id never duplicates the row).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from yialpha.ledger.sqlite import ledger_transaction, utc_now_iso

logger = logging.getLogger(__name__)


def attach_ticket(
    ticket_id: str,
    decision_id: str | None,
    run_id: str | None,
    payload: dict[str, Any],
    ticket_version: str,
) -> None:
    """Mirror one ticket into the ledger DB (idempotent per ticket_id).

    ``payload`` is the ticket's own ``model_dump()`` serialized with sorted
    keys so identical tickets compare byte-equal across processes. Fail-soft
    by the record-stage invariant: a ledger problem is logged and swallowed,
    never raised into the run it is describing.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    try:
        with ledger_transaction() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO tickets "
                "(ticket_id, decision_id, run_id, payload, ticket_version, written_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ticket_id,
                    decision_id,
                    run_id,
                    serialized,
                    ticket_version,
                    utc_now_iso(),
                ),
            )
    except Exception:  # noqa: BLE001 -- record stage must never abort a run
        logger.warning(
            "attach_ticket(%s) failed; ticket not mirrored", ticket_id, exc_info=True
        )


def ticket_for_run(run_id: str) -> dict[str, Any] | None:
    """The run's latest mirrored ticket payload (parsed dict), or None."""
    from yialpha.ledger.sqlite import get_connection, ledger_exists

    if not ledger_exists():
        return None
    try:
        row = get_connection(readonly=True).execute(
            "SELECT payload FROM tickets WHERE run_id = ? "
            "ORDER BY written_at DESC, rowid DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    except Exception:  # noqa: BLE001 -- read helpers degrade, never raise
        logger.warning("ticket_for_run(%s) read failed", run_id, exc_info=True)
        return None
    if row is None:
        return None
    try:
        parsed: dict[str, Any] = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    return parsed
