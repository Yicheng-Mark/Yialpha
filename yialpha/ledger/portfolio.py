"""Portfolio ledger — snapshots, positions, and the V2.4 atomic commit.

The V2.4 resolver path closes the key chain ``run_id → prediction_id →
decision_id → ticket_id → position_id``: the Final Ticket (tickets mirror),
its Portfolio Snapshot, and the resulting Position rows are written in ONE
``BEGIN IMMEDIATE`` transaction (:func:`commit_final_ticket`), so a crash can
never strand one without the others.

Append-first + idempotent: every write is ``INSERT OR IGNORE`` keyed by a
deterministic id, so a checkpoint retry re-rendering the same ticket never
duplicates a snapshot or opens a second position.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from yialpha.ledger.sqlite import get_connection, ledger_exists, ledger_transaction, utc_now_iso

logger = logging.getLogger(__name__)


def new_snapshot_id(run_id: str) -> str:
    """Deterministic per-run snapshot id (one snapshot per run per resolver)."""
    return "S" + hashlib.sha256(f"portfolio-snapshot|{run_id}".encode()).hexdigest()[:12]


def new_position_id(ticket_id: str, symbol: str) -> str:
    """Deterministic per-(ticket, symbol) position id (retry-idempotent)."""
    return "N" + hashlib.sha256(f"position|{ticket_id}|{symbol}".encode()).hexdigest()[:12]


def open_positions() -> list[dict[str, Any]]:
    """Every not-yet-closed position (the pre-candidate portfolio state)."""
    if not ledger_exists():
        return []
    try:
        rows = get_connection(readonly=True).execute(
            "SELECT position_id, ticket_id, run_id, symbol, side, signed_weight, "
            "opened_at, payload FROM positions WHERE closed_at IS NULL "
            "ORDER BY opened_at"
        ).fetchall()
    except Exception:  # noqa: BLE001 -- read helpers degrade, never raise
        logger.warning("open_positions() read failed", exc_info=True)
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            payload = {}
        out.append(
            {
                "position_id": row["position_id"],
                "ticket_id": row["ticket_id"],
                "run_id": row["run_id"],
                "symbol": row["symbol"],
                "side": row["side"],
                "signed_weight": row["signed_weight"],
                "opened_at": row["opened_at"],
                "payload": payload,
            }
        )
    return out


def snapshot_by_id(snapshot_id: str) -> dict[str, Any] | None:
    """One portfolio snapshot payload (read-only web/API surface)."""
    if not ledger_exists():
        return None
    try:
        row = get_connection(readonly=True).execute(
            "SELECT payload FROM portfolio_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
    except Exception:  # noqa: BLE001 -- read helpers degrade, never raise
        return None
    if row is None:
        return None
    try:
        parsed: dict[str, Any] = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    return parsed


def commit_final_ticket(
    *,
    ticket_payload: dict[str, Any],
    ticket_id: str,
    decision_id: str | None,
    run_id: str | None,
    ticket_version: str,
    snapshot_payload: dict[str, Any],
    snapshot_id: str,
    position_symbol: str,
    position_side: str,
    position_signed_weight: float,
    open_position: bool,
) -> bool:
    """Atomically write the Final Ticket + Snapshot + (optional) Position.

    All three rows commit inside ONE ``BEGIN IMMEDIATE`` transaction; every
    INSERT is OR IGNORE, so re-running the same resolver output for the same
    run (checkpoint retry) is a no-op. ``open_position=False`` (FLAT / VETO /
    REDUCE-to-zero) still commits the ticket + snapshot but writes no
    position row — the audit trail exists without a position having been
    opened. Fail-soft by the record-stage invariant: a ledger error is
    logged and swallowed (returns False), never raised into the run.
    """
    now = utc_now_iso()
    serialized_ticket = json.dumps(ticket_payload, sort_keys=True, default=str)
    serialized_snapshot = json.dumps(snapshot_payload, sort_keys=True, default=str)
    serialized_position = json.dumps(
        {
            "ticket_id": ticket_id,
            "symbol": position_symbol,
            "side": position_side,
            "signed_weight": position_signed_weight,
            "run_id": run_id,
        },
        sort_keys=True,
        default=str,
    )
    try:
        with ledger_transaction() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO tickets "
                "(ticket_id, decision_id, run_id, payload, ticket_version, written_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticket_id, decision_id, run_id, serialized_ticket, ticket_version, now),
            )
            cur.execute(
                "INSERT OR IGNORE INTO portfolio_snapshots "
                "(snapshot_id, run_id, payload, created_at) VALUES (?, ?, ?, ?)",
                (snapshot_id, run_id, serialized_snapshot, now),
            )
            if open_position:
                # Same-symbol REPLACE semantics (retry invariant): opening a
                # position for a symbol closes that symbol's prior open row
                # first — one open position per symbol, so a checkpoint
                # retry / re-decision can never stack duplicates. This is a
                # lifecycle transition (closed_at), not a correction: the
                # append-first rule outlaws rewriting predictions/evidence,
                # not advancing a position's state machine.
                cur.execute(
                    "UPDATE positions SET closed_at = ? "
                    "WHERE symbol = ? AND closed_at IS NULL",
                    (now, position_symbol),
                )
                cur.execute(
                    "INSERT OR IGNORE INTO positions "
                    "(position_id, ticket_id, run_id, symbol, side, signed_weight, "
                    "opened_at, closed_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                    (
                        new_position_id(ticket_id, position_symbol),
                        ticket_id,
                        run_id,
                        position_symbol,
                        position_side,
                        position_signed_weight,
                        now,
                        serialized_position,
                    ),
                )
        return True
    except Exception:  # noqa: BLE001 -- record stage must never abort a run
        logger.warning(
            "commit_final_ticket(%s) failed; final ticket not recorded", ticket_id,
            exc_info=True,
        )
        return False
