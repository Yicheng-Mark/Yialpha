"""Portfolio ledger — snapshots, positions, and the V2.4 atomic commit.

The V2.4 resolver path closes the key chain ``run_id → prediction_id →
decision_id → ticket_id → position_id``: the Final Ticket (tickets mirror),
its Portfolio Snapshot, and the resulting Position rows are written in ONE
``BEGIN IMMEDIATE`` transaction (:func:`commit_final_ticket`), so a crash can
never strand one without the others.

Append-first + idempotent: every write is ``INSERT OR IGNORE`` keyed by a
deterministic id, so a checkpoint retry re-rendering the same ticket never
duplicates a snapshot or opens a second position. A committed-operation
registry (``portfolio_committed_ops``) extends that idempotency to EVERY
committed transition — CLOSEs and no-change commits included, which create
no position row of their own.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
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


def load_positions_input() -> tuple[list[dict[str, Any]], str]:
    """READ-ONLY portfolio snapshot input for shadow/enforced previews.

    Shadow acceptance item 1: shadow mode writes no positions, so previews
    against the (empty) ledger alone would prove nothing about the
    constraints. The input is therefore composed of TWO read-only layers:

    1. the ledger's open positions (populated only when the enforced mode
       has actually committed positions);
    2. an operator-maintained JSON file (config ``portfolio_positions_file``,
       a list of ``{"symbol", "side", "signed_weight", "instrument_class"?}``
       objects) OVERLAID on top — same-symbol file rows replace ledger rows.

    Returns ``(rows, source_label)`` where the label names exactly what fed
    the snapshot (``"empty"`` / ``"ledger"`` / ``"file:<name>"`` /
    ``"ledger+file:<name>"``) so every preview is auditable. The label ALSO
    separates the file's zero-row / bad-input states, which a bare row
    count cannot prove:

    * ``+empty`` — the file is an explicit ``[]``: a LEGAL zero-row book;
    * ``+invalid:<n>`` — ``<n>`` of the file's rows were invalid (not
      objects, missing ``symbol``, unparseable numbers) and were DROPPED
      with a WARNING; the surviving rows are NOT a validated book;
    * an unreadable/malformed file (bad JSON, not a list) degrades to the
      ledger-only view with a WARNING (``positions_file_status=unreadable``)
      — the preview never fails on its input book, and nothing is silently
      swallowed: every dropped row and parse failure is counted and logged.
    """
    ledger_rows = open_positions()
    rows = list(ledger_rows)
    source = "ledger" if ledger_rows else "empty"

    from yialpha.dataflows.config import get_config

    path = str(get_config().get("portfolio_positions_file") or "").strip()
    if not path:
        return rows, source
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError("positions file must be a JSON list")
    except Exception:  # noqa: BLE001 -- input book degrades, never breaks
        logger.warning(
            "portfolio_positions_file %r unreadable/malformed "
            "(positions_file_status=unreadable); previewing against the "
            "ledger view only — this file proves no valid book",
            path,
            exc_info=True,
        )
        return rows, source

    # Row-level validation: a bad row is DROPPED and COUNTED — never
    # silently skipped. (The pre-2026-09 fix batch ``continue``d bad rows
    # away, so a file of all-bad rows was indistinguishable — and equally
    # silent — as an explicit legal empty book.)
    file_rows: list[dict[str, Any]] = []
    bad_rows = 0
    for entry in entries:
        try:
            if not isinstance(entry, dict) or "symbol" not in entry:
                raise ValueError("row is not an object with a 'symbol' key")
            file_rows.append(
                {
                    "position_id": f"file:{entry['symbol']}",
                    "ticket_id": None,
                    "run_id": None,
                    "symbol": str(entry["symbol"]),
                    "side": str(entry.get("side") or (
                        "LONG" if float(entry.get("signed_weight", 0.0)) >= 0 else "SHORT"
                    )),
                    "signed_weight": float(entry.get("signed_weight", 0.0)),
                    "opened_at": "",
                    "payload": {
                        "instrument_class": entry.get("instrument_class")
                        or "unknown_perp",
                        "source": "positions_file",
                    },
                }
            )
        except Exception:  # noqa: BLE001 -- one bad row is dropped, not fatal
            bad_rows += 1
    if bad_rows:
        logger.warning(
            "portfolio_positions_file %s: %d of %d rows invalid and dropped "
            "(positions_file_status=invalid_rows:%d/%d); the surviving rows "
            "are NOT a validated book",
            path,
            bad_rows,
            len(entries),
            bad_rows,
            len(entries),
        )

    # Overlay: a file row REPLACES the same-symbol ledger row (the file is
    # the operator's current statement of the book).
    by_symbol = {row["symbol"]: row for row in rows}
    for file_row in file_rows:
        by_symbol[file_row["symbol"]] = file_row
    rows = list(by_symbol.values())
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    base = f"ledger+file:{name}" if ledger_rows else f"file:{name}"
    if not entries:
        source = f"{base}+empty"
    elif bad_rows:
        source = f"{base}+invalid:{bad_rows}"
    else:
        source = base
    return rows, source


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
    close_existing: bool = False,
    instrument_class: str | None = None,
) -> bool:
    """Atomically write the Final Ticket + Snapshot + position transition.

    All rows commit inside ONE ``BEGIN IMMEDIATE`` transaction; every INSERT
    is OR IGNORE, so re-running the same resolver output for the same run
    (checkpoint retry) is a no-op. The position transition distinguishes the
    two sanctioned mutations (shadow acceptance item 2 — 拒绝开仓 ≠ 批准平仓):

    * ``open_position=True``  — an approved/resized ENTRY: the same symbol's
      prior open row closes first (replace semantics; one open position per
      symbol), then the new row opens.
    * ``close_existing=True`` (mutually exclusive) — an APPROVED
      close-intent ticket: the same symbol's open row closes, and NOTHING
      opens.
    * both False — a VETO / FLAT candidate: existing rows are UNTOUCHED.
      Refusing a new trade never modifies the current book.

    Transaction-level idempotency covers all three outcomes: every committed
    operation is registered under its full content key (ticket_id + action +
    symbol + side + weight) in ``portfolio_committed_ops`` INSIDE the same
    transaction as its effects. A re-presented ticket_id — an
    already-committed CLOSE retry included, which creates no position row
    and would otherwise slip past the position_id gate below — is a full
    no-op that returns the original result and can never act on a position
    opened by a LATER ticket; a same-id-different-content replay is refused
    with a warning (the registered operation is authoritative,
    append-first).

    An EXACT replay (same ticket_id → same deterministic position_id) skips
    the position transition entirely — the close-UPDATE alone would close
    the very row the retry re-opens — while a different ticket for the same
    symbol keeps the replace semantics. ``instrument_class`` resolves
    explicit kwarg → ``ticket_payload`` key; a resolved class is stamped
    onto the ticket mirror and the position payload (unresolved → key
    omitted; the graph read path falls back to ``unknown_perp``).
    """
    # Fallback chain: explicit kwarg → the ticket payload's own key (the
    # class may legitimately arrive only inside ticket_payload).
    resolved_class = (
        instrument_class
        if instrument_class is not None
        else ticket_payload.get("instrument_class")
    )
    if resolved_class is not None:
        # Stamp the resolved class onto the ticket mirror as well (INSERT OR
        # IGNORE keeps a replay byte-stable) so the class survives the
        # ledger roundtrip even when the caller passed it only via payload.
        ticket_payload = {**ticket_payload, "instrument_class": resolved_class}
    position_id = new_position_id(ticket_id, position_symbol)
    now = utc_now_iso()
    serialized_ticket = json.dumps(ticket_payload, sort_keys=True, default=str)
    serialized_snapshot = json.dumps(snapshot_payload, sort_keys=True, default=str)
    position_payload: dict[str, Any] = {
        "ticket_id": ticket_id,
        "symbol": position_symbol,
        "side": position_side,
        "signed_weight": position_signed_weight,
        "run_id": run_id,
    }
    if resolved_class is not None:
        # The class rides the position row so the next candidate's
        # asset-class aggregation reads the book's true composition; omitted
        # when unresolved — the graph read path falls back to unknown_perp.
        position_payload["instrument_class"] = resolved_class
    serialized_position = json.dumps(
        position_payload, sort_keys=True, default=str
    )
    try:
        # Committed-operation identity: the FULL operation content (action +
        # symbol + side + quantity) under the ticket_id (the operation's
        # sequence number). Computed inside the try: float() may raise on a
        # malformed weight, and a bad call must degrade to the record-stage
        # False contract, never raise into the run.
        op_action = (
            "open" if open_position else ("close" if close_existing else "hold")
        )
        op_key = "O" + hashlib.sha256(
            "|".join((
                "portfolio-op",
                ticket_id,
                op_action,
                position_symbol,
                position_side,
                format(float(position_signed_weight), ".17g"),
            )).encode()
        ).hexdigest()[:12]
        with ledger_transaction() as cur:
            # Committed-operation registry — transaction-level idempotency
            # for ALL three outcomes. The position_id gate below only
            # protects OPENs (they create a position row); a CLOSE creates
            # none, so an already-committed CLOSE replayed after the symbol
            # was re-opened by a LATER ticket used to fall through and close
            # that new position. Every committed operation is registered in
            # the SAME transaction as its effects, so any re-presented
            # ticket_id is a no-op before a single row is touched. The
            # runtime IF NOT EXISTS DDL upgrades ledgers written before
            # this table existed (move into sqlite._migrate() when the
            # schema version next bumps).
            cur.execute(
                "CREATE TABLE IF NOT EXISTS portfolio_committed_ops ("
                "op_key TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, "
                "action TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, "
                "signed_weight REAL NOT NULL, committed_at TEXT NOT NULL)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_committed_ops_ticket "
                "ON portfolio_committed_ops (ticket_id)"
            )
            prior = cur.execute(
                "SELECT op_key FROM portfolio_committed_ops WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if prior is not None:
                if prior[0] != op_key:
                    # Same ticket re-presented with DIFFERENT content: the
                    # registered operation is authoritative (append-first)
                    # — re-applying anything could act on a book state the
                    # original operation never saw.
                    logger.warning(
                        "commit_final_ticket(%s): ticket already committed "
                        "with a different operation (registered %s, replay "
                        "%s); keeping the committed one, book untouched",
                        ticket_id,
                        prior[0],
                        op_key,
                    )
                return True
            cur.execute(
                "INSERT OR IGNORE INTO portfolio_committed_ops "
                "(op_key, ticket_id, action, symbol, side, signed_weight, "
                "committed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    op_key,
                    ticket_id,
                    op_action,
                    position_symbol,
                    position_side,
                    position_signed_weight,
                    now,
                ),
            )
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
            # Exact-replay gate: the deterministic position_id already in the
            # table means this very ticket committed before — re-running the
            # close-UPDATE would close the position the retry "re-opens"
            # (the INSERT is OR IGNORE and lands nothing), so the whole
            # transition is skipped and an identical replay is a full no-op.
            # A DIFFERENT ticket for the same symbol hashes to a different
            # position_id and keeps the replace semantics below. Read via
            # the transaction cursor, not ledger_exists(): replay idempotency
            # is a property of the positions table, not of the DB file.
            replayed = (
                cur.execute(
                    "SELECT 1 FROM positions WHERE position_id = ?",
                    (position_id,),
                ).fetchone()
                is not None
            )
            if open_position and not replayed:
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
                        position_id,
                        ticket_id,
                        run_id,
                        position_symbol,
                        position_side,
                        position_signed_weight,
                        now,
                        serialized_position,
                    ),
                )
            elif close_existing:
                # Approved CLOSE intent: close the same symbol's open row and
                # open nothing. The anti-case is deliberate — a VETOed or
                # FLAT ticket takes this branch NEVER (both flags False) and
                # therefore cannot touch the current book.
                cur.execute(
                    "UPDATE positions SET closed_at = ? "
                    "WHERE symbol = ? AND closed_at IS NULL",
                    (now, position_symbol),
                )
        return True
    except Exception:  # noqa: BLE001 -- record stage must never abort a run
        logger.warning(
            "commit_final_ticket(%s) failed; final ticket not recorded", ticket_id,
            exc_info=True,
        )
        return False
