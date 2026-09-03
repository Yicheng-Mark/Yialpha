"""Ledger-group regression tests — v2.4 portfolio-control fix batch.

Two hermetic seams (in-memory SQLite only, no network, no config files):

* R2 — transaction-level idempotency of ``commit_final_ticket``. The
  position_id replay gate only protects OPENs (a CLOSE creates no position
  row), so an already-committed CLOSE replayed after the symbol was
  re-opened by a LATER ticket used to fall through and close that new
  position. The committed-operation registry pins the whole sequence:
  OPEN / CLOSE / no-change replays are side-effect-free no-ops, a
  conflicting same-id replay is refused with a warning, and a different
  ticket keeps the one-open-position-per-symbol replace semantics.

* P2 — the operator positions-file signal: an explicit empty book (a legal
  ``[]``) must be distinguishable from invalid input (bad rows dropped
  with a structured warning plus a ``+invalid:<n>`` label marker), and
  both from an unreadable file (degrade + structured warning).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest

import yialpha.ledger.portfolio as portfolio_module
from yialpha.ledger.portfolio import (
    commit_final_ticket,
    load_positions_input,
    new_position_id,
    open_positions,
)

# The three harness tables (the reviewer-harness seam). The committed-op
# registry is deliberately NOT pre-created: commit_final_ticket must
# upgrade older ledgers itself via its runtime IF NOT EXISTS DDL.
SCHEMA = """
CREATE TABLE tickets (
    ticket_id TEXT PRIMARY KEY, decision_id TEXT, run_id TEXT,
    payload TEXT, ticket_version TEXT, written_at TEXT
);
CREATE TABLE portfolio_snapshots (
    snapshot_id TEXT PRIMARY KEY, run_id TEXT, payload TEXT, created_at TEXT
);
CREATE TABLE positions (
    position_id TEXT PRIMARY KEY, ticket_id TEXT, run_id TEXT, symbol TEXT,
    side TEXT, signed_weight REAL, opened_at TEXT, closed_at TEXT, payload TEXT
);
"""


@contextmanager
def private_ledger():
    """One private :memory: ledger wired through the portfolio seams."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)

    @contextmanager
    def transaction():
        cursor = connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            yield cursor
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            cursor.close()

    try:
        with ExitStack() as mocks:
            mocks.enter_context(
                patch.object(portfolio_module, "ledger_transaction", transaction)
            )
            mocks.enter_context(
                patch.object(portfolio_module, "ledger_exists", return_value=True)
            )
            mocks.enter_context(
                patch.object(
                    portfolio_module,
                    "get_connection",
                    lambda *, readonly=False: connection,
                )
            )
            yield connection
    finally:
        connection.close()


def commit(
    ticket_id: str,
    *,
    symbol: str = "BTCUSDT",
    weight: float = 0.12,
    close: bool = False,
) -> bool:
    """One final-ticket commit (a byte-identical retry = same arguments)."""
    return commit_final_ticket(
        ticket_payload={"ticket_id": ticket_id, "status": "APPROVED"},
        ticket_id=ticket_id,
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id="S-" + ticket_id,
        position_symbol=symbol,
        position_side="FLAT" if close else ("LONG" if weight > 0 else "SHORT"),
        position_signed_weight=0.0 if close else weight,
        open_position=not close,
        close_existing=close,
        instrument_class="pure_crypto_perp",
    )


def commit_hold(ticket_id: str, *, symbol: str = "BTCUSDT") -> bool:
    """A VETO / FLAT ticket: both transition flags False, book untouched."""
    return commit_final_ticket(
        ticket_payload={"ticket_id": ticket_id, "status": "VETOED"},
        ticket_id=ticket_id,
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id="S-" + ticket_id,
        position_symbol=symbol,
        position_side="FLAT",
        position_signed_weight=0.0,
        open_position=False,
        close_existing=False,
        instrument_class="pure_crypto_perp",
    )


def book() -> list[dict[str, Any]]:
    return [
        {
            "ticket_id": row["ticket_id"],
            "symbol": row["symbol"],
            "signed_weight": row["signed_weight"],
        }
        for row in open_positions()
    ]


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    row = connection.execute(
        "SELECT (SELECT COUNT(*) FROM tickets) AS tickets, "
        "(SELECT COUNT(*) FROM portfolio_snapshots) AS snapshots, "
        "(SELECT COUNT(*) FROM positions) AS positions, "
        "(SELECT COUNT(*) FROM portfolio_committed_ops) AS ops"
    ).fetchone()
    return {
        "tickets": row["tickets"],
        "snapshots": row["snapshots"],
        "positions": row["positions"],
        "ops": row["ops"],
    }


# --------------------------------------------------------------------------- #
# R2 — transaction-level idempotency of committed operations
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_close_replay_never_closes_a_later_same_symbol_position():
    """R2 repro (reviewer Case 1): T1 open → C1 close → T2 open → replay C1.

    The replayed CLOSE must return the original True result and leave T2
    open — an already-committed CLOSE is a no-op, never a close of whatever
    position happens to be open at retry time.
    """
    with private_ledger() as conn:
        assert commit("T1") is True
        assert commit("C1", close=True) is True
        assert commit("T2") is True
        before = book()
        assert [row["ticket_id"] for row in before] == ["T2"]

        assert commit("C1", close=True) is True
        assert book() == before
        # The replay was a byte-stable transaction: nothing written anywhere.
        assert table_counts(conn) == {
            "tickets": 3,
            "snapshots": 3,
            "positions": 2,
            "ops": 3,
        }


@pytest.mark.unit
def test_every_committed_action_kind_replays_as_a_noop():
    """OPEN, CLOSE and no-change (VETO) replays are all side-effect-free."""
    with private_ledger() as conn:
        assert commit("T1") is True
        open_book = book()
        assert commit("T1") is True  # exact OPEN replay
        assert book() == open_book

        assert commit("C1", close=True) is True
        after_close = table_counts(conn)
        assert commit("C1", close=True) is True  # exact CLOSE replay
        assert book() == []
        assert table_counts(conn) == after_close

        assert commit("T3") is True
        held_book = book()
        assert commit_hold("V1") is True
        assert commit_hold("V1") is True  # exact no-change replay
        assert book() == held_book
        assert table_counts(conn) == {
            "tickets": 4,
            "snapshots": 4,
            "positions": 2,
            "ops": 4,
        }


@pytest.mark.unit
def test_conflicting_same_ticket_replay_never_mutates_the_book(caplog):
    """Same ticket_id re-presented with different content: no-op + warning."""
    with private_ledger():
        assert commit("T1") is True
        assert commit("C1", close=True) is True
        assert commit("T2") is True
        assert commit("TE1", symbol="ETHUSDT") is True
        before = book()
        with caplog.at_level(logging.WARNING, logger="yialpha.ledger.portfolio"):
            # The registered C1 was "close BTCUSDT"; this replay claims
            # "close ETHUSDT" — a different operation under the same ticket.
            assert commit("C1", close=True, symbol="ETHUSDT") is True
        assert book() == before  # ETH stays open, BTC stays closed
        messages = [record.getMessage() for record in warnings_in(caplog)]
        assert any("different operation" in message for message in messages)


@pytest.mark.unit
def test_different_ticket_same_symbol_still_replaces():
    """The registry gates by ticket, never by symbol: replace semantics keep."""
    with private_ledger() as conn:
        assert commit("T1") is True
        assert commit("T2") is True  # different ticket, same symbol
        assert [row["ticket_id"] for row in open_positions()] == ["T2"]
        assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 2
        assert table_counts(conn)["ops"] == 2


@pytest.mark.unit
def test_pre_registry_ledger_keeps_the_position_id_gate_for_open_replays():
    """Upgrade path: a ledger written before the registry existed (an OPEN
    row present, no registry row) still must not stack or replace on an
    OPEN replay — the position_id gate (D3) remains the last line."""
    with private_ledger() as conn:
        conn.execute(
            "INSERT INTO positions (position_id, ticket_id, run_id, symbol, "
            "side, signed_weight, opened_at, closed_at, payload) "
            "VALUES (?, 'T1', NULL, 'BTCUSDT', 'LONG', 0.12, "
            "'2026-09-03T00:00:00+00:00', NULL, '{}')",
            (new_position_id("T1", "BTCUSDT"),),
        )
        conn.commit()
        assert commit("T1") is True  # falls through the empty registry...
        # ...and the position_id gate must still block the transition.
        assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 1
        assert [row["ticket_id"] for row in open_positions()] == ["T1"]


# --------------------------------------------------------------------------- #
# P2 — empty vs invalid positions-file signal
# --------------------------------------------------------------------------- #


@contextmanager
def positions_file(path: str):
    from yialpha.dataflows.config import set_config

    set_config({"portfolio_positions_file": path})
    try:
        yield
    finally:
        set_config({"portfolio_positions_file": ""})


def write_positions_file(tmp_path, content: str) -> str:
    path = tmp_path / "positions.json"
    path.write_text(content, encoding="utf-8")
    return str(path)


def warnings_in(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


@pytest.mark.unit
def test_positions_file_explicit_empty_book_is_legal_and_labeled(tmp_path, caplog):
    with (
        positions_file(write_positions_file(tmp_path, "[]")),
        caplog.at_level(logging.WARNING, logger="yialpha.ledger.portfolio"),
    ):
        rows, source = load_positions_input()
    assert rows == []
    assert source == "file:positions.json+empty"
    assert warnings_in(caplog) == []  # a legal empty book is not an error


@pytest.mark.unit
def test_positions_file_all_bad_rows_warn_and_mark_the_label(tmp_path, caplog):
    with (
        positions_file(write_positions_file(tmp_path, "[1, null, {}]")),
        caplog.at_level(logging.WARNING, logger="yialpha.ledger.portfolio"),
    ):
        rows, source = load_positions_input()
    assert rows == []
    assert source == "file:positions.json+invalid:3"
    messages = [record.getMessage() for record in warnings_in(caplog)]
    assert any("invalid and dropped" in message for message in messages)


@pytest.mark.unit
def test_positions_file_partial_bad_rows_keep_valid_rows_and_warn(tmp_path, caplog):
    content = json.dumps(
        [
            {"symbol": "ETHUSDT", "side": "LONG", "signed_weight": 0.2},
            "junk-row",
            {"symbol": "SOLUSDT", "signed_weight": "not-a-number"},
        ]
    )
    with (
        positions_file(write_positions_file(tmp_path, content)),
        caplog.at_level(logging.WARNING, logger="yialpha.ledger.portfolio"),
    ):
        rows, source = load_positions_input()
    assert [row["symbol"] for row in rows] == ["ETHUSDT"]
    assert source == "file:positions.json+invalid:2"
    messages = [record.getMessage() for record in warnings_in(caplog)]
    assert any("invalid and dropped" in message for message in messages)


@pytest.mark.unit
def test_positions_file_valid_rows_keep_the_bare_label(tmp_path):
    content = json.dumps(
        [{"symbol": "ETHUSDT", "side": "LONG", "signed_weight": 0.2}]
    )
    with positions_file(write_positions_file(tmp_path, content)):
        rows, source = load_positions_input()
    assert source == "file:positions.json"
    assert rows[0]["signed_weight"] == pytest.approx(0.2)


@pytest.mark.unit
def test_positions_file_unreadable_degrades_with_structured_warning(tmp_path, caplog):
    with (
        positions_file(write_positions_file(tmp_path, "{not json")),
        caplog.at_level(logging.WARNING, logger="yialpha.ledger.portfolio"),
    ):
        rows, source = load_positions_input()
    assert rows == []
    assert source == "empty"  # degraded to the (empty) ledger view
    messages = [record.getMessage() for record in warnings_in(caplog)]
    assert any("unreadable/malformed" in message for message in messages)


@pytest.mark.unit
def test_positions_file_not_a_list_degrades_with_warning(tmp_path, caplog):
    with (
        positions_file(write_positions_file(tmp_path, '{"symbol": "BTCUSDT"}')),
        caplog.at_level(logging.WARNING, logger="yialpha.ledger.portfolio"),
    ):
        rows, source = load_positions_input()
    assert rows == []
    assert source == "empty"
    messages = [record.getMessage() for record in warnings_in(caplog)]
    assert any("unreadable/malformed" in message for message in messages)


@pytest.mark.unit
def test_explicit_empty_file_over_ledger_rows_is_labeled(tmp_path):
    with private_ledger():
        assert commit("T1") is True
        with positions_file(write_positions_file(tmp_path, "[]")):
            rows, source = load_positions_input()
    assert [row["symbol"] for row in rows] == ["BTCUSDT"]  # ledger survives
    assert source == "ledger+file:positions.json+empty"
