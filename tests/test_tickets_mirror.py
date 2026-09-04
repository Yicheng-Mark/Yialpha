"""Tickets mirror: fail-soft writes, degraded reads, latest-ticket order.

Covers yialpha/ledger/tickets_mirror.py — ``attach_ticket`` (append-first
``INSERT OR IGNORE`` mirror of a run's execution ticket) and
``ticket_for_run`` (the newest mirrored payload for a run, or None).

Fail-soft contract under test: a ledger problem is logged and swallowed at
the record stage — it must never abort the run being described — and the
read helpers degrade to ``None`` instead of raising. conftest's autouse
``_runtime_ledger_isolated`` fixture already points ``ledger_db_path`` at a
per-test tmp file and resets thread-local connections around each test.
No network anywhere: everything below is local SQLite.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3

import pytest

from yialpha.ledger.sqlite import get_connection, ledger_exists
from yialpha.ledger.tickets_mirror import attach_ticket, ticket_for_run

_LATER = "2099-01-01T00:00:00+00:00"  # far-future written_at beats any real clock
_EARLIER = "2026-01-01T00:00:00+00:00"


def _insert_raw_ticket(
    ticket_id, run_id, payload_text, written_at, decision_id=None, version="v1"
):
    """Direct row insert (schema must already exist via a prior attach)."""
    get_connection().execute(
        "INSERT INTO tickets "
        "(ticket_id, decision_id, run_id, payload, ticket_version, written_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ticket_id, decision_id, run_id, payload_text, version, written_at),
    )


# --------------------------------------------------------------------------- #
# attach_ticket: append-first mirror
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_attach_ticket_roundtrip_is_idempotent_per_ticket_id():
    attach_ticket("T1", "D1", "R1", {"ticket_id": "T1", "alpha": 1}, "v1")
    # A checkpoint retry re-renders the same ticket id: ignored, not duplicated.
    attach_ticket("T1", "D1", "R1", {"ticket_id": "T1", "alpha": 999}, "v1")

    assert ledger_exists()
    rows = get_connection(readonly=True).execute(
        "SELECT ticket_id, decision_id, run_id, payload, ticket_version "
        "FROM tickets WHERE ticket_id = 'T1'"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["decision_id"] == "D1"
    assert row["run_id"] == "R1"
    assert row["ticket_version"] == "v1"
    # Payload round-trips as the dict (serialized with sorted keys).
    assert json.loads(row["payload"]) == {"ticket_id": "T1", "alpha": 1}
    mirrored = ticket_for_run("R1")
    assert mirrored == {"ticket_id": "T1", "alpha": 1}


# --------------------------------------------------------------------------- #
# attach_ticket: fail-soft when the ledger is broken
# --------------------------------------------------------------------------- #


class _ExplodingCursor:
    """Cursor stand-in whose execute raises (mid-transaction failure)."""

    def execute(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("database table is locked")


@contextlib.contextmanager
def _transaction_raising_on_entry(*_args, **_kwargs):
    raise sqlite3.OperationalError("database is locked")
    yield  # pragma: no cover - never reached


@contextlib.contextmanager
def _transaction_raising_on_execute(*_args, **_kwargs):
    yield _ExplodingCursor()


@pytest.mark.unit
@pytest.mark.parametrize("broken_txn", [_transaction_raising_on_entry,
                                        _transaction_raising_on_execute])
def test_attach_ticket_failsoft_when_ledger_raises(
    monkeypatch, caplog, broken_txn
):
    """A broken ledger (locked DB, failed execute) is a WARNING, never an
    exception out of attach_ticket — the run it describes must survive."""
    monkeypatch.setattr(
        "yialpha.ledger.tickets_mirror.ledger_transaction", broken_txn
    )
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.tickets_mirror"):
        attach_ticket("T-BROKEN", "D1", "R1", {"ticket_id": "T-BROKEN"}, "v1")

    warnings = [
        rec for rec in caplog.records
        if rec.levelno == logging.WARNING and "T-BROKEN" in rec.getMessage()
    ]
    assert warnings, "expected a fail-soft warning naming the ticket"
    assert warnings[0].exc_info is not None  # the cause is preserved for ops


@pytest.mark.unit
def test_attach_ticket_failure_does_not_poison_later_writes(monkeypatch):
    """After a mocked ledger failure the process keeps working: once the
    seam is healthy again, the next attach lands and is readable."""
    monkeypatch.setattr(
        "yialpha.ledger.tickets_mirror.ledger_transaction",
        _transaction_raising_on_entry,
    )
    attach_ticket("T-FAIL", "D1", "R1", {"seq": 0}, "v1")  # swallowed

    monkeypatch.undo()  # restore the real ledger_transaction
    attach_ticket("T-OK", "D1", "R1", {"seq": 1}, "v1")
    mirrored = ticket_for_run("R1")
    assert mirrored == {"seq": 1}


# --------------------------------------------------------------------------- #
# ticket_for_run: degraded reads
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_ticket_for_run_missing_db_returns_none():
    """No ledger file yet (conftest gave us a fresh tmp path): the read-only
    surface reports None instead of creating or raising."""
    assert ledger_exists() is False
    assert ticket_for_run("R-NOBODY") is None


@pytest.mark.unit
def test_ticket_for_run_unknown_run_returns_none():
    attach_ticket("T1", "D1", "R1", {"ticket_id": "T1"}, "v1")  # creates schema
    assert ticket_for_run("R-UNKNOWN") is None


@pytest.mark.unit
def test_ticket_for_run_corrupt_payload_degrades_to_none():
    """Latest row wins; if that row's payload JSON is corrupt the read
    degrades to None (it does not fall back to older rows or raise)."""
    attach_ticket("T-GOOD", "D1", "R1", {"ticket_id": "T-GOOD"}, "v1")
    _insert_raw_ticket("T-CORRUPT", "R1", "{not-json", _LATER)

    assert ticket_for_run("R1") is None


@pytest.mark.unit
def test_ticket_for_run_read_error_degrades_to_none(monkeypatch, caplog):
    """A read-side DB error (e.g. ro open fails) is a logged warning + None."""
    attach_ticket("T1", "D1", "R1", {"ticket_id": "T1"}, "v1")  # ledger exists

    def _broken_connection(*_args, **_kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(
        "yialpha.ledger.sqlite.get_connection", _broken_connection
    )
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.tickets_mirror"):
        assert ticket_for_run("R1") is None
    assert any(
        rec.levelno == logging.WARNING and "ticket_for_run(R1)" in rec.getMessage()
        for rec in caplog.records
    )


# --------------------------------------------------------------------------- #
# "Latest ticket wins" determinism for one run_id
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("reverse_order", [False, True],
                         ids=["chronological", "reversed_insert"])
def test_latest_ticket_wins_regardless_of_insert_order(monkeypatch, reverse_order):
    """Two distinct tickets for one run_id with DISTINCT written_at values:
    the mirror must surface the max-written_at payload no matter which one
    was appended first. The clock is pinned via the module-level
    ``utc_now_iso`` seam and each timestamp stays bound to its own ticket,
    so reversing the insert order really does flip which row lands first."""
    entries = [
        ("T-OLD", {"ticket_id": "T-OLD", "seq": 1}, _EARLIER),
        ("T-NEW", {"ticket_id": "T-NEW", "seq": 2}, _LATER),
    ]
    if reverse_order:
        entries.reverse()

    clock = iter(ts for _tid, _payload, ts in entries)
    monkeypatch.setattr(
        "yialpha.ledger.tickets_mirror.utc_now_iso", lambda: next(clock)
    )
    for ticket_id, payload, _ts in entries:
        attach_ticket(ticket_id, "D1", "R-LATEST", payload, "v1")

    mirrored = ticket_for_run("R-LATEST")
    assert mirrored["ticket_id"] == "T-NEW"
    assert mirrored["seq"] == 2


@pytest.mark.unit
def test_same_second_tie_breaks_by_insertion_order(monkeypatch):
    """written_at is second-granularity ISO: two tickets for one run written
    in the same second must still resolve deterministically — the row
    appended last (higher rowid) wins, not an arbitrary SQLite tie order."""
    monkeypatch.setattr(
        "yialpha.ledger.tickets_mirror.utc_now_iso", lambda: _LATER
    )
    attach_ticket("T-FIRST", "D1", "R-TIE", {"ticket_id": "T-FIRST", "seq": 1}, "v1")
    attach_ticket("T-SECOND", "D1", "R-TIE", {"ticket_id": "T-SECOND", "seq": 2}, "v1")

    mirrored = ticket_for_run("R-TIE")
    assert mirrored["ticket_id"] == "T-SECOND"
    assert mirrored["seq"] == 2
