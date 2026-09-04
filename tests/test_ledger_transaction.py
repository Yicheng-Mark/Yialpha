"""Commit-boundary safety of ``ledger_transaction`` (yialpha/ledger/sqlite).

Pins the COMMIT failure path: a commit that fails (disk full, SQLITE_BUSY,
...) must ROLLBACK the still-open transaction before re-raising — otherwise
the thread-local connection stays inside a dangling transaction and every
later ``BEGIN IMMEDIATE`` dies with "cannot start a transaction within a
transaction", permanently bricking that thread's ledger.
"""

from __future__ import annotations

import sqlite3

import pytest

from yialpha.ledger.sqlite import ledger_transaction


class _FakeConn:
    """Minimal connection double recording every executed statement.

    ``fail_on`` maps a statement to the exception it raises, so COMMIT can
    fail while BEGIN/ROLLBACK succeed — exactly the disk-full/BUSY shape.
    """

    def __init__(self, fail_on: dict[str, Exception] | None = None) -> None:
        self.statements: list[str] = []
        self.fail_on = fail_on or {}
        self._cursor = object()  # identity is all the CM's yield needs

    def execute(self, sql: str, *args: object):
        self.statements.append(sql)
        exc = self.fail_on.get(sql)
        if exc is not None:
            raise exc
        return None

    def cursor(self):
        return self._cursor


@pytest.mark.unit
def test_commit_failure_rolls_back_then_reraises():
    conn = _FakeConn(fail_on={"COMMIT": sqlite3.OperationalError("disk I/O error")})
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"), ledger_transaction(conn=conn) as cur:
        assert cur is conn._cursor
    # BEGIN ran, the failed COMMIT was followed by ROLLBACK, and the ORIGINAL
    # commit exception surfaced (not a secondary rollback error).
    assert conn.statements == ["BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"]


@pytest.mark.unit
def test_body_failure_rolls_back_without_commit():
    conn = _FakeConn()
    with pytest.raises(RuntimeError, match="boom"), ledger_transaction(conn=conn):
        raise RuntimeError("boom")
    assert conn.statements == ["BEGIN IMMEDIATE", "ROLLBACK"]


@pytest.mark.unit
def test_successful_transaction_commits_without_rollback():
    conn = _FakeConn()
    with ledger_transaction(conn=conn):
        pass
    assert conn.statements == ["BEGIN IMMEDIATE", "COMMIT"]
