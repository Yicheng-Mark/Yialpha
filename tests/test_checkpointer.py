"""Unit tests for ``yialpha.graph.checkpointer`` pure functions and edge cases.

``test_checkpoint_resume.py`` already covers the crash-and-resume integration
round-trip with a real StateGraph. These tests pin the *pure* contracts that
the integration test doesn't directly assert: ``thread_id`` determinism and
signature-sensitivity, ``_db_path`` path-traversal safety, and
``clear_all_checkpoints`` counting / idempotency. All use ``tmp_path`` so no
DB state leaks between tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from yialpha.graph.checkpointer import (
    _db_path,
    clear_all_checkpoints,
    clear_checkpoint,
    thread_id,
)

pytestmark = pytest.mark.unit


class TestThreadId:
    def test_deterministic(self):
        assert thread_id("AAPL", "2026-01-01") == thread_id("AAPL", "2026-01-01")

    def test_ticker_case_insensitive(self):
        assert thread_id("aapl", "2026-01-01") == thread_id("AAPL", "2026-01-01")

    def test_different_ticker_different_id(self):
        assert thread_id("AAPL", "2026-01-01") != thread_id("MSFT", "2026-01-01")

    def test_different_date_different_id(self):
        assert thread_id("AAPL", "2026-01-01") != thread_id("AAPL", "2026-01-02")

    def test_signature_changes_id(self):
        base = thread_id("AAPL", "2026-01-01")
        with_sig = thread_id("AAPL", "2026-01-01", signature="analysts=market")
        assert base != with_sig

    def test_different_signature_different_id(self):
        a = thread_id("AAPL", "2026-01-01", signature="debate=1")
        b = thread_id("AAPL", "2026-01-01", signature="debate=2")
        assert a != b

    def test_id_is_hex_and_128_bit(self):
        # 32 hex chars = 128 bits of SHA-256. Was 16 hex chars (64 bits);
        # widened to remove the residual cross-run collision risk at zero cost.
        tid = thread_id("AAPL", "2026-01-01")
        assert len(tid) == 32
        int(tid, 16)  # raises ValueError if not hex


class TestDbPathSafety:
    def test_uppercases_ticker_in_filename(self, tmp_path):
        p = _db_path(tmp_path, "aapl")
        assert p.name == "AAPL.db"
        assert p.parent == tmp_path / "checkpoints"

    def test_creates_checkpoints_dir(self, tmp_path):
        cp_dir = tmp_path / "checkpoints"
        assert not cp_dir.exists()
        _db_path(tmp_path, "AAPL")
        assert cp_dir.exists()

    def test_path_traversal_ticker_rejected(self, tmp_path):
        # A malicious ticker must not escape the checkpoints directory.
        # safe_ticker_component rejects path separators outright (ValueError),
        # so _db_path never builds a path outside data_dir.
        with pytest.raises(ValueError):
            _db_path(tmp_path, "../../../etc/passwd")


class TestClearAllCheckpoints:
    def test_returns_zero_when_no_checkpoints_dir(self, tmp_path):
        assert clear_all_checkpoints(tmp_path) == 0

    def test_deletes_all_dbs_and_returns_count(self, tmp_path):
        for t in ("AAPL", "MSFT", "NVDA"):
            _db_path(tmp_path, t).touch()
        assert clear_all_checkpoints(tmp_path) == 3
        assert not list((tmp_path / "checkpoints").glob("*.db"))

    def test_idempotent(self, tmp_path):
        _db_path(tmp_path, "AAPL").touch()
        clear_all_checkpoints(tmp_path)
        # Second call on now-empty dir returns 0, no error.
        assert clear_all_checkpoints(tmp_path) == 0

    def test_ignores_non_db_files(self, tmp_path):
        db = _db_path(tmp_path, "AAPL")
        db.touch()
        (tmp_path / "checkpoints" / "README.txt").touch()
        assert clear_all_checkpoints(tmp_path) == 1
        # Non-db files are left alone.
        assert (tmp_path / "checkpoints" / "README.txt").exists()

    def test_permission_error_skips_db_but_deletes_rest(
        self, tmp_path, monkeypatch, caplog
    ):
        """Round-13: one undeletable DB (a live SQLite handle on Windows)
        must not abort the sweep — the rest still delete, the failure is
        reported in a WARNING, and the return counts successes only."""
        for t in ("AAPL", "MSFT", "NVDA"):
            _db_path(tmp_path, t).touch()
        held = _db_path(tmp_path, "MSFT")

        real_unlink = Path.unlink

        def guarded_unlink(self, missing_ok=False):
            if self == held:
                raise PermissionError(
                    5, "Access is denied (another process holds the handle)"
                )
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", guarded_unlink)
        with caplog.at_level(logging.WARNING, logger="yialpha.graph.checkpointer"):
            deleted = clear_all_checkpoints(tmp_path)
        # The other two DBs were deleted despite the first failure.
        assert deleted == 2
        assert held.exists()
        assert not _db_path(tmp_path, "AAPL").exists()
        assert not _db_path(tmp_path, "NVDA").exists()
        # The skipped file is named in the summary warning.
        assert any(
            "MSFT.db" in r.message and "could not remove" in r.message
            for r in caplog.records
        )

    def test_vanishing_db_mid_sweep_is_skipped(self, tmp_path, monkeypatch):
        """Round-13: a DB that disappears between glob and unlink (sibling
        sweep racing us) counts as neither deleted nor failed — the sweep
        finishes with the files that are actually still there."""
        for t in ("AAPL", "MSFT"):
            _db_path(tmp_path, t).touch()
        vanished = _db_path(tmp_path, "AAPL")

        real_unlink = Path.unlink

        def guarded_unlink(self, missing_ok=False):
            if self == vanished:
                raise FileNotFoundError(2, "raced a sibling sweep")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", guarded_unlink)
        assert clear_all_checkpoints(tmp_path) == 1
        assert not _db_path(tmp_path, "MSFT").exists()

    def test_sidecar_unlink_failure_does_not_crash_or_undercount(
        self, tmp_path, monkeypatch
    ):
        """Round-13: a -wal sidecar that cannot be removed is garbage-
        collection noise, not a DB failure — the db still counts as deleted
        and the sweep never crashes on it."""
        db = _db_path(tmp_path, "AAPL")
        db.touch()
        wal = db.with_name(db.name + "-wal")
        wal.touch()

        real_unlink = Path.unlink

        def guarded_unlink(self, missing_ok=False):
            if self == wal:
                raise PermissionError(5, "handle still open on the sidecar")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", guarded_unlink)
        assert clear_all_checkpoints(tmp_path) == 1
        assert not db.exists()


class TestClearCheckpointRobustness:
    def test_corrupt_db_only_warns(self, tmp_path, caplog):
        """Round-13: a corrupt DB makes SQLite raise DatabaseError (NOT an
        OperationalError subclass) on the first execute; clear_checkpoint
        must warn and return, never raise — its callers guard completed runs
        and this is the innermost layer of that defense."""

        db = _db_path(tmp_path, "AAPL")
        # Enough non-SQLite garbage that connect->execute refuses the file.
        db.write_bytes(b"definitely not a sqlite database" * 8)
        with caplog.at_level(logging.WARNING, logger="yialpha.graph.checkpointer"):
            clear_checkpoint(tmp_path, "AAPL", "2026-01-01")  # must not raise
        assert any(
            "checkpoint cleanup failed" in r.message for r in caplog.records
        )
        # The corrupt file itself is untouched (deletion is not this API's job).
        assert db.exists()

    def test_operational_error_still_warns(self, tmp_path, caplog, monkeypatch):
        """The pre-round-13 handled class (locked DB -> OperationalError)
        keeps its warning path — widening the except must not narrow it."""
        import sqlite3

        _db_path(tmp_path, "AAPL").touch()

        # Simulate the locked-DB failure by handing the module a connection
        # whose first execute raises OperationalError.
        class _LockedConn:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("database is locked")

            def commit(self):
                pass

            def close(self):
                pass

        import yialpha.graph.checkpointer as cp

        monkeypatch.setattr(cp.sqlite3, "connect", lambda *a, **k: _LockedConn())
        with caplog.at_level(logging.WARNING, logger="yialpha.graph.checkpointer"):
            clear_checkpoint(tmp_path, "AAPL", "2026-01-01")  # must not raise
        assert any(
            "checkpoint cleanup failed" in r.message for r in caplog.records
        )
