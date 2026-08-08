"""Unit tests for ``yiagents.graph.checkpointer`` pure functions and edge cases.

``test_checkpoint_resume.py`` already covers the crash-and-resume integration
round-trip with a real StateGraph. These tests pin the *pure* contracts that
the integration test doesn't directly assert: ``thread_id`` determinism and
signature-sensitivity, ``_db_path`` path-traversal safety, and
``clear_all_checkpoints`` counting / idempotency. All use ``tmp_path`` so no
DB state leaks between tests.
"""

from __future__ import annotations

import pytest

from yiagents.graph.checkpointer import (
    _db_path,
    clear_all_checkpoints,
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

    def test_id_is_hex_and_short(self):
        tid = thread_id("AAPL", "2026-01-01")
        assert len(tid) == 16
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
