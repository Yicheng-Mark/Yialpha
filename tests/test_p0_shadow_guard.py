"""Temporary-file-only coverage for the offline P0 isolation boundary."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.p0_shadow.guard import (
    GuardError,
    PathGuard,
    isolated_runtime,
    require_runtime,
    resolve_cohort_root,
)


@pytest.fixture(autouse=True)
def _fresh_yfinance_cache_managers():
    """Re-arm yfinance's process-wide cache singletons before each runtime.

    isolated_runtime refuses already-initialized managers to mirror the
    fresh-process rule; in a full-suite run earlier dataflow tests have
    opened them. set_location() closes _db and resets it in place.
    """
    from yfinance import cache as yf_cache

    for name in ("_TzDBManager", "_CookieDBManager", "_ISINDBManager"):
        manager = getattr(yf_cache, name, None)
        if manager is not None:
            manager.set_location(manager.get_location())
    yield


@pytest.mark.parametrize("value", ["", " ", ".", "relative", "../escape"])
def test_invalid_roots_fail_without_creating(tmp_path, value):
    before = list(tmp_path.iterdir())
    with pytest.raises(GuardError):
        resolve_cohort_root(value, project_root=tmp_path / "project")
    assert list(tmp_path.iterdir()) == before


@pytest.mark.parametrize("relative", ["", "../outside", "a/../../outside", ".env", ".env.local",
                                      "file:stream", "trailing.", "trailing "])
def test_child_paths_fail_before_writes(tmp_path, relative):
    paths = PathGuard(tmp_path / "cohort")
    with pytest.raises(GuardError):
        paths.write_json(relative, {"fixture": True})
    assert not paths.root.exists()


def test_default_and_protected_roots_rejected(tmp_path):
    project = tmp_path / "project"
    for path in (project, tmp_path, project / "results", project / "analysis_output",
                 project / "analysis_output" / "shadow-glm-round6b-20260905" / "nested",
                 Path.home() / ".yialpha" / "ledger"):
        with pytest.raises(GuardError):
            resolve_cohort_root(path, project_root=project)
    expected = project / "analysis_output" / "new-cohort"
    assert resolve_cohort_root(expected, project_root=project) == expected
    assert not project.exists()


def test_absolute_child_is_rejected(tmp_path):
    paths = PathGuard(tmp_path / "cohort")
    with pytest.raises(GuardError):
        paths.resolve(tmp_path / "outside.json")


def test_evidence_creation_is_exclusive(tmp_path):
    paths = PathGuard(tmp_path / "cohort")
    target = paths.write_json("raw/request/record.json", {"sample_kind": "offline_fixture"})
    original = target.read_bytes()
    with pytest.raises(GuardError, match="overwrite"):
        paths.write_json("raw/request/record.json", {"different": True})
    assert target.read_bytes() == original
    assert json.loads(original)["sample_kind"] == "offline_fixture"


def test_hardlink_writes_are_rejected(tmp_path):
    paths = PathGuard(tmp_path / "cohort")
    paths.root.mkdir()
    protected = tmp_path / "outside.txt"
    protected.write_text("unchanged", encoding="utf-8")
    target = paths.root / "linked.txt"
    os.link(protected, target)
    with pytest.raises(GuardError, match="hardlinked"):
        paths.write_bytes("linked.txt", b"changed", exclusive=False)
    assert protected.read_text(encoding="utf-8") == "unchanged"


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "cohort"
    root.mkdir()
    try:
        (root / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this host: {exc}")
    with pytest.raises(GuardError, match="symlink/junction"):
        PathGuard(root).write_json("linked/wrong.json", {})
    assert not (outside / "wrong.json").exists()


def test_runtime_rejects_live_or_unsafe_attempt_before_import(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    with patch("scripts.p0_shadow.guard.importlib.import_module") as importer:
        for attempt, mode in (("safe", "live"), ("../escape", "offline")):
            with pytest.raises(GuardError), isolated_runtime(root, attempt, network_mode=mode):
                pytest.fail("invalid runtime entered")
        importer.assert_not_called()
    assert list(root.iterdir()) == []


def test_runtime_config_and_sqlite_are_scoped_and_restore(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    from yialpha.dataflows.config import get_config

    previous = get_config()
    with isolated_runtime(root, "config-test") as summary:
        import yfinance.cache as cache

        from yialpha.dataflows.disk_cache import cache_base_dir
        from yialpha.ledger.sqlite import get_connection, ledger_db_path

        assert require_runtime(root).root == root
        assert get_config()["analysis_only"] is True
        assert get_config()["live_execution_enabled"] is False
        assert get_config()["llm_cache"] is False
        assert ledger_db_path() == str(root / "ledger" / "portfolio.db")
        assert cache_base_dir() == str(root / "cache" / "config-test" / "data")
        for manager in (cache._TzDBManager, cache._CookieDBManager, cache._ISINDBManager):
            assert manager.get_location() == str(root / "cache" / "config-test" / "yfinance-internal")
        connection = get_connection()
        assert connection.execute("PRAGMA temp_store").fetchone()[0] == 2
        assert sys.dont_write_bytecode
        assert summary["network_attempts"] == 0
    assert get_config() == previous
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(GuardError, match="active"):
        require_runtime(root)
    (tmp_path / "outside-after.txt").write_text("scope ended", encoding="utf-8")


def _denied_guard(operation, *args, **kwargs):
    with pytest.raises(GuardError):
        operation(*args, **kwargs)


def _denied(operation, *args, **kwargs):
    """Assert a dotenv-family access is denied by the guard or an earlier
    regression-entry audit hook (RuntimeError) when one owns the process."""
    with pytest.raises((GuardError, RuntimeError), match="forbidden"):
        operation(*args, **kwargs)


def _denied_spawn():
    """The guard denies spawning external programs on any platform."""
    with pytest.raises(GuardError):
        if os.name == "nt":  # Windows
            os.startfile(".")
        else:  # POSIX
            os.posix_spawn(sys.executable, [sys.executable], os.environ)


def test_runtime_blocks_external_io_before_files_exist(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    outside = tmp_path / "outside.db"
    with isolated_runtime(root, "write-test"):
        _denied_guard(outside.write_bytes, b"bad")
        _denied_guard(sqlite3.connect, outside)
        _denied_guard(sqlite3.connect, f"file:{outside}?mode=ro", uri=True)
        _denied_guard((tmp_path / "new-directory").mkdir)
        _denied_spawn()
        _denied((root / ".env.local").read_bytes)
        _denied((root / ".env.enterprise").write_bytes, b"bad")
    assert not outside.exists()
    assert not (tmp_path / "new-directory").exists()
    assert list(root.iterdir()) == []


def test_sqlite_attach_and_temp_escape_are_denied(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    with isolated_runtime(root, "sqlite-test"):
        connection = sqlite3.connect(root / "fixture.db")
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            connection.execute("ATTACH DATABASE ? AS external", (str(tmp_path / "escape.db"),))
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            connection.execute("PRAGMA temp_store=FILE")
        with pytest.raises(GuardError, match="extensions"):
            connection.enable_load_extension(True)
    assert not (tmp_path / "escape.db").exists()


def test_socket_dns_and_native_curl_are_blocked(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    with isolated_runtime(root, "network-test") as summary:
        import curl_cffi
        import curl_cffi.requests

        curl = curl_cffi.Curl()
        try:
            for operation in (
                lambda: socket.getaddrinfo("example.invalid", 443),
                lambda: socket.create_connection(("127.0.0.1", 9)),
                lambda: curl.perform(),
                lambda: curl_cffi.requests.get("https://example.invalid"),
            ):
                with pytest.raises(GuardError, match="disabled"):
                    operation()
        finally:
            curl.close()
    assert summary["network_attempts"] == 4


def test_dotenv_functions_never_open_files(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    with isolated_runtime(root, "dotenv-test"):
        import dotenv
        import dotenv.main

        assert dotenv.find_dotenv(usecwd=True) == ""
        assert dotenv.load_dotenv(root / ".env") is False
        assert dotenv.main.dotenv_values(root / ".env.enterprise") == {}
        # Read an ordinary source file outside the cohort: imports remain legal.
        assert Path(__file__).read_text(encoding="utf-8").startswith('"""')


def test_existing_supplier_cache_is_rejected_without_io(tmp_path):
    root = tmp_path / "cohort"
    (root / "cache" / "attempt" / "data").mkdir(parents=True)
    with pytest.raises(GuardError, match="fresh supplier cache"), \
            isolated_runtime(root, "attempt"):
        pytest.fail("reused cache accepted")
    assert not (root / "ledger").exists()


@pytest.mark.parametrize("platform,uri,expected", [
    ("nt", "///C:/cohort/ledger/db.sqlite", "C:/cohort/ledger/db.sqlite"),
    ("nt", "//C:/cohort/db.sqlite", "C:/cohort/db.sqlite"),
    ("nt", "//server/share/db.sqlite", "//server/share/db.sqlite"),
    ("posix", "///tmp/cohort/ledger/db.sqlite", "/tmp/cohort/ledger/db.sqlite"),
    ("posix", "//server/share/db.sqlite", "//server/share/db.sqlite"),
    ("posix", "/tmp/cohort/db.sqlite", "/tmp/cohort/db.sqlite"),
])
def test_sqlite_uri_normalization_is_platform_aware(platform, uri, expected):
    from scripts.p0_shadow.guard import _normalize_sqlite_uri

    assert _normalize_sqlite_uri(uri, platform) == expected
