"""Fail-closed, single-threaded isolation for the offline P0 runner.

This is an application guard for the runner's narrow Python entry points, not
an OS sandbox for hostile native extensions. Importing this module uses only
the standard library and does not import yialpha or inspect dotenv files.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sqlite3
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import unquote


class GuardError(RuntimeError):
    """An operation falls outside the offline cohort boundary."""


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ACTIVE: dict[str, Any] | None = None
_HOOK_INSTALLED = False
_DENY_DOTENV = False
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _env_path(value: object) -> bool:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    return any(part.lower().startswith(".env") for part in Path(os.fsdecode(value)).parts)


def _no_links(path: Path, *, writing: bool = False) -> None:
    """Reject all reparse points, including Windows junctions, and hardlinks."""
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise GuardError("symlink/junction paths are not permitted")
        if writing and stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise GuardError("hardlinked write targets are not permitted")


def _absolute(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise GuardError("an explicit absolute path is required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise GuardError("relative paths and directory traversal are forbidden")
    for component in path.parts[1:]:
        if ":" in component or component.endswith((" ", ".")):
            raise GuardError("ambiguous path components are forbidden")
    if _env_path(path):
        raise GuardError("dotenv family paths are forbidden")
    _no_links(path)
    return path.resolve()


def resolve_cohort_root(
    value: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
    must_exist: bool = False,
) -> Path:
    """Validate without creating directories or opening a ledger.

    Project-local cohorts must be new descendants of ``analysis_output``;
    external roots (including test temp directories) are supported. Existence
    and cohort identity for create/resume are the manifest layer's concern.
    """
    project = _absolute(project_root)
    path = _absolute(value)
    defaults = (
        Path.home() / ".yialpha",
        project / "results",
        project / "logs",
        project / "cache",
        project / "data_cache",
        project / "memory",
        project / "analysis_output" / "shadow-glm-round6-20260905",
        project / "analysis_output" / "shadow-glm-round6b-20260905",
    )
    if path == Path(path.anchor) or _inside(project, path):
        raise GuardError("project root and its ancestors cannot be cohort roots")
    if any(_inside(path, item.resolve()) or path == item.resolve() for item in defaults):
        raise GuardError("default and protected directories are forbidden")
    output = project / "analysis_output"
    if _inside(path, project) and (path == output or not _inside(path, output)):
        raise GuardError("project-local cohorts must be below analysis_output")
    if path.exists() and not path.is_dir():
        raise GuardError("cohort root must be a directory")
    if must_exist and not path.is_dir():
        raise GuardError("cohort directory does not exist")
    return path


class PathGuard:
    """Resolve cohort-relative files and publish evidence without overwrite."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = resolve_cohort_root(root, project_root=_PROJECT_ROOT)

    def resolve(self, relative: str | os.PathLike[str]) -> Path:
        if not isinstance(relative, (str, os.PathLike)) or not str(relative).strip():
            raise GuardError("a nonempty cohort-relative path is required")
        rel = Path(relative)
        if rel.is_absolute() or rel.drive or any(p in ("..", ".") for p in rel.parts):
            raise GuardError("absolute paths and directory traversal are forbidden")
        path = _absolute(self.root / rel)
        if not _inside(path, self.root) or path == self.root:
            raise GuardError("target must be a file below the cohort root")
        _no_links(self.root)
        _no_links(path, writing=True)
        return path

    def write_bytes(
        self, relative: str | os.PathLike[str], data: bytes, *, exclusive: bool = True
    ) -> Path:
        path = self.resolve(relative)
        if exclusive and path.exists():
            raise GuardError("refusing to overwrite an existing evidence file")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Revalidate immediately before opening; x mode prevents overwrite even
        # if another writer creates the name after the existence check.
        path = self.resolve(relative)
        try:
            with path.open("xb" if exclusive else "wb") as stream:
                stream.write(data)
        except FileExistsError as exc:
            raise GuardError("refusing to overwrite an existing evidence file") from exc
        return path

    def write_json(
        self, relative: str | os.PathLike[str], payload: Any, *, exclusive: bool = True
    ) -> Path:
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n").encode("utf-8")
        return self.write_bytes(relative, encoded, exclusive=exclusive)


def require_runtime(root: str | os.PathLike[str] | None = None) -> PathGuard:
    """Require the active, configured runtime, optionally for an exact root."""
    if _ACTIVE is None or not _ACTIVE.get("ready"):
        raise GuardError("an active isolated_runtime is required")
    guard = _ACTIVE["guard"]
    if root is not None and _absolute(root) != guard.root:
        raise GuardError("the active runtime belongs to a different cohort")
    return guard


def _write_path(value: object) -> Path:
    assert _ACTIVE is not None
    if not isinstance(value, (str, bytes, os.PathLike)):
        raise GuardError("unverifiable file descriptor writes are forbidden")
    path = Path(os.fsdecode(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    path = _absolute(path)
    if not _inside(path, _ACTIVE["guard"].root):
        raise GuardError("file mutation outside the cohort root is forbidden")
    _no_links(path, writing=True)
    return path


def _sqlite_path(value: object) -> Path:
    if not isinstance(value, (str, bytes, os.PathLike)):
        raise GuardError("SQLite requires an explicit cohort path")
    raw = os.fsdecode(value)
    if raw.startswith("file:"):
        raw = unquote(raw[5:].split("?", 1)[0])
        # sqlite accepts file:///C:/... drive URIs (any number of leading
        # slashes) on Windows; only UNC //server/share targets stay rejected.
        if os.name == "nt":
            drive = re.match(r"^/+([A-Za-z]:/.+)$", raw)
            if drive:
                raw = drive.group(1)
    if not raw or raw == ":memory:" or raw.startswith("//"):
        raise GuardError("SQLite requires an explicit cohort file")
    target = _write_path(raw)
    for suffix in ("-journal", "-wal", "-shm"):
        _write_path(str(target) + suffix)
    return target


def _sqlite_authorizer(action: int, arg1: str | None, arg2: str | None,
                       _database: str | None, _trigger: str | None) -> int:
    if action == sqlite3.SQLITE_ATTACH:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA:
        key = (arg1 or "").lower()
        if key in ("temp_store_directory", "data_store_directory"):
            return sqlite3.SQLITE_DENY
        if key == "temp_store" and arg2 is not None and arg2.lower() not in ("2", "memory"):
            return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in (
        "load_extension", "writefile",
    ):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and _DENY_DOTENV and args and _env_path(args[0]):
        raise GuardError("dotenv family file access is forbidden")
    if _ACTIVE is None:
        return
    if threading.get_ident() != _ACTIVE["thread_id"]:
        raise GuardError("the offline runtime is restricted to one thread")
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & _WRITE_FLAGS
        ):
            _write_path(args[0])
    elif event == "sqlite3.connect":
        _sqlite_path(args[0])
        # Authorizer/PRAGMA setup happens in the sqlite3.connect wrapper inside
        # isolated_runtime: the connect/handle audit event also fires for
        # connections whose __init__ was aborted by a rejected path above, and
        # touching those raises ProgrammingError before setup can fail closed.
    elif event in ("sqlite3.enable_load_extension", "sqlite3.load_extension"):
        raise GuardError("SQLite extensions are forbidden")
    elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime",
                   "os.truncate"):
        _write_path(args[0])
        if event in ("os.remove", "os.rmdir") and len(args) > 1 and args[1] not in (-1, None):
            raise GuardError("relative directory descriptors are forbidden")
    elif event == "os.rename":
        _write_path(args[0])
        _write_path(args[1])
        if any(item not in (-1, None) for item in args[2:]):
            raise GuardError("relative directory descriptors are forbidden")
    elif event in ("os.link", "os.symlink"):
        raise GuardError("creating links is forbidden in an isolated runtime")
    elif event.startswith("socket."):
        if event == "socket.gethostname":
            # Local host-name lookup used by platform.uname() at pandas import;
            # performs no network I/O.
            return
        _ACTIVE["network_attempts"] += 1
        raise GuardError("network and DNS are disabled for B0")
    elif event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn",
                   "os.exec", "os.fork", "os.forkpty", "os.startfile", "os.startfile/2"):
        raise GuardError("subprocesses and external execution are forbidden")


def disable_dotenv() -> None:
    """Install the dotenv read/write denial before any yialpha import.

    This protection deliberately remains enabled for the process lifetime;
    disabling a context must never make a later package import read .env.
    """
    global _HOOK_INSTALLED, _DENY_DOTENV
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True
    _DENY_DOTENV = True
    dotenv = importlib.import_module("dotenv")
    # dotenv re-exports these lazily; mypy cannot see the module attributes.
    dotenv.find_dotenv = lambda *args, **kwargs: ""  # type: ignore[attr-defined]
    dotenv.load_dotenv = lambda *args, **kwargs: False  # type: ignore[attr-defined]
    dotenv.dotenv_values = lambda *args, **kwargs: {}  # type: ignore[attr-defined]
    main = importlib.import_module("dotenv.main")
    main.find_dotenv = dotenv.find_dotenv  # type: ignore[attr-defined]
    main.load_dotenv = dotenv.load_dotenv  # type: ignore[attr-defined]
    main.dotenv_values = dotenv.dotenv_values  # type: ignore[attr-defined]


def _blocked_network(*_args: Any, **_kwargs: Any) -> Any:
    if _ACTIVE is not None:
        _ACTIVE["network_attempts"] += 1
    raise GuardError("network and native curl are disabled for B0")


def _blocked_execution(*_args: Any, **_kwargs: Any) -> Any:
    raise GuardError("LLM, worker threads and execution are disabled for B0")


def _profile(frame: Any, event: str, arg: Any) -> None:
    if event != "call":
        return
    name = frame.f_globals.get("__name__", "")
    if name.startswith(("yialpha.llm_clients", "yialpha.execution",
                        "yialpha.graph.trading_graph", "yialpha.graph.propagation",
                        "yialpha.graph.setup", "yialpha.graph.reflection")):
        raise GuardError("LLM and execution entry points are disabled for B0")


@contextmanager
def isolated_runtime(
    root: str | os.PathLike[str], attempt_id: str, *, network_mode: str = "offline"
) -> Iterator[dict[str, Any]]:
    """Configure the actual project inside a scoped, offline write allowlist.

    The caller creates/verifies its cohort manifest first. Each CLI attempt
    must use a fresh process; sequential contexts exist only for offline tests.
    No supplier call or SQL is made during configuration verification.
    """
    global _ACTIVE
    if network_mode != "offline":
        raise GuardError("B0 only supports network_mode='offline'")
    if _ACTIVE is not None:
        raise GuardError("nested isolated runtimes are forbidden")
    if not isinstance(attempt_id, str) or not _SAFE_COMPONENT.fullmatch(attempt_id):
        raise GuardError("attempt_id must be one safe path component")
    guard = PathGuard(root)
    if not guard.root.is_dir():
        raise GuardError("create and verify the cohort before entering its runtime")
    paths = {
        "ledger_db_path": str(guard.resolve("ledger/portfolio.db")),
        "data_cache_dir": str(guard.resolve(f"cache/{attempt_id}/data")),
        "results_dir": str(guard.resolve(f"attempts/{attempt_id}")),
        "memory_log_path": str(guard.resolve(f"attempts/{attempt_id}/memory/trading_memory.md")),
    }
    yf_path = guard.resolve(f"cache/{attempt_id}/yfinance-internal")
    if Path(paths["data_cache_dir"]).exists() or yf_path.exists():
        raise GuardError("each attempt requires a fresh supplier cache directory")
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    disable_dotenv()
    state: dict[str, Any] = {
        "guard": guard, "ready": False, "thread_id": threading.get_ident(),
        "connections": [], "network_attempts": 0,
    }
    _ACTIVE = state
    old_profile = sys.getprofile()
    try:
        with ExitStack() as stack:
            import socket

            for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr",
                         "create_connection"):
                stack.enter_context(patch.object(socket, name, _blocked_network))
            for name in ("connect", "connect_ex", "sendto", "send", "sendall"):
                stack.enter_context(patch.object(socket.socket, name, _blocked_network))
            stack.enter_context(patch.object(threading.Thread, "start", _blocked_execution))

            def guarded_connect(*args: Any, _connect: Any = sqlite3.connect,
                                **kwargs: Any) -> sqlite3.Connection:
                # The audit hook path-checks sqlite3.connect before any file is
                # created; this wrapper only instruments fully constructed
                # connections (the connect/handle audit event cannot be used
                # because aborted connects also emit it).
                assert _ACTIVE is not None
                connection = _connect(*args, **kwargs)
                connection.execute("PRAGMA temp_store=MEMORY")
                connection.set_authorizer(_sqlite_authorizer)
                _ACTIVE["connections"].append(connection)
                return connection

            stack.enter_context(patch.object(sqlite3, "connect", guarded_connect))
            # curl_cffi uses native libcurl and bypasses Python socket.connect.
            # Patch both sync perform and the asynchronous multi-handle seam.
            curl = importlib.import_module("curl_cffi")
            stack.enter_context(patch.object(curl.Curl, "perform", _blocked_network))
            stack.enter_context(patch.object(curl.AsyncCurl, "add_handle", _blocked_network))
            curl_requests = importlib.import_module("curl_cffi.requests")
            stack.enter_context(patch.object(curl_requests.Session, "request", _blocked_network))
            stack.enter_context(patch.object(curl_requests.AsyncSession, "request", _blocked_network))

            cfg_module = importlib.import_module("yialpha.dataflows.config")
            isolated = dict(paths, analysis_only=True, live_execution_enabled=False,
                            live_execution=False, llm_cache=False, batch_concurrency=False,
                            analyst_parallel=False, portfolio_control_mode="shadow")
            config_token = cfg_module._config_var.set(cfg_module.get_config())
            stack.callback(cfg_module._config_var.reset, config_token)
            cfg_module.set_config(isolated)
            yf = importlib.import_module("yfinance")
            yf_cache = importlib.import_module("yfinance.cache")
            managers = [getattr(yf_cache, name) for name in
                        ("_TzDBManager", "_CookieDBManager", "_ISINDBManager")]
            if any(manager._db is not None for manager in managers):
                raise GuardError("yfinance cache is already initialized; use a fresh process")
            old_locations = [manager.get_location() for manager in managers]
            for manager, location in zip(managers, old_locations, strict=True):
                stack.callback(manager.set_location, location)
            for manager_name, attr in (("_TzCacheManager", "_tz_cache"),
                                       ("_CookieCacheManager", "_Cookie_cache"),
                                       ("_ISINCacheManager", "_isin_cache")):
                manager = getattr(yf_cache, manager_name)
                if hasattr(manager, attr):
                    stack.enter_context(patch.object(manager, attr, None))
            yf.set_tz_cache_location(str(yf_path))
            for manager in managers:
                if Path(manager.get_location()).resolve() != yf_path:
                    raise GuardError("yfinance internal cache redirection failed")

            ledger = importlib.import_module("yialpha.ledger.sqlite")
            # Do not reuse a caller's cached connection, even when it happens
            # to use the same filename. Remove all runtime connections on exit.
            stack.enter_context(patch.object(ledger._local, "connections", {}, create=True))
            disk_cache = importlib.import_module("yialpha.dataflows.disk_cache")
            readback = cfg_module.get_config()
            if any(readback.get(key) != value for key, value in isolated.items()):
                raise GuardError("isolated project configuration did not read back")
            if Path(ledger.ledger_db_path()) != Path(paths["ledger_db_path"]):
                raise GuardError("ledger path redirection failed")
            if Path(disk_cache.cache_base_dir()) != Path(paths["data_cache_dir"]):
                raise GuardError("supplier cache path redirection failed")
            state["ready"] = True
            sys.setprofile(_profile)
            summary = dict(isolated, yfinance_internal_cache=str(yf_path),
                           network_mode="offline", network_attempts=0)
            try:
                yield summary
            finally:
                summary["network_attempts"] = state["network_attempts"]
                state["ready"] = False
                sys.setprofile(old_profile)
                for connection in state["connections"]:
                    with suppress(sqlite3.Error):
                        connection.close()
    finally:
        sys.setprofile(old_profile)
        _ACTIVE = None
        sys.dont_write_bytecode = old_bytecode
