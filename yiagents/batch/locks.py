"""File locks for shared resources under multi-ticker concurrency.

The batch runner fans tickers out across worker threads in ONE process, so the
contention that matters is between those threads. A process-local
``threading.Lock`` (keyed by path) fully serializes them. When the optional
``filelock`` package is installed we ALSO take an OS-level lock, so two
separate processes (e.g. a batch run plus a live analysis in another terminal)
coordinate on the same backing file. OS file locks are per-process, not
per-thread, so the threading lock is always taken first to serialize threads
within this process regardless of whether ``filelock`` is available.

This module has NO dependency on the rest of ``yiagents.batch`` (or anything in
``yiagents``) so that core modules — the memory log and the OHLCV cache — can
import it at module top level without an import cycle. ``yiagents/batch/__init__``
must stay empty for the same reason.
"""
from __future__ import annotations

import threading
from contextlib import nullcontext

try:
    from filelock import FileLock as _OSFileLock

    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover - exercised only without the extra dep
    _HAS_FILELOCK = False


# One threading.Lock per lock path, shared across every FileLock instance in
# this process so all workers racing on the same backing file serialize on one
# lock object (two FileLock instances with the same path must not each get
# their own threading.Lock, or they would not exclude each other).
_process_locks: dict[str, threading.Lock] = {}
_registry_guard = threading.Lock()


def _process_lock(path: str) -> threading.Lock:
    with _registry_guard:
        lock = _process_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _process_locks[path] = lock
        return lock


class FileLock:
    """Blocking file lock shared across threads (and processes when possible).

    Blocks until acquired — the critical sections it guards (memory-log
    read-modify-write, OHLCV cache read/write) are sub-millisecond, so
    unbounded blocking is safe and is what prevents the lost-update / torn-read
    races. Always serializes threads in this process; additionally serializes
    across processes when ``filelock`` is installed. A single instance may be
    reused across calls (acquire/release are balanced by the ``with`` block).
    """

    def __init__(self, path):
        self._path = str(path)
        self._plock = _process_lock(self._path)
        self._oslock = _OSFileLock(self._path + ".lock") if _HAS_FILELOCK else None

    def __enter__(self):
        # Process-local first: serializes threads (OS locks do not).
        self._plock.acquire()
        if self._oslock is not None:
            self._oslock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._oslock is not None:
            self._oslock.release()
        self._plock.release()
        return False


def shared_file_lock(path, enabled: bool = True):
    """Return a ``FileLock`` for ``path`` when ``enabled``, else a no-op.

    Lets callers gate locking behind a config flag without branching at every
    call site: ``with shared_file_lock(p, cfg.get("batch_memory_lock", True)):``.
    """
    return FileLock(path) if enabled else nullcontext()
