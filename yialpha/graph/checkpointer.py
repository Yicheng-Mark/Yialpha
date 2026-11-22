"""LangGraph checkpoint support for resumable analysis runs.

Per-ticker SQLite databases so concurrent tickers don't contend.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from yialpha.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)


def _db_path(data_dir: str | Path, ticker: str) -> Path:
    """Return the SQLite checkpoint DB path for a ticker."""
    # Reject ticker values that would escape the checkpoints directory.
    safe = safe_ticker_component(ticker).upper()
    p = Path(data_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.db"


def thread_id(ticker: str, date: str, signature: str = "") -> str:
    """Deterministic thread ID for a ticker+date pair.

    ``signature`` folds in graph-shape-affecting run choices (analysts selected,
    debate/risk depth, asset type, parallel mode) so a resume under a different
    graph can't reuse a stale checkpoint. Omitting it keeps the legacy ID.

    The ID is the first 32 hex chars of a SHA-256 (128 bits — collision-free
    for any realistic number of ticker/date/signature rows). It was previously
    truncated to 16 hex chars (64 bits); widening costs nothing and removes
    the residual cross-run collision risk of two different runs hashing to
    the same thread (which would silently resume the wrong graph state).
    """
    base = f"{ticker.upper()}:{date}"
    if signature:
        base = f"{base}:{signature}"
    return hashlib.sha256(base.encode()).hexdigest()[:32]


@contextmanager
def get_checkpointer(data_dir: str | Path, ticker: str) -> Generator[SqliteSaver, None, None]:
    """Context manager yielding a SqliteSaver backed by a per-ticker DB."""
    db = _db_path(data_dir, ticker)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def has_checkpoint(data_dir: str | Path, ticker: str, date: str, signature: str = "") -> bool:
    """Check whether a resumable checkpoint exists for ticker+date."""
    return checkpoint_step(data_dir, ticker, date, signature) is not None


def checkpoint_step(data_dir: str | Path, ticker: str, date: str, signature: str = "") -> int | None:
    """Return the step number of the latest checkpoint, or None if none exists."""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return None
    tid = thread_id(ticker, date, signature)
    with get_checkpointer(data_dir, ticker) as saver:
        config: RunnableConfig = {"configurable": {"thread_id": tid}}
        cp = saver.get_tuple(config)
        if cp is None:
            return None
        return cp.metadata.get("step")


def clear_all_checkpoints(data_dir: str | Path) -> int:
    """Remove all checkpoint DBs. Returns number of files deleted.

    Round-13: an undeletable DB no longer aborts the sweep. On Windows any
    live SQLite handle (another analysis run, the web store, antivirus)
    denies the unlink with ``PermissionError``; the old bare call crashed the
    ``--clear-checkpoints`` CLI mid-loop, leaving a half-cleared directory
    and no summary. Failures are now counted per file, logged as one WARNING,
    and the sweep continues — the return stays the honest count of successes
    (the caller prints it as a tally, the WARNING carries what was skipped).
    """
    cp_dir = Path(data_dir) / "checkpoints"
    if not cp_dir.exists():
        return 0
    dbs = list(cp_dir.glob("*.db"))
    deleted = 0
    failed: list[str] = []
    for db in dbs:
        # WAL/SHM sidecars are left orphaned when another process still held
        # the connection at unlink time (round-8): a stale sidecar never gets
        # applied to a fresh DB (salt mismatch) but it does accumulate as
        # garbage — delete alongside its db. Only AFTER the db itself is
        # gone: if the db unlink was denied the connection is still live, and
        # removing its -wal out from under it could drop committed frames.
        try:
            db.unlink()
        except (PermissionError, FileNotFoundError) as exc:
            # PermissionError: a live handle holds the file. FileNotFound:
            # raced a sibling sweep between glob and unlink. Either way the
            # file's fate is not ours — record it and keep clearing the rest.
            failed.append(f"{db.name} ({exc})")
            continue
        deleted += 1
        for suffix in ("-wal", "-shm"):
            try:
                db.with_name(db.name + suffix).unlink(missing_ok=True)
            except (PermissionError, FileNotFoundError):
                # Sidecar garbage outlives this sweep; harmless (a fresh DB
                # never applies a stale salt-mismatched WAL) and the next
                # sweep retries. Not a db-level failure — don't count it.
                pass
    if failed:
        logger.warning(
            "clear_all_checkpoints: deleted %d of %d checkpoint DB(s); could "
            "not remove %d (held open by another process or raced): %s",
            deleted, len(dbs), len(failed), "; ".join(failed),
        )
    return deleted


def clear_checkpoint(data_dir: str | Path, ticker: str, date: str, signature: str = "") -> None:
    """Remove checkpoint for a specific ticker+date by deleting the thread's rows."""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return
    tid = thread_id(ticker, date, signature)
    conn = sqlite3.connect(str(db))
    try:
        for table in ("writes", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        conn.commit()
    except sqlite3.Error as exc:
        # Benign on a freshly-created empty DB (no tables yet), but a locked
        # or corrupt DB means the thread's rows survive — a later resume would
        # silently continue from a stale checkpoint. Round-13: widened from
        # OperationalError to the sqlite3.Error base — a corrupt DB raises
        # DatabaseError ("file is not a database"), which is NOT an
        # OperationalError subclass and used to escape straight through this
        # handler into the caller.
        logger.warning(
            "checkpoint cleanup failed for %s (thread %s): %s — a later "
            "resume may continue from stale graph state",
            db.name, tid, exc,
        )
    finally:
        conn.close()
