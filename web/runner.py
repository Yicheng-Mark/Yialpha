"""Spawn ``scripts/run_robust.py`` as a subprocess and watch its progress.

We never call ``propagate()`` in-process: a DeepSeek mid-response hang freezes
the interpreter (``ssl.read`` cannot be interrupted) and there is no in-process
recovery. ``run_robust`` runs each ticker in its own OS process behind a
wall-clock watchdog with ``taskkill /F /T`` so a hang is killed and retried.

We deliberately do NOT add an outer timeout or force-kill here — ``run_robust``
owns that contract, and a second outer watchdog would race it (double-kill,
spurious retries). The watcher only reads stdout for the coarse-grained
``attempt N/M`` banner and records the final exit code.

Success criterion is ``proc.returncode == 0``. ``run_robust`` internally
treats a run as successful only when a fresh ``complete_report.md`` appeared
(rc 0 with no new report → retry), so rc 0 is a reliable completion signal.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from web import store

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ROBUST = _PROJECT_ROOT / "scripts" / "run_robust.py"

# Matches run_robust's per-attempt stdout banner:
#   "[AAPL] ▶️ attempt 2/3 → log robust_AAPL_..._a2.log"
# Only this banner carries N/M; the success/retry lines carry a single number.
_ATTEMPT_RE = re.compile(r"attempt\s+(\d+)/(\d+)")

# Keep this many recent stdout lines for the (P1) log-tail view.
_LOG_TAIL_MAX = 50


@dataclass
class TaskState:
    task_id: str
    ticker: str
    date: str
    asset_type: str
    started_at: float
    status: str = "running"  # running | done | error
    attempt: int = 0
    max_attempts: int = 0
    report_dir: str | None = None  # newest <TICKER>_<stamp>/ after success
    error: str | None = None
    finished_at: float | None = None
    pid: int | None = None
    log_tail: list[str] = field(default_factory=list)


class TaskRegistry:
    """In-memory task registry. A server restart loses history (acceptable)."""

    def __init__(self) -> None:
        self.tasks: dict[str, TaskState] = {}
        self._running_id: str | None = None

    def is_busy(self) -> bool:
        """True iff a task is currently in the running state.

        Single-slot concurrency: VPN single connection + DeepSeek rpm + one form.
        """
        rid = self._running_id
        st = self.tasks.get(rid) if rid else None
        return st is not None and st.status == "running"

    def get(self, task_id: str) -> TaskState | None:
        return self.tasks.get(task_id)


registry = TaskRegistry()


def _build_cmd(ticker: str, date: str, asset_type: str) -> list[str]:
    cmd = [
        sys.executable,
        str(_ROBUST),
        "--tickers",
        ticker,
        "--date",
        date,
        "--workers",
        "1",
    ]
    # asset_type == "auto" → omit --asset-type entirely so run_robust's argv
    # is byte-identical to a plain CLI run (detect_asset_type auto-resolves).
    if asset_type and asset_type != "auto":
        cmd += ["--asset-type", asset_type]
    return cmd


async def spawn(ticker: str, date: str, asset_type: str) -> str:
    """Start a run_robust subprocess and register it; return the task_id.

    Caller must have already checked ``registry.is_busy()``.
    """
    task_id = uuid.uuid4().hex
    state = TaskState(
        task_id=task_id,
        ticker=ticker,
        date=date,
        asset_type=asset_type,
        started_at=time.time(),
    )
    registry.tasks[task_id] = state
    registry._running_id = task_id

    # PYTHONUTF8 so the child's emoji/中文 stdout never trips GBK (cp936).
    env = {**os.environ, "PYTHONUTF8": "1"}
    cmd = _build_cmd(ticker, date, asset_type)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_PROJECT_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    state.pid = proc.pid
    # Fire-and-forget watcher; the registry is its only state.
    asyncio.create_task(_watch(proc, state))
    return task_id


async def _watch(proc: asyncio.subprocess.Process, state: TaskState) -> None:
    """Read stdout lines for attempt banners, then record the final exit code."""
    assert proc.stdout is not None
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            m = _ATTEMPT_RE.search(text)
            if m:
                state.attempt = int(m.group(1))
                state.max_attempts = int(m.group(2))
            if len(state.log_tail) >= _LOG_TAIL_MAX:
                state.log_tail.pop(0)
            state.log_tail.append(text)
    except Exception as e:  # noqa: BLE001 -- never let the watcher raise
        state.error = f"watch error: {e!r}"

    rc = await proc.wait()
    state.finished_at = time.time()
    if rc == 0:
        state.status = "done"
        # Point at the newest complete report dir for this ticker (download).
        reports = store.list_reports(state.ticker)
        if reports:
            state.report_dir = reports[0]["dir"]
    else:
        state.status = "error"
        state.error = state.error or f"run_robust exited {rc}"
    registry._running_id = None
