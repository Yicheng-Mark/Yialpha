"""Regression tests for the web analysis subprocess lifecycle."""

import asyncio

import pytest

from web import runner


@pytest.mark.unit
def test_spawn_failure_releases_busy_slot(monkeypatch):
    registry = runner.TaskRegistry()
    monkeypatch.setattr(runner, "registry", registry)

    async def fail_to_start(*args, **kwargs):
        raise OSError("process creation failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_to_start)

    with pytest.raises(OSError, match="process creation failed"):
        asyncio.run(runner.spawn("AAPL", "2026-01-10", "stock"))

    assert not registry.is_busy()
    state = next(iter(registry.tasks.values()))
    assert state.status == "error"
    assert state.finished_at is not None
    assert "failed to start analysis" in (state.error or "")


@pytest.mark.unit
def test_report_lookup_failure_after_child_exit_releases_busy_slot(monkeypatch):
    registry = runner.TaskRegistry()
    monkeypatch.setattr(runner, "registry", registry)

    state = runner.TaskState(
        task_id="task-1",
        ticker="AAPL",
        date="2026-01-10",
        asset_type="stock",
        started_at=0.0,
    )
    registry.tasks[state.task_id] = state
    registry._running_id = state.task_id

    class EmptyStdout:
        async def readline(self):
            return b""

    class SuccessfulProcess:
        stdout = EmptyStdout()

        async def wait(self):
            return 0

    def fail_lookup(_ticker):
        raise OSError("report directory unavailable")

    monkeypatch.setattr(runner.store, "list_reports", fail_lookup)

    asyncio.run(runner._watch(SuccessfulProcess(), state))

    assert not registry.is_busy()
    assert state.status == "error"
    assert "report lookup failed" in (state.error or "")


@pytest.mark.unit
class TestStartupReset:
    """Busy-state startup cleanup (P1-D2): a stale busy slot from a previous
    process is cleared at startup (fail-closed, no reconnect)."""

    def test_reset_clears_stale_busy_slot(self):
        registry = runner.TaskRegistry()
        # Simulate a task stuck in "running" from a crashed previous process.
        state = runner.TaskState(
            task_id="stale-1",
            ticker="AAPL",
            date="2026-01-10",
            asset_type="stock",
            started_at=0.0,
        )
        state.status = "running"
        registry.tasks[state.task_id] = state
        registry._running_id = state.task_id

        assert registry.is_busy()  # pre-condition: stuck busy

        registry.reset()

        assert not registry.is_busy()
        assert registry._running_id is None
        assert registry.tasks == {}

    def test_reset_is_idempotent(self):
        registry = runner.TaskRegistry()
        registry.reset()  # already empty
        assert not registry.is_busy()
        assert registry.tasks == {}

    def test_reset_does_not_reconnect_orphan_pid(self):
        """Fail-closed: reset clears state but never reattaches to an orphan.

        The orphan's PID is lost with the old process; we never try to kill or
        monitor it. It writes its result to disk and exits naturally.
        """
        registry = runner.TaskRegistry()
        state = runner.TaskState(
            task_id="orphan-1",
            ticker="NVDA",
            date="2026-01-10",
            asset_type="stock",
            started_at=0.0,
            pid=99999,  # a PID that belongs to the dead previous process
        )
        state.status = "running"
        registry.tasks[state.task_id] = state
        registry._running_id = state.task_id

        registry.reset()

        # The orphan's state is gone — we did not keep it to reattach.
        assert registry.get("orphan-1") is None
        assert not registry.is_busy()
