"""Regression tests for BatchRunner's per-thread ticker log attribution.

``_TickerLogFilter`` was originally attached to the root *logger*, which
Python's logging semantics never apply to records propagated up from child
loggers — every ``yialpha.*`` module logs through its own child logger, so
the "[TICKER]" tagging never fired in real concurrent batches. The filter now
rides on the root logger's *handlers*, where propagated records do pass
through it.
"""

from __future__ import annotations

import logging
import threading

import pytest

from yialpha.batch.runner import BatchRunner


class _ListHandler(logging.Handler):
    """Capture records (and survive level checks) without touching stderr."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _FakeGraph:
    """Minimal stand-in for YiAlphaGraph used by BatchRunner."""

    def __init__(self, config):
        self.config = config


@pytest.fixture()
def isolated_root_handler():
    """Swap the root logger to exactly one capturing handler; restore after.

    The filter attaches to the root logger's handlers at BatchRunner
    construction, so the test must control that handler set precisely.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers
    saved_level = root.level
    handler = _ListHandler()
    root.handlers = [handler]
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def _emit_child(message: str, *args) -> None:
    # The realistic emission path: a yialpha.* child logger propagating to
    # the root handlers (what a worker thread's vendor/agent code does).
    logging.getLogger("yialpha.dataflows.utils").info(message, *args)


@pytest.mark.unit
def test_child_logger_record_from_worker_thread_is_tagged(isolated_root_handler):
    with BatchRunner({}, graph_factory=_FakeGraph, progress=False) as br:
        def worker():
            br._worker_ctx.ticker = "AAPL"
            _emit_child("worker message %s", "arg")

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert len(isolated_root_handler.records) == 1
    record = isolated_root_handler.records[0]
    assert record.getMessage().startswith("[AAPL] worker message")
    # The filter rewrites msg in place (args consumed) — formatting must not
    # double-interpolate or crash on the emptied args tuple.
    assert "[AAPL]" in record.getMessage()


@pytest.mark.unit
def test_untagged_records_pass_through_unmodified(isolated_root_handler):
    with BatchRunner({}, graph_factory=_FakeGraph, progress=False):
        # Main thread: no ticker in this thread's context.
        _emit_child("untagged message")

    assert len(isolated_root_handler.records) == 1
    assert isolated_root_handler.records[0].getMessage() == "untagged message"


@pytest.mark.unit
def test_filter_attached_to_handlers_not_logger(isolated_root_handler):
    root = logging.getLogger()
    with BatchRunner({}, graph_factory=_FakeGraph, progress=False) as br:
        # Logger-level filters never see propagated child records; assert the
        # handler carries the filter and the root logger itself does not.
        assert br._log_filter in isolated_root_handler.filters
        assert br._log_filter not in root.filters

    # close() removes it symmetrically from the handlers.
    assert br._log_filter is None
    assert isolated_root_handler.filters == []


@pytest.mark.unit
def test_two_runners_isolated_close(isolated_root_handler):
    # Closing one runner must not strip another live runner's filter.
    br1 = BatchRunner({}, graph_factory=_FakeGraph, progress=False)
    br2 = BatchRunner({}, graph_factory=_FakeGraph, progress=False)
    br1.close()
    assert br2._log_filter in isolated_root_handler.filters
    br2.close()
    assert isolated_root_handler.filters == []


@pytest.mark.unit
def test_tagging_is_idempotent_per_record(isolated_root_handler):
    with BatchRunner({}, graph_factory=_FakeGraph, progress=False) as br:
        br._worker_ctx.ticker = "NVDA"
        child = logging.getLogger("yialpha.agents.test")
        child.info("once %s", "x")

    assert isolated_root_handler.records[0].getMessage() == "[NVDA] once x"
