"""Unit tests for ``yialpha.graph.perf_telemetry``.

Covers:
1. ``wrap_node`` transparency (return value + exception propagation).
2. Wall-time + call-count accumulation, including multiple calls.
3. ``NodePerfTokenCallback.on_llm_end`` token attribution to the active node,
   and fall-through to ``_unattributed_`` when no node is active.
4. Thread-safety smoke under a real ``ThreadPoolExecutor`` (no lost updates).
5. ``dump_perf_report`` writes JSON matching ``serialize()``.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from yialpha.graph.perf_telemetry import (
    UNATTRIBUTED,
    NodePerfTokenCallback,
    NodePerfTracker,
    dump_perf_report,
    wrap_node,
    wrap_runnable_node,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _llm_result(usage: dict | None = None) -> LLMResult:
    """Build a minimal LLMResult whose only generation carries an AIMessage
    with the given ``usage_metadata``."""
    message = AIMessage(content="ok")
    if usage is not None:
        # langchain_core AIMessage accepts usage_metadata via constructor in
        # some versions; setting the attribute directly works across versions.
        message.usage_metadata = usage
    generation = ChatGeneration(message=message)
    return LLMResult(generations=[[generation]])


# --------------------------------------------------------------------------- #
# 1. wrap_node transparency
# --------------------------------------------------------------------------- #
def test_wrap_node_returns_same_value():
    tracker = NodePerfTracker()

    def handler(state):
        return {"doubled": state["x"] * 2}

    wrapped = wrap_node(handler, "double_node", tracker)
    result = wrapped({"x": 21})
    assert result == {"doubled": 42}


def test_wrap_node_propagates_exceptions_and_still_records_time():
    tracker = NodePerfTracker()

    def handler(state):
        raise ValueError("boom")

    wrapped = wrap_node(handler, "faulty", tracker)
    with pytest.raises(ValueError):
        wrapped({})

    # Even on failure the timing sample is recorded.
    node = tracker.serialize()["nodes"]["faulty"]
    assert node["calls"] == 1
    assert node["wall_seconds"] >= 0.0


def test_wrap_node_restores_previous_active_node_on_exception():
    tracker = None
    tracker = NodePerfTracker()
    tracker.set_active_node("outer")

    def handler(state):
        raise RuntimeError("nope")

    wrapped = wrap_node(handler, "inner", tracker)
    with pytest.raises(RuntimeError):
        wrapped({})
    # After the exception, the outer context must be restored.
    assert tracker.get_active_node() == "outer"


def test_wrap_node_works_for_callable_objects():
    """Callable instances (``__call__`` defined) wrap fine under the generic
    wrapper. langgraph Runnables such as ToolNode are NOT plain callables —
    see wrap_runnable_node tests below (round-11)."""
    tracker = NodePerfTracker()

    class FakeCallableNode:
        def __call__(self, state):
            return {"tools_ran": True}

    wrapped = wrap_node(FakeCallableNode(), "tools_market", tracker)
    out = wrapped({"some": "state"})
    assert out == {"tools_ran": True}
    assert tracker.serialize()["nodes"]["tools_market"]["calls"] == 1


# --------------------------------------------------------------------------- #
# 1b. wrap_runnable_node (round-11: ToolNode telemetry)
# --------------------------------------------------------------------------- #
class _RecordingRunnable:
    """Minimal stand-in for a langgraph Runnable (e.g. ToolNode): records the
    config it was invoked with; raises when asked to."""

    def __init__(self, *, boom: bool = False) -> None:
        self.seen_config = "unset"
        self.boom = boom

    def invoke(self, state, config=None, **kwargs):
        self.seen_config = config
        if self.boom:
            raise ValueError("boom")
        return {"invoked": True}


def test_wrap_runnable_node_returns_value_and_records():
    tracker = NodePerfTracker()
    runnable = _RecordingRunnable()
    wrapped = wrap_runnable_node(runnable, "tools_market", tracker)

    out = wrapped({"messages": []}, {"recursion_limit": 25})
    assert out == {"invoked": True}
    node = tracker.serialize()["nodes"]["tools_market"]
    assert node["calls"] == 1
    assert node["wall_seconds"] >= 0.0


def test_wrap_runnable_node_forwards_config_to_invoke():
    """The wrapper's 2nd parameter is named ``config`` so LangGraph's
    RunnableCallable hands it the real run config; it must reach
    ``invoke`` (callback propagation into tool execution)."""
    tracker = NodePerfTracker()
    runnable = _RecordingRunnable()
    wrapped = wrap_runnable_node(runnable, "tools_market", tracker)

    cfg = {"recursion_limit": 25, "tags": ["x"]}
    wrapped({"messages": []}, cfg)
    assert runnable.seen_config is cfg


def test_wrap_runnable_node_state_only_call_defaults_config():
    """Called with state only: config=None forwards as invoke's own default."""
    tracker = NodePerfTracker()
    runnable = _RecordingRunnable()
    wrapped = wrap_runnable_node(runnable, "tools_market", tracker)

    wrapped({"messages": []})
    assert runnable.seen_config is None
    assert tracker.serialize()["nodes"]["tools_market"]["calls"] == 1


def test_wrap_runnable_node_propagates_exceptions_and_restores_context():
    tracker = NodePerfTracker()
    tracker.set_active_node("Market Analyst")
    wrapped = wrap_runnable_node(
        _RecordingRunnable(boom=True), "tools_market", tracker
    )
    with pytest.raises(ValueError):
        wrapped({"messages": []})
    # Timing sample booked despite the failure, context restored.
    assert tracker.serialize()["nodes"]["tools_market"]["calls"] == 1
    assert tracker.get_active_node() == "Market Analyst"


def test_wrap_runnable_node_sets_active_node_during_invoke():
    """While the runnable runs, the tracker's active node is the tool node —
    so any LLM call a tool triggers mid-flight attributes to it."""
    tracker = NodePerfTracker()

    class _NestedRunnable:
        def invoke(self, state, config=None, **kwargs):
            tracker.record_tokens("probe", input_tokens=1)
            return tracker.get_active_node(), True

    wrapped = wrap_runnable_node(_NestedRunnable(), "tools_market", tracker)
    active_inside, _ = wrapped({"messages": []})
    assert active_inside == "tools_market"


def test_wrap_runnable_node_compiles_without_config_type_warning():
    """Round-11 remediation: the wrapper's ``config`` parameter is typed
    ``Optional[RunnableConfig]`` — compiling a graph that carries the
    wrapped node must not emit langgraph's "'config' parameter should be
    typed as 'RunnableConfig'" UserWarning (the previous ``config: Any``
    annotation fired it once per serial ToolNode at compile time; the
    spelling matters because this module stringifies annotations via
    ``from __future__ import annotations`` and langgraph only accepts the
    ``Optional[RunnableConfig]`` string form, not ``RunnableConfig | None``)."""
    import warnings

    from langgraph.graph import END, START, StateGraph

    tracker = NodePerfTracker()
    wrapped = wrap_runnable_node(_RecordingRunnable(), "tools_market", tracker)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph = StateGraph(dict)
        graph.add_node("tools_market", wrapped)
        graph.add_edge(START, "tools_market")
        graph.add_edge("tools_market", END)
        graph.compile()
    offending = [
        w for w in caught
        if "should be typed as 'RunnableConfig'" in str(w.message)
    ]
    assert offending == []


# --------------------------------------------------------------------------- #
# 2. Wall-time + call accumulation
# --------------------------------------------------------------------------- #
def test_single_call_records_non_negative_wall_time_and_one_call():
    tracker = NodePerfTracker()

    def handler(state):
        time.sleep(0.005)
        return state

    wrapped = wrap_node(handler, "sleeper", tracker)
    wrapped({})

    node = tracker.serialize()["nodes"]["sleeper"]
    assert isinstance(node["wall_seconds"], float)
    assert node["wall_seconds"] >= 0.0
    assert node["calls"] == 1


def test_two_calls_accumulate_wall_time_and_count():
    tracker = NodePerfTracker()

    def handler(state):
        time.sleep(0.01)
        return state

    wrapped = wrap_node(handler, "sleeper", tracker)
    wrapped({})
    wrapped({})

    node = tracker.serialize()["nodes"]["sleeper"]
    assert node["calls"] == 2
    # Two ~10ms sleeps should sum to something >= a single sleep; we only
    # assert non-decreasing to keep the test deterministic across platforms.
    assert node["wall_seconds"] >= 0.0


# --------------------------------------------------------------------------- #
# 3. Token attribution
# --------------------------------------------------------------------------- #
def test_token_callback_attributes_to_active_node():
    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)

    tracker.set_active_node("Market Analyst")
    try:
        cb.on_llm_end(
            _llm_result(
                {
                    "input_tokens": 120,
                    "output_tokens": 80,
                    "output_token_details": {"reasoning": 17},
                }
            )
        )
    finally:
        tracker.set_active_node(None)

    node = tracker.serialize()["nodes"]["Market Analyst"]
    assert node["tokens_in"] == 120
    assert node["tokens_out"] == 80
    assert node["tokens_reasoning"] == 17


def test_token_callback_falls_back_to_unattributed_when_no_active_node():
    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)

    # No active node set on this thread.
    assert tracker.get_active_node() is None
    cb.on_llm_end(
        _llm_result(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "output_token_details": {"reasoning": 1},
            }
        )
    )

    nodes = tracker.serialize()["nodes"]
    assert UNATTRIBUTED in nodes
    assert nodes[UNATTRIBUTED]["tokens_in"] == 10
    assert nodes[UNATTRIBUTED]["tokens_out"] == 5
    assert nodes[UNATTRIBUTED]["tokens_reasoning"] == 1


def test_token_callback_noop_when_no_usage_metadata():
    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)
    tracker.set_active_node("X")
    cb.on_llm_end(_llm_result(usage=None))
    # No tokens recorded -> node slot for X should not even exist.
    assert "X" not in tracker.serialize()["nodes"]


# --------------------------------------------------------------------------- #
# 4. Thread-safety smoke
# --------------------------------------------------------------------------- #
def test_concurrent_record_does_not_lose_updates():
    """N threads each record ``per_thread`` samples on distinct node names;
    the totals must equal the expected sum exactly."""
    tracker = NodePerfTracker()
    n_threads = 8
    per_thread = 200
    expected_calls = n_threads * per_thread
    expected_wall = float(n_threads * per_thread)  # each sample = 1.0s

    def worker(tid):
        name = f"node_{tid}"
        for _ in range(per_thread):
            tracker.record(name, 1.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = tracker.serialize()
    assert sum(n["calls"] for n in data["nodes"].values()) == expected_calls
    assert data["totals"]["wall_seconds"] == pytest.approx(expected_wall)


def test_concurrent_record_tokens_does_not_lose_updates():
    """Token accumulation must also be lossless under contention on a SHARED
    node name (worst case for the lock)."""
    tracker = NodePerfTracker()
    n_threads = 8
    per_thread = 200
    per_call_in = 3
    per_call_out = 2
    expected = n_threads * per_thread

    def worker():
        for _ in range(per_thread):
            tracker.record_tokens(
                "shared",
                input_tokens=per_call_in,
                output_tokens=per_call_out,
            )

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    node = tracker.serialize()["nodes"]["shared"]
    assert node["tokens_in"] == expected * per_call_in
    assert node["tokens_out"] == expected * per_call_out


def test_thread_local_active_node_is_isolated():
    """A worker thread setting active_node must not leak to the main thread."""
    tracker = NodePerfTracker()
    barrier = threading.Barrier(2)

    def worker():
        tracker.set_active_node("worker_node")
        barrier.wait()  # let main thread observe
        barrier.wait()  # wait for main to finish observing

    t = threading.Thread(target=worker)
    t.start()
    barrier.wait()
    # Main thread's active_node is untouched by the worker's write.
    assert tracker.get_active_node() is None
    barrier.wait()
    t.join()


# --------------------------------------------------------------------------- #
# 5. dump_perf_report
# --------------------------------------------------------------------------- #
def test_dump_perf_report_writes_valid_json_matching_serialize(tmp_path):
    tracker = NodePerfTracker()
    tracker.record("A", 1.5)
    tracker.record("A", 0.5)
    tracker.record_tokens("A", input_tokens=10, output_tokens=4)

    out_file = tmp_path / "sub" / "node_perf_2026-07-05.json"
    dump_perf_report(tracker, out_file)

    assert out_file.exists()
    with open(out_file, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == tracker.serialize()
    # Spot-check structure & totals.
    assert on_disk["nodes"]["A"]["calls"] == 2
    assert on_disk["nodes"]["A"]["wall_seconds"] == pytest.approx(2.0)
    assert on_disk["totals"]["tokens_in"] == 10
    # Parent dirs were created.
    assert (tmp_path / "sub").is_dir()


def test_dump_perf_report_accepts_string_path(tmp_path):
    tracker = NodePerfTracker()
    tracker.record("n", 0.1)
    out_file = str(tmp_path / "out.json")
    dump_perf_report(tracker, out_file)
    assert os.path.exists(out_file)
    with open(out_file, encoding="utf-8") as fh:
        assert json.load(fh) == tracker.serialize()


def test_dump_perf_report_atomic_no_tmp_leftover(tmp_path):
    # Atomic write (tmp + os.replace) must not leave a sibling .tmp file behind
    # on success. A leftover .tmp would accumulate run-over-run and a truncated
    # final file (the failure mode this guards against) can't be produced
    # deterministically in-process, so we assert the clean-path contract.
    tracker = NodePerfTracker()
    tracker.record("A", 1.0)
    out_file = tmp_path / "node_perf_2026-07-06.json"
    dump_perf_report(tracker, out_file)
    assert out_file.exists()
    assert not (tmp_path / "node_perf_2026-07-06.json.tmp").exists()
    # And the file is valid JSON (not half-written).
    with open(out_file, encoding="utf-8") as fh:
        assert "nodes" in json.load(fh)


# --------------------------------------------------------------------------- #
# Bonus: serialize() shape sanity
# --------------------------------------------------------------------------- #
def test_serialize_shape_for_empty_tracker():
    data = NodePerfTracker().serialize()
    assert data == {
        "nodes": {},
        "totals": {
            "wall_seconds": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "tokens_reasoning": 0,
        },
    }


def test_token_callback_reasoning_only_generations_skip_llm_output_fallback():
    """Round-10: generations carrying ONLY reasoning tokens (input=0,
    output=0, reasoning=N — some providers report just that) plus a gateway
    that double-writes usage on llm_output must count N ONCE. The old
    input==0-and-output==0 fallback test consulted llm_output anyway and
    added N a second time."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)

    message = AIMessage(content="thought")
    message.usage_metadata = {
        "input_tokens": 0,
        "output_tokens": 0,
        "output_token_details": {"reasoning": 17},
    }
    result = LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "output_token_details": {"reasoning": 17},
            }
        },
    )

    tracker.set_active_node("Market Analyst")
    try:
        cb.on_llm_end(result)
    finally:
        tracker.set_active_node(None)

    node = tracker.serialize()["nodes"]["Market Analyst"]
    assert node["tokens_reasoning"] == 17  # not 34


def test_token_callback_reasoning_only_metadata_recovers_io_from_llm_output():
    """Round-10 (G6 follow-up): the gateway shape where generations carry
    ONLY reasoning (usage_metadata = {input:0, output:0, reasoning:N}) while
    llm_output.token_usage holds the FULL counts must recover input/output
    from llm_output AND keep reasoning from the generations — the strict
    all-zero fallback gate recorded input/output as 0 here, while per-field
    merge preserves both the G6 no-double-count fix and the I/O counts."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)

    message = AIMessage(content="thought")
    message.usage_metadata = {
        "input_tokens": 0,
        "output_tokens": 0,
        "output_token_details": {"reasoning": 17},
    }
    result = LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={
            "token_usage": {
                "input_tokens": 120,
                "output_tokens": 80,
                "output_token_details": {"reasoning": 17},
            }
        },
    )

    tracker.set_active_node("Market Analyst")
    try:
        cb.on_llm_end(result)
    finally:
        tracker.set_active_node(None)

    node = tracker.serialize()["nodes"]["Market Analyst"]
    # Input/output recovered from llm_output (previously recorded as 0)...
    assert node["tokens_in"] == 120
    assert node["tokens_out"] == 80
    # ...while reasoning stays the generations' count — not added on top of
    # llm_output's copy (17, not 34): the G6 double-count fix survives.
    assert node["tokens_reasoning"] == 17


def test_token_callback_partial_generation_usage_not_overwritten_by_llm_output():
    """Per-field means per-field: generations carrying real input/output keep
    them even when llm_output.double-writes different values — the fallback
    fills ONLY still-zero fields, never overrides a generation-provided one."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    tracker = NodePerfTracker()
    cb = NodePerfTokenCallback(tracker)

    message = AIMessage(content="ok")
    message.usage_metadata = {
        "input_tokens": 7,
        "output_tokens": 0,
        "output_token_details": {"reasoning": 3},
    }
    result = LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={
            "token_usage": {
                "input_tokens": 999,
                "output_tokens": 42,
                "output_token_details": {"reasoning": 9},
            }
        },
    )

    tracker.set_active_node("Trader")
    try:
        cb.on_llm_end(result)
    finally:
        tracker.set_active_node(None)

    node = tracker.serialize()["nodes"]["Trader"]
    # Generation-provided input/reasoning win; only the zero output is filled.
    assert node["tokens_in"] == 7
    assert node["tokens_out"] == 42
    assert node["tokens_reasoning"] == 3
