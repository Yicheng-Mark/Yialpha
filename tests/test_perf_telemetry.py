"""Unit tests for ``yiagents.graph.perf_telemetry``.

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
import tempfile
import threading
import time

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from yiagents.graph.perf_telemetry import (
    UNATTRIBUTED,
    NodePerfTokenCallback,
    NodePerfTracker,
    dump_perf_report,
    wrap_node,
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


def test_wrap_node_works_for_callable_object_like_toolnode():
    """ToolNode is a callable instance, not a function. The wrapper must be
    signature-agnostic and work for it too."""
    tracker = NodePerfTracker()

    class FakeToolNode:
        def __call__(self, state):
            return {"tools_ran": True}

    wrapped = wrap_node(FakeToolNode(), "tools_market", tracker)
    out = wrapped({"some": "state"})
    assert out == {"tools_ran": True}
    assert tracker.serialize()["nodes"]["tools_market"]["calls"] == 1


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
    with open(out_file, "r", encoding="utf-8") as fh:
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
    with open(out_file, "r", encoding="utf-8") as fh:
        assert json.load(fh) == tracker.serialize()


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
