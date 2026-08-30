"""Regression: data-quality sentinels recorded inside langgraph nodes must
reach the caller's context after ``invoke``/``stream``.

LangGraph executes every node task inside ``copy_context().run(...)``. A
ContextVar value first bound inside a node is invisible to the caller — which
is exactly how the DEGRADED evidence chain went dead in real runs while
``tests/test_data_quality_evidence.py`` (no graph in between) kept passing.
``quality.ensure_run_context()`` binds the shared event list in the *parent*
context before the graph runs; node contexts then inherit the same list
object and appends mutate it.

These tests run a real ``StateGraph`` end to end. If langgraph ever changes
its context semantics so inherited list objects are no longer shared across
node boundaries, they fail and the accumulator must move into graph state.
"""

from __future__ import annotations

from typing import TypedDict

import pytest

from yialpha.dataflows import quality

pytestmark = pytest.mark.unit


class _State(TypedDict, total=False):
    value: str


@pytest.fixture(autouse=True)
def _clean_quality():
    quality.reset_quality()
    yield
    quality.reset_quality()


def _build_graph():
    from langgraph.graph import END, START, StateGraph

    def record_node(state: _State) -> dict:
        # Simulates a vendor tool call degrading inside an agent node.
        quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "node context")
        return {}

    builder = StateGraph(_State)
    builder.add_node("recorder", record_node)
    builder.add_edge(START, "recorder")
    builder.add_edge("recorder", END)
    return builder.compile()


def test_sentinel_inside_node_reaches_parent_after_invoke():
    quality.ensure_run_context()
    graph = _build_graph()

    graph.invoke({"value": "x"})

    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["method"] == "get_stock_data"
    assert events[0]["detail"] == "node context"
    assert quality.summarize_quality(events)["core_sentinel_count"] == 1


def test_sentinel_inside_node_reaches_parent_after_stream():
    quality.ensure_run_context()
    graph = _build_graph()

    for _chunk in graph.stream({"value": "x"}):
        pass

    assert [e["kind"] for e in quality.snapshot_quality()] == [quality.KIND_NO_DATA]


def test_ensure_run_context_installs_fresh_list_each_run():
    # A crashed prior run (no _log_state snapshot/reset) must not leak its
    # events into the next run in the same context.
    quality.ensure_run_context()
    quality.record_sentinel("m", quality.KIND_NO_DATA, "first run")

    quality.ensure_run_context()

    assert quality.snapshot_quality() == []
