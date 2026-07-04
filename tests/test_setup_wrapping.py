"""Regression tests for GraphSetup._wrap_node (T0 perf telemetry wiring).

Locks the contract that:
* with no perf tracker, every handler passes through unchanged (byte-identical
  to the historical graph);
* with a tracker, plain-function handlers get wrapped, BUT ``ToolNode``
  instances do NOT — ToolNode is a Runnable LangGraph invokes via ``.invoke()``,
  so wrapping it (which then calls ``tool_node(state)``) raises
  ``'ToolNode' object is not callable`` (caught by the --profile smoke on
  2026-07-05).
"""

from __future__ import annotations

from langgraph.prebuilt import ToolNode

from yiagents.graph.perf_telemetry import NodePerfTracker, wrap_node
from yiagents.graph.setup import GraphSetup


def _make_setup(perf_tracker=None) -> GraphSetup:
    # Stub LLMs are fine: setup_graph only captures them in closures; it never
    # invokes them at compile time (bind_tools runs at node-invoke time).
    class _StubLLM:
        pass

    return GraphSetup(
        _StubLLM(),
        _StubLLM(),
        {k: ToolNode([]) for k in ("market", "social", "news", "fundamentals")},
        conditional_logic=None,  # not exercised by _wrap_node
        perf_tracker=perf_tracker,
    )


def test_passthrough_when_no_tracker():
    """No tracker = handler returned as-is (byte-identical to no telemetry)."""
    gs = _make_setup(perf_tracker=None)

    def handler(state):
        return state

    tn = ToolNode([])
    assert gs._wrap_node(handler, "x") is handler
    assert gs._wrap_node(tn, "y") is tn


def test_plain_function_is_wrapped_when_tracker_set():
    """A plain ``def fn(state)`` handler gets wrapped (different object)."""
    gs = _make_setup(perf_tracker=NodePerfTracker())

    def handler(state):
        return {"ok": True}

    wrapped = gs._wrap_node(handler, "Market Analyst")
    assert wrapped is not handler
    # Wrapper is transparent: same return value, and records a sample.
    out = wrapped({"messages": []})
    assert out == {"ok": True}


def test_toolnode_is_not_wrapped_even_with_tracker():
    """ToolNode must pass through UNWRAPPED even when a tracker is configured.

    This is the regression guard for the --profile smoke failure
    ('ToolNode object is not callable'). ToolNode is a Runnable LangGraph
    invokes via .invoke(); wrapping it breaks that contract.
    """
    gs = _make_setup(perf_tracker=NodePerfTracker())
    tn = ToolNode([])
    assert gs._wrap_node(tn, "tools_market") is tn


def test_wrap_node_itself_is_transparent_for_plain_callables():
    """The underlying wrap_node still forwards *args/**kwargs for any callable
    that IS directly callable (the setup layer's job is to keep ToolNode out
    of it)."""
    tk = NodePerfTracker()

    class Callable:
        def __call__(self, state):
            return {"called": True}

    obj = Callable()
    wrapped = wrap_node(obj, "custom", tk)
    assert wrapped({"x": 1}) == {"called": True}
    assert tk.serialize()["nodes"]["custom"]["calls"] == 1
