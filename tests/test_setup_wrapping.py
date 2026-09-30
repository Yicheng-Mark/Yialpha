"""Regression tests for GraphSetup._wrap_node (T0 perf telemetry wiring).

Locks the contract that:
* with no perf tracker, every handler passes through unchanged (byte-identical
  to the historical graph) — ToolNode included;
* with a tracker, plain-function handlers get the generic wrapper, and
  ``ToolNode`` instances get the Runnable-aware wrapper (round-11): LangGraph
  invokes Runnables via ``.invoke(state, config)``, never as a plain
  function, so the generic ``handler(*args)`` closure raised
  ``'ToolNode' object is not callable`` (caught by the --profile smoke on
  2026-07-05; ToolNode telemetry was excluded until round-11 closed the A/B
  speedup gate's serial-leg undercount).
"""

from __future__ import annotations

import pytest
from langgraph.prebuilt import ToolNode

from yialpha.graph.conditional_logic import ConditionalLogic
from yialpha.graph.perf_telemetry import NodePerfTracker, wrap_node
from yialpha.graph.setup import GraphSetup


def _make_setup(perf_tracker=None) -> GraphSetup:
    # Stub LLMs are fine: setup_graph never *invokes* an LLM at compile time
    # (bind_tools runs at node-invoke time). It does, however, *dereference*
    # each LLM eagerly (e.g. create_bull_researcher(self.debate_llm)), so all
    # three LLM attributes must exist — hence three stubs, not two.
    class _StubLLM:
        pass

    return GraphSetup(
        _StubLLM(),
        _StubLLM(),
        _StubLLM(),
        {k: ToolNode([]) for k in ("market", "social", "news", "fundamentals")},
        conditional_logic=ConditionalLogic(),  # not exercised by _wrap_node
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


class _ProbeToolNode(ToolNode):
    """ToolNode subclass whose invoke records the config it was handed.

    Lets the tests observe config forwarding without depending on real
    ToolNode internals (empty-tools behavior varies across langgraph
    versions).
    """

    def __init__(self):
        super().__init__([])
        self.seen_config = "unset"

    def invoke(self, state, config=None, **kwargs):
        self.seen_config = config
        return {"tools_ran": True}


def test_toolnode_is_wrapped_with_runnable_aware_wrapper():
    """ToolNode telemetry (round-11): with a tracker configured, ToolNode is
    wrapped by a DIFFERENT object that invokes via .invoke(state, config) —
    not the object itself, and not the generic handler(*args) closure whose
    call raised 'ToolNode' object is not callable on 2026-07-05."""
    gs = _make_setup(perf_tracker=NodePerfTracker())
    probe = _ProbeToolNode()
    wrapped = gs._wrap_node(probe, "tools_market")
    assert wrapped is not probe

    tracker = gs.perf_tracker
    out = wrapped({"messages": []}, {"recursion_limit": 25})
    assert out == {"tools_ran": True}
    # The wrapper forwarded the real config through to .invoke (LangGraph's
    # RunnableCallable passes it because the wrapper's 2nd param is named
    # ``config``) — callback propagation into tool execution survives.
    assert probe.seen_config == {"recursion_limit": 25}
    # And the wall time landed under the node name the A/B serial-leg sum
    # resolves from ANALYST_NODE_SPECS.
    node = tracker.serialize()["nodes"]["tools_market"]
    assert node["calls"] == 1
    assert node["wall_seconds"] >= 0.0


def test_toolnode_wrapper_handles_state_only_call():
    """If LangGraph ever calls the wrapper with state only, config=None
    forwards as invoke's own default — no crash, telemetry still recorded."""
    gs = _make_setup(perf_tracker=NodePerfTracker())
    probe = _ProbeToolNode()
    wrapped = gs._wrap_node(probe, "tools_market")

    out = wrapped({"messages": []})
    assert out == {"tools_ran": True}
    assert probe.seen_config is None
    assert gs.perf_tracker.serialize()["nodes"]["tools_market"]["calls"] == 1


def test_toolnode_wrapper_records_and_restores_on_exception():
    """Same finally-block guarantees as the generic wrapper: an exception
    raised inside invoke still records the sample and restores the previous
    active-node context."""
    gs = _make_setup(perf_tracker=NodePerfTracker())

    class _BoomToolNode(_ProbeToolNode):
        def invoke(self, state, config=None, **kwargs):
            raise ValueError("boom")

    wrapped = gs._wrap_node(_BoomToolNode(), "tools_market")
    tracker = gs.perf_tracker
    tracker.set_active_node("Market Analyst")
    with pytest.raises(ValueError):
        wrapped({"messages": []})
    node = tracker.serialize()["nodes"]["tools_market"]
    assert node["calls"] == 1
    assert tracker.get_active_node() == "Market Analyst"


def test_wrap_node_itself_is_transparent_for_plain_callables():
    """The underlying wrap_node still forwards *args/**kwargs for any callable
    that IS directly callable (Runnables belong to wrap_runnable_node)."""
    tk = NodePerfTracker()

    class Callable:
        def __call__(self, state):
            return {"called": True}

    obj = Callable()
    wrapped = wrap_node(obj, "custom", tk)
    assert wrapped({"x": 1}) == {"called": True}
    assert tk.serialize()["nodes"]["custom"]["calls"] == 1
