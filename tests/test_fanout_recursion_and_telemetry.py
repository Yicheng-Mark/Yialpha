"""C2 + C6 regressions for ``yialpha.graph.analyst_fanout``.

C2 — recursion-budget parity: the per-analyst subgraph budget must be derived
from the serial whole-graph budget (``max_recur_limit``), not a fixed constant,
so a deep tool loop does not trip ``GraphRecursionError`` in serial mode while
succeeding in parallel mode (identical-decision-distribution iron law).

C6 — telemetry attribution: worker threads must set the perf tracker's
thread-local ``active_node`` to the analyst's serial node name while the
subgraph runs, so parallel-leg LLM tokens attribute to the analyst instead of
``_unattributed_``.

Hermetic: scripted stub agents + stub tool nodes, no LLM, no network.
"""

from __future__ import annotations

import threading

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yialpha.dataflows.config import set_config
from yialpha.graph.analyst_execution import build_analyst_execution_plan
from yialpha.graph.analyst_fanout import (
    create_analyst_fanout_node,
    derive_subgraph_recursion_limit,
)
from yialpha.graph.conditional_logic import ConditionalLogic
from yialpha.graph.perf_telemetry import NodePerfTracker

# ---------------------------------------------------------------------------
# Shared stub helpers (same shape as tests/test_analyst_fanout.py)
# ---------------------------------------------------------------------------

def _make_minimal_state(ticker: str = "AAPL", trade_date: str = "2026-07-01"):
    return {
        "messages": [HumanMessage(content=ticker)],
        "company_of_interest": ticker,
        "asset_type": "stock",
        "instrument_context": f"The instrument to analyze is `{ticker}`.",
        "trade_date": trade_date,
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
    }


def _make_scripted_agent_factory(
    report_key: str,
    report_text: str,
    on_invoke=None,
):
    """Scripted agent: tool-call first, clean report second.

    ``on_invoke(active_node, call_idx)`` is called (if given) inside the node
    so tests can observe the thread-local active-node context mid-flight.
    """

    def factory():
        call_count = [0]

        def agent_node(state):
            call_count[0] += 1
            if on_invoke is not None:
                on_invoke(call_count[0])
            if call_count[0] == 1:
                return {
                    "messages": [
                        AIMessage(
                            content="thinking",
                            tool_calls=[
                                {
                                    "name": "get_stock_data",
                                    "args": {"ticker": "AAPL"},
                                    "id": "call1",
                                }
                            ],
                        )
                    ]
                }
            return {
                "messages": [AIMessage(content="done", tool_calls=[])],
                report_key: report_text,
            }

        return agent_node

    return factory


def _make_stub_tool_node(content: str = "result"):
    def tool_node(state):
        last = state["messages"][-1]
        tool_call_id = last.tool_calls[0]["id"]
        return {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}

    return tool_node


def _build_fanout(plan_keys, *, on_invoke=None, **kwargs):
    plan = build_analyst_execution_plan(plan_keys)
    factories = {
        spec.key: _make_scripted_agent_factory(spec.report_key, f"R:{spec.key}",
                                               on_invoke=on_invoke)
        for spec in plan.specs
    }
    tools = {spec.key: _make_stub_tool_node() for spec in plan.specs}
    cond = ConditionalLogic()
    return create_analyst_fanout_node(plan, factories, tools, cond, **kwargs)


# ---------------------------------------------------------------------------
# C2: derive_subgraph_recursion_limit
# ---------------------------------------------------------------------------

class TestDeriveSubgraphRecursionLimit:
    def test_default_budget_four_analysts(self):
        # 100 // 4 = 25: parallel worst case 4*25 = 100 ~= serial pool of 100.
        assert derive_subgraph_recursion_limit(100, 4) == 25

    def test_single_analyst_gets_full_budget(self):
        assert derive_subgraph_recursion_limit(100, 1) == 100

    def test_floor_protects_tiny_budgets(self):
        # 40 // 4 = 10 (at floor); 8 // 4 = 2 -> clamped up to 10 so one
        # agent -> tool -> agent -> clear cycle can still finish.
        assert derive_subgraph_recursion_limit(40, 4) == 10
        assert derive_subgraph_recursion_limit(8, 4) == 10

    def test_zero_or_negative_specs_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            derive_subgraph_recursion_limit(100, 0)

    def test_fanout_default_derives_from_config_serial_budget(self):
        # The default (recursion_limit=None) must read the thread-local
        # config's max_recur_limit — the same budget the serial propagate()
        # path uses — instead of a hardcoded constant.
        set_config({"max_recur_limit": 48})
        fanout = _build_fanout(("market", "social", "news", "fundamentals"))
        assert fanout.resolved_recursion_limit == 12

        set_config({"max_recur_limit": 100})
        fanout = _build_fanout(("market", "social", "news", "fundamentals"))
        assert fanout.resolved_recursion_limit == 25

    def test_explicit_recursion_limit_wins(self):
        set_config({"max_recur_limit": 100})
        fanout = _build_fanout(
            ("market", "social"), recursion_limit=7,
        )
        assert fanout.resolved_recursion_limit == 7


# ---------------------------------------------------------------------------
# C6: per-analyst token attribution in worker threads
# ---------------------------------------------------------------------------

class TestFanoutTelemetryAttribution:
    def test_worker_threads_set_active_node_to_serial_node_name(self):
        seen: dict[str, str] = {}
        seen_lock = threading.Lock()
        plan = build_analyst_execution_plan(
            ("market", "social", "news", "fundamentals")
        )

        def on_invoke_factory(spec_key):
            def on_invoke(_call_idx):
                # Read the thread-local active node INSIDE the worker thread.
                value = tracker.get_active_node()
                with seen_lock:
                    seen[spec_key] = value

            return on_invoke

        factories = {}
        for spec in plan.specs:
            hook = on_invoke_factory(spec.key)

            def factory(hook=hook, spec=spec):
                return _make_scripted_agent_factory(
                    spec.report_key, f"R:{spec.key}", on_invoke=hook,
                )()

            factories[spec.key] = factory
        tools = {spec.key: _make_stub_tool_node() for spec in plan.specs}

        tracker = NodePerfTracker()
        fanout = create_analyst_fanout_node(
            plan, factories, tools, ConditionalLogic(), perf_tracker=tracker,
        )

        result = fanout(_make_minimal_state())
        assert set(result) == {
            "market_report", "sentiment_report", "news_report", "fundamentals_report",
        }
        # Every analyst's agent node observed its own serial node name as the
        # active node (not None / _unattributed_).
        assert seen == {
            "market": "Market Analyst",
            "social": "Sentiment Analyst",
            "news": "News Analyst",
            "fundamentals": "Fundamentals Analyst",
        }
        # The main thread's context was never touched by the worker threads.
        assert tracker.get_active_node() is None

    def test_no_tracker_is_inert(self):
        # perf_tracker=None (telemetry off) must not change behaviour.
        fanout = _build_fanout(("market", "news"))
        result = fanout(_make_minimal_state())
        assert result["market_report"] == "R:market"
        assert result["news_report"] == "R:news"

    def test_tokens_attributed_to_analyst_nodes(self):
        """End-to-end: record_tokens inside the worker lands on the analyst."""
        plan = build_analyst_execution_plan(("market",))
        tracker = NodePerfTracker()

        def factory():
            def agent_node(state):
                # Simulate the LangChain callback firing on this thread while
                # the subgraph (and its active-node context) is live.
                tracker.record_tokens(
                    tracker.get_active_node() or "_unattributed_",
                    input_tokens=10, output_tokens=5,
                )
                return {
                    "messages": [AIMessage(content="done", tool_calls=[])],
                    "market_report": "r",
                }

            return agent_node

        # Direct-call routing needs no tool round: agent -> clear -> END.
        fanout = create_analyst_fanout_node(
            plan, {"market": factory},
            {"market": _make_stub_tool_node()},
            ConditionalLogic(), perf_tracker=tracker,
        )
        fanout(_make_minimal_state())
        snapshot = tracker.serialize()
        assert snapshot["nodes"]["Market Analyst"]["tokens_in"] == 10
        assert snapshot["nodes"]["Market Analyst"]["tokens_out"] == 5
        assert "_unattributed_" not in snapshot["nodes"]
