"""Unit tests for ``yiagents.graph.analyst_fanout``.

These tests exercise the per-analyst ``agent <-> tool <-> clear`` cluster
(``build_analyst_subgraph``) and the parallel fan-out node
(``create_analyst_fanout_node``) using scripted stub agents + stub tool nodes,
while keeping the REAL ``create_msg_delete`` clear-node and the REAL
``ConditionalLogic.should_continue_*`` router.

The scripted stub agent advances state across two invocations:

* call 1 -> ``AIMessage`` WITH a ``tool_call`` (routes to the tool node);
* call 2 -> ``AIMessage`` with ``tool_calls=[]`` AND writes the report key
  (routes to the clear node, which reduces ``messages`` to ``[placeholder]``).

All side-effect collection in stubs is thread-safe (the fan-out runs a
``ThreadPoolExecutor``); per-spec closures get fresh counters because
``agent_factory()`` is called once per spec at subgraph-build time.
"""

from __future__ import annotations

import threading
import time
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yiagents.graph.analyst_execution import build_analyst_execution_plan
from yiagents.graph.analyst_fanout import (
    build_analyst_subgraph,
    create_analyst_fanout_node,
)
from yiagents.graph.conditional_logic import ConditionalLogic


# ---------------------------------------------------------------------------
# State + stub helpers
# ---------------------------------------------------------------------------

def _make_minimal_state(ticker: str = "AAPL", trade_date: str = "2026-07-01"):
    """Build a minimal ``AgentState``-shaped dict for subgraph invocation.

    Includes exactly the keys ``create_msg_delete`` /
    ``get_instrument_context_from_state`` read (``messages``,
    ``company_of_interest``, ``instrument_context``, ``trade_date``) plus the
    four ``*_report`` keys so the scripted agents can write their reports.
    """
    return {
        "messages": [HumanMessage(content=ticker)],
        "company_of_interest": ticker,
        "asset_type": "stock",
        "instrument_context": (
            f"The instrument to analyze is `{ticker}`. "
            "Use this exact ticker in every tool call, report, and "
            "recommendation, preserving any exchange suffix "
            "(e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        ),
        "trade_date": trade_date,
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
    }


def _expected_placeholder(state: dict) -> str:
    """Reproduce the placeholder text emitted by ``create_msg_delete``.

    ``create_msg_delete`` and the fan-out's ``_clone_for_spec`` build the
    placeholder from the SAME helpers (``get_instrument_context_from_state`` +
    ``trade_date``), so this text is byte-identical in both places for a given
    state.
    """
    instrument_context = state["instrument_context"]
    trade_date = state["trade_date"]
    return (
        f"Proceed with your assigned analysis for this workflow. "
        f"{instrument_context} The analysis date is {trade_date}."
    )


def _make_scripted_agent_factory(
    report_key: str,
    report_text: str,
    first_invoke_seen: list | None = None,
    lock: threading.Lock | None = None,
    raise_on_first: Exception | None = None,
):
    """Return a zero-arg factory producing a scripted agent node fn.

    * call 1 -> ``AIMessage`` WITH a ``tool_call`` (routes to the tool node),
    unless ``raise_on_first`` is set (then the agent raises immediately).
    * call 2 -> ``AIMessage`` with ``tool_calls=[]`` + writes the report key
    (routes to the clear node).

    If ``first_invoke_seen`` is supplied, the stub records the message count
    and first message content observed on the FIRST invoke (thread-safe via
    ``lock``). Used to verify the placeholder rule indirectly.
    """

    def factory():
        call_count = [0]

        def agent_node(state):
            call_count[0] += 1
            if first_invoke_seen is not None and call_count[0] == 1:
                msgs = state["messages"]
                record = {
                    "num_messages": len(msgs),
                    "first_content": msgs[0].content if msgs else None,
                }
                if lock is not None:
                    with lock:
                        first_invoke_seen.append(record)
                else:
                    first_invoke_seen.append(record)
            if raise_on_first is not None and call_count[0] == 1:
                raise raise_on_first
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
    """Return a callable echoing a ``ToolMessage`` for the pending tool-call.

    Reads the tool-call id from ``state["messages"][-1].tool_calls[0]["id"]``,
    mirroring what a real ``ToolNode`` does, without requiring real tools.
    """

    def tool_node(state):
        last = state["messages"][-1]
        tool_call_id = last.tool_calls[0]["id"]
        return {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}

    return tool_node


# ---------------------------------------------------------------------------
# Tests for build_analyst_subgraph (single cluster, synchronous)
# ---------------------------------------------------------------------------

class BuildAnalystSubgraphTests(unittest.TestCase):
    def test_build_analyst_subgraph_runs_cluster(self):
        """A single market cluster: agent -> tool -> agent -> clear -> END.

        Verifies the cluster terminates, the report key is written, and the
        REAL ``create_msg_delete`` reduces ``messages`` to the single
        context-anchored placeholder.
        """
        spec = build_analyst_execution_plan(("market",)).specs[0]
        agent_factory = _make_scripted_agent_factory(
            "market_report", "STUB MARKET REPORT"
        )
        tool_node = _make_stub_tool_node()
        subgraph = build_analyst_subgraph(
            spec,
            agent_factory,
            tool_node,
            ConditionalLogic().should_continue_market,
        )

        state = _make_minimal_state()
        final_state = subgraph.invoke(state, config={"recursion_limit": 30})

        # Report written by the agent's second invoke.
        self.assertEqual(final_state["market_report"], "STUB MARKET REPORT")

        # After create_msg_delete, messages reduce to exactly the placeholder.
        final_messages = final_state["messages"]
        self.assertEqual(len(final_messages), 1)
        self.assertEqual(final_messages[0].content, _expected_placeholder(state))

    def test_build_analyst_subgraph_tool_loop_routes_via_conditional_logic(self):
        """Conditional logic routes to the tool node when tool_calls present,
        and to the clear node when absent (verified via a stub that records
        the routing by completing the loop)."""
        spec = build_analyst_execution_plan(("social",)).specs[0]
        agent_factory = _make_scripted_agent_factory(
            "sentiment_report", "STUB SENTIMENT"
        )
        tool_node = _make_stub_tool_node(content="social-data")
        subgraph = build_analyst_subgraph(
            spec,
            agent_factory,
            tool_node,
            ConditionalLogic().should_continue_social,
        )

        state = _make_minimal_state(ticker="INTC")
        final_state = subgraph.invoke(state, config={"recursion_limit": 30})

        self.assertEqual(final_state["sentiment_report"], "STUB SENTIMENT")
        self.assertEqual(len(final_state["messages"]), 1)


# ---------------------------------------------------------------------------
# Tests for create_analyst_fanout_node (parallel fan-out)
# ---------------------------------------------------------------------------

class CreateAnalystFanoutNodeTests(unittest.TestCase):
    def test_fanout_node_returns_only_reports(self):
        """The fan-out returns exactly ``{report_key: ...}`` per spec and
        NEVER writes ``messages`` (or any other key) back to the parent."""
        plan = build_analyst_execution_plan(("market", "social"))
        factories = {
            "market": _make_scripted_agent_factory("market_report", "STUB MARKET"),
            "social": _make_scripted_agent_factory("sentiment_report", "STUB SENTIMENT"),
        }
        tools = {
            "market": _make_stub_tool_node(),
            "social": _make_stub_tool_node(),
        }
        fanout = create_analyst_fanout_node(plan, factories, tools, ConditionalLogic())

        state = _make_minimal_state()
        result = fanout(state)

        self.assertEqual(
            set(result.keys()), {"market_report", "sentiment_report"}
        )
        self.assertEqual(result["market_report"], "STUB MARKET")
        self.assertEqual(result["sentiment_report"], "STUB SENTIMENT")
        self.assertNotIn("messages", result)

    def test_fanout_placeholder_rule(self):
        """specs[0] sees the parent's messages verbatim; specs[i>0] see
        ``[placeholder]``. Asserted indirectly via the stub agents recording
        what they observed on their first invoke."""
        market_seen: list[dict] = []
        market_lock = threading.Lock()
        social_seen: list[dict] = []
        social_lock = threading.Lock()
        news_seen: list[dict] = []
        news_lock = threading.Lock()

        plan = build_analyst_execution_plan(("market", "social", "news"))
        factories = {
            "market": _make_scripted_agent_factory(
                "market_report", "MKT", first_invoke_seen=market_seen, lock=market_lock
            ),
            "social": _make_scripted_agent_factory(
                "sentiment_report", "SOC", first_invoke_seen=social_seen, lock=social_lock
            ),
            "news": _make_scripted_agent_factory(
                "news_report", "NWS", first_invoke_seen=news_seen, lock=news_lock
            ),
        }
        tools = {key: _make_stub_tool_node() for key in ("market", "social", "news")}
        fanout = create_analyst_fanout_node(plan, factories, tools, ConditionalLogic())

        state = _make_minimal_state(ticker="MSFT", trade_date="2026-07-01")
        expected_ph_text = _expected_placeholder(state)
        fanout(state)

        # specs[0] (market) receives the parent's messages verbatim — the
        # original single HumanMessage(ticker).
        self.assertEqual(len(market_seen), 1)
        self.assertEqual(market_seen[0]["num_messages"], 1)
        self.assertEqual(market_seen[0]["first_content"], "MSFT")

        # specs[1] (social) and specs[2] (news) receive [placeholder].
        for seen in (social_seen, news_seen):
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]["num_messages"], 1)
            self.assertEqual(seen[0]["first_content"], expected_ph_text)

    def test_fanout_failfast_reraises(self):
        """If one spec's subgraph raises, the fan-out re-raises that exception
        (never swallows), and a sibling that would set a completion marker
        never reaches it.

        Approach: the fast-raiser sets a shared ``cancel_flag`` then raises
        immediately. The slow sibling's first invoke waits on the flag; once
        set (by the raiser), it aborts before reaching the second invoke where
        it would record completion. The small delay guarantees the raiser's
        future is processed first by ``as_completed`` (deterministic exception
        propagation)."""
        cancel_flag = threading.Event()
        completed: list[str] = []
        completed_lock = threading.Lock()

        def fast_raiser_factory():
            def agent(state):
                cancel_flag.set()
                raise RuntimeError("fast boom")

            return agent

        def slow_sibling_factory():
            call_count = [0]

            def agent(state):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Block until the fast-raiser signals (it sets the flag
                    # before raising). Returns True almost immediately.
                    cancel_flag.wait(timeout=5.0)
                    # Small delay so the raiser's future is yielded first by
                    # as_completed (the re-raised exception is deterministic).
                    time.sleep(0.1)
                    raise RuntimeError("sibling aborted")
                # Second invoke — the completion marker. Unreachable because
                # the first invoke aborts above once the raiser fires.
                with completed_lock:
                    completed.append("social")
                return {
                    "messages": [AIMessage(content="done", tool_calls=[])],
                    "sentiment_report": "SIB",
                }

            return agent

        plan = build_analyst_execution_plan(("market", "social"))
        factories = {
            "market": fast_raiser_factory,
            "social": slow_sibling_factory,
        }
        tools = {
            "market": _make_stub_tool_node(),
            "social": _make_stub_tool_node(),
        }
        fanout = create_analyst_fanout_node(plan, factories, tools, ConditionalLogic())

        state = _make_minimal_state()
        with self.assertRaisesRegex(RuntimeError, "fast boom"):
            fanout(state)

        # The sibling never reached its second invoke (where it would set the
        # completion marker).
        self.assertEqual(completed, [])

    def test_fanout_subgraphs_are_cached_across_invokes(self):
        """The fan-out builds subgraphs lazily on the first invoke and reuses
        them on subsequent invokes (verified by counting factory calls)."""
        factory_calls = {"market": 0, "social": 0}
        factory_lock = threading.Lock()

        def counting_factory(key, report_key, report_text):
            def factory():
                with factory_lock:
                    factory_calls[key] += 1

                call_count = [0]

                def agent(state):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return {
                            "messages": [
                                AIMessage(
                                    content="t",
                                    tool_calls=[
                                        {
                                            "name": "x",
                                            "args": {},
                                            "id": "call1",
                                        }
                                    ],
                                )
                            ]
                        }
                    return {
                        "messages": [AIMessage(content="d", tool_calls=[])],
                        report_key: report_text,
                    }

                return agent

            return factory

        plan = build_analyst_execution_plan(("market", "social"))
        factories = {
            "market": counting_factory("market", "market_report", "M1"),
            "social": counting_factory("social", "sentiment_report", "S1"),
        }
        tools = {"market": _make_stub_tool_node(), "social": _make_stub_tool_node()}
        fanout = create_analyst_fanout_node(plan, factories, tools, ConditionalLogic())

        state = _make_minimal_state()
        r1 = fanout(state)
        self.assertEqual(r1["market_report"], "M1")
        first_call_counts = dict(factory_calls)

        # Second invoke — factories should NOT be called again (cached).
        state2 = _make_minimal_state()
        state2["messages"] = [HumanMessage(content="GOOG")]
        r2 = fanout(state2)
        self.assertEqual(factory_calls, first_call_counts)
        self.assertEqual(r2["market_report"], "M1")


# ---------------------------------------------------------------------------
# Optional: wall-time tracker is exercised
# ---------------------------------------------------------------------------

class WallTimeTrackerTests(unittest.TestCase):
    def test_wall_time_tracker_is_called(self):
        """When a duck-typed tracker is supplied, ``mark_started`` and
        ``mark_completed`` are each called once per spec (thread-safe
        collection)."""
        calls: list[tuple[str, str]] = []
        calls_lock = threading.Lock()

        class FakeTracker:
            def mark_started(self, key):
                with calls_lock:
                    calls.append(("started", key))

            def mark_completed(self, key):
                with calls_lock:
                    calls.append(("completed", key))

        plan = build_analyst_execution_plan(("market", "social"))
        factories = {
            "market": _make_scripted_agent_factory("market_report", "M"),
            "social": _make_scripted_agent_factory("sentiment_report", "S"),
        }
        tools = {"market": _make_stub_tool_node(), "social": _make_stub_tool_node()}
        fanout = create_analyst_fanout_node(
            plan,
            factories,
            tools,
            ConditionalLogic(),
            wall_time_tracker=FakeTracker(),
        )

        fanout(_make_minimal_state())

        with calls_lock:
            started = sorted(k for evt, k in calls if evt == "started")
            completed = sorted(k for evt, k in calls if evt == "completed")

        self.assertEqual(started, ["market", "social"])
        self.assertEqual(completed, ["market", "social"])


if __name__ == "__main__":
    unittest.main()
