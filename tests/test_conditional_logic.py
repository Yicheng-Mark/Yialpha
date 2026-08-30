"""Unit tests for ``yialpha.graph.conditional_logic.ConditionalLogic``.

These tests pin the debate-routing contract. The investment debate's speaker
alternation was previously driven by ``current_response.startswith("Bull")``,
which coupled routing to the LLM's output prefix; the parity-based route is
behaviourally equivalent under the normal flow (Bull first, then Bear, …) but
robust to malformed/missing prefixes. The risk-debate side already used an
explicit ``latest_speaker`` field.
"""

from __future__ import annotations

import pytest

from yialpha.graph.conditional_logic import ConditionalLogic

pytestmark = pytest.mark.unit


def _invest_state(count: int, current_response: str = "") -> dict:
    return {"investment_debate_state": {"count": count, "current_response": current_response}}


def _risk_state(count: int, latest_speaker: str = "") -> dict:
    return {"risk_debate_state": {"count": count, "latest_speaker": latest_speaker}}


class TestInvestmentDebateRouting:
    def test_count_zero_goes_to_bull(self):
        cl = ConditionalLogic(max_debate_rounds=1)
        assert cl.should_continue_debate(_invest_state(0)) == "Bull Researcher"

    def test_count_one_goes_to_bear(self):
        cl = ConditionalLogic(max_debate_rounds=1)
        assert cl.should_continue_debate(_invest_state(1)) == "Bear Researcher"

    def test_count_two_goes_to_bull(self):
        cl = ConditionalLogic(max_debate_rounds=2)
        assert cl.should_continue_debate(_invest_state(2)) == "Bull Researcher"

    def test_threshold_reached_goes_to_research_manager(self):
        # max_debate_rounds=1 -> threshold is 2*1=2.
        cl = ConditionalLogic(max_debate_rounds=1)
        assert cl.should_continue_debate(_invest_state(2)) == "Research Manager"

    def test_threshold_exceeded_goes_to_research_manager(self):
        cl = ConditionalLogic(max_debate_rounds=1)
        assert cl.should_continue_debate(_invest_state(3)) == "Research Manager"

    def test_higher_round_count_threshold(self):
        cl = ConditionalLogic(max_debate_rounds=3)
        assert cl.should_continue_debate(_invest_state(5)) == "Bear Researcher"
        assert cl.should_continue_debate(_invest_state(6)) == "Research Manager"

    def test_routing_ignores_current_response_content(self):
        """The whole point of the parity fix: routing must NOT depend on the
        LLM output prefix. Any current_response value yields the same route as
        an empty one for a given count."""
        cl = ConditionalLogic(max_debate_rounds=1)
        # count=1 -> Bear regardless of what current_response says.
        for resp in ("", "Bull Analyst: ...", "Bear Analyst: ...", "garbage", "BULL"):
            assert cl.should_continue_debate(_invest_state(1, resp)) == "Bear Researcher"
        # count=0 -> Bull regardless.
        for resp in ("", "Bull Analyst: ...", "Bear Analyst: ...", "garbage"):
            assert cl.should_continue_debate(_invest_state(0, resp)) == "Bull Researcher"

    def test_full_alternation_sequence(self):
        """A complete 2-round debate alternates Bull, Bear, Bull, Bear, then RM."""
        cl = ConditionalLogic(max_debate_rounds=2)
        seq = [cl.should_continue_debate(_invest_state(c)) for c in range(5)]
        assert seq == [
            "Bull Researcher",
            "Bear Researcher",
            "Bull Researcher",
            "Bear Researcher",
            "Research Manager",
        ]


class TestRiskDebateRouting:
    def test_threshold_reached_goes_to_portfolio_manager(self):
        cl = ConditionalLogic(max_risk_discuss_rounds=1)
        assert cl.should_continue_risk_analysis(_risk_state(3)) == "Portfolio Manager"

    def test_speaker_rotation(self):
        cl = ConditionalLogic(max_risk_discuss_rounds=2)
        assert cl.should_continue_risk_analysis(_risk_state(0, "")) == "Aggressive Analyst"
        assert cl.should_continue_risk_analysis(_risk_state(1, "Aggressive")) == "Conservative Analyst"
        assert cl.should_continue_risk_analysis(_risk_state(2, "Conservative")) == "Neutral Analyst"
        assert cl.should_continue_risk_analysis(_risk_state(3, "Neutral")) == "Aggressive Analyst"


class _Msg:
    """Minimal stand-in for a LangChain message: just needs ``tool_calls``."""

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls


class TestAnalystToolRouting:
    """The four ``should_continue_*`` analysts route to the tool node when the
    last message carries pending tool calls, otherwise to the clear node.
    Previously each method indexed ``messages[-1]`` with no empty-list guard —
    an abnormal empty ``messages`` state crashed the graph with IndexError.
    """

    @pytest.mark.parametrize("method,tool_node,clear_node", [
        ("should_continue_market", "tools_market", "Msg Clear Market"),
        ("should_continue_social", "tools_social", "Msg Clear Sentiment"),
        ("should_continue_news", "tools_news", "Msg Clear News"),
        ("should_continue_fundamentals", "tools_fundamentals", "Msg Clear Fundamentals"),
    ])
    def test_routes_to_tool_node_on_tool_calls(self, method, tool_node, clear_node):
        cl = ConditionalLogic()
        state = {"messages": [_Msg(tool_calls=[{"name": "get_price"}])]}
        assert getattr(cl, method)(state) == tool_node

    @pytest.mark.parametrize("method,clear_node", [
        ("should_continue_market", "Msg Clear Market"),
        ("should_continue_social", "Msg Clear Sentiment"),
        ("should_continue_news", "Msg Clear News"),
        ("should_continue_fundamentals", "Msg Clear Fundamentals"),
    ])
    def test_routes_to_clear_node_when_no_tool_calls(self, method, clear_node):
        cl = ConditionalLogic()
        state = {"messages": [_Msg(tool_calls=[])]}
        assert getattr(cl, method)(state) == clear_node

    @pytest.mark.parametrize("method,clear_node", [
        ("should_continue_market", "Msg Clear Market"),
        ("should_continue_social", "Msg Clear Sentiment"),
        ("should_continue_news", "Msg Clear News"),
        ("should_continue_fundamentals", "Msg Clear Fundamentals"),
    ])
    def test_empty_messages_does_not_crash(self, method, clear_node):
        """The guard: an empty messages list must fall through to the clear
        node, not raise IndexError. Pre-fix this crashed the graph."""
        cl = ConditionalLogic()
        state = {"messages": []}
        assert getattr(cl, method)(state) == clear_node


class TestDebateRoundValidation:
    """C10: 0 debate rounds must be rejected loudly, not silently accepted.

    The graph topology runs Bull Researcher / Aggressive Analyst once before
    the router fires, so ``max_*_rounds=0`` never meant "skip the debate" —
    it silently burned one LLM call per run while pretending to disable the
    stage. The validation layer now rejects it.
    """

    @pytest.mark.parametrize("kwargs", [
        {"max_debate_rounds": 0},
        {"max_risk_discuss_rounds": 0},
        {"max_debate_rounds": -1},
        {"max_risk_discuss_rounds": -2},
        {"max_debate_rounds": 1.5},
        {"max_debate_rounds": True},
        {"max_risk_discuss_rounds": "2"},
    ])
    def test_non_positive_or_non_int_rounds_rejected(self, kwargs):
        with pytest.raises(ValueError, match="integer >= 1"):
            ConditionalLogic(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"max_debate_rounds": 1},
        {"max_debate_rounds": 2},
        {"max_risk_discuss_rounds": 1},
        {"max_risk_discuss_rounds": 5},
    ])
    def test_valid_rounds_accepted(self, kwargs):
        ConditionalLogic(**kwargs)
