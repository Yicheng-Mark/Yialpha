"""Unit tests for ``yiagents.graph.conditional_logic.ConditionalLogic``.

These tests pin the debate-routing contract. The investment debate's speaker
alternation was previously driven by ``current_response.startswith("Bull")``,
which coupled routing to the LLM's output prefix; the parity-based route is
behaviourally equivalent under the normal flow (Bull first, then Bear, …) but
robust to malformed/missing prefixes. The risk-debate side already used an
explicit ``latest_speaker`` field.
"""

from __future__ import annotations

import pytest

from yiagents.graph.conditional_logic import ConditionalLogic

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
