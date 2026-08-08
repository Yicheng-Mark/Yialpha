"""Unit tests for ``yiagents.agents.trader.trader`` node-level contract.

``test_structured_agents.py`` already covers the structured-output happy path,
the free-text fallback, and ``render_trader_proposal``. These tests focus on
the *node* contract that was previously unasserted: the returned dict's shape
(``sender``/``trader_investment_plan`` keys), multi-action round-trips (HOLD /
SELL, not just BUY), and robustness against the empty-string state fields a
checkpoint-resumed run can feed in (see propagation.create_initial_state).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yiagents.agents.schemas import TraderAction, TraderProposal
from yiagents.agents.trader.trader import create_trader

pytestmark = pytest.mark.unit


def _structured_llm(proposal: TraderProposal, captured: dict | None = None) -> MagicMock:
    """An LLM mock whose structured binding returns ``proposal`` and captures
    the prompt messages into ``captured`` (if given)."""
    structured = MagicMock()
    if captured is None:
        structured.invoke.return_value = proposal
    else:
        structured.invoke.side_effect = lambda prompt: (
            captured.__setitem__("prompt", prompt) or proposal
        )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _state(company="NVDA", plan="**Recommendation**: Buy\n..."):
    return {"company_of_interest": company, "investment_plan": plan}


class TestTraderNodeShape:
    """The node returns a dict with exactly the three keys downstream nodes
    read via state[key] indexing (sender / trader_investment_plan / messages)."""

    @pytest.mark.parametrize("action", list(TraderAction))
    def test_every_action_round_trips(self, action):
        proposal = TraderProposal(action=action, reasoning="r")
        trader = create_trader(_structured_llm(proposal))
        result = trader(_state())
        assert result["sender"] == "Trader"
        assert isinstance(result["trader_investment_plan"], str)
        # render_trader_proposal upper-cases the action value in the final line.
        assert f"FINAL TRANSACTION PROPOSAL: **{action.value.upper()}**" in result["trader_investment_plan"]
        assert len(result["messages"]) == 1

    def test_sender_is_trader_regardless_of_binding_name(self):
        # create_trader binds name="Trader"; the node must echo it, not a stale
        # default. This pins the contract the risk-debate / PM nodes rely on.
        proposal = TraderProposal(action=TraderAction.HOLD, reasoning="wait")
        trader = create_trader(_structured_llm(proposal))
        assert trader(_state())["sender"] == "Trader"


class TestTraderEmptyStateRobustness:
    """A checkpoint-resumed run can seed ``investment_plan`` to "" (propagation
    does this now). The node must not crash on empty fields."""

    def test_empty_investment_plan_does_not_crash(self):
        proposal = TraderProposal(action=TraderAction.HOLD, reasoning="no plan")
        trader = create_trader(_structured_llm(proposal))
        result = trader(_state(plan=""))
        assert result["trader_investment_plan"]  # still rendered something

    def test_missing_asset_type_defaults_without_crash(self):
        # state has no asset_type key; get_instrument_context_from_state must
        # tolerate its absence (the node should not KeyError).
        proposal = TraderProposal(action=TraderAction.BUY, reasoning="x")
        trader = create_trader(_structured_llm(proposal))
        result = trader({"company_of_interest": "AAPL", "investment_plan": "buy"})
        assert result["sender"] == "Trader"


class TestTraderPromptConstruction:
    def test_company_and_plan_appear_in_user_message(self):
        captured: dict = {}
        proposal = TraderProposal(action=TraderAction.BUY, reasoning="x")
        trader = create_trader(_structured_llm(proposal, captured))
        trader(_state(company="MSFT", plan="Accumulate on dips."))
        user_msg = captured["prompt"][-1]["content"]
        assert "MSFT" in user_msg
        assert "Accumulate on dips." in user_msg

    def test_instrument_context_propagates_when_present(self):
        captured: dict = {}
        proposal = TraderProposal(action=TraderAction.BUY, reasoning="x")
        trader = create_trader(_structured_llm(proposal, captured))
        state = _state()
        state["instrument_context"] = "PRODUCT: perp futures"
        trader(state)
        assert "PRODUCT: perp futures" in captured["prompt"][-1]["content"]
