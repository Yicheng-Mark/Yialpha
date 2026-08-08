"""Unit tests for ``yiagents.graph.propagation.Propagator.create_initial_state``.

Pins the contract that the initial state carries every key that downstream
nodes read via direct ``state[key]`` indexing. This matters for checkpoint
resume: if a run crashes before (e.g.) the Research Manager writes
``investment_plan``, the resumed state must still have the key so the Trader
node degrades gracefully instead of raising ``KeyError`` and masking the
original crash.
"""

from __future__ import annotations

import pytest

from yiagents.graph.propagation import Propagator

pytestmark = pytest.mark.unit


class TestCreateInitialStateCompleteness:
    def setup_method(self):
        self.prop = Propagator()
        self.state = self.prop.create_initial_state(
            company_name="AAPL",
            trade_date="2026-01-15",
        )

    def test_company_and_date(self):
        assert self.state["company_of_interest"] == "AAPL"
        assert self.state["trade_date"] == "2026-01-15"

    def test_downstream_plan_keys_initialized(self):
        """Keys read via direct indexing by Trader / PortfolioManager / debators."""
        for key in (
            "investment_plan",
            "trader_investment_plan",
            "final_trade_decision",
            "pm_rating",
        ):
            assert key in self.state, f"missing key: {key}"
            assert self.state[key] == ""

    def test_report_keys_initialized(self):
        for key in (
            "market_report",
            "fundamentals_report",
            "sentiment_report",
            "news_report",
        ):
            assert key in self.state
            assert self.state[key] == ""

    def test_debate_states_initialized(self):
        assert self.state["investment_debate_state"]["count"] == 0
        assert self.state["investment_debate_state"]["current_response"] == ""
        assert self.state["risk_debate_state"]["count"] == 0
        assert self.state["risk_debate_state"]["latest_speaker"] == ""

    def test_optional_contexts(self):
        assert self.state["asset_type"] == "stock"
        assert self.state["instrument_context"] == ""
        assert self.state["past_context"] == ""
        assert self.state["portfolio_state"] is None

    def test_asset_type_override(self):
        state = self.prop.create_initial_state(
            company_name="BTCUSDT",
            trade_date="2026-01-15",
            asset_type="crypto",
        )
        assert state["asset_type"] == "crypto"
