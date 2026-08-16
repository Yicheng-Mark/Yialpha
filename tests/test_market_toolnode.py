"""Wiring guard: every tool an analyst can bind must be executable in its ToolNode.

The market analyst is prompt-instructed to call ``get_verified_market_snapshot``
and (2026-08-15 TA expansion) to cite ``get_indicators_weekly`` /
``get_support_resistance`` / ``get_volume_features`` / ``get_candlestick_patterns``
/ ``get_relative_strength``. Commit 6c90703 added those five to the analyst's
bind list and prompt but NOT to the market ToolNode — every stock run's
mandated price-structure call then died with "not a valid tool" and the whole
2134-test suite stayed green because nothing compared the bind list against the
ToolNode registry (P0, round-5 audit 2026-08-16).

This module pins the registry side of that contract. The bind-list side is
pinned in test_crypto_spot_mode.py / test_crypto_perp_mode.py; keep all three
in sync when an analyst's tool list changes.
"""
from types import SimpleNamespace

import pytest

from yiagents.graph.trading_graph import YiAgentsGraph


class _StubLLM:
    """Duck-typed LLM for make_pot_compute_tool (PotAnalyzer stores it only)."""


@pytest.mark.unit
def test_market_toolnode_can_execute_verified_snapshot():
    nodes = _tool_nodes()
    market_tools = set(nodes["market"].tools_by_name)
    assert "get_verified_market_snapshot" in market_tools, (
        "get_verified_market_snapshot is bound to the market analyst but not "
        "registered in the market ToolNode, so the model's call fails."
    )
    # the other core market tools must remain too
    assert {"get_stock_data", "get_indicators"} <= market_tools


@pytest.mark.unit
def test_market_toolnode_registers_prompt_mandated_expansion_tools():
    """The five TA-expansion tools are prompt-MANDATED for stock runs."""
    market_tools = set(_tool_nodes()["market"].tools_by_name)
    missing = {
        "get_indicators_weekly",
        "get_support_resistance",
        "get_volume_features",
        "get_candlestick_patterns",
        "get_relative_strength",
    } - market_tools
    assert not missing, (
        f"Prompt-mandated tools missing from the market ToolNode: {missing}. "
        "The analyst binds and is instructed to call them; unregistered means "
        "every citation requirement silently degrades."
    )


@pytest.mark.unit
def test_market_toolnode_registers_flag_gated_tools():
    """A-share market tools the analyst binds when YIAGENTS_A_SHARE_NATIVE is on."""
    market_tools = set(_tool_nodes()["market"].tools_by_name)
    missing = {
        "get_a_share_northbound_native",
        "get_a_share_sector_flow_native",
        "get_a_share_realtime_quote_native",
        "get_a_share_market_breadth_native",
    } - market_tools
    assert not missing, f"A-share market tools bound by the analyst but not executable: {missing}"


@pytest.mark.unit
def test_fundamentals_toolnode_registers_flag_gated_tools():
    """Valuation/PoT tools (valuation_tools) + A-share financials (a_share_native)."""
    fundamentals_tools = set(_tool_nodes()["fundamentals"].tools_by_name)
    missing = {
        "get_valuation_metrics",
        "pot_compute",
        "get_a_share_money_flow_native",
        "get_a_share_dragon_tiger_native",
        "get_a_share_income_statement_native",
        "get_a_share_balance_sheet_native",
        "get_a_share_cashflow_statement_native",
    } - fundamentals_tools
    assert not missing, (
        f"Flag-gated fundamentals tools bound by the analyst but not executable: {missing}"
    )


def _tool_nodes() -> dict:
    # _create_tool_nodes needs only self.quick_thinking_llm (the PoT tool
    # closure); a stub keeps this a pure-construction unit test.
    fake_self = SimpleNamespace(quick_thinking_llm=_StubLLM())
    return YiAgentsGraph._create_tool_nodes(fake_self)
