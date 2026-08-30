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

from yialpha.graph.trading_graph import YiAlphaGraph


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
    """A-share market tools the analyst binds when YIALPHA_A_SHARE_NATIVE is on."""
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


@pytest.mark.unit
def test_news_toolnode_registers_web_search():
    """The news analyst binds web_search whenever web_search_enabled is on
    (default): it must be executable in the news ToolNode or every call dies
    with "not a valid tool" (same wiring-gap class as the round-5 audit)."""
    news_tools = set(_tool_nodes()["news"].tools_by_name)
    assert "web_search" in news_tools, (
        "web_search is bound to the news analyst but not registered in the "
        "news ToolNode, so the model's call fails."
    )


@pytest.mark.unit
def test_market_and_fundamentals_toolnodes_register_web_search():
    """Scoped-budget allocation (2026-08-17): the market and fundamentals
    analysts bind their own web_search instances on live dates — each must
    be executable in its ToolNode. All instances share the tool NAME
    "web_search"; each ToolNode holds its own scope-charging instance, so
    one registry entry per node is the correct shape."""
    nodes = _tool_nodes()
    market_tools = set(nodes["market"].tools_by_name)
    fundamentals_tools = set(nodes["fundamentals"].tools_by_name)
    assert "web_search" in market_tools, (
        "web_search is bound to the market analyst (live runs) but not "
        "registered in the market ToolNode, so the model's call fails."
    )
    assert "web_search" in fundamentals_tools, (
        "web_search is bound to the fundamentals analyst (live runs) but "
        "not registered in the fundamentals ToolNode, so the model's call "
        "fails."
    )


@pytest.mark.unit
def test_market_toolnode_registers_vision_and_depth_tools():
    """Perp accuracy expansion (2026-08-17): the live order-book snapshot and
    the two data.binance.vision archive tools are bound by the market analyst
    for crypto_perp runs (the archive tools for historical replays too) —
    they must be executable in the market ToolNode or the deep-history
    positioning evidence silently degrades (same wiring-gap class as the
    round-5 audit)."""
    market_tools = set(_tool_nodes()["market"].tools_by_name)
    missing = {
        "get_binance_depth_snapshot",
        "get_binance_vision_metrics",
        "get_binance_vision_book_depth",
    } - market_tools
    assert not missing, (
        f"Perp accuracy tools bound by the analyst but not executable: {missing}"
    )


def _tool_nodes() -> dict:
    # _create_tool_nodes needs only self.quick_thinking_llm (the PoT tool
    # closure); a stub keeps this a pure-construction unit test.
    fake_self = SimpleNamespace(quick_thinking_llm=_StubLLM())
    return YiAlphaGraph._create_tool_nodes(fake_self)
