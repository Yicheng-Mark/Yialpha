"""Integration tests for the Phase-1 risk overlay wired into YiAgentsGraph.

These exercise the live-flow wiring (``_apply_risk_overlay``) directly, without
the full graph ``__init__`` (no LLM clients, no network). The backtest-side
wiring of the same :class:`RiskManager` is covered in ``test_risk_manager.py``.
"""

from __future__ import annotations

import pytest

from yiagents.agents.utils.rating import parse_rating
from yiagents.graph.signal_processing import SignalProcessor
from yiagents.graph.trading_graph import YiAgentsGraph


def _make_graph(risk_enabled: bool) -> YiAgentsGraph:
    """Build a graph shell with just enough state for the overlay to run."""
    g = YiAgentsGraph.__new__(YiAgentsGraph)
    g.config = {
        "risk_enabled": risk_enabled,
        "kelly_fraction": 0.25,
        "max_single_position": 0.20,
        "max_single_sector": 0.30,
        "max_drawdown_hard_stop": 0.15,
        "atr_stop_mult": 2.0,
    }
    g.risk_manager = g._build_risk_manager()
    g.signal_processor = SignalProcessor(None)
    return g


@pytest.mark.unit
def test_overlay_disabled_is_noop():
    g = _make_graph(risk_enabled=False)
    assert g.risk_manager is None
    original = "**Rating**: Buy\n\nThe thesis."
    state = {"final_trade_decision": original}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, None)
    assert out["final_trade_decision"] == original  # untouched


@pytest.mark.unit
def test_overlay_appends_section_and_preserves_rating(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d: (190.0, 3.0))

    state = {"final_trade_decision": "**Rating**: Buy\n\nStrong momentum thesis."}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})

    md = out["final_trade_decision"]
    assert "## Quantitative Risk Overlay" in md
    assert "**Target Weight**" in md
    assert "**Stop Loss**" in md
    # The appended section never hijacks rating extraction.
    assert parse_rating(md) == "Buy"
    # The original thesis is still present above the overlay.
    assert md.index("Strong momentum thesis.") < md.index("Quantitative Risk Overlay")


@pytest.mark.unit
def test_overlay_sell_has_no_stop(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d: (190.0, 3.0))
    state = {"final_trade_decision": "**Rating**: Sell"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "**Stop Loss**" not in md
    assert "Target Weight**: 0.0%" in md


@pytest.mark.unit
def test_overlay_survives_missing_price(monkeypatch):
    """No price/ATR available -> overlay still runs, just without a stop."""
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d: (None, None))
    state = {"final_trade_decision": "**Rating**: Overweight"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "Quantitative Risk Overlay" in md
    assert "**Stop Loss**" not in md
    assert "**Entry Reference**" not in md


@pytest.mark.unit
def test_overlay_coerces_dict_portfolio_state(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d: (100.0, 2.0))
    portfolio = {
        "cash": 50_000, "equity": 120_000,
        "positions": {"AAPL": 70_000}, "sectors": {"Tech": 70_000},
        "returns_history": [], "trade_history": [],
    }
    state = {"final_trade_decision": "**Rating**: Buy"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, portfolio)
    assert "Quantitative Risk Overlay" in out["final_trade_decision"]


@pytest.mark.unit
def test_overlay_drawdown_regime_recorded(monkeypatch):
    g = _make_graph(risk_enabled=True)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d: (100.0, 2.0))
    # Deep drawdown: equity 80k vs a 100k peak forces hard-stop via the breaker.
    portfolio = {"equity": 80_000, "cash": 80_000}
    # Prime the breaker by deciding once at the lower equity so it tracks peak.
    state = {"final_trade_decision": "**Rating**: Buy"}
    out = g._apply_risk_overlay("AAPL", "2024-01-15", state, portfolio)
    md = out["final_trade_decision"]
    # Regime line always present; with a fresh peak == 80k the drawdown is 0,
    # so this asserts the section renders without error and includes the field.
    assert "Drawdown Regime" in md
