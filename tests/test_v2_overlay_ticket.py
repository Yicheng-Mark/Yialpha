"""V2.0 — overlay integration: Candidate ExecutionTicket + evidence log.

I1 on the wire: the ticket is BUILT and RENDERED by the deterministic
overlay, but the PM's rating decision text is never rewritten — a NO_TRADE
ticket coexists with an unchanged "**Rating**: Buy" markdown.
"""

from __future__ import annotations

import json

import pytest

from yialpha.agents.utils.rating import parse_rating
from yialpha.graph.signal_processing import SignalProcessor
from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.risk.tradeability import TicketSide, Tradeability


def _make_graph(tmp_path) -> YiAlphaGraph:
    """Graph shell with just enough state for the overlay + log (no LLMs)."""
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {
        "risk_enabled": True,
        "kelly_fraction": 0.25,
        "max_single_position": 0.20,
        "max_single_sector": 0.30,
        "max_drawdown_hard_stop": 0.15,
        "atr_stop_mult": 2.0,
        "results_dir": str(tmp_path),
    }
    g._risk_overlay_degraded = False
    g.risk_manager = g._build_risk_manager()
    g.signal_processor = SignalProcessor(None)
    g.ticker = "AAPL"
    return g


def _state(**over) -> dict:
    base = {
        "final_trade_decision": "**Rating**: Buy\n\nStrong momentum thesis.",
        "pm_decision_fields": {"price_target": 220.0, "confidence": 0.7},
    }
    base.update(over)
    return base


@pytest.mark.unit
def test_overlay_builds_tradeable_candidate_ticket(monkeypatch, tmp_path):
    g = _make_graph(tmp_path)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))

    out = g._apply_risk_overlay("AAPL", "2024-01-15", _state(), {"equity": 100_000})
    ticket = out["execution_ticket"]
    assert ticket["status"] == "CANDIDATE"
    assert ticket["side"] == TicketSide.LONG.value
    assert ticket["tradeability"] == Tradeability.TRADEABLE.value
    assert ticket["reference_price"] == 190.0
    assert ticket["target_price"] == 220.0
    assert ticket["confidence"] == 0.7
    assert ticket["proposed_size"] > 0.0
    assert ticket["final_size"] is None          # V2.4 resolver owns it
    assert ticket["prediction_ids"] == []        # V2.1 linkage keys present
    assert ticket["regime_id"] is None

    md = out["final_trade_decision"]
    assert "**Tradeability**: TRADEABLE" in md
    assert ticket["ticket_id"] in md
    # I1: the overlay appended, never rewrote the PM's opinion.
    assert parse_rating(md) == "Buy"
    assert "Strong momentum thesis." in md


@pytest.mark.unit
def test_no_trade_ticket_never_touches_pm_rating(monkeypatch, tmp_path):
    g = _make_graph(tmp_path)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))

    out = g._apply_risk_overlay(
        "AAPL", "2024-01-15",
        _state(
            final_trade_decision="**Rating**: Buy\n\nLate-stage chase.",
            pm_decision_fields={"price_target": 150.0},
        ),
        {"equity": 100_000},
    )
    ticket = out["execution_ticket"]
    assert ticket["tradeability"] == Tradeability.NO_TRADE.value
    assert ticket["gross_edge"] < 0.0
    md = out["final_trade_decision"]
    assert "**Tradeability**: NO_TRADE" in md
    # The rating the accuracy loop reads is untouched by the gate.
    assert parse_rating(md) == "Buy"


@pytest.mark.unit
def test_hold_rating_yields_unevaluated_flat_ticket(monkeypatch, tmp_path):
    g = _make_graph(tmp_path)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))
    out = g._apply_risk_overlay(
        "AAPL", "2024-01-15",
        _state(
            final_trade_decision="**Rating**: Hold",
            pm_decision_fields={},
        ),
        {"equity": 100_000},
    )
    ticket = out["execution_ticket"]
    assert ticket["side"] == TicketSide.FLAT.value
    assert ticket["tradeability"] == Tradeability.UNEVALUATED.value


@pytest.mark.unit
def test_perp_overlay_carries_stress_disclosure_for_historical_dates(
    monkeypatch, tmp_path,
):
    g = _make_graph(tmp_path)
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (60000.0, 900.0)
    )
    # 2024 date: funding fetch policy AND stress window both skip live calls.
    out = g._apply_risk_overlay(
        "BTCUSDT", "2024-01-15",
        _state(
            final_trade_decision="**Rating**: Buy",
            pm_decision_fields={"price_target": 66000.0},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    md = out["final_trade_decision"]
    assert "**Derivatives Stress**: n/a (historical run" in md
    assert "Suggested Leverage" in md
    ticket = out["execution_ticket"]
    assert ticket["leverage"] is not None and ticket["leverage"] >= 1.0


@pytest.mark.unit
def test_ticket_failure_degrades_without_breaking_the_overlay(
    monkeypatch, tmp_path,
):
    g = _make_graph(tmp_path)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (190.0, 3.0))

    import yialpha.tickets as tickets

    def _explode(**kwargs):
        raise RuntimeError("ticket machinery broken")

    monkeypatch.setattr(tickets, "build_candidate_ticket", _explode)
    out = g._apply_risk_overlay("AAPL", "2024-01-15", _state(), {"equity": 100_000})
    md = out["final_trade_decision"]
    assert "Quantitative Risk Overlay" in md          # overlay still ran
    assert "execution_ticket" not in out              # ...just without a ticket


def _full_final_state() -> dict:
    debate = {
        "bull_history": "", "bear_history": "", "history": "",
        "current_response": "", "judge_decision": "", "count": 0,
    }
    risk = {
        "aggressive_history": "", "conservative_history": "", "neutral_history": "",
        "history": "", "latest_speaker": "", "current_aggressive_response": "",
        "current_conservative_response": "", "current_neutral_response": "",
        "judge_decision": "", "count": 0,
    }
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2024-01-15",
        "market_report": "m", "sentiment_report": "s",
        "news_report": "n", "fundamentals_report": "f",
        "investment_debate_state": debate,
        "trader_investment_plan": "t",
        "risk_debate_state": risk,
        "investment_plan": "p",
        "final_trade_decision": "**Rating**: Buy",
        "pm_rating": "Buy",
        "pm_decision_fields": {"price_target": 220.0, "schema_version": "v1"},
        "execution_ticket": {"ticket_id": "Tabc123", "tradeability": "TRADEABLE"},
        "asset_type": "stock",
        "price_at_decision": 190.0,
        "price_at_decision_basis": "risk_overlay_close",
    }


@pytest.mark.unit
def test_log_state_persists_ticket_and_pm_fields(monkeypatch, tmp_path):
    g = _make_graph(tmp_path)
    monkeypatch.setattr(g, "_latest_close_and_atr", lambda t, d, at="stock": (None, None))

    final_state = _full_final_state()
    g._log_state("2024-01-15", final_state)

    log_path = (
        tmp_path / "AAPL" / "YiAlphaStrategy_logs" / "full_states_log_2024-01-15.json"
    )
    entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert entry["pm_decision_fields"]["price_target"] == 220.0
    assert entry["execution_ticket"]["ticket_id"] == "Tabc123"
    assert entry["execution_ticket"]["tradeability"] == "TRADEABLE"
    assert entry["pm_rating"] == "Buy"
