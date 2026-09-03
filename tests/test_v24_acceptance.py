"""V2.4 acceptance batch — the five convergence items (2026-09-03).

Post-delivery acceptance, NOT new features. The user's acceptance criteria:

1. Ticket-first ORDER verified: PM → [strategy initial sizing — single-
   instrument Kelly, NOT portfolio risk] → Candidate → Snapshot →
   RiskDecision[] → Resolver → Final. Pinned here as an execution trace.
2. Enforced HARD REFUSALS: unknown_perp / missing FX / NO_TRADE verdicts
   actually BLOCK the Final Ticket (status VETOED, no position) — record-
   stage disclosure becomes binding.
3. Short sizing labeled as what it is: a TRANSITIONAL HEURISTIC (kelly x
   max_single), never signed-Kelly; resize preserves the short sign.
4. 261 as calendar assumption: pinned in test_backtest_perp_classes.py
   (config_summary caveat) + report renders it.
5. Independent security scan: run separately (Mimosa deep scan).
"""

from __future__ import annotations

import pytest

import yialpha.ledger.portfolio as portfolio_module
import yialpha.risk.constraints as constraints_module
import yialpha.risk.resolver as resolver_module
import yialpha.tickets as tickets_module
from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.ledger.evidence import register_run
from yialpha.ledger.portfolio import commit_final_ticket, new_snapshot_id, open_positions
from yialpha.ledger.run_context import (
    reset_ledger_run_context,
    set_ledger_run_context,
)
from yialpha.tickets import TicketStatus

_TRADE_DATE = "2026-09-03"
_RUN_ID = "RV24ACCEPT01"


def _make_graph(tmp_path, **config_over) -> YiAlphaGraph:
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
    g.config.update(config_over)
    g._risk_overlay_degraded = False
    g.risk_manager = g._build_risk_manager()
    g.ticker = "BTCUSDT"
    return g


def _perp_state(**over) -> dict:
    base = {
        "final_trade_decision": "**Rating**: Buy\n\nThesis.",
        "pm_rating": "Buy",
        "pm_decision_fields": {"price_target": 70.0, "confidence": 0.7},
    }
    base.update(over)
    return base


def _stub_prices(g, monkeypatch) -> None:
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (60.0, 1.5)
    )
    monkeypatch.setattr(g, "_trailing_funding_total", lambda t, d: None)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: None)


def _bind_run(instrument_class: str = "pure_crypto_perp") -> None:
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", instrument_class, _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", instrument_class, _TRADE_DATE
    )


def _seed_open_position(symbol: str, weight: float) -> None:
    commit_final_ticket(
        ticket_payload={"ticket_id": f"T-{symbol}"},
        ticket_id=f"T-{symbol}",
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id=new_snapshot_id(f"seed-{symbol}"),
        position_symbol=symbol,
        position_side="LONG",
        position_signed_weight=weight,
        open_position=True,
    )


@pytest.fixture(autouse=True)
def _clean_run_context():
    reset_ledger_run_context()
    yield
    reset_ledger_run_context()


# --------------------------------------------------------------------------- #
# Item 1 — ticket-first order (execution trace)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_enforced_execution_order_is_ticket_first(tmp_path, monkeypatch):
    """PM → strategy sizing → CANDIDATE → snapshot → constraints → resolver
    → final commit, as ONE observable execution trace.

    The pre-candidate decide() is SINGLE-INSTRUMENT strategy sizing (Kelly ×
    breaker × CVaR × ATR — it becomes the candidate's proposed_size); the
    portfolio-level risk (snapshot/constraints/resolver) runs strictly AFTER
    the candidate exists. No portfolio-risk step may precede the candidate.
    """
    _bind_run()
    trace: list[str] = []

    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)

    real_decide = g.risk_manager.decide

    def traced_decide(*args, **kwargs):
        trace.append("strategy_sizing")
        return real_decide(*args, **kwargs)

    monkeypatch.setattr(g.risk_manager, "decide", traced_decide)

    real_build = tickets_module.build_candidate_ticket

    def traced_build(*args, **kwargs):
        trace.append("candidate")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(tickets_module, "build_candidate_ticket", traced_build)

    real_open = portfolio_module.open_positions

    def traced_open():
        trace.append("snapshot")
        return real_open()

    monkeypatch.setattr(portfolio_module, "open_positions", traced_open)

    real_eval = constraints_module.evaluate_constraints

    def traced_eval(*args, **kwargs):
        trace.append("constraints")
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(constraints_module, "evaluate_constraints", traced_eval)

    real_resolve = resolver_module.resolve_constraints

    def traced_resolve(*args, **kwargs):
        trace.append("resolver")
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(resolver_module, "resolve_constraints", traced_resolve)

    real_commit = portfolio_module.commit_final_ticket

    def traced_commit(*args, **kwargs):
        trace.append("final")
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(portfolio_module, "commit_final_ticket", traced_commit)

    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    assert out["execution_ticket"]["status"] == TicketStatus.APPROVED.value
    assert trace == [
        "strategy_sizing", "candidate", "snapshot", "constraints", "resolver", "final",
    ]


# --------------------------------------------------------------------------- #
# Item 2 — enforced hard refusals (eligibility gates bind)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_enforced_unknown_perp_is_vetoed_not_just_disclosed(tmp_path, monkeypatch):
    _bind_run(instrument_class="unknown_perp")
    monkeypatch.setattr(
        "yialpha.graph.routing.instrument_class", lambda a, t: "unknown_perp"
    )
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["status"] == TicketStatus.VETOED.value
    assert ticket["final_size"] == 0.0
    assert any(
        "unknown_instrument_class" in r for r in ticket["veto_reasons"]
    )
    assert "VETOED (eligibility)" in out["final_trade_decision"]
    assert open_positions() == []
    # The five frozen constraints still ran for the audit trail.
    assert "eligibility" in ticket["risk_decision_ids"]
    assert "global_gross" in ticket["risk_decision_ids"]


@pytest.mark.unit
def test_enforced_no_trade_verdict_blocks_the_final_ticket(tmp_path, monkeypatch):
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    # Target 30 vs close 60: deeply negative edge → tradeability NO_TRADE.
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(pm_decision_fields={"price_target": 30.0, "confidence": 0.7}),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["tradeability"] == "NO_TRADE"
    assert ticket["status"] == TicketStatus.VETOED.value
    assert ticket["final_size"] == 0.0
    assert any("tradeability_no_trade" in r for r in ticket["veto_reasons"])
    assert open_positions() == []


@pytest.mark.unit
def test_enforced_missing_fx_blocks_stock_perp_final_ticket(tmp_path, monkeypatch):
    _bind_run(instrument_class="stock_perp")
    monkeypatch.setattr(
        "yialpha.graph.routing.instrument_class", lambda a, t: "stock_perp"
    )
    monkeypatch.setattr("yialpha.perp.quote_fx.usdt_usd_as_of", lambda as_of: None)
    g = _make_graph(tmp_path, portfolio_control_mode="enforced", stock_perp_fair_value=True)
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            pm_decision_fields={"underlying_price_target": 100.0, "confidence": 0.7}
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    # The bridge recorded the gap (underlying set, no conversion)...
    assert ticket["underlying_target"] == 100.0
    assert ticket["contract_target"] is None
    # ...and in enforced mode that gap VETOES instead of shadow-marking.
    assert ticket["status"] == TicketStatus.VETOED.value
    assert any(
        "fair_value_bridge_incomplete" in r for r in ticket["veto_reasons"]
    )
    assert open_positions() == []


@pytest.mark.unit
def test_shadow_mode_shows_would_veto_lines_without_blocking(tmp_path, monkeypatch):
    _bind_run(instrument_class="unknown_perp")
    monkeypatch.setattr(
        "yialpha.graph.routing.instrument_class", lambda a, t: "unknown_perp"
    )
    g = _make_graph(tmp_path, portfolio_control_mode="shadow")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    md = out["final_trade_decision"]
    assert "WOULD VETO (eligibility)" in md
    assert "unknown_instrument_class" in md
    # Shadow never writes rows and never alters the ticket verdict.
    assert out["execution_ticket"]["status"] == TicketStatus.CANDIDATE.value
    assert open_positions() == []


# --------------------------------------------------------------------------- #
# Item 3 — short sizing honesty
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_short_sizing_labeled_as_transitional_heuristic(tmp_path, monkeypatch):
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nThesis.",
            pm_rating="Sell",
            pm_decision_fields={"desired_side": "SHORT"},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    assert "TRANSITIONAL HEURISTIC" in out["final_trade_decision"]
    assert "not signed-Kelly" in out["final_trade_decision"]
    control = out["portfolio_control"]
    assert control["short_sizing"] == "heuristic"
    (position,) = open_positions()
    assert position["signed_weight"] == pytest.approx(-0.05)


@pytest.mark.unit
def test_short_resize_preserves_the_short_sign(tmp_path, monkeypatch):
    _bind_run()
    # Gross-heavy book: the heuristic 5% short must be clipped by
    # global_gross while staying NEGATIVE (resize shrinks, never flips).
    _seed_open_position("ETHUSDT", 0.60)
    _seed_open_position("SOLUSDT", 0.38)
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nThesis.",
            pm_rating="Sell",
            pm_decision_fields={"desired_side": "SHORT"},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["status"] == TicketStatus.RESIZED.value
    assert 0.0 < ticket["final_size"] < 0.05
    (position,) = [p for p in open_positions() if p["symbol"] == "BTCUSDT"]
    assert position["side"] == "SHORT"
    assert -0.05 < position["signed_weight"] < 0.0
