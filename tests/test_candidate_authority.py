"""R1 — the authoritative Candidate and its ExecutionTicket are ONE object.

The portfolio-control stage resolves the AUTHORITATIVE candidate: the
explicit ``desired_side`` (or the frozen conservative rating mapping) for
direction, and the strategy sizing layer (RiskManager.decide, with the
transitional short heuristic) for the proposed size. The Candidate ticket,
however, was built EARLIER from the legacy rating/target_weight pair —
long-only sizing — so an explicit SHORT kept ``proposed_size=0``, no stop
and a rating-derived direction that diverged from the resolver's input
(pm-pipeline-recheck: ticket proposed 0.0 vs resolver 0.05, stop None,
Hold+SHORT ticket FLAT while the ledger went SHORT).

Enforced mode now syncs the ticket to that candidate BEFORE the
constraints run — direction, proposed size, directional tradeability
verdict (the SAME evaluate_tradeability gate via the side's proxy rating),
directional ATR stop and the perp leverage advisory — so the Final stage
can only shrink or refuse the candidate, never introduce a direction.
Shadow mode keeps the legacy ticket untouched (diffing is the point).

PositionView shape/semantics are unchanged; no new constraint, no
signed-Kelly sizing.
"""

from __future__ import annotations

import pytest

from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.ledger.evidence import register_run
from yialpha.ledger.portfolio import commit_final_ticket, new_snapshot_id, open_positions
from yialpha.ledger.run_context import (
    reset_ledger_run_context,
    set_ledger_run_context,
)
from yialpha.tickets import TicketStatus

_TRADE_DATE = "2026-09-03"
_RUN_ID = "RCANDAUTH001"


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


def _seed_open_position(symbol: str, weight: float, side: str = "LONG") -> None:
    commit_final_ticket(
        ticket_payload={"ticket_id": f"T-{symbol}", "status": "APPROVED"},
        ticket_id=f"T-{symbol}",
        decision_id=None,
        run_id=None,
        ticket_version="v1",
        snapshot_payload={"positions": []},
        snapshot_id=new_snapshot_id(f"seed-{symbol}"),
        position_symbol=symbol,
        position_side=side,
        position_signed_weight=weight,
        open_position=True,
    )


@pytest.fixture(autouse=True)
def _clean_run_context():
    reset_ledger_run_context()
    yield
    reset_ledger_run_context()


def _bind_run() -> None:
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TRADE_DATE
    )


# The reviewer's four contract checks, pinned per case.
def _assert_candidate_contract(ticket: dict, control: dict, positions: list) -> None:
    assert ticket["side"] == control["side"]
    assert ticket["proposed_size"] == pytest.approx(control["proposed_size"])
    assert ticket["final_size"] <= ticket["proposed_size"] + 1e-12
    assert not positions or ticket["stop_price"] is not None


@pytest.mark.unit
def test_sell_with_explicit_short_syncs_ticket_to_the_candidate(tmp_path, monkeypatch):
    """Sell + desired_side SHORT: the ticket IS the resolver's input.

    Before the sync the ticket kept the legacy long-only build: side SHORT
    but proposed_size 0.0, final 0.05 ABOVE its own proposed, stop None —
    a sized short with no protection.
    """
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nThesis.",
            pm_rating="Sell",
            pm_decision_fields={"desired_side": "SHORT", "price_target": 50.0},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    control = out["portfolio_control"]
    positions = open_positions()
    assert ticket["side"] == "SHORT"
    assert ticket["status"] == TicketStatus.APPROVED.value
    # Transitional heuristic sizing (kelly 0.25 x max_single 0.20), not
    # signed-Kelly.
    assert ticket["proposed_size"] == pytest.approx(0.05)
    assert ticket["final_size"] == pytest.approx(0.05)
    # Directional ATR stop: SHORT entry + mult*atr = 60 + 2*1.5.
    assert ticket["stop_price"] == pytest.approx(63.0)
    assert ticket["tradeability"] == "TRADEABLE"
    # Directional edge: 1 - target/reference, net of the 14 bps round trip
    # (taker + slippage, funding None) and the 10 bps risk buffer.
    assert ticket["gross_edge"] == pytest.approx(1.0 - 50.0 / 60.0)
    assert ticket["net_edge"] == pytest.approx(
        ticket["gross_edge"] - 0.0014 - 0.0010
    )
    assert control["side"] == "SHORT"
    assert control["proposed_size"] == pytest.approx(0.05)
    (position,) = positions
    assert position["side"] == "SHORT"
    assert position["signed_weight"] == pytest.approx(-0.05)
    _assert_candidate_contract(ticket, control, positions)


@pytest.mark.unit
def test_hold_with_explicit_short_flips_ticket_side_to_the_candidate(
    tmp_path, monkeypatch
):
    """Hold + desired_side SHORT: the ticket follows the explicit side.

    Before the sync the ticket stayed FLAT/UNEVALUATED
    ("hold_rating_no_directional_trade") while the ledger opened a SHORT —
    the Final stage had introduced a direction the Candidate never carried.
    """
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Hold\n\nThesis.",
            pm_rating="Hold",
            pm_decision_fields={"desired_side": "SHORT", "price_target": 50.0},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    control = out["portfolio_control"]
    positions = open_positions()
    assert ticket["side"] == "SHORT"
    assert ticket["tradeability"] == "TRADEABLE"
    assert ticket["tradeability_reason"] == "net_edge_above_costs"
    assert ticket["proposed_size"] == pytest.approx(0.05)
    assert ticket["stop_price"] == pytest.approx(63.0)
    assert ticket["status"] == TicketStatus.APPROVED.value
    (position,) = positions
    assert position["side"] == "SHORT"
    _assert_candidate_contract(ticket, control, positions)


@pytest.mark.unit
def test_sell_rating_alone_flattens_the_ticket_to_match_close_intent(
    tmp_path, monkeypatch,
):
    """Sell with NO explicit side: the candidate is FLAT, so is the ticket.

    Before the sync the ticket stayed SHORT (rating mapping) while the
    control closed/flat — the ledger could carry a direction the ticket
    never proposed.
    """
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nThesis.",
            pm_rating="Sell",
            pm_decision_fields={"price_target": 50.0},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    control = out["portfolio_control"]
    assert ticket["side"] == "FLAT"
    assert ticket["tradeability"] == "UNEVALUATED"
    assert ticket["tradeability_reason"] == "hold_rating_no_directional_trade"
    assert ticket["proposed_size"] == pytest.approx(0.0)
    assert ticket["final_size"] == pytest.approx(0.0)
    assert control["side"] == "FLAT"
    assert open_positions() == []
    _assert_candidate_contract(ticket, control, [])


@pytest.mark.unit
def test_short_candidate_with_negative_edge_is_vetoed_by_the_synced_verdict(
    tmp_path, monkeypatch,
):
    """A SHORT whose target sits ABOVE the reference is a negative-edge
    mistake — the synced NO_TRADE verdict must veto the Final Ticket, not
    just decorate it (the sync runs BEFORE the eligibility gate)."""
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Hold\n\nThesis.",
            pm_rating="Hold",
            pm_decision_fields={"desired_side": "SHORT", "price_target": 70.0},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["side"] == "SHORT"
    assert ticket["tradeability"] == "NO_TRADE"
    assert ticket["status"] == TicketStatus.VETOED.value
    assert ticket["final_size"] == pytest.approx(0.0)
    assert any(
        "tradeability_no_trade" in r for r in ticket["veto_reasons"]
    )
    assert open_positions() == []


@pytest.mark.unit
def test_resolver_resize_shrinks_the_synced_short_never_flips_it(
    tmp_path, monkeypatch,
):
    _bind_run()
    # A near-cap SHORT book leaves exactly a sliver of same-direction
    # headroom: the synced 5% short must be clipped while staying negative.
    # (A short candidate against a net-LONG book over the 0.8 directional
    # cap is an opposite-direction VETO by the frozen constraints — refused,
    # never resized — so same-direction headroom is the resize scenario.)
    _seed_open_position("ETHUSDT", -0.78, side="SHORT")
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nThesis.",
            pm_rating="Sell",
            pm_decision_fields={"desired_side": "SHORT", "price_target": 50.0},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    control = out["portfolio_control"]
    positions = open_positions()
    assert ticket["side"] == "SHORT"
    assert ticket["status"] == TicketStatus.RESIZED.value
    assert 0.0 < ticket["final_size"] < ticket["proposed_size"]
    btc = [p for p in positions if p["symbol"] == "BTCUSDT"]
    assert btc and btc[0]["side"] == "SHORT"
    assert -ticket["proposed_size"] < btc[0]["signed_weight"] < 0.0
    _assert_candidate_contract(ticket, control, positions)


@pytest.mark.unit
def test_long_candidate_keeps_the_legacy_stop_and_size(tmp_path, monkeypatch):
    """A plain Buy/LONG candidate is byte-identical in direction and stop:
    the directional formula reproduces the legacy long-only stop exactly."""
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="enforced")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE, _perp_state(), {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    control = out["portfolio_control"]
    assert ticket["side"] == "LONG"
    assert ticket["stop_price"] == pytest.approx(60.0 - 2.0 * 1.5)
    assert ticket["status"] == TicketStatus.APPROVED.value
    assert ticket["final_size"] == pytest.approx(ticket["proposed_size"])
    assert control["side"] == "LONG"
    _assert_candidate_contract(ticket, control, open_positions())


@pytest.mark.unit
def test_shadow_mode_keeps_the_candidate_ticket_untouched(tmp_path, monkeypatch):
    """Shadow still computes the pipeline but the ticket keeps the legacy
    build (CANDIDATE, unsized short) — diffing the two is the point."""
    _bind_run()
    g = _make_graph(tmp_path, portfolio_control_mode="shadow")
    _stub_prices(g, monkeypatch)
    out = g._apply_risk_overlay(
        "BTCUSDT", _TRADE_DATE,
        _perp_state(
            final_trade_decision="**Rating**: Sell\n\nThesis.",
            pm_rating="Sell",
            pm_decision_fields={"desired_side": "SHORT", "price_target": 50.0},
        ),
        {"equity": 100_000},
        asset_type="crypto_perp",
    )
    ticket = out["execution_ticket"]
    assert ticket["status"] == TicketStatus.CANDIDATE.value
    assert ticket["final_size"] is None
    # Legacy build: rating Sell -> SHORT side, but long-only sizing kept
    # the size at zero and no stop.
    assert ticket["side"] == "SHORT"
    assert ticket["proposed_size"] == pytest.approx(0.0)
    assert ticket["stop_price"] is None
    # The shadow record still carries the WOULD-BE candidate + advisory.
    shadow = out["portfolio_control_shadow"]
    assert shadow["side"] == "SHORT"
    assert shadow["proposed_size"] == pytest.approx(0.05)
    assert shadow["advisory"]
    assert shadow["equity"] == pytest.approx(100_000.0)
    assert open_positions() == []
