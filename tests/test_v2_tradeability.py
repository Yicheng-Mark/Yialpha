"""V2.0 P0.2 — directional edge + NO_TRADE gate.

The edge is DIRECTIONAL (never abs): a BUY whose target sits below the
reference price is negative edge. The gate is ``net_edge <= 0 -> NO_TRADE``
— equality (edge exactly equals cost + buffer) has zero trade value.
"""

from __future__ import annotations

import pytest

from yialpha.risk.cost_model import CostEstimate
from yialpha.risk.tradeability import (
    REASON_CRITICAL_MISSING,
    REASON_DATA_INVALID,
    REASON_EDGE,
    REASON_HOLD,
    REASON_NO_REFERENCE,
    REASON_NO_TARGET,
    TicketSide,
    Tradeability,
    evaluate_tradeability,
    gross_edge,
    side_from_rating,
)


def _zero_cost() -> CostEstimate:
    return CostEstimate()


def _cost(total: float) -> CostEstimate:
    # Split symmetrically; only total_bps matters to the gate.
    half = total / 2.0
    return CostEstimate(entry_fee_bps=half, exit_fee_bps=half)


@pytest.mark.unit
def test_side_from_rating_maps_all_five_tiers():
    assert side_from_rating("Buy") == TicketSide.LONG
    assert side_from_rating("Overweight") == TicketSide.LONG
    assert side_from_rating("Sell") == TicketSide.SHORT
    assert side_from_rating("Underweight") == TicketSide.SHORT
    assert side_from_rating("Hold") == TicketSide.FLAT
    assert side_from_rating("") == TicketSide.FLAT
    assert side_from_rating("garbage") == TicketSide.FLAT


@pytest.mark.unit
def test_gross_edge_is_directional_not_absolute():
    # LONG toward a target BELOW the reference is negative edge — the exact
    # bug the abs() formulation would have hidden as a "10% edge".
    assert gross_edge(TicketSide.LONG, 110.0, 100.0) == pytest.approx(0.10)
    assert gross_edge(TicketSide.LONG, 90.0, 100.0) == pytest.approx(-0.10)
    # SHORT mirrors: profit when the target is below the reference.
    assert gross_edge(TicketSide.SHORT, 90.0, 100.0) == pytest.approx(0.10)
    assert gross_edge(TicketSide.SHORT, 110.0, 100.0) == pytest.approx(-0.10)
    assert gross_edge(TicketSide.FLAT, 110.0, 100.0) is None
    assert gross_edge(TicketSide.LONG, None, 100.0) is None
    assert gross_edge(TicketSide.LONG, 110.0, 0.0) is None


@pytest.mark.unit
def test_hold_rating_is_unevaluated_not_no_trade():
    v = evaluate_tradeability(
        rating="Hold", target_price=110.0, reference_price=100.0, cost=_zero_cost()
    )
    assert v.tradeability == Tradeability.UNEVALUATED
    assert v.tradeability_reason == REASON_HOLD


@pytest.mark.unit
def test_invalid_tier_refuses_even_with_big_edge():
    v = evaluate_tradeability(
        rating="Buy", target_price=200.0, reference_price=100.0,
        cost=_zero_cost(), quality_tier="INVALID",
    )
    assert v.tradeability == Tradeability.NO_TRADE
    assert v.tradeability_reason == REASON_DATA_INVALID


@pytest.mark.unit
def test_critical_tier_vetoes_but_auxiliary_does_not():
    critical = evaluate_tradeability(
        rating="Buy", target_price=110.0, reference_price=100.0,
        cost=_zero_cost(), quality_tier="DEGRADED_CRITICAL",
    )
    assert critical.tradeability == Tradeability.NO_TRADE
    assert critical.tradeability_reason == REASON_CRITICAL_MISSING

    # A Reddit/Binance-Square outage (auxiliary degradation) must not veto a
    # fully-priced trade (freeze check #2).
    aux = evaluate_tradeability(
        rating="Buy", target_price=110.0, reference_price=100.0,
        cost=_zero_cost(), quality_tier="DEGRADED_AUXILIARY",
    )
    assert aux.tradeability == Tradeability.TRADEABLE


@pytest.mark.unit
def test_missing_target_or_reference_is_unevaluated():
    no_target = evaluate_tradeability(
        rating="Buy", target_price=None, reference_price=100.0, cost=_zero_cost()
    )
    assert no_target.tradeability == Tradeability.UNEVALUATED
    assert no_target.tradeability_reason == REASON_NO_TARGET

    no_ref = evaluate_tradeability(
        rating="Buy", target_price=110.0, reference_price=None, cost=_zero_cost()
    )
    assert no_ref.tradeability == Tradeability.UNEVALUATED
    assert no_ref.tradeability_reason == REASON_NO_REFERENCE


@pytest.mark.unit
def test_edge_exactly_at_cost_is_no_trade():
    # gross 100 bps, cost 90 bps, buffer 10 bps -> net exactly 0: refused.
    v = evaluate_tradeability(
        rating="Buy", target_price=101.0, reference_price=100.0,
        cost=_cost(90.0), risk_buffer_bps=10.0,
    )
    assert v.gross_edge_bps == pytest.approx(100.0)
    assert v.tradeability == Tradeability.NO_TRADE
    assert v.tradeability_reason == REASON_EDGE
    assert v.net_edge_bps == pytest.approx(0.0)


@pytest.mark.unit
def test_edge_one_basis_point_above_cost_is_tradeable():
    v = evaluate_tradeability(
        rating="Buy", target_price=101.0, reference_price=100.0,
        cost=_cost(89.0), risk_buffer_bps=10.0,
    )
    assert v.tradeability == Tradeability.TRADEABLE
    assert v.net_edge_bps == pytest.approx(1.0)


@pytest.mark.unit
def test_reverse_target_buy_is_no_trade_on_edge():
    # BUY with target 90 vs reference 100: negative edge, refused with the
    # edge reason (not a data problem — a directional mistake).
    v = evaluate_tradeability(
        rating="Buy", target_price=90.0, reference_price=100.0, cost=_zero_cost()
    )
    assert v.tradeability == Tradeability.NO_TRADE
    assert v.tradeability_reason == REASON_EDGE
    assert v.gross_edge_bps == pytest.approx(-1000.0)


@pytest.mark.unit
def test_short_symmetry_through_the_full_gate():
    # SHORT target 99 vs reference 100: +100 bps gross; cost 90 + buffer 10
    # -> refused at equality, exactly like the long case.
    refused = evaluate_tradeability(
        rating="Sell", target_price=99.0, reference_price=100.0,
        cost=_cost(90.0), risk_buffer_bps=10.0,
    )
    assert refused.tradeability == Tradeability.NO_TRADE
    ok = evaluate_tradeability(
        rating="Sell", target_price=99.0, reference_price=100.0,
        cost=_cost(89.0), risk_buffer_bps=10.0,
    )
    assert ok.tradeability == Tradeability.TRADEABLE
    assert ok.side == TicketSide.SHORT
