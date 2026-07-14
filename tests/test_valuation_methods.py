"""Unit tests for the deterministic valuation engine (pure math, no I/O)."""

from __future__ import annotations

import math

import pytest

from yiagents.dataflows.valuation_methods import (
    earnings_yield,
    graham_number,
    intrinsic_value_two_stage_dcf,
    margin_of_safety,
    net_current_asset_value_per_share,
    owner_earnings,
    pe_ratio,
    peg_ratio,
    summarize,
    weighted_average_cost_of_capital,
)


@pytest.mark.unit
def test_pe_and_earnings_yield():
    assert pe_ratio(100.0, 5.0) == 20.0
    assert pe_ratio(100.0, -2.0) is None      # loss-maker: P/E undefined
    assert pe_ratio(100.0, 0.0) is None
    assert earnings_yield(100.0, 5.0) == pytest.approx(0.05)
    assert earnings_yield(0.0, 5.0) is None


@pytest.mark.unit
def test_graham_number():
    # sqrt(22.5 * 2 * 30) = sqrt(1350)
    assert graham_number(2.0, 30.0) == pytest.approx(math.sqrt(1350.0))
    assert graham_number(-1.0, 30.0) is None  # negative earnings
    assert graham_number(2.0, -5.0) is None   # negative book value


@pytest.mark.unit
def test_net_current_asset_value_per_share():
    assert net_current_asset_value_per_share(500.0, 300.0, 10.0) == 20.0
    assert net_current_asset_value_per_share(500.0, 300.0, 0.0) is None


@pytest.mark.unit
def test_peg_ratio():
    assert peg_ratio(20.0, 15.0) == pytest.approx(20.0 / 15.0)
    assert peg_ratio(20.0, 0.0) is None       # flat/shrinking -> undefined
    assert peg_ratio(20.0, -5.0) is None
    assert peg_ratio(-10.0, 15.0) is None     # invalid PE


@pytest.mark.unit
def test_owner_earnings():
    assert owner_earnings(100.0, 20.0, 15.0) == 105.0
    # Negative maintenance capex would be nonsensical but the formula stays
    # arithmetic; only non-finite inputs return None.
    assert owner_earnings(float("nan"), 20.0, 15.0) is None


@pytest.mark.unit
def test_dcf_flat_perpetuity_reduces_to_fcf_over_r():
    """g=0, gt=0 -> two-stage DCF collapses to the Gordon perpetuity fcf/r."""
    value = intrinsic_value_two_stage_dcf(10.0, 0.0, 0.10, 0.0, high_growth_years=10)
    assert value is not None
    assert value == pytest.approx(100.0, abs=0.01)


@pytest.mark.unit
def test_dcf_ill_posed_returns_none():
    # terminal growth >= discount rate -> terminal sum diverges
    assert intrinsic_value_two_stage_dcf(10.0, 0.10, 0.10, 0.12, 10) is None
    # non-positive horizon
    assert intrinsic_value_two_stage_dcf(10.0, 0.05, 0.10, 0.03, 0) is None


@pytest.mark.unit
def test_wacc():
    # 0.6*0.10 + 0.4*0.05 = 0.08
    assert weighted_average_cost_of_capital(600.0, 400.0, 0.10, 0.05) == pytest.approx(0.08)
    assert weighted_average_cost_of_capital(0.0, 0.0, 0.10, 0.05) is None


@pytest.mark.unit
def test_margin_of_safety_sign():
    assert margin_of_safety(100.0, 70.0) == pytest.approx(0.30)   # discount
    assert margin_of_safety(100.0, 130.0) == pytest.approx(-0.30)  # premium
    assert margin_of_safety(0.0, 50.0) is None


@pytest.mark.unit
def test_summarize_only_uses_present_inputs():
    out = summarize(price=100.0, eps=5.0, book_value_per_share=30.0)
    assert out["pe_ratio"] == 20.0
    assert out["earnings_yield"] == pytest.approx(0.05)
    assert out["graham_number"] == pytest.approx(math.sqrt(22.5 * 5.0 * 30.0))
    # Metrics whose inputs were not supplied come back as None, never a guess.
    assert out["ncav_per_share"] is None
    assert out["peg_ratio"] is None
    assert out["owner_earnings"] is None
    assert out["intrinsic_value_dcf"] is None
    assert out["wacc"] is None
    assert out["margin_of_safety_dcf"] is None


@pytest.mark.unit
def test_summarize_dcf_threads_through_to_margin_of_safety():
    out = summarize(
        price=70.0,
        free_cash_flow_per_share=10.0,
        growth_rate=0.0,
        discount_rate=0.10,
        terminal_growth_rate=0.0,
        high_growth_years=10,
    )
    assert out["intrinsic_value_dcf"] == pytest.approx(100.0, abs=0.01)
    # (100 - 70) / 100 = 0.30
    assert out["margin_of_safety_dcf"] == pytest.approx(0.30)
