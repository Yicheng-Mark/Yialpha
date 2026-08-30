"""V2.0 P0.1 — round-trip cost model (single source of truth).

The numbers here are contract: the ExecutionTicket's ``estimated_cost``, the
tradeability gate, and the backtester's fill pricing must all read the same
constants (moved from backtest/engine.py, which re-exports them).
"""

from __future__ import annotations

import pytest

from yialpha.backtest import engine
from yialpha.risk import cost_model
from yialpha.risk.cost_model import CostEstimate, estimate_round_trip_cost
from yialpha.versions import COST_MODEL_VERSION


@pytest.mark.unit
def test_backtest_reexports_are_the_same_constants():
    # The engine module keeps its historical names; they must BE the cost
    # model's objects (identity), not copies that can drift.
    assert engine.BINANCE_USDT_M_TAKER_BPS is cost_model.BINANCE_USDT_M_TAKER_BPS
    assert engine.BINANCE_USDT_M_MAKER_BPS is cost_model.BINANCE_USDT_M_MAKER_BPS
    assert engine.BNB_FEE_DISCOUNT is cost_model.BNB_FEE_DISCOUNT
    assert cost_model.BINANCE_USDT_M_TAKER_BPS == 5.0


@pytest.mark.unit
def test_stock_long_is_slippage_only():
    est = estimate_round_trip_cost("stock", "long", horizon_days=5.0)
    assert est.entry_fee_bps == 0.0
    assert est.exit_fee_bps == 0.0
    assert est.entry_slippage_bps == cost_model.STOCK_SLIPPAGE_BPS
    assert est.borrow_bps == 0.0
    assert est.total_bps == 2 * cost_model.STOCK_SLIPPAGE_BPS


@pytest.mark.unit
def test_stock_short_accrues_borrow_over_horizon():
    short5 = estimate_round_trip_cost("stock", "short", horizon_days=5.0)
    short30 = estimate_round_trip_cost("stock", "short", horizon_days=30.0)
    assert short5.borrow_bps > 0.0
    # Linear accrual: 30d costs ~6x the 5d borrow leg.
    assert short30.borrow_bps == pytest.approx(6.0 * short5.borrow_bps)
    long5 = estimate_round_trip_cost("stock", "long", horizon_days=5.0)
    assert long5.borrow_bps == 0.0


@pytest.mark.unit
def test_spot_like_pays_spot_fee_per_side():
    for asset in ("crypto", "crypto_spot"):
        est = estimate_round_trip_cost(asset, "long")
        assert est.entry_fee_bps == cost_model.BINANCE_SPOT_FEE_BPS
        assert est.exit_fee_bps == cost_model.BINANCE_SPOT_FEE_BPS
        assert est.entry_slippage_bps == cost_model.CRYPTO_SLIPPAGE_BPS
        assert est.funding_bps == 0.0


@pytest.mark.unit
def test_perp_long_pays_positive_funding():
    # 10%/yr funding over ~18.25d of a 5d-horizon... check the exact accrual:
    # 0.10 * 1e4 * 5/365 = 13.7 bps paid by the long.
    est = estimate_round_trip_cost(
        "crypto_perp", "long", horizon_days=5.0, funding_rate_annualized=0.10
    )
    assert est.entry_fee_bps == 5.0
    assert est.funding_bps == pytest.approx(0.10 * 1e4 * 5.0 / 365.0)
    assert est.funding_bps > 0.0


@pytest.mark.unit
def test_perp_short_receives_positive_funding():
    # Same funding rate is a CREDIT for the short: signed negative leg that
    # reduces the round trip instead of being clamped away.
    est = estimate_round_trip_cost(
        "crypto_perp", "short", horizon_days=5.0, funding_rate_annualized=0.10
    )
    assert est.funding_bps == pytest.approx(-0.10 * 1e4 * 5.0 / 365.0)
    assert est.total_bps < 2 * 5.0 + 2 * cost_model.CRYPTO_SLIPPAGE_BPS


@pytest.mark.unit
def test_perp_without_funding_rate_omits_carry():
    est = estimate_round_trip_cost("crypto_perp", "long", funding_rate_annualized=None)
    assert est.funding_bps == 0.0


@pytest.mark.unit
def test_bnb_discount_reduces_perp_fees():
    plain = estimate_round_trip_cost("crypto_perp", "long")
    bnb = estimate_round_trip_cost("crypto_perp", "long", bnb_discount=True)
    assert bnb.entry_fee_bps == pytest.approx(5.0 * 0.9)
    assert plain.entry_fee_bps == 5.0


@pytest.mark.unit
def test_flat_side_prices_everything_at_zero():
    est = estimate_round_trip_cost("crypto_perp", "flat", funding_rate_annualized=0.5)
    assert est.total_bps == 0.0


@pytest.mark.unit
def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        estimate_round_trip_cost("warrant", "long")
    with pytest.raises(ValueError):
        estimate_round_trip_cost("stock", "buyish")
    with pytest.raises(ValueError):
        estimate_round_trip_cost("stock", "long", horizon_days=-1.0)


@pytest.mark.unit
def test_cost_estimate_total_sums_legs_and_as_dict_round_trips():
    est = CostEstimate(
        entry_fee_bps=5.0, exit_fee_bps=5.0,
        entry_slippage_bps=2.0, exit_slippage_bps=2.0,
        funding_bps=-1.0, borrow_bps=0.5,
    )
    assert est.total_bps == pytest.approx(13.5)
    d = est.as_dict()
    assert d["total_bps"] == pytest.approx(13.5)
    assert set(d) == {
        "entry_fee_bps", "exit_fee_bps", "entry_slippage_bps",
        "exit_slippage_bps", "funding_bps", "borrow_bps", "total_bps",
    }
    assert COST_MODEL_VERSION == "v1"
