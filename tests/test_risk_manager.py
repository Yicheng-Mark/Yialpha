"""Unit tests for the RiskManager orchestrator (Phase 1 integration)."""

from __future__ import annotations

import pandas as pd
import pytest

from yiagents.backtest.engine import run_backtest
from yiagents.risk.manager import (
    PortfolioState,
    RiskDecision,
    RiskManager,
    build_backtest_weight_fn,
)


@pytest.mark.unit
def test_buy_rating_produces_sized_long_position_with_atr_stop():
    rm = RiskManager()
    state = PortfolioState(cash=100_000, equity=100_000)
    d = rm.decide("AAPL", "Buy", state, price=190.0, atr=3.0)
    assert isinstance(d, RiskDecision)
    assert d.action == "enter"
    assert 0.0 < d.target_weight <= rm.breaker.max_single_position
    assert d.stop_loss is not None
    assert d.stop_loss == pytest.approx(190.0 - 2.0 * 3.0)
    assert d.entry_price == 190.0
    assert d.exposure_ok


@pytest.mark.unit
def test_sell_goes_flat_no_stop():
    rm = RiskManager()
    state = PortfolioState(cash=100_000, equity=100_000)
    d = rm.decide("AAPL", "Sell", state, price=190.0, atr=3.0)
    assert d.target_weight == 0.0
    assert d.action in ("exit",)
    assert d.stop_loss is None  # no long to protect


@pytest.mark.unit
def test_drawdown_breaker_blocks_new_entries_then_hard_stops():
    rm = RiskManager()  # thresholds 5/10/15%
    state = PortfolioState(cash=0, equity=100_000)

    # Normal: new Buy allowed.
    d0 = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0)
    assert d0.breaker.regime == "normal"
    assert d0.target_weight > 0.0

    # Equity drops 12% -> no_new regime: a NEW long is blocked, existing kept.
    state.equity = 88_000
    state.positions = {"AAPL": 0.0}  # treat as a fresh ticker
    d1 = rm.decide("MSFT", "Buy", state, price=100.0, atr=2.0)
    assert d1.action == "blocked"
    assert d1.target_weight == 0.0
    assert not d1.exposure_ok

    # Equity drops 16% -> hard stop: multiplier 0 flattens everything.
    state.equity = 84_000
    d2 = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0)
    assert d2.target_weight == 0.0


@pytest.mark.unit
def test_cvar_derisks_when_tail_loss_breaches():
    rm = RiskManager(cvar_breach=-0.02)
    # 60 returns with a fat negative tail -> CVaR well below -2%.
    bad_history = [0.001] * 50 + [-0.08, -0.07, -0.06, -0.05, -0.04, -0.03,
                                   -0.09, -0.10, -0.08, -0.07]
    state = PortfolioState(equity=100_000, returns_history=bad_history)
    d = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0)
    assert d.cvar_multiplier == 0.5  # de-risked
    # Compare against a calm-history decision on the same rating.
    calm = PortfolioState(equity=100_000, returns_history=[0.001] * 60)
    d_calm = rm.decide("AAPL", "Buy", calm, price=100.0, atr=2.0)
    assert d.target_weight < d_calm.target_weight


@pytest.mark.unit
def test_kelly_uses_trade_history_win_rate():
    rm = RiskManager()
    # Strong ledger: 80 wins / 20 losses -> high win rate -> larger Kelly.
    good = PortfolioState(
        equity=100_000,
        trade_history=[{"return": 0.05}] * 80 + [{"return": -0.03}] * 20,
    )
    bad = PortfolioState(
        equity=100_000,
        trade_history=[{"return": 0.05}] * 20 + [{"return": -0.03}] * 80,
    )
    dg = rm.decide("AAPL", "Buy", good, price=100.0, atr=2.0)
    db = rm.decide("AAPL", "Buy", bad, price=100.0, atr=2.0)
    assert dg.kelly_raw >= db.kelly_raw
    assert dg.target_weight >= db.target_weight


@pytest.mark.unit
def test_sector_cap_trims_position():
    rm = RiskManager(max_single_position=0.20, max_single_sector=0.30)
    # Already 25% in "Tech"; a new 20% Buy would breach the 30% sector cap.
    state = PortfolioState(
        equity=100_000,
        positions={"OTHER": 25_000},
        sectors={"Tech": 25_000},
    )
    d = rm.decide("MSFT", "Buy", state, price=100.0, atr=2.0, sector="Tech")
    assert d.target_weight <= 0.05 + 1e-9  # 30% cap - 25% existing = 5% headroom


@pytest.mark.unit
def test_from_config_reads_keys_and_scales_breaker():
    cfg = {
        "kelly_fraction": 0.5, "max_single_position": 0.25, "max_single_sector": 0.40,
        "max_drawdown_hard_stop": 0.12, "atr_stop_mult": 3.0,
    }
    rm = RiskManager.from_config(cfg)
    assert rm.kelly_fraction == 0.5
    assert rm.breaker.max_single_position == 0.25
    assert rm.atr_mult == 3.0
    # Hard stop 12% -> warn ~4%, no_new ~8%, hard 12%.
    assert rm.breaker.warn_drawdown <= 0.05
    assert rm.breaker.hard_stop_drawdown == 0.12


@pytest.mark.unit
def test_unknown_rating_treated_as_hold():
    rm = RiskManager()
    state = PortfolioState(equity=100_000)
    d = rm.decide("AAPL", "GarbageRating", state, price=100.0, atr=2.0)
    assert d.rating == "Hold"
    assert d.action == "hold"


@pytest.mark.unit
def test_backtest_weight_fn_integration_with_engine():
    """The RiskManager's weight_fn must plug into run_backtest and produce sane results."""
    import pandas as pd

    dates = list(pd.bdate_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d"))[::5][:10]

    class FakeGraph:
        def propagate(self, c, d, asset_type="stock"):
            return {"final_trade_decision": "**Rating**: Buy"}, "Buy"
        def _resolve_benchmark(self, t):
            return "SPY"

    def rising(t, s, e):
        idx = pd.bdate_range(s, e)
        return pd.Series([100.0 * (1 + 0.002 * i) for i in range(len(idx))],
                         index=idx.strftime("%Y-%m-%d"), dtype=float)

    rm = RiskManager(max_single_position=0.20)
    wfn = build_backtest_weight_fn(rm, "AAPL")
    result = run_backtest(
        FakeGraph(), "AAPL", dates, holding_days=5,
        price_provider=rising, weight_fn=wfn,
    )
    # Every rebalance obeyed the 20% single-position cap.
    assert all(t.executed_weight <= 0.20 + 1e-9 for t in result.trades)
    assert result.metrics is not None
