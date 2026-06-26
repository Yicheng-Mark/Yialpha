"""Unit tests for ``tradingagents.backtest.metrics``.

Pure-numpy, no network, no scipy required. Marked ``@pytest.mark.unit``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tradingagents.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    returns_from_equity,
)


@pytest.mark.unit
def test_returns_from_equity_known_array():
    equity = [100.0, 110.0, 99.0]
    # 110/100 - 1 = 0.1 ; 99/110 - 1 = -0.1
    got = returns_from_equity(equity)
    assert isinstance(got, np.ndarray)
    assert got.shape == (2,)
    assert np.allclose(got, [0.1, -0.1])


@pytest.mark.unit
def test_returns_from_equity_rejects_short_input():
    with pytest.raises(ValueError):
        returns_from_equity([100.0])
    with pytest.raises(ValueError):
        returns_from_equity([])


@pytest.mark.unit
def test_compute_metrics_monotonic_uptrend():
    eq = [100, 102, 104, 106, 108, 110, 112]
    m = compute_metrics(eq)
    assert isinstance(m, BacktestMetrics)
    assert m.total_return > 0.0
    assert m.sharpe > 0.0
    # Strictly increasing equity -> no drawdown.
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-12)
    # Calmar is 0 when there is no drawdown, per spec.
    assert m.calmar == 0.0
    assert m.num_periods == len(eq) - 1
    assert m.periods_per_year == 252


@pytest.mark.unit
def test_max_drawdown_known_dip():
    eq = [100.0, 120.0, 90.0, 110.0]
    m = compute_metrics(eq)
    # Trough 90 vs prior peak 120 -> 90/120 - 1 = -0.25
    assert m.max_drawdown == pytest.approx(-0.25, abs=1e-12)
    assert m.max_drawdown < 0.0


@pytest.mark.unit
def test_sortino_differs_from_sharpe_when_downside_volatility():
    # Mostly up with one sharp down tick -> upside and downside dispersion differ.
    eq = [100, 105, 110, 115, 90, 100]
    m = compute_metrics(eq)
    # When downside and total dispersion differ, Sortino and Sharpe diverge.
    # (They could coincide only if returns were symmetric; here they are not.)
    assert m.sharpe != pytest.approx(m.sortino, abs=1e-6)


@pytest.mark.unit
def test_alpha_vs_buyhold_beats_flat_benchmark():
    strategy = [100, 110, 120, 130]
    flat = [100, 100, 100, 100]
    m = compute_metrics(strategy, benchmark_equity=flat)
    assert m.alpha_vs_buyhold is not None
    # Strategy mean return > 0, benchmark mean return == 0 -> positive alpha.
    assert m.alpha_vs_buyhold > 0.0
    # Annualization: mean strat return ~0.10, * 252.
    strat_rets = returns_from_equity(strategy)
    expected_alpha = (strat_rets.mean() - 0.0) * 252
    assert m.alpha_vs_buyhold == pytest.approx(expected_alpha)


@pytest.mark.unit
def test_alpha_none_when_no_benchmark():
    m = compute_metrics([100, 110, 120])
    assert m.alpha_vs_buyhold is None


@pytest.mark.unit
def test_deflated_sharpe_in_range_and_beats_hurdle():
    # A strong, realistic uptrend with modest noise -> DSR > 0.5.
    # A perfectly linear ramp is degenerate: its per-period Sharpe is so high
    # that the Bailey-Lopez de Prado non-IID denominator goes imaginary, so we
    # use a realistic positive-mean / moderate-volatility series instead.
    rng = np.random.default_rng(42)
    returns = 0.0025 + rng.normal(0.0, 0.01, size=120)  # positive drift, noise
    eq = np.cumprod(1.0 + returns) * 100.0
    eq = np.concatenate([[100.0], eq])
    m = compute_metrics(eq, n_trials=1)
    assert 0.0 <= m.deflated_sharpe <= 1.0
    assert m.deflated_sharpe > 0.5


@pytest.mark.unit
def test_deflated_sharpe_more_trials_lowers_dsr():
    eq = [100, 101, 102, 103, 104, 105, 106, 107, 108]
    single = compute_metrics(eq, n_trials=1).deflated_sharpe
    many = compute_metrics(eq, n_trials=50).deflated_sharpe
    # More trials -> higher hurdle -> lower DSR (or equal, never higher here).
    assert many <= single + 1e-12


@pytest.mark.unit
def test_compute_metrics_rejects_short_equity():
    with pytest.raises(ValueError):
        compute_metrics([100.0])
    with pytest.raises(ValueError):
        compute_metrics([])


@pytest.mark.unit
def test_compute_metrics_invalid_args():
    with pytest.raises(ValueError):
        compute_metrics([100, 110], periods_per_year=0)
    with pytest.raises(ValueError):
        compute_metrics([100, 110], n_trials=0)


@pytest.mark.unit
def test_numpy_array_input_works():
    eq = np.array([100.0, 105.0, 103.0, 108.0, 112.0, 120.0], dtype=float)
    m = compute_metrics(eq)
    assert m.total_return == pytest.approx(120.0 / 100.0 - 1.0)
    assert m.num_periods == 5


@pytest.mark.unit
def test_inline_smoke_example():
    # Mirrors the one-liner from the task description.
    m = compute_metrics([100, 105, 103, 108, 112, 120])
    assert math.isfinite(m.total_return)
    assert math.isfinite(m.cagr)
    assert math.isfinite(m.sharpe)
    assert math.isfinite(m.deflated_sharpe)
