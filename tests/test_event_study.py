"""Unit tests for the event-study engine (hermetic: synthetic prices, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yiagents.backtest.event_study import (
    abnormal_returns,
    bootstrap_ci,
    cross_sectional_ttest,
    event_study,
    fit_market_model,
)


def _prices_from_returns(returns: np.ndarray, start: float = 100.0) -> pd.Series:
    """Build a price series from per-period returns (price[0]=start)."""
    prices = start * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    return pd.Series(prices, dtype=float)


def _bday_index(n: int, start: str = "2020-01-02") -> pd.Index:
    return pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")


@pytest.mark.unit
def test_fit_market_model_recovers_known_beta():
    rng = np.random.default_rng(1)
    bench = rng.normal(0.0005, 0.01, size=300)
    # Noise-free linear model -> OLS recovers alpha, beta to machine precision.
    asset = 0.001 + 1.5 * bench
    alpha, beta = fit_market_model(asset, bench)
    assert alpha == pytest.approx(0.001, abs=1e-9)
    assert beta == pytest.approx(1.5, abs=1e-9)


@pytest.mark.unit
def test_abnormal_returns_sum_is_car():
    bench = np.array([0.01, 0.02, -0.01])
    asset = np.array([0.02, 0.03, 0.0])
    # alpha=0, beta=1 -> AR = asset - bench
    ar = abnormal_returns(asset, bench, alpha=0.0, beta=1.0)
    assert ar == pytest.approx(np.array([0.01, 0.01, 0.01]))
    assert ar.sum() == pytest.approx(0.03)


@pytest.mark.unit
def test_cross_sectional_ttest_basic():
    cars = np.array([0.05, 0.06, 0.04, 0.055, 0.045])
    t_stat, p_value = cross_sectional_ttest(cars)
    # All-positive CARs well away from zero -> large positive t.
    assert t_stat > 5
    # The exact p-value needs scipy's t-distribution; when scipy is absent the
    # t-statistic is still returned and p_value is None (documented behaviour).
    if p_value is not None:
        assert p_value < 0.01


@pytest.mark.unit
def test_bootstrap_ci_deterministic_with_seed():
    cars = np.array([0.01, 0.02, -0.005, 0.015, 0.03, 0.005, 0.012, -0.002])
    ci_a = bootstrap_ci(cars, n_bootstrap=2000, rng_seed=7)
    ci_b = bootstrap_ci(cars, n_bootstrap=2000, rng_seed=7)
    assert ci_a is not None and ci_b is not None
    assert ci_a == pytest.approx(ci_b, rel=1e-9)


def _build_drift_series(n_days: int = 700, base_seed: int = 123):
    """Asset that follows the benchmark (beta=1, alpha=0) everywhere EXCEPT
    injected per-event abnormal drift in event windows."""
    rng = np.random.default_rng(base_seed)
    bench_ret = rng.normal(0.0004, 0.01, size=n_days)
    asset_ret = bench_ret.copy()

    # Event price-positions, spaced well apart so no event window leaks into
    # another event's estimation window (estimation_size+gap < spacing).
    event_positions = [200, 290, 380, 470, 560]
    deltas = 0.005 + rng.normal(0.0, 0.0015, size=len(event_positions))
    window_len = 11  # event_window (0, +10) inclusive
    for pos, d in zip(event_positions, deltas):
        ret_idx = pos - 1
        asset_ret[ret_idx:ret_idx + window_len] += d

    idx = _bday_index(n_days)
    asset_prices = _prices_from_returns(asset_ret).set_axis(idx)
    bench_prices = _prices_from_returns(bench_ret).set_axis(idx)
    event_dates = [idx[p] for p in event_positions]
    return asset_prices, bench_prices, event_dates


@pytest.mark.unit
def test_event_study_detects_positive_abnormal_drift():
    asset_prices, bench_prices, event_dates = _build_drift_series()
    result = event_study(
        asset_prices, bench_prices, event_dates,
        estimation_size=60, pre_event_gap=5, event_window=(0, 10),
        rng_seed=11,
    )
    # Every event had a decidable window -> none skipped.
    assert result.n_events == len(event_dates)
    # Injected drift is ~0.5%/day over 11 days -> mean CAR strongly positive.
    assert result.mean_car is not None and result.mean_car > 0.03
    assert result.t_stat is not None and result.t_stat > 3
    # p_value requires scipy; the t-stat and bootstrap CI are scipy-free.
    if result.p_value is not None:
        assert result.p_value < 0.05
    assert result.ci_low is not None and result.ci_low > 0


@pytest.mark.unit
def test_event_study_zero_abnormal_is_null():
    """Asset == benchmark everywhere -> CARs all ~0, null result."""
    rng = np.random.default_rng(5)
    bench_ret = rng.normal(0.0004, 0.01, size=700)
    idx = _bday_index(700)
    asset_prices = _prices_from_returns(bench_ret).set_axis(idx)
    bench_prices = _prices_from_returns(bench_ret).set_axis(idx)
    event_positions = [200, 290, 380, 470, 560]
    event_dates = [idx[p] for p in event_positions]
    result = event_study(
        asset_prices, bench_prices, event_dates,
        estimation_size=60, pre_event_gap=5, event_window=(0, 10),
        rng_seed=11,
    )
    assert result.n_events == len(event_dates)
    # Perfect beta=1, alpha=0 fit -> abnormal returns are ~0 (only float
    # round-trip residuals from cumprod->division), so every CAR is ~0.
    assert result.mean_car == pytest.approx(0.0, abs=1e-6)
    assert all(abs(e.car) < 1e-6 for e in result.events)
    # With near-zero variance the t-statistic is an unstable ratio of two tiny
    # numbers; we only assert it is finite, not that it equals zero.
    assert result.t_stat is None or np.isfinite(result.t_stat)


@pytest.mark.unit
def test_event_study_skips_events_without_estimation_window():
    """An event too early in the series (estimation window runs off the start)
    is skipped rather than fit on a truncated window."""
    asset_prices, bench_prices, event_dates = _build_drift_series()
    early_date = asset_prices.index[10]  # ret_idx=9, est_start = 9-5-60 < 0
    result = event_study(
        asset_prices, bench_prices, event_dates + [early_date],
        estimation_size=60, pre_event_gap=5, event_window=(0, 10),
    )
    assert result.n_events == len(event_dates)  # the early one was dropped
    assert all(e.event_date != early_date for e in result.events)
