"""Event-study engine: market-model abnormal returns around decision dates.

The backtest harness already records, per rebalance, a naive ``alpha_vs_index``
that is simply the asset's holding-window return minus the index's over the
same window (:mod:`yiagents.backtest.engine`). That number has no beta control
-- a high-beta stock in a bull market reads as positive "alpha" -- and no
significance test, so any claim that "the analyst's signal works" is not
falsifiable.

This module turns that claim into a statistical proposition. For each event
(rebalanced decision date) it:

1. Fits a market model ``R_asset = alpha + beta * R_benchmark`` over an
   estimation window preceding the event (OLS via ``numpy.linalg.lstsq``).
2. Computes abnormal returns ``AR_t = R_asset_t - (alpha + beta*R_benchmark_t)``
   over the event window.
3. Sums them into a cumulative abnormal return (CAR).
4. Aggregates across events: mean CAR, a cross-sectional t-test
   (Brown & Warner 1980, 1985), and a percentile bootstrap confidence interval.

Method references: Sharpe (1963) on the single-index/market model; Brown &
Warner (1980, 1985) on event-time test statistics and the cross-sectional t.

**Scope.** This is a pure offline post-evaluation over *already realized*
ratings and prices. It runs entirely outside the agent graph: it changes no
agent input, capability, or depth, and adds no non-determinism to the graph.
``scipy`` is optional (``scipy.stats.ttest_1samp`` gives an exact t-distribution
p-value; without it the t-statistic is still returned and ``p_value`` is
``None``). The event anchor is the decision date -- the question the engine
answers is "did the asset move abnormally after *our* analyst decided?" -- so
unlike a classic earnings-announcement study it depends on no filing-date
source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Optional scipy -- the exact two-sided p-value needs the t-distribution CDF.
try:  # pragma: no cover - exercised only by environment
    from scipy.stats import ttest_1samp as _scipy_ttest_1samp

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only by environment
    _scipy_ttest_1samp = None
    _HAVE_SCIPY = False


@dataclass
class EventCAR:
    """One event's cumulative abnormal return."""

    event_date: str
    car: float
    alpha: float
    beta: float
    n_estimation: int
    n_event: int
    rating: str | None = None


@dataclass
class EventStudyResult:
    """Aggregate event-study output across all decidable events."""

    events: list[EventCAR]
    mean_car: float | None
    t_stat: float | None
    p_value: float | None
    ci_low: float | None
    ci_high: float | None
    n_events: int
    estimation_size: int
    event_window: tuple[int, int]
    benchmark: str


def _price_returns(prices: pd.Series) -> pd.Series:
    """Per-period simple returns of a close-price series, as a float Series
    indexed by the price index shifted by one (return[t] belongs to date[t])."""
    p = prices.astype(float).sort_index()
    if len(p) < 2:
        return pd.Series(dtype=float)
    rets = p.values[1:] / p.values[:-1] - 1.0
    return pd.Series(rets, index=p.index[1:])


def fit_market_model(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> tuple[float, float]:
    """OLS fit of ``R_asset = alpha + beta * R_benchmark``.

    Uses the normal equations (``numpy.linalg.lstsq``) on a design matrix with
    an intercept column. Returns ``(alpha, beta)``. The two arrays must be the
    same length; align them before calling.
    """
    a = np.asarray(asset_returns, dtype=float).ravel()
    b = np.asarray(benchmark_returns, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return 0.0, 0.0
    X = np.column_stack([np.ones_like(b), b])
    coeffs, _, _, _ = np.linalg.lstsq(X, a, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def abnormal_returns(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """ ``AR_t = R_asset_t - (alpha + beta * R_benchmark_t)`` over the window."""
    a = np.asarray(asset_returns, dtype=float).ravel()
    b = np.asarray(benchmark_returns, dtype=float).ravel()
    n = min(a.size, b.size)
    return a[:n] - (alpha + beta * b[:n])


def cross_sectional_ttest(cars: np.ndarray) -> tuple[float, float | None]:
    """Brown & Warner cross-sectional t-test on a stack of CARs.

    Returns ``(t_stat, p_value)``. With scipy the p-value is the exact
    two-sided t-distribution tail; without scipy ``p_value`` is ``None`` (the
    t-statistic is still exact -- it is just ``mean / (sd / sqrt(n))``).
    """
    arr = np.asarray(cars, dtype=float).ravel()
    n = arr.size
    if n < 2:
        return 0.0, None
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return 0.0, (1.0 if _HAVE_SCIPY and _scipy_ttest_1samp is not None else None)
    t_stat = mean / (sd / math.sqrt(n))
    if _HAVE_SCIPY and _scipy_ttest_1samp is not None:
        _, p = _scipy_ttest_1samp(arr, popmean=0.0)
        return float(t_stat), float(p)
    return float(t_stat), None


def bootstrap_ci(
    cars: np.ndarray,
    n_bootstrap: int = 10_000,
    rng_seed: int | None = None,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean CAR.

    Resamples the CAR stack with replacement and takes the 2.5/97.5 percentiles
    of the bootstrapped means. Returns ``None`` when fewer than 2 CARs are
    supplied. Deterministic when ``rng_seed`` is set.
    """
    arr = np.asarray(cars, dtype=float).ravel()
    n = arr.size
    if n < 2:
        return None
    rng = np.random.default_rng(rng_seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        means[i] = rng.choice(arr, size=n, replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def event_study(
    asset_prices: pd.Series,
    benchmark_prices: pd.Series,
    event_dates: list[str],
    *,
    ratings: list[str] | None = None,
    estimation_size: int = 250,
    pre_event_gap: int = 10,
    event_window: tuple[int, int] = (0, 20),
    n_bootstrap: int = 10_000,
    rng_seed: int | None = None,
    benchmark_name: str = "benchmark",
) -> EventStudyResult:
    """Run a market-model event study over ``event_dates``.

    Parameters
    ----------
    asset_prices / benchmark_prices:
        Chronological close-price series (date-string indexed). Both must span
        every event's estimation + event windows or that event is skipped.
    event_dates:
        Decision dates acting as the event anchor (t=0). A date that is not an
        exact index label snaps forward to the next available trading day.
    ratings:
        Optional, aligned 1:1 with ``event_dates``; recorded on each ``EventCAR``
        for slicing results by signal. Not used in the statistics.
    estimation_size:
        Number of returns in the estimation window (default 250 trading days).
    pre_event_gap:
        Returns between the estimation window and the event, kept clean of
        event leakage (default 10).
    event_window:
        ``(start_offset, end_offset)`` return offsets relative to the event-day
        return, inclusive (default ``(0, 20)`` -- the event day through 20
        sessions after).
    """
    if ratings is not None and len(ratings) != len(event_dates):
        raise ValueError(
            "ratings must align 1:1 with event_dates "
            f"({len(ratings)} vs {len(event_dates)})"
        )

    # Align the two series to their common trading days so the market model
    # regresses return-on-return at matching dates.
    common = asset_prices.index.intersection(benchmark_prices.index)
    if len(common) < 2:
        return EventStudyResult(
            events=[], mean_car=None, t_stat=None, p_value=None,
            ci_low=None, ci_high=None, n_events=0,
            estimation_size=estimation_size, event_window=event_window,
            benchmark=benchmark_name,
        )
    asset_prices = asset_prices.loc[common]
    benchmark_prices = benchmark_prices.loc[common]
    asset_ret = _price_returns(asset_prices)
    bench_ret = _price_returns(benchmark_prices)

    price_idx = asset_prices.index
    start_off, end_off = event_window

    events: list[EventCAR] = []
    for k, event_date in enumerate(event_dates):
        # Map the event date to a *return* index (the event-day return). A price
        # at index `pos` realizes its return at return-index `pos-1`, so the
        # event-day return is ret_idx = pos-1 (requires pos >= 1).
        pos = _index_at_or_after(price_idx, str(event_date))
        if pos is None or pos < 1:
            continue
        ret_idx = pos - 1

        # Estimation window: a block of `estimation_size` returns ending
        # `pre_event_gap` returns before the event (clean of event leakage).
        est_start = ret_idx - pre_event_gap - estimation_size
        est_end = ret_idx - pre_event_gap
        # Event window: returns [ret_idx+start_off, ret_idx+end_off], inclusive.
        ev_start = ret_idx + start_off
        ev_end = ret_idx + end_off

        # Skip events whose estimation or event window is not fully inside both
        # return series (the common edge cases: the earliest events, where the
        # estimation window runs off the start; and the latest, where the event
        # window runs past the end).
        if est_start < 0 or ev_end >= asset_ret.size or ev_end >= bench_ret.size:
            continue

        a_est = asset_ret.values[est_start:est_end]
        b_est = bench_ret.values[est_start:est_end]
        alpha, beta = fit_market_model(a_est, b_est)

        a_ev = asset_ret.values[ev_start:ev_end + 1]
        b_ev = bench_ret.values[ev_start:ev_end + 1]
        ar = abnormal_returns(a_ev, b_ev, alpha, beta)
        car = float(ar.sum())

        events.append(EventCAR(
            event_date=str(event_date),
            car=car,
            alpha=alpha,
            beta=beta,
            n_estimation=int(est_end - est_start),
            n_event=int(ev_end - ev_start + 1),
            rating=(ratings[k] if ratings is not None else None),
        ))

    cars = np.array([e.car for e in events], dtype=float)
    mean_car: float | None = None
    t_stat: float | None = None
    p_value: float | None = None
    ci: tuple[float, float] | None = None
    if cars.size >= 1:
        mean_car = float(cars.mean())
    if cars.size >= 2:
        t_stat, p_value = cross_sectional_ttest(cars)
        ci = bootstrap_ci(cars, n_bootstrap=n_bootstrap, rng_seed=rng_seed)

    return EventStudyResult(
        events=events,
        mean_car=mean_car,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=(ci[0] if ci else None),
        ci_high=(ci[1] if ci else None),
        n_events=len(events),
        estimation_size=estimation_size,
        event_window=event_window,
        benchmark=benchmark_name,
    )


def _index_at_or_after(index: pd.Index, target: str) -> int | None:
    """Position of ``target`` in a chronological date-string index, else the
    first strictly-later date. YYYY-MM-DD strings sort chronologically, so
    lexicographic comparison is correct (mirrors engine._index_position_at_or_after)."""
    arr = np.asarray(index, dtype=object)
    exact = np.where(arr == str(target))[0]
    if exact.size:
        return int(exact[0])
    after = np.where(arr > str(target))[0]
    return int(after[0]) if after.size else None


def render_event_study(result: EventStudyResult) -> str:
    """Markdown summary of an :class:`EventStudyResult`."""
    lines = [
        "## Event study (market-model abnormal returns)",
        "",
        f"- Benchmark: `{result.benchmark}`  |  Events used: {result.n_events}",
        f"- Estimation window: {result.estimation_size} returns  |  "
        f"Event window: [{result.event_window[0]}, +{result.event_window[1]}]",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Mean CAR | {_fmt(result.mean_car)} |",
        f"| t-statistic | {_fmt(result.t_stat)} |",
        f"| p-value (two-sided) | {_fmt(result.p_value)} |",
        f"| Bootstrap 95% CI (mean CAR) | "
        f"[{_fmt(result.ci_low)}, {_fmt(result.ci_high)}] |",
        "",
    ]
    if result.events:
        lines += [
            "A positive mean CAR with a t-statistic whose p-value is below your "
            "threshold (e.g. 0.05) and a CI excluding zero is evidence the asset "
            "moved abnormally in the event window after the analyst's decision. "
            "This is a beta-controlled test; it is not the same as the naive "
            "raw-minus-index return, and it does not by itself prove a tradeable edge.",
        ]
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}"
