"""Round-16 audit tests for the backtest / risk / perp money path.

Hermetic: every price, funding or benchmark series is injected; nothing here
touches the network. Each test pins ONE contract named in its docstring.
"""

from __future__ import annotations

import json
import math
import warnings
from contextlib import contextmanager, nullcontext
from datetime import date as _date

import numpy as np
import pandas as pd
import pytest

from yialpha.backtest import ic as ic_mod, metrics as mm
from yialpha.backtest.engine import BacktestResult, run_backtest
from yialpha.backtest.ic import information_coefficient, rolling_ic
from yialpha.backtest.metrics import (
    BacktestMetrics,
    _expected_max_z,
    compute_metrics,
)
from yialpha.backtest.validation_gate import evaluate_gate
from yialpha.perp import quote_fx as qfx
from yialpha.perp.fair_value import fair_value_bridge, render_bridge_block
from yialpha.perp.quote_fx import QuoteFxResult
from yialpha.risk import constraints, perp_ticket, resolver, signed_math, tradeability as ta
from yialpha.risk.cost_model import (
    BINANCE_USDT_M_TAKER_BPS,
    BNB_FEE_DISCOUNT,
    CRYPTO_SLIPPAGE_BPS,
    estimate_round_trip_cost,
)
from yialpha.risk.cvar import historical_cvar
from yialpha.risk.kelly import RATING_TO_CONFIDENCE, bayesian_win_rate, kelly_fraction, kelly_sizing
from yialpha.risk.manager import PortfolioState, RiskManager, _trade_stats

_EULER_GAMMA = 0.5772156649


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _skewed_returns() -> np.ndarray:
    """Deterministic 90-period return series with real skew and fat tails."""
    rng = np.random.default_rng(16)
    rets = rng.normal(0.0004, 0.01, 90)
    rets[10] = -0.08
    rets[40] = 0.06
    rets[77] = -0.05
    return rets


def _equity_from_returns(rets: np.ndarray) -> np.ndarray:
    """Equity curve (len(rets)+1 points) whose period returns ARE ``rets``."""
    return 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)])


def _gate_result(
    total_return: float = 0.05,
    dsr: float = 0.6,
    bh_return: float = 0.02,
    with_metrics: bool = True,
) -> BacktestResult:
    """Minimal BacktestResult carrying only what evaluate_gate reads."""
    metrics = None
    if with_metrics:
        metrics = BacktestMetrics(
            total_return=total_return, cagr=total_return, volatility=0.1,
            sharpe=1.0, sortino=1.0, max_drawdown=-0.1, calmar=1.0,
            deflated_sharpe=dsr, alpha_vs_buyhold=None, num_periods=10,
            periods_per_year=252,
        )
    return BacktestResult(
        ticker="T", initial_capital=100.0, holding_days=5,
        equity=[100.0, 100.0 * (1.0 + total_return)],
        equity_dates=["2024-01-01", "2024-01-02"], trades=[],
        benchmark_equity=[100.0, 100.0 * (1.0 + bh_return)],
        benchmark_name="T buy-and-hold", metrics=metrics,
    )


class _ScriptedGraph:
    """Stand-in graph returning scripted ratings and a fixed benchmark name."""

    def __init__(self, ratings: dict[str, str] | None = None, benchmark: str = "SPY"):
        self._ratings = dict(ratings or {})
        self._benchmark = benchmark
        self.calls: list[str] = []

    def propagate(self, company_name, trade_date, asset_type="stock"):
        self.calls.append(trade_date)
        rating = self._ratings.get(trade_date, "Hold")
        return {"final_trade_decision": f"Rating: {rating}"}, rating

    def _resolve_benchmark(self, ticker):
        return self._benchmark


_MONTH = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=20)]
# Linear tape: _MONTH[i] closes at 100 + i.
_LINEAR = {d: 100.0 + float(i) for i, d in enumerate(_MONTH)}


def _tape(closes: dict[str, float]):
    """Price provider over an explicit {date: close} tape (sorted by date)."""
    dates = sorted(closes)

    def provider(ticker, start, end):
        return pd.Series(
            [float(closes[d]) for d in dates],
            index=pd.Index(dates, dtype=object), dtype=float,
        )

    return provider


def _flat_funding(dates: list[str], rate: float):
    def provider(ticker, start, end):
        return pd.Series(
            [rate] * len(dates), index=pd.Index(dates, dtype=object), dtype=float,
        )

    return provider


# =========================================================================== #
# DSR: hurdle formula, scipy-free equivalence, fail-closed degenerates
# =========================================================================== #
@pytest.mark.unit
@pytest.mark.parametrize("n_trials", [2, 3, 10, 45])
def test_r16_dsr_expected_max_z_matches_published_approximation(n_trials):
    """E[max of N std normals] is (1-g)*ppf(1-1/N) + g*ppf(1-1/(N*e)) (B&LdP)."""
    scipy_stats = pytest.importorskip("scipy.stats")
    expected = (1.0 - _EULER_GAMMA) * float(
        scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
    ) + _EULER_GAMMA * float(
        scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    assert _expected_max_z(n_trials) == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("n_trials", [1, 0, -5])
def test_r16_dsr_expected_max_z_single_trial_has_no_penalty(n_trials):
    """N <= 1 means no multiple testing: the hurdle term is exactly zero."""
    assert _expected_max_z(n_trials) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("n_trials", [1, 2, 10])
def test_r16_dsr_scipy_free_path_equals_scipy_path(n_trials, monkeypatch):
    """Without scipy the whole compute_metrics battery is numerically identical.

    The DSR gate verdict must never depend on whether scipy is importable.
    """
    eq = _equity_from_returns(_skewed_returns())
    with_scipy = compute_metrics(eq, n_trials=n_trials)
    monkeypatch.setattr(mm, "_HAVE_SCIPY", False)
    without = compute_metrics(eq, n_trials=n_trials)
    for name in (
        "total_return", "cagr", "volatility", "sharpe", "sortino",
        "max_drawdown", "calmar",
    ):
        assert getattr(without, name) == pytest.approx(
            getattr(with_scipy, name), rel=1e-12, abs=1e-15,
        ), name
    # Acklam's inverse-normal is a 1.15e-9 relative approximation, so the
    # hurdle -- and only the hurdle -- may differ at the 1e-10 level.
    assert without.deflated_sharpe == pytest.approx(
        with_scipy.deflated_sharpe, abs=1e-9,
    )
    # The gate verdict is identical in both environments.
    verdicts = [
        evaluate_gate(_gate_result(), [_gate_result(dsr=value)]).passes
        for value in (with_scipy.deflated_sharpe, without.deflated_sharpe)
    ]
    assert verdicts[0] == verdicts[1]
    # ... and the battery is not vacuous: 1-2 trials clear, ten do not.
    assert verdicts[0] is (n_trials <= 2)


@pytest.mark.unit
def test_r16_dsr_single_trial_is_the_probabilistic_sharpe_ratio():
    """n_trials=1: DSR == Phi(SR*sqrt(n-1)/sqrt(1-sk*SR+((ku+2)/4)*SR^2))."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rets = _skewed_returns()
    n = rets.size
    sr = float(rets.mean() / rets.std(ddof=0))
    sk = float(scipy_stats.skew(rets, bias=True))
    ku = float(scipy_stats.kurtosis(rets, fisher=True, bias=True))
    stat = sr * math.sqrt(n - 1) / math.sqrt(
        1.0 - sk * sr + ((ku + 2.0) / 4.0) * sr ** 2
    )
    expected = float(scipy_stats.norm.cdf(stat))
    dsr = compute_metrics(_equity_from_returns(rets), n_trials=1).deflated_sharpe
    assert dsr == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_r16_dsr_hurdle_strictly_tightens_with_trial_count():
    """Every extra trial raises the hurdle, so DSR is strictly decreasing in N."""
    eq = _equity_from_returns(_skewed_returns())
    dsr_by_trials = [
        compute_metrics(eq, n_trials=n).deflated_sharpe
        for n in (1, 2, 5, 20, 200)
    ]
    assert all(
        later < earlier
        for earlier, later in zip(dsr_by_trials, dsr_by_trials[1:], strict=False)
    )
    assert all(0.0 <= value <= 1.0 for value in dsr_by_trials)


@pytest.mark.unit
def test_r16_dsr_single_observation_scores_zero():
    """One return cannot estimate a Sharpe: the DSR is 0.0, never NaN."""
    assert mm._deflated_sharpe_ratio(
        np.array([0.01]), sharpe_annualized=1.0, n_trials=3, periods_per_year=252,
        skew_override=None, kurt_override=None,
    ) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("constant", [0.0, 0.004])
def test_r16_dsr_zero_variance_scores_zero_even_under_float_noise(constant):
    """A zero-variance return series must score DSR 0.0 (fail-closed).

    Built through ``cumprod`` the period returns are only constant to
    ~1e-16; the ``std == 0.0`` guard used to miss that float noise and report
    DSR 0.9995 with a Sharpe of 1e13 for a mathematically variance-free curve.
    """
    metrics = compute_metrics(_equity_from_returns(np.full(20, constant)), n_trials=3)
    assert metrics.deflated_sharpe == 0.0
    assert metrics.sharpe == 0.0


@pytest.mark.unit
def test_r16_dsr_stays_inside_unit_interval_for_a_huge_stat():
    """A near-deterministic edge approaches 1.0 but never leaves [0, 1]."""
    rets = np.full(40, 0.01)
    rets[::7] = 0.0100001  # break the zero-variance degenerate
    dsr = compute_metrics(_equity_from_returns(rets)).deflated_sharpe
    assert 0.999999 < dsr <= 1.0


# =========================================================================== #
# Validation gate: fail-closed on missing / non-finite evidence
# =========================================================================== #
@pytest.mark.unit
@pytest.mark.parametrize(
    "baseline, improved, kwargs, match",
    [
        (_gate_result(), [_gate_result()], {"hurdle_dsr": 1.5}, "between 0 and 1"),
        (_gate_result(), [_gate_result()], {"hurdle_dsr": -0.1}, "between 0 and 1"),
        (_gate_result(), [_gate_result()], {"hurdle_dsr": float("nan")}, "between 0 and 1"),
        ([], [_gate_result()], {}, "at least one baseline"),
        (_gate_result(), [], {}, "at least one improved"),
        (_gate_result(), (_r for _r in ()), {}, "at least one improved"),
    ],
)
def test_r16_gate_rejects_empty_runs_and_out_of_range_hurdle(
    baseline, improved, kwargs, match,
):
    """Empty run lists (list or generator) and a hurdle outside [0, 1] raise."""
    with pytest.raises(ValueError, match=match):
        evaluate_gate(baseline, improved, **kwargs)


@pytest.mark.unit
def test_r16_gate_control_case_passes():
    """Sanity anchor: a real edge (DSR 0.6, beats B&H) does pass the gate."""
    verdict = evaluate_gate(_gate_result(), [_gate_result()])
    assert verdict.passes is True
    assert verdict.clears_hurdle is True
    assert verdict.beats_buyhold is True


@pytest.mark.unit
def test_r16_gate_unmeasured_improved_run_fails_closed():
    """An improved run with metrics=None is NO evidence: the gate must be red.

    Previously such runs were silently dropped from the mean, so with a zero
    hurdle and a losing B&H the gate passed on zero measurements.
    """
    baseline = _gate_result()
    unmeasured = _gate_result(bh_return=-0.30, with_metrics=False)
    verdict = evaluate_gate(
        baseline, [unmeasured], hurdle_dsr=0.0, min_dsr=-1.0,
    )
    assert verdict.passes is False
    assert any("no metrics" in note for note in verdict.notes)
    # Mixed: one measured winner + one unmeasured run is still not enough.
    mixed = evaluate_gate(
        baseline, [_gate_result(dsr=0.9), unmeasured],
        hurdle_dsr=0.0, min_dsr=-1.0,
    )
    assert mixed.passes is False
    assert mixed.n_runs == 2


@pytest.mark.unit
def test_r16_gate_non_finite_dsr_fails_closed():
    """A NaN DSR (NaN returns) is not coerced to a pass: the gate stays red."""
    improved = _gate_result(dsr=0.9)
    assert improved.metrics is not None
    improved.metrics.deflated_sharpe = float("nan")
    verdict = evaluate_gate(_gate_result(), [improved])
    assert verdict.passes is False
    assert verdict.clears_hurdle is False
    assert math.isnan(verdict.mean_dsr)
    # One NaN run poisons the mean: a sibling winner cannot rescue the verdict.
    mixed = evaluate_gate(_gate_result(), [improved, _gate_result(dsr=0.95)])
    assert mixed.passes is False


@pytest.mark.unit
def test_r16_metrics_nan_return_yields_nan_dsr_and_zero_sharpe():
    """An equity curve hitting exactly 0 produces a NaN return -> NaN DSR.

    NaN must propagate (so the gate fails closed on the comparison) rather
    than be clamped into a confident-looking number inside [0, 1]; Sharpe has
    no measurable volatility and degrades to the module's 0.0 convention.
    """
    # The zero equity makes the 0/0 and x/0 returns on purpose. The metrics
    # are computed under np.errstate, so numpy no longer reports them.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = compute_metrics([100.0, 0.0, 0.0, 10.0])
    assert math.isnan(metrics.deflated_sharpe)
    assert metrics.sharpe == 0.0
    improved = _gate_result(total_return=metrics.total_return)
    assert improved.metrics is not None
    improved.metrics.deflated_sharpe = metrics.deflated_sharpe
    assert evaluate_gate(_gate_result(), [improved]).passes is False


@pytest.mark.unit
def test_r16_gate_non_finite_total_return_fails_closed():
    """A NaN strategy total return cannot 'beat' buy-and-hold."""
    improved = _gate_result(total_return=0.9)
    assert improved.metrics is not None
    improved.metrics.total_return = float("nan")
    verdict = evaluate_gate(_gate_result(), [improved])
    assert verdict.passes is False
    assert verdict.beats_buyhold is False


@pytest.mark.unit
def test_r16_gate_benchmark_without_data_does_not_crash():
    """A run with an empty benchmark curve contributes 0.0 instead of raising."""
    improved = _gate_result(total_return=0.5, dsr=0.9)
    improved.benchmark_equity = []
    verdict = evaluate_gate(_gate_result(), [improved])
    assert verdict.buyhold_total_return == 0.0
    assert verdict.passes is True


@pytest.mark.unit
def test_r16_gate_hurdle_is_inclusive_and_min_dsr_is_strict():
    """DSR exactly at the hurdle clears it; min_dsr is a STRICT floor."""
    at_hurdle = _gate_result(dsr=0.5)
    assert evaluate_gate(_gate_result(), [at_hurdle]).passes is True
    assert evaluate_gate(
        _gate_result(), [at_hurdle], min_dsr=0.5,
    ).passes is False


# =========================================================================== #
# IC: scipy-free Spearman must agree with scipy on ties and non-finite pairs
# =========================================================================== #
_TIED_FACTOR = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 1.0, 4.0, 2.0]
_TIED_FWD = [0.3, -0.1, 0.2, 0.0, 0.0, -0.4, 0.5, 0.1, 0.1, -0.2, 0.6, 0.05]


@pytest.mark.unit
def test_r16_ic_scipy_free_spearman_agrees_with_scipy_on_ties(monkeypatch):
    """Average ranks for ties: the pandas-rank fallback equals scipy.spearmanr."""
    scipy_stats = pytest.importorskip("scipy.stats")
    expected = float(scipy_stats.spearmanr(_TIED_FACTOR, _TIED_FWD).statistic)
    assert information_coefficient(_TIED_FACTOR, _TIED_FWD) == pytest.approx(
        expected, abs=1e-12,
    )
    monkeypatch.setattr(ic_mod, "_HAVE_SCIPY", False)
    assert information_coefficient(_TIED_FACTOR, _TIED_FWD) == pytest.approx(
        expected, abs=1e-12,
    )


@pytest.mark.unit
def test_r16_ic_non_finite_pairs_are_dropped_before_ranking(monkeypatch):
    """inf/NaN drop PAIRWISE in both environments; the survivor set is identical."""
    scipy_stats = pytest.importorskip("scipy.stats")
    factor = list(_TIED_FACTOR)
    fwd = list(_TIED_FWD)
    factor[0] = math.inf
    fwd[5] = math.nan
    keep = [i for i in range(len(factor)) if i not in (0, 5)]
    expected = float(scipy_stats.spearmanr(
        [_TIED_FACTOR[i] for i in keep], [_TIED_FWD[i] for i in keep],
    ).statistic)
    with_scipy = information_coefficient(factor, fwd)
    monkeypatch.setattr(ic_mod, "_HAVE_SCIPY", False)
    without = information_coefficient(factor, fwd)
    assert with_scipy == pytest.approx(expected, abs=1e-12)
    assert without == pytest.approx(expected, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "factor, fwd, expected",
    [
        ([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], 1.0),
        ([1, 2, 3, 4, 5], [50, 40, 30, 20, 10], -1.0),
        ([3, 3, 3, 3, 3], [1, 2, 3, 4, 5], None),
        ([1, 2, 3], [1, 2, 3], None),  # fewer than _MIN_PAIRS
    ],
)
def test_r16_ic_monotone_degenerate_and_short_inputs(
    factor, fwd, expected, monkeypatch,
):
    """Perfect/inverse monotone give exactly +-1; zero variance or <5 pairs -> None."""
    if expected is None:
        # Zero variance: scipy's spearmanr warns on the constant input before
        # returning NaN (the numpy fallback below is silent). Too few pairs:
        # returned before either path is reached.
        with _constant_input_warning() if len(set(factor)) == 1 else nullcontext():
            assert information_coefficient(factor, fwd) is None
        monkeypatch.setattr(ic_mod, "_HAVE_SCIPY", False)
        assert information_coefficient(factor, fwd) is None
    else:
        assert information_coefficient(factor, fwd) == pytest.approx(
            expected, abs=1e-12,
        )
        monkeypatch.setattr(ic_mod, "_HAVE_SCIPY", False)
        assert information_coefficient(factor, fwd) == pytest.approx(
            expected, abs=1e-12,
        )


@contextmanager
def _constant_input_warning():
    """Context for ``information_coefficient`` on a constant input: nothing
    may warn, with or without scipy. A constant input is refused before
    ``scipy.stats.spearmanr`` is reached (round-17 follow-up); it used to warn
    and return NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", UserWarning)  # scipy's ConstantInputWarning
        yield


@pytest.mark.unit
def test_r16_ic_rolling_scipy_free_equals_scipy_path(monkeypatch):
    """rolling_ic's tied-window fallback gives the same series without scipy."""
    rng = np.random.default_rng(5)
    n = 48
    # Coarse grid guarantees plenty of tied values inside every window.
    factor = np.round(rng.normal(size=n), 1)
    fwd = 0.4 * factor + np.round(rng.normal(scale=0.7, size=n), 1)
    factor[7] = math.nan
    fwd[20] = math.inf
    with_scipy = rolling_ic(factor, fwd, window=12)
    monkeypatch.setattr(ic_mod, "_HAVE_SCIPY", False)
    without = rolling_ic(factor, fwd, window=12)
    pd.testing.assert_series_equal(without, with_scipy)
    # The series is not vacuous: most windows are decidable.
    assert int(with_scipy.notna().sum()) >= n - 12 - 2


# =========================================================================== #
# Engine: fill timing and execution costs on a hand-computed linear tape
# =========================================================================== #
def _run_stock(graph, dates, **kwargs):
    return run_backtest(
        graph, "AAPL", dates, initial_capital=100_000.0,
        price_provider=_tape(_LINEAR), compute_index_alpha=False, **kwargs,
    )


@pytest.mark.unit
def test_r16_engine_buy_fills_next_close_not_signal_close():
    """A Buy decided on bar t is filled at bar t+1's close, never at t's close."""
    result = _run_stock(_ScriptedGraph({_MONTH[0]: "Buy"}), [_MONTH[0]])
    trade = result.trades[0]
    assert trade.date == _MONTH[0]
    assert trade.execution_date == _MONTH[1]
    assert trade.price == 101.0
    shares = 100_000.0 / 101.0
    for date, equity in zip(result.equity_dates, result.equity, strict=True):
        expected = 100_000.0 if date <= _MONTH[0] else shares * _LINEAR[date]
        assert equity == pytest.approx(expected), date
    # The signal-day close (100) is NOT captured: the fill is at 101.
    assert result.equity[1] == pytest.approx(100_000.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "cost_kwargs, expected_cost",
    [
        ({}, 0.0),
        ({"cost_bps": 10.0}, 100.0),
        ({"taker_bps": 5.0, "slippage_bps": 20.0}, 100_200.0 * 0.0005),
        (
            {"taker_bps": 5.0, "slippage_bps": 20.0, "bnb_discount": True},
            100_200.0 * 0.0005 * (1.0 - BNB_FEE_DISCOUNT),
        ),
    ],
)
def test_r16_engine_fill_costs_hand_computed(cost_kwargs, expected_cost):
    """Slippage moves the fill price; the fee is charged on the SLIPPED notional."""
    result = _run_stock(_ScriptedGraph({_MONTH[0]: "Buy"}), [_MONTH[0]], **cost_kwargs)
    trade = result.trades[0]
    assert trade.transaction_cost == pytest.approx(expected_cost)
    # Traded notional is reported at the mark price, not the slipped fill.
    assert trade.traded_notional == pytest.approx(100_000.0)
    slip_cost = 100_000.0 * cost_kwargs.get("slippage_bps", 0.0) / 10_000.0
    assert result.equity[1] == pytest.approx(100_000.0 - slip_cost - expected_cost)
    # The episode return is net of the entry cost (opening equity is pre-fill).
    assert trade.position_return == pytest.approx(result.equity[-1] / 100_000.0 - 1.0)


@pytest.mark.unit
def test_r16_engine_round_trip_costs_and_episode_accounting():
    """Buy then Sell: both legs pay cost_bps on their own notional; 1 episode."""
    graph = _ScriptedGraph({_MONTH[0]: "Buy", _MONTH[3]: "Sell"})
    result = _run_stock(graph, [_MONTH[0], _MONTH[3]], cost_bps=10.0)
    buy, sell = result.trades
    assert buy.execution_date == _MONTH[1]
    assert sell.execution_date == _MONTH[4]  # close 104 on the linear tape
    shares = 100_000.0 / 101.0
    assert buy.transaction_cost == pytest.approx(100.0)
    assert sell.transaction_cost == pytest.approx(shares * 104.0 * 0.001)
    # Flat from the sell bar on: the book is the post-buy cash (-100) plus
    # the net sale proceeds; equity never moves again.
    flat_equity = -100.0 + shares * 104.0 * 0.999
    assert result.equity[-1] == pytest.approx(flat_equity)
    assert result.equity[-1] == pytest.approx(
        result.equity[result.equity_dates.index(_MONTH[4])]
    )
    assert result.metrics is not None
    assert result.metrics.num_trades == 1
    assert result.metrics.win_rate == pytest.approx(1.0)
    assert sell.position_return is None  # the closing row is not a new episode
    assert buy.position_return == pytest.approx(flat_equity / 100_000.0 - 1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "rating, allow_short, expected_sign",
    [("Buy", False, 1.0), ("Sell", True, -1.0)],
)
def test_r16_engine_perp_funding_sign_for_long_and_short(rating, allow_short, expected_sign):
    """Longs PAY positive funding, shorts RECEIVE it; the entry day pays nothing."""
    closes = {d: 100.0 + float(i) for i, d in enumerate(_MONTH)}
    result = run_backtest(
        _ScriptedGraph({_MONTH[0]: rating}), "BTCUSDT", [_MONTH[0]],
        initial_capital=100_000.0, price_provider=_tape(closes),
        funding_provider=_flat_funding(_MONTH, 0.001), asset_type="crypto_perp",
        allow_short=allow_short, compute_index_alpha=False,
    )
    trade = result.trades[0]
    assert trade.execution_date == _MONTH[1]
    assert result.equity[1] == pytest.approx(100_000.0)  # entry day: no settlement
    # Cash-account book: a long spends all cash (base 0), a 1x short holds
    # proceeds + capital (base 2E). Funding then moves cash by -shares*px*r.
    shares = expected_sign * 100_000.0 / 101.0
    cash_base = 100_000.0 - shares * 101.0
    px2 = _LINEAR[_MONTH[2]]
    assert result.equity[2] == pytest.approx(cash_base + shares * px2 * (1.0 - 0.001))
    charged_days = [d for d in result.equity_dates if d > _MONTH[1]]
    assert charged_days, "the marking window must extend past the entry bar"
    expected_paid = sum(shares * _LINEAR[d] * 0.001 for d in charged_days)
    assert result.metrics is not None
    assert result.metrics.funding_paid_total == pytest.approx(round(expected_paid, 2))
    assert math.copysign(1.0, result.metrics.funding_paid_total) == expected_sign
    # Buy-and-hold is the same 1x long and carries the identical drag.
    if expected_sign > 0.0:
        assert result.benchmark_equity == pytest.approx(result.equity)
        assert result.benchmark_equity[1] == pytest.approx(100_000.0)


# =========================================================================== #
# signed_math: the frozen sign vocabulary, and the engine obeying it
# =========================================================================== #
@pytest.mark.unit
@pytest.mark.parametrize(
    "sign, rate, expected",
    [
        (1, 0.0001, -0.1),    # long pays positive funding
        (-1, 0.0001, 0.1),    # short receives it
        (0, 0.0001, 0.0),     # flat accrues nothing
        (1, -0.0001, 0.1),    # negative funding: the long is credited
        (-1, -0.0001, -0.1),  # negative funding: the short pays
    ],
)
def test_r16_signed_funding_pnl_hand_computed(sign, rate, expected):
    """-sign*notional*rate: a credit stays a credit, it is never abs()'d."""
    assert signed_math.funding_pnl(sign, 1000.0, rate) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "side, mark, liq, expected",
    [
        ("LONG", 90.0, 90.0, True),     # inclusive at the trigger
        ("LONG", 90.01, 90.0, False),
        ("SHORT", 110.0, 110.0, True),
        ("SHORT", 109.99, 110.0, False),
        ("FLAT", 0.0, 90.0, False),     # a flat position is never in breach
        ("long", 50.0, 90.0, False),    # an unknown side is not LONG
    ],
)
def test_r16_signed_liquidation_breach_inclusive_and_side_aware(side, mark, liq, expected):
    """LONG breaches at mark <= liq, SHORT at mark >= liq, both inclusive."""
    assert signed_math.liquidation_breach(side, mark, liq) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "side, expected", [("LONG", 96.0), ("SHORT", 104.0), ("FLAT", None)],
)
def test_r16_signed_atr_stop_is_the_mirror_of_the_long_form(side, expected):
    """LONG stop = entry - mult*ATR; SHORT = entry + mult*ATR; FLAT has none."""
    stop = signed_math.atr_stop(side, 100.0, 2.0, 2.0)
    if expected is None:
        assert stop is None
    else:
        assert stop == pytest.approx(expected)
        # A fractional multiplier scales the distance linearly from entry.
        assert signed_math.atr_stop(side, 100.0, 2.0, 1.5) == pytest.approx(
            100.0 + (1.0 if side == "SHORT" else -1.0) * 3.0,
        )


@pytest.mark.unit
def test_r16_signed_stop_and_liquidation_use_opposite_sides_of_entry():
    """A long's stop sits BELOW entry and its liquidation trigger is hit from
    above; a short's mirror. The engine relies on this to order the checks."""
    entry, atr, mult = 100.0, 2.0, 2.0
    long_stop = signed_math.atr_stop("LONG", entry, atr, mult)
    short_stop = signed_math.atr_stop("SHORT", entry, atr, mult)
    assert long_stop == pytest.approx(entry - mult * atr)
    assert short_stop == pytest.approx(entry + mult * atr)
    assert (long_stop - entry) == pytest.approx(-(short_stop - entry))  # mirror invariant
    assert signed_math.same_bar_resolution(hit_stop=True, hit_liquidation=True) == "liquidation"
    assert signed_math.same_bar_resolution(hit_stop=True, hit_liquidation=False) == "stop"
    assert signed_math.same_bar_resolution(hit_stop=False, hit_liquidation=False) is None


@pytest.mark.unit
def test_r16_engine_funding_obeys_the_signed_math_convention():
    """Engine cash drag == -funding_pnl(+1, notional, rate): one sign language."""
    result = run_backtest(
        _ScriptedGraph({_MONTH[0]: "Buy"}), "BTCUSDT", [_MONTH[0]],
        initial_capital=100_000.0, price_provider=_tape(_LINEAR),
        funding_provider=_flat_funding(_MONTH, 0.001), asset_type="crypto_perp",
        compute_index_alpha=False,
    )
    shares = 100_000.0 / 101.0
    pnl = [
        signed_math.funding_pnl(signed_math.position_sign("LONG"), shares * _LINEAR[d], 0.001)
        for d in result.equity_dates if d > _MONTH[1]
    ]
    assert result.metrics is not None
    # funding_paid_total is positive when the long PAID, i.e. minus the PnL sum.
    assert result.metrics.funding_paid_total == pytest.approx(round(-sum(pnl), 2))


# =========================================================================== #
# Kelly / ledger: the payoff ratio must not be fabricated on a one-sided ledger
# =========================================================================== #
_WIDE_BANDS = dict.fromkeys(RATING_TO_CONFIDENCE, (0.0, 1.0))


@pytest.mark.unit
@pytest.mark.parametrize(
    "wins, losses, expected",
    [(0, 0, 0.5), (10, 0, 11.0 / 12.0), (0, 10, 1.0 / 12.0), (-5, 3, 1.0 / 5.0)],
)
def test_r16_kelly_bayesian_win_rate_hand_computed(wins, losses, expected):
    """Beta(1+win, 1+loss) posterior mean; negatives clamp to zero, 0/0 -> 0.5."""
    assert bayesian_win_rate(wins, losses) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "p, avg_win, avg_loss, expected",
    [
        (0.6, 2.0, 1.0, 0.4),      # (2*0.6 - 0.4)/2
        (0.4, 2.0, 1.0, 0.1),      # (2*0.4 - 0.6)/2
        (0.2, 2.0, 1.0, 0.0),      # negative edge truncated to 0
        (1.0, 3.0, 1.0, 1.0),      # a certain win is the full Kelly fraction
        (0.6, 0.05, 0.05, 0.2),    # b = 1: p - q
        (0.6, 0.05, 1.0, 0.0),     # b = 0.05: 0.6 - 0.4/0.05 < 0
        (0.6, 2.0, 0.0, 0.2),      # missing avg_loss -> symmetric b = 1: p - q
        (0.6, 2.0, -1.0, 0.2),     # a signed avg_loss reads as "missing"
        (float("nan"), 2.0, 1.0, 0.0),
        (1.5, 2.0, 1.0, 1.0),      # p is clipped into [0, 1]
    ],
)
def test_r16_kelly_fraction_full_formula_matrix(p, avg_win, avg_loss, expected):
    """f* = (b*p - q)/b with b = avg_win/|avg_loss|; a missing side gives b = 1."""
    assert kelly_fraction(p, avg_win, avg_loss) == pytest.approx(expected)


@pytest.mark.unit
def test_r16_kelly_sizing_blends_history_with_conviction_then_clips():
    """p_eff = p * confidence, quarter-Kelly, then the band clip (hand-computed)."""
    # p = 0.6, confidence(Buy) = 0.90 -> p_eff 0.54; b = 2 -> f* = 0.31.
    size = kelly_sizing(
        "Buy", win_rate=0.6, avg_win=2.0, avg_loss=1.0, band_map=_WIDE_BANDS,
    )
    assert size == pytest.approx(0.31 * 0.25)
    # The ledger beats win_rate: p = 8/12, confidence(Overweight) = 0.72 ->
    # p_eff 0.48 < 0.5 with b = 1 -> a NEGATIVE raw Kelly clipped to 0.
    clipped, raw = kelly_sizing(
        "Overweight", win_rate=0.99, wins=7, losses=3,
        avg_win=0.05, avg_loss=0.05, band_map=_WIDE_BANDS, return_raw=True,
    )
    assert raw == 0.0
    assert clipped == 0.0


@pytest.mark.unit
def test_r16_risk_one_sided_ledger_does_not_fabricate_payoff_ratio():
    """A ledger with no losses (or no wins) has an UNKNOWN opposite magnitude.

    ``_trade_stats`` used to return 1.0 for the missing side, which
    ``kelly_fraction`` read as a real 100% win / 100% loss: a 10-0 record
    scored Kelly 0 (b = 0.05) and a 0-3 record scored a POSITIVE Kelly
    (b = 20).
    """
    # 10 wins of +5%, no losses: p = 11/12, p_eff = 0.825, b = 1 -> f* = 0.65.
    wins, losses, avg_win, avg_loss = _trade_stats([{"return": 0.05}] * 10)
    assert (wins, losses) == (10, 0)
    assert avg_win == pytest.approx(0.05)
    assert avg_loss == 0.0  # "no losing trade" is NOT "every trade loses 100%"
    manager = RiskManager()
    perfect = manager.decide(
        "X", "Buy",
        PortfolioState(equity=100_000.0, trade_history=[{"return": 0.05}] * 10),
        price=100.0, atr=2.0,
    )
    assert perfect.kelly_preclip == pytest.approx(0.65 * 0.25)
    assert perfect.kelly_raw == pytest.approx(0.08)  # band CAP, not the floor
    # 0 wins, 3 losses of -5%: p = 0.2, p_eff = 0.07, b = 1 -> f* < 0 -> 0.
    wins, losses, avg_win, avg_loss = _trade_stats([{"return": -0.05}] * 3)
    assert (wins, losses) == (0, 3)
    assert avg_win == 0.0
    assert avg_loss == pytest.approx(0.05)
    losing = manager.decide(
        "X", "Underweight",
        PortfolioState(equity=100_000.0, trade_history=[{"return": -0.05}] * 3),
        price=100.0, atr=2.0,
    )
    assert losing.kelly_preclip == 0.0
    assert losing.target_weight == 0.0
    assert losing.action == "reduce"


@pytest.mark.unit
def test_r16_risk_manager_multipliers_compose_by_hand():
    """target = clip(Kelly x breaker x CVaR, cap) with every factor hand-set."""
    manager = RiskManager(max_single_position=0.20)
    # 7-3 ledger of +-5%: p = 8/12, p_eff(Buy) = 0.60, b = 1 -> f* = 0.20.
    state = PortfolioState(
        equity=100_000.0,
        trade_history=[{"return": 0.05}] * 7 + [{"return": -0.05}] * 3,
        returns_history=[-0.001] * 10 + [-0.10] * 25,  # CVaR95 = -0.10 < -0.05
    )
    decision = manager.decide("X", "Buy", state, price=100.0, atr=2.0)
    assert decision.kelly_raw == pytest.approx(0.05)      # 0.20*0.25, inside the band
    assert decision.cvar_multiplier == pytest.approx(0.5)  # breached
    assert decision.breaker.position_multiplier == pytest.approx(1.0)
    assert decision.target_weight == pytest.approx(0.025)  # 0.05 * 0.5
    assert decision.position_value == pytest.approx(2_500.0)
    assert decision.stop_loss == pytest.approx(96.0)       # 100 - 2*2
    assert decision.action == "enter"


# =========================================================================== #
# CVaR: tail count and tail side
# =========================================================================== #
@pytest.mark.unit
@pytest.mark.parametrize(
    "n, confidence, expected_tail",
    [
        (10, 0.5, 5), (10, 0.8, 2), (10, 0.95, 1), (10, 0.99, 1),
        (100, 0.9, 10), (100, 0.95, 5), (100, 0.99, 1),
        (40, 0.95, 2),  # 40*0.05 = 2.0000000000000004: float dust must not add one
        (7, 0.999, 1),
    ],
)
def test_r16_cvar_tail_count_matrix(n, confidence, expected_tail):
    """CVaR averages the worst ceil(n*(1-confidence)) (min 1) LEFT-tail returns."""
    rets = [-(i + 1) / 100.0 for i in range(n)]  # -0.01, -0.02, ..., -n/100
    # The worst `expected_tail` observations are the deepest: -(n-t+1) .. -n.
    worst = range(n - expected_tail + 1, n + 1)
    expected = -sum(worst) / expected_tail / 100.0
    assert historical_cvar(rets, confidence=confidence) == pytest.approx(expected)


@pytest.mark.unit
def test_r16_cvar_reads_the_loss_tail_not_the_gain_tail():
    """CVaR is the mean of the worst observations, not of the largest |moves|."""
    rets = [0.30, 0.02, 0.01, -0.01, -0.02, -0.40, 0.015, -0.015]
    # n = 8, confidence 0.75 -> 2 observations in each tail.
    assert historical_cvar(rets, confidence=0.75) == pytest.approx(-0.21)   # -0.40, -0.02
    # The mirrored sample exposes the GAIN tail: 0.30, 0.02.
    assert historical_cvar([-r for r in rets], confidence=0.75) == pytest.approx(-0.16)


# =========================================================================== #
# perp_ticket: leverage is the MINIMUM of four caps; liquidation mirrors entry
# =========================================================================== #
@pytest.mark.unit
@pytest.mark.parametrize(
    "stop_dist, atr_pct, asset_type, strength, profile, bracket, expected",
    [
        # 12.5 (liq) / 15 (vol) / 5 (conv) / 20 (hard) -> 5
        (0.04, 0.02, "crypto_perp", 2, "conservative", None, 5.0),
        # |Sell| == |Buy| strength: conviction 10 (moderate) binds
        (0.04, 0.02, "crypto_perp", -2, "moderate", None, 10.0),
        # a neutral rating has no conviction cap -> 0 -> floored to 1x
        (0.04, 0.02, "crypto_perp", 0, "conservative", None, 1.0),
        # a 60% stop distance: 1/(2*0.6) < 1 -> floored to 1x
        (0.60, 0.02, "crypto_perp", 2, "conservative", None, 1.0),
        # missing stop distance: the liq cap steps aside, the hard ceiling fills in
        (None, 0.02, "crypto_perp", 2, "aggressive", None, 15.0),
        # the exchange's initialLeverage cap binds below every other cap
        (0.04, 0.02, "crypto_perp", 2, "conservative", 3.0, 3.0),
        # an equity perp: hard ceiling 5, vol cap 0.20/0.02 = 10
        (0.04, 0.02, "us_stock", 2, "conservative", None, 5.0),
    ],
)
def test_r16_ticket_leverage_is_the_minimum_of_the_caps(
    stop_dist, atr_pct, asset_type, strength, profile, bracket, expected,
):
    """L = max(1, min(L_liq, L_vol, L_conv, L_hard[, L_bracket]))."""
    leverage, caps = perp_ticket.compute_leverage(
        stop_dist, atr_pct, asset_type, strength, profile=profile,
        bracket_max_leverage=bracket,
    )
    assert leverage == pytest.approx(expected)
    assert caps["L_liq"] == pytest.approx(
        1.0 / (perp_ticket.LIQ_SAFETY * stop_dist) if stop_dist
        else perp_ticket.HARD_CEILING[asset_type]
    )
    assert caps["L_vol"] == pytest.approx(
        perp_ticket.VOL_K[asset_type] / atr_pct if atr_pct
        else perp_ticket.HARD_CEILING[asset_type]
    )
    assert ("L_bracket" in caps) is (bracket is not None)
    # Exactly the smallest cap, unless that is below the 1x floor.
    assert leverage == pytest.approx(max(1.0, min(caps.values())))
    assert leverage <= perp_ticket.HARD_CEILING[asset_type]


@pytest.mark.unit
@pytest.mark.parametrize(
    "direction, notional, expected",
    [
        ("long", None, 100.0 * (1.0 - 0.2 + 0.005 + 0.0005)),      # assumed 50k -> tier 2
        ("short", None, 100.0 * (1.0 + 0.2 - 0.005 - 0.0005)),
        ("long", 10_000.0, 100.0 * (0.8 + 0.004 + 0.0005)),       # tier 1
        ("long", 200_000.0, 100.0 * (0.8 + 0.006 + 0.0005)),      # tier 3
        ("long", 1_000_000.0, 100.0 * (0.8 + 0.0125 + 0.0005)),   # tier 7
        ("LONG", None, None),      # direction vocabulary is lowercase
        ("flat", None, None),
        ("long", 1.0, 100.0 * (0.8 + 0.004 + 0.0005)),       # tier-1 notional
    ],
)
def test_r16_ticket_liquidation_price_mirrors_around_entry(direction, notional, expected):
    """entry*(1 - 1/L + MMR + fee) long, entry*(1 + 1/L - MMR - fee) short."""
    liq = perp_ticket.liquidation_price(100.0, 5.0, direction, "crypto_perp", notional)
    if direction in ("LONG", "flat"):
        assert liq is None
        return
    assert liq == pytest.approx(expected)
    if direction == "long":
        assert liq < 100.0
    else:
        assert liq > 100.0


@pytest.mark.unit
def test_r16_ticket_liquidation_not_applicable_at_or_below_1x_or_for_equities():
    """L <= 1 and non-perp assets have no USDT-M isolated-margin liquidation."""
    assert perp_ticket.liquidation_price(100.0, 1.0, "long", "crypto_perp") is None
    assert perp_ticket.liquidation_price(100.0, 0.5, "short", "crypto_perp") is None
    assert perp_ticket.liquidation_price(100.0, 5.0, "long", "us_stock") is None


@pytest.mark.unit
def test_r16_ticket_engine_and_ticket_share_one_liquidation_formula():
    """The advisory ticket and the engine's trigger must not diverge."""
    mmr = 0.005
    ticket_long = perp_ticket.liquidation_price(
        250.0, 4.0, "long", "crypto_perp", notional=60_000.0,
    )
    engine_long = 250.0 * (1.0 - 1.0 / 4.0 + mmr + 0.0005)
    assert ticket_long == pytest.approx(engine_long)
    ticket_short = perp_ticket.liquidation_price(
        250.0, 4.0, "short", "crypto_perp", notional=60_000.0,
    )
    assert ticket_short == pytest.approx(250.0 * (1.0 + 0.25 - mmr - 0.0005))
    # Mirror invariant around the entry price.
    assert (ticket_long - 250.0) == pytest.approx(-(ticket_short - 250.0))


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry, stop, direction, expected",
    [
        (100.0, 96.0, "long", [106.0, 112.0, 120.0]),     # R = 4: 1.5R / 3R / 5R
        (100.0, 104.0, "short", [94.0, 88.0, 80.0]),
        (100.0, 100.0, "long", []),                        # R == 0 -> no targets
        (None, 96.0, "long", []),
        (100.0, None, "long", []),
        (1.3e-5, 1.0e-5, "long", [1.75e-5, 2.2e-5, 2.8e-5]),  # significant digits
    ],
)
def test_r16_ticket_take_profits_are_r_multiples(entry, stop, direction, expected):
    """TP = entry +- mult*|entry-stop| for mult in (1.5, 3, 5), 8 significant digits."""
    targets = perp_ticket.take_profits(entry, stop, direction)
    assert len(targets) == len(expected)
    for got, want in zip(targets, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-6)


@pytest.mark.unit
def test_r16_ticket_numbers_long_hand_computed():
    """Buy, ATR 2% with a 2*ATR stop -> 5x (conviction), liq 80.55, stop 96."""
    lev, detail, liq, stop_used = perp_ticket.perp_ticket_numbers(
        100.0, 2.0, "Buy", None, 0.10,
    )
    assert stop_used == pytest.approx(96.0)
    assert lev == pytest.approx(5.0)
    assert detail["L_liq"] == pytest.approx(12.5)
    assert detail["L_vol"] == pytest.approx(15.0)
    assert detail["L_conv"] == pytest.approx(5.0)
    assert detail["L_hard"] == pytest.approx(20.0)
    assert liq == pytest.approx(80.55)
    assert detail["liq_mmr"] == pytest.approx(0.005)
    assert detail["liq_fee_rate"] == pytest.approx(0.0005)
    assert "assumed 50,000 USDT notional" in detail["liq_mmr_basis"]


@pytest.mark.unit
def test_r16_ticket_numbers_short_and_real_notional_hand_computed():
    """A short ticket mirrors the long one; a real notional re-tiers the MMR."""
    lev, detail, liq, stop_used = perp_ticket.perp_ticket_numbers(
        100.0, 2.0, "Sell", None, -0.10, notional=200_000.0,
    )
    assert stop_used == pytest.approx(104.0)       # entry + 2*ATR
    assert lev == pytest.approx(5.0)
    assert liq == pytest.approx(100.0 * (1.0 + 0.2 - 0.006 - 0.0005))
    assert detail["liq_mmr"] == pytest.approx(0.006)
    assert "notional 200,000 USDT" in detail["liq_mmr_basis"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "rating, weight, entry, stop, expected_stop",
    [
        # Direction-correct explicit stops are honoured on both sides.
        ("Buy", 0.1, 100.0, 97.0, 97.0),
        ("Sell", -0.1, 100.0, 103.0, 103.0),
        # A stop on the wrong side is rebuilt (round-8): long stop >= entry,
        # short stop <= entry.
        ("Buy", 0.1, 100.0, 101.0, 96.0),
        ("Buy", 0.1, 100.0, 100.0, 96.0),
        ("Sell", -0.1, 100.0, 99.0, 104.0),
        # Non-positive / NaN / inf stops are unusable and rebuilt too.
        ("Buy", 0.1, 100.0, 0.0, 96.0),
        ("Buy", 0.1, 100.0, -5.0, 96.0),
        ("Buy", 0.1, 100.0, float("nan"), 96.0),
        ("Sell", -0.1, 100.0, float("inf"), 104.0),
        # The caller's ATR multiplier drives the rebuilt stop.
        ("Buy", 0.1, 100.0, None, 96.0),
    ],
)
def test_r16_ticket_stop_is_honoured_only_when_direction_correct(
    rating, weight, entry, stop, expected_stop,
):
    """Explicit stops must be finite AND on the losing side; else mult*ATR."""
    _lev, _detail, _liq, stop_used = perp_ticket.perp_ticket_numbers(
        entry, 2.0, rating, stop, weight,
    )
    assert stop_used == pytest.approx(expected_stop)


@pytest.mark.unit
def test_r16_ticket_rebuilt_stop_uses_the_configured_atr_multiplier():
    """atr_mult threads through: a 3*ATR run renders advice off a 3*ATR stop."""
    lev, detail, _liq, stop_used = perp_ticket.perp_ticket_numbers(
        100.0, 2.0, "Buy", None, 0.10, atr_mult=3.0,
    )
    assert stop_used == pytest.approx(94.0)
    assert detail["L_liq"] == pytest.approx(1.0 / (2.0 * 0.06))
    assert lev == pytest.approx(5.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry, atr, rating, stop, weight",
    [
        (None, 2.0, "Buy", None, 0.1),
        (0.0, 2.0, "Buy", None, 0.1),
        (100.0, None, "Buy", None, 0.1),
        (100.0, 0.0, "Buy", None, 0.1),
        (100.0, 2.0, "Buy", None, 0.0),     # flat weight: no ticket
    ],
)
def test_r16_ticket_numbers_none_on_unusable_inputs(entry, atr, rating, stop, weight):
    """No price / no ATR / zero weight -> None (the caller renders nothing)."""
    assert perp_ticket.perp_ticket_numbers(entry, atr, rating, stop, weight) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "rating, expected_strength",
    [("Buy", 2), ("buy", 2), ("BUY", 2), ("Overweight", 1), ("Hold", 0),
     ("Underweight", -1), ("sell", -2), ("Strong Buy", 0), (None, 0), (3, 0)],
)
def test_r16_ticket_rating_strength_is_case_insensitive(rating, expected_strength):
    """rating_strength normalizes case; unknown / non-string ratings are 0."""
    assert perp_ticket.rating_strength(rating) == expected_strength


@pytest.mark.unit
def test_r16_ticket_neutral_rating_gets_no_leverage_and_no_liquidation():
    """Hold has strength 0 -> L_conv 0 -> 1x, and 1x has no liquidation row."""
    lev, detail, liq, stop_used = perp_ticket.perp_ticket_numbers(
        100.0, 2.0, "Hold", None, 0.10,
    )
    assert lev == 1.0
    assert liq is None
    assert "liq_mmr" not in detail
    assert stop_used == pytest.approx(96.0)


# =========================================================================== #
# quote_fx: fail-soft payload handling; fair_value: USD -> USDT divides
# =========================================================================== #
def _fx_payload(close, available_at="2026-11-20"):
    return json.dumps({
        "fetched_at": "2026-11-20T08:00:00+00:00",
        "available_at": available_at,
        "close": close,
    }).encode("utf-8")


@pytest.mark.unit
def test_r16_quote_fx_cache_machinery_failure_degrades_to_none(monkeypatch, caplog):
    """cached_or_fetch itself raising must not break a run (never a 1.0 peg)."""
    seen: list[tuple] = []

    def _explode(cache_dir, name, fetch, ttl_days, vendor, fail_open):
        seen.append((name, ttl_days, vendor, fail_open))
        raise OSError("disk full")

    monkeypatch.setattr(qfx, "cached_or_fetch", _explode)
    with caplog.at_level("WARNING"):
        assert qfx.fetch_usdt_usd() is None
    assert len(seen) == 1
    name, ttl_days, vendor, fail_open = seen[0]
    assert name.startswith("usdcusdt_") and name.endswith(".json")
    assert ttl_days == pytest.approx(1.0 / 24.0)   # one-hour TTL
    assert vendor == "quote_fx" and fail_open is True
    assert any("payload fetch failed" in r.message for r in caplog.records)


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        b"not json at all",
        json.dumps({"close": "abc", "available_at": "2026-11-20", "fetched_at": "t"}).encode(),
        json.dumps({"available_at": "2026-11-20", "fetched_at": "t"}).encode(),  # no close
        json.dumps({"close": 1.0, "fetched_at": "t"}).encode(),                   # no date
        json.dumps({"close": None, "available_at": "2026-11-20", "fetched_at": "t"}).encode(),
    ],
)
def test_r16_quote_fx_unreadable_payload_degrades_to_none(monkeypatch, payload):
    """A cached payload that cannot be parsed is UNAVAILABLE, never a guess."""
    monkeypatch.setattr(qfx, "cached_or_fetch", lambda *a, **k: payload)
    assert qfx.fetch_usdt_usd() is None


@pytest.mark.unit
def test_r16_quote_fx_round_trip_is_the_inverted_close(monkeypatch):
    """rate = 1/close is USD per 1 USDT; provenance travels with the payload."""
    monkeypatch.setattr(qfx, "cached_or_fetch", lambda *a, **k: _fx_payload(0.9993))
    result = qfx.fetch_usdt_usd()
    assert result is not None
    assert result.rate == pytest.approx(1.0 / 0.9993)
    assert result.available_at == "2026-11-20"
    assert result.fetched_at == "2026-11-20T08:00:00+00:00"
    assert result.source == qfx.QUOTE_FX_SOURCE
    assert result.replayability == qfx.LIVE_ONLY == "LIVE_ONLY"
    assert qfx.render_fx_line(result) == (
        f"USDT/USD fx: {result.rate:.6f} ({qfx.QUOTE_FX_SOURCE}, live-only)"
    )


@pytest.mark.unit
def test_r16_quote_fx_as_of_gate_uses_both_live_anchors(monkeypatch):
    """Only a LIVE label (UTC or host-local today) may fetch; else None, no fetch."""
    calls: list[str] = []
    sentinel = QuoteFxResult(1.0, "2026-03-02", "2026-03-02T00:00:00+00:00")

    def _fetch():
        calls.append("fetch")
        return sentinel

    monkeypatch.setattr(qfx, "fetch_usdt_usd", _fetch)
    monkeypatch.setattr(qfx, "live_anchor_dates", lambda: {_date(2026, 3, 2)})
    assert qfx.usdt_usd_as_of("2026-03-02") is sentinel
    assert calls == ["fetch"]
    assert qfx.usdt_usd_as_of("2026-03-01") is None   # yesterday: no PIT archive
    assert qfx.usdt_usd_as_of("2026-03-03") is None   # tomorrow: no observation
    assert qfx.usdt_usd_as_of("02/03/2026") is None   # unparseable: no raise
    assert calls == ["fetch"]
    assert qfx.render_fx_line(None) == (
        "USDT/USD fx: UNAVAILABLE (live-only feed; not PIT-replayable)"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "usd, fx, basis, expected",
    [
        (100.0, 0.9993, 0.0, 100.0 / 0.9993),      # a discounted USDT needs MORE units
        (100.0, 1.0, 0.0, 100.0),                   # par: identity
        (100.0, 1.02, 0.0, 100.0 / 1.02),           # a premium USDT needs FEWER
        (100.0, 1.0, 0.01, 101.0),                  # +1% expected premium
        (100.0, 0.9993, -0.02, 100.0 / 0.9993 * 0.98),
        (0.004, 1.0, 0.0, 0.004),                   # micro-price target
    ],
)
def test_r16_fair_value_bridge_divides_by_usd_per_usdt(usd, fx, basis, expected):
    """contract_usdt = usd / (USD per USDT) * (1 + expected_basis)."""
    result = fair_value_bridge(usd, fx, current_basis=0.003, expected_basis=basis)
    assert result.contract_target_usdt == pytest.approx(expected)
    assert result.missing_inputs == ()           # the basis snapshot was supplied
    assert len(result.chain) == 4
    # The stored target keeps full precision; only the rendering is rounded.
    assert render_bridge_block(result).splitlines()[-1] == (
        "- = contract target "
        + (f"{expected:.6f}" if abs(expected) >= 1.0 else f"{expected:.6g}")
        + " USDT"
    )


@pytest.mark.unit
def test_r16_fair_value_current_basis_never_enters_the_math():
    """current_basis is an audit snapshot: with or without it the target is equal."""
    with_basis = fair_value_bridge(100.0, 0.9993, current_basis=0.25)
    without_basis = fair_value_bridge(100.0, 0.9993, current_basis=None)
    assert with_basis.contract_target_usdt == pytest.approx(
        without_basis.contract_target_usdt
    )
    assert without_basis.missing_inputs == ("current_basis",)
    assert "current basis snapshot not provided" in render_bridge_block(without_basis)


# =========================================================================== #
# constraints: headroom arithmetic, frozen order, fail-closed vetoes
# =========================================================================== #
def _pos(symbol, klass, side, weight):
    return constraints.PositionView(
        symbol=symbol, instrument_class=klass, side=side, weight=weight,
    )


def _snap(equity, *positions, returns=()):
    return constraints.PortfolioSnapshot(
        equity=equity, positions=tuple(positions), returns_history=returns,
    )


_C = "stock_perp"


@pytest.mark.unit
@pytest.mark.parametrize(
    "book, candidate, expected_action, expected_mult, expected_reason, expected_metric",
    [
        # others 0.6, candidate 0.5, cap 1.0 -> headroom 0.4 -> x0.8
        ((_pos("A", _C, "LONG", 0.3), _pos("B", _C, "SHORT", -0.3)),
         _pos("C", _C, "LONG", 0.5), "RESIZE", 0.8, "exceeds_limit", 1.1),
        # re-sizing a holding replaces it: others is only B's 0.3 -> fits
        ((_pos("A", _C, "LONG", 0.3), _pos("B", _C, "SHORT", -0.3)),
         _pos("A", _C, "LONG", 0.5), "PASS", 1.0, "within_limit", 0.8),
        # the old book is already over the cap: zero headroom, candidate -> 0
        ((_pos("A", _C, "LONG", 0.6), _pos("B", _C, "LONG", 0.6)),
         _pos("C", _C, "LONG", 0.1), "RESIZE", 0.0, "book_already_over_limit", 1.3),
        # a long and a short BOTH consume gross (offsetting is not gross's job)
        ((_pos("A", _C, "LONG", 0.6), _pos("B", _C, "SHORT", -0.6)),
         _pos("C", _C, "SHORT", -0.1), "RESIZE", 0.0, "book_already_over_limit", 1.3),
        # a risk-reducing CLOSE passes unconditionally, even on an over-cap book
        # (the candidate REPLACES A: the post book is B's 0.6 alone)
        ((_pos("A", _C, "LONG", 0.6), _pos("B", _C, "LONG", 0.6)),
         _pos("A", _C, "LONG", 0.0), "PASS", 1.0, "within_limit", 0.6),
        # exactly on the cap passes
        ((), _pos("C", _C, "LONG", 1.0), "PASS", 1.0, "within_limit", 1.0),
    ],
)
def test_r16_constraints_global_gross_headroom_matrix(
    book, candidate, expected_action, expected_mult, expected_reason, expected_metric,
):
    """Over the cap the candidate is sized into UNUSED headroom, never VETOed."""
    decision = constraints.global_gross(candidate, _snap(100_000.0, *book), constraints.PortfolioLimits())
    assert decision.action == expected_action
    assert decision.multiplier == pytest.approx(expected_mult)
    assert decision.reasons == [expected_reason]
    assert decision.metrics["post_gross"] == pytest.approx(expected_metric)
    assert decision.metrics["max_gross"] == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("weight, expected", [(0.25, 0.8), (0.20, 1.0), (0.10, 1.0), (0.0, 1.0)])
def test_r16_constraints_single_concentration_is_limit_over_proposed(weight, expected):
    """A lone position over max_single is RESIZED to limit/|w|, never vetoed."""
    decision = constraints.single_concentration(
        _pos("A", _C, "LONG", weight), _snap(100_000.0), constraints.PortfolioLimits(),
    )
    assert decision.multiplier == pytest.approx(expected)
    assert decision.action == ("PASS" if expected == 1.0 else "RESIZE")
    assert decision.metrics["post_single"] == pytest.approx(weight)


@pytest.mark.unit
def test_r16_constraints_asset_class_budget_is_per_class():
    """Each sleeve has its own budget: crypto gross never eats the stock cap."""
    limits = constraints.PortfolioLimits()
    book = _snap(
        100_000.0,
        _pos("X", "stock_perp", "LONG", 0.4),
        _pos("Y", "pure_crypto_perp", "SHORT", 0.5),
    )
    stock = constraints.asset_class(
        _pos("Z", "stock_perp", "LONG", 0.3), book, limits,
    )
    # stock-perp others 0.4, cap 0.6 -> headroom 0.2 for a 0.3 candidate.
    assert (stock.action, stock.multiplier) == ("RESIZE", pytest.approx(2.0 / 3.0))
    assert stock.metrics["post_class_gross"] == pytest.approx(0.7)
    crypto = constraints.asset_class(
        _pos("W", "pure_crypto_perp", "LONG", 0.1), book, limits,
    )
    # crypto others 0.5, cap 0.6 -> exactly the 0.1 headroom: PASS.
    assert (crypto.action, crypto.multiplier) == ("PASS", 1.0)
    # The same symbol on another venue is a DIFFERENT row: it is not replaced
    # and does not consume this class's budget.
    mixed = _snap(100_000.0, _pos("Z", "pure_crypto_perp", "LONG", 0.55))
    other_venue = constraints.asset_class(
        _pos("Z", "stock_perp", "LONG", 0.3), mixed, limits,
    )
    assert (other_venue.action, other_venue.multiplier) == ("PASS", 1.0)
    gross = constraints.global_gross(_pos("Z", "stock_perp", "LONG", 0.3), mixed, limits)
    assert gross.metrics["post_gross"] == pytest.approx(0.85)   # both rows counted


@pytest.mark.unit
@pytest.mark.parametrize(
    "others_net, side, weight, expected_action, expected_mult, expected_reason",
    [
        (0.5, "LONG", 0.5, "RESIZE", 0.6, "exceeds_limit"),       # 0.5 + 0.5 > 0.8
        (-0.7, "LONG", 0.4, "PASS", 1.0, "within_limit"),         # hedged: nets -0.3
        (0.5, "SHORT", -0.5, "PASS", 1.0, "within_limit"),        # nets to 0.0
        (0.85, "SHORT", -0.05, "PASS", 1.0, "within_limit"),      # nets exactly 0.80
        # A weakened offsetting position pushes the NET long past the cap:
        # shrinking only deepens it, so no admissible size exists -> VETO.
        (0.85, "SHORT", -0.01, "VETO", 0.0, "opposite_direction_exceeds_limit"),
        # Boundary: the weakened hedge lands the net exactly on 0.80 -> allowed.
        (0.81, "SHORT", -0.01, "PASS", 1.0, "within_limit"),
    ],
)
def test_r16_constraints_directional_nets_and_refuses_a_broken_hedge(
    others_net, side, weight, expected_action, expected_mult, expected_reason,
):
    """Directional caps NET exposure; a hedge reduction that breaks the opposite
    side is refused outright rather than shrunk."""
    decision = constraints.directional_concentration(
        _pos("A", _C, side, weight),
        _snap(100_000.0, _pos("B", _C, "LONG" if others_net > 0 else "SHORT", others_net)),
        constraints.PortfolioLimits(),
    )
    assert (decision.action, decision.multiplier) == (expected_action, pytest.approx(expected_mult))
    assert decision.reasons == [expected_reason]
    if expected_reason == "opposite_direction_exceeds_limit":
        assert decision.metrics["opposite_directional"] == pytest.approx(others_net - 0.01)


@pytest.mark.unit
def test_r16_constraints_directional_flat_side_and_side_weight_disagreement():
    """FLAT adds no direction; a weight pointing against its declared side is
    sized into the OPPOSITE headroom (the tighter multiplier wins)."""
    limits = constraints.PortfolioLimits()
    flat = constraints.directional_concentration(
        _pos("A", _C, "FLAT", 0.0), _snap(100_000.0, _pos("B", _C, "LONG", 0.5)), limits,
    )
    assert (flat.action, flat.multiplier) == ("PASS", 1.0)
    assert flat.metrics["post_directional"] == 0.0
    disagree = constraints.directional_concentration(
        _pos("A", _C, "LONG", -0.5),
        _snap(100_000.0, _pos("B", _C, "SHORT", -0.9)),
        limits,
    )
    # Own side (long) sees 0 exposure, but the short side is already 0.9 over
    # 0.8: the disagreement branch pins the multiplier at 0.
    assert (disagree.action, disagree.multiplier) == ("RESIZE", 0.0)
    assert disagree.reasons == ["book_already_over_limit"]


@pytest.mark.unit
def test_r16_constraints_non_finite_limit_is_a_veto_on_every_rule():
    """A NaN/inf cap cannot be judged: the rule refuses instead of waving through."""
    limits = constraints.PortfolioLimits(max_gross=float("nan"), max_single=float("inf"))
    cand = _pos("A", _C, "LONG", 0.1)
    snap = _snap(100_000.0)
    for rule in (constraints.global_gross, constraints.single_concentration):
        decision = rule(cand, snap, limits)
        assert (decision.action, decision.multiplier, decision.reasons) == (
            "VETO", 0.0, ["non_finite_limit"],
        ), rule.__name__
    directional = constraints.directional_concentration(cand, snap, limits)
    assert directional.action == "PASS"   # max_directional is finite here
    insane = constraints.PortfolioLimits(max_directional=float("nan"))
    vetoed = constraints.directional_concentration(cand, snap, insane)
    assert (vetoed.action, vetoed.reasons) == ("VETO", ["non_finite_limit"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "candidate, snapshot, reason",
    [
        (_pos("A", _C, "LONG", float("nan")), _snap(100.0), "non_finite_candidate_weight"),
        (_pos("A", _C, "SHORT", float("inf")), _snap(100.0), "non_finite_candidate_weight"),
        (_pos("A", _C, "LONG", 0.1), _snap(0.0), "non_positive_equity"),
        (_pos("A", _C, "LONG", 0.1), _snap(-5.0), "non_positive_equity"),
        (_pos("A", _C, "LONG", 0.1), _snap(float("nan")), "non_positive_equity"),
        (_pos("A", _C, "LONG", 0.1),
         _snap(100.0, _pos("B", _C, "LONG", float("nan"))), "non_finite_position_weight"),
    ],
)
def test_r16_constraints_degenerate_inputs_veto_through_the_public_entry(
    candidate, snapshot, reason,
):
    """Non-finite weights / non-positive equity are hard refusals on all rules."""
    decisions = constraints.evaluate_constraints(candidate, snapshot, constraints.PortfolioLimits())
    assert [d.rule for d in decisions] == [
        "global_gross", "asset_class", "single_concentration",
        "directional_concentration", "correlation_cluster",
    ]
    assert all(d.action == "VETO" and d.multiplier == 0.0 for d in decisions)
    assert all(d.reasons == [reason] for d in decisions)


@pytest.mark.unit
def test_r16_constraints_evaluate_runs_in_frozen_order_and_fails_closed(monkeypatch):
    """Order is the contract; a crashing constraint yields VETO constraint_error."""
    decisions = constraints.evaluate_constraints(
        _pos("A", _C, "LONG", 0.1), _snap(100.0), constraints.PortfolioLimits(),
    )
    assert [d.rule for d in decisions] == [
        "global_gross", "asset_class", "single_concentration",
        "directional_concentration", "correlation_cluster",
    ]
    assert [fn.__name__ for fn in constraints.FIVE_HARD_CONSTRAINTS] == [d.rule for d in decisions]

    def _boom(candidate, snapshot, limits):
        raise ZeroDivisionError("risk check exploded")

    original = constraints.FIVE_HARD_CONSTRAINTS
    monkeypatch.setattr(
        constraints, "FIVE_HARD_CONSTRAINTS", (original[0], _boom) + original[2:],
    )
    patched = constraints.evaluate_constraints(
        _pos("A", _C, "LONG", 0.1), _snap(100.0), constraints.PortfolioLimits(),
    )
    assert patched[0].action == "PASS"
    assert (patched[1].action, patched[1].multiplier, patched[1].reasons) == (
        "VETO", 0.0, ["constraint_error"],
    )
    # A broken check never produces an out-of-range multiplier either.
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        constraints.RiskDecision(rule="x", action="RESIZE", multiplier=1.5, reasons=[], metrics={})
    with pytest.raises(ValueError, match="VETO must carry multiplier == 0.0"):
        constraints.RiskDecision(rule="x", action="VETO", multiplier=0.5, reasons=[], metrics={})
    with pytest.raises(ValueError, match="PASS/RESIZE/VETO"):
        constraints.RiskDecision(rule="x", action="HOLD", multiplier=1.0, reasons=[], metrics={})


@pytest.mark.unit
def test_r16_constraints_cluster_budget_uses_the_cluster_map():
    """Only the candidate's cluster shares a budget; unmapped symbols cluster alone."""
    mapped = constraints.PortfolioLimits(
        cluster_of={"A": "meme", "B": "meme", "C": "ai"},
    )
    book = _snap(100_000.0, _pos("A", _C, "LONG", 0.25), _pos("B", _C, "LONG", 0.1))
    # C is alone in "ai": its 0.5 over the 0.4 cap -> x0.8.
    c_new = constraints.correlation_cluster(_pos("C", _C, "LONG", 0.5), book, mapped)
    assert (c_new.action, c_new.multiplier) == ("RESIZE", pytest.approx(0.8))
    # Re-sizing B: A's 0.25 stays in "meme" -> headroom 0.15 for a 0.2 proposal.
    b_resize = constraints.correlation_cluster(_pos("B", _C, "LONG", 0.2), book, mapped)
    assert (b_resize.action, b_resize.multiplier) == ("RESIZE", pytest.approx(0.75))
    assert b_resize.metrics["post_cluster"] == pytest.approx(0.45)
    # Without a map every symbol is its own cluster: B fits.
    alone = constraints.correlation_cluster(_pos("B", _C, "LONG", 0.2), book, constraints.PortfolioLimits())
    assert (alone.action, alone.multiplier) == ("PASS", 1.0)


@pytest.mark.unit
def test_r16_constraints_percentile_beta_pearson_degenerates_match_numpy():
    """Advisory stats refuse to fabricate: empty / misaligned / flat -> None."""
    vals = [-0.05, -0.02, 0.0, 0.01, 0.03]
    for q in (0.0, 0.05, 0.25, 0.5, 0.9, 1.0):
        assert constraints._percentile_linear(vals, q) == pytest.approx(
            float(np.percentile(vals, q * 100.0)),
        )
    assert math.isnan(constraints._percentile_linear([], 0.05))
    assert constraints._percentile_linear([0.02], 0.05) == 0.02
    # beta: misaligned or flat benchmark -> None; exact cov/var otherwise.
    assert constraints._beta([0.01, 0.02, 0.03], [0.01, 0.02]) is None
    assert constraints._beta([0.01, 0.02], [0.0, 0.0]) is None
    # cov 0.0002 / var 0.0002: an exact unit beta.
    assert constraints._beta([0.01, 0.03], [0.02, 0.04]) == pytest.approx(1.0)
    assert constraints._beta([0.01], [0.01]) is None
    # pearson: misaligned / <2 points / zero variance -> None.
    assert constraints._pearson([1.0, 2.0], [1.0]) is None
    assert constraints._pearson([1.0], [1.0]) is None
    assert constraints._pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert constraints._pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert constraints._pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)
    # A zero-variance series keeps its own diagonal but gets no pairwise cell.
    matrix = constraints._correlation_matrix(
        {"A": [1.0, 2.0, 3.0, 4.0], "B": [1.0, 1.0, 1.0, 1.0], "C": [2.0, 4.0, 6.0, 8.0]},
    )
    assert matrix["A"] == {"A": 1.0, "C": pytest.approx(1.0)}
    assert matrix["B"] == {"B": 1.0}
    assert list(matrix) == ["A", "B", "C"]   # sorted for determinism
    assert constraints._correlation_matrix(None) == {}


@pytest.mark.unit
def test_r16_constraints_advisory_metrics_hand_computed():
    """Advisory numbers are exact and NEVER resize (display + archive only)."""
    returns = tuple([0.01, -0.01] * 15)
    btc = [0.02, -0.02] * 15
    snapshot = _snap(
        1000.0,
        _pos("B", "pure_crypto_perp", "SHORT", -0.1),
        returns=returns,
    )
    candidate = _pos("A", "stock_perp", "LONG", 0.4)
    out = constraints.advisory_metrics(
        candidate, snapshot, constraints.PortfolioLimits(), btc_returns=btc,
    )
    stdev = math.sqrt(0.003 / 29.0)          # ddof=1 of the +-1% series
    assert out["usdt_collateral_concentration"] == 1.0
    assert out["venue_exposure"] == 1.0
    assert out["var_95"] == pytest.approx(-0.01)
    assert out["cvar_95"] == pytest.approx(-0.01)      # ceil(30*0.05) = 2 worst
    assert out["beta_vs_btc"] == pytest.approx(0.5)    # cov 0.006 / var 0.012
    assert out["correlation_matrix"] == {}
    assert out["usdt_depeg_shock"] == pytest.approx(-100.0)
    # gross 0.5: -equity * gross * 2 * stdev
    assert out["binance_outage_shock"] == pytest.approx(-1000.0 * 0.5 * 2.0 * stdev)
    # -0.20 * beta(0.5) - 0.08 * stock_perp gross (0.4), scaled by equity.
    assert out["btc_crash_x_earnings_gap"] == pytest.approx(1000.0 * (-0.10 - 0.032))


@pytest.mark.unit
def test_r16_constraints_advisory_refuses_thin_or_unpaired_history():
    """<30 finite returns -> VaR/CVaR None; beta drops non-finite pairs by DATE."""
    flat5 = constraints.advisory_metrics(
        _pos("A", _C, "LONG", 0.1), _snap(100.0, returns=(0.01,) * 5),
        constraints.PortfolioLimits(),
    )
    assert flat5["var_95"] is None and flat5["cvar_95"] is None
    assert flat5["beta_vs_btc"] is None              # no bench series supplied
    assert flat5["btc_crash_x_earnings_gap"] is None  # no beta -> no compound shock
    assert flat5["correlation_matrix"] == {}
    assert flat5["binance_outage_shock"] == 0.0      # constant series: stdev 0
    single = constraints.advisory_metrics(
        _pos("A", _C, "LONG", 0.1), _snap(100.0, returns=(0.01,)),
        constraints.PortfolioLimits(),
    )
    assert single["binance_outage_shock"] is None    # <2 returns: no stdev
    # Pairwise finite filter (round-7): a day is kept only when BOTH sides
    # are finite on that day. The NaN (portfolio, day 2) and the inf (bench,
    # day 3) fall on DIFFERENT days, so both days are dropped -- day 3's
    # healthy 0.03 must not be paired with day 2's 0.02.
    history = (0.01, float("nan"), 0.03, -0.01)
    bench = (0.02, 0.02, float("inf"), -0.02)
    paired = constraints.advisory_metrics(
        _pos("A", _C, "LONG", 0.1), _snap(100.0, returns=history),
        constraints.PortfolioLimits(), btc_returns=bench,
    )
    # Survivors: (0.01, 0.02) and (-0.01, -0.02) only.
    assert paired["beta_vs_btc"] == pytest.approx(0.5)   # exact: cov/var
    assert paired["beta_vs_btc"] == pytest.approx(constraints._beta(
        [0.01, -0.01], [0.02, -0.02],
    ))
    # Had the two sides been filtered independently, day 3's 0.03 would have
    # been date-shifted onto day 2's 0.02 -- a different, fabricated beta.
    assert paired["beta_vs_btc"] != pytest.approx(constraints._beta(
        [0.01, 0.03, -0.01], [0.02, 0.02, -0.02],
    ))


# =========================================================================== #
# resolver: min-multiplier composition (frozen)
# =========================================================================== #
def _dec(rule, action, multiplier, reasons=("within_limit",)):
    return constraints.RiskDecision(
        rule=rule, action=action, multiplier=multiplier, reasons=list(reasons), metrics={},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "decisions, proposed, expected_action, expected_size, expected_rules",
    [
        ([], 3.0, "APPROVED", 3.0, []),                         # no rule ran
        ([_dec("a", "PASS", 1.0), _dec("b", "PASS", 1.0)], 3.0, "APPROVED", 3.0, []),
        ([_dec("a", "PASS", 1.0), _dec("b", "RESIZE", 0.5)], 3.0, "RESIZED", 1.5, ["b"]),
        ([_dec("a", "RESIZE", 0.2), _dec("b", "RESIZE", 0.5)], 3.0, "RESIZED", 0.6, ["a", "b"]),
        ([_dec("a", "VETO", 0.0, ("nope",)), _dec("b", "PASS", 1.0)], 3.0, "VETOED", 0.0, ["a"]),
        ([_dec("a", "RESIZE", 0.5), _dec("b", "VETO", 0.0, ("nope",))], 3.0, "VETOED", 0.0, ["a", "b"]),
        ([_dec("a", "PASS", 1.0)], 0.0, "APPROVED", 0.0, []),
    ],
)
def test_r16_resolver_takes_the_minimum_multiplier(
    decisions, proposed, expected_action, expected_size, expected_rules,
):
    """final = proposed * min(multipliers); any VETO zeroes the size.

    ``risk_decision_rules`` lists EVERY rule that contributed (VETO or a
    multiplier below 1), not only the minimum.
    """
    result = resolver.resolve_constraints(decisions, proposed, "LONG")
    assert result.action == expected_action
    assert result.final_size == pytest.approx(expected_size)
    assert result.final_multiplier == pytest.approx(
        min((d.multiplier for d in decisions), default=1.0),
    )
    assert result.risk_decision_rules == expected_rules
    assert result.side == "LONG"


@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.5])
def test_r16_resolver_rejects_invalid_proposed_size(bad):
    """A non-finite or negative size is vetoed before any constraint math."""
    result = resolver.resolve_constraints([_dec("a", "PASS", 1.0)], bad, "SHORT")
    assert (result.action, result.final_size, result.final_multiplier) == ("VETOED", 0.0, 0.0)
    assert result.reasons == ["invalid_proposed_size"]
    assert result.risk_decision_rules == []


@pytest.mark.unit
def test_r16_resolver_render_lines_are_deterministic():
    """One verdict line, the binding rules, then veto reasons."""
    resized = resolver.resolve_constraints(
        [_dec("a", "PASS", 1.0), _dec("b", "RESIZE", 0.5, ("exceeds_limit",))], 3.0, "LONG",
    )
    assert resolver.render_resolver_lines(resized).splitlines() == [
        "- **Resolver**: RESIZED · multiplier 0.50 · final size 1.5000",
        "- **Binding constraints**: b",
    ]
    vetoed = resolver.resolve_constraints(
        [_dec("a", "VETO", 0.0, ("bad", "worse"))], 3.0, "LONG",
    )
    assert resolver.render_resolver_lines(vetoed).splitlines()[-1] == (
        "- **Veto reasons**: bad; worse"
    )


# =========================================================================== #
# cost_model / tradeability: signed carry legs and the net-edge gate
# =========================================================================== #
@pytest.mark.unit
@pytest.mark.parametrize(
    "asset_type, side, kwargs, expected",
    [
        ("stock", "long", {}, {"entry_fee_bps": 0.0, "exit_fee_bps": 0.0,
                                   "entry_slippage_bps": 5.0, "exit_slippage_bps": 5.0,
                                   "funding_bps": 0.0, "borrow_bps": 0.0}),
        ("stock", "short", {"horizon_days": 36.5},
         {"entry_fee_bps": 0.0, "exit_fee_bps": 0.0, "entry_slippage_bps": 5.0,
              "exit_slippage_bps": 5.0, "funding_bps": 0.0, "borrow_bps": 10.0}),
        ("crypto", "long", {}, {"entry_fee_bps": 10.0, "exit_fee_bps": 10.0,
                                    "entry_slippage_bps": 2.0, "exit_slippage_bps": 2.0,
                                    "funding_bps": 0.0, "borrow_bps": 0.0}),
        ("crypto_spot", "short", {}, {"entry_fee_bps": 10.0, "exit_fee_bps": 10.0,
                                          "entry_slippage_bps": 2.0, "exit_slippage_bps": 2.0,
                                          "funding_bps": 0.0, "borrow_bps": 0.0}),
        # 10%/yr over 36.5 days = 100 bps; a long PAYS it, a short RECEIVES it.
        ("crypto_perp", "long", {"horizon_days": 36.5, "funding_rate_annualized": 0.10},
         {"entry_fee_bps": 5.0, "exit_fee_bps": 5.0, "entry_slippage_bps": 2.0,
              "exit_slippage_bps": 2.0, "funding_bps": 100.0, "borrow_bps": 0.0}),
        ("crypto_perp", "short", {"horizon_days": 36.5, "funding_rate_annualized": 0.10},
         {"entry_fee_bps": 5.0, "exit_fee_bps": 5.0, "entry_slippage_bps": 2.0,
              "exit_slippage_bps": 2.0, "funding_bps": -100.0, "borrow_bps": 0.0}),
        # A short's carry credit can push the whole round trip NEGATIVE (reported).
        ("crypto_perp", "short", {"horizon_days": 365.0, "funding_rate_annualized": 0.10},
         {"entry_fee_bps": 5.0, "exit_fee_bps": 5.0, "entry_slippage_bps": 2.0,
              "exit_slippage_bps": 2.0, "funding_bps": -1000.0, "borrow_bps": 0.0}),
        ("crypto_perp", "long", {"bnb_discount": True},
         {"entry_fee_bps": 4.5, "exit_fee_bps": 4.5, "entry_slippage_bps": 2.0,
              "exit_slippage_bps": 2.0, "funding_bps": 0.0, "borrow_bps": 0.0}),
        # flat "side": every leg is zero (case/whitespace-insensitive side).
        ("crypto_perp", "FLAT", {}, {"entry_fee_bps": 0.0, "exit_fee_bps": 0.0,
                                         "entry_slippage_bps": 0.0, "exit_slippage_bps": 0.0,
                                         "funding_bps": 0.0, "borrow_bps": 0.0}),
        ("stock", " flat ", {}, {"entry_fee_bps": 0.0, "exit_fee_bps": 0.0,
                                     "entry_slippage_bps": 0.0, "exit_slippage_bps": 0.0,
                                     "funding_bps": 0.0, "borrow_bps": 0.0}),
    ],
)
def test_r16_cost_model_legs_hand_computed(asset_type, side, kwargs, expected):
    """Fees+slippage per side; funding/borrow are SIGNED over the horizon."""
    cost = estimate_round_trip_cost(asset_type, side, **kwargs)
    assert cost.as_dict() == pytest.approx({**expected, "total_bps": sum(expected.values())})


@pytest.mark.unit
@pytest.mark.parametrize(
    "args, kwargs, match",
    [
        (("stock", "long"), {"horizon_days": -1.0}, "horizon_days must be >= 0"),
        (("stock", "sideways"), {}, "side must be long/short/flat"),
        (("stock", None), {}, "side must be long/short/flat"),
        (("forex", "long"), {}, "unknown asset_type"),
    ],
)
def test_r16_cost_model_rejects_invalid_inputs(args, kwargs, match):
    """Bad horizon / side / asset type raise instead of pricing a guess."""
    with pytest.raises(ValueError, match=match):
        estimate_round_trip_cost(*args, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "rating, target, expected_side, expected_edge",
    [
        ("Buy", 104.0, "LONG", 0.04),
        ("Overweight", 95.0, "LONG", -0.05),     # a target on the WRONG side is negative
        ("Sell", 96.0, "SHORT", 0.04),
        ("underweight", 101.0, "SHORT", -0.01),
        ("Hold", 110.0, "FLAT", None),
        ("Buy", None, "LONG", None),
        ("Buy", 0.0, "LONG", None),               # zero target: short-rewarding junk
        ("Buy", float("nan"), "LONG", None),
        ("Buy", 4.0, "LONG", None),               # 0.04x: outside the [0.05, 20x] band
        ("Sell", 2001.0, "SHORT", None),          # 20.01x
        ("Buy", 5.0, "LONG", -0.95),              # 0.05x is the inclusive edge
        ("Buy", 2000.0, "LONG", 19.0),            # 20x is the inclusive edge
    ],
)
def test_r16_tradeability_gross_edge_sign_and_sanity_band(
    rating, target, expected_side, expected_edge,
):
    """LONG: target/ref-1; SHORT: 1-target/ref; refuses non-positive / absurd."""
    side = ta.side_from_rating(rating)
    assert (side.value, side) == (expected_side, expected_side)   # StrEnum
    edge = ta.gross_edge(ta.side_from_rating(rating), target, 100.0)
    if expected_edge is None:
        assert edge is None
    else:
        assert edge == pytest.approx(expected_edge)


@pytest.mark.unit
@pytest.mark.parametrize(
    "target, expected_verdict, expected_net",
    [
        (100.25, "TRADEABLE", 1.0),       # 25 bps gross - 14 cost - 10 buffer
        (100.24, "NO_TRADE", 0.0),        # edge == cost + buffer: not worth executing
        (100.2399, "NO_TRADE", -0.01),
        (99.0, "NO_TRADE", -124.0),
    ],
)
def test_r16_tradeability_net_edge_is_gross_minus_cost_minus_buffer(
    target, expected_verdict, expected_net,
):
    """net = gross_bps - total_bps - risk_buffer; <= 0 (with 1e-9 eps) -> NO_TRADE."""
    cost = estimate_round_trip_cost("crypto_perp", "long", funding_rate_annualized=None)
    assert cost.total_bps == pytest.approx(14.0)       # 2x5 fee + 2x2 slippage
    assert ta.RISK_BUFFER_BPS == 10.0
    verdict = ta.evaluate_tradeability(
        rating="Buy", target_price=target, reference_price=100.0, cost=cost,
    )
    assert verdict.tradeability.value == expected_verdict
    assert verdict.net_edge_bps == pytest.approx(expected_net, abs=1e-9)
    assert verdict.estimated_cost_bps == pytest.approx(14.0)


@pytest.mark.unit
def test_r16_tradeability_rule_order_and_fail_closed_on_nan_cost():
    """FLAT beats everything; a NaN cost is NO_TRADE, never TRADEABLE."""
    good_cost = estimate_round_trip_cost("crypto_perp", "long")
    nan_cost = estimate_round_trip_cost(
        "crypto_perp", "long", horizon_days=5.0, funding_rate_annualized=float("nan"),
    )
    assert math.isnan(nan_cost.total_bps)
    verdict = ta.evaluate_tradeability(
        rating="Buy", target_price=110.0, reference_price=100.0, cost=nan_cost,
    )
    assert verdict.tradeability.value == "NO_TRADE"
    assert verdict.tradeability_reason == ta.REASON_COST_NON_FINITE

    def _run(rating, target, cost, tier=ta.TIER_GOOD):
        return ta.evaluate_tradeability(
            rating=rating, target_price=target, reference_price=100.0, cost=cost,
            quality_tier=tier,
        )

    # Order: FLAT side is UNEVALUATED even under an INVALID quality tier ...
    assert (_run("Hold", 110.0, nan_cost, ta.TIER_INVALID).tradeability_reason
            == ta.REASON_HOLD)
    # ... an INVALID tier outranks a missing target ...
    assert (_run("Buy", None, good_cost, ta.TIER_INVALID).tradeability_reason
            == ta.REASON_DATA_INVALID)
    assert (_run("Buy", None, good_cost, ta.TIER_DEGRADED_CRITICAL).tradeability_reason
            == ta.REASON_CRITICAL_MISSING)
    # ... then a missing target, then a garbage target, then the cost check.
    assert _run("Buy", None, good_cost).tradeability_reason == ta.REASON_NO_TARGET
    assert _run("Buy", 0.0, good_cost).tradeability_reason == ta.REASON_INVALID_TARGET
    assert _run("Buy", 2.0, good_cost).tradeability_reason == ta.REASON_INVALID_TARGET
    assert _run("Buy", 110.0, good_cost).tradeability.value == "TRADEABLE"


@pytest.mark.unit
def test_r16_engine_fill_matches_cost_model_single_leg():
    """One engine fill costs exactly entry_fee + entry_slippage from cost_model."""
    result = _run_stock(
        _ScriptedGraph({_MONTH[0]: "Buy"}), [_MONTH[0]],
        taker_bps=BINANCE_USDT_M_TAKER_BPS, slippage_bps=CRYPTO_SLIPPAGE_BPS,
    )
    single_leg = estimate_round_trip_cost("crypto_perp", "long")
    expected_bps = single_leg.entry_slippage_bps + single_leg.entry_fee_bps * (
        1.0 + CRYPTO_SLIPPAGE_BPS / 1e4
    )
    assert result.equity[1] == pytest.approx(100_000.0 * (1.0 - expected_bps / 1e4))
    assert result.trades[0].transaction_cost == pytest.approx(
        100_000.0 * (1.0 + CRYPTO_SLIPPAGE_BPS / 1e4) * BINANCE_USDT_M_TAKER_BPS / 1e4,
    )
    # The engine's BNB knob and the cost model's apply the same 10% discount.
    discounted = _run_stock(
        _ScriptedGraph({_MONTH[0]: "Buy"}), [_MONTH[0]],
        taker_bps=BINANCE_USDT_M_TAKER_BPS, slippage_bps=CRYPTO_SLIPPAGE_BPS,
        bnb_discount=True,
    )
    assert discounted.trades[0].transaction_cost == pytest.approx(
        result.trades[0].transaction_cost * (1.0 - BNB_FEE_DISCOUNT),
    )


# =========================================================================== #
# constraints: a candidate that lands the book EXACTLY on the limit passes
# =========================================================================== #
@pytest.mark.unit
@pytest.mark.parametrize(
    "others, contribution, limit",
    [
        (0.5, 0.1, 0.6),     # 0.6 - 0.5 = 0.0999...98 in binary floating point
        (0.3, 0.2, 0.5),     # 0.5 - 0.3 = 0.1999...98
        (0.7, 0.2, 0.9),     # 0.9 - 0.7 = 0.2000...02 (rounds the other way)
        (0.05, 0.05, 0.1),
    ],
)
def test_r16_constraints_at_limit_candidate_passes_not_resized_by_float_dust(
    others, contribution, limit,
):
    """A candidate that fills the book to exactly the limit in REAL arithmetic
    is PASS at full size.

    ``limit - others`` carries binary rounding dust (``0.6 - 0.5`` is
    ``0.0999...98``, below ``0.1``), so the strict ``>=`` comparison sent the
    candidate down the clip path as ``RESIZE x0.9999999999999998`` with the
    reason ``exceeds_limit`` -- an at-limit trade reported as clipped by a
    binding constraint (verified against the unmodified v1.0.0 checkout).
    """
    decision = constraints._limit_decision(
        "max_single", "single", "limit", others, contribution, limit,
    )
    assert decision.action == "PASS"
    assert decision.multiplier == 1.0
    assert decision.reasons == ["within_limit"]


@pytest.mark.unit
@pytest.mark.parametrize("excess", [1e-6, 1e-5, 1e-3, 0.05])
def test_r16_constraints_tolerance_does_not_admit_a_real_breach(excess):
    """The 1e-12 tolerance is a float-dust allowance, not slack: a candidate
    that really exceeds the limit -- by as little as 1e-6 -- is still clipped
    to the headroom ratio."""
    limit, others = 0.6, 0.5
    contribution = (limit - others) + excess
    decision = constraints._limit_decision(
        "max_single", "single", "limit", others, contribution, limit,
    )
    assert decision.action == "RESIZE"
    assert decision.reasons == ["exceeds_limit"]
    assert decision.multiplier == pytest.approx((limit - others) / contribution)
    assert decision.multiplier < 1.0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q", "-p no:cacheprovider"])
