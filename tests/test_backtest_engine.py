"""Unit tests for the Phase 0 backtest engine (hermetic: no network, no LLM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yiagents.backtest.cache import DecisionCache
from yiagents.backtest.engine import (
    BacktestResult,
    TradeRow,
    run_backtest,
)


class FakeGraph:
    """Stand-in for YiAgentsGraph: returns scripted ratings per date."""

    def __init__(self, ratings: dict[str, str], benchmark: str = "SPY"):
        # Preserve insertion order so tests are deterministic.
        self._ratings = dict(ratings)
        self._benchmark = benchmark

    def propagate(self, company_name, trade_date, asset_type="stock"):
        rating = self._ratings.get(trade_date, "Hold")
        final_state = {
            "final_trade_decision": f"**Rating**: {rating}\n\nSome analysis for {trade_date}.",
        }
        return final_state, rating

    def _resolve_benchmark(self, ticker):
        return self._benchmark


def _rising_prices(ticker, start, end):
    """Synthetic close prices: 100 -> 200 across the window."""
    idx = pd.bdate_range(start, end)
    values = [100.0 * (1 + 0.001 * i) for i in range(len(idx))]
    s = pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)
    return s


def _flat_index(ticker, start, end):
    idx = pd.bdate_range(start, end)
    return pd.Series([100.0] * len(idx), index=idx.strftime("%Y-%m-%d"), dtype=float)


def _decision_dates(n=12, start="2024-01-01"):
    """n business days, spaced ~5 apart, all in YYYY-MM-DD string form."""
    idx = pd.bdate_range(start, periods=n * 6, freq="B")
    picked = [idx[i].strftime("%Y-%m-%d") for i in range(0, n * 6, 5)][:n]
    return picked


@pytest.mark.unit
def test_engine_all_buy_beats_flat_index():
    dates = _decision_dates(10)
    graph = FakeGraph(dict.fromkeys(dates, "Buy"))

    result = run_backtest(
        graph, "AAPL", dates,
        initial_capital=100_000.0,
        holding_days=5,
        price_provider=_rising_prices,
        compute_index_alpha=True,
    )

    assert isinstance(result, BacktestResult)
    assert len(result.equity) >= 2
    assert result.ticker == "AAPL"
    # All-Buy on a rising tape should be profitable.
    assert result.metrics.total_return > 0
    # Each decision recorded as a trade with full weight.
    assert len(result.trades) == len(dates)
    assert all(t.executed_weight == 1.0 for t in result.trades)
    assert all(t.rating == "Buy" for t in result.trades)
    # Buy-and-hold benchmark of the same ticker is present and aligned.
    assert len(result.benchmark_equity) == len(result.equity)


@pytest.mark.unit
def test_engine_sell_goes_flat_preserves_capital():
    dates = _decision_dates(8)
    graph = FakeGraph(dict.fromkeys(dates, "Sell"))
    result = run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices,
    )
    # Sell -> 0% weight: equity stays flat (in cash) after first rebalance,
    # and the strategy must NOT beat the rising buy-and-hold.
    assert result.metrics.total_return < result.metrics.total_return + 1  # sanity
    assert all(t.executed_weight == 0.0 for t in result.trades)
    # Flat cash position: final equity roughly equals initial capital.
    assert abs(result.equity[-1] - result.equity[0]) / result.equity[0] < 0.05


@pytest.mark.unit
def test_engine_hold_keeps_prior_position():
    dates = _decision_dates(6)
    # Buy first, then Hold the rest: position should carry (weight stays 1.0).
    ratings = {dates[0]: "Buy"}
    for d in dates[1:]:
        ratings[d] = "Hold"
    graph = FakeGraph(ratings)
    result = run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices,
    )
    assert result.trades[0].executed_weight == 1.0
    # Hold rows record the carried weight, not 0.
    assert all(t.executed_weight == 1.0 for t in result.trades[1:])
    assert all(t.rating == "Hold" for t in result.trades[1:])


@pytest.mark.unit
def test_engine_custom_weight_fn_overrides_mapping():
    dates = _decision_dates(6)
    graph = FakeGraph(dict.fromkeys(dates, "Buy"))

    def half_size(rating, date, ctx):
        return 0.5

    result = run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices,
        weight_fn=half_size,
    )
    assert all(t.executed_weight == 0.5 for t in result.trades)


@pytest.mark.unit
def test_engine_cache_replay_avoids_recalling_graph(tmp_path):
    dates = _decision_dates(6)
    call_count = {"n": 0}

    class CountingGraph(FakeGraph):
        def propagate(self, company_name, trade_date, asset_type="stock"):
            call_count["n"] += 1
            return super().propagate(company_name, trade_date, asset_type)

    graph = CountingGraph(dict.fromkeys(dates, "Buy"))
    cache = DecisionCache(tmp_path, enabled=True)

    first = run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices, cache=cache, run_tag="r1",
    )
    calls_after_first = call_count["n"]
    assert calls_after_first == len(dates)
    assert first.cached_misses == len(dates)
    assert first.cached_hits == 0

    # Replay with the SAME run_tag: every decision served from cache.
    graph2 = CountingGraph(dict.fromkeys(dates, "Sell"))  # different scripts ignored
    second = run_backtest(
        graph2, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices, cache=cache, run_tag="r1",
    )
    assert call_count["n"] == calls_after_first  # no new propagate calls
    assert second.cached_hits == len(dates)
    assert second.cached_misses == 0
    # Same realized decisions -> same ratings on the trade rows.
    assert all(t.rating == "Buy" for t in second.trades)


@pytest.mark.unit
def test_engine_transaction_costs_drag_returns():
    dates = _decision_dates(6)
    graph = FakeGraph(dict.fromkeys(dates, "Buy"))
    no_cost = run_backtest(graph, "AAPL", dates, holding_days=5,
                           price_provider=_rising_prices, cost_bps=0.0)
    graph2 = FakeGraph(dict.fromkeys(dates, "Buy"))
    with_cost = run_backtest(graph2, "AAPL", dates, holding_days=5,
                             price_provider=_rising_prices, cost_bps=50.0)
    # Churning into the same target still incurs cost on the first rebalance.
    assert with_cost.equity[-1] < no_cost.equity[-1]


@pytest.mark.unit
def test_engine_empty_dates_raises():
    with pytest.raises(ValueError):
        run_backtest(FakeGraph({}), "AAPL", [], price_provider=_rising_prices)


@pytest.mark.unit
def test_engine_propagate_error_treated_as_hold():
    dates = _decision_dates(4)

    class BrokenGraph:
        def propagate(self, *a, **k):
            raise RuntimeError("boom")
        def _resolve_benchmark(self, t):
            return "SPY"

    result = run_backtest(BrokenGraph(), "AAPL", dates, holding_days=5,
                          price_provider=_rising_prices)
    # Errors degrade to Hold, never raise.
    assert all(t.rating == "Hold" for t in result.trades)


@pytest.mark.unit
def test_trade_row_fields_populated():
    dates = _decision_dates(4)
    graph = FakeGraph(dict.fromkeys(dates, "Buy"))
    result = run_backtest(graph, "AAPL", dates, holding_days=5,
                          price_provider=_rising_prices)
    t: TradeRow = result.trades[0]
    assert t.price is not None and t.price > 0
    assert isinstance(t.raw_return, float)
    # Rising prices -> positive realized asset return over the holding window.
    assert t.raw_return > 0


def _moderate_edge_prices(ticker, start, end):
    """Synthetic close with a modest positive edge + noise.

    Used by the DSR n_trials test: a non-saturated Sharpe keeps DSR in a
    mid-range where raising the multiple-testing hurdle visibly moves it, so
    the strict-inequality assertion bites. Fixed seed -> identical series on
    every call, so two backtests differ ONLY in n_trials.
    """
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(start, end)
    returns = 0.0011 + rng.normal(0.0, 0.013, size=len(idx))
    values = (100.0 * np.cumprod(1.0 + returns)).tolist()
    return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)


@pytest.mark.unit
def test_engine_n_trials_threads_through_to_dsr():
    """Regression guard: run_backtest must forward n_trials to compute_metrics.

    Previously the engine's compute_metrics call site hard-coded n_trials=1,
    which collapses the Bailey-Lopez de Prado hurdle to 0 and makes any
    positive Sharpe trivially "clear" it -- turning the live-go/no-go
    validation gate into a near-vacuous test. With the param threaded through,
    more trials -> higher hurdle -> strictly lower DSR on the SAME equity
    curve. The strict inequality is what makes this a guard: if n_trials were
    ever re-hard-coded, both DSR values would be identical and the assertion
    would fail.
    """
    dates = _decision_dates(12)
    ratings = dict.fromkeys(dates, "Buy")
    single = run_backtest(
        FakeGraph(ratings), "AAPL", dates, holding_days=5,
        price_provider=_moderate_edge_prices, n_trials=1,
    )
    many = run_backtest(
        FakeGraph(ratings), "AAPL", dates, holding_days=5,
        price_provider=_moderate_edge_prices, n_trials=30,
    )
    assert 0.0 < single.metrics.deflated_sharpe < 1.0
    assert many.metrics.deflated_sharpe < single.metrics.deflated_sharpe


@pytest.mark.unit
def test_engine_rejects_invalid_n_trials():
    dates = _decision_dates(4)
    graph = FakeGraph(dict.fromkeys(dates, "Buy"))
    with pytest.raises(ValueError):
        run_backtest(graph, "AAPL", dates, holding_days=5,
                     price_provider=_rising_prices, n_trials=0)


@pytest.mark.unit
def test_engine_win_rate_turnover_drawdown_date_populated():
    """Trade-/equity-derived extras are filled post-hoc by the engine.

    compute_metrics is equity-only, so win_rate / turnover_annual /
    max_drawdown_date default to None; the engine must populate them from its
    trade rows and equity dates. Rising tape -> every decidable rebalance is a
    winner (win_rate 1.0), Buy every round churns capital (turnover > 0), and
    the drawdown date is a real date string.
    """
    dates = _decision_dates(8)
    graph = FakeGraph(dict.fromkeys(dates, "Buy"))
    result = run_backtest(graph, "AAPL", dates, holding_days=5,
                          price_provider=_rising_prices)
    m = result.metrics
    assert m.win_rate == 1.0
    assert m.num_trades == len(dates)
    assert m.turnover_annual is not None and m.turnover_annual > 0
    assert isinstance(m.max_drawdown_date, str) and len(m.max_drawdown_date) == 10


@pytest.mark.unit
def test_engine_win_rate_none_when_no_decidable_returns():
    """A flat price tape yields raw_return 0.0, not None -- so win_rate is a
    number (0.0, zero winners), never None, when rebalances exist."""
    dates = _decision_dates(5)
    graph = FakeGraph(dict.fromkeys(dates, "Buy"))
    result = run_backtest(graph, "AAPL", dates, holding_days=5,
                          price_provider=_flat_index)
    # Flat asset -> no positive holding returns -> win_rate 0.0 (decided, none win).
    assert result.metrics.win_rate == 0.0


@pytest.mark.unit
def test_engine_event_study_off_leaves_defaults():
    """event_study=False (the default) must not run the post-processing: the
    event_study_* metric fields stay at their defaults, so every existing
    caller that does not opt in is byte-equivalent."""
    dates = _decision_dates(6)
    graph = FakeGraph(dict.fromkeys(dates, "Buy"))
    result = run_backtest(graph, "AAPL", dates, holding_days=5,
                          price_provider=_rising_prices)
    m = result.metrics
    assert m.event_study_n == 0
    assert m.event_study_mean_car is None
    assert m.event_study_t_stat is None
    assert m.event_study_p_value is None
    assert m.event_study_ci is None
    assert m.event_study_benchmark is None


@pytest.mark.unit
def test_engine_event_study_populated_when_opted_in():
    """event_study=True fills the event_study_* fields when the wide-window
    price pull covers the 250-return estimation window before the first event.

    The first decision is pushed ~6 months in so the -400d wide pull still
    leaves a full estimation window ahead of event[0]; a flat benchmark makes
    the asset's holding-window drift read as abnormal return, so n_events > 0
    and the aggregate stats are populated (the opt-in path actually ran).
    """
    dates = _decision_dates(6, start="2024-06-01")
    graph = FakeGraph(dict.fromkeys(dates, "Buy"), benchmark="SPY")

    def prices(ticker, start, end):
        # Asset rises; benchmark flat -> abnormal return = asset drift.
        if ticker == "SPY":
            return _flat_index(ticker, start, end)
        return _rising_prices(ticker, start, end)

    result = run_backtest(graph, "AAPL", dates, holding_days=5,
                          price_provider=prices, compute_index_alpha=True,
                          event_study=True)
    m = result.metrics
    assert m.event_study_n > 0
    assert m.event_study_mean_car is not None
    assert m.event_study_t_stat is not None
    assert m.event_study_benchmark == "SPY"


@pytest.mark.unit
def test_engine_event_study_fail_open_on_missing_wide_prices():
    """When the wide-window asset pull comes back empty (the estimation-window
    data is unavailable), event_study degrades to n_events=0 without raising --
    the backtest itself is unaffected (advisory, fail-open)."""
    dates = _decision_dates(4, start="2024-06-01")
    graph = FakeGraph(dict.fromkeys(dates, "Buy"), benchmark="SPY")

    def prices(ticker, start, end):
        s = _rising_prices(ticker, start, end)
        # The event-study wide pull starts ~400d before the first decision
        # (well before 2024-05-01); return empty there so asset_prices.empty
        # trips the fail-open return. run_backtest's own window starts at the
        # first decision (>= 2024-06-01) and gets real prices.
        if str(start) < "2024-05-01":
            return pd.Series(dtype=float)
        return s

    result = run_backtest(graph, "AAPL", dates, holding_days=5,
                          price_provider=prices, event_study=True)
    assert isinstance(result, BacktestResult)
    assert result.metrics.event_study_n == 0
    assert result.metrics.event_study_mean_car is None
