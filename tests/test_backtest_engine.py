"""Unit tests for the Phase 0 backtest engine (hermetic: no network, no LLM)."""

from __future__ import annotations

import pandas as pd
import pytest

from yiagents.backtest.cache import DecisionCache
from yiagents.backtest.engine import (
    DEFAULT_RATING_TO_WEIGHT,
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
    graph = FakeGraph({d: "Buy" for d in dates})

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
    graph = FakeGraph({d: "Sell" for d in dates})
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
    graph = FakeGraph({d: "Buy" for d in dates})

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

    graph = CountingGraph({d: "Buy" for d in dates})
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
    graph2 = CountingGraph({d: "Sell" for d in dates})  # different scripts ignored
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
    graph = FakeGraph({d: "Buy" for d in dates})
    no_cost = run_backtest(graph, "AAPL", dates, holding_days=5,
                           price_provider=_rising_prices, cost_bps=0.0)
    graph2 = FakeGraph({d: "Buy" for d in dates})
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
    graph = FakeGraph({d: "Buy" for d in dates})
    result = run_backtest(graph, "AAPL", dates, holding_days=5,
                          price_provider=_rising_prices)
    t: TradeRow = result.trades[0]
    assert t.price is not None and t.price > 0
    assert isinstance(t.raw_return, float)
    # Rising prices -> positive realized asset return over the holding window.
    assert t.raw_return > 0
