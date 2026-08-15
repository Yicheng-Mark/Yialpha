"""C1 regression: degraded decisions must not pollute the DecisionCache.

When ``propagate`` raises, the engine fabricates a Hold so one bad date does
not abort the backtest. That fake Hold must NEVER be written to the
DecisionCache — otherwise a replay would serve the fabricated Hold as a
``was_degraded=False`` cache hit, permanently miscounting a network fault as
a genuine agent judgement and distorting win-rate / DSR.

Hermetic: no network, no LLM (fake graphs + synthetic prices only).
"""

from __future__ import annotations

import pandas as pd
import pytest

from yiagents.backtest.cache import DecisionCache
from yiagents.backtest.engine import run_backtest


def _rising_prices(ticker: str, start: str, end: str) -> pd.Series:
    idx = pd.bdate_range(start, end)
    values = [100.0 * (1 + 0.001 * i) for i in range(len(idx))]
    return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)


def _decision_dates(n: int = 4, start: str = "2024-01-01") -> list[str]:
    idx = pd.bdate_range(start, periods=n * 6, freq="B")
    return [idx[i].strftime("%Y-%m-%d") for i in range(0, n * 6, 5)][:n]


class _FlakyGraph:
    """Propagate raises while ``broken`` is set; otherwise scripts a rating."""

    def __init__(self, rating: str = "Buy") -> None:
        self.rating = rating
        self.broken = True
        self.calls = 0

    def propagate(self, company_name, trade_date, asset_type="stock"):
        self.calls += 1
        if self.broken:
            raise RuntimeError("simulated network fault")
        return (
            {"final_trade_decision": f"**Rating**: {self.rating}\n\nanalysis"},
            self.rating,
        )

    def _resolve_benchmark(self, ticker):
        return "SPY"


@pytest.mark.unit
def test_propagate_failure_is_not_cached(tmp_path):
    """A propagate failure degrades to Hold but must not touch the cache."""
    dates = _decision_dates()
    graph = _FlakyGraph()
    cache = DecisionCache(tmp_path, enabled=True)

    first = run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices, cache=cache, run_tag="r1",
    )
    assert first.degraded_decision_count == len(dates)
    assert all(t.rating == "Hold" for t in first.trades)
    # Nothing was remembered: every key misses on a fresh lookup.
    for d in dates:
        assert cache.get("AAPL", d, "r1") is None

    # Heal the graph: the replay retries (does not serve the fake Hold) and
    # now caches the genuine decision.
    graph.broken = False
    second = run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices, cache=cache, run_tag="r1",
    )
    assert graph.calls == 2 * len(dates)  # retried every date
    assert second.degraded_decision_count == 0
    assert all(t.rating == "Buy" for t in second.trades)
    for d in dates:
        assert cache.get("AAPL", d, "r1") is not None


@pytest.mark.unit
def test_genuine_decision_still_cached_after_fix(tmp_path):
    """Unrelated cache behaviour is unchanged: genuine decisions ARE cached."""
    dates = _decision_dates()
    graph = _FlakyGraph(rating="Sell")
    graph.broken = False
    cache = DecisionCache(tmp_path, enabled=True)

    run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices, cache=cache, run_tag="r1",
    )
    calls_after_first = graph.calls
    for d in dates:
        assert cache.get("AAPL", d, "r1") is not None

    run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices, cache=cache, run_tag="r1",
    )
    assert graph.calls == calls_after_first  # replay served from cache


@pytest.mark.unit
def test_unparseable_rating_is_cached_but_counted_degraded(tmp_path):
    """A genuine decision whose rating text is garbage stays cacheable.

    propagate succeeded, so the decision markdown is real agent output; the
    engine degrades the rating to Hold but must still remember the decision so
    a replay does not re-bill the LLM.
    """
    dates = _decision_dates(2)

    class GarbageRatingGraph(_FlakyGraph):
        def propagate(self, company_name, trade_date, asset_type="stock"):
            self.calls += 1
            return {"final_trade_decision": "real analysis text"}, 12345  # not a str

    graph = GarbageRatingGraph()
    cache = DecisionCache(tmp_path, enabled=True)

    result = run_backtest(
        graph, "AAPL", dates, holding_days=5,
        price_provider=_rising_prices, cache=cache, run_tag="r1",
    )
    assert result.degraded_decision_count == len(dates)
    assert all(t.rating == "Hold" for t in result.trades)
    for d in dates:
        cached = cache.get("AAPL", d, "r1")
        assert cached is not None
        assert cached.rating == "Hold"
