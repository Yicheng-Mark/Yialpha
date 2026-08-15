"""C8 regression: ``(close, atr)`` for the risk overlay is memoized per
``(ticker, trade_date)``.

Backtests / A-B legs hit the same (ticker, date) repeatedly; recomputing the
14-period stockstats ATR each time was pure waste. The memo key includes the
date, so point-in-time correctness is preserved (a new analysis date
re-computes). Failures are not memoized, so transient vendor faults retry.

Hermetic: loader + ATR computation monkeypatched, no network.
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

import yiagents.graph.trading_graph as tg


def _synthetic_frame() -> pd.DataFrame:
    n = 40
    return pd.DataFrame({
        "Open": [100.0] * n,
        "High": [101.0] * n,
        "Low": [99.0] * n,
        "Close": [100.0 + i * 0.1 for i in range(n)],
    })


@pytest.fixture(autouse=True)
def _clear_memo():
    tg._memoized_close_and_atr.cache_clear()
    yield
    tg._memoized_close_and_atr.cache_clear()


@pytest.mark.unit
def test_close_and_atr_computed_once_per_ticker_date():
    calls = {"n": 0}

    def fake_loader(ticker, date):
        calls["n"] += 1
        return _synthetic_frame()

    with (
        mock.patch("yiagents.dataflows.stockstats_utils.load_ohlcv", fake_loader),
        mock.patch(
            "yiagents.risk.atr_stop.latest_atr_from_frame",
            return_value=(104.9, 2.5),
        ),
    ):
        first = tg._memoized_close_and_atr("AAPL", "2026-06-01")
        second = tg._memoized_close_and_atr("AAPL", "2026-06-01")
        other_date = tg._memoized_close_and_atr("AAPL", "2026-06-02")

    assert first == (104.9, 2.5)
    assert second == first
    assert other_date == first
    # 2 distinct keys -> exactly 2 loads; the repeated date hit the memo.
    assert calls["n"] == 2


@pytest.mark.unit
def test_method_delegates_and_converts_none_on_failure(caplog):
    """The instance method keeps its (None, None) fail-soft contract."""
    import logging

    calls = {"n": 0}

    def boom(ticker, date):
        calls["n"] += 1
        raise RuntimeError("vendor down")

    with (
        mock.patch("yiagents.dataflows.stockstats_utils.load_ohlcv", boom),
        caplog.at_level(logging.WARNING),
    ):
        out1 = tg.YiAgentsGraph._latest_close_and_atr(object(), "MSFT", "2026-01-05")
        out2 = tg.YiAgentsGraph._latest_close_and_atr(object(), "MSFT", "2026-01-05")

    assert out1 == (None, None)
    assert out2 == (None, None)
    # Failures are NOT memoized: the vendor was retried both times.
    assert calls["n"] == 2
    assert any("price/ATR" in r.message for r in caplog.records)


@pytest.mark.unit
def test_distinct_tickers_do_not_share_memo_entries():
    seen: list[tuple[str, str]] = []

    def fake_loader(ticker, date):
        seen.append((ticker, date))
        return _synthetic_frame()

    with (
        mock.patch("yiagents.dataflows.stockstats_utils.load_ohlcv", fake_loader),
        mock.patch(
            "yiagents.risk.atr_stop.latest_atr_from_frame",
            side_effect=lambda frame: (10.0, 1.0),
        ),
    ):
        tg._memoized_close_and_atr("AAPL", "2026-06-01")
        tg._memoized_close_and_atr("NVDA", "2026-06-01")

    assert set(seen) == {("AAPL", "2026-06-01"), ("NVDA", "2026-06-01")}
