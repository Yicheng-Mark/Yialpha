"""Symbol normalization must apply on every yfinance path, not just price fetch.

Regression tests for #983 (instrument identity), #984 (reflection returns), and
the news path: a broker-style crypto symbol like BTCUSD must resolve to the
same Yahoo symbol (BTC-USD) that the price path uses, so identity,
realized-return, and news lookups hit the right instrument instead of
failing/mismatching.
"""
import pandas as pd
import pytest

import yialpha.agents.utils.agent_utils as au
import yialpha.dataflows.y_finance as yfin
import yialpha.dataflows.yfinance_news as ynews
from yialpha.graph.trading_graph import YiAlphaGraph


def test_identity_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        @property
        def info(self):
            return {"longName": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY"}

    monkeypatch.setattr(au.yf, "Ticker", FakeTicker)
    au.resolve_instrument_identity.cache_clear()

    identity = au.resolve_instrument_identity("BTCUSD")

    assert seen["symbol"] == "BTC-USD"  # normalized, not the raw broker symbol
    assert identity.get("company_name") == "Bitcoin USD"


def test_fetch_returns_normalizes_symbol(monkeypatch):
    queried = []

    class FakeTicker:
        def __init__(self, symbol):
            queried.append(symbol)

        def history(self, *args, **kwargs):
            return pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]})

    # _fetch_returns fetches through the vendor layer's cached history
    # wrapper (y_finance), so that is the yfinance entry point to patch.
    monkeypatch.setattr(yfin.yf, "Ticker", FakeTicker)

    # _fetch_returns does not use ``self``; call unbound to avoid building the graph.
    raw, alpha, days = YiAlphaGraph._fetch_returns(
        None, "BTCUSD", "2025-01-02", holding_days=5, benchmark="SPY"
    )

    assert queried[0] == "BTC-USD"  # crypto symbol normalized (#984)
    assert queried[1] == "SPY"   # benchmark left as the canonical symbol
    assert raw is not None and days is not None


def test_news_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def get_news(self, count):
            return []

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ynews, "yf_retry", lambda fn, **kw: fn())

    # PR5 typed soft-miss: an EMPTY result raises NoMarketDataError (carrying
    # both the user's ticker and the canonical symbol) instead of returning
    # an "No news found" string — a plain string counted as success and
    # masked the router's alpha_vantage fallback.
    from yialpha.dataflows.errors import NoMarketDataError

    with pytest.raises(NoMarketDataError) as excinfo:
        ynews.get_news_yfinance("BTCUSD", "2025-01-01", "2025-01-10")

    assert seen["symbol"] == "BTC-USD"   # news queried with the canonical symbol
    assert "BTCUSD" in str(excinfo.value)   # the user's ticker stays in the error
    assert "BTC-USD" in str(excinfo.value)  # provenance noted
