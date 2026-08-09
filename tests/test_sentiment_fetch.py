"""Byte-equivalence guard for the sentiment source fetch.

``_fetch_sentiment_sources`` fans the three independent fetches (Yahoo news /
StockTwits / Reddit) across a thread pool when ``YIAGENTS_SENTIMENT_PARALLEL_FETCH``
is on, and runs them sequentially when off. Each block lands in a fixed slot,
so the two paths must produce identical results -- this test pins that.
"""
import pytest

import yiagents.agents.analysts.sentiment_analyst as sent
from yiagents.dataflows.config import get_config, set_config


class _Stub:
    """Stand-in for ``get_news`` (accessed as ``get_news.func(...)``)."""

    def __init__(self, payload: str):
        self._payload = payload

    @property
    def func(self):
        return lambda *a, **k: self._payload


def _stub_sources(monkeypatch):
    monkeypatch.setattr(sent, "get_news", _Stub("NEWS"))
    monkeypatch.setattr(
        sent, "fetch_stocktwits_messages", lambda ticker, limit=30: "STOCKTWITS"
    )
    monkeypatch.setattr(sent, "fetch_reddit_posts", lambda ticker: "REDDIT")


@pytest.mark.unit
def test_sequential_fetch_returns_all_three(monkeypatch):
    _stub_sources(monkeypatch)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    out = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    assert out == ("NEWS", "STOCKTWITS", "REDDIT")


@pytest.mark.unit
def test_parallel_fetch_byte_equivalent_to_sequential(monkeypatch):
    _stub_sources(monkeypatch)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    sequential = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", True)
    parallel = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    # Fixed slots: completion order must not change the assembled result.
    assert parallel == sequential == ("NEWS", "STOCKTWITS", "REDDIT")


@pytest.mark.unit
def test_parallel_fetch_workers_inherit_dataflow_config(monkeypatch):
    seen = []

    def source(name):
        def fetch(*args, **kwargs):
            seen.append(get_config()["context_probe"])
            return name

        return fetch

    set_config({"context_probe": 987654})
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", True)
    monkeypatch.setattr(sent, "_get_news_impl", source("NEWS"))
    monkeypatch.setattr(sent, "fetch_stocktwits_messages", source("STOCKTWITS"))
    monkeypatch.setattr(sent, "fetch_reddit_posts", source("REDDIT"))

    result = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")

    assert result == ("NEWS", "STOCKTWITS", "REDDIT")
    assert seen == [987654, 987654, 987654]


@pytest.mark.unit
def test_historical_fetch_omits_current_social_feeds(monkeypatch):
    _stub_sources(monkeypatch)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: True)

    def _must_not_fetch(*_args, **_kwargs):
        raise AssertionError("current social endpoint called during historical run")

    monkeypatch.setattr(sent, "fetch_stocktwits_messages", _must_not_fetch)
    monkeypatch.setattr(sent, "fetch_reddit_posts", _must_not_fetch)
    out = sent._fetch_sentiment_sources("AAPL", "2020-01-01", "2020-01-08")

    assert out[0] == "NEWS"
    assert "historical analysis" in out[1]
    assert "historical analysis" in out[2]


@pytest.mark.unit
def test_default_flag_is_off(monkeypatch):
    # Re-import-free: the module constant must default to False (byte-equivalent
    # to today's sequential behaviour) unless the env var is set.
    import importlib

    monkeypatch.delenv("YIAGENTS_SENTIMENT_PARALLEL_FETCH", raising=False)
    assert importlib.reload(sent)._SENTIMENT_PARALLEL_FETCH is False
    monkeypatch.setenv("YIAGENTS_SENTIMENT_PARALLEL_FETCH", "true")
    assert importlib.reload(sent)._SENTIMENT_PARALLEL_FETCH is True
    importlib.reload(sent)  # restore module to its process-default state
