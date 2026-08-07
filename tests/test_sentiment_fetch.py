"""Byte-equivalence guard for the sentiment source fetch.

``_fetch_sentiment_sources`` fans the three independent fetches (Yahoo news /
StockTwits / Reddit) across a thread pool when ``YIAGENTS_SENTIMENT_PARALLEL_FETCH``
is on, and runs them sequentially when off. Each block lands in a fixed slot,
so the two paths must produce identical results -- this test pins that.
"""
import pytest

import yiagents.agents.analysts.sentiment_analyst as sent


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
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    out = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    assert out == ("NEWS", "STOCKTWITS", "REDDIT")


@pytest.mark.unit
def test_parallel_fetch_byte_equivalent_to_sequential(monkeypatch):
    _stub_sources(monkeypatch)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    sequential = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", True)
    parallel = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    # Fixed slots: completion order must not change the assembled result.
    assert parallel == sequential == ("NEWS", "STOCKTWITS", "REDDIT")


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
