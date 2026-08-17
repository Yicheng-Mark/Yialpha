"""Byte-equivalence + gating guard for the sentiment source fetch.

``_fetch_sentiment_sources`` fans the independent fetches (Yahoo news /
StockTwits / Reddit, plus the crypto-only Binance Square block) across a
thread pool when ``YIAGENTS_SENTIMENT_PARALLEL_FETCH`` is on, and runs them
sequentially when off. Each block lands in a fixed slot, so the two paths
must produce identical results -- this test pins that.

The fourth slot is the Binance Square crypto-sentiment block:
* stock / unset asset_type -> ``None`` (prompt byte-identical to the
  three-source version);
* crypto asset types, live date, ``binance_square_enabled`` -> fetched;
* crypto asset types, historical date -> explicit unavailable placeholder,
  with the current-feed endpoint never called (PIT fail-closed);
* crypto asset types, config disabled -> ``None``, endpoint never called.
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


def _stub_sources(monkeypatch, square="UNUSED"):
    monkeypatch.setattr(sent, "get_news", _Stub("NEWS"))
    monkeypatch.setattr(
        sent, "fetch_stocktwits_messages", lambda ticker, limit=30: "STOCKTWITS"
    )
    monkeypatch.setattr(sent, "fetch_reddit_posts", lambda ticker: "REDDIT")
    monkeypatch.setattr(sent, "fetch_binance_square_block", lambda ticker: square)


def _square_must_not_fetch(monkeypatch):
    def _must_not_fetch(*_args, **_kwargs):
        raise AssertionError("Binance Square current feed called when gated off")

    monkeypatch.setattr(sent, "fetch_binance_square_block", _must_not_fetch)


@pytest.mark.unit
def test_sequential_fetch_returns_all_three_plus_none_slot(monkeypatch):
    _stub_sources(monkeypatch)
    _square_must_not_fetch(monkeypatch)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    out = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    assert out == ("NEWS", "STOCKTWITS", "REDDIT", None)


@pytest.mark.unit
def test_parallel_fetch_byte_equivalent_to_sequential(monkeypatch):
    _stub_sources(monkeypatch)
    _square_must_not_fetch(monkeypatch)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    sequential = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", True)
    parallel = sent._fetch_sentiment_sources("AAPL", "2026-01-01", "2026-01-08")
    # Fixed slots: completion order must not change the assembled result.
    assert parallel == sequential == ("NEWS", "STOCKTWITS", "REDDIT", None)


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

    assert result == ("NEWS", "STOCKTWITS", "REDDIT", None)
    assert seen == [987654, 987654, 987654]


@pytest.mark.unit
@pytest.mark.parametrize("asset_type", ["crypto", "crypto_spot", "crypto_perp"])
def test_crypto_live_fetches_binance_square(monkeypatch, asset_type):
    _stub_sources(monkeypatch, square="SQUARE")
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    out = sent._fetch_sentiment_sources(
        "BTCUSDT", "2026-01-01", "2026-01-08", asset_type=asset_type
    )
    assert out == ("NEWS", "STOCKTWITS", "REDDIT", "SQUARE")


@pytest.mark.unit
def test_crypto_parallel_fetch_byte_equivalent_to_sequential(monkeypatch):
    _stub_sources(monkeypatch, square="SQUARE")
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    sequential = sent._fetch_sentiment_sources(
        "BTCUSDT", "2026-01-01", "2026-01-08", asset_type="crypto"
    )
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", True)
    parallel = sent._fetch_sentiment_sources(
        "BTCUSDT", "2026-01-01", "2026-01-08", asset_type="crypto"
    )
    assert parallel == sequential == ("NEWS", "STOCKTWITS", "REDDIT", "SQUARE")


@pytest.mark.unit
def test_crypto_disabled_config_gates_square_off(monkeypatch):
    _stub_sources(monkeypatch)
    _square_must_not_fetch(monkeypatch)
    set_config({"binance_square_enabled": False})
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    out = sent._fetch_sentiment_sources(
        "BTCUSDT", "2026-01-01", "2026-01-08", asset_type="crypto_perp"
    )
    assert out == ("NEWS", "STOCKTWITS", "REDDIT", None)


@pytest.mark.unit
def test_stock_asset_type_never_calls_binance_square(monkeypatch):
    _stub_sources(monkeypatch)
    _square_must_not_fetch(monkeypatch)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    out = sent._fetch_sentiment_sources(
        "AAPL", "2026-01-01", "2026-01-08", asset_type="stock"
    )
    assert out[3] is None


@pytest.mark.unit
def test_historical_fetch_omits_current_social_feeds(monkeypatch):
    _stub_sources(monkeypatch)
    _square_must_not_fetch(monkeypatch)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: True)

    def _must_not_fetch(*_args, **_kwargs):
        raise AssertionError("current social endpoint called during historical run")

    monkeypatch.setattr(sent, "fetch_stocktwits_messages", _must_not_fetch)
    monkeypatch.setattr(sent, "fetch_reddit_posts", _must_not_fetch)
    out = sent._fetch_sentiment_sources("AAPL", "2020-01-01", "2020-01-08")

    assert out[0] == "NEWS"
    assert "historical analysis" in out[1]
    assert "historical analysis" in out[2]
    # Stock run: the fourth slot stays absent (None), not a placeholder.
    assert out[3] is None


@pytest.mark.unit
def test_crypto_historical_gets_placeholder_and_never_calls_feed(monkeypatch):
    _stub_sources(monkeypatch)
    _square_must_not_fetch(monkeypatch)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: True)

    out = sent._fetch_sentiment_sources(
        "BTCUSDT", "2020-01-01", "2020-01-08", asset_type="crypto"
    )

    # The historical branch never calls the social fetchers: every current-
    # feed slot is an explicit unavailable placeholder, Square included.
    assert out[0] == "NEWS"
    assert out[1] == sent._HISTORICAL_STOCKTWITS_UNAVAILABLE
    assert out[2] == sent._HISTORICAL_REDDIT_UNAVAILABLE
    assert out[3] == sent._HISTORICAL_BINANCE_SQUARE_UNAVAILABLE
    assert "historical as-of boundary" in out[3]


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


# ---------------------------------------------------------------------------
# Prompt assembly: stock runs byte-identical, crypto runs gain the block
# ---------------------------------------------------------------------------
_PROMPT_KWARGS = {
    "ticker": "BTCUSDT",
    "start_date": "2026-01-01",
    "end_date": "2026-01-08",
    "news_block": "NEWS",
    "stocktwits_block": "STOCKTWITS",
    "reddit_block": "REDDIT",
}


@pytest.mark.unit
def test_system_message_without_square_matches_three_source_prompt():
    msg = sent._build_system_message(**_PROMPT_KWARGS, binance_square_block=None)

    assert "drawing on three complementary data sources" in msg
    assert "<start_of_binance_square>" not in msg
    assert "Treat Binance Square as crypto-native retail chatter" not in msg
    assert "<start_of_news>" in msg and "<start_of_reddit>" in msg


@pytest.mark.unit
def test_system_message_with_square_adds_fourth_source():
    msg = sent._build_system_message(
        **_PROMPT_KWARGS, binance_square_block="SQUARE DATA"
    )

    assert "drawing on four complementary data sources" in msg
    assert "<start_of_binance_square>\nSQUARE DATA\n<end_of_binance_square>" in msg
    assert "Treat Binance Square as crypto-native retail chatter" in msg
    # The crypto message is the stock message plus exactly three insertions.
    stock = sent._build_system_message(**_PROMPT_KWARGS, binance_square_block=None)
    stripped = (
        msg.replace("four complementary", "three complementary")
        .replace(
            "\n### Binance Square posts — crypto-native social feed (current "
            "snapshot)\nCrypto-native retail chatter from Binance Square, "
            "filtered for the target asset and ranked by view/like counts, "
            "plus feed-wide hot-coin mentions for overall market mood. Posts "
            "are opinions (frequently shilling or sarcasm), not data.\n\n"
            "<start_of_binance_square>\nSQUARE DATA\n<end_of_binance_square>\n",
            "",
        )
        .replace(
            "\n9. **Treat Binance Square as crypto-native retail chatter.** "
            "Weight posts by their view/like counts (a 200k-view post reflects "
            "real attention; a 300-view post is noise), stay alert to shilling "
            "and sarcasm, and read it against the news framing — Square posts "
            "are opinion, never price data. If the block reports zero posts "
            "for the target asset, say so explicitly instead of generalizing "
            "from the hot-coin list.\n",
            "",
        )
    )
    assert stripped == stock
