"""Policy + gating guard for the sentiment source fetch.

``_fetch_sentiment_sources`` fans the independent fetches across a thread
pool when ``YIALPHA_SENTIMENT_PARALLEL_FETCH`` is on, and runs them
sequentially when off. Each block lands in a fixed slot, so the two paths
must produce identical results -- this test pins that.

The source SET is the deterministic per-instrument policy (PR4, 2026-09):

* plain stocks — news + StockTwits + Reddit (Square slot None);
* pure crypto (crypto / crypto_spot / crypto_perp without an equity
  underlying) — news + Binance Square; StockTwits/Reddit carry explicit
  policy-off placeholders and their fetchers are never called;
* crypto family, historical date — explicit unavailable placeholders, with
  the current-feed endpoints never called (PIT fail-closed);
* crypto family, config disabled — Square slot None, endpoint never called.
"""

import pytest

import yialpha.agents.analysts.sentiment_analyst as sent
from yialpha.dataflows.config import get_config, set_config


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
    # as_of kwarg is part of the PR4 freshness contract.
    monkeypatch.setattr(
        sent, "fetch_binance_square_block", lambda ticker, as_of=None: square
    )


def _square_must_not_fetch(monkeypatch):
    def _must_not_fetch(*_args, **_kwargs):
        raise AssertionError("Binance Square current feed called when gated off")

    monkeypatch.setattr(sent, "fetch_binance_square_block", _must_not_fetch)


def _must_not_call(monkeypatch, name):
    def _guard(*_args, **_kwargs):
        raise AssertionError(f"{name} called under its source policy")

    monkeypatch.setattr(sent, name, _guard)


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
def test_pure_crypto_live_is_square_only_policy(monkeypatch, asset_type):
    # PR4: pure crypto keeps news + Square; StockTwits/Reddit fetchers are
    # never called and their slots carry explicit policy placeholders.
    _stub_sources(monkeypatch, square="SQUARE")
    _must_not_call(monkeypatch, "fetch_stocktwits_messages")
    _must_not_call(monkeypatch, "fetch_reddit_posts")
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)
    out = sent._fetch_sentiment_sources(
        "BTCUSDT", "2026-01-01", "2026-01-08", asset_type=asset_type
    )
    assert out == (
        "NEWS",
        sent._CRYPTO_STOCKTWITS_OFF,
        sent._CRYPTO_REDDIT_OFF,
        "SQUARE",
    )
    assert "source policy" in out[1] and "source policy" in out[2]


@pytest.mark.unit
def test_stock_perp_hybrid_queries_underlying_stocktwits(monkeypatch):
    # Tokenized-stock perp: Square on the contract side, StockTwits on the
    # UNDERLYING equity ticker; Reddit off.
    seen = {}

    def fake_stocktwits(ticker, limit=30):
        seen["stocktwits"] = ticker
        return "STOCKTWITS"

    def fake_square(ticker, as_of=None):
        seen["square"] = (ticker, as_of)
        return "SQUARE"

    monkeypatch.setattr(sent, "get_news", _Stub("NEWS"))
    monkeypatch.setattr(sent, "fetch_stocktwits_messages", fake_stocktwits)
    monkeypatch.setattr(sent, "fetch_reddit_posts", lambda t: "REDDIT")
    monkeypatch.setattr(sent, "fetch_binance_square_block", fake_square)
    monkeypatch.setattr(sent, "is_historical_date", lambda _date: False)
    monkeypatch.setattr(sent, "_SENTIMENT_PARALLEL_FETCH", False)

    out = sent._fetch_sentiment_sources(
        "MUUSDT", "2026-01-01", "2026-01-08", asset_type="crypto_perp"
    )
    assert out == ("NEWS", "STOCKTWITS", sent._CRYPTO_REDDIT_OFF, "SQUARE")
    assert seen["stocktwits"] == "MU"
    assert seen["square"] == ("MUUSDT", "2026-01-08")


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
    expected = ("NEWS", sent._CRYPTO_STOCKTWITS_OFF, sent._CRYPTO_REDDIT_OFF, "SQUARE")
    assert parallel == sequential == expected


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
    assert out == ("NEWS", sent._CRYPTO_STOCKTWITS_OFF, sent._CRYPTO_REDDIT_OFF, None)


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

    monkeypatch.delenv("YIALPHA_SENTIMENT_PARALLEL_FETCH", raising=False)
    assert importlib.reload(sent)._SENTIMENT_PARALLEL_FETCH is False
    monkeypatch.setenv("YIALPHA_SENTIMENT_PARALLEL_FETCH", "true")
    assert importlib.reload(sent)._SENTIMENT_PARALLEL_FETCH is True
    importlib.reload(sent)  # restore module to its process-default state


# ---------------------------------------------------------------------------
# System message: instructions only — data blocks ride the evidence message
# ---------------------------------------------------------------------------
_SYS_KWARGS = {
    "ticker": "BTCUSDT",
    "start_date": "2026-01-01",
    "end_date": "2026-01-08",
}


@pytest.mark.unit
def test_system_message_carries_instructions_not_data():
    msg = sent._build_system_message(**_SYS_KWARGS, has_square=False)
    assert "drawing on three complementary data sources" in msg
    assert "EXTERNAL EVIDENCE" in msg  # describes where the blocks live
    # Third-party content must never appear in the system role.
    assert "<start_of_news>" not in msg
    assert "<start_of_binance_square>" not in msg
    # No cap rule without a cap.
    assert "Deterministic source policy" not in msg


@pytest.mark.unit
def test_system_message_with_square_and_cap_rule():
    msg = sent._build_system_message(
        **_SYS_KWARGS, has_square=True, confidence_cap="medium"
    )
    assert "drawing on four complementary data sources" in msg
    assert "Treat Binance Square as crypto-native retail chatter" in msg
    assert "Neutral / insufficient evidence" in msg
    assert "at most 'medium' confidence" in msg


@pytest.mark.unit
def test_evidence_message_holds_all_blocks_in_user_role():
    from langchain_core.messages import HumanMessage

    evidence = sent._render_evidence_message(
        news_block="NEWS DATA",
        stocktwits_block="STOCKTWITS DATA",
        reddit_block="REDDIT DATA",
        binance_square_block="SQUARE DATA",
    )
    assert isinstance(evidence, HumanMessage)
    content = evidence.content
    assert content.startswith("[EXTERNAL EVIDENCE")
    for tag, payload in (
        ("news", "NEWS DATA"),
        ("stocktwits", "STOCKTWITS DATA"),
        ("reddit", "REDDIT DATA"),
        ("binance_square", "SQUARE DATA"),
    ):
        assert f"<start_of_{tag}>" in content
        assert payload in content
    # Stock runs: the Square section is absent from the evidence message.
    stock_evidence = sent._render_evidence_message(
        news_block="N", stocktwits_block="S", reddit_block="R",
        binance_square_block=None,
    )
    assert "<start_of_binance_square>" not in stock_evidence.content
