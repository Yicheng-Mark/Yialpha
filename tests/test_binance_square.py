"""Binance Square feed vendor — contract tests.

Pinned to the REAL response shape captured live from
``bapi/composite/v9/friendly/pgc/feed/feed-recommend/list`` (2026-08-17):
top-level ``success``/``data.vos``, post cards with ``cardType``
BUZZ_SHORT/BUZZ_LONG (widget cards like KOL_RECOMMEND_GROUP and
FEAR_GREED_HIGHEST_SEARCHED are mixed in and must be skipped), ``date`` as
unix seconds, engagement counts, structured coin tags in
``tradingPairsV2[].code`` + ``$COIN`` ``coinPairList`` markers, and BUZZ_LONG
posts whose body lives in ``title``/``subTitle`` with empty ``content``.

Mock style follows test_tavily_web_search.py: a fake with the REAL vendor
call signature is wrapped by ``mock.Mock(wraps=...)`` so a kwarg drift in the
vendor's ``requests.post`` call raises TypeError here (this repo's
"signature smoke" defense against fake-field mocks), while ``call_args``
stays inspectable. Network is always mocked; the disk cache is isolated to a
per-test tmp dir by conftest's ``_isolated_vendor_cache`` (vendor_cache_dir
resolves ``data_cache_dir`` at call time).
"""

import json
from unittest import mock

import pytest
import requests

from yiagents.dataflows import binance_square as bs, quality
from yiagents.dataflows.utils import proxy_map


# ---------------------------------------------------------------------------
# Fixtures: real-shape response payloads
# ---------------------------------------------------------------------------
def _post(**overrides):
    """A real-shape BUZZ_SHORT post (fields seen in the live capture)."""
    post = {
        "id": "356622028540210",
        "cardType": "BUZZ_SHORT",
        "content": "The crypto industry is evolving beyond trading.",
        "date": 1786957124,
        "authorName": "Crypto_lens_",
        "username": "crypto_lens",
        "likeCount": 135,
        "viewCount": 218376,
        "replyCount": 12,
        "shareCount": 3,
        "quoteCount": 0,
        "webLink": "https://www.binance.com/en/square/post/356622028540210",
        "shareLink": "https://www.binance.com/en/square/post/356622028540210",
        "tradingPairsV2": [],
        "tradingPairs": [],
        "coinPairList": [],
        "hashtagList": [],
        "tendency": 0,
        "aiSummary": None,
        "isCreatedByAI": False,
        "title": "",
        "subTitle": "",
    }
    post.update(overrides)
    return post


def _feed_body(posts):
    return {"success": True, "code": "000000", "message": None, "data": {"vos": posts}}


_FULL_FEED = [
    # BTC via structured tradingPairsV2 tag, highest views.
    _post(
        id="1",
        content="My plan: $63K → $49K. Buy $BTC around 43K.",
        viewCount=75669,
        likeCount=61,
        tradingPairsV2=[{"symbol": "BTCUSDT", "code": "BTC", "price": "63000"}],
    ),
    # BTC via content regex ONLY (no structured tags) — the fallback layer.
    _post(
        id="2",
        content="CZ: Soon, millionaires won't be able to afford 1 full $BTC.",
        viewCount=32261,
        likeCount=37,
    ),
    # BUZZ_LONG: empty content, body lives in title/subTitle; BTC structured.
    _post(
        id="3",
        cardType="BUZZ_LONG",
        content="",
        title="Market News Today: Goldman Says September Hike Very Unlikely",
        subTitle="Goldman described the chances of a September increase as very "
        "unlikely, with CME odds dropping to 30.6%.",
        viewCount=2316,
        likeCount=4,
        tradingPairsV2=[{"symbol": "BTCUSDT", "code": "BTC"}],
    ),
    # HEMI post: matched via coinPairList "$HEMI " marker; must NOT match BTC.
    _post(
        id="4",
        content="Today's Gainers Are Pretty Bad! $HEMI spiked and dropped.",
        viewCount=103773,
        likeCount=63,
        tradingPairsV2=[{"symbol": "HEMIUSDT", "code": "HEMI"}],
        coinPairList=["$HEMI "],
    ),
    # Word-boundary guard: $BTCP must not count as a BTC mention.
    _post(id="5", content="I love $BTCP, the future of finance", viewCount=999),
    # Widget cards with null/absent fields — must be skipped entirely.
    {"id": "6", "cardType": "KOL_RECOMMEND_GROUP", "date": None, "viewCount": None},
    {"id": "7", "cardType": "FEAR_GREED_HIGHEST_SEARCHED", "content": ""},
]


class _FakeResponse:
    def __init__(self, payload=None, *, status=200, text="error"):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.content = (
            json.dumps(payload).encode() if payload is not None else text.encode()
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Error")

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class _FakePost:
    """``requests.post`` stand-in with the vendor's REAL call signature.

    Any kwarg the vendor passes but this signature lacks -> TypeError, which
    is exactly the drift the wraps-style mock is meant to catch.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, json=None, headers=None, timeout=None, proxies=None):
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
                "proxies": proxies,
            }
        )
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


@pytest.fixture()
def _quality_ledger():
    quality.ensure_run_context()
    yield
    quality.reset_quality()


def _patch_post(monkeypatch, fake):
    """Patch the vendor module's ``requests.post`` with a wraps-style Mock."""
    return mock.patch.object(bs.requests, "post", mock.Mock(wraps=fake))


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_success_parses_filters_and_ranks_symbol_posts(monkeypatch, _quality_ledger):
    fake = _FakePost(_FakeResponse(_feed_body(_FULL_FEED)))
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert block.startswith("Binance Square recommended feed")
    assert "Posts mentioning BTC: 3" in block
    # Ranked by views desc: 75669 (structured) -> 32261 (regex) -> 2316 (long)
    assert block.index("75,669") < block.index("32,261") < block.index("2,316")
    assert block.count("Source: https://www.binance.com/en/square/post/") == 3
    # $BTCP is a word-boundary non-match: the post must not be rendered.
    assert "$BTCP" not in block
    # Hot-coin table counts STRUCTURED tags only (regex fallback excluded).
    assert "BTC (2 posts · 77,985 views)" in block
    assert "HEMI (1 post · 103,773 views)" in block
    assert "never as price/volume data" in block
    assert quality.snapshot_quality() == []
    # Two scenes fetched, each a real-shaped request body.
    assert [c["json"]["scene"] for c in fake.calls] == list(bs._SCENES)


@pytest.mark.unit
def test_outbound_request_shape_pinned(monkeypatch, _quality_ledger):
    fake = _FakePost(_FakeResponse(_feed_body(_FULL_FEED)))
    with _patch_post(monkeypatch, fake):
        bs.fetch_binance_square_block("BTCUSDT")

    call = fake.calls[0]
    assert call["url"] == bs._FEED_URL
    # Body fields pinned to the endpoint's real contract (pageIndex is
    # decorative but required; scene drives the feed; contentIds pages).
    assert call["json"] == {
        "pageIndex": 1,
        "pageSize": bs._PAGE_SIZE,
        "scene": "web-homepage",
        "contentIds": [],
    }
    headers = call["headers"]
    assert headers["Clienttype"] == "web"
    assert headers["Bnc-Uuid"] == bs._DEVICE_UUID
    assert headers["Cookie"].startswith(f"bnc-uid={bs._DEVICE_UUID}")
    assert "binance.com" in headers["Referer"]
    assert call["timeout"] == bs._TIMEOUT
    assert call["proxies"] == proxy_map()


@pytest.mark.unit
def test_buzz_long_body_comes_from_title_and_subtitle(monkeypatch, _quality_ledger):
    fake = _FakePost(_FakeResponse(_feed_body([_FULL_FEED[2]])))
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert "Market News Today: Goldman Says September Hike Very Unlikely" in block


@pytest.mark.unit
def test_widget_only_feed_is_honest_empty(monkeypatch, _quality_ledger):
    widgets = [_FULL_FEED[5], _FULL_FEED[6]]
    fake = _FakePost(_FakeResponse(_feed_body(widgets)))
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert block.startswith("<no Binance Square posts in the current feed")
    assert quality.snapshot_quality() == []


@pytest.mark.unit
def test_zero_mentions_is_honest_not_degraded(monkeypatch, _quality_ledger):
    fake = _FakePost(_FakeResponse(_feed_body([_FULL_FEED[3]])))
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert "No posts mentioning BTC appear in the current feed" in block
    assert "HEMI (1 post · 103,773 views)" in block
    assert quality.snapshot_quality() == []


@pytest.mark.unit
def test_multiplier_symbols_match_unmultiplied_cashtags(monkeypatch, _quality_ledger):
    pepe_post = _post(
        id="8",
        content="$PEPE to the moon",
        coinPairList=["$PEPE "],
        tradingPairsV2=[{"symbol": "1000PEPEUSDT", "code": "PEPE"}],
    )
    fake = _FakePost(_FakeResponse(_feed_body([pepe_post])))
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("1000PEPEUSDT")

    assert "Posts mentioning 1000PEPE: 1" in block
    assert "$PEPE to the moon" in block


@pytest.mark.unit
def test_cache_collapses_burst_reasks(monkeypatch, _quality_ledger):
    fake = _FakePost(_FakeResponse(_feed_body(_FULL_FEED)))
    with _patch_post(monkeypatch, fake):
        first = bs.fetch_binance_square_block("BTCUSDT")
        second = bs.fetch_binance_square_block("ETHUSDT")

    # One network round per scene total — the second call served from cache.
    assert len(fake.calls) == len(bs._SCENES)
    assert first != second  # per-symbol filtering still differs


@pytest.mark.unit
def test_base_asset_candidates():
    assert bs.base_asset_candidates("BTCUSDT") == ("BTC",)
    assert bs.base_asset_candidates("btcusdt") == ("BTC",)
    assert bs.base_asset_candidates("BTC") == ("BTC",)
    assert bs.base_asset_candidates("XRPUSDC") == ("XRP",)
    assert bs.base_asset_candidates("1000SHIBUSDT") == ("1000SHIB", "SHIB")
    assert bs.base_asset_candidates("1MBABYDOGEUSDT") == ("1MBABYDOGE", "BABYDOGE")


@pytest.mark.unit
def test_bad_ticker_fails_closed_before_any_request(_quality_ledger):
    def _no_request(**_kwargs):
        raise AssertionError("no request may be attempted for a malformed ticker")

    with mock.patch.object(bs.requests, "post", _no_request), pytest.raises(ValueError):
        bs.fetch_binance_square_block("../etc/passwd")


# ---------------------------------------------------------------------------
# Degradation paths (placeholder + data-quality sentinel, never raise)
# ---------------------------------------------------------------------------
def _degrades(monkeypatch, response, expect_placeholder):
    quality.ensure_run_context()
    fake = _FakePost(response)
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert block == expect_placeholder
    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["method"] == "fetch_binance_square_block"
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    quality.reset_quality()


@pytest.mark.unit
def test_http_error_degrades_with_sentinel(monkeypatch):
    _degrades(
        monkeypatch,
        _FakeResponse({"success": False}, status=403, text="forbidden"),
        "<binance_square unavailable: HTTPError>",
    )


@pytest.mark.unit
def test_transport_error_retries_then_degrades(monkeypatch):
    quality.ensure_run_context()
    # Avoid the transient-retry backoff sleeping 2s in tests.
    from yiagents.dataflows import netretry

    monkeypatch.setattr(netretry.time, "sleep", lambda _s: None)

    def _conn_error(url, json=None, headers=None, timeout=None, proxies=None):
        raise requests.exceptions.ConnectionError("connection reset")

    with mock.patch.object(bs.requests, "post", _conn_error):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert block == "<binance_square unavailable: ConnectionError>"
    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    quality.reset_quality()


@pytest.mark.unit
def test_non_json_body_degrades_with_sentinel(monkeypatch):
    _degrades(
        monkeypatch,
        _FakeResponse(None, status=200, text="<html>blocked</html>"),
        "<binance_square unavailable: JSONDecodeError>",
    )


@pytest.mark.unit
def test_success_false_shape_degrades_with_sentinel(monkeypatch):
    _degrades(
        monkeypatch,
        _FakeResponse({"success": False, "code": "100001", "message": "rejected"}),
        "<binance_square unavailable: ValueError>",
    )


# ---------------------------------------------------------------------------
# URL guard (SSRF): validate before any request
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/bapi",
        "https://127.0.0.1/bapi",
        "https://10.0.0.5/bapi",
        "https://192.168.1.4/bapi",
        "https://169.254.169.254/bapi",
        "https://[fd00::1]/bapi",
        "https://sub.localhost/bapi",
        "ftp://www.binance.com/bapi",
        "https:///bapi",
    ],
)
def test_url_guard_rejects_unsafe_targets(url):
    with pytest.raises(ValueError):
        bs._validated_feed_url(url)


@pytest.mark.unit
def test_url_guard_accepts_the_real_endpoint():
    assert bs._validated_feed_url(bs._FEED_URL) == bs._FEED_URL
