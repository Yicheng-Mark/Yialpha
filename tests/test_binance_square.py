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
import os
import time
from unittest import mock

import pytest
import requests

from yialpha.dataflows import binance_square as bs, quality
from yialpha.dataflows.utils import proxy_map


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


@pytest.fixture(autouse=True)
def _isolated_square_cache(tmp_path):
    """Per-test disk cache for EVERY test in this file.

    conftest's ``_isolated_vendor_cache`` is opt-in (not autouse); without
    this override the vendor cache resolves to the developer's REAL
    ``~/.yialpha/cache`` — mocked responses would pollute it and later
    tests (or real runs) would serve the fake bytes back. This file's
    fetch-and-cache tests must never touch the real cache dir.
    """
    from yialpha.dataflows.config import set_config

    set_config({"data_cache_dir": str(tmp_path / "square-cache")})


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
_ALL_FAILED = "<binance_square unavailable: all feed scenes failed ({detail})>"


def _degrades(monkeypatch, response, exc_name, expect_placeholder):
    quality.ensure_run_context()
    fake = _FakePost(response)
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert block == expect_placeholder
    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["method"] == "fetch_binance_square_block"
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    assert exc_name in events[0]["detail"]
    quality.reset_quality()


@pytest.mark.unit
def test_http_error_degrades_with_sentinel(monkeypatch):
    # Per-scene isolation (PR4): both scenes fail -> one sentinel whose
    # detail carries each scene's exception type.
    _degrades(
        monkeypatch,
        _FakeResponse({"success": False}, status=403, text="forbidden"),
        "HTTPError",
        _ALL_FAILED.format(
            detail="web-homepage: HTTPError; web-trending: HTTPError"
        ),
    )


@pytest.mark.unit
def test_transport_error_retries_then_degrades(monkeypatch):
    quality.ensure_run_context()
    # Avoid the transient-retry backoff sleeping 2s in tests.
    from yialpha.dataflows import netretry

    monkeypatch.setattr(netretry.time, "sleep", lambda _s: None)

    def _conn_error(url, json=None, headers=None, timeout=None, proxies=None):
        raise requests.exceptions.ConnectionError("connection reset")

    with mock.patch.object(bs.requests, "post", _conn_error):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert block == _ALL_FAILED.format(
        detail="web-homepage: ConnectionError; web-trending: ConnectionError"
    )
    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    quality.reset_quality()


@pytest.mark.unit
def test_non_json_body_degrades_with_sentinel(monkeypatch):
    _degrades(
        monkeypatch,
        _FakeResponse(None, status=200, text="<html>blocked</html>"),
        "JSONDecodeError",
        _ALL_FAILED.format(
            detail="web-homepage: JSONDecodeError; web-trending: JSONDecodeError"
        ),
    )


@pytest.mark.unit
def test_success_false_shape_degrades_with_sentinel(monkeypatch):
    _degrades(
        monkeypatch,
        _FakeResponse({"success": False, "code": "100001", "message": "rejected"}),
        "ValueError",
        _ALL_FAILED.format(
            detail="web-homepage: ValueError; web-trending: ValueError"
        ),
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


# ---------------------------------------------------------------------------
# PR4 freshness contract: true fetched_at, STALE marker, 15-minute stale cap
# ---------------------------------------------------------------------------
def _write_scene_cache(posts, fetched_ms, age_s, scene="web-homepage"):
    """Seed a v2 wrapper cache file whose payload/mtime are ``age_s`` old."""
    import pathlib

    cache_dir = pathlib.Path(bs.vendor_cache_dir("binance_square"))
    response = {"success": True, "data": {"vos": posts}}
    wrapper = {
        "fetched_at_ms": int(fetched_ms),
        "post_count": len(posts),
        "response": response,
    }
    path = cache_dir / f"feed_v2_{scene}.json"
    path.write_bytes(json.dumps(wrapper).encode())
    old = time.time() - age_s
    os.utime(path, (old, old))
    return path


def _fresh_post(**overrides):
    return _post(
        id="fresh-1",
        content="$BTC looking strong this week",
        date=int(time.time()) - 3_600,
        viewCount=5_000,
        tradingPairsV2=[{"symbol": "BTCUSDT", "code": "BTC"}],
        **overrides,
    )


@pytest.mark.unit
def test_fresh_cache_served_without_network(monkeypatch, _quality_ledger):
    _write_scene_cache([_fresh_post()], time.time() * 1000 - 60_000, age_s=60)
    _write_scene_cache([], time.time() * 1000 - 60_000, age_s=60, scene="web-trending")
    monkeypatch.setattr(
        bs.requests, "post", mock.Mock(side_effect=AssertionError("network hit"))
    )
    block = bs.fetch_binance_square_block("BTCUSDT")
    assert "Posts mentioning BTC: 1" in block
    assert "(fresh)" in block
    assert "STALE" not in block


@pytest.mark.unit
def test_stale_within_cap_renders_true_fetch_time_and_marker(monkeypatch, _quality_ledger):
    # Cache 10 min old (> 5-min TTL, < 15-min cap): fetch fails -> stale
    # serve, rendered with the TRUE stored fetch time and an explicit STALE
    # marker — never today's clock on old bytes.
    fetched_ms = time.time() * 1000 - 600_000
    _write_scene_cache([_fresh_post()], fetched_ms, age_s=600)
    # Same failing cache for the second scene (beyond-cap there is fine —
    # per-scene isolation keeps the first scene's partial feed).
    _write_scene_cache([], time.time() * 1000 - 600_000, age_s=600, scene="web-trending")

    def _fail(url, json=None, headers=None, timeout=None, proxies=None):
        raise requests.exceptions.ConnectionError("down")

    with mock.patch.object(bs.requests, "post", _fail):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert "Posts mentioning BTC: 1" in block
    assert "STALE — live fetch failed" in block
    assert "min old" in block
    # The rendered timestamp is the STORED fetch time, not the render clock.
    from datetime import UTC, datetime

    true_ts = datetime.fromtimestamp(fetched_ms / 1000, tz=UTC).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    assert f"fetched {true_ts}" in block
    # Partial feed: the second scene's stale serve still renders (cap not
    # exceeded), but both fetch failures are visible as stale-cache evidence.
    events = quality.snapshot_quality()
    assert all(e["kind"] == quality.KIND_STALE_CACHE for e in events)


@pytest.mark.unit
def test_stale_beyond_cap_degrades_instead_of_serving_old_bytes(monkeypatch, _quality_ledger):
    quality.ensure_run_context()
    # Cache 30 min old — past the 15-minute square cap even though the
    # global data_cache_max_stale_days default is 30 days.
    _write_scene_cache([_fresh_post()], time.time() * 1000 - 1_800_000, age_s=1_800)

    def _fail(url, json=None, headers=None, timeout=None, proxies=None):
        raise requests.exceptions.ConnectionError("down")

    from yialpha.dataflows import netretry

    monkeypatch.setattr(netretry.time, "sleep", lambda _s: None)
    with mock.patch.object(bs.requests, "post", _fail):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert block.startswith("<binance_square unavailable: all feed scenes failed")
    events = quality.snapshot_quality()
    assert any(e["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE for e in events)
    quality.reset_quality()


@pytest.mark.unit
def test_one_scene_failing_renders_partial_feed(monkeypatch, _quality_ledger):
    """Scene isolation: web-trending 500s, web-homepage serves -> partial."""
    good = _FakeResponse(_feed_body(_FULL_FEED))

    def _flaky(url, json=None, headers=None, timeout=None, proxies=None):
        if json and json.get("scene") == "web-trending":
            return _FakeResponse({"success": False}, status=500, text="boom")
        return good

    with mock.patch.object(bs.requests, "post", mock.Mock(wraps=_flaky)):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert "Posts mentioning BTC: 3" in block
    assert "FAILED: web-trending — partial feed" in block
    assert "scenes ok: web-homepage" in block
    # A partial feed is a rendered answer, not a full degradation: no
    # optional_unavailable sentinel fires.
    assert quality.snapshot_quality() == []


@pytest.mark.unit
def test_future_posts_are_pit_dropped_and_recency_partitioned(monkeypatch, _quality_ledger):
    now = time.time()
    posts = [
        # In-window (today) — matches BTC.
        _post(
            id="today",
            content="$BTC today rally",
            date=int(now) - 3_600,
            viewCount=2_000,
            tradingPairsV2=[{"symbol": "BTCUSDT", "code": "BTC"}],
        ),
        # Old (20 days) — still evidence, but partitioned behind recent.
        _post(
            id="old",
            content="$BTC macro take",
            date=int(now) - 20 * 86_400,
            viewCount=900_000,
            tradingPairsV2=[{"symbol": "BTCUSDT", "code": "BTC"}],
        ),
        # Future (tomorrow) — PIT-dropped for an as_of of today.
        _post(
            id="future",
            content="$BTC tomorrow news",
            date=int(now) + 86_400,
            viewCount=999_999,
            tradingPairsV2=[{"symbol": "BTCUSDT", "code": "BTC"}],
        ),
    ]
    fake = _FakePost(_FakeResponse(_feed_body(posts)))
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("BTCUSDT")

    # Future post dropped: 2 matching, only 1 within the recency window.
    assert "Posts mentioning BTC: 2 (1 of 2 within the last 3 days" in block
    assert "tomorrow news" not in block
    # The in-window post outranks the 900k-view old one despite fewer views.
    assert block.index("today rally") < block.index("macro take")
    assert "older matching post" in block
    # oldest/newest span disclosed.
    assert "matching posts span" in block


@pytest.mark.unit
def test_future_post_excluded_from_hot_coins_and_scan_count(
    monkeypatch, _quality_ledger
):
    """The PIT filter applies to the FEED-WIDE tallies too: a future-dated
    post (clock skew / publisher error) must not enter the hot-coin table or
    the scanned count — only the as-of-visible posts do."""
    now = time.time()
    posts = [
        _post(
            id="today-btc",
            content="$BTC today rally",
            date=int(now) - 3_600,
            viewCount=2_000,
            tradingPairsV2=[{"symbol": "BTCUSDT", "code": "BTC"}],
        ),
        # Future post shilling PEPE — would win the hot-coin table by views
        # if the feed-wide tallies forgot the PIT filter.
        _post(
            id="future-pepe",
            content="$PEPE tomorrow moon",
            date=int(now) + 3 * 86_400,
            viewCount=99_999_999,
            tradingPairsV2=[{"symbol": "PEPEUSDT", "code": "PEPE"}],
        ),
    ]
    fake = _FakePost(_FakeResponse(_feed_body(posts)))
    with _patch_post(monkeypatch, fake):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert "PEPE" not in block
    assert "Feed-wide hot coins" in block and "BTC (1 post" in block
    # Scanned count sees only the as-of-visible post, not the future one.
    assert "— 1 posts scanned" in block


@pytest.mark.unit
def test_mixed_scene_freshness_labelled_mixed(monkeypatch, _quality_ledger):
    """One scene fresh (cache within TTL), the other stale-served (fetch
    failed, within the 15-min cap): the block is labelled MIXED naming the
    stale scene — never wholesale "(fresh)" off the fresh scene's clock."""
    # web-homepage: cache 60s old — within the 5-min TTL, served fresh.
    _write_scene_cache([_fresh_post()], time.time() * 1000 - 60_000, age_s=60)
    # web-trending: fetched 10 min ago (> TTL, < cap) — network fails, the
    # stale snapshot is served.
    _write_scene_cache([], time.time() * 1000 - 600_000, age_s=600, scene="web-trending")

    def _fail(url, json=None, headers=None, timeout=None, proxies=None):
        raise requests.exceptions.ConnectionError("down")

    with mock.patch.object(bs.requests, "post", _fail):
        block = bs.fetch_binance_square_block("BTCUSDT")

    assert "MIXED — live fetch failed for: web-trending" in block
    assert "remaining scenes fresh" in block
    assert "(fresh)." not in block  # the wholesale-fresh label is gone


@pytest.mark.unit
def test_singleflight_collapses_concurrent_burst_onto_one_round(
    monkeypatch, tmp_path, _quality_ledger
):
    """Four threads racing a cold cache: ONE network round per scene.

    The singleflight lock holds across the whole read-or-fetch critical
    section, so the three losers block on the LOCK (never reaching the
    network) and then serve the fresh cache the winner wrote.

    The cache dir is patched at module scope (NOT via config): config rides
    a ContextVar that raw worker threads do not inherit, so the config
    route would resolve the developer's REAL cache dir inside the threads.
    """
    import threading

    monkeypatch.setattr(
        bs, "vendor_cache_dir", lambda name: str(tmp_path / name)
    )
    network_calls: list[str] = []
    call_lock = threading.Lock()

    def _slow_post(url, json=None, headers=None, timeout=None, proxies=None):
        scene = (json or {}).get("scene", "?")
        with call_lock:
            network_calls.append(scene)
        time.sleep(0.05)
        return _FakeResponse(_feed_body(_FULL_FEED))

    with mock.patch.object(bs.requests, "post", mock.Mock(wraps=_slow_post)):
        blocks: list[str] = []
        results_lock = threading.Lock()

        def _run():
            block = bs.fetch_binance_square_block("BTCUSDT")
            with results_lock:
                blocks.append(block)

        threads = [threading.Thread(target=_run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    assert len(blocks) == 4
    assert all(b.startswith("Binance Square recommended feed") for b in blocks)
    assert sorted(network_calls) == sorted(bs._SCENES)
