"""Binance Square recommended-feed fetcher (crypto-native retail sentiment).

Binance Square (币安广场) is Binance's social platform. Its web client loads
the recommended post feed from an internal ``bapi`` endpoint that requires no
API key and no login — only browser-shaped headers carrying an anonymous
device id (``Bnc-Uuid`` + matching ``bnc-uid`` cookie). Each post is plain
social content plus engagement counts (``viewCount``/``likeCount``) and, on
most coin-related posts, Binance's own structured coin tags
(``tradingPairsV2[].code`` and ``$COIN`` ``coinPairList`` markers), which are
far more reliable for coin matching than content regexes.

The feed is a GLOBAL latest-posts snapshot (two scenes fetched, deduped by
post id) — there is no per-symbol query and no as-of parameter. Per-symbol
filtering therefore happens in memory after the fetch, and the raw feed is
cached on disk per scene (5-minute TTL) so a batch run over many symbols
collapses to one network round per scene instead of one per symbol.

Degradation contract: this fetcher is invoked DIRECTLY by the sentiment
analyst (not through ``route_to_vendor``), whose node contract requires a
string return — so a transport/HTTP/shape failure still returns a
``<binance_square unavailable: ...>`` placeholder, BUT it also records a
``KIND_OPTIONAL_UNAVAILABLE`` data-quality sentinel so the run's evidence
chain captures the degradation (mirroring stocktwits.py). A feed that simply
has zero posts mentioning the target coin is an HONEST answer (placeholder
text + feed-wide hot-coin table), not a degradation, and records no sentinel.

Point-in-time: the endpoint exposes only the current feed. Callers gate on
``is_historical_date`` (the sentiment analyst does, same as StockTwits/Reddit)
— this module never serves today's posts into a historical replay.

Security: the request URL is a module constant, validated by
:func:`_validated_feed_url` (https/http scheme only; localhost/loopback/
private/reserved hosts rejected) before every request. The ticker-derived base
asset passes :func:`yiagents.dataflows.utils.safe_ticker_component` before it
is used in any regex.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import urllib.parse
import uuid
from collections import Counter
from datetime import UTC, datetime
from functools import partial

import requests

from . import quality
from .disk_cache import cached_or_fetch, vendor_cache_dir
from .netretry import with_transient_retry
from .utils import proxy_map, safe_ticker_component

logger = logging.getLogger(__name__)

#: Internal web endpoint behind the Binance Square "recommended" feed. Reverse-
#: engineered publicly (github.com/skingchan/Binance-Square-Analysis); free,
#: keyless, but unofficial — it can change or throttle without notice, which
#: the degradation contract above is designed to absorb.
_FEED_URL = "https://www.binance.com/bapi/composite/v9/friendly/pgc/feed/feed-recommend/list"

#: Two scenes are fetched and deduped by post id. ``pageIndex`` is decorative
#: on this endpoint (verified experimentally); real paging would require the
#: contentIds exclusion trick, which a sentiment snapshot does not need.
_SCENES = ("web-homepage", "web-trending")

_PAGE_SIZE = 50
_LANG = "en-US"
_TIMEOUT: tuple[float, float] = (5.0, 20.0)

#: Same short-TTL policy as stocktwits.py: the feed is minute-scale social
#: chatter, but burst re-asks (batch runs over many symbols) must collapse to
#: one network round per scene.
_CACHE_TTL_DAYS = 5.0 / 1440.0

#: Anonymous device identity, generated per process. The reference project
#: hard-codes one shared UUID across all its users — a shared fingerprint that
#: invites collective rate-limiting. A fresh uuid4 per process is equally
#: anonymous and does not inherit anyone's reputation.
_DEVICE_UUID = str(uuid.uuid4())

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

#: Only these card types are real user posts. The feed also mixes in widget
#: cards (KOL_RECOMMEND_GROUP, FEAR_GREED_HIGHEST_SEARCHED, ...) whose
#: date/view/content fields are None/absent.
_POST_CARD_TYPES = frozenset({"BUZZ_SHORT", "BUZZ_LONG"})

_MAX_SYMBOL_POSTS = 10
_MAX_BODY_CHARS = 280
_HOT_COINS_TOP_N = 8

#: Quote suffixes stripped to recover the base asset, longest first so e.g.
#: BTCUSDT never strips to BTCUS+T territory (USDT is checked before USD).
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "PERP")

#: Binance prefixes some leveraged/low-price symbols with a multiplier that
#: Square posts never use as a cashtag ($PEPE, not $1000PEPE). Both forms are
#: matched when the multiplier is present.
_MULTIPLIER_PREFIXES = ("1000", "1M")

_METHOD = "fetch_binance_square_block"


def _validated_feed_url(url: str) -> str:
    """Return ``url`` unchanged iff it is a safe http(s) target.

    Rejects non-http(s) schemes and localhost/loopback/private/reserved/
    link-local/multicast/unspecified hosts, checked BEFORE any request is
    attempted. The URL is a module constant today; this guard exists so a
    future edit to that constant (or a test injecting one) cannot silently
    point vendor traffic at an internal address.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"feed URL scheme must be http/https, got {url!r}")
    host = parts.hostname
    if not host:
        raise ValueError(f"feed URL has no host: {url!r}")
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        raise ValueError(f"feed URL host is localhost: {url!r}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return url  # ordinary domain name
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_reserved
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise ValueError(f"feed URL host {host} is a non-public address")
    return url


def base_asset_candidates(ticker: str) -> tuple[str, ...]:
    """Return the base-asset code(s) a Square post may tag for ``ticker``.

    ``BTCUSDT`` -> ``("BTC",)``; ``1000PEPEUSDT`` -> ``("1000PEPE", "PEPE")``
    (Square cashtags use the unmultiplied name). The ticker is validated with
    :func:`safe_ticker_component` (fail-closed ValueError) before any use.
    """
    safe = safe_ticker_component(ticker.upper())
    base = safe
    for suffix in _QUOTE_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[: -len(suffix)]
            break
    candidates = [base]
    for prefix in _MULTIPLIER_PREFIXES:
        if base.startswith(prefix) and len(base) - len(prefix) >= 3:
            candidates.append(base[len(prefix):])
    return tuple(dict.fromkeys(candidates))


def _feed_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
        "Bnc-Uuid": _DEVICE_UUID,
        "Clienttype": "web",
        "Versioncode": "web",
        "Origin": "https://www.binance.com",
        "Referer": "https://www.binance.com/en/square",
        "Cookie": f"bnc-uid={_DEVICE_UUID}; lang={_LANG}",
        "Accept-Language": _LANG,
    }


def _fetch_scene_bytes(scene: str) -> bytes:
    """POST one scene of the recommended feed and return the raw body."""

    def _post() -> bytes:
        resp = requests.post(
            _validated_feed_url(_FEED_URL),
            json={"pageIndex": 1, "pageSize": _PAGE_SIZE, "scene": scene, "contentIds": []},
            headers=_feed_headers(),
            proxies=proxy_map(),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.content

    # Transport hiccups only: HTTP status errors must not be retried (a 403
    # from the anti-bot layer would just double the fingerprint).
    return with_transient_retry(
        _post,
        vendor="binance_square",
        retry_on=(requests.exceptions.ConnectionError, requests.exceptions.Timeout),
    )


def _parse_posts(raw: bytes) -> list[dict]:
    """Extract real posts (``vos`` entries) from one scene's response body.

    Raises ``ValueError`` on any shape deviation — the caller degrades that to
    the placeholder + sentinel instead of feeding half-parsed data onward.
    """
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("success") is not True:
        raise ValueError(f"feed response not success: {str(data)[:120]}")
    inner = data.get("data")
    vos = inner.get("vos") if isinstance(inner, dict) else None
    if not isinstance(vos, list):
        raise ValueError("feed response has no data.vos list")
    return [p for p in vos if isinstance(p, dict) and p.get("cardType") in _POST_CARD_TYPES]


def _post_codes(post: dict) -> set[str]:
    """Structured coin tags on a post (tradingPairsV2 codes + $COIN markers)."""
    codes: set[str] = set()
    for tp in post.get("tradingPairsV2") or []:
        if isinstance(tp, dict):
            code = str(tp.get("code") or "").strip().upper()
            if code:
                codes.add(code)
    for marker in post.get("coinPairList") or []:
        token = str(marker or "").strip().lstrip("$#").strip().upper()
        if token:
            codes.add(token)
    return codes


def _post_body(post: dict) -> str:
    """Post text for prompt display: content, else title/subTitle (BUZZ_LONG)."""
    content = str(post.get("content") or "").strip()
    if content:
        return content
    title = str(post.get("title") or "").strip()
    subtitle = str(post.get("subTitle") or "").strip()
    if title and subtitle:
        return f"{title} — {subtitle}"
    return title or subtitle


def _mentions(post: dict, candidates: tuple[str, ...]) -> bool:
    codes = _post_codes(post)
    if any(c in codes for c in candidates):
        return True
    # Regex fallback: plain-text cashtags ($BTC / #BTC) the structured fields
    # missed. Word-bounded so $BTCP does not match BTC.
    text = _post_body(post)
    return any(
        re.search(rf"[#$]{re.escape(c)}\b", text, re.IGNORECASE) for c in candidates
    )


def _utc_date(ts: object) -> str:
    try:
        return datetime.fromtimestamp(
            int(ts), tz=UTC  # type: ignore[call-overload]
        ).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return "?"


def _int_or_zero(value: object) -> int:
    """Coerce a JSON count field (int, numeric str, or None) to int."""
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


def _hot_coins_line(posts: list[dict]) -> str:
    counts: Counter[str] = Counter()
    views: Counter[str] = Counter()
    for post in posts:
        post_views = _int_or_zero(post.get("viewCount"))
        for code in _post_codes(post):
            counts[code] += 1
            views[code] += post_views
    ranked = sorted(counts, key=lambda c: (-counts[c], -views[c], c))[:_HOT_COINS_TOP_N]
    if not ranked:
        return "Feed-wide hot coins: none tagged in the scanned posts."
    parts = [
        f"{code} ({counts[code]} post{'s' if counts[code] != 1 else ''} · "
        f"{views[code]:,} views)"
        for code in ranked
    ]
    return "Feed-wide hot coins (structured tags across all scanned posts): " + "; ".join(
        parts
    ) + "."


_FOOTER = (
    "Note: Binance Square posts are unverified crypto-native social content. "
    "Use as qualitative retail-sentiment context only — never as price/volume "
    "data, and do not quote numbers from posts as market data."
)


def _render_block(posts: list[dict], candidates: tuple[str, ...], base: str) -> str:
    fetched_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    matching = [p for p in posts if _mentions(p, candidates)]
    matching.sort(
        key=lambda p: (-_int_or_zero(p.get("viewCount")), -_int_or_zero(p.get("date")))
    )
    head = (
        f"Binance Square recommended feed (global crypto social) — {len(posts)} "
        f"posts scanned (scenes: {', '.join(_SCENES)}), fetched {fetched_at}. "
        f"Posts mentioning {base}: {len(matching)}."
    )

    if not matching:
        return (
            head
            + f"\n\nNo posts mentioning {base} appear in the current feed — the "
            "asset is not a current topic of Square chatter. The feed-wide hot "
            f"coins below still show overall market mood.\n\n{_hot_coins_line(posts)}\n\n{_FOOTER}"
        )

    lines = []
    for post in matching[:_MAX_SYMBOL_POSTS]:
        author = str(post.get("authorName") or post.get("username") or "?")
        views = _int_or_zero(post.get("viewCount"))
        likes = _int_or_zero(post.get("likeCount"))
        body = _post_body(post).replace("\n", " ").strip()
        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + "…"
        link = post.get("webLink") or post.get("shareLink") or ""
        entry = (
            f"- [{_utc_date(post.get('date'))} UTC · {author} · "
            f"views {views:,} · likes {likes:,}] {body}"
        )
        if link:
            entry += f"\n  Source: {link}"
        lines.append(entry)

    return (
        head
        + "\n\n"
        + "\n".join(lines)
        + "\n\n"
        + _hot_coins_line(posts)
        + "\n\n"
        + _FOOTER
    )


def fetch_binance_square_block(ticker: str) -> str:
    """Fetch the Binance Square feed and render a prompt block for ``ticker``.

    Returns a formatted plaintext block (symbol-matching posts ranked by
    views, feed-wide hot-coin mentions, anti-fabrication footer) ready for
    prompt injection. Returns a placeholder string — never raises — when the
    endpoint is unreachable or the response shape is unexpected, recording a
    data-quality sentinel so the degradation stays visible. A malformed ticker
    raises ``ValueError`` (fail-closed, same contract as stocktwits.py).
    """
    candidates = base_asset_candidates(ticker)

    try:
        posts: list[dict] = []
        seen_ids: set[str] = set()
        for scene in _SCENES:
            raw = cached_or_fetch(
                vendor_cache_dir("binance_square"),
                f"feed_{scene}.json",
                partial(_fetch_scene_bytes, scene),
                ttl_days=_CACHE_TTL_DAYS,
                vendor="binance_square",
            )
            assert raw is not None  # fail_open never set: fetch errors re-raise
            for post in _parse_posts(raw):
                post_id = str(post.get("id") or "")
                if post_id and post_id in seen_ids:
                    continue
                if post_id:
                    seen_ids.add(post_id)
                posts.append(post)
    except (requests.RequestException, json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("Binance Square fetch failed for %s: %s", ticker, exc)
        quality.record_sentinel(
            _METHOD,
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"transport/parse failure: {type(exc).__name__}: {exc}",
        )
        return f"<binance_square unavailable: {type(exc).__name__}>"

    if not posts:
        return (
            "<no Binance Square posts in the current feed — the endpoint "
            "returned an empty feed>"
        )

    return _render_block(posts, candidates, candidates[0])
