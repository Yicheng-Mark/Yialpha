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

Freshness contract (2026-09 PR4): the cache wrapper stores the TRUE fetch
timestamp next to the payload, and that — never the render clock — is what
the block reports. A fetch failure serves stale data only up to
:data:`_MAX_STALE_MINUTES` (15), renders an explicit ``STALE`` marker with
the real age, and records a stale-cache sentinel; beyond the cap the scene
degrades honestly instead of wearing today's timestamp on old bytes. Scenes
fetch independently (one scene failing drops only itself — partial feeds are
rendered with a failure note), and a per-scene singleflight lock collapses
concurrent batch workers onto one network round.

Recency contract: posts dated after the ``as_of`` day are dropped (PIT), the
matching set is partitioned into a trailing :data:`_POST_RECENCY_DAYS` window
vs older posts (recent first, engagement-ranked), and the block header
discloses the split plus the oldest/newest matching post times — an old
high-view post can still be evidence, but never masquerades as today's
chatter. Zero matching posts is an HONEST answer (explicit
Neutral/insufficient guidance + feed-wide hot-coin table), not a degradation.

Degradation contract: this fetcher is invoked DIRECTLY by the sentiment
analyst (not through ``route_to_vendor``), whose node contract requires a
string return — so a transport/HTTP/shape failure of EVERY scene still
returns a ``<binance_square unavailable: ...>`` placeholder, BUT it also
records a ``KIND_OPTIONAL_UNAVAILABLE`` data-quality sentinel so the run's
evidence chain captures the degradation (mirroring stocktwits.py).

Point-in-time: the endpoint exposes only the current feed. Callers gate on
``is_historical_date`` (the sentiment analyst does, same as StockTwits/Reddit)
— this module never serves today's posts into a historical replay.

Security: the request URL is a module constant, validated by
:func:`_validated_feed_url` (https/http scheme only; localhost/loopback/
private/reserved hosts rejected) before every request. The ticker-derived base
asset passes :func:`yialpha.dataflows.utils.safe_ticker_component` before it
is used in any regex. Rendered post text is capped at
:data:`_MAX_BODY_CHARS` and travels to the LLM as untrusted evidence content,
never as system instructions.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import time
import urllib.parse
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta

import requests

from ..batch.locks import FileLock
from . import quality
from .disk_cache import cache_file_path, cached_or_fetch, vendor_cache_dir
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

#: Minute-scale social chatter must never be served as "current" when it is
#: not: a fetch failure may serve stale cache for at most this many minutes
#: (rendered with an explicit STALE marker + the true fetch time), after
#: which the scene degrades to unavailable instead of wearing today's clock
#: on old bytes. Tighter than the global data_cache_max_stale_days on purpose.
_MAX_STALE_MINUTES = 15.0

#: Matching posts inside this trailing window (ending at the run's as-of day)
#: are "recent"; older ones still render but are partitioned behind them and
#: disclosed (oldest/newest post times in the header).
_POST_RECENCY_DAYS = 3

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

#: Cache payload wrapper version. v2 files carry ``fetched_at_ms`` beside the
#: response so the TRUE fetch time survives cache hits; the new filename
#: (feed_v2_*) simply ignores any legacy v1 bytes still on disk.
_CACHE_WRAPPER_VERSION = "v2"


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


def _posts_from_response(data: object) -> list[dict]:
    """Extract real posts from one scene's PARSED response object.

    Raises ``ValueError`` on any shape deviation — the caller degrades that
    scene to a failure note instead of feeding half-parsed data onward.
    """
    if not isinstance(data, dict) or data.get("success") is not True:
        raise ValueError(f"feed response not success: {str(data)[:120]}")
    inner = data.get("data")
    vos = inner.get("vos") if isinstance(inner, dict) else None
    if not isinstance(vos, list):
        raise ValueError("feed response has no data.vos list")
    return [p for p in vos if isinstance(p, dict) and p.get("cardType") in _POST_CARD_TYPES]


def _parse_posts(raw: bytes) -> list[dict]:
    """Bytes-shaped wrapper kept for callers/tests parsing a raw scene body."""
    return _posts_from_response(json.loads(raw))


def _cached_scene(scene: str) -> tuple[list[dict], float]:
    """(posts, fetched_at_ms) for one scene, through the freshness-aware cache.

    The cache file stores ``{"fetched_at_ms": ..., "response": {...}}`` so the
    TRUE fetch time travels with the payload (the render clock must never
    masquerade as the fetch clock). A per-scene singleflight lock — threads
    AND processes, via the shared file lock — collapses concurrent batch
    workers racing the same expiring cache entry onto one network round
    instead of last-writer-wins fetch storms.
    """
    cache_dir = vendor_cache_dir("binance_square")
    filename = f"feed_{_CACHE_WRAPPER_VERSION}_{scene}.json"

    def _fetch_wrapped() -> bytes:
        body = _fetch_scene_bytes(scene)
        # Validate the shape BEFORE caching: a wrapper whose response cannot
        # be parsed would poison the cache for the whole TTL window.
        data = json.loads(body)
        posts = _posts_from_response(data)
        wrapper = {
            "fetched_at_ms": int(time.time() * 1000),
            "post_count": len(posts),
            "response": data,
        }
        return json.dumps(wrapper).encode()

    lock = FileLock(str(cache_file_path(cache_dir, filename)))
    with lock:
        raw = cached_or_fetch(
            cache_dir,
            filename,
            _fetch_wrapped,
            ttl_days=_CACHE_TTL_DAYS,
            vendor="binance_square",
            stale_cap_days=_MAX_STALE_MINUTES / (60.0 * 24.0),
        )
    assert raw is not None  # fail_open never set: fetch errors re-raise
    wrapper = json.loads(raw)
    if not isinstance(wrapper, dict) or "response" not in wrapper:
        raise ValueError("cache wrapper has no response payload")
    fetched_ms = float(wrapper.get("fetched_at_ms") or 0.0)
    return _posts_from_response(wrapper["response"]), fetched_ms


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


def _post_ts(post: dict) -> float | None:
    """Post date as unix SECONDS (the feed's ``date`` field), or None."""
    try:
        ts = float(post.get("date"))  # type: ignore[arg-type]
        return ts if ts > 0 else None
    except (TypeError, ValueError):
        return None


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

_ZERO_MATCH_GUIDANCE = (
    "Guidance: report social sentiment for the target as Neutral / insufficient "
    "evidence — do NOT substitute the feed-wide hot-coin mood for the target "
    "asset."
)


def _render_block(
    posts: list[dict],
    candidates: tuple[str, ...],
    base: str,
    fetched_ms: float,
    as_of: str | None,
    failed_scenes: list[str],
    scene_fetched_ms: dict[str, float] | None = None,
) -> str:
    """Render the prompt block: freshness, recency split, ranked posts.

    Every temporal claim comes from real data: ``fetched_ms`` is the cache
    wrapper's stored fetch time (never the render clock), the STALE marker
    fires exactly when the served snapshot is older than the fresh TTL, and
    the recency split/oldest-newest range come from the posts' own dates.
    ``scene_fetched_ms`` (per successfully-served scene) makes freshness
    PER-SCENE: a mixed feed — one scene fresh, another stale-served from
    cache — is labelled MIXED with the stale scenes named, never
    wholesale "fresh" off the newest scene's timestamp.
    """
    now_ms = time.time() * 1000.0
    fetched_dt = (
        datetime.fromtimestamp(fetched_ms / 1000.0, tz=UTC) if fetched_ms > 0 else None
    )
    fetched_at = (
        fetched_dt.strftime("%Y-%m-%d %H:%M UTC") if fetched_dt is not None else "?"
    )
    stale_ttl_ms = _CACHE_TTL_DAYS * 86_400_000.0
    freshness = "fresh"
    if fetched_dt is not None and (now_ms - fetched_ms) > stale_ttl_ms:
        freshness = (
            f"STALE — live fetch failed; snapshot is "
            f"{(now_ms - fetched_ms) / 60_000.0:.0f} min old"
        )
    if scene_fetched_ms:
        stale_scenes = sorted(
            s for s, ms in scene_fetched_ms.items()
            if (now_ms - ms) > stale_ttl_ms
        )
        if stale_scenes and len(stale_scenes) < len(scene_fetched_ms):
            ages = ", ".join(
                f"{s} {max(0.0, now_ms - scene_fetched_ms[s]) / 60_000.0:.0f} min old"
                for s in stale_scenes
            )
            freshness = (
                f"MIXED — live fetch failed for: {ages} (stale snapshots "
                "served from cache); remaining scenes fresh"
            )

    # Recency window anchored on the run's as-of day (default: today UTC).
    try:
        as_of_day = datetime.strptime(
            as_of or datetime.now(UTC).strftime("%Y-%m-%d"), "%Y-%m-%d"
        ).replace(tzinfo=UTC)
    except ValueError:
        as_of_day = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = as_of_day - timedelta(days=_POST_RECENCY_DAYS)
    as_of_eod = as_of_day + timedelta(days=1)

    # PIT: a post dated after the as-of day is future content for this run.
    def _in_window(post: dict) -> bool:
        ts = _post_ts(post)
        return ts is not None and window_start.timestamp() <= ts < as_of_eod.timestamp()

    def _is_future(post: dict) -> bool:
        ts = _post_ts(post)
        return ts is not None and ts >= as_of_eod.timestamp()

    matching = [
        p for p in posts
        if _mentions(p, candidates) and not _is_future(p)
    ]
    # PIT hot coins: a future-dated post (clock skew, publisher error) is
    # future content for this run and must not pollute the feed-wide coin
    # tally or the scanned count either — the same filter `matching` gets.
    pit_posts = [p for p in posts if not _is_future(p)]
    recent = [p for p in matching if _in_window(p)]
    older = [p for p in matching if not _in_window(p)]
    match_ts = [t for t in (_post_ts(p) for p in matching) if t is not None]
    recency_note = (
        f"{len(recent)} of {len(matching)} within the last "
        f"{_POST_RECENCY_DAYS} days (as of {as_of_day.strftime('%Y-%m-%d')})"
    )
    if match_ts:
        recency_note += (
            f"; matching posts span {_utc_date(min(match_ts))} → "
            f"{_utc_date(max(match_ts))} UTC"
        )
    if len(older) > 0:
        recency_note += (
            f"; {len(older)} older matching post{'s' if len(older) != 1 else ''} "
            "kept for context, dated in-line"
        )

    scenes_note = (
        f"scenes ok: {', '.join(s for s in _SCENES if s not in failed_scenes)}"
        if failed_scenes
        else ", ".join(_SCENES)
    )
    head = (
        f"Binance Square recommended feed (global crypto social) — {len(pit_posts)} "
        f"posts scanned ({scenes_note}"
        + (f"; FAILED: {', '.join(failed_scenes)} — partial feed" if failed_scenes else "")
        + f"), fetched {fetched_at} ({freshness}). "
        f"Posts mentioning {base}: {len(matching)} ({recency_note})."
    )

    if not matching:
        return (
            head
            + f"\n\nNo posts mentioning {base} appear in the current feed — the "
            "asset is not a current topic of Square chatter. The feed-wide hot "
            f"coins below still show overall market mood.\n\n{_ZERO_MATCH_GUIDANCE}"
            f"\n\n{_hot_coins_line(pit_posts)}\n\n{_FOOTER}"
        )

    # Recent matching posts first (engagement-ranked), older ones behind them.
    ranked = sorted(
        recent + older,
        key=lambda p: (
            0 if _in_window(p) else 1,
            -_int_or_zero(p.get("viewCount")),
            -_int_or_zero(p.get("date")),
        ),
    )
    lines = []
    for post in ranked[:_MAX_SYMBOL_POSTS]:
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
        + _hot_coins_line(pit_posts)
        + "\n\n"
        + _FOOTER
    )


def fetch_binance_square_block(ticker: str, as_of: str | None = None) -> str:
    """Fetch the Binance Square feed and render a prompt block for ``ticker``.

    ``as_of`` (``YYYY-MM-DD``, default today UTC) anchors the recency window
    and the point-in-time post filter. Returns a formatted plaintext block
    (symbol-matching posts partitioned by recency and ranked by views,
    feed-wide hot-coin mentions, anti-fabrication footer) ready for evidence
    injection. Returns a placeholder string — never raises — when EVERY scene
    is unreachable/unparseable/stale-beyond-cap, recording a data-quality
    sentinel so the degradation stays visible; one scene failing renders a
    partial feed with a failure note. A malformed ticker raises ``ValueError``
    (fail-closed, same contract as stocktwits.py).
    """
    candidates = base_asset_candidates(ticker)

    posts: list[dict] = []
    seen_ids: set[str] = set()
    failed: list[tuple[str, Exception]] = []
    fetched_ms = 0.0
    scene_fetched_ms: dict[str, float] = {}
    for scene in _SCENES:
        try:
            scene_posts, scene_fetched_ms_value = _cached_scene(scene)
        except (requests.RequestException, json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(
                "Binance Square scene %r failed for %s: %s", scene, ticker, exc
            )
            failed.append((scene, exc))
            continue
        scene_fetched_ms[scene] = scene_fetched_ms_value
        if scene_fetched_ms_value > fetched_ms:
            fetched_ms = scene_fetched_ms_value
        for post in scene_posts:
            post_id = str(post.get("id") or "")
            if post_id and post_id in seen_ids:
                continue
            if post_id:
                seen_ids.add(post_id)
            posts.append(post)

    failed_scenes = [scene for scene, _exc in failed]
    if failed and not posts:
        detail = "; ".join(f"{scene}: {type(exc).__name__}" for scene, exc in failed)
        quality.record_sentinel(
            _METHOD,
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"all scenes failed ({detail})",
        )
        return f"<binance_square unavailable: all feed scenes failed ({detail})>"

    if not posts:
        return (
            "<no Binance Square posts in the current feed — the endpoint "
            "returned an empty feed>"
        )

    return _render_block(
        posts, candidates, candidates[0], fetched_ms, as_of, failed_scenes,
        scene_fetched_ms=scene_fetched_ms,
    )
