"""Reddit search fetcher for ticker-specific discussion posts.

OAuth-API-first, RSS fallback. When ``REDDIT_CLIENT_ID`` /
``REDDIT_CLIENT_SECRET`` are configured, the OAuth JSON search endpoint
(``oauth.reddit.com/r/{sub}/search``) is the default path: it carries score /
comment counts (which the sentiment prompt weighs posts by) and bypasses both
the public JSON endpoint's WAF ``403`` (issue #862) and the RSS feed's per-IP
``429``. The bearer token is fetched once via the ``client_credentials`` grant,
cached in process memory with a safety margin, and short negative-cached on
failure so a multi-subreddit / multi-ticker batch coasts on RSS rather than
hammering the token endpoint.

Without creds — or whenever the OAuth path fails (no token, ``401``/``403``/
``429``/network/JSON error) — the path is byte-equivalent to the legacy public
Atom/RSS search feed (``reddit.com/r/{sub}/search.rss``). On a 429 the RSS path
backs off once (honouring ``Retry-After``). RSS lacks score / comment counts, so
those posts are tagged ``source="rss"`` and the formatter omits the metrics
rather than printing fake zeros.

OAuth optional. Returns formatted plaintext blocks ready for prompt injection
and degrades gracefully — returns a placeholder string rather than raising, so
callers never special-case missing data.
"""

from __future__ import annotations

import base64
import html
import http.client
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Public JSON search endpoint — reliably WAF-blocked (HTTP 403) for non-OAuth
# clients (issue #862). Kept for reference; the OAuth path uses _OAUTH_API on
# oauth.reddit.com, which serves authenticated requests without the WAF wall.
_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
_OAUTH_API = "https://oauth.reddit.com/r/{sub}/search?{qs}"
_OAUTH_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
# Refresh a still-valid token this far before its real expiry, so a fetch mid-
# batch doesn't race the clock into using a token that expires in-flight.
_TOKEN_SAFETY_MARGIN = 300
# After a token fetch fails, skip retrying for this long — a multi-subreddit /
# multi-ticker batch otherwise hammers the token endpoint and trips 429s while
# the OAuth backend is having a bad minute. The RSS path still serves throughout.
_NEG_CACHE_TTL = 300
# A descriptive, identified User-Agent (per Reddit's API etiquette). Reddit
# blocks generic/anonymous tokens like bare "Mozilla/5.0" or "curl/…" but
# serves this one on both endpoints; the RSS feed accepts it even when the
# JSON search endpoint 403s, so no browser-spoofing is needed.
_UA = "yiagents/0.3 (+https://github.com/zhang12120113-creator/Yiagents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# In-memory OAuth bearer-token cache. ``None`` = never fetched. A populated
# dict carries either a live token (``token`` set, ``expires_at`` in the future)
# or a negative-cache window (``token`` empty, ``neg_until`` in the future).
# Lives only in process memory; the secret never touches disk or logs.
_token_cache: dict | None = None
_token_lock = threading.Lock()

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at 30s."""
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 30.0) if val else None
    except (ValueError, TypeError, AttributeError):
        return None


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict]:
    """Default path: parse the public Atom search feed for a subreddit.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display. On a 429 (Reddit's
    per-IP rate limit) we back off once — honouring ``Retry-After`` when
    present — before giving up, so a transient burst doesn't blank the feed.
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 5.0
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _reddit_oauth_creds() -> tuple[str, str] | None:
    """Return ``(client_id, secret)`` when both Reddit OAuth env vars are set.

    Mirrors ``fred.py``'s direct ``os.getenv`` read. A blank value (e.g. a stub
    copied from ``.env.example``) counts as unset — OAuth only engages when real
    creds are present, so the keyless RSS path stays the byte-level default.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not client_id or not secret:
        return None
    return client_id, secret


def _oauth_user_agent() -> str:
    """User-Agent for OAuth requests — the feed's ``_UA`` unless
    ``REDDIT_USER_AGENT`` personalizes it (Reddit's API etiquette asks for a
    unique, descriptive UA per app)."""
    return os.getenv("REDDIT_USER_AGENT", "").strip() or _UA


def _request_oauth_token(creds: tuple[str, str], timeout: float) -> tuple[str, int]:
    """POST the ``client_credentials`` grant; return ``(token, expires_in)``.

    Uses HTTP Basic with ``base64(client_id:secret)`` per Reddit's spec. The
    ``client_credentials`` grant is read-only public data — no user identity, no
    redirect — so the secret never leaves this process and the redirect URI is
    irrelevant (it never redirects).
    """
    client_id, secret = creds
    auth = "Basic " + base64.b64encode(f"{client_id}:{secret}".encode()).decode()
    body = urlencode({"grant_type": "client_credentials"}).encode()
    req = Request(
        _OAUTH_TOKEN_URL,
        data=body,
        method="POST",
        headers={
            "User-Agent": _oauth_user_agent(),
            "Authorization": auth,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    token = payload.get("access_token")
    if not token:
        raise ValueError("token endpoint returned no access_token")
    return token, int(payload.get("expires_in", 3600))


def _get_oauth_token(timeout: float) -> str | None:
    """Return a live bearer token, or ``None`` if unavailable.

    ``None`` means: no creds configured, the token endpoint failed (then
    negative-cached for ``_NEG_CACHE_TTL`` seconds), or the cache is empty.
    Double-checked locking keeps the fast path (cache hit) lock-free; only a
    miss/expiry acquires ``_token_lock``, re-checks, then fetches — so a
    concurrent burst (analyst fan-out / batch) shares one token rather than
    racing N parallel fetches. The worst case is two threads each fetching once
    (last writer wins); never incorrect.
    """
    global _token_cache
    now = time.time()
    cache = _token_cache
    if cache and cache.get("token") and cache["expires_at"] > now + _TOKEN_SAFETY_MARGIN:
        return cache["token"]
    if cache and cache.get("neg_until", 0) > now:
        return None
    creds = _reddit_oauth_creds()
    if not creds:
        return None
    with _token_lock:
        # Re-check under the lock: another thread may have populated the cache
        # while we were waiting.
        now = time.time()
        cache = _token_cache
        if cache and cache.get("token") and cache["expires_at"] > now + _TOKEN_SAFETY_MARGIN:
            return cache["token"]
        if cache and cache.get("neg_until", 0) > now:
            return None
        try:
            token, expires_in = _request_oauth_token(creds, timeout)
        except (OSError, http.client.HTTPException, json.JSONDecodeError, ValueError) as exc:
            # OSError covers HTTPError (4xx/5xx from the token endpoint incl.
            # 429), URLError, timeouts. Negative-cache so the batch coasts on
            # RSS instead of hammering a sick token endpoint.
            logger.warning(
                "Reddit OAuth token fetch failed: %s — RSS fallback for %.0fs",
                exc, _NEG_CACHE_TTL,
            )
            _token_cache = {"token": None, "expires_at": 0.0, "neg_until": now + _NEG_CACHE_TTL}
            return None
        _token_cache = {"token": token, "expires_at": now + expires_in, "neg_until": 0.0}
        return token


def _fetch_subreddit_json(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """OAuth JSON search path — richer posts (score / comment counts), bypasses
    the public JSON endpoint's WAF 403 and the RSS feed's per-IP 429.

    Requires a bearer token. Without one (no creds, fetch failure, or within the
    negative-cache window) this degrades straight to RSS. A search failure
    (401/403/429/network/JSON) also falls back to RSS — ``HTTPError`` is an
    ``OSError`` subclass, so the existing ``except`` covers OAuth error
    responses with no new exception handling.
    """
    token = _get_oauth_token(timeout)
    if not token:
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)
    url = _OAUTH_API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(
        url,
        headers={
            "User-Agent": _oauth_user_agent(),
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        posts = []
        for c in children:
            if not isinstance(c, dict):
                continue
            d = c.get("data", {})
            if not isinstance(d, dict):
                continue
            posts.append({
                "title": d.get("title", "") or "",
                "score": d.get("score"),
                "num_comments": d.get("num_comments"),
                # OAuth already returns created_utc as a float epoch; RSS
                # converts an ISO string to the same shape.
                "created_utc": d.get("created_utc"),
                "selftext": d.get("selftext", "") or "",
                "source": "oauth",
            })
        return posts[:limit]
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit OAuth JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Fetch one subreddit — OAuth-API-first, RSS fallback.

    When ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` are set, the OAuth JSON
    search endpoint (``oauth.reddit.com``) carries score / comment counts and
    bypasses both the public JSON endpoint's WAF ``403`` and the RSS feed's
    per-IP ``429``. Without creds the byte-level default is the RSS feed —
    Reddit serves our identified User-Agent there reliably.
    """
    if _reddit_oauth_creds():
        return _fetch_subreddit_json(ticker, sub, limit, timeout)
    return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 1.0,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    ``inter_request_delay`` paces per-subreddit requests to stay under Reddit's
    public per-IP rate limit on the RSS path; with OAuth configured the limit is
    per-client instead, so the delay is generous-but-harmless pacing. Either way
    429s are rare even when several analyses run back-to-back.
    """
    blocks = []
    total_posts = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(ticker, sub, limit_per_sub, timeout)
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            # Score / comment counts are absent on the RSS fallback path —
            # show them only when present rather than printing fake zeros.
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)
