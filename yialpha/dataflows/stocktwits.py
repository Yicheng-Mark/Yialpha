"""StockTwits public symbol-stream fetcher.

StockTwits exposes a per-symbol message stream at
``api.stocktwits.com/api/2/streams/symbol/{ticker}.json`` that requires no
API key, no OAuth, and no registration. Each message includes a
user-labeled sentiment field (``Bullish``/``Bearish``/null), the message
body, timestamp, and posting user.

The ticker is validated with :func:`yialpha.dataflows.utils.safe_ticker_component`
before it is interpolated into the URL path — the value ultimately comes from
LLM/user input and must never inject path segments.

Degradation contract: this fetcher is invoked DIRECTLY by the sentiment
analyst (not through ``route_to_vendor``), whose node contract requires a
string return — so a transport/parse failure still returns the
``<stocktwits unavailable: ...>`` placeholder, BUT it now also records a
``KIND_OPTIONAL_UNAVAILABLE`` data-quality sentinel so the run's evidence
chain captures the degradation instead of silently losing it (mirroring what
the router does for its optional categories). Repeat calls for the same
ticker within a short TTL are served from the shared on-disk cache
(:func:`yialpha.dataflows.disk_cache.cached_or_fetch`), so burst re-asks
(batch runs, retries) do not re-hit the API.
"""

from __future__ import annotations

import http.client
import json
import logging
from urllib.request import Request, urlopen

from . import quality
from .disk_cache import cached_or_fetch, safe_cache_component, vendor_cache_dir
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_UA = "yialpha/0.3 (+https://github.com/zhang12120113-creator/YiAlpha)"

#: Message-stream sentiment is prompt-noise that moves minute-to-minute, so the
#: TTL is short (5 minutes): burst re-asks for the same ticker (batch runs,
#: retries) collapse to one network hit while the stream stays near-live.
_CACHE_TTL_DAYS = 5.0 / 1440.0


def fetch_stocktwits_messages(ticker: str, limit: int = 30, timeout: float = 10.0) -> str:
    """Fetch recent StockTwits messages for ``ticker`` and return them as a
    formatted plaintext block ready for prompt injection.

    Returns a placeholder string when the endpoint is unreachable, the
    symbol has no messages, or the response shape is unexpected — the
    caller never has to special-case None or exceptions. A transport/parse
    degradation additionally records an optional-unavailable data-quality
    sentinel so the run's report reflects the missing source. A malformed
    ticker raises ``ValueError`` (fail-closed): it must never be silently
    interpolated into the request URL.
    """
    # Validate BEFORE URL interpolation: the ticker reaches us via LLM/user
    # input, and an unvalidated value would inject arbitrary path segments.
    safe = safe_ticker_component(ticker.upper())
    url = _API.format(ticker=safe)
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    def _fetch() -> bytes:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()

    try:
        raw = cached_or_fetch(
            vendor_cache_dir("stocktwits"),
            f"stream_{safe_cache_component(safe)}.json",
            _fetch,
            ttl_days=_CACHE_TTL_DAYS,
            vendor="stocktwits",
        )
        assert raw is not None  # fail_open never set: fetch errors re-raise
        data = json.loads(raw)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        # Keep the placeholder contract (the sentiment node requires a string),
        # but record the degradation as data-quality evidence — this fetcher is
        # not routed through interface.route_to_vendor, so without recording it
        # here the run's evidence chain would lose this optional source's
        # failure entirely.
        logger.warning("StockTwits fetch failed for %s: %s", ticker, exc)
        quality.record_sentinel(
            "fetch_stocktwits_messages",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"transport/parse failure: {type(exc).__name__}: {exc}",
        )
        return f"<stocktwits unavailable: {type(exc).__name__}>"

    messages = data.get("messages", []) if isinstance(data, dict) else []
    if not messages:
        return f"<no StockTwits messages found for ${ticker.upper()}>"

    lines = []
    bullish = bearish = unlabeled = 0
    for m in messages[:limit]:
        created = m.get("created_at", "")
        user = (m.get("user") or {}).get("username", "?")
        entities = m.get("entities") or {}
        sentiment_obj = entities.get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        body = (m.get("body") or "").replace("\n", " ").strip()
        if len(body) > 280:
            body = body[:280] + "…"

        if sentiment == "Bullish":
            bullish += 1
            tag = "Bullish"
        elif sentiment == "Bearish":
            bearish += 1
            tag = "Bearish"
        else:
            unlabeled += 1
            tag = "no-label"
        lines.append(f"[{created} · @{user} · {tag}] {body}")

    total = bullish + bearish + unlabeled
    bull_pct = round(100 * bullish / total) if total else 0
    bear_pct = round(100 * bearish / total) if total else 0
    summary = (
        f"Bullish: {bullish} ({bull_pct}%) · "
        f"Bearish: {bearish} ({bear_pct}%) · "
        f"Unlabeled: {unlabeled} · "
        f"Total: {total} most-recent messages"
    )
    return summary + "\n\n" + "\n".join(lines)
