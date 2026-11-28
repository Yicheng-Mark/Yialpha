"""yfinance-based news data fetching functions."""

import contextlib
import hashlib
import json
import logging
from datetime import UTC, datetime

import yfinance as yf
from dateutil.relativedelta import relativedelta

from .config import get_config
from .disk_cache import cached_or_fetch, vendor_cache_dir
from .errors import NoMarketDataError
from .stockstats_utils import yf_retry
from .symbol_utils import normalize_symbol
from .utils import current_pit_end

logger = logging.getLogger(__name__)

#: TTL for cached yfinance Search results (minutes -> fractional days below).
#: Global-news queries are identical across every ticker in a batch run; a
#: 30-minute window collapses that to one network call per query while news
#: stays fresh enough for an intraday analysis.
_SEARCH_CACHE_TTL_DAYS = 30.0 / (24.0 * 60.0)

#: A fetch failure may serve a cached Search result for at most this long —
#: three times the TTL, as for the social feeds (binance_square,
#: stocktwits). Beyond it the error propagates to the router, which tries the
#: next news vendor or records the miss; the global 30-day stale cap used to
#: replay weeks-old headlines as today's.
_SEARCH_MAX_STALE_DAYS = 3.0 * _SEARCH_CACHE_TTL_DAYS


def _cached_search_news(query: str, news_count: int) -> list[dict]:
    """Run one yfinance Search with a short on-disk cache.

    Batch runs re-issue the exact same global-news queries for every ticker;
    the cache (keyed by query + count) serves repeats from disk. The raw
    article dicts round-trip through JSON; downstream dedup + window
    filtering run on every call regardless of cache origin.
    """
    key_blob = json.dumps({"q": query, "n": news_count}, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(key_blob).hexdigest()[:12]
    filename = f"search_{digest}.json"

    def _fetch() -> bytes:
        search = yf_retry(
            lambda: yf.Search(
                query, news_count=news_count, enable_fuzzy_query=True,
            ),
            # No ticker here; the query is the target the error should name.
            symbol=f"news search: {query}",
        )
        return json.dumps(search.news, default=str).encode("utf-8")

    raw = cached_or_fetch(
        vendor_cache_dir("yfnews"), filename, _fetch,
        ttl_days=_SEARCH_CACHE_TTL_DAYS, vendor="yfnews",
        stale_cap_days=_SEARCH_MAX_STALE_DAYS,
    )
    assert raw is not None  # fail_open never set: fetch errors re-raise
    return json.loads(raw)


def _extract_article_data(article: dict) -> dict:
    """Extract article data from yfinance news format (handles nested 'content' structure)."""
    # Handle nested content structure
    if "content" in article:
        content = article["content"]
        title = content.get("title", "No title")
        summary = content.get("summary", "")
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "Unknown")

        # Get URL from canonicalUrl or clickThroughUrl
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = url_obj.get("url", "")

        # Get publish date
        pub_date_str = content.get("pubDate", "")
        pub_date = None
        if pub_date_str:
            with contextlib.suppress(ValueError, AttributeError):
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))

        return {
            "title": title,
            "summary": summary,
            "publisher": publisher,
            "link": link,
            "pub_date": pub_date,
        }
    else:
        # Fallback for flat structure. Parse the epoch publish time so flat
        # articles are date-filterable too (otherwise they bypass the
        # historical window and leak future news, #992/#1007). The epoch is
        # converted as UTC — ``fromtimestamp(ts)`` without tz yields the HOST's
        # wall clock, which on a non-UTC host (e.g. UTC+8) mixes two time
        # bases in the same window comparison and skews inclusion near
        # midnight by the offset.
        pub_date = None
        ts = article.get("providerPublishTime")
        if ts:
            with contextlib.suppress(ValueError, OSError, TypeError):
                pub_date = datetime.fromtimestamp(ts, tz=UTC)
        return {
            "title": article.get("title", "No title"),
            "summary": article.get("summary", ""),
            "publisher": article.get("publisher", "Unknown"),
            "link": article.get("link", ""),
            "pub_date": pub_date,
        }


def _to_naive_utc(pub_date) -> datetime:
    """Normalize a pub datetime to naive-UTC for window comparisons.

    ``start_dt``/``end_dt`` are naive date-derived datetimes, so an aware
    ``pub_date`` must be reduced to a comparable naive value. Merely
    ``replace(tzinfo=None)`` keeps the ORIGINAL offset's wall clock — an
    article at ``2025-05-10T01:30+08:00`` (i.e. 2025-05-09 17:30 UTC) was
    compared as if it were May 10th, skewing the window by up to the offset.
    Converting to UTC first makes the comparison a single calendar standard
    on both sides. Naive datetimes pass through unchanged.
    """
    if getattr(pub_date, "tzinfo", None) is not None:
        return pub_date.astimezone(UTC).replace(tzinfo=None)
    return pub_date


def _in_news_window(pub_date, start_dt, end_dt) -> bool:
    """Whether an article belongs in the [start_dt, end_dt] window.

    Dated articles are kept only if they fall in the window (aware timestamps
    are normalized to UTC first — see :func:`_to_naive_utc`). An undated article
    is kept only when the window reaches the present (live run) — in a
    historical/backtest window it's excluded, since we can't prove it isn't
    future news (look-ahead safety, #992/#1007).
    """
    if pub_date is not None:
        naive = _to_naive_utc(pub_date)
        return start_dt <= naive <= end_dt + relativedelta(days=1)
    return end_dt >= datetime.now() - relativedelta(days=1)


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    article_limit = get_config()["news_article_limit"]
    # PIT guard: like get_YFin_data_online, the tool carries no analysis-date
    # argument (the LLM picks end_date from its prompt context), so clamp the
    # window against the run's pinned analysis date. Live mode (no analysis
    # date pinned) is a no-op pass-through.
    end_date = current_pit_end(end_date) or end_date
    # Query Yahoo with the canonical symbol, like every other yfinance path —
    # a raw broker/forex/crypto alias (XAUUSD, BTCUSD) otherwise silently
    # returns no news. Keep the user's ticker in the report header.
    canonical = normalize_symbol(ticker)
    resolved = "" if canonical == ticker else f" (resolved to {canonical})"
    try:
        stock = yf.Ticker(canonical)
        news = yf_retry(
            lambda: stock.get_news(count=article_limit),
            symbol=ticker,
            canonical=canonical,
        )

        if not news:
            # EMPTY is a typed soft-miss, not a success: raising lets the
            # router's vendor chain fall through to the next configured
            # vendor (alpha_vantage). A plain "No news found" string used
            # to count as a successful fetch and silently masked the whole
            # fallback chain (PR5, 2026-09).
            raise NoMarketDataError(
                ticker, canonical, "yfinance returned no news items",
            )

        # Parse date range for filtering
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        news_str = ""
        filtered_count = 0

        for article in news:
            data = _extract_article_data(article)

            # Keep only articles within the requested window (look-ahead safe).
            if not _in_news_window(data["pub_date"], start_dt, end_dt):
                continue

            news_str += f"### {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary']}\n"
            if data["link"]:
                news_str += f"Link: {data['link']}\n"
            news_str += "\n"
            filtered_count += 1

        if filtered_count == 0:
            # Same soft-miss contract: the symbol may simply be covered by
            # the next vendor in the chain for this window.
            raise NoMarketDataError(
                ticker, canonical,
                f"no yfinance news within {start_date}..{end_date}",
            )

        return f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n{news_str}"

    except NoMarketDataError:
        # Typed soft-miss (empty feed / nothing in the window): re-raise
        # untouched — this is the designed degradation path (the router tries
        # the next vendor in the chain), not a fault worth an ERROR-level
        # stack trace (same contract as alpha_vantage_indicator, round-11).
        raise
    except Exception:
        # Raise instead of returning an "Error fetching news…" string: news_data
        # is a core category, so the router (route_to_vendor) must see the
        # failure to either fall through to the next vendor or let the
        # all-vendors-failed error propagate. Returning prose here made the
        # router treat the error message as successfully fetched news data.
        logger.exception("news retrieval failed for %s", ticker)
        raise


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back. ``None`` falls back to
            ``global_news_lookback_days`` from the active config.
        limit: Maximum number of articles to return. ``None`` falls back to
            ``global_news_article_limit`` from the active config.

    Returns:
        Formatted string containing global news articles
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]
    search_queries = config["global_news_queries"]
    # PIT guard: curr_date comes from the LLM's tool call; clamp it against the
    # run's pinned analysis date so a backtest cannot receive today's headlines
    # (same guard as get_news_yfinance above; no-op in live mode).
    curr_date = current_pit_end(curr_date) or curr_date
    # Double-form guard (round-11 remediation, same normalize as the
    # sentiment/news tools): curr_date may arrive as "2026-09-30 16:00:00" —
    # reduce to the ISO date so the strptime below never sees a bare
    # ValueError on a datetime-form value.
    curr_date = str(curr_date)[:10]

    all_news = []
    seen_titles = set()

    try:
        # Date range BEFORE accumulating (round-11): the per-query candidates
        # are window-filtered as they accumulate. The old loop counted
        # out-of-window articles toward ``limit`` and broke before the later
        # queries were ever consulted — a first query full of out-window
        # headlines masked the rest and ended in a false NoMarketDataError.
        # Parsed INSIDE the try (round-11 remediation): a malformed date is a
        # real fault worth an ERROR-level stack trace, and the re-raise keeps
        # the router's view (a raw ValueError) unchanged.
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - relativedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        for query in search_queries:
            for article in _cached_search_news(query, limit):
                # Extract uniformly (flat + nested) and window-filter AT
                # ACCUMULATION (round-11): an out-of-window candidate must not
                # consume the ``limit`` quota.
                data = _extract_article_data(article)
                if not _in_news_window(data["pub_date"], start_dt, curr_dt):
                    continue
                # Titleless flat entries carry no signal ("No title" is the
                # extractor's placeholder): they used to consume the ``limit``
                # quota and dedup-collapse into a single "No title" row —
                # exclude them at the gate (nested entries always carry a
                # real title; an empty nested summary is already skipped by
                # the renderer).
                if not data["title"] or data["title"] == "No title":
                    continue
                # Deduplicate by title
                if data["title"] not in seen_titles:
                    seen_titles.add(data["title"])
                    all_news.append(data)

            if len(all_news) >= limit:
                break

        if not all_news:
            # Typed soft-miss (same contract as get_news_yfinance): nothing
            # in-window (or nothing at all) must reach the router's vendor
            # chain, not count as a successful fetch that masks the fallback.
            raise NoMarketDataError(
                curr_date, curr_date,
                f"no yfinance global news within {start_date}..{curr_date}",
            )

        # Render reuses the extraction results accumulated above — the old
        # loop re-extracted every article (and re-applied a dead window
        # filter that the accumulation gate had already made unreachable).
        news_str = ""
        for data in all_news[:limit]:
            news_str += f"### {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary']}\n"
            if data["link"]:
                news_str += f"Link: {data['link']}\n"
            news_str += "\n"

        return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"

    except NoMarketDataError:
        # Typed soft-miss: re-raise untouched — the designed degradation path
        # (the router tries the next vendor), not a fault worth an ERROR-level
        # stack trace (same contract as alpha_vantage_indicator, round-11).
        raise
    except Exception:
        # Same contract as get_news_yfinance above: global_news is a core
        # category, so failures must reach the router rather than masquerade
        # as a news payload.
        logger.exception("global news retrieval failed")
        raise
