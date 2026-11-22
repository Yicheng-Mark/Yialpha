from typing import Annotated

from langchain_core.tools import tool

from yialpha.dataflows.interface import route_to_vendor
from yialpha.dataflows.utils import current_pit_end, get_analysis_date


def _pit_curr_date(curr_date):
    """Clamp an LLM-supplied curr_date to the run's pinned analysis date.

    Same guard as the a_share_native / sec_ownership tools: vendors
    PIT-filter (or refuse historical snapshots) by the passed ``curr_date``
    verbatim, so an LLM echoing a cached header's today-date must not
    disarm the gate. An EMPTY/omitted curr_date in a pinned (backtest) run
    falls back to the pinned date — "or empty for live mode" cannot be
    used to sidestep a historical gate (review P2-1); genuinely live runs
    have no pinned date and keep the empty value.
    """
    return current_pit_end(curr_date) or get_analysis_date() or curr_date


def _vendor_ticker(ticker: str) -> str:
    """Stock-vendor-ready ticker: a tokenized-stock perp's underlying.

    The LLM-facing tools receive the contract ticker ("MUUSDT") per the
    shared instrument-context contract, but these routes end at equity-only
    vendors (Yahoo / Alpha Vantage) that cannot answer the contract symbol.
    Resolve the Yahoo-ready underlying here — the same deterministic mapping
    the sentiment/news prefetch legs apply above the tool layer — so a tool
    call that follows the prompt still reaches real company data.
    Pure-crypto perps and plain equities resolve to None and pass through
    unchanged. Deferred import idiom as in sentiment_analyst (binance pulls
    a heavy module graph).
    """
    from yialpha.dataflows.binance import stock_perp_underlying

    return stock_perp_underlying(ticker) or ticker


def _iso_date_bound(value: str) -> str:
    """Round-13: reduce a dual-form date bound to its ``YYYY-MM-DD`` head.

    The graph entry normalizes ``propagate()``'s trade_date, but this tool
    still sees timestamp forms ("2026-08-01 16:00:00") by two paths: the LLM
    echoing the prompt's full-ISO header date into a bound tool call, and a
    programmatic direct call (the sentiment / news prefetch legs, tests).
    Both end at vendors that ``strptime`` the value verbatim —
    ``yfinance_news.get_news_yfinance`` and Alpha Vantage's
    ``format_datetime_for_api`` ValueError on the time part, and news_data
    is a CORE category (fail-closed), so the error aborted the whole run.
    ``str(x)[:10]`` is byte-identical on an already-canonical date; a
    non-date string passes through and still fails at the vendor — we
    normalize the SHAPE, never invent a date.
    """
    return str(value)[:10]


@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor(
        "get_news",
        _vendor_ticker(ticker),
        _iso_date_bound(start_date),
        _iso_date_bound(end_date),
    )

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int | None, "Days to look back; omit to use the configured default"] = None,
    limit: Annotated[int | None, "Max articles to return; omit to use the configured default"] = None,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor. Defaults for look_back_days and
    limit come from DEFAULT_CONFIG (global_news_lookback_days,
    global_news_article_limit); pass explicit values to override.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back; omit to inherit config
        limit (int): Maximum number of articles to return; omit to inherit config

    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str | None, "current/as-of date in yyyy-mm-dd (the trade date, or empty for live mode)"] = None,
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor. Dormant today (no analyst
    advertises it); the curr_date gate mirrors the vendor-side historical
    refusal so a future wiring cannot leak today's snapshot into a backtest.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str | None): As-of date yyyy-mm-dd (empty for live mode)
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor(
        "get_insider_transactions", _vendor_ticker(ticker),
        _pit_curr_date(curr_date),
    )
