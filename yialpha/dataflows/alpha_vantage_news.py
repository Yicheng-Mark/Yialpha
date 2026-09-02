from .alpha_vantage_common import _make_api_request, format_datetime_for_api
from .errors import NoMarketDataError
from .utils import current_pit_end


def _refuse_empty_feed(raw, symbol: str, canonical: str):
    """An empty NEWS_SENTIMENT feed is a soft miss, not a success.

    Alpha Vantage answers a covered-but-empty ticker/window with HTTP 200
    ``{"feed": []}``. Returning that text as a payload used to mask the
    router's fall-through to the next vendor AND record a core success for
    an empty answer. Raise the typed no-data error instead (the same
    contract as yfinance's empty-news path) so the router records
    KIND_NO_DATA and tries the next vendor. Non-JSON bodies and non-empty
    feeds pass through untouched.
    """
    import json

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("feed"), list)
        and not parsed["feed"]
    ):
        raise NoMarketDataError(
            symbol, canonical,
            "alpha_vantage returned an empty news feed for this "
            "symbol/window",
        )
    return raw


def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """
    # PIT guard: the tool carries no analysis-date argument (the LLM picks
    # end_date from its prompt context), so clamp the query window against the
    # run's pinned analysis date — otherwise future headlines enter a backtest
    # prompt. Live mode (no analysis date pinned) is a no-op pass-through.
    end_date = current_pit_end(end_date) or end_date

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    return _refuse_empty_feed(
        _make_api_request("NEWS_SENTIMENT", params), ticker, ticker,
    )

def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, macro, and monetary policy.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    from datetime import datetime, timedelta

    # PIT guard: clamp the window to the run's pinned analysis date (see
    # get_news above); no-op in live mode.
    curr_date = current_pit_end(curr_date) or curr_date

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    return _refuse_empty_feed(
        _make_api_request("NEWS_SENTIMENT", params),
        "GLOBAL_NEWS", "GLOBAL_NEWS",
    )


def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)
