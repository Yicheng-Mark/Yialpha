from typing import Annotated

from langchain_core.tools import tool

from yialpha.dataflows.binance import stock_perp_underlying
from yialpha.dataflows.interface import route_to_vendor


def _underlying_symbol(ticker: str) -> str:
    """Remap a Binance tokenized-stock perp symbol to its US-equity underlying.

    ``MUUSDT`` -> ``MU`` so the stock vendors (yfinance/SEC) receive a symbol
    they actually cover, no matter which spelling the LLM passes; anything
    else passes through unchanged. Byte-equivalent for every non-perp run: a
    non-USDT/USDC symbol never triggers the (cached) listing lookup, and no
    pre-existing run could carry a whitelisted perp symbol into these tools.
    """
    return stock_perp_underlying(ticker) or ticker


def _pit_curr_date(curr_date: str | None) -> str | None:
    """Clamp an LLM-supplied curr_date to the run's pinned analysis date.

    Same guard the round-7 sweep added to the indicator/weekly/validation
    tools; the fundamentals chain was the remaining gap. An LLM can echo a
    cached report header's "Data retrieved on" date (today) instead of the
    pinned analysis date — unclamped, filter_financials_by_date and the
    overview/SEC leak gates would then admit post-decision quarters and
    today-snapshot overviews into a historical run. Unchanged in live mode
    (no analysis date pinned).
    """
    from yialpha.dataflows.utils import current_pit_end

    return current_pit_end(curr_date) or curr_date


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", _underlying_symbol(ticker), _pit_curr_date(curr_date))


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str | None, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", _underlying_symbol(ticker), freq, _pit_curr_date(curr_date))


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str | None, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", _underlying_symbol(ticker), freq, _pit_curr_date(curr_date))


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str | None, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", _underlying_symbol(ticker), freq, _pit_curr_date(curr_date))
