"""LangChain ``@tool`` wrappers for the SEC ownership/short-interest vendors.

Two thin pass-throughs to ``route_to_vendor`` — same shape as the Binance tool
modules — so the optional-category fallback (429/network/non-US -> sentinel) is
reused verbatim. Only the fundamentals analyst advertises these, and only when
``YIAGENTS_SEC_OWNERSHIP`` is on; the fundamentals ``ToolNode`` carries them too
so dispatched tool_calls resolve, but for a default run the LLM never names them
so they stay dormant.

* ``get_form4_insider_trading`` — insider buys/sells (Form 4). US-only (needs a
  CIK); a non-US ticker degrades to the ``NO_DATA_AVAILABLE`` sentinel.
* ``get_ftd_data`` — fails-to-deliver balances (CNS). Ticker-keyed; a non-US
  ticker yields an honest "no fails reported" string rather than an error.

Both are point-in-time correct (Form 4 by ``filingDate``; FTD by
cutoff + publication lag) so they are safe to use in backtests.
"""

from typing import Annotated

from langchain_core.tools import tool

from yiagents.dataflows.interface import route_to_vendor


@tool
def get_form4_insider_trading(
    ticker: Annotated[str, "US-listed ticker symbol, e.g. AAPL"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "look-back window in days (default 180)"] = 180,
) -> str:
    """Retrieve recent SEC Form 4 insider transactions for a US-listed ticker.

    Insider (officer/director/>10% holder) non-derivative buys and sells, pulled
    from the latest Form 4 filings whose filingDate is on or before curr_date
    (point-in-time). Returns a per-trade table (date, insider, title, BUY/SELL,
    shares, price, post-transaction holding) plus a window net-flow summary.
    US-listed issuers only: a non-US ticker returns the NO_DATA_AVAILABLE
    sentinel — report "data not available" rather than estimating.
    Args:
        ticker: US-listed ticker, e.g. AAPL.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Look-back window in days (default 180).
    Returns:
        str: Header + insider-trade table + window summary, or an explicit
        "no recent insider activity" string if none filed in the window.
    """
    return route_to_vendor("get_form4_insider_trading", ticker, curr_date, look_back_days)


@tool
def get_ftd_data(
    ticker: Annotated[str, "US-listed ticker symbol, e.g. AAPL"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "look-back window in days (default 180)"] = 180,
) -> str:
    """Retrieve recent SEC fails-to-deliver (FTD) balances for a US-listed ticker.

    Fails-to-deliver are settlement balances that signal naked-short / bearish
    pressure. Data comes from the SEC's semi-monthly CNS dissemination files,
    filtered by ticker and by Date <= curr_date, and only files already public
    by curr_date (cutoff + publication lag) are read — point-in-time. Returns a
    per-fail-day table (date, total fails, price) plus a peak/total summary.
    Args:
        ticker: US-listed ticker, e.g. AAPL.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Look-back window in days (default 180).
    Returns:
        str: Header + fail-day table + window summary, or an explicit
        "no fails reported" string if the ticker had no FTDs in the window.
    """
    return route_to_vendor("get_ftd_data", ticker, curr_date, look_back_days)
