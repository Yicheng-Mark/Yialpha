"""LangChain ``@tool`` wrapper for the Eastmoney A-share data vendor.

A thin pass-through to ``route_to_vendor`` — same shape as the SEC ownership
tool module — so the optional-category fallback (429/network/non-A-share ->
sentinel) is reused verbatim. Only the fundamentals analyst advertises this,
and only when ``YIALPHA_A_STOCK`` is on **and** ``is_a_stock(ticker)`` holds;
the fundamentals ``ToolNode`` carries it too so dispatched tool_calls resolve,
but for a default run (or any non-A-share ticker) the LLM never names it so it
stays dormant.

* ``get_margin_trading`` — 融资融券 (margin trading: 融资余额 / 融券余额 /
  融资买入额 / 融资净买入额). China A-share only; a non-A-share ticker degrades
  to the ``NO_DATA_AVAILABLE`` sentinel.

Point-in-time correct (rows filtered by ``date <= curr_date``) so it is safe to
use in backtests.
"""

from typing import Annotated

from langchain_core.tools import tool

from yialpha.dataflows.interface import route_to_vendor


@tool
def get_margin_trading(
    ticker: Annotated[str, "China A-share ticker, e.g. 600519.SS (Shanghai) or 000001.SZ (Shenzhen)"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "look-back window in days (default 180)"] = 180,
) -> str:
    """Retrieve recent 融资融券 (margin trading) balances for a China A-share ticker.

    融资余额 (margin balance), 融券余额 (short balance), 融资买入额 (margin buy),
    融券卖出量 (short sell, shares), and 融资净买入额 (net margin buy), pulled
    from Eastmoney's exchange-disclosed daily series filtered by date <= curr_date
    (point-in-time; full history). Returns a per-day table plus a window summary.
    A rising 融资余额 signals bullish leverage build-up; a rising 融券余额 signals
    growing short interest.
    China A-share only (Shanghai .SS/.SH or Shenzhen .SZ): a non-A-share ticker
    returns the NO_DATA_AVAILABLE sentinel — report "data not available" rather
    than estimating.
    Args:
        ticker: China A-share ticker, e.g. 600519.SS.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Look-back window in days (default 180).
    Returns:
        str: Header + per-day margin table + window summary, or an explicit
        "data not available" string if the name is not on the margin-trading list.
    """
    return route_to_vendor("get_margin_trading", ticker, curr_date, look_back_days)
