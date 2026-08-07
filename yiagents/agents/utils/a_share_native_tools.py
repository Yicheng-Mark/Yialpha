"""LangChain ``@tool`` wrappers for the native A-share data vendors.

A thin pass-through to ``route_to_vendor`` — same shape as the Eastmoney
(``eastmoney_tools``) and SEC-ownership tool modules — so the optional-category
fallback (block / non-A-share / missing optional dependency -> sentinel) is
reused verbatim. Only the fundamentals analyst advertises these, and only when
``YIAGENTS_A_SHARE_NATIVE`` is on **and** ``is_a_stock(ticker)`` holds; the
fundamentals ``ToolNode`` carries them too so dispatched tool_calls resolve,
but for a default run (or any non-A-share ticker) the LLM never names them so
they stay dormant.

* ``get_a_share_fundamentals_native`` — daily TTM valuation (PE/PB/PS/PCF,
  server-computed so no single-period inflation) from BaoStock.
* ``get_a_share_ohlc_native`` — daily 前复权 (qfq) OHLCV from BaoStock.
* ``get_a_share_money_flow_native`` — daily 资金流 (主力/超大单/大单/中单/小单
  net inflow) from AKShare (东财).
* ``get_a_share_dragon_tiger_native`` — recent 龙虎榜 appearances (net buy/sell,
  reason) from AKShare (东财).

China A-share only (.SS/.SH/.SZ); a non-A-share ticker degrades to the
``NO_DATA_AVAILABLE`` sentinel. Point-in-time correct (rows filtered by
``date <= curr_date``) so they are safe to use in backtests.
"""

from typing import Annotated

from langchain_core.tools import tool

from yiagents.dataflows.interface import route_to_vendor


@tool
def get_a_share_fundamentals_native(
    ticker: Annotated[str, "China A-share ticker, e.g. 600519.SS (Shanghai) or 000001.SZ (Shenzhen)"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "look-back window in days (default 180)"] = 180,
) -> str:
    """Retrieve recent daily TTM valuation (PE/PB/PS/PCF) for a China A-share ticker.

    Server-computed trailing-twelve-month / MRQ multiples (peTTM / pbMRQ / psTTM
    / pcfNcfTTM) plus the ST flag, pulled from BaoStock's exchange daily series
    filtered by date <= curr_date (point-in-time). This is the fundamentals
    signal the default yfinance path supplies sparsely for A-shares; it
    complements (does not replace) ``get_fundamentals``. Returns a sampled daily
    table plus a latest-snapshot summary. A negative PE = trailing loss.
    China A-share only (Shanghai .SS/.SH or Shenzhen .SZ): a non-A-share ticker
    returns the NO_DATA_AVAILABLE sentinel — report "data not available" rather
    than estimating multiples.
    Args:
        ticker: China A-share ticker, e.g. 600519.SS.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Look-back window in days (default 180).
    Returns:
        str: Header + sampled valuation table + latest summary, or an explicit
        "data not available" string if no rows fall in the window.
    """
    return route_to_vendor("get_a_share_fundamentals_native", ticker, curr_date, look_back_days)


@tool
def get_a_share_ohlc_native(
    ticker: Annotated[str, "China A-share ticker, e.g. 600519.SS (Shanghai) or 000001.SZ (Shenzhen)"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "look-back window in days (default 180)"] = 180,
) -> str:
    """Retrieve recent daily OHLCV (前复权 qfq) for a China A-share ticker.

    Open / high / low / close / volume / amount / turnover / %change, pulled
    from BaoStock's exchange daily series (adjustflag='2' = qfq) filtered by
    date <= curr_date (point-in-time). Returns a per-day table plus a window
    summary. More reliable for A-shares than the default yfinance path.
    China A-share only (Shanghai .SS/.SH or Shenzhen .SZ): a non-A-share ticker
    returns the NO_DATA_AVAILABLE sentinel — report "data not available" rather
    than estimating prices.
    Args:
        ticker: China A-share ticker, e.g. 600519.SS.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Look-back window in days (default 180).
    Returns:
        str: Header + per-day OHLCV table + window summary, or an explicit
        "data not available" string if no rows fall in the window.
    """
    return route_to_vendor("get_a_share_ohlc_native", ticker, curr_date, look_back_days)


@tool
def get_a_share_news_native(
    ticker: Annotated[str, "China A-share ticker, e.g. 600519.SS (Shanghai) or 000001.SZ (Shenzhen)"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "look-back window in days (default 14)"] = 14,
) -> str:
    """Retrieve recent per-stock Chinese news (东财) for a China A-share ticker.

    Pulled from AKShare's ``stock_news_em`` (Eastmoney per-stock feed), reached
    directly (proxy bypassed), filtered by publish date <= curr_date
    (point-in-time). Chinese-language headlines (title, date, source, snippet).
    This is the A-share news signal the default Reddit/StockTwits/yfinance-news
    path covers thinly. Note: the feed has no deep historical archive, so for a
    curr_date far in the past it honestly returns few/no items — report 'no
    coverage found' rather than fabricating.
    China A-share only (Shanghai .SS/.SH or Shenzhen .SZ): a non-A-share ticker
    returns the NO_DATA_AVAILABLE sentinel — report "data not available" rather
    than inventing headlines.
    Args:
        ticker: China A-share ticker, e.g. 600519.SS.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Look-back window in days (default 14).
    Returns:
        str: Header + bulleted news list, or an explicit "no coverage found"
        string if no items fall in the window.
    """
    return route_to_vendor("get_a_share_news_native", ticker, curr_date, look_back_days)


@tool
def get_a_share_money_flow_native(
    ticker: Annotated[str, "China A-share ticker, e.g. 600519.SS (Shanghai) or 000001.SZ (Shenzhen)"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "look-back window in days (default 30)"] = 30,
) -> str:
    """Retrieve recent daily capital flow (资金流) for a China A-share ticker.

    主力 / 超大单 / 大单 / 中单 / 小单 net inflow per day (CNY), plus a window
    summary (latest main net, cumulative main net, consecutive main-inflow
    days), pulled from AKShare's ``stock_individual_fund_flow`` (Eastmoney 个股
    资金流), reached directly (proxy bypassed) and filtered to
    ``[curr_date - look_back_days, curr_date]`` (point-in-time). A persistent
    positive 主力净流入 = institutional accumulation (bullish); persistent
    negative = distribution (bearish). This is an A-share smart-money signal the
    default yfinance path cannot supply.
    China A-share only (Shanghai .SS/.SH or Shenzhen .SZ): a non-A-share ticker
    returns the NO_DATA_AVAILABLE sentinel — report "data not available" rather
    than estimating capital flow.
    Args:
        ticker: China A-share ticker, e.g. 600519.SS.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Look-back window in days (default 30).
    Returns:
        str: Header + per-day money-flow table + window summary, or an explicit
        "no coverage found" string if no rows fall in the window.
    """
    return route_to_vendor("get_a_share_money_flow_native", ticker, curr_date, look_back_days)


@tool
def get_a_share_dragon_tiger_native(
    ticker: Annotated[str, "China A-share ticker, e.g. 600519.SS (Shanghai) or 000001.SZ (Shenzhen)"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "look-back window in days (default 90)"] = 90,
) -> str:
    """Retrieve recent dragon-tiger board (龙虎榜) appearances for a China A-share.

    For each appearance in the window: 上榜日, 解读, 上榜原因, 龙虎榜净买额 /
    买入额 / 卖出额, 净买额占总成交比. Pulled from AKShare's
    ``stock_lhb_detail_em`` (Eastmoney 龙虎榜), reached directly (proxy
    bypassed) and filtered by ``上榜日 <= curr_date`` (point-in-time). A net
    institutional buy-in (positive 净买额) is a bullish smart-money signal; a net
    sell-out is bearish. The post-event 上榜后N日 forward-return columns are
    intentionally omitted (they would leak future returns for a backtest
    curr_date). Most stocks do not appear on the board in a given window — an
    honest empty result is normal, not an error.
    China A-share only (Shanghai .SS/.SH or Shenzhen .SZ): a non-A-share ticker
    returns the NO_DATA_AVAILABLE sentinel — report "data not available" rather
    than inventing appearances.
    Args:
        ticker: China A-share ticker, e.g. 600519.SS.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Look-back window in days (default 90).
    Returns:
        str: Header + per-appearance breakdown, or an explicit "no dragon-tiger
        activity" string if the stock did not appear on the board in the window.
    """
    return route_to_vendor("get_a_share_dragon_tiger_native", ticker, curr_date, look_back_days)
