"""LangChain ``@tool`` wrappers for the Binance USDT-M perp data vendors.

Each tool is a thin pass-through to ``route_to_vendor`` — same shape as
:mod:`yiagents.agents.utils.core_stock_tools` — so the optional-category
fallback (429/network -> sentinel) is reused verbatim. Only the market analyst
advertises these (and only for ``asset_type == "crypto_perp"``); the market
``ToolNode`` carries them too so the dispatched tool_calls resolve, but for
non-perp runs the LLM never names them so they stay dormant.
"""

from typing import Annotated

from langchain_core.tools import tool

from yiagents.dataflows.interface import route_to_vendor


@tool
def get_binance_klines(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    interval: Annotated[str, "Kline interval, e.g. '1d' (default) or '1h'"] = "1d",
) -> str:
    """Retrieve daily OHLCV candles for a Binance USDT-M perpetual contract.

    Prefer this over ``get_stock_data`` for perpetuals: it prices the actual
    USDT-M perp (not the Yahoo spot pair). Returns a CSV with Open, High, Low,
    Close, Adj Close, Volume columns (Adj Close == Close for perps).
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT, ETHUSDT, 1000PEPEUSDT.
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format.
        interval: Kline interval (default '1d').
    Returns:
        str: Header + CSV of OHLCV candles for the requested range.
    """
    return route_to_vendor("get_binance_klines", symbol, start_date, end_date, interval)


@tool
def get_binance_funding_rate(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve funding-rate history for a Binance USDT-M perpetual contract.

    Funding is charged every 8h; persistently positive funding means longs pay
    shorts (long crowding / cost-of-carry). Returns fundingTime, fundingRate,
    symbol columns.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format.
    Returns:
        str: Header + CSV of funding-rate rows for the requested range.
    """
    return route_to_vendor("get_binance_funding_rate", symbol, start_date, end_date)


@tool
def get_binance_open_interest(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    look_back_days: Annotated[int, "Number of past days of OI history (default 7)"] = 7,
) -> str:
    """Retrieve open-interest history + live snapshot for a Binance USDT-M perp.

    Rising open interest confirms new money entering a trend; combined with
    price direction it distinguishes trending conviction from crowded
    liquidation setups. Returns time, openInterest, openInterestValue columns
    (last row is the live snapshot).
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        look_back_days: Days of daily OI history to include (default 7).
    Returns:
        str: Header + CSV of open-interest rows (history then live snapshot).
    """
    return route_to_vendor("get_binance_open_interest", symbol, look_back_days)


@tool
def get_binance_long_short_ratio(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    look_back_days: Annotated[int, "Number of past days of history (default 7)"] = 7,
) -> str:
    """Retrieve trader long/short positioning for a Binance USDT-M perpetual.

    The perp-native "sentiment" signal: how the leveraged crowd is actually
    positioned (not what social media says). Returns three series in one table —
    top-trader account ratio, top-trader position ratio, and global account
    ratio (column ``series``) — with ``longAccount`` (long share 0-1),
    ``longShortRatio`` (>1 = longs dominate), and ``shortAccount``.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT, AAPLUSDT.
        look_back_days: Days of daily history (default 7, capped at 30).
    Returns:
        str: Header + CSV of long/short ratio rows across the 3 series.
    """
    return route_to_vendor("get_binance_long_short_ratio", symbol, look_back_days)


@tool
def get_binance_taker_buy_sell(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    look_back_days: Annotated[int, "Number of past days of history (default 7)"] = 7,
) -> str:
    """Retrieve taker buy/sell volume for a Binance USDT-M perpetual.

    Aggressive market order-flow: ``buySellRatio`` > 1 = takers buying more
    than selling (urgent long pressure); < 1 = selling pressure. A rally on a
    sub-1 ratio is low-conviction; a dump on a above-1 ratio is often
    capitulation. Returns ``time, buySellRatio, buyVol, sellVol``.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        look_back_days: Days of daily history (default 7, capped at 30).
    Returns:
        str: Header + CSV of taker buy/sell rows.
    """
    return route_to_vendor("get_binance_taker_buy_sell", symbol, look_back_days)


@tool
def get_binance_basis(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    look_back_days: Annotated[int, "Number of past days of history (default 7)"] = 7,
) -> str:
    """Retrieve the perp-vs-index basis for a Binance USDT-M perpetual.

    Premium/discount of the perpetual vs its underlying index. Positive basis
    = perp trades rich (long demand); negative = discount (short pressure).
    Returns ``time, basis, futuresPrice, indexPrice, basisRate``. Unsupported
    for newer TRADIFI stock-perps (e.g. AAPLUSDT/MUUSDT) — degrades to a
    sentinel rather than failing the run.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        look_back_days: Days of daily history (default 7, capped at 30).
    Returns:
        str: Header + CSV of basis rows.
    """
    return route_to_vendor("get_binance_basis", symbol, look_back_days)
