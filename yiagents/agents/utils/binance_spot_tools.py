"""LangChain ``@tool`` wrappers for the Binance SPOT data vendors.

Each tool is a thin pass-through to ``route_to_vendor`` — same shape as
:mod:`yiagents.agents.utils.binance_perp_tools` — so the optional-category
fallback (429/network -> sentinel) is reused verbatim. Only the market analyst
advertises these (and only for ``asset_type == "crypto_spot"``); the market
``ToolNode`` carries them too so the dispatched tool_calls resolve, but for
non-spot runs the LLM never names them so they stay dormant.

Spot carries OHLCV + 24h ticker only (no funding/OI/leverage — those are
perp-only). The spot-perp basis tool is the cross-venue signal spot unlocks:
it exposes the perpetual's premium/discount vs this spot reference.
"""

from typing import Annotated

from langchain_core.tools import tool

from yiagents.dataflows.interface import route_to_vendor


@tool
def get_binance_spot_klines(
    symbol: Annotated[str, "Binance SPOT symbol, e.g. BTCUSDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    interval: Annotated[str, "Kline interval, e.g. '1d' (default) or '1h'"] = "1d",
) -> str:
    """Retrieve daily OHLCV candles for a Binance SPOT pair.

    Prices the actual Binance spot book (not the perpetual, not the Yahoo
    pair). Returns a CSV with Open, High, Low, Close, Adj Close, Volume columns
    (Adj Close == Close for spot) — the same shape as the perp klines tool, so
    the indicator path is reusable.
    Args:
        symbol: Binance spot symbol, e.g. BTCUSDT, ETHUSDT.
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format.
        interval: Kline interval (default '1d').
    Returns:
        str: Header + CSV of OHLCV candles for the requested range.
    """
    return route_to_vendor("get_binance_spot_klines", symbol, start_date, end_date, interval)


@tool
def get_binance_spot_ticker24(
    symbol: Annotated[str, "Binance SPOT symbol, e.g. BTCUSDT"],
) -> str:
    """Retrieve 24h rolling stats for a Binance SPOT pair.

    The spot market snapshot: latest price, 24h change %, high/low, base +
    quote volume. Spot has no funding/OI, so this plus the OHLCV klines are the
    spot market's complete picture. Returns lastPrice, priceChangePercent,
    highPrice, lowPrice, volume, quoteVolume.
    Args:
        symbol: Binance spot symbol, e.g. BTCUSDT.
    Returns:
        str: Header + single-row CSV of 24h stats.
    """
    return route_to_vendor("get_binance_spot_ticker24", symbol)


@tool
def get_binance_spot_perp_basis(
    symbol: Annotated[str, "Binance USDT symbol with both a spot and perp book, e.g. BTCUSDT"],
    look_back_days: Annotated[int, "Number of past days of history (default 7)"] = 7,
) -> str:
    """Retrieve the spot-perp basis (Binance USDT-M perp vs Binance SPOT).

    The cross-venue signal spot unlocks: ``basis`` = perpClose - spotClose and
    ``basisRate`` = basis / spotClose, aligned by date over the lookback window.
    Positive basis = the perpetual trades rich vs spot (long demand paying a
    premium); negative = discount (short pressure / flight to spot). This is
    the "real" basis traders watch, distinct from the perp-native perp-vs-index
    basis. Degrades to a sentinel when either leg is unavailable (e.g. a perp
    with no spot listing).
    Args:
        symbol: USDT symbol that has BOTH a spot and a perp book, e.g. BTCUSDT.
        look_back_days: Days of daily history (default 7, capped at 30).
    Returns:
        str: Header + CSV of date, perpClose, spotClose, basis, basisRate.
    """
    return route_to_vendor("get_binance_spot_perp_basis", symbol, look_back_days)
