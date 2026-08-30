"""LangChain ``@tool`` wrappers for the Binance USDT-M perp data vendors.

Each tool is a thin pass-through to ``route_to_vendor`` — same shape as
:mod:`yialpha.agents.utils.core_stock_tools` — so the optional-category
fallback (429/network -> sentinel) is reused verbatim. Only the market analyst
advertises these (and only for ``asset_type == "crypto_perp"``); the market
``ToolNode`` carries them too so the dispatched tool_calls resolve, but for
non-perp runs the LLM never names them so they stay dormant.
"""

from typing import Annotated

from langchain_core.tools import tool

from yialpha.dataflows.interface import route_to_vendor


@tool
def get_binance_klines(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    interval: Annotated[str, "Kline interval, e.g. '1d' (default) or '1h'"] = "1d",
    price_type: Annotated[str, "'last' (default, traded price), 'mark' (price Binance liquidates against) or 'index' (settlement reference; volume is 0)"] = "last",
) -> str:
    """Retrieve daily OHLCV candles for a Binance USDT-M perpetual contract.

    Prefer this over ``get_stock_data`` for perpetuals: it prices the actual
    USDT-M perp (not the Yahoo spot pair). Returns a CSV with Open, High, Low,
    Close, Adj Close, Volume columns (Adj Close == Close for perps). Use
    ``price_type='mark'`` when assessing liquidation risk — Binance triggers
    liquidations on the mark price, not the last traded price — and
    ``price_type='index'`` for the index-price series (mark-vs-index
    displacement as two aligned series; volume column reads 0).
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT, ETHUSDT, 1000PEPEUSDT.
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format.
        interval: Kline interval (default '1d').
        price_type: 'last' (default), 'mark' (mark-price klines) or 'index'.
    Returns:
        str: Header + CSV of OHLCV candles for the requested range.
    """
    return route_to_vendor(
        "get_binance_klines", symbol, start_date, end_date, interval, price_type
    )


@tool
def get_binance_funding_rate(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve funding-rate history for a Binance USDT-M perpetual contract.

    Funding settles on the contract's own cadence (8h on most contracts, 4h/1h
    on many newer ones — the header states the inferred cadence); persistently
    positive funding means longs pay shorts (long crowding / cost-of-carry).
    Returns fundingTime, fundingRate, symbol columns.
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
    period: Annotated[str, "Series granularity: '1d' (default), '5m', '15m', '30m', '1h', '2h', '4h', '6h' or '12h'"] = "1d",
) -> str:
    """Retrieve open-interest history + live snapshot for a Binance USDT-M perp.

    Rising open interest confirms new money entering a trend; combined with
    price direction it distinguishes trending conviction from crowded
    liquidation setups. Returns time, openInterest, openInterestValue columns
    (last row is the live snapshot, daily windows only). ``period`` selects
    the granularity — e.g. '1h' for intraday OI (rows timestamped to the
    hour). The endpoint retains only the last 30 days; for deeper history use
    get_binance_vision_metrics.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        look_back_days: Days of OI history to include (default 7).
        period: Series granularity (default '1d').
    Returns:
        str: Header + CSV of open-interest rows (history then live snapshot).
    """
    return route_to_vendor(
        "get_binance_open_interest", symbol, look_back_days, period=period,
    )


@tool
def get_binance_long_short_ratio(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    look_back_days: Annotated[int, "Number of past days of history (default 7)"] = 7,
    period: Annotated[str, "Series granularity: '1d' (default), '5m', '15m', '30m', '1h', '2h', '4h', '6h' or '12h'"] = "1d",
) -> str:
    """Retrieve trader long/short positioning for a Binance USDT-M perpetual.

    The perp-native "sentiment" signal: how the leveraged crowd is actually
    positioned (not what social media says). Returns three series in one table —
    top-trader account ratio, top-trader position ratio, and global account
    ratio (column ``series``) — with ``longAccount`` (long share 0-1),
    ``longShortRatio`` (>1 = longs dominate), and ``shortAccount``.
    ``period`` selects the granularity ('1d' default; intraday for
    event-window positioning). The endpoints retain only the last 30 days;
    for deeper history use get_binance_vision_metrics.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT, AAPLUSDT.
        look_back_days: Days of history (default 7; capped at 30 for '1d').
        period: Series granularity (default '1d').
    Returns:
        str: Header + CSV of long/short ratio rows across the 3 series.
    """
    return route_to_vendor(
        "get_binance_long_short_ratio", symbol, look_back_days, period=period,
    )


@tool
def get_binance_taker_buy_sell(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    look_back_days: Annotated[int, "Number of past days of history (default 7)"] = 7,
    period: Annotated[str, "Series granularity: '1d' (default), '5m', '15m', '30m', '1h', '2h', '4h', '6h' or '12h'"] = "1d",
) -> str:
    """Retrieve taker buy/sell volume for a Binance USDT-M perpetual.

    Aggressive market order-flow: ``buySellRatio`` > 1 = takers buying more
    than selling (urgent long pressure); < 1 = selling pressure. A rally on a
    sub-1 ratio is low-conviction; a dump on a above-1 ratio is often
    capitulation. Returns ``time, buySellRatio, buyVol, sellVol``.
    ``period`` selects the granularity ('1d' default; '5m'/'15m' for
    event-window order flow). The endpoint retains only the last 30 days; for
    deeper history use get_binance_vision_metrics.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        look_back_days: Days of history (default 7; capped at 30 for '1d').
        period: Series granularity (default '1d').
    Returns:
        str: Header + CSV of taker buy/sell rows.
    """
    return route_to_vendor(
        "get_binance_taker_buy_sell", symbol, look_back_days, period=period,
    )


@tool
def get_binance_premium_index(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
) -> str:
    """Retrieve the mark-price snapshot for a Binance USDT-M perpetual.

    ``/fapi/v1/premiumIndex`` in one call: ``markPrice`` (the price Binance
    liquidates against — anchor liquidation-distance claims here, NOT the last
    traded price), ``indexPrice``, ``markVsIndexPct`` (mark premium vs index,
    %), ``lastFundingRate`` (the rate in effect for the NEXT settlement — more
    current than the history tool's last settled row), and ``nextFundingTime``.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
    Returns:
        str: Header + single-row CSV of the mark-price snapshot.
    """
    return route_to_vendor("get_binance_premium_index", symbol)


@tool
def get_binance_basis(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    look_back_days: Annotated[int, "Number of past days of history (default 7)"] = 7,
    period: Annotated[str, "Series granularity: '1d' (default), '5m', '15m', '30m', '1h', '2h', '4h', '6h' or '12h'"] = "1d",
) -> str:
    """Retrieve the perp-vs-index basis for a Binance USDT-M perpetual.

    Premium/discount of the perpetual vs its underlying index. Positive basis
    = perp trades rich (long demand); negative = discount (short pressure).
    Returns ``time, basis, futuresPrice, indexPrice, basisRate``. Unsupported
    for newer TRADIFI stock-perps (e.g. AAPLUSDT/MUUSDT) — degrades to a
    sentinel rather than failing the run. ``period`` selects the granularity
    ('1d' default; intraday for funding-window basis dynamics).
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        look_back_days: Days of history (default 7; capped at 30 for '1d').
        period: Series granularity (default '1d').
    Returns:
        str: Header + CSV of basis rows.
    """
    return route_to_vendor("get_binance_basis", symbol, look_back_days, period=period)


@tool
def get_binance_depth_snapshot(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    limit: Annotated[int, "Number of book levels per side: 5/10/20/50/100 (default)/500/1000"] = 100,
) -> str:
    """Retrieve the live order-book depth snapshot for a Binance USDT-M perp.

    ``/fapi/v1/depth`` in one call: a level-by-level bid/ask ladder plus mid,
    spread (bps) and the top-N quantity imbalance ((bids-asks)/total; positive
    = heavier bid side). This is execution reality — stops and targets fill
    INTO this book, and a thin side into the price's direction is slippage
    (or, at leverage, the liquidation-cascade accelerant). Live-only; for how
    depth persisted through past moves use get_binance_vision_book_depth.
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        limit: Book levels per side (default 100).
    Returns:
        str: Header + CSV ladder (level, bid_price, bid_qty, ask_price, ask_qty).
    """
    return route_to_vendor("get_binance_depth_snapshot", symbol, limit)


@tool
def get_binance_vision_metrics(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    interval: Annotated[str, "Output granularity: '1d' (default), '4h', '1h' or '5m' (raw)"] = "1d",
) -> str:
    """Retrieve DEEP-HISTORY positioning metrics for a Binance USDT-M perp.

    data.binance.vision public archives: 5-minute open interest, the
    top-trader ACCOUNT and POSITION long/short ratios, the GLOBAL account
    long/short ratio and the taker buy/sell ratio going back YEARS (BTCUSDT
    since 2020-09) — the same series the REST tools only retain for 30 days
    (the archive's count_/sum_ column prefixes are legacy naming for the
    VALUE, REST-verified). This is the tool for regime context across
    funding cycles and the ONLY point-in-time-correct positioning source
    for a historical analysis date (each archive file contains only its own
    day's rows; every zip is sha256-verified).
    Returns ``time, open_interest, open_interest_value,
    top_trader_account_long_short_ratio, top_trader_long_short_ratio,
    global_long_short_ratio, taker_buy_sell_ratio`` (day-close OI /
    day-mean ratios at '1d').
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format (end inclusive; archives lag ~1 day).
        interval: Output granularity (default '1d').
    Returns:
        str: Header + CSV of positioning metric rows for the requested range.
    """
    return route_to_vendor(
        "get_binance_vision_metrics", symbol, start_date, end_date, interval,
    )


@tool
def get_binance_vision_book_depth(
    symbol: Annotated[str, "Binance USDT-M perpetual symbol, e.g. BTCUSDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    interval: Annotated[str, "Output granularity: '1d' (default), '1h' or '5m'"] = "1d",
) -> str:
    """Retrieve historical order-book DEPTH for a Binance USDT-M perp (archived).

    data.binance.vision ``bookDepth`` archives: resting depth within SIGNED
    ±percentage price bands of mid (negative = bid side, positive = ask
    side; bands at ±0.2/1/2/3/4/5%), ~30-second source grain, back years.
    Depth persistence is the liquidation-cascade context — a thin book into
    a falling price means slippage amplifies forced selling; a thick book
    absorbing a dump signals real demand. PIT-correct for historical dates;
    every zip is sha256-verified. Returns ``time, percentage, depth,
    notional`` (~12 band rows per day at '1d').
    Args:
        symbol: Binance USDT-M perp symbol, e.g. BTCUSDT.
        start_date: Start date in yyyy-mm-dd format.
        end_date: End date in yyyy-mm-dd format (end inclusive; archives lag ~1 day).
        interval: Output granularity (default '1d').
    Returns:
        str: Header + CSV of per-band depth rows for the requested range.
    """
    return route_to_vendor(
        "get_binance_vision_book_depth", symbol, start_date, end_date, interval,
    )
