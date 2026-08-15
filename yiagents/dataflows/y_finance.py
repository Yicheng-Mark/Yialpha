import json
import logging
from collections.abc import Callable
from datetime import datetime
from io import StringIO
from typing import Annotated

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from .disk_cache import cached_or_fetch, safe_cache_component, vendor_cache_dir
from .feature_registry import compute_derived
from .indicator_catalog import TOOL_DESCRIPTIONS
from .stockstats_utils import (
    OHLCV_CACHE_TTL_SECONDS,
    _assert_ohlcv_not_stale,
    compute_indicator,
    filter_financials_by_date,
    load_ohlcv,
    read_cached_ohlcv,
    yf_retry,
)
from .symbol_utils import NoMarketDataError, normalize_symbol
from .utils import current_pit_end, overview_would_leak_future

logger = logging.getLogger(__name__)

#: The indicator-window tools' gate + report trailer, rendered from the
#: shared catalog (yiagents/dataflows/indicator_catalog.py — the single
#: source of truth this dict used to be a drifted copy of). Kept as a named
#: module attribute so cross-file consistency tests assert against exactly
#: what ``get_stock_stats_indicators_window`` uses.
best_ind_params = TOOL_DESCRIPTIONS

#: Statements / snapshot payloads change at most daily (filings and Yahoo's
#: refresh cadence), so a ~24h on-disk cache collapses the 3-4 yfinance
#: round trips per statement per ticker per run into one. Windows of daily
#: history that end STRICTLY IN THE PAST are immutable once written, so the
#: same TTL bounds them too.
_STATEMENT_CACHE_TTL_DAYS = 1.0


def _history_cache_ttl_days(end_date: str) -> float:
    """TTL for a cached daily-history window, keyed on whether it reaches today.

    A window whose ``end_date`` is today or later can still gain rows — the
    current day's bar appears only after the close, so a snapshot taken in the
    morning would be served for a full 24h and miss today's close. Such windows
    reuse the OHLCV pipeline's same-day refresh TTL (:data:
    `OHLCV_CACHE_TTL_SECONDS`, 15 minutes): recent enough that an intraday run
    picks up today's close soon after it publishes, long enough that a day
    with no bar at all (weekend, holiday) cannot trigger a download on every
    call. Windows entirely in the past are immutable and keep the ~24h TTL.
    Unparseable dates fall back to the daily TTL (statements-grade caching).
    """
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return _STATEMENT_CACHE_TTL_DAYS
    if end_dt >= datetime.now().date():
        return OHLCV_CACHE_TTL_SECONDS / 86400.0
    return _STATEMENT_CACHE_TTL_DAYS


class _EmptyVendorPayload(Exception):
    """Internal marker: the vendor returned an empty payload.

    Raised from inside a ``cached_or_fetch`` fetch callable so an empty
    result is NOT persisted for the TTL — an empty frame cached for 24h
    would stop a later retry from seeing data that has since appeared
    (e.g. a reflection entry whose outcome rows were not published yet).
    When a *stale* cache exists, ``cached_or_fetch`` serves it with the
    usual stale-serve warning before this marker ever propagates, which is
    the right precedence: old statements beat no statements. Callers catch
    the marker and take their existing empty-path handling.
    """


def _cached_ticker_info(ticker: str, canonical: str) -> dict:
    """``yf.Ticker(canonical).info`` behind the ~24h statements cache.

    The raw dict round-trips through JSON (``default=str`` mirrors the
    yfnews payload wrapping). A corrupt cache raises the vendor's typed
    error rather than returning stale garbage.
    """
    filename = f"stmt_{safe_cache_component(canonical)}_fundamentals.json"

    def _fetch() -> bytes:
        ticker_obj = yf.Ticker(canonical)
        info = yf_retry(lambda: ticker_obj.info)
        if not info:
            raise _EmptyVendorPayload(f"{canonical}: empty info")
        return json.dumps(info, default=str).encode("utf-8")

    try:
        raw = cached_or_fetch(
            vendor_cache_dir("yfinance"), filename, _fetch,
            ttl_days=_STATEMENT_CACHE_TTL_DAYS, vendor="yfinance",
        )
    except _EmptyVendorPayload:
        return {}
    assert raw is not None  # fail_open never set: fetch errors re-raise
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as err:
        raise NoMarketDataError(
            ticker, canonical,
            f"cached fundamentals payload for {canonical} is corrupt",
        ) from err
    if not isinstance(payload, dict):
        raise NoMarketDataError(
            ticker, canonical,
            f"cached fundamentals payload for {canonical} is not an object",
        )
    return payload


def _cached_yf_frame(
    canonical: str,
    filename: str,
    loader: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Serve a yfinance statement/insider frame behind the ~24h cache.

    The RAW vendor frame is cached (CSV bytes); PIT filtering by
    ``curr_date`` stays at the call site, so a cached serve is filtered per
    request exactly like fresh data. Empty vendor frames return uncached
    via :class:`_EmptyVendorPayload` (see its docstring).
    """
    def _fetch() -> bytes:
        frame = yf_retry(loader)
        if frame is None or frame.empty:
            raise _EmptyVendorPayload(f"{canonical}: empty frame for {filename}")
        return frame.to_csv().encode("utf-8")

    try:
        raw = cached_or_fetch(
            vendor_cache_dir("yfinance"), filename, _fetch,
            ttl_days=_STATEMENT_CACHE_TTL_DAYS, vendor="yfinance",
        )
    except _EmptyVendorPayload:
        return pd.DataFrame()
    assert raw is not None  # fail_open never set: fetch errors re-raise
    return pd.read_csv(StringIO(raw.decode("utf-8")), index_col=0)


def get_YFin_history_cached(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """``yf.Ticker(symbol).history(start, end)`` behind the TTL cache.

    Same frame, same window (inclusive start, exclusive end), same retry
    policy as the direct call — used by the reflection layer, where a batch
    run over N tickers re-downloads the SAME benchmark window once per
    ticker. Windows that end in the past are immutable and cached ~24h;
    windows reaching today are cached only :data:`OHLCV_CACHE_TTL_SECONDS`
    (15 min) so today's close is picked up minutes after it publishes
    instead of the next day (see :func:`_history_cache_ttl_days`). The index
    is restored as a DatetimeIndex when it round-tripped as date strings
    (yfinance daily history), so downstream ``index.date`` slicing keeps
    working; other index shapes are returned exactly as the CSV round-trip
    produced them. Empty vendor frames return uncached (see
    :class:`_EmptyVendorPayload`).
    """
    filename = (
        f"hist_{safe_cache_component(symbol)}_{start_date}_{end_date}.csv"
    )

    def _fetch() -> bytes:
        ticker = yf.Ticker(symbol)
        frame = yf_retry(lambda: ticker.history(start=start_date, end=end_date))
        if frame is None or frame.empty:
            raise _EmptyVendorPayload(
                f"{symbol}: empty history {start_date}..{end_date}"
            )
        return frame.to_csv().encode("utf-8")

    try:
        raw = cached_or_fetch(
            vendor_cache_dir("yfinance"), filename, _fetch,
            ttl_days=_history_cache_ttl_days(end_date), vendor="yfinance",
        )
    except _EmptyVendorPayload:
        return pd.DataFrame()
    assert raw is not None  # fail_open never set: fetch errors re-raise
    frame = pd.read_csv(StringIO(raw.decode("utf-8")), index_col=0)
    if not isinstance(frame.index, pd.DatetimeIndex) and not (
        pd.api.types.is_numeric_dtype(frame.index)
    ):
        # Date-string index (yfinance daily history): restore DatetimeIndex
        # semantics. utc=True tolerates a window straddling a DST change
        # (mixed UTC offsets) and preserves each bar's calendar date.
        # Numeric indexes (mocks / odd frames) are left exactly as parsed.
        parsed = pd.to_datetime(frame.index, utc=True, errors="coerce")
        if not parsed.isna().any():
            frame.index = parsed
    return frame


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    # PIT guard: a backtest must never see rows after the analysis date. The
    # stock-data tool carries no analysis-date argument (the LLM picks end_date
    # from its prompt context), so clamp here against the run's pinned analysis
    # date. Live mode (no analysis date pinned) is a no-op pass-through.
    end_date = current_pit_end(end_date) or end_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Resolve broker/forex symbols to Yahoo's convention (XAUUSD+ -> GC=F).
    canonical = normalize_symbol(symbol)

    # Opportunistic cache reuse: the indicator pipeline's per-symbol OHLCV
    # cache (5y window, PIT-filtered to end_date, freshness + staleness rules
    # enforced by read_cached_ohlcv) often already holds exactly this window —
    # serving it here avoids a second Yahoo round-trip for the same symbol in
    # one run. Any miss (no cache, stale, window not covered) falls through
    # to the original online path unchanged.
    data = None
    cached = read_cached_ohlcv(canonical, end_date)
    if cached is not None:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        sliced = cached[cached["Date"] >= start_dt]
        if not sliced.empty and cached["Date"].min() <= start_dt:
            sliced = sliced.copy()
            sliced.index = pd.DatetimeIndex(sliced.pop("Date"))
            data = sliced

    if data is None:
        ticker = yf.Ticker(canonical)
        # yfinance treats ``end`` as EXCLUSIVE, so it would drop the requested
        # end_date row (and the current day when end_date is today). Request
        # one day past end_date so the requested range is actually inclusive
        # (#986/#987).
        end_inclusive = (end_dt + relativedelta(days=1)).strftime("%Y-%m-%d")
        data = yf_retry(
            lambda: ticker.history(start=start_date, end=end_inclusive),
            symbol=symbol,
            canonical=canonical,
        )
        # Empty result means the symbol is unknown/delisted. Raise a typed
        # error instead of returning prose: the routing layer turns it into a
        # single unambiguous "no data" signal so the agent never fabricates
        # a price.
        if data.empty:
            raise NoMarketDataError(
                symbol, canonical, f"no rows between {start_date} and {end_date}"
            )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Reject a stale frame (e.g. a year-old partial response) before it is
    # formatted into the report. Raises NoMarketDataError, which the router
    # turns into one clear unavailable signal (#1021).
    _assert_ohlcv_not_stale(data, end_date, symbol, canonical)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information; note the resolved symbol when it differs so the
    # agent (and user) can see which instrument was actually priced.
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Stock data for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string

def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)

        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []

        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')

            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"

            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)

        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"

    except NoMarketDataError:
        raise  # Unknown/delisted symbol — let the router emit the sentinel
    except Exception:
        # Do NOT fall back to the per-day N+1 loop: that silently recomputes
        # one row at a time (re-loading OHLCV each call) and masks the real
        # failure behind a partial result. Let the router emit its sentinel so
        # the agent reports "unavailable" rather than a truncated series.
        logger.exception(
            "bulk stockstats calc failed for %s/%s — surfacing to router",
            symbol, indicator,
        )
        raise

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: str,
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, curr_date)

    # Derived features (vol estimators, OBV, ...) compute on the raw frame
    # via the feature registry; stockstats names go through wrap().
    derived = compute_derived(data, indicator)
    if derived is not None:
        dates = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")
        result_dict = {}
        for date_str, value in zip(dates, derived, strict=True):
            result_dict[date_str] = "N/A" if pd.isna(value) else str(float(value))
        return result_dict

    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Calculate the indicator for all rows at once (this triggers stockstats'
    # lazy computation; compute_indicator also applies vendor-scale fixes such
    # as mfi 0-1 -> 0-100 so the values match the Alpha Vantage scale).
    compute_indicator(df, indicator)

    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]

        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)

    return result_dict


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str | None, "current date, yyyy-mm-dd"] = None
):
    """Get company fundamentals overview from yfinance.

    yfinance ``.info`` is a single current-point snapshot with no date
    dimension -- PE, marketCap, EPS, beta are always *today's* values. On an
    explicit past ``curr_date`` (a backtest decision date) surfacing them would
    leak the future, so we refuse and let the router emit its NO_DATA_AVAILABLE
    sentinel (the fundamentals analyst is grounded to report "data not
    available" rather than fabricate). Live mode (``curr_date`` empty or
    today/future) keeps the snapshot -- it is legitimately current then.
    """
    canonical = normalize_symbol(ticker)
    if overview_would_leak_future(curr_date):
        raise NoMarketDataError(
            ticker, canonical,
            f"overview snapshot is point-in-time (today only); not valid as of {curr_date}",
        )
    try:
        info = _cached_ticker_info(ticker, canonical)

        if not info:
            raise NoMarketDataError(ticker, canonical, "no fundamentals returned")

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")

        # yfinance returns a stub dict (e.g. {"trailingPegRatio": None}) for
        # unknown symbols, so `info` is truthy but every field is empty. Treat
        # "no usable fields" as no data rather than emitting a bare header the
        # agent might fabricate around.
        if not lines:
            raise NoMarketDataError(ticker, canonical, "no fundamental fields returned")

        header = f"# Company Fundamentals for {canonical}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except NoMarketDataError:
        raise
    except Exception:
        # Raise instead of returning an "Error retrieving…" string: the router
        # (route_to_vendor) logs the failure and either falls through to the
        # next vendor or emits its sentinel. Returning prose here made the
        # agent treat the error message as if it were fundamentals data.
        logger.exception("fundamentals retrieval failed for %s", ticker)
        raise


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str | None, "current date in YYYY-MM-DD format"] = None
):
    """Get balance sheet data from yfinance."""
    canonical = normalize_symbol(ticker)
    freq_key = "quarterly" if freq.lower() == "quarterly" else "annual"
    try:
        if freq_key == "quarterly":
            loader = lambda: yf.Ticker(canonical).quarterly_balance_sheet  # noqa: E731
        else:
            loader = lambda: yf.Ticker(canonical).balance_sheet  # noqa: E731
        data = _cached_yf_frame(
            canonical,
            f"stmt_{safe_cache_component(canonical)}_balance_sheet_{freq_key}.csv",
            loader,
        )

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            raise NoMarketDataError(ticker, canonical, "no balance sheet data")

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()

        # Add header information
        header = f"# Balance Sheet data for {canonical} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except NoMarketDataError:
        raise
    except Exception:
        logger.exception("balance sheet retrieval failed for %s", ticker)
        raise


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str | None, "current date in YYYY-MM-DD format"] = None
):
    """Get cash flow data from yfinance."""
    canonical = normalize_symbol(ticker)
    freq_key = "quarterly" if freq.lower() == "quarterly" else "annual"
    try:
        if freq_key == "quarterly":
            loader = lambda: yf.Ticker(canonical).quarterly_cashflow  # noqa: E731
        else:
            loader = lambda: yf.Ticker(canonical).cashflow  # noqa: E731
        data = _cached_yf_frame(
            canonical,
            f"stmt_{safe_cache_component(canonical)}_cashflow_{freq_key}.csv",
            loader,
        )

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            raise NoMarketDataError(ticker, canonical, "no cash flow data")

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()

        # Add header information
        header = f"# Cash Flow data for {canonical} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except NoMarketDataError:
        raise
    except Exception:
        logger.exception("cash flow retrieval failed for %s", ticker)
        raise


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str | None, "current date in YYYY-MM-DD format"] = None
):
    """Get income statement data from yfinance."""
    canonical = normalize_symbol(ticker)
    freq_key = "quarterly" if freq.lower() == "quarterly" else "annual"
    try:
        if freq_key == "quarterly":
            loader = lambda: yf.Ticker(canonical).quarterly_income_stmt  # noqa: E731
        else:
            loader = lambda: yf.Ticker(canonical).income_stmt  # noqa: E731
        data = _cached_yf_frame(
            canonical,
            f"stmt_{safe_cache_component(canonical)}_income_statement_{freq_key}.csv",
            loader,
        )

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            raise NoMarketDataError(ticker, canonical, "no income statement data")

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()

        # Add header information
        header = f"# Income Statement data for {canonical} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except NoMarketDataError:
        raise
    except Exception:
        logger.exception("income statement retrieval failed for %s", ticker)
        raise


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    canonical = normalize_symbol(ticker)
    try:
        data = _cached_yf_frame(
            canonical,
            f"stmt_{safe_cache_component(canonical)}_insider_transactions.csv",
            lambda: yf.Ticker(canonical).insider_transactions,
        )

        # Empty is normal here (many valid symbols have no insider filings),
        # so report it plainly rather than treating the symbol as invalid.
        if data is None or data.empty:
            return f"No insider transactions reported for symbol '{canonical}'"

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()

        # Add header information
        header = f"# Insider Transactions data for {canonical}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string

    except Exception:
        logger.exception("insider transactions retrieval failed for %s", ticker)
        raise
