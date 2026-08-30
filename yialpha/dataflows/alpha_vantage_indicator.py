import logging

from .alpha_vantage_common import AlphaVantageNotConfiguredError, _make_api_request
from .errors import NoMarketDataError
from .indicator_catalog import (
    AV_COLUMNS,
    AV_DESCRIPTIONS,
    AV_SUPPORTED,
    YFINANCE_ONLY,
)

logger = logging.getLogger(__name__)

# The indicator gate, descriptions, and response-column map are rendered from
# the shared catalog (yialpha/dataflows/indicator_catalog.py) — the same
# single source the y_finance tool gate and the market analyst prompt use, so
# the three copies cannot drift again (the descriptions here had already lost
# mfi before the unification).
supported_indicators = AV_SUPPORTED
indicator_descriptions = AV_DESCRIPTIONS
col_name_map = AV_COLUMNS


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close"
) -> str:
    """
    Returns Alpha Vantage technical indicator values over a time window.

    Args:
        symbol: ticker symbol of the company
        indicator: technical indicator to get the analysis and report of
        curr_date: The current trading date you are trading on, YYYY-mm-dd
        look_back_days: how many days to look back
        interval: Time interval (daily, weekly, monthly)
        time_period: Number of data points for calculation
        series_type: The desired price type (close, open, high, low)

    Returns:
        String containing indicator values and description
    """
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    if indicator not in supported_indicators:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(supported_indicators.keys())}"
        )

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Get the full data for the period instead of making individual calls
    _, required_series_type = supported_indicators[indicator]

    # Use the provided series_type or fall back to the required one
    if required_series_type:
        series_type = required_series_type

    try:
        # Get indicator data for the period
        if indicator == "close_50_sma":
            data = _make_api_request("SMA", {
                "symbol": symbol,
                "interval": interval,
                "time_period": "50",
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "close_200_sma":
            data = _make_api_request("SMA", {
                "symbol": symbol,
                "interval": interval,
                "time_period": "200",
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "close_10_ema":
            data = _make_api_request("EMA", {
                "symbol": symbol,
                "interval": interval,
                "time_period": "10",
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "macd" or indicator == "macds" or indicator == "macdh":
            data = _make_api_request("MACD", {
                "symbol": symbol,
                "interval": interval,
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "rsi":
            data = _make_api_request("RSI", {
                "symbol": symbol,
                "interval": interval,
                "time_period": str(time_period),
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator in ["boll", "boll_ub", "boll_lb"]:
            data = _make_api_request("BBANDS", {
                "symbol": symbol,
                "interval": interval,
                "time_period": "20",
                "series_type": series_type,
                "datatype": "csv"
            })
        elif indicator == "atr":
            data = _make_api_request("ATR", {
                "symbol": symbol,
                "interval": interval,
                "time_period": str(time_period),
                "datatype": "csv"
            })
        elif indicator == "mfi":
            # Volume-based like ATR: no series_type parameter.
            data = _make_api_request("MFI", {
                "symbol": symbol,
                "interval": interval,
                "time_period": str(time_period),
                "datatype": "csv"
            })
        elif indicator in YFINANCE_ONLY:
            # Alpha Vantage has no endpoint for this indicator (kdj family, adx,
            # supertrend, cci, wr, stochrsi, roc, cmo, trix, vr — same shape as
            # the original vwma case). RAISE (do not return prose): a returned
            # message is a SUCCESS to the router, which then never falls through
            # to the yfinance vendor — the one that CAN compute it from OHLCV
            # via stockstats. NoMarketDataError routes to the next vendor in
            # the chain and, if none serves, to the sentinel.
            raise NoMarketDataError(
                symbol, symbol,
                f"Alpha Vantage has no {indicator} endpoint (compute from "
                f"OHLCV via the yfinance indicator vendor instead)",
            )
        else:
            # Unreachable behind the supported_indicators gate above; kept as a
            # typed raise (never a returned "Error:" string) for defence.
            raise NoMarketDataError(
                symbol, symbol, f"indicator {indicator} not implemented yet"
            )

        # Parse CSV data and extract values for the date range
        if not isinstance(data, str):
            raise NoMarketDataError(
                symbol, symbol,
                f"no CSV data returned for {indicator} (got {type(data).__name__})",
            )
        lines = data.strip().split('\n')
        if len(lines) < 2:
            raise NoMarketDataError(
                symbol, symbol, f"no data rows returned for {indicator}"
            )

        # Parse header and data
        header = [col.strip() for col in lines[0].split(',')]
        try:
            date_col_idx = header.index('time')
        except ValueError:
            raise NoMarketDataError(
                symbol, symbol,
                f"'time' column not found for {indicator}; "
                f"available columns: {header}",
            ) from None

        # Map internal indicator names to expected CSV column names from Alpha
        # Vantage (rendered from the shared catalog as ``col_name_map``).
        target_col_name = col_name_map.get(indicator)

        if not target_col_name:
            # Default to the second column if no specific mapping exists
            value_col_idx = 1
        else:
            try:
                value_col_idx = header.index(target_col_name)
            except ValueError:
                raise NoMarketDataError(
                    symbol, symbol,
                    f"column {target_col_name!r} not found for indicator "
                    f"{indicator!r}; available columns: {header}",
                ) from None

        result_data = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(',')
            if len(values) > value_col_idx:
                try:
                    date_str = values[date_col_idx].strip()
                    # Parse the date
                    date_dt = datetime.strptime(date_str, "%Y-%m-%d")

                    # Check if date is in our range
                    if before <= date_dt <= curr_date_dt:
                        value = values[value_col_idx].strip()
                        result_data.append((date_dt, value))
                except (ValueError, IndexError):
                    continue

        # Sort by date and format output
        result_data.sort(key=lambda x: x[0])

        ind_string = ""
        for date_dt, value in result_data:
            ind_string += f"{date_dt.strftime('%Y-%m-%d')}: {value}\n"

        if not ind_string:
            ind_string = "No data available for the specified date range.\n"

        result_str = (
            f"## {indicator.upper()} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + ind_string
            + "\n\n"
            + indicator_descriptions.get(indicator, "No description available.")
        )

        return result_str

    except AlphaVantageNotConfiguredError:
        # Vendor unavailable (no API key). Let it propagate so the router can
        # fall back / emit the no-data sentinel instead of returning this as a
        # successful-looking error string.
        raise
    except NoMarketDataError:
        # Typed no-data (including the vwma hand-off above): re-raise untouched
        # so the router tries the next vendor / emits its sentinel. Not logged
        # as an exception — this is the expected degradation path, not a fault.
        raise
    except Exception:
        # Raise instead of returning an "Error retrieving …" string: returning
        # prose made the agent treat the failure message as indicator data.
        # The router (route_to_vendor) logs the failure and either falls through
        # to the next vendor or emits its sentinel.
        logger.exception(
            "Alpha Vantage indicator %s failed for %s", indicator, symbol,
        )
        raise
