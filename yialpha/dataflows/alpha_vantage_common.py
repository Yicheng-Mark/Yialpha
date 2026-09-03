import hashlib
import json
import logging
import os
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from ..logging_config import redact_secrets
from .disk_cache import cached_or_fetch, vendor_cache_dir
from .errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from .netretry import with_transient_retry

logger = logging.getLogger(__name__)

API_BASE_URL = "https://www.alphavantage.co/query"

# Network timeout (seconds) so a stalled Alpha Vantage request can't hang the
# CLI/agents indefinitely (#990).
REQUEST_TIMEOUT = 30


class AlphaVantageNotConfiguredError(VendorNotConfiguredError):
    """Raised when Alpha Vantage is selected but no API key is configured.

    A VendorNotConfiguredError (and thus still a ValueError), so the routing
    layer's "vendor unavailable" handling and existing ValueError callers both
    keep working.
    """
    pass


def get_api_key() -> str:
    """Retrieve the API key for Alpha Vantage from environment variables."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise AlphaVantageNotConfiguredError(
            "ALPHA_VANTAGE_API_KEY environment variable is not set."
        )
    return api_key

def format_datetime_for_api(date_input) -> str:
    """Convert various date formats to YYYYMMDDTHHMM format required by Alpha Vantage API."""
    if isinstance(date_input, str):
        # If already in correct format, return as-is
        if len(date_input) == 13 and 'T' in date_input:
            return date_input
        # Try to parse common date formats
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            return dt.strftime("%Y%m%dT0000")
        except ValueError:
            try:
                dt = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
                return dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                raise ValueError(f"Unsupported date format: {date_input}") from None
    elif isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%dT%H%M")
    else:
        raise ValueError(f"Date must be string or datetime object, got {type(date_input)}")

class AlphaVantageRateLimitError(VendorRateLimitError):
    """Raised when the Alpha Vantage API rate limit is exceeded."""
    pass

def _raw_api_request(function_name: str, params: dict) -> dict | str:
    """Raw (uncached) API call + response classification. Internal: use
    :func:`_make_api_request`, which caches around this.

    Raises:
        AlphaVantageRateLimitError: When API rate limit is exceeded
    """
    # Create a copy of params to avoid modifying the original
    api_params = params.copy()
    # Kept for exception-message redaction below: vendor notices echo the key
    # back verbatim, and the message must never carry it into logs (#R6).
    api_key = get_api_key()
    api_params.update({
        "function": function_name,
        "apikey": api_key,
        "source": "yialpha",
    })

    # Handle the entitlement parameter: an explicit per-call params entry
    # only (it selects a premium data tier). There is no module-level
    # entitlement state.
    entitlement = api_params.get("entitlement")

    if entitlement:
        api_params["entitlement"] = entitlement
    elif "entitlement" in api_params:
        # Remove entitlement if it's None or empty
        api_params.pop("entitlement", None)

    response = requests.get(API_BASE_URL, params=api_params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    response_text = response.text

    # Error responses are JSON; data responses are usually CSV (or data-keyed
    # JSON). A non-JSON body is normal data.
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    # Alpha Vantage reports problems via "Information" / "Note". Classify so a
    # genuine rate limit and an invalid/missing key aren't conflated (#991):
    # rate-limit phrasing is checked first because those notices also mention
    # "API key" ("your API key ... 25 requests per day"). The notice is
    # redacted BEFORE classification and raising: the vendor echoes the
    # apikey back verbatim, and these messages propagate into caller logs
    # (#R6). Scrubbing keeps the surrounding wording (and thus the
    # rate-limit / bad-key classification) intact.
    notice = response_json.get("Information") or response_json.get("Note")
    if notice:
        notice = redact_secrets(notice, extra_secrets=(api_key,))
        low = notice.lower()
        if any(m in low for m in ("rate limit", "requests per day", "call frequency", "premium")):
            raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit exceeded: {notice}")
        if "api key" in low or "apikey" in low:
            # Reuse the existing "not configured" error so a bad key surfaces as
            # a real, actionable failure rather than a mislabeled rate limit (#991).
            raise AlphaVantageNotConfiguredError(f"Alpha Vantage API key invalid or missing: {notice}")

    # "Error Message" is AV's hard-failure body (e.g. "Invalid API call" for an
    # unknown/invalid symbol or parameter). Returning it as text made every
    # caller treat the error prose as a successful payload; classify it as the
    # taxonomy's no-data error so the router falls through to the next vendor
    # or emits its sentinel instead.
    error_message = response_json.get("Error Message")
    if error_message:
        symbol = params.get("symbol") or params.get("tickers") or function_name
        raise NoMarketDataError(
            str(symbol),
            detail=f"Alpha Vantage error: {redact_secrets(str(error_message), extra_secrets=(api_key,))}",
        )

    return response_text


def _make_api_request(function_name: str, params: dict) -> dict | str:
    """Cached + retried Alpha Vantage request (the vendor entry point).

    Wraps :func:`_raw_api_request` with a 15-minute on-disk cache keyed by
    function + params (sans API key): the free tier updates at most daily,
    and a batch run over N tickers re-requests the *same* symbol data per
    ticker — the cache collapses that to one network call. Rate-limit /
    bad-key classification still happens inside the raw call on every actual
    fetch. Transport hiccups (connection reset, timeout) get one retry.
    """
    # Cache key excludes the API key but includes the entitlement (an
    # explicit params entry selecting a different data tier, i.e. different
    # bytes for the same query).
    key_blob = json.dumps(
        {"function": function_name, "params": params,
         "entitlement": params.get("entitlement")},
        sort_keys=True, default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(key_blob).hexdigest()[:12]
    filename = f"{function_name}_{digest}.txt"

    def _fetch() -> bytes:
        text = with_transient_retry(
            lambda: _raw_api_request(function_name, params),
            vendor="alphavantage",
            retry_on=(
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ),
        )
        # _raw_api_request's declared type is dict | str (its legacy contract);
        # in practice it always returns the response text. Serialize the
        # unlikely dict so the cache stores bytes either way.
        if not isinstance(text, str):
            text = json.dumps(text)
        return text.encode("utf-8")

    raw = cached_or_fetch(
        vendor_cache_dir("alphavantage"), filename, _fetch,
        ttl_days=15.0 / (24.0 * 60.0), vendor="alphavantage",
    )
    assert raw is not None  # fail_open never set: raw errors re-raise
    return raw.decode("utf-8", errors="replace")


def _filter_csv_by_date_range(csv_data: str, start_date: str, end_date: str) -> str:
    """
    Filter CSV data to include only rows within the specified date range.

    Args:
        csv_data: CSV string from Alpha Vantage API
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Filtered CSV string
    """
    if not csv_data or csv_data.strip() == "":
        return csv_data

    try:
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_data))

        # Assume the first column is the date column (timestamp)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        filtered_df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]

        # Convert back to CSV string
        return filtered_df.to_csv(index=False)

    except Exception:
        # Do NOT return the unfiltered CSV on filter failure — that would leak
        # future rows into a backtest (lookahead) and over-feed the analyst in
        # live mode. Surface the failure so the router emits its sentinel,
        # matching the y_finance.py contract (see _get_stock_stats_bulk,
        # which also raises on calculation failure).
        logger.warning("Failed to filter CSV data by date range", exc_info=True)
        raise
