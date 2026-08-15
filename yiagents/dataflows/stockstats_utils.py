import glob
import logging
import os
import socket
import threading
import time
from contextlib import nullcontext

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from yiagents.batch.locks import FileLock

from .config import get_config
from .symbol_utils import NoMarketDataError, normalize_symbol
from .utils import is_filing_public, safe_ticker_component

logger = logging.getLogger(__name__)

# HTTP read-timeout safety net for yfinance. yfinance has no default read
# timeout, so under Yahoo rate-limiting a stalled socket blocks forever and
# hangs the whole batch — the same half-open-socket class as the LLM read
# timeout in llm_clients/openai_client.py. Two layers, both opt-in via
# YIAGENTS_HTTP_TIMEOUT_S (seconds), off by default:
#   1. socket.setdefaulttimeout — process-wide backstop for the yfinance calls
#      that take no per-call timeout (Ticker.info, get_news, Search). Every
#      other network path here already passes an explicit timeout (Reddit,
#      FRED, Alpha Vantage, the LLM clients), so this effectively binds only
#      yfinance.
#   2. timeout= on yf.download — the OHLCV path, the call that actually hangs
#      the pipeline.
_HTTP_TIMEOUT_ENV = os.environ.get("YIAGENTS_HTTP_TIMEOUT_S")
YF_HTTP_TIMEOUT: float | None = None
if _HTTP_TIMEOUT_ENV:
    try:
        parsed = float(_HTTP_TIMEOUT_ENV)
        if parsed > 0:
            YF_HTTP_TIMEOUT = parsed
            socket.setdefaulttimeout(YF_HTTP_TIMEOUT)
        else:
            logger.warning(
                "YIAGENTS_HTTP_TIMEOUT_S=%r is not positive; ignoring it "
                "(no HTTP timeout will be applied)", _HTTP_TIMEOUT_ENV,
            )
    except ValueError:
        # Same contract as the LLM timeout (llm_clients/_timeout.py): a bad
        # value must be visible, never silently swallowed.
        logger.warning(
            "YIAGENTS_HTTP_TIMEOUT_S=%r is not a number; ignoring it "
            "(no HTTP timeout will be applied)", _HTTP_TIMEOUT_ENV,
        )

# A vendor's latest OHLCV row this many calendar days before the requested date
# is treated as stale. Generous enough to span long holiday weekends, tight
# enough to catch the year-old frames yfinance occasionally returns (#1021).
MAX_OHLCV_STALE_DAYS = 10

# How long a same-day cache that does not yet reach the requested day may be
# reused before it is refetched (#1150). Short enough that an intraday run picks
# up today's close soon after it publishes, long enough that a day with no bar
# at all (weekend, holiday) cannot trigger a download on every call. Historical
# days are immutable, so they never trigger a refresh (see _needs_same_day_refresh).
OHLCV_CACHE_TTL_SECONDS = 900

# Transport-level failures from yfinance's HTTP stack. ``OSError`` is the single
# root: socket.timeout and ConnectionError are OSError subclasses; requests'
# RequestException derives from IOError (== OSError); and curl_cffi's CurlError
# (Timeout/ConnectionError/etc., yfinance's browser-impersonation backend) also
# derives from OSError. Used in yf_retry to convert "Yahoo unreachable" into the
# typed NoMarketDataError so the routing layer degrades instead of crashing the
# node — Yahoo unreachability must never abort an analysis (perps fall back to
# Binance; stock/crypto runs degrade rather than hard-fail).
_YF_NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError,)


def yf_retry(func, max_retries=3, base_delay=2.0, symbol=None, canonical=None):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Transport-level failures (connection timeout, DNS, TLS,
    curl_cffi errors) and exhausted rate-limit retries are converted to
    NoMarketDataError so the routing layer degrades gracefully instead of
    crashing the node — Yahoo unreachability must never abort an analysis.
    The success path (Yahoo returns data) is unchanged.

    ``symbol``/``canonical`` are passed so the typed error names the right
    instrument; callers without context leave them None and the sentinel
    carries a placeholder.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError as err:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            raise NoMarketDataError(
                symbol or "?", canonical or symbol or "?",
                "Yahoo Finance rate-limited after retries",
            ) from err
        except _YF_NETWORK_ERRORS as exc:
            raise NoMarketDataError(
                symbol or "?", canonical or symbol or "?",
                f"Yahoo unreachable ({type(exc).__name__})",
            ) from exc


def _ensure_date_column(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the date column to ``Date``.

    Some yfinance builds leave the index unnamed (so ``reset_index()`` yields
    ``index``) or use ``Datetime`` for intraday data. Rename the first
    date-like column so indicators don't silently drop when it isn't ``Date``.
    """
    if "Date" in data.columns:
        return data
    for candidate in ("index", "Datetime", "date"):
        if candidate in data.columns:
            return data.rename(columns={candidate: "Date"})
    return data


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps.

    Gap filling is forward-only (``ffill``). The previous ``.ffill().bfill()``
    filled LEADING NaNs with the next row's value — when the caller's
    ``curr_date`` filter lands at the head of the 5y window that next row is
    dated AFTER curr_date, i.e. a look-ahead: the "current" price at the
    decision date was actually tomorrow's. Leading NaNs now stay NaN (honest
    missing data; indicators report N/A during warm-up instead of borrowing
    the future).
    """
    data = _ensure_date_column(data)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill()

    return data


def _coerce_ohlcv_dates(data: pd.DataFrame) -> pd.Series:
    """Return parsed dates from an OHLCV frame, whether Date is a column or the index."""
    if "Date" in data.columns:
        return pd.to_datetime(data["Date"], errors="coerce").dropna()
    # yfinance keeps the dates in the index (a DatetimeIndex, sometimes unnamed).
    if isinstance(data.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(data.index, errors="coerce")).dropna()
    # Fallback: expose the index and look for any date-like column.
    df = data.reset_index()
    for col in ("Date", "Datetime", "date", "index"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if not parsed.empty:
                return parsed
    return pd.Series(dtype="datetime64[ns]")


def _assert_ohlcv_not_stale(
    data: pd.DataFrame,
    curr_date: str,
    symbol: str,
    canonical: str | None = None,
    *,
    max_stale_days: int = MAX_OHLCV_STALE_DAYS,
) -> None:
    """Reject OHLCV whose latest row is far older than curr_date.

    Raises NoMarketDataError (with a stale-specific detail) so the router treats
    it like any other "no usable data from this vendor" — try the next vendor,
    then emit one clear unavailable signal. Empty frames are left to the
    caller's existing no-data handling; this guards only the dangerous case of
    present-but-stale rows (a vendor returning a year-old frame that would
    otherwise feed wrong prices to the agent, #1021).
    """
    if data is None or data.empty:
        return
    requested = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(requested):
        return
    requested = requested.normalize()
    dates = _coerce_ohlcv_dates(data)
    if dates.empty:
        return
    latest = dates.max().normalize()
    stale_days = (requested - latest).days
    if stale_days > max_stale_days:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"latest row is {latest.date()}, {stale_days} days before the "
            f"requested {requested.date()} (stale) — refusing to use it",
        )


def _needs_same_day_refresh(data_file, curr_date_dt, today_date) -> bool:
    """Whether a cached frame must be refetched to reflect the requested day.

    The cache is now ONE fixed-name file per symbol (see ``_ohlcv_cache_path``),
    so without this a run started before the day's bar was final would keep
    serving that snapshot to every later run (#1150). Only the current day is
    affected: a historical date's rows are immutable, so the cache is reused
    unconditionally.
    """
    if curr_date_dt.date() < today_date.date():
        return False
    return time.time() - os.path.getmtime(data_file) > OHLCV_CACHE_TTL_SECONDS


def _ohlcv_cache_window() -> tuple[str, str]:
    """The download window for every OHLCV fetch (NOT part of the cache filename).

    yfinance ``end`` is EXCLUSIVE; request tomorrow so today's row is included
    when curr_date is the current day (#986). Look-ahead is still prevented by
    the caller's curr_date filter. The window shifts daily but the cache file
    name is fixed; freshness of the reused file is governed by its mtime via
    ``_needs_same_day_refresh``.
    """
    today_date = pd.Timestamp.today()
    start_str = (today_date - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end_str = (today_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return start_str, end_str


def _ohlcv_cache_path(config: dict, safe_symbol: str) -> str:
    """Per-symbol cache file: ONE fixed name over the rolling 5y window.

    The name deliberately embeds no dates. The previous
    ``{symbol}-YFin-data-{start}-{end}.csv`` scheme re-derived the window every
    call (``end`` is always tomorrow), so each symbol minted a NEW ~1250-row
    CSV every day and nothing ever deleted the old ones — unbounded cache
    growth per symbol per day. With a fixed name the file is simply rewritten
    on every refresh and its mtime drives freshness; the leftover dated files
    from the old scheme are removed by :func:`_purge_legacy_ohlcv_caches`.
    """
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    return os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-YFin-data.csv",
    )


def _purge_legacy_ohlcv_caches(config: dict, safe_symbol: str) -> None:
    """Best-effort removal of this symbol's legacy date-stamped cache files.

    Called right after a fresh cache write, so a one-time upgrade cleans up the
    per-day files the old filename scheme accumulated. ``safe_symbol`` is
    already path-validated (``safe_ticker_component``), so the glob cannot
    escape the cache directory; failures to delete are logged, never raised —
    cache hygiene must not fail a data call that already succeeded.
    """
    pattern = os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-YFin-data-*.csv",
    )
    for legacy in glob.glob(pattern):
        try:
            os.remove(legacy)
        except OSError as exc:
            logger.warning("could not remove legacy OHLCV cache %s: %s", legacy, exc)


# In-process memo of CLEANED OHLCV frames, keyed by the cache-file path and
# validated against the file's mtime. A single-ticker run re-enters
# load_ohlcv 10+ times (indicator windows, market regime, the validator, the
# risk overlay, bulk stats); the disk cache already dedups the network, but
# every call still paid config deepcopy + FileLock + a full CSV re-read +
# re-cleaning. The memo serves the date-INDEPENDENT cleaned frame (per-call
# curr_date filtering stays at the call site, after the memo), so no call
# sequence can observe a different result than the disk path alone would
# produce — the same-day-refresh rule is re-checked on every memo hit, and
# any cache rewrite (this process or another) bumps the mtime and
# invalidates the entry. Batch workers are threads, so the dict is guarded.
_OHLCV_MEMO_LOCK = threading.Lock()
_OHLCV_MEMO: dict[str, tuple[float, pd.DataFrame]] = {}
#: Upper bound so a very long-lived batch process over hundreds of tickers
#: cannot grow the memo without limit (each 5y daily frame is ~1250 rows).
_OHLCV_MEMO_MAX_ENTRIES = 64


def _ohlcv_memo_get(data_file: str, mtime: float) -> pd.DataFrame | None:
    """The memoized cleaned frame for ``data_file`` iff keyed to ``mtime``."""
    with _OHLCV_MEMO_LOCK:
        entry = _OHLCV_MEMO.get(data_file)
    if entry is None or entry[0] != mtime:
        return None
    return entry[1]


def _ohlcv_memo_set(data_file: str, mtime: float, frame: pd.DataFrame) -> None:
    """Remember the cleaned frame under (data_file, mtime)."""
    with _OHLCV_MEMO_LOCK:
        if len(_OHLCV_MEMO) >= _OHLCV_MEMO_MAX_ENTRIES and data_file not in _OHLCV_MEMO:
            _OHLCV_MEMO.pop(next(iter(_OHLCV_MEMO)))  # FIFO eviction
        _OHLCV_MEMO[data_file] = (mtime, frame)


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 5 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices. Within one process
    the CLEANED frame is additionally memoized (keyed by cache-file mtime,
    see ``_OHLCV_MEMO``); the curr_date filter stays per-call, after the
    memo, so results are identical to the disk path alone.
    """
    # Resolve broker/forex symbols (XAUUSD+ -> GC=F) to Yahoo's convention,
    # then reject values that would escape the cache directory when
    # interpolated into the cache filename (e.g. ``../../tmp/x``).
    canonical = normalize_symbol(symbol)
    safe_symbol = safe_ticker_component(canonical)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    data_file = _ohlcv_cache_path(config, safe_symbol)
    today_date = pd.Timestamp.today()
    start_str, end_str = _ohlcv_cache_window()

    # The cache read + (on miss) download + write must be atomic per symbol:
    # two workers fetching the SAME symbol would otherwise both miss the cache,
    # both download, and race the non-atomic to_csv (a concurrent reader could
    # see a half-written file). Keyed by file path, so different symbols use
    # different locks and stay fully concurrent. Holding the lock across the
    # network download only blocks other workers fetching THIS symbol (rare —
    # the batch runner dedups tickers), and the second waiter then hits the
    # freshly-written cache instead of re-downloading.
    lock = (
        FileLock(data_file)
        if config.get("batch_ohlcv_lock", True)
        else nullcontext()
    )
    with lock:
        # A cached file may be empty if a prior fetch failed (unknown symbol,
        # transient rate limit). Treat an empty/columnless cache as a miss and
        # re-fetch rather than serving the poisoned file forever. The same-day
        # refresh rule is checked BEFORE the (memo or CSV) read, exactly like
        # the original condition ordering: a file that must be refetched is
        # never served from any cache level.
        data = None
        from_memo = False
        if os.path.exists(data_file) and not _needs_same_day_refresh(
            data_file, curr_date_dt, today_date
        ):
            memoized = _ohlcv_memo_get(data_file, os.path.getmtime(data_file))
            if memoized is not None:
                data = memoized  # already cleaned
                from_memo = True
            else:
                cached = pd.read_csv(
                    data_file, on_bad_lines="skip", encoding="utf-8"
                )
                if not cached.empty and "Close" in cached.columns:
                    data = cached

        if data is None:
            downloaded = yf_retry(
                lambda: yf.download(
                    canonical,
                    start=start_str,
                    end=end_str,
                    multi_level_index=False,
                    progress=False,
                    auto_adjust=True,
                    timeout=YF_HTTP_TIMEOUT,
                ),
                symbol=symbol,
                canonical=canonical,
            )
            downloaded = _ensure_date_column(downloaded.reset_index())
            # Only cache real data — never persist an empty frame.
            if downloaded.empty or "Close" not in downloaded.columns:
                raise NoMarketDataError(
                    symbol, canonical, "Yahoo Finance returned no rows"
                )
            downloaded.to_csv(data_file, index=False, encoding="utf-8")
            # The cache name is now fixed; drop this symbol's legacy dated
            # files (one per day under the old scheme) so they cannot pile up.
            _purge_legacy_ohlcv_caches(config, safe_symbol)
            data = downloaded

        if not from_memo:
            data = _clean_dataframe(data)
            # Key the memo to the cache file's mtime as seen inside the lock:
            # any later rewrite (this process or another) changes the mtime
            # and invalidates the entry on the next _ohlcv_memo_get.
            _ohlcv_memo_set(data_file, os.path.getmtime(data_file), data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

    # Reject a stale frame (latest row far older than curr_date) rather than
    # feeding year-old prices into indicators (#1021).
    _assert_ohlcv_not_stale(data, curr_date, symbol, canonical)

    return data


def read_cached_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame | None:
    """Read the per-symbol OHLCV cache with ZERO network fallback.

    Same freshness/staleness/PIT rules as :func:`load_ohlcv`, but a miss
    returns ``None`` instead of downloading — so callers (the stock-data
    tool) can opportunistically reuse the cache and only hit Yahoo when it
    cannot serve the request. The frame is PIT-filtered to ``curr_date``.

    Serves from the in-process memo (``_OHLCV_MEMO``) when it is keyed to the
    cache file's current mtime — a run that already cleaned this frame via
    ``load_ohlcv`` skips the full CSV re-read + re-clean on every call. The
    mtime keying means any cache rewrite invalidates the memo, and the
    same-day-refresh rule above already rejected over-age files, so a memo hit
    is exactly as fresh as the CSV read it replaces.
    """
    canonical = normalize_symbol(symbol)
    safe_symbol = safe_ticker_component(canonical)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)
    today_date = pd.Timestamp.today()

    data_file = _ohlcv_cache_path(config, safe_symbol)
    if not os.path.exists(data_file):
        return None
    if _needs_same_day_refresh(data_file, curr_date_dt, today_date):
        return None
    mtime = os.path.getmtime(data_file)
    data = _ohlcv_memo_get(data_file, mtime)
    if data is None:
        try:
            cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        except (OSError, ValueError):
            return None
        if cached.empty or "Close" not in cached.columns:
            return None
        data = _clean_dataframe(cached)
        _ohlcv_memo_set(data_file, mtime, data)

    data = data[data["Date"] <= curr_date_dt]
    if data.empty:
        return None
    try:
        _assert_ohlcv_not_stale(data, curr_date, symbol, canonical)
    except NoMarketDataError:
        return None
    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str | None) -> pd.DataFrame:
    """Drop financial-statement columns that were not yet public on ``curr_date``.

    yfinance financial statements use fiscal period end dates as columns. A
    column whose period ends on or before ``curr_date`` is NOT necessarily
    public knowledge on that date -- the report is filed days to weeks later
    (see :func:`yiagents.dataflows.utils.is_filing_public`). We keep a column
    only once its fiscal period end + filing lag is on/before ``curr_date``,
    so a backtest cannot read a report the market could not yet have seen.

    Non-date columns (NaT after coerce -- e.g. a "symbol"/"currency" annotation,
    or the trailing-"ttm" aggregate) are not fiscal periods and are kept as
    metadata; they survive the filter just as before.
    """
    if not curr_date or data.empty:
        return data
    parsed = pd.to_datetime(data.columns, errors="coerce")
    keep = []
    for col, col_ts in zip(data.columns, parsed, strict=True):
        if pd.isna(col_ts):
            keep.append(True)  # non-date metadata column
        else:
            keep.append(is_filing_public(str(col), curr_date))
    return data.loc[:, keep]
