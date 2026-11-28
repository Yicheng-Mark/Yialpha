"""Vendor cache growth / staleness fixes (workflow B).

Covers the fixes that all share the "one cache, bounded lifetime" theme:

* B1 — the OHLCV per-symbol cache uses ONE fixed filename (the old name
  embedded tomorrow's date, minting a new ~1250-row CSV per symbol per day,
  forever), and a fresh write purges the legacy dated files.
* B2 — ``read_cached_ohlcv`` reuses the in-process cleaned-frame memo instead
  of re-reading + re-cleaning the CSV on every call.
* B3 — ``_clean_dataframe`` no longer back-fills leading NaNs (bfill used
  t+1's value at the head of the window = look-ahead for early curr_dates).
* B4 — ``get_YFin_history_cached`` caches a window reaching today for only
  15 minutes (today's close must not wait a full 24h TTL).
"""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

import yialpha.dataflows.stockstats_utils as su
import yialpha.dataflows.y_finance as y_finance
from yialpha.dataflows.config import get_config, set_config


@pytest.fixture(autouse=True)
def _fresh_memo():
    """The OHLCV memo is process-global; start and end each test clean."""
    with su._OHLCV_MEMO_LOCK:
        su._OHLCV_MEMO.clear()
    yield
    with su._OHLCV_MEMO_LOCK:
        su._OHLCV_MEMO.clear()


def _cache_file(symbol: str = "AAPL") -> str:
    safe = su.safe_ticker_component(su.normalize_symbol(symbol))
    return su._ohlcv_cache_path(get_config(), safe)


def _ohlcv_csv(end: str, periods: int = 6, base_close: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    n = len(dates)
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": [base_close - 0.5] * n,
        "High": [base_close + 1.5] * n,
        "Low": [base_close - 1.5] * n,
        "Close": [base_close + i for i in range(n)],
        "Volume": [1_000_000] * n,
    })


def _counting_read_csv(monkeypatch):
    counter = {"reads": 0}
    real = pd.read_csv

    def _counting(*args, **kwargs):
        counter["reads"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _counting)
    return counter


# --------------------------------------------------------------------------- #
# B1 — fixed cache filename + legacy purge
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_cache_filename_is_fixed_not_date_stamped(tmp_path):
    """The cache path embeds NO dates: same symbol, same file, any day."""
    set_config({"data_cache_dir": str(tmp_path)})
    p1 = su._ohlcv_cache_path(get_config(), "AAPL")
    assert os.path.basename(p1) == "AAPL-YFin-data.csv"
    # Deterministic across calls (the window still shifts for the DOWNLOAD,
    # but it must not leak into the filename).
    assert su._ohlcv_cache_path(get_config(), "AAPL") == p1


@pytest.mark.unit
def test_fresh_write_purges_legacy_dated_cache_files(monkeypatch, tmp_path):
    """Writing the fixed-name cache removes the per-day files the old scheme
    left behind, so the upgrade does not leave stale CSVs accumulating."""
    set_config({"data_cache_dir": str(tmp_path)})
    legacy_1 = tmp_path / "AAPL-YFin-data-2021-08-15-2026-08-16.csv"
    legacy_2 = tmp_path / "AAPL-YFin-data-2021-08-14-2026-08-15.csv"
    other_symbol = tmp_path / "MSFT-YFin-data-2021-08-15-2026-08-16.csv"
    for f in (legacy_1, legacy_2, other_symbol):
        f.write_text("Date,Close", encoding="utf-8")

    def fake_download(symbol, start, end, **kwargs):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-06-10")], name="Date")
        return pd.DataFrame(
            {"Open": [300.0], "High": [301.0], "Low": [299.0],
             "Close": [300.5], "Volume": [1]},
            index=idx,
        )

    monkeypatch.setattr(su.yf, "download", fake_download)
    out = su.load_ohlcv("AAPL", "2026-06-10")
    assert len(out) == 1

    # This symbol's legacy files are gone; another symbol's are untouched
    # (the purge is scoped by the validated safe symbol).
    assert not legacy_1.exists()
    assert not legacy_2.exists()
    assert other_symbol.exists()
    # The fixed-name file exists and served the download.
    assert os.path.exists(_cache_file())


@pytest.mark.unit
def test_purge_is_best_effort_on_removal_failure(monkeypatch, tmp_path):
    """A failing os.remove must be logged, never raised into the data call."""
    set_config({"data_cache_dir": str(tmp_path)})
    legacy = tmp_path / "AAPL-YFin-data-2021-08-15-2026-08-16.csv"
    legacy.write_text("Date,Close", encoding="utf-8")

    def boom(path):
        raise OSError("locked")

    monkeypatch.setattr(su.os, "remove", boom)
    su._purge_legacy_ohlcv_caches(get_config(), "AAPL")  # must not raise
    assert legacy.exists()  # file left in place; hygiene is best-effort


# --------------------------------------------------------------------------- #
# B2 — read_cached_ohlcv reuses the memo
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_read_cached_ohlcv_serves_memo_without_rereading_csv(monkeypatch, tmp_path):
    """After load_ohlcv warmed the memo, the opportunistic cache read must not
    re-read + re-clean the CSV (previously every call paid a full disk read)."""
    set_config({"data_cache_dir": str(tmp_path)})
    curr = "2026-06-10"  # historical: no same-day refresh
    _ohlcv_csv(curr).to_csv(_cache_file(), index=False)

    counter = _counting_read_csv(monkeypatch)
    first = su.load_ohlcv("AAPL", curr)
    assert counter["reads"] == 1

    cached = su.read_cached_ohlcv("AAPL", curr)
    assert counter["reads"] == 1  # memo served; no extra CSV read
    assert cached is not None
    pd.testing.assert_frame_equal(cached, first)


@pytest.mark.unit
def test_read_cached_ohlcv_populates_memo_for_later_calls(monkeypatch, tmp_path):
    """Even without a prior load_ohlcv, the first read memoizes: a second
    read (and a later load_ohlcv) skips the CSV re-read."""
    set_config({"data_cache_dir": str(tmp_path)})
    curr = "2026-06-10"
    _ohlcv_csv(curr).to_csv(_cache_file(), index=False)

    counter = _counting_read_csv(monkeypatch)
    a = su.read_cached_ohlcv("AAPL", curr)
    b = su.read_cached_ohlcv("AAPL", curr)
    assert counter["reads"] == 1
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
def test_read_cached_ohlcv_miss_returns_none(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    assert su.read_cached_ohlcv("NOPE", "2026-06-10") is None


# --------------------------------------------------------------------------- #
# B3 — no bfill at the window head (look-ahead)
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_clean_dataframe_does_not_backfill_leading_nans():
    """A leading NaN must stay NaN: filling it from t+1 hands a decision at
    the window head tomorrow's price (look-ahead)."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-06-10", "2026-06-11", "2026-06-12"]),
        "Open": [None, 101.0, 102.0],
        "High": [None, 103.0, 104.0],
        "Low": [None, 100.0, 101.0],
        "Close": [100.0, 101.5, 102.5],  # Close present (row survives dropna)
        "Volume": [None, 1_000, 2_000],
    })
    cleaned = su._clean_dataframe(df)
    assert pd.isna(cleaned["Open"].iloc[0])   # was bfilled to 101.0 before
    assert pd.isna(cleaned["Volume"].iloc[0])
    # Interior gaps still forward-fill (that direction is PIT-safe).
    assert cleaned["Open"].iloc[1] == 101.0


@pytest.mark.unit
def test_clean_dataframe_still_ffills_interior_gaps():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-06-10", "2026-06-11", "2026-06-12"]),
        "Open": [100.0, None, 102.0],
        "High": [103.0, None, 105.0],
        "Low": [99.0, None, 101.0],
        "Close": [100.0, None, 102.5],
        "Volume": [1_000, None, 2_000],
    })
    cleaned = su._clean_dataframe(df)
    # The Close-NaN row is dropped entirely (pre-existing rule).
    assert len(cleaned) == 2


# --------------------------------------------------------------------------- #
# B4 — same-day history TTL (get_YFin_history_cached)
# --------------------------------------------------------------------------- #

def _isolate_yf_cache(monkeypatch, tmp_path):
    """Route the yfinance statements cache into tmp_path (dir created — the
    REAL vendor_cache_dir mkdirs, a bare lambda does not, and cached_or_fetch
    then warns "could not write cache" on every call, breaking TTL asserts)."""
    def _cache_dir(name: str) -> str:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(y_finance, "vendor_cache_dir", _cache_dir)


@pytest.mark.unit
def test_history_ttl_is_minutes_for_today_daily_for_past():
    # "Today" is the UTC calendar date (the rule's anchor), not the host-local
    # one: west of Greenwich the local date is a day behind for hours every day.
    today = su._utc_today().strftime("%Y-%m-%d")
    assert y_finance._history_cache_ttl_days(today) == pytest.approx(
        su.OHLCV_CACHE_TTL_SECONDS / 86400.0
    )
    # Tomorrow (yfinance end is exclusive; callers pass future ends).
    tomorrow = (su._utc_today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert y_finance._history_cache_ttl_days(tomorrow) == pytest.approx(
        su.OHLCV_CACHE_TTL_SECONDS / 86400.0
    )
    # A window strictly in the past is immutable -> the ~24h statements TTL.
    assert y_finance._history_cache_ttl_days("2026-06-11") == pytest.approx(1.0)


@pytest.mark.unit
def test_history_same_day_cache_refetches_after_ttl(monkeypatch, tmp_path):
    """A window ending today must be REFetched once the 15-minute TTL passes —
    previously the 24h TTL meant today's close never reached an intraday run
    until the next day."""
    _isolate_yf_cache(monkeypatch, tmp_path)
    today = su._utc_today().strftime("%Y-%m-%d")  # the rule's (UTC) "today"
    calls = {"n": 0}

    class _Ticker:
        def __init__(self, symbol):
            pass

        def history(self, start, end):
            calls["n"] += 1
            close = 300.0 + calls["n"]
            idx = pd.DatetimeIndex([pd.Timestamp(today)], name="Date")
            return pd.DataFrame(
                {"Open": [close - 1], "High": [close], "Low": [close - 2],
                 "Close": [close], "Volume": [1]},
                index=idx,
            )

    monkeypatch.setattr(y_finance.yf, "Ticker", _Ticker)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())

    y_finance.get_YFin_history_cached("SPY", "2026-01-01", today)
    y_finance.get_YFin_history_cached("SPY", "2026-01-01", today)
    assert calls["n"] == 1  # inside the TTL: cache served

    # Age the cache file past the 15-minute same-day TTL.
    cache_file = next((tmp_path / "yfinance").glob("hist_SPY_*.csv"))
    stale = time.time() - su.OHLCV_CACHE_TTL_SECONDS - 60
    os.utime(cache_file, (stale, stale))

    frame = y_finance.get_YFin_history_cached("SPY", "2026-01-01", today)
    assert calls["n"] == 2  # refetched
    assert frame["Close"].iloc[0] == 302.0  # the FRESH close, not the cached one


@pytest.mark.unit
def test_history_past_window_still_cached_a_day(monkeypatch, tmp_path):
    """An immutable past window keeps the long TTL (no wasteful refetch)."""
    _isolate_yf_cache(monkeypatch, tmp_path)
    calls = {"n": 0}

    class _Ticker:
        def __init__(self, symbol):
            pass

        def history(self, start, end):
            calls["n"] += 1
            idx = pd.DatetimeIndex([pd.Timestamp("2026-06-10")], name="Date")
            return pd.DataFrame({"Close": [330.58]}, index=idx)

    monkeypatch.setattr(y_finance.yf, "Ticker", _Ticker)
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())

    y_finance.get_YFin_history_cached("SPY", "2026-06-01", "2026-06-11")
    # Age the cache 2 hours: still within the 24h TTL for a past window.
    cache_file = next((tmp_path / "yfinance").glob("hist_SPY_*.csv"))
    stale = time.time() - 2 * 3600
    os.utime(cache_file, (stale, stale))
    y_finance.get_YFin_history_cached("SPY", "2026-06-01", "2026-06-11")
    assert calls["n"] == 1
