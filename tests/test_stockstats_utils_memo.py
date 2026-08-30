"""In-process memoization of ``load_ohlcv``'s cleaned frame (P3-16).

A single-ticker run calls load_ohlcv 10+ times; the disk cache dedups the
network, but every call still re-reads and re-cleans the same 5y CSV. The
memo serves the date-INDEPENDENT cleaned frame, keyed by the cache file's
mtime. Pinned here:

* two consecutive calls read the CSV once and return equal frames;
* a rewritten cache file (mtime bump) invalidates the memo and re-reads;
* the same-day-refresh rule is still enforced on memo hits (an over-age
  same-day cache is refetched, not served from memory);
* an empty download is never memoized, so a retry re-fetches.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

import yialpha.dataflows.stockstats_utils as su
from yialpha.dataflows.config import get_config, set_config
from yialpha.dataflows.symbol_utils import NoMarketDataError


@pytest.fixture(autouse=True)
def _fresh_memo():
    """The memo is process-global; start and end each test clean."""
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


@pytest.mark.unit
def test_second_call_serves_memo_without_rereading_csv(monkeypatch, tmp_path):
    """Two consecutive calls for the same symbol read the CSV exactly once."""
    set_config({"data_cache_dir": str(tmp_path)})
    curr = "2026-06-10"  # historical date: no same-day refresh, rows not stale
    _ohlcv_csv(curr).to_csv(_cache_file(), index=False)

    counter = _counting_read_csv(monkeypatch)
    first = su.load_ohlcv("AAPL", curr)
    second = su.load_ohlcv("AAPL", curr)

    assert counter["reads"] == 1
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6


@pytest.mark.unit
def test_rewritten_cache_file_invalidates_memo(monkeypatch, tmp_path):
    """A rewritten cache file (mtime change) triggers a re-read."""
    set_config({"data_cache_dir": str(tmp_path)})
    curr = "2026-06-10"
    data_file = _cache_file()
    _ohlcv_csv(curr, base_close=100.0).to_csv(data_file, index=False)
    before = os.path.getmtime(data_file)

    counter = _counting_read_csv(monkeypatch)
    first = su.load_ohlcv("AAPL", curr)
    assert counter["reads"] == 1

    # Rewrite with different prices and force a distinct mtime (filesystem
    # timestamp granularity is not guaranteed to differ on its own).
    _ohlcv_csv(curr, base_close=200.0).to_csv(data_file, index=False)
    forced = before + 10
    os.utime(data_file, (forced, forced))

    second = su.load_ohlcv("AAPL", curr)
    assert counter["reads"] == 2  # memo invalidated -> CSV re-read
    assert second["Close"].iloc[-1] == first["Close"].iloc[-1] + 100.0


@pytest.mark.unit
def test_same_day_refresh_still_refetches_on_memo_hit(monkeypatch, tmp_path):
    """An over-age same-day cache is refetched even when the memo is warm.

    The disk-cache TTL rule (#1150) must survive memoization: a run started
    before today's bar was final must pick it up once the 15-minute window
    has passed, instead of serving the cached frame forever.
    """
    set_config({"data_cache_dir": str(tmp_path)})
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    data_file = _cache_file()
    _ohlcv_csv(today, periods=3, base_close=100.0).to_csv(data_file, index=False)

    counter = _counting_read_csv(monkeypatch)
    first = su.load_ohlcv("AAPL", today)
    assert counter["reads"] == 1  # fresh same-day cache served from CSV
    assert first["Close"].iloc[-1] == 102.0

    # Age the file past OHLCV_CACHE_TTL_SECONDS so the same-day rule fires.
    stale = os.path.getmtime(data_file) - su.OHLCV_CACHE_TTL_SECONDS - 60
    os.utime(data_file, (stale, stale))

    def fake_download(symbol, start, end, **kwargs):
        idx = pd.DatetimeIndex([pd.Timestamp.today().normalize()], name="Date")
        return pd.DataFrame(
            {"Open": [300.0], "High": [301.0], "Low": [299.0],
             "Close": [300.5], "Volume": [1]},
            index=idx,
        )

    monkeypatch.setattr(su.yf, "download", fake_download)
    out = su.load_ohlcv("AAPL", today)

    # The over-age same-day cache was NOT served (neither memo nor CSV):
    # the refresh path downloaded today's fresh row instead.
    assert counter["reads"] == 1
    assert out["Close"].iloc[-1] == 300.5


@pytest.mark.unit
def test_empty_download_not_memoized(monkeypatch, tmp_path):
    """A failed fetch memoizes nothing, so the retry re-fetches (#1150 guard)."""
    set_config({"data_cache_dir": str(tmp_path)})
    calls = {"n": 0}

    def empty_download(symbol, start, end, **kwargs):
        calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(su.yf, "download", empty_download)
    with pytest.raises(NoMarketDataError):
        su.load_ohlcv("FAKE", "2026-01-01")
    with pytest.raises(NoMarketDataError):
        su.load_ohlcv("FAKE", "2026-01-01")
    assert calls["n"] == 2
    with su._OHLCV_MEMO_LOCK:
        assert su._OHLCV_MEMO == {}
