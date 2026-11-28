"""Round-15 coverage: edge branches of the indicator-side dataflow modules.

Fresh assertions for the branches the round-14 report still lists as missed:

* ``ohlcv_resample`` -- the all-invalid-rows early return, the tz-aware
  live-run guard branches, and the fail-visible unparseable-``curr_date``
  refusal (a PIT boundary, not a crash path).
* ``stockstats_utils`` -- the ``YIALPHA_HTTP_TIMEOUT_S`` parse fallbacks,
  the yfinance rate-limit retry/exhaustion conversion, the date-coercion
  fallbacks feeding the stale guard, memo FIFO eviction, and every
  degrade-to-``None`` leg of ``read_cached_ohlcv``.
* ``candlestick_patterns`` -- pattern detectors on edge rows (flat bars,
  short trios, trend-context and flat-trend naming, the two-bar reversal
  family, the evening-star mirror, and the pivot-shape guard clauses).
* ``valuation_methods`` -- the non-finite / non-numeric input guards that
  keep a confabulated number out of the deterministic engine.

All frames are built inline; no network, no LLM, no sleeps (retry backoff
is stubbed), and the module reload for env parsing restores prior state.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yialpha.dataflows.candlestick_patterns as cpat
import yialpha.dataflows.stockstats_utils as su
from yialpha.dataflows.config import get_config, set_config
from yialpha.dataflows.ohlcv_resample import resample_weekly
from yialpha.dataflows.symbol_utils import NoMarketDataError
from yialpha.dataflows.valuation_methods import (
    earnings_yield,
    graham_number,
    margin_of_safety,
    net_current_asset_value_per_share,
    pe_ratio,
    peg_ratio,
    summarize,
    weighted_average_cost_of_capital,
)


@pytest.fixture(autouse=True)
def _r15_fresh_memo():
    """The OHLCV memo is process-global; start and end each test clean."""
    with su._OHLCV_MEMO_LOCK:
        su._OHLCV_MEMO.clear()
    yield
    with su._OHLCV_MEMO_LOCK:
        su._OHLCV_MEMO.clear()


def _r15_bars(rows, start="2025-01-01"):
    """OHLCV frame from (open, high, low, close, volume) tuples."""
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    frame.insert(0, "Date", pd.bdate_range(start, periods=len(frame)))
    return frame


def _r15_daily_frame(dates, closes):
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "Date": pd.to_datetime(list(dates)),
        "Open": closes - 1.0,
        "High": closes + 2.0,
        "Low": closes - 2.0,
        "Close": closes,
        "Volume": [100.0] * len(closes),
    })


def _r15_cache_file(symbol: str = "AAPL") -> str:
    safe = su.safe_ticker_component(su.normalize_symbol(symbol))
    return su._ohlcv_cache_path(get_config(), safe)


def _r15_ohlcv_csv(end: str, periods: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=periods)
    n = len(dates)
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": [99.5] * n,
        "High": [101.5] * n,
        "Low": [98.5] * n,
        "Close": [100.0 + i for i in range(n)],
        "Volume": [1_000_000] * n,
    })


# --------------------------------------------------------------------------- #
# ohlcv_resample: edge rows + tz-aware live guard + fail-visible PIT refusal
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15ResampleEdgeBranches:
    def test_r15_resample_all_rows_invalid_returns_empty_columned_frame(self):
        """A non-empty frame whose dates all fail to parse must return the
        empty columned frame (the honest "no weekly bars" answer), not raise
        and not fabricate a bin."""
        daily = pd.DataFrame({
            "Date": ["not-a-date", "also bad"],
            "Open": [1.0, 1.0], "High": [2.0, 2.0], "Low": [0.5, 0.5],
            "Close": [1.5, 1.5], "Volume": [10.0, 10.0],
        })
        # pandas finds no common format in the junk dates and says so before
        # coercing them one by one (ohlcv_resample.py:51 passes no format=).
        with pytest.warns(UserWarning, match="Could not infer format"):
            weekly = resample_weekly(daily, curr_date="2025-01-10")
        assert weekly.empty
        assert list(weekly.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]

    def test_r15_resample_tz_aware_live_friday_drops_forming_bin(self, monkeypatch):
        """tz-aware daily dates must flow through the live guard: on a live
        run dated the bin's own Friday the trailing bin is still forming and
        must drop even when every timestamp carries an explicit UTC offset."""
        monkeypatch.setattr(
            "yialpha.dataflows.utils.is_historical_date", lambda d: False,
        )
        monkeypatch.setattr(
            pd.Timestamp, "now",
            staticmethod(lambda *a, **k: pd.Timestamp("2026-09-25", tz="UTC")),
        )
        dates = pd.date_range("2026-09-14", "2026-09-25", tz="UTC")
        daily = _r15_daily_frame(dates, range(1, len(dates) + 1))
        weekly = resample_weekly(daily, curr_date="2026-09-25 00:00:00+00:00")
        labels = [ts.date().isoformat() for ts in weekly["Date"]]
        assert labels == ["2026-09-18"]  # the same-day Friday bin dropped

    def test_r15_resample_tz_aware_completed_bin_survives_later_live_day(
        self, monkeypatch,
    ):
        """Mirror boundary: on a Monday live run the trailing Friday-labelled
        tz-aware bin is complete (label day is the cache tail) and survives
        both live-guard clauses."""
        monkeypatch.setattr(
            "yialpha.dataflows.utils.is_historical_date", lambda d: False,
        )
        monkeypatch.setattr(
            pd.Timestamp, "now",
            staticmethod(lambda *a, **k: pd.Timestamp("2026-09-28", tz="UTC")),
        )
        dates = pd.date_range("2026-09-14", "2026-09-25", tz="UTC")
        daily = _r15_daily_frame(dates, range(1, len(dates) + 1))
        weekly = resample_weekly(daily, curr_date="2026-09-28 00:00:00+00:00")
        labels = [ts.date().isoformat() for ts in weekly["Date"]]
        assert labels == ["2026-09-18", "2026-09-25"]

    def test_r15_resample_unparseable_curr_date_refuses_with_value_error(self):
        """Fail-visible PIT contract: an unparseable curr_date must refuse to
        resample rather than silently keep the trailing incomplete week."""
        daily = _r15_daily_frame(pd.date_range("2025-01-06", "2025-01-10"), [1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="PIT cutoff"):
            resample_weekly(daily, curr_date="not-a-date")


# --------------------------------------------------------------------------- #
# stockstats_utils: YIALPHA_HTTP_TIMEOUT_S parse fallbacks (module reload)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15HttpTimeoutEnvParsing:
    @pytest.fixture(autouse=True)
    def _undo_stockstats_reload(self):
        """Put ``stockstats_utils`` back as it was after each test.

        ``reload`` re-executes the module into its existing namespace; the
        tests below do it to re-read ``YIALPHA_HTTP_TIMEOUT_S``. The module's
        timeout is re-derived at the end of each test, its functions are not:
        without this the later tests in the session get new function objects.
        """
        saved = dict(su.__dict__)
        yield
        su.__dict__.clear()
        su.__dict__.update(saved)

    def _reload_with(self, monkeypatch, value: str | None):
        if value is None:
            monkeypatch.delenv("YIALPHA_HTTP_TIMEOUT_S", raising=False)
        else:
            monkeypatch.setenv("YIALPHA_HTTP_TIMEOUT_S", value)
        return importlib.reload(su)

    def test_r15_env_timeout_non_positive_falls_back_with_warning(self, caplog, monkeypatch):
        with caplog.at_level(logging.WARNING, logger=su.__name__):
            module = self._reload_with(monkeypatch, "-5")
        assert module.YF_HTTP_TIMEOUT == 30.0
        assert "not positive" in caplog.text
        assert "YIALPHA_HTTP_TIMEOUT_S" in caplog.text

    def test_r15_env_timeout_non_numeric_falls_back_with_warning(self, caplog, monkeypatch):
        with caplog.at_level(logging.WARNING, logger=su.__name__):
            module = self._reload_with(monkeypatch, "soon")
        assert module.YF_HTTP_TIMEOUT == 30.0
        assert "not a number" in caplog.text

    def test_r15_env_timeout_zero_opts_out_positive_sets(self, monkeypatch):
        assert self._reload_with(monkeypatch, "0").YF_HTTP_TIMEOUT is None
        assert self._reload_with(monkeypatch, "12.5").YF_HTTP_TIMEOUT == 12.5


# --------------------------------------------------------------------------- #
# stockstats_utils: yf_retry rate-limit retry / exhaustion conversion
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15YfRetryRateLimit:
    def test_r15_rate_limit_backs_off_then_succeeds(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(su.time, "sleep", sleeps.append)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise su.YFRateLimitError()
            return "DATA"

        assert su.yf_retry(flaky, max_retries=3, base_delay=1.0) == "DATA"
        assert calls["n"] == 2
        assert sleeps == [1.0]  # one exponential backoff, then success

    def test_r15_rate_limit_exhaustion_raises_no_market_data(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(su.time, "sleep", sleeps.append)
        attempts = {"n": 0}

        def always_limited():
            attempts["n"] += 1
            raise su.YFRateLimitError()

        with pytest.raises(NoMarketDataError) as ctx:
            su.yf_retry(
                always_limited, max_retries=2, base_delay=1.0,
                symbol="TST", canonical="TST",
            )
        # max_retries + 1 attempts, max_retries backoffs (2.0s, 4.0s...).
        assert attempts["n"] == 3
        assert sleeps == [1.0, 2.0]
        assert ctx.value.symbol == "TST"
        assert "rate-limited" in ctx.value.detail


# --------------------------------------------------------------------------- #
# stockstats_utils: date-coercion fallbacks feeding the stale guard
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15StaleGuardInputs:
    def test_r15_stale_guard_ignores_unparseable_curr_date(self):
        """curr_date garbage: the guard cannot reason about freshness and
        must not raise (the resampler owns the fail-visible refusal)."""
        frame = pd.DataFrame({"Date": [pd.Timestamp("2026-06-10")], "Close": [1.0]})
        su._assert_ohlcv_not_stale(frame, "not-a-date", "X")  # no raise

    def test_r15_stale_guard_ignores_frame_without_any_dates(self):
        frame = pd.DataFrame({"Close": [1.0, 2.0]}, index=["x", "y"])
        # "x" / "y" are not dates: pandas finds no common format and says so
        # before coercing them to NaT (stockstats_utils.py:213). Expected.
        with pytest.warns(UserWarning, match="Could not infer format"):
            assert su._coerce_ohlcv_dates(frame).empty
            su._assert_ohlcv_not_stale(frame, "2026-06-11", "X")  # no raise

    def test_r15_stale_guard_reads_datetime_column_fallback(self):
        """A frame whose date column is named ``Datetime`` (yfinance intraday
        layout) still feeds the guard: staleness must fire through it."""
        frame = pd.DataFrame({
            "Datetime": [pd.Timestamp("2025-06-11")],
            "Close": [330.58],
        })
        parsed = su._coerce_ohlcv_dates(frame)
        assert list(parsed) == [pd.Timestamp("2025-06-11")]
        with pytest.raises(NoMarketDataError) as ctx:
            su._assert_ohlcv_not_stale(frame, "2026-06-11", "CB", "CB")
        assert "2025-06-11" in str(ctx.value)
        assert "stale" in str(ctx.value)


# --------------------------------------------------------------------------- #
# stockstats_utils: memo FIFO eviction at capacity
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15OhlcvMemoEviction:
    def test_r15_memo_evicts_oldest_entry_at_capacity(self):
        frame = pd.DataFrame({"Close": [1.0]})
        first = "/r15-memo/entry-000.csv"
        su._ohlcv_memo_set(first, 1.0, frame)
        for i in range(1, su._OHLCV_MEMO_MAX_ENTRIES):
            su._ohlcv_memo_set(f"/r15-memo/entry-{i:03d}.csv", 1.0, frame)
        assert su._ohlcv_memo_get(first, 1.0) is not None  # at cap, resident

        su._ohlcv_memo_set("/r15-memo/new-key.csv", 1.0, frame)  # over cap
        assert su._ohlcv_memo_get(first, 1.0) is None  # FIFO evicted
        assert su._ohlcv_memo_get("/r15-memo/entry-001.csv", 1.0) is not None
        assert su._ohlcv_memo_get("/r15-memo/new-key.csv", 1.0) is not None

    def test_r15_memo_refresh_of_existing_key_does_not_evict(self):
        frame = pd.DataFrame({"Close": [1.0]})
        for i in range(su._OHLCV_MEMO_MAX_ENTRIES):
            su._ohlcv_memo_set(f"/r15-memo/hold-{i:03d}.csv", 1.0, frame)
        # Re-setting a PRESENT key at cap is a refresh, not a new entry.
        su._ohlcv_memo_set("/r15-memo/hold-000.csv", 2.0, frame)
        assert su._ohlcv_memo_get("/r15-memo/hold-000.csv", 2.0) is not None
        assert su._ohlcv_memo_get("/r15-memo/hold-001.csv", 1.0) is not None


# --------------------------------------------------------------------------- #
# stockstats_utils: read_cached_ohlcv degrade-to-None legs
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15ReadCachedOhlcvDegradation:
    def test_r15_cached_over_age_same_day_file_returns_none(self, tmp_path):
        set_config({"data_cache_dir": str(tmp_path)})
        today = su._utc_today().strftime("%Y-%m-%d")
        path = _r15_cache_file()
        _r15_ohlcv_csv(today).to_csv(path, index=False)
        aged = time.time() - su.OHLCV_CACHE_TTL_SECONDS - 60
        os.utime(path, (aged, aged))
        # The 15-minute same-day TTL fired: refuse the file, do not download.
        assert su.read_cached_ohlcv("AAPL", today) is None

    def test_r15_cached_unreadable_csv_returns_none(self, tmp_path):
        set_config({"data_cache_dir": str(tmp_path)})
        path = Path(_r15_cache_file())
        # Unterminated quoted field: a torn write, raises ParserError
        # (a ValueError subclass) even with on_bad_lines="skip".
        path.write_text(
            'Date,Open,High,Low,Close,Volume\n"2026-06-10,100,101,99,100.5,10\n',
            encoding="utf-8",
        )
        assert su.read_cached_ohlcv("AAPL", "2026-06-10") is None

    def test_r15_cached_header_only_file_returns_none(self, tmp_path):
        set_config({"data_cache_dir": str(tmp_path)})
        path = Path(_r15_cache_file())
        path.write_text("Date,Open,High,Low,Close,Volume\n", encoding="utf-8")
        assert su.read_cached_ohlcv("AAPL", "2026-06-10") is None

    def test_r15_cached_post_pit_filter_empty_returns_none(self, tmp_path):
        set_config({"data_cache_dir": str(tmp_path)})
        _r15_ohlcv_csv("2026-06-10").to_csv(_r15_cache_file(), index=False)
        # Analysis date before the frame's first row: nothing survives the
        # PIT filter, so the opportunistic read reports a miss.
        assert su.read_cached_ohlcv("AAPL", "2026-01-02") is None

    def test_r15_cached_stale_frame_returns_none(self, tmp_path):
        set_config({"data_cache_dir": str(tmp_path)})
        _r15_ohlcv_csv("2025-06-11").to_csv(_r15_cache_file(), index=False)
        # Rows survive the PIT filter but the latest is ~a year old: the
        # stale guard's NoMarketDataError degrades to a cache miss here.
        assert su.read_cached_ohlcv("AAPL", "2026-06-11") is None


# --------------------------------------------------------------------------- #
# candlestick_patterns: edge rows and guards
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15ScanEdgeRows:
    def test_r15_scan_returns_empty_for_none_and_short_frames(self):
        assert cpat.scan_candlestick_patterns(None) == []
        assert cpat.scan_candlestick_patterns(_r15_bars([(100, 100.9, 99.4, 100.2, 1)] * 2)) == []

    def test_r15_scan_skips_flat_zero_range_bar(self):
        """A perfectly flat bar (O=H=L=C) has no range: it must be skipped
        entirely rather than scored as a full-body bar (body/range floors to
        1.0 on the 1e-12 range guard)."""
        rows = [
            (100.0, 100.9, 99.4, 100.2, 1),
            (100.0, 100.0, 100.0, 100.0, 1),  # flat bar
            (100.0, 100.9, 99.4, 100.2, 1),
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(rows))
        assert hits == []
        assert not any(h["date"] == "2025-01-02" for h in hits)

    def test_r15_soldiers_and_crows_reject_short_trios(self):
        two = _r15_bars([(100.0, 101.0, 99.0, 100.5, 1)] * 2)
        assert cpat._detect_three_white_soldiers(two) is False
        assert cpat._detect_three_black_crows(two) is False


# --------------------------------------------------------------------------- #
# candlestick_patterns: trend-context naming on the edge bars
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15TrendContextPatterns:
    def test_r15_hanging_man_after_advance(self):
        rows = [
            (100.0, 100.9, 99.6, 100.8, 1),  # advancing lead-in
            (100.8, 102.0, 100.6, 101.8, 1),  # advancing lead-in
            (102.0, 102.5, 96.0, 102.4, 1),   # long lower shadow, small body
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(rows))
        hang = next(h for h in hits if h["pattern"] == "hanging_man")
        assert hang["direction"] == "bearish"
        assert hang["date"] == "2025-01-03"

    def test_r15_inverted_hammer_after_decline(self):
        rows = [
            (104.0, 104.4, 103.4, 103.6, 1),  # declining lead-in
            (103.6, 103.9, 102.4, 102.6, 1),  # declining lead-in
            (102.2, 108.0, 102.1, 102.6, 1),  # long upper shadow, small body
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(rows))
        inv = next(h for h in hits if h["pattern"] == "inverted_hammer")
        assert inv["direction"] == "bullish"
        assert inv["date"] == "2025-01-03"

    def test_r15_star_family_flat_trend_falls_back_to_color(self):
        """With a flat lead-in the trend is unclassifiable, so the color
        heuristic names the bar: green upper-shadow body -> inverted hammer
        (bullish), red -> shooting star (bearish)."""
        green_rows = [
            (100.0, 100.5, 99.5, 100.0, 1),
            (100.0, 100.5, 99.5, 100.0, 1),
            (100.0, 106.0, 99.9, 100.4, 1),  # green: close above open
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(green_rows))
        star = next(h for h in hits if h["pattern"] in ("shooting_star", "inverted_hammer"))
        assert star["pattern"] == "inverted_hammer"
        assert star["direction"] == "bullish"

        red_rows = [
            (100.0, 100.5, 99.5, 100.0, 1),
            (100.0, 100.5, 99.5, 100.0, 1),
            (100.4, 106.0, 99.9, 100.0, 1),  # red: close below open
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(red_rows))
        star = next(h for h in hits if h["pattern"] in ("shooting_star", "inverted_hammer"))
        assert star["pattern"] == "shooting_star"
        assert star["direction"] == "bearish"


@pytest.mark.unit
class TestR15TwoBarPatterns:
    def test_r15_bearish_engulfing(self):
        rows = [
            (99.5, 100.0, 99.0, 99.6, 1),    # bland lead-in
            (99.0, 102.2, 98.8, 101.5, 1),   # decent green body
            (102.0, 102.5, 98.3, 98.5, 1),   # bigger red body swallows it
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(rows))
        engulf = next(h for h in hits if h["pattern"] == "bearish_engulfing")
        assert engulf["direction"] == "bearish"
        assert engulf["date"] == "2025-01-03"

    def test_r15_harami(self):
        rows = [
            (99.5, 99.9, 99.1, 99.4, 1),     # bland lead-in
            (99.0, 102.2, 98.8, 102.0, 1),   # long body
            (100.0, 101.4, 98.9, 101.0, 1),  # small body inside it
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(rows))
        harami = next(h for h in hits if h["pattern"] == "harami")
        assert harami["direction"] == "neutral"
        assert harami["date"] == "2025-01-03"

    def test_r15_piercing(self):
        rows = [
            (99.0, 99.8, 98.9, 99.6, 1),     # bland lead-in
            (102.0, 102.3, 98.8, 99.0, 1),   # red bar, body spans 99..102
            (98.5, 101.3, 98.3, 101.0, 1),   # opens below, closes past midpoint
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(rows))
        pierce = next(h for h in hits if h["pattern"] == "piercing")
        assert pierce["direction"] == "bullish"
        assert pierce["date"] == "2025-01-03"

    def test_r15_dark_cloud_cover(self):
        rows = [
            (99.5, 99.9, 99.1, 99.4, 1),     # bland lead-in
            (99.0, 102.2, 98.8, 102.0, 1),   # green bar, body spans 99..102
            (102.5, 102.7, 99.7, 100.0, 1),  # opens above, closes under midpoint
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(rows))
        dark = next(h for h in hits if h["pattern"] == "dark_cloud_cover")
        assert dark["direction"] == "bearish"
        assert dark["date"] == "2025-01-03"


@pytest.mark.unit
class TestR15ThreeBarPatterns:
    def test_r15_evening_star(self):
        rows = [
            (99.0, 99.6, 98.8, 99.2, 1),     # bland lead-in
            (99.5, 103.2, 99.3, 103.0, 1),   # long green
            (103.3, 103.8, 103.1, 103.5, 1),  # small star body
            (103.4, 103.6, 99.8, 100.0, 1),  # long red below the midpoint
        ]
        hits = cpat.scan_candlestick_patterns(_r15_bars(rows))
        star = next(h for h in hits if h["pattern"] == "evening_star")
        assert star["direction"] == "bearish"
        assert star["date"] == "2025-01-06"  # attaches to the LAST bar


@pytest.mark.unit
class TestR15PivotShapes:
    def test_r15_find_pivots_too_short_returns_empty(self):
        assert cpat.find_pivots(_r15_bars([(100, 100.9, 99.4, 100.2, 1)] * 6)) == ([], [])
        assert cpat.find_pivots(None) == ([], [])

    def test_r15_double_extreme_too_short_returns_empty(self):
        assert cpat.detect_double_top_bottom(
            _r15_bars([(100, 100.9, 99.4, 100.2, 1)] * 10)
        ) == []
        assert cpat.detect_double_top_bottom(None) == []

    def test_r15_peaks_beyond_tolerance_are_not_a_double_top(self):
        """Two swing highs 8/114 (~7%) apart exceed the 2% equality
        tolerance: no double-top may be reported for them."""
        rows = [(100, 101, 99, 100, 1)] * 40
        closes = list(np.linspace(100, 110, 12))
        closes += list(np.linspace(109.5, 104, 8))
        closes += list(np.linspace(104.2, 118, 10))
        closes += list(np.linspace(117.5, 115.0, 4))
        for c in closes:
            rows.append((c - 0.5, c + 0.5, c - 0.7, c, 1))
        shapes = cpat.detect_double_top_bottom(_r15_bars(rows))
        assert [s for s in shapes if s["pattern"] == "double_top"] == []

    def test_r15_adjacent_pivots_without_intervening_bars_skipped(
        self, monkeypatch,
    ):
        """Two pivots with no bar strictly between them have no neckline to
        break, so the pair is skipped rather than fabricating a level."""
        frame = _r15_bars([(100, 101, 99, 100, 1)] * 20)
        monkeypatch.setattr(cpat, "find_pivots", lambda df, wing=3: ([5, 6], []))
        assert cpat.detect_double_top_bottom(frame) == []

    def _r15_bottom_bars(self, confirmed: bool):
        rows = [(100, 101, 99, 100, 1)] * 40
        closes = list(np.linspace(100, 90, 12))
        closes += list(np.linspace(90.5, 96, 8))
        closes += list(np.linspace(95.8, 90.2, 10))
        closes += list(np.linspace(90.5, 93.0, 4))
        for c in closes:
            rows.append((c - 0.5, c + 0.5, c - 0.7, c, 1))
        tail = (96.5, 97.2, 96.0, 97.0, 1) if confirmed else (93.5, 94.2, 93.0, 94.0, 1)
        rows.append(tail)
        return _r15_bars(rows)

    def test_r15_double_bottom_confirmed_on_neckline_break(self):
        shapes = cpat.detect_double_top_bottom(self._r15_bottom_bars(True))
        bottoms = [s for s in shapes if s["pattern"] == "double_bottom"]
        assert bottoms, "expected a double-bottom shape"
        shape = bottoms[0]
        assert shape["confirmed"] is True
        assert shape["direction"] == "bullish"
        assert shape["neckline"] > shape["level"]  # bounce high above the lows

    def test_r15_double_bottom_unconfirmed_without_break(self):
        shapes = cpat.detect_double_top_bottom(self._r15_bottom_bars(False))
        bottoms = [s for s in shapes if s["pattern"] == "double_bottom"]
        assert bottoms, "expected a double-bottom shape"
        assert bottoms[0]["confirmed"] is False


# --------------------------------------------------------------------------- #
# valuation_methods: non-finite / non-numeric input guards
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestR15ValuationGuards:
    def test_r15_non_finite_inputs_return_none_never_a_number(self):
        nan = float("nan")
        assert pe_ratio(None, 5.0) is None
        assert pe_ratio(nan, 5.0) is None
        assert earnings_yield(None, 5.0) is None
        assert earnings_yield(100.0, nan) is None
        assert graham_number(nan, 30.0) is None
        assert graham_number(2.0, float("inf")) is None
        assert net_current_asset_value_per_share(None, 300.0, 10.0) is None
        assert net_current_asset_value_per_share(500.0, nan, 10.0) is None
        assert peg_ratio(None, 15.0) is None
        assert peg_ratio(20.0, nan) is None
        assert weighted_average_cost_of_capital(None, 400.0, 0.10, 0.05) is None
        assert weighted_average_cost_of_capital(600.0, float("inf"), 0.10, 0.05) is None

    def test_r15_non_numeric_values_rejected_by_finite_guard(self):
        """A string or arbitrary object that float() cannot parse must fall
        through the try/except in the finiteness guard, not crash the math."""
        assert pe_ratio("not-a-number", 5.0) is None
        assert pe_ratio(object(), 5.0) is None
        assert graham_number("oops", 30.0) is None
        assert weighted_average_cost_of_capital("bad", 400.0, 0.10, 0.05) is None

    def test_r15_summarize_keeps_guards_when_inputs_missing_or_bad(self):
        out = summarize(price=100.0, eps=float("nan"))
        assert out["pe_ratio"] is None
        assert out["earnings_yield"] is None
        # A valid DCF still threads through to the margin of safety.
        out = summarize(
            price=70.0,
            free_cash_flow_per_share=10.0,
            growth_rate=0.0,
            discount_rate=0.10,
            terminal_growth_rate=0.0,
        )
        assert out["intrinsic_value_dcf"] == pytest.approx(100.0, abs=0.01)
        assert margin_of_safety(out["intrinsic_value_dcf"], 70.0) == pytest.approx(0.30)
