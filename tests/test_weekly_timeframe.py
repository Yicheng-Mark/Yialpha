"""Weekly multi-timeframe: resample correctness, incomplete-week drop, PIT.

The daily->weekly resampler (ohlcv_resample) and the weekly indicator tool
(get_indicators_weekly) extend the analyst's evidence to the higher
timeframe. The load-bearing rules pinned here: correct OHLCV rollup, the
trailing still-open week is never quoted as a completed bar, and no weekly
bar may contain a daily row after the analysis date.
"""

from __future__ import annotations

import pandas as pd
import pytest

from yialpha.dataflows.ohlcv_resample import resample_weekly, weekly_ohlcv


def _daily_frame(dates, closes):
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": closes - 1.0,
        "High": closes + 2.0,
        "Low": closes - 2.0,
        "Close": closes,
        "Volume": [100.0] * len(closes),
    })


@pytest.mark.unit
class TestResampleWeekly:
    def test_basic_rollup_labels_friday(self):
        # Mon 2025-01-06 .. Fri 2025-01-10 = one full week.
        daily = _daily_frame(pd.date_range("2025-01-06", "2025-01-10"), [1, 2, 3, 4, 5])
        weekly = resample_weekly(daily, curr_date="2025-01-10")
        assert len(weekly) == 1
        row = weekly.iloc[0]
        assert row["Date"] == pd.Timestamp("2025-01-10")  # W-FRI label
        assert row["Open"] == 0.0   # first open
        assert row["High"] == 7.0   # max of closes+2
        assert row["Low"] == -1.0   # min of closes-2
        assert row["Close"] == 5.0  # last close
        assert row["Volume"] == 500.0

    def test_trailing_incomplete_week_is_dropped(self):
        # Full week Mon-Fri, then Mon-Tue of the NEXT week (its Friday label
        # 2025-01-17 is after curr_date 2025-01-14).
        dates = list(pd.date_range("2025-01-06", "2025-01-10")) + [
            pd.Timestamp("2025-01-13"), pd.Timestamp("2025-01-14"),
        ]
        daily = _daily_frame(dates, range(1, 8))
        weekly = resample_weekly(daily, curr_date="2025-01-14")
        assert len(weekly) == 1
        assert weekly.iloc[0]["Close"] == 5.0

    def test_complete_week_at_curr_date_is_kept(self):
        # curr_date exactly the Friday: that week is complete, keep it.
        daily = _daily_frame(pd.date_range("2025-01-06", "2025-01-10"), [1, 2, 3, 4, 5])
        weekly = resample_weekly(daily, curr_date="2025-01-10")
        assert len(weekly) == 1

    def test_no_curr_date_keeps_every_bin(self):
        dates = pd.date_range("2025-01-06", "2025-01-14")
        daily = _daily_frame(dates, range(1, len(dates) + 1))
        weekly = resample_weekly(daily)  # caller takes responsibility
        assert len(weekly) == 2

    def test_empty_input_returns_empty_frame(self):
        weekly = resample_weekly(pd.DataFrame(), curr_date="2025-01-10")
        assert weekly.empty
        assert "Close" in weekly.columns

    def test_nan_close_row_is_dropped_not_forward_filled(self):
        daily = _daily_frame(pd.date_range("2025-01-06", "2025-01-10"), [1, 2, 3, 4, 5])
        daily.loc[daily.index[2], "Close"] = float("nan")
        weekly = resample_weekly(daily, curr_date="2025-01-10")
        # The NaN row is dropped; the week's close is the last VALID close.
        assert weekly.iloc[0]["Close"] == 5.0


@pytest.mark.unit
class TestWeeklyPit:
    def test_weekly_never_contains_rows_after_curr_date(self, monkeypatch):
        import yialpha.dataflows.ohlcv_resample as mod

        # 8 full weeks Mon-Fri, curr_date = the Wednesday of week 7.
        all_days = pd.date_range("2025-01-06", periods=8 * 5, freq="B")
        daily = _daily_frame(all_days, range(1, len(all_days) + 1))
        monkeypatch.setattr(mod, "load_ohlcv", lambda s, d: daily[daily["Date"] <= pd.Timestamp(d)])

        curr = "2025-02-19"  # Wednesday inside week 7 (label Fri 2025-02-21)
        weekly = weekly_ohlcv("TEST", curr)
        assert (pd.to_datetime(weekly["Date"]) <= pd.Timestamp(curr)).all()
        # Week 7's Friday (2025-02-21) is after curr -> only 6 complete weeks.
        assert len(weekly) == 6


@pytest.mark.unit
class TestLiveFormingGuardRound11:
    """Round-11: the still-forming judgment asks whether the BIN's own
    interval extends to today (a W-FRI label L covers (L-6, L]; today > L
    means no new row can ever land in it), not whether the feed has printed
    today's bar — that bar belongs to the NEXT bin, which the cutoff filter
    already dropped. The round-8 criterion (``last_day >= today``) deleted
    the last COMPLETED Friday bin on every Tue-Thu live run (crypto always
    carries a today-dated bar; intraday equities too), so weekly pivots /
    weekly indicators read the week BEFORE last — the exact "Wednesday
    analysis returned week-before-last" symptom round-5 chased while its
    Saturday test path never ran on a real Saturday.

    All three scenarios pin the clock (pd.Timestamp.now) and the live gate
    (is_historical_date) so they are deterministic on any host date."""

    _WEEK_A = pd.date_range("2026-09-14", "2026-09-18")  # label Fri 09-18
    _WEEK_B = pd.date_range("2026-09-21", "2026-09-25")  # label Fri 09-25

    def _patch_live_clock(self, monkeypatch, today: str):
        import yialpha.dataflows.ohlcv_resample as mod

        monkeypatch.setattr(
            "yialpha.dataflows.utils.is_historical_date", lambda d: False,
        )
        monkeypatch.setattr(
            mod.pd.Timestamp, "now",
            staticmethod(lambda *a, **k: pd.Timestamp(today, tz="UTC")),
        )
        return mod

    def test_tuesday_live_keeps_completed_friday_bin(self, monkeypatch):
        """Crypto Tuesday live run (today's bar present in the feed): the
        last completed Friday bin must survive — the round-8 guard deleted
        it because ``last_day >= today`` held for the today-dated bar."""
        mod = self._patch_live_clock(monkeypatch, "2026-09-29")
        days = list(self._WEEK_A) + list(self._WEEK_B) + [
            pd.Timestamp("2026-09-28"), pd.Timestamp("2026-09-29"),
        ]
        weekly = mod.resample_weekly(
            _daily_frame(days, range(1, len(days) + 1)), curr_date="2026-09-29",
        )
        labels = [ts.date().isoformat() for ts in weekly["Date"]]
        assert labels == ["2026-09-18", "2026-09-25"]

    def test_friday_live_drops_same_day_bin(self, monkeypatch):
        """Live ON the bin's own Friday: the day's bar is still intraday —
        the trailing bin is forming and must drop (unchanged behavior)."""
        mod = self._patch_live_clock(monkeypatch, "2026-09-25")
        days = list(self._WEEK_A) + list(self._WEEK_B)
        weekly = mod.resample_weekly(
            _daily_frame(days, range(1, len(days) + 1)), curr_date="2026-09-25",
        )
        labels = [ts.date().isoformat() for ts in weekly["Date"]]
        assert labels == ["2026-09-18"]

    def test_saturday_live_keeps_completed_friday_bin(self, monkeypatch):
        """True-Saturday crypto run (Saturday row landed, UTC rolled): the
        Friday bin is complete and must survive. The round-8 guard deleted
        it (``last_day >= today`` for the Saturday row) — round-5's suite
        never exercised this path on a real Saturday."""
        mod = self._patch_live_clock(monkeypatch, "2026-09-26")
        days = (
            list(self._WEEK_A) + list(self._WEEK_B)
            + [pd.Timestamp("2026-09-26")]  # the rolled-UTC Saturday row
        )
        weekly = mod.resample_weekly(
            _daily_frame(days, range(1, len(days) + 1)), curr_date="2026-09-26",
        )
        labels = [ts.date().isoformat() for ts in weekly["Date"]]
        assert labels == ["2026-09-18", "2026-09-25"]

    def test_saturday_live_cache_missing_friday_bar_drops_truncated_bin(
        self, monkeypatch,
    ):
        """Round-11 remediation: the calendar clause alone cannot judge a
        bin whose label day is past but whose CACHE never received the
        label-day bar — a Saturday-early live run against a feed still
        ending Thursday serves a Mon–Thu bin wearing Friday's label
        (Thursday's close quoted as a weekly close). The restored
        ``last_label > last_daily`` clause drops it (round-5 doctrine: the
        label-day bar must be in the cache)."""
        mod = self._patch_live_clock(monkeypatch, "2026-09-26")
        truncated_week_b = [
            pd.Timestamp("2026-09-21"), pd.Timestamp("2026-09-22"),
            pd.Timestamp("2026-09-23"), pd.Timestamp("2026-09-24"),
        ]  # Mon–Thu only — Friday 09-25's bar never landed in the cache
        days = list(self._WEEK_A) + truncated_week_b
        weekly = mod.resample_weekly(
            _daily_frame(days, range(1, len(days) + 1)), curr_date="2026-09-26",
        )
        labels = [ts.date().isoformat() for ts in weekly["Date"]]
        assert labels == ["2026-09-18"]  # truncated Fri-labelled bin dropped

    def test_monday_preopen_live_friday_tail_bin_kept(self, monkeypatch):
        """The truncation clause must not over-drop: on a pre-open Monday
        equities live run there are no weekend rows, so Friday's bar IS the
        cache tail (``last_label == last_daily``) and the completed bin
        survives."""
        mod = self._patch_live_clock(monkeypatch, "2026-09-28")
        days = list(self._WEEK_A) + list(self._WEEK_B)  # tail = Fri 09-25
        weekly = mod.resample_weekly(
            _daily_frame(days, range(1, len(days) + 1)), curr_date="2026-09-28",
        )
        labels = [ts.date().isoformat() for ts in weekly["Date"]]
        assert labels == ["2026-09-18", "2026-09-25"]


@pytest.mark.unit
class TestWeeklyTool:
    def test_tool_renders_table_and_summary(self, monkeypatch):
        import yialpha.dataflows.ohlcv_resample as mod
        from yialpha.agents.utils.weekly_indicators_tools import get_indicators_weekly
        all_days = pd.date_range("2025-01-06", periods=60 * 5, freq="B")
        daily = _daily_frame(all_days, [100.0 + i * 0.5 for i in range(60 * 5)])
        monkeypatch.setattr(
            mod, "load_ohlcv",
            lambda s, d: daily[daily["Date"] <= pd.Timestamp(d)],
        )
        out = get_indicators_weekly.invoke({"symbol": "TEST", "curr_date": "2026-02-27"})
        assert "Weekly (" in out
        assert "close_10_sma" in out
        assert "weekly RSI (14)" in out
        assert "trailing incomplete week is excluded" in out
        # 300 business days -> ~60 weeks; warm-up (30w SMA, MACD) is done, so
        # the latest row carries finite values.
        assert "N/A" not in out.splitlines()[-4]

    def test_tool_degrades_typed_on_failure(self, monkeypatch):
        import yialpha.dataflows.ohlcv_resample as mod
        from yialpha.agents.utils.weekly_indicators_tools import get_indicators_weekly

        def boom(symbol, curr_date):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(mod, "load_ohlcv", boom)
        out = get_indicators_weekly.invoke({"symbol": "TEST", "curr_date": "2025-04-04"})
        assert out.startswith("DATA_UNAVAILABLE")
        assert "RuntimeError" in out
