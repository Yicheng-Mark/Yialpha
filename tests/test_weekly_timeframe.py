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

from yiagents.dataflows.ohlcv_resample import resample_weekly, weekly_ohlcv


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
        import yiagents.dataflows.ohlcv_resample as mod

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
class TestWeeklyTool:
    def test_tool_renders_table_and_summary(self, monkeypatch):
        import yiagents.dataflows.ohlcv_resample as mod
        from yiagents.agents.utils.weekly_indicators_tools import get_indicators_weekly
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
        import yiagents.dataflows.ohlcv_resample as mod
        from yiagents.agents.utils.weekly_indicators_tools import get_indicators_weekly

        def boom(symbol, curr_date):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(mod, "load_ohlcv", boom)
        out = get_indicators_weekly.invoke({"symbol": "TEST", "curr_date": "2025-04-04"})
        assert out.startswith("DATA_UNAVAILABLE")
        assert "RuntimeError" in out
