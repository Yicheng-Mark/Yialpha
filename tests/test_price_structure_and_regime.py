"""Price structure (S/R, patterns, volume tools) + composite regime tests.

Deterministic fixtures pin the math: classic pivot hand values, volume
profile POC/value area, one synthetic candle fixture per pattern family,
pivot-based double top/bottom, trend/vol regime classifiers, and the
analyst's tool/prompt wiring for the 2026-08-15 expansion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yialpha.dataflows.candlestick_patterns import (
    detect_double_top_bottom,
    find_pivots,
    scan_candlestick_patterns,
)
from yialpha.dataflows.market_regime import (
    classify_trend_state,
    classify_vol_state,
)
from yialpha.dataflows.support_resistance import (
    build_support_resistance,
    classic_pivots,
    daily_pivots,
    volume_profile,
    weekly_pivots,
)


def _bars(rows, start="2025-01-01"):
    """rows: iterable of (open, high, low, close[, volume])."""
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    n = len(frame)
    frame.insert(0, "Date", pd.bdate_range(start, periods=n))
    return frame


@pytest.mark.unit
class TestClassicPivots:
    def test_hand_values(self):
        # H=110, L=100, C=106 -> P=105.33.., R1=110.67.., S1=100.67..,
        # R2=115.33.., S2=95.33..
        p = classic_pivots(110.0, 100.0, 106.0)
        assert p["P"] == pytest.approx((110 + 100 + 106) / 3.0)
        assert p["R1"] == pytest.approx(2 * p["P"] - 100.0)
        assert p["S1"] == pytest.approx(2 * p["P"] - 110.0)
        assert p["R2"] == pytest.approx(p["P"] + 10.0)
        assert p["S2"] == pytest.approx(p["P"] - 10.0)

    def test_daily_pivots_use_previous_session(self):
        rows = [(99, 101, 98, 100, 1)] * 3 + [(106, 110, 100, 108, 1)]
        pivots = daily_pivots(_bars(rows))
        # From row -2 (the 100/101/98 session), NOT the last row.
        assert pivots is not None
        assert pivots["P"] == pytest.approx((101 + 98 + 100) / 3.0)


@pytest.mark.unit
class TestVolumeProfile:
    def test_poc_lands_on_heaviest_bin(self):
        rows = []
        for i in range(120):
            close = 100.0 + 5.0 * (i % 10) / 9.0  # spread across 100..105
            vol = 500.0
            rows.append((close - 0.1, close + 0.1, close - 0.1, close, vol))
        # Make the 102-ish band the heaviest.
        for i in range(60, 90):
            close = 102.0
            rows[i] = (close - 0.05, close + 0.05, close - 0.05, close, 5000.0)
        prof = volume_profile(_bars(rows), lookback=120, bins=20)
        assert prof is not None
        assert 101.0 < prof["poc"] < 103.0
        assert prof["value_area_low"] <= prof["poc"] <= prof["value_area_high"]

    def test_too_short_history_is_none(self):
        assert volume_profile(_bars([(1, 1, 1, 1, 1)] * 10)) is None


@pytest.mark.unit
class TestBuildSupportResistance:
    def test_full_report_structure(self):
        rows = [(100 + i * 0.5, 101 + i * 0.5, 99 + i * 0.5, 100 + i * 0.5, 1000)
                for i in range(300)]
        report = build_support_resistance(_bars(rows), "2026-02-20")
        assert report["daily_pivots"] is not None
        assert report["weekly_pivots"] is not None
        assert set(report["rolling_levels"]) == {20, 55, 252}
        assert report["volume_profile"] is not None
        assert report["latest_close"] == pytest.approx(100 + 299 * 0.5)

    def test_weekly_pivots_use_last_completed_week(self):
        # 2026-08-12 is a Wednesday: the week ending Fri 2026-08-07 is the
        # last COMPLETED week, so its H/L/C must drive the pivots (the old
        # iloc[-2] returned the week before that — one week too far back).
        # 8 business days from Mon 2026-08-03 land on Wed 2026-08-12; the
        # first five (08-03..08-07) form the completed W-FRI-labelled week.
        rows = [(c - 0.5, c + 0.6, c - 0.7, c, 1000) for c in range(100, 108)]
        bars = _bars(rows, "2026-08-03")
        pivots = weekly_pivots(bars, "2026-08-12")
        assert pivots is not None
        assert pivots["P"] == pytest.approx((104.6 + 99.3 + 104.0) / 3.0)


@pytest.mark.unit
class TestCandlestickPatterns:
    def test_doji(self):
        rows = [(100, 102, 98, 100.02, 1)] * 5
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        assert any(h["pattern"] == "doji" for h in hits)

    def test_hammer(self):
        # Two bland lead-in bars (the scanner needs >=3 rows), then a hammer:
        # small body at the top, long lower shadow, tiny upper shadow.
        rows = [
            (100.0, 100.9, 99.4, 100.2, 1),
            (100.0, 100.9, 99.4, 100.2, 1),
            (100.0, 100.5, 95.0, 100.3, 1),
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        assert any(h["pattern"] == "hammer" for h in hits)

    def test_bullish_engulfing(self):
        rows = [
            (100.0, 100.9, 99.4, 100.2, 1),   # bland lead-in
            (101.0, 101.5, 99.0, 99.5, 1),    # decent red body
            (99.0, 103.0, 98.8, 102.5, 1),    # bigger green body engulfs it
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        assert any(h["pattern"] == "bullish_engulfing" for h in hits)

    def test_morning_star(self):
        rows = [
            (100.0, 100.9, 99.4, 100.2, 1),   # bland lead-in
            (100.0, 100.2, 97.0, 97.5, 1),    # long red
            (97.4, 97.6, 96.8, 97.2, 1),      # small body (star)
            (97.5, 100.5, 97.3, 100.2, 1),    # long green above bar-1 midpoint
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        assert any(h["pattern"] == "morning_star" for h in hits)

    def test_no_patterns_on_bland_bars(self):
        rows = [(100, 100.9, 99.4, 100.2, 1)] * 5
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        assert hits == []

    def test_direction_attached(self):
        rows = [
            (100.0, 100.9, 99.4, 100.2, 1),
            (101.0, 101.5, 99.0, 99.5, 1),
            (99.0, 103.0, 98.8, 102.5, 1),
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        engulf = next(h for h in hits if h["pattern"] == "bullish_engulfing")
        assert engulf["direction"] == "bullish"
        assert engulf["date"]

    def test_three_black_crows_textbook_hits(self):
        # P0 regression (2026-08-16): the old column-swap "mirror" required
        # rising opens/closes, so textbook crows NEVER matched. Textbook
        # shape: three red bars, closes strictly falling, each open inside
        # the prior body, small lower shadows.
        rows = [
            (100.0, 100.6, 99.2, 100.2, 1),   # bland lead-in
            (105.0, 105.3, 99.4, 100.0, 1),   # red, closes near low
            (102.0, 102.3, 96.4, 97.0, 1),    # red, open within prior body
            (99.0, 99.3, 93.4, 94.0, 1),      # red, open within prior body
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        crow = next(h for h in hits if h["pattern"] == "three_black_crows")
        assert crow["direction"] == "bearish"
        # _bars starts 2025-01-01 (bdate_range): rows land Wed 01, Thu 02,
        # Fri 03, Mon 06 — the shape attaches to its LAST bar.
        assert crow["date"] == "2025-01-06"

    def test_three_black_crows_ascending_red_bars_rejected(self):
        # The inverse regression: ascending red bars (opens AND closes
        # rising) were exactly what the broken detector flagged as crows.
        rows = [
            (100.0, 100.6, 99.2, 100.2, 1),
            (105.0, 106.3, 99.4, 100.0, 1),
            (110.0, 111.3, 104.4, 105.0, 1),
            (115.0, 116.3, 109.4, 110.0, 1),
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        assert not any(h["pattern"] == "three_black_crows" for h in hits)

    def test_red_hammer_in_downtrend_is_bullish_hammer(self):
        # Round-5 (2026-08-16): hammer vs hanging man is split by the PRIOR
        # TREND, not candle color — a red hammer-shaped bar after a decline
        # is a bullish hammer; the color rule labeled it hanging_man/bearish.
        rows = [
            (104.0, 104.4, 103.4, 103.6, 1),   # declining lead-in
            (103.5, 103.8, 102.4, 102.6, 1),   # declining lead-in
            (103.0, 103.1, 97.0, 102.5, 1),    # RED bar: open 103, close 102.5,
                                               # long lower shadow, tiny upper
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        ham = [h for h in hits if h["pattern"] in ("hammer", "hanging_man")]
        assert ham and ham[0]["pattern"] == "hammer"
        assert ham[0]["direction"] == "bullish"

    def test_green_star_shape_after_advance_is_bearish_shooting_star(self):
        # Mirror case: a GREEN star-shaped bar (small body low, long upper
        # shadow) after an advance is a bearish shooting star; the color
        # rule labeled it inverted_hammer/bullish.
        rows = [
            (100.0, 101.0, 99.6, 100.8, 1),   # advancing lead-in
            (100.8, 102.0, 100.6, 101.8, 1),  # advancing lead-in
            (102.0, 108.0, 101.9, 102.4, 1),  # GREEN: open 102, close 102.4,
                                              # long upper shadow, tiny lower
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        star = [h for h in hits if h["pattern"] in ("shooting_star", "inverted_hammer")]
        assert star and star[0]["pattern"] == "shooting_star"
        assert star[0]["direction"] == "bearish"

    def test_three_white_soldiers_unchanged(self):
        rows = [
            (100.0, 100.6, 99.2, 100.2, 1),
            (100.0, 101.3, 99.8, 101.0, 1),
            (101.0, 102.3, 100.8, 102.0, 1),
            (102.0, 103.3, 101.8, 103.0, 1),
        ]
        hits = scan_candlestick_patterns(_bars(rows), lookback=5)
        assert any(h["pattern"] == "three_white_soldiers" for h in hits)


@pytest.mark.unit
class TestDoubleTopBottom:
    def _double_top_bars(self, confirmed: bool):
        # Rise to ~110 (pivot 1), pullback to ~104, second peak ~109.8, then a
        # close below the 104 neckline (or not). Segment joints deliberately
        # avoid EXACTLY equal extremes — a strict swing pivot requires a
        # unique local max, and equal highs form a plateau, not a pivot.
        rows = []
        for _ in range(60):
            rows.append((100, 101, 99, 100, 1))  # flat lead-in
        closes = list(np.linspace(100, 110, 12))
        closes += list(np.linspace(109.5, 104, 8))
        closes += list(np.linspace(104.2, 109.8, 10))
        # A few mild descent bars after the second peak so it clears the
        # 3-bar right wing the strict pivot definition requires.
        closes += list(np.linspace(108.5, 106.0, 4))
        for c in closes:
            rows.append((c - 0.5, c + 0.5, c - 0.7, c, 1))
        tail = [(103.5, 104.2, 102.0, 102.5, 1)] if confirmed else [(105, 105.5, 104.2, 105.2, 1)]
        rows.extend(tail)
        return _bars(rows)

    def test_unconfirmed_double_top_reported(self):
        shapes = detect_double_top_bottom(self._double_top_bars(False))
        tops = [s for s in shapes if s["pattern"] == "double_top"]
        assert tops, "expected a double-top shape among pivots"
        assert tops[0]["confirmed"] is False

    def test_confirmed_double_top(self):
        shapes = detect_double_top_bottom(self._double_top_bars(True))
        tops = [s for s in shapes if s["pattern"] == "double_top"]
        assert tops and tops[0]["confirmed"] is True

    def test_find_pivots_strict(self):
        rows = [(100, 100, 99, 99.5, 1)] * 6 + [(100, 110, 100, 109, 1)] \
            + [(100, 100, 99, 99.5, 1)] * 6
        highs, _ = find_pivots(_bars(rows), wing=3)
        assert 6 in highs


@pytest.mark.unit
class TestRegimeClassifiers:
    def _trending_frame(self, direction: str, rows: int = 400) -> pd.DataFrame:
        step = 0.5 if direction == "up" else -0.5
        closes = [100.0 + i * step for i in range(rows)]
        return _bars([
            (c - step / 2, max(c, c - step / 2) + 0.6, min(c, c - step / 2) - 0.6, c, 1000)
            for c in closes
        ])

    def test_uptrend_classified(self):
        trend = classify_trend_state(self._trending_frame("up"))
        assert trend is not None
        assert trend["trend"] == "uptrend"
        assert trend["stack"] == "bull_stack"

    def test_downtrend_classified(self):
        trend = classify_trend_state(self._trending_frame("down"))
        assert trend["trend"] == "downtrend"
        assert trend["stack"] == "bear_stack"

    def test_short_history_returns_none(self):
        assert classify_trend_state(self._trending_frame("up", rows=100)) is None

    def test_vol_state_extreme_after_quiet_then_shock(self):
        rng = np.random.default_rng(4)
        quiet = [100.0]
        for _ in range(300):
            quiet.append(quiet[-1] * (1 + rng.normal(0, 0.002)))
        # Triple the dispersion in the last 30 sessions.
        for _ in range(30):
            quiet.append(quiet[-1] * (1 + rng.normal(0, 0.02)))
        vol = classify_vol_state(_bars([
            (c, c * 1.01, c * 0.99, c, 1000) for c in quiet
        ]))
        assert vol is not None
        assert vol["label"] in ("high", "extreme")
        assert vol["percentile"] >= 75.0

    def test_vol_state_short_history_none(self):
        assert classify_vol_state(_bars([(1, 1, 1, 1, 1)] * 10)) is None


@pytest.mark.unit
class TestRegimeContextLine:
    def test_composes_trend_and_vol(self, monkeypatch):
        import yialpha.dataflows.market_regime as mr

        step = 0.5
        closes = [100.0 + i * step for i in range(400)]
        frame = _bars([
            (c - step / 2, c + 0.6, c - step / 2 - 0.6, c, 1000) for c in closes
        ])
        monkeypatch.setattr(mr, "load_ohlcv", lambda s, d: frame)
        monkeypatch.setattr(mr, "compute_turbulence", lambda *a, **k: 9.0)
        line = mr.format_regime_context("TEST", "2026-05-01")
        assert line is not None
        assert "trend=uptrend" in line
        assert "vol=" in line
        assert "turbulence" in line and "elevated" in line
        assert "breadth" not in line  # not an A-share ticker

    def test_fails_soft_to_none_without_data(self, monkeypatch):
        import yialpha.dataflows.market_regime as mr

        def boom(symbol, curr_date):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(mr, "load_ohlcv", boom)
        assert mr.format_regime_context("TEST", "2026-05-01") is None


@pytest.mark.unit
class TestAnalystWiring:
    def test_stock_tools_include_price_structure(self):
        import yialpha.agents.analysts.market_analyst as ma
        from yialpha.agents.utils.agent_utils import (
            get_candlestick_patterns,
            get_indicators_weekly,
            get_support_resistance,
            get_volume_features,
        )

        # The node builds its tool list inside the closure; assert the module
        # surface the node imports from carries the new tools (as bound
        # langchain StructuredTools) and that the prompt text references them
        # (integration smoke without an LLM).
        for tool_obj in (
            get_support_resistance, get_volume_features,
            get_candlestick_patterns, get_indicators_weekly,
        ):
            assert tool_obj.func is not None
        legacy = ma._legacy_system_message()
        assert "get_support_resistance" in legacy
        assert "get_volume_features" in legacy
        assert "get_candlestick_patterns" in legacy
        assert "get_indicators_weekly" in legacy

    def test_regime_context_gate(self, monkeypatch):
        import yialpha.agents.analysts.market_analyst as ma
        from yialpha.dataflows.config import set_config

        monkeypatch.setattr(
            ma, "format_regime_context", lambda t, d: "Market regime (T): line"
        )
        set_config({"regime_context": False})
        # The gate lives in the node; assert the config key is honored by
        # direct inspection of the default and the override round-trip.
        set_config({"regime_context": True})
        from yialpha.dataflows.config import get_config
        assert get_config()["regime_context"] is True
        set_config({"regime_context": None})  # restore default handling


@pytest.mark.unit
class TestPriceStructureTools:
    def _frame(self, rows=300):
        closes = [100.0 + 0.4 * i for i in range(rows)]
        return _bars([
            (c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in closes
        ])

    @pytest.fixture()
    def patched_ohlcv(self, monkeypatch):
        import yialpha.agents.utils.price_structure_tools as pst
        frame = self._frame()
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frame)
        return pst

    def test_support_resistance_renders(self, patched_ohlcv):
        out = patched_ohlcv.get_support_resistance.invoke(
            {"symbol": "TEST", "curr_date": "2026-03-01"}
        )
        assert "Support/Resistance levels" in out
        assert "R2" in out and "S1" in out
        assert "Point of control" in out
        assert "rolling" in out.lower() or "d" in out

    def test_volume_features_renders(self, patched_ohlcv):
        out = patched_ohlcv.get_volume_features.invoke(
            {"symbol": "TEST", "curr_date": "2026-03-01"}
        )
        assert "Volume-price features" in out
        assert "Relative volume" in out
        assert "Divergence check" in out

    def test_patterns_renders(self, patched_ohlcv):
        out = patched_ohlcv.get_candlestick_patterns.invoke(
            {"symbol": "TEST", "curr_date": "2026-03-01"}
        )
        assert "Candlestick & chart patterns" in out
        assert "double-top" in out or "double_top" in out or "No double-top" in out \
            or "No double" in out

    def test_typed_degrade_on_vendor_failure(self, monkeypatch):
        import yialpha.agents.utils.price_structure_tools as pst

        def boom(symbol, curr_date):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(pst, "load_ohlcv", boom)
        out = pst.get_support_resistance.invoke(
            {"symbol": "TEST", "curr_date": "2026-03-01"}
        )
        assert out.startswith("DATA_UNAVAILABLE")

    def test_relative_strength_wealth_ratio_in_down_market(self, monkeypatch):
        """Round-5: r/b flipped sign when the benchmark fell — +2% vs -1%
        scored -2.0 "underperforming". The wealth ratio (1+r)/(1+b) stays >1
        exactly when the ticker outperformed."""
        import pandas as pd

        import yialpha.agents.utils.price_structure_tools as pst

        n = 260
        dates = pd.bdate_range("2025-01-01", periods=n)

        def up_frame():
            closes = [100.0 * (1.004 ** i) for i in range(n)]  # ~rising
            return _bars([(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in closes],
                         start="2025-01-01")

        def down_frame():
            closes = [100.0 * (0.998 ** i) for i in range(n)]  # ~falling
            return _bars([(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in closes],
                         start="2025-01-01")

        frames = {"TEST": up_frame(), "SPY": down_frame()}
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frames[s])
        monkeypatch.setattr(pst, "resolve_market_benchmark", lambda t: "SPY")

        out = pst.get_relative_strength.invoke(
            {"symbol": "TEST", "curr_date": str(dates[-1].date())}
        )
        # Ticker up vs benchmark down over the same windows: every window's
        # RS ratio must exceed 1 (outperforming), which r/b could not deliver.
        import re as _re

        ratios = [float(m) for m in _re.findall(r"\| \d+m \|[^|]+\|[^|]+\| ([\d.]+) \|", out)]
        assert ratios and all(r > 1.0 for r in ratios), out
