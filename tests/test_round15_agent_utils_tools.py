"""Round-15 coverage hardening: the four agents/utils tool modules.

Targets the honest-degrade seams and render contracts the earlier rounds'
tests only reach on their happy paths:

* ``valuation_tools`` -- the whole PoT table render (hand-computed Graham /
  PEG / owner-earnings / two-stage-DCF / WACC values, and the all-``n/a``
  contract when the analyst supplies only a price);
* ``technical_indicators_tools`` -- the per-indicator ValueError degrade;
* ``weekly_indicators_tools`` -- sub-2-week degrade, one-column indicator
  failure, and the ``_fmt_num`` garbage guards;
* ``price_structure_tools`` -- DATA_UNAVAILABLE degrades, short-history
  placeholders, the pattern-hit / double-top render branches, the
  relative-strength window guards, and the formatter guards.

Every vendor/data seam is monkeypatched; nothing here touches the network.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _bars_round15(rows, start="2025-01-06"):
    """OHLCV frame from (open, high, low, close, volume) tuples."""
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    frame.insert(0, "Date", pd.bdate_range(start, periods=len(frame)))
    return frame


def _daily_round15(dates, closes):
    """Daily OHLCV frame from parallel date/close sequences."""
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": closes - 1.0,
            "High": closes + 2.0,
            "Low": closes - 2.0,
            "Close": closes,
            "Volume": [100.0] * len(closes),
        }
    )


# ---------------------------------------------------------------------------
# valuation_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15ValuationMetrics:
    def test_full_battery_renders_hand_computed_values(self):
        """The PoT sink returns the textbook formulas' exact values.

        Every expected number below is hand-computed from the public formula
        (P/E 50/2.5, Graham sqrt(22.5*2.5*20), NCAV (1000-400)/50, PEG
        20/10, owner earnings 100+30-20, WACC 0.9*0.10+0.1*0.05, and the
        Damodaran two-stage DCF for FCF=4, g=8%, r=10%, gt=3%, n=10 ->
        85.2007 with margin of safety (85.2007-50)/85.2007 = 41.32%)."""
        from yialpha.agents.utils.valuation_tools import get_valuation_metrics

        out = get_valuation_metrics.invoke(
            {
                "price": 50.0,
                "eps": 2.5,
                "book_value_per_share": 20.0,
                "current_assets": 1000.0,
                "total_liabilities": 400.0,
                "shares_outstanding": 50.0,
                "earnings_growth_pct": 10.0,
                "net_income": 100.0,
                "depreciation_amortization": 30.0,
                "maintenance_capex": 20.0,
                "free_cash_flow_per_share": 4.0,
                "growth_rate": 0.08,
                "discount_rate": 0.10,
                "terminal_growth_rate": 0.03,
                "high_growth_years": 10,
                "market_value_equity": 900.0,
                "market_value_debt": 100.0,
                "cost_of_equity": 0.10,
                "after_tax_cost_of_debt": 0.05,
            }
        )
        assert "## Deterministic valuation metrics" in out
        assert "- Market price / share: 50.0000" in out
        assert "| Metric | Value |" in out
        for row in (
            "| P/E ratio | 20.0000 |",
            "| Earnings yield | 5.00% |",
            "| Graham number | 33.5410 |",
            "| NCAV per share (net-net) | 12.0000 |",
            "| PEG ratio | 2.0000 |",
            "| Owner earnings | 110.0000 |",
            "| DCF intrinsic value / share | 85.2007 |",
            "| Margin of safety (vs DCF) | 41.32% |",
            "| WACC | 9.50% |",
        ):
            assert row in out, row
        # The n/a semantics note rides along on every render.
        assert "not zero" in out

    def test_price_only_renders_every_metric_na(self):
        """A price with no line items must render n/a, never zero/fabricated."""
        from yialpha.agents.utils.valuation_tools import get_valuation_metrics

        out = get_valuation_metrics.invoke({"price": 50.0})
        assert "- Market price / share: 50.0000" in out
        for name in (
            "P/E ratio",
            "Earnings yield",
            "Graham number",
            "NCAV per share (net-net)",
            "PEG ratio",
            "Owner earnings",
            "DCF intrinsic value / share",
            "Margin of safety (vs DCF)",
            "WACC",
        ):
            assert f"| {name} | n/a |" in out, name

    def test_loss_maker_degrades_pe_but_keeps_negative_yield(self):
        """eps <= 0 kills P/E / Graham / PEG; the negative earnings yield is
        itself a signal and must be preserved (-2/50 = -4.00%)."""
        from yialpha.agents.utils.valuation_tools import get_valuation_metrics

        out = get_valuation_metrics.invoke({"price": 50.0, "eps": -2.0})
        assert "| P/E ratio | n/a |" in out
        assert "| Earnings yield | -4.00% |" in out
        assert "| Graham number | n/a |" in out
        assert "| PEG ratio | n/a |" in out

    def test_private_fmt_helpers(self):
        from yialpha.agents.utils.valuation_tools import _fmt, _fmt_pct

        assert _fmt(None) == "n/a"
        assert _fmt(12.0) == "12.0000"
        # _fmt_pct is the shared yialpha.fmt implementation aliased locally.
        assert _fmt_pct(None) == "n/a"
        assert _fmt_pct(0.05) == "5.00%"


# ---------------------------------------------------------------------------
# technical_indicators_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15GetIndicators:
    def test_comma_list_is_stripped_lowercased_and_joined(self, monkeypatch):
        """LLMs pass ' MACD ,, rsi ' -- blank segments drop, names normalize,
        results join with a blank line."""
        import yialpha.agents.utils.technical_indicators_tools as tit

        seen: list[tuple[str, str, int]] = []

        def fake_route(method, symbol, ind, curr_date, look_back_days):
            seen.append((symbol, ind, look_back_days))
            return f"ok:{ind}:{curr_date}:{look_back_days}"

        monkeypatch.setattr(tit, "route_to_vendor", fake_route)
        out = tit.get_indicators.func("AAPL", " MACD ,, rsi ", "2026-01-05", 30)
        assert seen == [("AAPL", "macd", 30), ("AAPL", "rsi", 30)]
        assert out == "ok:macd:2026-01-05:30\n\nok:rsi:2026-01-05:30"

    def test_valueerror_for_one_indicator_degrades_to_its_message(self, monkeypatch):
        """An unknown indicator appends the vendor's message instead of
        sinking the whole multi-indicator call."""
        import yialpha.agents.utils.technical_indicators_tools as tit

        def fake_route(method, symbol, ind, curr_date, look_back_days):
            if ind == "foo":
                raise ValueError("unsupported indicator: foo")
            return f"ok:{ind}"

        monkeypatch.setattr(tit, "route_to_vendor", fake_route)
        out = tit.get_indicators.func("AAPL", "macd,rsi,foo", "2026-01-05", 30)
        assert out == "ok:macd\n\nok:rsi\n\nunsupported indicator: foo"


# ---------------------------------------------------------------------------
# weekly_indicators_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15WeeklyIndicators:
    def test_fmt_num_rejects_nonfinite_and_unparseable(self):
        from yialpha.agents.utils.weekly_indicators_tools import _fmt_num

        assert _fmt_num(None) == "N/A"
        assert _fmt_num(float("nan")) == "N/A"
        assert _fmt_num(float("inf")) == "N/A"
        assert _fmt_num(float("-inf")) == "N/A"
        assert _fmt_num("not-a-number") == "N/A"  # ValueError
        assert _fmt_num(object()) == "N/A"  # TypeError

    def test_fewer_than_two_weekly_bars_degrades_typed(self, monkeypatch):
        import yialpha.agents.utils.weekly_indicators_tools as weekly

        one_week = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-01-09")],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [10.0],
            }
        )
        monkeypatch.setattr(weekly, "weekly_ohlcv", lambda s, d: one_week)
        out = weekly.get_indicators_weekly.func("TEST", "2026-01-09")
        assert out.startswith("DATA_UNAVAILABLE: fewer than 2 complete weekly bars")
        assert "TEST" in out
        assert "weekly timeframe not assessable" in out

    def test_one_indicator_failure_renders_na_column_not_dead_table(
        self, monkeypatch
    ):
        """A stockstats failure on ONE column (macd) must leave the rest of
        the weekly table intact and mark only that column N/A."""
        import yialpha.agents.utils.weekly_indicators_tools as weekly
        import yialpha.dataflows.ohlcv_resample as resample_mod
        from yialpha.dataflows.stockstats_utils import (
            compute_indicator as real_compute,
        )

        dates = pd.date_range("2025-01-06", periods=300, freq="B")
        daily = _daily_round15(dates, [100.0 + 0.5 * i for i in range(300)])
        monkeypatch.setattr(
            resample_mod,
            "load_ohlcv",
            lambda s, d: daily[daily["Date"] <= pd.Timestamp(d)],
        )

        def flaky(sdf, name):
            if name == "macd":
                raise RuntimeError("stockstats macd boom")
            return real_compute(sdf, name)

        monkeypatch.setattr(weekly, "compute_indicator", flaky)
        out = weekly.get_indicators_weekly.invoke(
            {"symbol": "TEST", "curr_date": "2026-02-27"}
        )

        # The failed column keeps its header (keyed by dict key, honest N/A
        # values); the healthy columns still render numbers.
        assert "| Week ending | Close |" in out
        assert "macd (weekly MACD (12,26,9))" in out
        summary = [ln for ln in out.splitlines() if ln.startswith("Latest complete week")]
        assert summary, out
        assert "weekly MACD (12,26,9)=N/A" in summary[0]
        assert "weekly RSI (14)=N/A" not in summary[0]


# ---------------------------------------------------------------------------
# price_structure_tools: formatter guards + degrades
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15PriceStructureFormatters:
    def test_fmt_rejects_none_nan_and_garbage(self):
        from yialpha.agents.utils.price_structure_tools import _fmt

        assert _fmt(None) == "N/A"
        assert _fmt(float("nan")) == "N/A"
        assert _fmt("junk") == "N/A"  # ValueError
        assert _fmt(object()) == "N/A"  # TypeError
        assert _fmt(1.5) == "1.50"

    def test_fmt_price_rejects_garbage(self):
        from yialpha.agents.utils.price_structure_tools import _fmt_price

        assert _fmt_price("junk") == "N/A"  # ValueError
        assert _fmt_price(object()) == "N/A"  # TypeError

    def test_fmt_pct_rejects_garbage(self):
        from yialpha.agents.utils.price_structure_tools import _fmt_pct

        assert _fmt_pct("junk") == "n/a"
        assert _fmt_pct(object()) == "n/a"
        assert _fmt_pct(0.0123) == "+1.23%"
        assert _fmt_pct(None) == "n/a"


@pytest.mark.unit
class TestRound15PriceStructureDegrades:
    def test_volume_features_data_unavailable_on_load_failure(self, monkeypatch):
        import yialpha.agents.utils.price_structure_tools as pst

        def boom(symbol, curr_date):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(pst, "load_ohlcv", boom)
        out = pst.get_volume_features.invoke(
            {"symbol": "TEST", "curr_date": "2026-03-01"}
        )
        assert out.startswith("DATA_UNAVAILABLE: OHLCV for 'TEST'")
        assert "RuntimeError" in out

    def test_candlestick_patterns_data_unavailable_on_load_failure(self, monkeypatch):
        import yialpha.agents.utils.price_structure_tools as pst

        def boom(symbol, curr_date):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(pst, "load_ohlcv", boom)
        out = pst.get_candlestick_patterns.invoke(
            {"symbol": "TEST", "curr_date": "2026-03-01"}
        )
        assert out.startswith("DATA_UNAVAILABLE: OHLCV for 'TEST'")
        assert "RuntimeError" in out

    def test_relative_strength_no_ohlcv_history_degrades_typed(self, monkeypatch):
        import yialpha.agents.utils.price_structure_tools as pst

        empty = pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"),
                              "Close": pd.Series(dtype=float)})
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: empty)
        out = pst.get_relative_strength.func("TEST", "2026-03-02")
        assert out.startswith("DATA_UNAVAILABLE: no OHLCV history for 'TEST'")


# ---------------------------------------------------------------------------
# price_structure_tools: render branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15PriceStructureRender:
    def test_support_resistance_short_history_uses_placeholders(self, monkeypatch):
        """Four bars inside one still-open week: daily pivots render, but the
        weekly pivots / rolling extremes / volume profile each fall back to
        the honest '_Not enough history._' placeholder."""
        import yialpha.agents.utils.price_structure_tools as pst

        frame = _bars_round15([(99, 101, 98, 100, 1000)] * 4, "2025-01-06")
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frame)
        out = pst.get_support_resistance.invoke(
            {"symbol": "TEST", "curr_date": "2025-01-09"}
        )
        assert "Support/Resistance levels for TEST" in out
        # Daily pivots DID render (>= 2 rows -> previous session exists).
        assert "### Previous session pivots (classic)" in out
        assert "R2 " in out and "S2 " in out
        # The three short-history families each degraded honestly.
        assert out.count("_Not enough history._") == 3
        assert "| Window |" not in out  # no rolling table without 20 rows
        assert "Point of control" not in out  # no profile without 60 rows

    def test_volume_features_vr_failure_renders_na_not_crash(self, monkeypatch):
        """A stockstats 'vr' failure must not sink the volume render -- only
        that one reading degrades to N/A while OBV still reports."""
        import yialpha.agents.utils.price_structure_tools as pst

        closes = [100.0 + 0.4 * i for i in range(300)]
        frame = _bars_round15(
            [(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in closes], "2025-01-06"
        )
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frame)

        def boom_vr(sdf, name):
            raise ValueError("vr unsupported")

        monkeypatch.setattr(pst, "compute_indicator", boom_vr)
        out = pst.get_volume_features.invoke(
            {"symbol": "TEST", "curr_date": "2026-03-01"}
        )
        assert "Volume-price features for TEST" in out
        assert "- Volume Ratio (vr, 26-session): N/A" in out
        # Monotonic 1000-volume advance: OBV = 299 * 1000 with digits=0.
        assert "- OBV (latest): 299000" in out

    def test_volume_features_short_history_divergence_not_enough(self, monkeypatch):
        """8 bars: the 20-session divergence check and both 10-session
        slopes honestly report missing history instead of guessing."""
        import yialpha.agents.utils.price_structure_tools as pst

        closes = [100.0 + i for i in range(8)]
        frame = _bars_round15(
            [(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in closes], "2025-01-06"
        )
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frame)
        out = pst.get_volume_features.invoke(
            {"symbol": "TEST", "curr_date": "2025-01-15"}
        )
        assert "Volume-price features for TEST" in out
        assert "Divergence check: _not enough history (needs 20 sessions)._" in out
        assert "- Price 10-session change: N/A" in out

    def test_candlestick_patterns_renders_hits_table(self, monkeypatch):
        """A real doji bar produces the pattern-hit table (pattern + date +
        direction) the analyst is told to cite."""
        import yialpha.agents.utils.price_structure_tools as pst

        rows = [(99, 101, 98, 100, 1000)] * 6
        rows += [(100, 101, 99, 100, 1000)]  # doji: open == close
        rows += [(100, 102, 100, 100.5, 1000)]
        frame = _bars_round15(rows, "2025-01-06")
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frame)
        out = pst.get_candlestick_patterns.invoke(
            {"symbol": "TEST", "curr_date": "2025-01-15"}
        )
        assert "Candlestick & chart patterns for TEST" in out
        assert "| Pattern | Date | Direction |" in out
        assert "| doji | 2025-01-14 | neutral |" in out
        # Short frame: no pivots wide enough for a double top/bottom.
        assert "No double-top/bottom shape among recent swing pivots" in out

    def test_candlestick_patterns_renders_double_top_shapes(self, monkeypatch):
        """Confirmed and unconfirmed double-extreme shapes render with their
        pivots, level, neckline, and the confirmation status sentence."""
        import yialpha.agents.utils.price_structure_tools as pst

        frame = _bars_round15([(99, 101, 98, 100, 1000)] * 8, "2025-01-06")
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frame)
        monkeypatch.setattr(
            pst, "scan_candlestick_patterns", lambda df, lookback: []
        )
        monkeypatch.setattr(
            pst,
            "detect_double_top_bottom",
            lambda df, lookback: [
                {
                    "pattern": "double_top",
                    "direction": "bearish",
                    "first_pivot_date": "2025-06-02",
                    "second_pivot_date": "2025-08-01",
                    "level": 110.5,
                    "neckline": 100.25,
                    "confirmed": True,
                },
                {
                    "pattern": "double_bottom",
                    "direction": "bullish",
                    "first_pivot_date": "2025-09-01",
                    "second_pivot_date": "2025-10-15",
                    "level": 90.125,
                    "neckline": 95.5,
                    "confirmed": False,
                },
            ],
        )
        out = pst.get_candlestick_patterns.invoke(
            {"symbol": "TEST", "curr_date": "2025-01-15"}
        )
        assert "No classic candlestick patterns" in out
        assert "**double_top** (bearish): pivots 2025-06-02 & 2025-08-01" in out
        assert "near 110.5; neckline 100.25" in out
        assert "CONFIRMED (neckline broken on a close)" in out
        assert "**double_bottom** (bullish): pivots 2025-09-01 & 2025-10-15" in out
        assert (
            "UNCONFIRMED (neckline not yet broken — a setup, not a signal)" in out
        )


# ---------------------------------------------------------------------------
# price_structure_tools: relative-strength window guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRound15RelativeStrengthHelpers:
    def test_window_return_guards(self):
        from yialpha.agents.utils.price_structure_tools import _window_return

        close = pd.Series(
            [100.0 + i for i in range(10)],
            index=pd.bdate_range("2025-01-01", periods=10),
        )
        assert _window_return(close, 10) is None  # needs len > rows
        assert _window_return(close, 1) == pytest.approx(109.0 / 108.0 - 1.0)

        with_zero = pd.Series(
            [100.0, 100.0, 0.0, 104.0, 105.0, 106.0],
            index=pd.bdate_range("2025-01-01", periods=6),
        )
        assert _window_return(with_zero, 3) is None  # anchor close <= 0

    def test_bench_return_over_guards(self):
        from yialpha.agents.utils.price_structure_tools import _bench_return_over

        jan1, jan8 = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-08")
        bench = pd.Series([100.0, 110.0], index=[jan1, jan8])
        assert _bench_return_over(bench, jan1, 121.0) == pytest.approx(0.21)

        # Empty benchmark / missing end value: not computable.
        assert _bench_return_over(pd.Series(dtype=float), jan1, 121.0) is None
        assert _bench_return_over(bench, jan1, None) is None
        assert _bench_return_over(bench, jan1, 0.0) is None

        # Unsorted benchmark index: asof() raises, the guard returns None.
        unsorted = pd.Series([1.0, 2.0], index=[jan8, jan1])
        assert _bench_return_over(unsorted, pd.Timestamp("2025-01-03"), 1.5) is None

        # asof before the first benchmark row returns NaN -> None.
        late = pd.Series([1.0], index=[pd.Timestamp("2025-06-01")])
        assert _bench_return_over(late, jan1, 1.5) is None

        # Anchor lands on a zero benchmark close -> None, not inf.
        zero_anchor = pd.Series([0.0, 1.0], index=[jan1, jan8])
        assert _bench_return_over(zero_anchor, pd.Timestamp("2025-01-03"), 1.5) is None

    def test_short_history_trend_line_says_unjudgable(self, monkeypatch):
        """30 ticker rows vs a full benchmark: only the 1m window yields a
        ratio, so the trend verdict must be the honest 'not enough
        overlapping windows' line -- never a fabricated trend."""
        import yialpha.agents.utils.price_structure_tools as pst

        ticker = _bars_round15(
            [(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in
             (100.0 + 0.5 * i for i in range(30))],
            "2025-01-06",
        )
        bench = _bars_round15(
            [(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in
             (400.0 + 0.2 * i for i in range(300))],
            "2025-01-06",
        )
        frames = {"TEST": ticker, "SPY": bench}
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frames[s])
        out = pst.get_relative_strength.func("TEST", "2026-02-27")
        assert "Relative strength: TEST vs SPY" in out
        assert "RS trend: not enough overlapping windows to judge the trend." in out
        assert "| 12m | n/a | n/a | n/a |" in out
        # The 1m window had enough rows for a real ratio.
        assert "| 1m |" in out and out.count("| 1m |") == 1

    def test_benchmark_unavailable_reports_absolute_returns_only(self, monkeypatch):
        """When SPY cannot be loaded the RS ratios degrade to n/a and the
        verdict says absolute-returns-only instead of inventing a relative
        read (the benchmark leg's typed degrade)."""
        import yialpha.agents.utils.price_structure_tools as pst

        ticker = _bars_round15(
            [(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in
             (100.0 + 0.5 * i for i in range(30))],
            "2025-01-06",
        )

        def frames(symbol, curr_date):
            if symbol == "SPY":
                raise RuntimeError("benchmark vendor down")
            return ticker

        monkeypatch.setattr(pst, "load_ohlcv", frames)
        out = pst.get_relative_strength.func("TEST", "2026-02-27")
        assert "Relative strength: TEST vs SPY" in out
        assert "| 1m |" in out  # ticker leg still rendered
        assert (
            "Benchmark SPY unavailable as of 2026-02-27 — absolute "
            "returns only; report relative strength as unavailable." in out
        )
        # No ratio column was fabricated without the benchmark.
        assert all(
            ln.endswith("| n/a |")
            for ln in out.splitlines()
            if ln.startswith("| 1m |") or ln.startswith("| 12m |")
        )

    def test_nonpositive_window_anchor_skips_that_ratio(self, monkeypatch):
        """A zero close 21 rows back makes the 1m window uncomputable (None,
        not a fabricated ratio) while longer windows still report."""
        import yialpha.agents.utils.price_structure_tools as pst

        closes = [100.0 + 0.1 * i for i in range(260)]
        closes[-1 - 21] = 0.0  # the 21-row anchor is worthless
        ticker = _bars_round15(
            [(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in closes], "2025-01-06"
        )
        bench = _bars_round15(
            [(c - 0.2, c + 0.5, c - 0.6, c, 1000) for c in
             (400.0 + 0.2 * i for i in range(300))],
            "2025-01-06",
        )
        frames = {"TEST": ticker, "SPY": bench}
        monkeypatch.setattr(pst, "load_ohlcv", lambda s, d: frames[s])
        out = pst.get_relative_strength.func("TEST", "2026-01-05")
        one_m = [ln for ln in out.splitlines() if ln.startswith("| 1m |")]
        assert one_m, out
        assert one_m[0].endswith("| n/a |")  # ratio column
        three_m = [ln for ln in out.splitlines() if ln.startswith("| 3m |")]
        assert three_m and not three_m[0].endswith("| n/a |")
