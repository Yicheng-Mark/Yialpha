"""WP10: trade-quality metrics, IR/TE, CSI-300 benchmark, RS tool.

Pure-math pins for profit factor / avg win-loss / payoff and
information-ratio / tracking-error; the A-share benchmark map now resolves
to CSI 300 on both .SS and .SZ; the relative-strength tool renders the
1/3/6/12m comparison against the resolved benchmark.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from yiagents.backtest.metrics import (
    BacktestMetrics,
    benchmark_comparison,
    compute_metrics,
    trade_quality_stats,
)


@pytest.mark.unit
class TestTradeQualityStats:
    def test_mixed_trades_hand_values(self):
        # wins +10%, +6% (gross 16%); losses -4%, -2% (gross 6%) ->
        # PF = 16/6, avg_win +8%, avg_loss -3%, payoff 8/3.
        stats = trade_quality_stats([0.10, -0.04, 0.06, -0.02])
        assert stats["profit_factor"] == pytest.approx(16.0 / 6.0)
        assert stats["avg_win"] == pytest.approx(0.08)
        assert stats["avg_loss"] == pytest.approx(-0.03)
        assert stats["payoff_ratio"] == pytest.approx(8.0 / 3.0)

    def test_all_wins_give_infinite_pf(self):
        stats = trade_quality_stats([0.05, 0.10])
        assert math.isinf(stats["profit_factor"])  # type: ignore[arg-type]
        assert stats["avg_loss"] is None
        assert stats["payoff_ratio"] is None

    def test_no_trades_all_none(self):
        assert trade_quality_stats([]) == {
            "profit_factor": None, "avg_win": None,
            "avg_loss": None, "payoff_ratio": None,
        }

    def test_all_losses_pf_zero(self):
        stats = trade_quality_stats([-0.05, -0.10])
        assert stats["profit_factor"] == pytest.approx(0.0)


@pytest.mark.unit
class TestBenchmarkComparison:
    def test_hand_values(self):
        # active = +1% every period over 4 periods: mean .01, std(ddof=1)=0
        # -> TE 0, IR None (no denominator).
        strat = np.array([0.03, 0.03, 0.03, 0.03])
        bench = np.array([0.02, 0.02, 0.02, 0.02])
        ir, te = benchmark_comparison(strat, bench, 252)
        assert te == pytest.approx(0.0)
        assert ir is None

    def test_ir_te_hand_values(self):
        # active = [+0.01, -0.01, +0.01, -0.01]: mean 0; std(ddof=1) over the
        # 4 values is 0.01*sqrt(4/3) (divide by n-1=3) -> TE scales by
        # sqrt(252); IR = 0 / TE = 0.
        strat = np.array([0.03, 0.01, 0.03, 0.01])
        bench = np.array([0.02, 0.02, 0.02, 0.02])
        ir, te = benchmark_comparison(strat, bench, 252)
        assert te == pytest.approx(0.01 * math.sqrt(4.0 / 3.0) * math.sqrt(252))
        assert ir == pytest.approx(0.0)

    def test_length_mismatch_returns_none(self):
        ir, te = benchmark_comparison([0.01, 0.02], [0.01], 252)
        assert ir is None and te is None

    def test_compute_metrics_fills_ir_te(self):
        # Equity that beats a flat benchmark by a constant 1%/period.
        eq = [100.0]
        bh = [100.0]
        for _ in range(10):
            eq.append(eq[-1] * 1.01)
            bh.append(bh[-1] * 1.00)
        m = compute_metrics(eq, benchmark_equity=bh)
        assert m.tracking_error == pytest.approx(0.0)
        assert m.information_ratio is None  # zero-variance active series
        assert m.alpha_vs_buyhold == pytest.approx(0.01 * 252)
        # Fields default to None without a benchmark.
        m2 = compute_metrics(eq)
        assert m2.information_ratio is None and m2.tracking_error is None

    def test_dataclass_defaults_constructible(self):
        m = BacktestMetrics(
            total_return=0.1, cagr=0.1, volatility=0.1, sharpe=1.0,
            sortino=1.0, max_drawdown=-0.1, calmar=1.0, deflated_sharpe=0.5,
            alpha_vs_buyhold=0.0, num_periods=10, periods_per_year=252,
        )
        assert m.profit_factor is None
        assert m.benchmark_name is None


@pytest.mark.unit
class TestBenchmarkMapCsi300:
    def test_a_share_suffixes_resolve_to_csi300(self):
        from yiagents.dataflows.config import get_config
        from yiagents.dataflows.market_regime import resolve_market_benchmark

        bmap = get_config()["benchmark_map"]
        assert bmap[".SS"] == "000300.SS"
        assert bmap[".SZ"] == "000300.SS"
        assert resolve_market_benchmark("600519.SS") == "000300.SS"
        assert resolve_market_benchmark("000001.SZ") == "000300.SS"
        assert resolve_market_benchmark("AAPL") == "SPY"


@pytest.mark.unit
class TestRelativeStrengthTool:
    def _frame(self, closes, start="2024-01-01"):
        frame = pd.DataFrame({
            "Date": pd.bdate_range(start, periods=len(closes)),
            "Open": closes, "High": closes, "Low": closes,
            "Close": closes, "Volume": [1000.0] * len(closes),
        })
        return frame

    @pytest.fixture()
    def patched(self, monkeypatch):
        import yiagents.agents.utils.price_structure_tools as pst

        # Ticker doubles over ~2y; benchmark up 20% over the same rows.
        n = 300
        ticker = self._frame(list(np.linspace(100, 200, n)))
        bench = self._frame(list(np.linspace(100, 120, n)))
        monkeypatch.setattr(pst, "resolve_market_benchmark", lambda s: "BMKT")
        monkeypatch.setattr(
            pst, "load_ohlcv",
            lambda s, d: bench if s == "BMKT" else ticker,
        )
        return pst

    def test_renders_windows_and_trend(self, patched):
        out = patched.get_relative_strength.invoke({
            "symbol": "TEST", "curr_date": "2025-12-31",
        })
        assert "Relative strength: TEST vs BMKT" in out
        for w in ("1m", "3m", "6m", "12m"):
            assert w in out
        assert "RS ratio" in out
        assert "improving" in out or "fading" in out

    def test_benchmark_unavailable_degrades_named(self, monkeypatch):
        import yiagents.agents.utils.price_structure_tools as pst

        n = 300
        ticker = self._frame(list(np.linspace(100, 200, n)))

        def fake_load(symbol, curr_date):
            if symbol == "BMKT":
                raise RuntimeError("bench down")
            return ticker

        monkeypatch.setattr(pst, "resolve_market_benchmark", lambda s: "BMKT")
        monkeypatch.setattr(pst, "load_ohlcv", fake_load)
        out = pst.get_relative_strength.invoke({
            "symbol": "TEST", "curr_date": "2025-12-31",
        })
        assert "BMKT unavailable" in out
        assert "relative strength as unavailable" in out

    def test_ticker_failure_is_typed(self, monkeypatch):
        import yiagents.agents.utils.price_structure_tools as pst

        def boom(symbol, curr_date):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(pst, "load_ohlcv", boom)
        out = pst.get_relative_strength.invoke({
            "symbol": "TEST", "curr_date": "2025-12-31",
        })
        assert out.startswith("DATA_UNAVAILABLE")


@pytest.mark.unit
class TestReportRendersNewMetrics:
    def test_metric_rows_present(self):
        from yiagents.backtest.engine import BacktestResult
        from yiagents.backtest.report import render_backtest_report

        metrics = BacktestMetrics(
            total_return=0.2, cagr=0.2, volatility=0.1, sharpe=1.5,
            sortino=1.8, max_drawdown=-0.1, calmar=2.0, deflated_sharpe=0.6,
            alpha_vs_buyhold=0.05, num_periods=50, periods_per_year=252,
            win_rate=0.6, profit_factor=1.8, avg_win=0.03, avg_loss=-0.02,
            payoff_ratio=1.5, information_ratio=0.9, tracking_error=0.04,
            benchmark_name="X buy-and-hold", num_trades=10,
        )
        result = BacktestResult(
            ticker="X", initial_capital=100_000.0, holding_days=5,
            equity=[100_000.0, 120_000.0], equity_dates=["2025-01-01", "2025-03-01"],
            trades=[], benchmark_equity=[100_000.0, 115_000.0],
            benchmark_name="X buy-and-hold", metrics=metrics,
            config_summary={}, cached_hits=0, cached_misses=0,
            degraded_decision_count=0, unexecuted_decision_count=0,
        )
        text = render_backtest_report(result)
        assert "Profit factor" in text
        assert "1.80" in text
        assert "Information ratio vs B&H" in text
        assert "Tracking error vs B&H" in text
        assert "Payoff ratio" in text
        assert "Avg win / avg loss" in text
