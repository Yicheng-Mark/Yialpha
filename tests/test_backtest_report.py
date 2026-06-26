"""Unit tests for the backtest report / multi-run distribution (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from yiagents.backtest.engine import BacktestResult, TradeRow, run_backtest
from yiagents.backtest.report import (
    multi_run,
    render_backtest_report,
    render_multi_run_report,
    summarize_distribution,
    write_report,
)


class FakeGraph:
    def __init__(self, ratings):
        self._ratings = dict(ratings)

    def propagate(self, company_name, trade_date, asset_type="stock"):
        r = self._ratings.get(trade_date, "Hold")
        return {"final_trade_decision": f"**Rating**: {r}"}, r

    def _resolve_benchmark(self, t):
        return "SPY"


def _rising(ticker, start, end):
    idx = pd.bdate_range(start, end)
    vals = [100.0 * (1 + 0.002 * i) for i in range(len(idx))]
    return pd.Series(vals, index=idx.strftime("%Y-%m-%d"), dtype=float)


def _dates(n=8, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n * 6, freq="B")
    return [idx[i].strftime("%Y-%m-%d") for i in range(0, n * 6, 5)][:n]


@pytest.fixture()
def buy_result():
    dates = _dates(8)
    g = FakeGraph({d: "Buy" for d in dates})
    return run_backtest(g, "AAPL", dates, holding_days=5, price_provider=_rising)


@pytest.mark.unit
def test_single_report_renders_sections(buy_result: BacktestResult):
    md = render_backtest_report(buy_result)
    assert "# Backtest: AAPL" in md
    assert "## Equity curve" in md
    assert "## Metrics" in md
    assert "Sharpe" in md
    assert "Max drawdown" in md
    assert "BEATS" in md or "TRAILS" in md
    # Sparkline present (non-empty unicode bars).
    assert "▁" in md or "█" in md


@pytest.mark.unit
def test_report_includes_trades_table(buy_result: BacktestResult):
    md = render_backtest_report(buy_result)
    assert "## Trades" in md
    assert "| Rating |" in md


@pytest.mark.unit
def test_distribution_aggregates_runs(buy_result: BacktestResult):
    # Build a few synthetic variants by perturbing the equity on copies.
    variants = []
    for scale in (0.9, 1.0, 1.1):
        r = buy_result
        # Mutate total_return via equity scaling for a real distribution signal.
        scaled = BacktestResult(
            ticker=r.ticker, initial_capital=r.initial_capital, holding_days=r.holding_days,
            equity=[v * scale for v in r.equity], equity_dates=r.equity_dates,
            trades=r.trades, benchmark_equity=r.benchmark_equity, benchmark_name=r.benchmark_name,
            metrics=r.metrics,
        )
        variants.append(scaled)
    summary = summarize_distribution(variants)
    assert "total_return" in summary
    assert summary["total_return"]["n"] == 3
    assert summary["total_return"]["std"] >= 0.0


@pytest.mark.unit
def test_multi_run_report_table_shape(buy_result: BacktestResult):
    variants = [buy_result, buy_result]
    md = render_multi_run_report(variants)
    assert "Multi-run distribution" in md
    assert "| mean | std |" in md
    assert "Deflated Sharpe" in md


@pytest.mark.unit
def test_write_report_to_disk(buy_result: BacktestResult, tmp_path):
    path = write_report(buy_result, results_dir=tmp_path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Backtest: AAPL" in text


@pytest.mark.unit
def test_write_multi_run_report(buy_result: BacktestResult, tmp_path):
    path = write_report([buy_result, buy_result], results_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Multi-run distribution" in text
    # Each individual run also rendered.
    assert text.count("# Backtest: AAPL") == 2


@pytest.mark.unit
def test_multi_run_helper_runs_factory_n_times():
    calls = []

    def factory(i):
        calls.append(i)
        return buy_result

    out = multi_run(factory, 3)
    assert len(out) == 3
    assert calls == [0, 1, 2]


@pytest.mark.unit
def test_summarize_distribution_empty():
    assert summarize_distribution([]) == {}


@pytest.mark.unit
def test_none_metrics_handled():
    """A result with metrics=None must not crash rendering."""
    r = BacktestResult(
        ticker="X", initial_capital=100_000, holding_days=5,
        equity=[100_000, 101_000], equity_dates=["2024-01-01", "2024-01-02"],
        trades=[TradeRow(date="2024-01-01", rating="Buy", target_weight=1.0,
                         executed_weight=1.0, price=100.0)],
        benchmark_equity=[100_000, 101_000], benchmark_name="X buy-and-hold",
        metrics=None,
    )
    md = render_backtest_report(r)
    assert "n/a" in md  # metrics gracefully absent
