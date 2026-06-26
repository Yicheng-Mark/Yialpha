"""Unit tests for the Phase-4 monitoring dashboard generator."""

from __future__ import annotations

import pytest

from tradingagents.backtest.engine import BacktestResult, TradeRow
from tradingagents.backtest.metrics import BacktestMetrics
from tradingagents.monitoring.dashboard import render_dashboard, write_dashboard


def _result(ticker="AAPL", total_return=0.15, mdd=-0.08, dsr=0.6) -> BacktestResult:
    metrics = BacktestMetrics(
        total_return=total_return, cagr=total_return, volatility=0.15,
        sharpe=1.2, sortino=1.6, max_drawdown=mdd, calmar=abs(total_return / mdd),
        deflated_sharpe=dsr, alpha_vs_buyhold=0.05, num_periods=252, periods_per_year=252,
    )
    return BacktestResult(
        ticker=ticker, initial_capital=100_000, holding_days=5,
        equity=[100_000, 105_000, 110_000, 115_000],
        equity_dates=["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"],
        trades=[TradeRow(date="2024-01-01", rating="Buy", target_weight=1.0,
                         executed_weight=1.0, price=100.0)],
        benchmark_equity=[100_000, 105_000], benchmark_name="AAPL buy-and-hold",
        metrics=metrics,
    )


@pytest.mark.unit
def test_dashboard_renders_equity_and_metrics():
    md = render_dashboard(_result())
    assert "<svg" in md
    assert "polyline" in md
    assert "Sharpe" in md
    assert "Max drawdown" in md
    assert "AAPL equity curve" in md


@pytest.mark.unit
def test_kill_switch_banner():
    md = render_dashboard(_result(), kill_switch=True)
    assert "KILL SWITCH ENGAGED" in md


@pytest.mark.unit
def test_no_kill_banner_by_default():
    md = render_dashboard(_result())
    assert "KILL SWITCH" not in md


@pytest.mark.unit
def test_drawdown_alert_when_deep():
    deep = _result(mdd=-0.20)
    md = render_dashboard(deep)
    assert "Drawdown alert" in md


@pytest.mark.unit
def test_no_alert_when_shallow():
    md = render_dashboard(_result(mdd=-0.05))
    assert "Drawdown alert" not in md


@pytest.mark.unit
def test_positions_table_from_dict():
    md = render_dashboard(_result(), portfolio_state={"positions": {"AAPL": 25000, "MSFT": 0}})
    assert "Current positions" in md
    assert "AAPL" in md
    assert "25,000" in md


@pytest.mark.unit
def test_positions_table_from_object():
    class PS:
        positions = {"AAPL": 25000}
    md = render_dashboard(_result(), portfolio_state=PS())
    assert "Current positions" in md


@pytest.mark.unit
def test_multi_run_table():
    md = render_dashboard([_result("AAPL"), _result("MSFT")])
    assert "Runs" in md
    assert "AAPL" in md and "MSFT" in md


@pytest.mark.unit
def test_write_dashboard_to_disk(tmp_path):
    path = write_dashboard(_result(), results_dir=tmp_path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "<html" in text
    assert "AAPL" in text


@pytest.mark.unit
def test_html_escaping_in_ticker():
    """A malicious ticker string must be escaped, not injected as HTML."""
    bad = _result(ticker="<script>alert(1)</script>")
    md = render_dashboard(bad)
    assert "<script>alert(1)</script>" not in md  # escaped
    assert "&lt;script&gt;" in md
