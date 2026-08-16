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
    g = FakeGraph(dict.fromkeys(dates, "Buy"))
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


# ---------------------------------------------------------------------------
# Round-5 audit (2026-08-16): perp-simulation observability + honest display
# ---------------------------------------------------------------------------


def _rising_subcent(ticker, start, end):
    idx = pd.bdate_range(start, end)
    vals = [1.23456e-5 * (1 + 0.002 * i) for i in range(len(idx))]
    return pd.Series(vals, index=idx.strftime("%Y-%m-%d"), dtype=float)


def _perp_result(**config_extra) -> BacktestResult:
    """A real backtest run with the engine's perp config fields overlaid."""
    dates = _dates(8)
    g = FakeGraph(dict.fromkeys(dates, "Buy"))
    r = run_backtest(
        g, "PEPEUSDT", dates, holding_days=5, price_provider=_rising_subcent
    )
    r.config_summary.update({
        "asset_type": "crypto_perp",
        "cost_bps": 0.0,
        "periods_per_year": 365,
        "perp_fees": "taker 4.5bps x0.9 BNB, slippage 2bps",
        "perp_funding_paid_total": -12.34,
        "perp_leverage": 3,
        "perp_short": True,
        "perp_fill_quantization": "stepSize/minNotional",
        "perp_model_note": "USDT-M perp simulation: daily funding drag",
        "perp_liquidations": [
            {
                "date": "2026-08-01", "side": "long", "shares": 2.0,
                "entry": 0.0000123456, "liquidation_price": 0.0000041,
                "exit_price": 0.0000041, "fee": 3.2,
            }
        ],
        **config_extra,
    })
    return r


@pytest.mark.unit
def test_perp_section_renders_funding_liquidations_and_fees():
    md = render_backtest_report(_perp_result())
    assert "## Perp simulation" in md
    # Effective fees the engine actually charged, not the raw cost_bps=0.
    assert "taker 4.5bps x0.9 BNB" in md
    assert "0.0 bps" not in md
    # Signed funding: negative = net received (short-biased run).
    assert "-12.34" in md and "net received" in md
    # Liquidation visibility: count + entry/liquidation prices (adaptive
    # precision — the raw 4.1e-6 must not collapse to "0.00").
    assert "Liquidation events: 1" in md
    assert "4.1e-06" in md
    # Leverage / side / model note.
    assert "3x" in md and "long + short" in md
    assert "USDT-M perp simulation" in md


@pytest.mark.unit
def test_non_perp_report_has_no_perp_section():
    r = BacktestResult(
        ticker="AAPL", initial_capital=100_000, holding_days=5,
        equity=[100_000, 101_000], equity_dates=["2024-01-01", "2024-01-02"],
        trades=[], benchmark_equity=[100_000, 101_000],
        benchmark_name="AAPL buy-and-hold", metrics=None,
        config_summary={"cost_bps": 5.0, "periods_per_year": 252},
    )
    md = render_backtest_report(r)
    assert "Perp simulation" not in md
    assert "5.0 bps" in md  # plain runs still show the raw parameter
    assert "252/yr" in md


@pytest.mark.unit
def test_sub_cent_perp_prices_render_nonzero():
    """Prices ~1e-5 must not collapse to '0.00' in the trades table."""
    md = render_backtest_report(_perp_result())
    assert "## Trades" in md
    assert "0.00" not in md.split("## Trades")[1].split("\n\n")[0]
    assert "1.235e-05" in md or "0.00001235" in md or "1.2346e-05" in md


@pytest.mark.unit
def test_multi_run_table_shares_n_column():
    """Partial-coverage metrics show their run count, not a fake distribution."""
    variants = []
    for scale in (0.9, 1.0, 1.1):
        r = _perp_result()
        variants.append(
            BacktestResult(
                ticker=r.ticker, initial_capital=r.initial_capital,
                holding_days=r.holding_days,
                equity=[v * scale for v in r.equity],
                equity_dates=r.equity_dates, trades=r.trades,
                benchmark_equity=r.benchmark_equity,
                benchmark_name=r.benchmark_name, metrics=r.metrics,
                config_summary=r.config_summary,
            )
        )
    md = render_multi_run_report(variants)
    assert "| n |" in md
    # The header row plus at least one metric row carrying n=3.
    assert "| 3 |" in md


@pytest.mark.unit
def test_fmt_price_adaptive_precision():
    from yiagents.backtest.report import _fmt_price

    assert _fmt_price(123.456) == "123.46"
    assert _fmt_price(0.0000123456) == "1.235e-05"
    assert _fmt_price(None) == "n/a"
    assert float(_fmt_price(0.5)) == 0.5
