"""End-to-end integration test tying Phase 0 + 1 + 3 + 4 together.

Demonstrates the full A/B pipeline the roadmap is built around:
  baseline backtest (LLM sizing) vs improved backtest (Phase-1 risk sizing),
  fed through the Phase-3 validation gate and rendered into the Phase-4
  dashboard + Phase-0 report -- all with a mock graph and synthetic prices, so
  it runs hermetically without an LLM or network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from yiagents.backtest.engine import run_backtest
from yiagents.backtest.report import write_report
from yiagents.backtest.validation_gate import evaluate_gate
from yiagents.monitoring.dashboard import write_dashboard
from yiagents.risk.manager import RiskManager, build_backtest_weight_fn


class FakeGraph:
    """Scripted ratings: a calm Buy-and-occasional-Hold tape."""

    def __init__(self, ratings):
        self._ratings = dict(ratings)

    def propagate(self, company_name, trade_date, asset_type="stock"):
        r = self._ratings.get(trade_date, "Hold")
        return {"final_trade_decision": f"**Rating**: {r}"}, r

    def _resolve_benchmark(self, t):
        return "SPY"


def _volatile_prices(ticker, start, end):
    """Rising-but-noisy tape so risk sizing can differ from full-Buy."""
    idx = pd.bdate_range(start, end)
    import math
    vals = [100.0 * (1 + 0.003 * i + 0.01 * math.sin(i)) for i in range(len(idx))]
    return pd.Series(vals, index=idx.strftime("%Y-%m-%d"), dtype=float)


def _dates(n=14, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n * 6, freq="B")
    return [idx[i].strftime("%Y-%m-%d") for i in range(0, n * 6, 5)][:n]


@pytest.mark.unit
def test_end_to_end_ab_gate_dashboard(tmp_path):
    dates = _dates(14)
    # Mostly Buy with a couple Holds -> both variants trade, sizing differs.
    ratings = {d: "Buy" for d in dates}
    ratings[dates[3]] = "Hold"
    ratings[dates[8]] = "Hold"
    graph = FakeGraph(ratings)
    pp = _volatile_prices

    # --- Baseline: simple rating->weight (Phase 0 default, no risk layer) ---
    baseline = run_backtest(graph, "AAPL", dates, holding_days=5, price_provider=pp)

    # --- Improved: Phase-1 risk sizing via the backtest weight_fn ---
    rm = RiskManager(max_single_position=0.20)
    wfn = build_backtest_weight_fn(rm, "AAPL")
    graph2 = FakeGraph(ratings)
    improved = run_backtest(graph2, "AAPL", dates, holding_days=5,
                            price_provider=pp, weight_fn=wfn)

    # Both variants ran and produced metrics.
    assert baseline.metrics is not None and improved.metrics is not None
    # Risk variant obeyed the 20% single-position cap on every rebalance.
    assert all(t.executed_weight <= 0.20 + 1e-9 for t in improved.trades)

    # --- Phase 3 gate: decide whether the edge is real (multiple runs) ---
    verdict = evaluate_gate(baseline, [improved])
    assert verdict.n_runs == 1
    assert isinstance(verdict.passes, bool)
    assert "margin_vs_baseline" in verdict.__dict__
    rendered = verdict.render()
    assert "Validation Gate" in rendered

    # --- Phase 4 dashboard + Phase 0 report both render to disk ---
    dash_path = write_dashboard([baseline, improved], results_dir=tmp_path,
                                kill_switch=False)
    assert dash_path.exists()
    assert "<svg" in dash_path.read_text(encoding="utf-8")

    report_path = write_report([baseline, improved], results_dir=tmp_path)
    assert report_path.exists()
    assert "Backtest: AAPL" in report_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_kill_switch_config_env_override(monkeypatch):
    """The kill switch must be settable via the documented env var."""
    from yiagents.default_config import _ENV_OVERRIDES, _apply_env_overrides
    # The override mapping is registered.
    assert _ENV_OVERRIDES["YIAGENTS_KILL_SWITCH"] == "kill_switch"

    # _apply_env_overrides coerces the env value into the config dict.
    monkeypatch.setenv("YIAGENTS_KILL_SWITCH", "true")
    cfg = _apply_env_overrides({"kill_switch": False})
    assert cfg["kill_switch"] is True

    # And the browser broker reads the same env var live at order time.
    from yiagents.execution.browser_broker import KillSwitch
    assert KillSwitch.is_halted() is True
