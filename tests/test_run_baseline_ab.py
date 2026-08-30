"""Regression tests for paired, state-isolated A/B orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.run_baseline as runner
from yialpha.dataflows.config import get_config, reset_config, set_config


@pytest.mark.unit
def test_validation_graph_disables_persistent_memory(monkeypatch):
    captured = {}

    def fake_graph(*args, **kwargs):
        captured.update(kwargs["config"])
        return object()

    monkeypatch.setattr(runner, "YiAlphaGraph", fake_graph)
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "memory_enabled", True)

    runner._build_graph(risk_enabled=False)

    assert captured["memory_enabled"] is False


@pytest.mark.unit
def test_full_ab_pairs_decision_tapes_and_fresh_risk_managers(monkeypatch, tmp_path):
    calls = []
    managers = []
    map_modes = []
    gate_args = []

    monkeypatch.setattr(runner, "_rebalance_dates", lambda *args: ["2024-01-02"])

    def fake_map(tickers, task, workers, risk_enabled=None):
        map_modes.append(risk_enabled)
        return [task(ticker, object()) for ticker in tickers]

    monkeypatch.setattr(runner, "_map_tickers", fake_map)

    def fake_from_config(config):
        manager = object()
        managers.append(manager)
        return manager

    monkeypatch.setattr(runner.RiskManager, "from_config", fake_from_config)
    monkeypatch.setattr(
        runner,
        "build_backtest_weight_fn",
        lambda manager, ticker: (lambda rating, date, ctx: 0.05),
    )

    def fake_backtest(graph, ticker, dates, **kwargs):
        calls.append({"ticker": ticker, **kwargs})
        return SimpleNamespace(
            ticker=ticker,
            metrics=SimpleNamespace(
                total_return=0.10,
                sharpe=1.0,
                max_drawdown=-0.05,
                deflated_sharpe=0.6,
            ),
        )

    monkeypatch.setattr(runner, "run_backtest", fake_backtest)

    verdict = SimpleNamespace(
        passes=True,
        mean_dsr=0.6,
        beats_buyhold=True,
        recommendation="analysis only",
        render=lambda: "# gate",
    )

    def fake_gate(baselines, improved):
        gate_args.append((baselines, improved))
        return verdict

    monkeypatch.setattr(runner, "evaluate_gate", fake_gate)
    monkeypatch.setattr(runner, "write_report", lambda *args, **kwargs: tmp_path / "report.md")
    monkeypatch.setattr(runner, "write_dashboard", lambda *args, **kwargs: None)

    assert runner.full_ab(
        ["AAPL", "MSFT"], "2024-01-01", "2024-02-01", 5, 1, 5, 1.0, 2,
        str(tmp_path), workers=1,
    ) is True

    assert map_modes == [False]  # no graph-level double risk overlay
    assert len(managers) == 4 and len({id(manager) for manager in managers}) == 4
    assert len(calls) == 8
    for baseline_call, improved_call in zip(calls[::2], calls[1::2], strict=True):
        assert baseline_call["run_tag"] == improved_call["run_tag"]
        assert baseline_call["cache"] is improved_call["cache"]
        assert "weight_fn" not in baseline_call
        assert callable(improved_call["weight_fn"])
    assert len(gate_args) == 2
    assert all(len(base) == len(improved) == 2 for base, improved in gate_args)


@pytest.mark.unit
def test_map_tickers_propagates_context_config_to_workers(monkeypatch):
    monkeypatch.setattr(runner, "_build_graph", lambda **kwargs: object())
    set_config({"run_baseline_context_probe": 987654})
    try:
        got = runner._map_tickers(
            ["AAPL", "MSFT"],
            lambda ticker, graph: get_config()["run_baseline_context_probe"],
            workers=2,
            risk_enabled=False,
        )
    finally:
        reset_config()
    assert got == [987654, 987654]
