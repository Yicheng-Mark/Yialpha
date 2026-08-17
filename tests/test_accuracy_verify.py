"""Tests for the rating↔outcome verification loop (yiagents/accuracy).

Covers the history scan, asset routing, forward-outcome scoring, aggregate
report + markdown, the artifact write, the extracted memory-resolution core,
and the /api/accuracy artifact-serving contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from web import runner, store
from web.app import app as web_app
from yiagents import accuracy
from yiagents.accuracy import DecisionRecord


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(runner, "registry", runner.TaskRegistry())
    with TestClient(web_app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Fixtures: a small on-disk history
# --------------------------------------------------------------------------- #
def _write_log(
    root: Path, ticker: str, date: str, *, rating_md: str = "", pm_rating: str = "",
    asset_type: str | None = None, price: float | None = None,
    basis: str | None = None,
) -> None:
    d = root / ticker / "YiAgentsStrategy_logs"
    d.mkdir(parents=True, exist_ok=True)
    entry = {
        "company_of_interest": ticker,
        "trade_date": date,
        "final_trade_decision": rating_md or f"**Rating**: {pm_rating or 'Hold'} …",
        "pm_rating": pm_rating,
    }
    if asset_type is not None:
        entry["asset_type"] = asset_type
    if price is not None:
        entry["price_at_decision"] = price
        entry["price_at_decision_basis"] = basis or "risk_overlay_close"
    (d / f"full_states_log_{date}.json").write_text(
        json.dumps(entry), encoding="utf-8"
    )


@pytest.fixture()
def history(tmp_path):
    _write_log(tmp_path, "AAPL", "2026-01-05", pm_rating="Rating: BUY",
               asset_type="stock", price=100.0)
    _write_log(tmp_path, "AAPL", "2026-01-12", pm_rating="Rating: Sell",
               asset_type="stock")
    _write_log_log_legacy(tmp_path)
    _write_log(tmp_path, "BTCUSDT", "2026-01-05", pm_rating="Rating: Hold",
               asset_type="crypto_perp", price=42000.0)
    # Unreadable file must be skipped, not fatal.
    bad = tmp_path / "ZZZZ" / "YiAgentsStrategy_logs"
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "full_states_log_2026-01-05.json").write_text("{not json", encoding="utf-8")
    return tmp_path


def _write_log_log_legacy(root: Path) -> None:
    """A pre-T0-5 log: no asset_type, no price, no pm_rating (markdown only)."""
    _write_log(root, "MSFT", "2026-01-05", rating_md="**Rating**: Buy\noverlay …")


# --------------------------------------------------------------------------- #
# Scan + asset routing
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_scan_extracts_records(history):
    records = accuracy.scan_history(history)
    by_key = {(r.ticker, r.trade_date): r for r in records}

    assert set(by_key) == {
        ("AAPL", "2026-01-05"), ("AAPL", "2026-01-12"),
        ("MSFT", "2026-01-05"), ("BTCUSDT", "2026-01-05"),
    }  # unreadable ZZZZ skipped

    a = by_key[("AAPL", "2026-01-05")]
    assert a.rating == "Buy"
    assert a.price_at_decision == 100.0
    assert a.price_basis == "risk_overlay_close"
    assert a.asset_type == "stock" and a.asset_source == "logged"

    # Legacy markdown-only rating still parses.
    assert by_key[("MSFT", "2026-01-05")].rating == "Buy"
    assert by_key[("MSFT", "2026-01-05")].price_basis == "legacy"


@pytest.mark.unit
def test_asset_type_routing_rules():
    assert accuracy._resolve_asset_type("BTCUSDT", "crypto_perp") == ("crypto_perp", "logged")
    assert accuracy._resolve_asset_type("X", "crypto") == ("crypto_perp", "logged")
    assert accuracy._resolve_asset_type("AAPL", "stock") == ("stock", "logged")
    # Legacy log (no asset_type): USDT suffix inferred as perp, and marked.
    assert accuracy._resolve_asset_type("BTCUSDT", None) == ("crypto_perp", "inferred")
    assert accuracy._resolve_asset_type("AAPL", None) == ("stock", "inferred")


# --------------------------------------------------------------------------- #
# Forward outcome + aggregation
# --------------------------------------------------------------------------- #
def _series(closes):
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    s = pd.Series(closes, index=idx, dtype=float)
    s.index = s.index.date
    return s


@pytest.mark.unit
def test_forward_outcome_scoring(monkeypatch):
    up = DecisionRecord("A", "2026-01-05", "Buy", "stock", None, "legacy", "inferred", "p")
    monkeypatch.setattr(accuracy, "_dated_close", lambda *a, **k: _series([100, 110, 120, 130, 140, 150]))
    out = accuracy.forward_outcome(up, holding_days=5)
    assert out["raw_return"] == pytest.approx(0.5)
    assert out["baseline_close"] == 100.0

    # Not enough elapsed sessions → pending (None), never a partial score.
    short = DecisionRecord("A", "2026-01-05", "Buy", "stock", None, "legacy", "inferred", "p")
    monkeypatch.setattr(accuracy, "_dated_close", lambda *a, **k: _series([100, 105, 106]))
    assert accuracy.forward_outcome(short, holding_days=5) is None


@pytest.mark.unit
def test_report_aggregates_direction_and_hold(monkeypatch):
    # Buy +10% (hit), Buy -5% (miss), Sell -8% (hit), Sell +4% (miss),
    # Hold +2% (neutral — excluded from direction, reported separately).
    records = [
        DecisionRecord("A", "2026-01-05", "Buy", "stock", None, "legacy", "inferred", "p"),
        DecisionRecord("A", "2026-01-06", "Buy", "stock", None, "legacy", "inferred", "p"),
        DecisionRecord("B", "2026-01-05", "Sell", "stock", None, "legacy", "inferred", "p"),
        DecisionRecord("B", "2026-01-06", "Sell", "stock", None, "legacy", "inferred", "p"),
        DecisionRecord("C", "2026-01-05", "Hold", "stock", None, "legacy", "inferred", "p"),
    ]
    returns = iter([0.10, -0.05, -0.08, 0.04, 0.02])
    monkeypatch.setattr(
        accuracy, "forward_outcome",
        lambda record, holding_days=5: {"raw_return": next(returns)},
    )
    report = accuracy.build_accuracy_report(records, holding_days=5)

    assert report["direction"] == {"n": 4, "hits": 2, "hit_rate": 0.5}
    assert report["hold_n"] == 1
    assert report["hold_mean_return"] == pytest.approx(0.02)
    assert report["by_rating"]["Buy"] == {
        "n": 2, "mean_return": pytest.approx(0.025), "directional_n": 2,
        "hits": 1, "hit_rate": 0.5,
    }
    assert report["by_ticker"]["C"]["directional_n"] == 0

    md = accuracy.render_markdown(report)
    assert "2/4 correct (50.0%)" in md
    assert "| Buy | 2 |" in md  # sample sizes ship with every table row


@pytest.mark.unit
def test_verify_history_writes_artifacts(history, monkeypatch):
    monkeypatch.setattr(
        accuracy, "_dated_close",
        lambda ticker, asset_type, start, end: _series(
            [100, 102, 104, 106, 108, 110]
        ),
    )
    report, json_path, md_path = accuracy.verify_history(history, holding_days=5)

    assert json_path.is_file() and md_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["total_runs"] == 4
    assert (json_path.parent / "accuracy_report.md").name == md_path.name
    # Every scored record carries a forward_return; markdown notes the
    # inferred legacy record so a misrouted asset source stays visible.
    assert "Legacy-log note" in md_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# fetch_returns_yf (memory twin of the graph's _fetch_returns)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_fetch_returns_yf_cutoff_and_math(monkeypatch):
    import yiagents.accuracy as acc

    closes = pd.DataFrame(
        {"Close": [100.0, 110.0, 121.0]},
        index=pd.bdate_range("2026-01-05", periods=3),
    )
    bench = pd.DataFrame(
        {"Close": [200.0, 202.0, 204.0]},
        index=pd.bdate_range("2026-01-05", periods=3),
    )

    def fake_history(symbol, start, end):
        return bench if symbol == "SPY" else closes

    monkeypatch.setattr(
        "yiagents.dataflows.y_finance.get_YFin_history_cached", fake_history
    )
    monkeypatch.setattr(
        "yiagents.dataflows.symbol_utils.normalize_symbol", lambda s: s
    )

    raw, alpha, days = acc.fetch_returns_yf("AAPL", "2026-01-05", "SPY", holding_days=2)
    assert raw == pytest.approx(0.21)
    assert alpha == pytest.approx(0.21 - 0.02)
    assert days == 2

    # as_of cutoff that leaves a partial horizon (baseline + only 1 of the 2
    # required sessions) → (None, None, None): a reflection labelled a
    # five-session outcome must not come from a partial horizon.
    assert acc.fetch_returns_yf(
        "AAPL", "2026-01-05", "SPY", holding_days=2, as_of_date="2026-01-06"
    ) == (None, None, None)


# --------------------------------------------------------------------------- #
# memory_resolution core (shared by graph + CLI)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_resolve_pending_entries(monkeypatch, caplog):
    from yiagents.graph import memory_resolution as mr

    class FakeMemoryLog:
        def get_pending_entries(self):
            return [
                {"ticker": "AAPL", "date": "2026-01-05", "decision": "d1"},
                {"ticker": "NVDA", "date": "2026-01-05", "decision": "d2"},
                {"ticker": "AAPL", "date": "2999-01-01", "decision": "future"},
            ]

        updates = None

        def batch_update_with_outcomes(self, updates):
            self.updates = updates

    class FakeReflector:
        def reflect_on_final_decision(self, final_decision, raw_return,
                                      alpha_return, benchmark_name):
            return f"reflected {final_decision} {raw_return:+.2%}"

    monkeypatch.setattr(
        accuracy, "fetch_returns_yf", lambda *a, **k: (0.05, 0.01, 5),
    )
    log = FakeMemoryLog()
    resolved = mr.resolve_pending_entries(log, FakeReflector(), "AAPL", as_of_date="2026-01-20")
    assert resolved == 1  # only the same-ticker, precedes-cutoff entry
    assert log.updates[0]["ticker"] == "AAPL"
    assert log.updates[0]["reflection"].startswith("reflected d1")

    # Unresolvable + old → stale WARNING (observable, not silent).
    monkeypatch.setattr(accuracy, "fetch_returns_yf", lambda *a, **k: (None, None, None))
    log2 = FakeMemoryLog()
    with caplog.at_level("WARNING"):
        mr.resolve_pending_entries(log2, FakeReflector(), "AAPL", as_of_date="2026-03-01")
    assert any("still unresolved" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# /api/accuracy serves the artifact (never fabricates)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_api_accuracy_serves_artifact(client, monkeypatch, tmp_path):
    report = {
        "generated_at": "2026-08-16T00:00:00", "holding_days": 5,
        "total_runs": 1, "scored": 1, "pending": 0,
        "direction": {"n": 1, "hits": 1, "hit_rate": 1.0},
        "hold_mean_return": None, "hold_n": 0, "by_rating": {}, "by_ticker": {},
        "records": [],
    }
    acc_dir = tmp_path / "accuracy"
    acc_dir.mkdir()
    (acc_dir / "accuracy_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    monkeypatch.setattr(store, "LOGS_ROOT", tmp_path)

    resp = client.get("/api/accuracy")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["direction"]["hit_rate"] == 1.0


@pytest.mark.unit
def test_api_accuracy_unavailable_is_honest(client, monkeypatch, tmp_path):
    monkeypatch.setattr(store, "LOGS_ROOT", tmp_path)  # no artifact on disk
    body = client.get("/api/accuracy").json()
    assert body["available"] is False
    assert body["hint"] == "yiagents verify-history"
