"""Unit tests for ``web.store.load_run``'s run-record mapping.

Pins the degraded-run evidence contract: the ``data_quality`` block the graph
writes into ``full_states_log_<date>.json`` must reach the web API payload so
the report view can render the degraded-run banner. ``None`` for logs written
before the field existed.
"""

from __future__ import annotations

import json

import pytest

from web import store


@pytest.fixture
def logs_root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LOGS_ROOT", tmp_path)
    return tmp_path


def _write_log(root, date: str, extra_state: dict) -> None:
    d = root / "AAPL" / "YiAgentsStrategy_logs"
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "company_of_interest": "AAPL",
        "trade_date": date,
        "market_report": "m",
        "sentiment_report": "s",
        "news_report": "n",
        "fundamentals_report": "f",
        "investment_debate_state": {},
        "trader_investment_decision": "t",
        "risk_debate_state": {},
        "investment_plan": "i",
        "final_trade_decision": "**Rating**: Buy",
    }
    state.update(extra_state)
    (d / f"full_states_log_{date}.json").write_text(
        json.dumps(state), encoding="utf-8"
    )


@pytest.mark.unit
def test_load_run_passes_data_quality_through(logs_root):
    _write_log(logs_root, "2026-06-10", {"data_quality": {
        "core_sentinel_count": 1,
        "optional_sentinel_count": 2,
        "sentinels": [{"method": "get_news", "kind": "no_data", "detail": "x"}],
    }})
    run = store.load_run("AAPL", "2026-06-10")
    assert run is not None
    assert run["data_quality"]["core_sentinel_count"] == 1
    assert run["data_quality"]["optional_sentinel_count"] == 2
    assert run["data_quality"]["sentinels"][0]["method"] == "get_news"


@pytest.mark.unit
def test_load_run_data_quality_none_when_absent(logs_root):
    _write_log(logs_root, "2026-06-10", {})
    run = store.load_run("AAPL", "2026-06-10")
    assert run is not None
    assert run["data_quality"] is None
