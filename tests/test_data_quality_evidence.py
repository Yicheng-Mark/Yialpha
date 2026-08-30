"""Tests for the run-level data-quality evidence (dataflows/quality).

Covers the recorder primitives and the router integration: every sentinel a
vendor chain degrades to must land in the context's event list, and the
``_log_state`` write must carry ``pm_rating`` + a ``data_quality`` block.
"""

from __future__ import annotations

import json

import pytest

from yialpha.dataflows import quality
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.dataflows.interface import route_to_vendor


@pytest.fixture(autouse=True)
def _clean_quality():
    quality.reset_quality()
    yield
    quality.reset_quality()


# --------------------------------------------------------------------------- #
# Recorder primitives
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_record_snapshot_reset_roundtrip():
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "no rows")
    quality.record_sentinel("get_macro_data", quality.KIND_OPTIONAL_UNAVAILABLE, "net")
    events = quality.snapshot_quality()
    assert [e["method"] for e in events] == ["get_stock_data", "get_macro_data"]
    summary = quality.summarize_quality(events)
    assert summary["core_sentinel_count"] == 1
    assert summary["optional_sentinel_count"] == 1
    assert summary["sentinels"] == events

    quality.reset_quality()
    assert quality.snapshot_quality() == []
    assert quality.snapshot_core_successes() == set()
    assert quality.summarize_quality(None) == {
        "sentinels": [], "core_sentinel_count": 0, "core_error_count": 0,
        "core_ok_count": 0, "optional_sentinel_count": 0,
        "stale_cache_count": 0, "degraded_count": 0, "data_vacuum": False,
    }


@pytest.mark.unit
def test_record_never_raises():
    # A recorder failure must not be able to break the data call it describes.
    quality.record_sentinel(None, None, None)  # type: ignore[arg-type]
    events = quality.snapshot_quality()
    assert events[0]["method"] == "None"


@pytest.mark.unit
def test_record_success_never_raises():
    quality.record_success(None)  # type: ignore[arg-type]
    assert "None" in quality.snapshot_core_successes()


@pytest.mark.unit
def test_core_error_kind_counts_as_core():
    """A router hard error on a core category is a core degradation — the
    gate reads core_sentinel_count, so KIND_CORE_ERROR must be folded in
    or a total-outage run would look clean."""
    quality.record_sentinel("get_stock_data", quality.KIND_CORE_ERROR, "net down")
    summary = quality.summarize_quality(quality.snapshot_quality())
    assert summary["core_sentinel_count"] == 1
    assert summary["core_error_count"] == 1


@pytest.mark.unit
def test_degraded_count_folds_in_stale_cache():
    quality.record_sentinel("get_stock_data", quality.KIND_STALE_CACHE, "stale served")
    summary = quality.summarize_quality(quality.snapshot_quality())
    assert summary["core_sentinel_count"] == 0  # stale is NOT a vacuum signal
    assert summary["stale_cache_count"] == 1
    assert summary["degraded_count"] == 1


@pytest.mark.unit
def test_is_data_vacuum_verdicts():
    # Nothing attempted -> no verdict (the gate must not reject what it can't judge).
    assert quality.is_data_vacuum([]) is False
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "x")
    events = quality.snapshot_quality()
    assert quality.is_data_vacuum(events) is True  # attempted, none succeeded
    assert quality.is_data_vacuum(events, {"get_stock_data"}) is False


@pytest.mark.unit
def test_summarize_data_vacuum_flag():
    quality.record_sentinel("get_news", quality.KIND_NO_DATA, "x")
    assert quality.summarize_quality(
        quality.snapshot_quality(), set()
    )["data_vacuum"] is True
    assert quality.summarize_quality(
        quality.snapshot_quality(), {"get_stock_data"}
    )["data_vacuum"] is False


# --------------------------------------------------------------------------- #
# Data-vacuum gate (policy enforcement)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def _policy_ctx():
    from yialpha.dataflows import config as dfconfig

    dfconfig.reset_config()
    yield dfconfig
    dfconfig.reset_config()


@pytest.mark.unit
def test_check_data_vacuum_rejects(_policy_ctx):
    _policy_ctx.set_config({"data_vacuum_policy": "reject"})
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "all vendors down")
    with pytest.raises(quality.DataVacuumError, match="data vacuum"):
        quality.check_data_vacuum()


@pytest.mark.unit
def test_check_data_vacuum_warns(_policy_ctx, caplog):
    _policy_ctx.set_config({"data_vacuum_policy": "warn"})
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "all vendors down")
    with caplog.at_level("WARNING"):
        quality.check_data_vacuum()  # must not raise
    assert any("DATA VACUUM" in r.message for r in caplog.records)


@pytest.mark.unit
def test_check_data_vacuum_passes_when_data_ok(_policy_ctx):
    _policy_ctx.set_config({"data_vacuum_policy": "reject"})
    quality.record_sentinel("get_news", quality.KIND_NO_DATA, "news only")
    quality.record_success("get_stock_data")
    quality.check_data_vacuum()  # not a vacuum: market data succeeded


@pytest.mark.unit
def test_invalid_policy_fails_closed(_policy_ctx, caplog):
    _policy_ctx.set_config({"data_vacuum_policy": "lenient-typo"})
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "x")
    with caplog.at_level("WARNING"), pytest.raises(quality.DataVacuumError):
        quality.check_data_vacuum()  # unknown value -> reject, never warn


@pytest.mark.unit
def test_gate_wrapper_passthrough():
    calls = []

    def handler(state):
        calls.append(state)
        return {"ok": True}

    gated = quality.gate_on_data_vacuum(handler)
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "x")
    quality.record_success("get_stock_data")
    assert gated({"s": 1}) == {"ok": True}
    assert calls == [{"s": 1}]


# --------------------------------------------------------------------------- #
# Router integration
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_no_data_sentinel_is_recorded(monkeypatch):
    from yialpha.dataflows import interface as iface

    def no_data(*a, **k):
        raise NoMarketDataError("ZZZZ", "ZZZZ", "no rows anywhere")

    monkeypatch.setattr(
        iface, "get_vendor", lambda category, method: "default"
    )
    # Route a core category whose every vendor fails with NoMarketDataError:
    # monkeypatch the whole chain via VENDOR_METHODS.
    method = "get_stock_data"
    saved = dict(iface.VENDOR_METHODS[method])
    try:
        iface.VENDOR_METHODS[method] = dict.fromkeys(saved, no_data)
        out = route_to_vendor(method, "ZZZZ", "2026-01-01", "2026-01-31")
    finally:
        iface.VENDOR_METHODS[method] = saved

    assert out.startswith("NO_DATA_AVAILABLE")
    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["method"] == method
    assert events[0]["kind"] == quality.KIND_NO_DATA
    assert "no rows anywhere" in events[0]["detail"]


@pytest.mark.unit
def test_core_hard_error_records_sentinel_then_raises(monkeypatch):
    """The router's core-category raise path must leave evidence BEFORE
    raising — a caller that swallows the error into a degraded report
    otherwise produces zero data AND zero proof there was no data."""
    from yialpha.dataflows import interface as iface

    def boom(*a, **k):
        raise RuntimeError("network unreachable")

    method = "get_stock_data"
    monkeypatch.setattr(iface, "get_vendor", lambda category, m: "default")
    saved = dict(iface.VENDOR_METHODS[method])
    try:
        iface.VENDOR_METHODS[method] = dict.fromkeys(saved, boom)
        with pytest.raises(RuntimeError, match="network unreachable"):
            route_to_vendor(method, "AAPL", "2026-01-01", "2026-01-31")
    finally:
        iface.VENDOR_METHODS[method] = saved

    events = quality.snapshot_quality()
    assert [e["kind"] for e in events] == [quality.KIND_CORE_ERROR]
    assert quality.summarize_quality(events)["core_sentinel_count"] == 1


@pytest.mark.unit
def test_core_success_records_success(monkeypatch):
    from yialpha.dataflows import interface as iface

    method = "get_stock_data"
    monkeypatch.setattr(iface, "get_vendor", lambda category, m: "default")
    saved = dict(iface.VENDOR_METHODS[method])
    try:
        iface.VENDOR_METHODS[method] = {"yfinance": lambda *a, **k: "rows"}
        out = route_to_vendor(method, "AAPL", "2026-01-01", "2026-01-31")
    finally:
        iface.VENDOR_METHODS[method] = saved

    assert out == "rows"
    assert quality.snapshot_core_successes() == {method}
    assert quality.snapshot_quality() == []  # no sentinels on a clean run


@pytest.mark.unit
def test_optional_success_records_nothing(monkeypatch):
    from yialpha.dataflows import interface as iface

    method = "get_macro_indicators"  # macro_data is optional
    monkeypatch.setattr(iface, "get_vendor", lambda category, m: "default")
    saved = dict(iface.VENDOR_METHODS[method])
    try:
        iface.VENDOR_METHODS[method] = {"fred": lambda *a, **k: "macro"}
        route_to_vendor(method, "US", "2026-01-01", "2026-01-31")
    finally:
        iface.VENDOR_METHODS[method] = saved

    # Optional successes do not participate in the vacuum verdict.
    assert quality.snapshot_core_successes() == set()


@pytest.mark.unit
def test_optional_no_data_sentinel_counts_as_optional(monkeypatch):
    """A US-only optional tool raising NoMarketDataError by design (e.g. Form 4
    for a non-US ticker) must be evidence of *optional* unavailability — not a
    core-data degradation that would flag every non-US run as DEGRADED."""
    from yialpha.dataflows import interface as iface

    def no_data(*a, **k):
        raise NoMarketDataError("0700.HK", "0700.HK", "US-listed only")

    method = "get_form4_insider_trading"
    monkeypatch.setattr(
        iface, "get_vendor", lambda category, m: "default"
    )
    saved = dict(iface.VENDOR_METHODS[method])
    try:
        iface.VENDOR_METHODS[method] = dict.fromkeys(saved, no_data)
        out = route_to_vendor(method, "0700.HK", "2026-01-01", "2026-01-31")
    finally:
        iface.VENDOR_METHODS[method] = saved

    assert out.startswith("NO_DATA_AVAILABLE")
    summary = quality.summarize_quality(quality.snapshot_quality())
    assert summary["core_sentinel_count"] == 0
    assert summary["optional_sentinel_count"] == 1


# --------------------------------------------------------------------------- #
# _log_state integration (graph)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_log_state_writes_pm_rating_and_data_quality(tmp_path, monkeypatch):
    from yialpha.graph.trading_graph import YiAlphaGraph

    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "stale")
    quality.record_sentinel("get_macro_data", quality.KIND_OPTIONAL_UNAVAILABLE, "x")

    graph = YiAlphaGraph.__new__(YiAlphaGraph)  # skip __init__: only attrs below used
    graph.ticker = "NVDA"
    graph.log_states_dict = {}
    monkeypatch.setattr(
        graph, "propagate", None, raising=False
    )
    # Price-at-decision fallback (overlay did not run in this fixture): the
    # memoized loader must not hit the network from a unit test.
    monkeypatch.setattr(
        graph, "_latest_close_and_atr", lambda *a, **k: (123.45, 2.5)
    )

    debate = {"bull_history": "", "bear_history": "", "history": "",
              "current_response": "", "judge_decision": ""}
    risk_debate = dict.fromkeys(("aggressive_history", "conservative_history", "neutral_history", "history", "judge_decision"), "")
    final_state = {
        "company_of_interest": "NVDA",
        "trade_date": "2026-06-10",
        "market_report": "m", "sentiment_report": "s", "news_report": "n",
        "fundamentals_report": "f",
        "investment_debate_state": debate,
        "trader_investment_plan": "t",
        "risk_debate_state": risk_debate,
        "investment_plan": "i",
        "final_trade_decision": "d",
        "pm_rating": "Rating: BUY",
    }


    results_dir = tmp_path / "results"
    object.__setattr__(graph, "config", {"results_dir": str(results_dir)})
    returned = graph._log_state("2026-06-10", final_state)

    log_path = (
        results_dir / "NVDA" / "YiAlphaStrategy_logs" / "full_states_log_2026-06-10.json"
    )
    assert log_path.exists()
    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert data["pm_rating"] == "Rating: BUY"
    assert data["data_quality"]["core_sentinel_count"] == 1
    assert data["data_quality"]["optional_sentinel_count"] == 1
    assert data["data_quality"]["sentinels"][0]["method"] == "get_stock_data"
    # Price-at-decision fallback: overlay keys absent -> loader fallback wins.
    assert data["price_at_decision"] == 123.45
    assert data["price_at_decision_basis"] == "fallback_loader"
    assert data["asset_type"] == "stock"

    # Overlay-provided prices win over the fallback.
    final_state["price_at_decision"] = 130.0
    final_state["price_at_decision_basis"] = "risk_overlay_close"
    graph._log_state("2026-06-10", final_state)
    data2 = json.loads(log_path.read_text(encoding="utf-8"))
    assert data2["price_at_decision"] == 130.0
    assert data2["price_at_decision_basis"] == "risk_overlay_close"

    # Consuming the snapshot resets the accumulator for the next run.
    assert quality.snapshot_quality() == []

    # The consumed block is returned so _run_graph can attach it to
    # final_state for the report writer / web UI degraded-run banner.
    assert returned["core_sentinel_count"] == 1
    assert returned["optional_sentinel_count"] == 1
