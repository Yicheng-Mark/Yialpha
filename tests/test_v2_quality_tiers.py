"""V2.0 P0.3 — four-tier data-quality classification (critical-data gate).

GOOD / DEGRADED_AUXILIARY / DEGRADED_CRITICAL / INVALID. Only critical
categories (price/indicators/fundamentals/perp-spot books) and vacuums can
escalate; auxiliary outages (news/macro/social) never do.
"""

from __future__ import annotations

import pytest

from yialpha.dataflows import quality
from yialpha.dataflows.quality import (
    TIER_DEGRADED_AUXILIARY,
    TIER_DEGRADED_CRITICAL,
    TIER_GOOD,
    TIER_INVALID,
    classify_quality,
)


def _event(method: str, kind: str) -> dict:
    return {"method": method, "kind": kind, "detail": "test"}


@pytest.mark.unit
def test_clean_run_is_good():
    out = classify_quality([], set())
    assert out["tier"] == TIER_GOOD
    assert out["critical_data_available"] is True
    assert out["critical_missing"] == []
    assert out["auxiliary_degraded"] == []


@pytest.mark.unit
def test_auxiliary_only_degradation_stays_auxiliary():
    # Reddit down, Binance Square down, macro missing: news/macro are not in
    # CRITICAL_CATEGORIES — this must stay DEGRADED_AUXILIARY, never veto.
    # Price DID succeed, so this is degradation, not a vacuum.
    events = [
        _event("get_news", quality.KIND_NO_DATA),
        _event("get_macro_indicators", quality.KIND_OPTIONAL_UNAVAILABLE),
    ]
    out = classify_quality(events, {"get_stock_data"})
    assert out["tier"] == TIER_DEGRADED_AUXILIARY
    assert out["critical_data_available"] is True
    assert out["auxiliary_degraded"] == [
        "get_news(no_data)", "get_macro_indicators(optional_unavailable)",
    ]


@pytest.mark.unit
def test_price_failure_is_critical():
    events = [
        _event("get_stock_data", quality.KIND_NO_DATA),
        _event("get_news", quality.KIND_NO_DATA),
    ]
    out = classify_quality(events, {"get_indicators", "get_fundamentals"})
    assert out["tier"] == TIER_DEGRADED_CRITICAL
    assert out["critical_data_available"] is False
    assert out["critical_missing"] == ["get_stock_data(no_data)"]
    assert "get_news(no_data)" in out["auxiliary_degraded"]


@pytest.mark.unit
def test_core_error_on_critical_category_is_critical():
    # Every vendor ERRORED for a critical method (not a clean no-data).
    out = classify_quality(
        [_event("get_binance_klines", quality.KIND_CORE_ERROR)],
        {"get_stock_data"},
    )
    assert out["tier"] == TIER_DEGRADED_CRITICAL


@pytest.mark.unit
def test_vacuum_is_invalid_even_under_aux_successes():
    # Core attempted, none succeeded: vacuum -> INVALID regardless of the
    # success set being empty (successes for core only count).
    events = [_event("get_stock_data", quality.KIND_NO_DATA)]
    out = classify_quality(events, set())
    assert out["tier"] == TIER_INVALID
    assert out["critical_data_available"] is False


@pytest.mark.unit
def test_stale_cache_is_auxiliary_disclosure():
    out = classify_quality(
        [_event("get_stock_data", quality.KIND_STALE_CACHE)],
        {"get_stock_data"},
    )
    assert out["tier"] == TIER_DEGRADED_AUXILIARY
    assert "get_stock_data(stale_cache)" in out["auxiliary_degraded"]


@pytest.mark.unit
def test_unknown_method_classifies_auxiliary_not_crash():
    # A sentinel recorded for a method the category table doesn't know:
    # fail-open classification, raw name still visible. Core succeeded
    # elsewhere, so this is degradation rather than a vacuum.
    out = classify_quality(
        [_event("get_future_tool", quality.KIND_NO_DATA)], {"get_stock_data"}
    )
    assert out["tier"] == TIER_DEGRADED_AUXILIARY
    assert out["auxiliary_degraded"] == ["get_future_tool(no_data)"]


@pytest.mark.unit
def test_critical_categories_cover_price_and_fundamentals_only():
    assert "core_stock_apis" in quality.CRITICAL_CATEGORIES
    assert "binance_perp" in quality.CRITICAL_CATEGORIES
    assert "binance_spot" in quality.CRITICAL_CATEGORIES
    # The auxiliary crowd — the whole point of the split.
    assert "news_data" not in quality.CRITICAL_CATEGORIES
    assert "macro_data" not in quality.CRITICAL_CATEGORIES
    assert "prediction_markets" not in quality.CRITICAL_CATEGORIES


# ---------------------------------------------------------------------------
# Method-level severity inside the Binance price categories: the klines /
# indicator engines are critical; enrichment tools never veto.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_perp_enrichment_failure_never_vetoes():
    # A tokenized-stock perp structurally lacks a spot leg, so basis tools
    # error; vision/depth can be unavailable. None of these may push the
    # tier to DEGRADED_CRITICAL (the historical false-NO_TRADE bug) — they
    # degrade to auxiliary and are disclosed.
    events = [
        _event("get_binance_basis", quality.KIND_OPTIONAL_UNAVAILABLE),
        _event("get_binance_spot_perp_basis", quality.KIND_OPTIONAL_UNAVAILABLE),
        _event("get_binance_vision_metrics", quality.KIND_OPTIONAL_UNAVAILABLE),
        _event("get_binance_depth_snapshot", quality.KIND_OPTIONAL_UNAVAILABLE),
        _event("get_binance_funding_rate", quality.KIND_OPTIONAL_UNAVAILABLE),
    ]
    out = classify_quality(events, {"get_binance_klines"})
    assert out["tier"] == TIER_DEGRADED_AUXILIARY
    assert out["critical_data_available"] is True
    assert out["critical_missing"] == []
    assert len(out["auxiliary_degraded"]) == 5


@pytest.mark.unit
def test_perp_indicator_engine_failure_is_critical():
    # The actual kline/indicator book failing MUST veto — historically this
    # was the inverse: the unregistered method name classified as auxiliary
    # while a decorative basis failure was critical.
    for method in ("get_binance_klines", "get_binance_indicators",
                   "get_binance_spot_klines", "get_binance_spot_indicators"):
        out = classify_quality(
            [_event(method, quality.KIND_OPTIONAL_UNAVAILABLE)],
            {"get_stock_data"},
        )
        assert out["tier"] == TIER_DEGRADED_CRITICAL, method
        assert out["critical_missing"] == [f"{method}(optional_unavailable)"]


@pytest.mark.unit
def test_indicator_methods_are_registered_in_category_table():
    # The indicator tools record sentinels under their own names; those names
    # must resolve in TOOLS_CATEGORIES or classify_quality silently drops
    # them to the unknown-method auxiliary branch.
    from yialpha.dataflows.interface import get_category_for_method

    assert get_category_for_method("get_binance_indicators") == "binance_perp"
    assert get_category_for_method("get_binance_spot_indicators") == "binance_spot"
    assert "get_binance_indicators" in quality._PERP_CORE_METHODS
    assert "get_binance_spot_indicators" in quality._PERP_CORE_METHODS


# ---------------------------------------------------------------------------
# Qualifier-level severity: an INDEX-kline miss is enrichment (settlement
# fair-value anchor), not a lost price book — last/mark stay critical.
# ---------------------------------------------------------------------------


def _qevent(method: str, kind: str, qualifier: str) -> dict:
    return {
        "method": method, "kind": kind, "detail": "test", "qualifier": qualifier,
    }


@pytest.mark.unit
def test_index_kline_failure_is_auxiliary():
    out = classify_quality(
        [_qevent("get_binance_klines", quality.KIND_OPTIONAL_UNAVAILABLE, "index")],
        {"get_binance_klines[index]"},
    )
    assert out["tier"] == TIER_DEGRADED_AUXILIARY
    assert out["critical_missing"] == []
    # "(recovered)" (2026-09-19): recovery keys on method AND qualifier —
    # an INDEX-basis success recovers the INDEX-basis sentinel only.
    assert out["auxiliary_degraded"] == [
        "get_binance_klines[index](optional_unavailable) (recovered)",
    ]


@pytest.mark.unit
def test_qualified_success_does_not_recover_other_basis():
    # Round-2 fix pin: an index-klines success (auxiliary basis) must NOT
    # recover a last/mark book outage of the same method name.
    out = classify_quality(
        [_qevent("get_binance_klines", quality.KIND_OPTIONAL_UNAVAILABLE, "last")],
        {"get_binance_klines[index]"},
    )
    assert out["tier"] == TIER_DEGRADED_CRITICAL
    assert out["critical_missing"] == [
        "get_binance_klines[last](optional_unavailable)",
    ]


@pytest.mark.unit
def test_mark_kline_failure_stays_critical():
    out = classify_quality(
        [_qevent("get_binance_klines", quality.KIND_OPTIONAL_UNAVAILABLE, "mark")],
        {"get_stock_data"},
    )
    assert out["tier"] == TIER_DEGRADED_CRITICAL
    assert out["critical_missing"] == [
        "get_binance_klines[mark](optional_unavailable)",
    ]


@pytest.mark.unit
def test_unqualified_kline_failure_stays_critical():
    # Defensive back-compat: a klines sentinel recorded WITHOUT a qualifier
    # (e.g. the bundle's core-leg sentinel) still grades critical.
    out = classify_quality(
        [_event("get_binance_klines", quality.KIND_OPTIONAL_UNAVAILABLE)],
        {"get_stock_data"},
    )
    assert out["tier"] == TIER_DEGRADED_CRITICAL


@pytest.mark.unit
def test_record_sentinel_carries_qualifier():
    quality.ensure_run_context()
    try:
        quality.record_sentinel(
            "get_binance_klines", quality.KIND_OPTIONAL_UNAVAILABLE,
            "down", qualifier="index",
        )
        events = quality.snapshot_quality()
    finally:
        quality.reset_quality()
    assert events[0]["qualifier"] == "index"
    # Events without a qualifier keep the empty-string shape (uniform dict).
    quality.ensure_run_context()
    try:
        quality.record_sentinel("get_news", quality.KIND_NO_DATA)
        assert quality.snapshot_quality()[0]["qualifier"] == ""
    finally:
        quality.reset_quality()


@pytest.mark.unit
def test_qualifier_flows_through_the_router(monkeypatch):
    # The tool layer passes _qualifier=price_type; the router's optional-
    # category sentinel must carry it so classify grades index as auxiliary.
    import yialpha.dataflows.interface as iface

    def refuse(symbol, start, end, interval="1d", price_type="last"):
        from yialpha.dataflows.errors import NoMarketDataError

        raise NoMarketDataError(symbol, symbol, "index endpoint down")

    monkeypatch.setattr(
        iface, "VENDOR_METHODS",
        {
            **iface.VENDOR_METHODS,
            "get_binance_klines": {"binance": refuse},
        },
    )
    quality.ensure_run_context()
    try:
        out = iface.route_to_vendor(
            "get_binance_klines", "BTCUSDT", "2026-01-01", "2026-01-31",
            "1d", "index", _qualifier="index",
        )
        events = quality.snapshot_quality()
    finally:
        quality.reset_quality()
    assert out.startswith("NO_DATA_AVAILABLE")
    assert events[0]["method"] == "get_binance_klines"
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    assert events[0]["qualifier"] == "index"
