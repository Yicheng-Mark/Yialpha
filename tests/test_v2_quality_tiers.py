"""V2.0 P0.3 — four-tier data-quality classification (critical-data gate).

GOOD / DEGRADED_AUXILIARY / DEGRADED_CRITICAL / INVALID. Only critical
categories (price/indicators/fundamentals/perp-spot books) and vacuums can
escalate; auxiliary outages (news/macro/social) never do.
"""

from __future__ import annotations

import pytest

from yiagents.dataflows import quality
from yiagents.dataflows.quality import (
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
