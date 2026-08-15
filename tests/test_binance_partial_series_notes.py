"""In-band notes for partial Binance perp series (silent-visibility hardening).

A dropped L/S-ratio series or a missing live open-interest snapshot used to be
log-only: the CSV's absence of a series was ambiguous between "no data" and
"fetch failed". These tests pin the header notes that disambiguate.
"""

from __future__ import annotations

import pytest

from yiagents.dataflows import binance as bn
from yiagents.dataflows.errors import NoMarketDataError


def _ls_rows():
    return [{
        "timestamp": 1735689600000,
        "longAccount": "0.6",
        "longShortRatio": "1.5",
        "shortAccount": "0.4",
    }]


@pytest.mark.unit
def test_ls_ratio_dropped_series_noted(monkeypatch):
    def fake_http(path, params, symbol, canonical):
        if "topLongShortPositionRatio" in path:
            raise NoMarketDataError(symbol, canonical, "429")
        return _ls_rows()

    monkeypatch.setattr(bn, "_http_get", fake_http)
    out = bn.get_binance_long_short_ratio("SPCXUSDT", 5)
    assert "⚠ unavailable series: top_position" in out
    assert "absence ≠ no positioning" in out
    assert "top_account" in out          # surviving series still rendered
    assert "global_account" in out


@pytest.mark.unit
def test_ls_ratio_all_series_present_has_no_note(monkeypatch):
    monkeypatch.setattr(bn, "_http_get", lambda *a, **k: _ls_rows())
    out = bn.get_binance_long_short_ratio("SPCXUSDT", 5)
    assert "unavailable series" not in out


@pytest.mark.unit
def test_open_interest_live_snapshot_failure_noted(monkeypatch):
    def fake_http(path, params, symbol, canonical):
        if path == "/fapi/v1/openInterest":
            raise NoMarketDataError(symbol, canonical, "timeout")
        return [{
            "timestamp": 1735689600000,
            "sumOpenInterest": "1000",
            "sumOpenInterestValue": "2000",
        }]

    monkeypatch.setattr(bn, "_http_get", fake_http)
    out = bn.get_binance_open_interest("SPCXUSDT", 5)
    assert "⚠ live openInterest snapshot unavailable" in out
    assert "history only" in out


@pytest.mark.unit
def test_open_interest_healthy_has_no_note(monkeypatch):
    def fake_http(path, params, symbol, canonical):
        if path == "/fapi/v1/openInterest":
            return {"openInterest": "1234.5"}
        return [{
            "timestamp": 1735689600000,
            "sumOpenInterest": "1000",
            "sumOpenInterestValue": "2000",
        }]

    monkeypatch.setattr(bn, "_http_get", fake_http)
    out = bn.get_binance_open_interest("SPCXUSDT", 5)
    assert "snapshot unavailable" not in out
