"""Offline acceptance-time snapshots: exact closed bars and immutable clocks."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from yialpha.ledger import time_contract as tc
from yialpha.versions import PREDICTION_TIME_VERSION


def _frame(dates=("2026-09-04", "2026-09-05"), prices=(100.0, 999.0)):
    return pd.DataFrame({"Close": prices}, index=pd.to_datetime(dates))


def _clock(monkeypatch, instant="2026-09-05T12:00:00+00:00"):
    monkeypatch.setattr(tc, "_now_utc", lambda: datetime.fromisoformat(instant))


@pytest.mark.unit
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "MUUSDT"])
def test_capture_only_uses_exact_closed_perp_reference(monkeypatch, symbol):
    _clock(monkeypatch)
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return _frame()

    monkeypatch.setattr(tc, "binance_klines_frame", fetch)
    timing = tc.capture_prediction_timing(symbol, "CONTRACT")
    assert timing["version"] == PREDICTION_TIME_VERSION
    assert timing["reference_price"] == 100.0
    assert timing["reference_price_at"] == "2026-09-05T00:00:00+00:00"
    assert timing["reference_available_at"] == timing["reference_price_at"]
    assert timing["reference_observed_at"] == timing["prediction_formed_at"]
    assert timing["reference_source"] == "binance_perp:1d:last"
    assert timing["reference_error"] is None
    assert calls[0][0] == (symbol, "2026-09-04", "2026-09-04")
    assert calls[0][1]["closed_as_of"] == int(datetime(2026, 9, 5, 12, tzinfo=UTC).timestamp() * 1000)


@pytest.mark.unit
def test_reference_observation_precedes_actual_acceptance_after_fetch(monkeypatch):
    instants = iter(datetime.fromisoformat(s) for s in (
        "2026-09-05T12:00:00+00:00", "2026-09-05T12:00:03+00:00",
        "2026-09-05T12:00:04+00:00",
    ))
    monkeypatch.setattr(tc, "_now_utc", lambda: next(instants))
    monkeypatch.setattr(tc, "binance_klines_frame", lambda *a, **k: _frame())
    timing = tc.capture_prediction_timing("BTCUSDT", "CONTRACT")
    assert timing["reference_observed_at"] == "2026-09-05T12:00:03+00:00"
    assert timing["prediction_formed_at"] == "2026-09-05T12:00:04+00:00"


@pytest.mark.unit
@pytest.mark.parametrize("day", ["2026-09-05", "2026-09-06", "2026-09-07"])
def test_weekend_reference_is_exact_friday_equity_close(monkeypatch, day):
    _clock(monkeypatch, day + "T12:00:00+00:00")
    monkeypatch.setattr(tc, "get_YFin_history_cached", lambda *a, **k: _frame())
    timing = tc.capture_prediction_timing("MU", "UNDERLYING")
    assert timing["reference_price"] == 100.0
    assert timing["reference_price_at"] == "2026-09-05T00:00:00+00:00"
    assert timing["reference_source"] == "yfinance:1d:close"


@pytest.mark.unit
@pytest.mark.parametrize("scope", ["CONTRACT", "UNDERLYING"])
def test_missing_expected_bar_never_uses_older_bar(monkeypatch, scope):
    _clock(monkeypatch)
    older = _frame(("2026-09-03",), (100.0,))
    monkeypatch.setattr(tc, "binance_klines_frame", lambda *a, **k: older)
    monkeypatch.setattr(tc, "get_YFin_history_cached", lambda *a, **k: older)
    timing = tc.capture_prediction_timing("MU", scope)
    assert timing["reference_price"] is None
    assert "2026-09-04" in timing["reference_error"]
    assert timing["version"] == PREDICTION_TIME_VERSION


@pytest.mark.unit
@pytest.mark.parametrize("price", [float("inf"), float("nan"), 0.0, -1.0])
def test_invalid_reference_is_recorded_as_unavailable(monkeypatch, price):
    _clock(monkeypatch)
    monkeypatch.setattr(tc, "binance_klines_frame", lambda *a, **k: _frame(("2026-09-04",), (price,)))
    timing = tc.capture_prediction_timing("BTCUSDT", "CONTRACT")
    assert timing["reference_price"] is None
    assert timing["reference_error"] == "reference_price_invalid"


@pytest.mark.unit
def test_vendor_failure_keeps_new_version_and_formation_time(monkeypatch):
    _clock(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(tc, "binance_klines_frame", fail)
    timing = tc.capture_prediction_timing("BTCUSDT", "CONTRACT")
    assert timing["version"] == PREDICTION_TIME_VERSION
    assert timing["reference_price"] is None
    assert timing["reference_error"] == "reference_fetch_failed:RuntimeError"
    assert timing["prediction_formed_at"] == "2026-09-05T12:00:00+00:00"


@pytest.mark.unit
@pytest.mark.parametrize("scope", ["MACRO", "POSITIONING"])
def test_nonprice_scope_records_formation_without_fetch(monkeypatch, scope):
    _clock(monkeypatch)

    def fail(*args, **kwargs):
        pytest.fail("nonprice prediction must not fetch a price")

    monkeypatch.setattr(tc, "_reference_frame", fail)
    timing = tc.capture_prediction_timing("BTCUSDT", scope)
    assert timing["prediction_formed_at"] == "2026-09-05T12:00:00+00:00"
    assert timing["reference_price"] is None
    assert timing["reference_error"] is None


@pytest.mark.unit
@pytest.mark.parametrize("changes", [
    {"prediction_formed_at": "2026-09-05"},
    {"reference_observed_at": "2026-09-05T12:00:00"},
    {"reference_available_at": "2026-09-06T00:00:00+00:00"},
    {"reference_price_at": "2026-09-05T13:00:00+00:00"},
    {"reference_price": float("nan")},
    {"reference_price": True},
    {"version": "unknown"},
])
def test_timing_rejects_naive_future_nonfinite_or_unknown_contract(monkeypatch, changes):
    _clock(monkeypatch)
    monkeypatch.setattr(tc, "binance_klines_frame", lambda *a, **k: _frame())
    timing = tc.capture_prediction_timing("BTCUSDT", "CONTRACT")
    with pytest.raises(ValueError):
        tc.validate_timing({**timing, **changes})


@pytest.mark.unit
def test_fetch_crossing_midnight_cannot_promote_a_forming_reference(monkeypatch):
    moments = iter([
        datetime(2026, 9, 5, 23, 59, 59, tzinfo=UTC),
        datetime(2026, 9, 6, 0, 0, 1, tzinfo=UTC),
        datetime(2026, 9, 6, 0, 0, 2, tzinfo=UTC),
    ])
    monkeypatch.setattr(tc, "_now_utc", lambda: next(moments))
    monkeypatch.setattr(tc, "binance_klines_frame", lambda *a, **k: _frame())
    timing = tc.capture_prediction_timing("BTCUSDT", "CONTRACT")
    assert timing["reference_price"] is None
    assert timing["reference_error"] == "reference_bar_not_closed"
