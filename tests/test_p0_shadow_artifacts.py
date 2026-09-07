"""Offline, temporary-directory artifact integrity and provenance checks."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from scripts.p0_shadow.artifacts import EvidenceRecorder, create_manifest
from scripts.p0_shadow.guard import GuardError

STARTED = datetime(2026, 9, 6, 12, tzinfo=UTC)
OBSERVED = STARTED + timedelta(seconds=1)


@pytest.fixture(autouse=True)
def _fresh_yfinance_cache_managers():
    """Re-arm yfinance's process-wide cache singletons before each runtime.

    isolated_runtime refuses already-initialized managers to mirror the
    fresh-process rule; in a full-suite run earlier dataflow tests have
    opened them. set_location() closes _db and resets it in place.
    """
    from yfinance import cache as yf_cache

    for name in ("_TzDBManager", "_CookieDBManager", "_ISINDBManager"):
        manager = getattr(yf_cache, name, None)
        if manager is not None:
            manager.set_location(manager.get_location())
    yield


def _record(recorder, **changes):
    kwargs = {
        "source": "fixture:binance_perp",
        "request_metadata": {
            "endpoint": "/fapi/v1/klines", "method": "GET",
            "query": {"symbol": "BTCUSDT", "interval": "1d"},
        },
        "raw_bytes": b'[[123, "100.5"]]',
        "normalized_payload": [{"Date": "2026-09-05", "Close": 100.5}],
        "started_at": STARTED,
        "observed_at": OBSERVED,
        "evidence_level": "offline_fixture",
        "call_id": "capture-btc-1",
        "prediction_ids": ["pred-1", "pred-5", "pred-21"],
    }
    kwargs.update(changes)
    return recorder.record_response(**kwargs)


def test_same_call_response_can_be_replayed_and_normalized_is_separate(tmp_path):
    record = _record(EvidenceRecorder(tmp_path, "attempt-1"))
    raw = (tmp_path / record["raw"]["path"]).read_bytes()
    normalized = (tmp_path / record["normalized"]["path"]).read_bytes()
    payload = json.loads(normalized)
    assert json.loads(raw)[0][1] == "100.5"
    assert payload["payload"][0]["Close"] == 100.5
    assert not payload["is_vendor_raw_response"]
    assert record["raw"]["sha256"] == sha256(raw).hexdigest()
    assert record["normalized"]["sha256"] == sha256(normalized).hexdigest()
    assert record["raw"]["sha256"] != record["normalized"]["sha256"]
    assert record["attempt_id"] == "attempt-1"
    assert record["call_id"] == "capture-btc-1"
    assert record["prediction_ids"] == ["pred-1", "pred-5", "pred-21"]
    assert not record["origin_verified_by_recorder"]
    assert not record["vendor_http_hook_installed"]
    assert json.loads((tmp_path / record["record_path"]).read_text()) == record


def test_sensitive_headers_query_and_url_parameters_are_never_written(tmp_path):
    record = _record(
        EvidenceRecorder(tmp_path, "attempt-1"),
        request_metadata={
            "endpoint": "/fapi/v1/klines",
            "url": "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&api_key=secret-url#private",
            "headers": {
                "Content-Type": "application/json", "Authorization": "secret-auth",
                "Cookie": "secret-cookie", "X-MBX-APIKEY": "secret-header",
            },
            "response_headers": {"Set-Cookie": "secret-set-cookie", "Date": "source-date"},
            "params": {"symbol": "BTCUSDT", "apiKey": "secret-param", "signature": "secret-sig"},
            "query": {"interval": "1d", "access_token": "secret-token"},
            "proxy": "secret-proxy", "unknown": {"api_key": "secret-nested"},
        },
    )
    metadata = record["request_metadata"]
    assert metadata["headers"] == {"content-type": "application/json"}
    assert metadata["response_headers"] == {"date": "source-date"}
    assert metadata["query"] == {"interval": "1d"}
    assert metadata["params"] == {"symbol": "BTCUSDT"}
    assert metadata["url"] == "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"secret-" not in path.read_bytes()


@pytest.mark.parametrize("changes", [
    {"started_at": "2026-09-06T12:00:00"},
    {"observed_at": STARTED - timedelta(seconds=1)},
    {"normalized_payload": {"Close": float("nan")}},
    {"normalized_payload": {"Close": float("inf")}},
    {"evidence_level": "vendor_http_response"},
    {"request_metadata": {"url": "https://user:secret@example.com/history"}},
    {"request_metadata": {"endpoint": "/history?api_key=secret"}},
    {"prediction_ids": "pred-1"},
    {"call_id": "../outside"},
])
def test_invalid_input_is_rejected_before_any_artifact_write(tmp_path, changes):
    with pytest.raises((ValueError, TypeError)):
        _record(EvidenceRecorder(tmp_path, "attempt-1"), **changes)
    assert list(tmp_path.iterdir()) == []


def test_same_response_parallel_records_do_not_overwrite_an_observation(tmp_path):
    recorder = EvidenceRecorder(tmp_path, "attempt-1")
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda _: _record(recorder), range(8)))
    assert len({row["request_id"] for row in records}) == 8
    assert len(list((tmp_path / "raw").glob("*/record.json"))) == 8
    assert len({row["raw"]["sha256"] for row in records}) == 1
    for row in records:
        assert json.loads((tmp_path / row["record_path"]).read_text()) == row


def test_hash_detects_changed_replay_bytes(tmp_path):
    record = _record(EvidenceRecorder(tmp_path, "attempt-1"))
    original = (tmp_path / record["raw"]["path"]).read_bytes()
    changed_replay = original.replace(b"100.5", b"900.5")
    assert sha256(changed_replay).hexdigest() != record["raw"]["sha256"]
    assert sha256(original).hexdigest() == record["raw"]["sha256"]


def test_manifest_covers_twelve_distinct_records_and_versions(tmp_path):
    manifest = create_manifest(
        tmp_path, "offline-cohort-1", sample_kind="offline_fixture",
        code_identity={"head": "fixture-head", "dirty_diff_sha256": "fixture-hash"},
        created_at=STARTED, versions={"sqlite": 4, "outcome": "close_reference_v1"},
    )
    matrix = manifest["matrix"]
    assert len(matrix) == 12
    assert len({row["matrix_id"] for row in matrix}) == 12
    expected = {
        (symbol, scope, horizon)
        for symbol, scope in (
            ("BTCUSDT", "CONTRACT"), ("ETHUSDT", "CONTRACT"),
            ("MUUSDT", "CONTRACT"), ("MU", "UNDERLYING"),
        )
        for horizon in (1, 5, 21)
    }
    assert {(row["instrument_id"], row["prediction_scope"], row["horizon_days"]) for row in matrix} == expected
    assert all(row["requires_real_utc_weekend"] for row in matrix if row["instrument_id"] == "MU")
    assert manifest["production_contract_versions"]["sqlite"] == 4
    assert manifest["production_versions_captured"]
    assert manifest["analysis_only"] and not manifest["live_execution_enabled"]
    assert not manifest["boundaries"]["real_natural_maturity_verified"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_manifest_is_immutable_and_omitted_versions_are_disclosed(tmp_path):
    kwargs = {
        "sample_kind": "offline_fixture", "code_identity": {"head": "fixture-head"},
        "created_at": STARTED,
    }
    first = create_manifest(tmp_path, "offline-cohort-1", **kwargs)
    before = (tmp_path / "manifest.json").read_bytes()
    assert first["production_contract_versions"] is None
    assert not first["production_versions_captured"]
    with pytest.raises(GuardError, match="overwrite"):
        create_manifest(tmp_path, "offline-cohort-2", **kwargs)
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_offset_timestamps_are_normalized_to_utc(tmp_path):
    record = _record(
        EvidenceRecorder(tmp_path, "attempt-1"),
        started_at="2026-09-06T20:00:00+08:00", observed_at="2026-09-06T20:00:01+08:00",
    )
    assert record["started_at"] == STARTED.isoformat()
    assert record["observed_at"] == OBSERVED.isoformat()


def test_partial_response_write_does_not_publish_record(tmp_path, monkeypatch):
    recorder = EvidenceRecorder(tmp_path, "attempt-1")
    original = recorder.guard.write_json

    def fail_normalized(relative, payload, *, exclusive=True):
        if str(relative).endswith("normalized.json"):
            raise OSError("simulated interrupted evidence write")
        return original(relative, payload, exclusive=exclusive)

    monkeypatch.setattr(recorder.guard, "write_json", fail_normalized)
    with pytest.raises(OSError, match="interrupted"):
        _record(recorder)
    assert len(list((tmp_path / "raw").glob("*/response.bin"))) == 1
    assert list((tmp_path / "raw").glob("*/record.json")) == []


def test_offline_demo_end_to_end_then_read_only_inspect(tmp_path):
    """The B0 runner CLI path: fixture cohort, synthetic maturity, audit."""
    from scripts.p0_shadow.runner import inspect_cohort, run_offline_demo

    summary = run_offline_demo(tmp_path / "cohort", cohort_id="e2e-baseline")
    assert summary["predictions"] == 12
    # Only the weekend U-MU 1d same-session window is unscoreable by design.
    assert summary["outcomes_by_status"] == {"complete": 11, "incomplete": 1}
    assert summary["real_network_enabled"] is False
    inspected = inspect_cohort(tmp_path / "cohort", cohort_id="e2e-baseline")
    assert inspected["scoring_invoked"] is False
    assert (tmp_path / "cohort" / "ledger" / "portfolio.db").is_file()


def test_offline_demo_missing_funding_reaches_terminal_incomplete(tmp_path):
    from scripts.p0_shadow.runner import run_offline_demo

    summary = run_offline_demo(tmp_path / "cohort", cohort_id="e2e-gap",
                               scenario="missing-funding")
    # BTC funding never backfills: 1/5/21d stay pending then terminate at the
    # fixed 7-day deadline, plus the weekend same-session 1d sample.
    assert summary["outcomes_by_status"] == {"complete": 8, "incomplete": 4}
