"""Ledger evidence seam tests (yialpha/ledger/evidence + models ids).

conftest's autouse ``_runtime_ledger_isolated`` fixture points
``ledger_db_path`` at a per-test tmp DB, so these tests hit the real sqlite
seam directly (no mocks): registration idempotence, the run foreign key,
content dedupe, the PIT guard, and deterministic id stability.
"""

from __future__ import annotations

import sqlite3
from hashlib import sha256

import pytest

from yialpha.ledger.evidence import (
    evidence_for_run,
    evidence_ids_for_run,
    record_evidence,
    register_run,
)
from yialpha.ledger.models import (
    REPLAYABILITY_LIVE_ONLY,
    REPLAYABILITY_PIT_REPLAYABLE,
    SCOPE_CONTRACT,
    SCOPE_UNDERLYING,
    EvidenceRecord,
    new_evidence_id,
    new_outcome_id,
    new_prediction_id,
)
from yialpha.ledger.sqlite import get_connection
from yialpha.versions import SCHEMA_VERSION

_RUN_ID = "run-mu-2026-09-01"


def _register_run() -> str:
    register_run(_RUN_ID, "MU", "stock", "equity_common", "2026-09-01")
    return _RUN_ID


@pytest.mark.unit
def test_register_run_creates_schema_versioned_row():
    _register_run()
    rows = get_connection(readonly=True).execute(
        "SELECT run_id, ticker, asset_type, instrument_class, analysis_as_of, "
        "config_digest, schema_version FROM runs"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["run_id"] == _RUN_ID
    assert rows[0]["ticker"] == "MU"
    assert rows[0]["instrument_class"] == "equity_common"
    assert rows[0]["analysis_as_of"] == "2026-09-01"
    assert rows[0]["config_digest"] is None
    assert rows[0]["schema_version"] == SCHEMA_VERSION


@pytest.mark.unit
def test_register_run_is_idempotent_on_run_id():
    _register_run()
    first_created = get_connection(readonly=True).execute(
        "SELECT created_at FROM runs WHERE run_id = ?", (_RUN_ID,)
    ).fetchone()["created_at"]
    # A retried pipeline replays the registration with different optional
    # fields — the original row must survive untouched.
    register_run(
        _RUN_ID, "MU", "stock", None, "2026-09-01", config_digest="digest-2"
    )
    rows = get_connection(readonly=True).execute(
        "SELECT created_at, config_digest FROM runs"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["created_at"] == first_created
    assert rows[0]["config_digest"] is None  # replay did not overwrite


@pytest.mark.unit
def test_evidence_for_unknown_run_hits_foreign_key():
    with pytest.raises(sqlite3.IntegrityError):
        record_evidence(
            "run-never-registered",
            "news",
            "news",
            "MU",
            SCOPE_UNDERLYING,
            "payload",
            replayability=REPLAYABILITY_PIT_REPLAYABLE,
        )


@pytest.mark.unit
def test_same_payload_dedupes_to_one_row():
    run_id = _register_run()
    first = record_evidence(
        run_id,
        "news",
        "news",
        "MU",
        SCOPE_UNDERLYING,
        "identical block",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
        available_at="2026-09-01T08:00:00+00:00",
    )
    second = record_evidence(
        run_id,
        "news",
        "news",
        "MU",
        SCOPE_UNDERLYING,
        "identical block",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
        available_at="2026-09-01T08:00:00+00:00",
    )
    expected_hash = sha256(b"identical block").hexdigest()
    assert first == second == new_evidence_id(run_id, expected_hash)
    assert evidence_ids_for_run(run_id) == [first]
    assert len(evidence_for_run(run_id)) == 1


@pytest.mark.unit
def test_different_payload_is_a_separate_row():
    run_id = _register_run()
    record_evidence(
        run_id,
        "news",
        "news",
        "MU",
        SCOPE_UNDERLYING,
        "block A",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
    )
    record_evidence(
        run_id,
        "news",
        "news",
        "MU",
        SCOPE_UNDERLYING,
        "block B",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
    )
    ids = evidence_ids_for_run(run_id)
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert len(evidence_for_run(run_id)) == 2


@pytest.mark.unit
def test_same_payload_in_different_run_does_not_dedupe():
    run_id = _register_run()
    register_run("run-mu-2026-09-02", "MU", "stock", "equity_common", "2026-09-02")
    id_a = record_evidence(
        run_id,
        "news",
        "news",
        "MU",
        SCOPE_UNDERLYING,
        "shared payload",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
    )
    id_b = record_evidence(
        "run-mu-2026-09-02",
        "news",
        "news",
        "MU",
        SCOPE_UNDERLYING,
        "shared payload",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
    )
    assert id_a != id_b  # evidence identity is (run, payload), not payload alone


@pytest.mark.unit
def test_pit_guard_rejects_late_available_at():
    run_id = _register_run()
    with pytest.raises(ValueError, match="PIT violation"):
        record_evidence(
            run_id,
            "news",
            "news",
            "MU",
            SCOPE_UNDERLYING,
            "late block",
            replayability=REPLAYABILITY_LIVE_ONLY,
            available_at="2026-09-02T00:00:00+00:00",
            analysis_as_of="2026-09-01",
        )
    assert evidence_ids_for_run(run_id) == []  # nothing was written


@pytest.mark.unit
def test_pit_guard_admits_same_day_evidence_for_date_only_as_of():
    # A date-only analysis_as_of means the run executes DURING that day, so
    # evidence available at 18:00 on the run date is admissible.
    run_id = _register_run()
    evidence_id = record_evidence(
        run_id,
        "news",
        "news",
        "MU",
        SCOPE_UNDERLYING,
        "same-day block",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
        available_at="2026-09-01T18:00:00+00:00",
        analysis_as_of="2026-09-01",
    )
    assert evidence_ids_for_run(run_id) == [evidence_id]


@pytest.mark.unit
def test_invalid_scope_and_replayability_rejected():
    run_id = _register_run()
    with pytest.raises(ValueError, match="scope"):
        record_evidence(
            run_id,
            "news",
            "news",
            "MU",
            "PLANET",
            "payload",
            replayability=REPLAYABILITY_PIT_REPLAYABLE,
        )
    with pytest.raises(ValueError, match="replayability"):
        record_evidence(
            run_id,
            "news",
            "news",
            "MU",
            SCOPE_CONTRACT,
            "payload",
            replayability="MAYBE",
        )


@pytest.mark.unit
def test_deterministic_ids_stable_across_calls():
    hash_a = sha256(b"payload").hexdigest()
    assert new_evidence_id("run-1", hash_a) == new_evidence_id("run-1", hash_a)
    assert new_prediction_id("run-1", "fundamentals", SCOPE_UNDERLYING, 5) == (
        new_prediction_id("run-1", "fundamentals", SCOPE_UNDERLYING, 5)
    )
    assert new_outcome_id("Pabc", 5) == new_outcome_id("Pabc", 5)
    # and discriminated by every preimage component
    assert new_prediction_id("run-1", "market", SCOPE_UNDERLYING, 5) != (
        new_prediction_id("run-1", "fundamentals", SCOPE_UNDERLYING, 5)
    )
    assert new_prediction_id("run-1", "fundamentals", SCOPE_UNDERLYING, 21) != (
        new_prediction_id("run-1", "fundamentals", SCOPE_UNDERLYING, 5)
    )
    assert new_outcome_id("Pabc", 5) != new_outcome_id("Pabc", 21)


@pytest.mark.unit
def test_id_prefixes_and_length():
    hash_a = sha256(b"payload").hexdigest()
    evidence_id = new_evidence_id("run-1", hash_a)
    prediction_id = new_prediction_id("run-1", "fundamentals", SCOPE_CONTRACT, 1)
    outcome_id = new_outcome_id("P" + "0" * 12, 5)
    assert evidence_id.startswith("E") and len(evidence_id) == 13
    assert prediction_id.startswith("P") and len(prediction_id) == 13
    assert outcome_id.startswith("O") and len(outcome_id) == 13


@pytest.mark.unit
def test_evidence_for_run_returns_typed_records():
    run_id = _register_run()
    payload = "SEC filing text"
    evidence_id = record_evidence(
        run_id,
        "edgar",
        "filings",
        "MU",
        SCOPE_UNDERLYING,
        payload,
        source_url="https://example.gov/filing",
        event_time="2026-08-30",
        available_at="2026-08-31T10:00:00+00:00",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
        quality_status="ok",
        analysis_as_of="2026-09-01",
    )
    records = evidence_for_run(run_id)
    assert all(isinstance(record, EvidenceRecord) for record in records)
    (record,) = records
    assert record.evidence_id == evidence_id
    assert record.run_id == run_id
    assert record.source == "edgar"
    assert record.category == "filings"
    assert record.scope == SCOPE_UNDERLYING
    assert record.payload_hash == sha256(payload.encode("utf-8")).hexdigest()
    assert record.source_url == "https://example.gov/filing"
    assert record.event_time == "2026-08-30"
    assert record.available_at == "2026-08-31T10:00:00+00:00"
    assert record.replayability == REPLAYABILITY_PIT_REPLAYABLE
    assert record.quality_status == "ok"


@pytest.mark.unit
def test_readers_on_absent_ledger_return_empty():
    # No write has happened in this test, so the per-test tmp DB does not
    # exist — optional reads degrade to empty, not OperationalError.
    assert evidence_for_run("run-none") == []
    assert evidence_ids_for_run("run-none") == []
