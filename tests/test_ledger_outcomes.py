"""Outcome ledger tests (yialpha/ledger/outcomes).

Pins the append-only outcome contract (deterministic ids, idempotent
identical rewrite, ``ValueError`` on conflicting rewrite), ``legs_missing``
round-trip, the due/unscored ``pending_predictions`` worklist with
date-vs-datetime ``analysis_as_of``, and the typed readers.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from yialpha.ledger.evidence import register_run
from yialpha.ledger.models import SCOPE_CONTRACT, OutcomeRecord, new_outcome_id
from yialpha.ledger.outcomes import (
    all_outcomes,
    outcomes_for_prediction,
    pending_predictions,
    write_outcome,
)
from yialpha.ledger.predictions import revise_prediction, submit_predictions
from yialpha.ledger.sqlite import get_connection
from yialpha.versions import OUTCOME_COMPUTE_VERSION

_RUN_ID = "run-btc-2026-08-25"


def _setup_run() -> dict[int, str]:
    """Register the run and submit 3-horizon predictions; return {horizon: id}."""
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "perp_linear", "2026-08-25")
    ids = submit_predictions(
        _RUN_ID,
        "fundamentals",
        "BTCUSDT",
        SCOPE_CONTRACT,
        [
            {"horizon_days": 1, "direction": "up", "prob_up": 0.6},
            {"horizon_days": 5, "direction": "up", "prob_up": 0.55},
            {"horizon_days": 21, "direction": "flat", "prob_up": 0.5},
        ],
        "2026-08-25",
    )
    return {1: ids[0], 5: ids[1], 21: ids[2]}


@pytest.mark.unit
def test_write_outcome_is_deterministic_and_idempotent():
    predictions = _setup_run()
    outcome_id = write_outcome(
        predictions[5],
        _RUN_ID,
        5,
        status="complete",
        contract_price_return=0.031,
        underlying_return=0.028,
        basis_return=0.003,
        funding_pnl=-0.001,
        fees=-0.0008,
        slippage=-0.0002,
        net_return=0.029,
        outcome_available_at="2026-08-30T23:59:59+00:00",
        ticket_id="T-001",
    )
    assert outcome_id == new_outcome_id(predictions[5], 5)
    replay = write_outcome(
        predictions[5],
        _RUN_ID,
        5,
        status="complete",
        contract_price_return=0.031,
        underlying_return=0.028,
        basis_return=0.003,
        funding_pnl=-0.001,
        fees=-0.0008,
        slippage=-0.0002,
        net_return=0.029,
        outcome_available_at="2026-08-30T23:59:59+00:00",
        ticket_id="T-001",
    )
    assert replay == outcome_id
    records = outcomes_for_prediction(predictions[5])
    assert len(records) == 1  # one row, not two
    (record,) = records
    assert isinstance(record, OutcomeRecord)
    assert record.outcome_id == outcome_id
    assert record.status == "complete"
    assert record.net_return == pytest.approx(0.029)
    assert record.funding_pnl == pytest.approx(-0.001)
    assert record.ticket_id == "T-001"


@pytest.mark.unit
def test_conflicting_outcome_rewrite_raises():
    predictions = _setup_run()
    write_outcome(predictions[1], _RUN_ID, 1, status="complete", net_return=0.01)
    with pytest.raises(ValueError, match="is immutable"):
        write_outcome(predictions[1], _RUN_ID, 1, status="complete", net_return=0.99)
    records = outcomes_for_prediction(predictions[1])
    assert len(records) == 1
    assert records[0].net_return == pytest.approx(0.01)  # original survived


@pytest.mark.unit
def test_legs_missing_round_trips_as_sorted_tuple():
    predictions = _setup_run()
    write_outcome(
        predictions[21],
        _RUN_ID,
        21,
        status="incomplete",
        net_return=None,
        legs_missing=["underlying_close", "funding"],
        outcome_available_at="2026-09-15",
    )
    (record,) = outcomes_for_prediction(predictions[21])
    assert record.legs_missing == ("funding", "underlying_close")
    assert record.status == "incomplete"
    assert record.outcome_available_at == "2026-09-15"


@pytest.mark.unit
def test_invalid_status_and_horizon_rejected():
    predictions = _setup_run()
    with pytest.raises(ValueError, match="status"):
        write_outcome(predictions[1], _RUN_ID, 1, status="done")
    with pytest.raises(ValueError, match="horizon ladder"):
        write_outcome(predictions[1], _RUN_ID, 7, status="complete")
    assert all_outcomes() == []  # nothing was written


@pytest.mark.unit
def test_outcome_for_unknown_prediction_hits_foreign_key():
    _setup_run()
    with pytest.raises(sqlite3.IntegrityError):
        write_outcome("P" + "0" * 12, _RUN_ID, 1, status="complete")


@pytest.mark.unit
def test_pending_predictions_respects_horizon_elapsed():
    _setup_run()
    # as_of 2026-08-25 → due at 08-26 (1d), 08-30 (5d), 09-15 (21d)
    pending = pending_predictions("2026-09-03")
    horizons = {item["horizon_days"] for item in pending}
    assert horizons == {1, 5}
    by_horizon = {item["horizon_days"]: item for item in pending}
    entry = by_horizon[5]
    # scorer-facing fields, incl. the joined run columns
    assert entry["prediction_id"].startswith("P")
    assert entry["analyst"] == "fundamentals"
    assert entry["instrument_id"] == "BTCUSDT"
    assert entry["prediction_scope"] == SCOPE_CONTRACT
    assert entry["direction"] == "up"
    assert entry["prob_up"] == 0.55
    assert entry["analysis_as_of"] == "2026-08-25"
    assert entry["ticker"] == "BTCUSDT"
    assert entry["asset_type"] == "crypto_perp"
    assert entry["instrument_class"] == "perp_linear"
    # before any horizon elapses, nothing is pending
    assert pending_predictions("2026-08-25") == []


@pytest.mark.unit
def test_complete_outcome_excludes_prediction_from_pending():
    predictions = _setup_run()
    write_outcome(predictions[1], _RUN_ID, 1, status="complete", net_return=0.01)
    pending = pending_predictions("2026-09-03")
    assert {item["horizon_days"] for item in pending} == {5}
    # Any persisted outcome is immutable and leaves the automatic worklist.
    write_outcome(
        predictions[5],
        _RUN_ID,
        5,
        status="pending",
        legs_missing=["underlying_close"],
    )
    assert pending_predictions("2026-09-03") == []


@pytest.mark.unit
def test_pending_predictions_handles_datetime_analysis_as_of():
    run_id = "run-eth-2026-08-25-intraday"
    register_run(run_id, "ETHUSDT", "crypto_perp", "perp_linear", "2026-08-25T12:00:00+00:00")
    (prediction_id,) = submit_predictions(
        run_id,
        "market",
        "ETHUSDT",
        SCOPE_CONTRACT,
        [{"horizon_days": 1, "direction": "down", "prob_up": 0.7}],
        "2026-08-25T12:00:00+00:00",
    )
    # due at 2026-08-26T12:00 — not yet at midnight, yes by the next day
    assert pending_predictions("2026-08-26T00:00:00+00:00") == []
    pending = pending_predictions("2026-08-27")
    assert [item["prediction_id"] for item in pending] == [prediction_id]


@pytest.mark.unit
def test_pending_predictions_carries_revision_rows_separately():
    predictions = _setup_run()
    revise_prediction(
        predictions[5],
        [{"horizon_days": 5, "direction": "down", "prob_up": 0.65}],
        "debate flipped the signal",
        "2026-08-26",
    )
    pending = [item for item in pending_predictions("2026-09-03") if item["horizon_days"] == 5]
    assert len(pending) == 2  # original + revision are distinct worklist entries
    revision_entry = next(item for item in pending if item["debate_revision"] == 1)
    assert revision_entry["original_prediction_id"] == predictions[5]
    assert revision_entry["direction"] == "down"


@pytest.mark.unit
def test_all_outcomes_returns_typed_records_with_limit():
    predictions = _setup_run()
    write_outcome(predictions[1], _RUN_ID, 1, status="complete", net_return=0.01)
    write_outcome(predictions[5], _RUN_ID, 5, status="pending")
    records = all_outcomes()
    assert len(records) == 2
    assert all(isinstance(record, OutcomeRecord) for record in records)
    limited = all_outcomes(limit=1)
    assert len(limited) == 1
    assert limited == records[:1]


@pytest.mark.unit
def test_readers_on_absent_ledger_return_empty():
    # No write has happened in this test, so the per-test tmp DB does not
    # exist — optional reads degrade to empty, not OperationalError.
    assert outcomes_for_prediction("P" + "0" * 12) == []
    assert all_outcomes() == []
    assert pending_predictions("2026-09-03") == []


@pytest.mark.unit
@pytest.mark.parametrize("status", ["pending", "incomplete"])
def test_append_only_noncomplete_outcomes_retire_from_automatic_worklist(status):
    """An immutable result cannot remain eligible for a conflicting retry."""
    predictions = _setup_run()
    write_outcome(predictions[1], _RUN_ID, 1, status=status)
    with pytest.raises(ValueError, match="is immutable"):
        write_outcome(predictions[1], _RUN_ID, 1, status="complete", net_return=0.01)
    assert predictions[1] not in {
        item["prediction_id"] for item in pending_predictions("2026-09-20")
    }
    # Later horizons remain accessible despite the retired earliest row.
    assert {item["horizon_days"] for item in pending_predictions("2026-09-20")} == {5, 21}


@pytest.mark.unit
def test_scoring_context_roundtrip_canonical_identity_and_conflict():
    predictions = _setup_run()
    context = {"version": OUTCOME_COMPUTE_VERSION, "reason": "no_new_trading_session"}
    first = write_outcome(predictions[1], _RUN_ID, 1, status="incomplete", scoring_context=context)
    assert write_outcome(
        predictions[1], _RUN_ID, 1, status="incomplete",
        scoring_context=dict(reversed(list(context.items()))),
    ) == first
    assert outcomes_for_prediction(predictions[1])[0].scoring_context == context
    assert all_outcomes()[0].scoring_context == context
    with pytest.raises(ValueError, match="immutable"):
        write_outcome(
            predictions[1], _RUN_ID, 1, status="incomplete",
            scoring_context={**context, "reason": "rewritten"},
        )


@pytest.mark.unit
def test_due_uses_prediction_formation_and_returns_parsed_timing():
    predictions = _setup_run()
    timing = {"version": OUTCOME_COMPUTE_VERSION, "prediction_formed_at": "2026-08-26T12:00:00+00:00"}
    # Simulate a persisted timing contract independently of reference capture.
    get_connection().execute("UPDATE predictions SET timing=? WHERE prediction_id=?", (json.dumps(timing), predictions[1]))
    assert pending_predictions("2026-08-27T11:59:59+00:00") == []
    (item,) = pending_predictions("2026-08-27T12:00:00+00:00")
    assert item["prediction_id"] == predictions[1]
    assert item["timing"] == timing
    assert item["timing_error"] is None


@pytest.mark.unit
@pytest.mark.parametrize("raw", [
    "not json", "null", "[]", "{}",
    '{"version":"unknown","prediction_formed_at":"2026-08-25T12:00:00+00:00"}',
    '{"version":"close_reference_v1","prediction_formed_at":"2026-08-25"}',
])
def test_invalid_timing_is_diagnosable_without_hiding_other_candidates(raw):
    predictions = _setup_run()
    get_connection().execute("UPDATE predictions SET timing=? WHERE prediction_id=?", (raw, predictions[1]))
    rows = pending_predictions("2026-09-20")
    assert {item["horizon_days"] for item in rows} == {1, 5, 21}
    invalid = next(item for item in rows if item["prediction_id"] == predictions[1])
    assert isinstance(invalid["timing"], dict)
    assert invalid["timing_error"]


@pytest.mark.unit
def test_readers_accept_old_schema_without_mutation():
    predictions = _setup_run()
    write_outcome(predictions[1], _RUN_ID, 1, status="complete", net_return=0.01)
    connection = get_connection()
    connection.execute("ALTER TABLE predictions DROP COLUMN timing")
    connection.execute("ALTER TABLE outcomes DROP COLUMN scoring_context")
    before = connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    assert outcomes_for_prediction(predictions[1])[0].scoring_context is None
    assert all_outcomes()[0].scoring_context is None
    assert all(item["timing"] is None for item in pending_predictions("2026-09-20"))
    after = connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    assert [tuple(row) for row in before] == [tuple(row) for row in after]
