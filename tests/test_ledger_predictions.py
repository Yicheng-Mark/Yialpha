"""Blind prediction ledger tests (yialpha/ledger/predictions).

Pins the immutability contract end to end: deterministic ids, one row per
ladder horizon with version stamps, idempotent identical resubmission,
``ValueError`` on conflicting resubmission or invalid entries, the revision
chain (root linkage + 1-based numbering), and the typed readers.
"""

from __future__ import annotations

import sqlite3

import pytest

from yialpha.ledger.evidence import register_run
from yialpha.ledger.models import (
    SCOPE_CONTRACT,
    AnalystPrediction,
    PredictionEntry,
    new_prediction_id,
)
from yialpha.ledger.predictions import (
    prediction_by_id,
    predictions_for_run,
    revise_prediction,
    revisions_of,
    submit_predictions,
)
from yialpha.versions import FEATURE_VERSION, SCHEMA_VERSION

_RUN_ID = "run-btc-2026-08-25"
_ANALYST = "fundamentals"
_INSTRUMENT = "BTCUSDT"


def _setup_run(as_of: str = "2026-08-25") -> str:
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "perp_linear", as_of)
    return _RUN_ID


def _entries() -> list[dict[str, object]]:
    return [
        {"horizon_days": 1, "direction": "up", "prob_up": 0.6, "confidence": 0.55},
        {
            "horizon_days": 5,
            "direction": "up",
            "prob_up": 0.55,
            "expected_return": 0.03,
            "target_price": 65000.0,
            "target_currency": "USDT",
            "price_basis": "mark_price_close",
        },
        {"horizon_days": 21, "direction": "flat", "prob_up": 0.5},
    ]


def _submit(run_id: str = _RUN_ID, as_of: str = "2026-08-25") -> list[str]:
    return submit_predictions(
        run_id,
        _ANALYST,
        _INSTRUMENT,
        SCOPE_CONTRACT,
        _entries(),
        as_of,
        evidence_ids=["Ebbb", "Eaaa"],
    )


@pytest.mark.unit
def test_three_horizons_three_rows_stamped_with_versions():
    run_id = _setup_run()
    ids = _submit()
    assert len(ids) == 3
    assert len(set(ids)) == 3
    records = predictions_for_run(run_id)
    assert len(records) == 3
    assert {record.horizon_days for record in records} == {1, 5, 21}
    assert all(record.schema_version == SCHEMA_VERSION for record in records)
    assert all(record.feature_version == FEATURE_VERSION for record in records)
    assert all(record.run_id == run_id for record in records)
    assert all(record.analyst == _ANALYST for record in records)
    assert all(record.instrument_id == _INSTRUMENT for record in records)
    assert all(record.prediction_scope == SCOPE_CONTRACT for record in records)
    # evidence ids round-trip as a sorted tuple (canonical set semantics)
    assert all(record.evidence_ids == ("Eaaa", "Ebbb") for record in records)
    assert all(record.original_prediction_id is None for record in records)
    assert all(record.debate_revision is None for record in records)


@pytest.mark.unit
def test_prediction_ids_are_deterministic_from_preimage():
    run_id = _setup_run()
    ids = _submit()
    assert ids[0] == new_prediction_id(run_id, _ANALYST, SCOPE_CONTRACT, 1)
    assert ids[1] == new_prediction_id(run_id, _ANALYST, SCOPE_CONTRACT, 5)
    assert ids[2] == new_prediction_id(run_id, _ANALYST, SCOPE_CONTRACT, 21)
    by_id = prediction_by_id(ids[1])
    assert by_id is not None and isinstance(by_id, AnalystPrediction)
    assert by_id.horizon_days == 5
    assert by_id.direction == "up"
    assert by_id.prob_up == 0.55
    assert by_id.expected_return == 0.03
    assert by_id.target_price == 65000.0
    assert by_id.target_currency == "USDT"
    assert by_id.price_basis == "mark_price_close"


@pytest.mark.unit
def test_identical_resubmission_is_idempotent():
    run_id = _setup_run()
    first = _submit()
    second = _submit()
    assert first == second
    assert len(predictions_for_run(run_id)) == 3  # still three rows, not six


@pytest.mark.unit
def test_evidence_order_does_not_change_content_identity():
    run_id = _setup_run()
    first = _submit()
    second = submit_predictions(
        run_id,
        _ANALYST,
        _INSTRUMENT,
        SCOPE_CONTRACT,
        _entries(),
        "2026-08-25",
        evidence_ids=["Eaaa", "Ebbb"],  # same set, different order
    )
    assert first == second
    assert len(predictions_for_run(run_id)) == 3


@pytest.mark.unit
def test_conflicting_resubmission_raises_immutable():
    run_id = _setup_run()
    _submit()
    conflicting = [
        entry if entry["horizon_days"] != 5 else {**entry, "direction": "down"}
        for entry in _entries()
    ]
    with pytest.raises(ValueError, match="is immutable"):
        submit_predictions(
            run_id, _ANALYST, _INSTRUMENT, SCOPE_CONTRACT, conflicting, "2026-08-25"
        )
    assert len(predictions_for_run(run_id)) == 3  # nothing new written


@pytest.mark.unit
def test_unknown_run_hits_foreign_key():
    with pytest.raises(sqlite3.IntegrityError):
        submit_predictions(
            "run-never-registered",
            _ANALYST,
            _INSTRUMENT,
            SCOPE_CONTRACT,
            _entries(),
            "2026-08-25",
        )


@pytest.mark.unit
def test_invalid_horizon_direction_and_probabilities_rejected():
    run_id = _setup_run()
    with pytest.raises(ValueError, match="horizon ladder"):
        submit_predictions(
            run_id,
            _ANALYST,
            _INSTRUMENT,
            SCOPE_CONTRACT,
            [{"horizon_days": 7, "direction": "up"}],
            "2026-08-25",
        )
    with pytest.raises(ValueError, match="direction"):
        submit_predictions(
            run_id,
            _ANALYST,
            _INSTRUMENT,
            SCOPE_CONTRACT,
            [{"horizon_days": 1, "direction": "UP"}],
            "2026-08-25",
        )
    with pytest.raises(ValueError, match="prob_up"):
        submit_predictions(
            run_id,
            _ANALYST,
            _INSTRUMENT,
            SCOPE_CONTRACT,
            [{"horizon_days": 1, "direction": "up", "prob_up": 1.5}],
            "2026-08-25",
        )
    with pytest.raises(ValueError, match="confidence"):
        submit_predictions(
            run_id,
            _ANALYST,
            _INSTRUMENT,
            SCOPE_CONTRACT,
            [{"horizon_days": 1, "direction": "up", "confidence": -0.1}],
            "2026-08-25",
        )
    assert predictions_for_run(run_id) == []  # validation precedes any write


@pytest.mark.unit
def test_unknown_entry_keys_and_duplicate_horizons_rejected():
    run_id = _setup_run()
    with pytest.raises(ValueError, match="unknown prediction entry keys"):
        submit_predictions(
            run_id,
            _ANALYST,
            _INSTRUMENT,
            SCOPE_CONTRACT,
            [{"horizon_days": 1, "direction": "up", "horizon": 1}],
            "2026-08-25",
        )
    with pytest.raises(ValueError, match="duplicate horizon_days"):
        submit_predictions(
            run_id,
            _ANALYST,
            _INSTRUMENT,
            SCOPE_CONTRACT,
            [
                {"horizon_days": 5, "direction": "up"},
                {"horizon_days": 5, "direction": "down"},
            ],
            "2026-08-25",
        )
    with pytest.raises(ValueError, match="scope"):
        submit_predictions(
            run_id,
            _ANALYST,
            _INSTRUMENT,
            "GALAXY",
            [{"horizon_days": 1, "direction": "up"}],
            "2026-08-25",
        )


@pytest.mark.unit
def test_prediction_entry_dataclass_entries_accepted():
    run_id = _setup_run()
    ids = submit_predictions(
        run_id,
        "market",
        _INSTRUMENT,
        SCOPE_CONTRACT,
        [PredictionEntry(horizon_days=1, direction="down", prob_up=0.7)],
        "2026-08-25",
    )
    assert ids == [new_prediction_id(run_id, "market", SCOPE_CONTRACT, 1)]
    record = prediction_by_id(ids[0])
    assert record is not None
    assert record.direction == "down"
    assert record.prob_up == 0.7
    assert record.evidence_ids == ()  # no evidence chain → empty tuple


@pytest.mark.unit
def test_revision_chain_links_and_numbers_revisions():
    run_id = _setup_run()
    original_ids = _submit()
    revision_one = revise_prediction(
        original_ids[1],  # revise the 5-day horizon
        [{"horizon_days": 5, "direction": "down", "prob_up": 0.65}],
        "debate evidence flipped the signal",
        "2026-08-26",
    )
    assert len(revision_one) == 1
    revision = prediction_by_id(revision_one[0])
    assert revision is not None
    assert revision.original_prediction_id == original_ids[1]
    assert revision.debate_revision == 1
    assert revision.revision_reason == "debate evidence flipped the signal"
    assert revision.analysis_as_of == "2026-08-26"
    assert revision.evidence_ids == ("Eaaa", "Ebbb")  # evidence chain reused
    assert revision.run_id == run_id

    revision_two = revise_prediction(
        original_ids[1],
        [{"horizon_days": 5, "direction": "down", "prob_up": 0.6}],
        "sharpened probability after PM challenge",
        "2026-08-26",
    )
    second = prediction_by_id(revision_two[0])
    assert second is not None
    assert second.debate_revision == 2
    assert second.original_prediction_id == original_ids[1]

    chain = revisions_of(original_ids[1])
    assert [row.debate_revision for row in chain] == [1, 2]
    assert len(predictions_for_run(run_id)) == 5  # 3 originals + 2 revisions


@pytest.mark.unit
def test_revising_a_revision_stays_anchored_to_root():
    _setup_run()
    original_ids = _submit()
    first = revise_prediction(
        original_ids[1],
        [{"horizon_days": 5, "direction": "down", "prob_up": 0.65}],
        "first correction",
        "2026-08-26",
    )
    # Revising the REVISION must link to the root, not fork a new chain.
    second = revise_prediction(
        first[0],
        [{"horizon_days": 5, "direction": "flat", "prob_up": 0.5}],
        "second correction",
        "2026-08-27",
    )
    record = prediction_by_id(second[0])
    assert record is not None
    assert record.original_prediction_id == original_ids[1]
    assert record.debate_revision == 2
    assert len(revisions_of(original_ids[1])) == 2


@pytest.mark.unit
def test_revising_nonexistent_prediction_raises():
    _setup_run()
    with pytest.raises(ValueError, match="not found in ledger"):
        revise_prediction(
            "P" + "0" * 12,
            [{"horizon_days": 1, "direction": "up"}],
            "reason",
            "2026-08-26",
        )


@pytest.mark.unit
def test_prediction_by_id_unknown_returns_none():
    assert prediction_by_id("P" + "f" * 12) is None


@pytest.mark.unit
def test_readers_on_absent_ledger_return_empty():
    # No write has happened in this test, so the per-test tmp DB does not
    # exist — optional reads degrade to empty, not OperationalError.
    assert predictions_for_run("run-none") == []
    assert prediction_by_id("P" + "0" * 12) is None
    assert revisions_of("P" + "0" * 12) == []
