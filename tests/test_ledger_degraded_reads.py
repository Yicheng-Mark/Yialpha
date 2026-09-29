"""Round-10: optional ledger readers degrade on an unreadable ledger file.

A file that EXISTS but is not a YiAlpha ledger passes ``ledger_exists()``;
the read-only connection never migrates, so the first statement raises. Two
crash classes, both pinned here with real bytes (not mocks):

* garbage bytes -> ``sqlite3.DatabaseError`` ("file is not a database") —
  the PARENT of ``OperationalError``, which round-10's first cut of pasted
  ``except sqlite3.OperationalError`` guards did NOT catch (every one of
  them escaped); connect-time errors also escaped where ``get_connection``
  sat outside the try (outcomes/scoreboard);
* a valid but foreign SQLite file (0-byte / unrelated tables) -> plain
  ``OperationalError`` ("no such table").

One shared chokepoint — :func:`yialpha.ledger.sqlite.optional_read_rows`
(``except sqlite3.Error``, connection inside the try, one WARNING naming
the reader) — now covers every optional reader. These pins assert the
[]-with-warning contract the round-10 comments claim, plus the two row-level
degradations that must never abort a whole sweep: a malformed legacy
``analysis_as_of`` and an unparseable evidence anchor.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from yialpha.dataflows.config import set_config
from yialpha.ledger.evidence import evidence_for_run, evidence_ids_for_run
from yialpha.ledger.outcomes import (
    all_outcomes,
    outcomes_for_prediction,
    pending_predictions,
)
from yialpha.ledger.predictions import (
    prediction_by_id,
    predictions_for_run,
    revisions_of,
)
from yialpha.ledger.sqlite import reset_ledger_state_for_test

_NOW = "2026-09-03"


def _install_ledger_bytes(tmp_path: Path, payload: bytes | None) -> Path:
    """Point the ledger at a file with exactly these bytes (None = 0-byte)."""
    path = tmp_path / "corrupt-ledger" / "portfolio.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload or b"")
    set_config({"ledger_db_path": str(path)})
    reset_ledger_state_for_test()
    return path


def _install_foreign_sqlite(tmp_path: Path) -> Path:
    """A valid SQLite file that is not a YiAlpha ledger (unrelated tables)."""
    path = tmp_path / "foreign-ledger" / "portfolio.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE other_app (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO other_app VALUES ('x')")
    conn.commit()
    conn.close()
    set_config({"ledger_db_path": str(path)})
    reset_ledger_state_for_test()
    return path


def _assert_degraded(caplog: pytest.LogCaptureFixture, context: str) -> None:
    """One WARNING from the chokepoint naming the calling reader."""
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == "yialpha.ledger.sqlite"
        and "ledger read degraded" in record.getMessage()
        and context in record.getMessage()
    ]
    assert warnings, f"no degraded-read warning naming {context!r}: {caplog.text}"


# --------------------------------------------------------------------------- #
# Corrupt bytes: sqlite3.DatabaseError escapes except-OperationalError guards
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_garbage_ledger_raises_databaseerror_not_operationalerror(tmp_path):
    """Live proof of the crash class the narrow guards let through: a corrupt
    file's first statement raises DatabaseError ("file is not a database"),
    which is OperationalError's PARENT — ``except sqlite3.OperationalError``
    cannot catch it. This pin fails if anyone narrows the chokepoint back."""
    from yialpha.ledger.sqlite import get_connection

    _install_ledger_bytes(tmp_path, b"not a sqlite file at all")
    with pytest.raises(sqlite3.DatabaseError, match="file is not a database") as info:
        get_connection(readonly=True).execute("SELECT 1 FROM predictions")
    assert not isinstance(info.value, sqlite3.OperationalError)


@pytest.mark.unit
def test_pending_predictions_degrades_on_garbage_ledger(tmp_path, caplog):
    _install_ledger_bytes(tmp_path, b"not a sqlite file at all")
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.sqlite"):
        assert pending_predictions(_NOW) == []
    _assert_degraded(caplog, "pending_predictions")


@pytest.mark.unit
def test_outcome_readers_degrade_on_garbage_ledger(tmp_path, caplog):
    _install_ledger_bytes(tmp_path, b"not a sqlite file at all")
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.sqlite"):
        assert outcomes_for_prediction("P-any") == []
        assert all_outcomes() == []
    _assert_degraded(caplog, "outcomes_for_prediction")
    _assert_degraded(caplog, "all_outcomes")


@pytest.mark.unit
def test_evidence_readers_degrade_on_garbage_ledger(tmp_path, caplog):
    _install_ledger_bytes(tmp_path, b"not a sqlite file at all")
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.sqlite"):
        assert evidence_for_run("R-any") == []
        assert evidence_ids_for_run("R-any") == []
    _assert_degraded(caplog, "evidence_for_run")
    _assert_degraded(caplog, "evidence_ids_for_run")


@pytest.mark.unit
def test_prediction_readers_degrade_on_garbage_ledger(tmp_path, caplog):
    _install_ledger_bytes(tmp_path, b"not a sqlite file at all")
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.sqlite"):
        assert predictions_for_run("R-any") == []
        assert prediction_by_id("P-any") is None
        assert revisions_of("P-any") == []
    _assert_degraded(caplog, "predictions_for_run")
    _assert_degraded(caplog, "prediction_by_id")
    _assert_degraded(caplog, "revisions_of")


@pytest.mark.unit
def test_scoreboard_degrades_on_garbage_ledger(tmp_path, caplog):
    from yialpha.ledger.scoreboard import _scored_rows, build_scoreboard

    _install_ledger_bytes(tmp_path, b"not a sqlite file at all")
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.sqlite"):
        assert _scored_rows() == []
        scoreboard = build_scoreboard()
    _assert_degraded(caplog, "_scored_rows")
    # An empty-but-honest scoreboard, structurally intact for the renderer.
    assert scoreboard["overall"]["n"] == 0
    assert scoreboard["overall"]["directional_accuracy"] is None


@pytest.mark.unit
def test_ticket_cost_legs_degrade_on_garbage_ledger(tmp_path, caplog):
    from yialpha.ledger.outcome_compute import _ticket_cost_legs

    _install_ledger_bytes(tmp_path, b"not a sqlite file at all")
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.sqlite"):
        assert _ticket_cost_legs("R-any") == (None, None, None)
    _assert_degraded(caplog, "_ticket_cost_legs")


# --------------------------------------------------------------------------- #
# Valid-but-foreign SQLite: plain OperationalError ("no such table")
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pending_predictions_degrades_on_foreign_sqlite(tmp_path, caplog):
    _install_foreign_sqlite(tmp_path)
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.sqlite"):
        assert pending_predictions(_NOW) == []
    _assert_degraded(caplog, "pending_predictions")


@pytest.mark.unit
def test_zero_byte_ledger_degrades_every_reader(tmp_path, caplog):
    _install_ledger_bytes(tmp_path, None)
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.sqlite"):
        assert pending_predictions(_NOW) == []
        assert predictions_for_run("R-any") == []
        assert evidence_for_run("R-any") == []
        assert all_outcomes() == []
    _assert_degraded(caplog, "pending_predictions")
    _assert_degraded(caplog, "predictions_for_run")
    _assert_degraded(caplog, "evidence_for_run")
    _assert_degraded(caplog, "all_outcomes")


# --------------------------------------------------------------------------- #
# Row-level degradations: one malformed row must not abort the sweep
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_malformed_legacy_as_of_degrades_row_not_sweep():
    """A hand-edited blank ``analysis_as_of`` used to raise ValueError out of
    pending_predictions (timestamp_as_utc('')) — BEFORE compute_outcomes'
    per-row isolation — killing every other row's batch. The row is now
    nominated with ``timing_error`` and retired by the terminal scorer
    diagnosis, exactly like an invalid timing blob (no vendor seam is ever
    touched on that path)."""
    from yialpha.ledger.evidence import register_run
    from yialpha.ledger.models import SCOPE_CONTRACT
    from yialpha.ledger.outcome_compute import compute_outcomes
    from yialpha.ledger.predictions import submit_predictions

    register_run("run-blank-asof", "BTCUSDT", "crypto_perp", "pure_crypto_perp", "")
    (prediction_id,) = submit_predictions(
        "run-blank-asof",
        "fundamentals",
        "BTCUSDT",
        SCOPE_CONTRACT,
        [{"horizon_days": 5, "direction": "up", "prob_up": 0.6}],
        "",  # the malformed legacy anchor
    )

    worklist = pending_predictions(_NOW)  # must NOT raise
    assert [item["prediction_id"] for item in worklist] == [prediction_id]
    assert worklist[0]["timing"] is None
    assert "analysis_as_of" in (worklist[0]["timing_error"] or "")

    report = compute_outcomes(_NOW)
    assert (report.considered, report.completed, report.incomplete, report.failed) == (
        1, 0, 1, 0,
    )
    (detail,) = report.details
    assert detail.status == "incomplete"
    assert "invalid_prediction_timing" in (detail.reason or "")
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"  # terminal diagnosis, row retired
    assert pending_predictions(_NOW) == []


@pytest.mark.unit
def test_malformed_evidence_anchor_keeps_evidence_flowing(caplog):
    """A truthy-but-unparseable ``analysis_as_of`` ("2026/07/01") used to
    reach pit_latest_instant, whose plain ValueError propagated and was
    swallowed upstream — dropping EVERY run-derived evidence block. Now the
    anchor is treated as unanchored with an explicit WARNING; a VALID anchor
    still raises PITEvidenceViolation on future-dated evidence."""
    from yialpha.ledger.evidence import (
        PITEvidenceViolation,
        record_evidence,
        register_run,
    )
    from yialpha.ledger.models import REPLAYABILITY_LIVE_ONLY, SCOPE_CONTRACT

    register_run("run-anchor", "BTCUSDT", "crypto_perp", "pure_crypto_perp", "2026-09-03")
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.evidence"):
        evidence_id = record_evidence(
            run_id="run-anchor",
            source="news",
            category="news_data",
            symbol="BTCUSDT",
            scope=SCOPE_CONTRACT,
            payload="PAYLOAD",
            replayability=REPLAYABILITY_LIVE_ONLY,
            analysis_as_of="2026/07/01",  # truthy but unparseable
            available_at="2026-09-04T00:00:00+00:00",  # even future-dated
        )
    # The evidence row is STILL written — losing the measurability chain over
    # one malformed string is the worse failure.
    rows = evidence_for_run("run-anchor")
    assert [row.evidence_id for row in rows] == [evidence_id]
    assert any(
        "unparseable analysis_as_of" in record.getMessage()
        and "2026/07/01" in record.getMessage()
        for record in caplog.records
    )
    # Byte-identical PIT guard for a parseable anchor: future-dated evidence
    # still raises the typed violation.
    with pytest.raises(PITEvidenceViolation, match="PIT violation"):
        record_evidence(
            run_id="run-anchor",
            source="news",
            category="news_data",
            symbol="BTCUSDT",
            scope=SCOPE_CONTRACT,
            payload="PAYLOAD-2",
            replayability=REPLAYABILITY_LIVE_ONLY,
            analysis_as_of="2026-09-03",
            available_at="2026-09-04T00:00:00+00:00",
        )


# --------------------------------------------------------------------------- #
# web store: the degraded read is not a phantom empty book
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_web_positions_reflect_degraded_read(tmp_path, caplog):
    """load_positions used to serve available=True + positions=[] on a
    degraded read — the phantom empty book presentation; the flag now
    reflects open_positions_with_status' ok."""
    from web import store

    _install_ledger_bytes(tmp_path, b"not a sqlite file at all")
    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.portfolio"):
        payload = store.load_positions()
    assert payload["available"] is False
    assert payload["positions"] == []
    assert any("read failed" in record.getMessage() for record in caplog.records)
