"""Round-17 audit tests for the ``web_runtime`` shard (app / store / runner / health).

Covers what earlier rounds left untested: the ledger-backed read surfaces
(tickets, snapshots, positions, outcomes, calibration, blind predictions), the
listing edges, the script launch and the route surface. Each test pins one
documented contract of the local Web UI — exact payloads from well-formed
on-disk state, the documented "unavailable" / ``None`` / 404 answer for
everything else (missing file, absent ledger DB, unreadable row, corrupt JSON)
— and the round-17 fixes: non-finite numbers and an unreadable ledger, which
were 500s or a fabricated "available" answer.

Hermetic: every analysis spawn and subprocess launch is stubbed except two
child interpreters that only import ``web/app.py`` (uvicorn is faked, no
server, no socket); the health probes replace the three network surfaces; the
logs root and the ledger DB are per-test tmp paths.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web import app as app_module, health, runner, store
from web.app import app

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def logs_root(monkeypatch, tmp_path):
    """Point the store at an empty per-test logs root (never ~/.yialpha).

    ``REPORTS_ROOT`` is repointed too: the app's lifespan does
    ``store.REPORTS_ROOT.mkdir(parents=True, exist_ok=True)`` on startup, so a
    client fixture that only moved ``LOGS_ROOT`` would touch the real
    ``~/.yialpha/logs/reports`` on every ``TestClient(app)``.
    """
    root = tmp_path / "logs"
    monkeypatch.setattr(store, "LOGS_ROOT", root)
    monkeypatch.setattr(store, "REPORTS_ROOT", root / "reports")
    return root


@pytest.fixture
def client(monkeypatch, logs_root):
    """App client: isolated task registry, isolated logs root, real lifespan."""
    monkeypatch.setattr(runner, "registry", runner.TaskRegistry())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_no_raise(monkeypatch, logs_root):
    """Same as ``client`` but server exceptions surface as 500 responses."""
    monkeypatch.setattr(runner, "registry", runner.TaskRegistry())
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _write_state_log(root, ticker, day, decision="**Rating**: Buy", *, extra=None, raw=None):
    """Write one ``full_states_log_<day>.json`` (``raw`` bytes win if given)."""
    d = root / ticker / "YiAlphaStrategy_logs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"full_states_log_{day}.json"
    if raw is not None:
        path.write_bytes(raw)
        return path
    state = {"company_of_interest": ticker, "trade_date": day, "final_trade_decision": decision}
    if extra:
        state.update(extra)
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


_STAMP = "2026-06-10T12:00:00+00:00"
_AS_OF = "2026-06-10"


def _create_ledger():
    """Create the per-test ledger DB through the real write path (one run row).

    ``tests/conftest.py`` points the ``ledger_db_path`` config at a per-test tmp
    file; ``register_run`` opens the write connection, which migrates the schema.
    """
    from yialpha.ledger.evidence import register_run

    register_run("run-1", "BTCUSDT", "crypto_perp", "pure_crypto_perp", _AS_OF)


def _scalar(sql: str, params: tuple = ()):
    from yialpha.ledger.sqlite import get_connection

    return get_connection(readonly=True).execute(sql, params).fetchone()[0]


def _insert_row(sql: str, params: tuple) -> None:
    from yialpha.ledger.sqlite import ledger_transaction

    with ledger_transaction() as cur:
        cur.execute(sql, params)


def _refuse_reads(monkeypatch, module):
    """Make ``module``'s ``get_connection`` behave like an unreadable ledger."""
    import yialpha.ledger.sqlite as ledger_sqlite

    def locked(*, readonly=False):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(ledger_sqlite, "get_connection", locked)
    monkeypatch.setattr(module, "get_connection", locked)


# --------------------------------------------------------------------------- #
# store: _dates_for / _latest_rating
# --------------------------------------------------------------------------- #


class TestStoreDatesFor:
    def test_r17_store_missing_strategy_dir_is_empty(self, logs_root):
        assert store._dates_for("AAPL") == []

    def test_r17_store_keeps_only_the_venue_vocabulary_and_sorts_ascending(self, logs_root):
        """Only ``full_states_log_<YYYY-MM-DD>[_perp|_spot].json`` names are
        analysis dates (the captured stem is the date key every downstream
        constructor rebuilds the filename from); sorted ascending."""
        d = logs_root / "AAPL" / "YiAlphaStrategy_logs"
        d.mkdir(parents=True)
        for name in (
            "full_states_log_2026-09-19_spot.json",
            "full_states_log_2026-09-19.json",
            "full_states_log_2026-09-19_perp.json",
            "full_states_log_2026-01-05.json",
            # not date keys
            "full_states_log_x.json",
            "full_states_log_2026-9-1.json",
            "full_states_log_2026-09-19_PERP.json",
            "full_states_log_2026-09-19_perpx.json",
            "full_states_log_2026-09-19.spot.json",
            "full_states_log_2026-09-19.json.bak",
            "node_perf_2026-09-20.json",
            "copy_full_states_log_2026-09-20.json",
        ):
            (d / name).write_text("{}", encoding="utf-8")
        assert store._dates_for("AAPL") == [
            "2026-01-05",
            "2026-09-19",
            "2026-09-19_perp",
            "2026-09-19_spot",
        ]

    def test_r17_store_same_date_venues_sort_after_the_plain_key(self, logs_root):
        d = logs_root / "ETHUSDT" / "YiAlphaStrategy_logs"
        d.mkdir(parents=True)
        for suffix in ("_spot", "_perp", ""):
            (d / f"full_states_log_2026-06-10{suffix}.json").write_text("{}", encoding="utf-8")
        assert store._dates_for("ETHUSDT") == [
            "2026-06-10",
            "2026-06-10_perp",
            "2026-06-10_spot",
        ]


class TestStoreLatestRating:
    def test_r17_store_missing_file_is_none(self, logs_root):
        """A listed date whose file is gone (or was never a file) reads as no
        rating — the documented "None only when the JSON can't be read"."""
        assert store._latest_rating("AAPL", "2026-06-10") is None

    def test_r17_store_well_formed_log_yields_the_rating_word(self, logs_root):
        _write_state_log(logs_root, "AAPL", "2026-06-10", decision="**Rating**: Sell")
        assert store._latest_rating("AAPL", "2026-06-10") == "Sell"
        assert store._latest_rating("AAPL", "2026-06-10_perp") is None

    def test_r17_store_readable_log_without_a_tier_word_defaults_to_hold(self, logs_root):
        _write_state_log(logs_root, "AAPL", "2026-06-10", decision="no rating here")
        assert store._latest_rating("AAPL", "2026-06-10") == "Hold"


# --------------------------------------------------------------------------- #
# store: load_ticket (V2.4 read-only ticket surface)
# --------------------------------------------------------------------------- #


_TICKET_COLUMNS = (
    "INSERT INTO tickets (ticket_id, decision_id, run_id, payload, ticket_version, written_at) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


class TestStoreLoadTicket:
    def test_r17_store_no_ledger_db_is_none(self):
        assert store.load_ticket("T-1") is None

    def test_r17_store_unknown_ticket_id_is_none(self):
        _create_ledger()
        assert store.load_ticket("nope") is None

    def test_r17_store_unreadable_ledger_is_none(self, monkeypatch):
        import yialpha.ledger.sqlite as ledger_sqlite

        _create_ledger()
        _refuse_reads(monkeypatch, ledger_sqlite)
        assert store.load_ticket("T-1") is None

    def test_r17_store_mirrored_row_round_trips_with_its_columns(self):
        from yialpha.ledger.tickets_mirror import attach_ticket

        _create_ledger()
        payload = {
            "ticket_id": "T-1",
            "decision_id": "D-1",
            "side": "LONG",
            "entry": {"price": 65432.5},
            "evidence_ids": ["E-1", "E-2"],
        }
        attach_ticket("T-1", "D-1", "run-1", payload, "ticket-v1")
        written_at = _scalar("SELECT written_at FROM tickets WHERE ticket_id = 'T-1'")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", written_at)
        assert store.load_ticket("T-1") == {
            "ticket_id": "T-1",
            "run_id": "run-1",
            "ticket_version": "ticket-v1",
            "written_at": written_at,
            "payload": payload,
        }

    def test_r17_store_corrupt_payload_json_is_served_as_a_null_payload(self):
        _create_ledger()
        _insert_row(_TICKET_COLUMNS, ("T-bad", "D-1", "run-1", "{not json", "v1", _STAMP))
        assert store.load_ticket("T-bad") == {
            "ticket_id": "T-bad",
            "run_id": "run-1",
            "ticket_version": "v1",
            "written_at": _STAMP,
            "payload": None,
        }

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("[]", []), ('"text"', "text"), ("null", None), ("42", 42)],
        ids=["list-root", "string-root", "null-root", "number-root"],
    )
    def test_r17_store_non_object_payload_does_not_raise(self, raw, expected):
        _create_ledger()
        _insert_row(_TICKET_COLUMNS, ("T-shape", None, None, raw, "v1", _STAMP))
        got = store.load_ticket("T-shape")
        assert got["payload"] == expected
        assert got["ticket_id"] == "T-shape" and got["run_id"] is None


# --------------------------------------------------------------------------- #
# store: load_portfolio_snapshot (V2.4 read-only surface)
# --------------------------------------------------------------------------- #


def _commit_snapshot(snapshot_id: str, snapshot: dict) -> None:
    from yialpha.ledger.portfolio import commit_final_ticket

    assert commit_final_ticket(
        ticket_payload={"ticket_id": "T-1", "side": "LONG"},
        ticket_id="T-1",
        decision_id="D-1",
        run_id="run-1",
        ticket_version="ticket-v1",
        snapshot_payload=snapshot,
        snapshot_id=snapshot_id,
        position_symbol="BTCUSDT",
        position_side="LONG",
        position_signed_weight=0.1,
        open_position=True,
        close_existing=False,
        instrument_class="pure_crypto_perp",
    )


class TestStoreLoadPortfolioSnapshot:
    def test_r17_store_no_ledger_db_is_none(self):
        assert store.load_portfolio_snapshot("S-1") is None

    def test_r17_store_unknown_snapshot_id_is_none(self):
        _create_ledger()
        assert store.load_portfolio_snapshot("nope") is None

    def test_r17_store_unreadable_ledger_is_none(self, monkeypatch):
        import yialpha.ledger.portfolio as ledger_portfolio

        _create_ledger()
        _refuse_reads(monkeypatch, ledger_portfolio)
        assert store.load_portfolio_snapshot("S-1") is None

    def test_r17_store_snapshot_written_by_the_resolver_path_round_trips(self):
        snapshot = {
            "snapshot_id": "S-abc",
            "run_id": "run-1",
            "mode": "enforced",
            "equity": 1000.0,
            "positions": [{"symbol": "ETHUSDT", "signed_weight": 0.05}],
            "decisions": [{"rule": "max_gross", "ok": True}],
        }
        _create_ledger()
        _commit_snapshot("S-abc", snapshot)
        assert store.load_portfolio_snapshot("S-abc") == snapshot

    def test_r17_store_corrupt_payload_json_is_none(self):
        _create_ledger()
        _insert_row(
            "INSERT INTO portfolio_snapshots (snapshot_id, run_id, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("S-bad", "run-1", "{'equity': nan}", _STAMP),
        )
        assert store.load_portfolio_snapshot("S-bad") is None

    def test_r17_store_non_object_payload_does_not_raise(self):
        _create_ledger()
        _insert_row(
            "INSERT INTO portfolio_snapshots (snapshot_id, run_id, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("S-shape", None, "[]", _STAMP),
        )
        assert store.load_portfolio_snapshot("S-shape") == []



# --------------------------------------------------------------------------- #
# store: list_tickers / list_compare / list_reports / load_run (listing edges)
# --------------------------------------------------------------------------- #


def _report_dir(root, name, *, complete=True, mtime=None):
    d = root / name
    d.mkdir(parents=True)
    if complete:
        cr = d / "complete_report.md"
        cr.write_text("# report\n", encoding="utf-8")
        if mtime is not None:
            os.utime(cr, (mtime, mtime))
    elif mtime is not None:
        os.utime(d, (mtime, mtime))
    return d


class TestStoreListTickers:
    def test_r17_store_skips_files_non_ticker_dirs_and_dirs_without_a_run(self, logs_root):
        """Only directories holding at least one completed run are tickers; the
        reports / robust roots, stray files and an empty ticker dir are not."""
        logs_root.mkdir()
        (logs_root / "reports").mkdir()
        (logs_root / "robust").mkdir()
        (logs_root / "accuracy").mkdir()
        (logs_root / "stray.txt").write_text("x", encoding="utf-8")
        (logs_root / "NORUN").mkdir()
        _write_state_log(logs_root, "msft", "2026-06-01", decision="**Rating**: Sell")
        assert store.list_tickers() == [
            {
                "ticker": "msft",
                "latest_date": "2026-06-01",
                "latest_rating": "Sell",
                "run_count": 1,
                "history": [{"date": "2026-06-01", "rating": "Sell"}],
            }
        ]

    def test_r17_store_missing_logs_root_is_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(store, "LOGS_ROOT", tmp_path / "nope")
        assert store.list_tickers() == []

    def test_r17_store_sorted_case_insensitively_with_latest_date_rating_and_count(self, logs_root):
        _write_state_log(logs_root, "ZVZZT", "2026-06-01", decision="**Rating**: Sell")
        _write_state_log(logs_root, "aapl", "2026-06-03", decision="**Rating**: Buy")
        _write_state_log(logs_root, "aapl", "2026-06-02", decision="**Rating**: Hold")
        _write_state_log(logs_root, "aapl", "2026-06-02_perp", decision="**Rating**: Sell")
        _write_state_log(logs_root, "MSFT", "2026-06-05", decision="**Rating**: Buy")
        entries = store.list_tickers()
        assert [e["ticker"] for e in entries] == ["aapl", "MSFT", "ZVZZT"]
        aapl = entries[0]
        assert aapl["latest_date"] == "2026-06-03"
        assert aapl["latest_rating"] == "Buy"
        assert aapl["run_count"] == 3
        assert aapl["history"] == [
            {"date": "2026-06-02", "rating": "Hold"},
            {"date": "2026-06-02_perp", "rating": "Sell"},
            {"date": "2026-06-03", "rating": "Buy"},
        ]
        assert entries[1]["history"] == [{"date": "2026-06-05", "rating": "Buy"}]

    def test_r17_store_history_keeps_only_the_most_recent_ratings(self, logs_root):
        """The home card's timeline dots are capped at ``_HISTORY_CAP`` newest
        dates; ``run_count`` still counts every completed run."""
        days = [f"2026-01-{d:02d}" for d in range(1, 13)]
        for i, day in enumerate(days):
            rating = ("Buy", "Hold", "Sell")[i % 3]
            _write_state_log(logs_root, "AAPL", day, decision=f"**Rating**: {rating}")
        [entry] = store.list_tickers()
        assert entry["run_count"] == 12
        assert entry["latest_date"] == "2026-01-12"
        assert entry["history"] == [
            {"date": day, "rating": ("Buy", "Hold", "Sell")[i % 3]}
            for i, day in enumerate(days)
            if day >= days[-store._HISTORY_CAP]
        ]
        assert [h["date"] for h in entry["history"]] == days[-store._HISTORY_CAP:]
        assert len(entry["history"]) == 8


class TestStoreListCompare:
    def test_r17_store_missing_logs_root_is_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(store, "LOGS_ROOT", tmp_path / "nope")
        assert store.list_compare() == {"tickers": []}

    def test_r17_store_skips_files_and_non_ticker_dirs(self, logs_root):
        logs_root.mkdir()
        (logs_root / "reports").mkdir()
        (logs_root / "robust").mkdir()
        (logs_root / "notes.txt").write_text("x", encoding="utf-8")
        (logs_root / "reports" / "tickerless").mkdir()
        _write_state_log(logs_root, "AAPL", "2026-06-01", decision="**Rating**: Buy")
        assert store.list_compare() == {
            "tickers": [
                {"ticker": "AAPL", "date_ratings": [{"date": "2026-06-01", "rating": "Buy"}]}
            ]
        }

    def test_r17_store_keeps_every_date_including_unreadable_ones_for_a_readable_ticker(self, logs_root):
        """A ticker is kept when ANY of its ratings is readable; its unreadable
        dates stay in the series as ``None`` (the chart shows the gap)."""
        _write_state_log(logs_root, "AAPL", "2026-06-01", decision="**Rating**: Buy")
        _write_state_log(logs_root, "AAPL", "2026-06-02", raw=b"{not json")
        _write_state_log(logs_root, "AAPL", "2026-06-03", raw=b"[]")
        assert store.list_compare() == {
            "tickers": [
                {
                    "ticker": "AAPL",
                    "date_ratings": [
                        {"date": "2026-06-01", "rating": "Buy"},
                        {"date": "2026-06-02", "rating": None},
                        {"date": "2026-06-03", "rating": None},
                    ],
                }
            ]
        }


class TestStoreListReports:
    def test_r17_store_unsafe_ticker_is_empty(self, logs_root):
        for bad in ("..", "../AAPL", "AAPL/..", "AAPL\\..", "", "A" * 33, "AAPL;rm"):
            assert store.list_reports(bad) == [], bad

    def test_r17_store_missing_reports_root_is_empty(self, logs_root):
        assert not store.REPORTS_ROOT.exists()
        assert store.list_reports("AAPL") == []

    def test_r17_store_well_formed_dirs_newest_first_with_completeness(self, logs_root):
        _report_dir(logs_root / "reports", "AAPL_20260610_120000", mtime=1_800_000_000)
        _report_dir(logs_root / "reports", "AAPL_20260610_090000", complete=False, mtime=1_799_996_400)
        assert store.list_reports("AAPL") == [
            {"dir": "AAPL_20260610_120000", "mtime": 1_800_000_000.0, "complete": True},
            {"dir": "AAPL_20260610_090000", "mtime": 1_799_996_400.0, "complete": False},
        ]

    def test_r17_store_plain_files_matching_the_glob_are_not_reports(self, logs_root):
        """A FILE under the reports root is never a report: a stamp-shaped one
        passes the name check and is rejected by ``is_dir``; any other is
        rejected by the stamp check."""
        reports = logs_root / "reports"
        reports.mkdir(parents=True)
        (reports / "AAPL_20260610_120000").write_text("x", encoding="utf-8")
        (reports / "AAPL_notes.txt").write_text("x", encoding="utf-8")
        _report_dir(reports, "AAPL_20260610_090000", mtime=1_800_000_000)
        assert store.list_reports("AAPL") == [
            {"dir": "AAPL_20260610_090000", "mtime": 1_800_000_000.0, "complete": True}
        ]

    def test_r17_store_unstattable_report_dir_is_skipped(self, logs_root, monkeypatch):
        """A report dir whose mtime cannot be read (permission, dangling link)
        is skipped; the other dirs still list."""
        reports = logs_root / "reports"
        good = _report_dir(reports, "AAPL_20260610_120000", mtime=1_800_000_000)
        bad = _report_dir(reports, "AAPL_20260610_110000", mtime=1_800_000_000)
        real_stat = Path.stat

        def stat(self, *args, **kwargs):
            if self == bad or bad in self.parents:
                raise PermissionError(13, "Permission denied")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)
        assert store.list_reports("AAPL") == [
            {"dir": good.name, "mtime": 1_800_000_000.0, "complete": True}
        ]


class TestStoreLoadRun:
    def test_r17_store_missing_file_is_none_even_when_the_date_is_listed(self, logs_root):
        """A DIRECTORY spelled like a states log is listed as a date by the glob
        but is not a readable run: the detail view answers the honest 404."""
        d = logs_root / "AAPL" / "YiAlphaStrategy_logs"
        d.mkdir(parents=True)
        (d / "full_states_log_2026-06-10.json").mkdir()
        assert store._dates_for("AAPL") == ["2026-06-10"]
        assert store.load_run("AAPL", "2026-06-10") is None
        assert store._latest_rating("AAPL", "2026-06-10") is None

    def test_r17_store_no_run_for_a_date_without_a_file(self, logs_root):
        _write_state_log(logs_root, "AAPL", "2026-06-10")
        assert store.load_run("AAPL", "2026-06-11") is None


# --------------------------------------------------------------------------- #
# store: ledger-backed loaders — well-formed / absent / unreadable
# --------------------------------------------------------------------------- #


def _submit(run_id: str, *, expected_return=None, prob_up=0.6):
    from yialpha.ledger.models import PredictionEntry
    from yialpha.ledger.predictions import submit_predictions

    return submit_predictions(
        run_id,
        "market",
        "BTCUSDT",
        "CONTRACT",
        [
            PredictionEntry(
                horizon_days=1,
                direction="up",
                prob_up=prob_up,
                expected_return=expected_return,
                target_price=70000.0,
                target_currency="USDT",
            )
        ],
        _AS_OF,
    )[0]


def _scored_ledger():
    """One run, one prediction, one COMPLETE legacy outcome: a scoreboard of n=1."""
    from yialpha.ledger.outcomes import write_outcome

    _create_ledger()
    prediction_id = _submit("run-1", expected_return=0.05)
    outcome_id = write_outcome(
        prediction_id,
        "run-1",
        1,
        status="complete",
        contract_price_return=0.05,
        fees=0.0,
        slippage=0.0,
        net_return=0.05,
    )
    return prediction_id, outcome_id


class TestStoreLoadRunPredictions:
    def test_r17_store_no_ledger_db_is_none(self):
        assert store.load_run_predictions("run-1") is None

    def test_r17_store_unknown_run_id_is_none(self):
        _create_ledger()
        assert store.load_run_predictions("nope") is None

    def test_r17_store_unreadable_ledger_is_none(self, monkeypatch):
        import yialpha.ledger.sqlite as ledger_sqlite

        _create_ledger()
        _refuse_reads(monkeypatch, ledger_sqlite)
        assert store.load_run_predictions("run-1") is None

    def test_r17_store_known_run_without_predictions_is_empty_but_known(self):
        _create_ledger()
        assert store.load_run_predictions("run-1") == {
            "run_id": "run-1",
            "prediction_ids": [],
            "predictions": [],
            "evidence_ids": [],
        }

    def test_r17_store_run_with_predictions_and_evidence_round_trips(self):
        from yialpha.ledger.evidence import record_evidence

        _create_ledger()
        prediction_id = _submit("run-1", expected_return=0.05)
        e1 = record_evidence(
            "run-1",
            "binance",
            "market",
            "BTCUSDT",
            "CONTRACT",
            "klines block",
            replayability="PIT_REPLAYABLE",
            available_at="2026-06-10T00:00:00+00:00",
        )
        e2 = record_evidence(
            "run-1",
            "binance",
            "funding",
            "BTCUSDT",
            "CONTRACT",
            "funding block",
            replayability="PIT_REPLAYABLE",
            available_at="2026-06-10T00:00:00+00:00",
        )
        payload = store.load_run_predictions("run-1")
        assert payload["run_id"] == "run-1"
        assert payload["prediction_ids"] == [prediction_id]
        assert payload["evidence_ids"] == sorted([e1, e2])
        [prediction] = payload["predictions"]
        created_at = prediction.pop("created_at")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", created_at)
        # compared as the JSON document the route serves
        assert json.loads(json.dumps(prediction)) == {
            "prediction_id": prediction_id,
            "run_id": "run-1",
            "analyst": "market",
            "instrument_id": "BTCUSDT",
            "prediction_scope": "CONTRACT",
            "horizon_days": 1,
            "direction": "up",
            "prob_up": 0.6,
            "expected_return": 0.05,
            "target_price": 70000.0,
            "target_currency": "USDT",
            "price_basis": None,
            "confidence": None,
            "evidence_ids": [],
            "analysis_as_of": _AS_OF,
            "schema_version": "v1",
            "feature_version": "v2",
            "original_prediction_id": None,
            "debate_revision": None,
            "revision_reason": None,
            "regime_id": None,
            "timing": None,
        }


class TestStoreLoadOutcomes:
    def test_r17_store_no_ledger_db_is_unavailable_with_hint(self):
        assert store.load_outcomes() == {
            "available": False,
            "hint": "yialpha scoreboard",
            "outcomes": [],
        }

    def test_r17_store_outcome_rows_round_trip_with_every_leg(self):
        _create_ledger()
        prediction_id = _submit("run-1")
        from yialpha.ledger.outcomes import write_outcome

        outcome_id = write_outcome(
            prediction_id,
            "run-1",
            1,
            status="complete",
            contract_price_return=0.05,
            underlying_return=0.04,
            basis_return=0.01,
            funding_pnl=-0.001,
            fees=0.0004,
            slippage=0.0002,
            net_return=0.0484,
            legs_missing=["liquidation_loss"],
            outcome_available_at="2026-06-11T00:00:00+00:00",
            ticket_id="T-1",
        )
        computed_at = _scalar("SELECT computed_at FROM outcomes WHERE outcome_id = ?", (outcome_id,))
        assert json.loads(json.dumps(store.load_outcomes())) == {
            "available": True,
            "outcomes": [
                {
                    "outcome_id": outcome_id,
                    "prediction_id": prediction_id,
                    "run_id": "run-1",
                    "ticket_id": "T-1",
                    "horizon_days": 1,
                    "status": "complete",
                    "contract_price_return": 0.05,
                    "underlying_return": 0.04,
                    "basis_return": 0.01,
                    "funding_pnl": -0.001,
                    "fees": 0.0004,
                    "slippage": 0.0002,
                    "liquidation_loss": None,
                    "net_return": 0.0484,
                    "legs_missing": ["liquidation_loss"],
                    "outcome_available_at": "2026-06-11T00:00:00+00:00",
                    "computed_at": computed_at,
                    "regime_id": None,
                    "scoring_context": None,
                }
            ],
        }

    def test_r17_store_limit_is_passed_to_the_reader(self, monkeypatch):
        import yialpha.ledger.outcomes as ledger_outcomes

        seen = []
        monkeypatch.setattr(ledger_outcomes, "all_outcomes", lambda limit: seen.append(limit) or [])
        _create_ledger()
        assert store.load_outcomes(limit=7) == {"available": True, "outcomes": []}
        store.load_outcomes()
        assert seen == [7, 500]


class TestStoreLoadCalibration:
    def test_r17_store_no_ledger_db_is_unavailable_with_hint(self):
        assert store.load_calibration() == {"available": False, "hint": "yialpha scoreboard"}

    def test_r17_store_scoreboard_failure_is_unavailable_with_hint(self, monkeypatch):
        import yialpha.ledger.scoreboard as ledger_scoreboard

        def boom():
            raise ValueError("poisoned row")

        _create_ledger()
        monkeypatch.setattr(ledger_scoreboard, "build_scoreboard", boom)
        assert store.load_calibration() == {"available": False, "hint": "yialpha scoreboard"}

    def test_r17_store_serves_the_scoreboard_unmodified(self):
        from yialpha.ledger.scoreboard import build_scoreboard

        _scored_ledger()
        got = store.load_calibration()
        expected = {"available": True, **build_scoreboard()}
        # generated_at is a wall-clock stamp: compare it by shape, not by value.
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", got.pop("generated_at")
        )
        expected.pop("generated_at")
        assert got == expected

    def test_r17_store_one_scored_row_yields_the_hand_computed_cell(self):
        """One ``up`` prediction with prob 0.6 that realized +5%: one hit, Brier
        (0.6−1)² = 0.16, log-loss −ln(0.6), one reliability bin [0.6, 0.7) and
        an ECE of |1.0 − 0.6|."""
        import math

        _scored_ledger()
        got = store.load_calibration()
        assert got["available"] is True
        assert got["scoring_versions"] == ["legacy_daily_v1"]
        assert got["mixed_scoring_versions"] is False
        assert (got["v3_min_samples_per_cell"], got["min_samples_display"]) == (30, 1)
        cell = {
            "n": 1,
            "directional_n": 1,
            "hits": 1,
            "directional_accuracy": 1.0,
            "brier_score": pytest.approx(0.16),
            "log_loss": pytest.approx(-math.log(0.6)),
            "calibration_error": pytest.approx(0.4),
            "reliability_bins": [
                {"bin_low": 0.6, "bin_high": 0.7, "n": 1, "mean_prob_up": 0.6, "accuracy": 1.0}
            ],
            "flat_n": 0,
            "missing_prob_up": 0,
            "below_v3_min_samples": True,
        }
        assert got["overall"] == cell
        assert got["by_analyst"] == {"market": cell}
        assert got["by_direction"] == {"up": cell}
        assert got["by_horizon_days"] == {"1": cell}
        assert got["by_regime"] == {"no_regime": cell}


class TestStoreLoadPositions:
    def test_r17_store_no_ledger_db_is_unavailable(self):
        assert store.load_positions() == {"available": False, "positions": []}

    def test_r17_store_position_written_by_the_resolver_path_round_trips(self):
        from yialpha.ledger.portfolio import new_position_id

        _create_ledger()
        _commit_snapshot("S-abc", {"equity": 1000.0})
        got = store.load_positions()
        assert got["available"] is True
        [position] = got["positions"]
        opened_at = position.pop("opened_at")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", opened_at)
        assert position == {
            "position_id": new_position_id("T-1", "BTCUSDT"),
            "ticket_id": "T-1",
            "run_id": "run-1",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "signed_weight": 0.1,
            "payload": {
                "ticket_id": "T-1",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "signed_weight": 0.1,
                "run_id": "run-1",
                "instrument_class": "pure_crypto_perp",
            },
        }

    def test_r17_store_degraded_read_is_unavailable_not_an_empty_book(self, monkeypatch):
        import yialpha.ledger.portfolio as ledger_portfolio

        _create_ledger()
        _refuse_reads(monkeypatch, ledger_portfolio)
        assert store.load_positions() == {"available": False, "positions": []}



# --------------------------------------------------------------------------- #
# app: path-ticker validation, run listing / detail, favicon redirect
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("ticker", "decoded"),
    [
        ("..%5C", "..\\"),
        ("%2E%2E", ".."),
        ("AAPL%5C..%5Cetc", "AAPL\\..\\etc"),
        ("A%00B", "A\x00B"),
        ("AAPL%20", "AAPL "),
        ("A" * 33, "A" * 33),
        ("AAPL%3Bx", "AAPL;x"),
        ("AAPL%2Fx", None),  # the decoded slash splits the path: no route at all
    ],
)
def test_r17_app_unsafe_path_ticker_is_404(client, ticker, decoded):
    """A path-parameter ticker that is unsafe as a filesystem component
    (traversal, drive/UNC separators, NUL, whitespace, overlong, metacharacters)
    never reaches the store: 404, with the validator's message (the decoded
    component in ``repr`` form, so a backslash shows doubled)."""
    response = client.get(f"/api/tickers/{ticker}/runs")
    assert response.status_code == 404, response.text
    if decoded is None:
        assert response.json() == {"detail": "Not Found"}
    else:
        assert response.json() == {"detail": f"unknown ticker: {decoded!r}"}


def test_r17_app_safe_ticker_without_runs_lists_empty(client):
    response = client.get("/api/tickers/AAPL/runs")
    assert response.status_code == 200, response.text
    assert response.json() == {"ticker": "AAPL", "dates": [], "date_ratings": [], "reports": []}


def test_r17_app_runs_listing_is_exact(client, logs_root):
    _write_state_log(logs_root, "AAPL", "2026-06-10", decision="**Rating**: Buy")
    _write_state_log(logs_root, "AAPL", "2026-06-12_perp", decision="**Rating**: Sell")
    _write_state_log(logs_root, "AAPL", "2026-06-11", raw=b"{oops")
    _report_dir(logs_root / "reports", "AAPL_20260612_120000", mtime=1_800_000_000)
    response = client.get("/api/tickers/AAPL/runs")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "ticker": "AAPL",
        "dates": ["2026-06-10", "2026-06-11", "2026-06-12_perp"],
        "date_ratings": [
            {"date": "2026-06-10", "rating": "Buy"},
            {"date": "2026-06-11", "rating": None},
            {"date": "2026-06-12_perp", "rating": "Sell"},
        ],
        "reports": [{"dir": "AAPL_20260612_120000", "mtime": 1_800_000_000.0, "complete": True}],
    }


_DECISION = (
    "**Rating**: Buy\n\n## Quantitative Risk Overlay\n"
    "- **Action**: overweight\n"
    "- **Target Weight**: 12.5% (1,250)\n"
    "- **Stop Loss**: 185.25\n"
    "- **Entry Reference**: 201.5\n"
    "- **Drawdown Regime**: NORMAL\n"
    "- **Rationale**: sized to volatility\n"
)
_FULL_STATE = {
    "company_of_interest": "AAPL",
    "trade_date": "2026-06-10",
    "asset_type": "stock",
    "market_report": "mkt",
    "sentiment_report": "snt",
    "news_report": "nws",
    "fundamentals_report": "fnd",
    "investment_debate_state": {
        "bull_history": "b",
        "bear_history": "be",
        "history": "h",
        "current_response": "cr",
        "judge_decision": "jd",
    },
    "trader_investment_decision": "trd",
    "risk_debate_state": {
        "aggressive_history": "a",
        "conservative_history": "c",
        "neutral_history": "n",
        "history": "h2",
        "judge_decision": "jd2",
    },
    "investment_plan": "plan",
    "final_trade_decision": _DECISION,
}


def test_r17_app_run_detail_maps_every_section_from_the_states_log(client, logs_root):
    _write_state_log(logs_root, "AAPL", "2026-06-10", raw=json.dumps(_FULL_STATE).encode("utf-8"))
    response = client.get("/api/tickers/AAPL/runs/2026-06-10")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "ticker": "AAPL",
        "trade_date": "2026-06-10",
        "rating": "Buy",
        "asset_type": "stock",
        "company_of_interest": "AAPL",
        "sections": {
            "market_report": "mkt",
            "positioning_report": None,
            "sentiment_report": "snt",
            "news_report": "nws",
            "fundamentals_report": "fnd",
            "investment_debate": _FULL_STATE["investment_debate_state"],
            "trader_decision": "trd",
            "risk_debate": _FULL_STATE["risk_debate_state"],
            "investment_plan": "plan",
            "final_trade_decision": _DECISION,
        },
        "overlay": {
            "action": "overweight",
            "target_weight": "12.5%",
            "position_value": "1,250",
            "stop_loss": "185.25",
            "entry": "201.5",
            "regime": "NORMAL",
            "rationale": "sized to volatility",
        },
        "data_quality": None,
        "node_perf": None,
    }


def test_r17_app_run_detail_serves_the_venue_suffixed_log_with_its_own_perf(client, logs_root):
    """A crypto venue run is two first-class files (states log + node perf)
    under the suffixed date key; the unsuffixed key stays a separate run."""
    state = {**_FULL_STATE, "asset_type": "crypto_perp", "positioning_report": "pos"}
    _write_state_log(logs_root, "BTCUSDT", "2026-06-12_perp", raw=json.dumps(state).encode())
    d = logs_root / "BTCUSDT" / "YiAlphaStrategy_logs"
    (d / "node_perf_2026-06-12_perp.json").write_text('{"nodes": [{"name": "market"}]}', encoding="utf-8")
    (d / "node_perf_2026-06-12.json").write_text('{"nodes": [{"name": "unsuffixed"}]}', encoding="utf-8")
    response = client.get("/api/tickers/BTCUSDT/runs/2026-06-12_perp")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["asset_type"] == "crypto_perp"
    assert payload["sections"]["positioning_report"] == "pos"
    assert payload["node_perf"] == {"nodes": [{"name": "market"}]}
    assert client.get("/api/tickers/BTCUSDT/runs/2026-06-12").status_code == 404


@pytest.mark.parametrize(
    ("date", "detail"),
    [
        ("2026-06-11", "no run for AAPL on 2026-06-11"),
        ("9999-99-99", "no run for AAPL on 9999-99-99"),  # shape-valid, simply absent
        ("not-a-date", "bad date: 'not-a-date'"),
        ("2026-06-10_spotx", "bad date: '2026-06-10_spotx'"),
        ("..%5C..%5Capp", "bad date: " + repr("..\\..\\app")),
        ("2026-06-10%00", "bad date: " + repr("2026-06-10\x00")),
    ],
)
def test_r17_app_run_detail_missing_or_bad_date_is_404(client, logs_root, date, detail):
    _write_state_log(logs_root, "AAPL", "2026-06-10")
    response = client.get(f"/api/tickers/AAPL/runs/{date}")
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": detail}


def test_r17_app_favicon_ico_redirects_to_the_svg(client):
    """The legacy favicon path answers a 302 to the SVG icon (never a 404);
    the redirect carries the security headers like every other response."""
    response = client.get("/favicon.ico", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/static/favicon.svg"
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    final = client.get("/static/favicon.svg")
    assert final.status_code == 200
    assert final.headers["content-type"].startswith("image/svg+xml")
    assert final.content.startswith(b"<svg")


# --------------------------------------------------------------------------- #
# app: static mounts, error responses, cwd guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", ["/reports/", "/static/", "/static/vendor", "/favicon.ico/x"])
def test_r17_app_directory_requests_are_404(client, path):
    """Both static mounts are file-only (``html=False``): no directory listing."""
    response = client.get(path)
    assert response.status_code == 404, response.text


@pytest.mark.parametrize(
    "path",
    [
        "/static/../app.py",
        "/static/..%5Capp.py",
        "/static/%2E%2E/app.py",
        "/static/..%2F..%2Fwebapp.py",
        "/reports/..%5C..%5Cpyproject.toml",
        "/reports/%2E%2E/%2E%2E/etc/passwd",
    ],
)
def test_r17_app_traversal_out_of_the_static_roots_is_404(client, path):
    response = client.get(path)
    assert response.status_code == 404, response.text
    assert b"fastapi" not in response.content


def test_r17_app_unknown_route_is_a_json_404_with_security_headers(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["referrer-policy"] == "no-referrer"


def test_r17_app_index_serves_the_spa(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert b"/static/app.js" in response.content


@pytest.mark.parametrize(
    ("method", "path", "allow"),
    [("GET", "/api/analyze", "POST"), ("POST", "/api/tickers", "GET"), ("POST", "/api/health", "GET")],
)
def test_r17_app_wrong_method_is_405_without_running_the_handler(client, monkeypatch, method, path, allow):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("handler ran for a wrong-method request")

    monkeypatch.setattr(store, "list_tickers", fail_if_called)
    monkeypatch.setattr(health, "run_health", fail_if_called)
    monkeypatch.setattr(runner, "spawn", fail_if_called)
    response = client.request(method, path, **({"json": {}} if method == "POST" else {}))
    assert response.status_code == 405, response.text
    assert allow in response.headers["allow"]


class TestAssertCwdIsProjectRoot:
    def test_a_project_root_passes_silently(self, monkeypatch, tmp_path, capsys):
        (tmp_path / "yialpha").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "run_robust.py").write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        app_module._assert_cwd_is_project_root()
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize(
        "present",
        [(), ("yialpha",), ("run_robust",)],
        ids=["both-missing", "only-run-robust-present", "only-yialpha-present"],
    )
    def test_missing_pieces_exit_2_naming_them(self, monkeypatch, tmp_path, capsys, present):
        """``web/app.py`` refuses to start unless cwd is the project root:
        exit code 2 and a stderr message naming the cwd and every missing piece."""
        if "yialpha" in present:
            (tmp_path / "yialpha").mkdir()
        if "run_robust" in present:
            (tmp_path / "scripts").mkdir()
            (tmp_path / "scripts" / "run_robust.py").write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exit_info:
            app_module._assert_cwd_is_project_root()
        assert exit_info.value.code == 2
        err = capsys.readouterr().err
        assert err.startswith("\nERROR: web/app.py must run from the YiAlpha project root.\n")
        assert f"  cwd     = {tmp_path}\n" in err
        expected = [
            piece for piece, there in (("yialpha/", "yialpha" in present), ("scripts/run_robust.py", "run_robust" in present)) if not there
        ]
        assert f"  missing = {', '.join(expected)}\n" in err
        assert "load_dotenv(usecwd=True)" in err
        assert err.rstrip("\n").endswith("python web/app.py")


# --------------------------------------------------------------------------- #
# app: ledger-backed read-only routes
# --------------------------------------------------------------------------- #


def test_r17_app_run_predictions_route(client):
    assert client.get("/api/runs/run-1/predictions").json() == {"detail": "no recorded run run-1"}
    _create_ledger()
    assert client.get("/api/runs/nope/predictions").status_code == 404
    _submit("run-1", expected_return=0.05)
    response = client.get("/api/runs/run-1/predictions")
    assert response.status_code == 200, response.text
    assert response.json() == json.loads(json.dumps(store.load_run_predictions("run-1")))


@pytest.mark.parametrize(
    ("query", "limit"),
    [("", 500), ("?limit=7", 7), ("?limit=2000", 2000), ("?limit=99999", 2000), ("?limit=0", 1), ("?limit=-5", 1)],
)
def test_r17_app_outcomes_route_clamps_the_limit(client, monkeypatch, query, limit):
    seen = []

    def fake_load(limit=500):
        seen.append(limit)
        return {"available": False, "hint": "yialpha scoreboard", "outcomes": []}

    monkeypatch.setattr(store, "load_outcomes", fake_load)
    response = client.get(f"/api/outcomes{query}")
    assert response.status_code == 200, response.text
    assert response.json() == {"available": False, "hint": "yialpha scoreboard", "outcomes": []}
    assert seen == [limit]


@pytest.mark.parametrize("query", ["?limit=abc", "?limit=1.5", "?limit=1e3", "?limit="])
def test_r17_app_outcomes_route_non_integer_limit_is_422(client, query):
    response = client.get(f"/api/outcomes{query}")
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["query", "limit"]


def test_r17_app_outcomes_route_serves_the_rows(client):
    prediction_id, outcome_id = _scored_ledger()
    response = client.get("/api/outcomes")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["available"] is True
    assert [(o["outcome_id"], o["prediction_id"], o["net_return"]) for o in payload["outcomes"]] == [
        (outcome_id, prediction_id, 0.05)
    ]
    assert response.json() == json.loads(json.dumps(store.load_outcomes()))


def test_r17_app_calibration_route(client):
    assert client.get("/api/calibration").json() == {"available": False, "hint": "yialpha scoreboard"}
    _scored_ledger()
    payload = client.get("/api/calibration").json()
    assert payload["available"] is True
    assert payload["overall"]["n"] == 1
    assert set(payload) == {
        "available", "generated_at", "v3_min_samples_per_cell", "min_samples_display",
        "scoring_version_filter", "scoring_versions", "mixed_scoring_versions", "overall",
        "by_analyst", "by_instrument_class", "by_horizon_days", "by_direction",
        "by_evidence_bucket", "by_regime", "by_scoring_version",
    }


def test_r17_app_ticket_route(client):
    assert client.get("/api/tickets/T-1").status_code == 404
    assert client.get("/api/tickets/T-1").json() == {"detail": "no ticket T-1"}
    _create_ledger()
    assert client.get("/api/tickets/T-1").json() == {"detail": "no ticket T-1"}
    from yialpha.ledger.tickets_mirror import attach_ticket

    attach_ticket("T-1", "D-1", "run-1", {"ticket_id": "T-1", "side": "LONG"}, "v1")
    response = client.get("/api/tickets/T-1")
    assert response.status_code == 200, response.text
    assert response.json() == store.load_ticket("T-1")


def test_r17_app_portfolio_snapshot_route(client):
    assert client.get("/api/portfolio/snapshots/S-abc").json() == {"detail": "no snapshot S-abc"}
    _create_ledger()
    assert client.get("/api/portfolio/snapshots/S-abc").status_code == 404
    _commit_snapshot("S-abc", {"equity": 1000.0, "positions": []})
    response = client.get("/api/portfolio/snapshots/S-abc")
    assert response.status_code == 200, response.text
    assert response.json() == {"equity": 1000.0, "positions": []}


def test_r17_app_positions_route(client):
    assert client.get("/api/positions").json() == {"available": False, "positions": []}
    _create_ledger()
    assert client.get("/api/positions").json() == {"available": True, "positions": []}
    _commit_snapshot("S-abc", {"equity": 1000.0})
    payload = client.get("/api/positions").json()
    assert payload["available"] is True
    assert [(p["symbol"], p["side"], p["signed_weight"]) for p in payload["positions"]] == [
        ("BTCUSDT", "LONG", 0.1)
    ]


# --------------------------------------------------------------------------- #
# app: /api/health
# --------------------------------------------------------------------------- #


def test_r17_app_health_route_passes_the_preflight_result_through(client, monkeypatch):
    canned = {"ok": False, "checks": [{"name": "dep socks", "ok": False, "hint": "pip install x"}]}

    def fake_run_health():
        return canned

    monkeypatch.setattr(health, "run_health", fake_run_health)
    response = client.get("/api/health")
    assert response.status_code == 200, response.text
    assert response.json() == canned
    # sync def: Starlette threadpools the blocking probes off the event loop
    # that drives the analysis watcher.
    import inspect

    assert inspect.iscoroutinefunction(app_module.api_health) is False


def test_r17_app_health_route_serves_the_real_preflight_without_leaking_the_key(client, monkeypatch):
    """The shared preflight (same implementation as ``run_baseline
    --preflight``) runs behind the route with every network surface stubbed;
    the API key that authorises the provider probe never appears in the body."""
    import types

    import httpx

    import yialpha.monitoring.preflight as preflight

    secret = "sk-r17-secret-0123456789"
    monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    monkeypatch.delenv("YIALPHA_LLM_BACKEND_URL", raising=False)
    monkeypatch.setenv("SOCKS5_PROXY", "socks5h://127.0.0.1:1080")
    for var in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        monkeypatch.delenv(var, raising=False)
    probes = []
    tickers = []

    class _FakeTicker:
        def __init__(self, ticker):
            tickers.append(ticker)

        def history(self, **kwargs):
            return [1.0, 2.0]

    def create_connection(addr, timeout=None):
        probes.append(addr)
        return types.SimpleNamespace(close=lambda: None)

    def fake_get(url, headers=None, timeout=None):
        probes.append((url, headers["Authorization"]))
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(
        preflight, "socket", types.SimpleNamespace(create_connection=create_connection)
    )
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_FakeTicker))
    monkeypatch.setattr(httpx, "get", fake_get)

    response = client.get("/api/health")
    assert response.status_code == 200, response.text
    payload = response.json()
    by_name = {c["name"]: c for c in payload["checks"]}
    assert set(by_name) >= {
        "DEEPSEEK_API_KEY set",
        "YIALPHA_LLM_PROVIDER set",
        "proxy 127.0.0.1:1080 reachable",
        "yfinance pulled SPY",
        "DeepSeek API reachable",
    }
    for name in ("DEEPSEEK_API_KEY set", "YIALPHA_LLM_PROVIDER set", "proxy 127.0.0.1:1080 reachable"):
        assert by_name[name]["ok"] is True, by_name[name]
    assert by_name["yfinance pulled SPY"]["ok"] is True
    # Every hint in the body is a static operator hint / status code.
    assert all(c["hint"] in ("", "HTTP 200", ".env not loaded — start from the project root (dir with .env)", "confirm V2Ray/Xray on that port", ".env missing YIALPHA_LLM_PROVIDER", "proxy / PySocks / Yahoo rate-limit") for c in payload["checks"])
    assert by_name["DeepSeek API reachable"] == {
        "name": "DeepSeek API reachable", "ok": True, "hint": "HTTP 200",
    }
    assert payload["ok"] is True
    assert ("127.0.0.1", 1080) in probes
    assert ("https://api.deepseek.com/v1/models", f"Bearer {secret}") in probes
    assert tickers == ["SPY"]
    assert secret not in response.text
    assert "Bearer" not in response.text


# --------------------------------------------------------------------------- #
# app + runner: single analysis slot under concurrent submits (no real subprocess)
# --------------------------------------------------------------------------- #


class _BlockingStdout:
    async def readline(self):
        import asyncio

        await asyncio.Event().wait()


class _BlockingProc:
    """A child that is still running: stdout never yields, ``wait`` would return 0."""

    pid = 4242

    def __init__(self):
        self.stdout = _BlockingStdout()

    async def wait(self):
        return 0


def test_r17_app_single_slot_holds_under_concurrent_submits(monkeypatch, logs_root):
    """Eight browsers submit at once: the reservation inside ``runner.spawn``
    precedes its first ``await``, so exactly one child is launched and every
    other request is 409. After that task ends the slot is free again."""
    import asyncio
    import threading

    launched = []

    async def fake_exec(*cmd, **kwargs):
        launched.append(list(cmd))
        return _BlockingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    statuses = []
    lock = threading.Lock()

    with TestClient(app) as shared:
        def submit():
            response = shared.post("/api/analyze", json={"ticker": "AAPL", "date": "2026-06-10"})
            with lock:
                statuses.append((response.status_code, response.json()))

        threads = [threading.Thread(target=submit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(launched) == 1
        assert launched[0][launched[0].index("--tickers") + 1] == "AAPL"
        accepted = [body for code, body in statuses if code == 200]
        refused = [body for code, body in statuses if code == 409]
        assert len(refused) == 7
        assert all(body == {"detail": "an analysis is already running"} for body in refused)
        [body] = accepted
        task_id = body["task_id"]
        assert runner.registry.is_busy() is True
        assert runner.registry.get(task_id).pid == 4242
        assert shared.get(f"/api/tasks/{task_id}").json()["status"] == "running"

        async def finish(watcher):
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

        (watcher,) = list(runner._WATCH_TASKS)
        shared.portal.call(finish, watcher)
        assert runner.registry.is_busy() is False

        again = shared.post("/api/analyze", json={"ticker": "MSFT", "date": "2026-06-10"})
        assert again.status_code == 200, again.text
        assert len(launched) == 2
        (watcher,) = list(runner._WATCH_TASKS)
        shared.portal.call(finish, watcher)
    assert runner.registry.is_busy() is False


def test_r17_app_spawn_returning_an_unregistered_id_is_500(client, monkeypatch):
    """``runner.spawn`` returning an id the registry does not know is reported
    as a 500 (type-honest branch), never as a 200 with a dangling task id."""
    async def ghost_spawn(ticker, analysis_date, asset_type, language="en"):
        return "ghost-task"

    monkeypatch.setattr(runner, "spawn", ghost_spawn)
    response = client.post("/api/analyze", json={"ticker": "AAPL", "date": "2026-06-10"})
    assert response.status_code == 500
    assert response.json() == {"detail": "task registration failed"}


@pytest.mark.parametrize(
    ("ticker", "asset_type", "detail_start"),
    [
        ("-AAPL", "auto", "Unsupported market"),
        ("--", "auto", "Unsupported market"),
        ("A-", "auto", "Unsupported market"),
        ("...", "auto", "bad ticker"),
        ("=AAPL", "auto", "Unsupported market"),
        ("A_B", "auto", "Unsupported market"),
        ("ＡＡＰＬ", "auto", "Unsupported market"),
        ("．AAPL", "auto", "bad ticker"),
        ("A\x00A", "auto", "bad ticker"),
        ("C:\\AAPL", "auto", "bad ticker"),
        ("A" * 33, "auto", "bad ticker"),
        ("AAPL/", "auto", "bad ticker"),
        ("..\\", "crypto_perp", "bad ticker"),
    ],
)
def test_r17_app_analyze_option_like_and_non_ascii_tickers_never_spawn(client, monkeypatch, ticker, asset_type, detail_start):
    """Option-shaped tickers (``-AAPL``, ``--``) cannot reach the child's argv:
    the market gate rejects them before spawn; non-ASCII and NUL die at the
    charset check; the venue normalizer never turns them into a spawnable
    symbol."""
    def fail_if_called(*args, **kwargs):
        raise AssertionError(f"invalid request reached process spawn: {args!r}")

    monkeypatch.setattr(runner, "spawn", fail_if_called)
    response = client.post("/api/analyze", json={"ticker": ticker, "date": "2026-06-10", "asset_type": asset_type})
    assert response.status_code == 400, response.text
    assert response.json()["detail"].startswith(detail_start)


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        (" aapl\r\n", "AAPL"),
        ("AAPL\u00a0", "AAPL"),
        ("BTC-USD-", "BTC-USD"),
        ("ETH-BTC", "ETH-BTC"),
    ],
)
def test_r17_app_analyze_surrounding_whitespace_and_trailing_dash_are_normalized(client, monkeypatch, ticker, expected):
    calls = []

    async def fake_spawn(ticker, analysis_date, asset_type, language="en"):
        calls.append(ticker)
        runner.registry.tasks["t"] = runner.TaskState(
            task_id="t", ticker=ticker, date=analysis_date, asset_type=asset_type, started_at=1.0
        )
        return "t"

    monkeypatch.setattr(runner, "spawn", fake_spawn)
    response = client.post("/api/analyze", json={"ticker": ticker, "date": "2026-06-10"})
    assert response.status_code == 200, response.text
    assert calls == [expected]



# --------------------------------------------------------------------------- #
# Defects found in round 17 (red on the unmodified tree): non-finite numbers
# from on-disk JSON / ledger rows, and an unreadable ledger
# --------------------------------------------------------------------------- #

_NAN = float("nan")
_INF = float("inf")
_OVERFLOW_STATE = (
    b'{"final_trade_decision": "**Rating**: Buy", '
    b'"data_quality": {"ratio": 1e999, "core_sentinel_count": 2, '
    b'"sentinels": [{"detail": "x", "elapsed": -1e400}]}}'
)

class TestOverflowNumberLiteralIsServedAsNull:
    """``1e999`` is not one of the three bare constants ``parse_constant`` sees:
    ``json`` parses it through ``float()`` into ``inf``, which the response
    render (``allow_nan=False``) rejects. The round-11/16 guard only covered
    ``NaN`` / ``Infinity`` / ``-Infinity``; one such number 500'd the page."""

    @pytest.mark.parametrize("literal", ["1e999", "1E999", "-1e999", "1e309", "1.8e308"])
    def test_accuracy_artifact_overflow_is_null(self, client_no_raise, logs_root, literal):
        acc = logs_root / "accuracy"
        acc.mkdir(parents=True)
        (acc / "accuracy_report.json").write_text(
            f'{{"hit_rate": {literal}, "n": 4}}', encoding="utf-8"
        )
        response = client_no_raise.get("/api/accuracy")
        assert response.status_code == 200, response.text
        assert response.json() == {"available": True, "hit_rate": None, "n": 4}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1e308", 1e308),
            ("1.7976931348623157e308", 1.7976931348623157e308),
            ("2.5", 2.5),
            ("-0.5", -0.5),
            ("1e-400", 0.0),  # underflow is a finite zero, kept
            ("123456789012345678901234567890", 123456789012345678901234567890),
            ("NaN", None),
            ("Infinity", None),
            ("-Infinity", None),
            ("1e999", None),
        ],
    )
    def test_only_overflow_and_bare_constants_are_nulled(self, client_no_raise, logs_root, text, expected):
        acc = logs_root / "accuracy"
        acc.mkdir(parents=True)
        (acc / "accuracy_report.json").write_text(f'{{"x": {text}}}', encoding="utf-8")
        response = client_no_raise.get("/api/accuracy")
        assert response.status_code == 200, response.text
        assert response.json() == {"available": True, "x": expected}

    def test_states_log_nested_overflow_is_null(self, client_no_raise, logs_root):
        _write_state_log(logs_root, "AAPL", "2026-06-10", raw=_OVERFLOW_STATE)
        response = client_no_raise.get("/api/tickers/AAPL/runs/2026-06-10")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["rating"] == "Buy"
        assert payload["data_quality"] == {
            "ratio": None,
            "core_sentinel_count": 2,
            "sentinels": [{"detail": "x", "elapsed": None}],
        }

    def test_node_perf_overflow_is_null(self, client_no_raise, logs_root):
        _write_state_log(logs_root, "AAPL", "2026-06-10")
        (logs_root / "AAPL" / "YiAlphaStrategy_logs" / "node_perf_2026-06-10.json").write_text(
            '{"nodes": [{"name": "market", "tokens_per_s": 1e999, "wall_s": 12.5}]}', encoding="utf-8"
        )
        response = client_no_raise.get("/api/tickers/AAPL/runs/2026-06-10")
        assert response.status_code == 200, response.text
        assert response.json()["node_perf"] == {
            "nodes": [{"name": "market", "tokens_per_s": None, "wall_s": 12.5}]
        }


def _served(response):
    """A 200 already proves the body is strict JSON (the render is
    ``allow_nan=False``: a surviving non-finite float would have been a 500)."""
    assert response.status_code == 200, response.text
    return response.json()


class TestNonFiniteLedgerRowIsServedAsNull:
    """The ledger writers do not validate finiteness: ``attach_ticket`` /
    ``commit_final_ticket`` dump payloads with ``json.dumps`` defaults (bare
    ``NaN`` / ``Infinity`` literals land in the DB), a REAL column stores
    ``inf``, and ``expected_return`` / the outcome legs are unvalidated. The
    response render is ``allow_nan=False``, so ONE such row 500'd the whole
    endpoint (for positions: the whole book)."""

    def test_ticket_payload_from_the_real_ticket_model(self, client_no_raise):
        from yialpha.ledger.tickets_mirror import attach_ticket
        from yialpha.tickets import ExecutionTicket

        _create_ledger()
        ticket = ExecutionTicket(symbol="BTCUSDT", stop_price=_NAN, leverage=_INF)
        attach_ticket("T-1", "D-1", "run-1", ticket.model_dump(), "v1")
        raw = _scalar("SELECT payload FROM tickets WHERE ticket_id = 'T-1'")
        assert "NaN" in raw and "Infinity" in raw  # the writer really stored the literals
        response = client_no_raise.get("/api/tickets/T-1")
        payload = _served(response)["payload"]
        assert payload["symbol"] == "BTCUSDT"
        assert payload["stop_price"] is None
        assert payload["leverage"] is None

    def test_ticket_payload_nested_and_finite_values(self, client_no_raise):
        from yialpha.ledger.tickets_mirror import attach_ticket

        _create_ledger()
        attach_ticket(
            "T-2",
            None,
            None,
            {"side": "LONG", "entry": 65432.5, "cost_detail": {"fees": _INF, "slippage": 0.0},
             "veto_reasons": ["x"], "quote_fx": {"rate": -_INF}},
            "v1",
        )
        response = client_no_raise.get("/api/tickets/T-2")
        assert _served(response)["payload"] == {
            "side": "LONG",
            "entry": 65432.5,
            "cost_detail": {"fees": None, "slippage": 0.0},
            "veto_reasons": ["x"],
            "quote_fx": {"rate": None},
        }

    def test_portfolio_snapshot_payload(self, client_no_raise):
        _create_ledger()
        _commit_snapshot(
            "S-nan",
            {"equity": _NAN, "advisory": {"sharpe": _INF, "trades": 3}, "positions": []},
        )
        response = client_no_raise.get("/api/portfolio/snapshots/S-nan")
        assert _served(response) == {"equity": None, "advisory": {"sharpe": None, "trades": 3}, "positions": []}

    def test_one_position_with_an_infinite_weight_does_not_kill_the_book(self, client_no_raise):
        from yialpha.ledger.portfolio import commit_final_ticket, new_position_id

        _create_ledger()
        _commit_snapshot("S-ok", {"equity": 1000.0})  # BTCUSDT LONG 0.1
        assert commit_final_ticket(
            ticket_payload={"ticket_id": "T-bad"},
            ticket_id="T-bad",
            decision_id="D-2",
            run_id="run-1",
            ticket_version="v1",
            snapshot_payload={"equity": 1000.0},
            snapshot_id="S-bad",
            position_symbol="ETHUSDT",
            position_side="SHORT",
            position_signed_weight=-_INF,
            open_position=True,
            close_existing=False,
            instrument_class="pure_crypto_perp",
        )
        assert _scalar("SELECT signed_weight FROM positions WHERE symbol = 'ETHUSDT'") == -_INF
        response = client_no_raise.get("/api/positions")
        payload = _served(response)
        assert payload["available"] is True
        [healthy, poisoned] = sorted(payload["positions"], key=lambda p: p["symbol"] == "ETHUSDT")
        assert (healthy["symbol"], healthy["side"], healthy["signed_weight"]) == ("BTCUSDT", "LONG", 0.1)
        assert healthy["payload"]["signed_weight"] == 0.1
        assert poisoned["position_id"] == new_position_id("T-bad", "ETHUSDT")
        assert poisoned["signed_weight"] is None
        assert poisoned["payload"]["signed_weight"] is None
        assert poisoned["payload"]["instrument_class"] == "pure_crypto_perp"

    def test_prediction_expected_return(self, client_no_raise):
        _create_ledger()
        prediction_id = _submit("run-1", expected_return=_INF)
        assert _scalar("SELECT expected_return FROM predictions WHERE prediction_id = ?", (prediction_id,)) == _INF
        [prediction] = _served(client_no_raise.get("/api/runs/run-1/predictions"))["predictions"]
        assert prediction["prediction_id"] == prediction_id
        assert prediction["expected_return"] is None
        assert prediction["prob_up"] == 0.6 and prediction["target_price"] == 70000.0

    def test_outcome_leg(self, client_no_raise):
        from yialpha.ledger.outcomes import write_outcome

        _create_ledger()
        prediction_id = _submit("run-1")
        outcome_id = write_outcome(
            prediction_id,
            "run-1",
            1,
            status="incomplete",
            contract_price_return=0.05,
            underlying_return=_INF,
            fees=_NAN,
            slippage=0.0,
            net_return=0.05,
        )
        [outcome] = _served(client_no_raise.get("/api/outcomes"))["outcomes"]
        assert outcome["outcome_id"] == outcome_id
        assert outcome["underlying_return"] is None
        assert outcome["fees"] is None  # NaN reached SQLite as NULL
        assert outcome["contract_price_return"] == 0.05


def _ledger_file() -> Path:
    from yialpha.ledger.sqlite import ledger_db_path

    return Path(ledger_db_path())


class TestUnreadableLedgerIsUnavailable:
    """``ledger_exists()`` only checks that the file is there. A present but
    unreadable ledger (never migrated, corrupt, locked) makes the ledger
    readers degrade to an empty result internally — which the aggregate
    loaders served as an EMPTY-BUT-AVAILABLE table (the "phantom empty book"
    round-10 fixed for positions only). The honesty contract: ``available``
    is true only when the read actually succeeded."""

    @pytest.mark.parametrize("fill", [b"", b"not a sqlite database " * 40], ids=["zero-byte", "garbage"])
    def test_present_but_unreadable_file_is_unavailable(self, client, fill):
        _ledger_file().parent.mkdir(parents=True, exist_ok=True)
        _ledger_file().write_bytes(fill)
        unavailable = {"available": False, "hint": "yialpha scoreboard"}
        assert client.get("/api/calibration").json() == unavailable
        assert client.get("/api/outcomes").json() == {**unavailable, "outcomes": []}

    def test_refused_read_is_unavailable(self, client, monkeypatch):
        import yialpha.ledger.sqlite as ledger_sqlite

        _scored_ledger()
        monkeypatch.setattr(
            ledger_sqlite, "get_connection", _locked_connection
        )
        unavailable = {"available": False, "hint": "yialpha scoreboard"}
        assert client.get("/api/calibration").json() == unavailable
        assert client.get("/api/outcomes").json() == {**unavailable, "outcomes": []}

    def test_readable_ledger_without_scored_rows_is_still_available(self, client):
        """The guard must not over-correct: a ledger that reads fine and simply
        has no complete outcomes yet is an honest empty-but-available answer."""
        _create_ledger()
        _submit("run-1")
        assert client.get("/api/outcomes").json() == {"available": True, "outcomes": []}
        calibration = client.get("/api/calibration").json()
        assert calibration["available"] is True
        assert calibration["overall"]["n"] == 0


def _locked_connection(*, readonly=False):
    raise sqlite3.OperationalError("database is locked")



# --------------------------------------------------------------------------- #
# app: the documented HTTP surface and the script launch (module level / __main__)
# --------------------------------------------------------------------------- #


def test_r17_app_route_surface_is_exactly_the_read_only_api_plus_one_write():
    """Every route the server exposes. Only ``POST /api/analyze`` writes
    anything (it spawns the analysis); everything else is read-only, and the
    two static mounts are file-only (no directory listing)."""
    from fastapi.routing import APIRoute
    from starlette.routing import Mount
    from starlette.staticfiles import StaticFiles

    surface = {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    assert surface == {
        ("GET", "/"),
        ("GET", "/favicon.ico"),
        ("GET", "/api/accuracy"),
        ("GET", "/api/calibration"),
        ("GET", "/api/compare"),
        ("GET", "/api/health"),
        ("GET", "/api/outcomes"),
        ("GET", "/api/portfolio/snapshots/{snapshot_id}"),
        ("GET", "/api/positions"),
        ("GET", "/api/runs/{run_id}/predictions"),
        ("GET", "/api/tasks/{task_id}"),
        ("GET", "/api/tickets/{ticket_id}"),
        ("GET", "/api/tickers"),
        ("GET", "/api/tickers/{ticker}/runs"),
        ("GET", "/api/tickers/{ticker}/runs/{date}"),
        ("POST", "/api/analyze"),
    }
    mounts = [route for route in app.routes if isinstance(route, Mount)]
    assert [(m.path, m.name) for m in mounts] == [("/static", "static"), ("/reports", "reports")]
    assert all(isinstance(m.app, StaticFiles) and m.app.html is False for m in mounts)


_APP_PATH = Path(app_module.__file__).resolve()


def _launch(cwd, *, out_file=None, prelude=""):
    """Run ``web/app.py`` the way ``python web/app.py`` does (as ``__main__``),
    in an isolated child interpreter whose ``sys.path`` does not hold the
    project root — the launch form the module's bootstrap exists for."""
    import subprocess

    env = {**os.environ, "YIALPHA_RESULTS_DIR": str(cwd / "child-results")}
    code = prelude + "import runpy\n" f"runpy.run_path({str(_APP_PATH)!r}, run_name='__main__')\n"
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, *([str(out_file)] if out_file else [])],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )


_UVICORN_STUB = (
    "import json, sys, types\n"
    "def run(app, **kwargs):\n"
    "    with open(sys.argv[1], 'w', encoding='utf-8') as fh:\n"
    "        json.dump({'app_type': type(app).__name__, 'title': app.title,\n"
    "                  'kwargs': kwargs, 'routes': len(app.routes)}, fh)\n"
    "sys.modules['uvicorn'] = types.SimpleNamespace(run=run)\n"
)


def test_r17_app_main_block_hands_the_app_object_to_uvicorn_on_loopback(tmp_path):
    """``python web/app.py`` starts uvicorn with the APP OBJECT (never the
    ``web.app:app`` import string — a re-import would re-run the cwd assert
    and re-mount static) and binds loopback only. uvicorn is stubbed: no
    server, no socket."""
    out = tmp_path / "uvicorn.json"
    completed = _launch(_APP_PATH.parent.parent, out_file=out, prelude=_UVICORN_STUB)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "app_type": "FastAPI",
        "title": "YiAlpha Web",
        "kwargs": {"host": "127.0.0.1", "port": 8000, "log_level": "info"},
        "routes": len(app.routes),
    }
    assert not (tmp_path / "child-results" / "reports").exists()  # no lifespan ran


def test_r17_app_launched_outside_the_project_root_exits_2(tmp_path):
    """The real launch path: imported from a directory that is not the project
    root, the module writes the operator hint to stderr and exits 2 before
    uvicorn is ever reached."""
    (tmp_path / "unrelated.txt").write_text("x", encoding="utf-8")
    completed = _launch(tmp_path)
    assert completed.returncode == 2, completed.stderr[-2000:]
    assert "ERROR: web/app.py must run from the YiAlpha project root." in completed.stderr
    assert f"  cwd     = {tmp_path}\n" in completed.stderr
    assert "  missing = yialpha/, scripts/run_robust.py\n" in completed.stderr


# --------------------------------------------------------------------------- #
# runner: a report dir the listing cannot stat must not fail a finished run
# --------------------------------------------------------------------------- #


class _DoneStdout:
    """A child's stdout after it exited cleanly: one banner line, then EOF."""

    def __init__(self):
        self._lines = [b"[AAPL] done\n"]

    async def readline(self):
        return self._lines.pop(0) if self._lines else b""


class _DoneProc:
    pid = 4242

    def __init__(self):
        self.stdout = _DoneStdout()

    async def wait(self):
        return 0


def test_r17_runner_unstattable_report_dir_does_not_fail_a_finished_run(monkeypatch, logs_root):
    """After a rc-0 child, the watcher attaches the newest report dir through
    ``store.list_reports``. An entry whose metadata cannot be read used to
    raise out of the listing, so the watcher marked the SUCCESSFUL analysis
    ``error`` ("post-run report lookup failed")."""
    import asyncio

    registry = runner.TaskRegistry()
    monkeypatch.setattr(runner, "registry", registry)

    async def fake_exec(*cmd, **kwargs):
        return _DoneProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    good = _report_dir(logs_root / "reports", "AAPL_20260610_120000", mtime=1_800_000_000)
    bad = _report_dir(logs_root / "reports", "AAPL_20260610_110000", mtime=1_799_999_000)
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self == bad or bad in self.parents:
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)

    async def main():
        task_id = await runner.spawn("AAPL", "2026-06-10", "stock")
        await asyncio.wait_for(asyncio.gather(*list(runner._WATCH_TASKS)), timeout=10)
        return task_id

    state = registry.get(asyncio.run(main()))
    assert state.status == "done"
    assert state.error is None
    assert state.report_dir == good.name
    assert registry.is_busy() is False



def test_r17_app_failed_process_launch_is_a_500_that_frees_the_slot(monkeypatch, logs_root):
    """Through the real route + runner: a child that cannot be created answers
    500 (the launch error is not an HTTP error), the task is recorded as
    ``error`` with the reason, the single slot is free again, and the very next
    submit starts normally."""
    import asyncio

    monkeypatch.setattr(runner, "registry", runner.TaskRegistry())

    async def no_process(*cmd, **kwargs):
        raise OSError("process creation failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_process)
    with TestClient(app, raise_server_exceptions=False) as c:
        body = {"ticker": "AAPL", "date": "2026-06-10"}
        failed = c.post("/api/analyze", json=body)
        assert failed.status_code == 500
        assert runner.registry.is_busy() is False
        [state] = runner.registry.tasks.values()
        assert state.status == "error"
        assert "failed to start analysis" in (state.error or "")
        assert state.finished_at is not None and state.pid is None
        assert c.get(f"/api/tasks/{state.task_id}").json()["status"] == "error"

        async def fake_exec(*cmd, **kwargs):
            return _BlockingProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        retried = c.post("/api/analyze", json=body)
        assert retried.status_code == 200, retried.text
        assert runner.registry.is_busy() is True

        async def finish(w):
            w.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await w

        (watcher,) = list(runner._WATCH_TASKS)
        c.portal.call(finish, watcher)
    assert runner.registry.is_busy() is False
