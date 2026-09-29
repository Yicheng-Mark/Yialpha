"""Round-10 regressions: Windows write-race + fail-soft finalize seams.

Three same-root defects (2026-09-29 audit, graph/scripts shards):

1. ``memory_resolution.resolve_pending_entries`` — the final durable write
   (``batch_update_with_outcomes``) had NO exception guard, so an IO/lock
   failure propagated out of ``propagate()`` and killed a whole analysis
   run before any node executed — the exact failure mode the round-6 fix
   already eliminated for the LLM leg.
2. ``os.replace`` onto a file a lockless reader holds open raises
   ``PermissionError`` on Windows (CPython opens without
   ``FILE_SHARE_DELETE``); the memory log / full_states_log / node_perf
   writers all hit this against the web store / sibling workers.
   ``replace_retry_on_windows`` retries through the milliseconds-wide race.
3. ``_run_graph``'s finalize seams (``_log_state`` / ``_dump_perf`` /
   ``memory_log.store_decision``) were unprotected inside the main try —
   an IO hiccup at the END of a finished run discarded the entire analysis
   (run_robust would re-bill the whole graph on retry).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_TODAY = date.today().isoformat()


# --------------------------------------------------------------------------- #
# 1. memory_resolution write leg is fail-soft
# --------------------------------------------------------------------------- #
class _FakeReflector:
    def reflect_on_final_decision(self, **kwargs):
        return "lesson"


@pytest.mark.unit
def test_memory_resolution_survives_outcome_write_failure(tmp_path, monkeypatch):
    """batch_update_with_outcomes raising must not escape the sweep: entries
    stay pending (next sweep retries), and the resolved count is honest 0."""
    import yialpha.accuracy as accuracy_mod
    from yialpha.agents.utils.memory import TradingMemoryLog
    from yialpha.graph.memory_resolution import resolve_pending_entries

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    old = (date.today() - timedelta(days=10)).isoformat()
    log.store_decision("NVDA", old, "Decision text.")

    monkeypatch.setattr(
        accuracy_mod, "fetch_returns_yf",
        lambda *a, **k: (0.05, 0.02, 5),
    )
    monkeypatch.setattr(
        log, "batch_update_with_outcomes",
        MagicMock(side_effect=OSError("disk full / replace raced a reader")),
    )
    # Must return (not raise) and report zero durably-resolved entries.
    resolved = resolve_pending_entries(
        log, _FakeReflector(), "NVDA", as_of_date=_TODAY
    )
    assert resolved == 0
    # The entry was never durably rewritten — still pending on disk.
    pending = [e for e in log.get_pending_entries() if e["ticker"] == "NVDA"]
    assert len(pending) == 1


# --------------------------------------------------------------------------- #
# 2. replace_retry_on_windows
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_replace_retry_succeeds_after_transient_permission_errors(tmp_path, monkeypatch):
    """A PermissionError on the first attempts (lockless reader mid-read)
    is retried through; the final replace lands."""
    from yialpha.dataflows.utils import replace_retry_on_windows

    target = tmp_path / "log.json"
    target.write_text("old", encoding="utf-8")
    tmp = tmp_path / "log.json.tmp"
    tmp.write_text("new", encoding="utf-8")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a, b):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(
                5, "Access is denied (reader holds the target)"
            )
        return real_replace(a, b)

    monkeypatch.setattr(
        "yialpha.dataflows.utils.os.replace", flaky_replace
    )
    replace_retry_on_windows(tmp, target)
    assert calls["n"] == 3
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.unit
def test_replace_retry_raises_after_exhaustion(tmp_path, monkeypatch):
    """A persistent denial re-raises after the retry budget — callers keep
    their own fail-soft policy; the helper never swallows indefinitely."""
    from yialpha.dataflows.utils import replace_retry_on_windows

    monkeypatch.setattr(
        "yialpha.dataflows.utils.os.replace",
        MagicMock(side_effect=PermissionError(5, "persistent denial")),
    )
    with pytest.raises(PermissionError):
        replace_retry_on_windows(
            tmp_path / "a.tmp", tmp_path / "a.json", attempts=3, delay_s=0
        )


@pytest.mark.unit
def test_memory_log_batch_write_survives_windows_replace_race(
    tmp_path, monkeypatch
):
    """The memory log's own write path must route through the retry helper:
    a same-process reader holding the file must not lose the update."""
    from yialpha.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    old = (date.today() - timedelta(days=10)).isoformat()
    log.store_decision("NVDA", old, "Decision text.")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(5, "get_past_context reader mid-read")
        return real_replace(a, b)

    monkeypatch.setattr(
        "yialpha.dataflows.utils.os.replace", flaky_replace
    )
    log.batch_update_with_outcomes([{
        "ticker": "NVDA",
        "trade_date": old,
        "raw_return": 0.05,
        "alpha_return": 0.02,
        "holding_days": 5,
        "reflection": "lesson",
        "available_date": _TODAY,
        "asset_type": None,
    }])
    assert calls["n"] >= 2  # the race was ridden through, not bypassed
    pending = [e for e in log.get_pending_entries() if e["ticker"] == "NVDA"]
    assert pending == []  # update durably landed


# --------------------------------------------------------------------------- #
# 3. _run_graph finalize seams are fail-soft
# --------------------------------------------------------------------------- #
def _finalize_harness_graph(tmp_path):
    """The test_memory_log.py propagate harness: MagicMock graph with the
    REAL _run_graph / _bind_run_record_stage bound, so the finalize block
    under test executes verbatim."""
    import functools

    from yialpha.graph.trading_graph import YiAlphaGraph

    fake_state = {
        "final_trade_decision": "Rating: Buy\nBuy NVDA.",
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-10",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "",
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "history": "", "judge_decision": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 1, "latest_speaker": "",
        },
    }
    mock_graph = MagicMock()
    mock_graph.log_states_dict = {}
    mock_graph.debug = False
    mock_graph.config = {"results_dir": str(tmp_path)}
    mock_graph.graph.invoke.return_value = fake_state
    # _run_graph unpacks the run result via _invoke_or_stream (not the bare
    # graph.invoke); returning the REAL dict (not an auto-mock) makes the
    # finalize block's data_quality assignment observable — MagicMock's
    # __setitem__ records the call without storing anything.
    mock_graph._invoke_or_stream = MagicMock(return_value=fake_state)
    mock_graph.propagator.create_initial_state.return_value = fake_state
    mock_graph.propagator.get_graph_args.return_value = {}
    mock_graph.signal_processor.process_signal.return_value = "Buy"
    mock_graph._run_graph = functools.partial(
        YiAlphaGraph._run_graph, mock_graph
    )
    mock_graph._bind_run_record_stage = functools.partial(
        YiAlphaGraph._bind_run_record_stage, mock_graph
    )
    # propagate() threads the final state through the risk overlay after
    # _run_graph returns; make it a passthrough so the assertions observe
    # the SAME state dict the finalize block wrote data_quality onto.
    # Signature: (company_name, trade_date, final_state, portfolio_state,
    # asset_type=...) — the state is the THIRD positional.
    mock_graph._apply_risk_overlay = MagicMock(
        side_effect=lambda *a, **k: a[2]
    )
    return mock_graph, fake_state


@pytest.mark.unit
def test_propagate_survives_states_log_write_failure(tmp_path):
    """A full_states_log IO failure at the END of a finished run must not
    discard the analysis — the run result is returned with the quality
    block reading None (unknown), and a WARNING is logged."""
    from yialpha.agents.utils.memory import TradingMemoryLog
    from yialpha.graph.trading_graph import YiAlphaGraph

    mock_graph, _state = _finalize_harness_graph(tmp_path)
    mock_graph.memory_log = TradingMemoryLog(
        {"memory_log_path": str(tmp_path / "mem.md")}
    )
    mock_graph._log_state = MagicMock(
        side_effect=OSError("replace raced a web-store reader")
    )
    final_state, signal = YiAlphaGraph.propagate(
        mock_graph, "NVDA", "2026-01-10"
    )
    # The completed analysis is returned, not raised away.
    assert final_state["final_trade_decision"].startswith("Rating: Buy")
    assert final_state["data_quality"] is None
    assert signal is not None


@pytest.mark.unit
def test_propagate_survives_store_decision_failure(tmp_path, caplog):
    """A memory store_decision failure costs one pending entry (logged),
    never the finished analysis."""
    import logging

    from yialpha.graph.trading_graph import YiAlphaGraph

    mock_graph, _state = _finalize_harness_graph(tmp_path)
    mock_graph._log_state = MagicMock(return_value=None)
    mock_graph.memory_log.store_decision = MagicMock(
        side_effect=OSError("lock acquisition failed")
    )
    with caplog.at_level(logging.WARNING):
        final_state, signal = YiAlphaGraph.propagate(
            mock_graph, "NVDA", "2026-01-10"
        )
    assert final_state["final_trade_decision"].startswith("Rating: Buy")
    assert signal is not None
    assert any("store_decision" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# 4. ledger round-10 fixes
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_degraded_ledger_read_is_labeled_not_empty(tmp_path, monkeypatch):
    """L1: a failed positions read must not masquerade as a legal empty book
    ("" was indistinguishable from "no positions"); the snapshot source now
    says ledger-unreadable, mirroring the file side's +unreadable contract."""
    from yialpha.ledger import portfolio

    monkeypatch.setattr(portfolio, "ledger_exists", lambda: True)
    monkeypatch.setattr(
        portfolio, "get_connection",
        MagicMock(side_effect=RuntimeError("SQLITE_BUSY / disk I/O error")),
    )
    rows, source = portfolio.load_positions_input()
    assert rows == []
    assert source == "ledger-unreadable"
    # The compat wrapper keeps its never-raise contract.
    assert portfolio.open_positions() == []


@pytest.mark.unit
def test_cost_legs_and_ticket_for_run_pick_same_latest_ticket():
    """L2: two same-second tickets on one run — the cost legs must book from
    the SAME row ticket_for_run discloses (last INSERTED), not from a
    ticket_id lexicographic tie-break."""
    import json as _json

    from yialpha.ledger import outcome_compute, tickets_mirror
    from yialpha.ledger.sqlite import ledger_transaction, utc_now_iso

    run_id = "R-r10-tie"
    same_second = utc_now_iso()  # second precision
    payloads = {
        "T-aaa": {"cost_detail": {
            "entry_fee_bps": 5, "exit_fee_bps": 5,
            "entry_slippage_bps": 2, "exit_slippage_bps": 2,
        }},  # fees 10bps, slippage 4bps
        "T-zzz": {"cost_detail": {
            "entry_fee_bps": 25, "exit_fee_bps": 25,
            "entry_slippage_bps": 10, "exit_slippage_bps": 10,
        }},  # fees 50bps, slippage 20bps
    }
    # Insert aaa FIRST, zzz second: latest written = zzz regardless of the
    # fact that "T-aaa" < "T-zzz" lexicographically would also pick zzz —
    # so insert the lexicographically SMALLER id LAST to expose a wrong
    # tie-break (ticket_id DESC would pick T-zzz, the wrong row).
    for ticket_id in ("T-zzz", "T-aaa"):
        with ledger_transaction() as cur:
            cur.execute(
                "INSERT INTO tickets (ticket_id, decision_id, run_id, "
                "payload, ticket_version, written_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticket_id, None, run_id,
                 _json.dumps(payloads[ticket_id]), "v1", same_second),
            )
    tid, fees, slip = outcome_compute._ticket_cost_legs(run_id)
    assert tid == "T-aaa"  # last inserted, not lexicographically largest
    assert fees == 0.0010 and slip == 0.0004
    disclosed = tickets_mirror.ticket_for_run(run_id)
    assert disclosed is not None
    # Same row: the disclosed payload carries T-aaa's bps fields.
    assert disclosed["cost_detail"]["entry_fee_bps"] == 5


@pytest.mark.unit
def test_row_to_outcome_tolerates_pre_regime_id_schema():
    """L10: a read-only row from an older outcomes table (no regime_id
    column) must decode, not IndexError."""
    from yialpha.ledger.models import row_to_outcome

    class _Row:
        # sqlite3.Row-like: keys() without regime_id.
        _data = {
            "outcome_id": "O1", "prediction_id": "P1", "run_id": "R1",
            "ticket_id": "T1", "horizon_days": 5, "status": "complete",
            "contract_price_return": 0.1, "underlying_return": None,
            "basis_return": None, "funding_pnl": 0.0, "fees": 0.0,
            "slippage": 0.0, "liquidation_loss": None, "net_return": 0.1,
            "legs_missing": "[]", "outcome_available_at": None,
            "computed_at": "2026-01-01T00:00:00+00:00",
        }

        def keys(self):
            return self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    rec = row_to_outcome(_Row())  # type: ignore[arg-type]
    assert rec.regime_id is None
    assert rec.outcome_id == "O1"


@pytest.mark.unit
def test_pit_date_only_admits_sub_second_end_of_day():
    """L8: a date-only as-of admits ANY instant of that day — including
    23:59:59.5 — per pit_latest_instant's own contract."""
    from yialpha.ledger.models import pit_latest_instant

    end_of_day = pit_latest_instant("2026-09-22")
    evidence = datetime.fromisoformat("2026-09-22T23:59:59.500000+00:00")
    assert end_of_day >= evidence


@pytest.mark.unit
def test_available_at_rejects_compact_iso_date():
    """L12: Python >=3.11 date.fromisoformat accepts "20260922"; that form
    breaks the lexicographic PIT contract, so the validator must refuse it
    (event_date's strict regex now applies to available_at too)."""
    from yialpha.instruments.corporate_actions import (
        _require_iso_date_or_datetime,
    )

    with pytest.raises(ValueError):
        _require_iso_date_or_datetime("20260922")
    # Dashed forms still accepted.
    _require_iso_date_or_datetime("2026-09-22")
    _require_iso_date_or_datetime("2026-09-22T12:00:00+00:00")


# --------------------------------------------------------------------------- #
# 5. scripts round-10 fixes (run_baseline / prune CLI / export)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_rebalance_dates_rejects_step_zero():
    """F7: step 0 repeated the same rebalance date n times into the engine."""
    import scripts.run_baseline as runner

    with pytest.raises(SystemExit) as cm:
        runner._rebalance_dates("2026-01-01", "2026-06-30", step=0, n=6)
    assert "step" in str(cm.value)


@pytest.mark.unit
def test_rebalance_dates_warns_when_calendar_shrinks(capsys):
    """F7: a window too narrow for n*step dates must SAY so, not silently
    run a shrunken calendar."""
    import scripts.run_baseline as runner

    dates = runner._rebalance_dates(
        "2026-06-01", "2026-06-30", step=10, n=6
    )
    assert len(dates) < 6
    err = capsys.readouterr().err
    assert "shrunk" in err


@pytest.mark.unit
def test_run_baseline_rejects_bad_numeric_flags(capsys):
    """F6: --runs 0 / --rebalance 0 exit 2 with a usage error instead of a
    bare ValueError from evaluate_gate (or a contradictory summary)."""
    import scripts.run_baseline as runner

    base_argv = ["--baseline", "--start", "2026-01-01",
                 "--end", "2026-03-01"]
    for flag in (["--runs", "0"], ["--rebalance", "0"],
                 ["--workers", "0"], ["--holding-days", "0"]):
        import sys as _sys
        from unittest import mock as _mock

        with pytest.raises(SystemExit) as cm, _mock.patch.object(
            _sys, "argv", ["run_baseline.py"] + base_argv + flag
        ):
            runner.main()
        assert cm.value.code == 2


def _load_prune_cli():
    """Load prune_indicators_cli.py in-process (the suite's established
    importlib pattern, cf. tests/test_ic_analysis.py:28-31) — the round-10
    pins below used to spawn a real interpreter per invocation (~1s each:
    pandas + the yialpha.backtest.ic import chain) on every full-suite run.
    In-process, main() returns the exit code directly (the __main__ guard is
    what wraps it in SystemExit)."""
    import importlib.util as _ilu

    _p = Path(__file__).resolve().parent.parent / "scripts" / "prune_indicators_cli.py"
    _spec = _ilu.spec_from_file_location("prune_cli_round10_under_test", _p)
    assert _spec is not None and _spec.loader is not None
    mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_prune_cli_rejects_degenerate_args(tmp_path):
    """F9: --window 0 made every IC window NaN (all "keep", exit 0);
    --min-consecutive 0 pruned everything — both now exit 1."""
    mod = _load_prune_cli()

    csv = tmp_path / "ic.csv"
    csv.write_text("date,forward_return,ind\n", encoding="utf-8")  # header only
    for extra in (["--window", "0"], ["--min-consecutive", "0"]):
        assert mod.main([str(csv), *extra]) == 1, extra


@pytest.mark.unit
def test_prune_cli_rejects_header_only_csv(tmp_path, capsys):
    """F9: a verdict manufactured from zero evidence must not exit 0."""
    mod = _load_prune_cli()

    csv = tmp_path / "empty.csv"
    csv.write_text("date,forward_return,ind\n", encoding="utf-8")
    assert mod.main([str(csv)]) == 1
    assert "no data rows" in capsys.readouterr().err


@pytest.mark.unit
def test_prune_cli_all_nan_forward_diagnosed_as_broken_construction(
    tmp_path, capsys,
):
    """Round-10: rows exist but every forward_return is NaN (exporter
    horizon bug) — refused with exit 1 like the header-only case, but the
    diagnosis must say the forward-return construction is broken (the old
    single counter misreported it as "header only", pointing the operator
    at the wrong fix)."""
    mod = _load_prune_cli()

    csv = tmp_path / "allnan.csv"
    csv.write_text(
        "date,forward_return,ind\n"
        "2026-01-02,,1.0\n"
        "2026-01-03,,2.0\n",
        encoding="utf-8",
    )
    assert mod.main([str(csv)]) == 1
    err = capsys.readouterr().err
    assert "2 rows" in err
    assert "0 non-NaN" in err
    assert "forward-return construction is broken" in err


@pytest.mark.unit
def test_export_ic_dataset_dedupes_tickers(monkeypatch, capsys, tmp_path):
    """F10: a repeated ticker counted twice against the written map — exit 1
    despite a fully successful export."""
    import importlib.util as _ilu

    _p = Path(__file__).resolve().parent.parent / "scripts" / "export_ic_dataset.py"
    _spec = _ilu.spec_from_file_location("export_ic_under_test", _p)
    mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(mod)

    def fake_export(tickers, **_kwargs):
        assert tickers == ["NVDA"], "duplicates must collapse before export"
        out = tmp_path / "NVDA_5d.csv"
        out.write_text("date,forward_return,macd\n", encoding="utf-8")
        return {"NVDA": str(out)}

    monkeypatch.setattr(mod, "export_ic_datasets", fake_export)
    monkeypatch.setattr(mod, "INDICATOR_NAMES", {"macd"})
    rc = mod.main(["NVDA", "NVDA", "--horizon", "5"])
    assert rc == 0
    assert "duplicate" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 6. round-10 misc pins (regime id re-derivation)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_compute_regime_id_rederives_from_stored_payload():
    """The docstring-exported invariant — compute_regime_id over a STORED
    payload (asdict + the ledger store's ticker/instrument_class context
    keys) reproduces its own id — was false: the context keys polluted the
    preimage. Fresh-state ids are unchanged (asdict never carries them)."""
    from dataclasses import asdict

    from yialpha.regime.state import RegimeState, compute_regime_id

    state = RegimeState(trend_regime="up", funding_pct=0.001)
    fresh_id = compute_regime_id(state)
    stored_payload = dict(asdict(state))
    stored_payload["regime_id"] = fresh_id  # as persisted
    stored_payload["ticker"] = "BTCUSDT"
    stored_payload["instrument_class"] = "pure_crypto_perp"
    assert compute_regime_id(stored_payload) == fresh_id
