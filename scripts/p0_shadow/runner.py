"""Offline B0 integration and read-only inspection. No live transport entry."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import tempfile
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch

from .artifacts import EvidenceRecorder, create_manifest
from .audit import build_audit, write_audit
from .guard import GuardError, PathGuard, isolated_runtime, resolve_cohort_root

# .../Yialpha/scripts/p0_shadow/runner.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMED = datetime(2026, 9, 5, 12, tzinfo=UTC)
GROUPS = (
    ("C-BTC", "BTCUSDT", "BTCUSDT", "pure_crypto_perp", "CONTRACT"),
    ("C-ETH", "ETHUSDT", "ETHUSDT", "pure_crypto_perp", "CONTRACT"),
    ("C-MU", "MUUSDT", "MUUSDT", "stock_perp", "CONTRACT"),
    ("U-MU-WE", "MUUSDT", "MU", "stock_perp", "UNDERLYING"),
)


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def code_identity() -> dict:
    """Read selected source identity before entering the no-subprocess runtime."""
    scope = ["scripts", "yialpha", "tests", "docs", "pyproject.toml", "requirements.lock"]
    exclusions = [":(exclude)**/.env", ":(exclude)**/.env.*"]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)

    extra = git("ls-files", "--others", "--exclude-standard", "--", *scope, *exclusions)
    hashes = {}
    for name in extra.decode().splitlines():
        if Path(name).name.startswith(".env"):
            continue
        path = PROJECT_ROOT / name
        if path.is_file() and not path.is_symlink():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dependencies = {}
    for name in ("pandas", "yfinance", "curl_cffi", "python-dotenv"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "unavailable"
    return {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "dirty_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD", "--", *scope, *exclusions)).hexdigest(),
        "untracked_source_sha256": hashes,
        "dependencies": dependencies,
    }


def _versions() -> dict:
    from yialpha import versions

    return {name: getattr(versions, name) for name in (
        "SCHEMA_VERSION", "FEATURE_VERSION", "COST_MODEL_VERSION", "TICKET_VERSION",
        "PREDICTION_TIME_VERSION", "OUTCOME_COMPUTE_VERSION",
    )}


def _backup(paths: PathGuard, ledger: Path, attempt: str) -> dict:
    target = paths.write_bytes(f"checkpoints/{attempt}/portfolio.db", b"", exclusive=True)
    with sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True) as source:
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
            if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise GuardError("SQLite consistent backup failed validation")
        finally:
            destination.close()
    return {"original_ledger_path": str(ledger), "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "method": "sqlite_backup"}


def _report(paths: PathGuard, cohort: str, attempt: str, now: str, *, sample_kind: str) -> dict:
    snapshot = _backup(paths, paths.resolve("ledger/portfolio.db"), attempt)
    audit = build_audit(paths.resolve("ledger/portfolio.db"), cohort_id=cohort, observed_at=now)
    audit.update(sample_kind=sample_kind, source_snapshot=snapshot,
                 prediction_ability_evidence=False, execution_observed_at=_utc())
    write_audit(paths, attempt, audit)
    return audit


class _FixtureMarket:
    """Generated fixture rows only; synthetic clock is never a live option."""

    def __init__(self, recorder: EvidenceRecorder, scenario: str):
        self.recorder, self.scenario = recorder, scenario
        self.now = FORMED
        self.call_id = "capture"
        self.records: list[dict] = []

    def record(self, source, metadata, normalized):
        raw = json.dumps(normalized, sort_keys=True, allow_nan=False).encode()
        result = self.recorder.record_response(
            source=source, request_metadata={**metadata, "clock_domain": "synthetic"},
            raw_bytes=raw, normalized_payload=normalized,
            started_at=self.now.isoformat(), observed_at=self.now.isoformat(),
            evidence_level="offline_fixture", call_id=self.call_id,
        )
        self.records.append(result)

    def prices(self, symbol, start, end, *, equity=False):
        import pandas as pd

        days = pd.date_range(start, end, freq="D")
        if equity:
            days = days[days.dayofweek < 5]
        rows = [{"date": d.date().isoformat(), "close": 100.0 + (d.date() - FORMED.date()).days}
                for d in days]
        self.record("yfinance:1d:close" if equity else "binance_perp:1d:last",
                    {"symbol": symbol, "start_date": str(start), "end_date": str(end), "interval": "1d"}, rows)
        return pd.DataFrame({"Close": [r["close"] for r in rows]}, index=pd.to_datetime([r["date"] for r in rows]))

    def reference(self, symbol, day, *, equity, as_of):
        return self.prices(symbol, day.isoformat(), day.isoformat(), equity=equity)

    def perp(self, symbol, start, end, **kwargs):
        return self.prices(symbol, start, end)

    def equity(self, symbol, start, end):
        last = datetime.fromisoformat(end).date() - timedelta(days=1)
        return self.prices(symbol, start, last.isoformat(), equity=True)

    def funding(self, symbol, start, end, **kwargs):
        import pandas as pd

        rows = []
        if not (self.scenario == "missing-funding" and symbol == "BTCUSDT"):
            for stamp in pd.date_range(start, end + " 23:59:59", freq="8h"):
                rows.append({"fundingTime": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                             "fundingRate": 0.0001, "symbol": symbol})
        self.record("binance_perp:funding", {"symbol": symbol, "start_date": start, "end_date": end}, rows)
        return "fundingTime,fundingRate,symbol\n" + "\n".join(
            f"{r['fundingTime']},{r['fundingRate']},{symbol}" for r in rows)


def _score_fixture(paths: PathGuard, market: _FixtureMarket, cohort: str, day: int):
    from yialpha.ledger.outcome_compute import compute_outcomes
    from yialpha.ledger.outcomes import pending_predictions
    from yialpha.ledger.sqlite import get_connection

    market.now = FORMED + timedelta(days=day)
    now = market.now.isoformat()
    attempt = f"sim-{day}d-{uuid.uuid4().hex}"
    market.call_id = attempt
    before = pending_predictions(now)
    reports = []
    for _ in range(len(before) + 1):
        old_count = get_connection().execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        report = compute_outcomes(now, limit=2)
        reports.append(asdict(report))
        new_count = get_connection().execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        if new_count == old_count or not pending_predictions(now):
            break
    paths.write_json(f"attempts/{attempt}/scoring.json", {
        "sample_kind": "offline_fixture", "synthetic_now_as_of": now,
        "execution_observed_at": _utc(), "queue_before": before,
        "reports": reports, "queue_after": pending_predictions(now),
        "request_ids": [r["request_id"] for r in market.records if r["call_id"] == attempt],
    })
    _report(paths, cohort, attempt, now, sample_kind="offline_fixture")
    return attempt


def run_offline_demo(root: Path, *, cohort_id: str, scenario: str = "baseline", identity: dict | None = None) -> dict:
    if scenario not in {"baseline", "missing-funding"}:
        raise GuardError("Unknown offline fixture scenario")
    root = resolve_cohort_root(root, project_root=PROJECT_ROOT)
    if root == Path(tempfile.gettempdir()).resolve() or not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise GuardError("offline-demo requires a new directory beneath the system temporary directory")
    if root.exists():
        raise GuardError("offline-demo never overwrites an existing cohort")
    if not cohort_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in cohort_id):
        raise GuardError("cohort ID must contain only letters, digits, dash and underscore")
    identity = code_identity() if identity is None else identity
    # Warm import-time probes (filelock symlink detection via the system temp
    # directory, platform/pandas hostname lookup) before the offline runtime
    # denies writes and socket calls; yfinance's cache managers stay lazy.
    import filelock  # noqa: F401
    import yfinance  # noqa: F401
    root.mkdir()
    session = "offline-" + uuid.uuid4().hex
    with isolated_runtime(root, session, network_mode="offline"):
        from yialpha.agents.utils import prediction_tools as pt
        from yialpha.ledger import outcome_compute as oc, time_contract as tc
        from yialpha.ledger.evidence import register_run
        from yialpha.ledger.predictions import predictions_for_run
        from yialpha.ledger.run_context import reset_ledger_run_context, set_ledger_run_context
        from yialpha.ledger.sqlite import get_connection
        from yialpha.ledger.tickets_mirror import attach_ticket, ticket_for_run

        paths = PathGuard(root)
        create_manifest(root, cohort_id, sample_kind="offline_fixture", code_identity=identity,
                        created_at=_utc(), versions=_versions())
        market = _FixtureMarket(EvidenceRecorder(root, session), scenario)
        with (patch.object(tc, "_now_utc", lambda: market.now),
              patch.object(tc, "_reference_frame", market.reference),
              patch.object(oc, "binance_klines_frame", market.perp),
              patch.object(oc, "get_YFin_history_cached", market.equity),
              patch.object(oc, "get_binance_funding_rate", market.funding)):
            try:
                pt.reset_prediction_capture_for_test()
                for group, ticker, instrument, instrument_class, scope in GROUPS:
                    run_id = f"{cohort_id}-{group}"
                    market.call_id = f"capture-{group}"
                    set_ledger_run_context(run_id, ticker, "crypto_perp", instrument_class, FORMED.isoformat())
                    register_run(run_id, ticker, "crypto_perp", instrument_class, FORMED.isoformat())
                    costs = {"entry_fee_bps": 4.0, "exit_fee_bps": 4.0,
                             "entry_slippage_bps": 1.0, "exit_slippage_bps": 1.0}
                    attach_ticket(run_id + "-cost", None, run_id, costs, "v1")
                    if ticket_for_run(run_id) != costs:
                        raise GuardError("shadow cost mirror did not persist exactly")
                    paths.write_json(f"costs/{run_id}.json", {
                        "run_id": run_id, "sample_kind": "offline_fixture", "costs": costs,
                        "provenance": "fixed synthetic assumption; no fills or account fees",
                        "synthetic_frozen_at": FORMED.isoformat(), "versions": _versions(),
                    })
                    pt.begin_prediction_capture("market", instrument, scope)
                    inputs = [{"horizon_days": h, "direction": "up", "prob_up": 0.5} for h in (1, 5, 21)]
                    tool = pt.make_submit_prediction_tool(instrument)
                    accepted = tool.invoke({"predictions": inputs})
                    count = len(market.records)
                    tool.invoke({"predictions": inputs})
                    if len(market.records) != count or len(pt.flush_predictions()) != 3:
                        raise GuardError("submission replay or capture persistence failed")
                    if pt.flush_predictions():
                        raise GuardError("flush replay unexpectedly wrote rows")
                    for row in predictions_for_run(run_id):
                        paths.write_json(f"snapshots/{row.prediction_id}.json", {
                            "sample_kind": "offline_fixture", "prediction": asdict(row),
                            "submission_inputs": inputs, "acceptance": accepted,
                            "call_id": market.call_id,
                            "request_ids": [r["request_id"] for r in market.records if r["call_id"] == market.call_id],
                        })
                attempts = [_score_fixture(paths, market, cohort_id, day) for day in (1, 5, 21, 28)]
                rows = get_connection().execute("SELECT status,COUNT(*) FROM outcomes GROUP BY status").fetchall()
                summary = {"cohort_id": cohort_id, "sample_kind": "offline_fixture", "scenario": scenario,
                           "predictions": 12, "outcomes_by_status": dict(rows), "attempt_ids": attempts,
                           "root": str(root), "prediction_ability_evidence": False,
                           "natural_maturity_observed": False, "real_network_enabled": False}
                paths.write_json("offline-result.json", summary)
                return summary
            finally:
                reset_ledger_run_context()
                pt.reset_prediction_capture_for_test()


def inspect_cohort(root: Path, *, cohort_id: str, identity: dict | None = None) -> dict:
    root = resolve_cohort_root(root, project_root=PROJECT_ROOT, must_exist=True)
    paths = PathGuard(root)
    manifest = json.loads(paths.resolve("manifest.json").read_text(encoding="utf-8"))
    if manifest.get("cohort_id") != cohort_id:
        raise GuardError("cohort ID does not match manifest")
    kind = manifest.get("sample_kind")
    if kind not in {"offline_fixture", "process_validation_fixed_input"}:
        raise GuardError("unsupported or missing sample kind")
    if not paths.resolve("ledger/portfolio.db").is_file():
        raise GuardError("cohort ledger is missing; inspect cannot initialize it")
    identity = code_identity() if identity is None else identity
    import filelock  # noqa: F401
    import yfinance  # noqa: F401
    attempt = "inspect-" + uuid.uuid4().hex
    with isolated_runtime(root, attempt, network_mode="offline"):
        now = _utc()
        report = _report(paths, cohort_id, attempt, now, sample_kind=kind)
        paths.write_json(f"attempts/{attempt}/inspection.json", {
            "code_identity": identity, "sample_kind": kind, "observed_at": now,
            "scoring_invoked": False, "network_enabled": False,
        })
    return {"cohort_id": cohort_id, "attempt_id": attempt, "root": str(root),
            "source_snapshot": report["source_snapshot"], "scoring_invoked": False}
