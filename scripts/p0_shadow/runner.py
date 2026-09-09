"""Offline B0, real B1 capture and B2 natural-expiry scoring. No LLM transport."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
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
PROCESS_KIND = "process_validation_fixed_input"
WEEKDAY_GROUPS = ("C-BTC", "C-ETH", "C-MU")
FIXED_COSTS = {"entry_fee_bps": 4.0, "exit_fee_bps": 4.0,
               "entry_slippage_bps": 1.0, "exit_slippage_bps": 1.0}
FIXED_INPUTS = [{"horizon_days": h, "direction": "up", "prob_up": 0.5} for h in (1, 5, 21)]


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
    audit = build_audit(paths.resolve("ledger/portfolio.db"), cohort_id=cohort,
                        observed_at=now, sample_kind=sample_kind)
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


def _vendor_payload(result: Any) -> dict[str, Any]:
    """JSON-safe representation of one library-level vendor return value."""
    if result is None:
        return {"kind": "none"}
    if isinstance(result, str):
        return {"kind": "text", "length": len(result), "text": result[:20000]}
    columns = getattr(result, "columns", None)
    if columns is not None and hasattr(result, "index"):
        names = [str(column) for column in columns]
        values: dict[str, Any] = {}
        for name, column in zip(names, columns, strict=True):
            cells: list[float | str | None] = []
            for value in result[column].tolist():
                if value is None or (isinstance(value, float) and not math.isfinite(value)):
                    cells.append(None)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    cells.append(float(value))
                else:
                    cells.append(str(value))
            values[name] = cells
        return {"kind": "dataframe", "columns": names, "length": len(result),
                "index": [str(item) for item in result.index.tolist()], "data": values}
    return {"kind": "repr", "repr": repr(result)[:2000]}


class _VendorEvidence:
    """Call-through wrappers that file library-level vendor output as evidence.

    This is not an HTTP hook: the recorder honestly labels responses as
    ``vendor_library_output``. Call ids let the runner prove that a replayed
    submission added no vendor fetch.
    """

    def __init__(self, recorder: EvidenceRecorder):
        self.recorder = recorder
        self.call_id = recorder.attempt_id
        self.calls: list[dict[str, str]] = []

    def ids_for(self, call_id: str) -> list[str]:
        return [item["request_id"] for item in self.calls if item["call_id"] == call_id]

    def _record(self, source: str, metadata: dict[str, Any], result: Any, started_at: str) -> None:
        payload = _vendor_payload(result)
        raw = (result.encode("utf-8") if isinstance(result, str)
               else json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8"))
        record = self.recorder.record_response(
            source=source, request_metadata=metadata, raw_bytes=raw,
            normalized_payload=payload, started_at=started_at, observed_at=_utc(),
            evidence_level="vendor_library_output", call_id=self.call_id,
        )
        self.calls.append({"call_id": self.call_id,
                           "request_id": record["request_id"],
                           "source": source})

    def _wrap(self, stack: ExitStack, module: Any, attr: str, source_of: Callable[..., str]) -> None:
        original = getattr(module, attr)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = _utc()
            result = original(*args, **kwargs)
            metadata: dict[str, Any] = {}
            if args:
                metadata["symbol"] = str(args[0])
            if len(args) > 1:
                metadata["start_date"] = str(args[1])
            if len(args) > 2:
                metadata["end_date"] = str(args[2])
            for key in ("start_date", "end_date", "interval", "venue", "price_type"):
                if key in kwargs:
                    metadata[key] = str(kwargs[key])
            self._record(source_of(*args, **kwargs), metadata, result, started)
            return result

        stack.enter_context(patch.object(module, attr, wrapped))

    @contextmanager
    def installed(self, *, include_reference_seam: bool):
        from yialpha.ledger import outcome_compute as oc, time_contract as tc

        with ExitStack() as stack:
            if include_reference_seam:
                self._wrap(stack, tc, "_reference_frame",
                           lambda *a, **k: "yfinance:1d:close" if k.get("equity") else "binance_perp:1d:last")
            self._wrap(stack, oc, "binance_klines_frame", lambda *a, **k: "binance_perp:1d:last")
            self._wrap(stack, oc, "get_YFin_history_cached", lambda *a, **k: "yfinance:1d:close")
            self._wrap(stack, oc, "get_binance_funding_rate", lambda *a, **k: "binance_perp:funding")
            yield self


def _load_manifest(paths: PathGuard) -> dict:
    return json.loads(paths.resolve("manifest.json").read_text(encoding="utf-8"))


def _timing_summary(row: Any) -> dict[str, Any]:
    timing = row.timing if isinstance(row.timing, dict) else None
    if timing is None:
        return {"prediction_id": row.prediction_id,
                "unparsed_timing": None if row.timing is None else str(row.timing)[:200]}
    return {
        "prediction_id": row.prediction_id, "horizon_days": row.horizon_days,
        "prediction_formed_at": timing.get("prediction_formed_at"),
        "reference_price": timing.get("reference_price"),
        "reference_price_at": timing.get("reference_price_at"),
        "reference_source": timing.get("reference_source"),
        "reference_error": timing.get("reference_error"),
    }


def capture_cohort(
    root: Path, *, cohort_id: str, groups: tuple[str, ...] = WEEKDAY_GROUPS,
    identity: dict | None = None, now_fn: Callable[[], datetime] | None = None,
) -> dict:
    """B1: form process-validation samples at the actual formation time.

    Fixed inputs only (``prob_up=0.5``): no research meaning, no LLM. The
    submit tool path freezes the reference price locally before acceptance.
    Re-running a group is a no-op once its three horizons are persisted.
    """
    now_fn = now_fn or (lambda: datetime.now(UTC))
    known = {group[0] for group in GROUPS}
    if not groups or any(group not in known for group in groups):
        raise GuardError("capture groups must be a nonempty subset of " + ",".join(sorted(known)))
    if "U-MU-WE" in groups and now_fn().weekday() < 5:
        raise GuardError("U-MU-WE must be formed on an actual UTC weekend day")
    root = resolve_cohort_root(root, project_root=PROJECT_ROOT)
    paths = PathGuard(root)
    created = not root.exists()
    if created:
        root.mkdir()
    else:
        manifest = _load_manifest(paths)
        if manifest.get("cohort_id") != cohort_id:
            raise GuardError("cohort ID does not match manifest")
        if manifest.get("sample_kind") != PROCESS_KIND:
            raise GuardError("only process-validation cohorts accept appended captures")
        if not paths.resolve("ledger/portfolio.db").is_file():
            raise GuardError("existing cohort is missing its ledger")
    identity = code_identity() if identity is None else identity
    # Warm import-time probes before the runtime denies side-effectful calls.
    import filelock  # noqa: F401
    import yfinance  # noqa: F401
    attempt = "capture-" + uuid.uuid4().hex
    with isolated_runtime(root, attempt, network_mode="allowlist") as runtime:
        from yialpha.agents.utils import prediction_tools as pt
        from yialpha.ledger.evidence import register_run
        from yialpha.ledger.predictions import predictions_for_run
        from yialpha.ledger.run_context import reset_ledger_run_context, set_ledger_run_context
        from yialpha.ledger.tickets_mirror import attach_ticket, ticket_for_run

        if created:
            create_manifest(root, cohort_id, sample_kind=PROCESS_KIND, code_identity=identity,
                            created_at=_utc(), versions=_versions())
        recorder = EvidenceRecorder(root, attempt, sample_kind=PROCESS_KIND)
        captured: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        try:
            with _VendorEvidence(recorder).installed(include_reference_seam=True) as evidence:
                for group_id, ticker, instrument, instrument_class, scope in GROUPS:
                    if group_id not in groups:
                        continue
                    run_id = f"{cohort_id}-{group_id}"
                    existing = predictions_for_run(run_id)
                    if len(existing) >= len(FIXED_INPUTS):
                        missing = [paths.resolve(f"snapshots/{row.prediction_id}.json")
                                   for row in existing
                                   if not paths.resolve(f"snapshots/{row.prediction_id}.json").exists()]
                        for row in existing:
                            if not paths.resolve(f"snapshots/{row.prediction_id}.json").exists():
                                paths.write_json(f"snapshots/{row.prediction_id}.json", {
                                    "sample_kind": PROCESS_KIND, "prediction": asdict(row),
                                    "recovered_from_ledger": True,
                                    "original_submission_inputs_unavailable": True,
                                })
                        skipped.append({"group_id": group_id, "run_id": run_id,
                                        "predictions": len(existing),
                                        "snapshots_recovered": len(missing)})
                        continue
                    if existing:
                        raise GuardError("partial capture state requires manual review before retry")
                    formed = now_fn().isoformat()
                    set_ledger_run_context(run_id, ticker, "crypto_perp", instrument_class, formed)
                    register_run(run_id, ticker, "crypto_perp", instrument_class, formed)
                    attach_ticket(run_id + "-cost", None, run_id, FIXED_COSTS, "v1")
                    if ticket_for_run(run_id) != FIXED_COSTS:
                        raise GuardError("shadow cost mirror did not persist exactly")
                    if not paths.resolve(f"costs/{run_id}.json").exists():
                        paths.write_json(f"costs/{run_id}.json", {
                            "run_id": run_id, "sample_kind": PROCESS_KIND, "costs": FIXED_COSTS,
                            "provenance": "fixed assumption frozen at formation; no fills or account fees",
                            "frozen_at": formed, "versions": _versions(),
                        })
                    evidence.call_id = f"capture-{group_id}"
                    pt.begin_prediction_capture("market", instrument, scope)
                    tool = pt.make_submit_prediction_tool(instrument)
                    accepted = tool.invoke({"predictions": FIXED_INPUTS})
                    before = len(evidence.calls)
                    tool.invoke({"predictions": FIXED_INPUTS})
                    if len(evidence.calls) != before:
                        raise GuardError("submission replay re-fetched vendor data")
                    if len(pt.flush_predictions()) != len(FIXED_INPUTS):
                        raise GuardError("capture persistence failed")
                    if pt.flush_predictions():
                        raise GuardError("flush replay unexpectedly wrote rows")
                    rows = predictions_for_run(run_id)
                    timings = []
                    for row in rows:
                        paths.write_json(f"snapshots/{row.prediction_id}.json", {
                            "sample_kind": PROCESS_KIND, "prediction": asdict(row),
                            "submission_inputs": FIXED_INPUTS, "acceptance": accepted,
                            "call_id": evidence.call_id,
                            "request_ids": evidence.ids_for(evidence.call_id),
                        })
                        timings.append(_timing_summary(row))
                    captured.append({
                        "group_id": group_id, "run_id": run_id, "instrument_id": instrument,
                        "prediction_scope": scope, "analysis_as_of": formed,
                        "predictions": len(rows), "timings": timings,
                        "request_ids": evidence.ids_for(evidence.call_id),
                    })
            paths.write_json(f"attempts/{attempt}/capture.json", {
                "sample_kind": PROCESS_KIND, "llm_used": False,
                "prediction_ability_evidence": False,
                "attempt_id": attempt, "observed_at": _utc(),
                "captured": captured, "skipped": skipped,
                "vendor_calls": evidence.calls,
                "network": {"mode": runtime["network_mode"],
                            "allowed_calls": runtime["allowed_calls"],
                            "denied_attempts": runtime["network_attempts"]},
            })
        finally:
            reset_ledger_run_context()
            pt.reset_prediction_capture_for_test()
    return {"cohort_id": cohort_id, "attempt_id": attempt, "root": str(root),
            "sample_kind": PROCESS_KIND, "captured_groups": [item["group_id"] for item in captured],
            "skipped_groups": [item["group_id"] for item in skipped],
            "network": {"allowed_calls": runtime["allowed_calls"],
                        "denied_attempts": runtime["network_attempts"]},
            "prediction_ability_evidence": False}


def score_cohort(
    root: Path, *, cohort_id: str, identity: dict | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> dict:
    """B2: run natural-expiry scoring with the real current UTC clock.

    Only process-validation cohorts; the CLI never exposes ``now_fn``. One
    attempt drains the due queue, files vendor evidence, snapshots the
    scoreboard and writes an independent row audit with a ledger backup.
    """
    now_fn = now_fn or (lambda: datetime.now(UTC))
    root = resolve_cohort_root(root, project_root=PROJECT_ROOT, must_exist=True)
    paths = PathGuard(root)
    manifest = _load_manifest(paths)
    if manifest.get("cohort_id") != cohort_id:
        raise GuardError("cohort ID does not match manifest")
    if manifest.get("sample_kind") != PROCESS_KIND:
        raise GuardError("scoring is restricted to process-validation cohorts")
    if not paths.resolve("ledger/portfolio.db").is_file():
        raise GuardError("cohort ledger is missing")
    identity = code_identity() if identity is None else identity
    import filelock  # noqa: F401
    import yfinance  # noqa: F401
    attempt = "score-" + uuid.uuid4().hex
    with isolated_runtime(root, attempt, network_mode="allowlist") as runtime:
        from yialpha.ledger.outcome_compute import compute_outcomes
        from yialpha.ledger.outcomes import pending_predictions
        from yialpha.ledger.scoreboard import build_scoreboard, render_scoreboard_markdown
        from yialpha.ledger.sqlite import get_connection

        recorder = EvidenceRecorder(root, attempt, sample_kind=PROCESS_KIND)
        with _VendorEvidence(recorder).installed(include_reference_seam=False) as evidence:
            now = now_fn().isoformat(timespec="seconds")
            before = pending_predictions(now)
            reports = []
            for _ in range(len(before) + 1):
                old_count = get_connection().execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
                report = compute_outcomes(now, limit=24)
                reports.append(asdict(report))
                new_count = get_connection().execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
                if new_count == old_count or not pending_predictions(now):
                    break
            paths.write_json(f"attempts/{attempt}/scoring.json", {
                "sample_kind": PROCESS_KIND, "now_as_of": now,
                "execution_observed_at": _utc(), "queue_before": before,
                "reports": reports, "queue_after": pending_predictions(now),
                "request_ids": [item["request_id"] for item in evidence.calls],
            })
            board = build_scoreboard()
            paths.write_json(f"attempts/{attempt}/scoreboard.json", {
                "note": "固定输入流程验证，非预测能力评估", "scoreboard": board,
            })
            paths.write_bytes(
                f"attempts/{attempt}/scoreboard.md",
                ("> 固定输入流程验证，非预测能力评估（process validation; not predictive-ability evidence）\n\n"
                 + render_scoreboard_markdown(board)).encode("utf-8"), exclusive=True)
            audit = _report(paths, cohort_id, attempt, now, sample_kind=PROCESS_KIND)
        rows = get_connection().execute(
            "SELECT status,COUNT(*) FROM outcomes GROUP BY status").fetchall()
        remaining = pending_predictions(now)
    return {"cohort_id": cohort_id, "attempt_id": attempt, "root": str(root),
            "now_as_of": audit["observed_at"], "scoring_invoked": True,
            "outcomes_by_status": dict(rows), "pending_remaining": len(remaining),
            "network": {"mode": runtime["network_mode"],
                        "allowed_calls": runtime["allowed_calls"],
                        "denied_attempts": runtime["network_attempts"]},
            "source_snapshot": audit["source_snapshot"],
            "prediction_ability_evidence": False}
