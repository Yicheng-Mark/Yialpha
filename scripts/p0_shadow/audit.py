"""Read-only, row-level audit for isolated P0 process-validation fixtures.

No project module is imported until an isolated runtime has been verified.
The production scoreboard predicate is a cross-check, never a scoring call.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .guard import PathGuard

_NEW = "close_reference_v1"
_LEGACY = "legacy_daily_v1"
_SAMPLE_KINDS = frozenset({"offline_fixture", "process_validation_fixed_input"})
_DIAGNOSTIC_LEGS = {"underlying", "underlying_close", "underlying_price_return", "basis_return"}
_NUMERIC_LEGS = ("contract_price_return", "funding_pnl", "fees", "slippage", "net_return")


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _stamp(value: Any, *, allow_date: bool = False) -> datetime:
    stamp = datetime.fromisoformat(value)
    if allow_date and isinstance(value, str) and len(value) == 10:
        stamp = stamp.replace(tzinfo=UTC)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("timestamp must carry a timezone")
    return stamp.astimezone(UTC)


def _object(raw: Any) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _safe(value: Any) -> Any:
    """Keep malformed numeric evidence serializable without concealing it."""
    if isinstance(value, float) and not math.isfinite(value):
        return f"nonfinite:{value}"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, bytes):
        return {"blob_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    return value


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        _safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _diagnose(row: dict[str, Any]) -> tuple[str | None, list[dict[str, str]], dict[str, Any]]:
    """Explain qualification independently, with named rules for each failure."""
    from yialpha.ledger.time_contract import validate_timing

    reasons: list[dict[str, str]] = []
    checks: dict[str, Any] = {"finite": {key: _finite(row.get(key)) for key in _NUMERIC_LEGS}}

    def reject(category: str, rule: str, detail: str) -> None:
        reasons.append({"category": category, "rule": rule, "detail": detail})

    def instant(value: Any, field: str) -> datetime | None:
        try:
            return _stamp(value)
        except (ValueError, TypeError, OverflowError) as exc:
            reject("time", "aware_timestamp_required", f"{field}: {exc}")
            return None

    if not checks["finite"]["net_return"]:
        reject("net", "finite_net_required", "net_return is missing or non-finite")
    raw = row.get("scoring_context")
    if raw is None:
        if row.get("prediction_timing") is not None:
            reject("version", "new_snapshot_requires_metadata", "new prediction cannot use historical NULL metadata")
        checks["qualification_policy"] = "legacy_null_metadata_compatibility"
        return (None if reasons else _LEGACY), reasons, checks
    try:
        context = _object(raw)
    except (ValueError, TypeError) as exc:
        reject("version", "metadata_object_required", str(exc))
        return None, reasons, checks
    version = context.get("version")
    checks["declared_version"] = version
    if version not in {_NEW, _LEGACY}:
        reject("version", "supported_version_required", str(version))
        return None, reasons, checks
    if version == _LEGACY and row.get("prediction_timing") is not None:
        reject("version", "snapshot_downgrade_forbidden", "versioned prediction cannot claim legacy outcome")

    times = {key: instant(context.get(key), key) for key in ("window_start", "window_end", "horizon_end")}
    available = instant(row.get("outcome_available_at"), "outcome_available_at")
    start, end, horizon = (times[key] for key in ("window_start", "window_end", "horizon_end"))
    scope = row.get("prediction_scope")
    if start is not None and end is not None and not start < end:
        reject("time", "nonempty_window_required", "window_start must precede window_end")
    if end is not None and available is not None:
        if version == _NEW and available != end:
            reject("time", "availability_equals_endpoint", "outcome_available_at differs from window_end")
        elif version == _LEGACY and scope == "POSITIONING":
            if start is not None and not start < available <= end:
                reject("time", "legacy_settlement_inside_window", "last settlement is outside the window")
        elif version == _LEGACY and available < end:
            reject("time", "endpoint_closed", "outcome is available before its endpoint")
    if version == _NEW and end is not None and horizon is not None and end > horizon:
        reject("time", "fixed_deadline", "window_end is after horizon_end")

    formed = None
    if version == _NEW:
        try:
            timing = validate_timing(_object(row.get("prediction_timing")))
            if timing is None:
                raise ValueError("frozen timing is absent")
        except (ValueError, TypeError, OverflowError) as exc:
            timing = None
            reject("snapshot", "valid_original_timing_required", str(exc))
        formed = instant(context.get("prediction_formed_at"), "prediction_formed_at")
        if timing is not None and formed is not None:
            if formed != _stamp(timing["prediction_formed_at"]):
                reject("snapshot", "formed_matches_prediction", "metadata formation differs from original timing")
            try:
                if int(row["horizon_days"]) != int(row["prediction_horizon_days"]):
                    reject("time", "horizon_matches_prediction", "outcome horizon differs from prediction")
                if horizon != formed + timedelta(days=int(row["horizon_days"])):
                    reject("time", "formation_plus_horizon", "horizon_end differs from formation plus calendar days")
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                reject("time", "valid_horizon_required", str(exc))
        if scope not in {"CONTRACT", "UNDERLYING", "POSITIONING"}:
            reject("version", "scoreable_scope_required", str(scope))
        if scope == "POSITIONING":
            if start != formed or end != horizon:
                reject("time", "positioning_formation_window", "POSITIONING must use formed_at through horizon_end")
        elif timing is not None:
            if timing.get("reference_price") is None or timing.get("reference_error") is not None:
                reject("snapshot", "frozen_price_required", str(timing.get("reference_error") or "reference price missing"))
            else:
                if start != _stamp(timing["reference_price_at"]):
                    reject("snapshot", "entry_matches_reference", "window_start differs from frozen reference_price_at")
                for key in ("reference_price", "reference_source"):
                    if context.get(key) != timing[key] or (key == "reference_price" and isinstance(context.get(key), bool)):
                        reject("snapshot", "reference_value_matches", f"{key} differs from original snapshot")
                for key in ("reference_available_at", "reference_observed_at"):
                    value = instant(context.get(key), key)
                    if value is not None and value != _stamp(timing[key]):
                        reject("snapshot", "reference_time_matches", f"{key} differs from original snapshot")

    try:
        missing = json.loads(row["legs_missing"]) if row.get("legs_missing") else []
        if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
            raise ValueError("legs_missing must be an array of strings")
        missing = set(missing)
    except (ValueError, TypeError) as exc:
        missing = {"invalid_legs_missing"}
        reject("missing", "valid_missing_legs_required", str(exc))
    checks["legs_missing"] = sorted(missing)
    if scope == "POSITIONING":
        if missing or not checks["finite"]["funding_pnl"]:
            reject("missing", "positioning_funding_required", "POSITIONING requires complete finite funding")
        elif checks["finite"]["net_return"]:
            expected_net = float(row["funding_pnl"])
            checks["recomputed_net_return"] = expected_net
            if not math.isclose(float(row["net_return"]), expected_net, abs_tol=1e-12):
                reject("net", "positioning_net_is_funding", "net_return differs from cumulative funding")
        return (None if reasons else version), reasons, checks

    is_perp = row.get("asset_type") == "crypto_perp" or "perp" in str(row.get("instrument_class") or "")
    equity = not is_perp or (scope == "UNDERLYING" and row.get("instrument_class") == "stock_perp")
    if version == _NEW:
        source = "yfinance:1d:close" if equity else "binance_perp:1d:last"
        if context.get("reference_source") != source:
            reject("snapshot", "source_matches_instrument", f"expected {source}")
        if equity and context.get("reference_basis_check") != "matched_frozen_close":
            reject("snapshot", "equity_basis_verified", "frozen equity price basis was not verified")
        for boundary, cutoff, field in ((start, formed, "window_start"), (end, horizon, "window_end")):
            if cutoff is None:
                continue
            try:
                day = cutoff.date() - timedelta(days=1)
                while equity and day.weekday() >= 5:
                    day -= timedelta(days=1)
                expected = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
                if boundary != expected:
                    reject("time", "exact_daily_boundary", f"{field} expected {expected.isoformat()}")
            except OverflowError as exc:
                reject("time", "representable_daily_boundary", str(exc))
    if missing - _DIAGNOSTIC_LEGS:
        reject("missing", "mandatory_legs_present", ",".join(sorted(missing - _DIAGNOSTIC_LEGS)))
    for key in ("contract_price_return", "fees", "slippage"):
        if not checks["finite"][key]:
            reject("cost" if key in {"fees", "slippage"} else "missing", "finite_attribution_required", key)
        elif version == _NEW and key in {"fees", "slippage"} and float(row[key]) < 0:
            reject("cost", "nonnegative_cost_required", key)
    funding: float | None = 0.0
    if scope == "CONTRACT" and is_perp:
        if not checks["finite"]["funding_pnl"]:
            reject("missing", "finite_funding_required", "CONTRACT perpetual requires funding_pnl")
            funding = None
        else:
            funding = float(row["funding_pnl"])
    if funding is not None and all(checks["finite"][key] for key in ("contract_price_return", "fees", "slippage")):
        expected_net = float(row["contract_price_return"]) + funding - float(row["fees"]) - float(row["slippage"])
        checks["recomputed_net_return"] = expected_net
        if checks["finite"]["net_return"] and not math.isclose(float(row["net_return"]), expected_net, rel_tol=1e-9, abs_tol=1e-12):
            reject("net", "attribution_composition", "net_return differs from price + funding - fees - slippage")
    return (None if reasons else version), reasons, checks


def _prediction_schedule(prediction: dict[str, Any], observed: datetime) -> dict[str, Any]:
    raw = prediction.get("timing")
    try:
        if raw is None:
            formed = _stamp(prediction["analysis_as_of"], allow_date=True)
            source = "legacy_analysis_as_of"
        else:
            timing = _object(raw)
            if timing.get("version") != _NEW:
                raise ValueError("unsupported prediction timing version")
            formed = _stamp(timing["prediction_formed_at"])
            source = "prediction_timing"
        horizon = formed + timedelta(days=int(prediction["horizon_days"]))
        return {
            "prediction_formed_at": formed.isoformat(), "formation_source": source,
            "horizon_end": horizon.isoformat(), "retry_deadline": (horizon + timedelta(days=7)).isoformat(),
            "due_at_observation": observed >= horizon, "schedule_error": None,
        }
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return {
            "prediction_formed_at": None, "formation_source": "invalid",
            "horizon_end": None, "retry_deadline": None, "due_at_observation": None,
            "schedule_error": str(exc),
        }


def build_audit(
    ledger_path: Path, *, cohort_id: str, observed_at: str,
    sample_kind: str = "offline_fixture",
) -> dict:
    """Read one consistent SQLite transaction; never migrate or compute outcomes.

    Call inside ``guard.isolated_runtime``. ``sample_kind`` labels the data
    mode: offline fixtures or B1/B2 process validation with real vendor calls.
    A runner must attach its SQLite backup fingerprint; the logical-row hash
    here is not a hash of a live main database file.
    """
    from .guard import require_runtime

    if not cohort_id or not isinstance(cohort_id, str):
        raise ValueError("cohort_id must be a nonempty string")
    if sample_kind not in _SAMPLE_KINDS:
        raise ValueError("unsupported sample_kind")
    ledger_path = Path(ledger_path)
    if not ledger_path.is_absolute():
        raise ValueError("ledger_path must be absolute")
    ledger_path = ledger_path.resolve(strict=True)
    paths = require_runtime(ledger_path.parent.parent)
    if paths.resolve("ledger/portfolio.db") != ledger_path:
        raise ValueError("audit source must be the isolated cohort ledger")
    observed = _stamp(observed_at)

    from yialpha import versions
    from yialpha.dataflows.config import get_config
    from yialpha.ledger.scoreboard import _scoring_version
    from yialpha.ledger.sqlite import ledger_db_path

    config = get_config()
    if Path(ledger_db_path()).resolve() != ledger_path:
        raise ValueError("active ledger configuration changed after runtime isolation")
    if config.get("analysis_only") is not True or config.get("live_execution_enabled") is not False:
        raise ValueError("audit requires analysis_only and disabled live execution")
    connection = sqlite3.connect(ledger_path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"predictions", "outcomes", "runs", "schema_meta"} <= tables:
            raise ValueError("source is missing required ledger tables; audit does not migrate")
        logical = {}
        for table in ("schema_meta", "runs", "predictions", "outcomes", "tickets", "evidence"):
            logical[table] = [dict(row) for row in connection.execute(f"SELECT * FROM {table}")] if table in tables else []
            logical[table].sort(key=lambda row: json.dumps(_safe(row), sort_keys=True))
    finally:
        connection.close()
    schema = next((row["value"] for row in logical["schema_meta"] if row["key"] == "schema_version"), None)
    runs = {row["run_id"]: row for row in logical["runs"]}
    tickets = {row["ticket_id"]: row for row in logical["tickets"]}
    evidence = {row["evidence_id"]: row for row in logical["evidence"]}
    outcomes_by_prediction: dict[str, list[dict]] = {}
    for row in logical["outcomes"]:
        outcomes_by_prediction.setdefault(row["prediction_id"], []).append(row)
    predictions = []
    eligible_ids: list[str] = []
    rejected_ids: list[str] = []
    versions_count: Counter = Counter()
    status_count: Counter = Counter()
    disagreements: list[str] = []
    queue_ids: list[str] = []
    for prediction in logical["predictions"]:
        prediction_id = prediction["prediction_id"]
        schedule = _prediction_schedule(prediction, observed)
        run = runs.get(prediction["run_id"])
        outcome_reports = []
        related = outcomes_by_prediction.get(prediction_id, [])
        matching_outcome = any(row["horizon_days"] == prediction["horizon_days"] for row in related)
        if not matching_outcome and (schedule["due_at_observation"] is True or schedule["schedule_error"] is not None):
            queue_ids.append(prediction_id)
        for outcome in related:
            status = outcome.get("status")
            status_count[status] += 1
            row = {
                **outcome, "prediction_timing": prediction.get("timing"),
                "prediction_horizon_days": prediction["horizon_days"],
                "prediction_scope": prediction["prediction_scope"],
                "instrument_class": (run or {}).get("instrument_class"),
                "asset_type": (run or {}).get("asset_type"),
                "scoring_context": outcome.get("scoring_context"),
            }
            version, reasons, checks = _diagnose(row)
            if run is None:
                reasons.append({"category": "source", "rule": "prediction_run_exists", "detail": prediction["run_id"]})
                version = None
            production_error = None
            try:
                production_version = _scoring_version(row) if status == "complete" and run is not None else None
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                production_version = None
                production_error = f"{type(exc).__name__}: {exc}"
            independent_version = version if status == "complete" else None
            if status != "complete":
                reasons.insert(0, {"category": "status", "rule": "complete_required", "detail": str(status)})
            agrees = production_version == independent_version and production_error is None
            if not agrees:
                disagreements.append(outcome["outcome_id"])
                reasons.append({"category": "qualification", "rule": "production_crosscheck_disagrees", "detail": production_error or f"independent={independent_version}; production={production_version}"})
            if production_version is not None:
                eligible_ids.append(outcome["outcome_id"])
                versions_count[production_version] += 1
                if not reasons:
                    reasons.append({"category": "qualification", "rule": "qualified_complete", "detail": production_version})
            else:
                rejected_ids.append(outcome["outcome_id"])
            ticket = tickets.get(outcome.get("ticket_id"))
            outcome_reports.append({
                "outcome_id": outcome["outcome_id"], "status": status,
                "scoreboard_eligible": production_version is not None,
                "scoring_version": production_version, "independent_scoring_version": independent_version,
                "qualification_crosscheck_agrees": agrees, "reasons": reasons, "checks": checks,
                "ledger_outcome": outcome, "ledger_outcome_sha256": _hash_json(outcome),
                "ticket_id": outcome.get("ticket_id"),
                "ticket_payload_sha256": hashlib.sha256(str(ticket["payload"]).encode("utf-8")).hexdigest() if ticket else None,
            })
        try:
            evidence_ids = json.loads(prediction.get("evidence_ids") or "[]")
            if not isinstance(evidence_ids, list):
                raise ValueError("evidence_ids is not an array")
            evidence_links = [{"evidence_id": item, "metadata": evidence.get(item)} for item in evidence_ids if isinstance(item, str)]
        except (ValueError, TypeError):
            evidence_links = [{"error": "invalid prediction evidence_ids JSON"}]
        if matching_outcome:
            state = "persisted_outcome"
            reason = "existing outcome retires this horizon from the automatic queue"
        elif schedule["schedule_error"]:
            state, reason = "invalid_schedule", schedule["schedule_error"]
        elif schedule["due_at_observation"]:
            state = "due_unscored"
            reason = "no persisted outcome; pending/failed/not_attempted cannot be distinguished from SQLite alone"
        else:
            state, reason = "not_due", "fixed horizon has not elapsed"
        predictions.append({
            "prediction_id": prediction_id, "run_id": prediction["run_id"],
            "instrument_id": prediction["instrument_id"], "scope": prediction["prediction_scope"],
            "horizon_days": prediction["horizon_days"], **schedule,
            "lifecycle_state": state, "lifecycle_reason": reason,
            "ledger_prediction": prediction, "ledger_prediction_sha256": _hash_json(prediction),
            "evidence_links": evidence_links, "outcomes": outcome_reports,
        })
    prediction_ids = {row["prediction_id"] for row in logical["predictions"]}
    orphan_outcomes = [row["outcome_id"] for row in logical["outcomes"] if row["prediction_id"] not in prediction_ids]
    return _safe({
        "audit_version": "p0_shadow_audit_v1", "cohort_id": cohort_id,
        "title": "固定输入流程验证，非预测能力评估",
        "sample_kind": sample_kind,
        "data_mode": "offline_fixture" if sample_kind == "offline_fixture" else "real_vendor_calls",
        "llm_used": False, "predictive_ability_evidence": False,
        "observed_at": observed.isoformat(), "source_ledger_path": str(ledger_path),
        "sqlite_schema_version": schema,
        "source_snapshot": {
            "read_consistency": "single SQLite read transaction",
            "logical_rows_sha256": _hash_json(logical),
            "database_backup": None, "backup_status": "runner_must_attach_consistent_backup_and_sha256",
            "main_database_file_hash_is_not_used": True,
        },
        "versions": {name: getattr(versions, name) for name in (
            "SCHEMA_VERSION", "FEATURE_VERSION", "COST_MODEL_VERSION", "TICKET_VERSION",
            "PREDICTION_TIME_VERSION", "OUTCOME_COMPUTE_VERSION",
        )},
        "summary": {
            "predictions": len(predictions), "outcomes": len(logical["outcomes"]),
            "due_predictions": sum(row["due_at_observation"] is True for row in predictions),
            "not_due_predictions": sum(row["due_at_observation"] is False for row in predictions),
            "invalid_schedules": sum(row["schedule_error"] is not None for row in predictions),
            "persisted_complete": status_count["complete"], "persisted_incomplete": status_count["incomplete"],
            "persisted_pending": status_count["pending"], "scoreboard_qualified_complete": len(eligible_ids),
            "rejected_complete": status_count["complete"] - len(eligible_ids),
            "in_memory_pending": None, "failed_attempts": None,
            "attempt_status_limitation": "pending/failed require the separate attempt report; never inferred from absent outcomes",
        },
        "queue_remaining_prediction_ids": queue_ids, "predictions": predictions,
        "scoreboard": {
            "eligible_outcome_ids": eligible_ids, "rejected_outcome_ids": rejected_ids,
            "by_scoring_version": dict(sorted(versions_count.items())),
            "mixed_scoring_versions": len(versions_count) > 1,
            "qualification_disagreements": disagreements,
            "metrics_interpretation": "qualification counts only; no prediction ability claim",
        },
        "source_integrity": {"orphan_outcome_ids": orphan_outcomes},
    })


def write_audit(paths: PathGuard, attempt_id: str, report: dict) -> None:
    """Create an immutable row-audit artifact under the cohort reports path."""
    from .guard import require_runtime

    if not isinstance(attempt_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", attempt_id):
        raise ValueError("attempt_id must be one safe path component")
    guard = require_runtime()
    if guard.root != paths.root:
        raise ValueError("audit writer must use the active runtime cohort")
    paths.write_json(f"reports/{attempt_id}/audit.json", _safe(report), exclusive=True)
