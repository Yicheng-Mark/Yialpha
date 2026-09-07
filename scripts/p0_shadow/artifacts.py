"""Append-only evidence artifacts for the P0 shadow validation runner.

This module records bytes supplied by a caller; it does not fetch data or
install a vendor HTTP hook. B0 exercises it with simulated vendor responses.
The caller must identify the same collection call and its attempt explicitly.
An SDK/normalized response is never labelled as a vendor HTTP response here.
No project package is imported, so dotenv must still be disabled by the runner
before the runner imports production ledger or dataflow modules.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from .guard import PathGuard

ARTIFACT_SCHEMA_VERSION = "p0_shadow_artifacts_v1"
SAMPLE_KINDS = frozenset({"offline_fixture", "process_validation_fixed_input"})
EVIDENCE_LEVELS = frozenset({
    "offline_fixture", "vendor_http_response", "vendor_library_output",
})

VALIDATION_MATRIX = tuple(
    {
        "matrix_id": f"{group}-{horizon}d",
        "group_id": group,
        "run_ticker": ticker,
        "asset_type": "crypto_perp",
        "instrument_class": instrument_class,
        "instrument_id": instrument,
        "prediction_scope": scope,
        "horizon_days": horizon,
        "requires_real_utc_weekend": scope == "UNDERLYING",
    }
    for group, ticker, instrument_class, instrument, scope in (
        ("C-BTC", "BTCUSDT", "pure_crypto_perp", "BTCUSDT", "CONTRACT"),
        ("C-ETH", "ETHUSDT", "pure_crypto_perp", "ETHUSDT", "CONTRACT"),
        ("C-MU", "MUUSDT", "stock_perp", "MUUSDT", "CONTRACT"),
        ("U-MU-WE", "MUUSDT", "stock_perp", "MU", "UNDERLYING"),
    )
    for horizon in (1, 5, 21)
)
MATRIX = VALIDATION_MATRIX

# Only public market request parameters and non-secret response headers survive.
# Unknown keys are omitted, including custom spelling of auth/cookie/API keys.
_QUERY_KEYS = frozenset({
    "symbol", "pair", "interval", "startTime", "endTime", "limit", "period",
    "start", "end", "start_date", "end_date", "price_type", "closed_as_of",
    "events", "includeAdjustedClose", "period1", "period2",
})
_HEADER_KEYS = frozenset({
    "content-type", "content-length", "date", "last-modified", "etag",
    "retry-after", "x-mbx-used-weight", "x-mbx-used-weight-1m",
})
_METADATA_KEYS = frozenset({
    "method", "endpoint", "symbol", "canonical_symbol", "interval", "venue",
    "price_type", "start_date", "end_date", "closed_as_of", "status_code",
    "cache_status", "cache_source_request_id", "page", "parser_version",
})
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")


def _identity(value: str, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a nonempty plain identifier")
    return value


def _utc(value: str | datetime, field: str) -> str:
    try:
        stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(stamp, datetime) or stamp.tzinfo is None:
            raise ValueError
        if stamp.utcoffset() is None:
            raise ValueError
        return stamp.astimezone(UTC).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc


def _canonical_bytes(payload: Any) -> bytes:
    """Validate JSON before any write; non-finite data is not evidence JSON."""
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _public_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        _canonical_bytes(value)
        return value
    raise ValueError("public request metadata values must be JSON scalars")


def _public_fields(value: Any, allowed: frozenset[str], *, lower_keys: bool = False) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("headers and query metadata must be mappings")
    cleaned = {}
    for key, item in value.items():
        name = str(key).lower() if lower_keys else str(key)
        if name in allowed:
            cleaned[name] = _public_scalar(item)
    return cleaned


def _public_url(value: Any) -> str:
    """Retain a reference URL without credentials, private query or fragment.

    The returned string is provenance only: this module has no network path.
    """
    if not isinstance(value, str):
        raise ValueError("url metadata must be a string")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url metadata must name an HTTP(S) source")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credential-bearing source URLs are not evidence metadata")
    # Access .port here to reject malformed ports before creating any files.
    port = parsed.port
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    public_query = [(key, val) for key, val in parse_qsl(parsed.query) if key in _QUERY_KEYS]
    return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(public_query), ""))


def _request_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("request_metadata must be a mapping")
    cleaned = _public_fields(value, _METADATA_KEYS)
    if "endpoint" in cleaned:
        endpoint = cleaned["endpoint"]
        if not isinstance(endpoint, str) or not endpoint.startswith("/"):
            raise ValueError("endpoint must be a path without query parameters")
        if "?" in endpoint or "#" in endpoint or endpoint.startswith("//"):
            raise ValueError("endpoint must be a path without query parameters")
    for key in ("query", "params"):
        if key in value:
            cleaned[key] = _public_fields(value[key], _QUERY_KEYS)
    for key in ("headers", "request_headers", "response_headers"):
        if key in value:
            cleaned[key] = _public_fields(value[key], _HEADER_KEYS, lower_keys=True)
    if "url" in value:
        cleaned["url"] = _public_url(value["url"])
    return cleaned


def create_manifest(
    root: str | Path,
    cohort_id: str,
    *,
    sample_kind: str,
    code_identity: Mapping[str, Any],
    created_at: str | datetime,
    versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new manifest once; never update an existing cohort in place.

    Production versions must be supplied from the runner after its import
    guard. Omitting them explicitly leaves version capture incomplete.
    ``code_identity`` is a caller-selected identity summary, not an environment
    dump; callers must never put secrets or full configuration in this mapping.
    """
    guard = PathGuard(root)
    _identity(cohort_id, "cohort_id")
    if sample_kind not in SAMPLE_KINDS:
        raise ValueError("unsupported sample_kind")
    if not isinstance(code_identity, Mapping) or not code_identity:
        raise ValueError("code_identity must be a nonempty identity mapping")
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "sample_kind": sample_kind,
        "created_at": _utc(created_at, "created_at"),
        "root": str(guard.resolve("manifest.json").parent),
        "code_identity": dict(code_identity),
        "production_contract_versions": dict(versions) if versions is not None else None,
        "production_versions_captured": versions is not None,
        "analysis_only": True,
        "live_execution": False,
        "live_execution_enabled": False,
        "llm_used": False,
        "matrix": [dict(item) for item in VALIDATION_MATRIX],
        "boundaries": {
            "vendor_http_hook_installed": False,
            "network_collection_performed_by_this_module": False,
            "caller_supplied_same_call_evidence": True,
            "prediction_performance_evidence": False,
            "real_natural_maturity_verified": False,
        },
        "paths": {
            "ledger": "ledger/portfolio.db",
            "cache": "cache/<attempt_id>/",
            "raw": "raw/<request_id>/",
            "attempts": "attempts/<attempt_id>/",
            "snapshots": "snapshots/",
            "costs": "costs/",
            "reports": "reports/<attempt_id>/",
            "checkpoints": "checkpoints/<attempt_id>/",
        },
    }
    _canonical_bytes(manifest)
    guard.write_json("manifest.json", manifest, exclusive=True)
    return manifest


class EvidenceRecorder:
    """Persist one caller-identified response and normalization append-only.

    ``record_response`` must receive bytes from that same collection call.
    The recorder cannot establish where supplied bytes actually originated;
    the caller's live HTTP integration is a separate, not-yet-installed step.
    """

    def __init__(
        self, root: str | Path, attempt_id: str, *, sample_kind: str = "offline_fixture",
    ) -> None:
        self.guard = PathGuard(root)
        self.attempt_id = _identity(attempt_id, "attempt_id")
        if sample_kind not in SAMPLE_KINDS:
            raise ValueError("unsupported sample_kind")
        self.sample_kind = sample_kind

    def record_response(
        self,
        *,
        source: str,
        request_metadata: Mapping[str, Any],
        raw_bytes: bytes,
        normalized_payload: Any,
        started_at: str | datetime,
        observed_at: str | datetime,
        evidence_level: str,
        call_id: str,
        prediction_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Return the committed record linking immutable files and their hashes.

        A request UUID always denotes a new response observation, even for
        identical content or repeated ``call_id``. ``record.json`` is written
        last: its presence means both payload files were successfully written.
        """
        _identity(source, "source")
        _identity(call_id, "call_id")
        if evidence_level not in EVIDENCE_LEVELS:
            raise ValueError("unsupported evidence_level")
        if self.sample_kind == "offline_fixture" and evidence_level != "offline_fixture":
            raise ValueError("offline fixtures cannot be labelled as real vendor evidence")
        if not isinstance(raw_bytes, bytes):
            raise TypeError("raw_bytes must be the original response bytes")
        if isinstance(prediction_ids, (str, bytes)):
            raise ValueError("prediction_ids must be a sequence of identifiers")
        predictions = [_identity(item, "prediction_id") for item in prediction_ids]
        start = _utc(started_at, "started_at")
        observed = _utc(observed_at, "observed_at")
        if datetime.fromisoformat(observed) < datetime.fromisoformat(start):
            raise ValueError("observed_at must not precede started_at")
        request = _request_metadata(request_metadata)
        normalized_bytes = _canonical_bytes(normalized_payload)
        request_id = uuid4().hex
        base = f"raw/{request_id}"
        raw_relative = f"{base}/response.bin"
        normalized_relative = f"{base}/normalized.json"
        record_relative = f"{base}/record.json"
        normalized_document = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "representation": "normalized_payload",
            "is_vendor_raw_response": False,
            "payload": normalized_payload,
        }
        _canonical_bytes(normalized_document)
        self.guard.write_bytes(raw_relative, raw_bytes, exclusive=True)
        normalized_path = self.guard.write_json(
            normalized_relative, normalized_document, exclusive=True,
        )
        normalized_file_bytes = normalized_path.read_bytes()
        record = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "request_id": request_id,
            "call_id": call_id,
            "attempt_id": self.attempt_id,
            "prediction_ids": predictions,
            "sample_kind": self.sample_kind,
            "source": source,
            "evidence_level": evidence_level,
            "started_at": start,
            "observed_at": observed,
            "request_metadata": request,
            "metadata_policy": "public_market_allowlist_v1",
            "vendor_http_hook_installed": False,
            "origin_verified_by_recorder": False,
            "raw": {
                "path": raw_relative,
                "sha256": sha256(raw_bytes).hexdigest(),
                "byte_count": len(raw_bytes),
                "representation": "caller_supplied_response_bytes",
            },
            "normalized": {
                "path": normalized_relative,
                "sha256": sha256(normalized_file_bytes).hexdigest(),
                "byte_count": len(normalized_file_bytes),
                "payload_sha256": sha256(normalized_bytes).hexdigest(),
                "representation": "normalized_payload",
                "is_vendor_raw_response": False,
            },
            "record_path": record_relative,
        }
        self.guard.write_json(record_relative, record, exclusive=True)
        return record
