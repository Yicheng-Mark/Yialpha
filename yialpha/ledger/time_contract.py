"""Versioned daily-close reference captured when a forecast is accepted.

This is a daily reference-return benchmark, not an executable submission
price. A session labelled D is conservatively knowable at D+1 00:00 UTC.
The exact latest eligible date is required: missing history never falls
back to an older bar. Vendor access is isolated here for offline tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from yialpha.dataflows.binance import binance_klines_frame
from yialpha.dataflows.y_finance import get_YFin_history_cached
from yialpha.versions import PREDICTION_TIME_VERSION

_TIME_FIELDS = (
    "prediction_formed_at",
    "reference_price_at",
    "reference_available_at",
    "reference_observed_at",
)
_FIELDS = frozenset({
    "version", *_TIME_FIELDS, "reference_price", "reference_source", "reference_error",
})


def aware_utc(value: str, *, field: str = "timestamp") -> datetime:
    """Require an explicit instant, rather than interpreting naive/date input."""
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    return stamp.astimezone(UTC)


def expected_reference_date(formed_at: datetime, *, equity: bool) -> date:
    """Latest UTC date whose daily bar is closed under this conservative rule."""
    day = formed_at.astimezone(UTC).date() - timedelta(days=1)
    while equity and day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def validate_timing(timing: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate and canonicalize new timing; None explicitly means legacy.

    Error snapshots keep the new version and formation time even when no
    reference exists. MACRO/POSITIONING also have no price, by definition.
    """
    if timing is None:
        return None
    if not isinstance(timing, Mapping):
        raise ValueError("timing must be a mapping or None")
    if set(timing) != _FIELDS:
        raise ValueError("timing must contain exactly the close-reference fields")
    result = dict(timing)
    if result["version"] != PREDICTION_TIME_VERSION:
        raise ValueError(f"unsupported prediction timing version {result['version']!r}")
    instants: dict[str, datetime] = {}
    for key in _TIME_FIELDS:
        value = result[key]
        if value is None and key != "prediction_formed_at":
            continue
        instants[key] = aware_utc(value, field=key)
        result[key] = instants[key].isoformat()
    for key in ("reference_source", "reference_error"):
        if result[key] is not None and (not isinstance(result[key], str) or not result[key]):
            raise ValueError(f"{key} must be a nonempty string or None")
    price = result["reference_price"]
    if price is not None:
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise ValueError("reference_price must be finite and positive")
        if not math.isfinite(price) or price <= 0:
            raise ValueError("reference_price must be finite and positive")
        result["reference_price"] = float(price)
        if len(instants) != len(_TIME_FIELDS) or result["reference_source"] is None:
            raise ValueError("a reference price requires all timestamps and its source")
        if result["reference_error"] is not None:
            raise ValueError("a valid reference price cannot carry reference_error")
    elif result["reference_price_at"] is not None or result["reference_available_at"] is not None:
        raise ValueError("reference price timestamps require a reference_price")
    ordered = [instants[key] for key in (
        "reference_price_at", "reference_available_at", "reference_observed_at",
        "prediction_formed_at",
    ) if key in instants]
    if ordered != sorted(ordered):
        raise ValueError("reference times must satisfy price <= available <= observed <= formed")
    return result


def encode_timing(timing: Mapping[str, Any] | None) -> str | None:
    """Canonical JSON is part of immutable prediction content identity."""
    canonical = validate_timing(timing)
    return None if canonical is None else json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _reference_frame(symbol: str, day: date, *, equity: bool, as_of: datetime) -> pd.DataFrame:
    """Narrow, deterministic vendor seam; fetch exactly one expected daily bar."""
    if equity:
        return get_YFin_history_cached(symbol, day.isoformat(), (day + timedelta(days=1)).isoformat())
    return binance_klines_frame(
        symbol, day.isoformat(), day.isoformat(), interval="1d", venue="binance_perp",
        price_type="last", closed_as_of=int(as_of.timestamp() * 1000),
    )


def capture_prediction_timing(instrument_id: str, prediction_scope: str) -> dict[str, Any]:
    """Freeze a locally sourced snapshot before accepting the tool submission.

    Formation is stamped after the reference fetch/observation; delayed
    flushes never re-fetch it. Failed fetches are explicit new-contract
    snapshots, never an implicit request for legacy historical scoring.
    """
    result: dict[str, Any] = dict.fromkeys(_FIELDS)
    result["version"] = PREDICTION_TIME_VERSION
    if prediction_scope in {"POSITIONING", "MACRO"}:
        result["prediction_formed_at"] = _now_utc().isoformat()
        return validate_timing(result)  # type: ignore[return-value]
    equity = prediction_scope == "UNDERLYING"
    result["reference_source"] = "yfinance:1d:close" if equity else "binance_perp:1d:last"
    requested_at = _now_utc()
    day = expected_reference_date(requested_at, equity=equity)
    try:
        frame = _reference_frame(instrument_id, day, equity=equity, as_of=requested_at)
        observed_at = _now_utc()
        result["reference_observed_at"] = observed_at.isoformat()
        formed_at = _now_utc()
        expected_day = expected_reference_date(formed_at, equity=equity)
        if frame is None or frame.empty or "Close" not in frame:
            result["reference_error"] = f"reference_bar_missing:{expected_day.isoformat()}"
        else:
            dates = pd.to_datetime(frame.index).date
            exact = frame.loc[dates == expected_day, "Close"]
            if len(exact) != 1:
                result["reference_error"] = f"reference_bar_missing_or_duplicate:{expected_day.isoformat()}"
            else:
                price = float(exact.iloc[0])
                close_at = datetime.combine(expected_day + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
                if not math.isfinite(price) or price <= 0:
                    result["reference_error"] = "reference_price_invalid"
                elif close_at > requested_at or close_at > observed_at:
                    result["reference_error"] = "reference_bar_not_closed"
                else:
                    result["reference_price"] = price
                    result["reference_price_at"] = close_at.isoformat()
                    result["reference_available_at"] = close_at.isoformat()
    except Exception as exc:  # noqa: BLE001 -- persist an explicit unscoreable snapshot
        formed_at = _now_utc()
        result["reference_error"] = f"reference_fetch_failed:{type(exc).__name__}"
    result["prediction_formed_at"] = formed_at.isoformat()
    return validate_timing(result)  # type: ignore[return-value]
