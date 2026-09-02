"""Corporate-action records for perp underlyings (V2.1 skeleton, no storage).

Record stage only: this module fixes the value vocabulary and the
disclosure/validation contract for corporate actions (dividends, splits,
mergers, ticker changes) on tokenized-stock perp underlyings. There is
deliberately NO ledger table and NO vendor wiring yet — persistence and
backtest replay integration arrive with V2.4, when the perp→underlying
return attribution must split raw price moves from corporate-action
adjustments. Defining the shape now keeps those later PRs additive.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from yialpha.ledger.sqlite import utc_now_iso

#: The corporate-action vocabulary (record stage: dividends + splits are the
#: V2.4 return-attribution drivers; merger/ticker_change cover identity
#: drift; ``other`` is the explicit escape hatch).
CorporateActionType = Literal[
    "dividend", "split", "merger", "ticker_change", "other"
]

_VALID_ACTION_TYPES = frozenset(
    {"dividend", "split", "merger", "ticker_change", "other"}
)


@dataclass(frozen=True)
class CorporateActionRecord:
    """One corporate action on an instrument or its underlying.

    ``available_at`` follows the ledger time discipline: when the action
    became knowable to this framework (ISO date or datetime), which V2.4
    replay will bound on exactly like ``instrument_snapshots.
    snapshot_available_at``. ``event_date`` is the effective date of the
    action itself (ISO date). ``details`` carries action-type-specific
    payload (e.g. ``{"amount": 0.25, "currency": "USD"}`` for a dividend).
    """

    symbol: str
    action_type: CorporateActionType
    event_date: str
    available_at: str | None = None
    source: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


def validate_corporate_action(record: CorporateActionRecord) -> None:
    """Raise ``ValueError`` on a bad action type or malformed ISO dates.

    ``event_date`` must be a strict ISO date (``YYYY-MM-DD``);
    ``available_at``, when present, may be an ISO date or datetime. Returns
    ``None`` when the record is well-formed. Validation is explicit (callers
    validate at ingest) — rendering never validates, so a malformed legacy
    row can still be disclosed.
    """
    if record.action_type not in _VALID_ACTION_TYPES:
        raise ValueError(
            f"unknown corporate action type {record.action_type!r} "
            f"for {record.symbol!r}"
        )
    try:
        date.fromisoformat(record.event_date)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"event_date must be an ISO date (YYYY-MM-DD), "
            f"got {record.event_date!r}"
        ) from exc
    if record.available_at is not None:
        _require_iso_date_or_datetime(record.available_at)


def _require_iso_date_or_datetime(value: str) -> None:
    """Raise ValueError unless ``value`` parses as an ISO date or datetime."""
    try:
        date.fromisoformat(value)
        return
    except (TypeError, ValueError):
        pass
    try:
        datetime.fromisoformat(value)
        return
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"available_at must be an ISO date or datetime, got {value!r}"
        ) from exc


def render_corporate_action_disclosure(
    records: Sequence[CorporateActionRecord],
) -> str:
    """One disclosure line per action (``"<SYM>: <type> effective <date> ...``").

    Lines carry the source and, when known, the knowable-at timestamp, plus
    any ``details`` as ``key=value`` pairs — enough for an LLM-facing report
    to state exactly which actions were already public at the analysis date.
    Empty input renders the empty string.
    """
    lines: list[str] = []
    for record in records:
        parts = [
            f"{record.symbol}: {record.action_type} effective {record.event_date}"
        ]
        if record.available_at:
            parts.append(f"knowable at {record.available_at}")
        parts.append(f"source {record.source}")
        if record.details:
            rendered = ", ".join(
                f"{key}={record.details[key]}" for key in sorted(record.details)
            )
            parts.append(f"({rendered})")
        lines.append(" ".join(parts))
    return "\n".join(lines)
