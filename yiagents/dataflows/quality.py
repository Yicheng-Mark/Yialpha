"""Structured data-quality evidence for each analysis run.

When every vendor for a category fails, the router degrades to a sentinel
string (``NO_DATA_AVAILABLE`` / ``DATA_UNAVAILABLE``) that the analysts' LLM
turns into prose ("data not available..."). That grounding is good behaviour,
but it left the *final artifacts* with no machine-readable record of which
categories ran degraded — a "new HOLD report" was indistinguishable from a
"data vacuum" HOLD report (the run_robust success criterion could not tell
them apart).

This module is the recorder half of closing that gap: the router calls
:func:`record_sentinel` whenever it emits a sentinel, and the graph writes
the accumulated events into ``full_states_log_<date>.json`` as a
``data_quality`` block next to the final decision.

Like :mod:`yiagents.dataflows.config`, state lives in a ``ContextVar`` so
concurrent batch workers (which copy the submitting context) keep separate
per-run evidence. Recording is append-only and never raises into the data
path.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

#: Sentinel kinds, mirroring the router's two degradation outcomes.
KIND_NO_DATA = "no_data"                    # core category: NO_DATA_AVAILABLE
KIND_OPTIONAL_UNAVAILABLE = "optional_unavailable"  # optional: DATA_UNAVAILABLE

_events_var: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "yiagents_data_quality_events", default=None
)


def record_sentinel(method: str, kind: str, detail: str = "") -> None:
    """Append one degradation event for the current run context.

    Never raises: quality evidence must not be able to fail the data call it
    is describing. ``method`` is the router method name (e.g.
    ``get_stock_data``); ``detail`` carries the typed error's reason.
    """
    try:
        events = _events_var.get()
        if events is None:
            events = []
            _events_var.set(events)
        events.append(
            {"method": str(method), "kind": str(kind), "detail": str(detail)[:500]}
        )
    except Exception:  # noqa: BLE001 -- evidence must never break the run
        pass


def snapshot_quality() -> list[dict[str, Any]]:
    """Copy the current run's sentinel events (empty list when none)."""
    events = _events_var.get()
    return [dict(e) for e in events] if events else []


def reset_quality() -> None:
    """Clear the current context's events (call after consuming a snapshot)."""
    _events_var.set(None)


def summarize_quality(events: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Build the ``data_quality`` block written into full_states_log.

    ``core_sentinel_count`` counts core-category ``NO_DATA_AVAILABLE``
    events — the ones that mean "the analysis decided without this data",
    as opposed to optional enrichment categories that were simply absent.
    """
    events = events or []
    return {
        "sentinels": events,
        "core_sentinel_count": sum(1 for e in events if e.get("kind") == KIND_NO_DATA),
        "optional_sentinel_count": sum(
            1 for e in events if e.get("kind") == KIND_OPTIONAL_UNAVAILABLE
        ),
    }
