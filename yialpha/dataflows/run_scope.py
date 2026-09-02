"""Per-run single-fetch scope for the deterministic analyst prefetches.

The LLM tool loop re-enters an analyst node on every tool call
(``setup.py`` wires ``tool_node -> current_analyst``), so a bundle prefetch
living in the node body re-executes from the top each round: a fundamentals
run that issues three tool calls refetches overview + three statements +
ETF data three times. Vendor disk caches blunt the HTTP cost but the
routing/rendering/quality-ledger work re-runs, and a live run re-hits live
endpoints within one decision.

This module is the same ContextVar contract the quality ledger
(:mod:`yialpha.dataflows.quality`) and the Tavily run budget use: the graph
runner binds a fresh cache dict in the PARENT context before the run
(``ensure_run_scope``), every node/worker context inherits the SAME dict
object, and ``reset_run_scope`` drops it at run end. A process serving many
runs (batch worker, web subprocess) therefore never serves one run's live
snapshot to the next — deliberately NOT a long-lived global cache.

Keys are caller-defined tuples — conventionally
``(kind, *identifiers)`` e.g. ``("fundamentals_bundle", asset_type, ticker,
date)``. Cached values are the FETCH results (dicts/strings), not rendered
blocks: rendering stays per-entry so status text can evolve.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_RUN_CACHE: ContextVar[dict[tuple[str, ...], Any] | None] = ContextVar(
    "yialpha_run_prefetch_cache", default=None
)

_T = TypeVar("_T")


def ensure_run_scope() -> None:
    """Bind a fresh cache dict in the CURRENT (parent) context.

    No-op when a scope is already bound — the graph runner calls this once
    per run; LangGraph node contexts and ``submit_with_context`` workers
    inherit the same dict object and mutations are visible everywhere.
    """
    if _RUN_CACHE.get() is None:
        _RUN_CACHE.set({})


def reset_run_scope() -> None:
    """Drop the bound cache (run end). The next run starts cold."""
    _RUN_CACHE.set(None)


def run_cached(key: tuple[str, ...], fn: Callable[[], _T]) -> _T:
    """Return the cached value for ``key`` or fetch it exactly once.

    When no scope is bound (unit tests, script callers that never ran
    ``ensure_run_scope``) this degrades to a direct call — never a silent
    cross-test leak. Exceptions are NOT cached: a transient fetch failure
    retries on the next node re-entry within the same run.
    """
    cache = _RUN_CACHE.get()
    if cache is None:
        return fn()
    if key not in cache:
        cache[key] = fn()
    return cache[key]
