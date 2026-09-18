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

Like :mod:`yialpha.dataflows.config`, state lives in a ``ContextVar`` so
concurrent batch workers (which copy the submitting context) keep separate
per-run evidence. Recording is append-only and never raises into the data
path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

#: Sentinel kinds, mirroring the router's degradation outcomes.
KIND_NO_DATA = "no_data"                    # core category: NO_DATA_AVAILABLE
KIND_OPTIONAL_UNAVAILABLE = "optional_unavailable"  # optional: DATA_UNAVAILABLE
KIND_STALE_CACHE = "stale_cache"            # vendor failed; stale disk cache served
KIND_CORE_ERROR = "core_error"              # core category: every vendor errored

#: Counted into ``core_sentinel_count`` alongside :data:`KIND_NO_DATA` — both
#: mean "the analysis decided without this core category's data".
_CORE_SENTINEL_KINDS = frozenset({KIND_NO_DATA, KIND_CORE_ERROR})

_POLICY_REJECT = "reject"
_POLICY_WARN = "warn"


class DataVacuumError(RuntimeError):
    """Raised when ``data_vacuum_policy=reject`` and a run reached the decision
    stage with zero successful core-category data calls.

    Every core vendor either errored or reported no data — the run would have
    produced a "data vacuum HOLD" (a report that looks normal but was decided
    without any market/fundamental/news data). Rejecting it here keeps the
    failure typed and visible instead of silently degrading.
    """


_events_var: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "yialpha_data_quality_events", default=None
)

# Distinct CORE-category methods that returned data this run. Set in the
# submitting context (same contract as _events_var) so node contexts mutate the
# shared object the parent reads back. Only the router records successes —
# direct-connect tools do not participate in the vacuum verdict.
_success_var: ContextVar[set[str] | None] = ContextVar(
    "yialpha_data_quality_core_successes", default=None
)


def ensure_run_context() -> None:
    """Bind a fresh event list in the *caller's* context before a graph run.

    LangGraph executes every node task inside ``copy_context().run(...)``.
    A ContextVar value first bound inside a node is invisible to the caller
    that later snapshots — without this call, ``record_sentinel`` fires inside
    node contexts and ``snapshot_quality`` in the parent always sees ``[]``
    (the entire DEGRADED evidence chain goes dead). Binding the list here,
    in the context that submits the graph, makes every copied node context
    inherit the *same list object*; ``list.append`` then mutates the object
    the parent reads back. Appends are GIL-atomic and recording stays
    append-only, so concurrent node contexts are safe.

    Always installs a fresh list (and a fresh core-success set): a crashed
    prior run in the same worker context cannot leak its events into this one.
    """
    _events_var.set([])
    _success_var.set(set())


def record_sentinel(
    method: str, kind: str, detail: str = "", qualifier: str = "",
) -> None:
    """Append one degradation event for the current run context.

    Never raises: quality evidence must not be able to fail the data call it
    is describing. ``method`` is the router method name (e.g.
    ``get_stock_data``); ``detail`` carries the typed error's reason;
    ``qualifier`` labels the call's severity-bearing parameter (e.g. a
    klines call's ``price_type``) so the classifier can grade enrichment
    misses below core misses — see :func:`classify_quality`.
    """
    try:
        events = _events_var.get()
        if events is None:
            events = []
            _events_var.set(events)
        events.append(
            {
                "method": str(method),
                "kind": str(kind),
                "detail": str(detail)[:500],
                "qualifier": str(qualifier or ""),
            }
        )
    except Exception:  # noqa: BLE001 -- evidence must never break the run
        pass


def _success_key(method: str, qualifier: str = "") -> str:
    """Composite success-set key: ``method[qualifier]`` when qualified.

    Recovery must not cross price bases: an INDEX-klines success must not
    forgive a LAST-book outage (and a qualified router success must not
    forgive the overlay's unqualified decision-time sentinel), so successes
    are keyed at the same (method, qualifier) granularity as sentinels.
    """
    m, q = str(method), str(qualifier or "")
    return f"{m}[{q}]" if q else m


def _is_recovered(event: dict[str, Any], core_successes: set[str]) -> bool:
    """True when the SAME method AND qualifier served data this run.

    An event with a qualifier matches only a qualified success of that exact
    basis; an unqualified event matches a bare-method success. This is what
    keeps the overlay's decision-time sentinels (unqualified, never retried)
    critical even while the analysts' qualified router calls succeeded.
    """
    method = str(event.get("method", ""))
    qualifier = str(event.get("qualifier", "") or "")
    if qualifier:
        return _success_key(method, qualifier) in core_successes
    return method in core_successes


def record_success(method: str, qualifier: str = "") -> None:
    """Record that a CORE-category router call returned data this run.

    The success side of the vacuum verdict: a run is a *data vacuum* only when
    core calls were attempted (sentinels exist) yet none succeeded. Any
    recorded success also un-vacuums. ``qualifier`` (the sentinel side's
    severity-bearing parameter, e.g. a klines call's ``price_type``) keys the
    success at the same granularity as the sentinel it may forgive — see
    :func:`_is_recovered`. Never raises — same contract as
    :func:`record_sentinel`.
    """
    try:
        methods = _success_var.get()
        if methods is None:
            methods = set()
            _success_var.set(methods)
        methods.add(_success_key(method, qualifier))
    except Exception:  # noqa: BLE001 -- evidence must never break the run
        pass


def snapshot_quality() -> list[dict[str, Any]]:
    """Copy the current run's sentinel events (empty list when none)."""
    events = _events_var.get()
    return [dict(e) for e in events] if events else []


def snapshot_core_successes() -> set[str]:
    """Copy the core-category methods that returned data this run."""
    methods = _success_var.get()
    return set(methods) if methods else set()


def reset_quality() -> None:
    """Clear the current context's events (call after consuming a snapshot)."""
    _events_var.set(None)
    _success_var.set(None)


def is_data_vacuum(
    events: list[dict[str, Any]] | None,
    core_successes: set[str] | None = None,
) -> bool:
    """True when core calls were attempted but none returned data.

    ``attempted`` means at least one core sentinel was recorded; without one
    (e.g. a context where no core tool ran) there is no evidence of a vacuum,
    so the verdict is False — the gate must not reject runs it cannot judge.
    """
    events = events or []
    core_sentinels = [e for e in events if e.get("kind") in _CORE_SENTINEL_KINDS]
    if not core_sentinels:
        return False
    return not (core_successes or set())


def summarize_quality(
    events: list[dict[str, Any]] | None,
    core_successes: set[str] | None = None,
) -> dict[str, Any]:
    """Build the ``data_quality`` block written into full_states_log.

    ``core_sentinel_count`` counts core-category degradation events — both
    ``no_data`` (clean "no usable data") and ``core_error`` (every vendor
    errored) — the ones that mean "the analysis decided without this data",
    as opposed to optional enrichment categories that were simply absent.
    ``degraded_count`` additionally folds in stale-cache serves for reporting;
    the vacuum verdict itself stays on core evidence only.
    """
    events = events or []
    core_successes = core_successes or set()
    # Recovery-aware: a core-kind sentinel whose own (method, qualifier)
    # later served data is "(recovered)" in classify_quality — the run's
    # FINAL data state has that basis, so run_robust's DEGRADED verdict
    # must not re-run a completed run over it either (the two consumers of
    # the same evidence cannot disagree about the same event).
    core_sentinel_count = sum(
        1 for e in events
        if e.get("kind") in _CORE_SENTINEL_KINDS
        and not _is_recovered(e, core_successes)
    )
    stale_cache_count = sum(1 for e in events if e.get("kind") == KIND_STALE_CACHE)
    return {
        "sentinels": events,
        "core_sentinel_count": core_sentinel_count,
        "core_error_count": sum(
            1 for e in events
            if e.get("kind") == KIND_CORE_ERROR
            and not _is_recovered(e, core_successes)
        ),
        "core_ok_count": len(core_successes),
        "optional_sentinel_count": sum(
            1 for e in events if e.get("kind") == KIND_OPTIONAL_UNAVAILABLE
        ),
        "stale_cache_count": stale_cache_count,
        "degraded_count": core_sentinel_count + stale_cache_count,
        # The vacuum verdict itself stays on raw evidence: "attempted and
        # zero successes" (any recorded success un-vacuums).
        "data_vacuum": (
            any(e.get("kind") in _CORE_SENTINEL_KINDS for e in events)
            and not core_successes
        ),
    }


def _vacuum_policy() -> str:
    """Resolve ``data_vacuum_policy`` from config; unknown values fail closed.

    An unrecognized value resolves to ``reject`` with a loud warning: treating
    a typo as "warn" would let data-vacuum runs slip through silently — the
    exact failure mode this gate exists to close. ``config-check`` validates
    the value up front so operators see the typo there first.
    """
    from yialpha.dataflows.config import get_config  # local: avoid import cycle

    try:
        raw = get_config().get("data_vacuum_policy", _POLICY_REJECT)
    except Exception:  # noqa: BLE001 -- config unavailable: fail closed
        return _POLICY_REJECT
    policy = str(raw or _POLICY_REJECT).strip().lower()
    if policy not in (_POLICY_REJECT, _POLICY_WARN):
        logger.warning(
            "Invalid data_vacuum_policy=%r (expected 'reject' or 'warn'); "
            "failing closed to 'reject'.", raw,
        )
        return _POLICY_REJECT
    return policy


def check_data_vacuum() -> None:
    """Enforce ``data_vacuum_policy`` against the current run's evidence.

    Called at the decision stage (Trader node entry). Under ``reject`` a data
    vacuum raises :class:`DataVacuumError` carrying the sentinel evidence, so
    batch/robust orchestrators record a typed failure instead of a
    normal-looking HOLD report. Under ``warn`` the vacuum is logged loudly and
    the run proceeds — the report still ships the DEGRADED banner.
    """
    events = snapshot_quality()
    if not is_data_vacuum(events, snapshot_core_successes()):
        return
    core_events = [e for e in events if e.get("kind") in _CORE_SENTINEL_KINDS]
    evidence = "; ".join(
        f"{e.get('method')}: {e.get('detail') or e.get('kind')}" for e in core_events
    )
    if _vacuum_policy() == _POLICY_REJECT:
        raise DataVacuumError(
            "data vacuum: every core data call failed — no market/fundamental/"
            f"news data reached the decision stage ({evidence}). Set "
            "data_vacuum_policy=warn to allow degraded runs."
        )
    logger.warning(
        "DATA VACUUM (policy=warn): proceeding without core data — %s", evidence
    )


def gate_on_data_vacuum(handler: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a graph node handler so :func:`check_data_vacuum` runs before it.

    Used at the Trader node: the analysts have finished by then, so the
    evidence is complete, and no decision-stage LLM call has been billed yet
    when the gate rejects. The wrapper is transparent for every non-vacuum
    run (byte-identical state in and out).
    """
    def _gated(*args: Any, **kwargs: Any) -> Any:
        check_data_vacuum()
        return handler(*args, **kwargs)

    return _gated


# ---------------------------------------------------------------------------
# V2.0 P0.3 — four-tier data-quality classification (critical-data gate)
# ---------------------------------------------------------------------------

#: Tiers consumed by the tradeability gate. ``DEGRADED_AUXILIARY`` degrades
#: confidence and is disclosed; only ``DEGRADED_CRITICAL`` and ``INVALID``
#: can veto a trade.
TIER_GOOD = "GOOD"
TIER_DEGRADED_AUXILIARY = "DEGRADED_AUXILIARY"
TIER_DEGRADED_CRITICAL = "DEGRADED_CRITICAL"
TIER_INVALID = "INVALID"

#: Categories whose failure means the decision lacks CRITICAL inputs: the
#: price/indicator books (price, ATR, perp/spot klines incl. mark/index) and
#: core fundamentals. News, macro, prediction markets, social and the A-share
#: enrichment stacks are deliberately absent — a Reddit or Binance Square
#: outage must not veto an otherwise fully-priced BTC perp trade (freeze
#: check #2). Extending this set is a reviewable semantic change.
CRITICAL_CATEGORIES = frozenset({
    "core_stock_apis",
    "technical_indicators",
    "fundamental_data",
    "binance_perp",
    "binance_spot",
})

#: The Binance price categories get METHOD+QUALIFIER severity, not bare
#: category-level: only the klines/indicator engines (the price/ATR book
#: every number in the decision rests on) are critical. The enrichment
#: tools — funding, OI, LSR, taker, premium, basis, depth, vision — are
#: auxiliary: their failure degrades and is disclosed but never vetoes. The
#: klines tool itself is qualifier-graded: last/mark are the entry and
#: liquidation price books (critical), while ``index`` is the settlement
#: fair-value ANCHOR — an index-only miss must not veto a run whose traded
#: and liquidation prices are intact. This fixes two inverse failures: a
#: tokenized-stock perp structurally lacking a spot leg (basis tools error)
#: was being NO_TRADE'd on an absent CAPABILITY, while the actual indicator
#: engine failing (previously an unregistered method name, classified
#: auxiliary) did not veto at all — and a third, an index-kline-only outage
#: reading as a lost price book.
_PERP_PRICE_CATEGORIES = frozenset({"binance_perp", "binance_spot"})
_PERP_CORE_METHODS = frozenset({
    "get_binance_klines",
    "get_binance_indicators",
    "get_binance_spot_klines",
    "get_binance_spot_indicators",
})
#: Qualifiers that downgrade an otherwise-core klines miss to auxiliary.
_KLINES_AUX_QUALIFIERS = frozenset({"index"})


def _is_critical_failure(method: str, category: str, qualifier: str = "") -> bool:
    """Per-method (+qualifier) severity: core-method failure in the Binance
    price categories is critical; their enrichment tools — and klines
    enrichment price bases — are auxiliary."""
    if category in _PERP_PRICE_CATEGORIES:
        if method not in _PERP_CORE_METHODS:
            return False
        return qualifier not in _KLINES_AUX_QUALIFIERS
    return category in CRITICAL_CATEGORIES


def is_perp_core_method(method: str) -> bool:
    """True when ``method`` is one of the perp price-book core engines.

    The Binance price categories are OPTIONAL as categories (their enrichment
    tools degrade fail-soft), yet the klines/indicator engines are the CORE
    price book of a perp run. Router call sites use this to record the
    SUCCESS side of those methods even when served from an optional category,
    so :func:`classify_quality` can tell a hard outage (sentinel, no later
    success) from a recoverable one (bad arg / transient throttle → sentinel
    → the model retried and got data).
    """
    return method in _PERP_CORE_METHODS


def classify_quality(
    events: list[dict[str, Any]] | None,
    core_successes: set[str] | None = None,
) -> dict[str, Any]:
    """Classify a run's sentinel evidence into the four-tier scale.

    Returns ``{tier, critical_data_available, critical_missing,
    auxiliary_degraded}``:

    - ``INVALID`` — data vacuum (core calls attempted, none succeeded). The
      vacuum gate refuses such runs under the default policy; this tier keeps
      the classification complete for ``warn``-policy runs and post-hoc reads.
    - ``DEGRADED_CRITICAL`` — at least one sentinel in a critical category
      (price/indicators/fundamentals; in the Binance price categories only
      the klines/indicator core methods count — see
      :data:`_PERP_CORE_METHODS`) from a method+qualifier that never served
      data this run. A sentinel whose own method AND qualifier later
      succeeded is downgraded to a "(recovered)" auxiliary line: the model
      hit an instructive sentinel (bad argument, transient throttle),
      retried the SAME basis, and got the data — the price book exists, so
      the attempt is disclosed but must not veto. Recovery never crosses
      bases (an index-klines success does not forgive a last-book outage)
      and never forgives the overlay's unqualified decision-time sentinels
      (no router success carries a bare-method key for them to match).
      The tradeability gate turns this tier into NO_TRADE.
    - ``DEGRADED_AUXILIARY`` — only auxiliary degradation (news/macro/social
      absent or stale-cache serves). Confidence penalty + disclosure, never a
      veto.
    - ``GOOD`` — no degradation evidence.

    A sentinel method whose category cannot be resolved counts as auxiliary
    (fail-open classification); its raw name is still listed so the report
    shows what actually happened.
    """
    events = events or []
    core_successes = core_successes or set()
    if is_data_vacuum(events, core_successes):
        return {
            "tier": TIER_INVALID,
            "critical_data_available": False,
            "critical_missing": ["(all core categories failed)"],
            "auxiliary_degraded": [],
        }

    critical_missing: list[str] = []
    auxiliary: list[str] = []
    for e in events:
        method = str(e.get("method", ""))
        kind = str(e.get("kind", ""))
        qualifier = str(e.get("qualifier", "") or "")
        try:
            from .interface import get_category_for_method  # local: avoid cycle

            category = get_category_for_method(method)
        except Exception:  # noqa: BLE001 -- unknown method: classify aux, keep name
            category = ""
        if kind in _CORE_SENTINEL_KINDS or kind == KIND_OPTIONAL_UNAVAILABLE:
            label = f"{method}[{qualifier}]({kind})" if qualifier else (
                f"{method}({kind})" if kind else method
            )
            if _is_recovered(e, core_successes):
                # Same method AND qualifier served data later in the run —
                # disclosed as auxiliary, never a veto.
                auxiliary.append(f"{label} (recovered)")
            elif _is_critical_failure(method, category, qualifier):
                critical_missing.append(label)
            else:
                auxiliary.append(label)
        elif kind == KIND_STALE_CACHE:
            auxiliary.append(f"{method}(stale_cache)")

    if critical_missing:
        tier = TIER_DEGRADED_CRITICAL
    elif auxiliary:
        tier = TIER_DEGRADED_AUXILIARY
    else:
        tier = TIER_GOOD
    return {
        "tier": tier,
        "critical_data_available": not critical_missing,
        "critical_missing": critical_missing,
        "auxiliary_degraded": auxiliary,
    }
