"""Blind-prediction capture: the ``submit_prediction`` tool + per-run buffer.

V2.1 record stage (Measurability): every perp-run analyst commits
direction / probability / expected return for ALL THREE ladder horizons
(1 / 5 / 21 days) BEFORE the debate, via ONE ``submit_prediction`` tool
call. The rows are written once by :func:`flush_predictions` and are
immutable from then on — accuracy attribution later scores exactly what
the analyst said in advance.

Design contracts:

* **Tool description is the only prompt surface.** No system prompt is
  edited anywhere; the binding :func:`make_submit_prediction_tool(instrument)`
  bakes the instrument id into the tool description so "direction refers
  to the instrument named in the description" is literal.
* **The tool never writes the ledger.** It validates each entry
  individually (invalid entries are rejected with their reason, valid ones
  accepted) and appends to a per-context capture buffer, returning a short
  confirmation string — a tool error can never crash a run.
* **ContextVar buffer, house pattern.** Same contract as
  :mod:`yialpha.dataflows.quality` / :mod:`yialpha.dataflows.run_scope`:
  the graph runner binds the buffer registry in the PARENT context
  (:func:`ensure_prediction_capture_scope`, alongside
  ``quality.ensure_run_context()``); node tasks then share the SAME
  registry object across the tool loop (analyst node -> ToolNode dispatch
  -> analyst node re-entry), and cross-task hand-off happens through
  registry mutations only (a ``ContextVar.set`` inside one node task is
  invisible to sibling tasks — that is exactly why quality binds in the
  parent). Without the parent binding the buffer still works within a
  single node's context (direct node invocation, tests).
* **No run context -> no-ops.** :func:`begin_prediction_capture` and
  :func:`flush_predictions` return without side effects when
  :func:`yialpha.ledger.run_context.current_ledger_run_context` is None;
  the tool then answers with an explanatory string instead of raising.

Settlement (:func:`settle_prediction_capture`) runs at the FINAL tool-loop
round only — a mid-loop flush could write rows whose evidence chain then
grows on later rounds, colliding with the rows' immutability check. An
empty buffer at settlement means the model never called the tool; that is
recorded as an ``optional_unavailable`` quality sentinel (the blind-
prediction capability is optional enrichment — the run's data quality is
unaffected without it, so ``no_data``/``core_error`` would overstate it).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from yialpha.dataflows import quality
from yialpha.ledger.evidence import evidence_ids_for_run
from yialpha.ledger.models import PredictionEntry, coerce_prediction_entry
from yialpha.ledger.predictions import submit_predictions
from yialpha.ledger.run_context import current_ledger_run_context

logger = logging.getLogger(__name__)

#: Stable tool name — the analyst-side binding and the graph ToolNode
#: registration must agree on it (dispatch is by name).
TOOL_NAME = "submit_prediction"

#: Free-form-but-checked optional fields (the ledger row does not enum
#: them; the tool rejects anything outside these sets, per entry).
_TARGET_CURRENCIES = frozenset({"USDT", "USD"})
_PRICE_BASES = frozenset({"last", "mark"})


class PredictionEntryInput(BaseModel):
    """One horizon of a blind forecast (fields mirror the ledger entry).

    Types are deliberately permissive: per-entry validation happens in the
    tool body (via :func:`yialpha.ledger.models.coerce_prediction_entry`)
    so one bad entry is rejected with a precise reason while the others in
    the same call are accepted.
    """

    horizon_days: int = Field(
        description="Forecast horizon in days: 1, 5, or 21 — exactly one entry per horizon."
    )
    direction: str = Field(
        description="'up', 'down' or 'flat' — the forecast price direction over the horizon."
    )
    prob_up: float | None = Field(
        default=None,
        description="P(return > 0) over the horizon, in [0, 1].",
    )
    expected_return: float | None = Field(
        default=None,
        description="Expected fractional return over the horizon (0.02 = +2%), or null.",
    )
    target_price: float | None = Field(
        default=None,
        description="Price target at horizon end, or null.",
    )
    target_currency: str | None = Field(
        default=None,
        description="'USDT' or 'USD' — the currency of target_price, or null.",
    )
    price_basis: str | None = Field(
        default=None,
        description="'last' or 'mark' — the price basis for target_price, or null.",
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence in this forecast set, in [0, 1].",
    )


_DESCRIPTION_TEMPLATE = """File YOUR blind forecasts for {instrument_phrase}.

Call this tool EXACTLY ONCE, in a single call carrying ALL THREE horizons
(horizon_days 1, 5 and 21 — one entry each), BEFORE you finish your
analysis and write your report. These are blind pre-debate forecasts: they
are committed before any debate, research or manager stage runs, they are
immutable once filed, and they will be scored later against realized
outcomes (directional accuracy, Brier score on prob_up, calibration), so
be honest and specific rather than hedged.

direction and prob_up always refer to {instrument_phrase} (the instrument
named in this description): direction is your forecast for its price move
over the horizon ('up' = positive return, 'down' = negative, 'flat' =
approximately unchanged) and prob_up is P(return > 0) in [0, 1].
expected_return is the expected fractional return over the horizon (0.02
= +2%) or null. target_price (with target_currency 'USDT' or 'USD' and
price_basis 'last' or 'mark') and confidence (0..1) are optional. Entries
are validated individually: invalid ones are rejected with the reason in
the tool response while valid ones are accepted, so fix and re-file only
the rejected horizons.

Args:
    predictions: all horizon entries in one list — one entry each for
        horizon_days 1, 5 and 21.
Returns:
    str: the accepted horizons plus any per-entry rejection reasons.
"""


def _instrument_phrase(instrument_id: str) -> str:
    if instrument_id:
        return f"the instrument {instrument_id}"
    return "the instrument this analysis is about (the run's ticker)"


def _coerce_tool_entry(
    entry: PredictionEntryInput | Mapping[str, Any],
) -> PredictionEntry:
    """Validate one tool entry into a ledger entry (raises ``ValueError``)."""
    data = entry.model_dump() if isinstance(entry, PredictionEntryInput) else dict(entry)
    currency = data.get("target_currency")
    if currency is not None and currency not in _TARGET_CURRENCIES:
        raise ValueError(
            f"target_currency {currency!r} not in {sorted(_TARGET_CURRENCIES)}"
        )
    basis = data.get("price_basis")
    if basis is not None and basis not in _PRICE_BASES:
        raise ValueError(f"price_basis {basis!r} not in {sorted(_PRICE_BASES)}")
    return coerce_prediction_entry(data)


def _record_submitted_entries(
    entries: Sequence[PredictionEntryInput | Mapping[str, Any]],
) -> str:
    """Validate every entry; append the valid ones to the active capture.

    Never raises: no active capture (recording not armed for this run) and
    invalid entries alike come back as explanatory strings — same posture
    as the data tools' sentinel returns, because a tool error must not
    crash a run.
    """
    capture = _active_capture()
    if capture is None:
        return (
            "submit_prediction ignored: no analysis-run capture is active "
            "(the blind-prediction ledger is not recording this run)."
        )
    accepted: list[int] = []
    errors: list[str] = []
    for index, entry in enumerate(entries):
        try:
            coerced = _coerce_tool_entry(entry)
        except (TypeError, ValueError) as exc:
            errors.append(f"entry {index + 1}: {exc}")
            continue
        capture.entries.append(coerced)
        accepted.append(coerced.horizon_days)
    head = (
        f"accepted horizons {sorted(accepted)} for {capture.analyst} on "
        f"{capture.instrument_id} ({capture.prediction_scope}) — committed."
        if accepted
        else "no entries accepted."
    )
    if errors:
        return f"submit_prediction: {head} Rejected entries: " + "; ".join(errors)
    return f"submit_prediction: {head}"


def make_submit_prediction_tool(instrument_id: str = "") -> BaseTool:
    """Build the ``submit_prediction`` tool with the instrument named.

    The description is the ONLY prompt surface the record stage gets, so
    the instrument id (the perp contract, e.g. MUUSDT, vs the underlying
    equity, e.g. MU) is baked in per run: the model must never guess which
    instrument its direction/prob_up refer to. The tool NAME is the
    constant :data:`TOOL_NAME` for every instance, so a graph ToolNode can
    hold one generic registration while analyst nodes bind per-run
    instances — dispatch is by name, and the buffer's instrument identity
    comes from :func:`begin_prediction_capture`, not from the description.
    """

    def submit_prediction(
        predictions: Annotated[
            list[PredictionEntryInput],
            "All forecast entries — one per horizon (1, 5 and 21 days), filed together.",
        ],
    ) -> str:
        return _record_submitted_entries(predictions)

    submit_prediction.__doc__ = _DESCRIPTION_TEMPLATE.format(
        instrument_phrase=_instrument_phrase(instrument_id)
    )
    return tool(submit_prediction)


#: Generic instance (instrument described as "the run's ticker"). Analyst
#: nodes bind per-run instances via :func:`make_submit_prediction_tool`;
#: this one serves direct tests and the graph ToolNode registration.
submit_prediction = make_submit_prediction_tool()


# --------------------------------------------------------------------------- #
# Per-context capture buffer (house ContextVar pattern)
# --------------------------------------------------------------------------- #
@dataclass
class _PredictionCapture:
    """One analyst's accumulated blind-prediction entries for one run."""

    run_id: str
    analyst: str
    instrument_id: str
    prediction_scope: str
    entries: list[PredictionEntry] = field(default_factory=list)
    submitted: bool = False

    def matches(
        self, run_id: str, analyst: str, instrument_id: str, prediction_scope: str
    ) -> bool:
        """True when this capture names the same (run, analyst, instrument, scope)."""
        return (
            self.run_id == run_id
            and self.analyst == analyst
            and self.instrument_id == instrument_id
            and self.prediction_scope == prediction_scope
        )


#: Registry key holding the analyst whose tool loop is currently active.
#: Lives INSIDE the shared registry dict (not a ContextVar.set) so the
#: ToolNode task — which inherits the registry object but not sibling-node
#: ContextVar writes — still resolves the right capture.
_ACTIVE_CAPTURE_KEY = "__active__"

_CAPTURE_SCOPE: ContextVar[dict[str, Any] | None] = ContextVar(
    "yialpha_prediction_capture_scope", default=None
)


def _capture_scope() -> dict[str, Any]:
    scope = _CAPTURE_SCOPE.get()
    if scope is None:
        scope = {}
        _CAPTURE_SCOPE.set(scope)
    return scope


def _active_capture() -> _PredictionCapture | None:
    """The capture tool calls currently append to, or ``None``."""
    scope = _capture_scope()
    analyst = scope.get(_ACTIVE_CAPTURE_KEY)
    capture = scope.get(analyst) if isinstance(analyst, str) else None
    return capture if isinstance(capture, _PredictionCapture) else None


def ensure_prediction_capture_scope() -> None:
    """Bind a fresh capture registry in the CURRENT (parent) context.

    Graph-runner call, same contract as ``quality.ensure_run_context()``:
    binding here makes the registry object shared by every node task
    (analyst node, ToolNode dispatch, re-entries). No-op when already
    bound. A fresh run for the same analyst replaces its capture via
    :func:`begin_prediction_capture` (keyed by run id), so a crashed prior
    run in a reused worker context cannot leak entries into this one.
    """
    if _CAPTURE_SCOPE.get() is None:
        _CAPTURE_SCOPE.set({})


def reset_prediction_capture_for_test() -> None:
    """Drop the bound registry (test isolation / forced cold start)."""
    _CAPTURE_SCOPE.set(None)


def begin_prediction_capture(
    analyst: str, instrument_id: str, prediction_scope: str
) -> None:
    """Arm the capture buffer for ``analyst`` before its LLM tool loop.

    No-op without a bound ledger run context. Idempotent across tool-loop
    re-entries: when a capture already names the same
    (run, analyst, instrument, scope) it is kept — the node body re-runs
    on every tool-call round and must not drop earlier entries. A
    different identity (new run, new instrument) replaces the capture.
    Marking the analyst active routes tool-call buffering to it even when
    the ToolNode task executes the generic tool instance.
    """
    context = current_ledger_run_context()
    if context is None:
        return
    scope = _capture_scope()
    existing = scope.get(analyst)
    if not (
        isinstance(existing, _PredictionCapture)
        and existing.matches(context.run_id, analyst, instrument_id, prediction_scope)
    ):
        existing = _PredictionCapture(
            run_id=context.run_id,
            analyst=analyst,
            instrument_id=instrument_id,
            prediction_scope=prediction_scope,
        )
        scope[analyst] = existing
    scope[_ACTIVE_CAPTURE_KEY] = analyst


def prediction_capture_pending() -> bool:
    """True when any capture holds unsubmitted entries."""
    return any(
        isinstance(capture, _PredictionCapture)
        and not capture.submitted
        and bool(capture.entries)
        for capture in (_CAPTURE_SCOPE.get() or {}).values()
    )


def flush_predictions() -> list[str]:
    """Submit every pending capture; ONE ``submit_predictions`` call each.

    Uses the bound run context (``analysis_as_of`` from it, evidence ids
    via :func:`yialpha.ledger.evidence.evidence_ids_for_run`); no-op
    returning ``[]`` without a context or with nothing captured. Idempotent
    by design — the ledger collapses identical resubmissions — and a
    successfully flushed capture is marked submitted so a later settlement
    cannot rewrite its rows against a grown evidence chain. Any exception
    is caught, logged as a WARNING, and ``[]`` returned: prediction
    recording must never abort the run.
    """
    context = current_ledger_run_context()
    if context is None:
        return []
    pending = [
        capture
        for capture in (_CAPTURE_SCOPE.get() or {}).values()
        if isinstance(capture, _PredictionCapture)
        and not capture.submitted
        and capture.entries
        and capture.run_id == context.run_id
    ]
    if not pending:
        return []
    try:
        evidence_ids = evidence_ids_for_run(context.run_id)
        written: list[str] = []
        for capture in sorted(pending, key=lambda item: item.analyst):
            written.extend(
                submit_predictions(
                    run_id=context.run_id,
                    analyst=capture.analyst,
                    instrument_id=capture.instrument_id,
                    prediction_scope=capture.prediction_scope,
                    entries=list(capture.entries),
                    analysis_as_of=context.analysis_as_of,
                    evidence_ids=evidence_ids,
                )
            )
            capture.submitted = True
        return written
    except Exception:  # noqa: BLE001 -- never abort the run on ledger failure
        logger.warning(
            "flush_predictions failed for run %s; blind predictions not recorded",
            context.run_id,
            exc_info=True,
        )
        return []


def settle_prediction_capture(analyst: str, instrument_id: str) -> None:
    """Settle one analyst's capture at the END of its tool loop.

    Entries captured -> :func:`flush_predictions` writes them. Buffer empty
    (the model never called the tool, or every entry was rejected) -> one
    ``optional_unavailable`` quality sentinel for ``submit_prediction``
    (blind predictions are optional enrichment; the run's data quality is
    unaffected, so ``no_data``/``core_error`` would overstate the gap).
    Call this ONLY on the final loop round (no pending tool calls) — see
    the module docstring. Fail-soft: never raises.
    """
    try:
        capture = (_CAPTURE_SCOPE.get() or {}).get(analyst)
        if not (isinstance(capture, _PredictionCapture) and capture.entries):
            quality.record_sentinel(
                "submit_prediction",
                quality.KIND_OPTIONAL_UNAVAILABLE,
                detail=(
                    f"{analyst} analyst never filed blind predictions for "
                    f"{instrument_id}; no prediction rows recorded for this run"
                ),
            )
            return
        flush_predictions()
    except Exception:  # noqa: BLE001 -- settlement must never abort the node
        logger.warning(
            "prediction capture settlement failed for %s", analyst, exc_info=True
        )


def dispatch_prediction_tool_calls(response: Any) -> int:
    """Execute ``submit_prediction`` tool calls carried by an LLM response.

    Used by the tool-less sentiment analyst: its structured-output path
    cannot carry extra tools, so the node runs ONE dedicated bound round
    and executes the returned tool calls itself (nobody dispatches them —
    the node, not the graph, drives that round). Returns the number of
    dispatched calls; per-call failures are logged and skipped.
    """
    dispatched = 0
    for call in getattr(response, "tool_calls", None) or []:
        if not isinstance(call, dict) or str(call.get("name", "")) != TOOL_NAME:
            continue
        try:
            args = dict(call.get("args") or {})
            outcome = submit_prediction.invoke(
                {"predictions": args.get("predictions") or []}
            )
            logger.info("submit_prediction dispatch: %s", outcome)
            dispatched += 1
        except Exception:  # noqa: BLE001 -- one bad call must not drop the rest
            logger.warning("submit_prediction dispatch failed", exc_info=True)
    return dispatched
