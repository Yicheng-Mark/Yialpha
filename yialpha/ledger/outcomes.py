"""Forward outcomes: realized results + the net-return attribution legs.

After a horizon elapses, the outcome writer resolves what actually happened
and appends ONE row per ``(prediction_id, horizon_days)``: the realized
return legs (contract price / underlying / basis / funding / fees /
slippage / liquidation) and their composition into ``net_return``. All legs
are signed fractions of notional (FEATURE_VERSION v2).

Immutability follows the ledger-wide contract — the table's
``UNIQUE (prediction_id, horizon_days)`` is never hit by this code path
(check-then-insert inside ONE ``BEGIN IMMEDIATE`` transaction):

* rewriting an existing (prediction, horizon) with IDENTICAL content (all
  stored columns except ``computed_at``) is an idempotent no-op returning
  the same outcome id;
* rewriting with different content raises ``ValueError`` — outcomes are
  append-only facts, not fields to be tuned.

``legs_missing`` is stored as a sorted JSON array (the same canonical form
as prediction ``evidence_ids``); ``None``/empty is stored as NULL. A row
with ``status='pending'`` (horizon not fully resolvable yet) or
``'incomplete'`` (data legs missing) deliberately stays on the
:func:`pending_predictions` worklist — only ``'complete'`` retires a
prediction from it.

Date-vs-datetime normalization: see
:func:`yialpha.ledger.models.timestamp_as_utc` — a pure date counts as
midnight UTC.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Any, Literal

from yialpha.ledger.models import (
    OutcomeRecord,
    encode_json_list,
    new_outcome_id,
    row_to_outcome,
    timestamp_as_utc,
    validate_horizon_days,
)
from yialpha.ledger.sqlite import (
    get_connection,
    ledger_exists,
    ledger_transaction,
    utc_now_iso,
)

#: Every admissible outcome ``status`` (checked at runtime despite the
#: Literal signature — the ledger must never carry a typo'd status).
OUTCOME_STATUSES = frozenset({"complete", "pending", "incomplete"})

#: Every ``outcomes`` column, in INSERT/SELECT order (v1 DDL order + the
#: migration-v2 ``regime_id`` tail column).
_OUTCOME_COLUMNS = (
    "outcome_id",
    "prediction_id",
    "run_id",
    "ticket_id",
    "horizon_days",
    "status",
    "contract_price_return",
    "underlying_return",
    "basis_return",
    "funding_pnl",
    "fees",
    "slippage",
    "liquidation_loss",
    "net_return",
    "legs_missing",
    "outcome_available_at",
    "computed_at",
    "regime_id",
)


def write_outcome(
    prediction_id: str,
    run_id: str,
    horizon_days: int,
    *,
    status: Literal["complete", "pending", "incomplete"],
    contract_price_return: float | None = None,
    underlying_return: float | None = None,
    basis_return: float | None = None,
    funding_pnl: float | None = None,
    fees: float | None = None,
    slippage: float | None = None,
    liquidation_loss: float | None = None,
    net_return: float | None = None,
    legs_missing: Sequence[str] | None = None,
    outcome_available_at: str | None = None,
    ticket_id: str | None = None,
    regime_id: str | None = None,
) -> str:
    """Append one outcome row; returns the deterministic ``outcome_id``.

    ``outcome_id`` derives from ``(prediction_id, horizon_days)``
    (:func:`yialpha.ledger.models.new_outcome_id`); ``horizon_days`` must sit
    on the frozen ladder and ``status`` in ``{complete, pending,
    incomplete}``. ``regime_id`` (V2.2) is the prediction's regime carried
    through — part of content identity like every stored column. Unknown
    predictions raise ``sqlite3.IntegrityError`` (foreign key). Rewrites
    follow the immutability contract described in the module docstring.
    """
    if status not in OUTCOME_STATUSES:
        raise ValueError(f"status {status!r} not in {sorted(OUTCOME_STATUSES)}")
    horizon = validate_horizon_days(horizon_days)
    outcome_id = new_outcome_id(prediction_id, horizon)
    values: dict[str, Any] = {
        "outcome_id": outcome_id,
        "prediction_id": prediction_id,
        "run_id": run_id,
        "ticket_id": ticket_id,
        "horizon_days": horizon,
        "status": status,
        "contract_price_return": contract_price_return,
        "underlying_return": underlying_return,
        "basis_return": basis_return,
        "funding_pnl": funding_pnl,
        "fees": fees,
        "slippage": slippage,
        "liquidation_loss": liquidation_loss,
        "net_return": net_return,
        "legs_missing": encode_json_list(legs_missing) if legs_missing else None,
        "outcome_available_at": outcome_available_at,
        "computed_at": utc_now_iso(),
        "regime_id": regime_id,
    }
    with ledger_transaction() as cur:
        existing = cur.execute(
            "SELECT outcome_id, prediction_id, run_id, ticket_id, horizon_days, "
            "status, contract_price_return, underlying_return, basis_return, "
            "funding_pnl, fees, slippage, liquidation_loss, net_return, "
            "legs_missing, outcome_available_at, computed_at, regime_id "
            "FROM outcomes WHERE outcome_id = ?",
            (outcome_id,),
        ).fetchone()
        if existing is not None:
            if all(
                existing[column] == values[column]
                for column in _OUTCOME_COLUMNS
                if column != "computed_at"
            ):
                return outcome_id  # byte-identical replay
            raise ValueError(
                f"outcome {outcome_id} for prediction {prediction_id} "
                f"(horizon {horizon}d) is immutable; outcomes are append-only"
            )
        cur.execute(
            "INSERT INTO outcomes "
            "(outcome_id, prediction_id, run_id, ticket_id, horizon_days, status, "
            "contract_price_return, underlying_return, basis_return, funding_pnl, "
            "fees, slippage, liquidation_loss, net_return, legs_missing, "
            "outcome_available_at, computed_at, regime_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(values[column] for column in _OUTCOME_COLUMNS),
        )
    return outcome_id


def pending_predictions(now_as_of: str) -> list[dict[str, Any]]:
    """Due-and-unscored predictions: the outcome writer's worklist.

    A prediction is pending when its horizon has elapsed —
    ``timestamp_as_utc(analysis_as_of) + horizon_days`` calendar days
    ``<= timestamp_as_utc(now_as_of)`` (date-only strings are midnight UTC) —
    AND no outcome row with ``status='complete'`` exists for that
    ``(prediction_id, horizon_days)``; a ``pending``/``incomplete`` outcome
    keeps the prediction on the list for retry.

    Date-only note: a date-only ``analysis_as_of`` therefore falls due at the
    START of the day ``horizon_days`` after it, which can nominate a
    candidate up to one session before the horizon close provably exists.
    That is safe because this list only nominates candidates — the outcome
    writer verifies the endpoint data and records ``'pending'`` /
    ``'incomplete'`` when it is not there yet. Calendar-day arithmetic on
    midnight-normalized timestamps is the documented approximation of the
    ladder's session-calendar days.

    Revision rows appear as their own entries (carrying
    ``original_prediction_id`` / ``debate_revision``) so the scorer can
    prefer the latest link of a chain. Returns an empty list when no ledger
    DB exists yet (optional read).
    """
    if not ledger_exists():
        return []
    now_dt = timestamp_as_utc(now_as_of)
    rows = (
        get_connection(readonly=True)
        .execute(
            "SELECT p.prediction_id, p.run_id, p.analyst, p.instrument_id, "
            "p.prediction_scope, p.horizon_days, p.direction, p.prob_up, "
            "p.expected_return, p.target_price, p.target_currency, "
            "p.analysis_as_of, p.original_prediction_id, p.debate_revision, "
            "p.regime_id, "
            "r.ticker, r.asset_type, r.instrument_class "
            "FROM predictions p JOIN runs r ON r.run_id = p.run_id "
            "WHERE NOT EXISTS (SELECT 1 FROM outcomes o "
            "WHERE o.prediction_id = p.prediction_id "
            "AND o.horizon_days = p.horizon_days AND o.status = 'complete') "
            "ORDER BY p.analysis_as_of, p.prediction_id"
        )
        .fetchall()
    )
    pending: list[dict[str, Any]] = []
    for row in rows:
        due_at = timestamp_as_utc(str(row["analysis_as_of"])) + timedelta(
            days=int(row["horizon_days"])
        )
        if due_at > now_dt:
            continue
        pending.append(
            {
                "prediction_id": row["prediction_id"],
                "run_id": row["run_id"],
                "analyst": row["analyst"],
                "instrument_id": row["instrument_id"],
                "prediction_scope": row["prediction_scope"],
                "horizon_days": int(row["horizon_days"]),
                "direction": row["direction"],
                "prob_up": row["prob_up"],
                "expected_return": row["expected_return"],
                "target_price": row["target_price"],
                "target_currency": row["target_currency"],
                "analysis_as_of": row["analysis_as_of"],
                "original_prediction_id": row["original_prediction_id"],
                "debate_revision": row["debate_revision"],
                "regime_id": row["regime_id"],
                "ticker": row["ticker"],
                "asset_type": row["asset_type"],
                "instrument_class": row["instrument_class"],
            }
        )
    return pending


def outcomes_for_prediction(prediction_id: str) -> list[OutcomeRecord]:
    """All outcome rows linked to one prediction (``horizon_days`` order)."""
    if not ledger_exists():
        return []
    rows = (
        get_connection(readonly=True)
        .execute(
            "SELECT outcome_id, prediction_id, run_id, ticket_id, horizon_days, "
            "status, contract_price_return, underlying_return, basis_return, "
            "funding_pnl, fees, slippage, liquidation_loss, net_return, "
            "legs_missing, outcome_available_at, computed_at, regime_id "
            "FROM outcomes WHERE prediction_id = ? ORDER BY horizon_days",
            (prediction_id,),
        )
        .fetchall()
    )
    return [row_to_outcome(row) for row in rows]


def all_outcomes(limit: int = 500) -> list[OutcomeRecord]:
    """Most recent outcome rows across all predictions (``computed_at`` order)."""
    if not ledger_exists():
        return []
    rows = (
        get_connection(readonly=True)
        .execute(
            "SELECT outcome_id, prediction_id, run_id, ticket_id, horizon_days, "
            "status, contract_price_return, underlying_return, basis_return, "
            "funding_pnl, fees, slippage, liquidation_loss, net_return, "
            "legs_missing, outcome_available_at, computed_at, regime_id "
            "FROM outcomes ORDER BY computed_at DESC, outcome_id DESC LIMIT ?",
            (limit,),
        )
        .fetchall()
    )
    return [row_to_outcome(row) for row in rows]
