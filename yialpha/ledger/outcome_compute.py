"""Forward outcome computation: due blind predictions -> realized net returns.

:func:`compute_outcomes` walks the
:func:`yialpha.ledger.outcomes.pending_predictions` worklist and appends one
outcome row per due ``(prediction, horizon)`` carrying the full attribution
decomposition the calibration scoreboard later joins on. Every vendor /
mirror read goes through three module-level seams — imported HERE (not at
their home modules) so tests monkeypatch this module's bindings:

* ``binance_klines_frame`` — perp daily last-price klines (venue
  ``binance_perp``, ``price_type="last"``), the same frame the analysts read;
* ``get_binance_funding_rate`` — the perp funding-rate history CSV (the
  funding seam the perp bundle maps to);
* ``get_YFin_history_cached`` — the equity history seam used for stock-perp
  underlyings (the same one ``yialpha.accuracy`` prices equities on).

Conventions pinned here (each is a measurable contract, not an implementation
detail):

* **Price window.** Daily bars are labeled by their session date and a bar
  labeled ``D`` counts as available from ``D 00:00 UTC`` (the same
  baseline/endpoint semantics as ``yialpha.accuracy``: entry = last bar at or
  before ``analysis_as_of``, exit = last bar at or before
  ``min(analysis_as_of + horizon_days, now_as_of)``; intraday
  ``analysis_as_of`` therefore resolves at daily granularity — disclosed
  approximation). The fetch window reaches ``_LOOKBACK_BUFFER_DAYS`` before
  the analysis date so a weekend/holiday analysis date still finds an entry
  bar.
* **Skip-vs-incomplete.** When the exit bar is NOT strictly after the entry
  bar (horizon end not yet bar-visible, e.g. a weekend-bound equity horizon)
  nothing is written: the prediction stays ``pending`` on the worklist and is
  retried next batch. When the vendor frame cannot supply an ENTRY bar at
  all, the row is written ``incomplete`` with leg ``"contract_price"`` — a
  visible, fail-closed record that the analysis date itself is unpriceable.
* **Funding sign.** ``funding_pnl = -sign(direction) * sum(funding_rate)``
  over settlements in ``(analysis_as_of, horizon_end]`` with sign +1 for
  ``up`` / -1 for ``down`` / 0 for ``flat``: a long in positive funding PAYS
  (negative pnl), a short RECEIVES (positive pnl). The leg is signed, never
  an absolute value. A gap in the funding grid inside the window (mode
  settlement cadence inferred from the fetched series, then every expected
  slot verified present) fails the leg closed: ``funding_pnl=None`` +
  ``legs_missing += "funding"``.
* **legs_missing lists only legs that SHOULD exist.** Funding exists only
  for CONTRACT-scope predictions on perp runs; the underlying leg exists
  only for ``stock_perp`` runs. A leg that is structurally not applicable
  (e.g. funding on an UNDERLYING-scope row) is stored ``None`` and is NOT a
  missing leg.
* **Net composition.** ``net_return = contract_price_return + funding_pnl -
  fees - slippage`` where ``funding_pnl`` contributes only when it should
  exist (UNDERLYING-scope rows omit it — the call was about the equity, not
  the perp's carry). Status is ``complete`` exactly when all four
  contributing legs are present; ``basis_return`` / ``underlying_return``
  are diagnostic legs and their absence never blocks ``net_return``.
* **Ticket costs.** Fees and slippage come from the run's ticket mirror
  (latest ``tickets`` row for the run). When the payload carries the cost
  model's decomposition (``entry_fee_bps`` / ``exit_fee_bps`` /
  ``entry_slippage_bps`` / ``exit_slippage_bps``, flat or nested under
  ``cost_detail`` / ``cost_estimate`` / ``cost``) the carry legs are excluded
  — funding is measured from realized history, never double charged. When
  the payload carries only the scalar ``estimated_cost`` total, the whole
  total is booked under ``fees`` with ``slippage=0.0`` (``net_return`` is
  invariant to the label split; the scalar may embed expected carry —
  disclosed approximation). No usable ticket row → both ``None`` +
  ``"ticket_cost"``.
* **basis_return** is the same-window approximation
  ``contract_price_return - underlying_return`` (a true basis decomposition
  would need matched contract/underlying marks at both endpoints).
* **Failure isolation.** Every per-prediction vendor/ledger error is caught:
  the row counts ``failed`` in the report and the batch continues. Writes go
  through :func:`yialpha.ledger.outcomes.write_outcome` (idempotent on
  identical content; a conflicting rewrite of an existing outcome row raises
  and surfaces as ``failed`` — outcomes are append-only facts).
* **outcome_available_at** is the exit bar's date (ISO) — the bar whose
  close the exit leg used; when the horizon end is not yet bar-visible this
  is earlier than the calendar horizon end and discloses it.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from yialpha.dataflows.binance import binance_klines_frame, get_binance_funding_rate
from yialpha.dataflows.y_finance import get_YFin_history_cached
from yialpha.ledger.models import (
    SCOPE_CONTRACT,
    SCOPE_MACRO,
    SCOPE_UNDERLYING,
    timestamp_as_utc,
)
from yialpha.ledger.outcomes import pending_predictions, write_outcome
from yialpha.ledger.sqlite import get_connection

logger = logging.getLogger(__name__)

#: Days fetched before the analysis date so a weekend/holiday analysis date
#: still finds an entry bar ("last close at/before analysis_as_of").
_LOOKBACK_BUFFER_DAYS = 10

#: Registry instrument classes that denote a perpetual contract (anything
#: else — plain equities — prices on the equity seam).
_PERP_INSTRUMENT_CLASSES = frozenset({"stock_perp", "pure_crypto_perp", "unknown_perp"})

#: Payload keys under which a ticket may nest the cost-model decomposition.
_TICKET_COST_KEYS = ("cost_detail", "cost_estimate", "cost")

#: The cost-model decomposition fields (bps) a ticket payload may carry.
_FEE_BPS_KEYS = ("entry_fee_bps", "exit_fee_bps")
_SLIPPAGE_BPS_KEYS = ("entry_slippage_bps", "exit_slippage_bps")

#: Legs that can appear in ``legs_missing`` (legs that SHOULD exist but
#: could not be sourced): the price leg, the funding leg, the ticket cost
#: legs (booked as one), and the stock-perp underlying diagnostic leg.
LEG_CONTRACT_PRICE = "contract_price"
LEG_FUNDING = "funding"
LEG_TICKET_COST = "ticket_cost"
LEG_UNDERLYING = "underlying"


# --------------------------------------------------------------------------- #
# Report shape
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OutcomeComputeDetail:
    """Per-prediction result of one :func:`compute_outcomes` pass.

    ``status`` is the outcome row's status for written rows; ``"pending"``
    marks a skipped row (horizon end not yet bar-visible — nothing written,
    the prediction stays on the worklist); ``"skipped"`` marks a row this
    writer deliberately does not score (MACRO scope); ``"failed"`` marks an
    isolated per-row error (``error`` carries the message).
    """

    prediction_id: str
    run_id: str
    horizon_days: int
    status: str
    legs_missing: tuple[str, ...] = ()
    net_return: float | None = None
    outcome_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class OutcomeComputeReport:
    """Batch totals over one :func:`compute_outcomes` pass.

    ``considered`` counts every worklist row (including pending skips,
    MACRO skips and failures); ``completed`` / ``incomplete`` count rows
    actually written with that status; ``failed`` counts isolated errors.
    """

    considered: int
    completed: int
    incomplete: int
    failed: int
    now_as_of: str = ""
    details: list[OutcomeComputeDetail] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Vendor seam adapters (tests monkeypatch the module-level bindings above)
# --------------------------------------------------------------------------- #
def _dated_close_series(frame: pd.DataFrame) -> pd.Series:
    """Normalize a vendor OHLCV frame to a date-indexed close series."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    series = frame["Close"].astype(float)
    series.index = pd.to_datetime(series.index).date
    return series[~series.index.duplicated(keep="last")].sort_index()


def _perp_close_series(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """Perp daily last-price closes (the contract series)."""
    frame = binance_klines_frame(
        symbol, start_date, end_date, interval="1d", venue="binance_perp",
        price_type="last",
    )
    return _dated_close_series(frame)


def _equity_close_series(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """Equity daily closes; +1 day on the end because the seam's end is EXCLUSIVE."""
    end_exclusive = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    frame = get_YFin_history_cached(symbol, start_date, end_exclusive)
    return _dated_close_series(frame)


def _underlying_equity_symbol(instrument_id: str) -> str:
    """Equity ticker under a stock perp (strip the quote/contract suffix).

    Mirrors the instrument registry's convention: the base asset of e.g.
    ``TSLAUSDT`` is ``TSLA``. A symbol carrying no known suffix is returned
    as-is (the run's instrument_id is already the equity ticker then).
    """
    symbol = instrument_id.strip().upper()
    for suffix in ("USDT", "USDC", "PERP", "USD"):
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            return symbol[: -len(suffix)]
    return symbol


@dataclass(frozen=True)
class _Bar:
    """One dated close used as an entry/exit reference."""

    bar_date: str
    close: float


def _last_close_at_or_before(series: pd.Series, on_date: str) -> _Bar | None:
    """Last bar whose session date is at/before ``on_date`` (ISO date)."""
    cutoff = datetime.strptime(on_date, "%Y-%m-%d").date()
    eligible = series[[d <= cutoff for d in series.index]]
    if eligible.empty:
        return None
    return _Bar(bar_date=str(eligible.index[-1]), close=float(eligible.iloc[-1]))


# --------------------------------------------------------------------------- #
# Funding leg
# --------------------------------------------------------------------------- #
def _parse_funding_csv(text: str) -> list[tuple[datetime, float]]:
    """Parse the funding seam's CSV (``fundingTime, fundingRate``; ``#`` header).

    ``fundingTime`` values are UTC; malformed rows are skipped (the gap check
    below fails closed on anything missing).
    """
    lines = [
        line for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    settlements: list[tuple[datetime, float]] = []
    for row in csv.DictReader(lines):
        time_raw, rate_raw = row.get("fundingTime"), row.get("fundingRate")
        if time_raw is None or rate_raw is None:
            continue
        try:
            moment = datetime.strptime(str(time_raw).strip(), "%Y-%m-%d %H:%M:%S")
            rate = float(rate_raw)
        except ValueError:
            continue
        settlements.append((moment.replace(tzinfo=UTC), rate))
    return sorted(settlements, key=lambda item: item[0])


def _modal_cadence_hours(times: list[datetime]) -> int | None:
    """Modal whole-hour spacing between adjacent settlements (None if unknown).

    The mode (not min/median) is robust to one missing settlement — the same
    inference the vendor seam itself reports in its header.
    """
    if len(times) < 2:
        return None
    diffs = [
        round((b - a).total_seconds() / 3600.0)
        for a, b in zip(times, times[1:], strict=False)
        if b > a
    ]
    diffs = [hours for hours in diffs if hours > 0]
    if not diffs:
        return None
    mode = max(set(diffs), key=diffs.count)
    return mode if mode > 0 else None


def _funding_pnl_leg(
    symbol: str,
    analysis_dt: datetime,
    horizon_end_dt: datetime,
    direction: str,
) -> tuple[float | None, bool]:
    """Signed funding pnl over ``(analysis_dt, horizon_end_dt]`` or a miss.

    Returns ``(pnl, missing)``: ``missing`` is True when the settlement grid
    inside the window cannot be proven complete (no settlements, cadence
    uninferrable, or an expected slot absent) — fail closed per leg. The pnl
    itself is ``-sign(direction) * sum(rate)``: a long pays positive funding.
    """
    csv_text = get_binance_funding_rate(
        symbol,
        analysis_dt.date().isoformat(),
        horizon_end_dt.date().isoformat(),
    )
    settlements = _parse_funding_csv(csv_text)
    if not settlements:
        return None, True
    in_window = [
        (moment, rate) for moment, rate in settlements if analysis_dt < moment <= horizon_end_dt
    ]
    if not in_window:
        return None, True
    cadence = _modal_cadence_hours([moment for moment, _ in settlements])
    if cadence is None:
        return None, True
    # Verify every expected settlement slot from the first fetched one up to
    # the horizon end is present; an absent slot is a history gap.
    present = {moment for moment, _ in settlements}
    step = timedelta(hours=cadence)
    expected = settlements[0][0]
    while expected <= horizon_end_dt:
        if expected > analysis_dt and expected not in present:
            return None, True
        expected += step
    sign = {"up": 1.0, "down": -1.0}.get(direction, 0.0)
    total = sum(rate for _, rate in in_window)
    return -sign * total, False


# --------------------------------------------------------------------------- #
# Ticket cost legs
# --------------------------------------------------------------------------- #
def _cost_split_from_mapping(source: dict[str, Any]) -> tuple[float, float] | None:
    """(fees, slippage) from a mapping carrying the cost-model bps fields."""
    if not all(key in source for key in _FEE_BPS_KEYS + _SLIPPAGE_BPS_KEYS):
        return None
    try:
        fees = sum(float(source[key]) for key in _FEE_BPS_KEYS) / 1e4
        slippage = sum(float(source[key]) for key in _SLIPPAGE_BPS_KEYS) / 1e4
    except (TypeError, ValueError):
        return None
    return fees, slippage


def _ticket_cost_legs(run_id: str) -> tuple[str | None, float | None, float | None]:
    """Latest ticket mirror for the run -> (ticket_id, fees, slippage).

    Carries legs are never booked here (funding is measured from realized
    history). A payload with only the scalar ``estimated_cost`` total books
    the whole total under fees with slippage 0.0 — see module docstring. No
    usable row -> (None, None, None) and the caller marks ``ticket_cost``.
    """
    row = (
        get_connection(readonly=True)
        .execute(
            "SELECT ticket_id, payload FROM tickets WHERE run_id = ? "
            "ORDER BY written_at DESC, ticket_id DESC LIMIT 1",
            (run_id,),
        )
        .fetchone()
    )
    if row is None:
        return None, None, None
    ticket_id = str(row["ticket_id"])
    try:
        payload = json.loads(str(row["payload"]))
    except ValueError:
        return ticket_id, None, None
    if not isinstance(payload, dict):
        return ticket_id, None, None
    sources: list[dict[str, Any]] = [payload]
    sources += [
        payload[key]
        for key in _TICKET_COST_KEYS
        if isinstance(payload.get(key), dict)
    ]
    for source in sources:
        split = _cost_split_from_mapping(source)
        if split is not None:
            return ticket_id, split[0], split[1]
    total = payload.get("estimated_cost")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return ticket_id, float(total), 0.0
    return ticket_id, None, None


# --------------------------------------------------------------------------- #
# One prediction
# --------------------------------------------------------------------------- #
def _compute_one(item: dict[str, Any], now_as_of: str) -> OutcomeComputeDetail:
    """Resolve one pending prediction into a written (or skipped) outcome."""
    prediction_id = str(item["prediction_id"])
    run_id = str(item["run_id"])
    horizon = int(item["horizon_days"])
    scope = str(item["prediction_scope"])
    direction = str(item["direction"])
    analysis_as_of = str(item["analysis_as_of"])
    instrument_id = str(item["instrument_id"])
    instrument_class = item["instrument_class"]
    asset_type = str(item["asset_type"] or "")

    def detail(
        status: str,
        *,
        legs_missing: tuple[str, ...] = (),
        net_return: float | None = None,
        outcome_id: str | None = None,
    ) -> OutcomeComputeDetail:
        return OutcomeComputeDetail(
            prediction_id=prediction_id,
            run_id=run_id,
            horizon_days=horizon,
            status=status,
            legs_missing=legs_missing,
            net_return=net_return,
            outcome_id=outcome_id,
        )

    if scope == SCOPE_MACRO:
        # A macro call is not a claim about this instrument's return; scoring
        # it as one would fabricate precision. No row is ever written.
        return detail("skipped")

    is_perp = (
        instrument_class in _PERP_INSTRUMENT_CLASSES or asset_type == "crypto_perp"
    )
    is_stock_perp = instrument_class == "stock_perp"
    underlying_symbol = _underlying_equity_symbol(instrument_id) if is_stock_perp else None

    analysis_dt = timestamp_as_utc(analysis_as_of)
    horizon_end_dt = min(
        analysis_dt + timedelta(days=horizon), timestamp_as_utc(now_as_of)
    )
    start_date = (analysis_dt.date() - timedelta(days=_LOOKBACK_BUFFER_DAYS)).isoformat()
    end_date = horizon_end_dt.date().isoformat()

    # --- predicted-series price leg (contract_price_return) -----------------
    # CONTRACT scope prices the run's own instrument (perp klines for perp
    # runs, the equity seam otherwise); UNDERLYING scope prices the equity
    # seam when a stock-perp underlying exists, else the run's primary series
    # (disclosed approximation — a pure-crypto underlying has no equity seam).
    if scope == SCOPE_UNDERLYING and is_stock_perp:
        series = _equity_close_series(
            underlying_symbol or instrument_id, start_date, end_date
        )
    elif is_perp:
        series = _perp_close_series(instrument_id, start_date, end_date)
    else:
        series = _equity_close_series(instrument_id, start_date, end_date)

    entry = _last_close_at_or_before(series, analysis_dt.date().isoformat())
    if entry is None:
        # The analysis date itself is unpriceable (vendor window does not
        # reach back): visible fail-closed record, prediction stays due.
        outcome_id = write_outcome(
            prediction_id, run_id, horizon, status="incomplete",
            legs_missing=[LEG_CONTRACT_PRICE], regime_id=item.get("regime_id"),
        )
        return detail("incomplete", legs_missing=(LEG_CONTRACT_PRICE,), outcome_id=outcome_id)
    exit_bar = _last_close_at_or_before(series, end_date)
    if exit_bar is None or exit_bar.bar_date <= entry.bar_date:
        # Horizon end not yet bar-visible: nothing written, stays pending.
        return detail("pending", legs_missing=(LEG_CONTRACT_PRICE,))
    contract_price_return = exit_bar.close / entry.close - 1.0

    # --- underlying diagnostic leg (stock perps only) ------------------------
    underlying_return: float | None = None
    missing: set[str] = set()
    if is_stock_perp:
        if scope == SCOPE_UNDERLYING:
            underlying_series = series  # same seam call, same window
        else:
            underlying_series = _equity_close_series(
                underlying_symbol or instrument_id, start_date, end_date
            )
        u_entry = _last_close_at_or_before(
            underlying_series, analysis_dt.date().isoformat()
        )
        u_exit = _last_close_at_or_before(underlying_series, end_date)
        if (
            u_entry is not None
            and u_exit is not None
            and u_exit.bar_date > u_entry.bar_date
        ):
            underlying_return = u_exit.close / u_entry.close - 1.0
        else:
            missing.add(LEG_UNDERLYING)

    basis_return = (
        contract_price_return - underlying_return
        if underlying_return is not None
        else None
    )

    # --- funding leg (CONTRACT scope on a perp: it should exist) -------------
    funding_should_exist = scope == SCOPE_CONTRACT and is_perp
    funding_pnl: float | None = None
    if funding_should_exist:
        funding_pnl, funding_missing = _funding_pnl_leg(
            instrument_id, analysis_dt, horizon_end_dt, direction
        )
        if funding_missing:
            missing.add(LEG_FUNDING)
            funding_pnl = None

    # --- ticket cost legs -----------------------------------------------------
    ticket_id, fees, slippage = _ticket_cost_legs(run_id)
    if fees is None or slippage is None:
        missing.add(LEG_TICKET_COST)
        fees = slippage = None

    # --- composition -----------------------------------------------------------
    net_return: float | None = None
    status: str = "incomplete"
    if fees is not None and slippage is not None and (
        not funding_should_exist or funding_pnl is not None
    ):
        funding_addend = (
            funding_pnl if funding_should_exist and funding_pnl is not None else 0.0
        )
        net_return = contract_price_return + funding_addend - fees - slippage
        status = "complete"

    outcome_id = write_outcome(
        prediction_id,
        run_id,
        horizon,
        status=status,  # type: ignore[arg-type]
        contract_price_return=contract_price_return,
        underlying_return=underlying_return,
        basis_return=basis_return,
        funding_pnl=funding_pnl,
        fees=fees,
        slippage=slippage,
        net_return=net_return,
        legs_missing=sorted(missing) or None,
        outcome_available_at=exit_bar.bar_date,
        ticket_id=ticket_id,
        regime_id=item.get("regime_id"),
    )
    return detail(
        status,
        legs_missing=tuple(sorted(missing)),
        net_return=net_return,
        outcome_id=outcome_id,
    )


# --------------------------------------------------------------------------- #
# Batch entry point
# --------------------------------------------------------------------------- #
def compute_outcomes(now_as_of: str, *, limit: int = 200) -> OutcomeComputeReport:
    """Score every due-and-unscored prediction; returns the batch report.

    Walks :func:`yialpha.ledger.outcomes.pending_predictions` (first
    ``limit`` rows), resolves each one through the module seams, and appends
    the outcome row via the idempotent :func:`write_outcome`. Any per-row
    error — vendor failure, immutable-rewrite conflict, bad payload — is
    isolated: the row counts ``failed`` (message in its detail) and the
    batch continues. Never raises for per-row causes.
    """
    worklist = pending_predictions(now_as_of)[:limit]
    details: list[OutcomeComputeDetail] = []
    completed = incomplete = failed = 0
    for item in worklist:
        try:
            result = _compute_one(item, now_as_of)
        except Exception as exc:  # one row must never sink the batch
            failed += 1
            logger.warning(
                "outcome computation failed for %s: %s",
                item.get("prediction_id"),
                exc,
            )
            details.append(
                OutcomeComputeDetail(
                    prediction_id=str(item.get("prediction_id", "")),
                    run_id=str(item.get("run_id", "")),
                    horizon_days=int(item.get("horizon_days", 0)),
                    status="failed",
                    error=str(exc),
                )
            )
            continue
        if result.status == "complete":
            completed += 1
        elif result.status == "incomplete":
            incomplete += 1
        details.append(result)
    return OutcomeComputeReport(
        considered=len(worklist),
        completed=completed,
        incomplete=incomplete,
        failed=failed,
        now_as_of=now_as_of,
        details=details,
    )
