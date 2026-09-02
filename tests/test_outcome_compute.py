"""Outcome computation tests (yialpha/ledger/outcome_compute).

All vendor seams are monkeypatched at THIS module's bindings (the module
imports ``binance_klines_frame`` / ``get_binance_funding_rate`` /
``get_YFin_history_cached`` at its own top level precisely for this). The
pinned base scenario:

* perp closes: entry = 2026-08-25 close 110, exit = 2026-08-30 close 121
  -> ``contract_price_return = 121/110 - 1 = 0.10`` (bars through 09-02
  exist and must NOT move the exit — the horizon end bounds it);
* funding: +0.0001 per 8h settlement, 08-25T00:00..08-30T00:00 fetched
  (16), settlements in the window (08-25T00:00, 08-30T00:00] = 15
  -> sum = 0.0015;
* ticket ``cost_detail``: fees 10bps = 0.0010, slippage 4bps = 0.0004.

Sign convention pinned: ``funding_pnl = -sign(direction) * sum`` — an
``up`` prediction pays positive funding (negative pnl), a ``down``
prediction receives it (positive pnl); never an absolute value.
``net = 0.10 - 0.0015 - 0.0010 - 0.0004 = 0.0971`` for the ``up`` row.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta

import pandas as pd
import pytest

from yialpha.ledger import outcome_compute
from yialpha.ledger.evidence import register_run
from yialpha.ledger.models import SCOPE_CONTRACT, SCOPE_MACRO, SCOPE_UNDERLYING
from yialpha.ledger.outcome_compute import compute_outcomes
from yialpha.ledger.outcomes import (
    all_outcomes,
    outcomes_for_prediction,
    pending_predictions,
)
from yialpha.ledger.predictions import predictions_for_run, submit_predictions
from yialpha.ledger.sqlite import ledger_transaction, utc_now_iso

_NOW = "2026-09-03"
_RUN = "run-btc-2026-08-25"

_PERP_CLOSES = {
    "2026-08-23": 99.0,
    "2026-08-24": 100.0,
    "2026-08-25": 110.0,  # entry reference
    "2026-08-26": 112.0,
    "2026-08-27": 115.0,
    "2026-08-28": 118.0,
    "2026-08-29": 119.0,
    "2026-08-30": 121.0,  # exit reference (horizon end)
    "2026-09-01": 130.0,  # beyond the horizon — must not move the exit
    "2026-09-02": 131.0,
}

_COST_DETAIL = {
    "entry_fee_bps": 5.0,
    "exit_fee_bps": 5.0,
    "entry_slippage_bps": 2.0,
    "exit_slippage_bps": 2.0,
}


def _frame(closes: dict[str, float]) -> pd.DataFrame:
    """Vendor-shaped daily close frame (DatetimeIndex + ``Close`` column)."""
    dates = sorted(closes)
    return pd.DataFrame({"Close": [closes[d] for d in dates]}, index=pd.to_datetime(dates))


def _funding_csv(
    rate: float,
    start: str = "2026-08-25",
    end: str = "2026-08-30",
    drop: frozenset[str] = frozenset(),
) -> str:
    """Funding seam CSV: 8h settlements from start 00:00 through end 00:00."""
    moment = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end, "%Y-%m-%d")
    lines = [
        "# Perp USDT-M funding rate for BTCUSDT (fixture)",
        "fundingTime,fundingRate,symbol",
    ]
    while moment <= stop:
        stamp = moment.strftime("%Y-%m-%d %H:%M:%S")
        if stamp not in drop:
            lines.append(f"{stamp},{rate},BTCUSDT")
        moment += timedelta(hours=8)
    return "\n".join(lines) + "\n"


class _Seams:
    """Synthetic vendor responses installed onto outcome_compute's bindings."""

    def __init__(
        self,
        perp_closes: dict[str, float] | None = None,
        funding_text: str | None = None,
        equity_closes: dict[str, float] | None = None,
        raise_symbols: frozenset[str] = frozenset(),
    ) -> None:
        self.perp_closes = dict(perp_closes or _PERP_CLOSES)
        self.funding_text = funding_text if funding_text is not None else _funding_csv(0.0001)
        self.equity_closes = dict(equity_closes or {})
        self.raise_symbols = raise_symbols
        self.perp_calls: list[tuple[str, str, str]] = []
        self.equity_calls: list[tuple[str, str, str]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(outcome_compute, "binance_klines_frame", self._klines)
        monkeypatch.setattr(outcome_compute, "get_binance_funding_rate", self._funding)
        monkeypatch.setattr(outcome_compute, "get_YFin_history_cached", self._equity)

    def _klines(self, symbol: str, start_date: str, end_date: str, **_kwargs: object):
        if symbol in self.raise_symbols:
            raise RuntimeError(f"vendor down for {symbol}")
        self.perp_calls.append((symbol, start_date, end_date))
        return _frame(self.perp_closes)

    def _funding(self, symbol: str, start_date: str, end_date: str) -> str:
        return self.funding_text

    def _equity(self, symbol: str, start_date: str, end_date: str):
        self.equity_calls.append((symbol, start_date, end_date))
        return _frame(self.equity_closes)


def _seed(
    run_id: str = _RUN,
    analyst: str = "fundamentals",
    direction: str = "up",
    scope: str = SCOPE_CONTRACT,
    ticker: str = "BTCUSDT",
    asset_type: str = "crypto_perp",
    instrument_class: str = "pure_crypto_perp",
    analysis_as_of: str = "2026-08-25",
    horizon: int = 5,
    prob_up: float = 0.6,
) -> str:
    """One run + one CONTRACT/UNDERLYING/MACRO prediction; returns its id."""
    register_run(run_id, ticker, asset_type, instrument_class, analysis_as_of)
    (prediction_id,) = submit_predictions(
        run_id,
        analyst,
        ticker,
        scope,
        [{"horizon_days": horizon, "direction": direction, "prob_up": prob_up}],
        analysis_as_of,
    )
    return prediction_id


def _write_ticket(run_id: str, payload: dict, ticket_id: str = "T-test1") -> None:
    """Insert one ticket-mirror row for the run (payload as JSON text)."""
    with ledger_transaction() as cur:
        cur.execute(
            "INSERT INTO tickets (ticket_id, decision_id, run_id, payload, "
            "ticket_version, written_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                ticket_id,
                None,
                run_id,
                json.dumps(payload),
                "v1",
                utc_now_iso(),
            ),
        )


@pytest.fixture()
def seams(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Seams]:
    """Base-scenario seams installed for every test that overrides pieces."""
    synthetic = _Seams()
    synthetic.install(monkeypatch)
    yield synthetic


# --------------------------------------------------------------------------- #
# The complete CONTRACT-scope row
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_contract_scope_complete_row_with_all_legs(seams: _Seams):
    prediction_id = _seed()
    _write_ticket(_RUN, {"ticket_id": "T-test1", "cost_detail": _COST_DETAIL})

    report = compute_outcomes(_NOW)
    assert (report.considered, report.completed, report.incomplete, report.failed) == (
        1, 1, 0, 0,
    )

    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "complete"
    assert record.legs_missing == ()
    assert record.contract_price_return == pytest.approx(0.10)
    assert record.funding_pnl == pytest.approx(-0.0015)  # up pays positive funding
    assert record.fees == pytest.approx(0.0010)
    assert record.slippage == pytest.approx(0.0004)
    assert record.net_return == pytest.approx(0.0971)
    assert record.outcome_available_at == "2026-08-30"  # exit bar's date
    assert record.ticket_id == "T-test1"
    # pure-crypto perp: no equity underlying, so that leg is n/a, not missing
    assert record.underlying_return is None
    assert record.basis_return is None
    # the perp seam saw exactly one call, windowed around the analysis date
    assert len(seams.perp_calls) == 1
    symbol, start, end = seams.perp_calls[0]
    assert symbol == "BTCUSDT"
    assert end == "2026-08-30"  # horizon end, not `now`
    assert start < "2026-08-25"  # lookback buffer reaches before the entry


@pytest.mark.unit
def test_second_pass_is_idempotent(seams: _Seams):
    prediction_id = _seed()
    _write_ticket(_RUN, {"cost_detail": _COST_DETAIL})
    first = compute_outcomes(_NOW)
    assert first.completed == 1
    # the complete outcome retired the prediction from the worklist
    second = compute_outcomes(_NOW)
    assert second.considered == 0
    assert len(outcomes_for_prediction(prediction_id)) == 1


# --------------------------------------------------------------------------- #
# Funding sign conventions (signed, never abs)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_funding_sign_follows_direction(seams: _Seams):
    """up pays positive funding; down receives it; flat carries zero pnl."""
    for analyst, direction in (
        ("bull", "up"),
        ("bear", "down"),
        ("neutral", "flat"),
    ):
        _seed(analyst=analyst, direction=direction)
    _write_ticket(_RUN, {"cost_detail": _COST_DETAIL})

    report = compute_outcomes(_NOW)
    assert report.completed == 3
    directions = {
        row.prediction_id: row.direction
        for row in predictions_for_run(_RUN)
    }
    records = {record.prediction_id: record for record in all_outcomes()}
    assert set(records) == set(directions)
    seen: dict[str, float] = {}
    for prediction_id, direction in directions.items():
        pnl = records[prediction_id].funding_pnl
        assert pnl is not None
        seen[direction] = pnl
    # window funding sum is +0.0015 for every row; only the sign flips —
    # the leg is signed, never an absolute value
    assert seen["up"] == pytest.approx(-0.0015)
    assert seen["down"] == pytest.approx(0.0015)
    assert seen["flat"] == 0.0


# --------------------------------------------------------------------------- #
# Fail-closed legs
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_funding_gap_fails_the_leg_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    seams = _Seams(
        funding_text=_funding_csv(0.0001, drop=frozenset({"2026-08-27 08:00:00"}))
    )
    seams.install(monkeypatch)
    prediction_id = _seed()
    _write_ticket(_RUN, {"cost_detail": _COST_DETAIL})

    report = compute_outcomes(_NOW)
    assert (report.completed, report.incomplete) == (0, 1)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.legs_missing == ("funding",)
    assert record.funding_pnl is None
    assert record.net_return is None
    assert record.contract_price_return == pytest.approx(0.10)  # price leg survives


@pytest.mark.unit
def test_ticket_absent_marks_ticket_cost(seams: _Seams):
    prediction_id = _seed()
    report = compute_outcomes(_NOW)
    assert report.incomplete == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.legs_missing == ("ticket_cost",)
    assert record.fees is None
    assert record.slippage is None
    assert record.net_return is None
    assert record.ticket_id is None


@pytest.mark.unit
def test_scalar_estimated_cost_booked_under_fees(seams: _Seams):
    prediction_id = _seed()
    _write_ticket(_RUN, {"ticket_id": "T-scalar", "estimated_cost": 0.0014})
    report = compute_outcomes(_NOW)
    assert report.completed == 1
    (record,) = outcomes_for_prediction(prediction_id)
    # no decomposition in the payload: the total lands under fees, slip 0
    assert record.fees == pytest.approx(0.0014)
    assert record.slippage == pytest.approx(0.0)
    # 0.10 - 0.0015 - 0.0014 = 0.0971 (same total frictions as the split)
    assert record.net_return == pytest.approx(0.0971)


# --------------------------------------------------------------------------- #
# Skip / pending / entry rules
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_horizon_end_not_bar_visible_skips_row(monkeypatch: pytest.MonkeyPatch):
    closes = {d: c for d, c in _PERP_CLOSES.items() if d <= "2026-08-25"}
    seams = _Seams(perp_closes=closes)
    seams.install(monkeypatch)
    prediction_id = _seed()

    report = compute_outcomes(_NOW)
    assert (report.considered, report.completed, report.incomplete, report.failed) == (
        1, 0, 0, 0,
    )
    (detail,) = report.details
    assert detail.status == "pending"
    assert detail.legs_missing == ("contract_price",)
    assert all_outcomes() == []  # nothing written
    # still on the worklist for the next batch
    assert [item["prediction_id"] for item in pending_predictions(_NOW)] == [prediction_id]


@pytest.mark.unit
def test_entry_bar_missing_writes_incomplete(monkeypatch: pytest.MonkeyPatch):
    late_closes = {"2026-09-01": 130.0, "2026-09-02": 131.0}  # nothing <= 08-25
    seams = _Seams(perp_closes=late_closes)
    seams.install(monkeypatch)
    prediction_id = _seed()

    report = compute_outcomes(_NOW)
    assert report.incomplete == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.legs_missing == ("contract_price",)
    assert record.net_return is None


@pytest.mark.unit
def test_macro_scope_is_skipped_not_scored(seams: _Seams):
    _seed(scope=SCOPE_MACRO)
    report = compute_outcomes(_NOW)
    assert (report.considered, report.completed, report.incomplete, report.failed) == (
        1, 0, 0, 0,
    )
    (detail,) = report.details
    assert detail.status == "skipped"
    assert all_outcomes() == []


# --------------------------------------------------------------------------- #
# Stock perps: equity seam, underlying + basis legs, funding n/a
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_underlying_scope_uses_equity_seam_and_skips_funding(
    monkeypatch: pytest.MonkeyPatch,
):
    seams = _Seams(
        equity_closes={"2026-08-25": 200.0, "2026-08-30": 210.0},
    )
    seams.install(monkeypatch)
    prediction_id = _seed(
        run_id="run-tsla-u",
        ticker="TSLAUSDT",
        instrument_class="stock_perp",
        scope=SCOPE_UNDERLYING,
    )
    _write_ticket("run-tsla-u", {"cost_detail": _COST_DETAIL})

    report = compute_outcomes(_NOW)
    assert report.completed == 1
    (record,) = outcomes_for_prediction(prediction_id)

    # priced on the EQUITY seam (TSLA), never on the perp seam
    assert seams.equity_calls and seams.equity_calls[0][0] == "TSLA"
    assert seams.perp_calls == []
    assert record.contract_price_return == pytest.approx(0.05)
    assert record.underlying_return == pytest.approx(0.05)
    assert record.basis_return == pytest.approx(0.0)
    # funding is structurally n/a for UNDERLYING scope: None, NOT a missing leg
    assert record.funding_pnl is None
    assert "funding" not in record.legs_missing
    assert record.legs_missing == ()
    assert record.status == "complete"
    # net = 0.05 - 0.0010 - 0.0004 (no funding addend)
    assert record.net_return == pytest.approx(0.05 - 0.0010 - 0.0004)


@pytest.mark.unit
def test_contract_scope_stock_perp_computes_basis(monkeypatch: pytest.MonkeyPatch):
    seams = _Seams(
        equity_closes={"2026-08-24": 199.0, "2026-08-25": 200.0, "2026-08-30": 212.0},
    )
    seams.install(monkeypatch)
    prediction_id = _seed(
        run_id="run-tsla-c",
        ticker="TSLAUSDT",
        instrument_class="stock_perp",
        scope=SCOPE_CONTRACT,
    )
    _write_ticket("run-tsla-c", {"cost_detail": _COST_DETAIL})

    report = compute_outcomes(_NOW)
    assert report.completed == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.contract_price_return == pytest.approx(0.10)  # perp seam
    assert record.underlying_return == pytest.approx(212.0 / 200.0 - 1.0)  # 0.06
    assert record.basis_return == pytest.approx(0.10 - 0.06)
    # CONTRACT scope on a perp: funding SHOULD exist and did
    assert record.funding_pnl == pytest.approx(-0.0015)
    assert record.status == "complete"


@pytest.mark.unit
def test_missing_underlying_is_diagnostic_not_blocking(monkeypatch: pytest.MonkeyPatch):
    seams = _Seams(equity_closes={})  # equity vendor serves nothing
    seams.install(monkeypatch)
    prediction_id = _seed(
        run_id="run-tsla-gap",
        ticker="TSLAUSDT",
        instrument_class="stock_perp",
        scope=SCOPE_CONTRACT,
    )
    _write_ticket("run-tsla-gap", {"cost_detail": _COST_DETAIL})

    report = compute_outcomes(_NOW)
    assert report.completed == 1  # all four net legs present
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "complete"
    assert record.legs_missing == ("underlying",)  # diagnostic gap disclosed
    assert record.underlying_return is None
    assert record.basis_return is None
    assert record.net_return == pytest.approx(0.0971)


# --------------------------------------------------------------------------- #
# Failure isolation + helpers
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_per_row_exception_isolated(monkeypatch: pytest.MonkeyPatch):
    seams = _Seams(raise_symbols=frozenset({"ETHUSDT"}))
    seams.install(monkeypatch)
    _seed(run_id="run-btc-ok", ticker="BTCUSDT")
    _seed(run_id="run-eth-bad", ticker="ETHUSDT")
    _write_ticket("run-btc-ok", {"cost_detail": _COST_DETAIL}, ticket_id="T-ok")
    _write_ticket("run-eth-bad", {"cost_detail": _COST_DETAIL}, ticket_id="T-bad")

    report = compute_outcomes(_NOW)
    assert (report.considered, report.completed, report.failed) == (2, 1, 1)
    failed = [detail for detail in report.details if detail.status == "failed"]
    assert len(failed) == 1
    assert "vendor down for ETHUSDT" in str(failed[0].error)
    assert len(all_outcomes()) == 1  # only the healthy row landed


@pytest.mark.unit
def test_underlying_equity_symbol_strips_quote_suffixes():
    from yialpha.ledger.outcome_compute import _underlying_equity_symbol

    assert _underlying_equity_symbol("TSLAUSDT") == "TSLA"
    assert _underlying_equity_symbol("1000PEPEUSDT") == "1000PEPE"
    assert _underlying_equity_symbol("AAPLUSD") == "AAPL"
    assert _underlying_equity_symbol("AAPL") == "AAPL"
