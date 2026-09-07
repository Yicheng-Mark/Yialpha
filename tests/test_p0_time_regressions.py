"""P0 price/time contract regressions, using only synthetic market data.

Before repair, 11 characterization cases reproduced the daily intraday gate,
the perpetual same-bar weekend retry, and append-only retry conflicts. These
tests now pin the fail-closed legacy policy and the versioned new contract.
The shared conftest isolates ledgers/caches under tmp_path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from yialpha.ledger import outcome_compute
from yialpha.ledger.evidence import register_run
from yialpha.ledger.models import SCOPE_CONTRACT, SCOPE_UNDERLYING
from yialpha.ledger.outcomes import outcomes_for_prediction, pending_predictions, write_outcome
from yialpha.ledger.predictions import submit_predictions
from yialpha.ledger.sqlite import ledger_transaction, utc_now_iso

_ANALYSIS = "2026-09-05T14:30:00+00:00"
_NOW = "2026-09-30T00:00:00+00:00"
_START = "2026-09-05T00:00:00+00:00"
_COSTS = {
    "entry_fee_bps": 5.0, "exit_fee_bps": 5.0,
    "entry_slippage_bps": 2.0, "exit_slippage_bps": 2.0,
}


def _frame(closes: dict[str, float]) -> pd.DataFrame:
    days = sorted(closes)
    return pd.DataFrame({"Close": [closes[d] for d in days]}, index=pd.to_datetime(days))


@dataclass
class _Market:
    perp: dict[str, float]
    equity: dict[str, float]
    funding: dict[datetime, float]
    calls: list[tuple] = field(default_factory=list)

    def funding_csv(self, *args, **kwargs) -> str:
        self.calls.append(args)
        rows = ["fundingTime,fundingRate,symbol"]
        rows.extend(f"{moment:%Y-%m-%d %H:%M:%S},{rate},BTCUSDT" for moment, rate in sorted(self.funding.items()))
        return "\n".join(rows)


@pytest.fixture()
def p0_market(monkeypatch: pytest.MonkeyPatch) -> _Market:
    days = pd.date_range("2026-08-26", "2026-09-29", freq="D")
    perp = {day.strftime("%Y-%m-%d"): 100.0 + i for i, day in enumerate(days)}
    perp["2026-09-04"] = 100.0
    equity = {day: price for day, price in perp.items() if datetime.fromisoformat(day).weekday() < 5}
    funding = {}
    moment = datetime(2026, 8, 26, tzinfo=UTC)
    while moment <= datetime(2026, 9, 30, tzinfo=UTC):
        funding[moment] = 0.0001
        moment += timedelta(hours=8)
    market = _Market(perp, equity, funding)
    monkeypatch.setattr(outcome_compute, "binance_klines_frame", lambda *a, **k: _frame(market.perp))
    monkeypatch.setattr(outcome_compute, "get_YFin_history_cached", lambda *a, **k: _frame(market.equity))
    monkeypatch.setattr(outcome_compute, "get_binance_funding_rate", market.funding_csv)
    return market


def _timing(*, equity=False, **changes) -> dict:
    return {
        "version": "close_reference_v1",
        "prediction_formed_at": _ANALYSIS,
        "reference_price": 100.0,
        "reference_price_at": _START,
        "reference_available_at": _START,
        "reference_observed_at": "2026-09-05T14:29:59+00:00",
        "reference_source": "yfinance:1d:close" if equity else "binance_perp:1d:last",
        "reference_error": None,
        **changes,
    }


def _seed(
    symbol: str, horizon: int, *, scope: str = SCOPE_CONTRACT,
    timing: dict | None = None, analysis_as_of: str = _ANALYSIS,
    cost_payload: dict | None = None,
) -> str:
    run_id = f"p0-{symbol}-{scope}-{horizon}"
    instrument_class = "stock_perp" if symbol == "MUUSDT" else "pure_crypto_perp"
    register_run(run_id, symbol, "crypto_perp", instrument_class, analysis_as_of)
    kwargs = {"timing": timing} if timing is not None else {}
    (prediction_id,) = submit_predictions(
        run_id, "market", symbol, scope,
        [{"horizon_days": horizon, "direction": "up", "prob_up": 0.6}], analysis_as_of,
        **kwargs,
    )
    with ledger_transaction() as cur:
        cur.execute(
            "INSERT INTO tickets (ticket_id, decision_id, run_id, payload, ticket_version, written_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"ticket-{run_id}", None, run_id,
                json.dumps(cost_payload if cost_payload is not None else {"cost_detail": _COSTS}),
                "v1", utc_now_iso(),
            ),
        )
    return prediction_id


@pytest.mark.unit
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "MUUSDT"])
@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_intraday_legacy_daily_entry_stays_unscoreable_and_retires(p0_market, symbol, horizon):
    prediction_id = _seed(symbol, horizon)
    first = outcome_compute.compute_outcomes(_NOW)
    assert (first.completed, first.incomplete, first.failed) == (0, 1, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.net_return is None
    assert record.scoring_context["version"] != "close_reference_v1"
    assert record.scoring_context["reason"]
    assert pending_predictions(_NOW) == []
    later = outcome_compute.compute_outcomes("2026-10-30T00:00:00+00:00")
    assert later.considered == 0
    assert outcomes_for_prediction(prediction_id) == [record]


@pytest.mark.unit
@pytest.mark.parametrize("versioned", [False, True])
def test_weekend_underlying_same_bar_has_terminal_reason(p0_market, versioned):
    prediction_id = _seed("MUUSDT", 1, scope=SCOPE_UNDERLYING, timing=_timing(equity=True) if versioned else None)
    report = outcome_compute.compute_outcomes(_NOW)
    assert (report.considered, report.completed, report.incomplete, report.failed) == (1, 0, 1, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.net_return is None
    assert record.scoring_context["reason"] == "no_new_trading_session"
    assert pending_predictions(_NOW) == []
    assert outcome_compute.compute_outcomes("2026-10-30T00:00:00+00:00").considered == 0


@pytest.mark.unit
def test_legacy_incomplete_cannot_be_rewritten_or_retried(p0_market):
    prediction_id = _seed("BTCUSDT", 1)
    assert outcome_compute.compute_outcomes(_NOW).incomplete == 1
    (original,) = outcomes_for_prediction(prediction_id)
    p0_market.perp["2026-09-05"] = 200.0
    assert outcome_compute.compute_outcomes(_NOW).considered == 0
    with pytest.raises(ValueError, match="immutable"):
        write_outcome(prediction_id, original.run_id, 1, status="complete", net_return=0.5)
    assert outcomes_for_prediction(prediction_id) == [original]


@pytest.mark.unit
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "MUUSDT"])
@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_intraday_versioned_contract_uses_known_reference_and_closed_endpoint(p0_market, symbol, horizon):
    prediction_id = _seed(symbol, horizon, timing=_timing())
    report = outcome_compute.compute_outcomes(_NOW)
    assert (report.completed, report.incomplete, report.failed) == (1, 0, 0)
    (record,) = outcomes_for_prediction(prediction_id)
    deadline = datetime.fromisoformat(_ANALYSIS) + timedelta(days=horizon)
    window_end = datetime.combine(deadline.date(), datetime.min.time(), tzinfo=UTC)
    endpoint = (window_end - timedelta(days=1)).date().isoformat()
    expected_price_return = p0_market.perp[endpoint] / 100.0 - 1.0
    assert record.contract_price_return == pytest.approx(expected_price_return)
    assert record.funding_pnl == pytest.approx(-horizon * 3 * 0.0001)
    assert record.fees == pytest.approx(0.001)
    assert record.slippage == pytest.approx(0.0004)
    assert record.net_return == pytest.approx(expected_price_return - horizon * 0.0003 - 0.0014)
    assert record.outcome_available_at == window_end.isoformat()
    assert record.scoring_context["version"] == "close_reference_v1"
    assert record.scoring_context["window_start"] == _START
    assert record.scoring_context["window_end"] == window_end.isoformat()
    assert record.scoring_context["horizon_end"] == deadline.isoformat()
    assert record.scoring_context["reference_source"] == "binance_perp:1d:last"
    assert pending_predictions(_NOW) == []
    assert outcome_compute.compute_outcomes(_NOW).considered == 0


@pytest.mark.unit
def test_future_market_updates_cannot_change_persisted_prediction_reference(p0_market):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing())
    # Neither vendor history revisions nor future closes may reprice entry.
    p0_market.perp["2026-09-04"] = 999.0
    p0_market.perp["2026-09-05"] = 220.0
    assert outcome_compute.compute_outcomes(_NOW).completed == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.contract_price_return == pytest.approx(1.2)
    assert record.scoring_context["window_start"] == _START


@pytest.mark.unit
def test_versioned_horizon_is_based_on_formation_not_run_start(p0_market):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing(), analysis_as_of="2026-09-05T00:00:00+00:00")
    assert outcome_compute.compute_outcomes("2026-09-06T14:29:59+00:00").considered == 0
    assert outcomes_for_prediction(prediction_id) == []
    assert outcome_compute.compute_outcomes("2026-09-06T14:30:00+00:00").completed == 1


@pytest.mark.unit
def test_missing_closed_endpoint_cannot_use_forming_bar_and_can_retry(p0_market):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing())
    required_close = p0_market.perp.pop("2026-09-05")
    p0_market.perp["2026-09-06"] = 9000.0  # Still forming at scoring time.
    now = "2026-09-06T15:00:00+00:00"
    pending = outcome_compute.compute_outcomes(now)
    assert (pending.completed, pending.incomplete, pending.failed) == (0, 0, 0)
    assert pending.details[0].status == "pending"
    assert outcomes_for_prediction(prediction_id) == []
    p0_market.perp["2026-09-05"] = required_close
    assert outcome_compute.compute_outcomes(now).completed == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.contract_price_return == pytest.approx(required_close / 100.0 - 1.0)


@pytest.mark.unit
def test_history_gap_reaches_terminal_exactly_at_seven_day_deadline(p0_market):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing())
    del p0_market.perp["2026-09-05"]
    before = outcome_compute.compute_outcomes("2026-09-13T14:29:59+00:00")
    assert before.details[0].status == "pending"
    assert outcomes_for_prediction(prediction_id) == []
    deadline = outcome_compute.compute_outcomes("2026-09-13T14:30:00+00:00")
    assert deadline.incomplete == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.net_return is None
    assert record.scoring_context["reason"]
    assert pending_predictions(_NOW) == []


@pytest.mark.unit
def test_versioned_funding_integrates_exact_price_window_and_costs_once(p0_market, monkeypatch):
    start = datetime.fromisoformat(_START)
    end = start + timedelta(days=1)
    p0_market.funding[start] = 9.0  # Open boundary is excluded.
    p0_market.funding[end] = 0.0002  # Closed boundary is included.
    p0_market.funding[end + timedelta(hours=8)] = 9.0
    windows = []
    original_sum = outcome_compute._funding_window_sum

    def capture(symbol, window_start, window_end):
        windows.append((window_start, window_end))
        return original_sum(symbol, window_start, window_end)

    monkeypatch.setattr(outcome_compute, "_funding_window_sum", capture)
    prediction_id = _seed(
        "BTCUSDT", 1, timing=_timing(),
        cost_payload={"cost_detail": {**_COSTS, "estimated_funding_bps": 9000.0}, "estimated_cost": 0.9},
    )
    assert outcome_compute.compute_outcomes(_NOW).completed == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert windows == [(start, end)]
    assert record.funding_pnl == pytest.approx(-0.0004)
    assert record.net_return == pytest.approx(0.1 - 0.0004 - 0.0014)


@pytest.mark.unit
def test_versioned_scalar_cost_is_retryable_then_terminal(p0_market):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing(), cost_payload={"estimated_cost": 0.01})
    pending = outcome_compute.compute_outcomes("2026-09-06T15:00:00+00:00")
    assert pending.details[0].status == "pending"
    assert outcomes_for_prediction(prediction_id) == []
    assert outcome_compute.compute_outcomes("2026-09-13T14:30:00+00:00").incomplete == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.net_return is None
    assert pending_predictions(_NOW) == []


@pytest.mark.unit
def test_unscoreable_oldest_prediction_does_not_block_later_complete(p0_market):
    old = _seed("BTCUSDT", 1, analysis_as_of="2026-09-04T14:30:00+00:00")
    new = _seed("ETHUSDT", 1, timing=_timing())
    report = outcome_compute.compute_outcomes(_NOW, limit=1)
    assert (report.considered, report.completed, report.incomplete, report.failed) == (2, 1, 1, 0)
    assert outcomes_for_prediction(old)[0].status == "incomplete"
    assert outcomes_for_prediction(new)[0].status == "complete"
    assert pending_predictions(_NOW) == []


@pytest.mark.unit
def test_missing_funding_retries_without_writing_then_completes(p0_market):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing())
    missing_slot = datetime(2026, 9, 5, 8, tzinfo=UTC)
    rate = p0_market.funding.pop(missing_slot)
    now = "2026-09-06T15:00:00+00:00"
    report = outcome_compute.compute_outcomes(now)
    assert report.details[0].status == "pending"
    assert report.details[0].legs_missing == ("funding",)
    assert outcomes_for_prediction(prediction_id) == []
    p0_market.funding[missing_slot] = rate
    assert outcome_compute.compute_outcomes(now).completed == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.funding_pnl == pytest.approx(-0.0003)


@pytest.mark.unit
def test_missing_prediction_reference_is_not_reconstructed_from_future_history(p0_market):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing(
        reference_price=None, reference_price_at=None, reference_available_at=None,
        reference_error="reference_bar_missing:2026-09-04",
    ))
    # Complete market history now exists, but was never captured at formation.
    report = outcome_compute.compute_outcomes("2026-09-06T15:00:00+00:00")
    assert report.incomplete == 1
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.net_return is None
    assert "reference_unavailable_at_prediction" in record.scoring_context["reason"]
    assert pending_predictions(_NOW) == []


@pytest.mark.unit
@pytest.mark.parametrize("seam", ["binance_klines_frame", "get_binance_funding_rate"])
def test_vendor_failures_obey_retry_deadline(p0_market, monkeypatch, seam):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing())

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic vendor outage")

    monkeypatch.setattr(outcome_compute, seam, fail)
    first = outcome_compute.compute_outcomes("2026-09-06T15:00:00+00:00")
    assert first.details[0].status == "pending"
    assert outcomes_for_prediction(prediction_id) == []
    final = outcome_compute.compute_outcomes("2026-09-13T14:30:00+00:00")
    assert final.incomplete == 1
    assert outcomes_for_prediction(prediction_id)[0].net_return is None
    assert pending_predictions(_NOW) == []


@pytest.mark.unit
def test_leading_funding_gap_cannot_claim_a_complete_holding_window(p0_market):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing())
    first_remaining = datetime(2026, 9, 5, 16, tzinfo=UTC)
    p0_market.funding = {t: r for t, r in p0_market.funding.items() if t >= first_remaining}
    result = outcome_compute.compute_outcomes("2026-09-06T15:00:00+00:00")
    assert result.details[0].status == "pending"
    assert "funding" in result.details[0].legs_missing
    assert outcomes_for_prediction(prediction_id) == []


@pytest.mark.unit
def test_duplicate_funding_evidence_is_never_double_charged(p0_market, monkeypatch):
    prediction_id = _seed("BTCUSDT", 1, timing=_timing())
    original = outcome_compute.get_binance_funding_rate

    def duplicate(*args, **kwargs):
        text = original(*args, **kwargs)
        return text + "\n2026-09-05 08:00:00,0.0001,BTCUSDT\n"

    monkeypatch.setattr(outcome_compute, "get_binance_funding_rate", duplicate)
    assert outcome_compute.compute_outcomes(_NOW).completed == 1
    assert outcomes_for_prediction(prediction_id)[0].funding_pnl == pytest.approx(-0.0003)


@pytest.mark.unit
def test_legacy_intraday_entry_gap_cannot_bypass_knowability_gate(p0_market):
    prediction_id = _seed("BTCUSDT", 1)
    del p0_market.perp["2026-09-05"]
    # The previous close is available, but this legacy claim never froze it
    # as a reference. An entry-history gap cannot switch its price contract.
    report = outcome_compute.compute_outcomes(_NOW)
    assert report.completed == 0
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "incomplete"
    assert record.net_return is None
    assert record.scoring_context["reason"] == "legacy_entry_bar_missing"
    assert pending_predictions(_NOW) == []


@pytest.mark.unit
@pytest.mark.parametrize("revised_price", [50.0, 99.0])
def test_equity_revised_price_basis_never_mixes_with_frozen_reference(p0_market, revised_price):
    prediction_id = _seed("MUUSDT", 5, scope=SCOPE_UNDERLYING, timing=_timing(equity=True))
    p0_market.equity["2026-09-04"] = revised_price  # split/dividend-adjusted history
    p0_market.equity["2026-09-09"] = revised_price * 1.1
    report = outcome_compute.compute_outcomes(_NOW)
    assert report.completed == 0
    record = outcomes_for_prediction(prediction_id)[0]
    assert record.status == "incomplete"
    assert record.net_return is None
    assert record.scoring_context["reference_price"] == 100.0
    assert record.scoring_context["reason"] == "equity_reference_basis_changed"


@pytest.mark.unit
@pytest.mark.parametrize("horizon", [5, 21])
def test_equity_matching_reference_basis_keeps_valid_window(p0_market, horizon):
    from yialpha.ledger.scoreboard import build_scoreboard

    prediction_id = _seed("MUUSDT", horizon, scope=SCOPE_UNDERLYING, timing=_timing(equity=True))
    report = outcome_compute.compute_outcomes(_NOW)
    assert report.completed == 1
    record = outcomes_for_prediction(prediction_id)[0]
    assert record.scoring_context["reference_price"] == 100.0
    assert record.funding_pnl is None
    assert record.scoring_context["reference_basis_check"] == "matched_frozen_close"
    assert build_scoreboard(scoring_version="close_reference_v1")["overall"]["n"] == 1


@pytest.mark.unit
def test_equity_missing_basis_evidence_retries_without_reconstructing_entry(p0_market):
    prediction_id = _seed("MUUSDT", 5, scope=SCOPE_UNDERLYING, timing=_timing(equity=True))
    del p0_market.equity["2026-09-04"]
    pending = outcome_compute.compute_outcomes("2026-09-10T15:00:00+00:00")
    assert pending.details[0].status == "pending"
    assert outcomes_for_prediction(prediction_id) == []
    terminal = outcome_compute.compute_outcomes(_NOW)
    assert terminal.incomplete == 1
    record = outcomes_for_prediction(prediction_id)[0]
    assert record.scoring_context["reference_price"] == 100.0
    assert record.scoring_context["reason"] == "equity_reference_basis_unverifiable"
