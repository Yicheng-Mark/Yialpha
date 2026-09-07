"""Independent P0 review regressions using synthetic data and temporary ledgers."""

import json

import pandas as pd
import pytest

from yialpha.ledger import outcome_compute
from yialpha.ledger.evidence import register_run
from yialpha.ledger.outcomes import outcomes_for_prediction, pending_predictions, write_outcome
from yialpha.ledger.predictions import submit_predictions
from yialpha.ledger.scoreboard import build_scoreboard
from yialpha.ledger.sqlite import ledger_transaction, utc_now_iso
from yialpha.versions import LEGACY_OUTCOME_VERSION, OUTCOME_COMPUTE_VERSION


@pytest.mark.unit
def test_legacy_positioning_last_settlement_before_deadline_remains_eligible(monkeypatch):
    formed = "2026-09-05T10:00:00+00:00"
    register_run("legacy-positioning-review", "BTCUSDT", "crypto_perp", "pure_crypto_perp", formed)
    (prediction_id,) = submit_predictions(
        "legacy-positioning-review", "positioning", "BTCUSDT", "POSITIONING",
        [{"horizon_days": 1, "direction": "up", "prob_up": 0.6}], formed,
    )
    monkeypatch.setattr(
        outcome_compute, "get_binance_funding_rate",
        lambda *a, **k: "\n".join([
            "fundingTime,fundingRate,symbol",
            "2026-09-05 00:00:00,0.0001,BTCUSDT",
            "2026-09-05 08:00:00,0.0001,BTCUSDT",
            "2026-09-05 16:00:00,0.0001,BTCUSDT",
            "2026-09-06 00:00:00,0.0001,BTCUSDT",
            "2026-09-06 08:00:00,0.0001,BTCUSDT",
        ]),
    )
    assert outcome_compute.compute_outcomes("2026-09-06T11:00:00+00:00").completed == 1
    (outcome,) = outcomes_for_prediction(prediction_id)
    assert outcome.outcome_available_at == "2026-09-06T08:00:00+00:00"
    assert outcome.scoring_context["window_end"] == "2026-09-06T10:00:00+00:00"
    board = build_scoreboard()
    assert board["overall"]["n"] == 1
    assert board["scoring_versions"] == [LEGACY_OUTCOME_VERSION]


@pytest.mark.unit
def test_new_outcome_metadata_cannot_promote_a_legacy_prediction():
    formed = "2026-09-05T12:00:00+00:00"
    register_run("legacy-upgrade-review", "BTCUSDT", "crypto_perp", "pure_crypto_perp", formed)
    (prediction_id,) = submit_predictions(
        "legacy-upgrade-review", "market", "BTCUSDT", "CONTRACT",
        [{"horizon_days": 1, "direction": "up", "prob_up": 0.6}], formed,
    )
    write_outcome(
        prediction_id, "legacy-upgrade-review", 1, status="complete",
        contract_price_return=0.05, funding_pnl=-0.001,
        fees=0.0008, slippage=0.0002, net_return=0.048,
        outcome_available_at="2026-09-06T00:00:00+00:00",
        scoring_context={
            "version": OUTCOME_COMPUTE_VERSION,
            "prediction_formed_at": formed,
            "window_start": "2026-09-05T00:00:00+00:00",
            "window_end": "2026-09-06T00:00:00+00:00",
            "horizon_end": "2026-09-06T12:00:00+00:00",
            "reference_source": "binance_perp:1d:last",
        },
    )
    assert build_scoreboard()["overall"]["n"] == 0


def _seed_daily_close_review(monkeypatch, *, equity=False):
    formed = "2026-09-03T12:00:00+00:00" if equity else "2026-09-05T12:00:00+00:00"
    start = "2026-09-03T00:00:00+00:00" if equity else "2026-09-05T00:00:00+00:00"
    symbol = "MUUSDT" if equity else "BTCUSDT"
    run_id = "daily-close-review"
    register_run(run_id, symbol, "crypto_perp", "stock_perp" if equity else "pure_crypto_perp", formed)
    (prediction_id,) = submit_predictions(
        run_id, "market", symbol, "UNDERLYING" if equity else "CONTRACT",
        [{"horizon_days": 1, "direction": "up", "prob_up": 0.6}], formed,
        timing={
            "version": OUTCOME_COMPUTE_VERSION, "prediction_formed_at": formed,
            "reference_price": 100.0, "reference_price_at": start,
            "reference_available_at": start, "reference_observed_at": formed,
            "reference_source": "yfinance:1d:close" if equity else "binance_perp:1d:last",
            "reference_error": None,
        },
    )
    with ledger_transaction() as cur:
        cur.execute(
            "INSERT INTO tickets (ticket_id, run_id, payload, ticket_version, written_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("daily-close-review-ticket", run_id, json.dumps({
                "entry_fee_bps": 4.0, "exit_fee_bps": 4.0,
                "entry_slippage_bps": 1.0, "exit_slippage_bps": 1.0,
            }), "v1", utc_now_iso()),
        )
    monkeypatch.setattr(outcome_compute, "get_binance_funding_rate", lambda *a, **k: "\n".join([
        "fundingTime,fundingRate,symbol",
        "2026-09-05 00:00:00,0.0001,BTCUSDT",
        "2026-09-05 08:00:00,0.0001,BTCUSDT",
        "2026-09-05 16:00:00,0.0001,BTCUSDT",
        "2026-09-06 00:00:00,0.0001,BTCUSDT",
    ]))
    return prediction_id


def _daily_review_frame(rows):
    return pd.DataFrame({"Close": [price for _, price in rows]}, index=pd.to_datetime([day for day, _ in rows]))


@pytest.mark.unit
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_endpoint_closes_retry_then_expire_independent_of_order(monkeypatch, reverse):
    prediction_id = _seed_daily_close_review(monkeypatch)
    rows = [("2026-09-05", 105.0), ("2026-09-05", 120.0)]
    monkeypatch.setattr(outcome_compute, "binance_klines_frame", lambda *a, **k: _daily_review_frame(rows[::-1] if reverse else rows))
    report = outcome_compute.compute_outcomes("2026-09-06T12:00:00+00:00")
    assert report.details[0].status == "pending"
    assert report.details[0].reason == "price_bar_conflict:2026-09-05"
    assert outcomes_for_prediction(prediction_id) == []
    report = outcome_compute.compute_outcomes("2026-09-13T12:00:00+00:00")
    assert (report.completed, report.incomplete, report.failed) == (0, 1, 0)
    (outcome,) = outcomes_for_prediction(prediction_id)
    assert outcome.scoring_context["reason"] == "price_bar_conflict:2026-09-05"
    assert outcome.net_return is None
    assert pending_predictions("2026-09-13T12:00:00+00:00") == []


@pytest.mark.unit
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("conflict_day", ["2026-09-02", "2026-09-03"])
def test_equity_reference_or_endpoint_conflict_can_recover_before_deadline(monkeypatch, reverse, conflict_day):
    prediction_id = _seed_daily_close_review(monkeypatch, equity=True)
    rows = [("2026-09-02", 100.0), ("2026-09-03", 105.0), (conflict_day, 120.0)]
    monkeypatch.setattr(outcome_compute, "get_YFin_history_cached", lambda *a, **k: _daily_review_frame(rows[::-1] if reverse else rows))
    report = outcome_compute.compute_outcomes("2026-09-04T12:00:00+00:00")
    assert report.details[0].status == "pending"
    assert report.details[0].reason == f"price_bar_conflict:{conflict_day}"
    assert outcomes_for_prediction(prediction_id) == []
    rows.pop()
    report = outcome_compute.compute_outcomes("2026-09-05T12:00:00+00:00")
    assert report.completed == 1
    assert outcomes_for_prediction(prediction_id)[0].net_return == pytest.approx(0.049)


@pytest.mark.unit
@pytest.mark.parametrize("equity", [False, True])
def test_identical_daily_closes_and_unrelated_conflicts_do_not_block_scoring(monkeypatch, equity):
    prediction_id = _seed_daily_close_review(monkeypatch, equity=equity)
    rows = [("2026-08-31", 80.0), ("2026-08-31", 90.0)]
    if equity:
        rows += [("2026-09-02", 100.0)] * 2 + [("2026-09-03", 105.0)] * 2
    else:
        rows += [("2026-09-05", 105.0)] * 2
    seam = "get_YFin_history_cached" if equity else "binance_klines_frame"
    monkeypatch.setattr(outcome_compute, seam, lambda *a, **k: _daily_review_frame(rows))
    report = outcome_compute.compute_outcomes("2026-09-06T12:00:00+00:00")
    assert report.completed == 1
    assert outcomes_for_prediction(prediction_id)[0].net_return == pytest.approx(0.049 if equity else 0.0487)
    assert build_scoreboard()["overall"]["n"] == 1
