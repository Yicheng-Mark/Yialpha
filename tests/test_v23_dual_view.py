"""V2.3 — PM dual view, legacy side mapping, POSITIONING-scope outcomes.

* Dual view: PortfolioDecision's three stock-perp fields render a compact
  block ONLY when filled; all-None renders byte-identically to pre-V2.3
  (pinned). ``_decision_fields_dict`` passes them through only when set, so
  absent fields keep the logged state JSON byte-identical.
* ``desired_side_from_decision``: the frozen legacy rating->intent table —
  it can NEVER return SHORT (a short requires V2.4's explicit structured
  field).
* POSITIONING outcome pricing: the realized quantity IS the cumulative
  funding sum over the horizon window — direction-independent net_return,
  sign-based hits, fail-closed on a settlement-grid gap.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import yialpha.ledger.outcome_compute as oc
from yialpha.agents.managers.portfolio_manager import _decision_fields_dict
from yialpha.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    render_pm_decision,
)
from yialpha.ledger.evidence import register_run
from yialpha.ledger.models import SCOPE_POSITIONING
from yialpha.ledger.outcome_compute import compute_outcomes
from yialpha.ledger.outcomes import all_outcomes
from yialpha.ledger.predictions import submit_predictions
from yialpha.tickets import desired_side_from_decision

_RUN = "run-v23-dual-1"
_NOW = "2026-09-05"


def _decision(**over) -> PortfolioDecision:
    base: dict = {
        "rating": PortfolioRating.BUY,
        "executive_summary": "s",
        "investment_thesis": "t",
        "price_target": 60.0,
        "time_horizon": "3-6 months",
        "confidence": 0.7,
    }
    base.update(over)
    return PortfolioDecision(**base)


# ---- dual view rendering ------------------------------------------------------


@pytest.mark.unit
def test_dual_view_absent_renders_byte_identical():
    without = render_pm_decision(_decision())
    # Same decision built with explicit None dual fields: identical bytes.
    with_nones = render_pm_decision(
        _decision(
            underlying_direction=None,
            contract_direction=None,
            basis_view=None,
        )
    )
    assert without == with_nones
    assert "Underlying view" not in without
    assert "Contract view" not in without


@pytest.mark.unit
def test_dual_view_block_renders_when_filled():
    md = render_pm_decision(
        _decision(
            underlying_direction="bullish",
            contract_direction="avoid_long",
            basis_view="perp premium rich with expensive positive funding",
        )
    )
    assert "**Underlying view**: bullish" in md
    assert "**Contract view**: avoid_long" in md
    assert "**Basis view**: perp premium rich" in md
    # Partial fill renders only the filled lines.
    partial = render_pm_decision(_decision(contract_direction="avoid"))
    assert "Underlying view" not in partial
    assert "**Contract view**: avoid" in partial


@pytest.mark.unit
def test_decision_fields_dict_passthrough_is_none_silent():
    empty = _decision_fields_dict(_decision())
    assert "underlying_direction" not in empty
    filled = _decision_fields_dict(
        _decision(
            underlying_direction="bearish",
            contract_direction="avoid_short",
            basis_view="discounted perp vs index",
        )
    )
    assert filled["underlying_direction"] == "bearish"
    assert filled["contract_direction"] == "avoid_short"
    assert filled["basis_view"] == "discounted perp vs index"
    # None decision stays {} (free-text fallback).
    assert _decision_fields_dict(None) == {}


# ---- legacy side mapping ------------------------------------------------------


@pytest.mark.unit
def test_desired_side_table_never_invents_a_short():
    table = {
        "Buy": "LONG",
        "Overweight": "LONG",
        "Hold": "FLAT",
        "Underweight": "REDUCE",
        "Sell": "CLOSE",
    }
    for rating, expected in table.items():
        assert desired_side_from_decision({"rating": rating}) == expected
    # Missing / unknown / dual-view fields never produce SHORT (or anything
    # else): fail neutral, and opinions are not consulted.
    assert desired_side_from_decision(None) == "FLAT"
    assert desired_side_from_decision({}) == "FLAT"
    assert desired_side_from_decision({"rating": "nonsense"}) == "FLAT"
    assert (
        desired_side_from_decision(
            {"rating": "Sell", "contract_direction": "bullish"}
        )
        == "CLOSE"
    )


# ---- POSITIONING outcome pricing ----------------------------------------------


def _funding_csv(rate: float, drop: str | None = None) -> str:
    start = datetime(2026, 8, 25)
    stop = datetime(2026, 8, 30)
    lines = ["# fixture", "fundingTime,fundingRate,symbol"]
    moment = start
    while moment <= stop:
        stamp = moment.strftime("%Y-%m-%d %H:%M:%S")
        if stamp != drop:
            lines.append(f"{stamp},{rate},BTCUSDT")
        moment += timedelta(hours=8)
    return "\n".join(lines) + "\n"


def _expected_sum(rate: float) -> float:
    start = datetime(2026, 8, 25)
    stop = datetime(2026, 8, 30)
    count = 0
    moment = start
    while moment <= stop:
        if moment > start:  # window is (analysis_as_of, horizon_end]
            count += 1
        moment += timedelta(hours=8)
    return rate * count


def _seed_positioning(direction: str) -> str:
    register_run(_RUN, "BTCUSDT", "crypto_perp", "pure_crypto_perp", "2026-08-25")
    (prediction_id,) = submit_predictions(
        _RUN,
        "positioning",
        "BTCUSDT",
        SCOPE_POSITIONING,
        [{"horizon_days": 5, "direction": direction, "prob_up": 0.7}],
        "2026-08-25",
    )
    return prediction_id


@pytest.mark.unit
def test_positioning_outcome_is_the_realized_funding_sum(monkeypatch):
    monkeypatch.setattr(oc, "get_binance_funding_rate", lambda *a: _funding_csv(0.0001))
    prediction_id = _seed_positioning("up")
    report = compute_outcomes(_NOW)
    assert (report.considered, report.completed, report.incomplete) == (1, 1, 0)
    (row,) = all_outcomes()
    assert row.prediction_id == prediction_id
    assert row.status == "complete"
    assert row.net_return == pytest.approx(_expected_sum(0.0001))
    assert row.funding_pnl == pytest.approx(_expected_sum(0.0001))
    # No price legs fabricated for a funding-sign call.
    assert row.contract_price_return is None
    assert row.fees is None


@pytest.mark.unit
def test_positioning_net_return_is_direction_independent(monkeypatch):
    monkeypatch.setattr(oc, "get_binance_funding_rate", lambda *a: _funding_csv(0.0001))
    up = _seed_positioning("up")
    (down,) = submit_predictions(
        _RUN,
        "positioning-b",
        "BTCUSDT",
        SCOPE_POSITIONING,
        [{"horizon_days": 5, "direction": "down", "prob_up": 0.3}],
        "2026-08-25",
    )
    compute_outcomes(_NOW)
    rows = {row.prediction_id: row for row in all_outcomes()}
    # The realized quantity does not depend on what was predicted; only the
    # scoreboard's hit rule (direction up hit iff sum > 0) does.
    assert rows[up].net_return == pytest.approx(rows[down].net_return)


@pytest.mark.unit
def test_positioning_funding_gap_fails_closed(monkeypatch):
    monkeypatch.setattr(
        oc,
        "get_binance_funding_rate",
        lambda *a: _funding_csv(0.0001, drop="2026-08-27 08:00:00"),
    )
    _seed_positioning("up")
    report = compute_outcomes(_NOW)
    assert (report.completed, report.incomplete) == (0, 1)
    (row,) = all_outcomes()
    assert row.status == "incomplete"
    assert "funding_window" in (row.legs_missing or "")
    assert row.net_return is None
