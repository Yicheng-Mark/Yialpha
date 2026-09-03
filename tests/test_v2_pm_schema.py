"""V2.0 — PortfolioDecision extension + PM field capture wiring.

The five new optional fields must be exactly that: absent -> the rendered
markdown is byte-identical to pre-V2 (legacy parsers keep working); present
-> appended lines the ticket builder and evidence log consume.
"""

from __future__ import annotations

import pytest

from yialpha.agents.managers.portfolio_manager import _decision_fields_dict
from yialpha.agents.schemas import (
    OutcomeProbabilities,
    PortfolioDecision,
    render_pm_decision,
)
from yialpha.tickets import desired_side_from_decision
from yialpha.versions import SCHEMA_VERSION


def _old_shape_decision() -> PortfolioDecision:
    return PortfolioDecision(
        rating="Buy",
        executive_summary="Add on strength.",
        investment_thesis="Momentum with support.",
        price_target=210.0,
        time_horizon="3-6 months",
    )


@pytest.mark.unit
def test_old_shape_decision_has_none_for_all_new_fields():
    d = _old_shape_decision()
    assert d.confidence is None
    assert d.probabilities is None
    assert d.expected_return is None
    assert d.invalidation is None
    assert d.evidence_coverage is None


@pytest.mark.unit
def test_render_without_new_fields_is_pre_v2_shape():
    md = render_pm_decision(_old_shape_decision())
    assert md == (
        "**Rating**: Buy\n\n"
        "**Executive Summary**: Add on strength.\n\n"
        "**Investment Thesis**: Momentum with support.\n\n"
        "**Price Target**: 210.0\n\n"
        "**Time Horizon**: 3-6 months"
    )
    # Nothing V2 leaks into the legacy render.
    for absent in ("Confidence", "Probabilities", "Expected Return",
                   "Invalidation", "Evidence Coverage"):
        assert absent not in md


@pytest.mark.unit
def test_render_with_all_new_fields():
    d = PortfolioDecision(
        rating="Overweight",
        executive_summary="s",
        investment_thesis="t",
        confidence=0.71,
        probabilities=OutcomeProbabilities(bull=0.61, neutral=0.24, bear=0.15),
        expected_return=0.047,
        invalidation=["closes below 180", "funding flips negative"],
        evidence_coverage=0.86,
    )
    md = render_pm_decision(d)
    assert "**Confidence**: 71%" in md
    assert "**Probabilities**: bull 61% / neutral 24% / bear 15%" in md
    assert "**Expected Return**: +4.7%" in md
    assert "**Invalidation**: closes below 180; funding flips negative" in md
    assert "**Evidence Coverage**: 86%" in md


@pytest.mark.unit
def test_nullish_strings_coerce_to_none():
    d = PortfolioDecision(
        rating="Hold",
        executive_summary="s",
        investment_thesis="t",
        confidence="N/A",
        expected_return="-",
        evidence_coverage="unknown",
    )
    assert d.confidence is None
    assert d.expected_return is None
    assert d.evidence_coverage is None


@pytest.mark.unit
def test_decision_fields_dict_none_is_empty():
    assert _decision_fields_dict(None) == {}


@pytest.mark.unit
def test_decision_fields_dict_flattens_everything():
    d = PortfolioDecision(
        rating="Buy",
        executive_summary="s",
        investment_thesis="t",
        price_target=220.0,
        time_horizon="1 week",
        confidence=0.7,
        probabilities=OutcomeProbabilities(bull=0.6, neutral=0.3, bear=0.1),
        expected_return=0.05,
        invalidation=["break of 190"],
        evidence_coverage=0.9,
    )
    fields = _decision_fields_dict(d)
    assert fields["rating"] == "Buy"
    assert fields["price_target"] == 220.0
    assert fields["confidence"] == 0.7
    assert fields["probabilities"] == {"bull": 0.6, "neutral": 0.3, "bear": 0.1}
    assert fields["expected_return"] == 0.05
    assert fields["invalidation"] == ["break of 190"]
    assert fields["evidence_coverage"] == 0.9
    assert fields["schema_version"] == SCHEMA_VERSION


@pytest.mark.unit
def test_decision_fields_dict_carries_explicit_short_desired_side():
    # Mirror of acceptance repro Case 4: an explicit structured SHORT must
    # survive the real PM -> state projection (not a hand-injected dict).
    d = PortfolioDecision(
        rating="Sell",
        executive_summary="Offline fixture: explicitly propose a short position.",
        investment_thesis="Offline acceptance fixture; no market recommendation.",
        price_target=90.0,
        confidence=0.7,
        desired_side="SHORT",
    )
    fields = _decision_fields_dict(d)
    assert fields["desired_side"] == "SHORT"
    # Replay the graph's real side-selection expression (trading_graph.py,
    # read-only): the explicit side wins where the legacy rating mapping
    # alone could never produce SHORT (Sell maps to CLOSE).
    explicit_side = str(fields.get("desired_side") or "").upper()
    legacy_intent = desired_side_from_decision(fields)
    side = (
        explicit_side
        if explicit_side in ("LONG", "SHORT", "FLAT")
        else ("LONG" if legacy_intent == "LONG" else "FLAT")
    )
    assert side == "SHORT"
    # Byte-identical contract: a PM that omits desired_side keeps the key
    # absent from the serialized state JSON.
    no_side = PortfolioDecision(
        rating="Sell",
        executive_summary="Offline fixture: explicitly propose a short position.",
        investment_thesis="Offline acceptance fixture; no market recommendation.",
        price_target=90.0,
        confidence=0.7,
    )
    assert "desired_side" not in _decision_fields_dict(no_side)


@pytest.mark.unit
def test_probabilities_bounds_enforced():
    with pytest.raises(ValueError):
        OutcomeProbabilities(bull=1.5, neutral=0.2, bear=0.1)
