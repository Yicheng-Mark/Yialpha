"""V2.4 batch K — signed perp math pins (direction-decoupled formulas).

The sign conventions are contract: funding mirrors the frozen outcome-compute
convention (a long PAYS positive funding, a short RECEIVES it — never abs),
the ATR stop mirrors across direction, liquidation is judged on MARK with an
inclusive trigger, and a same-bar stop+liquidation tie conservatively
resolves to the liquidation.
"""

from __future__ import annotations

import random

import pytest

from yialpha.risk.signed_math import (
    atr_stop,
    funding_pnl,
    liquidation_breach,
    position_sign,
    same_bar_resolution,
    signed_position,
)


@pytest.mark.unit
def test_position_sign_pins():
    assert position_sign("LONG") == 1
    assert position_sign("SHORT") == -1
    assert position_sign("FLAT") == 0


@pytest.mark.unit
def test_signed_position_contracts():
    assert signed_position("LONG", 5.0) == 5.0
    assert signed_position("SHORT", 5.0) == -5.0
    # FLAT is 0.0 regardless of the size passed — no magnitude to inherit.
    assert signed_position("FLAT", 5.0) == 0.0
    assert signed_position("FLAT", -3.0) == 0.0


@pytest.mark.unit
def test_funding_pnl_long_pays_positive_funding():
    # +0.01% funding on a 10,000 USDT notional long -> the long PAYS 1.0.
    assert funding_pnl(1, 10_000.0, 0.0001) == pytest.approx(-1.0)
    assert funding_pnl(1, 10_000.0, 0.0001) < 0.0


@pytest.mark.unit
def test_funding_pnl_short_receives_positive_funding():
    # The same positive rate is a CREDIT for the short — sign survives,
    # never abs()'d.
    assert funding_pnl(-1, 10_000.0, 0.0001) == pytest.approx(1.0)
    # Negative funding flips both sides (long receives, short pays).
    assert funding_pnl(1, 10_000.0, -0.0001) == pytest.approx(1.0)
    assert funding_pnl(-1, 10_000.0, -0.0001) == pytest.approx(-1.0)


@pytest.mark.unit
def test_funding_pnl_flat_sign_is_zero():
    assert funding_pnl(0, 10_000.0, 0.0001) == 0.0


@pytest.mark.unit
def test_atr_stop_mirror_symmetry_pinned():
    # entry 100, atr 2, mult 2 -> long 96 / short 104.
    assert atr_stop("LONG", 100.0, 2.0, 2.0) == pytest.approx(96.0)
    assert atr_stop("SHORT", 100.0, 2.0, 2.0) == pytest.approx(104.0)


@pytest.mark.unit
def test_atr_stop_none_conditions():
    assert atr_stop("FLAT", 100.0, 2.0, 2.0) is None
    assert atr_stop("LONG", 100.0, 0.0, 2.0) is None
    assert atr_stop("LONG", 100.0, -2.0, 2.0) is None
    assert atr_stop("LONG", 100.0, 2.0, 0.0) is None
    assert atr_stop("LONG", 100.0, 2.0, -1.0) is None
    for bad_atr in (float("nan"), float("inf")):
        assert atr_stop("LONG", 100.0, bad_atr, 2.0) is None
    assert atr_stop("LONG", float("nan"), 2.0, 2.0) is None


@pytest.mark.unit
def test_liquidation_breach_mirror_pinned():
    # Long: mark at/below the trigger is gone; just above survives.
    assert liquidation_breach("LONG", 79.9, 80.0) is True
    assert liquidation_breach("LONG", 80.0, 80.0) is True
    assert liquidation_breach("LONG", 80.1, 80.0) is False
    # Short mirrors: mark at/above the trigger is gone.
    assert liquidation_breach("SHORT", 120.1, 120.0) is True
    assert liquidation_breach("SHORT", 120.0, 120.0) is True
    assert liquidation_breach("SHORT", 119.9, 120.0) is False
    # FLAT is never in breach.
    assert liquidation_breach("FLAT", 0.0, 80.0) is False


@pytest.mark.unit
def test_same_bar_resolution_conservative():
    # Both tripped -> liquidation counted FIRST (conservative tie rule).
    assert same_bar_resolution(True, True) == "liquidation"
    assert same_bar_resolution(False, True) == "liquidation"
    assert same_bar_resolution(True, False) == "stop"
    assert same_bar_resolution(False, False) is None


@pytest.mark.unit
def test_property_atr_stop_mirror_invariant():
    rng = random.Random(20260903)
    for _ in range(50):
        entry = rng.uniform(1.0, 1000.0)
        atr = rng.uniform(0.01, 50.0)
        mult = rng.uniform(0.1, 5.0)
        long_stop = atr_stop("LONG", entry, atr, mult)
        short_stop = atr_stop("SHORT", entry, atr, mult)
        assert long_stop is not None
        assert short_stop is not None
        assert entry - long_stop == pytest.approx(short_stop - entry)
        assert long_stop < entry < short_stop
