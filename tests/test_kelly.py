"""Pins for the pure Kelly-sizing helpers (yialpha/risk/kelly.py).

The prior review verified these contracts by reading but never pinned them:

* a negative-expectancy input clamps to zero — sizing is long-only, a negative
  edge can never produce a negative (short) size;
* the Sell rating band is ``(0.0, 0.0)`` so Sell sizes to exactly zero no
  matter how good the history is (the clamp short-circuits before ``p`` is
  even resolved);
* Hold sizes to zero on a negative edge and to at most its 3% band cap on a
  strong edge;
* conviction monotonicity — with identical history, a stronger rating yields a
  size >= the weaker rating's (confidence blend AND the band clip are both
  ordered);
* the ``kelly_fraction_mult`` multiplier (default 0.25 = quarter-Kelly) scales
  the raw Kelly linearly until the rating band clips it.

Everything here is a pure function: no config, no env, no LLM.
"""

import math

import pytest

from yialpha.risk.kelly import (
    RATING_TO_BAND,
    RATING_TO_CONFIDENCE,
    bayesian_win_rate,
    kelly_fraction,
    kelly_sizing,
)

# A band map with the caps blown open isolates the raw Kelly math from the
# band clipping (used to observe the multiplier and p-resolution contracts).
_WIDE_BANDS = dict.fromkeys(RATING_TO_BAND, (0.0, 100.0))

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# kelly_fraction: the raw formula
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_kelly_fraction_positive_edge_known_values():
    assert kelly_fraction(0.6, 1.0, 1.0) == pytest.approx(0.2)
    # b = avg_win/avg_loss = 2 -> f* = (2*0.6 - 0.4)/2 = 0.4
    assert kelly_fraction(0.6, 2.0, 1.0) == pytest.approx(0.4)
    assert kelly_fraction(0.9, 1.0, 1.0) == pytest.approx(0.8)


@pytest.mark.unit
def test_kelly_fraction_negative_edge_clamps_to_zero():
    """A negative-expectancy input must yield 0.0 — never a negative (short) size."""
    # Symmetric payoff, p < 0.5.
    assert kelly_fraction(0.2, 1.0, 1.0) == 0.0
    # Asymmetric: losses twice the wins -> b = 0.5, f* = (0.25-0.5)/0.5 < 0.
    assert kelly_fraction(0.5, 1.0, 2.0) == 0.0
    assert kelly_fraction(0.4, 0.5, 1.0) == 0.0
    # Zero edge exactly at p*b == q is the clamp boundary: 0.0, not negative.
    assert kelly_fraction(0.5, 1.0, 1.0) == 0.0


@pytest.mark.unit
def test_kelly_fraction_monotone_in_win_rate():
    prev = kelly_fraction(0.30, 2.0, 1.0)
    for p in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.80):
        current = kelly_fraction(p, 2.0, 1.0)
        assert current >= prev
        prev = current


@pytest.mark.unit
def test_kelly_fraction_degenerate_inputs():
    # Non-finite p -> 0.0 (guard before the formula).
    assert kelly_fraction(_NAN, 1.0, 1.0) == 0.0
    assert kelly_fraction(float("inf"), 1.0, 1.0) == 0.0
    # p clamped into [0, 1]: >1 acts as a sure thing, <0 as a sure loss.
    assert kelly_fraction(1.5, 1.0, 1.0) == pytest.approx(1.0)
    assert kelly_fraction(-2.0, 1.0, 1.0) == 0.0
    # avg_loss <= 0 (missing magnitude) -> symmetric payoff b = 1, even when
    # avg_win would otherwise imply a different ratio.
    assert kelly_fraction(0.6, 5.0, 0.0) == pytest.approx(0.2)
    assert kelly_fraction(0.6, 5.0, -3.0) == pytest.approx(0.2)
    # Non-finite avg_win/avg_loss fall back to 1.0 (symmetric).
    assert kelly_fraction(0.6, _NAN, 1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.6, None, 1.0) == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# bayesian_win_rate: the p resolution for historical stats
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_bayesian_win_rate_smoothing():
    # No observations at all: exactly the uniform-prior mean 0.5.
    assert bayesian_win_rate(0, 0) == 0.5
    # Beta(1+10, 1+0) mean = 11/12 — smoothed away from the raw 100%.
    assert bayesian_win_rate(10, 0) == pytest.approx(11.0 / 12.0)
    assert bayesian_win_rate(1, 1) == pytest.approx(0.5)
    assert bayesian_win_rate(0, 100) == pytest.approx(1.0 / 102.0)
    # Negative counts are clamped to zero, never into a malformed posterior.
    assert bayesian_win_rate(-5, 1) == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# kelly_sizing: rating bands + confidence blend
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sell_band_sizes_to_exactly_zero():
    """Sell never takes a long position: the (0.0, 0.0) band short-circuits to
    0.0 before p is resolved, so even a perfect history cannot size it."""
    assert kelly_sizing("Sell") == 0.0
    assert kelly_sizing("Sell", win_rate=0.99) == 0.0
    assert kelly_sizing(
        "Sell", wins=100, losses=0, avg_win=3.0, avg_loss=1.0,
        kelly_fraction_mult=1.0,
    ) == 0.0
    # The short-circuit is generic: ANY band with high <= 0.0 returns 0.0
    # regardless of the rating's confidence.
    assert kelly_sizing(
        "Hold", wins=100, losses=0, band_map={"Hold": (0.0, 0.0)},
    ) == 0.0


@pytest.mark.unit
def test_hold_sizes_zero_on_negative_edge_and_caps_at_band():
    """Hold's contract: zero on a negative edge, at most its 3% band cap."""
    # Negative edge: raw Kelly is 0 and Hold's floor is 0.0 -> exactly 0.0.
    assert kelly_sizing("Hold", win_rate=0.1) == 0.0
    assert kelly_sizing("Hold", wins=0, losses=100) == 0.0
    # Strong edge: raw Kelly (0.0837 here) exceeds the 0.03 cap -> clipped.
    assert kelly_sizing(
        "Hold", wins=90, losses=10, avg_win=3.0, avg_loss=1.0,
    ) == pytest.approx(RATING_TO_BAND["Hold"][1])


@pytest.mark.unit
def test_negative_edge_never_yields_negative_size():
    """Across every rating and a grid of nasty inputs the output stays in
    [0, 1]: the negative-edge clamp plus the band clip make short/negative
    sizes structurally impossible.

    Note the documented band nuance: a Buy-band rating with a NEGATIVE edge
    still floors at 0.05 (the band lifts raw Kelly 0 up to the tier minimum)
    — the guarantee is "never negative", not "negative edge => 0" at the
    sizing level (that clamp is exact at the kelly_fraction level).
    """
    ratings = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
    win_rates = (0.0, 0.1, 0.5, 0.9, 1.0, _NAN, 5.0)
    payoff_pairs = (
        (1.0, 1.0), (1.0, 2.0), (0.5, 1.0), (3.0, 1.0),
        (1.0, 0.0), (0.0, 1.0), (_NAN, _NAN),
    )
    for rating in ratings:
        for p in win_rates:
            for avg_win, avg_loss in payoff_pairs:
                size = kelly_sizing(rating, win_rate=p, avg_win=avg_win,
                                    avg_loss=avg_loss)
                assert 0.0 <= size <= 1.0, (rating, p, avg_win, avg_loss, size)
        # Buy with a hard negative edge floors at the band minimum, not below 0.
        assert kelly_sizing("Buy", win_rate=0.05) == pytest.approx(RATING_TO_BAND["Buy"][0])


@pytest.mark.unit
@pytest.mark.parametrize(
    "history",
    [
        # Strong edge: every tier clips at its own cap.
        {"wins": 90, "losses": 10, "avg_win": 3.0, "avg_loss": 1.0},
        # Break-even history: raw Kelly is 0, only band floors survive.
        {"wins": 50, "losses": 50, "avg_win": 1.0, "avg_loss": 1.0},
        # Terrible history: everything raw is clamped to 0.
        {"wins": 0, "losses": 100, "avg_win": 1.0, "avg_loss": 1.0},
        # Neutral scalar win rate, no counts.
        {"win_rate": 0.55},
    ],
    ids=["strong", "breakeven", "terrible", "neutral"],
)
def test_conviction_monotonicity_identical_history(history):
    """Identical history: size(Buy) >= size(Overweight) >= size(Hold) >=
    size(Underweight) >= size(Sell). Both the confidence blend (p_eff = p *
    confidence, conf ordered) and the bands (ordered, non-overlapping) respect
    the conviction ordering, so the composition is monotone end to end."""
    chain = []
    for rating in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
        counts = {
            k: history[k] for k in ("wins", "losses") if k in history
        } or {}
        chain.append(kelly_sizing(rating, **counts,
                                  avg_win=history.get("avg_win", 1.0),
                                  avg_loss=history.get("avg_loss", 1.0)))
    assert chain == sorted(chain, reverse=True), chain
    # The task's headline pin, explicitly: Buy >= Overweight with the same
    # history (strict > whenever the raw Kelly is not already zero for both).
    assert chain[0] >= chain[1]


@pytest.mark.unit
def test_rating_table_structural_invariants():
    """The wiring (risk/manager.py) imports RATING_TO_CONFIDENCE as the rating
    vocabulary; the two tables must agree on keys and stay conviction-ordered."""
    assert set(RATING_TO_CONFIDENCE) == set(RATING_TO_BAND)
    assert set(RATING_TO_CONFIDENCE) == {
        "Buy", "Overweight", "Hold", "Underweight", "Sell",
    }
    order = ["Buy", "Overweight", "Hold", "Underweight", "Sell"]
    confs = [RATING_TO_CONFIDENCE[r] for r in order]
    assert confs == sorted(confs, reverse=True)
    # Bands are conviction-ordered: each weaker tier's cap sits at or below
    # the stronger tier's cap, and its floor never sits above the stronger
    # tier's floor. (The bottom tiers SHARE the 0.0 floor — Hold and
    # Underweight overlap there, so full disjointness does not hold.)
    for stronger, weaker in zip(order, order[1:], strict=False):
        s_lo, s_hi = RATING_TO_BAND[stronger]
        w_lo, w_hi = RATING_TO_BAND[weaker]
        assert 0.0 <= s_lo <= s_hi <= 1.0
        assert 0.0 <= w_lo <= w_hi <= 1.0
        assert w_hi <= s_hi
        assert w_lo <= s_lo
    # Sell's band is exactly the zero band.
    assert RATING_TO_BAND["Sell"] == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# kelly_fraction_mult: the quarter-Kelly multiplier
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_multiplier_scales_raw_kelly_linearly_below_the_band():
    """With the band caps opened up, size scales linearly in
    kelly_fraction_mult: quarter -> half -> full Kelly is a clean 1x/2x/4x."""
    common = {
        "win_rate": 0.7, "avg_win": 1.5, "avg_loss": 1.0, "band_map": _WIDE_BANDS,
    }
    quarter = kelly_sizing("Buy", kelly_fraction_mult=0.25, **common)
    half = kelly_sizing("Buy", kelly_fraction_mult=0.5, **common)
    full = kelly_sizing("Buy", kelly_fraction_mult=1.0, **common)
    zero = kelly_sizing("Buy", kelly_fraction_mult=0.0, **common)
    # Raw Kelly for p_eff = 0.7*0.9 = 0.63, b = 1.5:
    # f* = (1.5*0.63 - 0.37)/1.5 = 0.383333...
    assert quarter == pytest.approx(0.575 / 1.5 * 0.25, rel=1e-9)
    assert half == pytest.approx(2.0 * quarter, rel=1e-9)
    assert full == pytest.approx(4.0 * quarter, rel=1e-9)
    assert zero == 0.0
    # The default multiplier IS quarter-Kelly.
    assert kelly_sizing("Buy", **common) == pytest.approx(quarter)


@pytest.mark.unit
def test_multiplier_default_and_degenerate_fallbacks():
    common = {"win_rate": 0.7, "avg_win": 1.5, "avg_loss": 1.0, "band_map": _WIDE_BANDS}
    # Omitted and None both mean the documented 0.25 default.
    assert kelly_sizing("Buy", **common) == kelly_sizing(
        "Buy", kelly_fraction_mult=0.25, **common
    )
    assert kelly_sizing("Buy", kelly_fraction_mult=None, **common) == kelly_sizing(
        "Buy", kelly_fraction_mult=0.25, **common
    )
    # A non-finite multiplier falls back to 0.25 rather than poisoning the size.
    assert kelly_sizing("Buy", kelly_fraction_mult=_NAN, **common) == pytest.approx(
        kelly_sizing("Buy", kelly_fraction_mult=0.25, **common)
    )


@pytest.mark.unit
def test_multiplier_monotone_under_default_bands():
    """A larger multiplier can only raise (or leave) the clipped size: the raw
    f is linear in mult and the band clip is monotone in f."""
    prev = None
    for mult in (0.0, 0.1, 0.25, 0.5, 1.0, 4.0):
        size = kelly_sizing(
            "Buy", wins=90, losses=10, avg_win=3.0, avg_loss=1.0,
            kelly_fraction_mult=mult,
        )
        if prev is not None:
            assert size >= prev
        prev = size


# --------------------------------------------------------------------------- #
# p resolution order + unknown-rating fallback
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_wins_losses_take_precedence_over_win_rate():
    """Resolution order: both counts present -> Bayesian smoothing wins and
    the scalar win_rate is ignored entirely."""
    common = {"avg_win": 1.0, "avg_loss": 1.0, "band_map": _WIDE_BANDS}
    with_counts = kelly_sizing("Buy", wins=60, losses=40, **common)
    without_rate = kelly_sizing("Buy", wins=60, losses=40, win_rate=0.99, **common)
    assert with_counts == without_rate
    # Beta(1, 1) prior: p = (1+60)/(1+60+1+40) = 61/102, p_eff = 0.9*p and
    # f = 2*p_eff - 1 (symmetric payoff), quartered.
    assert with_counts == pytest.approx((2.0 * 0.9 * 61.0 / 102.0 - 1.0) * 0.25,
                                        rel=1e-9)
    # The ignored win_rate=0.99 would have produced ~0.196 — different path.
    assert kelly_sizing("Buy", win_rate=0.99, **common) == pytest.approx(
        (2.0 * 0.9 * 0.99 - 1.0) * 0.25, rel=1e-9
    )


@pytest.mark.unit
def test_no_history_falls_back_to_rating_confidence():
    """Neither counts nor win_rate: p IS the rating confidence (Buy 0.9), so
    p_eff = 0.81 and the quarter-Kelly size is 0.62 * 0.25."""
    size = kelly_sizing("Buy", avg_win=1.0, avg_loss=1.0, band_map=_WIDE_BANDS)
    assert size == pytest.approx((2.0 * 0.9 * 0.9 - 1.0) * 0.25, rel=1e-9)
    # A non-finite win_rate takes the same confidence fallback.
    assert kelly_sizing(
        "Buy", win_rate=_NAN, avg_win=1.0, avg_loss=1.0, band_map=_WIDE_BANDS,
    ) == pytest.approx(size)


@pytest.mark.unit
def test_unknown_rating_falls_back_to_hold_band_with_half_confidence():
    """A stray LLM string must not break sizing: unknown ratings use the Hold
    band (0.0, 0.03) with confidence 0.5 — no raise, output in band."""
    weird = "DEFINITELY-BUY?!?"
    size = kelly_sizing(weird, wins=90, losses=10, avg_win=3.0, avg_loss=1.0)
    assert 0.0 <= size <= 0.03
    # Same math as an explicit Hold with confidence 0.5 and the Hold band.
    equivalent = kelly_sizing(
        "Hold", wins=90, losses=10, avg_win=3.0, avg_loss=1.0,
        confidence_map={"Hold": 0.5}, band_map={"Hold": (0.0, 0.03)},
    )
    assert size == pytest.approx(equivalent)
    assert not math.isnan(size)


@pytest.mark.unit
def test_custom_confidence_and_band_maps_drive_the_output():
    """The maps are injectable: a confidence_map override alone changes the
    blend, a band_map override alone changes the clip."""
    base = {"wins": 90, "losses": 10, "avg_win": 3.0, "avg_loss": 1.0}
    default = kelly_sizing("Buy", **base)
    confident = kelly_sizing("Buy", confidence_map={**RATING_TO_CONFIDENCE,
                                                    "Buy": 1.0}, **base)
    # Raw quarter-Kelly here is ~0.187 (91/101 smoothed x 0.9 blend, b=3).
    rebanded = kelly_sizing("Buy", band_map={"Buy": (0.0, 0.1)}, **base)
    assert default == pytest.approx(0.08)          # capped by the default band
    assert confident == pytest.approx(0.08)        # still capped
    assert rebanded == pytest.approx(0.1)          # tighter cap bites earlier
