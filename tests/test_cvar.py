"""Pins for the CVaR monitor (yialpha/risk/cvar.py).

Prior review verified but never pinned:

* empty / thin history is fail-SAFE: fewer than ``min_observations`` finite
  returns (default 30) returns the normal multiplier 1.0 — no de-risking on
  no evidence, and non-finite values do not count toward the quota;
* a tail breach beyond the configured threshold cuts the multiplier to the
  documented 0.5 (default) breached value; custom multiplier values pass
  through verbatim (including an exact 0.0 zero-weight hook);
* calm history leaves the multiplier at 1.0;
* the multiplier carries NO rating band of its own — it is purely
  returns-driven (the rating-aware layer is kelly_sizing; the risk manager
  multiplies the two: target = kelly_raw * breaker * cvar_mult);
* the breach comparison is strict: CVaR exactly equal to the threshold is
  NOT a breach.

Pure functions: returns are passed in; nothing reads config or env.
"""

import pytest

from yialpha.risk.cvar import cvar_position_multiplier, historical_cvar

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# historical_cvar: the tail estimator
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_historical_cvar_empty_inputs():
    assert historical_cvar([]) == 0.0
    assert historical_cvar(None) == 0.0
    # An all-non-finite sample filters down to empty -> 0.0, never nan/inf.
    assert historical_cvar([_NAN, float("inf"), float("-inf")]) == 0.0


@pytest.mark.unit
def test_historical_cvar_is_mean_of_worst_tail():
    # Float subtlety worth pinning: 1.0 - 0.95 == 0.050000000000000044 in
    # binary, so 100 observations at 95% give ceil(5.000000000000004) = SIX
    # tail observations, not five. The tail mean here averages the five -0.10
    # observations plus one +0.01: (-0.50 + 0.01) / 6.
    returns = [0.01] * 95 + [-0.10] * 5
    assert historical_cvar(returns, confidence=0.95) == pytest.approx(
        (-0.50 + 0.01) / 6.0
    )
    # 99% over 100 observations: 1.0-0.99 rounds UP, ceil -> 2-deep tail of
    # the two -0.10 observations -> exactly -0.10.
    assert historical_cvar(returns, confidence=0.99) == pytest.approx(-0.10)
    # Mixed tail: worst two of {-0.5, +0.01} average to -0.245.
    mixed = [0.01] * 30 + [-0.5]
    assert historical_cvar(mixed, confidence=0.95) == pytest.approx((-0.5 + 0.01) / 2.0)


@pytest.mark.unit
def test_historical_cvar_tail_count_rounds_up():
    """41 observations at 95% -> ceil(41 * 0.05) = 3 in the tail (floor would
    give 2 and a different mean): the discriminator between -0.02 and -0.025."""
    returns = [0.01] * 38 + [-0.01, -0.02, -0.03]
    cvar = historical_cvar(returns, confidence=0.95)
    assert cvar == pytest.approx((-0.01 - 0.02 - 0.03) / 3.0)
    assert cvar != pytest.approx((-0.02 - 0.03) / 2.0)


@pytest.mark.unit
def test_historical_cvar_tail_is_at_least_one_observation():
    """Even when (1 - confidence) * n < 1 the tail keeps one observation, so a
    single bad day is still captured."""
    assert historical_cvar([0.01, 0.02, 0.03], confidence=0.95) == pytest.approx(0.01)
    assert historical_cvar([-0.25], confidence=0.95) == pytest.approx(-0.25)


@pytest.mark.unit
def test_historical_cvar_filters_non_finite():
    """nan/inf/-inf are dropped BEFORE the tail selection: -inf must not
    poison the mean into -inf, nan must not propagate."""
    returns = [_NAN, float("inf"), float("-inf"), 0.01, 0.02, -0.5]
    assert historical_cvar(returns, confidence=0.95) == pytest.approx(-0.5)


@pytest.mark.unit
def test_historical_cvar_all_positive_sample_is_positive():
    """With no non-positive observation in the tail the mean is positive —
    the docstring's "never positive" guarantee is conditional on a
    non-positive tail observation existing."""
    assert historical_cvar([0.01] * 10, confidence=0.95) == pytest.approx(0.01)


@pytest.mark.unit
@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1, 0, 1])
def test_historical_cvar_rejects_degenerate_confidence(confidence):
    with pytest.raises(ValueError, match="open interval"):
        historical_cvar([0.01, -0.02], confidence=confidence)


@pytest.mark.unit
def test_historical_cvar_empty_short_circuits_before_confidence_check():
    """Guard order: the empty-input 0.0 return fires before confidence
    validation, so empty + degenerate confidence is 0.0, not a raise."""
    assert historical_cvar([], confidence=1.5) == 0.0


# --------------------------------------------------------------------------- #
# cvar_position_multiplier: the de-risk gate
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_empty_or_short_history_returns_normal_multiplier():
    """No evidence -> no de-risk: empty, None, and histories below
    min_observations (default 30) all return the normal multiplier, even
    when the few available returns are catastrophic."""
    assert cvar_position_multiplier([]) == 1.0
    assert cvar_position_multiplier(None) == 1.0
    assert cvar_position_multiplier([-0.9] * 29) == 1.0
    # Exactly at the boundary (30 finite) the gate IS evaluated.
    assert cvar_position_multiplier([-0.9] * 30) == 0.5


@pytest.mark.unit
def test_min_observations_counts_only_finite_values():
    """Non-finite returns don't count toward the 30-observation quota: 28
    finite catastrophes plus 12 NaNs stays fail-safe at 1.0; 30 finite ones
    plus 10 NaNs is evaluated and breaches."""
    diluted = [-0.9] * 28 + [_NAN] * 12
    assert cvar_position_multiplier(diluted) == 1.0
    evaluated = [-0.9] * 30 + [_NAN] * 10
    assert cvar_position_multiplier(evaluated) == 0.5


@pytest.mark.unit
def test_tail_breach_cuts_multiplier_to_documented_value():
    """60 observations whose tail (ceil(60*(1-0.95)) = 4 deep in float
    arithmetic) averages -10% breach the default -5% threshold -> the
    documented 0.5 de-risk multiplier."""
    returns = [0.01] * 56 + [-0.10] * 4
    assert cvar_position_multiplier(returns) == 0.5


@pytest.mark.unit
def test_calm_history_stays_at_normal_multiplier():
    returns = [0.01] * 55 + [-0.01] * 5
    assert cvar_position_multiplier(returns) == 1.0
    # Even a fully flat history is calm.
    assert cvar_position_multiplier([0.0] * 40) == 1.0


@pytest.mark.unit
def test_custom_threshold_moves_the_breach_line():
    """A tail mean of -3% is calm under the default -5% threshold but breaches
    an explicitly configured -2% threshold."""
    returns = [0.01] * 56 + [-0.03] * 4
    assert cvar_position_multiplier(returns) == 1.0
    assert cvar_position_multiplier(returns, breach_threshold=-0.02) == 0.5


@pytest.mark.unit
def test_breach_comparison_is_strict_at_the_boundary():
    """CVaR exactly equal to the threshold is NOT a breach (< is strict).
    Built with a binary-exact -0.0625 worst observation: 40 samples at 99%
    confidence put exactly that one value in the tail, so the CVaR equals the
    threshold bit-for-bit."""
    returns = [-0.0625] + [0.01] * 39
    calm = cvar_position_multiplier(
        returns, confidence=0.99, breach_threshold=-0.0625,
    )
    assert calm == 1.0
    breached = cvar_position_multiplier(
        returns, confidence=0.99, breach_threshold=-0.06,
    )
    assert breached == 0.5


@pytest.mark.unit
def test_custom_multipliers_pass_through_verbatim():
    returns_breach = [0.01] * 56 + [-0.10] * 4
    returns_calm = [0.01] * 55 + [-0.01] * 5
    assert cvar_position_multiplier(
        returns_breach, normal_multiplier=1.25, breached_multiplier=0.25,
    ) == 0.25
    assert cvar_position_multiplier(
        returns_calm, normal_multiplier=1.25, breached_multiplier=0.25,
    ) == 1.25


@pytest.mark.unit
def test_no_rating_band_encoded_zero_weight_comes_from_the_caller():
    """The multiplier is returns-driven only: there is no rating parameter
    and no Sell band inside it (that layer is kelly_sizing; the manager
    multiplies the two). A caller may encode a zero-weight policy by passing
    breached_multiplier=0.0, which passes through exactly."""
    returns_breach = [0.01] * 56 + [-0.10] * 4
    assert cvar_position_multiplier(returns_breach, breached_multiplier=0.0) == 0.0
    assert cvar_position_multiplier([], breached_multiplier=0.0) == 1.0
    # Same input -> same output on every call (purity, no hidden state).
    first = cvar_position_multiplier(returns_breach)
    second = cvar_position_multiplier(list(returns_breach))
    assert first == second == 0.5


@pytest.mark.unit
def test_degenerate_confidence_fails_safe_to_normal():
    """A confidence outside (0, 1) makes historical_cvar raise; the gate
    catches it and fails safe to the normal multiplier instead of raising."""
    returns = [0.01] * 56 + [-0.10] * 4
    assert cvar_position_multiplier(returns, confidence=1.0) == 1.0
    assert cvar_position_multiplier(returns, confidence=0.0) == 1.0
