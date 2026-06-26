"""Unit tests for ``yiagents.backtest.ic`` (IC-based indicator pruning).

Pure numpy/pandas, no network, no LLM, scipy optional. Marked
``@pytest.mark.unit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yiagents.backtest.ic import (
    build_ic_report,
    consecutive_below_threshold,
    information_coefficient,
    prune_indicators,
    rolling_ic,
)


# ---------------------------------------------------------------------------
# information_coefficient
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ic_perfectly_monotonic_is_near_one():
    factor = list(range(1, 9))
    returns = [2 * x for x in factor]  # perfectly monotonic with factor
    ic = information_coefficient(factor, returns)
    assert ic is not None
    assert ic >= 0.9


@pytest.mark.unit
def test_ic_anticorrelated_is_near_minus_one():
    factor = list(range(1, 9))
    returns = [-2 * x for x in factor]
    ic = information_coefficient(factor, returns)
    assert ic is not None
    assert ic <= -0.9


@pytest.mark.unit
def test_ic_random_is_small():
    # Fixed-seed "random" array with no real rank relationship to the factor.
    rng = np.random.default_rng(42)
    factor = np.arange(1, 21, dtype=float)
    returns = rng.permutation(20) * 1.0
    ic = information_coefficient(factor, returns)
    assert ic is not None
    assert abs(ic) < 0.5


@pytest.mark.unit
def test_ic_too_few_finite_pairs_returns_none():
    # Only 3 finite paired observations -> below the ~5 threshold.
    factor = [1.0, 2.0, 3.0, np.nan, np.nan]
    returns = [2.0, 4.0, 6.0, np.nan, np.nan]
    assert information_coefficient(factor, returns) is None


@pytest.mark.unit
def test_ic_length_mismatch_raises():
    with pytest.raises(ValueError):
        information_coefficient([1, 2, 3], [1, 2])


@pytest.mark.unit
def test_ic_drops_nonfinite_pairwise():
    # 6 finite pairs remain after dropping the inf pair -> valid IC near 1.
    factor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf]
    returns = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    ic = information_coefficient(factor, returns)
    assert ic is not None
    assert ic >= 0.9


@pytest.mark.unit
def test_ic_zero_variance_returns_none():
    # Constant factor -> zero variance -> None.
    assert information_coefficient([5.0] * 8, list(range(8))) is None


# ---------------------------------------------------------------------------
# rolling_ic
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rolling_ic_drops_when_correlation_breaks():
    rng = np.random.default_rng(7)
    n = 120
    factor = np.arange(1, n + 1, dtype=float)

    # First half: returns are monotonic in factor (strong positive IC).
    # Second half: returns are a random shuffle (no stable rank relationship).
    first_half = factor[: n // 2] * 2.0
    second_half = rng.permutation(factor[n // 2 :]) * 2.0
    returns = np.concatenate([first_half, second_half])

    ric = rolling_ic(factor, returns, window=30)
    assert isinstance(ric, pd.Series)
    assert len(ric) == n

    # The trailing windows ending in the first half should be strongly positive.
    first_half_ics = ric.iloc[30 : n // 2].dropna()
    assert len(first_half_ics) > 0
    assert first_half_ics.mean() > 0.5

    # The trailing windows ending in the second half should be materially lower.
    second_half_ics = ric.iloc[n // 2 :].dropna()
    assert len(second_half_ics) > 0
    assert second_half_ics.mean() < first_half_ics.mean()


@pytest.mark.unit
def test_rolling_ic_empty_and_short_inputs():
    assert len(rolling_ic([], [], window=10)) == 0
    out = rolling_ic([1.0, 2.0], [2.0, 4.0], window=5)
    # Window larger than data -> all NaN.
    assert out.isna().all()


@pytest.mark.unit
def test_rolling_ic_length_mismatch_raises():
    with pytest.raises(ValueError):
        rolling_ic([1, 2, 3], [1, 2], window=2)


# ---------------------------------------------------------------------------
# consecutive_below_threshold
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_consecutive_below_threshold_known_run():
    # Hand-built: a run of 4 low values, then a spike, then 4 more low values.
    # Longest low run (abs < 0.03) is 4.
    ic_series = [0.01, 0.02, 0.01, -0.02, 0.5, 0.01, 0.02, 0.005, -0.01]
    run = consecutive_below_threshold(ic_series, threshold=0.03, min_consecutive=30)
    assert run == 4


@pytest.mark.unit
def test_consecutive_below_threshold_nan_breaks_run():
    # NaN and non-finite values break a run, they do not extend it.
    ic_series = [0.01, 0.01, np.nan, 0.01, 0.01, 0.01]
    run = consecutive_below_threshold(ic_series, threshold=0.03, min_consecutive=30)
    assert run == 3


@pytest.mark.unit
def test_consecutive_below_threshold_none_when_all_high():
    ic_series = [0.4, 0.5, 0.6, 0.4]
    assert consecutive_below_threshold(ic_series, threshold=0.03) == 0


# ---------------------------------------------------------------------------
# prune_indicators
# ---------------------------------------------------------------------------

def _ic_series(values, index=None):
    return pd.Series(values, index=index)


@pytest.mark.unit
def test_prune_indicators_classifies_each_case():
    # Indicator A: IC consistently strong (~0.4) -> KEPT.
    strong = _ic_series([0.4] * 80)
    # Indicator B: IC consistently tiny (~0.005) for 60+ windows -> PRUNED.
    weak = _ic_series([0.005] * 80)
    # Indicator C: only 5 IC values -> too little data -> KEPT.
    short = _ic_series([0.005] * 5)

    result = prune_indicators(
        {"A": strong, "B": weak, "C": short},
        min_abs_ic=0.03,
        min_consecutive_days=30,
        min_observation_windows=30,
    )

    assert "A" in result["keep"]
    assert "C" in result["keep"]
    assert "B" in result["prune"]
    assert "B" not in result["keep"]
    # No indicator appears in both buckets.
    assert set(result["keep"]).isdisjoint(set(result["prune"]))


@pytest.mark.unit
def test_prune_indicators_strong_but_short_run_is_kept():
    # IC dips below threshold for 15 consecutive days (below the 30 bar),
    # but otherwise strong -> KEPT (run does not reach min_consecutive_days).
    vals = [0.4] * 40 + [0.01] * 15 + [0.4] * 40
    result = prune_indicators(
        {"X": _ic_series(vals)},
        min_abs_ic=0.03,
        min_consecutive_days=30,
        min_observation_windows=30,
    )
    assert result["keep"] == ["X"]
    assert result["prune"] == []


@pytest.mark.unit
def test_prune_indicators_empty_map():
    result = prune_indicators({})
    assert result == {"keep": [], "prune": []}


# ---------------------------------------------------------------------------
# build_ic_report
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_ic_report_contains_names_and_structure():
    strong = _ic_series([0.4] * 80)
    weak = _ic_series([0.005] * 80)
    short = _ic_series([0.005] * 5)

    ic_by_indicator = {"A": strong, "B": weak, "C": short}
    pruning = prune_indicators(ic_by_indicator)

    md = build_ic_report(pruning, ic_by_indicator)

    assert isinstance(md, str)
    # Structure markers present (report uses past-tense "kept"/"pruned").
    assert "kept" in md.lower()
    assert "pruned" in md.lower()
    # Each indicator name appears.
    for name in ("A", "B", "C"):
        assert name in md
    # Insufficient-data annotation for the short series.
    assert "insufficient data" in md.lower()


@pytest.mark.unit
def test_build_ic_report_smoke_no_raise_on_empty():
    md = build_ic_report({"keep": [], "prune": []}, {})
    assert isinstance(md, str)
    assert "IC Indicator Pruning Report" in md
