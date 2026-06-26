"""Quarter-Kelly position sizing scaled by analyst-rating confidence.

These helpers are pure functions: the decision-node wiring passes the
historical trade stats, the analyst rating and the Kelly parameters in.
Nothing here reads from the global config dict or any env var, so the
output is deterministic and unit-testable.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# Rating -> confidence multiplier applied to the historical win probability.
# Stronger ratings pull the effective win rate toward conviction; weaker ones
# pull it toward 0.5 / below so Kelly shrinks.
RATING_TO_CONFIDENCE: dict[str, float] = {
    "Buy": 0.90,
    "Overweight": 0.72,
    "Hold": 0.55,
    "Underweight": 0.35,
    "Sell": 0.12,
}

# Rating -> (min, max) target weight band the Kelly output is clipped to.
# Keeps each conviction tier in a sane position-size range regardless of
# what raw Kelly spits out.
RATING_TO_BAND: dict[str, tuple[float, float]] = {
    "Buy": (0.05, 0.08),
    "Overweight": (0.03, 0.05),
    "Hold": (0.0, 0.03),
    "Underweight": (0.0, 0.01),
    "Sell": (0.0, 0.0),
}


def bayesian_win_rate(
    wins: int,
    losses: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> float:
    """Beta(alpha+win, beta+loss) posterior mean.

    Smooths tiny samples toward 0.5 (a uniform Beta(1,1) prior). With no
    observations at all this returns exactly 0.5.
    """
    wins = max(int(wins), 0)
    losses = max(int(losses), 0)
    alpha = prior_alpha + wins
    beta = prior_beta + losses
    return float(alpha / (alpha + beta))


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly fraction ``f* = (b*p - q) / b``.

    ``b`` is the payoff ratio ``avg_win / |avg_loss|`` and ``q = 1 - p``.
    A non-positive Kelly (negative edge) is clamped to 0 — we never short
    via sizing. ``avg_loss`` is expected as a positive magnitude; if it is
    missing (``<= 0``) we treat ``b = 1`` (symmetric payoff).
    """
    p = float(win_rate)
    # Guard degenerate inputs: p outside [0,1] makes the formula meaningless.
    if not np.isfinite(p):
        return 0.0
    p = min(max(p, 0.0), 1.0)

    avg_win_f = float(avg_win) if avg_win is not None and np.isfinite(avg_win) else 1.0
    avg_loss_f = float(avg_loss) if avg_loss is not None and np.isfinite(avg_loss) else 1.0

    if avg_loss_f <= 0.0:
        b = 1.0
    else:
        b = avg_win_f / avg_loss_f
    if b <= 0.0:
        b = 1.0

    q = 1.0 - p
    f = (b * p - q) / b
    return float(max(f, 0.0))


def kelly_sizing(
    rating: str,
    win_rate: float | None = None,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
    wins: int | None = None,
    losses: int | None = None,
    kelly_fraction_mult: float = 0.25,
    confidence_map: Mapping[str, float] | None = None,
    band_map: Mapping[str, tuple[float, float]] | None = None,
) -> float:
    """Return a target portfolio weight in ``[0, 1]``.

    Resolution order for the effective win probability ``p``:

    1. If ``wins`` and ``losses`` are both provided, smooth the historical
       rate via :func:`bayesian_win_rate` (handles 0/0 gracefully).
    2. Else if ``win_rate`` is given, use it directly.
    3. Else fall back to the rating's confidence multiplier.

    The resolved ``p`` is then blended with the rating's directional
    conviction (``p_eff = p * confidence``) so a high win rate on a weak
    rating still produces a small position. The result is quarter-Kelly by
    default and clipped to the rating's target band.

    Unknown ratings fall back to the ``Hold`` band with confidence ``0.5``
    rather than raising, so a stray string from an LLM cannot break sizing.
    """
    conf = confidence_map if confidence_map is not None else RATING_TO_CONFIDENCE
    bands = band_map if band_map is not None else RATING_TO_BAND

    if rating not in conf:
        confidence = 0.5
        band = bands.get(rating, (0.0, 0.03))  # treat unknown as Hold band
    else:
        confidence = conf[rating]
        band = bands.get(rating, (0.0, 0.03))

    # Sell never takes a long position.
    low, high = band
    if high <= 0.0:
        return 0.0

    # Resolve effective p.
    if wins is not None and losses is not None:
        p = bayesian_win_rate(wins, losses)
    elif win_rate is not None:
        p = float(win_rate)
        if not np.isfinite(p):
            p = confidence
        else:
            p = min(max(p, 0.0), 1.0)
    else:
        p = confidence

    # Blend with the rating conviction.
    p_eff = p * confidence

    mult = float(kelly_fraction_mult) if kelly_fraction_mult is not None else 0.25
    if not np.isfinite(mult):
        mult = 0.25

    f = kelly_fraction(p_eff, avg_win, avg_loss) * mult
    if not np.isfinite(f):
        f = 0.0

    return float(min(max(f, low), high))
