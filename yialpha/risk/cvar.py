"""Conditional Value at Risk (CVaR) monitor.

CVaR is the expected loss in the worst tail of the return distribution.
The monitor turns a CVaR breach into a position-size multiplier so the
decision node can de-risk without rejecting a trade outright.

Pure functions: ``returns`` is passed in by the caller; nothing here reads
from the global config dict or any env var.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def historical_cvar(returns: Iterable[float], confidence: float = 0.95) -> float:
    """Conditional Value at Risk from a sample of periodic returns.

    Returns the mean of the worst ``(1 - confidence)`` tail of ``returns``
    — a negative number representing the expected tail loss. For a 95%
    confidence level that is the average of the worst 5% of observations.

    Guarantees:

    * Always returns a finite float (returns 0.0 for empty input).
    * The result is never positive when there is at least one non-positive
      observation in the tail, matching the "expected loss" interpretation.
    """
    arr = np.asarray(list(returns), dtype=float) if returns is not None else np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0

    if not (0.0 < confidence < 1.0):
        raise ValueError("historical_cvar: confidence must be in the open interval (0, 1)")

    # Number of observations in the tail. At least one so a single bad
    # observation is still captured.
    tail_count = int(np.ceil(arr.size * (1.0 - confidence)))
    if tail_count < 1:
        tail_count = 1
    if tail_count > arr.size:
        tail_count = arr.size

    worst = np.sort(arr)[:tail_count]
    return float(worst.mean())


def cvar_position_multiplier(
    returns: Iterable[float],
    confidence: float = 0.95,
    breach_threshold: float | None = None,
    normal_multiplier: float = 1.0,
    breached_multiplier: float = 0.5,
    min_observations: int = 30,
) -> float:
    """Return a position multiplier (1.0 normal, 0.5 de-risked on breach).

    * If fewer than ``min_observations`` finite returns are available the
      function is fail-safe: return ``normal_multiplier`` (never block a
      trade on thin data).
    * Otherwise compute CVaR; if it is worse (more negative) than
      ``breach_threshold`` return ``breached_multiplier``, else
      ``normal_multiplier``.

    ``breach_threshold`` defaults to ``-0.05`` (a 5% expected tail loss)
    when ``None``.
    """
    threshold = -0.05 if breach_threshold is None else float(breach_threshold)

    arr = np.asarray(list(returns), dtype=float) if returns is not None else np.array([], dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < min_observations:
        return float(normal_multiplier)

    try:
        cvar = historical_cvar(arr, confidence=confidence)
    except ValueError:
        # Degenerate confidence input — fail safe.
        return float(normal_multiplier)

    if cvar < threshold:
        return float(breached_multiplier)
    return float(normal_multiplier)
