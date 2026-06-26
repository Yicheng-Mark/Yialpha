"""Information-Coefficient (IC) based indicator pruning for YiAgents.

Phase 2c of the roadmap: the market analyst computes a fixed battery of
technical indicators but never validates which ones actually predict forward
returns. This module ranks each indicator by its rolling *Information
Coefficient* (Spearman rank correlation between the indicator value and the
realized forward return) and prunes indicators whose predictive power
persistently collapses.

The pruning rule encoded here is the roadmap rule:

    "drop any indicator whose rolling 60-day IC stays below 0.03
     for 30 consecutive days"

The module is pure math: no network, no LLM. ``numpy`` and ``pandas`` are the
only hard numeric dependencies. ``scipy`` is *optional* -- when
``scipy.stats.spearmanr`` is importable it is used, otherwise Spearman rank
correlation is computed manually (rank both arrays with ``pandas.Series.rank``
then take the Pearson correlation of the ranks).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional scipy import -- kept optional so the module works without it.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only by environment
    from scipy.stats import spearmanr as _scipy_spearmanr

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only by environment
    _scipy_spearmanr = None
    _HAVE_SCIPY = False


# Minimum number of finite paired observations required to trust an IC value.
_MIN_PAIRS = 5

# Type alias for the array-likes we accept as 1-D factor / return inputs.
ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def _as_clean_series(x: ArrayLike) -> pd.Series:
    """Coerce an array-like into a float pandas Series (no index assumption)."""
    return pd.Series(np.asarray(x, dtype=float))


def _spearman_from_ranks(fr: pd.Series, rr: pd.Series) -> Optional[float]:
    """Pearson correlation of two already-ranked series.

    Returns ``None`` when either series has zero variance (the correlation is
    undefined) or the inputs are empty.
    """
    if fr.empty:
        return None
    std_f = float(fr.std(ddof=0))
    std_r = float(rr.std(ddof=0))
    if std_f == 0.0 or std_r == 0.0:
        return None
    # Pearson on ranks: population covariance / (pop std_f * pop std_r).
    # mean of ranks is identical for both when lengths match, so centering is
    # equivalent for the covariance numerator -- use ddof=0 consistently.
    cov = float(fr.cov(rr, ddof=0))
    return cov / (std_f * std_r)


def information_coefficient(factor: ArrayLike, forward_returns: ArrayLike) -> Optional[float]:
    """Spearman rank IC between a factor series and forward returns.

    Inputs are equal-length 1-D array-likes (lists / np arrays / pd.Series).
    Returns IC in ``[-1, 1]``, or ``None`` if fewer than ~5 finite paired
    observations remain after pairwise dropping NaN/inf, or if either input has
    zero variance. NaNs/inf in either input are dropped *pairwise* before
    ranking.

    Raises ``ValueError`` (with a clear message) when the two inputs differ in
    length.
    """
    f = _as_clean_series(factor)
    r = _as_clean_series(forward_returns)

    if len(f) != len(r):
        raise ValueError(
            "factor and forward_returns must have equal length; got "
            f"{len(f)} and {len(r)}"
        )

    if len(f) == 0:
        return None

    # Pairwise drop of non-finite observations.
    finite = np.isfinite(f.values) & np.isfinite(r.values)
    f_clean = f[finite]
    r_clean = r[finite]

    if len(f_clean) < _MIN_PAIRS:
        return None

    if _HAVE_SCIPY and _scipy_spearmanr is not None:
        try:
            rho, _p = _scipy_spearmanr(f_clean.values, r_clean.values)
        except Exception:
            rho = _spearman_from_ranks(
                f_clean.rank(), r_clean.rank()
            )
        # spearmanr can return NaN on zero-variance input -> normalize to None.
        if rho is None or not np.isfinite(rho):
            return None
        # Defensive clamp into [-1, 1] (floating noise can push slightly out).
        return float(max(-1.0, min(1.0, rho)))

    return _spearman_from_ranks(f_clean.rank(), r_clean.rank())


def rolling_ic(
    factor: ArrayLike,
    forward_returns: ArrayLike,
    window: int = 60,
) -> "pd.Series":
    """Rolling Spearman IC over ``window``-sized trailing windows.

    Inputs are equal-length Series indexed identically (e.g. by date). The
    output is a ``pd.Series`` of IC values, indexed by the *last* date of each
    window (trailing-window alignment), with ``NaN`` where the window lacks
    enough finite paired data.

    Raises ``ValueError`` when the two inputs differ in length. The output
    shares the index of ``factor`` (which must match that of
    ``forward_returns``); positions before the first complete window are
    ``NaN``.
    """
    f = pd.Series(np.asarray(factor, dtype=float))
    r = pd.Series(np.asarray(forward_returns, dtype=float))

    if len(f) != len(r):
        raise ValueError(
            "factor and forward_returns must have equal length; got "
            f"{len(f)} and {len(r)}"
        )

    if len(f) == 0:
        return pd.Series(dtype=float)

    # Preserve a meaningful index even when callers pass plain lists: default
    # to a RangeIndex aligned with the inputs.
    if isinstance(factor, pd.Series):
        index = factor.index
    elif isinstance(forward_returns, pd.Series):
        index = forward_returns.index
    else:
        index = pd.RangeIndex(len(f))
    out = pd.Series(np.nan, index=index, dtype=float)

    if window <= 0:
        return out

    n = len(f)
    for end in range(n):
        start = end - window + 1
        if start < 0:
            continue
        ic = information_coefficient(f.iloc[start : end + 1], r.iloc[start : end + 1])
        out.iloc[end] = ic
    return out


def consecutive_below_threshold(
    ic_series: ArrayLike,
    threshold: float = 0.03,
    min_consecutive: int = 30,
) -> int:
    """Longest run of consecutive *finite* IC values with ``abs(IC) < threshold``.

    Used by the pruning rule. ``min_consecutive`` is informational only here
    (it does not truncate the result); the function returns the full longest
    run length, and callers compare it against ``min_consecutive``. NaN and
    non-finite values break a run (they are not "below threshold" -- they are
    simply missing). Returns ``0`` if no such run exists.

    Parameters
    ----------
    ic_series : array-like of IC values (may contain NaN).
    threshold : positive abs-IC level below which an indicator is "useless".
    min_consecutive : the roadmap run length (kept for API symmetry; the
        returned value is the true longest run regardless of this argument).
    """
    s = pd.Series(np.asarray(ic_series, dtype=float))

    best = 0
    run = 0
    for v in s.values:
        if np.isfinite(v) and abs(float(v)) < threshold:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def prune_indicators(
    ic_by_indicator: dict,
    min_abs_ic: float = 0.03,
    min_consecutive_days: int = 30,
    min_observation_windows: int = 30,
) -> dict:
    """Apply the roadmap pruning rule to a ``{indicator_name: rolling_ic_series}`` map.

    Rule: an indicator is **PRUNED** if it has at least
    ``min_observation_windows`` finite IC observations AND its longest
    consecutive run of ``abs(IC) < min_abs_ic`` reaches
    ``min_consecutive_days``. Otherwise it is **KEPT**.

    Indicators with too little data are KEPT (not enough evidence to prune).

    Returns ``{"keep": [...], "prune": [...]}`` with indicator names, each
    list sorted for deterministic output.
    """
    keep: list = []
    prune: list = []

    for name, series in ic_by_indicator.items():
        s = pd.Series(np.asarray(series, dtype=float))
        finite_count = int(np.isfinite(s.values).sum())

        # Too little data -> not enough evidence to prune.
        if finite_count < min_observation_windows:
            keep.append(name)
            continue

        longest_low_run = consecutive_below_threshold(
            s, threshold=min_abs_ic, min_consecutive=min_consecutive_days
        )

        if longest_low_run >= min_consecutive_days:
            prune.append(name)
        else:
            keep.append(name)

    return {"keep": sorted(keep), "prune": sorted(prune)}


def build_ic_report(
    pruning_result: dict,
    ic_by_indicator: dict,
) -> str:
    """Render a short markdown summary of which indicators survived IC pruning.

    For each KEPT indicator the mean absolute IC over its finite observations
    is shown; indicators whose rolling-IC series carries too few finite values
    to be prunable (fewer than 30) are noted as *insufficient data*. PRUNED
    indicators are listed by name.

    Parameters
    ----------
    pruning_result : ``{"keep": [...], "prune": [...]}`` from ``prune_indicators``.
    ic_by_indicator : the underlying ``{name: rolling_ic_series}`` map.
    """
    keep = sorted(pruning_result.get("keep", []))
    prune = sorted(pruning_result.get("prune", []))

    lines: list = []
    lines.append("# IC Indicator Pruning Report")
    lines.append("")
    lines.append(
        f"- **Kept**: {len(keep)} indicator(s)"
        + (f" ({', '.join(keep)})" if keep else "")
    )
    lines.append(
        f"- **Pruned**: {len(prune)} indicator(s)"
        + (f" ({', '.join(prune)})" if prune else "")
    )
    lines.append("")

    lines.append("## Kept indicators")
    if not keep:
        lines.append("_None._")
    else:
        lines.append("")
        lines.append("| Indicator | Mean |IC| | Note |")
        lines.append("|---|---|---|")
        for name in keep:
            s = pd.Series(np.asarray(ic_by_indicator.get(name, []), dtype=float))
            finite = s[np.isfinite(s.values)]
            if len(finite) < 30:
                note = "insufficient data"
                mean_abs = "n/a"
            else:
                mean_abs = f"{float(finite.abs().mean()):.3f}"
                note = "survived"
            lines.append(f"| {name} | {mean_abs} | {note} |")

    lines.append("")
    lines.append("## Pruned indicators")
    if not prune:
        lines.append("_None._")
    else:
        for name in prune:
            s = pd.Series(np.asarray(ic_by_indicator.get(name, []), dtype=float))
            finite = s[np.isfinite(s.values)]
            mean_abs = (
                f"{float(finite.abs().mean()):.3f}" if len(finite) else "n/a"
            )
            lines.append(f"- **{name}** (mean |IC| {mean_abs})")

    lines.append("")
    return "\n".join(lines)
