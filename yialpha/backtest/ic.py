"""Information-Coefficient (IC) based indicator pruning for YiAlpha.

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

Beyond the level+persistence pruning rule, the 2026-08-15 expansion adds the
diagnostics that make IC evidence actionable: :func:`ic_decay` (how fast
predictive power fades with the forward horizon), :func:`quantile_spread`
(whether the factor is monotone across quantiles or only tail-effective) and
:func:`factor_turnover` (how churny the signal is to actually capture).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

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
ArrayLike: TypeAlias = "Sequence[float] | np.ndarray | pd.Series"


def _as_clean_series(x: ArrayLike) -> pd.Series:
    """Coerce an array-like into a float pandas Series (no index assumption)."""
    return pd.Series(np.asarray(x, dtype=float))


def _spearman_from_ranks(fr: pd.Series, rr: pd.Series) -> float | None:
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


def information_coefficient(factor: ArrayLike, forward_returns: ArrayLike) -> float | None:
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
) -> pd.Series:
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
    if window <= 0 or n < window:
        return out

    # Vectorized (2026-08-16): the per-window loop called
    # ``information_coefficient`` ~n times, re-ranking every window through
    # scipy — ~1,200 redundant rank passes per indicator on a 5y frame, times
    # the whole battery in the pruning CLI. The closed form computes ranks
    # per window via a strided window view and row-wise Pearson on them.
    # Pairwise-finite semantics are preserved by NaN-masking one input where
    # the other is non-finite before windowing; windows with too few valid
    # pairs stay NaN (the old None normalization), zero-variance windows
    # divide 0/0 -> NaN (scipy's NaN), and floating noise is clamped to
    # [-1, 1]. Windows containing TIED values fall back to the exact
    # per-window call — positional ranks are only exact for tie-free windows.
    fvals = f.values
    rvals = r.values
    bad = ~(np.isfinite(fvals) & np.isfinite(rvals))
    if bad.any():
        fvals = np.where(bad, np.nan, fvals)
        rvals = np.where(bad, np.nan, rvals)

    from numpy.lib.stride_tricks import sliding_window_view

    fw = sliding_window_view(fvals, window)
    rw = sliding_window_view(rvals, window)
    valid = np.isfinite(fw) & np.isfinite(rw)
    counts = valid.sum(axis=1)

    sf = np.sort(fw, axis=1)
    sr = np.sort(rw, axis=1)
    tie_rows = (
        (np.isfinite(sf[:, :-1]) & (sf[:, :-1] == sf[:, 1:])).any(axis=1)
        | (np.isfinite(sr[:, :-1]) & (sr[:, :-1] == sr[:, 1:])).any(axis=1)
    )

    def _positional_ranks(x):
        order = np.argsort(x, axis=1, kind="stable")  # NaN sorts last
        ranks = np.empty(x.shape, dtype=float)
        rows = np.arange(x.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, x.shape[1] + 1)[None, :]
        return ranks

    rank_f = _positional_ranks(fw)
    rank_r = _positional_ranks(rw)
    rank_f[~valid] = np.nan
    rank_r[~valid] = np.nan

    n_pairs = counts.astype(float)
    safe_n = np.maximum(n_pairs, 1.0)
    mean_f = np.where(valid, rank_f, 0.0).sum(axis=1) / safe_n
    mean_r = np.where(valid, rank_r, 0.0).sum(axis=1) / safe_n
    dev_f = np.where(valid, rank_f - mean_f[:, None], 0.0)
    dev_r = np.where(valid, rank_r - mean_r[:, None], 0.0)
    cov = (dev_f * dev_r).sum(axis=1)
    var_f = (dev_f * dev_f).sum(axis=1)
    var_r = (dev_r * dev_r).sum(axis=1)
    denom = np.sqrt(var_f * var_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where(denom > 0.0, cov / denom, np.nan)

    for i in np.nonzero(tie_rows & (counts >= _MIN_PAIRS))[0]:
        exact = information_coefficient(fw[i], rw[i])
        ic[i] = np.nan if exact is None else float(exact)

    ic[counts < _MIN_PAIRS] = np.nan
    ic = np.where(np.isfinite(ic), np.clip(ic, -1.0, 1.0), np.nan)
    out.iloc[window - 1:] = ic
    return out


def ic_decay(
    factor: ArrayLike,
    forward_returns_by_horizon: dict[str, ArrayLike],
) -> dict[str, float | None]:
    """Full-sample Spearman IC per forward horizon — the IC decay curve.

    ``forward_returns_by_horizon`` maps a horizon label (e.g. ``"1d"``,
    ``"10d"``) to an equal-length forward-return series; the result maps the
    same labels to :func:`information_coefficient` values. A curve decaying
    toward zero as the horizon grows marks a short-lived signal; one that
    holds up at 20d supports slower rebalancing. ``None`` where a horizon
    lacks enough finite pairs.

    Raises ``ValueError`` when any horizon series differs in length from
    ``factor``.
    """
    f = _as_clean_series(factor)
    return {
        label: information_coefficient(f, fwd)
        for label, fwd in forward_returns_by_horizon.items()
    }


def quantile_spread(
    factor: ArrayLike,
    forward_returns: ArrayLike,
    n_quantiles: int = 5,
) -> dict | None:
    """Mean forward return per factor quantile, the Q_hi−Q_lo spread, and
    monotonicity of the profile.

    Rows are bucketed by factor value into ``n_quantiles`` equal-count groups
    (Q1 = lowest factor values). Returns::

        {
            "quantile_means": [mean forward return of Q1..Qk],
            "spread": mean(Qk) − mean(Q1),
            "monotonic": True/False,
            "n_quantiles": k,
        }

    ``monotonic`` means the quantile means move in ONE direction across
    buckets (weakly, ties allowed) — the shape that makes a factor usable as
    a ranked long-short signal rather than only at its tails. ``duplicates``
    in the factor collapse buckets; fewer than 2 usable buckets or too few
    rows returns ``None``.

    Raises ``ValueError`` when the inputs differ in length or
    ``n_quantiles < 2``.
    """
    f = _as_clean_series(factor)
    r = _as_clean_series(forward_returns)
    if len(f) != len(r):
        raise ValueError(
            "factor and forward_returns must have equal length; got "
            f"{len(f)} and {len(r)}"
        )
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2; got {n_quantiles}")

    finite = np.isfinite(f.values) & np.isfinite(r.values)
    f_ok = f[finite]
    r_ok = r[finite]
    if len(f_ok) < n_quantiles:
        return None

    try:
        buckets = pd.qcut(f_ok, q=n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return None

    frame = pd.DataFrame({"q": np.asarray(buckets), "r": r_ok.values})
    means = frame.groupby("q")["r"].mean().sort_index()
    k = len(means)
    if k < 2:
        return None

    diffs = means.diff().dropna()
    monotonic = bool((diffs >= 0).all() or (diffs <= 0).all())
    return {
        "quantile_means": [float(m) for m in means],
        "spread": float(means.iloc[-1] - means.iloc[0]),
        "monotonic": monotonic,
        "n_quantiles": k,
    }


def factor_turnover(factor: ArrayLike) -> float | None:
    """Mean absolute rank change between consecutive rows, normalized to
    ``[0, 1]``.

    ``0.0`` means the row ranking never changes (a static factor); values
    approaching ``1.0`` mean the ranking reshuffles almost completely row to
    row — a high-churn signal whose realized IC is expensive to capture after
    costs. Non-finite rows are dropped before ranking. Returns ``None`` for
    fewer than 2 usable rows or a zero-variance factor (all ties — the
    ranking is meaningless).
    """
    s = _as_clean_series(factor)
    s = s[np.isfinite(s.values)]
    if len(s) < 2:
        return None
    ranks = s.rank()
    if int(ranks.nunique()) <= 1:
        return None
    denom = float(len(s) - 1)  # max possible |Δrank| between adjacent rows
    diffs = ranks.diff().dropna()
    return float(diffs.abs().mean() / denom)


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
