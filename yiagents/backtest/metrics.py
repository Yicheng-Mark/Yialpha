"""Backtest performance metrics for the YiAgents backtest harness.

This module is pure math: given a chronological sequence of portfolio
mark-to-market (equity) values it computes the standard battery of
performance statistics -- total return, CAGR, annualized volatility, Sharpe,
Sortino, maximum drawdown, Calmar, the Deflated Sharpe Ratio of Bailey &
Lopez de Prado (2014), and an alpha versus an optional buy-and-hold
benchmark.

No external services are contacted. ``scipy`` is *optional*: when it is
installed its ``skew``/``kurtosis``/``norm`` are used, otherwise numpy/math
fallbacks keep the module fully functional. ``numpy`` (a transitive
dependency via pandas/stockstats) is the only hard numeric dependency.

Units convention
----------------
All return-based statistics are annualized using ``periods_per_year``
(default 252 daily trading periods). The periodic (per-observation) returns
are ``equity[1:] / equity[:-1] - 1``. Annualization multiplies the mean
return by ``periods_per_year`` and the standard deviation by
``sqrt(periods_per_year)``.

Deflated Sharpe Ratio (DSR)
---------------------------
DSR is the probability that the observed Sharpe is genuinely above the
multiple-testing "hurdle" Sharpe that would be expected as the best of
``n_trials`` independent strategies under the null (Bailey & Lopez de
Prado, 2014). To avoid unit confusion it is computed in per-observation
units:

    sr_per       = mean(returns) / std(returns)
    sr0_per      = z / sqrt(n)            # hurdle in per-obs units (0 if N==1)
    denom        = sqrt(1 - sk*sr_per + (ku*sr_per**2)/4)
    stat         = (sr_per - sr0_per) * sqrt(n - 1) / denom
    DSR          = norm.cdf(stat)         # in [0, 1]

where ``z`` is the expected maximum of ``N = n_trials`` standard normals
(built from the inverse normal CDF), ``sk`` is sample skew and ``ku`` is
sample excess kurtosis of the periodic returns. DSR > 0.5 means the
strategy clears the multiple-testing hurdle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy import -- kept optional so the module works without it.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only by environment
    from scipy.stats import kurtosis as _scipy_kurtosis, norm as _scipy_norm, skew as _scipy_skew

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only by environment
    _scipy_kurtosis = None
    _scipy_norm = None
    _scipy_skew = None
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Pure-numpy / pure-math fallbacks for the normal distribution moments.
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    """Standard normal CDF, preferring scipy; math.erf fallback otherwise."""
    if _HAVE_SCIPY and _scipy_norm is not None:
        return float(_scipy_norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (the ``ndtri`` function).

    scipy's ``norm.ppf`` is used when available; otherwise the Acklam
    rational approximation (relative error < 1.15e-9) is used as a
    pure-python fallback so the module has no hard scipy dependency.
    """
    if _HAVE_SCIPY and _scipy_norm is not None:
        return float(_scipy_norm.ppf(p))

    # Acklam's algorithm -- public-domain rational approximation.
    # Coefficients for the rational approximation.
    a = (-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143,
         3.75440866190742)

    plow = 0.02425
    phigh = 1.0 - plow

    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                               + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _sample_skew(returns: np.ndarray) -> float:
    """Sample skewness (Fisher, biased) -- scipy if present else numpy."""
    if _HAVE_SCIPY and _scipy_skew is not None:
        return float(_scipy_skew(returns, bias=True))
    n = returns.size
    if n < 2:
        return 0.0
    mean = returns.mean()
    s = returns.std()  # population std (ddof=0) to match scipy bias=True denominator
    if s == 0.0:
        return 0.0
    return float(np.mean(((returns - mean) / s) ** 3))


def _sample_excess_kurtosis(returns: np.ndarray) -> float:
    """Sample excess kurtosis (Fisher, biased) -- scipy if present else numpy."""
    if _HAVE_SCIPY and _scipy_kurtosis is not None:
        return float(_scipy_kurtosis(returns, fisher=True, bias=True))
    n = returns.size
    if n < 2:
        return 0.0
    mean = returns.mean()
    s = returns.std()  # ddof=0 to match scipy bias=True
    if s == 0.0:
        return 0.0
    g2 = np.mean(((returns - mean) / s) ** 4)
    return float(g2 - 3.0)


# ---------------------------------------------------------------------------
# Public data container.
# ---------------------------------------------------------------------------
@dataclass
class BacktestMetrics:
    """Bundle of backtest performance statistics."""

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    deflated_sharpe: float
    alpha_vs_buyhold: float | None
    num_periods: int
    periods_per_year: int


# ---------------------------------------------------------------------------
# Core helpers.
# ---------------------------------------------------------------------------
def returns_from_equity(equity: Sequence[float] | np.ndarray) -> np.ndarray:
    """Simple per-period returns ``equity[1:] / equity[:-1] - 1``.

    Raises ``ValueError`` if fewer than two equity points are supplied (no
    return can be formed). Returns a 1-D float ``np.ndarray`` of length
    ``len(equity) - 1``.
    """
    eq = np.asarray(equity, dtype=float).ravel()
    if eq.size < 2:
        raise ValueError(
            "returns_from_equity requires at least 2 equity points, "
            f"got {eq.size}"
        )
    prev = eq[:-1]
    # Guard against division by zero: numpy emits inf/nan naturally, but a
    # zero in the denominator would warn. We let it pass through and rely on
    # callers to handle non-finite returns; equity values are assumed > 0.
    return eq[1:] / prev - 1.0


def _expected_max_z(n_trials: int) -> float:
    """Expected value of the maximum of ``n_trials`` standard normals.

    Uses the approximation
    ``z = (1 - gamma) * ndtri(1 - 1/N) + gamma * ndtri(1 - 1/(N*e))``
    where ``gamma`` is the Euler-Mascheroni constant. Returns 0.0 for
    ``N == 1`` (no multiple-testing penalty).
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649
    e = math.e
    n = float(n_trials)
    z = ((1.0 - gamma) * _norm_ppf(1.0 - 1.0 / n)
         + gamma * _norm_ppf(1.0 - 1.0 / (n * e)))
    return float(z)


def _deflated_sharpe_ratio(
    returns: np.ndarray,
    sharpe_annualized: float,
    n_trials: int,
    periods_per_year: int,
    skew_override: float | None,
    kurt_override: float | None,
) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio in [0, 1].

    See module docstring for the exact closed form used. ``sharpe_annualized``
    is accepted for API symmetry but the statistic is computed in
    per-observation units to avoid annualization ambiguity.
    """
    n = int(returns.size)
    if n < 2:
        return 0.0

    std = float(returns.std(ddof=0))
    if std == 0.0:
        return 0.0

    sr_per = float(returns.mean()) / std

    sk = float(skew_override) if skew_override is not None else _sample_skew(returns)
    ku = (float(kurt_override)
          if kurt_override is not None else _sample_excess_kurtosis(returns))

    z = _expected_max_z(n_trials)
    sr0_per = z / math.sqrt(n)  # hurdle Sharpe in per-observation units

    denom_sq = 1.0 - sk * sr_per + (ku * sr_per ** 2) / 4.0
    if denom_sq <= 0.0:
        return 0.0
    denom = math.sqrt(denom_sq)

    stat = (sr_per - sr0_per) * math.sqrt(n - 1) / denom
    dsr = _norm_cdf(stat)
    # Clamp into [0, 1] to absorb tiny floating-point excursions at the tails.
    if dsr < 0.0:
        return 0.0
    if dsr > 1.0:
        return 1.0
    return float(dsr)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def compute_metrics(
    equity: Sequence[float] | np.ndarray,
    benchmark_equity: Sequence[float] | np.ndarray | None = None,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
    n_trials: int = 1,
    strategy_skew: float | None = None,
    strategy_kurtosis: float | None = None,
) -> BacktestMetrics:
    """Compute the full battery of backtest statistics from an equity curve.

    Parameters
    ----------
    equity:
        Chronological sequence (>= 2 points) of portfolio mark-to-market
        values. Must be strictly positive for return math to be meaningful.
    benchmark_equity:
        Optional buy-and-hold equity curve of the same length as ``equity``.
        When supplied, ``alpha_vs_buyhold`` is the annualized difference of
        mean periodic returns (strategy minus benchmark); otherwise it is
        ``None``.
    periods_per_year:
        Annualization factor (252 for daily, 12 for monthly, ...). Must be
        a positive integer.
    risk_free:
        Annualized risk-free rate used for excess-return Sharpe/Sortino.
    n_trials:
        Number of strategies tried, for the DSR multiple-testing deflation.
        ``>= 1``; ``1`` means no deflation penalty.
    strategy_skew / strategy_kurtosis:
        Optional precomputed sample skew / excess kurtosis of the periodic
        returns. If ``None`` they are computed internally.

    Returns
    -------
    BacktestMetrics
        The computed statistics. Raises ``ValueError`` if ``equity`` has
        fewer than two points or ``periods_per_year`` is not positive.
    """
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError(
            f"periods_per_year must be a positive integer, got {periods_per_year!r}"
        )
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError(f"n_trials must be an integer >= 1, got {n_trials!r}")

    eq = np.asarray(equity, dtype=float).ravel()
    if eq.size < 2:
        raise ValueError(
            "compute_metrics requires at least 2 equity points, "
            f"got {eq.size}"
        )

    ppy = periods_per_year
    returns = returns_from_equity(eq)
    n = int(returns.size)

    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))

    # --- Return levels ----------------------------------------------------
    total_return = float(eq[-1] / eq[0] - 1.0)
    cagr = float((eq[-1] / eq[0]) ** (ppy / n) - 1.0)

    # --- Annualized volatility -------------------------------------------
    volatility = std_ret * math.sqrt(ppy)

    # --- Sharpe -----------------------------------------------------------
    ann_excess = mean_ret * ppy - risk_free
    ann_vol = std_ret * math.sqrt(ppy)
    sharpe = ann_excess / ann_vol if ann_vol > 0.0 else 0.0

    # --- Sortino ----------------------------------------------------------
    target_per_period = risk_free / ppy
    downside_dev_per = math.sqrt(float(np.mean(np.minimum(0.0, returns - target_per_period) ** 2)))
    downside_dev_ann = downside_dev_per * math.sqrt(ppy)
    sortino = ann_excess / downside_dev_ann if downside_dev_ann > 0.0 else 0.0

    # --- Maximum drawdown (peak-to-trough) -------------------------------
    running_max = np.maximum.accumulate(eq)
    drawdowns = eq / running_max - 1.0
    max_drawdown = float(drawdowns.min())  # a negative number (or 0)

    # --- Calmar -----------------------------------------------------------
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0.0 else 0.0

    # --- Deflated Sharpe Ratio -------------------------------------------
    deflated_sharpe = _deflated_sharpe_ratio(
        returns,
        sharpe_annualized=sharpe,
        n_trials=n_trials,
        periods_per_year=ppy,
        skew_override=strategy_skew,
        kurt_override=strategy_kurtosis,
    )

    # --- Alpha vs buy-and-hold -------------------------------------------
    alpha_vs_buyhold: float | None = None
    if benchmark_equity is not None:
        bench = np.asarray(benchmark_equity, dtype=float).ravel()
        if bench.size == eq.size and bench.size >= 2:
            bret = returns_from_equity(bench)
            alpha_vs_buyhold = float((mean_ret - float(bret.mean())) * ppy)

    return BacktestMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        deflated_sharpe=deflated_sharpe,
        alpha_vs_buyhold=alpha_vs_buyhold,
        num_periods=n,
        periods_per_year=ppy,
    )
