"""Market turbulence index — a market-level, ex-ante risk-stress signal.

Adapted from FinRL's ``calculate_turbulence``
(``finrl/meta/preprocessor/preprocessors.py``, FinRL MIT license,
https://github.com/AI4Finance-Foundation/FinRL/blob/master/LICENSE). FinRL
computes a cross-sectional Mahalanobis distance over a panel (e.g. Dow 30)
using a 252-day rolling covariance. YiAgents analyses one symbol at a time, so
this module reduces that to the **single-asset** case: the squared z-score of
the benchmark's latest daily return against its trailing 252-day distribution.
That preserves the signal's meaning ("how abnormal is today's market move
relative to the recent regime") without needing a multi-ticker panel.

Why this exists alongside ``yiagents.risk.breaker.DrawdownBreaker``: the
breaker is **portfolio-level and reactive** (it trips after drawdown has
already happened). Turbulence is **market-level and ex-ante** (today's return
is abnormal vs the recent calm) — a leading stress cue the conservative risk
debater can weigh. The two are complementary, not redundant.

Point-in-time safety is inherited from ``load_ohlcv`` (already filters rows to
``<= curr_date``), so a backtest cannot peek at a future return.

This module is advisory-only and **fail-soft**: any data/numerical failure
returns ``None`` (and logs) rather than propagating — the caller simply omits
the turbulence reading. It never changes an agent's tools or capabilities; the
opt-in is wired in ``conservative_debator`` behind ``YIAGENTS_MARKET_REGIME``
(default off = byte-equivalent).
"""

from __future__ import annotations

import logging
import math

from .config import get_config
from .stockstats_utils import load_ohlcv

logger = logging.getLogger(__name__)

# Default trailing window (one trading year). Matches FinRL's 252.
_DEFAULT_WINDOW = 252
# Minimum returns required in the window before we trust μ/σ². Below this the
# turbulence estimate is too noisy to be a meaningful stress signal.
_DEFAULT_MIN_PERIODS = 60
# Coarse "elevated stress" cutoff in rolling-σ units (sqrt of the index). A
# benchmark daily return beyond ~2.5σ is flagged elevated. Advisory only — the
# raw index is always surfaced so the reader can judge.
_ELEVATED_SIGMA = 2.5


def resolve_market_benchmark(ticker: str) -> str:
    """Resolve a market benchmark symbol for ``ticker``.

    Replicates ``yiagents.graph.trading_graph._resolve_benchmark`` at the
    dataflow layer (this module has no graph instance): ``benchmark_ticker``
    overrides everything; otherwise the suffix map in config matches the
    ticker's exchange suffix; the empty-suffix entry (SPY by default) is the
    fallback. Note: crypto tickers have no suffix and therefore resolve to SPY
    — a cross-asset risk-on/off proxy. Set ``benchmark_ticker`` for a
    crypto-native benchmark (e.g. BTCUSDT).
    """
    config = get_config()
    explicit = config.get("benchmark_ticker")
    if explicit:
        return explicit
    benchmark_map = config.get("benchmark_map", {})
    ticker_upper = ticker.upper()
    for suffix, benchmark in benchmark_map.items():
        if suffix and ticker_upper.endswith(suffix.upper()):
            return benchmark
    return benchmark_map.get("", "SPY")


def compute_turbulence(
    symbol: str,
    curr_date: str,
    *,
    window: int = _DEFAULT_WINDOW,
    min_periods: int = _DEFAULT_MIN_PERIODS,
) -> float | None:
    """Compute the single-asset turbulence index for ``symbol`` at ``curr_date``.

    Returns the squared z-score of the most recent daily return against the
    trailing ``window`` returns (excluding the current day), or ``None`` when
    there is too little history, the variance is degenerate, or data fetch
    fails. PIT-safe via ``load_ohlcv``.
    """
    try:
        data = load_ohlcv(symbol, curr_date)
    except Exception as exc:  # noqa: BLE001 — advisory signal must be fail-soft
        logger.warning("turbulence: OHLCV fetch failed for %s: %s", symbol, exc)
        return None

    if data is None or data.empty or "Close" not in data.columns:
        return None

    close = (
        data.sort_values("Date")["Close"]
        .astype(float)
        .dropna()
    )
    returns = close.pct_change().dropna()
    # Need the current return PLUS `window` prior returns.
    if len(returns) < window + 1:
        # Fall back to whatever history exists, as long as it clears min_periods.
        if len(returns) < min_periods + 1:
            return None
        hist = returns.iloc[-(len(returns) - 1) : -1] if len(returns) > 1 else returns.iloc[:0]
    else:
        hist = returns.iloc[-(window + 1) : -1]

    curr = returns.iloc[-1]
    if len(hist) < min_periods:
        return None

    mu = float(hist.mean())
    var = float(hist.var(ddof=1))
    if not math.isfinite(var) or var <= 0:
        return None

    delta = float(curr) - mu
    return delta * delta / var


def format_market_regime(
    ticker: str,
    curr_date: str,
    *,
    window: int = _DEFAULT_WINDOW,
    min_periods: int = _DEFAULT_MIN_PERIODS,
) -> str | None:
    """Resolve the benchmark for ``ticker`` and return a one-line stress reading.

    Returns ``None`` (→ caller omits the line) when turbulence can't be
    computed. Example output::

        Market turbulence index (SPY, 252d): 6.12 (≈2.5σ rolling) — elevated
    """
    benchmark = resolve_market_benchmark(ticker)
    turb = compute_turbulence(
        benchmark, curr_date, window=window, min_periods=min_periods
    )
    if turb is None:
        return None

    sigma = math.sqrt(turb) if turb > 0 else 0.0
    label = "elevated" if sigma >= _ELEVATED_SIGMA else "normal"
    return (
        f"Market turbulence index ({benchmark}, {window}d): "
        f"{turb:.2f} (≈{sigma:.1f}σ rolling) — {label}"
    )
