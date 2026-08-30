"""Market turbulence index — a market-level, ex-ante risk-stress signal.

Adapted from FinRL's ``calculate_turbulence``
(``finrl/meta/preprocessor/preprocessors.py``, FinRL MIT license,
https://github.com/AI4Finance-Foundation/FinRL/blob/master/LICENSE). FinRL
computes a cross-sectional Mahalanobis distance over a panel (e.g. Dow 30)
using a 252-day rolling covariance. YiAlpha analyses one symbol at a time, so
this module reduces that to the **single-asset** case: the squared z-score of
the benchmark's latest daily return against its trailing 252-day distribution.
That preserves the signal's meaning ("how abnormal is today's market move
relative to the recent regime") without needing a multi-ticker panel.

Why this exists alongside ``yialpha.risk.breaker.DrawdownBreaker``: the
breaker is **portfolio-level and reactive** (it trips after drawdown has
already happened). Turbulence is **market-level and ex-ante** (today's return
is abnormal vs the recent calm) — a leading stress cue the conservative risk
debater can weigh. The two are complementary, not redundant.

Point-in-time safety is inherited from ``load_ohlcv`` (already filters rows to
``<= curr_date``), so a backtest cannot peek at a future return.

This module is advisory-only and **fail-soft**: any data/numerical failure
returns ``None`` (and logs) rather than propagating — the caller simply omits
the turbulence reading. It never changes an agent's tools or capabilities; the
opt-in is wired in ``conservative_debator`` behind ``YIALPHA_MARKET_REGIME``
(default off = byte-equivalent).

The 2026-08-15 expansion adds the composite :func:`format_regime_context`:
trend state (close vs 50/200 SMA + ADX strength), volatility state (rvol_20
percentile vs its trailing year), the benchmark turbulence above, and — for
A-share live runs — whole-market breadth. It feeds the market analyst's
prompt behind the ``regime_context`` config key (default ON, fail-soft
omission, env ``YIALPHA_REGIME_CONTEXT`` to disable).
"""

from __future__ import annotations

import logging
import math

from stockstats import wrap

from .config import get_config
from .stockstats_utils import load_ohlcv
from .symbol_utils import is_a_stock
from .utils import is_historical_date
from .vol_estimators import close_to_close_vol, periods_per_year_for

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

    The single implementation of the benchmark-resolution rule (the graph's
    ``YiAlphaGraph._resolve_benchmark`` delegates here since 2026-08-16 —
    it was a verbatim copy before): ``benchmark_ticker`` overrides
    everything; otherwise the suffix map in config matches the ticker's
    exchange suffix; the empty-suffix entry (SPY by default) is the
    fallback. Note: crypto tickers have no suffix and therefore resolve to
    SPY — a cross-asset risk-on/off proxy. Set ``benchmark_ticker`` for a
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


# ---------------------------------------------------------------------------
# Composite regime context (2026-08-15 expansion): trend + volatility states
# on the analyzed ticker, plus the existing benchmark turbulence and (A-share
# live runs only) whole-market breadth. One structured line the market
# analyst and the conservative debater can weigh — advisory, fail-soft per
# component, and gated by the ``regime_context`` config key.
# (The pre-expansion turbulence-only renderer ``format_market_regime`` was
# removed 2026-08-16: format_regime_context is its strict superset and its
# only remaining callers were its own tests.)
# ---------------------------------------------------------------------------

#: Rows required for the 200-SMA/ADX trend state to be meaningful.
_TREND_MIN_ROWS = 200
#: Rolling rvol window for the volatility state.
_VOL_WINDOW = 20
#: Trailing rvol values the latest one is percentile-ranked against.
_VOL_RANK_BASE = 252


def classify_trend_state(df) -> dict | None:
    """Trend state of the analyzed ticker: close vs SMA50/200, MA stack,
    ADX strength (stockstats EMA-smoothed ADX — ranking use only).

    Returns ``None`` when there is not enough history (the caller omits the
    component rather than guessing).
    """
    if df is None or len(df) < _TREND_MIN_ROWS:
        return None
    try:
        sdf = wrap(df.copy())
        close = float(sdf["close"].iloc[-1])
        sma50 = float(sdf["close_50_sma"].iloc[-1])
        sma200 = float(sdf["close_200_sma"].iloc[-1])
        adx = float(sdf["adx"].iloc[-1])
    except Exception:  # noqa: BLE001 — advisory component, fail-soft
        return None
    if not all(math.isfinite(v) for v in (close, sma50, sma200, adx)):
        return None

    if close > sma50 > sma200:
        trend = "uptrend"
    elif close < sma50 < sma200:
        trend = "downtrend"
    else:
        trend = "mixed"
    stack = "bull_stack" if sma50 > sma200 else "bear_stack"
    if adx >= 25.0:
        strength = "strong"
    elif adx <= 20.0:
        strength = "weak"
    else:
        strength = "moderate"
    return {
        "trend": trend,
        "stack": stack,
        "adx": adx,
        "adx_strength": strength,
        "close": close,
        "sma50": sma50,
        "sma200": sma200,
    }


def classify_vol_state(df, asset_type: str | None = None) -> dict | None:
    """Volatility state: rvol_20 percentile vs its trailing 252 values.

    Labels: low (<25th), normal, high (>75th), extreme (>90th). ``None``
    when the rvol history is too short to rank against. The percentile is
    scale-invariant, but the reported annualized level uses the asset's own
    calendar (365 for 24/7 crypto, 252 otherwise) so the displayed number
    matches the indicator tools' scale.
    """
    rvol = close_to_close_vol(
        df, window=_VOL_WINDOW,
        periods_per_year=periods_per_year_for(asset_type),
    )
    history = rvol.dropna()
    if len(history) < 30:
        return None
    base = history.tail(_VOL_RANK_BASE)
    latest = float(base.iloc[-1])
    if not math.isfinite(latest):
        return None
    pct = float((base < latest).mean()) * 100.0
    if pct >= 90.0:
        label = "extreme"
    elif pct >= 75.0:
        label = "high"
    elif pct <= 25.0:
        label = "low"
    else:
        label = "normal"
    return {"rvol_20": latest, "percentile": pct, "label": label}


def _a_share_breadth_part(ticker: str, curr_date: str) -> str | None:
    """Live-only A-share breadth fragment; ``None`` for non-A-share,
    historical dates, or any fetch failure (fail-soft omission)."""
    if not is_a_stock(ticker) or is_historical_date(curr_date):
        return None
    try:
        from .akshare_vendor import _cached_breadth_counts

        counts = _cached_breadth_counts()
    except Exception:  # noqa: BLE001 — advisory, must never break the run
        return None
    if not counts:
        return None
    adv, dec = counts["advancing"], counts["declining"]
    ratio = (adv / dec) if dec else float("inf")
    tone = "broad advance" if ratio >= 2.0 else "broad decline" if ratio <= 0.5 else "mixed"
    return (
        f"breadth (A-share live): {adv} up / {dec} down ({tone}), "
        f"涨停 {counts['limit_up']} / 跌停 {counts['limit_down']}"
    )


def format_regime_context(
    ticker: str, curr_date: str, asset_type: str | None = None,
) -> str | None:
    """One structured regime line for the analyzed ticker.

    Composes trend state + volatility state + benchmark turbulence + (A-share
    live) market breadth. Every component is fail-soft: a component that
    cannot be computed is omitted, and the whole line returns ``None`` when
    neither the trend nor the volatility state is available. Advisory only —
    it informs the analyst and the risk debate, it never gates tools.

    ``asset_type`` (e.g. from the graph state) annualizes the vol state on
    the asset's own calendar — 365 for 24/7 crypto, 252 otherwise — so the
    reported level matches the Binance indicator tools' scale.
    """
    try:
        data = load_ohlcv(ticker, curr_date)
    except Exception:  # noqa: BLE001 — advisory signal must be fail-soft
        data = None

    parts: list[str] = []
    trend = classify_trend_state(data) if data is not None else None
    if trend is not None:
        parts.append(
            f"trend={trend['trend']} ({trend['stack']}, ADX {trend['adx']:.0f} "
            f"{trend['adx_strength']})"
        )
    vol = classify_vol_state(data, asset_type) if data is not None else None
    if vol is not None:
        parts.append(
            f"vol={vol['label']} (rvol20 {vol['rvol_20']:.0%} annualized, "
            f"{vol['percentile']:.0f}th pct of trailing year)"
        )
    if trend is None and vol is None:
        return None

    benchmark = resolve_market_benchmark(ticker)
    turb = compute_turbulence(benchmark, curr_date)
    if turb is not None:
        sigma = math.sqrt(turb) if turb > 0 else 0.0
        label = "elevated" if sigma >= _ELEVATED_SIGMA else "normal"
        parts.append(f"turbulence({benchmark})={label} ({sigma:.1f}σ)")

    breadth = _a_share_breadth_part(ticker, curr_date)
    if breadth is not None:
        parts.append(breadth)

    return f"Market regime ({ticker}, {curr_date}): " + " | ".join(parts)
