"""Phase 0 backtest engine: run the agent graph over history and price the result.

The engine is the measuring stick for the whole profitability roadmap. It loops
``graph.propagate(ticker, date)`` over a list of decision dates, translates each
5-tier rating into a target portfolio weight, simulates holding that weight on a
daily mark-to-market calendar, and returns an equity curve plus a full metric
suite (delegated to :mod:`yiagents.backtest.metrics`).

Design choices driven by the roadmap:

* **Reuse, do not rebuild.** The graph already produces the rating and already
  owns PIT-safe data loading; the engine only adds the portfolio simulation on
  top. ``_fetch_returns`` / ``_resolve_benchmark`` are reused where useful.
* **Pluggable sizing.** ``rating_to_weight`` (the simple baseline mapping) is the
  default. Phase 1 swaps in a ``weight_fn`` driven by the risk layer so the
  *same* realized decisions can be re-priced under different risk rules -- the
  A/B comparison every later phase runs against.
* **Honest about LLM cost/non-determinism.** A :class:`~yiagents.backtest.cache.DecisionCache`
  memoizes realized decisions per ``(ticker, date, run_tag)``; the multi-run
  distribution comes from re-realizing under fresh ``run_tag`` values, not from
  silently re-billing the LLM on every replay.
* **Hermetic in tests.** A ``price_provider`` callable lets tests inject
  synthetic prices; production uses a yfinance-backed default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from yiagents.agents.utils.rating import parse_rating
from yiagents.backtest.cache import CachedDecision, DecisionCache
from yiagents.backtest.metrics import BacktestMetrics, compute_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default rating -> target long-weight mapping (the Phase-0 *baseline*).
# Buy/Overweight commit capital, Hold keeps the prior position (encoded as
# ``None`` so the simulator knows to "do nothing"), Underweight/Sell go flat.
# ---------------------------------------------------------------------------
DEFAULT_RATING_TO_WEIGHT: dict[str, float | None] = {
    "Buy": 1.0,
    "Overweight": 0.8,
    "Hold": None,           # hold the existing position (no rebalance)
    "Underweight": 0.0,
    "Sell": 0.0,
}


class _GraphLike(Protocol):
    """Structural type the engine needs from a YiAgentsGraph."""

    def propagate(self, company_name: str, trade_date: str, asset_type: str = "stock") -> Any: ...


# A weight function receives everything the risk layer needs to size a position
# deterministically: the realized rating, the trade date, and a mutable context
# dict the engine threads through the backtest (equity history, returns, etc.).
WeightFn = Callable[[str, str, dict[str, Any]], float | None]


def _default_weight_fn(
    rating_to_weight: Mapping[str, float | None],
) -> WeightFn:
    """Bind a rating->weight table into the ``WeightFn`` signature."""

    def _fn(rating: str, date: str, ctx: dict[str, Any]) -> float | None:
        return rating_to_weight.get(rating, rating_to_weight.get("Hold"))

    return _fn


def _yfinance_price_provider(ticker: str, start: str, end: str) -> pd.Series:
    """Default daily close-price provider, indexed by date string (YYYY-MM-DD).

    Lives here rather than in the graph so the engine stays decoupled from the
    graph internals and testable with synthetic prices.
    """
    import yfinance as yf
    from yiagents.dataflows.symbol_utils import normalize_symbol

    canonical = normalize_symbol(ticker)
    hist = yf.Ticker(canonical).history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return pd.Series(dtype=float)
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    s = hist["Close"].dropna()
    s.index = s.index.strftime("%Y-%m-%d")
    return s.astype(float)


@dataclass
class TradeRow:
    """One rebalance event: what the agents said, and what the simulator did."""

    date: str
    rating: str
    target_weight: float | None
    executed_weight: float
    price: float | None
    raw_return: float | None = None        # realized asset return over the holding period
    alpha_vs_index: float | None = None    # raw_return minus the index's return over the same window
    decision_excerpt: str = ""


@dataclass
class BacktestResult:
    """Full output of a single backtest run."""

    ticker: str
    initial_capital: float
    holding_days: int
    equity: list[float]
    equity_dates: list[str]
    trades: list[TradeRow]
    benchmark_equity: list[float]
    benchmark_name: str
    metrics: BacktestMetrics | None = None
    config_summary: dict[str, Any] = field(default_factory=dict)
    cached_hits: int = 0
    cached_misses: int = 0

    def equity_series(self) -> pd.Series:
        return pd.Series(self.equity, index=self.equity_dates, dtype=float)

    def benchmark_series(self) -> pd.Series:
        return pd.Series(self.benchmark_equity, index=self.equity_dates, dtype=float)


def _resolve_index_benchmark(graph: Any | None, ticker: str) -> str:
    """Pick the index used for alpha, reusing the graph's benchmark logic."""
    if graph is not None and hasattr(graph, "_resolve_benchmark"):
        try:
            return graph._resolve_benchmark(ticker)
        except Exception:  # noqa: BLE001 -- benchmark resolution must never break a backtest
            pass
    return "SPY"


def _index_position_at_or_after(index: pd.Index, target: str) -> int | None:
    """Position of ``target`` in a chronological date-string index, else the
    first strictly-later date. YYYY-MM-DD strings sort chronologically, so plain
    lexicographic comparison is correct and avoids pandas' numeric-only
    ``get_indexer(method='nearest')`` (which does arithmetic on the index)."""
    arr = np.asarray(index, dtype=object)
    exact = np.where(arr == str(target))[0]
    if exact.size:
        return int(exact[0])
    after = np.where(arr > str(target))[0]
    return int(after[0]) if after.size else None


def _asset_return(prices: pd.Series, start_date: str, horizon: int) -> float | None:
    """Return of holding the asset from ``start_date`` for ``horizon`` sessions."""
    if prices.empty:
        return None
    idx = _index_position_at_or_after(prices.index, start_date)
    if idx is None or idx >= len(prices) - 1:
        return None
    end_idx = min(idx + horizon, len(prices) - 1)
    p0 = float(prices.iloc[idx])
    p1 = float(prices.iloc[end_idx])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return p1 / p0 - 1.0


def run_backtest(
    graph: _GraphLike,
    ticker: str,
    dates: list[str],
    initial_capital: float = 100_000.0,
    holding_days: int = 5,
    rating_to_weight: Mapping[str, float | None] | None = None,
    weight_fn: WeightFn | None = None,
    asset_type: str = "stock",
    cache: DecisionCache | None = None,
    run_tag: str = "default",
    price_provider: Callable[[str, str, str], pd.Series] = _yfinance_price_provider,
    periods_per_year: int = 252,
    cost_bps: float = 0.0,
    compute_index_alpha: bool = True,
    progress: bool = False,
) -> BacktestResult:
    """Run the agent graph over ``dates`` and price the resulting strategy.

    Parameters
    ----------
    graph:
        Anything with a ``propagate(ticker, date, asset_type=...)`` method
        returning ``(final_state, rating)`` (a :class:`YiAgentsGraph`).
    ticker:
        Instrument to analyze and trade.
    dates:
        Decision / rebalance dates (YYYY-MM-DD), in chronological order.
    rating_to_weight / weight_fn:
        Position-sizing rule. If ``weight_fn`` is given it wins; otherwise the
        engine builds one from ``rating_to_weight`` (default
        :data:`DEFAULT_RATING_TO_WEIGHT`). A ``weight_fn`` may return ``None``
        to mean "hold the existing position" (no rebalance this round).
    cache:
        Optional :class:`DecisionCache`. When set, realized decisions are read
        from / written to disk keyed by ``(ticker, date, run_tag)`` so replays
        do not re-bill the LLM.
    cost_bps:
        Transaction cost in basis points applied to the traded notional on each
        rebalance (round-trip costs can be split across two calls by the caller).
    compute_index_alpha:
        If True, fetch the regional index benchmark and record ``alpha_vs_index``
        per trade (reuses ``graph._resolve_benchmark``).
    """
    if not dates:
        raise ValueError("run_backtest requires at least one decision date")
    if holding_days < 1:
        raise ValueError("holding_days must be >= 1")

    weight_fn = weight_fn or _default_weight_fn(rating_to_weight or DEFAULT_RATING_TO_WEIGHT)

    # --- Price window: span every decision date plus one holding period ----
    sorted_dates = sorted(str(d) for d in dates)
    start_date = sorted_dates[0]
    # Buffer the end so the last decision still has a full holding window to mark.
    from datetime import datetime, timedelta
    end_dt = datetime.strptime(sorted_dates[-1], "%Y-%m-%d") + timedelta(days=holding_days + 10)
    end_date = end_dt.strftime("%Y-%m-%d")

    prices = price_provider(ticker, start_date, end_date)
    if prices.empty:
        raise ValueError(
            f"No price data for {ticker} between {start_date} and {end_date}; "
            "cannot mark the backtest to market."
        )
    prices = prices.sort_index()

    index_prices: pd.Series | None = None
    index_name = ""
    if compute_index_alpha:
        index_name = _resolve_index_benchmark(graph, ticker)
        try:
            index_prices = price_provider(index_name, start_date, end_date).sort_index()
        except Exception as exc:  # noqa: BLE001 -- index data is advisory only
            logger.warning("Could not load index benchmark %s: %s", index_name, exc)
            index_prices = None

    # --- Portfolio simulation: daily walk, rebalance only on decision dates --
    decision_set = set(sorted_dates)
    # Threaded through every sizing call so a risk layer can use history.
    ctx: dict[str, Any] = {
        "equity_history": [initial_capital],
        "returns_history": [],
        "holding_days": holding_days,
        "ticker": ticker,
    }

    cash = initial_capital
    shares = 0.0
    current_weight = 0.0
    equity_curve: list[float] = []
    equity_dates: list[str] = []
    trades: list[TradeRow] = []
    cached_hits = 0
    cached_misses = 0

    for trade_date, price in prices.items():
        if trade_date < sorted_dates[0]:
            continue

        # Mark to market at today's close.
        equity = cash + shares * float(price)
        equity_curve.append(float(equity))
        equity_dates.append(trade_date)
        ctx["equity_history"].append(float(equity))
        if len(equity_curve) >= 2:
            prev = equity_curve[-2]
            if prev > 0:
                ctx["returns_history"].append(float(equity / prev - 1.0))

        # Rebalance on a decision date.
        if trade_date in decision_set:
            rating, decision_md, was_cached = _resolve_decision(
                graph, ticker, trade_date, asset_type, cache, run_tag,
            )
            if was_cached:
                cached_hits += 1
            else:
                cached_misses += 1

            target_weight = weight_fn(rating, trade_date, ctx)

            # None == "hold": carry the prior weight, still record the decision.
            if target_weight is None:
                target_weight = current_weight
            target_weight = float(np.clip(target_weight, 0.0, 1.0))

            desired_value = target_weight * equity
            current_value = shares * float(price)
            traded_notional = abs(desired_value - current_value)
            cost = traded_notional * (cost_bps / 10_000.0)

            # Execute rebalance at this close, net of cost.
            cash += (current_value - desired_value) - cost
            shares = desired_value / float(price) if float(price) > 0 else 0.0
            current_weight = target_weight

            raw_ret = _asset_return(prices, trade_date, holding_days)
            alpha = None
            if raw_ret is not None and index_prices is not None and not index_prices.empty:
                idx_ret = _asset_return(index_prices, trade_date, holding_days)
                if idx_ret is not None:
                    alpha = raw_ret - idx_ret

            trades.append(TradeRow(
                date=trade_date,
                rating=rating,
                target_weight=target_weight,
                executed_weight=target_weight,
                price=float(price),
                raw_return=raw_ret,
                alpha_vs_index=alpha,
                decision_excerpt=_excerpt(decision_md),
            ))

            if progress:
                logger.info(
                    "%s %s: rating=%s weight=%.2f equity=%.0f",
                    trade_date, ticker, rating, target_weight, equity,
                )

    if len(equity_curve) < 2:
        raise ValueError(
            f"Backtest produced fewer than 2 equity points for {ticker}; "
            "need a wider date window."
        )

    # --- Buy & hold benchmark of the SAME ticker (the "can we beat holding?") -
    first_price = float(prices.loc[prices.index >= sorted_dates[0]].iloc[0])
    bh_shares = initial_capital / first_price if first_price > 0 else 0.0
    bh_curve = [bh_shares * float(p) for p in prices.loc[prices.index >= sorted_dates[0]].values]

    metrics = compute_metrics(
        equity_curve,
        benchmark_equity=bh_curve,
        periods_per_year=periods_per_year,
        n_trials=1,
    )

    return BacktestResult(
        ticker=ticker,
        initial_capital=initial_capital,
        holding_days=holding_days,
        equity=equity_curve,
        equity_dates=equity_dates,
        trades=trades,
        benchmark_equity=bh_curve,
        benchmark_name=f"{ticker} buy-and-hold",
        metrics=metrics,
        config_summary={
            "asset_type": asset_type,
            "run_tag": run_tag,
            "cost_bps": cost_bps,
            "rating_to_weight": dict(rating_to_weight or DEFAULT_RATING_TO_WEIGHT),
            "index_benchmark": index_name,
        },
        cached_hits=cached_hits,
        cached_misses=cached_misses,
    )


def _resolve_decision(
    graph: _GraphLike,
    ticker: str,
    trade_date: str,
    asset_type: str,
    cache: DecisionCache | None,
    run_tag: str,
) -> tuple[str, str, bool]:
    """Return ``(rating, final_decision_markdown, was_cached)``.

    Uses the cache when available; otherwise calls the graph. The graph returns
    ``(final_state, rating)``; the markdown decision lives in
    ``final_state['final_trade_decision']``.
    """
    if cache is not None:
        cached = cache.get(ticker, trade_date, run_tag)
        if cached is not None:
            return cached.rating, cached.final_decision, True

    final_state, rating = _call_graph(graph, ticker, trade_date, asset_type)
    rating = parse_rating(rating) if isinstance(rating, str) else "Hold"
    decision_md = ""
    if isinstance(final_state, Mapping):
        decision_md = str(final_state.get("final_trade_decision", ""))  # type: ignore[union-attr]
    elif isinstance(final_state, dict):
        decision_md = str(final_state.get("final_trade_decision", ""))

    if cache is not None:
        cache.remember(ticker, trade_date, rating, decision_md, run_tag)
    return rating, decision_md, False


def _call_graph(graph: _GraphLike, ticker: str, trade_date: str, asset_type: str):
    """Invoke propagate defensively; a failing node should not abort the backtest."""
    try:
        return graph.propagate(ticker, trade_date, asset_type=asset_type)
    except Exception as exc:  # noqa: BLE001 -- one bad date must not kill the run
        logger.warning("propagate failed for %s on %s: %s (treating as Hold)", ticker, trade_date, exc)
        return {"final_trade_decision": f"[propagate error: {exc}]"}, "Hold"


def _excerpt(text: str, limit: int = 200) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")
