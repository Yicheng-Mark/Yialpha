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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from yiagents.agents.utils.rating import parse_rating
from yiagents.backtest.cache import DecisionCache
from yiagents.backtest.metrics import BacktestMetrics, compute_metrics, returns_from_equity

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
    """One decision event: what the agents said, and what the simulator did.

    ``date`` is the signal/as-of date.  ``execution_date`` is deliberately a
    later price bar: the graph is allowed to consume the completed signal-day
    bar, therefore that same close is not an executable price.  A row may be a
    genuine hold (``is_rebalance=False`` and ``traded_notional=0``).
    """

    date: str
    rating: str
    target_weight: float | None
    executed_weight: float
    price: float | None
    raw_return: float | None = None        # realized asset return over the holding period
    alpha_vs_index: float | None = None    # raw_return minus the index's return over the same window
    decision_excerpt: str = ""
    execution_date: str | None = None
    previous_weight: float = 0.0
    traded_notional: float = 0.0
    transaction_cost: float = 0.0
    is_rebalance: bool = False
    # Net portfolio return of the actual long-position episode opened by this
    # row.  It is filled when the position is closed (or at the backtest end),
    # and is the source for win-rate.  It intentionally stays None for cash and
    # repeated Hold decisions.
    position_return: float | None = None
    stop_loss: float | None = None
    risk_action: str | None = None
    risk_warning: str | None = None


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
    degraded_decision_count: int = 0
    unexecuted_decision_count: int = 0

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


def _execution_date_after_signal(
    index: pd.Index,
    signal_date: str,
    lag_bars: int,
) -> str | None:
    """Return the executable bar strictly after ``signal_date``.

    A daily analysis made as of ``signal_date`` may use that day's completed
    OHLCV.  Consequently the earliest honest close-price fill is the next
    available bar.  ``lag_bars=1`` means that first strictly-later bar, including
    the Monday after a weekend signal.
    """
    arr = np.asarray(index, dtype=object)
    later = np.where(arr > str(signal_date))[0]
    offset = lag_bars - 1
    if later.size <= offset:
        return None
    return str(arr[int(later[offset])])


def _finish_open_position_episode(
    opening_trade: TradeRow | None,
    opening_equity: float | None,
    ending_equity: float,
) -> float | None:
    """Stamp and return net P&L for one actual long-position episode."""
    if opening_trade is None or opening_equity is None or opening_equity <= 0:
        return None
    realized = float(ending_equity / opening_equity - 1.0)
    opening_trade.position_return = realized
    return realized


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
    periods_per_year: int | None = None,
    cost_bps: float = 0.0,
    n_trials: int = 1,
    execution_lag_bars: int = 1,
    compute_index_alpha: bool = True,
    factor_model: str | None = None,
    event_study: bool = False,
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
    execution_lag_bars:
        Number of available price bars between a completed daily signal and its
        fill.  The minimum/default is one: a signal that can read the complete
        signal-day OHLCV can never fill at that same close.
    n_trials:
        Number of independent strategy variants being compared in this research
        run, forwarded to :func:`compute_metrics` for the Deflated Sharpe Ratio
        multiple-testing deflation. ``1`` (default) applies no deflation penalty
        and is byte-equivalent to the prior behaviour. Set to the count of
        strategies/configurations actually tried (e.g. in ``run_baseline --full``
        the baseline + improved variants) so the DSR hurdle reflects the real
        selection bias rather than being deflated to a near-trivial test.
    compute_index_alpha:
        If True, fetch the regional index benchmark and record ``alpha_vs_index``
        per trade (reuses ``graph._resolve_benchmark``).
    factor_model:
        Optional Fama-French attribution: ``"3"`` (Mkt-RF / SMB / HML) or ``"5"``
        (+ RMW / CMA). When set, daily strategy returns are regressed on French
        factor data and ``metrics.factor_alpha`` / ``factor_betas`` /
        ``factor_r_squared`` are filled. ``None`` (default) skips attribution
        entirely — byte-equivalent to the prior behaviour. Fail-open: a missing
        factor file leaves the fields at None without breaking the backtest.
    event_study:
        If True, run a market-model event study over the decision dates: for
        each rebalance, fit ``R_asset = a + b*R_benchmark`` on a preceding
        estimation window, then test whether cumulative abnormal returns in the
        holding window are reliably non-zero (mean CAR, Brown & Warner
        cross-sectional t, bootstrap 95% CI). Fills ``metrics.event_study_*``.
        ``False`` (default) skips it entirely — byte-equivalent. Needs ~250
        trading days of asset + benchmark prices *before* the first decision,
        so an extra wide-window price pull is made when opted in. Advisory /
        fail-open: too few decidable events or a missing benchmark leaves the
        fields at their defaults without breaking the backtest.
    """
    if not dates:
        raise ValueError("run_backtest requires at least one decision date")
    if holding_days < 1:
        raise ValueError("holding_days must be >= 1")
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError(f"n_trials must be an integer >= 1, got {n_trials!r}")
    if not isinstance(execution_lag_bars, int) or execution_lag_bars < 1:
        raise ValueError("execution_lag_bars must be an integer >= 1")
    if asset_type == "crypto_perp":
        raise NotImplementedError(
            "crypto_perp backtesting is disabled: this engine is a long-only "
            "cash/spot simulator and does not model short exposure, funding, "
            "leverage, margin, or liquidation. Labeling its output as a "
            "perpetual-futures backtest would be misleading."
        )
    if periods_per_year is None:
        periods_per_year = 365 if asset_type.startswith("crypto") else 252
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be >= 1")

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

    # Resolve the LLM signal on its explicit as-of date, then map it to a
    # strictly later executable bar.  Resolution is kept chronological; only
    # portfolio sizing waits until execution so the risk layer sees the equity
    # and exposure that really existed at that point.
    scheduled: dict[str, list[tuple[str, str, str]]] = {}
    cached_hits = 0
    cached_misses = 0
    degraded_decision_count = 0
    unexecuted_decision_count = 0
    for signal_date in sorted_dates:
        rating, decision_md, was_cached, was_degraded = _resolve_decision(
            graph, ticker, signal_date, asset_type, cache, run_tag,
        )
        cached_hits += int(was_cached)
        cached_misses += int(not was_cached)
        degraded_decision_count += int(was_degraded)
        execution_date = _execution_date_after_signal(
            prices.index, signal_date, execution_lag_bars,
        )
        if execution_date is None:
            unexecuted_decision_count += 1
            logger.warning(
                "No price bar %d bar(s) after %s; %s decision cannot execute",
                execution_lag_bars, signal_date, ticker,
            )
            continue
        scheduled.setdefault(execution_date, []).append(
            (signal_date, rating, decision_md)
        )

    # Threaded through every sizing call so a risk layer can use only history
    # available at the actual execution point.
    ctx: dict[str, Any] = {
        "equity_history": [initial_capital],
        "returns_history": [],
        "realized_returns": [],
        "holding_days": holding_days,
        "ticker": ticker,
    }

    cash = initial_capital
    shares = 0.0
    equity_curve: list[float] = []
    equity_dates: list[str] = []
    trades: list[TradeRow] = []
    total_traded_notional = 0.0
    opening_trade: TradeRow | None = None
    opening_equity: float | None = None

    for trade_date, price in prices.items():
        if trade_date < sorted_dates[0]:
            continue

        px = float(price)
        if px <= 0.0 or not np.isfinite(px):
            raise ValueError(f"Invalid execution price for {ticker} on {trade_date}: {price!r}")

        # Mark to market at today's completed close before any close-price fill.
        equity = cash + shares * px
        equity_curve.append(float(equity))
        equity_dates.append(str(trade_date))
        ctx["equity_history"].append(float(equity))
        if len(equity_curve) >= 2:
            prev = equity_curve[-2]
            ctx["returns_history"].append(
                float(equity / prev - 1.0) if prev > 0 else 0.0
            )

        for signal_date, rating, decision_md in scheduled.get(str(trade_date), []):
            equity_before = cash + shares * px
            current_value = shares * px
            previous_weight = (
                current_value / equity_before if equity_before > 0 else 0.0
            )
            ctx.update({
                "signal_date": signal_date,
                "execution_date": str(trade_date),
                "execution_price": px,
                "current_weight": previous_weight,
                "current_position_value": current_value,
            })
            # Avoid accidentally carrying metadata from an earlier sizing call.
            ctx.pop("risk_decision", None)
            ctx.pop("risk_warning", None)
            requested_weight = weight_fn(rating, signal_date, ctx)
            risk_decision = ctx.get("risk_decision")
            risk_warning = ctx.get("risk_warning")

            # None means a literal Hold: do not recompute the old target against
            # the new equity, do not touch shares/cash, and charge no fee.
            if requested_weight is None:
                desired_value = current_value
                traded_notional = 0.0
                cost = 0.0
                target_weight: float | None = None
            else:
                target_weight = float(np.clip(requested_weight, 0.0, 1.0))
                desired_value = target_weight * equity_before
                traded_notional = abs(desired_value - current_value)
                cost = traded_notional * (cost_bps / 10_000.0)
                cash += (current_value - desired_value) - cost
                shares = desired_value / px

            total_traded_notional += traded_notional
            equity_after = cash + shares * px
            executed_weight = (
                shares * px / equity_after if equity_after > 0 else 0.0
            )
            tolerance = max(1e-8, abs(equity_before) * 1e-12)
            is_rebalance = traded_notional > tolerance

            raw_ret = _asset_return(prices, str(trade_date), holding_days)
            alpha = None
            if raw_ret is not None and index_prices is not None and not index_prices.empty:
                idx_ret = _asset_return(index_prices, str(trade_date), holding_days)
                if idx_ret is not None:
                    alpha = raw_ret - idx_ret

            row = TradeRow(
                date=signal_date,
                rating=rating,
                target_weight=target_weight,
                executed_weight=executed_weight,
                price=px,
                raw_return=raw_ret,
                alpha_vs_index=alpha,
                decision_excerpt=_excerpt(decision_md),
                execution_date=str(trade_date),
                previous_weight=previous_weight,
                traded_notional=traded_notional,
                transaction_cost=cost,
                is_rebalance=is_rebalance,
                stop_loss=getattr(risk_decision, "stop_loss", None),
                risk_action=getattr(risk_decision, "action", None),
                risk_warning=str(risk_warning) if risk_warning else None,
            )
            trades.append(row)

            was_invested = previous_weight > 1e-12
            is_invested = executed_weight > 1e-12
            if not was_invested and is_invested:
                opening_trade = row
                # Pre-fill equity makes the eventual episode return net of the
                # entry transaction cost already deducted from the portfolio.
                opening_equity = equity_before
            elif was_invested and not is_invested:
                realized = _finish_open_position_episode(
                    opening_trade, opening_equity, equity_after,
                )
                if realized is not None:
                    ctx["realized_returns"].append(realized)
                opening_trade = None
                opening_equity = None

            # Costs are part of same-day equity and the next risk decision must
            # see them.  Refresh the latest history values after every fill.
            equity_curve[-1] = float(equity_after)
            ctx["equity_history"][-1] = float(equity_after)
            if len(equity_curve) >= 2:
                prev = equity_curve[-2]
                ctx["returns_history"][-1] = (
                    float(equity_after / prev - 1.0) if prev > 0 else 0.0
                )

            if progress:
                logger.info(
                    "%s signal %s %s: rating=%s weight=%.2f notional=%.2f equity=%.0f",
                    signal_date, trade_date, ticker, rating, executed_weight,
                    traded_notional, equity_after,
                )

    if len(equity_curve) < 2:
        raise ValueError(
            f"Backtest produced fewer than 2 equity points for {ticker}; "
            "need a wider date window."
        )

    # Close any still-open position episode at the final marked equity.  This is
    # the actual strategy P&L (including costs and any intervening rebalances),
    # not the underlying asset's hypothetical forward direction.
    if opening_trade is not None:
        _finish_open_position_episode(
            opening_trade, opening_equity, float(equity_curve[-1]),
        )

    # --- Buy & hold benchmark of the SAME ticker --------------------------
    # Align entry with the first executable strategy bar rather than granting
    # B&H (or the strategy) the unavailable signal-day close.
    first_execution_date = min(scheduled) if scheduled else None
    bh_cash = initial_capital
    bh_shares = 0.0
    bh_curve: list[float] = []
    for date, price in zip(
        equity_dates,
        prices.loc[prices.index >= sorted_dates[0]].values,
        strict=True,
    ):
        px = float(price)
        if first_execution_date is not None and date >= first_execution_date and bh_shares == 0.0:
            bh_shares = bh_cash / px
            bh_cash = 0.0
        bh_curve.append(float(bh_cash + bh_shares * px))

    metrics = compute_metrics(
        equity_curve,
        benchmark_equity=bh_curve,
        periods_per_year=periods_per_year,
        n_trials=n_trials,
    )

    # Pure post-processing: fill the trade-/equity-derived metric fields
    # (win-rate, annualized turnover, max-drawdown date) that the equity-only
    # ``compute_metrics`` cannot. Deterministic, no I/O, no agent interaction.
    _augment_metrics(metrics, trades, equity_curve, equity_dates,
                     total_traded_notional, periods_per_year)

    # Optional Fama-French factor attribution. Advisory only, fail-open: a
    # missing factor file / failed fit leaves the metrics fields at None and
    # never changes the equity-derived statistics above.
    if factor_model is not None:
        _attribute_factors(
            metrics, equity_curve, equity_dates,
            start_date, end_date, factor_model, periods_per_year,
        )

    # Optional market-model event study over the decision dates. Advisory only,
    # fail-open: too few decidable events / a missing wide-window price pull
    # leaves the metrics fields at their defaults and never changes anything
    # above. Needs benchmark prices, so it resolves + pulls the index itself
    # when compute_index_alpha was off.
    if event_study:
        _run_event_study(
            metrics, graph, ticker, price_provider, trades,
            index_name, end_date,
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
            "execution_lag_bars": execution_lag_bars,
            "execution_price": "next available close",
            "periods_per_year": periods_per_year,
            "rating_to_weight": dict(rating_to_weight or DEFAULT_RATING_TO_WEIGHT),
            "index_benchmark": index_name,
            "risk_warnings": sorted({
                t.risk_warning for t in trades if t.risk_warning
            }),
        },
        cached_hits=cached_hits,
        cached_misses=cached_misses,
        degraded_decision_count=degraded_decision_count,
        unexecuted_decision_count=unexecuted_decision_count,
    )


def _augment_metrics(
    metrics: BacktestMetrics,
    trades: list[TradeRow],
    equity: list[float],
    equity_dates: list[str],
    total_traded_notional: float,
    periods_per_year: int,
) -> None:
    """Fill the trade-/equity-derived metric fields in place.

    :func:`compute_metrics` is a pure function of the equity curve (no trade
    rows, no calendar), so win-rate / annualized turnover / max-drawdown date --
    which need exactly those -- are computed here. Deterministic post-processing
    only: no I/O, no effect on any agent input or on the equity-derived metrics
    themselves.
    """
    # Win rate is based on net P&L of actual long-position episodes.  Signal
    # direction and the asset's hypothetical forward return are not trades:
    # an all-cash strategy therefore has no win rate instead of reporting 100%
    # merely because the underlying happened to rise.
    position_returns = [
        float(t.position_return)
        for t in trades
        if t.position_return is not None and np.isfinite(t.position_return)
    ]
    if position_returns:
        metrics.win_rate = sum(ret > 0.0 for ret in position_returns) / len(position_returns)
    metrics.num_trades = len(position_returns)

    # Annualized turnover: traded notional per unit of average equity, per year.
    if equity and len(equity) >= 2 and periods_per_year > 0:
        years = (len(equity) - 1) / periods_per_year
        mean_equity = sum(float(x) for x in equity) / len(equity)
        if years > 0 and mean_equity > 0:
            metrics.turnover_annual = total_traded_notional / (mean_equity * years)

    # Date the equity curve bottomed relative to its running peak (pairs with
    # ``max_drawdown``). On a monotonically rising curve drawdown is always 0
    # and argmin returns the first index, so the date is the window start.
    if len(equity) >= 2 and len(equity_dates) == len(equity):
        eq = np.asarray(equity, dtype=float)
        running_max = np.maximum.accumulate(eq)
        drawdowns = eq / running_max - 1.0
        trough_idx = int(np.argmin(drawdowns))
        metrics.max_drawdown_date = equity_dates[trough_idx]


def _attribute_factors(
    metrics: BacktestMetrics,
    equity_curve: list[float],
    equity_dates: list[str],
    start_date: str,
    end_date: str,
    factor_model: str,
    periods_per_year: int,
) -> None:
    """Fill the Fama-French attribution fields on ``metrics`` in place.

    Loads French daily factor returns over the backtest window, builds the
    strategy's daily return series from the equity curve, regresses excess
    returns on the factor matrix, and stamps ``factor_model`` / ``factor_alpha``
    / ``factor_betas`` / ``factor_r_squared``. Pure post-processing: no agent
    interaction, no effect on any other metric. Fail-open on every step so a
    bad/offline factor file degrades to "no attribution" rather than aborting.
    """
    try:
        from yiagents.backtest.factor_model import (
            factor_attribution,
            label_for,
            load_factor_returns,
        )

        factors = load_factor_returns(start_date, end_date, model=factor_model)
        if factors is None or factors.empty:
            return
        # returns_from_equity yields len(equity)-1 returns; pair them with the
        # dates of each return's *end* point (equity_dates[1:]).
        rets = returns_from_equity(equity_curve)
        if len(rets) != len(equity_dates) - 1 or len(rets) < 2:
            return
        strat = pd.Series(rets, index=pd.to_datetime(equity_dates[1:]))
        label = label_for(factor_model)
        attr = factor_attribution(
            strat, factors, periods_per_year=periods_per_year, model=label
        )
        if attr is None:
            return
        metrics.factor_model = label
        metrics.factor_alpha = attr.alpha_annual
        metrics.factor_betas = attr.betas
        metrics.factor_r_squared = attr.r_squared
    except Exception as exc:  # noqa: BLE001 -- attribution is advisory; never break a run
        logger.warning(
            "factor attribution failed for %s..%s (%s): %s",
            start_date, end_date, factor_model, exc,
        )


def _run_event_study(
    metrics: BacktestMetrics,
    graph: _GraphLike,
    ticker: str,
    price_provider: Callable[[str, str, str], pd.Series],
    trades: list[TradeRow],
    index_name: str,
    end_date: str,
) -> None:
    """Fill the event-study fields on ``metrics`` in place.

    Runs a market-model event study over the rebalance dates: for each event,
    fit ``R_asset = a + b*R_benchmark`` on a preceding estimation window, then
    test whether cumulative abnormal returns in the holding window are reliably
    non-zero. Pure post-processing: no agent interaction, no effect on any
    other metric. Fail-open on every step so a bad/missing price window or too
    few decidable events degrades to "no event study" rather than aborting.

    ``run_backtest``'s ``prices`` start at the first decision date, but the
    estimation window needs ~250 trading days *before* the earliest event, so a
    wider asset + benchmark window is re-pulled here. The benchmark is resolved
    via the same helper the index-alpha path uses (``_resolve_index_benchmark``)
    and pulled over the wide window; run_backtest's narrower ``index_prices``
    copy is not reused because it does not cover the estimation window.
    """
    try:
        from datetime import datetime, timedelta

        from yiagents.backtest.event_study import event_study as _event_study

        if not trades:
            return
        # The price reaction window starts when the signal was executable, not
        # at the already-consumed signal-day close.
        event_dates = [t.execution_date or t.date for t in trades]
        ratings = [t.rating for t in trades]

        # Widen backwards ~400 calendar days (~270 trading days) to cover the
        # default 250-return estimation window + pre-event gap before event[0].
        first_dt = datetime.strptime(str(event_dates[0]), "%Y-%m-%d")
        wide_start = (first_dt - timedelta(days=400)).strftime("%Y-%m-%d")

        asset_prices = price_provider(ticker, wide_start, end_date).sort_index()
        if asset_prices.empty:
            return

        # Benchmark must span the SAME wide window as the asset (the estimation
        # window sits before the first event), so pull it from wide_start
        # regardless of whether compute_index_alpha already fetched a copy.
        bench_name = index_name or _resolve_index_benchmark(graph, ticker)
        bench = price_provider(bench_name, wide_start, end_date).sort_index()
        if bench.empty:
            return

        result = _event_study(
            asset_prices, bench, event_dates,
            ratings=ratings,
            benchmark_name=bench_name or "benchmark",
        )
        if result.n_events == 0:
            return
        metrics.event_study_n = result.n_events
        metrics.event_study_mean_car = result.mean_car
        metrics.event_study_t_stat = result.t_stat
        metrics.event_study_p_value = result.p_value
        metrics.event_study_ci = (
            (result.ci_low, result.ci_high)
            if result.ci_low is not None and result.ci_high is not None
            else None
        )
        metrics.event_study_benchmark = result.benchmark
    except Exception as exc:  # noqa: BLE001 -- event study is advisory; never break a run
        logger.warning("event study failed for %s: %s", ticker, exc)


def _resolve_decision(
    graph: _GraphLike,
    ticker: str,
    trade_date: str,
    asset_type: str,
    cache: DecisionCache | None,
    run_tag: str,
) -> tuple[str, str, bool, bool]:
    """Return ``(rating, final_decision_markdown, was_cached, was_degraded)``.

    Uses the cache when available; otherwise calls the graph. The graph returns
    ``(final_state, rating)``; the markdown decision lives in
    ``final_state['final_trade_decision']``.

    ``was_degraded`` is True when the decision is a fallback rather than a
    genuine agent output — ``propagate`` raised (rating forced to Hold) or the
    rating was not a parseable string. This lets the backtest engine count
    degraded decisions separately so they are not silently confused with real
    "Hold" calls. (``parse_rating`` fallback also emits its own warning via
    ``warn_on_default=True``.)
    """
    if cache is not None:
        cached = cache.get(ticker, trade_date, run_tag)
        if cached is not None:
            return cached.rating, cached.final_decision, True, False

    final_state, rating, propagate_failed = _call_graph(graph, ticker, trade_date, asset_type)
    was_degraded = propagate_failed or not isinstance(rating, str)
    if propagate_failed or not isinstance(rating, str):
        rating = "Hold"
    else:
        rating = parse_rating(rating, warn_on_default=True)

    decision_md = ""
    if isinstance(final_state, Mapping):
        decision_md = str(final_state.get("final_trade_decision", ""))  # type: ignore[union-attr]
    elif isinstance(final_state, dict):
        decision_md = str(final_state.get("final_trade_decision", ""))

    if cache is not None:
        cache.remember(ticker, trade_date, rating, decision_md, run_tag)
    return rating, decision_md, False, was_degraded


def _call_graph(
    graph: _GraphLike, ticker: str, trade_date: str, asset_type: str,
) -> tuple[Any, Any, bool]:
    """Invoke propagate defensively; a failing node should not abort the backtest.

    Returns ``(final_state, rating, propagate_failed)``. ``propagate_failed`` is
    True when an exception was caught and the rating was forced to ``"Hold"``,
    so callers can distinguish a degraded fallback from a genuine Hold.
    """
    try:
        final_state, rating = graph.propagate(ticker, trade_date, asset_type=asset_type)
        return final_state, rating, False
    except Exception as exc:  # noqa: BLE001 -- one bad date must not kill the run
        logger.warning("propagate failed for %s on %s: %s (treating as Hold)", ticker, trade_date, exc)
        return {"final_trade_decision": f"[propagate error: {exc}]"}, "Hold", True


def _excerpt(text: str, limit: int = 200) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")
