"""Phase 0 backtest engine: run the agent graph over history and price the result.

The engine is the measuring stick for the whole profitability roadmap. It loops
``graph.propagate(ticker, date)`` over a list of decision dates, translates each
5-tier rating into a target portfolio weight, simulates holding that weight on a
daily mark-to-market calendar, and returns an equity curve plus a full metric
suite (delegated to :mod:`yialpha.backtest.metrics`).

Design choices driven by the roadmap:

* **Reuse, do not rebuild.** The graph already produces the rating and already
  owns PIT-safe data loading; the engine only adds the portfolio simulation on
  top. ``_fetch_returns`` / ``_resolve_benchmark`` are reused where useful.
* **Pluggable sizing.** ``rating_to_weight`` (the simple baseline mapping) is the
  default. Phase 1 swaps in a ``weight_fn`` driven by the risk layer so the
  *same* realized decisions can be re-priced under different risk rules -- the
  A/B comparison every later phase runs against.
* **Honest about LLM cost/non-determinism.** A :class:`~yialpha.backtest.cache.DecisionCache`
  memoizes realized decisions per ``(ticker, date, run_tag)``; the multi-run
  distribution comes from re-realizing under fresh ``run_tag`` values, not from
  silently re-billing the LLM on every replay.
* **Hermetic in tests.** A ``price_provider`` callable lets tests inject
  synthetic prices; production uses a yfinance-backed default.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from yialpha.backtest.cache import DecisionCache
from yialpha.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    returns_from_equity,
    trade_quality_stats,
)
from yialpha.dataflows.vol_estimators import periods_per_year_for

# V2.0 P0.1: the Binance fee constants moved to yialpha.risk.cost_model as
# the single source of truth shared with the decision-time tradeability
# gate; they are re-exported below so existing imports keep working.
from yialpha.risk.cost_model import (  # noqa: F401  (re-export)
    BINANCE_USDT_M_MAKER_BPS,
    BINANCE_USDT_M_TAKER_BPS,
    BNB_FEE_DISCOUNT,
)

if TYPE_CHECKING:
    # Engine-side corporate-action wiring (V2.4) is shape-only: the records
    # arrive fully built from the caller, so the dataclass is needed for the
    # signature annotation alone and never at import time.
    from yialpha.instruments.corporate_actions import CorporateActionRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# V2.4 perp-class split: the perp-family classes and the classification seam.
# The engine historically keyed perp semantics on the raw CLI asset_type
# string ("crypto_perp"); a tokenized-stock perp enters through the SAME
# string, so the string alone cannot tell a BTCUSDT from a MUUSDT run. The
# seam delegates to the ONE shared classifier (yialpha.graph.routing) —
# registry/warm consults are flag-gated and fetch-free — so a stock perp
# routed via any entrance gets perp semantics where the gates use it.
# ---------------------------------------------------------------------------
#: The perpetual instrument family (see
#: :func:`yialpha.graph.routing.instrument_class`).
_PERP_INSTRUMENT_CLASSES = frozenset(
    {"stock_perp", "pure_crypto_perp", "unknown_perp"}
)

#: Per-``(asset_type, symbol)`` memo for the classification seam. Routing
#: never fetches (warmed snapshot or static seed, flag-gated registry), so one
#: classification per instrument per process is enough; tests clear this dict
#: alongside ``refresh_equity_perp_bases`` to keep classification ordering-
#: deterministic.
_INSTRUMENT_CLASS_CACHE: dict[tuple[str, str], str] = {}


def _perp_instrument_class(asset_type: str, symbol: str) -> str:
    """Classify one instrument via the shared routing seam, memoized.

    Delegates to :func:`yialpha.graph.routing.instrument_class` (equity /
    crypto_spot / stock_perp / pure_crypto_perp / unknown_perp) — the same
    predicate every other entrance consults, so the engine and the live graph
    can never disagree about what an instrument is. The import stays inside
    the function to keep the engine importable without the routing graph and
    to honor test monkeypatching of the routing classifier. The pure fallback
    is the routing function's own logic (cache-or-seed, fetch-free).
    """
    key = (str(asset_type), str(symbol))
    cached = _INSTRUMENT_CLASS_CACHE.get(key)
    if cached is not None:
        return cached
    from yialpha.graph.routing import instrument_class as _routing_class

    klass = str(_routing_class(asset_type, symbol))
    _INSTRUMENT_CLASS_CACHE[key] = klass
    return klass


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

# The same map with shorting enabled (perp ``allow_short``): Sell goes to a
# full short instead of flat. Underweight stays flat — the 5-tier scale has
# no "weak short" rung, so inventing -0.8 would be a silent strategy change.
SHORT_RATING_TO_WEIGHT: dict[str, float | None] = {
    **DEFAULT_RATING_TO_WEIGHT,
    "Sell": -1.0,
}


class _GraphLike(Protocol):
    """Structural type the engine needs from a YiAlphaGraph."""

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

    from yialpha.dataflows.symbol_utils import normalize_symbol

    canonical = normalize_symbol(ticker)
    hist = yf.Ticker(canonical).history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return pd.Series(dtype=float)
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    s = hist["Close"].dropna()
    s.index = s.index.strftime("%Y-%m-%d")
    return s.astype(float)


def _binance_perp_price_provider(price_type: str = "last"):
    """Daily close-price provider backed by the perp's OWN Binance klines.

    The engine's historical default marks a perp backtest on Yahoo SPOT data
    (BTC-USD) — wrong instrument, wrong basis. This provider prices and marks
    on the contract the strategy actually trades, via the shared PIT-clamped
    data layer; ``price_type="mark"`` serves the mark-price klines Binance
    liquidates against.
    """

    def provider(ticker: str, start: str, end: str) -> pd.Series:
        from ..dataflows.binance import binance_klines_frame

        df = binance_klines_frame(
            ticker, start, end, interval="1d", venue="binance_perp",
            price_type=price_type,
        )
        s = df["Close"].dropna()
        s.index = s.index.strftime("%Y-%m-%d")
        return s.astype(float)

    return provider


def _binance_perp_extremes_provider(price_type: str = "last"):
    """Daily (low, high) series for intrabar liquidation/stop triggers.

    Liquidation is checked against the bar's adverse extreme (low for longs,
    high for shorts), not only the close — a wick through the liquidation
    price is a forced close even when the bar recovers. Returns a callable
    ``(ticker, start, end) -> (lows, highs)`` on the perp's own kline data;
    ``price_type="mark"`` serves the mark-price wicks Binance actually
    liquidates against, ``"last"`` (default) the ordinary last-traded book
    that resting stop orders live on.
    """

    def provider(ticker: str, start: str, end: str) -> tuple[pd.Series, pd.Series]:
        from ..dataflows.binance import binance_klines_frame

        df = binance_klines_frame(
            ticker, start, end, interval="1d", venue="binance_perp",
            price_type=price_type,
        )
        lows = df["Low"].dropna()
        highs = df["High"].dropna()
        lows.index = lows.index.strftime("%Y-%m-%d")
        highs.index = highs.index.strftime("%Y-%m-%d")
        return lows.astype(float), highs.astype(float)

    return provider


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
            # The fallback keeps the backtest alive, but alpha/beta vs SPY are
            # plain wrong for A-share/crypto tickers — the substitution must
            # be observable, not silent.
            logger.warning(
                "benchmark resolution failed for %s; falling back to SPY "
                "(alpha/beta will be computed against the wrong index)",
                ticker, exc_info=True,
            )
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


def _binance_funding_provider(ticker: str, start: str, end: str) -> pd.Series:
    """Daily aggregated funding rates for a Binance USDT-M perp.

    Returns a float Series indexed by ``YYYY-MM-DD`` strings; each value is
    the SUM of that UTC day's settlement rates (most perps settle every 8h →
    ~3 entries/day; 4h/1h contracts have more). A positive sum means longs
    paid that day — the drag a long-only perp strategy must carry.
    """
    from datetime import datetime

    from ..dataflows.binance import _FAPI_FUNDING_LIMIT, _paginate_history
    from ..dataflows.symbol_utils import normalize_symbol_for_venue
    from ..dataflows.utils import current_pit_end

    canonical = normalize_symbol_for_venue(ticker, "binance_perp")
    # Same PIT contract as every vendor call: a backtest for a past date must
    # never receive settlements after it (benign today only because the
    # backtest end is already past, but the engine must not rely on that).
    end = current_pit_end(end) or end
    start_ms = int(
        datetime.strptime(start, "%Y-%m-%d")
        .replace(tzinfo=UTC).timestamp() * 1000
    )
    end_ms = int(
        (
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
            + 86399
        ) * 1000
    )
    rows = _paginate_history(
        "/fapi/v1/fundingRate", {"symbol": canonical}, _FAPI_FUNDING_LIMIT,
        lambda r: r["fundingTime"], start_ms, end_ms, ticker, canonical,
    )
    agg: dict[str, float] = {}
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict) or r.get("fundingTime") is None:
            continue
        day = datetime.fromtimestamp(
            int(r["fundingTime"]) / 1000, tz=UTC
        ).strftime("%Y-%m-%d")
        try:
            raw_rate = r.get("fundingRate")
            rate = float(raw_rate) if raw_rate is not None else None
        except (TypeError, ValueError):
            continue
        if rate is None:
            continue
        agg[day] = agg.get(day, 0.0) + rate
    return pd.Series(agg, index=list(agg), dtype=float)


def _action_symbol_universe(ticker: str) -> frozenset[str]:
    """Symbols a corporate-action record may carry and still apply.

    Accepts the exact instrument symbol (``MUUSDT``) and its base form
    (``MU`` — the underlying the action is actually about), with dashed
    perp spellings normalized. A record for any OTHER symbol is data for a
    different instrument and is ignored, never mis-applied.
    """
    compact = str(ticker).strip().upper().replace("-", "")
    symbols = {compact}
    for suffix in ("USDT", "USDC"):
        if compact.endswith(suffix) and len(compact) > len(suffix):
            symbols.add(compact[: -len(suffix)])
    return frozenset(symbols)


def _split_ratio(record: CorporateActionRecord) -> float | None:
    """New-shares-per-old-share from a split record's ``details``, else None.

    ``details["ratio"]`` wins (``2.0`` = a 2:1 split halves prior prices);
    the ``details["to"] / details["from"]`` pair is accepted as the equally
    common venue spelling (``{"to": 2, "from": 1}``).
    """
    details = record.details or {}
    raw_ratio = details.get("ratio")
    if raw_ratio is not None:
        try:
            return float(raw_ratio)
        except (TypeError, ValueError):
            return None
    raw_to, raw_from = details.get("to"), details.get("from")
    if raw_to is not None and raw_from is not None:
        try:
            to_value, from_value = float(raw_to), float(raw_from)
        except (TypeError, ValueError):
            return None
        if from_value != 0.0:
            return to_value / from_value
    return None


def _apply_corporate_actions(
    prices: pd.Series,
    ticker: str,
    records: Sequence[CorporateActionRecord],
) -> tuple[pd.Series, int]:
    """Back-adjust ``prices`` for splits/dividends; returns ``(series, n)``.

    V2.4 honest skeleton: records arrive from the caller (there is no vendor
    and no storage yet — ``None``, today's only production value, changes
    nothing). Splits adjust MULTIPLICATIVELY (close bars strictly before the
    ``event_date`` divide by the split ratio) and dividends ADDITIVELY (prior
    bars subtract the per-share ``details["amount"]``); the ``event_date``
    bar itself is the ex bar and stays unadjusted. Records apply in
    chronological order so chained actions compose on the running adjusted
    series, and the adjusted series feeds the mark-to-market, the
    buy-and-hold benchmark and the asset-return windows alike (funding and
    bar extremes stay on raw contract prices — they are venue observables,
    not underlying economics). Malformed records raise ``ValueError``
    (:func:`~yialpha.instruments.corporate_actions.validate_corporate_action`
    plus ratio/amount checks): a wrong adjustment is worse than no backtest.
    ``merger`` / ``ticker_change`` / ``other`` records are identity actions
    with no price adjustment in this skeleton. The returned count is the
    number of records that actually moved a price bar.
    """
    from yialpha.instruments.corporate_actions import validate_corporate_action

    universe = _action_symbol_universe(ticker)
    matching = [
        record
        for record in records
        if str(record.symbol).strip().upper().replace("-", "") in universe
    ]
    adjusted = prices
    applied = 0
    for record in sorted(matching, key=lambda r: str(r.event_date)):
        validate_corporate_action(record)
        prior = adjusted.index < str(record.event_date)
        if not prior.any():
            continue
        if record.action_type == "split":
            ratio = _split_ratio(record)
            if ratio is None or not np.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(
                    f"split record for {record.symbol!r} effective "
                    f"{record.event_date!r} needs a positive finite ratio "
                    f"(details['ratio'], or details['to']/'from'); got "
                    f"{record.details!r}"
                )
            values = adjusted.to_numpy(dtype=float, copy=True)
            values[prior] /= ratio
            adjusted = pd.Series(values, index=adjusted.index, dtype=float)
            applied += 1
        elif record.action_type == "dividend":
            raw_amount = (record.details or {}).get("amount")
            try:
                amount = float(raw_amount) if raw_amount is not None else None
            except (TypeError, ValueError):
                amount = None
            if amount is None or not np.isfinite(amount) or amount <= 0.0:
                raise ValueError(
                    f"dividend record for {record.symbol!r} effective "
                    f"{record.event_date!r} needs a positive finite per-share "
                    f"details['amount']; got {record.details!r}"
                )
            values = adjusted.to_numpy(dtype=float, copy=True)
            values[prior] -= amount
            if (values[prior] <= 0.0).any():
                # A dividend larger than a prior price is a units error (raw
                # vs split-adjusted amounts mixed); fail closed rather than
                # emit a non-positive price the simulator would choke on.
                raise ValueError(
                    f"dividend adjustment for {record.symbol!r} effective "
                    f"{record.event_date!r} drives a prior close to <= 0; "
                    "check the amount's share-count basis"
                )
            adjusted = pd.Series(values, index=adjusted.index, dtype=float)
            applied += 1
    return adjusted, applied


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
    funding_provider: Callable[[str, str, str], pd.Series] | None = None,
    corporate_actions: Sequence[CorporateActionRecord] | None = None,
    periods_per_year: int | None = None,
    cost_bps: float = 0.0,
    taker_bps: float | None = None,
    slippage_bps: float = 0.0,
    bnb_discount: bool = False,
    filters_provider: Callable[[str], Any] | None = None,
    allow_funding_gaps: bool = False,
    leverage: float = 1.0,
    allow_short: bool = False,
    simulate_stop_triggers: bool | None = None,
    liquidation_price_type: str | None = None,
    brackets_provider: Callable[[str], list] | None = None,
    extremes_provider: Callable[[str, str, str], tuple[pd.Series, pd.Series]] | None = None,
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
        returning ``(final_state, rating)`` (a :class:`YiAlphaGraph`).
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
    funding_provider:
        Daily funding-rate source for ``asset_type="crypto_perp"`` — a callable
        ``(ticker, start, end) -> pd.Series`` indexed by ``YYYY-MM-DD`` whose
        values are each day's summed settlement rates. Defaults to the Binance
        USDT-M vendor (paginated ``/fapi/v1/fundingRate``, PIT-clamped). The
        perp mode charges the funding drag on held notional — longs pay
        positive rates, shorts (``allow_short``) receive them — for the
        strategy and the buy-and-hold benchmark alike. Missing funding DATA
        fails closed (ValueError), and so do missing COVERAGE days (a day
        absent from the series would silently accrue zero drag;
        ``allow_funding_gaps=True`` accepts that approximation explicitly).
    corporate_actions:
        Optional caller-supplied corporate-action records for the traded
        instrument or its underlying (V2.4 honest skeleton — no vendor, no
        fetch, no storage; ``None``, today's only production value, changes
        nothing). When provided, split records divide the close bars
        STRICTLY before their ``event_date`` by the split ratio
        (``details["ratio"]``, or ``details["to"]/details["from"]``) and
        dividend records subtract the per-share ``details["amount"]`` from
        those bars, BEFORE any returns are computed — the adjusted series
        feeds the strategy, the buy-and-hold benchmark and the asset-return
        windows alike. Records whose ``symbol`` is neither the instrument
        symbol nor its base are ignored; the applied count is disclosed as
        ``config_summary["corporate_actions_applied"]``.
    taker_bps / slippage_bps / bnb_discount / filters_provider:
        Execution-cost model for perp fills: taker fee in bps on the traded
        notional (``taker_bps=None`` falls back to the single ``cost_bps``),
        adverse slippage in bps baked into the fill price, an optional 10% BNB
        discount, and an optional ``filters_provider(ticker) -> SymbolFilters``
        for exchange order rules (fills floor the share delta to stepSize and
        refuse sub-minNotional deltas). Defaults reproduce the historical
        single-``cost_bps`` behaviour exactly.
    leverage / brackets_provider / extremes_provider:
        Perp-only leverage (default 1x = the historical cash simulation,
        byte-identical). ``leverage > 1`` scales the target weight's notional
        and models isolated-margin liquidation against the maintenance-margin
        ladder — from ``/fapi/v1/leverageBracket`` when operator API keys
        exist, else the documented default ladder, else an injected
        ``brackets_provider``. The trigger checks the bar's adverse extreme
        (from ``extremes_provider``, defaulting to Binance perp kline
        low/high) rather than only the close. Liquidation events are recorded
        in ``config_summary["perp_liquidations"]``.
    allow_short:
        Perp-only opt-in short side: ``Sell`` maps to weight -1.0, positions
        and funding flip sign, and liquidation triggers on the upside. Default
        off keeps the long-only semantics byte-identical.
    simulate_stop_triggers:
        Simulate the risk overlay's ATR stop as a GTC stop-market order: a
        bar whose adverse extreme crosses the stop level force-exits at the
        stop price (plus adverse slippage and the taker fee), even when the
        bar closes back through it. ``None`` (default) enables this for
        ``asset_type="crypto_perp"`` and keeps spot/stock runs byte-identical
        (the stop stays advisory metadata, as historically). Runs only when a
        ``weight_fn`` actually supplies ``risk_decision.stop_loss``; the
        baseline rating→weight table never arms a stop. Liquidation is
        checked first — a long's stop always sits above its liquidation
        level, so a bar piercing both is conservatively liquidated. Exits are
        recorded in ``config_summary["perp_stop_triggers"]``.
    liquidation_price_type:
        ``"mark"`` or ``"last"`` — which price series triggers the liquidation
        check. Binance USDT-M liquidates on the MARK price while the last
        price is what the equity curve marks on, so ``"mark"`` (the default
        for crypto_perp) fetches mark-price kline wicks for the trigger while
        keeping last-price marking; ``"last"`` restores the single-series
        behaviour. Only meaningful with ``leverage > 1``; an injected
        ``extremes_provider`` serves as the trigger series verbatim whatever
        this says. If mark klines cannot be fetched the engine falls back to
        last-price extremes LOUDLY — warning plus
        ``config_summary["perp_liq_price_source"]`` disclosure — never
        silently relabelling a last-price check as mark.
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
    if not (np.isfinite(leverage) and leverage >= 1.0):
        raise ValueError(f"leverage must be a finite number >= 1.0, got {leverage!r}")
    # V2.4 perp-class split: the perp-family gates key on the shared routing
    # classification, not the raw asset_type string — a tokenized-stock perp
    # (or an unresolvable perp) enters through asset_type="crypto_perp" and
    # must get perp semantics instead of the spot/stock rejection. Pure-crypto
    # perp and non-perp behaviour is unchanged (the family membership of
    # "crypto_perp"+BTCUSDT and of every non-perp asset_type is the same
    # verdict the string comparison produced).
    instrument_klass = _perp_instrument_class(asset_type, ticker)
    is_perp_family = instrument_klass in _PERP_INSTRUMENT_CLASSES
    if leverage != 1.0 and not is_perp_family:
        raise ValueError(
            "leverage is a crypto_perp-only parameter (perp family: "
            "stock_perp / pure_crypto_perp / unknown_perp); a levered "
            "spot/stock simulation would be a margin model this engine "
            "does not have"
        )
    if allow_short and not is_perp_family:
        raise ValueError(
            "allow_short is a crypto_perp-only parameter (perp family: "
            "stock_perp / pure_crypto_perp / unknown_perp); shorting "
            "spot/stock involves locate/borrow mechanics this engine "
            "does not model"
        )
    if liquidation_price_type is not None and liquidation_price_type not in ("mark", "last"):
        raise ValueError(
            f"liquidation_price_type must be 'mark' or 'last', got "
            f"{liquidation_price_type!r}"
        )
    if liquidation_price_type == "mark" and not is_perp_family:
        raise ValueError(
            "mark-price liquidation is a crypto_perp-only concept "
            "(perp family: stock_perp / pure_crypto_perp / unknown_perp; "
            "markPriceKlines is a perp endpoint)"
        )
    if slippage_bps < 0.0 or (taker_bps is not None and taker_bps < 0.0) or cost_bps < 0.0:
        raise ValueError("cost_bps / taker_bps / slippage_bps must be >= 0")
    if periods_per_year is None:
        # V2.2 class-aware annualization (V2.4 wires it into the backtest):
        # a stock perp annualizes at 365 like a pure crypto perp — klines
        # machine evidence (2026-09-04) shows Binance stock perps trade 24/7
        # (full US-market holidays and weekends carry volume), so the retired
        # 261 weekday count understated vol — and an unresolvable perp keeps
        # the 252 equity convention. Only the perp classes REFINE the
        # answer — passing e.g.
        # the "crypto_spot" class would drag spot crypto to 252 and break the
        # historical asset-type rule, so every non-perp class resolves via
        # the None-class path (crypto 365 / everything else 252, unchanged).
        periods_per_year = int(
            periods_per_year_for(
                asset_type, instrument_klass if is_perp_family else None,
            )
        )
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be >= 1")

    if allow_short and rating_to_weight is None and weight_fn is None:
        rating_to_weight = SHORT_RATING_TO_WEIGHT
    weight_fn = weight_fn or _default_weight_fn(rating_to_weight or DEFAULT_RATING_TO_WEIGHT)

    # --- Price window: span every decision date plus one holding period ----
    sorted_dates = sorted(str(d) for d in dates)
    start_date = sorted_dates[0]
    # Buffer the end so the last decision still has a full holding window to mark.
    from datetime import datetime, timedelta
    end_dt = datetime.strptime(sorted_dates[-1], "%Y-%m-%d") + timedelta(days=holding_days + 10)
    end_date = end_dt.strftime("%Y-%m-%d")

    # Perp mode prices/marks on the perp's own candles: the historical default
    # (Yahoo spot, e.g. BTC-USD) is the wrong instrument with the wrong basis.
    # An explicit provider always wins — tests inject synthetic prices this
    # way. The benchmark index keeps the ORIGINAL provider: index names are
    # yfinance-shaped (SPY / 000300.SS), never Binance symbols.
    index_price_provider = price_provider
    price_provider_swapped = False
    if is_perp_family and price_provider is _yfinance_price_provider:
        price_provider = _binance_perp_price_provider()
        price_provider_swapped = True

    prices = price_provider(ticker, start_date, end_date)
    if prices.empty:
        raise ValueError(
            f"No price data for {ticker} between {start_date} and {end_date}; "
            "cannot mark the backtest to market."
        )
    prices = prices.sort_index()

    # --- Corporate-action price adjustment (V2.4 honest skeleton) -----------
    # Splits/dividends on the underlying are mechanics, not P&L: adjust the
    # close series BEFORE any return, mark or benchmark is computed. None
    # (the only production value today) leaves the series byte-identical.
    corporate_actions_applied = 0
    if corporate_actions:
        prices, corporate_actions_applied = _apply_corporate_actions(
            prices, ticker, corporate_actions,
        )

    # --- Perp funding (long-only crypto_perp mode, 2026-08-16) ---------------
    # The engine remains a cash simulator at 1x; for a USDT-M perpetual it
    # additionally charges the daily funding drag on the held notional (longs
    # pay positive funding). Fail-closed: a perp backtest without funding
    # history would silently relabel a spot simulation, so missing funding
    # data raises instead — and so do COVERAGE gaps: a day absent from the
    # funding series used to default to 0.0 drag, silently flattering a
    # delisted/short-history contract (allow_funding_gaps overrides for
    # research runs that accept the approximation).
    perp_funding: pd.Series | None = None
    funding_paid_total = 0.0
    if is_perp_family:
        provider = funding_provider or _binance_funding_provider
        try:
            perp_funding = provider(ticker, start_date, end_date)
        except Exception as exc:  # noqa: BLE001 — converted to the loud gate below
            raise ValueError(
                f"crypto_perp backtest requires funding-rate history for "
                f"{ticker}; the funding fetch failed "
                f"({type(exc).__name__}: {exc}). Pass funding_provider with "
                "the data, or use asset_type='crypto' for an explicit "
                "spot-style simulation."
            ) from exc
        if perp_funding is None or perp_funding.empty:
            raise ValueError(
                f"crypto_perp backtest requires funding-rate history; none "
                f"was returned for {ticker} between {start_date} and "
                f"{end_date}. Use asset_type='crypto' for a spot-style "
                "simulation instead."
            )
        if not allow_funding_gaps:
            funding_days = {str(d) for d in perp_funding.index}
            # Days the simulator will actually charge: every price bar from
            # the first decision onward (earlier bars never accrue drag). The
            # FINAL bar is tolerated: a still-open current day can have
            # settlements pending, and the last decision's window is buffered.
            charge_days = [str(d) for d in prices.index if str(d) >= sorted_dates[0]]
            missing = [d for d in charge_days[:-1] if d not in funding_days]
            if missing:
                shown = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
                raise ValueError(
                    f"crypto_perp funding history for {ticker} is missing "
                    f"{len(missing)} priced day(s) ({shown}); those days would "
                    "silently accrue zero drag. Pass allow_funding_gaps=True "
                    "to accept the approximation explicitly."
                )

    # --- Execution-cost model (2026-08-16) -----------------------------------
    # Fills are market orders at the daily close: taker fee (BNB-discountable)
    # on the traded notional plus adverse slippage baked into the fill price.
    # Defaults (taker_bps=None, slippage_bps=0, bnb off) reproduce the
    # historical single cost_bps exactly.
    fee_rate = (taker_bps if taker_bps is not None else cost_bps) / 10_000.0
    if bnb_discount:
        fee_rate *= 1.0 - BNB_FEE_DISCOUNT
    slip_rate = slippage_bps / 10_000.0

    # Exchange order rules (stepSize / minNotional), when a filters source is
    # available: simulated fills floor the share DELTA to the symbol's step so
    # the simulation cannot trade quantities the venue would reject. Offline
    # runs pass filters_provider=None and fill fractionally (stated in
    # config_summary).
    symbol_filters: Any = None
    if filters_provider is not None:
        try:
            symbol_filters = filters_provider(ticker)
        except Exception as exc:  # noqa: BLE001 — quantization is an accuracy add-on
            logger.warning(
                "filters unavailable for %s (%s: %s); fills stay fractional",
                ticker, type(exc).__name__, exc,
            )
            symbol_filters = None

    # --- Leverage / margin / liquidation (2026-08-16, perp-only) -------------
    # leverage=1.0 keeps the historical cash simulation byte-identical: no
    # brackets are resolved, no liquidation is modeled. leverage>1 sizes the
    # target weight's notional up by L (margin = notional / L inside the same
    # cash account) and models isolated-margin forced closes against the
    # maintenance-margin ladder: liquidation price = entry x (1 - 1/L + MMR +
    # fee buffer), triggered by the bar's adverse extreme when extremes are
    # available (else the close), exited at the liquidation price or the worse
    # close when the bar gapped through it.
    model_liquidation = is_perp_family and leverage > 1.0
    # Stop-trigger simulation defaults ON for perps (a levered strategy whose
    # overlay publishes a stop must be priced with that stop actually firing)
    # and OFF everywhere else so spot/stock runs stay byte-identical.
    simulate_stops = (
        simulate_stop_triggers
        if simulate_stop_triggers is not None
        else is_perp_family
    )
    track_entry_basis = model_liquidation or simulate_stops
    stop_events: list[dict[str, Any]] = []
    mmr_brackets: list | None = None
    mmr_source = ""
    liq_events: list[dict[str, Any]] = []
    # Liquidation trigger series vs stop trigger series: Binance USDT-M
    # liquidates on the MARK price while resting stop orders live on the
    # last-traded book, and the equity curve marks on last-price closes —
    # three different observables, so "mark" mode (the perp default) fetches
    # mark kline wicks for the liquidation check and last kline wicks for
    # stops instead of conflating them into one series.
    liq_price_type = liquidation_price_type or (
        "mark" if is_perp_family else "last"
    )
    liq_price_source = ""
    bar_lows: pd.Series | None = None      # liquidation trigger series
    bar_highs: pd.Series | None = None
    stop_lows: pd.Series | None = None     # stop trigger series (last price)
    stop_highs: pd.Series | None = None
    if model_liquidation:
        from ..dataflows.binance_brackets import (
            default_brackets,
            get_leverage_brackets,
        )

        if brackets_provider is not None:
            mmr_brackets = list(brackets_provider(ticker))
            mmr_source = "injected"
        else:
            try:
                mmr_brackets = get_leverage_brackets(ticker)
                mmr_source = "leverageBracket"
            except Exception as exc:  # noqa: BLE001 — documented approximation below
                logger.warning(
                    "leverageBracket unavailable for %s (%s: %s); using the "
                    "default USDT-M MMR ladder (approximation — per-symbol "
                    "ladders differ)", ticker, type(exc).__name__, exc,
                )
                mmr_brackets = default_brackets()
                mmr_source = "default-ladder"

    def _fetch_extremes_or_none(
        price_type: str,
    ) -> tuple[pd.Series, pd.Series] | None:
        try:
            return _binance_perp_extremes_provider(price_type)(
                ticker, start_date, end_date,
            )
        except Exception as exc:  # noqa: BLE001 — close-only trigger is the fallback
            logger.warning(
                "%s-price bar extremes unavailable for %s (%s: %s); the "
                "matching trigger checks run on closes only",
                price_type, ticker, type(exc).__name__, exc,
            )
            return None

    if model_liquidation or simulate_stops:
        if extremes_provider is not None:
            # An injected provider IS the synthetic world: it serves both
            # trigger roles verbatim, whatever liq_price_type says.
            try:
                lows, highs = extremes_provider(ticker, start_date, end_date)
            except Exception as exc:  # noqa: BLE001 — close-only fallback
                logger.warning(
                    "bar extremes unavailable for %s (%s: %s); liquidation/"
                    "stop checks run on closes only",
                    ticker, type(exc).__name__, exc,
                )
            else:
                bar_lows, bar_highs = lows, highs
                stop_lows, stop_highs = lows, highs
                if model_liquidation:
                    liq_price_source = "injected"
        elif price_provider_swapped:
            if simulate_stops:
                got = _fetch_extremes_or_none("last")
                if got is not None:
                    stop_lows, stop_highs = got
            if model_liquidation:
                if liq_price_type == "mark":
                    got = _fetch_extremes_or_none("mark")
                    if got is not None:
                        bar_lows, bar_highs = got
                        liq_price_source = "mark"
                    else:
                        # Loud fallback: never relabel a last-price check as
                        # mark — the config discloses what actually ran.
                        got = _fetch_extremes_or_none("last")
                        if got is not None:
                            bar_lows, bar_highs = got
                            liq_price_source = "last (mark unavailable)"
                else:
                    got = _fetch_extremes_or_none("last")
                    if got is not None:
                        bar_lows, bar_highs = got
                        liq_price_source = "last"
        # else: an explicitly injected price provider (tests, offline
        # research) has no matching Binance extremes source — fetching a
        # real-market low/high series for synthetic prices would be worse
        # than useless, so extremes stay None and triggers run on closes.

    index_prices: pd.Series | None = None
    index_name = ""
    if compute_index_alpha:
        index_name = _resolve_index_benchmark(graph, ticker)
        try:
            index_prices = index_price_provider(index_name, start_date, end_date).sort_index()
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
    pos_avg_entry = 0.0            # weighted-average fill price of the open position
    current_stop: float | None = None  # armed GTC stop from the risk overlay

    for trade_date, price in prices.items():
        if trade_date < sorted_dates[0]:
            continue

        px = float(price)
        if px <= 0.0 or not np.isfinite(px):
            raise ValueError(f"Invalid execution price for {ticker} on {trade_date}: {price!r}")

        # Perp funding drag: charge the day's settlements on the marked
        # notional BEFORE recording equity, so the equity curve, the sizing
        # context and the metrics all see the post-funding value. Daily
        # granularity attributes the day's settlements (UTC) to that day's
        # close — the standard bar-level approximation. Signed shares make
        # the direction automatic: longs PAY positive funding, shorts RECEIVE it.
        if perp_funding is not None and shares != 0.0:
            rate = float(perp_funding.get(str(trade_date), 0.0) or 0.0)
            if rate and np.isfinite(rate):
                charge = shares * px * rate
                cash -= charge
                funding_paid_total += charge

        # Isolated-margin liquidation check (perp, leverage > 1 only). Runs
        # after funding, before the mark, so the equity curve reflects the
        # forced close on its own day. Trigger level: entry x (1 - 1/L + MMR
        # + fee buffer) for longs (mirrored for shorts); the bar's adverse
        # extreme triggers it even when the close recovers. The forced close
        # executes AT the trigger level — in this cash-account model that
        # credit equals the isolated-margin remainder exactly (margin + P&L
        # at the trigger price), so the loss stays capped near the posted
        # margin even when the bar gapped far through the level (gap
        # slippage beyond that is the exchange's liquidation fee / insurance
        # mechanics, approximated by the fee charged below).
        if model_liquidation and shares != 0.0:
            from ..dataflows.binance_brackets import mmr_for_notional as _mmr_for

            notional = abs(shares) * px
            mmr = _mmr_for(mmr_brackets, notional)
            if shares > 0.0:
                liq_price = pos_avg_entry * (1.0 - 1.0 / leverage + mmr + fee_rate)
                adverse = (
                    float(bar_lows.get(str(trade_date), px))
                    if bar_lows is not None else px
                )
                triggered = adverse <= liq_price
            else:
                liq_price = pos_avg_entry * (1.0 + 1.0 / leverage - mmr - fee_rate)
                adverse = (
                    float(bar_highs.get(str(trade_date), px))
                    if bar_highs is not None else px
                )
                triggered = adverse >= liq_price
            if triggered:
                exit_px = liq_price
                exit_notional = abs(shares) * exit_px
                liq_fee = exit_notional * fee_rate
                cash += shares * exit_px - liq_fee
                liq_events.append({
                    "date": str(trade_date),
                    "side": "long" if shares > 0.0 else "short",
                    "shares": shares,
                    "entry": pos_avg_entry,
                    "liquidation_price": liq_price,
                    "exit_price": exit_px,
                    "fee": liq_fee,
                })
                shares = 0.0
                pos_avg_entry = 0.0
                current_stop = None  # position gone; the resting order dies with it

        # Stop-trigger simulation: the overlay's ATR stop behaves like a GTC
        # stop-market order — the bar's adverse extreme crossing the level
        # force-exits AT the stop price with adverse slippage and the taker
        # fee, even when the bar closes back through it (a gap far beyond the
        # level fills at the level itself, the same documented bar-level
        # approximation the liquidation exit makes). Runs AFTER the
        # liquidation check: a long's stop always sits above its liquidation
        # level, so a bar piercing both is conservatively liquidated first.
        # A signal landing on this same bar may still re-enter at the close.
        if simulate_stops and shares != 0.0 and current_stop is not None:
            if shares > 0.0:
                adverse = (
                    float(stop_lows.get(str(trade_date), px))
                    if stop_lows is not None else px
                )
                hit = adverse <= current_stop
            else:
                adverse = (
                    float(stop_highs.get(str(trade_date), px))
                    if stop_highs is not None else px
                )
                hit = adverse >= current_stop
            if hit:
                exit_px = (
                    current_stop * (1.0 - slip_rate) if shares > 0.0
                    else current_stop * (1.0 + slip_rate)
                )
                stop_fee = abs(shares) * exit_px * fee_rate
                cash += shares * exit_px - stop_fee
                stop_events.append({
                    "date": str(trade_date),
                    "side": "long" if shares > 0.0 else "short",
                    "shares": shares,
                    "entry": pos_avg_entry,
                    "stop_price": current_stop,
                    "exit_price": exit_px,
                    "fee": stop_fee,
                })
                shares = 0.0
                pos_avg_entry = 0.0
                current_stop = None

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
                # Lets a risk weight_fn state an accurate caveat: whether the
                # engine will actually simulate the stop it is being handed.
                "stop_simulation_mode": simulate_stops,
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
                weight_lo = -1.0 if allow_short else 0.0
                target_weight = float(np.clip(requested_weight, weight_lo, 1.0))
                # Leverage scales the target weight's NOTIONAL (margin =
                # notional / L inside the same cash account): at L=1 this is
                # the historical target weight verbatim.
                desired_value = target_weight * leverage * equity_before
                delta_notional = desired_value - current_value
                fill_px = (
                    px * (1.0 + slip_rate) if delta_notional > 0.0
                    else px * (1.0 - slip_rate)
                )
                # Exchange order rules: floor the share DELTA to the symbol's
                # stepSize and refuse sub-minNotional deltas, so the simulator
                # never trades a quantity the venue would reject. Fractional
                # fills (no filters / offline) are the documented fallback.
                delta_shares = delta_notional / px
                if symbol_filters is not None:
                    from ..dataflows.binance_filters import quantize_order

                    q = quantize_order(fill_px, abs(delta_shares), symbol_filters)
                    qty = q["quantity"]
                    stepped = float(qty) if isinstance(qty, (int, float)) else 0.0
                    if q["below_min_qty"] or q["below_min_notional"]:
                        stepped = 0.0
                    delta_shares = math.copysign(stepped, delta_shares) if stepped else 0.0
                prev_shares = shares
                shares = prev_shares + delta_shares
                traded_notional = abs(delta_shares) * px
                cost = abs(delta_shares) * fill_px * fee_rate
                # Cash leg pays/receives the slipped fill price per share; at
                # slip=0 and fee=cost_bps this reduces to the historical
                # (current_value - desired_value) - cost exactly.
                cash += -delta_shares * fill_px - cost
                # Weighted-average entry basis for the liquidation/stop levels:
                # extending a position blends fill prices; REDUCING one keeps
                # the basis (a trim does not change where the surviving
                # contracts entered); flipping restarts it at the new fill.
                # (The old condition mis-routed any partial reduction into
                # the flip branch, resetting the basis to the trim's fill and
                # shifting the liquidation level after every rebalance-down.)
                if track_entry_basis:
                    if prev_shares == 0.0:
                        if shares != 0.0:
                            pos_avg_entry = fill_px
                    elif shares == 0.0:
                        pos_avg_entry = 0.0
                    elif (shares > 0.0) != (prev_shares > 0.0):
                        # flip through zero: restart at the new fill
                        pos_avg_entry = fill_px
                    elif delta_shares != 0.0 and (delta_shares > 0.0) == (shares > 0.0):
                        total = abs(prev_shares) + abs(delta_shares)
                        pos_avg_entry = (
                            pos_avg_entry * abs(prev_shares)
                            + fill_px * abs(delta_shares)
                        ) / total
                    # else: same-direction reduction — basis unchanged

                # GTC stop semantics: this decision's stop replaces the
                # resting order while a position stays open; a full close
                # disarms it. A decision without a stop keeps the previous
                # level resting (broker GTC behaviour). The direction guard
                # rejects stops that sit on the wrong side of the entry —
                # e.g. the overlay's long-formula stop surviving a flip to a
                # short — instead of arming an instantly-triggering order.
                if simulate_stops:
                    candidate = getattr(risk_decision, "stop_loss", None)
                    # Arm/replace with this decision's stop when it is valid
                    # for the resulting position.
                    if candidate is not None and shares != 0.0:
                        valid = (
                            candidate < pos_avg_entry if shares > 0.0
                            else candidate > pos_avg_entry
                        )
                        if valid:
                            current_stop = float(candidate)
                    # Disarm: position fully closed, or a flip left the old
                    # stop on the wrong side of the entry.
                    if current_stop is not None:
                        if shares == 0.0:
                            current_stop = None
                        else:
                            still_valid = (
                                current_stop < pos_avg_entry if shares > 0.0
                                else current_stop > pos_avg_entry
                            )
                            if not still_valid:
                                current_stop = None

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

            was_invested = abs(previous_weight) > 1e-12
            is_invested = abs(executed_weight) > 1e-12
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
        # Buy-and-hold pays the same funding drag (it is also a perp long) —
        # charging only the strategy would bias the comparison in its favour.
        # The charge runs on the start-of-day position, mirroring the strategy
        # loop (which charges before the day's fills): a position acquired at
        # THIS close has not yet sat through any settlement, so the entry day
        # itself pays nothing. Charging after the entry block used to bill
        # B&H one extra funding day per backtest and inflate alpha_vs_buyhold.
        if perp_funding is not None and bh_shares > 0.0:
            rate = float(perp_funding.get(date, 0.0) or 0.0)
            if rate and np.isfinite(rate):
                bh_cash -= bh_shares * px * rate
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
                     total_traded_notional, periods_per_year, ticker)

    # Perp run facts as first-class metrics (None for every non-perp run):
    # the forced-exit tallies and the signed funding totals behind the
    # config_summary event lists, so downstream consumers (report, web,
    # multi-run comparisons) do not have to re-derive them from configs.
    if is_perp_family:
        charge_day_count = max(
            1, sum(1 for d in prices.index if str(d) >= sorted_dates[0]),
        )
        metrics.liquidation_count = len(liq_events)
        metrics.stop_trigger_count = len(stop_events)
        metrics.funding_paid_total = round(funding_paid_total, 2)
        metrics.funding_drag_annualized = round(
            (funding_paid_total / initial_capital) / charge_day_count
            * periods_per_year, 6,
        )

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
            # V2.4 perp-class split: the resolved routing classification the
            # run actually keyed on (perp gates / annualization).
            "instrument_class": instrument_klass,
            "run_tag": run_tag,
            "cost_bps": cost_bps,
            "execution_lag_bars": execution_lag_bars,
            "execution_price": "next available close",
            "periods_per_year": periods_per_year,
            # Calendar disclosure: a stock perp's 24/7 calendar is klines-
            # VERIFIED (2026-09-04 machine evidence: MUUSDT traded with
            # volume through every full US-market holiday and all weekend
            # bars, ~1/4–1/3 of a normal day's volume; zero-volume days = 0),
            # so the 365 factor is measured, not assumed. The disclosure
            # rides the config summary of every stock-perp backtest.
            **(
                {
                    "session_calendar_assumption": (
                        f"{periods_per_year} sessions/year — klines-verified "
                        "24/7 calendar (2026-09-04 probe: full US-market "
                        "holidays and weekends traded with volume); factor "
                        "measured, not assumed"
                    )
                }
                if instrument_klass == "stock_perp"
                else {}
            ),
            "rating_to_weight": dict(rating_to_weight or DEFAULT_RATING_TO_WEIGHT),
            "index_benchmark": index_name,
            "risk_warnings": sorted({
                t.risk_warning for t in trades if t.risk_warning
            }),
            **(
                {
                    "perp_funding_drag": True,
                    "perp_funding_paid_total": round(funding_paid_total, 2),
                    "perp_fees": (
                        f"taker {round(fee_rate * 10_000.0, 4)}bps"
                        + (" x0.9 BNB" if bnb_discount else "")
                        + f", slippage {slippage_bps}bps"
                    ),
                    "perp_fill_quantization": (
                        "stepSize/minNotional" if symbol_filters is not None
                        else "fractional (no filters source)"
                    ),
                    "perp_price_source": (
                        "binance perp klines"
                        if is_perp_family
                        and index_price_provider is _yfinance_price_provider
                        and price_provider is not index_price_provider
                        else "caller-provided"
                    ),
                    "perp_leverage": leverage,
                    "perp_short": allow_short,
                    "perp_stop_simulation": simulate_stops,
                    "perp_stop_trigger_count": len(stop_events),
                    **(
                        {"perp_liq_price_source": liq_price_source}
                        if model_liquidation and liq_price_source else {}
                    ),
                    "perp_model_note": (
                        "USDT-M perp simulation: daily funding drag (strategy "
                        "and buy-and-hold), taker fees + slippage on fills"
                        + (
                            ", isolated-margin liquidation vs the "
                            f"{mmr_source} MMR ladder triggered on "
                            + ("mark-price" if liq_price_source == "mark"
                               else "bar")
                            + " extremes" if model_liquidation
                            else "; leverage/margin/liquidation NOT modeled "
                            "(leverage=1x cash sim)"
                        )
                        + (
                            ", overlay stop-losses simulated as GTC stop-market "
                            "orders triggered on bar extremes" if simulate_stops
                            else "; stop-losses advisory only (not simulated)"
                        )
                        + (", opt-in short side" if allow_short else
                           "; long-only")
                        + ". Buy-and-hold stays a 1x long."
                    ),
                }
                if is_perp_family else {}
            ),
            **(
                {"corporate_actions_applied": corporate_actions_applied}
                if corporate_actions else {}
            ),
            **(
                {"perp_liquidations": liq_events} if liq_events else {}
            ),
            **(
                {"perp_stop_triggers": stop_events} if stop_events else {}
            ),
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
    ticker: str = "",
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

    # Trade-quality battery over the same closed episodes (profit factor,
    # average win/loss, payoff). Same fail-open contract: no closed trades
    # leaves the fields at None.
    quality = trade_quality_stats(position_returns)
    metrics.profit_factor = quality["profit_factor"]
    metrics.avg_win = quality["avg_win"]
    metrics.avg_loss = quality["avg_loss"]
    metrics.payoff_ratio = quality["payoff_ratio"]
    # compute_metrics filled IR/TE against the buy-and-hold curve; name that
    # benchmark explicitly so the report can say what it was measured against.
    metrics.benchmark_name = f"{ticker} buy-and-hold"

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
        from yialpha.backtest.factor_model import (
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

        from yialpha.backtest.event_study import event_study as _event_study

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
    "Hold" calls.

    Cache policy (fail-closed on infrastructure faults):

    * A degraded-but-genuine decision (propagate succeeded, rating text was
      unparseable) IS cached — the decision markdown is real agent output, so
      a replay must not re-bill the LLM for it.
    * A ``propagate`` failure is NOT cached. The fake Hold exists only because
      one node raised (e.g. a transient network fault); caching it would make
      every later replay return the fake Hold as ``was_degraded=False`` and
      permanently distort win-rate / DSR. A replay instead retries the graph,
      which is the honest measurement.
    """
    if cache is not None:
        cached = cache.get(ticker, trade_date, run_tag)
        if cached is not None:
            return cached.rating, cached.final_decision, True, False

    final_state, rating, propagate_failed = _call_graph(graph, ticker, trade_date, asset_type)
    # ``propagate`` already returns a canonical 5-tier rating (it runs
    # ``parse_rating`` on the PM's markdown via ``process_signal``), so re-
    # parsing here would only duplicate the default-fallback warning. Trust
    # the contract; anything that is not a non-empty string is degraded.
    was_degraded = propagate_failed or not isinstance(rating, str) or not rating
    if was_degraded:
        rating = "Hold"

    decision_md = ""
    if isinstance(final_state, Mapping):
        decision_md = str(final_state.get("final_trade_decision", ""))

    if cache is not None and not propagate_failed:
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
