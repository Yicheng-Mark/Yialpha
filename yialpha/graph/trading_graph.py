# yialpha/graph/trading_graph.py

import json
import logging
import os
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.prebuilt import ToolNode

# Import the abstract tool methods from agent_utils
from yialpha.agents.utils.agent_utils import (
    build_instrument_context,
    get_a_share_balance_sheet_native,
    get_a_share_cashflow_statement_native,
    get_a_share_dragon_tiger_native,
    get_a_share_fundamentals_native,
    get_a_share_income_statement_native,
    get_a_share_market_breadth_native,
    get_a_share_money_flow_native,
    get_a_share_news_native,
    get_a_share_northbound_native,
    get_a_share_ohlc_native,
    get_a_share_realtime_quote_native,
    get_a_share_sector_flow_native,
    get_balance_sheet,
    get_binance_basis,
    get_binance_depth_snapshot,
    get_binance_funding_rate,
    get_binance_indicators,
    get_binance_klines,
    get_binance_long_short_ratio,
    get_binance_open_interest,
    get_binance_premium_index,
    get_binance_spot_indicators,
    get_binance_spot_klines,
    get_binance_spot_perp_basis,
    get_binance_spot_ticker24,
    get_binance_taker_buy_sell,
    get_binance_vision_book_depth,
    get_binance_vision_metrics,
    get_candlestick_patterns,
    get_cashflow,
    get_form4_insider_trading,
    get_ftd_data,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_indicators_weekly,
    get_insider_transactions,
    get_institutional_holdings,
    get_macro_indicators,
    get_margin_trading,
    get_news,
    get_prediction_markets,
    get_relative_strength,
    get_stock_data,
    get_support_resistance,
    get_verified_market_snapshot,
    get_volume_features,
    resolve_instrument_identity,
    web_search,
    web_search_fundamentals,
    web_search_market,
)
from yialpha.agents.utils.memory import TradingMemoryLog
from yialpha.agents.utils.pot_tool import make_pot_compute_tool
from yialpha.agents.utils.prediction_tools import submit_prediction
from yialpha.agents.utils.valuation_tools import get_valuation_metrics
from yialpha.dataflows.binance import stock_perp_underlying, warm_equity_perp_bases
from yialpha.dataflows.config import set_config
from yialpha.dataflows.utils import safe_ticker_component, set_analysis_date
from yialpha.default_config import DEFAULT_CONFIG
from yialpha.llm_clients import create_llm_client
from yialpha.reporting import write_report_tree

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .overlay_fields import OVERLAY_MARKER
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)

# Stale-pending observability for the memory log lives in
# yialpha.graph.memory_resolution (extracted with the resolution core so the
# graph and the `yialpha memory-resolve` CLI share one implementation).


@lru_cache(maxsize=4096)
def _memoized_close_and_atr(
    ticker: str, trade_date: str, asset_type: str = "stock",
) -> tuple[float, float]:
    """Load and compute the deterministic PIT ``(close, atr)`` for one date.

    Module-level LRU keyed by ``(ticker, trade_date, asset_type)`` — see
    ``YiAlphaGraph._latest_close_and_atr`` for the PIT-safety argument.
    Raises on any failure so failures are never cached (the caller converts
    them to ``(None, None)`` and retries on the next run).

    ``asset_type`` selects the price venue. A crypto_perp run must not price
    its overlay (entry reference / ATR stop / position value) on the Yahoo
    series ``load_ohlcv`` normalizes BTCUSDT into (BTC-USD spot): that is a
    different instrument trading at a different basis than the Binance
    perp the rest of the run analyzes, so a stop computed from it sits at
    the wrong level. Perp runs load the perp's own daily candles via
    ``binance_klines_frame`` (PIT-clamped by the same analysis-date
    ContextVar the graph pins in ``_run_graph``) and compute the identical
    14-period ATR on them. All other asset types keep the historical
    ``load_ohlcv`` path byte-for-byte (crypto spot deliberately stays on
    the Yahoo index source it has always used).
    """
    from yialpha.risk.atr_stop import latest_atr_from_frame

    if asset_type == "crypto_perp":
        from datetime import datetime, timedelta

        from yialpha.dataflows.binance import binance_klines_frame

        lookback = (
            datetime.strptime(str(trade_date), "%Y-%m-%d") - timedelta(days=120)
        ).strftime("%Y-%m-%d")
        frame = binance_klines_frame(ticker, lookback, str(trade_date))
        close, atr = latest_atr_from_frame(frame)
        return float(close), float(atr)

    from yialpha.dataflows.stockstats_utils import load_ohlcv

    frame = load_ohlcv(ticker, str(trade_date))
    close, atr = latest_atr_from_frame(frame)
    return float(close), float(atr)


@lru_cache(maxsize=4096)
def _memoized_mark_close(ticker: str, trade_date: str) -> float:
    """Mark-price close for one date (perp only; mirrors the (close, ATR)
    memo's PIT argument). Mark is the price Binance liquidates against —
    the overlay reports it next to the last-price entry reference so the
    liquidation distance claims carry their true trigger basis. Raises on
    failure so failures are never cached; the caller degrades to None."""
    from datetime import datetime, timedelta

    from yialpha.dataflows.binance import KLINE_CLOSE_COLUMN, binance_klines_frame

    lookback = (
        datetime.strptime(str(trade_date), "%Y-%m-%d") - timedelta(days=10)
    ).strftime("%Y-%m-%d")
    frame = binance_klines_frame(
        ticker, lookback, str(trade_date), "1d", "binance_perp", "mark",
    )
    return float(frame[KLINE_CLOSE_COLUMN].iloc[-1])


class YiAlphaGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=False,
        config: dict[str, Any] | None = None,
        callbacks: list | None = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # T0: optional node-level perf telemetry. Create the shared tracker up
        # front so it can be (a) wrapped around every node handler in setup.py,
        # (b) fed LLM tokens via NodePerfTokenCallback below, and (c)
        # serialized to node_perf_<date>.json at the end of each run. Off by
        # default = no tracker, no wrapping, byte-identical to today.
        self.perf_tracker = None
        if self.config.get("node_perf_telemetry"):
            from yialpha.graph.perf_telemetry import (
                NodePerfTokenCallback,
                NodePerfTracker,
            )
            self.perf_tracker = NodePerfTracker()
            # The token callback rides the LLM callbacks channel so on_llm_end
            # attributes token usage to the active node (set by wrap_node).
            self.callbacks = list(self.callbacks) + [
                NodePerfTokenCallback(self.perf_tracker)
            ]

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # T3: optional per-call LLM response disk cache (langchain global
        # set_llm_cache). Installed BEFORE any LLM is created/invoked so the
        # first call is already covered. Off by default = no cache installed,
        # byte-equivalent. See yialpha/llm_clients/response_cache.py for the
        # distribution-safety caveat (never use with the A/B gate / DSR runs).
        if self.config.get("llm_cache"):
            from langchain_core.globals import set_llm_cache

            from yialpha.llm_clients.response_cache import DiskLLMCache

            _llm_cache_dir = Path(self.config["data_cache_dir"]) / "llm_responses"
            set_llm_cache(DiskLLMCache(_llm_cache_dir))
            logger.info(
                "LLM response cache enabled at %s (iteration replay only; "
                "do NOT use with analyst A/B gate or DSR multi-run).",
                _llm_cache_dir,
            )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        # Debate-tier client: bull/bear researchers + risk debators. Falls back
        # to deep_think_llm when debate_llm is unset (None), so the adversarial-
        # argumentation layer runs on the same model as the final-decision nodes
        # unless the user overrides YIALPHA_DEBATE_LLM for A/B testing.
        debate_model = self.config["debate_llm"] or self.config["deep_think_llm"]
        debate_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=debate_model,
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        self.debate_llm = debate_client.get_llm()

        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.debate_llm,
            self.tool_nodes,
            self.conditional_logic,
            perf_tracker=self.perf_tracker,
            analyst_parallel=self.config.get("analyst_parallel", False),
            analyst_parallel_max_threads=self.config.get(
                "analyst_parallel_max_threads", 16
            ),
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # Phase 1: optional quantitative risk overlay. Built only when the user
        # opts in via risk_enabled; otherwise every node behaves as before and
        # the Phase-0 baseline stays reproducible.
        self._risk_overlay_degraded = False
        self.risk_manager = self._build_risk_manager()

        # State tracking
        self.curr_state: dict[str, Any] | None = None
        self.ticker: str | None = None
        self.selected_analysts = tuple(selected_analysts)  # for P0 telemetry plan
        self.log_states_dict: dict[str, dict[str, Any]] = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx: AbstractContextManager[Any] | None = None

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        # Sampling temperature is cross-provider: forward it whenever set.
        # float() here so a value coming from a YIALPHA_TEMPERATURE env
        # string ("0.2") works the same as a programmatic float.
        temperature = self.config.get("temperature")
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)

        # Phase C: optional shared rate limiter — throttles REQUEST RATE only
        # (never prompts/reasoning/temperature). One limiter per (provider, rpm)
        # is shared across every worker graph in the process, so the RPM ceiling
        # bounds the whole batch rather than each graph independently. Attached
        # only for OpenAI-compatible providers (DeepSeek, OpenAI, xAI, ...) whose
        # client forwards it via _PASSTHROUGH_KWARGS. Off by default; enable only
        # after measuring the ceiling (G4 instrumentation).
        if self.config.get("llm_rate_limiter"):
            from yialpha.llm_clients.openai_client import is_openai_compatible
            if is_openai_compatible(provider):
                from yialpha.llm_clients.rate_limiter import get_shared_rate_limiter
                kwargs["rate_limiter"] = get_shared_rate_limiter(
                    provider, int(self.config.get("llm_rpm", 60))
                )

        # P1a: optional shared httpx.Client so the K worker graphs reuse TLS/proxy
        # connections. Transport-only; off by default. Forwarded for OpenAI-
        # compatible providers via _PASSTHROUGH_KWARGS (http_client is allowlisted).
        if self.config.get("http_keepalive"):
            from yialpha.llm_clients.openai_client import is_openai_compatible
            if is_openai_compatible(provider):
                from yialpha.llm_clients.http_client import get_shared_http_client
                shared = get_shared_http_client()
                if shared is not None:
                    kwargs["http_client"] = shared

        # T1.3: per-call retry count forwarded to provider clients via
        # _PASSTHROUGH_KWARGS (max_retries). Default 2 == langchain-openai's
        # own default, so the success path is byte-equivalent to today; only
        # the failure/retry path can differ when a user lowers it. Exposed as
        # YIALPHA_LLM_MAX_RETRIES so flaky periods can tune it down (the outer
        # safety net is run_robust's per-ticker rerun, not in-call retry).
        kwargs["max_retries"] = int(self.config.get("llm_max_retries", 2))

        return kwargs

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                    # Deterministic verification snapshot (bound to the analyst
                    # LLM and required by its prompt; must be executable here or
                    # the call fails and the model reports it "unavailable").
                    get_verified_market_snapshot,
                    # Price-structure evidence tools + weekly timeframe
                    # (2026-08-15 TA expansion). The market analyst's prompt
                    # MANDATES citations from these five tools for stock (and
                    # crypto_spot) runs, and its bind_tools list includes them —
                    # they must therefore be executable here. Missing from this
                    # map, every mandated call would die with "not a valid tool"
                    # and the evidence chain the prompt demands would silently
                    # degrade to model memory (P0 wiring gap, round-5 audit).
                    get_indicators_weekly,
                    get_support_resistance,
                    get_volume_features,
                    get_candlestick_patterns,
                    get_relative_strength,
                    # Binance USDT-M perp tools. Dormant for non-perp runs: the
                    # market analyst only advertises them when asset_type ==
                    # "crypto_perp", so the LLM never names them otherwise and
                    # they are simply extra entries in ToolNode's name->tool map.
                    get_binance_klines,
                    get_binance_funding_rate,
                    get_binance_open_interest,
                    get_binance_long_short_ratio,
                    get_binance_taker_buy_sell,
                    get_binance_basis,
                    get_binance_premium_index,
                    # Live order book + data.binance.vision archive tools
                    # (deep-history positioning/depth, PIT-correct for
                    # historical replay dates). Same dormant contract.
                    get_binance_depth_snapshot,
                    get_binance_vision_metrics,
                    get_binance_vision_book_depth,
                    # Classic indicators on Binance candles, one binding per
                    # venue default (perp / spot); dormant unless the matching
                    # crypto run advertises them.
                    get_binance_indicators,
                    get_binance_spot_indicators,
                    # Binance SPOT tools. Same dormant contract as the perp
                    # tools above: only advertised when asset_type ==
                    # "crypto_spot", so they sit unused in the name->tool map
                    # for stock/crypto/perp runs.
                    get_binance_spot_klines,
                    get_binance_spot_ticker24,
                    get_binance_spot_perp_basis,
                    # Native A-share market signal tools (a_share_native gate on
                    # the analyst side). Same dormant contract: only advertised
                    # when YIALPHA_A_SHARE_NATIVE is on AND the ticker is an
                    # A-share, so they sit unused otherwise.
                    get_a_share_northbound_native,
                    get_a_share_sector_flow_native,
                    get_a_share_realtime_quote_native,
                    get_a_share_market_breadth_native,
                    # Open-web search (Tavily), market-scoped instance. The
                    # market analyst binds it only when web_search_enabled is
                    # on AND the run date is live (PIT: Tavily has no as-of
                    # parameter), so it sits dormant otherwise. The vendor
                    # degrades to a WEB_SEARCH_UNAVAILABLE sentinel + quality
                    # event on key-missing / budget-exhausted, so registering
                    # it here is always safe (never aborts a run). Same tool
                    # NAME as the news/fundamentals instances — each analyst's
                    # ToolNode holds its own scope-charging instance.
                    web_search_market,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            # V2.3 positioning split. The positioning analyst is
            # structured-output (no tool loop): its report call binds NO
            # tools and its blind-prediction round is dispatched by the node
            # itself, so this ToolNode exists purely to satisfy the serial
            # plan's (agent <-> tool <-> clear) wiring contract. The generic
            # submit_prediction registration makes the (never-expected) case
            # of a stray submit_prediction tool call resolvable instead of a
            # hard "not a valid tool" failure. Dormant by construction.
            "positioning": ToolNode([submit_prediction]),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                    get_macro_indicators,
                    get_prediction_markets,
                    # Native A-share news (a_share_native). Dormant for default
                    # runs: the news analyst only advertises it when
                    # YIALPHA_A_SHARE_NATIVE is on AND the ticker is an A-share
                    # (.SS/.SH/.SZ), so the LLM never names it otherwise and it
                    # is simply an extra entry in ToolNode's name->tool map (same
                    # dormant contract as the fundamentals A-share tools).
                    get_a_share_news_native,
                    # Open-web search (Tavily). Gated on the analyst side by
                    # web_search_enabled; the vendor itself degrades to a
                    # WEB_SEARCH_UNAVAILABLE sentinel + quality event when the
                    # key is missing / budget is exhausted, so registering it
                    # here is always safe (never aborts a run).
                    web_search,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                    # SEC ownership & short-interest tools (Track B2). Dormant for
                    # default runs: the fundamentals analyst only advertises them
                    # when YIALPHA_SEC_OWNERSHIP is on, so the LLM never names
                    # them otherwise and they are simply extra entries in
                    # ToolNode's name->tool map (same dormant contract as the
                    # Binance tools in the market ToolNode).
                    get_form4_insider_trading,
                    get_ftd_data,
                    get_institutional_holdings,
                    # China A-share margin-trading tool (Track A). Dormant for
                    # default runs: the fundamentals analyst only advertises it
                    # when YIALPHA_A_STOCK is on AND the ticker is an A-share
                    # (.SS/.SH/.SZ), so the LLM never names it otherwise and it
                    # is simply an extra entry in ToolNode's name->tool map
                    # (same dormant contract as the SEC/Binance tools above).
                    get_margin_trading,
                    # Native A-share OHLC + TTM valuation (a_share_native). Dormant
                    # for default runs: the fundamentals analyst only advertises
                    # them when YIALPHA_A_SHARE_NATIVE is on AND the ticker is an
                    # A-share (.SS/.SH/.SZ), so the LLM never names them otherwise
                    # and they are simply extra entries in ToolNode's name->tool
                    # map (same dormant contract as the margin tool above).
                    get_a_share_fundamentals_native,
                    get_a_share_ohlc_native,
                    # Native A-share financials (a_share_native gate): the
                    # fundamentals analyst appends these five alongside the two
                    # above when the flag and ticker gates both hold; they must
                    # be executable in the same ToolNode or those calls fail
                    # (same wiring-gap class as the market price-structure
                    # tools, round-5 audit).
                    get_a_share_money_flow_native,
                    get_a_share_dragon_tiger_native,
                    get_a_share_income_statement_native,
                    get_a_share_balance_sheet_native,
                    get_a_share_cashflow_statement_native,
                    # Deterministic valuation + PoT compute (valuation_tools
                    # gate). The analyst binds make_pot_compute_tool(llm) with
                    # the SAME quick-tier client this graph was built with; the
                    # ToolNode instance below is a second closure over that
                    # same client, registered under the identical tool name so
                    # routed calls execute with the configured model.
                    get_valuation_metrics,
                    make_pot_compute_tool(self.quick_thinking_llm),
                    # Open-web search (Tavily), fundamentals-scoped instance.
                    # Same dormant contract as web_search_market above: bound
                    # by the analyst only when web_search_enabled is on AND
                    # the run date is live; always safe to register.
                    web_search_fundamentals,
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        Thin delegation to :func:`yialpha.dataflows.market_regime.
        resolve_market_benchmark` (the dataflow-layer implementation this
        method was previously a verbatim copy of — the two drifted once
        already). ``config["benchmark_ticker"]`` overrides everything when
        set; otherwise the suffix map matches the ticker's exchange suffix
        and the empty-suffix entry (SPY by default) is the fallback, which is
        right because the alpha calculation works in USD.
        """
        from yialpha.dataflows.market_regime import resolve_market_benchmark

        # The dataflow resolver reads the same config keys through
        # get_config(); the graph's self.config is the same mapping source.
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        return resolve_market_benchmark(ticker)

    def _build_risk_manager(self):
        """Construct the Phase-1 RiskManager when ``risk_enabled`` is set.

        Returns ``None`` when risk control is off so the rest of the graph can
        branch cheaply on a single truthy check. On build failure, sets
        ``self._risk_overlay_degraded`` so :meth:`_apply_risk_overlay` can mark
        the decision document visibly instead of silently dropping the overlay.
        """
        if not self.config.get("risk_enabled"):
            return None
        from yialpha.risk.manager import RiskManager
        try:
            return RiskManager.from_config(self.config)
        except Exception as exc:  # noqa: BLE001 -- never block a run on risk setup
            logger.warning("risk_enabled is set but RiskManager build failed (%s); "
                           "running without the risk overlay", exc)
            self._risk_overlay_degraded = True
            return None

    def _latest_close_and_atr(self, ticker, trade_date, asset_type: str = "stock"):
        """Best-effort (close, atr) as of ``trade_date`` for the risk overlay.

        PIT-safe: reuses the project's cached, date-truncated OHLCV loader.
        Any failure returns ``(None, None)`` so the overlay still runs without
        a stop rather than aborting the decision.

        Memoized per ``(ticker, trade_date, asset_type)`` at module level: the
        14-period stockstats ATR for a fixed PIT date is deterministic, and
        backtests hit the same key repeatedly (A/B legs, multi-run
        distributions). ``asset_type`` is part of the key so the same ticker
        on different venues (BTCUSDT perp vs Yahoo BTC-USD spot) never
        cross-contaminates cache entries. The date being part of the key is
        what keeps this PIT-safe — a new analysis date re-computes rather
        than reusing a prior day's value. Failures are NOT memoized (an
        lru_cache on a raising function re-invokes next call), so a transient
        vendor fault still retries on the next run.
        """
        try:
            return _memoized_close_and_atr(ticker, str(trade_date), asset_type)
        except Exception as exc:  # noqa: BLE001
            logger.warning("risk overlay could not load price/ATR for %s on %s: %s",
                           ticker, trade_date, exc)
            if asset_type == "crypto_perp":
                # The overlay's price book is a CORE input (PR3): its
                # absence must reach the quality chain — DEGRADED_CRITICAL →
                # ticket NO_TRADE — instead of vanishing into a "Stop-loss
                # not set" warning on an otherwise-tradable-looking report.
                from yialpha.dataflows import quality as _quality

                _quality.record_sentinel(
                    "get_binance_klines",
                    _quality.KIND_OPTIONAL_UNAVAILABLE,
                    f"{ticker}: overlay price/ATR unavailable for {trade_date}: "
                    f"{type(exc).__name__}: {exc}",
                )
            return None, None

    def _latest_mark_close(self, ticker, trade_date) -> float | None:
        """Best-effort mark-price close (perp overlay bullet; None degrades).

        Mark is a CORE price leg on a perp run (liquidations trigger on it):
        a fetch failure records the same core sentinel the bundle's price
        legs do, so a decision without its mark basis cannot pass as clean.
        """
        try:
            return _memoized_mark_close(ticker, str(trade_date))
        except Exception as exc:  # noqa: BLE001
            logger.info("mark close unavailable for %s on %s: %s",
                        ticker, trade_date, exc)
            from yialpha.dataflows import quality as _quality

            _quality.record_sentinel(
                "get_binance_klines",
                _quality.KIND_OPTIONAL_UNAVAILABLE,
                f"{ticker}: overlay mark-price close unavailable for {trade_date}: "
                f"{type(exc).__name__}: {exc}",
            )
            return None

    @staticmethod
    def _risk_disabled_warning(reason: str) -> str:
        """A visible markdown warning appended when the risk overlay is skipped.

        Placed at the *end* of the decision text so ``parse_rating`` still reads
        the PM's leading ``**Rating**:`` line first. This makes a silent fail-open
        visible to anyone reading the decision report.
        """
        return (
            "\n\n---\n\n## ⚠️ Quantitative Risk Overlay DISABLED\n\n"
            f"The risk overlay layer failed to run for this decision ({reason}). "
            "The rating and thesis above are unchanged, but **no Kelly sizing, "
            "ATR stop-loss, drawdown breaker, or CVaR protection was applied**. "
            "Treat the position sizing in the decision above with caution.\n"
        )

    def _apply_risk_overlay(
        self, company_name, trade_date, final_state, portfolio_state,
        asset_type: str = "stock",
    ):
        """Append the deterministic risk overlay to the PM's final decision.

        The LLM keeps the rating and the thesis; this layer overrides size,
        stop and exposure with math and records it as a clearly-marked section
        appended after the existing markdown.

        Rating is read from ``final_state["pm_rating"]`` (the structured
        ``PortfolioDecision.rating`` extracted directly by the PM node) so it
        never depends on the markdown's text ordering. When ``pm_rating`` is
        empty (PM fell back to free text, or a checkpoint-resumed run seeded
        it to ""), it falls back to ``parse_rating`` on the markdown — the
        legacy path, kept for backward compatibility.
        """
        if self.risk_manager is None:
            if self._risk_overlay_degraded:
                final_state["final_trade_decision"] = (
                    final_state.get("final_trade_decision", "")
                    + self._risk_disabled_warning("RiskManager build failed")
                )
            return final_state

        from yialpha.risk.manager import PortfolioState

        decision_md = final_state.get("final_trade_decision", "")
        # Prefer the structured rating extracted directly from the PM's
        # PortfolioDecision; fall back to markdown parsing only when it is
        # absent (free-text fallback or checkpoint resume).
        rating = final_state.get("pm_rating", "") or self.signal_processor.process_signal(
            decision_md
        )

        # Coerce the injected snapshot into a PortfolioState the manager reads.
        state = portfolio_state
        if state is not None and not isinstance(state, PortfolioState):
            state = PortfolioState(
                cash=float(state.get("cash", 0.0) or 0.0),
                equity=float(state.get("equity", 0.0) or 0.0),
                positions=dict(state.get("positions", {}) or {}),
                sectors=dict(state.get("sectors", {}) or {}),
                returns_history=list(state.get("returns_history", []) or []),
                trade_history=list(state.get("trade_history", []) or []),
            )
        if state is None:
            state = PortfolioState(equity=0.0)

        # Naming clarification: ``company_name`` IS the tradable ticker symbol
        # throughout this codebase (the historical parameter name predates the
        # ticker/company split); every data loader below keys on it as such.
        ticker = company_name
        close, atr = self._latest_close_and_atr(ticker, trade_date, asset_type)
        # Perp carry input for the risk gate AND the ticket's funding note —
        # ONE fetch feeds both. Live runs only: a historical run must not pay
        # a network call per decision, and the gate stays skipped (None)
        # there by design.
        funding_total_7d: float | None = None
        if asset_type == "crypto_perp":
            funding_total_7d = self._trailing_funding_total(ticker, str(trade_date))
        funding_annualized = (
            funding_total_7d / 7.0 * 365.0 if funding_total_7d is not None else None
        )
        try:
            decision = self.risk_manager.decide(
                ticker, rating, state, price=close, atr=atr, date=str(trade_date),
                funding_rate_annualized=funding_annualized,
            )
        except Exception as exc:  # noqa: BLE001 -- overlay must never break a run
            logger.warning("risk overlay failed for %s on %s: %s", ticker, trade_date, exc)
            final_state["final_trade_decision"] = (
                final_state.get("final_trade_decision", "")
                + self._risk_disabled_warning(f"overlay computation failed: {exc}")
            )
            return final_state

        overlay = (
            f"\n\n---\n\n{OVERLAY_MARKER}\n\n"
            f"- **Action**: {decision.action}\n"
            f"- **Target Weight**: {decision.target_weight:.1%}"
            + (f" of equity ({decision.position_value:,.0f})" if decision.position_value else "")
            + "\n"
        )
        # Perp live runs: the last daily candle is still forming — its close
        # IS the live price, but the ATR feeding the stop/leverage math is
        # computed on completed bars only (atr_stop). Disclose the two bases
        # so the reader does not treat them as one consistent snapshot.
        forming_price_note = ""
        if (
            asset_type == "crypto_perp"
            and str(trade_date) == datetime.now(UTC).strftime("%Y-%m-%d")
        ):
            forming_price_note = (
                " (live forming candle; ATR on completed bars only)"
            )
        if decision.stop_loss is not None:
            overlay += f"- **Stop Loss**: {decision.stop_loss:.2f}\n"
        if decision.entry_price is not None:
            overlay += (
                f"- **Entry Reference**: {decision.entry_price:.2f}"
                f"{forming_price_note}\n"
            )
        # A sized position with no stop-loss is the silent-degradation signal:
        # it fires both when price/ATR data failed to load (close is None) AND
        # when the stop computation itself threw inside RiskManager.decide
        # (close present but atr invalid). Key off stop_loss + target_weight so
        # both paths are surfaced (mirrors the DISABLED banner for build/decide
        # failures).
        if decision.target_weight > 0.0 and decision.stop_loss is None:
            overlay += (
                "- **⚠️ Stop-loss not set**: price/ATR data unavailable; this "
                "position has no ATR-based stop protection.\n"
            )
        # Perp advisory ticket: deterministic leverage / liquidation-price
        # math — the same formulas scripts/trade_ticket.py uses, extracted to
        # yialpha.risk.perp_ticket so the runtime overlay and the post-hoc
        # ticket can never drift apart. The LLM never fills these numbers.
        if asset_type == "crypto_perp":
            overlay += self._render_perp_ticket(
                ticker, str(trade_date), rating, decision, close, atr,
                funding_total_7d=funding_total_7d,
                mark_close=self._latest_mark_close(ticker, str(trade_date)),
            )
            overlay += self._render_stress_line(ticker, str(trade_date))
        # V2.0: the Candidate ExecutionTicket — the single cross-stage trading
        # object (I1). Deterministic tradeability verdict (directional edge vs
        # round-trip cost + four-tier data quality) with the linkage keys later
        # stages fill (prediction_ids/regime_id → V2.1/V2.2). Stored on state
        # and rendered here; the PM's rating text is never touched.
        ticket = None
        try:
            from yialpha.dataflows import quality as _quality
            from yialpha.tickets import build_candidate_ticket, render_ticket_lines

            ticket = build_candidate_ticket(
                symbol=ticker,
                asset_type=asset_type,
                rating=rating,
                target_weight=float(decision.target_weight),
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                reference_price=close,
                quality_events=_quality.snapshot_quality(),
                core_successes=_quality.snapshot_core_successes(),
                pm_fields=final_state.get("pm_decision_fields") or {},
                atr=atr,
                funding_rate_annualized=funding_annualized,
                trade_date=str(trade_date),
            )
        except Exception as exc:  # noqa: BLE001 -- ticket must never break a run
            logger.warning("candidate ticket build failed for %s: %s", ticker, exc)
        if ticket is not None:
            bridge_block = self._link_ticket_to_ledger(
                ticket, final_state, ticker, str(trade_date), asset_type, close
            )
            control_block = self._apply_portfolio_control(
                ticket, decision, final_state, ticker, str(trade_date), asset_type,
                equity=float(getattr(decision, "position_value", 0.0) or 0.0)
                or float(state.equity or 0.0)
                or 1.0,
            )
            final_state["execution_ticket"] = ticket.model_dump()
            overlay += render_ticket_lines(ticket)
            if bridge_block:
                overlay += bridge_block
            if control_block:
                overlay += control_block
        overlay += (
            f"- **Drawdown Regime**: {decision.breaker.regime}"
            f" ({decision.breaker.current_drawdown:.1%})\n"
            f"- **Rationale**: {decision.rationale}\n"
        )

        final_state["final_trade_decision"] = decision_md + overlay
        # Structured price-at-decision for the rating↔outcome verification
        # loop (verify-history / /api/accuracy): the rating was decided against
        # THIS close. Keys ride on final_state into full_states_log via
        # _log_state; the markdown "Entry Reference" line stays the
        # human-facing view (web/overlay_fields regex remains the fallback for
        # logs predating these fields).
        final_state["price_at_decision"] = decision.entry_price
        final_state["price_at_decision_basis"] = (
            "risk_overlay_close"
            if decision.entry_price is not None
            else "risk_overlay_no_price"
        )
        return final_state

    def _link_ticket_to_ledger(
        self,
        ticket,
        final_state,
        ticker: str,
        trade_date: str,
        asset_type: str,
        close: float | None,
    ) -> str:
        """V2.1 record stage: link the ticket into the ledger + bridge targets.

        Two flag-gated, fail-soft steps:

        1. With a ledger run context bound (``prediction_ledger`` on): fill
           the ticket's linkage keys (``run_id`` / ``evidence_ids`` /
           ``prediction_ids``) and mirror the ticket into the central ledger
           DB so the outcome computer can join decision-time cost legs.
        2. For stock perps with ``stock_perp_fair_value`` on: convert the
           PM's USD underlying target into a USDT contract target through
           the deterministic Fair Value Bridge (live-only USDT/USD from the
           inverted Binance spot USDCUSDT price), record the conversion
           chain on the ticket, and return a markdown disclosure block for
           the overlay. A missing FX rate records the gap and a shadow
           DEGRADED_CRITICAL verdict — enforcement arrives with the V2.4
           enforced mode.

        Returns "" (no overlay addition) whenever a gate is closed or a leg
        is genuinely absent; an exception anywhere is logged and swallowed —
        the record stage must never break the run it is describing.
        """
        from yialpha.ledger.run_context import current_ledger_run_context

        context = current_ledger_run_context()
        if context is not None and context.run_id:
            try:
                from yialpha.ledger.evidence import evidence_ids_for_run
                from yialpha.ledger.predictions import predictions_for_run
                from yialpha.ledger.tickets_mirror import attach_ticket

                ticket.run_id = context.run_id
                ticket.evidence_ids = evidence_ids_for_run(context.run_id)
                ticket.prediction_ids = [
                    record.prediction_id
                    for record in predictions_for_run(context.run_id)
                ]
                # V2.2: the run's regime id from the (re-bound) run context —
                # None while the regime stage is off/uncomputable, which is
                # exactly the ticket's pre-V2.2 value.
                ticket.regime_id = context.regime_id
                # V2.4 enforced mode: the atomic commit_final_ticket owns the
                # ticket write (with the snapshot + position, in ONE
                # transaction) — an earlier CANDIDATE mirror here would win
                # the INSERT OR IGNORE and strand the final payload. Legacy
                # and shadow modes mirror here as before.
                if not (
                    asset_type == "crypto_perp"
                    and self.config.get("portfolio_control_mode") == "enforced"
                ):
                    attach_ticket(
                        ticket_id=ticket.ticket_id,
                        decision_id=ticket.decision_id,
                        run_id=context.run_id,
                        payload=ticket.model_dump(),
                        ticket_version=ticket.ticket_version,
                    )
            except Exception:  # noqa: BLE001 -- record stage must never break a run
                logger.warning(
                    "ticket ledger linkage failed for %s on %s",
                    ticker,
                    trade_date,
                    exc_info=True,
                )
        if asset_type != "crypto_perp" or not self.config.get("stock_perp_fair_value"):
            return ""
        try:
            from yialpha.graph import routing as routing_module
            from yialpha.perp import quote_fx as quote_fx_module
            from yialpha.perp.fair_value import fair_value_bridge, render_bridge_block

            instrument_class = (
                context.instrument_class if context is not None else None
            ) or routing_module.instrument_class(asset_type, ticker)
            if instrument_class != "stock_perp":
                return ""
            pm_fields = final_state.get("pm_decision_fields") or {}

            def _as_float(value) -> float | None:
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            underlying_target = _as_float(pm_fields.get("underlying_price_target"))
            if (
                underlying_target is None
                and str(pm_fields.get("price_target_currency") or "").upper() == "USD"
            ):
                # PM quoted the target on the underlying in USD directly.
                underlying_target = _as_float(pm_fields.get("price_target"))
            if underlying_target is None:
                return ""
            fx = quote_fx_module.usdt_usd_as_of(trade_date)
            mark_close = self._latest_mark_close(ticker, trade_date)
            current_basis = (
                (close - mark_close) / mark_close
                if close is not None and mark_close
                else None
            )
            result = fair_value_bridge(
                underlying_target,
                fx.rate if fx is not None else None,
                current_basis,
            )
            ticket.underlying_target = underlying_target
            ticket.price_target_basis = "last"
            if result.contract_target_usdt is not None:
                ticket.contract_target = result.contract_target_usdt
            if fx is not None:
                ticket.quote_fx = {
                    "rate": fx.rate,
                    "source": fx.source,
                    "available_at": fx.available_at,
                    "replayability": fx.replayability,
                }
            if close is not None or mark_close is not None:
                ticket.basis_snapshot = {
                    "last_close": close,
                    "mark_close": mark_close,
                    "last_vs_mark": current_basis,
                }
            block = (
                "\n"
                + quote_fx_module.render_fx_line(fx)
                + "\n\n"
                + render_bridge_block(result)
                + "\n"
            )
            if fx is None:
                block += (
                    "- ⚠ shadow verdict: DEGRADED_CRITICAL / NO_TRADE recorded "
                    "(enforcement arrives with the V2.4 enforced mode)\n"
                )
            return block
        except Exception:  # noqa: BLE001 -- record stage must never break a run
            logger.warning(
                "fair-value bridge wiring failed for %s on %s",
                ticker,
                trade_date,
                exc_info=True,
            )
            return ""

    def _apply_portfolio_control(
        self,
        ticket,
        decision,
        final_state,
        ticker: str,
        trade_date: str,
        asset_type: str,
        equity: float = 1.0,
    ) -> str:
        """V2.4 Portfolio Control: candidate → snapshot → constraints → resolver.

        Mode-gated (config ``portfolio_control_mode``), crypto_perp runs
        only, fail-soft throughout:

        * ``legacy`` (default) — no-op (byte-identical decisions, pinned).
        * ``shadow`` — the legacy decision stands; the full pipeline ALSO
          runs and renders a clearly-marked SHADOW section for diffing; the
          result rides ``final_state["portfolio_control_shadow"]``; NO
          ticket/snapshot/position rows are written.
        * ``enforced`` — the resolver outcome is authoritative: ``final_size``
          lands on the ticket, the status advances to APPROVED / RESIZED /
          VETOED, and the final ticket + portfolio snapshot + position commit
          atomically to the ledger DB. The resolver can only shrink or
          refuse — it never changes the side (from the PM's explicit
          ``desired_side`` or the frozen conservative rating mapping) and
          never grows the size.

        REDUCE/CLOSE intents map to FLAT here: V2.4 enforces ENTRY-side
        control; shrinking/closing an EXISTING position rides the position
        lifecycle (close path), not a new entry candidate.
        """
        mode = str(self.config.get("portfolio_control_mode") or "legacy")
        if mode not in ("shadow", "enforced") or asset_type != "crypto_perp":
            return ""
        try:
            from dataclasses import asdict
            from typing import cast

            from yialpha.graph import routing as routing_module
            from yialpha.ledger.portfolio import (
                commit_final_ticket,
                new_snapshot_id,
                open_positions,
            )
            from yialpha.ledger.run_context import current_ledger_run_context
            from yialpha.risk.constraints import (
                PortfolioLimits,
                PortfolioSnapshot,
                PositionView,
                advisory_metrics,
                evaluate_constraints,
            )
            from yialpha.risk.resolver import render_resolver_lines, resolve_constraints
            from yialpha.risk.signed_math import Side, signed_position
            from yialpha.tickets import TicketStatus, desired_side_from_decision

            pm_fields = final_state.get("pm_decision_fields") or {}
            # The overlay's canonical rating source: the structured pm_rating
            # the PM node extracted (pm_decision_fields carries it too, but a
            # degenerate/free-text state may lack the flattened copy).
            rating_fields = dict(pm_fields)
            rating_fields.setdefault(
                "rating", final_state.get("pm_rating", "")
            )
            explicit_side = str(pm_fields.get("desired_side") or "").upper()
            legacy_intent = desired_side_from_decision(rating_fields)
            side = cast(
                Side,
                explicit_side
                if explicit_side in ("LONG", "SHORT", "FLAT")
                else ("LONG" if legacy_intent == "LONG" else "FLAT"),
            )
            proposed = (
                max(0.0, float(decision.target_weight))
                if side in ("LONG", "SHORT")
                else 0.0
            )
            if proposed <= 0.0 and explicit_side == "SHORT":
                # Legacy sizing is rating-tied — a bearish rating sizes to
                # zero, which would make an EXPLICIT short dead on arrival.
                # Interim V2.4 rule (disclosed): an explicit short enters at
                # kelly_fraction x max_single_position (conservative default
                # entry), replaced by signed-Kelly sizing when it lands.
                proposed = float(self.config.get("kelly_fraction", 0.25) or 0.25) * float(
                    self.config.get("max_single_position", 0.2) or 0.2
                )
            instrument_class = routing_module.instrument_class(asset_type, ticker)
            candidate = PositionView(
                symbol=ticker,
                instrument_class=instrument_class,
                side=side,
                weight=signed_position(side, proposed),
            )
            open_rows = open_positions()
            positions = tuple(
                PositionView(
                    symbol=row["symbol"],
                    instrument_class=str(
                        (row.get("payload") or {}).get("instrument_class")
                        or "unknown_perp"
                    ),
                    side=cast(Side, str(row["side"])),
                    weight=float(row["signed_weight"]),
                )
                for row in open_rows
            )
            snapshot = PortfolioSnapshot(
                equity=equity,
                positions=positions,
            )
            limits = PortfolioLimits(
                max_single=float(self.config.get("max_single_position", 0.2) or 0.2)
            )
            decisions = evaluate_constraints(candidate, snapshot, limits)
            result = resolve_constraints(decisions, proposed, side)
            advisory = advisory_metrics(candidate, snapshot, limits)
            context = current_ledger_run_context()
            run_id = context.run_id if context is not None else None
            snapshot_id = new_snapshot_id(run_id or ticket.ticket_id)

            rule_lines = "\n".join(
                f"- {d.rule}: {d.action} (x{d.multiplier:.3f})"
                + (f" — {d.reasons[0]}" if d.reasons else "")
                for d in decisions
            )
            if mode == "shadow":
                final_state["portfolio_control_shadow"] = {
                    "side": side,
                    "proposed_size": proposed,
                    "resolver": asdict(result),
                    "decisions": [asdict(d) for d in decisions],
                    "snapshot_id": snapshot_id,
                }
                return (
                    f"\n\n---\n\n[PORTFOLIO CONTROL — SHADOW, not enforced]\n\n"
                    f"- **Side**: {side} · **Proposed**: {proposed:.1%}\n"
                    + render_resolver_lines(result)
                    + "\n"
                    + rule_lines
                    + "\n- Advisory metrics computed and recorded only "
                    "(never resize): see portfolio_control_shadow.\n"
                )

            # enforced: resolver outcome is authoritative.
            ticket.status = {
                "APPROVED": TicketStatus.APPROVED,
                "RESIZED": TicketStatus.RESIZED,
                "VETOED": TicketStatus.VETOED,
            }[result.action]
            ticket.final_size = result.final_size
            ticket.margin_mode = "ISOLATED"
            ticket.portfolio_snapshot_id = snapshot_id
            ticket.risk_decision_ids = [d.rule for d in decisions]
            snapshot_payload = {
                "snapshot_id": snapshot_id,
                "run_id": run_id,
                "mode": "enforced",
                "equity": snapshot.equity,
                "positions": [asdict(p) for p in positions],
                "candidate": asdict(candidate),
                "decisions": [asdict(d) for d in decisions],
                "resolver": asdict(result),
                "advisory": advisory,
            }
            open_position = (
                result.action != "VETOED" and result.final_size > 0.0 and side != "FLAT"
            )
            commit_final_ticket(
                ticket_payload=ticket.model_dump(),
                ticket_id=ticket.ticket_id,
                decision_id=ticket.decision_id,
                run_id=run_id,
                ticket_version=ticket.ticket_version,
                snapshot_payload=snapshot_payload,
                snapshot_id=snapshot_id,
                position_symbol=ticker,
                position_side=side,
                position_signed_weight=signed_position(
                    side, result.final_size if open_position else 0.0
                ),
                open_position=open_position,
            )
            final_state["portfolio_control"] = {
                "mode": "enforced",
                "side": side,
                "proposed_size": proposed,
                "final_size": result.final_size,
                "action": result.action,
                "snapshot_id": snapshot_id,
            }
            return (
                f"\n\n---\n\n[PORTFOLIO CONTROL — ENFORCED]\n\n"
                f"- **Side**: {side} · **Proposed**: {proposed:.1%} · "
                f"**Final**: {result.final_size:.1%} ({result.action})\n"
                f"- **Margin**: ISOLATED · **Snapshot**: {snapshot_id}\n"
                + render_resolver_lines(result)
                + "\n"
                + rule_lines
                + "\n"
            )
        except Exception:  # noqa: BLE001 -- portfolio control must never break a run
            logger.warning(
                "portfolio control failed for %s on %s; legacy decision stands",
                ticker,
                trade_date,
                exc_info=True,
            )
            return ""

    @staticmethod
    def _render_perp_ticket(
        ticker: str,
        trade_date: str,
        rating: str,
        decision,
        close: float | None,
        atr: float | None,
        funding_total_7d: float | None = None,
        mark_close: float | None = None,
    ) -> str:
        """Deterministic perp advisory bullets for the risk overlay.

        Direction comes from the overlay's own final weight (never the LLM's
        prose): positive → long, negative → short, flat → no ticket. Today
        the risk manager clamps target_weight to >= 0 (long-only — kelly
        sizing never sells), so the short branch below is dormant, kept
        ready for the day the manager emits negative weights. The numeric
        core (leverage caps, stop construction, liquidation estimate) is
        shared with the ExecutionTicket via
        :func:`yialpha.risk.perp_ticket.perp_ticket_numbers` — one
        implementation, two consumers, zero drift.
        ``funding_total_7d`` is the SAME trailing 7-day settlement sum the
        risk gate priced (fetched once by the caller); None formats as an
        explicit n/a. ``mark_close`` carries the mark-price basis
        (liquidations trigger on MARK, not last) next to the last-price
        entry reference. Failures degrade to a missing bullet, never a
        broken overlay.
        """
        if close is None or close <= 0.0 or atr is None or atr <= 0.0:
            return ""
        weight = decision.target_weight
        if weight == 0.0:
            return ""
        # Signed side for the breach/ordering checks below — same source of
        # truth as perp_ticket_numbers (the sign of the final weight).
        direction = "long" if weight > 0.0 else "short"
        entry = decision.entry_price if decision.entry_price else close

        from yialpha.risk.perp_ticket import perp_ticket_numbers

        result = perp_ticket_numbers(entry, atr, rating, decision.stop_loss, weight)
        if result is None:
            return ""
        lev, detail, liq, stop = result

        def _p(x: float) -> str:
            return f"{x:.6g}"

        lines = [
            "- **Suggested Leverage**: ≤ "
            f"{lev:.1f}x (liq-dist {detail['L_liq']:.1f}x · vol "
            f"{detail['L_vol']:.1f}x · conviction {detail['L_conv']:.1f}x · "
            f"hard {detail['L_hard']:.0f}x)\n"
        ]
        if mark_close is not None and mark_close > 0:
            basis_bps = (entry / mark_close - 1.0) * 1e4
            mark_note = (
                f" (last−mark {basis_bps:+.1f} bps; liquidation judges on MARK)"
            )
            if liq is not None:
                # Liquidation judges on MARK; for a long the level sits below
                # entry (mirrored for a short). A mark already THROUGH the
                # level is a breach — render the warning, never a positive
                # "X% to liq" distance off an abs().
                breached = (
                    mark_close <= liq
                    if direction == "long"
                    else mark_close >= liq
                )
                if breached:
                    mark_note = (
                        f" (last−mark {basis_bps:+.1f} bps; ⚠️ MARK "
                        f"{_p(mark_close)} has ALREADY breached the estimated "
                        f"liquidation level {_p(liq)} — the stop-first "
                        "ordering no longer holds)"
                    )
                else:
                    mark_off = abs(liq / mark_close - 1.0)
                    mark_note = (
                        f" (last−mark {basis_bps:+.1f} bps; {mark_off:.1%} from "
                        f"current mark {_p(mark_close)} to liq — liquidations "
                        "trigger on MARK)"
                    )
            lines.append(f"- **Mark Reference**: {_p(mark_close)}{mark_note}\n")
        if liq is not None:
            off = abs(liq / entry - 1.0)
            # The stop-first claim is VERIFIED at render time, not asserted:
            # the stop triggers on CONTRACT_PRICE (last) while liquidation
            # judges on MARK, so it only holds when the stop sits beyond the
            # liq level AND (when known) mark has not already crossed it.
            # Today's LIQ_SAFETY≥1 construction makes stop_beyond_liq hold
            # by design; the branch stays as a guard against future math
            # changes and mark divergence at render time.
            stop_beyond_liq = (
                stop > liq if direction == "long" else stop < liq
            )
            mark_safe = mark_close is None or (
                mark_close > liq if direction == "long" else mark_close < liq
            )
            if stop_beyond_liq and mark_safe:
                order_note = f"the stop at {_p(stop)} fires first by design"
            elif not stop_beyond_liq:
                order_note = (
                    f"⚠️ the stop at {_p(stop)} does NOT sit beyond the "
                    "estimated liquidation level — the stop-first ordering "
                    "is NOT guaranteed"
                )
            else:
                order_note = (
                    "⚠️ MARK has crossed the estimated liquidation level — "
                    "the stop-first ordering no longer holds"
                )
            mmr_note = (
                f"includes MMR {detail['liq_mmr']:.2%} + fee "
                f"{detail['liq_fee_rate']:.2%}; {detail.get('liq_mmr_basis', '')}"
                if "liq_mmr" in detail
                else ""
            )
            lines.append(
                f"- **Est. Liquidation Price**: {_p(liq)} "
                f"({off:.1%} from entry; {order_note}"
                f"{'; ' + mmr_note if mmr_note else ''})\n"
            )
            from yialpha.risk.perp_ticket import STOP_TRIGGER_BASIS

            lines.append(
                f"- **Stop Trigger Basis**: {STOP_TRIGGER_BASIS}\n"
            )
        lines.append(
            "- **Funding (7d)**: "
            + YiAlphaGraph._format_funding_note(funding_total_7d, trade_date)
            + "\n"
        )
        return "".join(lines)

    @staticmethod
    def _render_stress_line(ticker: str, trade_date: str) -> str:
        """Derivatives Stress bullet for perp overlays (V2.0 P0.4).

        Live runs only — the same no-fetch-on-historical-dates policy as the
        funding note (a backtest must not pay five vendor calls per
        decision). Any fetch failure degrades to an explicit n/a line, never
        a silently missing section; the pure ``compute_stress`` itself is
        fail-open per component, so a partial window scores what it has and
        flags what it lacks.
        """
        from datetime import date, datetime, timedelta

        try:
            dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
        except ValueError:
            dt = None
        if dt is None or dt < date.today() - timedelta(days=3):
            return (
                "- **Derivatives Stress**: n/a (historical run — live-only "
                "positioning window)\n"
            )
        try:
            from yialpha.dataflows.binance import derivatives_stress_series
            from yialpha.risk.derivatives_stress import (
                compute_stress,
                render_stress_line,
            )

            series = derivatives_stress_series(ticker, trade_date)
            report = compute_stress(
                funding=series.get("funding"),
                global_lsr=series.get("global_lsr"),
                basis=series.get("basis"),
                open_interest=series.get("open_interest"),
                taker_ratio=series.get("taker_ratio"),
            )
            return render_stress_line(report)
        except Exception as exc:  # noqa: BLE001 -- disclosure, never break
            logger.warning("derivatives stress unavailable for %s: %s", ticker, exc)
            return "- **Derivatives Stress**: n/a (fetch failed)\n"

    @staticmethod
    def _trailing_funding_total(ticker: str, trade_date: str) -> float | None:
        """Trailing-7d funding settlement sum (PIT-clamped), or None.

        None for historical runs (explicit no-fetch policy — a backtest must
        not pay one vendor call per decision) and for fetch failures (the
        caller degrades, never blocks). The single fetch feeds both the risk
        gate (annualized) and the funding-note bullet.
        """
        from datetime import date, datetime, timedelta

        try:
            dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
        except ValueError:
            return None
        if dt < date.today() - timedelta(days=3):
            return None
        try:
            from yialpha.backtest.engine import _binance_funding_provider

            start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
            series = _binance_funding_provider(ticker, start, trade_date)
            if series is None or series.empty:
                return None
            return float(series.sum())
        except Exception as exc:  # noqa: BLE001 — advisory input only
            logger.warning("funding lookup unavailable for %s: %s", ticker, exc)
            return None

    @staticmethod
    def _format_funding_note(total_7d: float | None, trade_date: str) -> str:
        """Render the trailing-7d funding sum; None explains WHY it is absent.

        Distinguishes the two None causes explicitly — a historical run did
        not fetch by policy, a live fetch failed — instead of one opaque n/a.
        """
        if total_7d is None:
            from datetime import date, datetime, timedelta

            try:
                dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
            except ValueError:
                return "n/a (unreadable date)"
            if dt < date.today() - timedelta(days=3):
                return "n/a (historical run; not fetched)"
            return "n/a (fetch failed)"
        direction = "longs pay" if total_7d > 0 else (
            "longs receive" if total_7d < 0 else "flat")
        return f"{total_7d:+.3%} net over 7d ({direction})"

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5,
        benchmark: str = "SPY",
        as_of_date: str | None = None,
    ) -> tuple[float | None, float | None, int | None]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        ``benchmark`` is the index used as the alpha baseline (resolved by the
        caller via ``_resolve_benchmark``). Returns ``(raw_return, alpha_return,
        actual_holding_days)`` or ``(None, None, None)`` if price data is
        unavailable (too recent, delisted, or network error).

        Thin delegation to :func:`yialpha.accuracy.fetch_returns_yf` — the
        standalone ``memory-resolve`` CLI runs the same resolution without a
        graph instance, so the implementation lives where both callers share
        it (extracted verbatim; the graph's tests pin the semantics).
        """
        from yialpha.accuracy import fetch_returns_yf

        return fetch_returns_yf(
            ticker, trade_date,
            benchmark=benchmark, holding_days=holding_days, as_of_date=as_of_date,
        )

    def _resolve_pending_entries(
        self,
        ticker: str,
        as_of_date: str | None = None,
    ) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Delegates to :func:`yialpha.graph.memory_resolution.
        resolve_pending_entries` (extracted so the standalone ``memory-resolve``
        CLI resolves the same entries without a full analysis run).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again — or until
        ``yialpha memory-resolve`` sweeps them.
        """
        from yialpha.graph.memory_resolution import resolve_pending_entries

        resolve_pending_entries(
            self.memory_log,
            self.reflector,
            ticker,
            benchmark=self._resolve_benchmark(ticker),
            as_of_date=as_of_date,
        )

    def resolve_instrument_context(
        self, ticker: str, asset_type: str = "stock", trade_date: str | None = None,
    ) -> str:
        """Resolve ticker identity once and return the full instrument context.

        Deterministic yfinance lookup (cached, fail-open) injected into a
        context string so every agent anchors to the real company instead of
        hallucinating one from the price chart (#814). Both the propagate()
        path and the CLI call this so the resolved identity reaches the whole
        graph regardless of entry point.

        ``trade_date`` enforces the point-in-time guard in
        :func:`resolve_instrument_identity`: on a historical date the live
        ``.info`` snapshot (company name / sector / exchange) is refused and
        the context degrades to ticker-only, the same as a yfinance failure.
        """
        # Tokenized US-equity perps anchor to the UNDERLYING company's
        # identity (MUUSDT -> Micron via MU): resolving the perp symbol itself
        # fails on Yahoo and degrades to ticker-only exactly where the company
        # context matters most. Pure-crypto perps keep the perp symbol (the
        # lookup fails harmlessly and degrades to ticker-only as before).
        identity_ticker = ticker
        if asset_type == "crypto_perp":
            identity_ticker = stock_perp_underlying(ticker) or ticker
        identity = resolve_instrument_identity(identity_ticker, trade_date)
        return build_instrument_context(ticker, asset_type, identity)

    def _run_signature(self, asset_type: str) -> str:
        """Graph-shape inputs that must invalidate a checkpoint if changed.

        Folded into the checkpoint thread_id so resuming under a different
        selection of analysts, debate/risk depth, asset type, or parallel
        mode starts fresh instead of continuing a graph with a different
        shape. ``analyst_parallel`` is read from config (YiAlpha-only; the
        fan-out changes the graph topology, so it must be part of the
        signature even though the upstream contract doesn't carry it).
        """
        return "|".join([
            # Invalidate checkpoints produced before live-only data sources
            # were removed from historical tool bindings.
            "pit=v2",
            "analysts=" + ",".join(self.selected_analysts),
            f"debate={self.config['max_debate_rounds']}",
            f"risk={self.config['max_risk_discuss_rounds']}",
            f"asset={asset_type}",
            f"parallel={self.config.get('analyst_parallel', False)}",
        ])

    def propagate(self, company_name, trade_date, asset_type: str = "stock", portfolio_state=None):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.

        ``portfolio_state`` (Phase 1) is an optional live snapshot the Portfolio
        Manager sizes against and the risk overlay reads. ``None`` keeps the
        baseline behaviour; passing a dict or
        :class:`~yialpha.risk.manager.PortfolioState` only takes effect
        when ``risk_enabled`` is set in config.
        """
        self.ticker = company_name

        # Perp runs: warm the EQUITY-perp listing once (one exchangeInfo
        # fetch, fail-open to the static seed with a WARNING) so instrument-
        # context / identity / tool remaps all see the same fresh universe.
        # Stock/crypto runs skip this entirely — zero new network for them.
        if asset_type == "crypto_perp":
            warm_equity_perp_bases()

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name, as_of_date=str(trade_date))

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            # From the moment the saver's SQLite connection is open, every
            # subsequent step — including checkpoint_step, which opens its OWN
            # second connection and can raise — must run under the same
            # try/finally that closes the saver. Otherwise an exception between
            # __enter__ and the old try block leaked the saver connection.
            try:
                self.graph = self.workflow.compile(checkpointer=saver)

                step = checkpoint_step(
                    self.config["data_cache_dir"], company_name, str(trade_date),
                    self._run_signature(asset_type),
                )
                if step is not None:
                    logger.info(
                        "Resuming from step %d for %s on %s", step, company_name, trade_date
                    )
                else:
                    logger.info("Starting fresh for %s on %s", company_name, trade_date)

                return self._run_graph(
                    company_name, trade_date, asset_type=asset_type,
                    portfolio_state=portfolio_state,
                )
            finally:
                if self._checkpointer_ctx is not None:
                    self._checkpointer_ctx.__exit__(None, None, None)
                    self._checkpointer_ctx = None
                    self.graph = self.workflow.compile()

        try:
            return self._run_graph(
                company_name, trade_date, asset_type=asset_type, portfolio_state=portfolio_state,
            )
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def save_reports(self, final_state, ticker, save_path=None) -> Path:
        """Write the markdown report tree for a completed run, like the CLI does.

        Programmatic callers get the same on-disk reports the CLI produces. Pass
        an explicit ``save_path`` or let it default under ``results_dir``.
        """
        if save_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                Path(self.config["results_dir"])
                / "reports"
                / f"{safe_ticker_component(ticker)}_{stamp}"
            )
        return write_report_tree(final_state, ticker, save_path)

    def _invoke_or_stream(self, init_agent_state, args):
        """Execute the graph; optionally stream for per-analyst wall-time telemetry.

        Default (``stream_telemetry`` off): plain ``self.graph.invoke`` —
        byte-identical to the historical path. When telemetry is on, stream with
        ``stream_mode="values"`` and feed each chunk to an
        ``AnalystWallTimeTracker``. The graph is fully serial, so the last values
        chunk is exactly what ``invoke`` returns — telemetry adds observation
        only, never changing inputs, depth, or the final state. If a run emits no
        chunks (should not happen for a valid graph), raise rather than silently
        re-invoking — a hidden full-graph re-run would double the LLM cost.
        """
        if not self.config.get("stream_telemetry"):
            return self.graph.invoke(init_agent_state, **args)

        from yialpha.graph.analyst_execution import (
            AnalystWallTimeTracker,
            build_analyst_execution_plan,
            sync_analyst_tracker_from_chunk,
        )

        tracker = None
        try:
            tracker = AnalystWallTimeTracker(
                build_analyst_execution_plan(self.selected_analysts)
            )
        except ValueError:
            # Unknown analyst shape — skip timing but still stream.
            tracker = None

        final_state = None
        for chunk in self.graph.stream(init_agent_state, **args):
            final_state = chunk
            if tracker is not None:
                sync_analyst_tracker_from_chunk(tracker, chunk)
        if final_state is None:
            # A valid graph always emits at least one chunk. Treat zero chunks
            # as a hard failure rather than silently re-invoking the whole graph
            # — that would re-run every analyst/debate/risk/PM LLM call (~10min,
            # many DeepSeek requests) and double-bill the user with no signal.
            # Raise so the failure is visible and run_robust can retry cleanly.
            raise RuntimeError(
                "graph.stream emitted no chunks for this run; aborting instead "
                "of silently re-invoking (which would double the LLM cost). "
                "This usually indicates a langgraph regression or an exception "
                "swallowed inside stream()."
            )
        if tracker is not None:
            logger.info("%s", tracker.format_summary())
        return final_state

    def _run_graph(self, company_name, trade_date, asset_type: str = "stock", portfolio_state=None):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Reset per-node perf telemetry so this run's node_perf_<date>.json
        # reflects only this run (graph instances are reused across tickers in
        # batch mode). No-op when telemetry is off.
        if self.perf_tracker is not None:
            self.perf_tracker.reset()

        # Pin the analysis date for the PIT clamp in the data vendor layer
        # (get_stock_data / get_binance_klines clamp their fetch window to it so
        # a backtest never sees rows after the analysis date). The batch runner
        # copies the context into each worker via submit_with_context, so this
        # crosses the ThreadPoolExecutor boundary correctly. Live runs (no
        # explicit date) leave it unset = no-op pass-through.
        set_analysis_date(str(trade_date)) if str(trade_date) else set_analysis_date(None)

        # Bind the data-quality event accumulator in THIS (parent) context
        # before the graph runs: langgraph executes node tasks inside copied
        # contexts, so a list first created inside a node (via record_sentinel)
        # would never be visible to _log_state afterwards. Bound here, every
        # node context inherits the same list object and appends to it. Fresh
        # per run — a crashed prior run in the same context cannot leak events.
        from yialpha.dataflows import quality

        quality.ensure_run_context()
        # Fresh per-run Tavily search budget (same ContextVar semantics as the
        # quality ledger above): a worker process serving many runs must not
        # carry one run's spent web-search calls into the next.
        from yialpha.dataflows import tavily as tavily_vendor

        tavily_vendor.reset_run_budget()
        # Fresh per-run bundle/news prefetch scope: the LLM tool loop re-enters
        # analyst nodes, and the deterministic prefetches must fetch exactly
        # ONCE per run — not once per tool-call round, and never one run's
        # live snapshot carried into the next (same ContextVar contract as
        # the quality ledger and the Tavily budget).
        from yialpha.dataflows import run_scope

        run_scope.ensure_run_scope()

        # V2.1 record stage: bind the ledger run context (one run_id for the
        # whole run, inherited by every node task) and register the run row.
        # Flag-gated — off means the entire ledger pipeline (evidence rows,
        # blind predictions, ticket linkage/mirror) is a no-op and the run's
        # state/log bytes stay identical to the pre-V2.1 behavior.
        ledger_run_id: str | None = None
        regime_run_id: str | None = None
        if self.config.get("prediction_ledger"):
            try:
                from yialpha.agents.utils import prediction_tools as _prediction_tools
                from yialpha.graph import routing as _routing
                from yialpha.ledger import run_context as _ledger_run_context
                from yialpha.ledger.evidence import register_run
                from yialpha.tickets import new_run_id

                ledger_run_id = new_run_id()
                _instrument_class = _routing.instrument_class(asset_type, company_name)
                _ledger_run_context.set_ledger_run_context(
                    ledger_run_id,
                    company_name,
                    asset_type,
                    instrument_class=_instrument_class,
                    analysis_as_of=str(trade_date),
                )
                register_run(
                    run_id=ledger_run_id,
                    ticker=company_name,
                    asset_type=asset_type,
                    instrument_class=_instrument_class,
                    analysis_as_of=str(trade_date),
                )
                # Bind the prediction capture registry in this parent context
                # so ToolNode tasks share one buffer per run (same ContextVar
                # contract as the quality ledger above).
                _prediction_tools.ensure_prediction_capture_scope()
            except Exception:  # noqa: BLE001 -- record stage must never break a run
                logger.warning(
                    "ledger run-context binding failed for %s; recording disabled "
                    "for this run",
                    company_name,
                    exc_info=True,
                )
                ledger_run_id = None
            # V2.2 Context stage: compute the run's RegimeState (perp runs,
            # flag regime_state), persist it, and re-bind the run context
            # with regime_id so predictions/tickets carry it and the market
            # analyst can inject the stored block WITHOUT recomputing. A
            # None regime (uncomputable inputs) leaves regime_id unset — the
            # analyst discloses "regime unavailable", never a fake id.
            if (
                ledger_run_id
                and self.config.get("regime_state")
                and asset_type == "crypto_perp"
            ):
                try:
                    from dataclasses import asdict

                    from yialpha.ledger.regime_store import upsert_regime
                    from yialpha.ledger.run_context import set_ledger_run_context
                    from yialpha.regime.compute import compute_regime_state

                    _regime = compute_regime_state(
                        company_name,
                        asset_type,
                        _instrument_class,
                        str(trade_date),
                        end_date=str(trade_date),
                    )
                    if _regime is not None and _regime.regime_id:
                        upsert_regime({
                            **asdict(_regime),
                            "ticker": company_name,
                            "instrument_class": _instrument_class,
                        })
                        set_ledger_run_context(
                            ledger_run_id,
                            company_name,
                            asset_type,
                            instrument_class=_instrument_class,
                            analysis_as_of=str(trade_date),
                            regime_id=_regime.regime_id,
                        )
                        regime_run_id = _regime.regime_id
                except Exception:  # noqa: BLE001 -- context stage must never break a run
                    logger.warning(
                        "regime state computation failed for %s; regime unavailable "
                        "for this run",
                        company_name,
                        exc_info=True,
                    )
                    regime_run_id = None

        try:
            # Initialize state — inject memory log context for PM and the
            # deterministically resolved instrument identity for all agents.
            past_context = self.memory_log.get_past_context(
                company_name,
                as_of_date=str(trade_date),
                asset_type=asset_type,
            )
            instrument_context = self.resolve_instrument_context(
                company_name, asset_type, trade_date=str(trade_date),
            )
            init_agent_state = self.propagator.create_initial_state(
                company_name,
                trade_date,
                asset_type=asset_type,
                past_context=past_context,
                instrument_context=instrument_context,
                portfolio_state=portfolio_state,
            )
            args = self.propagator.get_graph_args()

            # Inject thread_id so same ticker+date+graph-shape resumes; a different
            # date or graph shape starts fresh (#1089).
            if self.config.get("checkpoint_enabled"):
                tid = thread_id(company_name, str(trade_date), self._run_signature(asset_type))
                args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

            if self.debug:
                trace = []
                last_printed = None
                for chunk in self.graph.stream(init_agent_state, **args):
                    if chunk["messages"]:
                        msg = chunk["messages"][-1]
                        # Nodes after the trader don't append to messages, so the
                        # same trailing message repeats across chunks. Print it only
                        # when it changes (#1027); the trace/state merge is unchanged.
                        signature = (type(msg).__name__, getattr(msg, "content", None))
                        if signature != last_printed:
                            msg.pretty_print()
                            last_printed = signature
                        trace.append(chunk)
                # Streamed chunks are per-node deltas. Merge them so the returned
                # state matches what graph.invoke() yields in the non-debug path.
                final_state = {}
                for chunk in trace:
                    final_state.update(chunk)
            else:
                final_state = self._invoke_or_stream(init_agent_state, args)

            # Phase 1: deterministically override size / stop / exposure (LLM kept
            # the direction). No-op when risk_enabled is off. asset_type routes
            # the overlay's price source (perp runs price on the perp's own
            # Binance candles, not the Yahoo spot series).
            final_state = self._apply_risk_overlay(
                company_name, trade_date, final_state, portfolio_state,
                asset_type=asset_type,
            )

            # V2.1: the ledger run id rides on state — key presence (not a
            # null value) is the "this run was recorded" signal downstream.
            if ledger_run_id:
                final_state["run_id"] = ledger_run_id
            # V2.2: same key-presence contract for the regime id (present
            # only when the regime stage computed and stored a state).
            if regime_run_id:
                final_state["regime_id"] = regime_run_id

            # Store current state for reflection.
            self.curr_state = final_state

            # Log state to disk. The returned quality block rides on
            # final_state so report writers / the web UI can render the
            # degraded-run banner from the same evidence the JSON log has.
            final_state["data_quality"] = self._log_state(trade_date, final_state)

            # T0: dump per-node perf telemetry next to full_states_log. No-op when
            # telemetry is off (perf_tracker is None).
            if self.perf_tracker is not None:
                self._dump_perf(trade_date)

            # Store decision for deferred reflection on the next same-ticker
            # run. Every other field read here uses .get(); this was the one
            # bare subscript — a KeyError from a degenerate state would mask
            # the underlying condition, so degrade to "" with a WARNING
            # instead (parse_rating on "" already falls back to Hold).
            decision_for_memory = final_state.get("final_trade_decision")
            if decision_for_memory is None:
                logger.warning(
                    "final_state lacked 'final_trade_decision' for %s on %s; "
                    "storing empty decision for reflection",
                    company_name, trade_date,
                )
                decision_for_memory = ""
            self.memory_log.store_decision(
                ticker=company_name,
                trade_date=trade_date,
                final_trade_decision=decision_for_memory,
                asset_type=asset_type,
            )

            # Clear checkpoint on successful completion to avoid stale state.
            if self.config.get("checkpoint_enabled"):
                clear_checkpoint(
                    self.config["data_cache_dir"], company_name, str(trade_date),
                    self._run_signature(asset_type),
                )

            return final_state, self.process_signal(final_state["final_trade_decision"])
        finally:
            # Clear the PIT analysis date so it cannot leak into a subsequent run
            # in the same context (e.g. a live analysis after a backtest in the
            # same process). The batch runner copies a fresh context per worker,
            # so this is belt-and-braces for non-batch callers.
            set_analysis_date(None)
            # Drop the ledger run context with it: a later run in the same
            # context must not attribute evidence/predictions/tickets to this
            # finished run.
            from yialpha.ledger import run_context as _ledger_run_context_final

            _ledger_run_context_final.reset_ledger_run_context()

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file.

        Includes the structured evidence block the self-improvement loop
        consumes: ``pm_rating`` (the Portfolio Manager's structured rating,
        previously flattened into markdown only), ``data_quality`` (the
        router's sentinel events for this run, so a degraded report is
        machine-distinguishable from a fully-fed one), and
        ``web_search_usage`` (per-scope Tavily call counts). All are
        additive — older readers ignore unknown keys.
        """
        from yialpha.dataflows import quality
        from yialpha.dataflows import tavily as tavily_vendor  # isort: skip

        quality_block = quality.summarize_quality(
            quality.snapshot_quality(), quality.snapshot_core_successes()
        )
        quality.reset_quality()
        # Drop the per-run prefetch scope with the ledger: a live bundle
        # snapshot is run-scoped by contract and must never serve a later
        # run in the same process.
        from yialpha.dataflows import run_scope

        run_scope.reset_run_scope()
        # Snapshot the scoped web-search usage BEFORE anything else in this
        # method could run another charge; the counters themselves were reset
        # at run start (_run_graph), so this is the run's final tally.
        web_search_usage = tavily_vendor.run_usage()

        # Price-at-decision: normally set by _apply_risk_overlay (the close the
        # decision anchored on). When the overlay did not run (risk disabled /
        # degraded / decide failed), fall back to the same memoized PIT loader
        # — a cached hit when the overlay already tried, one fetch when it
        # didn't — so every log entry carries a verifiable decision price
        # instead of only the ones where the overlay succeeded.
        price = final_state.get("price_at_decision")
        basis = final_state.get("price_at_decision_basis")
        if basis is None:
            close, _atr = (
                self._latest_close_and_atr(
                    self.ticker, trade_date,
                    final_state.get("asset_type") or "stock",
                )
                if self.ticker else (None, None)
            )
            price, basis = (
                (close, "fallback_loader") if close is not None
                else (None, "unavailable")
            )

        entry = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
            # Structured decision + data-quality evidence (self-improvement):
            # pm_rating is the PM's typed rating (empty string when absent,
            # e.g. a state produced without the PM node); data_quality carries
            # the router's sentinel events accumulated during this run;
            # price_at_decision / basis / asset_type anchor the rating to the
            # close it was decided against, so historical accuracy can be
            # verified without re-deriving prices. All additive — older
            # readers ignore unknown keys, and pre-T0.5 logs simply lack them.
            "pm_rating": final_state.get("pm_rating", ""),
            "price_at_decision": price,
            "price_at_decision_basis": basis,
            "asset_type": final_state.get("asset_type") or "stock",
            "data_quality": quality_block,
            # Per-scope Tavily call counts for this run (news / market /
            # fundamentals, 0-filled): answers "did web_search actually fire
            # and for which analyst" straight from the log, without replaying
            # message histories. Charged calls only; degradations live in
            # data_quality above. Additive like its neighbours.
            "web_search_usage": web_search_usage,
            # V2.0 Evidence/Prediction-ledger linkage: the PM's typed decision
            # fields (opinion layer) and the deterministic Candidate
            # ExecutionTicket (tradeability verdict + linkage keys). None/{}
            # when the PM fell back to free text or the ticket build degraded;
            # both additive keys older readers ignore.
            "pm_decision_fields": final_state.get("pm_decision_fields") or {},
            "execution_ticket": final_state.get("execution_ticket"),
        }
        # V2.1: ledger run id — present only when the record-stage pipeline
        # ran (prediction_ledger on). Key presence (not a null value) keeps
        # flag-off logs byte-identical to the pre-V2.1 shape.
        if final_state.get("run_id"):
            entry["run_id"] = final_state["run_id"]
        # V2.2: regime id — same key-presence contract (regime_state on AND
        # a computable regime; flag-off / uncomputable logs lack the key).
        if final_state.get("regime_id"):
            entry["regime_id"] = final_state["regime_id"]
        # Write-and-drop: nothing downstream reads PAST dates from this dict
        # (the on-disk JSON below is the durable record), but a multi-date
        # backtest or a long-lived web process would otherwise accumulate
        # every date's FULL state — analyst reports, debate histories, the
        # works — for the lifetime of the graph object.
        self.log_states_dict = {str(trade_date): entry}

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "YiAlphaStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        # Atomic write: dump to a sibling temp file then os.replace() onto the
        # final path (same-volume rename is atomic on both Windows and POSIX).
        # A kill mid-write (run_robust's taskkill /F /T) would otherwise leave a
        # half-truncated JSON that masquerades as the "run complete" signal this
        # file is used as. Mirrors the atomic pattern in memory.py.
        tmp_path = directory / f".full_states_log_{trade_date}.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=4)
        os.replace(tmp_path, log_path)

        # Return the consumed quality block so the caller can attach it to
        # final_state — the report writer and web UI render it as the
        # human-facing degraded-run banner.
        return quality_block

    def finalize_streamed_run(
        self, ticker: str, trade_date: str, final_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply the ``_run_graph`` evidence contract to a caller-streamed run.

        The interactive CLI streams ``self.graph.stream`` directly for its live
        UI, bypassing :meth:`_run_graph`. That path used to skip
        :meth:`_log_state`, so a CLI run never wrote
        ``full_states_log_<date>.json`` (invisible in the web history) and
        never carried ``data_quality`` (the DEGRADED banner never rendered).
        This method reuses ``_log_state`` unchanged so both entry points land
        the identical on-disk evidence. The caller owns the pre-stream
        ``quality.ensure_run_context()`` call (it must run before the first
        node executes, not after streaming ends).

        Returns ``final_state`` with the consumed ``data_quality`` block
        attached (same shape ``_run_graph`` returns).
        """
        self.ticker = ticker
        final_state["data_quality"] = self._log_state(trade_date, final_state)
        return final_state

    def _dump_perf(self, trade_date):
        """Write per-node perf telemetry to node_perf_<trade_date>.json.

        Sits beside ``full_states_log_<trade_date>.json`` in the same per-ticker
        ``YiAlphaStrategy_logs`` directory. Callers gate on
        ``self.perf_tracker is not None``; this method assumes a live tracker.
        """
        from yialpha.graph.perf_telemetry import dump_perf_report

        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "YiAlphaStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)
        dump_perf_report(self.perf_tracker, directory / f"node_perf_{trade_date}.json")

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
