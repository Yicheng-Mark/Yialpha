# yiagents/graph/trading_graph.py

import json
import logging
import os
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.prebuilt import ToolNode

# Import the abstract tool methods from agent_utils
from yiagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_a_share_fundamentals_native,
    get_a_share_news_native,
    get_a_share_ohlc_native,
    get_balance_sheet,
    get_binance_basis,
    get_binance_funding_rate,
    get_binance_klines,
    get_binance_long_short_ratio,
    get_binance_open_interest,
    get_binance_spot_klines,
    get_binance_spot_perp_basis,
    get_binance_spot_ticker24,
    get_binance_taker_buy_sell,
    get_cashflow,
    get_form4_insider_trading,
    get_ftd_data,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_institutional_holdings,
    get_macro_indicators,
    get_margin_trading,
    get_news,
    get_prediction_markets,
    get_stock_data,
    get_verified_market_snapshot,
    resolve_instrument_identity,
)
from yiagents.agents.utils.memory import TradingMemoryLog
from yiagents.dataflows.config import set_config
from yiagents.dataflows.utils import safe_ticker_component, set_analysis_date
from yiagents.default_config import DEFAULT_CONFIG
from yiagents.llm_clients import create_llm_client
from yiagents.reporting import write_report_tree

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .overlay_fields import OVERLAY_MARKER
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)

#: A pending memory-log entry older than this (in days) that still cannot be
#: resolved is logged at WARNING so it does not silently accumulate forever.
STALE_PENDING_DAYS = 7


@lru_cache(maxsize=4096)
def _memoized_close_and_atr(ticker: str, trade_date: str) -> tuple[float, float]:
    """Load and compute the deterministic PIT ``(close, atr)`` for one date.

    Module-level LRU keyed by ``(ticker, trade_date)`` — see
    ``YiAgentsGraph._latest_close_and_atr`` for the PIT-safety argument.
    Raises on any failure so failures are never cached (the caller converts
    them to ``(None, None)`` and retries on the next run).
    """
    from yiagents.dataflows.stockstats_utils import load_ohlcv
    from yiagents.risk.atr_stop import latest_atr_from_frame

    frame = load_ohlcv(ticker, str(trade_date))
    close, atr = latest_atr_from_frame(frame)
    return float(close), float(atr)


def _entry_precedes_cutoff(entry: dict[str, Any], cutoff: datetime) -> bool:
    """Fail-closed date gate for pending reflection outcomes."""
    try:
        return datetime.strptime(str(entry["date"])[:10], "%Y-%m-%d") < cutoff
    except (KeyError, TypeError, ValueError):
        return False


def _warn_if_stale_pending(ticker: str, trade_date: str, now: datetime) -> None:
    """Log a WARNING when a pending entry is old enough to be concerning.

    Pending entries that can never resolve (delisted ticker, permanent data
    gap) previously accumulated indefinitely with no signal. This makes the
    silent-stuck state observable without changing the log format.
    """
    try:
        entry_date = datetime.strptime(trade_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return
    age_days = (now - entry_date).days
    if age_days > STALE_PENDING_DAYS:
        logger.warning(
            "Pending memory-log entry for %s @ %s is %d days old and still "
            "unresolved (price data unavailable). It may be stuck due to "
            "delisting or a permanent data gap.",
            ticker, trade_date, age_days,
        )


class YiAgentsGraph:
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
            from yiagents.graph.perf_telemetry import (
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
        # byte-equivalent. See yiagents/llm_clients/response_cache.py for the
        # distribution-safety caveat (never use with the A/B gate / DSR runs).
        if self.config.get("llm_cache"):
            from langchain_core.globals import set_llm_cache

            from yiagents.llm_clients.response_cache import DiskLLMCache

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
        # unless the user overrides YIAGENTS_DEBATE_LLM for A/B testing.
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
        # float() here so a value coming from a YIAGENTS_TEMPERATURE env
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
            from yiagents.llm_clients.openai_client import is_openai_compatible
            if is_openai_compatible(provider):
                from yiagents.llm_clients.rate_limiter import get_shared_rate_limiter
                kwargs["rate_limiter"] = get_shared_rate_limiter(
                    provider, int(self.config.get("llm_rpm", 60))
                )

        # P1a: optional shared httpx.Client so the K worker graphs reuse TLS/proxy
        # connections. Transport-only; off by default. Forwarded for OpenAI-
        # compatible providers via _PASSTHROUGH_KWARGS (http_client is allowlisted).
        if self.config.get("http_keepalive"):
            from yiagents.llm_clients.openai_client import is_openai_compatible
            if is_openai_compatible(provider):
                from yiagents.llm_clients.http_client import get_shared_http_client
                shared = get_shared_http_client()
                if shared is not None:
                    kwargs["http_client"] = shared

        # T1.3: per-call retry count forwarded to provider clients via
        # _PASSTHROUGH_KWARGS (max_retries). Default 2 == langchain-openai's
        # own default, so the success path is byte-equivalent to today; only
        # the failure/retry path can differ when a user lowers it. Exposed as
        # YIAGENTS_LLM_MAX_RETRIES so flaky periods can tune it down (the outer
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
                    # Binance SPOT tools. Same dormant contract as the perp
                    # tools above: only advertised when asset_type ==
                    # "crypto_spot", so they sit unused in the name->tool map
                    # for stock/crypto/perp runs.
                    get_binance_spot_klines,
                    get_binance_spot_ticker24,
                    get_binance_spot_perp_basis,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
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
                    # YIAGENTS_A_SHARE_NATIVE is on AND the ticker is an A-share
                    # (.SS/.SH/.SZ), so the LLM never names it otherwise and it
                    # is simply an extra entry in ToolNode's name->tool map (same
                    # dormant contract as the fundamentals A-share tools).
                    get_a_share_news_native,
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
                    # when YIAGENTS_SEC_OWNERSHIP is on, so the LLM never names
                    # them otherwise and they are simply extra entries in
                    # ToolNode's name->tool map (same dormant contract as the
                    # Binance tools in the market ToolNode).
                    get_form4_insider_trading,
                    get_ftd_data,
                    get_institutional_holdings,
                    # China A-share margin-trading tool (Track A). Dormant for
                    # default runs: the fundamentals analyst only advertises it
                    # when YIAGENTS_A_STOCK is on AND the ticker is an A-share
                    # (.SS/.SH/.SZ), so the LLM never names it otherwise and it
                    # is simply an extra entry in ToolNode's name->tool map
                    # (same dormant contract as the SEC/Binance tools above).
                    get_margin_trading,
                    # Native A-share OHLC + TTM valuation (a_share_native). Dormant
                    # for default runs: the fundamentals analyst only advertises
                    # them when YIAGENTS_A_SHARE_NATIVE is on AND the ticker is an
                    # A-share (.SS/.SH/.SZ), so the LLM never names them otherwise
                    # and they are simply extra entries in ToolNode's name->tool
                    # map (same dormant contract as the margin tool above).
                    get_a_share_fundamentals_native,
                    get_a_share_ohlc_native,
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

    def _build_risk_manager(self):
        """Construct the Phase-1 RiskManager when ``risk_enabled`` is set.

        Returns ``None`` when risk control is off so the rest of the graph can
        branch cheaply on a single truthy check. On build failure, sets
        ``self._risk_overlay_degraded`` so :meth:`_apply_risk_overlay` can mark
        the decision document visibly instead of silently dropping the overlay.
        """
        if not self.config.get("risk_enabled"):
            return None
        from yiagents.risk.manager import RiskManager
        try:
            return RiskManager.from_config(self.config)
        except Exception as exc:  # noqa: BLE001 -- never block a run on risk setup
            logger.warning("risk_enabled is set but RiskManager build failed (%s); "
                           "running without the risk overlay", exc)
            self._risk_overlay_degraded = True
            return None

    def _latest_close_and_atr(self, ticker, trade_date):
        """Best-effort (close, atr) as of ``trade_date`` for the risk overlay.

        PIT-safe: reuses the project's cached, date-truncated OHLCV loader. Any
        failure returns ``(None, None)`` so the overlay still runs without a
        stop rather than aborting the decision.

        Memoized per ``(ticker, trade_date)`` at module level: the 14-period
        stockstats ATR for a fixed PIT date is deterministic, and backtests
        hit the same (ticker, date) repeatedly (A/B legs, multi-run
        distributions). The date being part of the key is what keeps this
        PIT-safe — a new analysis date re-computes rather than reusing a
        prior day's value. Failures are NOT memoized (an lru_cache on a
        raising function re-invokes next call), so a transient vendor fault
        still retries on the next run.
        """
        try:
            return _memoized_close_and_atr(ticker, str(trade_date))
        except Exception as exc:  # noqa: BLE001
            logger.warning("risk overlay could not load price/ATR for %s on %s: %s",
                           ticker, trade_date, exc)
            return None, None

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

    def _apply_risk_overlay(self, company_name, trade_date, final_state, portfolio_state):
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

        from yiagents.risk.manager import PortfolioState

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
        close, atr = self._latest_close_and_atr(ticker, trade_date)
        try:
            decision = self.risk_manager.decide(
                ticker, rating, state, price=close, atr=atr, date=str(trade_date),
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
        if decision.stop_loss is not None:
            overlay += f"- **Stop Loss**: {decision.stop_loss:.2f}\n"
        if decision.entry_price is not None:
            overlay += f"- **Entry Reference**: {decision.entry_price:.2f}\n"
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
        overlay += (
            f"- **Drawdown Regime**: {decision.breaker.regime}"
            f" ({decision.breaker.current_drawdown:.1%})\n"
            f"- **Rationale**: {decision.rationale}\n"
        )

        final_state["final_trade_decision"] = decision_md + overlay
        return final_state

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
        """
        from yiagents.dataflows.symbol_utils import normalize_symbol
        from yiagents.dataflows.y_finance import get_YFin_history_cached

        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            cutoff = None
            if as_of_date is not None:
                cutoff = datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d")
                if cutoff <= start:
                    return None, None, None
                # yfinance's end is exclusive.  Never request observations
                # beyond the simulated date, even when today's vendor has them.
                end = min(end, cutoff + timedelta(days=1))
            end_str = end.strftime("%Y-%m-%d")

            # Normalize so the realized-return lookup hits the same instrument
            # the analysis priced (e.g. XAUUSD -> GC=F) (#984). The benchmark is
            # already a canonical Yahoo symbol from ``_resolve_benchmark``.
            # Both legs go through the vendor layer's ~24h cached history
            # wrapper (same window, same frame as the direct .history call):
            # a batch run over N tickers would otherwise re-download the SAME
            # benchmark window once per ticker.
            stock = get_YFin_history_cached(
                normalize_symbol(ticker), trade_date, end_str)
            bench = get_YFin_history_cached(benchmark, trade_date, end_str)

            if cutoff is not None:
                # Belt-and-suspenders PIT guard: vendors and test doubles can
                # ignore the requested end boundary, so trim dated frames again
                # on the client before calculating any outcome.
                def _through_cutoff(frame):
                    try:
                        return frame[frame.index.date <= cutoff.date()]
                    except (AttributeError, TypeError):
                        return frame

                stock = _through_cutoff(stock)
                bench = _through_cutoff(bench)

            if len(stock) < 2 or len(bench) < 2:
                return None, None, None

            if cutoff is not None:
                # A reflection labelled as a five-session outcome must not be
                # produced from a partial horizon merely because the historical
                # run date falls two sessions after the decision.
                if len(stock) <= holding_days or len(bench) <= holding_days:
                    return None, None, None
                actual_days = holding_days
            else:
                # Backwards-compatible library behaviour for callers that do
                # not request an as-of boundary.
                actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(
        self,
        ticker: str,
        as_of_date: str | None = None,
    ) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.

        Stale-pending observability: an entry whose trade_date is older than
        ``STALE_PENDING_DAYS`` (7) that still cannot be resolved is logged at
        WARNING so indefinitely-stuck pending entries are not silent.
        """
        try:
            cutoff = (
                datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d")
                if as_of_date is not None
                else datetime.now()
            )
        except (TypeError, ValueError):
            logger.warning("Invalid memory as-of date %r; skipping reflection", as_of_date)
            return

        cutoff_str = cutoff.strftime("%Y-%m-%d")
        pending = [
            entry
            for entry in self.memory_log.get_pending_entries()
            if entry["ticker"] == ticker
            and _entry_precedes_cutoff(entry, cutoff)
        ]
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(
                ticker,
                entry["date"],
                benchmark=benchmark,
                as_of_date=cutoff_str,
            )
            if raw is None:
                # Price not available yet — but if this entry is old, flag it so
                # it does not silently accumulate forever (delisted / bad data).
                _warn_if_stale_pending(ticker, entry["date"], cutoff)
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha if alpha is not None else 0.0,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
                "available_date": cutoff_str,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

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
        identity = resolve_instrument_identity(ticker, trade_date)
        return build_instrument_context(ticker, asset_type, identity)

    def _run_signature(self, asset_type: str) -> str:
        """Graph-shape inputs that must invalidate a checkpoint if changed.

        Folded into the checkpoint thread_id so resuming under a different
        selection of analysts, debate/risk depth, asset type, or parallel
        mode starts fresh instead of continuing a graph with a different
        shape. ``analyst_parallel`` is read from config (YiAgents-only; the
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
        :class:`~yiagents.risk.manager.PortfolioState` only takes effect
        when ``risk_enabled`` is set in config.
        """
        self.ticker = company_name

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

        from yiagents.graph.analyst_execution import (
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
        from yiagents.dataflows import quality

        quality.ensure_run_context()

        try:
            # Initialize state — inject memory log context for PM and the
            # deterministically resolved instrument identity for all agents.
            past_context = self.memory_log.get_past_context(
                company_name,
                as_of_date=str(trade_date),
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
            # the direction). No-op when risk_enabled is off.
            final_state = self._apply_risk_overlay(
                company_name, trade_date, final_state, portfolio_state,
            )

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

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file.

        Includes the structured evidence block the self-improvement loop
        consumes: ``pm_rating`` (the Portfolio Manager's structured rating,
        previously flattened into markdown only) and ``data_quality`` (the
        router's sentinel events for this run, so a degraded report is
        machine-distinguishable from a fully-fed one). Both are additive —
        older readers ignore unknown keys.
        """
        from yiagents.dataflows import quality

        quality_block = quality.summarize_quality(quality.snapshot_quality())
        quality.reset_quality()

        self.log_states_dict[str(trade_date)] = {
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
            # the router's sentinel events accumulated during this run.
            "pm_rating": final_state.get("pm_rating", ""),
            "data_quality": quality_block,
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "YiAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        # Atomic write: dump to a sibling temp file then os.replace() onto the
        # final path (same-volume rename is atomic on both Windows and POSIX).
        # A kill mid-write (run_robust's taskkill /F /T) would otherwise leave a
        # half-truncated JSON that masquerades as the "run complete" signal this
        # file is used as. Mirrors the atomic pattern in memory.py.
        tmp_path = directory / f".full_states_log_{trade_date}.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)
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
        ``YiAgentsStrategy_logs`` directory. Callers gate on
        ``self.perf_tracker is not None``; this method assumes a live tracker.
        """
        from yiagents.graph.perf_telemetry import dump_perf_report

        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "YiAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)
        dump_perf_report(self.perf_tracker, directory / f"node_perf_{trade_date}.json")

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
