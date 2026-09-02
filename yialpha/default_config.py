import os

_YIALPHA_HOME = os.path.join(os.path.expanduser("~"), ".yialpha")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "YIALPHA_LLM_PROVIDER":         "llm_provider",
    "YIALPHA_DEEP_THINK_LLM":       "deep_think_llm",
    "YIALPHA_QUICK_THINK_LLM":      "quick_think_llm",
    # Mid-tier model for the debate layer (bull/bear researchers + 3 risk
    # debators). These nodes do free-text adversarial argumentation across
    # multiple rounds — a task where the quick-tier model is weakest. When
    # unset (None), the debate layer falls back to deep_think_llm, i.e. the
    # same model as the Research/Portfolio Managers. Set explicitly (e.g.
    # deepseek-v4-flash) to A/B test a cheaper debate tier.
    "YIALPHA_DEBATE_LLM":           "debate_llm",
    "YIALPHA_LLM_BACKEND_URL":      "backend_url",
    "YIALPHA_OUTPUT_LANGUAGE":      "output_language",
    "YIALPHA_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "YIALPHA_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "YIALPHA_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "YIALPHA_MEMORY_ENABLED":       "memory_enabled",
    "YIALPHA_BENCHMARK_TICKER":     "benchmark_ticker",
    "YIALPHA_TEMPERATURE":          "temperature",
    # Provider-specific reasoning/thinking knobs (None = each provider's own
    # default). Settable here for non-interactive runs; the CLI also offers an
    # interactive choice, which is skipped when the matching var is set.
    "YIALPHA_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "YIALPHA_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "YIALPHA_ANTHROPIC_EFFORT":        "anthropic_effort",
    # Quantitative risk-control layer (Phase 1). Off by default so the Phase-0
    # baseline (LLM-driven sizing) stays reproducible; flip on to let the risk
    # manager override position size, stop-loss and exposure deterministically.
    "YIALPHA_RISK_ENABLED":            "risk_enabled",
    "YIALPHA_KELLY_FRACTION":          "kelly_fraction",
    "YIALPHA_MAX_POSITION":            "max_single_position",
    "YIALPHA_MAX_SECTOR":              "max_single_sector",
    "YIALPHA_MAX_DRAWDOWN":            "max_drawdown_hard_stop",
    "YIALPHA_ATR_STOP_MULT":           "atr_stop_mult",
    # Phase 2b: swap the analyst persona prompts for compact FinCoT structured
    # prompts (task -> reasoning steps -> output constraints). Off by default so
    # the baseline prompt shape -- and model behaviour -- stays reproducible.
    "YIALPHA_FIN_COT_PROMPTS":         "fin_cot_prompts",
    # Phase 4: global kill switch. When truthy, the browser broker refuses to
    # submit any order (positions are flattened manually / via the broker). The
    # execution layer reads the same env var directly so a halt takes effect
    # without restarting the agent process.
    "YIALPHA_KILL_SWITCH":             "kill_switch",
    # Safety boundary for analysis-only installations. Live execution requires
    # analysis_only=False plus both execution enable switches at the network
    # edge; PoT host-side code execution is independently opt-in.
    "YIALPHA_ANALYSIS_ONLY":            "analysis_only",
    "YIALPHA_LIVE_EXECUTION_ENABLED":   "live_execution_enabled",
    "YIALPHA_POT_ENABLED":              "pot_enabled",
    # Multi-ticker batch concurrency (Phase A/B). Off by default so every path
    # stays strictly serial (one ticker at a time) and reproducible. See
    # yialpha/batch/runner.py — each ticker runs through propagate() unchanged;
    # concurrency is layered above the graph, never inside an agent.
    "YIALPHA_BATCH_CONCURRENCY":       "batch_concurrency",
    "YIALPHA_BATCH_WORKERS":           "batch_workers",
    "YIALPHA_BATCH_DEDUP_TICKERS":     "batch_dedup_tickers",
    "YIALPHA_BATCH_MEMORY_LOCK":       "batch_memory_lock",
    "YIALPHA_BATCH_OHLCV_LOCK":        "batch_ohlcv_lock",
    "YIALPHA_BATCH_FAIL_FAST":         "batch_fail_fast",
    # Phase C: optional shared LLM rate limiter (DeepSeek RPM ceiling). Off by
    # default — rely on per-call retries for transient 429s until the ceiling is
    # measured (run the G4 instrumentation). Never changes reasoning params.
    "YIALPHA_LLM_RATE_LIMITER":        "llm_rate_limiter",
    "YIALPHA_LLM_RPM":                 "llm_rpm",
    # P1a: share one httpx.Client (keepalive) across every LLM client so the K
    # worker graphs reuse TLS/proxy connections. Transport-only; off by default.
    "YIALPHA_HTTP_KEEPALIVE":          "http_keepalive",
    # P0: stream the graph + record per-analyst wall time. Observation only
    # (serial graph final state == invoke); off by default.
    "YIALPHA_STREAM_TELEMETRY":        "stream_telemetry",
    # P0+: node-level wall-time + token telemetry (NodePerfTracker wraps every
    # graph node handler). Observation only; off by default = handlers pass
    # through unwrapped (byte-equivalent). See yialpha/graph/perf_telemetry.py.
    "YIALPHA_NODE_PERF_TELEMETRY":     "node_perf_telemetry",
    # T1.3: per-call LLM retry count forwarded to the provider client via
    # _PASSTHROUGH_KWARGS (max_retries). Default 2 == langchain-openai's own
    # default, so UNSET behaviour is byte-equivalent; expose so flaky periods
    # can tune it (run_robust's per-ticker rerun is the outer safety net).
    "YIALPHA_LLM_MAX_RETRIES":         "llm_max_retries",
    # T3: per-call LLM response disk cache (langchain global set_llm_cache).
    # Off by default = byte-equivalent (no cache, no I/O). When on, an identical
    # (model + prompt + temperature + bound tools) replays the cached
    # ChatGeneration instead of re-calling the model — saves the ~11
    # intermediate agent calls when re-running the same smoke / single analysis
    # while iterating on prompts or risk code. DO NOT combine with
    # scripts/run_analyst_parallel_ab.py or run_baseline --full (DSR n_trials):
    # caching collapses the temperature>0 run-to-run variability those
    # distribution checks measure. Backtest whole-graph replay is already free
    # via yialpha/backtest/cache.py DecisionCache.
    "YIALPHA_LLM_CACHE":               "llm_cache",
    # T2: fan the 4 analysts out inside ONE wrapper node (each analyst runs in
    # its own sub-graph with its own state, so the shared `messages` /
    # clear_node coupling that assumes serial execution is structurally
    # avoided). OFF by default = today's serial analyst chain runs verbatim.
    # Flip on only after scripts/run_analyst_parallel_ab.py passes its gate.
    "YIALPHA_ANALYST_PARALLEL":             "analyst_parallel",
    "YIALPHA_ANALYST_PARALLEL_MAX_THREADS": "analyst_parallel_max_threads",
    # Deterministic intrinsic-value math (Graham number / NCAV / PEG / DCF /
    # WACC / margin of safety) exposed to the fundamentals analyst as a
    # Program-of-Thought tool. OFF by default = the analyst's tool list, and
    # therefore its prompt and capabilities, are byte-for-byte unchanged when
    # the flag is unset. The tool only adds a deterministic arithmetic sink for
    # line items the analyst already gathers; it changes no agent input.
    "YIALPHA_VALUATION_TOOLS":              "valuation_tools",
    # Binance IP-weight proactive backoff (perp data vendor). When on, the
    # vendor reads X-MBX-USED-WEIGHT-1M and backs off before the ceiling. Off
    # by default = the vendor neither reads the header nor sleeps (byte-equivalent).
    "YIALPHA_BINANCE_PROACTIVE_BACKOFF":    "binance_proactive_backoff",
    "YIALPHA_BINANCE_WEIGHT_THRESHOLD":     "binance_weight_threshold",
    # Binance SPOT host: when on, the spot vendor uses the key-free market-data
    # mirror data-api.binance.vision instead of api.binance.com. Off by default
    # (api.binance.com is the proven host through the SOCKS5 proxy); the mirror
    # is Binance's recommended read-only host and carries the same data.
    "YIALPHA_BINANCE_SPOT_MIRROR":          "binance_spot_mirror",
    # Binance transport resilience (read-only public market data). keepalive
    # reuses one requests.Session across calls; retries recover transient
    # DNS/timeout/TLS/5xx without an OS-level run_robust rerun; honor_retry_after
    # backs off a real IP ban. All default-off/zero = byte-equivalent to today.
    "YIALPHA_BINANCE_HTTP_KEEPALIVE":       "binance_http_keepalive",
    "YIALPHA_BINANCE_HTTP_RETRIES":         "binance_http_retries",
    "YIALPHA_BINANCE_HONOR_RETRY_AFTER":    "binance_honor_retry_after",
    # Track B2 + B2.1: SEC ownership & short-interest (Form 4 insider trading,
    # fails-to-deliver, 13F institutional holdings) exposed to the fundamentals
    # analyst as opt-in tools behind one flag. Off by default = the analyst's
    # tool list / prompt / capabilities are byte-for-byte unchanged when the
    # flag is unset (same contract as valuation_tools). The tools only add
    # PIT-correct US-only signals the analyst may consult; they change no agent
    # input.
    "YIALPHA_SEC_OWNERSHIP":                "sec_ownership",
    # FTD point-in-time: a semi-monthly cutoff file is treated as public
    # `cutoff + ftd_pub_lag_days` days later (conservative; SEC publishes a few
    # days after the cutoff). Default 10 keeps backtests from peeking at a file
    # that had not yet been disseminated on curr_date.
    "YIALPHA_FTD_PUB_LAG_DAYS":             "ftd_pub_lag_days",
    # 13F point-in-time: a bulk Form 13F Data Set ZIP (period-end = last day of
    # Feb/May/Aug/Nov) is treated as public `period-end + sec_13f_pub_lag_days`
    # days later. SEC publishes each quarterly ZIP ~45 days after the window
    # closes (statutory deadline), so the default mirrors that: during the
    # publication gap the tool reports the honest "not yet published" message
    # instead of fetching a 404 and blaming the symbol.
    "YIALPHA_SEC_13F_PUB_LAG_DAYS":         "sec_13f_pub_lag_days",
    # Stale-cache age cap for the shared disk cache (disk_cache.cached_or_fetch):
    # when a vendor fails, a cache older than this many days is refused instead
    # of served. Default 30 bounds how old "stale fallback" data can get during
    # an outage; 0 disables stale serving entirely (fully fail-closed).
    "YIALPHA_DATA_CACHE_MAX_STALE_DAYS":    "data_cache_max_stale_days",
    # Track A: China A-share margin trading (Eastmoney 融资融券) exposed to the
    # fundamentals analyst as an opt-in tool behind one flag, AND only for A-share
    # tickers (.SS/.SH/.SZ) via the is_a_stock double-gate. Off by default = the
    # analyst's tool list / prompt / capabilities are byte-for-byte unchanged when
    # the flag is unset (same contract as valuation_tools). The tool only adds a
    # PIT-correct A-share-only signal the analyst may consult; it changes no agent
    # input. The vendor connects directly (bypassing the SOCKS5 VPN proxy) since
    # Eastmoney is a domestic source.
    "YIALPHA_A_STOCK":                      "a_stock",
    # Native A-share OHLC + TTM valuation + news / money flow / dragon-tiger
    # (a_share_native) exposed to the fundamentals + news analysts as opt-in
    # tools behind one flag, AND only for A-share tickers (.SS/.SH/.SZ) via the
    # is_a_stock double-gate. Off by default = the analyst tool list / prompt /
    # capabilities are byte-for-byte unchanged when the flag is unset (same
    # contract as a_stock / sec_ownership). When on, PIT-correct 前复权 OHLC +
    # server-computed TTM/MRQ multiples (PE/PB/PS/PCF) are appended so the
    # analyst can consult native A-share data the default yfinance path covers
    # thinly, plus 资金流 (主力 net inflow) and 龙虎榜 (dragon-tiger) smart-money
    # signals via AKShare. The BaoStock vendor reaches a domestic TCP socket (no
    # proxy bypass needed); the AKShare vendor pops the SOCKS5 proxy env around
    # its domestic HTTP calls. Optional deps are lazy-imported, so default-off
    # runs never import baostock/akshare (zero overhead).
    "YIALPHA_A_SHARE_NATIVE":               "a_share_native",
    # Market turbulence index (FinRL-derived, see yialpha.dataflows.market_regime).
    # Off by default = the conservative risk debater's prompt is byte-for-byte
    # unchanged (same opt-in contract as sec_ownership / valuation_tools). When
    # on, a one-line market-stress reading (benchmark 252d squared-z) is appended
    # to the conservative debater's data sources — a market-level ex-ante cue
    # complementary to the portfolio-level reactive DrawdownBreaker. Advisory and
    # fail-soft; it changes agent input only when explicitly enabled.
    "YIALPHA_MARKET_REGIME":                "market_regime",
    # Composite regime context (see yialpha.dataflows.market_regime.
    # format_regime_context): one structured line — ticker trend state
    # (close vs 50/200 SMA + ADX), volatility state (rvol_20 percentile vs
    # trailing year), benchmark turbulence, and (A-share live runs)
    # whole-market breadth — appended to the market analyst's prompt. ON by
    # default (it is deterministic, advisory, and fail-soft: any component
    # that cannot be computed is omitted); set false to restore the bare
    # prompt. The pre-existing YIALPHA_MARKET_REGIME turbulence-only
    # opt-in for the conservative debater is unchanged.
    "YIALPHA_REGIME_CONTEXT":               "regime_context",
    # Deterministic perp market bundle (see the perp_market_bundle comment
    # in DEFAULT_CONFIG): crypto_perp market analyst prefetch injection.
    "YIALPHA_PERP_BUNDLE":                  "perp_market_bundle",
    # Deterministic fundamentals bundle (see the fundamentals_bundle comment
    # in DEFAULT_CONFIG): fundamentals analyst prefetch injection.
    "YIALPHA_FUNDAMENTALS_BUNDLE":          "fundamentals_bundle",
    # Vision archive summary mode (see the binance_vision_summary comment
    # in DEFAULT_CONFIG): distribution + tail instead of raw daily CSV.
    "YIALPHA_VISION_SUMMARY":               "binance_vision_summary",
    # Data-vacuum gate policy: "reject" (default — a run with zero successful
    # core data calls fails at the trader node with DataVacuumError instead of
    # producing a data-vacuum HOLD report) or "warn" (old behaviour: report +
    # DEGRADED banner). The interactive analyze CLI forces "warn" unless this
    # env var is explicitly set; batch / run_robust inherit the reject default.
    "YIALPHA_DATA_VACUUM_POLICY":           "data_vacuum_policy",
    # Trailing-IC advisory context for the market analyst (see the
    # indicator_ic_context comment in DEFAULT_CONFIG). Default off keeps the
    # prompt byte-equivalent to the A/B baseline.
    "YIALPHA_INDICATOR_IC_CONTEXT":         "indicator_ic_context",
    # V2.1 Measurability (record/shadow stage): persistent instrument
    # registry, blind-prediction ledger, stock-perp fair-value bridge. All
    # three record and disclose without changing legacy decisions; see the
    # matching comments in DEFAULT_CONFIG.
    "YIALPHA_INSTRUMENT_REGISTRY":          "instrument_registry",
    "YIALPHA_PREDICTION_LEDGER":            "prediction_ledger",
    "YIALPHA_STOCK_PERP_FAIR_VALUE":        "stock_perp_fair_value",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value.

    Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended run.
    """
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply YIALPHA_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("YIALPHA_RESULTS_DIR", os.path.join(_YIALPHA_HOME, "logs")),
    "data_cache_dir": os.getenv("YIALPHA_CACHE_DIR", os.path.join(_YIALPHA_HOME, "cache")),
    "memory_log_path": os.getenv("YIALPHA_MEMORY_LOG_PATH", os.path.join(_YIALPHA_HOME, "memory", "trading_memory.md")),
    # Reflections contain realised returns and therefore require strict
    # point-in-time bookkeeping. Keep persistence opt-in for an analysis-only
    # installation; when enabled, historical runs additionally filter every
    # lesson by its recorded outcome-availability date.
    "memory_enabled": False,
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    # Debate-tier model: bull/bear researchers + risk debators. None = fall
    # back to deep_think_llm (the debate layer runs on the same model as the
    # final-decision nodes). Override via YIALPHA_DEBATE_LLM to A/B test a
    # cheaper model for the adversarial-argumentation layer.
    "debate_llm": None,
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Sampling temperature, forwarded to every provider when set. None leaves
    # each provider at its own default. Lower values reduce run-to-run
    # variation on models that honor it; reasoning models largely ignore it
    # and no setting makes LLM output bit-identical across runs (see README).
    "temperature": None,
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings. conditional_logic terminates the
    # investment debate at 2*N beats and the risk debate at 3*N beats, so N=1
    # stops Bull before it can answer Bear's rebuttal; N=2 yields a full
    # Bull->Bear->Bull->Bear cycle (and a 6-beat risk debate) -- the first
    # setting where both sides actually exchange rebuttals. max_recur_limit=100
    # comfortably absorbs the extra beats (longest path stays well under 100).
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 2,
    "max_recur_limit": 100,
    # --- Quantitative risk-control layer (Phase 1) -------------------------
    # On by default for analysis reports. The risk manager overrides analytical
    # position size / stop-loss / exposure deterministically (LLM keeps
    # direction; math owns size and risk). scripts/run_baseline.py sets this
    # explicitly per mode (baseline=False, full=True) so the A/B gate stays
    # clean regardless of this default; smoke inherits the default to exercise
    # the full analysis path.
    "risk_enabled": True,
    "kelly_fraction": 0.25,            # quarter-Kelly by default
    "max_single_position": 0.20,       # one ticker <= 20% of equity
    "max_single_sector": 0.30,         # one sector <= 30% of equity
    "max_drawdown_hard_stop": 0.15,    # flatten + cool off beyond this drawdown
    "atr_stop_mult": 2.0,              # stop = last_close - mult*ATR (long)
    # Analysis is the product's default boundary. These config values document
    # the mode for callers/UI; execution gateways additionally read the same
    # environment flags at each submission so a stop takes effect immediately.
    "analysis_only": True,
    "live_execution_enabled": False,
    # The in-process PoT namespace is not an OS sandbox. Keep LLM-generated
    # Python disabled unless the operator makes a separate explicit opt-in.
    "pot_enabled": False,
    # Phase 2b: FinCoT de-persona structured prompts for analysts.
    "fin_cot_prompts": False,
    # Market-analyst indicator battery (self-improvement landing point).
    # None (default) = full catalog, byte-identical to the baseline prompt.
    # A list of indicator names prunes the catalog the market analyst selects
    # from — apply the output of scripts/prune_indicators_cli.py here after
    # human review (never auto-edited). Valid names are pinned by
    # INDICATOR_NAMES in agents/analysts/market_analyst.py and validated by
    # `yialpha config-check`.
    "indicator_battery": None,
    # Trailing-IC advisory context (env: YIALPHA_INDICATOR_IC_CONTEXT). Off by
    # default = the market analyst's prompt is byte-identical to the A/B
    # baseline. When on, one advisory line per indicator (trailing mean |IC|
    # averaged over ic_data/*.prune.json verdicts) is appended to the system
    # message — the runtime consumer of the IC-pruning evidence. Fail-soft:
    # no verdicts on disk → no line, prompt unchanged.
    "indicator_ic_context": False,
    # Deterministic valuation tool (env: YIALPHA_VALUATION_TOOLS). Off by
    # default = the fundamentals analyst's tool list is unchanged (byte-
    # equivalent). When on, a get_valuation_metrics PoT tool is appended so the
    # analyst can delegate intrinsic-value arithmetic to Python rather than
    # confabulating it.
    "valuation_tools": False,
    # Track B2: SEC ownership & short-interest (env: YIALPHA_SEC_OWNERSHIP).
    # Off by default = the fundamentals analyst's tool list is unchanged
    # (byte-equivalent). When on, two PIT-correct US-only tools (Form 4 insider
    # trading + fails-to-deliver) are appended so the analyst can consult
    # ownership/short-interest signals the other vendors don't supply.
    "sec_ownership": False,
    # FTD point-in-time publication lag (env: YIALPHA_FTD_PUB_LAG_DAYS). A
    # semi-monthly cutoff file is treated as public `cutoff + this` days later;
    # 10 is conservative so a backtest can't peek at a not-yet-disseminated file.
    "ftd_pub_lag_days": 10,
    # 13F bulk-data-set point-in-time publication lag (env:
    # YIALPHA_SEC_13F_PUB_LAG_DAYS). A quarterly ZIP is treated as public
    # `period-end + this` days later; SEC's statutory deadline is 45 days
    # after quarter-end, which the default mirrors.
    "sec_13f_pub_lag_days": 45,
    # Stale-cache age cap (env: YIALPHA_DATA_CACHE_MAX_STALE_DAYS). When a
    # vendor fails, a disk cache older than this many days is refused instead
    # of served (fail-closed bound on outage staleness). 0 = never serve stale.
    "data_cache_max_stale_days": 30,
    # Track A: China A-share margin trading (env: YIALPHA_A_STOCK). Off by
    # default = the fundamentals analyst's tool list is unchanged (byte-
    # equivalent). Double-gated: when on AND the ticker is an A-share
    # (.SS/.SH/.SZ), one PIT-correct A-share-only tool (margin trading 融资融券)
    # is appended so the analyst can consult a money-flow signal the default
    # yfinance path cannot supply. US/crypto/HK tickers never enter the branch.
    "a_stock": False,
    # Native A-share data (env: YIALPHA_A_SHARE_NATIVE). Off by default = the
    # fundamentals/news analyst tool lists are unchanged (byte-equivalent).
    # Double-gated: when on AND the ticker is an A-share (.SS/.SH/.SZ),
    # PIT-correct A-share-only tools are appended — OHLC qfq + TTM valuation
    # (BaoStock), 资金流 money flow + 龙虎榜 dragon-tiger (AKShare), and 东财 news
    # (AKShare) — so the analyst can consult native data the default yfinance
    # path covers thinly/sparsely. US/crypto/HK tickers never enter the branch.
    # baostock/akshare are optional lazy-imported dependencies, so default-off
    # runs (and machines without the 'a-share' extra) are unaffected.
    "a_share_native": False,
    # Market turbulence index (env: YIALPHA_MARKET_REGIME). Off by default =
    # the conservative risk debater's prompt is unchanged (byte-equivalent).
    # When on, a one-line benchmark market-stress reading (252d rolling
    # squared-z, FinRL-derived) is appended to the conservative debater's data
    # sources. Advisory + fail-soft; only the conservative debater sees it.
    "market_regime": False,
    # Composite regime context (env: YIALPHA_REGIME_CONTEXT). ON by default:
    # the market analyst's prompt gains one structured regime line (trend
    # state + volatility state + benchmark turbulence +, on A-share live
    # runs, market breadth). Deterministic and fail-soft — components that
    # cannot be computed are omitted, and the line disappears entirely when
    # neither trend nor volatility state has enough history. Set false to
    # restore the pre-2026-08-15 bare prompt.
    "regime_context": True,
    # Deterministic perp market bundle (see yialpha.dataflows.perp_bundle).
    # ON by default for crypto_perp runs: before the market analyst's LLM
    # turn, ONE parallel prefetch assembles the decision-critical numbers
    # (last/mark/index closes + bases, funding, OI, 3-vantage LSR, taker
    # flow, fixed-bps depth bands with slippage estimates, ADL, spot-perp
    # basis) and injects them as advisory context — the LLM no longer
    # decides WHETHER the core market facts get fetched. Every component is
    # fail-soft with per-field status disclosed; the tools stay bound for
    # drill-down; a missing CORE price leg records a quality sentinel that
    # vetoes the ticket (NO_TRADE). Set false to restore the tools-only
    # prompt.
    "perp_market_bundle": True,
    # Deterministic fundamentals bundle (see
    # yialpha.dataflows.fundamentals_bundle). ON by default: before the
    # fundamentals analyst's LLM turn, ONE parallel prefetch assembles the
    # core fundamentals evidence — the merged SEC+Yahoo overview and the
    # three quarterly statements (for ETFs the fund snapshot REPLACES the
    # company statements; native A-share statements ride along when
    # a_share_native is on) — and rides the final USER message as marked
    # untrusted evidence. The LLM no longer decides WHETHER the core
    # fundamentals get fetched; fetches run through route_to_vendor inside
    # submit_with_context workers so the quality ledger sees the same
    # evidence a model-issued tool call produces, and run_cached pins the
    # fetch to once per run however many tool-loop re-entries the node
    # sees. Fail-soft per component with status disclosed. Set false to
    # restore the tools-only prompt.
    "fundamentals_bundle": True,
    # Vision archive summary mode (see yialpha.dataflows.binance_vision):
    # at the daily grain, the vision tools render a distribution table over
    # the FULL multi-year window + a recent tail (metrics: per-column
    # stats + 30-day tail; bookDepth: per-band stats + liquidity-thin
    # streak + 14-day tail) instead of raw CSV rows. ON by default — the
    # LLM-facing answer for deep history is statistics, not thousands of
    # rows. Set false (or pass summary=false per call) for the raw daily
    # CSV, subject to the output cap.
    "binance_vision_summary": True,
    # --- V2.1 Measurability (record/shadow; perp runs only) --------------------
    # Persistent instrument registry (yialpha.instruments.registry). ON by
    # default: every exchangeInfo warm snapshots ALL USDT-M symbols
    # (underlyingType, filters, onboard dates) into the central ledger DB, so
    # perp classification (stock_perp / pure_crypto_perp / unknown_perp) is
    # evidence-based, point-in-time (available_at <= analysis_as_of) and
    # survives restarts. Record stage: an unclassifiable perp becomes
    # unknown_perp with disclosure but does NOT yet veto the ticket (that
    # enforcement arrives with the V2.4 enforced mode). Set false to keep the
    # pre-V2.1 in-memory classification only (byte-equivalent).
    "instrument_registry": True,
    # Blind analyst predictions (yialpha.ledger.predictions). ON by default:
    # the submit_prediction tool is appended to the four analysts' toolkits
    # on crypto_perp runs ONLY, filing one immutable, pre-debate,
    # multi-horizon forecast set per analyst per run into the prediction
    # ledger. Record stage: predictions never influence decisions. Set false
    # for byte-equivalent tool lists.
    "prediction_ledger": True,
    # Stock-perp fair-value bridge (yialpha.perp.fair_value). ON by default:
    # converts a USD underlying target into a USDT contract target via
    # USDT/USD (inverted Binance spot USDCUSDT price) plus expected basis,
    # recording the full conversion chain on the ticket. Record stage: a
    # missing FX rate records a shadow DEGRADED_CRITICAL verdict without
    # vetoing. Set false to disable the bridge entirely.
    "stock_perp_fair_value": True,
    # Central SQLite ledger DB (env YIALPHA_LEDGER_DB). One append-first
    # database holds instrument snapshots, run evidence, blind predictions,
    # outcomes and ticket mirrors (V2.4 adds positions/portfolio snapshots).
    "ledger_db_path": os.getenv(
        "YIALPHA_LEDGER_DB", os.path.join(_YIALPHA_HOME, "ledger", "portfolio.db")
    ),
    # Phase 4: global kill switch (env: YIALPHA_KILL_SWITCH). Halt = no
    # new orders submitted by the browser broker; read live at order time.
    "kill_switch": False,
    # --- Multi-ticker batch concurrency ---------------------------------------
    # Master switch OFF by default: every entry point runs strictly serial
    # (K=1, one ticker at a time) and is byte-equivalent to today. Flip on to
    # fan a ticker list out across a pool of worker graphs. Concurrency lives
    # above propagate() — agent inputs/depth/reasoning params are unchanged.
    "batch_concurrency": False,
    "batch_workers": 3,            # K: max concurrent tickers (pool size)
    "batch_dedup_tickers": True,   # forbid duplicate tickers in one batch
    "batch_memory_lock": True,     # serialize memory-log read-modify-write
    "batch_ohlcv_lock": True,      # serialize per-symbol OHLCV cache read/write
    "batch_fail_fast": False,      # False = a failed ticker doesn't abort the batch
    # Phase C: optional shared DeepSeek rate limiter (requests-per-minute).
    # Off by default; size llm_rpm from measured RPM before enabling.
    "llm_rate_limiter": False,
    "llm_rpm": 60,
    # Binance IP-weight proactive backoff for the perp data vendor. When on,
    # the vendor reads the X-MBX-USED-WEIGHT-1M response header and sleeps
    # before the per-minute ceiling so it avoids a 429 rather than only
    # reacting to one. Off by default = the vendor neither reads the header
    # nor sleeps, byte-equivalent to today. Per-product documented budgets
    # apply when binance_weight_threshold is unset: fapi 2400/min, spot
    # 6000/min (2026-08-16); set it only to override BOTH (e.g. a shared
    # VPN-exit IP that must stay extra conservative).
    "binance_proactive_backoff": False,
    "binance_weight_threshold": 0,
    # Binance SPOT host mirror switch (env: YIALPHA_BINANCE_SPOT_MIRROR). Off
    # by default = spot vendor hits api.binance.com (proven through the SOCKS5
    # proxy); on = key-free market-data mirror data-api.binance.vision. The
    # spot vendor reads this at call time, so it is byte-equivalent to today
    # when off (and spot is new code regardless, so no prior output to perturb).
    "binance_spot_mirror": False,
    # Binance transport resilience (env: YIALPHA_BINANCE_HTTP_*). All
    # default-off/zero = byte-equivalent to today (per-call fresh requests.get,
    # no retry, no Retry-After sleep). keepalive reuses the TLS/SOCKS5 connection
    # across calls (mirrors http_keepalive for the LLM client); retries recover
    # transient DNS/timeout/TLS/5xx with exponential backoff (mirrors yf_retry,
    # exhausted → NoMarketDataError); honor_retry_after sleeps the server's
    # Retry-After window (≤60s) before raising on 429/418.
    "binance_http_keepalive": False,
    "binance_http_retries": 0,
    "binance_honor_retry_after": False,
    # P1a: process-wide shared httpx.Client for LLM calls — concurrent worker
    # graphs reuse TLS/SOCKS5-proxy connections instead of opening one per call.
    # Transport-only (changes nothing sent to the model); off by default.
    "http_keepalive": False,
    # P0: stream the graph (stream_mode="values") and record per-analyst wall
    # time via AnalystWallTimeTracker. The graph is fully serial, so the final
    # values chunk is identical to graph.invoke(); this adds observation only.
    "stream_telemetry": False,
    # P0+: node-level perf telemetry — per-node wall time + token totals,
    # dumped to node_perf_<date>.json next to full_states_log. Off by default
    # = node handlers pass through unwrapped, i.e. byte-equivalent to today.
    "node_perf_telemetry": False,
    # T1.3: per-call retry count forwarded to provider clients via
    # _PASSTHROUGH_KWARGS (max_retries). Default 2 == langchain-openai's own
    # default, so leaving it at 2 is byte-equivalent to the prior behaviour.
    "llm_max_retries": 2,
    # T3: per-call LLM response disk cache (env: YIALPHA_LLM_CACHE). Off by
    # default = no set_llm_cache call, byte-equivalent to today. When on, a
    # DiskLLMCache is installed via langchain's global set_llm_cache and replay
    # identical (model, prompt, temperature, bound tools) from disk. Iteration-
    # speed tool only — never use with the A/B gate or DSR multi-run; see
    # yialpha/llm_clients/response_cache.py docstring.
    "llm_cache": False,
    # T2: parallel analysts inside one wrapper node. OFF by default = today's
    # serial analyst chain verbatim. max_threads caps nested concurrency
    # (batch_workers * 4); above the cap the runner silently falls back to
    # serial analysts per graph so total in-flight LLM calls stay predictable.
    "analyst_parallel": False,
    "analyst_parallel_max_threads": 16,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Open-web search (Tavily) for the news/market/fundamentals analysts
    # (live dates only — historical replays never see the tool, since Tavily
    # has no as-of parameter). On by default: the tool degrades to a
    # WEB_SEARCH_UNAVAILABLE sentinel + data_quality event when no key is
    # set (TAVILY_API_KEYS pool or single TAVILY_API_KEY) or the per-run
    # budget (YIALPHA_TAVILY_MAX_CALLS_PER_RUN, default 15 total; split
    # news:8/market:5/fundamentals:2, override with
    # YIALPHA_TAVILY_BUDGET_SPLIT) is exhausted, so enabling it never
    # aborts a run. Set False for byte-for-byte pre-Tavily prompts.
    "web_search_enabled": True,
    # Binance Square feed (crypto-native retail sentiment) injected into the
    # sentiment analyst's prompt. Live crypto runs only (asset_type
    # crypto/crypto_spot/crypto_perp): the feed is a current snapshot with no
    # as-of parameter, so historical replays omit it; stock runs never see it
    # (byte-identical three-source prompts). Free and keyless — an anonymous
    # device id is generated per process. Transport/parse failures degrade to
    # a placeholder + data_quality event, never aborting a run. Set False for
    # byte-for-byte pre-Binance-Square prompts.
    "binance_square_enabled": True,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category).
    # The configured value is the exact vendor chain — requests are NOT silently
    # routed to vendors you didn't choose. For ordered fallback, list several,
    # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
    # The four core categories chain yfinance -> alpha_vantage (fundamentals
    # additionally -> sec_edgar) so a single-vendor outage degrades to the
    # backup instead of a data vacuum; yfinance stays first in the price/
    # indicator/news chains because it is keyless and unlimited —
    # alpha_vantage's free tier is rate-limited, so it must only ever serve
    # as the tail of the chain. FUNDAMENTALS are the deliberate exception
    # (PR5, 2026-09): SEC EDGAR leads because its filings carry the REAL
    # ``filed`` date — period visibility is point-in-time ground truth, not
    # the period_end+45-day heuristic yfinance/AV are stuck with (a 10-Q
    # filed 38 days after period end is genuinely public at day 38).
    "data_vendors": {
        "core_stock_apis": "yfinance,alpha_vantage",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance,alpha_vantage",  # Options: alpha_vantage, yfinance
        "fundamental_data": "sec_edgar,yfinance,alpha_vantage",  # Options: alpha_vantage, yfinance, sec_edgar
        "news_data": "yfinance,alpha_vantage",             # Options: alpha_vantage, yfinance
        "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
        "prediction_markets": "polymarket",  # Options: polymarket (keyless)
        # a_stock tools are single-vendor (eastmoney); the category is optional
        # and only advertised for A-share tickers, so no category-level default
        # is needed — route_to_vendor uses the sole configured vendor.
    },
    # Data-vacuum gate (env: YIALPHA_DATA_VACUUM_POLICY). When "reject", a run
    # whose core data calls ALL failed raises DataVacuumError at the trader
    # node instead of producing a normal-looking HOLD report decided without
    # any market/fundamental/news data. "warn" keeps the old behaviour (report
    # + DEGRADED banner). The interactive CLI degrades this to "warn" on its
    # own (a human is watching); unattended batch/robust paths inherit reject.
    "data_vacuum_policy": "reject",
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # The overview MERGES SEC filing facts with Yahoo's real-time
        # valuation (market cap, PE, beta; live runs only) instead of riding
        # the category chain, where SEC-first would serve its seven filing
        # facts and never read the valuation fields. The STATEMENTS stay on
        # the SEC-first category chain above — filings are their PIT ground
        # truth. Set "get_fundamentals": "sec_edgar,yfinance,alpha_vantage"
        # (or any chain) to restore plain chaining.
        "get_fundamentals": "fundamentals_overview",
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",       # NSE India (Nifty 50)
        ".BO":  "^BSESN",      # BSE India (Sensex)
        ".T":   "^N225",       # Tokyo (Nikkei 225)
        ".HK":  "^HSI",        # Hong Kong (Hang Seng)
        ".L":   "^FTSE",       # London (FTSE 100)
        ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
        ".AX":  "^AXJO",       # Australia (ASX 200)
        # A-shares benchmark against CSI 300 (沪深300, 000300.SS) — the
        # investable large-cap index fund flows actually track — rather than
        # the SSE Composite, which is dominated by state-owned banks and
        # makes almost any growth stock look like it has index alpha. Both
        # .SS and .SZ map to the same cross-market index on purpose.
        ".SS":  "000300.SS",   # China A-share (CSI 300)
        ".SZ":  "000300.SS",   # China A-share (CSI 300)
        "":     "SPY",         # default for US-listed tickers (no suffix)
    },
})
