import os

_YIAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".yiagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "YIAGENTS_LLM_PROVIDER":         "llm_provider",
    "YIAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "YIAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "YIAGENTS_LLM_BACKEND_URL":      "backend_url",
    "YIAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "YIAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "YIAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "YIAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "YIAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "YIAGENTS_TEMPERATURE":          "temperature",
    # Provider-specific reasoning/thinking knobs (None = each provider's own
    # default). Settable here for non-interactive runs; the CLI also offers an
    # interactive choice, which is skipped when the matching var is set.
    "YIAGENTS_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "YIAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "YIAGENTS_ANTHROPIC_EFFORT":        "anthropic_effort",
    # Quantitative risk-control layer (Phase 1). Off by default so the Phase-0
    # baseline (LLM-driven sizing) stays reproducible; flip on to let the risk
    # manager override position size, stop-loss and exposure deterministically.
    "YIAGENTS_RISK_ENABLED":            "risk_enabled",
    "YIAGENTS_KELLY_FRACTION":          "kelly_fraction",
    "YIAGENTS_MAX_POSITION":            "max_single_position",
    "YIAGENTS_MAX_SECTOR":              "max_single_sector",
    "YIAGENTS_MAX_DRAWDOWN":            "max_drawdown_hard_stop",
    "YIAGENTS_ATR_STOP_MULT":           "atr_stop_mult",
    # Phase 2b: swap the analyst persona prompts for compact FinCoT structured
    # prompts (task -> reasoning steps -> output constraints). Off by default so
    # the baseline prompt shape -- and model behaviour -- stays reproducible.
    "YIAGENTS_FIN_COT_PROMPTS":         "fin_cot_prompts",
    # Phase 4: global kill switch. When truthy, the browser broker refuses to
    # submit any order (positions are flattened manually / via the broker). The
    # execution layer reads the same env var directly so a halt takes effect
    # without restarting the agent process.
    "YIAGENTS_KILL_SWITCH":             "kill_switch",
    # Multi-ticker batch concurrency (Phase A/B). Off by default so every path
    # stays strictly serial (one ticker at a time) and reproducible. See
    # yiagents/batch/runner.py — each ticker runs through propagate() unchanged;
    # concurrency is layered above the graph, never inside an agent.
    "YIAGENTS_BATCH_CONCURRENCY":       "batch_concurrency",
    "YIAGENTS_BATCH_WORKERS":           "batch_workers",
    "YIAGENTS_BATCH_DEDUP_TICKERS":     "batch_dedup_tickers",
    "YIAGENTS_BATCH_MEMORY_LOCK":       "batch_memory_lock",
    "YIAGENTS_BATCH_OHLCV_LOCK":        "batch_ohlcv_lock",
    "YIAGENTS_BATCH_FAIL_FAST":         "batch_fail_fast",
    # Phase C: optional shared LLM rate limiter (DeepSeek RPM ceiling). Off by
    # default — rely on per-call retries for transient 429s until the ceiling is
    # measured (run the G4 instrumentation). Never changes reasoning params.
    "YIAGENTS_LLM_RATE_LIMITER":        "llm_rate_limiter",
    "YIAGENTS_LLM_RPM":                 "llm_rpm",
    # P1a: share one httpx.Client (keepalive) across every LLM client so the K
    # worker graphs reuse TLS/proxy connections. Transport-only; off by default.
    "YIAGENTS_HTTP_KEEPALIVE":          "http_keepalive",
    # P0: stream the graph + record per-analyst wall time. Observation only
    # (serial graph final state == invoke); off by default.
    "YIAGENTS_STREAM_TELEMETRY":        "stream_telemetry",
    # P0+: node-level wall-time + token telemetry (NodePerfTracker wraps every
    # graph node handler). Observation only; off by default = handlers pass
    # through unwrapped (byte-equivalent). See yiagents/graph/perf_telemetry.py.
    "YIAGENTS_NODE_PERF_TELEMETRY":     "node_perf_telemetry",
    # T1.3: per-call LLM retry count forwarded to the provider client via
    # _PASSTHROUGH_KWARGS (max_retries). Default 2 == langchain-openai's own
    # default, so UNSET behaviour is byte-equivalent; expose so flaky periods
    # can tune it (run_robust's per-ticker rerun is the outer safety net).
    "YIAGENTS_LLM_MAX_RETRIES":         "llm_max_retries",
    # T2: fan the 4 analysts out inside ONE wrapper node (each analyst runs in
    # its own sub-graph with its own state, so the shared `messages` /
    # clear_node coupling that assumes serial execution is structurally
    # avoided). OFF by default = today's serial analyst chain runs verbatim.
    # Flip on only after scripts/run_analyst_parallel_ab.py passes its gate.
    "YIAGENTS_ANALYST_PARALLEL":             "analyst_parallel",
    "YIAGENTS_ANALYST_PARALLEL_MAX_THREADS": "analyst_parallel_max_threads",
    # Binance IP-weight proactive backoff (perp data vendor). When on, the
    # vendor reads X-MBX-USED-WEIGHT-1M and backs off before the ceiling. Off
    # by default = the vendor neither reads the header nor sleeps (byte-equivalent).
    "YIAGENTS_BINANCE_PROACTIVE_BACKOFF":    "binance_proactive_backoff",
    "YIAGENTS_BINANCE_WEIGHT_THRESHOLD":     "binance_weight_threshold",
    # Binance SPOT host: when on, the spot vendor uses the key-free market-data
    # mirror data-api.binance.vision instead of api.binance.com. Off by default
    # (api.binance.com is the proven host through the SOCKS5 proxy); the mirror
    # is Binance's recommended read-only host and carries the same data.
    "YIAGENTS_BINANCE_SPOT_MIRROR":          "binance_spot_mirror",
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
    """Apply YIAGENTS_* env vars to the config dict in-place."""
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
    "results_dir": os.getenv("YIAGENTS_RESULTS_DIR", os.path.join(_YIAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("YIAGENTS_CACHE_DIR", os.path.join(_YIAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("YIAGENTS_MEMORY_LOG_PATH", os.path.join(_YIAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
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
    # On by default: the recommended production form. The risk manager overrides
    # position size / stop-loss / exposure deterministically (LLM keeps
    # direction; math owns size and risk). scripts/run_baseline.py sets this
    # explicitly per mode (baseline=False, full=True) so the A/B gate stays
    # clean regardless of this default; smoke inherits the default to exercise
    # the production path.
    "risk_enabled": True,
    "kelly_fraction": 0.25,            # quarter-Kelly by default
    "max_single_position": 0.20,       # one ticker <= 20% of equity
    "max_single_sector": 0.30,         # one sector <= 30% of equity
    "max_drawdown_hard_stop": 0.15,    # flatten + cool off beyond this drawdown
    "atr_stop_mult": 2.0,              # stop = last_close - mult*ATR (long)
    # Phase 2b: FinCoT de-persona structured prompts for analysts.
    "fin_cot_prompts": False,
    # Phase 4: global kill switch (env: YIAGENTS_KILL_SWITCH). Halt = no
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
    # nor sleeps, byte-equivalent to today. binance_weight_threshold is the
    # fapi USDT-M default (2400/min); override for a VIP-tier IP.
    "binance_proactive_backoff": False,
    "binance_weight_threshold": 2400,
    # Binance SPOT host mirror switch (env: YIAGENTS_BINANCE_SPOT_MIRROR). Off
    # by default = spot vendor hits api.binance.com (proven through the SOCKS5
    # proxy); on = key-free market-data mirror data-api.binance.vision. The
    # spot vendor reads this at call time, so it is byte-equivalent to today
    # when off (and spot is new code regardless, so no prior output to perturb).
    "binance_spot_mirror": False,
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
    # Data vendor configuration
    # Category-level configuration (default for all tools in category).
    # The configured value is the exact vendor chain — requests are NOT silently
    # routed to vendors you didn't choose. For ordered fallback, list several,
    # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
        "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
        "prediction_markets": "polymarket",  # Options: polymarket (keyless)
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
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
        ".SS":  "000001.SS",   # Shanghai (SSE Composite)
        ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
        "":     "SPY",         # default for US-listed tickers (no suffix)
    },
})
