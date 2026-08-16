<p align="center">
  <img src="assets/yiagents-logo.svg" alt="YiAgents" width="480">
</p>

<h1 align="center">YiAgents</h1>

<p align="center">
  A <b>research-oriented multi-agent LLM quantitative trading framework</b>
</p>

<p align="center">
  <a href="README.md">English</a> &nbsp;|&nbsp; <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/version-0.3.0-blue">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="status" src="https://img.shields.io/badge/status-Research%20Only-orange">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

> The design methodology draws on **99 papers** spanning LLM financial reasoning, multi-agent decision-making, quantitative risk control, backtest rigor, and adversarial security (see [Research foundation](#research-foundation), full list in [REFERENCES.md](REFERENCES.md)).
>
> Package name / import / CLI command are unified as `yiagents`; environment-variable prefix is `YIAGENTS_*`; user data lives under `~/.yiagents/`.

---

## What is this

YiAgents deploys a team of specialized **LLM agents** that mirror how a real trading firm reasons: fundamentals / sentiment / news / technical analysts produce views, bull & bear researchers debate in a structured format, a trader proposes, and a risk team together with a portfolio manager make the final call. On top of this, the framework layers a **deterministic quantitative risk overlay** (Kelly sizing / ATR stops / drawdown breaker / CVaR) and a **four-stage validation + backtest gate** pipeline for more rigorous analysis.

> ⚠️ **Analysis only.** The default configuration cannot place orders and cannot execute LLM-generated Python. Keep `YIAGENTS_ANALYSIS_ONLY=true`; reports and backtests are research evidence, not instructions for capital allocation. Nothing here constitutes financial, investment, or trading advice.

---

## What it can analyze

Give YiAgents a **ticker + date** and it analyzes from four angles, runs multiple debate and risk-deliberation rounds, and outputs a structured trading decision with a rating, position size, and stop-loss. The full journey — input to on-disk report — is detailed in [Analysis pipeline](#analysis-pipeline).

**Supported assets** (Yahoo Finance coverage via exchange-suffix tickers; company identity and the alpha benchmark are resolved per market automatically):

| Market | Example ticker |
| ------ | ------ |
| US equities | `AAPL`, `SPY` |
| A-shares | `600519.SS` (Kweichow Moutai), `000001.SZ` |
| Hong Kong / Tokyo / London | `0700.HK` / `7203.T` / `AZN.L` |
| India / Canada / Australia | `RELIANCE.NS` / `SHOP.TO` / `BHP.AX` |
| Crypto | `BTC-USD`, `ETH-USD` |

**Four analyst dimensions** (selectable):

| Analyst | Dimension | Data source |
| ------ | ------ | ------ |
| Market Analyst | Technical: picks up to 8 complementary indicators from a 28-name catalog (trend / momentum / volatility / volume) by market regime; five evidence tools always bound — weekly timeframe, support/resistance, volume features, candlestick patterns, benchmark relative strength | yfinance / Alpha Vantage |
| Sentiment Analyst | Social sentiment | Reddit, StockTwits (current-date analysis only) |
| News Analyst | Ticker news + macro/global news (Fed, geopolitics, central-bank policy …) | yfinance / Alpha Vantage News |
| Fundamentals Analyst | Financial fundamentals | yfinance / Alpha Vantage |

Macro data flows through FRED (Federal Reserve), event probabilities through Polymarket (prediction markets), and alternative data can be collected via browser. Sources without a trustworthy as-of parameter are omitted from historical runs rather than silently substituting today's data.

### Crypto analysis modes

Crypto has three opt-in modes. `crypto` (Yahoo spot) is the default; `crypto_spot` and `crypto_perp` are read-only, **unauthenticated, public-market** Binance tracks (Track A — analysis only, no order placement).

| Mode | `--asset-type` | Source | Tools bound to the analyst | Notes |
| ------ | ------ | ------ | ------ | ------ |
| Yahoo spot (default) | `crypto` | Yahoo Finance (`BTC-USD`) | indicators + verified_snapshot | Default crypto path; no funding / OI / leverage |
| Binance spot | `crypto_spot` | Binance spot (`api.binance.com`, optional mirror) | all runs: spot klines + spot indicators + dated indicators / snapshot / evidence tools; current date also: ticker24 / **spot_perp_basis** | Rolling 24h and current cross-venue basis are never injected into historical runs |
| Binance perpetual | `crypto_perp` | Binance USDT-M perp (`fapi.binance.com`) | all runs: klines + funding + indicators; current date also: OI / long-short / taker flow / basis / mark-price premium index | Analysis only. Backtests supported with full perp modeling (funding / costs / leverage / liquidation) |

In crypto modes the Fundamentals Analyst is dropped automatically (perpetual/spot pairs have no fundamentals). All Binance requests are hand-written `requests` (not the official SDK) over the existing SOCKS5 proxy, with per-product-line rate limiting and reactive 429/418 backstop.

### A-share native data

Opt-in native data for A-share tickers: set `YIAGENTS_A_SHARE_NATIVE=true` (default off; install the vendors with `pip install "yiagents[a-share]"`). The gate is flag **and** `is_a_stock(ticker)`, so US / crypto / HK runs stay byte-identical. When both hold, 12 PIT-correct read-only tools from Chinese vendors are appended to the analysts:

| Analyst | Appended tools (vendor) |
| ------ | ------ |
| Market | northbound flows · sector fund-flow · realtime quote · market breadth (AKShare) |
| Fundamentals | TTM PE/PB/dividend · forward-adjusted OHLCV · income / balance-sheet / cash-flow statements (BaoStock); money flow · dragon-tiger list (AKShare) |
| News | per-stock Chinese headlines, Eastmoney via AKShare |

A Tushare quality tier (`pip install "yiagents[a-share-tushare]"` + `TUSHARE_TOKEN`) is also available for fundamentals/news. Vendors are lazy-imported: without the extras, default-off runs are unaffected.

---

## Analysis pipeline

Every entry point — interactive `yiagents analyze`, `yiagents batch`, and the Web UI — funnels into the same `YiAgentsGraph.propagate()` run. One ticker + one date walks the full pipeline below, in order:

| # | Stage | Who | Produces |
| --- | --- | --- | --- |
| 1 | Input & PIT context | `propagate()` | State seeded: the analysis date pins the vendor-layer as-of clamp (no future data leaks in), instrument identity resolved deterministically, past memory-log lessons injected (opt-in, off by default) |
| 2 | Analyst team | Market / Sentiment / News / Fundamentals Analyst | Each analyst loops against its own tool set (`analyst ⇄ ToolNode`) until it emits its report; serial by default, collapsed into one parallel fan-out node behind `analyst_parallel`. Exception: the Sentiment Analyst pre-fetches Reddit / StockTwits / Yahoo headlines directly, no tool loop |
| 3 | Research debate | Bull ⇄ Bear Researchers | Multi-round structured debate, `max_debate_rounds` (default 2 → 4 speeches) |
| 4 | Research verdict | Research Manager (deep LLM) | Structured `ResearchPlan` |
| 5 | Trade proposal | Trader | `TraderProposal` — three-tier Buy / Hold / Sell |
| 6 | Risk debate | Aggressive · Conservative · Neutral | Three-way deliberation, `max_risk_discuss_rounds` (default 2 → 6 speeches) |
| 7 | Final call | Portfolio Manager (deep LLM) | `PortfolioDecision` — five-tier rating, price target, time horizon |
| 8 | Quant overlay | RiskManager (deterministic, no LLM) | Kelly ¼ sizing / ATR stop / drawdown breaker / CVaR recalibrate position, stop and exposure; appended to the decision under a marked section |
| 9 | Output & evidence | report writer | Per-stage markdown reports + consolidated report + machine-readable full-state log; rating parsed into a signal |

**What a run produces** (under `./reports/<TICKER>_<stamp>/` for CLI runs, `~/.yiagents/logs/reports/` programmatically):

```text
1_analysts/{market,sentiment,news,fundamentals}.md
2_research/{bull,bear,manager}.md
3_trading/trader.md
4_risk/{aggressive,conservative,neutral}.md
5_portfolio/decision.md      ← final decision, incl. the quant-overlay block
complete_report.md           ← consolidated; "⚠ DEGRADED RUN" banner on top if data quality tripped

~/.yiagents/logs/<TICKER>/YiAgentsStrategy_logs/full_states_log_<date>.json
   ← machine-readable full state (evidence, data_quality, per-node telemetry); Web history reads this file
```

> **Data quality ships with the report.** Data vendors record degradation sentinels (`no_data` / `optional_unavailable` / `stale_cache`) into a `data_quality` block on every run; a run that lost core data is flagged `DEGRADED` in both the report banner and the Web UI — it never silently pretends nothing happened. Exact prices and indicator values cited by the Market Analyst must come from the verified market-data snapshot, so every numeric claim has a checkable source.

Two optional loops close the cycle after a run: the causal memory log (`memory_enabled`, off by default — see [Persistence & recovery](#persistence--recovery)) resolves past same-ticker decisions against realized returns and injects lessons into the next run, and the offline self-improvement chain (IC pruning → config snapshot → validation gate — see [Backtest & validation gate](#backtest--validation-gate)).

---

## Architecture

```text
Data layer (yfinance / Alpha Vantage / FRED / Polymarket / Reddit / StockTwits / Binance)
        │
   ┌────▼───── Analyst Team (serial, or parallel behind a flag) ─────┐
   │  Market · Sentiment · News · Fundamentals                       │
   └──────────────────────────────────────────────────────────────────┘
        │
   ┌────▼───── Researcher Team ──────────────────────────────────────┐
   │  Bull ⇄ Bear multi-round structured debate (max_debate_rounds)   │
   └──────────────────────────────────────────────────────────────────┘
        │
   Research Manager ──► Trader (proposal)
        │
   ┌────▼───── Risk Management three-way debate ─────────────────────┐
   │  Aggressive · Neutral · Conservative                             │
   │  (max_risk_discuss_rounds / env YIAGENTS_MAX_RISK_ROUNDS)        │
   └──────────────────────────────────────────────────────────────────┘
        │
   Portfolio Manager (approve / reject)
        │
   Quantitative risk overlay (risk_enabled) → position / stop / exposure re-calibration
        │
   Optional causal memory loop (off by default; as-of filtered)
```

- **Ratings**: Research Manager / Portfolio Manager use a **five-tier** scale (Buy / Overweight / Hold / Underweight / Sell); the Trader uses three tiers (Buy / Hold / Sell).
- **Numeric safety**: exact prices / indicators come from a verified market-data snapshot; the LLM only picks direction, math owns sizing and stops.

---

## Installation

```bash
conda create -n yiagents python=3.12
conda activate yiagents
pip install -e .

# Optional extras
pip install -e ".[web]"        # FastAPI + Uvicorn for the local Web UI
pip install -e ".[bedrock]"    # Amazon Bedrock provider (AWS SigV4)
pip install -e ".[a-share]"    # AKShare + BaoStock for A-share native data
pip install -e ".[dev]"        # ruff / pytest
```

For a fully reproducible environment (pinned transitive deps), install from
the lock instead:

```bash
pip install -r requirements.lock   # core + dev, pinned (regenerate: uv pip compile)
```

Docker:

```bash
cp .env.example .env        # fill in API keys
docker compose run --rm yiagents
```

> `PySocks` is a load-bearing dependency: yfinance and the Binance vendor reach the network through a SOCKS5 proxy, and PySocks is what lets `requests` resolve hosts through it instead of hanging on DNS.

---

## Configuration

YiAgents supports multiple LLM providers — set one in `.env` (**the scripts in this repo default to DeepSeek**):

```bash
DEEPSEEK_API_KEY=...            # DeepSeek (script default)
OPENAI_API_KEY=...              # OpenAI (GPT)
ANTHROPIC_API_KEY=...           # Anthropic (Claude)
GOOGLE_API_KEY=...              # Google (Gemini)
XAI_API_KEY=...                 # xAI (Grok)
DASHSCOPE_API_KEY=...           # Qwen (international)
DASHSCOPE_CN_API_KEY=...        # Qwen (China)
ZHIPU_API_KEY=...               # GLM (Z.AI international)
ZHIPU_CN_API_KEY=...            # GLM (BigModel China)
MINIMAX_API_KEY=...             # MiniMax (global)
ALPHA_VANTAGE_API_KEY=...       # Alpha Vantage
FRED_API_KEY=...                # Federal Reserve macro data
```

**Recommended DeepSeek split** — heavy deliberation on the deep channel, light multi-turn on the quick channel (~8–10 min wall-clock per ticker):

```bash
YIAGENTS_LLM_PROVIDER=deepseek
YIAGENTS_DEEP_THINK_LLM=deepseek-v4-pro     # Research Manager / Portfolio Manager
YIAGENTS_QUICK_THINK_LLM=deepseek-v4-flash  # 4 analysts / debates / Trader / reflection
YIAGENTS_OUTPUT_LANGUAGE=English             # analyst report & final-decision language
```

**Local proxy (important):** if you reach Yahoo / Binance / the LLM through a SOCKS5 proxy, install PySocks (`pip install "requests[socks]"`) and set `HTTPS_PROXY`/`HTTP_PROXY`. `preflight` auto-detects the proxy port and the PySocks dependency.

> Config keys and their type-checked env mapping live in [yiagents/default_config.py](yiagents/default_config.py) (`_ENV_OVERRIDES`). A misspelled boolean or non-numeric int raises at startup instead of silently falling back.

---

## CLI usage

After install you get the `yiagents` command; you can also run `python -m yiagents.cli.main` from source. The CLI lives inside the unique `yiagents` namespace so an unrelated top-level `cli` package cannot shadow it.

### Single analysis: `yiagents analyze`

```bash
yiagents analyze
```

Interactively pick ticker, analysis date, output language, analysts, research depth, and LLM provider/model. Results stream as they are computed; at the end a full five-section report is produced and you are asked whether to save it.

```bash
yiagents analyze --checkpoint          # checkpoint this run (resume after a crash)
yiagents analyze --clear-checkpoints   # clear all checkpoints before running
```

### Batch (concurrent): `yiagents batch`

Analyze multiple tickers of the **same asset class** concurrently. Each ticker runs the exact same full pipeline as a single-ticker run, sharing one API key.

```bash
# Multiple US equities, same date
yiagents batch -t AAPL -t NVDA -t MSFT -d 2026-06-30

# Multiple A-shares
yiagents batch -t 600519.SS -t 000858.SZ -t 601318.SS -d 2026-06-30

# A batch of crypto (Yahoo spot by default)
yiagents batch -t BTC-USD -t ETH-USD -t SOL-USD -d 2026-06-30 -w 3

# Binance perpetual analysis (Track A, read-only public market data)
yiagents batch -t BTCUSDT -t ETHUSDT -d 2026-06-30 --asset-type crypto_perp

# Binance spot analysis (with cross-venue spot-perp basis)
yiagents batch -t BTCUSDT -t ETHUSDT -d 2026-06-30 --asset-type crypto_spot
```

| Option | Description |
| ------ | ------ |
| `-t / --ticker` | Ticker; repeat `-t` for several. **One batch = one asset class** (all stocks or all crypto) |
| `-d / --date` | Analysis date `YYYY-MM-DD` |
| `--asset-type` | `stock` / `crypto` / `crypto_spot` / `crypto_perp` / `auto` (`auto` infers from the first ticker; mixed classes error out) |
| `-w / --workers` | Concurrency pool size K. **Default is strictly serial (K=1)**; the pool only runs when `YIAGENTS_BATCH_CONCURRENCY=true` is set or an explicit `-w K>1` is passed (`YIAGENTS_BATCH_WORKERS` then sizes the pool). `-w 1` is explicit serial |

**Concurrency is safe**: each worker thread owns its own graph instance (no races); the memory log and OHLCV cache are serialized with filelock; one failed ticker does not abort the batch; and each ticker's analysis is **byte-equivalent** to running it serially — the concurrency layer sits above `propagate()` and never touches any agent's input, depth, or reasoning parameters. See [yiagents/batch/runner.py](yiagents/batch/runner.py).

### Config validation & audit: `yiagents config-check` / `yiagents snapshot`

```bash
yiagents config-check     # validates the selected provider's key, reports optional
                          # data-source keys, and checks indicator_battery names
yiagents snapshot record -r "Pruned low-IC indicators" [-e evidence]
                          # append-only config snapshot (timestamp / reason /
                          # evidence / git commit); never edits the live config
yiagents snapshot diff    # diff the active config against the last snapshot
yiagents snapshot list    # list recorded snapshots
```

---

## Web UI

A small FastAPI app + vanilla HTML/JS front end ([web/](web/)) lets you browse past analyses and submit new ones from a browser — **no agent / graph / dataflow code is touched**. Dark "quant terminal" styling, with 🌐 Chinese/English i18n that switches both the UI chrome and the report language.

```bash
pip install -e ".[web]"          # fastapi + uvicorn (one-time)
python web/app.py                # serves http://127.0.0.1:8000
```

> Must be launched from the **project root** (the dir containing `.env`, `yiagents/`, `scripts/`): `yiagents/__init__.py` loads `.env` via `load_dotenv(usecwd=True)`, so starting elsewhere leaves the DeepSeek key and the SOCKS5 proxy unset, and the spawned `run_robust` subprocess would inherit that broken env.

- **Browse**: ticker grid → per-ticker dates → full report view (rating badge, quantitative risk-overlay card, 5 collapsible sections, optional node-perf bar chart), plus a rating-comparison view (`#/compare`) across tickers and dates.
- **Submit**: a form spawns `scripts/run_robust.py` (the same watchdog-backed path the CLI uses); the UI polls every 4 s and links the finished report. One analysis at a time (409 while one is running).
- **API**: `GET /api/tickers`, `GET /api/tickers/{t}/runs[/{date}]`, `GET /api/compare`, `POST /api/analyze`, `GET /api/tasks/{id}`, `GET /api/health`.

See [web/README.md](web/README.md) for the architecture and the full endpoint reference.

---

## Python API

```python
from yiagents.graph.trading_graph import YiAgentsGraph
from yiagents.default_config import DEFAULT_CONFIG

# DEFAULT_CONFIG already applies YIAGENTS_* env overrides
ta = YiAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

Tune the quantitative risk overlay:

```python
config = DEFAULT_CONFIG.copy()
config["risk_enabled"] = True          # on by default (production form)
config["kelly_fraction"] = 0.25        # quarter Kelly
config["max_single_position"] = 0.20   # one ticker ≤ 20% of equity
config["max_single_sector"] = 0.30     # one sector ≤ 30%
config["max_drawdown_hard_stop"] = 0.15# drawdown breaker
config["atr_stop_mult"] = 2.0          # stop = last close − 2×ATR (long)

ta = YiAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
```

`ta.save_reports(final_state, ticker)` writes the same report tree as the CLI for headless / API use. All config keys are documented in [yiagents/default_config.py](yiagents/default_config.py).

---

## Quantitative risk overlay

The LLM picks direction; math owns sizing and risk ([yiagents/risk/](yiagents/risk/)):

| Mechanism | File | Effect |
| ------ | ------ | ------ |
| Kelly sizing | [kelly.py](yiagents/risk/kelly.py) | Optimal size from win-rate / payoff, fractional |
| ATR stop | [atr_stop.py](yiagents/risk/atr_stop.py) | stop = close − N×ATR (long) |
| Drawdown breaker | [breaker.py](yiagents/risk/breaker.py) | Flatten + cool off beyond max drawdown |
| CVaR | [cvar.py](yiagents/risk/cvar.py) | Conditional value-at-risk, tail constraint |
| Bus | [manager.py](yiagents/risk/manager.py) | Combines the above; enforces per-ticker / sector / exposure caps |

`risk_enabled` defaults to **True**: the risk manager deterministically rewrites the analytical position size / stop / exposure while the LLM keeps direction. `scripts/run_baseline.py` sets it explicitly per mode (`--baseline` = off, to build the Phase-0 baseline; `--full` = on, for the paired A/B).

---

## Backtest & validation gate

The **four-stage validation** script [scripts/run_baseline.py](scripts/run_baseline.py) — run it in order, from zero-cost self-check up to the full A/B, so you don't burn LLM quota upfront:

```bash
# Stage -1 | preflight: zero LLM cost — checks deps (incl. PySocks) / env / proxy port / yfinance pull / DeepSeek ping
python scripts/run_baseline.py --preflight --ticker AAPL

# Stage  0 | smoke: 1 ticker × 1 date — confirms LLM / network / keys all work (cheapest full pipeline)
python scripts/run_baseline.py --smoke --ticker AAPL --date 2026-03-15

# Stage  1 | baseline: current system (LLM decision + simple sizing) — baseline report + dashboard
python scripts/run_baseline.py --baseline --tickers AAPL NVDA

# Stage  2 | full A/B: baseline vs Phase-1 risk overlay + gate PASS/FAIL verdict
python scripts/run_baseline.py --full --tickers AAPL NVDA --runs 2
```

Common options: `--tickers` (A-share `600519.SS`) / `--asset-type` (`stock` / `crypto` / `crypto_perp`) / `--start --end` / `--step` (rebalance interval, default 10) / `--rebalance` (number of rebalances, default 6) / `--holding-days` / `--cost-bps` (one-side cost, default 5bp) / `--runs` (LLM non-determinism — run each ticker several times for a distribution) / `--workers` (cross-ticker concurrency) / `--out` (default `./backtest_output`). Perp-only: `--leverage` (default 1.0; >1 enables isolated-margin liquidation modeling) / `--allow-short` / `--slippage-bps` (default 0) / `--bnb-discount`.

> Each `propagate()` = one full LLM graph (4 analysts + debates + trader + risk debate + PM). Cost scales linearly with `tickers × dates × runs`. **Get preflight green first, then smoke, then scale up.**

Stage 2 (`--full`) runs the baseline-vs-risk-overlay A/B and independently judges a **validation gate** per ticker ([yiagents/backtest/validation_gate.py](yiagents/backtest/validation_gate.py)):

- **Deflated Sharpe Ratio (DSR)** — multiple-testing correction across samples, penalizes overfit ([metrics.py](yiagents/backtest/metrics.py))
- Beats buy-and-hold? PASS / FAIL verdict, improvement suggestions
- Baseline and improved legs reuse the same cached LLM decision tape; every ticker/run gets a fresh stateful risk manager
- Reports, dashboards, and gate verdicts land in `--out` (default `backtest_output/`)

Daily signals execute on the **next available bar**, never on the completed bar the model just read. `Hold` creates no rebalance or fee, and win rate is computed from actual position P&L. Crypto runs annualize on 365 trading days, stocks on 252.

`crypto_perp` backtests (`--asset-type crypto_perp`) ship with full perp accuracy modeling:

- **Marking**: positions mark to the perp's own Binance candles, not Yahoo spot.
- **Funding drag**: charged per funding interval; **fail-closed** — a coverage gap in the funding series aborts the run instead of silently assuming zero.
- **Costs**: taker fee + `--slippage-bps` + optional BNB discount, with `stepSize` / `minNotional` fill quantization from live exchange filters.
- **Leverage / liquidation**: `--leverage > 1` enables isolated-margin modeling — bar-extreme (high/low) liquidation triggers, MMR ladder from `leverageBracket`, liquidation events recorded in the run summary. Default is 1× long-only.
- **Shorts**: `--allow-short` opts in (Sell → −1×, receiving funding); perp-only.

**Indicator self-improvement loop (offline, no LLM):** the market analyst's 28-name catalog can be pruned by evidence — `scripts/export_ic_dataset.py` derives a `date, forward_return, <indicator>…` table straight from the PIT-filtered OHLCV cache, `scripts/prune_indicators_cli.py` ranks rolling IC and prints a keep/prune report (never auto-applied), and the reviewed list lands in the `indicator_battery` config key, validated by `yiagents config-check`. `yiagents snapshot record` keeps the change on an append-only audit trail.

```text
[AAPL] gate verdict: ✅ PASS | DSR 1.42 | beats B&H True
[MSFT] gate verdict: ❌ FAIL | DSR -0.31 | beats B&H False
```

---

## Performance & observability

The concurrency / transport / observation layers never touch any agent's input. These switches are all **off by default (or at an equivalent value) = byte-equivalent to the serial baseline**; turn on as needed:

| Switch (env) | Default | Effect |
| ------ | ------ | ------ |
| `YIAGENTS_LLM_TIMEOUT_S` | off (unset) | Per-call read timeout (seconds); half-open connections raise `APITimeoutError` and recover via the SDK's built-in retry. **Off by default** — set to 120 in production to prevent indefinite hangs on half-open sockets (omit for slow local models like Ollama) |
| `YIAGENTS_HTTP_KEEPALIVE` | false | Process-wide shared `httpx.Client`; reuses TLS / SOCKS5 connections |
| `YIAGENTS_LLM_MAX_RETRIES` | 2 | Per-call retry count (= langchain default, equivalent) |
| `YIAGENTS_NODE_PERF_TELEMETRY` | false | Per-node wall-time + token telemetry; `--profile` turns it on, writes `node_perf_<date>.json` |
| `YIAGENTS_STREAM_TELEMETRY` | false | Stream the graph + record per-analyst wall time (final state identical to invoke) |
| `YIAGENTS_ANALYST_PARALLEL` | false | Run the 4 analysts in parallel inside one wrapper node (each in its own sub-graph). **Flip on only after `run_analyst_parallel_ab.py` passes its gate** |
| `YIAGENTS_LLM_RATE_LIMITER` | false | Optional shared RPM rate limiter |
| `YIAGENTS_BINANCE_PROACTIVE_BACKOFF` | false | Binance perp vendor reads `X-MBX-USED-WEIGHT-1M` and backs off before the ceiling |
| `YIAGENTS_BINANCE_SPOT_MIRROR` | false | Crypto-spot quotes via the key-free mirror `data-api.binance.vision` |

One-line telemetry: `python scripts/run_baseline.py --smoke --profile --ticker <T> --date <D>` prints a node → wall-clock-share + token table that pinpoints the real bottleneck.

---

## Persistence & recovery

**Decision log (opt-in):** persistence is off by default (`YIAGENTS_MEMORY_ENABLED=false`). When explicitly enabled, completed decisions are written to `~/.yiagents/memory/trading_memory.md`; historical runs only see decisions and realized outcomes known by their as-of date. Legacy reflections without an outcome-availability date are excluded from historical prompts. Override the path with `YIAGENTS_MEMORY_LOG_PATH`.

**Checkpoint resume (opt-in):** turn on with `--checkpoint`. LangGraph archives state after each node, so a crashed / interrupted run resumes from the last successful step; checkpoints are cleaned up on successful completion. Per-ticker SQLite databases live in `~/.yiagents/cache/checkpoints/<TICKER>.db` (override with `YIAGENTS_CACHE_DIR`).

**Analysis boundary:** `YIAGENTS_ANALYSIS_ONLY=true` is the default and blocks every live execution path. The legacy and new execution switches must both be explicitly armed after analysis-only mode is disabled; gateways re-check them and `YIAGENTS_KILL_SWITCH` on every submission. PoT host execution is also off, and is always refused on Windows until a real process-isolated sandbox exists.

---

## Scripts

| Script | Purpose |
| ------ | ------ |
| [scripts/run_baseline.py](scripts/run_baseline.py) | Four stages: preflight / smoke / baseline / full |
| [scripts/run_robust.py](scripts/run_robust.py) | Per-ticker subprocess orchestrator + watchdog + OS-level force-kill and rerun (recommended for ≥1 ticker) |
| [scripts/run_batch.py](scripts/run_batch.py) | Batch concurrent analysis (equivalent to `yiagents batch`) |
| [scripts/run_analyst_parallel_ab.py](scripts/run_analyst_parallel_ab.py) | A/B gate verifying analyst-parallel is distribution-equivalent to serial |
| [scripts/smoke_structured_output.py](scripts/smoke_structured_output.py) | Verify the three structured-output agents against any provider |
| [scripts/analyze_window.py](scripts/analyze_window.py) | Multi-day decision window over one ticker with the quantitative risk overlay (Kelly / ATR / CVaR) |
| [scripts/export_ic_dataset.py](scripts/export_ic_dataset.py) | Build the IC dataset (from the OHLCV cache) that feeds the pruning CLI |
| [scripts/prune_indicators_cli.py](scripts/prune_indicators_cli.py) | Offline keep/prune report for low-IC indicators (evidence for `indicator_battery`) |
| [scripts/rank_signals.py](scripts/rank_signals.py) | Rank a batch of tickers by conviction from existing reports (read-only) |
| [scripts/trade_ticket.py](scripts/trade_ticket.py) | Post-analysis execution-ticket generator (read-only enhancement layer) |

---

## Project structure

```text
.
├── yiagents/            # core package (internal package name; import name)
│   ├── agents/               # analysts / researchers / managers / trader / risk_mgmt
│   ├── dataflows/            # yfinance / Alpha Vantage / FRED / Polymarket / Reddit / StockTwits / Binance / browser
│   ├── graph/                # LangGraph orchestration: trading_graph / propagation / reflection / signal_processing / perf_telemetry
│   ├── risk/                 # kelly / atr_stop / breaker / cvar / manager (quant risk overlay)
│   ├── backtest/             # engine / metrics / validation_gate / ic / report / cache
│   ├── batch/                # multi-ticker concurrent runner + filelock
│   ├── execution/            # browser_broker (browser broker + kill switch)
│   ├── monitoring/           # dashboard (HTML dashboard)
│   ├── llm_clients/          # multi-provider adapters + rate limiter + shared httpx
│   ├── cli/                  # interactive CLI (analyze / batch / config-check / snapshot)
│   ├── default_config.py     # config + env mapping
│   └── reporting.py
├── cli/                      # source compatibility shims; packaged CLI is yiagents.cli
├── web/                      # FastAPI Web UI (run in place, not packaged)
├── scripts/                  # run_baseline / run_robust / run_batch / export_ic_dataset / prune_indicators_cli / rank_signals / …
├── tests/                    # test suite — data / risk / backtest / gate / multi-provider / i18n / concurrency
└── pyproject.toml
```

---

## Reproducibility

YiAgents is LLM-driven: **two runs of the same ticker + date may differ** — an inherent property of language-model research, not a bug. Sources:

- **Model sampling non-determinism**: even at a fixed temperature, providers do not guarantee byte-identical output; reasoning models sample over their internal reasoning and vary more.
- **Current-date data drifts**: news / StockTwits / Reddit return different content over time. Historical runs omit current-only social, prediction-market, rolling ticker, and positioning feeds; dated sources are still subject to vendor corrections and coverage changes.

Mitigations: lower `temperature` (`YIAGENTS_TEMPERATURE`), or pick a non-reasoning model explicitly. Already deterministic: company identity is resolved from the ticker and locked before any agent runs; the market analyst's exact prices / indicators come from a verified data snapshot.

Backtest results are not guaranteed to match any published number — treat this as **scaffolding for researching multi-agent analysis**, not a strategy with a fixed, reproducible return.

---

## Research foundation

Every key mechanism in YiAgents maps to published research, not invention. The table below maps 14 research directions to where they land in the framework (representative papers only; the full 99 are in [REFERENCES.md](REFERENCES.md)):

| Research pillar | Representative work | Where it lands in YiAgents |
| ------ | ------ | ------ |
| Benchmarking & "alpha illusion" | FINSABER (Li 2025) · The Alpha Illusion (Jang 2026) · AlphaQuanter (2025) | [validation_gate.py](yiagents/backtest/validation_gate.py): DSR + beats-buy-and-hold |
| Multi-agent decision-making | Debate or Vote (NeurIPS 2025) · MA-PoP (2026) · S2-MAD (2025) | Bull/Bear & risk multi-round debates (`max_debate_rounds` / `max_risk_discuss_rounds`) |
| Reasoning optimization | FinCoT (2025) · Program-of-Thoughts · Overthinking early-exit (2025) | [fin_cot_prompts](yiagents/default_config.py): de-persona structured prompts |
| Memory & anti-forgetting | FinMem (TBDATA 2025) · Reflexion (NeurIPS 2023) · AlphaAgent (2025) | [memory loop](yiagents/graph/): decision log + reflection + cross-ticker lessons |
| Hallucination & numeric verification | Chain-of-Verification (ICML 2024) · DeBERTa-NLI · HHEM | Numbers go through an interpreter / verification path; the LLM only picks direction |
| LLM + traditional quant | LLM-MAS-DRL (2024) · AlphaCrafter (2024) · FinCon (2024) | Hybrid architecture: LLM produces views, the quant layer owns size / stops |
| Market-state detection | HMM Regime · Cascaded controller (2024) | Trend / range / high-vol / crisis adaptation (enhancement module, roadmap) |
| Risk & position sizing | HRP (Lopez de Prado) · Sentinel/ATR · CVaR two-layer (FinCon) | [risk/](yiagents/risk/): Kelly + ATR stop + breaker + CVaR |
| Backtest rigor | FinCAD (2025) · CPCV · Deflated Sharpe (Lopez de Prado) | [backtest/](yiagents/backtest/): parameterized look-ahead-bias correction + DSR |
| Adversarial robustness | MemMorph (2025) · SMSR (2025) · Spotlighting (2025) | Tool-call / memory-poisoning / prompt-injection defenses (roadmap) |
| Cost engineering | GPTCache · model cascading · DAG orchestration (2025) | Multi-provider routing + checkpoint resume + four-stage cost-ascending validation |
| Sentiment & alternative data | FinAgent (KDD 2024) · Few-shot stock prediction (Deng 2024) | [dataflows/](yiagents/dataflows/): Reddit / StockTwits / Polymarket / browser |
| Explainability | CFA XAI report (2025) · CoT visualization | Structured reports + [dashboard](yiagents/monitoring/dashboard.py) + decision log |
| Compliance & security | EU AI Act · AIBOM (2025) · Zero-trust architecture | `YIAGENTS_KILL_SWITCH` + research-only disclaimer |

> Items marked "roadmap" are designed but not all landed yet.

---

## Citation

If this framework helps your work, please cite:

```bibtex
@misc{yiagents2026,
      title={YiAgents: A Research-Oriented Multi-Agent LLM Quantitative Trading Framework},
      author={Mark},
      year={2026},
      url={https://github.com/zhang12120113-creator/Yiagents},
}
```

Full references (99 papers, grouped into 14 research directions) are in [REFERENCES.md](REFERENCES.md).

## Disclaimer

This project is for academic and engineering research only and **does not constitute any financial, investment, or trading advice**. Trading in financial markets carries significant risk; past performance does not guarantee future returns. The author bears no responsibility for any decisions made or losses incurred based on this project.
