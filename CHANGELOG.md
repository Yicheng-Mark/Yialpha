# Changelog

> **YiAlpha**（弈·Alpha）是基于研究文献（见 [REFERENCES.md](REFERENCES.md)，收录 99 篇相关研究）独立设计的多智能体 LLM 金融交易框架。包名 / import / CLI / env 前缀 `YIALPHA_*` / 数据目录 `~/.yialpha/` 全程统一。

All notable changes to YiAlpha are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Breaking changes within the 0.x line are called out explicitly.

## [Unreleased]

### Fixed

- **Perp bug-fix sweep (2026-09-19, 14 fixes; crypto_perp focus).** Four
  parallel code reviews + runtime smokes across the perp data / risk /
  accuracy / CLI layers; every fix carries a regression pin in
  `tests/test_perp_bugfix_regressions.py`.
  - **P1 — interactive perp runs ran the risk overlay as `asset_type="stock"`**
    (`cli/main.py` never forwarded `selections["asset_type"]`): the entire
    perp machinery (ticket / leverage / liquidation / funding gate /
    fair-value bridge / forming-candle note) was silently void on the only
    entrance that can reach a perp run interactively, and
    `price_at_decision` anchored on the Yahoo spot series.
  - **P1 — one recoverable klines failure permanently vetoed the ticket
    NO_TRADE.** A model typo (`interval="4H"` → instructive sentinel →
    retry `"4h"` → data) left an append-only core sentinel that
    `classify_quality` read as DEGRADED_CRITICAL regardless of the later
    success (the router never recorded successes for the optional Binance
    categories). The router now records perp price-book engine successes
    (`quality.is_perp_core_method`), and a sentinel whose own method later
    served data classifies as auxiliary "(recovered)" — disclosed, never
    vetoing a run whose price book exists.
  - **Funding `sum_7d` spanned 8 days** (`_window(7)` = D−7..D inclusive →
    24 settlements at 8h, not 21) — `sum_7d` and the annualized carry were
    overstated ~14%; the perp bundle now requests exactly 7 days
    end-inclusive.
  - **Memory resolution priced perp outcomes on the Yahoo spot venue**
    (`fetch_returns_yf` had no asset routing: BTCUSDT → BTC-USD spot, and
    tokenized-stock perps like MUUSDT never resolved at all). Routing now
    keys on the entry's own `asset=` tag (venue identity), falling back to
    the run's asset type; the crypto leg fetches the venue it traded with
    the closed-bars seam.
  - **Boundary-day scoring on the still-forming daily bar**: accuracy's
    `_dated_close` / `forward_outcome` and the backtest engine's perp
    price/extremes providers now pass `closed_as_of` (same seam
    `outcome_compute._perp_close_series` already used), so an intraday
    verify-history keeps the horizon `pending` and a backtest window
    touching today never marks equity on a partial candle.
  - **`take_profits` rounded to 6 decimals**, destroying the R-multiple
    grid on micro-price USDT-M contracts (PEPEUSDT-class ~1e-5, where the
    grid is ~9% of price) — significant-digit rounding (`_sig_digits`)
    keeps six digits at every magnitude.
  - **Interactive streamed path never bound the ledger run context**:
    prediction capture no-oped ("no active capture" noise into the tool
    loop), no run row / evidence / ticket mirror / regime id. The
    V2.1/V2.2 record stage is extracted as
    `YiAlphaGraph._bind_run_record_stage` and called by both `_run_graph`
    and the CLI streamed path.
  - **Interactive decisions stored without the venue tag** — perp/spot
    lesson separation on the same ticker was silently void for interactive
    runs; `_store_cli_decision` forwards `asset_type` now.
  - **`reaches_now` used a single UTC anchor** (missed by the dual-anchor
    fix db79063): on hosts behind UTC the live OI snapshot row was dropped
    from runs every other gate classified live. Now keyed on the earlier
    of `live_anchor_dates()`.
  - **Perp bundle core-sentinel detail reported failed price legs as
    "ok"** (key-presence instead of status check) — a self-contradictory
    evidence string in `full_states_log`.
  - **Perp bundle's ThreadPoolExecutor dropped the parent contextvars**
    (bare `pool.submit`): workers re-initialized config/analysis-date from
    defaults — latent divergence; now `submit_with_context` like every
    sibling fan-out.
  - **Overlay price/mark failures recorded `optional_unavailable`** while
    their own docstrings (and run_robust's DEGRADED counter, which reads
    `core_sentinel_count`) claim core — now `KIND_CORE_ERROR`.
  - **Vision summary headers instructed `summary=false`** — a parameter
    the bound tools do not expose (unexecutable instruction to the model);
    reworded to name the config toggle.
  - Two pre-existing test stubs widened (`fake_frame` accepts
    `closed_as_of`), `test_memory_log` / `test_cli_quality_chain` updated
    to pin the new call contracts.

- **Round-2 self-review of the same diff (adversarial, 10 finder angles).**
  The sweep's own fixes carried defects the green suite could not see;
  all verified against the working tree and fixed with pins:
  - **The bare-method "(recovered)" rule collided with the overlay's core
    sentinels** — perp runs ALWAYS carry qualified router klines successes
    by decision time, so the overlay's own (never-retried, unqualified)
    price/mark failures were being downgraded to auxiliary, voiding the
    exact NO_TRADE veto the kind upgrade claimed to preserve; an
    index-klines success could likewise forgive a last/mark outage.
    Recovery now keys on (method, qualifier): successes record the router's
    `_qualifier`, an event recovers only on an exact basis match, and
    unqualified decision-time sentinels are never matched by qualified
    successes. `summarize_quality.core_sentinel_count` is recovery-aware
    too, so the ticket gate and run_robust's DEGRADED verdict can no longer
    disagree about the same event.
  - **`_trailing_funding_total` (the risk gate's annualized-funding input)
    had the same 8-day window** as the bundle's `_fetch_funding` — fixed to
    7 days end-inclusive; the funding-note bullet and ticket cost model
    read a true 7-day sum now.
  - **The crypto cutoff clamp in `fetch_returns_yf` computed epoch-ms from
    a naive datetime** (host-local midnight vs UTC bar closes) — extracted
    into `_crypto_daily_frame` (UTC-everywhere, `binance._now_ms` seam)
    shared by `_dated_close` and `fetch_returns_yf`.
  - **The CLI streamed path's hand-mirror had already drifted within the
    same diff**: `_resolve_pending_entries` without the new `asset_type`
    fallback (legacy untagged perp entries priced on Yahoo), and the
    run_id/regime_id stamping copy-pasted — the stamping half is now the
    shared `YiAlphaGraph._stamp_run_record_ids` seam.
  - **Memory entries' `asset=` tag forwarded raw** — legacy `crypto`
    umbrella / hand-edited variants silently fell to the Yahoo leg; the tag
    now normalizes through the same `_resolve_asset_type` mapping
    verify-history uses (incl. USDT-suffix ticker inference for untagged
    entries).
  - **`reaches_now` re-implemented the dual-anchor predicate inline** and
    disagreed with `is_historical_date` on future labels (live snapshot row
    appended to a future-dated window) — now delegates to the one shared
    predicate.
  - **Five more live-run consumers baked the forming bar into frozen
    state**: `regime/compute.py` (trend/realized-vol/overnight-gap +
    spot/index basis legs of the persisted RegimeState), `market_regime.py`
    (turbulence line), `backtest/ic_dataset.py` (exported forward returns)
    — all on the `closed_as_of` closed-bars seam now.
  - **Bundle core-price sentinel upgraded to `KIND_CORE_ERROR`** (same
    outage, same kind as the overlay's — run_robust's counter no longer
    depends on which call site caught the failure).
  - `take_profits` significant digits widened 6 → 8 (BTC-scale TPs keep
    sub-dollar precision instead of quantizing to whole dollars); test
    stubs widened; the PR3 veto pin restored to a strong assertion with a
    true comment.

- **Round 3: the last deferred data-layer findings closed with live
  evidence.**
  - **`/futures/data` retention clamp** — live probe confirmed the server
    REJECTS a `startTime` older than the 30-day horizon (HTTP 400 -1130),
    no silent truncation: a wide window (an LLM echoing a 90-day start, or
    the >500-row end-anchor putting the start ~500 days back) degraded the
    whole OI/LSR/taker/basis call to a sentinel. `_futures_data_window` now
    clamps the head into the retained tail with a disclosure note (the
    same clamp its `derivatives_stress_series` sibling already carried).
  - **Vision edge-day 404 undisclosed** — in the ~06:30 UTC
    pre-publication window (end == yesterday, file not landed),
    `_resolve_window`'s "exceeds available archives" note does not fire and
    the 404ing edge day reached none of `_coverage_notes`' disclosed
    categories: the newest, decision-adjacent day was silently absent while
    the header named the full window. Missing START/END edge days are
    disclosed now.
  - **Funding cadence header inverted** — `/fapi/v1/fundingInfo` states the
    contract's interval AS OF NOW and used to override the (ground-truth)
    in-window settlement spacing: on a cadence-changed contract the
    annualisation hint stamped a 2x-wrong multiplier on rows settled at the
    old cadence. The window's own modal spacing wins whenever measurable,
    fundingInfo remains the fallback for tiny windows, and a disagreement
    is disclosed.
  - **Web/overlay micro-price parsing** — the overlay bullet regexes'
    numeric class stopped at the exponent sign, so a PEPEUSDT-class
    liquidation price rendered as `1.07e-05` parsed as `1.07` on the web
    KPI tile (five orders of magnitude off); the overlay's own Stop Loss /
    Entry Reference rendered `:.2f` → "0.00" on ~1e-5 contracts. Regexes
    accept exponent notation now and the renderer uses `%.6g`.
  - **PM optional-qualifier validators coerced, not strict** — a junk
    `price_target_currency` ("EUR", "USDT (Binance)") used to raise, discard
    the WHOLE structured decision (rating → regex fallback) and silently
    drop the fair-value bridge's target; out-of-vocab qualifiers now coerce
    to None (#1058 semantics).
  - **Umbrella `asset_type="crypto"` aligned with routing** — accuracy
    scored those decisions on Binance PERP klines while the live pipeline
    prices the umbrella on the spot series; the venue mapping now resolves
    it as crypto_spot.
  - **onchain_flows**: the "30d mean" label now states the span actually
    covered (the vendor window anchors at fetch-time NOW, so replays see
    fewer days); a replay older than the trailing window records a ledger
    sentinel naming the STRUCTURAL gap instead of rendering "network/parse
    failure"; per-chart failures record optional-unavailable sentinels (the
    fifth direct-connect vendor the T0 ledger sweep missed);
    `fetched_at` travels with the cache bytes (oldest ingredient wins)
    instead of regenerating NOW at assembly time.
  - Web "new analysis" form defaults/max to the HOST-LOCAL date (UTC slice
    made local-today un-submittable ahead of UTC); zh accuracy label says
    "天" not "交易日" (crypto horizons are calendar days); the bear
    researcher's prompt no longer says "stock" on crypto runs.

- **Known issues left open by design (round-3 review, need a layout or
  product decision before changing)**: (1) `full_states_log_<date>.json`
  is keyed only by ticker+date, so same-date perp AND spot runs on one
  ticker silently overwrite each other (memory log got the `asset=` tag
  for exactly this; the states log would need a venue-suffixed layout with
  reader updates across web history + accuracy scanning); (2) a persisted
  instrument-registry row can classify a run `stock_perp` while the
  in-process `stock_perp_underlying()` resolves None (registry says
  stock-perp, every remap/context says crypto-perp — fail-soft DEGRADED,
  reachable when a fresh subprocess's warm fetch fails and the base is not
  in the static seed); (3) legacy logs without `asset_type` render as
  stock in the web UI while accuracy infers perp pricing (display-only;
  the inference caveat never reaches the UI).

- **Round 4: execution-advice path + memory-log venue invariants (final
  review pair; the perp coverage map is closed).**
  - **P1 — `trade_ticket._fnum` truncated exponent notation** (same drift
    class the overlay regexes carried): the overlay now renders stop/entry
    with `%.6g`, so a PEPEUSDT-class stop of `1.14e-05` parsed as `1.14` —
    the execution ticket's notional/quantity/TP/liquidation outputs all off
    by five orders of magnitude, and `rank_signals`' 盈亏比 leg contaminated
    the same way. `_fnum` accepts exponent form; `_money`/rank `_m` render
    sub-1e-3 prices in `%.4g` instead of "$0"/"0".
  - **`render_ticket` crashed on breakout layouts** — a long ticket whose
    every parsed resistance sits BELOW entry (ordinary breakout narrative)
    made `min()` consume an empty generator and the whole script die with a
    traceback; the structure-reference row is skipped honestly now (short
    mirror included).
  - **`store_decision`'s idempotency guard keyed on (date, ticker)
    only** — the second venue's same-day decision was silently dropped,
    defeating the asset= tag's stated purpose. The guard (and
    `update_with_outcome` / `batch_update_with_outcomes` matching) now
    carries the venue: same-day perp+spot entries coexist and each outcome
    attaches to the venue it was priced for (`memory_resolution` threads
    the entry's asset through).
  - **`_apply_rotation` misread asset-tagged pending entries as resolved**
    (the `endswith("| pending]")` suffix test broke when the tag grew a
    keyed field) — the exact "unprocessed work" rotation promises to keep
    had become droppable, and resolved-count inflation rotated genuine
    resolved entries out early. Pending detection goes through the field
    parser (`_is_pending`) now.
  - **Known issues (need a product decision, not silently changed)**: (a)
    `trade_ticket`'s USDT-suffix heuristic still labels every crypto
    report `crypto_perp` for the ticket header — the spot/perp discriminator
    (overlay perp-only bullets) exists but re-classing spot reports as
    `crypto_spot` changes the user-visible ticket semantics, so it waits
    for a call; (b) the funding gate is sign-blind for V2.4 SHORT candidates
    in the non-default shadow/enforced portfolio-control modes (positive
    carry penalises shorts when it should favour them, adverse carry passes
    undampened via the disclosed transitional heuristic) — sizing-policy
    semantics, deferred; (c) portfolio-control snapshot equity uses the
    position value where advisory dollar shocks expect book equity
    (advisory-only, shadow/enforced).

### Changed

- **Project renamed: YiAgents → YiAlpha（弈·Alpha）.** Full rename across the
  board with no compatibility layer (pre-1.0, no external dependents):
  distribution / package / CLI `yialpha`, env prefix `YIALPHA_*` (84 vars),
  main class `YiAlphaGraph`, user data dir `~/.yialpha/` (one-time wholesale
  move preserves caches / checkpoints / memory / config history / accuracy
  products; per-ticker log dirs renamed `YiAlphaStrategy_logs` in lockstep),
  web UI branding, Dockerfile / compose service + volume names. Bare-named
  secrets (`DEEPSEEK_API_KEY`, `TUSHARE_TOKEN`, `BINANCE_API_KEY`) are
  unchanged. Historical entries below keep their original spelling.

### Added

- **Binance Vision deep-history archives (`crypto_perp`).** Two new tools,
  `get_binance_vision_metrics` (5m-grain open interest, top-trader long/short
  and taker buy/sell ratios) and `get_binance_vision_book_depth` (resting
  depth per signed price band), serve the official data.binance.vision daily
  zips — sha256-verified, disk-cached forever, PIT-clamped — back YEARS past
  the 30-day retention of the REST positioning endpoints. Bound for both
  live and historical perp runs (per-day files are inherently PIT-correct),
  wired at all three layers (bind / ToolNode / guard tests). Same batch:
  REST perp tools gained `period` passthrough (5m..12h), index-price klines
  (`price_type="index"`) and a live `/fapi/v1/depth` snapshot tool.
- **Data-vacuum gate (`data_vacuum_policy`, default `reject`).** A run whose
  core data calls ALL failed — the "data vacuum HOLD" that looked like a
  normal report — now raises a typed `DataVacuumError` at the trader node
  before any decision-stage LLM call is billed. The verdict is
  "core calls were attempted and none succeeded" (router successes tracked
  via `quality.record_success`), so partial degradations still produce a
  DEGRADED report while total vacuums fail loudly. Interactive `analyze`
  softens to `warn` (a human is watching); batch / `run_robust` inherit
  `reject`; `run_robust --allow-degraded` is the escape hatch back to the
  old semantics. Invalid policy values fail closed to `reject`.
- **Evidence-ledger blind spots closed (7 sites).** The router's
  core-category hard-error branch records `KIND_CORE_ERROR` before raising
  (counted into `core_sentinel_count`); the four direct-connect tools
  (market-data validator, price-structure, weekly indicators, Binance
  indicators) record their `DATA_UNAVAILABLE` degrades; Reddit records
  genuine degrades (OAuth token unavailable / OAuth failure → RSS / RSS
  failure / zero posts) while the keyless RSS default stays silent.
  `summarize_quality` adds `core_error_count` / `core_ok_count` /
  `degraded_count` / `data_vacuum`.
- **Decision-time price archived.** `full_states_log` now carries
  `price_at_decision`, `price_at_decision_basis` and `asset_type` (from the
  risk overlay's entry price, with a memoized-loader fallback when the
  overlay did not run) — the anchor the upcoming rating↔outcome accuracy
  loop needs. Older logs simply lack the fields (readers stay compatible).
- **Rating↔outcome verification loop.** `yiagents verify-history` scans
  every archived rating, fetches its PIT forward return (spot via yfinance,
  Binance perps/spot via their klines frames) and writes
  `accuracy/accuracy_report.{json,md}`: directional hit rate
  (Buy/Overweight/Sell/Underweight), Hold opportunity-cost mean, per-rating
  and per-ticker tables — all with sample sizes; not-yet-elapsed horizons are
  pending, never scored; legacy logs without asset_type get a visible
  USDT-suffix inference. `GET /api/accuracy` + the Web *Accuracy* view serve
  the artifact read-only (`available:false` + hint until the CLI runs).
- **`yiagents memory-resolve`.** Resolves pending memory-log entries for ALL
  tickers on demand — previously an entry only resolved when its ticker was
  analyzed again. The resolution core moved to
  `yiagents/graph/memory_resolution.py` (graph delegates), and the shared
  return-fetch lives in `yiagents/accuracy.fetch_returns_yf`
  (`YiAgentsGraph._fetch_returns` is now a thin delegate to it).
- **Decision-time price archived** (see above) anchors the whole loop.
- **`yiagents ic-cycle` — the IC loop's mechanical half in one command.**
  Runs export → prune verdict per ticker (same `yiagents.backtest.ic` math
  the prune CLI uses) → writes `<TICKER>_<h>d.csv` + its `.prune.json`,
  prints per-ticker verdict tables, the cross-ticker `indicator_battery`
  intersection suggestion, and the `snapshot record` command that documents
  applying it. The dataset builder moved into the package
  (`yiagents/backtest/ic_dataset.py`; the export script is now a thin CLI
  wrapper, its argv contract unchanged). NEVER edits the live config.
- **Weekly IC evidence workflow** (`.github/workflows/ic-cycle.yml`): Saturday
  04:30 UTC cron + manual dispatch, own concurrency group, uploads `ic_data/`
  artifacts for 90 days; an empty dataset fails the job rather than shipping
  an empty artifact.
- **`snapshot record --evidence` hygiene**: a path-looking evidence value
  that does not exist warns loudly (the audit trail would point at nothing);
  free-text descriptions stay silent.
- **`indicator_ic_context` (default off)**: when enabled, the market analyst
  gets one advisory line per indicator with its trailing mean |IC| averaged
  over `ic_data/*.prune.json` — the runtime consumer of the pruning evidence,
  and the first reader of the `.prune.json` format. Off = prompt
  byte-equivalent to the A/B baseline.



- **Default multi-vendor fallback chains.** The four core categories chain
  `yfinance,alpha_vantage` (fundamentals additionally `sec_edgar`), with
  keyless unlimited yfinance always first and rate-limited Alpha Vantage as
  tail-only fallback. `config-check` warns when a chain includes
  alpha_vantage without `ALPHAVANTAGE_API_KEY`.

### Changed

- **HTTP timeouts on by default.** `YIAGENTS_HTTP_TIMEOUT_S` (yfinance)
  defaults to 30s — was opt-in/off; explicit `0` still disables. BaoStock's
  raw TCP session (unreachable by the requests-level shim) gets
  `YIAGENTS_BAOSTOCK_TIMEOUT_S`, default 30s, scoped to the session
  lifetime via save/restore `socket.setdefaulttimeout`.
- **`run_robust` quality gate inverted to default-ON.** A degraded report
  now counts as a failure and retries; `--allow-degraded` restores the old
  "accept + mark DEGRADED" behaviour (and downgrades the child's vacuum
  policy to `warn`). The old `--require-data-quality` flag is a no-op kept
  for compatibility.

### Fixed

- **Live-date anchors: a live perp run is no longer misclassified as a
  historical replay in the post-local-midnight window.** The pipeline
  carries TWO date labels — the interactive CLI mints the analysis date on
  the host-local clock while the crypto/perp data layer anchors on UTC
  end-to-end — and on any host off UTC the two disagree for one window a
  day (00:00–08:00 on UTC+8). A single-anchor "is this date today" check
  then misfired in both directions: `is_historical_date` (host-local
  anchor) judged a UTC-labelled LIVE perp run historical, silently dropping
  the bundle's funding/premium/depth/ADL legs, the six live REST tools and
  web search; `quote_fx.usdt_usd_as_of` (strict UTC anchor) returned None
  for the CLI-labelled live run, so the stock-perp fair-value bridge
  disclosed "USDT/USD fx: UNAVAILABLE" for 8h a day. Both gates (plus the
  overlay's forming-candle note, third instance of the same compare) now
  accept EITHER anchor via `utils.live_anchor_dates()`; the two dates
  differ by at most one day, so past and future labels stay historical and
  the A-share path (whose entry dates are local-minted) is unchanged —
  the 2026-09-05 sweep's "don't unilaterally switch to UTC" constraint is
  respected by widening, not by re-anchoring.
- **Vision metrics: the global (and top-trader account) long/short ratio is
  now served from the archives.** The Binance Vision metrics CSV's
  `count_`/`sum_` prefixes are legacy naming residue — every ratio column
  carries that 5m snapshot's VALUE (cross-checked against the live REST
  series on identical timestamps): `count_long_short_ratio` IS the global
  account ratio and `count_toptrader_long_short_ratio` the top-trader
  ACCOUNT ratio. The shaper mapped only a nonexistent `sum_long_short_ratio`
  spelling, silently dropping both series while three docstrings claimed the
  archives lack the global one. Output now carries
  `top_trader_account_long_short_ratio` and `global_long_short_ratio`, the
  docs are corrected, and the output header relabels
  `top_trader_long_short_ratio` as the POSITION ratio it actually is.
- **Tavily missing-key no longer burns per-run budget.** The key-pool check
  now precedes the budget charge, so a key-less run reports "configure a
  key" on every call instead of drifting into a misleading
  "budget exhausted — raise the caps" after the scope's charges are spent.
- **Batch analyst parity with the interactive CLI.** `yiagents batch` and
  `scripts/run_batch.py` now drop the Fundamentals Analyst for pure-crypto
  batches via the shared `batch_selected_analysts` helper (tokenized-stock
  perps keep it; mixed perp batches keep it for all) — previously batch
  always ran all four analysts, wasting a fundamentals LLM tool loop on
  pure cryptos with an honest-but-useless no-data section.
- **Windowed intraday `/futures/data/*` requests are now end-anchored and
  disclose truncation.** With an explicit date window needing more rows than
  the 500-row endpoint cap, the request kept the window's own `startTime` —
  and since these endpoints serve rows ascending from `startTime`, the
  OLDEST rows came back while the decision-critical tail nearest `end_date`
  was silently dropped (the header still claimed the full window). The
  request now anchors one full page before the last in-window row (the tail
  is always kept) and every affected tool's header discloses the dropped
  head with a pointer to the archive tool. Archive windows with missing
  interior days (pre-listing / unpublished) are likewise disclosed in the
  header instead of only logging — the resampling shaper would otherwise
  render the hole as a continuous series.
- **`rank_signals` / `trade_ticket` no longer crash printing to stderr on
  Windows legacy consoles.** Both scripts already forced `sys.stdout` to
  UTF-8, but ✗ markers and Chinese error text emitted on the error stream
  still hit the GBK (cp936) / cp1252 default and raised
  `UnicodeEncodeError` — the same family as the prune-CLI U+2212 crash.
  `sys.stderr` is now reconfigured alongside stdout (still inside the
  `contextlib.suppress` guard).
- **P0: Binance OHLC no longer rounded to 2 decimals.**
  `binance_klines_frame` mirrored yfinance's display rounding
  (`df.round(2)`), which zeroes sub-cent contracts outright (PEPE ≈ 1e-5 →
  0.0) and badly distorts others (1000PEPE ≈ 19% error) — and
  `get_binance_indicators` computed its whole battery on that destroyed
  frame. Prices now keep the exchange's own precision end-to-end, and the
  indicator markdown table formats adaptively by magnitude (the display twin
  of the same bug: `.2f` rendered low-price closes as `0.00`). Tests used
  ~100-priced fixtures, which is why this survived every prior audit.
- **P0: `_detect_three_black_crows` detected the wrong shape entirely.**
  The "mirror" implementation swapped Open/Close and High/Low columns
  without negating prices, so its conditions required RISING opens and
  closes: textbook three-black-crows never matched, while three ascending
  red bars were flagged as bearish crows and fed to the LLM as evidence.
  Rewritten to the textbook definition (three red bars, strictly falling
  closes, each open inside the prior body, small lower shadows) with
  regression tests for both directions; three white soldiers unchanged.
- **`weekly_pivots` off-by-one week.** `resample_weekly` already drops the
  incomplete week, so `iloc[-2]` returned the week-before-last's pivots
  (a Wednesday analysis cited levels from 8+ trading days ago). Now reads
  the last completed week directly, with a semantic test.
- **`relative_volume` division-by-zero → `inf`.** A zero prior-window mean
  (suspended/new listings) now yields NaN ("unmeasurable"), not an
  infinitely-unusual reading rendered as `inf`.
- **Cross-market relative-strength windows aligned by date.** The benchmark
  leg counted rows (21 crypto days ≈ 29 calendar days of SPY), stretching
  the benchmark window and skewing every RS ratio for crypto/US pairs; it
  now anchors on the ticker's window dates via asof.
- **IC exporter bypassed the vendor-scale fix.** `export_ic_dataset` read
  `wrapped[ind]` directly instead of `compute_indicator`, so mfi's 0–1→0–100
  rescale never applied to IC inputs (latent — the indicator gate currently
  excludes mfi). Same scale as every other consumer now.
- **`_futures_data_window` start-only lookahead.** With only `start_date`
  given, the window end defaulted to `datetime.now()` even under a pinned
  analysis date, handing backtests positioning rows past their decision
  point; the start-only branch now clamps through `current_pit_end` exactly
  like an explicit end date.
- **`resample_weekly` silently ignored an unparseable `curr_date`**, keeping
  the still-open week as if complete (a look-ahead-style weekly close); it
  now raises instead of guessing.

### Added

- **Mark-price system for USDT-M perps.** New
  `get_binance_premium_index` tool (`/fapi/v1/premiumIndex`, weight 1)
  surfaces markPrice / indexPrice / markVsIndexPct / lastFundingRate (the
  rate in effect for the NEXT settlement) / nextFundingTime, bound to the
  perp analyst in live mode; `binance_klines_frame`/`get_binance_klines`
  accept `price_type="mark"` for mark-price klines; and the analyst nudge
  now requires liquidation-distance claims to anchor to markPrice (Binance
  liquidates on mark, not last price — the old discussion was structurally
  biased). Funding cadence is inferred from settlement spacing (8h/4h/1h
  per contract) and stated in the header instead of hardcoded 8h.
- **exchangeInfo filters + order quantization.** New
  `dataflows/binance_filters.py`: TTL-cached symbol filter blocks
  (tickSize/stepSize/minNotional) with Decimal-exact `quantize_order`
  (quantity floors to stepSize, price rounds to tickSize, min-qty/notional/
  max-qty flags). The execution gateway quantizes every order through it
  before submit — LLM-sized floats otherwise draw `-1111 Precision` rejects
  on essentially every order — and fails closed when the filters can't be
  fetched or the sized order violates the symbol's rules (no silent
  clamping of intended exposure).
- **Position-mode + leverage handling in the execution gateway.** `connect`
  queries the account's perp position mode (`get_current_position_mode`);
  hedge (dual-side) accounts now map `position_side=LONG/SHORT` with the
  correct close-side flip and omit `reduce_only` (invalid in hedge mode)
  instead of guaranteed `-4061` rejects; one-way accounts keep BOTH +
  reduce_only. Opt-in `YIAGENTS_EXECUTION_LEVERAGE` sets initial leverage
  once per symbol (`change_initial_leverage`) and rejects the order if the
  call fails (leverage moves liquidation distance — an unconfirmed margin
  setup must not trade silently). Documented in `.env.example`.
- **Long-only perp backtesting with funding drag.** `run_backtest` now
  accepts `asset_type="crypto_perp"`: the long-only engine charges each
  day's funding settlements on the held notional (strategy AND buy-and-hold,
  so the comparison stays apples-to-apples), sourced from the Binance
  funding vendor or an injectable `funding_provider`. Missing funding data
  fails closed (a spot simulation relabeled as a perp backtest is worse
  than no answer); `config_summary` states the model's limits explicitly
  (shorting/leverage/margin/liquidation not modeled).
- **Binance live-window staleness guard.** When a klines window reaches
  near the present, a frame whose last candle is >10 days old (delisted /
  renamed contract) now raises the same typed stale error as the yfinance
  path instead of feeding months-old prices as "current". Historical
  windows stay exempt — an early-ending series is a legitimate backtest
  input.
- **Outbound-URL validation on the Binance transport.** `_do_request`
  refuses non-HTTP(S) schemes and localhost/loopback/private/reserved
  literal hosts before any request is issued, so a misconfigured base can
  never turn a market-data fetch into an internal-network probe.
- **Per-product Binance weight budgets.** The spot limiter now defaults to
  the documented 6000/min (was sharing fapi's 2400 — merely over-conservative);
  an explicit `binance_weight_threshold` still overrides both.
- **Spot-default indicator binding.** `get_binance_spot_indicators` mirrors
  the perp tool with venue defaulting to spot, bound in crypto_spot runs so
  an omitted venue argument can never silently compute spot indicators on
  perp candles (or vice versa).

### Changed

- **Reddit dataflow: OAuth-API-first.** When `REDDIT_CLIENT_ID` /
  `REDDIT_CLIENT_SECRET` are set, the sentiment analyst's Reddit source pulls
  posts from `oauth.reddit.com` via the `client_credentials` grant — carrying
  the score / comment-count engagement signals the analyst prompt weighs posts
  by, and bypassing the public JSON endpoint's WAF `403` (#862) and the RSS
  feed's per-IP `429`. The bearer token is fetched once, cached in process
  memory with a safety margin before expiry, and short negative-cached on
  failure so a multi-subreddit / multi-ticker batch coasts on RSS rather than
  hammering the token endpoint. Transport stays `urllib` (Reddit is reachable
  from China without SOCKS5), so no new dependency and no test-mock churn.
  Without creds — or on any OAuth failure (`401`/`403`/`429`/network/JSON) — the
  path is byte-equivalent to the previous RSS-only behavior, so
  `fetch_reddit_posts` keeps its signature/contract and the analyst is
  untouched. Optional `REDDIT_USER_AGENT` personalizes the UA per Reddit's API
  etiquette. The secret lives only in env / memory; it never touches disk or
  logs.
- **`get_binance_spot_perp_basis` is backtest-safe.** Optional
  `start_date`/`end_date` with `current_pit_end` clamping, and the default
  "now" end is clamped the same way — the previously documented look-ahead
  can no longer fire under a pinned analysis date.
- **A-share breadth: 90s TTL cache + vectorized aggregation.** One live run
  fetched the ~5400-row whole-market spot table per caller (breadth tool,
  regime line, each LLM re-call); the fetch is now shared behind a
  process-level TTL cache and the row loop is vectorized (per-symbol limit
  thresholds remain per-row). Tests reset the cache via
  `_patch_ak` automatically.
- **Deduplication cleanup.** `YiAgentsGraph._resolve_benchmark` now
  delegates to `market_regime.resolve_market_benchmark` (it was a verbatim
  copy that had already drifted once); the dead turbulence-only renderer
  `format_market_regime` is removed (its superset
  `format_regime_context` replaced it in 2026-08-15 and only its own tests
  still called it); double-top/bottom scanning reports the MOST RECENT
  qualifying pair instead of the oldest coincidence in the window.
- **Indicator tool docstring encourages batching.** `get_indicators` told
  the LLM to "call once per indicator" while already supporting
  comma-separated names — an 8-indicator analysis made 8 tool round-trips
  for identical data; it now asks for one batched call.
- **Simulated exchangeInfo in gateway tests.** The gateway tests'
  assertions that raw quantities/prices pass through untouched are
  superseded: they now pin a permissive tick/step grid while dedicated
  tests pin the quantization math (floor-to-step, nearest-tick,
  min-notional rejection, fail-closed on unavailable filters).

## [0.3.0] — 2026-06-22

Stabilization and extensibility release: a CI gate, a unified verified
data-access contract, a provider and data-vendor registry, and a maintenance
sweep that hardened config precedence, the model catalog, data resilience, and
structured output.

### Added

- **CI gate.** GitHub Actions runs the pytest suite across Python 3.10-3.13,
  strict `ruff`, and a clean-install smoke that imports the package and CLI to
  catch undeclared dependencies. (#994, #197)
- **Provider registry.** OpenAI-compatible providers register as a single spec,
  and a generic `openai_compatible` endpoint covers vLLM, LM Studio, and relays.
  Adds NVIDIA NIM, Kimi, Groq, Mistral, and a native Amazon Bedrock client.
- **Macro and prediction-market vendors.** FRED macro indicators and Polymarket
  event probabilities, surfaced to the news and macro analysts.
- **Programmatic report output.** `YiAgentsGraph.save_reports()` writes the
  same report tree the CLI produces, for headless and API runs. (#1037)
- **Env-configurable reasoning depth** via `YIAGENTS_OPENAI_REASONING_EFFORT`,
  `YIAGENTS_GOOGLE_THINKING_LEVEL`, and `YIAGENTS_ANTHROPIC_EFFORT`,
  each gated to the models that accept it.

### Changed

- **Verified data-access contract.** Symbol normalization on every vendor path
  (identity, returns, CLI, news); the configured vendor list is the exact
  resolution chain with no silent fallback to unselected vendors; a typed
  `VendorError` taxonomy; look-ahead-safe news windows; stale-OHLCV rejection;
  inclusive yfinance date ranges.
- **Config precedence.** An explicit `YIAGENTS_*` value or CLI flag now wins
  over interactive defaults for debate and risk round counts,
  `--checkpoint / --no-checkpoint`, and the Docker provider profile; invalid
  boolean env values fail loudly. (#975, #976, #977)
- **Current-generation model catalog.** Refreshed provider lineups; retired
  `gpt-4.1`, Claude Sonnet 4.5, and the Gemini 2.5 line.
- **Optional vendors degrade** instead of aborting a run: a failed macro or
  prediction-market lookup returns a no-data sentinel.
- **Analyst prompts lead with the current date** so tool-call date ranges anchor
  to the run date rather than the model's training cutoff. (#836)

### Fixed

- **Instrument identity.** Deterministic ticker-to-company resolution prevents
  wrong-company hallucination, and a verified market-data snapshot grounds price
  and indicator claims. (#814, #830)
- **Social and market data sources.** Reddit RSS-first with 429 backoff,
  StockTwits transport hardening, and Alpha Vantage timeout plus
  key-versus-rate-limit handling.
- **Structured output.** Local OpenAI-compatible servers no longer reject
  object-form `tool_choice`; a thinking model that returns no parsed result falls
  back to free text; null-ish strings in optional price fields coerce to `None`.
  (#1038, #1051, #1057)

### Removed

- The no-op `analyst_concurrency_limit` config knob; parallel analyst execution
  is planned for a later release. (#979)
- The unused committed `uv.lock`. (#1030)

### Contributors

Thanks to everyone who shaped this release through code, design, and reports:

[@CadeYu](https://github.com/CadeYu), [@Zavianx](https://github.com/Zavianx), [@weijianz-opc](https://github.com/weijianz-opc), [@naltun](https://github.com/naltun), [@brahmasky](https://github.com/brahmasky), [@nik2208](https://github.com/nik2208), [@thieucong98](https://github.com/thieucong98), [@Derekko-web](https://github.com/Derekko-web), [@LukiPrince](https://github.com/LukiPrince), [@Eddieargenal](https://github.com/Eddieargenal), [@Ghraven](https://github.com/Ghraven), [@ms32035](https://github.com/ms32035), [@yting27](https://github.com/yting27), [@nyxst4ck](https://github.com/nyxst4ck), [@KenCheung-AIxFinance](https://github.com/KenCheung-AIxFinance), [@yangyusheng2n](https://github.com/yangyusheng2n), [@fareloj](https://github.com/fareloj), [@haosenwang1018](https://github.com/haosenwang1018), [@octo-patch](https://github.com/octo-patch), [@seifenk](https://github.com/seifenk), [@CaoYuhaoCarl](https://github.com/CaoYuhaoCarl), [@mihailnica10](https://github.com/mihailnica10), [@Dado-hash](https://github.com/Dado-hash), [@Handsomemikezzz](https://github.com/Handsomemikezzz), [@ydhawesome](https://github.com/ydhawesome), [@macd2](https://github.com/macd2), [@AyushKar2005](https://github.com/AyushKar2005), [@wildhuman](https://github.com/wildhuman), [@robert23kim](https://github.com/robert23kim), [@bngness](https://github.com/bngness), [@tedix-rodrigo](https://github.com/tedix-rodrigo), [@malaccan](https://github.com/malaccan), [@rfalken78](https://github.com/rfalken78), [@dengli1971-droid](https://github.com/dengli1971-droid), [@proofconcept39](https://github.com/proofconcept39), [@prasta1](https://github.com/prasta1), [@liximin](https://github.com/liximin), [@jeffhuen](https://github.com/jeffhuen), [@mazar](https://github.com/mazar), [@soyangelromero](https://github.com/soyangelromero), [@CNQQC](https://github.com/CNQQC), [@dovetaill](https://github.com/dovetaill), [@fperdigon](https://github.com/fperdigon), [@gyx09212214-prog](https://github.com/gyx09212214-prog), [@RSXLX](https://github.com/RSXLX).

## [0.2.5] — 2026-05-11

### Added

- **Grounded Sentiment Analyst.** The renamed `sentiment_analyst` now reads
  real Yahoo News, StockTwits, and Reddit data before generating its report,
  replacing the prior flow that could fabricate social posts under prompt
  pressure. (#557, #607)
- **MiniMax provider** with the full M2.x catalog (M2.7 / M2.5 / M2.1 / M2
  plus highspeed variants, 204K context). Dual-region: Global
  (`MINIMAX_API_KEY`) and China (`MINIMAX_CN_API_KEY`).
- **Dual-region Qwen and GLM** with separate keys per region — international
  (`DASHSCOPE_API_KEY`, `ZHIPU_API_KEY`) and China (`DASHSCOPE_CN_API_KEY`,
  `ZHIPU_CN_API_KEY`), selectable via a secondary region prompt. (#758)
- **`YIAGENTS_*` env-var configurability for `DEFAULT_CONFIG`.** Override
  `llm_provider`, deep/quick model IDs, `backend_url`, `output_language`,
  debate-round counts, checkpoint flag, and benchmark ticker via `.env` with
  type-aware coercion (string / int / bool). (#602)
- **Interactive API-key detection in the CLI.** When the selected provider's
  key is missing, the CLI prompts for it and persists the value to `.env`
  so the analysis run continues without restart.
- **Remote Ollama support.** `OLLAMA_BASE_URL` points the CLI and the
  programmatic client at a remote `ollama-serve`. The CLI surfaces the
  resolved endpoint and warns on common malformed inputs. Adds a
  `"Custom model ID"` option for models pulled via `ollama pull`. (#648, #768)
- **Configurable news-fetch parameters** in `DEFAULT_CONFIG` — per-ticker
  article limit, macro headline limit, lookback window, and macro search
  queries. (#606, #683)
- **Configurable alpha benchmark** for non-US tickers. Replaces hardcoded
  SPY with regional indices for `.NS` (^NSEI), `.T` (^N225), `.HK` (^HSI),
  `.L` (^FTSE), `.TO` (^GSPTSE), `.AX` (^AXJO), `.BO` (^BSESN); explicit
  `benchmark_ticker` override available. Eliminates FX drift dominating
  alpha for non-USD listings. (#628, #684)
- **Multi-language output covers every user-facing agent** — researchers,
  risk debators, research manager, and trader, ending the previous
  partial-localization reports. (#575)
- **Model catalog refresh.** OpenAI GPT-5.5 frontier, Anthropic Claude Opus
  4.7, Gemini 3.1 Flash-Lite GA, xAI Grok 4.20, Qwen 3.6 line. Versioned IDs
  only; auto-shifting aliases moved to the `"Custom model ID"` option.

### Changed

- **Sentiment Analyst** is now consistently named across the CLI dropdown,
  status panel, and final reports (previously the backend was renamed but
  the CLI still said "Social Analyst"). The `AnalystType.SOCIAL = "social"`
  wire value is kept for saved-config back-compat.

### Fixed

- **Structured output works on DeepSeek V4 / reasoner and MiniMax M2.x.**
  Those providers reject `tool_choice` per their tool-calling docs; the
  binding flow now skips it automatically via a capability table.
- **`pip install .` installations pick up the project `.env`** when running
  the CLI as a console script. (#747)
- **Reports save end-to-end** — streamed chunks were previously dropped from
  `complete_report.md`. (#719, #736)
- **Ticker prompt preserves exchange suffixes** (`.SH`, `.SZ`, `.SS`, `.HK`,
  `.T`, etc.) for A-share, HK, Tokyo, and other non-US flows. (#770)
- **Docker permission errors** no longer block first-run write to
  `~/.yiagents/`. (#519, #627, #672, #771)
- **Config state no longer leaks between runs** when sub-dicts are mutated;
  `set_config` partial updates preserve sibling defaults. (#788)
- **`max_recur_limit` config actually applies** — previously read but not
  forwarded to the propagator. (#764)
- **Missing-API-key error** names the exact env var to set. (#680)
- **Quieter startup** — suppressed the noisy upstream
  `LangChainPendingDeprecationWarning` from langgraph-checkpoint; will be
  removed once that package ships its fix.

### Security

- **Ticker path-traversal validation** at every filesystem-path site (cache,
  checkpoint database, results) so a malicious ticker cannot escape its
  intended directory. (#618)

## [0.2.4] — 2026-04-25

### Added

- **Structured-output decision agents.** Research Manager, Trader, and Portfolio
  Manager now use `llm.with_structured_output(Schema)` on their primary call
  and return typed Pydantic instances. Each provider's native structured-output
  mode is used (`json_schema` for OpenAI / xAI, `response_schema` for Gemini,
  tool-use for Anthropic, function-calling for OpenAI-compatible providers).
  Render helpers preserve the existing markdown shape so memory log, CLI
  display, and saved reports keep working unchanged. (#434)
- **LangGraph checkpoint resume** — opt-in via `--checkpoint`. State is saved
  after each node so crashed or interrupted runs resume from the last
  successful step. Per-ticker SQLite databases under
  `~/.yiagents/cache/checkpoints/`. `--clear-checkpoints` resets them. (#594)
- **Persistent decision log** replacing the per-agent BM25 memory. Decisions
  are stored automatically at the end of `propagate()`; the next same-ticker
  run resolves prior pending entries with realised return, alpha vs SPY, and
  a one-paragraph reflection. Override path with `YIAGENTS_MEMORY_LOG_PATH`.
  Optional `memory_log_max_entries` config caps resolved entries; pending
  entries are never pruned. (#578, #563, #564, #579)
- **DeepSeek, Qwen (Alibaba DashScope), GLM (Zhipu), and Azure OpenAI**
  providers, plus dynamic OpenRouter model selection.
- **Docker support** — multi-stage build with separate dev and runtime images.
- **`scripts/smoke_structured_output.py`** — diagnostic that exercises the
  three structured-output agents against any provider so contributors can
  verify their setup with one command.
- **5-tier rating scale** (Buy / Overweight / Hold / Underweight / Sell) used
  consistently by Research Manager, Portfolio Manager, signal processor, and
  the memory log; Trader keeps 3-tier (Buy / Hold / Sell) since transaction
  direction is naturally ternary.
- **Pytest fixtures** — lazy LLM client imports plus placeholder API keys so
  the test suite runs cleanly without credentials. (#588)

### Changed

- **`backend_url` default is now `None`** rather than the OpenAI URL. Each
  provider client falls back to its native default. The previous default
  leaked the OpenAI URL into non-OpenAI clients (e.g. Gemini), producing
  malformed request URLs for Python users who switched providers without
  overriding `backend_url`. The CLI flow is unaffected.
- All file I/O passes explicit `encoding="utf-8"` so Windows users no longer
  hit `UnicodeEncodeError` with the cp1252 default. (#543, #550, #576)
- Cache and log directories moved to `~/.yiagents/` to resolve Docker
  permission issues. (#519)
- `SignalProcessor` reads the rating from the Portfolio Manager's rendered
  markdown via a deterministic heuristic — no extra LLM call.
- OpenAI structured-output calls default to `method="function_calling"` to
  avoid noisy `PydanticSerializationUnexpectedValue` warnings emitted by
  langchain-openai's Responses-API parse path. Same typed result, no warnings.

### Fixed

- Empty memory no longer triggers fabricated past-lessons in agent prompts;
  the memory-log redesign makes this structurally impossible since only the
  Portfolio Manager consults memory and only when entries exist. (#572)
- Tool-call logging processes every chunk message, not just the last one, and
  memory score normalization handles empty score arrays. (#534, #531)

### Removed

- `FinancialSituationMemory` (the per-agent BM25 system) and the dead
  `reflect_and_remember()` plumbing; subsumed by the persistent decision log.
- Hardcoded Google endpoint that caused 404 when `langchain-google-genai`
  changed its API path. (#493, #496)

### Contributors

Thanks to everyone who shaped this release through code, design, and reports:

- [@claytonbrown](https://github.com/claytonbrown) — checkpoint resume (#594), test fixtures (#588), design feedback on cost tracking (#582) and structured validation (#583)
- [@Bcardo](https://github.com/Bcardo) — memory-log redesign (#579), empty-memory hallucination report (#572), encoding fix proposal (#570)
- [@voidborne-d](https://github.com/voidborne-d) — memory persistence design (#564), portfolio manager state fix (#503)
- [@mannubaveja007](https://github.com/mannubaveja007) — structured-output feature request (#434)
- [@kelder66](https://github.com/kelder66) — RAM-only memory issue (#563)
- [@Gujiassh](https://github.com/Gujiassh) — tool-call logging fix (#534), test stub PR (#533)
- [@iuyup](https://github.com/iuyup) — memory score normalization fix (#531)
- [@kaihg](https://github.com/kaihg) — Google base_url fix (#496)
- [@32ryh98yfe](https://github.com/32ryh98yfe) — Gemini 404 report (#493)
- [@uppb](https://github.com/uppb) — OpenRouter dynamic model selection (#482)
- [@guoz14](https://github.com/guoz14) — OpenRouter limited-model report (#337)
- [@samchenku](https://github.com/samchenku) — indicator name normalization (#490)
- [@JasonOA888](https://github.com/JasonOA888) — y_finance pandas import fix (#488)
- [@tiffanychum](https://github.com/tiffanychum) — stale import cleanup (#499)
- [@zaizou](https://github.com/zaizou) — Docker permission issue (#519)
- [@Stosman123](https://github.com/Stosman123), [@mauropuga](https://github.com/mauropuga), [@hotwind2015](https://github.com/hotwind2015) — Windows encoding bug reports (#543, #550, #576)
- [@nnishad](https://github.com/nnishad), [@atharvajoshi01](https://github.com/atharvajoshi01) — encoding fix proposals (#568, #549)

## [0.2.3] — 2026-03-29

### Added

- **Multi-language output** for analyst reports and final decisions, with a
  CLI selector. Internal agent debate stays in English for reasoning quality. (#472)
- **GPT-5.4 family models** in the default catalog, with deep/quick model split.
- **Unified model catalog** as a single source of truth for CLI options and
  provider validation.

### Changed

- `base_url` is forwarded to Google and Anthropic clients so corporate proxies
  work consistently across providers. (#427)
- Standardised the Google `api_key` parameter to the unified `api_key` form.

### Fixed

- Backtesting fetchers no longer leak look-ahead data when `curr_date` is in
  the middle of a fetched window. (#475)
- Invalid indicator names from the LLM are caught at the tool boundary instead
  of crashing the run. (#429)
- yfinance news fetchers respect the same exponential-backoff retry as price
  fetchers. (#445)

### Contributors

- [@ahmedk20](https://github.com/ahmedk20) — multi-language output (#472)
- [@CadeYu](https://github.com/CadeYu) — model catalog typing (#464)
- [@javierdejesusda](https://github.com/javierdejesusda) — unified Google API key parameter (#453)
- [@voidborne-d](https://github.com/voidborne-d) — yfinance news retry (#445)
- [@kostakost2](https://github.com/kostakost2) — look-ahead bias report (#475)
- [@lu-zhengda](https://github.com/lu-zhengda) — proxy/base_url support request (#427)
- [@VamsiKrishna2021](https://github.com/VamsiKrishna2021) — invalid indicator crash report (#429)

## [0.2.2] — 2026-03-22

### Added

- **Five-tier rating scale** (Buy / Overweight / Hold / Underweight / Sell)
  introduced for the Portfolio Manager.
- **Anthropic effort level** support for Claude models.
- **OpenAI Responses API** path for native OpenAI models.

### Changed

- `risk_manager` renamed to `portfolio_manager` to match the role description
  shown in the CLI display.
- Exchange-qualified tickers (e.g. `7203.T`, `BRK.B`) preserved across all
  agent prompts and tool calls.
- Process-level UTF-8 default attempted for cross-platform consistency
  (note: this approach did not actually take effect; replaced in v0.2.4 with
  explicit per-call `encoding="utf-8"` arguments).

### Fixed

- yfinance rate-limit errors are retried with exponential backoff. (#426)
- HTTP client SSL customisation is supported for environments that need
  custom certificate bundles. (#379)
- Report-section writes handle list-of-string content gracefully.

### Contributors

- [@CadeYu](https://github.com/CadeYu) — exchange-qualified ticker preservation (#413)
- [@yang1002378395-cmyk](https://github.com/yang1002378395-cmyk) — HTTP client SSL customisation (#379)

## [0.2.1] — 2026-03-15

### Security

- Patched `langchain-core` vulnerability (LangGrinch). (#335)
- Removed `chainlit` dependency affected by CVE-2026-22218.

### Added

- `pyproject.toml` build-system configuration; the project now installs via
  modern packaging tooling.

### Removed

- `setup.py` — dependencies consolidated to `pyproject.toml`.

### Fixed

- Risk manager reads the correct fundamental report source. (#341)
- All `open()` calls receive an explicit UTF-8 encoding (initial pass).
- `get_indicators` tool handles comma-separated indicator names from the LLM. (#368)
- `Propagation` initialises every debate-state field so risk debaters never
  see missing keys.
- Stock data parsing tolerates malformed CSVs and NaN values.
- Conditional debate logic respects the configured round count. (#361)

### Contributors

- [@RinZ27](https://github.com/RinZ27) — `langchain-core` security patch (#335)
- [@Ljx-007](https://github.com/Ljx-007) — risk manager fundamental-report fix (#341)
- [@makk9](https://github.com/makk9) — debate-rounds config issue (#361)

## [0.2.0] — 2026-02-04

This is the largest release since the initial public version. The framework
moved from single-provider to a multi-provider architecture and grew several
production-ready surfaces.

### Added

- **Multi-provider LLM support** (OpenAI, Google, Anthropic, xAI, OpenRouter,
  Ollama) via a factory pattern, with provider-specific thinking configurations.
- **Alpha Vantage** integration as a configurable primary data provider, with
  yfinance as a community-stability fallback.
- **Footer statistics** in the CLI: real-time tracking of LLM calls, tool
  calls, and token usage via LangChain callbacks.
- **Post-analysis report saving** — the framework writes per-section markdown
  files (analyst reports, debate transcripts, final decision) when a run
  completes.
- **Announcements panel** — fetches updates from `api.tauric.ai/v1/announcements`
  for the CLI welcome screen.
- **Tool fallbacks** so a single vendor outage does not stop the pipeline.

### Changed

- Risky / Safe risk debaters renamed to **Aggressive / Conservative** for
  consistency with the displayed agent labels.
- Default data vendor switched to balance reliability and quota across
  community deployments.
- Ollama and OpenRouter model lists updated; default endpoints clarified.

### Fixed

- Analyst status tracking and message deduplication in the live display.
- Infinite-loop guard in the agent loop; reflection and logging hardened.
- Various data-vendor implementation bugs and tool-signature mismatches.

### Contributors

This release is the first with substantial outside contributions; many community
PRs from late 2025 also landed here.

- [@luohy15](https://github.com/luohy15) — Alpha Vantage data-vendor integration (#235)
- [@EdwardoSunny](https://github.com/EdwardoSunny) — yfinance fetching optimisations (#245)
- [@Mirza-Samad-Ahmed-Baig](https://github.com/Mirza-Samad-Ahmed-Baig) — infinite-loop guard, reflection, and logging fixes (#89)
- [@ZeroAct](https://github.com/ZeroAct) — saved results path support (#29)
- [@Zhongyi-Lu](https://github.com/Zhongyi-Lu) — `.env` gitignore (#49)
- [@csoboy](https://github.com/csoboy) — local Ollama setup (#53)
- [@chauhang](https://github.com/chauhang) — initial Docker support attempt (#47, later reverted; the merged Docker support shipped in v0.2.4)

## [0.1.1] — 2025-06-07

### Removed

- Static site assets that had been bundled with v0.1.0; the public site now
  lives separately.

## [0.1.0] — 2025-06-05

### Added

- **Initial public release** of the YiAgents multi-agent trading
  framework: market / sentiment / news / fundamentals analysts; bull and bear
  researchers; trader; aggressive, conservative, and neutral risk debaters;
  portfolio manager. LangGraph orchestration, yfinance data, per-agent
  BM25 memory, single-provider OpenAI integration, interactive CLI.

[0.2.4]: https://github.com/zhang12120113-creator/Yiagents/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/zhang12120113-creator/Yiagents/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/zhang12120113-creator/Yiagents/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/zhang12120113-creator/Yiagents/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/zhang12120113-creator/Yiagents/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/zhang12120113-creator/Yiagents/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/zhang12120113-creator/Yiagents/releases/tag/v0.1.0
