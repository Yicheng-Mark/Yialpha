# Changelog

> **YiAlpha**（弈·Alpha）是基于研究文献（见 [REFERENCES.md](REFERENCES.md)，收录 99 篇相关研究）独立设计的多智能体 LLM 金融交易框架。包名 / import / CLI / env 前缀 `YIALPHA_*` / 数据目录 `~/.yialpha/` 全程统一。

All notable changes to YiAlpha are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Breaking changes within the 0.x line are called out explicitly.

## [Unreleased]

### Fixed

- **The test suite was not hermetic — and CI had been red for 66 days.** The three
  test jobs of the `CI` workflow had failed on 25 consecutive runs since the last
  green one (`db79063`, 2026-09-18), including the `v1.0.0` release commit. Two
  independent causes, both invisible from the Windows dev box the suite was only
  ever verified on:
  - **22 unit tests performed real network I/O.** Fifteen Form 4 tests in
    `tests/test_round15_sec_tavily.py` resolved the ticker's CIK through the
    unpatched `_cik_for_ticker` (a real `sec.gov` request); two Tavily budget tests
    called the live API with a fake key; the sentiment analyst node's prefetch
    threads reached StockTwits and Reddit; the news-prefetch gate test let the
    company-angle vendor chain run; `get_user_selections` warmed Binance's
    equity-perp listing; two CLI quality-chain tests downloaded five years of
    NVDA / AAPL bars through the decision-price fallback. They passed only while
    the dev-box proxy was up — most vendors swallow the connection error — and
    turned red as soon as it was down.
  - **A Windows-only test fixture.** `TestStatesRootDotenv` used Windows-drive
    literals (`C:/from/dotenv`) as "absolute paths"; on Linux they are relative
    paths, which `_states_root` correctly anchors to the project root, so the test
    failed on every ubuntu lane from the day it was written.
  Verified under CI's own conditions — fresh virtualenvs with *unconstrained*
  `pip install -e ".[dev,web]"` (what the CI jobs resolve, not the lock), no
  `.env`, no proxy variables, `TZ=UTC` — on Windows and Linux (WSL Ubuntu 24.04)
  for py3.11 / 3.12 / 3.13, plus a hash-checked install from `requirements.lock`:
  8167–8168 passed, 0 failed on every lane (re-measured on the round-17 tree,
  2026-11-28; the one-test spread is a test that needs scikit-learn, which is
  not a declared dependency — see the round-17 test files under Added).
  On GitHub, the first run after the push (2026-11-28) passed every job but the
  two ubuntu test jobs, each on a single test — the directory-order test listed
  further down. The run after that fix was green on every job: 8168 passed and
  2 skipped on py3.11, py3.13 and Windows py3.12, 99.87% of 22,287 statements on
  each. It is the first green run since `db79063` (2026-09-18).
- **`scipy` was used but never declared.** `backtest/metrics.py`, `backtest/ic.py`,
  `backtest/event_study.py` and the A/B gate treat it as an optional accelerator
  and ship numpy / math fallbacks so the runtime install stays lean. The eight
  tests that pin "fallback == scipy to 1e-12" (DSR skew / kurtosis / normal
  CDF / PPF, Spearman IC, the chi-square gate) `importorskip` it, so in every
  clean install — CI included — they were silently skipped and the fallbacks were
  never verified outside the dev box. `scipy>=1.14` is now a `[dev]` dependency
  (the runtime install is unchanged); the lock gained the single
  `scipy==1.18.1` entry.
- **CI's mypy gate had drifted.** It type-checked a hand-maintained module list
  that no longer named `yialpha/instruments`, `ledger`, `perp`, `regime` or four
  top-level modules. The gate now names the package root, so a new sub-package is
  covered from the day it is created.
- The README "Project structure" tree listed 8 of the 13 `yialpha` sub-packages —
  the same four the mypy gate had lost.
- `ci.yml` claimed job summaries are anonymously readable; they are not (both the
  logs and the summary need a GitHub sign-in even on a public repository — checked
  as an anonymous visitor). The comment now says what an anonymous reader can get
  (job conclusions and step status through the REST API).
- `ic-cycle.yml` pinned `actions/checkout@v4` / `actions/setup-python@v5` while
  `ci.yml` had moved to v5 / v6.
- Three tests slept through real retry backoff (`test_binance_square` ×2,
  `test_tavily_web_search` ×1): ~6 s of wall clock for no assertion.
- **Five tests could fail for reasons outside the code they test.** None of them reproduces
  on an idle dev box; all five were found after the round-17 regression matrix had come back
  green.
  - **Three `_map_tickers` tests asserted the order in which worker threads start**
    (`tests/test_round17_scripts_e2e.py`, `TestMapTickers`). `_map_tickers`
    (`scripts/run_baseline.py`) runs one task per ticker on a pool and promises that the
    *results* come back in input order, which the pool path delivers. The tests also
    asserted the start order, and with two workers that is the scheduler's choice. Under CPU
    load on Linux the third ticker started before the second in 587 of 3,200 repetitions; on
    Windows it never did, in 3,200. The three tests now assert the contract — results in
    input order, each ticker run exactly once — and the counting task they share takes its
    counter under a lock.
  - **The real-thread fail-fast test assumed the second task had started**
    (`tests/test_round17_ops.py`,
    `test_r17_concurrent_fail_fast_with_real_threads_still_returns_every_ticker`). If BOOM
    fails before NVDA's task is picked up, fail-fast cancels NVDA, as it should; that
    happened in 5 of 480 runs on two saturated cores. NVDA now announces itself and BOOM
    waits for it.
  - **A CLI test's outcome depended on the length of the temporary directory**
    (`tests/test_round15_cli_helpers.py`, `TestRound15VerifyHistoryCommand`).
    `verify-history` prints the report paths through Rich, which folds at 80 columns when no
    terminal is attached, and where it folds depends on the path's length: with `tmp_path`
    one to nineteen characters longer than this dev box's, `accuracy_report.json` came out
    split over two lines. `/tmp/pytest-of-runner/pytest-0` on a GitHub ubuntu runner is in
    that range; the dev box's own path is one character outside it. The test sets
    `COLUMNS`, as the other CLI-output tests already do. The 17 CLI- and script-driving test
    files (1,656 tests) were then run once per temp-path length — 60 lengths on Linux, 35 on
    Windows — and no other test depends on it.
  The first of these surfaced in the westward-clock run the round-17 matrix had asked for
  (`TZ=Pacific/Honolulu`, Linux py3.13) and reproduces just as well under `TZ=UTC`: it is a
  scheduling race, not a clock issue. A read-only review of every test that starts threads or
  processes then listed 15 further candidates; pinned to two cores beside eight CPU burners,
  the fail-fast test above was the only one that failed (5 of 13,920 executions).
- **A test asserted a directory listing's order** (`tests/test_round17_scripts_e2e.py`,
  `TestRobustFindNewReport`). This one reached CI: on the first run of the pushed tree it
  was the single failure on both ubuntu lanes (the Windows lane passed). `_find_new_report`
  (`scripts/run_robust.py`) returns the first report whose mtime beats the pre-attempt
  snapshot, and among reports with the *same* mtime that is whichever one `Path.glob`
  happens to yield first — the file system's order, which is sorted by name on NTFS, is
  creation order on a fresh WSL disk, and is neither on a hosted runner's ext4. The test
  expected a particular one of two tied reports. It now checks the unambiguous cases while
  there is a single newest report and treats the tie as a tie. The suite was then run once
  per scrambled directory order — six orders on Linux py3.13, four on Linux py3.11, four on
  Windows py3.12 — and this was the only test whose outcome depended on it. The same sweep
  turned up a neighbour with a different defect:
  `test_snapshot_record_warns_on_missing_pathlike_evidence` (`tests/test_ic_cycle.py`)
  wrote a scratch file into the working directory, where two suite runs in one checkout
  could delete each other's copy mid-test (it happened once in six parallel full runs); the
  file now lives in the test's `tmp_path`.
- **Ledger connections were dropped without `close()`** (`yialpha/ledger/sqlite.py`,
  `_connect`). `get_connection()` keeps one connection per path per mode in a
  `threading.local`. A connection opened inside a worker thread (batch runner, prediction
  flush, the web app) has no other owner, and one whose PRAGMA setup fails never reaches the
  cache: both were dropped without `close()` when the cyclic collector got to them. Nothing
  leaked — the dealloc closes the database — but Python 3.13 reports an unclosed connection
  as `ResourceWarning: unclosed database`, pytest-cov lifts Python's default ignore for
  exactly that message, and every run under coverage on 3.13 printed one line per connection
  (21 on the tree this follows, Windows and Linux alike; 3.11 and 3.12 emit no such warning).
  `_connect` now asks for a `sqlite3.Connection` subclass whose finalizer closes the
  connection. Three tests in `tests/test_round17_ledger.py` pin it: a worker thread's
  connections are closed once the thread is gone, a connection whose setup fails is closed
  too, and a connection whose *open* fails — `sqlite3.connect()` allocates before it opens,
  leaving an instance whose `__init__` never ran, on which `close()` raises — is finalized
  silently. With them CI's coverage command reports 0 warnings on py3.13, on Windows and on
  Linux (8171 passed, 2 skipped on both).
- **Product bugs found by the round-16 audit.** Each one is pinned by a
  regression test that was run against the unmodified `v1.0.0` checkout to
  confirm it fails without the fix:
  - **The market gate let quoted CFD aliases through**
    (`yialpha/dataflows/symbol_utils.py`). `_gate_deny_reason` rejected the bare
    aliases (`GOLD`, `XAU`, `SPX500`, `GER40`, `US30`, `HK50`, `WTI`, `BRENT`) but
    not the broker form with a quote attached — `GOLDUSDT`, `XAU-USDT`,
    `SPX500USDC`, dashed or compact, `USD` / `USDT` / `USDC` — which satisfies
    the crypto-venue suffix check, so a commodity / index CFD was treated as
    crypto and the venue normalizer fabricated a Binance pair for it (48 cases).
  - **DSR and Sharpe were meaningless on a variance-free series**
    (`yialpha/backtest/metrics.py`). The `std == 0.0` guard missed a standard
    deviation of ~1e-15 — what a mathematically flat equity curve leaves after
    `equity[1:] / equity[:-1] - 1` is rounded — so `mean / std` blew up and the
    noise-driven skew / kurtosis reported a **Deflated Sharpe Ratio of 0.9995
    for a flat strategy**, and a Sharpe of ~1e15. A standard deviation below
    1e-12 is now handled like an exact zero; a NaN one still propagates.
  - **The validation gate failed open on unmeasured runs**
    (`yialpha/backtest/validation_gate.py`). Improved runs without metrics were
    silently dropped from the means, so with a zero hurdle and a losing
    buy-and-hold "zero measurements" evaluated to PASS. The gate now requires
    every improved run to carry metrics and says so in its notes.
  - **A one-sided trade ledger fabricated a payoff ratio**
    (`yialpha/risk/manager.py`). `_trade_stats` used `1.0` as the sentinel for
    "no wins" / "no losses", which then posed as a real 100% average: a 10-0
    record scored a Kelly fraction of 0 while a ledger that had **never won**
    (0-3) scored a *positive* Kelly fraction. An unobserved side is now `0.0`,
    which `kelly_fraction` already documents as "unknown payoff".
  - **An at-limit candidate was reported as clipped** (`yialpha/risk/constraints.py`).
    A candidate that fills the book to exactly the limit in real arithmetic
    (0.5 + 0.1 under a 0.6 cap) failed the strict `limit - others >= contribution`
    test because `0.6 - 0.5` is `0.0999…98` in binary, and came back as
    `RESIZE ×0.9999999999999998` with the reason `exceeds_limit` — a full-size
    trade labelled as resized by a binding constraint. The comparison now uses
    the same 1e-12 tolerance as the over-limit branch above it; a real breach
    of 1e-6 is still clipped.
  - **Trade tickets could carry a stop that cannot exist**
    (`scripts/trade_ticket.py`). Only the entry was validated: `Stop Loss: $0`,
    a signed percentage, or a 2·ATR-derived negative stop (ATR ≥ entry / 2) all
    produced an "ok" ticket with a stop distance of 100% or more. A non-positive
    stop is now a degenerate ticket. The execution plan also told **shorts** to
    move the stop *up* to breakeven — i.e. to the losing side.
  - **SEC EDGAR's point-in-time check compared dates as strings**
    (`yialpha/dataflows/sec_edgar.py`). `str(filed)[:10] <= str(curr_date)[:10]`
    with an LLM-supplied `curr_date`: `"2024-11-15" <= "2024/05/03"` is true
    because `-` sorts before `/`, so a May as-of date exposed every later 2024
    filing (and `"2024-05-10" <= "2024-5-3"` likewise). Both sides are now
    parsed; an unparseable `curr_date` fails closed, the same policy as
    `is_filing_public` — the three statements then raise `NoMarketDataError`
    (the router turns that into NO_DATA) instead of returning a leaking report.
    An FTD row whose quantity cannot be parsed also crashed `get_ftd_data`
    (`yialpha/dataflows/sec_ownership.py`) instead of rendering `n/a` for the
    peak.
  - **The stale-cache cap silently disappeared for `inf`**
    (`yialpha/dataflows/disk_cache.py`): a non-finite `data_cache_max_stale_days`
    removed the ceiling, so arbitrarily old cached bytes were served when a fetch
    failed. Non-finite values now fall back to the 30-day default.
  - **A negative filing lag reopened look-ahead**
    (`yialpha/dataflows/utils.py`): `YIALPHA_FUNDAMENTALS_FILING_LAG_DAYS=-5` was
    accepted, putting `period_end + lag` before the period end. Negative and
    garbage values now fall back to 45 (`0` remains the documented opt-out).
    `pinned_analysis_date` also wrote `None` on exit instead of restoring the
    previous value, so a nested use cleared the outer run's PIT clamp.
  - **One malformed log file took the Web UI's read views down**
    (`web/store.py`). The readers caught only `OSError` and `JSONDecodeError`,
    so a log that was not valid UTF-8, whose root was not an object, or that
    contained a `NaN` / `Infinity` literal raised and 500'd `/api/tickers`,
    `/api/compare` and the run-detail view — every one of those is now skipped or
    served as `null`, and non-string fields read as empty. In `web/app.py`:
    the Host-rejection 403 was the only response without CSP / `nosniff`;
    `_hostname_of` accepted malformed Host headers such as `[::1]evil.com` or
    `127.0.0.1:80@evil.com` as loopback (it now parses `host[:port]` /
    `[v6][:port]` strictly and fails closed); a malformed `YIALPHA_WEB_ALLOWED_HOSTS`
    entry could whitelist the empty host name; and `/api/analyze` spawned the
    child for tickers like `"BTC USDT"` or `"[A-Z]USDT"` that the child then
    rejects — it now applies the same `is_valid_ticker_input` check up front.
- **Product bugs found by the round-17 audit** (54 fixes in 28 files). As in
  round 16, every fix is pinned by a regression test in a `tests/test_round17_*.py`
  file that was run with the fixed file reverted to its pre-round-17 version to
  confirm it fails without the fix. Line numbers refer to this tree.

  - *Agents and tools*
    - **A blind forecast could be filed with a field silently missing, or with
      a JSON `true` in place of a number**
      (`yialpha/agents/utils/prediction_tools.py:91`, `:124-135`).
      `PredictionEntryInput` dropped unknown keys, so `prob_pu` typed for
      `prob_up` was accepted and the row — immutable once filed — carried no
      probability; unknown keys are now kept and the tool body rejects that
      entry with its reason. A JSON boolean in a numeric field was coerced to
      `1.0` / `0` (a 1-day horizon for `horizon_days`), filing a "certain
      forecast"; it is refused before the coercion. (That refusal is
      type-level: it rejects the whole call, not just the one entry — the
      existing behaviour for type errors.)
    - **The code-sandbox watchdog could be switched off after construction**
      (`yialpha/agents/utils/pot_executor.py:551-555`). The constructor clamps
      a `None` or non-positive `timeout_seconds` to the module minimum, but
      `_run_with_timeout` re-read that public attribute at call time and, on
      `None` or ≤ 0, ran the function with no watchdog at all. It now applies
      the constructor's clamp.
    - **The positioning analyst was asked for labels with zero positioning
      evidence** (`yialpha/agents/analysts/positioning_analyst.py:117-127`,
      `:294-312`). `fetch_perp_market_bundle` fails soft per component:
      during an exchange outage, or on a replay date past the REST retention
      with no archive to read, it returns a bundle whose every leg is
      `unavailable` and raises nothing. The round-12 gate only stopped a
      raised fetch, and the rendered block (header + availability footer) is
      non-empty, so the LLM was invoked on no evidence and the schema's
      required labels were rendered as a data-derived header. At least one
      component must now have `status == ok`; otherwise the node writes an
      honest skip note and logs a warning, with zero LLM calls.
    - **Structured decisions turned JSON booleans and non-finite text into
      numbers** (`yialpha/agents/schemas.py:56-81`, `:728-736`).
      `_coerce_optional_float` now degrades a boolean and `"nan"` / `"inf"` /
      `"1e999"` to `None` like any unparseable string: `confidence: true`
      rendered as `**Confidence**: 100%`, `price_target: true` became a 1.0
      target and `"nan"` rendered as `**Price Target**: nan`.
      `SentimentReport.overall_score` refuses a boolean (`true` was read as
      1.0 on its 0–10 scale).
    - **A malformed portfolio snapshot crashed the PM node, and NaN reached
      the prompt** (`yialpha/agents/managers/portfolio_manager.py`,
      `_format_portfolio_state` and `_safe_float`). `{"equity": "100k"}`
      raised `ValueError` out of the whole node, `Equity: nan` was rendered
      into the prompt, and a `positions` that is not a mapping raised
      `AttributeError`. `equity` / `cash` now take the guard the holdings
      already had, `_safe_float` refuses NaN / inf, and a non-mapping
      `positions` is dropped.

  - *Ledger, backtest and risk constraints*
    - **A bad vendor close fabricated a return in the prediction ledger**
      (`yialpha/ledger/outcome_compute.py:913-918`, `:970-971`, `:1018`). The
      legacy path's entry price had no gate: a NaN close was written as a
      `complete` row with a NULL return, and a 0 close produced −100%. Entry,
      terminal and underlying closes must now be finite and positive — the
      gate the reference path already had.
    - **Booleans were filed as probabilities and horizons**
      (`yialpha/ledger/models.py:207`, `:219`): `validate_probability` and
      `validate_horizon_days` refuse them.
    - **`run_backtest` accepted capital and costs that are not finite
      numbers** (`yialpha/backtest/engine.py:781-786`, `:827-835`). Zero,
      negative, NaN or inf `initial_capital` produced a flat, short or NaN
      book and was reported on; NaN fails every `< 0.0` test and +inf passes
      it, so a non-finite fee or slippage landed on every fill and the whole
      equity curve came back NaN with no exception. Both now raise
      `ValueError`.
    - **`periods_per_year` was validated only after the decision loop**
      (`yialpha/backtest/engine.py:854-860`): a float or NaN paid for every
      decision date before `compute_metrics` rejected it. It must now be an
      `int >= 1` before any decision is billed.
    - **A non-finite funding rate counted as coverage**
      (`yialpha/backtest/engine.py:979-985`): a priced day whose summed rate
      is NaN / inf was skipped by the charge loop — zero drag, exactly like
      an absent day — yet counted toward the funding-coverage gate. It is
      missing coverage now.
    - **A NaN target weight poisoned the equity curve**
      (`yialpha/backtest/engine.py:1420-1429`): `np.clip` passes NaN through,
      and a NaN `weight_fn` result reached the share count and the cash leg,
      turning every later equity point NaN. `None` remains the documented
      "hold"; anything that is not a real number raises.
    - **The event study counted events it could not decide**
      (`yialpha/backtest/event_study.py:298-312`): a NaN / inf return
      anywhere in the estimation or the event window left the fit or the CAR
      NaN, and the event was still counted with `car=nan`, sinking the
      cross-event mean and every statistic derived from it. Such events are
      now skipped with a warning, like a window that runs off the tape.
    - **`advisory_metrics` computed a beta from misaligned series**
      (`yialpha/risk/constraints.py:684-688`). Its docstring promises `None`
      for misaligned input, and finite values were paired by date — but two
      series of different lengths were truncated by `zip` and paired from the
      head: a 3-day history against a 2-day benchmark read as a beta of 0.5
      and `btc_crash_x_earnings_gap` produced a number. Unequal lengths now
      yield `None`. (The only production caller passes no benchmark series.)

  - *Data vendors*
    - **A perp price book with no usable price passed as GOOD tier**
      (`yialpha/dataflows/perp_bundle.py:209-222`). `_fetch_prices` took the
      tail close as-is, so a 0 rendered "last 0.00… (1d -100.00%)" and a
      −10,000 bps mark basis, NaN / inf rendered "nan" / "inf" — all with
      `core_complete=True`. A close that is not a finite positive number now
      fails the leg.
    - **A partly served bundle component said nothing about its missing
      legs** (`yialpha/dataflows/perp_bundle.py:921-939`): `_component_footer`
      noted nested legs only under a component whose overall status was not
      ok, so a core-complete price book without its index leg, or one LSR
      vantage out of two, rendered its surviving reading in silence. Nested
      legs get their own note.
    - **A zero close read as a crash in the market-regime index**
      (`yialpha/dataflows/market_regime.py:200`, `:210`, `:265`).
      `compute_turbulence` treated a zero or negative close as a −100% day —
      one zero-close bar rendered "elevated (107.4σ)" — and its numpy
      fallback dropped the oldest return; `classify_trend_state` read a zero
      close as a downtrend. Zero / negative closes are now missing rows and
      the trend component is omitted.
    - **A malformed subreddit feed aborted the whole analysis**
      (`yialpha/dataflows/reddit.py:171`): `safe_xml_root` rejects a body
      with a DOCTYPE / ENTITY by raising `ValueError`, which
      `_fetch_subreddit_rss` did not catch; the sentiment analyst node has no
      try/except, so the exception ended the run. It now takes the degraded
      fetch path.
    - **Indicator windows returned an error string as every indicator's
      value** (`yialpha/dataflows/y_finance.py:308`,
      `yialpha/dataflows/alpha_vantage_indicator.py:62`): with a `curr_date`
      that carries a time, both tools raised `ValueError` and the tool layer
      handed the error text to the model. The date is cut to its first ten
      characters.
    - **An empty Alpha Vantage body was a successful payload**
      (`yialpha/dataflows/alpha_vantage_fundamentals.py:8-24`): HTTP 200 with
      a blank body travelled on as `''`, and the router recorded three
      successes and zero sentinels. `_api_text` now raises
      `NoMarketDataError`, so the router records its sentinel and tries the
      next vendor.
    - **Two non-canonical date forms defeated CoinGecko's point-in-time
      filter** (`yialpha/dataflows/coingecko_vendor.py:512`): a basic ISO
      date (`20260627`) or an ISO week date was compared with the row dates
      as strings, letting every row of the same year through. The as-of label
      is normalised to `YYYY-MM-DD` first.
    - **One row without a `y` value dropped a whole chart**
      (`yialpha/dataflows/onchain_flows.py:120`): `_points` raised `KeyError`
      on a point that has `x` but no `y`. Such rows are skipped.
    - **DefiLlama chain-resolution failures were silent or mislabelled**
      (`yialpha/dataflows/defillama_vendor.py:468`, `:525`): the fallback
      `except` now logs a warning, and a directory outage is no longer
      reported as the symbol having no coverage; a 429 stays a
      `VendorRateLimitError`.
    - **A transient read failure read as "nothing is synced"**
      (`yialpha/dataflows/binance_vision_store.py:132-147`): the lock-free
      `_read_manifest` hit `PermissionError` while a concurrent sync's
      `os.replace` held the file, took it for a missing manifest and reported
      0 synced days — `qa()` came back blank and the bookDepth memory gate
      was skipped. It now sleeps 0.2 s and re-reads once.

  - *LLM clients*
    - **A cache replay broke the next call's cache key**
      (`yialpha/llm_clients/base_client.py:22-33`). On a cache hit
      langchain-core copies the replayed message and stamps
      `usage_metadata["total_cost"] = 0` on it (`_convert_cached_generations`;
      same on 1.5.3, 1.6.1 and 1.8.4). `usage_metadata` is part of the prompt
      serialization, so once a replayed assistant message sat in the history
      the next call's key did not match — in a tool loop only the first call
      was ever replayed. `normalize_content` now drops that zero marker (live
      messages never carry it; a no-op with the cache off).
    - **A transient "cannot open" evicted a valid cache entry**
      (`yialpha/llm_clients/response_cache.py:162-175`): `DiskLLMCache.lookup`
      treated every read failure as a corrupt entry and deleted it —
      including the `PermissionError` of the instant `os.replace` on Windows,
      i.e. a paid-for generation a writer had just landed. `OSError` is now a
      miss that keeps the entry and logs `could not read`; only unparsable
      content is evicted.

  - *Logging*
    - **Secrets reached the log sinks through four side doors**
      (`yialpha/logging_config.py`). `CredentialRedactionFilter` looked only
      at `record.exc_text`, which is still `None` on a fresh record while the
      handler renders the live exception, so a key inside an exception's
      message (`requests` puts the whole URL into its `HTTPError`) went out
      unredacted (`:235-257`); a record whose %-args were malformed was
      reported by the handler with `record.msg` and `record.args` verbatim
      (`:221-230`); the `apikey=` pattern did not match the quoted JSON form
      (`:128`); and the exact-match list of key environment variables lacked
      every provider in use, `DEEPSEEK_API_KEY` included (`:90-115`). Also
      added: the `sk-<32-hex>` key shape (`:156`) and a 6-character floor
      below which a configured value is not treated as a credential (`:122`,
      `:189-203`).

  - *Web UI*
    - **A non-finite number in the ledger 500'd the read-only views**
      (`web/store.py`). Ticket, snapshot and position payloads are stored
      with `json.dumps` defaults, so a bare `NaN` / `Infinity` lands in the
      cell and a REAL column holds `inf`; the readers returned it and the
      response — rendered with `allow_nan=False` — failed the whole endpoint
      (for positions, the entire book). `load_run_predictions` (`:365`),
      `load_outcomes` (`:394`), `load_portfolio_snapshot` (`:471`) and
      `load_positions` (`:490`) pass their results through
      `yialpha.accuracy.sanitize_nonfinite`.
    - **An overflowing number literal 500'd a run page** (`web/store.py:134`,
      `:449`): `1e999` parses to `inf` through `json.load`, and the existing
      `parse_constant` hook only sees the `NaN` / `Infinity` words.
      `_read_json_object` and `load_ticket` add a `parse_float` hook.
    - **An unreadable ledger was served as "empty but available"**
      (`web/store.py:307`, `_ledger_readable`, used by `load_outcomes` and
      `load_calibration`): a zero-byte, corrupt or locked ledger file
      degraded to an empty read and answered `available: true`. It now
      answers `available: false`.
    - **One entry that cannot be stat'ed marked a finished run as an error**
      (`web/store.py:217-224`): `Path.is_dir()` raises the stat error on
      Python 3.11–3.13, and in `list_reports` it sat outside the
      `except OSError`, so that ticker's runs page returned 500. `is_dir()`
      moved inside.

  - *Scripts*
    - **`run_baseline` ran a ticker named twice, twice**
      (`scripts/run_baseline.py:162-183`): the pool path keys its results by
      ticker, so a ticker named twice ran two backtests racing on that
      ticker's state logs and cache, returned the second result twice and
      discarded the first (the serial path returned two distinct results).
      Duplicates are dropped by `strip().upper()` and listed on stderr.
    - **`run_baseline` hid a shrunken rebalance calendar**
      (`scripts/run_baseline.py:83-102`): when the window held fewer trading
      days than `n`, the branch returned every trading day in the window
      (consecutive, no longer `step` apart) without a word. It prints the
      round-10 "calendar shrunk" warning too.
    - **`run_baseline --smoke` passed `--date` to the graph unvalidated**
      (`scripts/run_baseline.py:551-561`): the graph does not validate it
      either, so an impossible date paid for the analysts before the
      data-vacuum gate stopped the run, and a non-canonical one wrote
      `full_states_log_<as typed>.json` where the web store and
      `verify-history` never look. It now goes through `is_canonical_ymd`,
      the predicate `run_robust --date` and the web endpoint already use.
    - **`run_robust` accepted non-finite timeouts**
      (`scripts/run_robust.py:146-160`): the lower-bound checks compare with
      0, which NaN fails and +inf passes. `--per-ticker-timeout nan` was
      accepted and `elapsed > nan` is always false, so the wall-clock
      watchdog could never fire; a NaN / inf `--backoff` reached `time.sleep`
      and escaped the worker thread; a negative `--stall-cpu` never tripped.
      The four flags must be finite and `--stall-cpu` ≥ 0.
    - **`run_robust` analyzed a ticker named twice, twice**
      (`scripts/run_robust.py:191-212`): one child is spawned per list entry,
      so two children worked on one ticker's state logs and caches at once,
      both worker slots went to it and the summary showed two rows.
      `--tickers` is deduplicated the way `run_batch` / `run_baseline` do it,
      and the dropped spellings are listed.
    - **A degraded run was reported as "no new report"**
      (`scripts/run_robust.py:760-773`): in strict mode (the default) a
      DEGRADED verdict retries instead of breaking, but the "exit 0 but no
      new complete_report.md" line and the `result["reason"]` overwrite still
      ran after it, so the summary claimed that a run had written no report
      when it had — only with degraded data. The verdict keeps its own
      reason.
    - **Ctrl+C still slept the backoff** (`scripts/run_robust.py:779-785`):
      the abort flag was read only at the top of the next iteration, i.e.
      after the backoff — once main()'s handler had killed the children, a
      worker still slept `--backoff` (15 s by default) and
      `pool.shutdown(wait=True)` sat it out. The flag is checked before
      sleeping.
    - **The analyst-parallel A/B gate crashed when one leg had no usable
      run** (`scripts/run_analyst_parallel_ab.py:1005-1012`,
      `_compute_metrics`): `all([])` is true, so a leg whose every
      `propagate` failed passed the check and the mean divided by zero; the
      whole run died as an "infra crash" (exit 3, no report) instead of the
      criterion-6 / criterion-7 failure those criteria exist to report. An
      empty leg reports N/A.

  - *CLI*
    - **The live panel froze at midnight** (`yialpha/cli/main.py`,
      `update_display`): messages were sorted by their `HH:MM:SS` stamps as
      text, so in a run crossing 00:00:00 everything after midnight ranked
      below the pre-midnight rows and the newest-messages window stopped
      updating. They are ordered by seconds past midnight, with pre-midnight
      rows moved back one day when the stamps wrap.
    - **The header text was cropped** (`yialpha/cli/main.py`, `update_display`):
      the header region is fixed at 3 rows (border, text, border); a vertical
      padding of 1 made the panel 5 rows and the layout cropped the middle —
      the text itself. The padding is 0.

### Security

- **`requirements.lock` resolved 43 known vulnerabilities in 7 packages**
  (pip-audit on 2026-11-24: aiohttp 17, filelock 12, pyasn1 4, soupsieve 4,
  urllib3 3, langgraph-sdk 2, multidict 1). The lock was regenerated: 8 of its
  120 pins moved — aiohttp 3.14.3→3.14.5, filelock 3.32.2→4.3.1,
  langgraph-sdk 0.4.2→0.4.14, multidict 6.7.1→7.2.1, pyasn1 0.6.4→0.6.5,
  soupsieve 2.9.2→4.1.3, urllib3 2.7.0→2.8.0, yarl 1.24.5→1.25.1 — and nothing
  was added or removed by the audit. filelock crosses a major version because
  every advisory against 3.32.2 is fixed only in 4.x. The audit reports no
  known vulnerability on the regenerated lock (at the time of the audit).

### Added

- **A network guard for the whole suite** (`tests/conftest.py::NetworkGuard`,
  autouse): real network I/O raises where it happens, and — because most vendors
  catch `Exception` and degrade — every attempt is also recorded and re-checked at
  teardown, so an attempt whose error the code under test swallowed still fails the
  test. Three transport layers are covered, since a vendor only has to sit on one
  that is not: Python sockets (`requests` / `urllib3` / sync `httpx`), the Windows
  asyncio Proactor loop (`ConnectEx` never calls `socket.connect`), and
  `curl_cffi` (yfinance builds a libcurl session whenever the package is
  installed). Connections made from inside `socket.socketpair()` are let through
  (in-process plumbing on Windows); the hop to a loopback proxy is not.
  `tests/test_round16_network_guard.py` pins each layer, including the
  proxy hop. `-m integration` remains the opt-out.
- **Drift guards** (`tests/test_round16_packaging_ci.py`): `requirements.lock`
  must satisfy every dependency `pyproject.toml` declares for the scopes the lock
  covers and must stay exactly pinned (CI installed the lock but never validated
  the project's own metadata against it); the mypy gate must name the package
  root; the coverage floor must not be quietly lowered; every `yialpha <command>`
  the docs mention must be registered on the real CLI; the README project tree
  must list every sub-package and every CLI command.
- **The round-17 test files** — 13 files, 2,711 tests: `tests/test_round17_`
  `pot_tools.py` (170), `ledger.py` (186), `backtest_engine.py` (147),
  `perp_bundle.py` (95), `ops.py` (167), `agents.py` (192), `llm_clients.py`
  (233), `dataflows_vendors.py` (422), `web_runtime.py` (157),
  `scripts_e2e.py` (722), `test_hygiene.py` (67), `mutation_graph.py` (97)
  and `cli_ui.py` (56). On the final tree (2026-11-28), under the CI conditions
  described under Fixed: Windows py3.13 (the baseline interpreter), py3.11 /
  3.12 / 3.13 (fresh installs) and the locked 3.12 install; Linux (WSL Ubuntu
  24.04) py3.11 / 3.13 (fresh installs) and py3.12 with `requirements.lock`
  installed hash-checked — the first time the lock was installed on Linux.
  0 failed on every lane, 8167 or 8168 passed and 2 or 3 skipped: the skips are
  `langchain_aws` (the optional Bedrock extra, not installed), a symlink test
  Windows does not permit, and one test that needs scikit-learn, which is
  present only on the dev baseline interpreter. Also green with the collection
  order reversed (Windows, Linux py3.11) and on clocks whose calendar date is a
  day ahead of UTC (`TZ=XXX-14`: Windows, Linux py3.13) or a day behind it
  (`TZ=XXX24` and `TZ=XXX18` on Windows; `TZ=XXX24` on Linux py3.11 and py3.13).
  The literal westward settings — `TZ=HST10` on Windows and `TZ=Pacific/Honolulu`
  on Linux py3.11 and py3.13 — need the UTC clock between 00:00 and 10:00 and
  were run afterwards (2026-11-28), on the tree with the five test fixes
  described under Fixed: green on all three. On the round-17 tree itself the
  Linux py3.13 run came back with the `_map_tickers` failure listed there.
  A plain run reports 0 warnings. On the round-17 tree a run under coverage printed 21
  `ResourceWarning: unclosed database` lines on the py3.13 lanes — ledger
  connections that were never closed explicitly; see under Fixed, the tree now
  reports none — and the locked 3.12 install prints one
  `StarletteDeprecationWarning` from FastAPI's test client; neither comes from
  a test of its own.
- **Test-hygiene pins** (`tests/test_round17_test_hygiene.py`): 12 checks that
  hold the suite itself to the rules this round fell foul of. They look at
  every `.py` file under `tests/`: each one must parse; no test body may be
  empty or a constant `True`; `skip` / `xfail` must carry a `reason`; a real
  `time.sleep` longer than 0.05 s, a Windows drive-letter literal and a
  `sys.path` edit each need an allow-listed reason (the allow-lists are exact
  sets and are checked in both directions); pytest-style tests, fixtures and
  helpers never write `os.environ` directly; no warning filter outlives its
  block; `pyproject.toml` silences only `DeprecationWarning`; every
  `importlib.reload` a test reaches must be undone; and a list of session
  objects must still be what it was at collection when the last test file
  runs. 44 existing test files were brought up to them: the full suite went
  from 45 warnings to 0, nine files stopped leaking a reloaded module into
  the rest of the session (two round-17 sessions went red, 6 and 20 tests, on
  state leaked that way), three real sleeps went, and 16 tests that passed
  the local calendar date where the product reasons in UTC now use the
  product's own clock. A plain run of the full suite went from 45 warnings
  to 0.
- **A mutation-testing pass over the graph and the risk money path**
  (`tests/test_round17_mutation_graph.py`, 97 tests). 478 single-line
  mutations were applied, one at a time and only in throwaway copies, to 15
  modules (`yialpha/graph/*` and `yialpha/risk/{constraints,manager,resolver,
  perp_ticket}.py`), each at a decision that changes a trade, a risk limit, a
  price, a position or which node runs next. The existing suite killed 364;
  114 survived the whole suite, and the pass found the `advisory_metrics` fix
  above. 96 of the new tests each kill one of the survivors (99 mutations);
  15 are left alive on purpose — 9 equivalent, 5 on branches production
  cannot reach and 1 observable only in the last bit of a float. The gaps
  clustered in the backtest adapter `build_backtest_weight_fn` (latest
  equity, realized P&L merged once, Hold flattened under a hard stop, the
  engine's current position), in what the risk overlay forwards from the
  injected positions, return series and trade ledger, in `from_config`
  defaults and the funding thresholds' endpoints, in the perp advisory
  ticket's liquidation checks, and in the telemetry wall clock (the existing
  tests asserted `wall_seconds >= 0`, which a constant 0 satisfies).

### Changed

- **`web/` and `scripts/` joined the CI coverage scope.** The three test jobs
  ran `pytest --cov=yialpha`, and a command-line `--cov` replaces
  `[tool.coverage.run] source` — so the two run-in-place trees (2,772
  statements) were measured by no gate. The workflow and the config now both
  name `yialpha`, `web` and `scripts`, and `tests/test_round16_packaging_ci.py`
  fails when the two lists disagree or either tree drops out. With them every
  lane measures 99.87–99.90% of 22,287 statements.
- **Coverage floor raised from 75% to 90%.** The floor had been set when the
  baseline was 78%; by 2026-11-23 every OS × interpreter lane measured 95.5% of
  19,393 statements, so the gate would have passed a twenty-point regression.
  On the round-17 tree every lane measures 99.87–99.90% of 22,287 statements,
  so the floor is now a wide cushion; raise it once a CI run under the widened
  scope has reported. It has: every test job of the first green run (2026-11-28)
  measured 99.87% of 22,287 statements.

## [1.0.0] — 2026-11-23

### Added

- **Crypto-native fundamentals (`YIALPHA_CRYPTO_NATIVE`, default off):** pure-crypto
  instruments (BTCUSDT-style perps, crypto spot, legacy `crypto`) previously lost the
  Fundamentals Analyst entirely — no company, no statements. The new optional
  category `crypto_native` brings the analyst back with a REPLACEMENT tool set
  (market-analyst perp precedent) behind the a_share_native-style contract:
  flag off = the historical skip and every other run's tool list / prompt are
  byte-identical. Two keyless free-tier vendors via the existing SOCKS5 proxy —
  `yialpha/dataflows/coingecko_vendor.py` (`get_crypto_token_fundamentals`:
  market cap / rank, supply schedule, FDV, ATH/ATL; live snapshot, historical
  daily PIT series with the free tier's trailing ~365-day cap honestly
  degraded beyond it; `/search`-based canonical symbol resolution with
  1000-prefix fallback; browser UA required — the edge 403s default Python
  UAs) and `yialpha/dataflows/defillama_vendor.py`
  (`get_crypto_network_fundamentals`: chain TVL full history, protocol TVL for
  governance tokens with a $1M-TVL credibility floor against junk symbol
  collisions (SHIB→shibex), live-only chain ranking). Gate = flag AND
  `instrument_class ∈ {pure_crypto_perp, crypto_spot}` (unknown_perp honestly
  skips); CLI/batch analyst filtering and the runtime node skip both carry the
  bypass; the stock fundamentals bundle is skipped on the crypto branch
  (expected statement failures must not masquerade as data-quality
  degradation); `crypto_native` joined `_SIGNATURE_CONFIG_KEYS` so stale
  checkpoints don't replay pre-flag analyst reports. Pinned across
  `tests/test_crypto_native.py` (28 tests: wiring byte-equivalence, prompt
  reframe, vendor PIT windows, resolution/floor, router sentinels, tool-layer
  date clamp).

### Removed

- **Live execution layer (`yialpha/execution/`, 2026-11-22):** the whole package
  (`domain` / `gateway` / `bridge` / `binance_gateway` / `browser_broker`) plus
  its test files (`test_execution_*`, `test_browser_broker`, `test_base_gateway`,
  `test_pre_trade_risk_check`, `test_round8_ki_execution`) — it never wired into
  the LangGraph graph and had zero production callers; the repo is now
  analysis-only by construction (reports + backtests + tickets). Also removed:
  the `YIALPHA_EXECUTION_{ENABLED,MAINNET,TIMEOUT_MS,LEVERAGE}` direct-read env
  keys and their `.env.example` section, the mypy `yialpha.execution.*` override
  and CI mypy scope entry, and the `yialpha.execution` prefix in
  `p0_shadow/guard.py`'s B0 blocklist. Kept: `YIALPHA_ANALYSIS_ONLY` /
  `YIALPHA_LIVE_EXECUTION_ENABLED` (p0-shadow manifest snapshots assert them at
  their inert defaults), `YIALPHA_KILL_SWITCH` (dashboard halt banner), and
  `BINANCE_API_KEY` / `BINANCE_API_SECRET` (read-only perp-ticket data access —
  `binance_brackets` / `perp_bundle` / `risk/perp_ticket` — not order keys).
- **Docker packaging (2026-11-22):** `Dockerfile`, `docker-compose.yml`, and
  `.dockerignore` — single-user local install on Windows; docker was never
  installed, CI has no image build/publish job, and the web UI runs via
  `python web/app.py` directly. README Docker sections and the
  `.dockerignore`-content packaging test removed with them.
- **Dead code sweep (2026-11-22):** the no-op timeout-shim mechanism in
  `run_robust.py` (`YIALPHA_TIMEOUT_SHIM_DIR` + PYTHONPATH injection; the repo
  never shipped a shim and its test was already gone); the orphaned scripts
  `scripts/_pull_crypto_snapshot.py`, `scripts/smoke_binance_vision.py`, and
  `scripts/analyze_nvda.py` (plus its argv-validation regression pin); the
  self-declared `PRODUCTION-ORPHANED` `yialpha/dataflows/browser_data.py` and
  its test; and `.env.enterprise.example` (superseded by the annotated Azure
  section in `.env.example`). ~10 stale A-share wording remnants in
  docstrings/comments cleaned (incl. the KDJ catalog line that reached LLM
  prompts).
- **P0 shadow-validation harness (2026-11-22):** `scripts/p0_shadow/`
  (~2.1k lines), `scripts/p0_shadow_validation.py`, its three test files
  (~840 lines), and the stale planning docs `docs/P0_NEXT_SESSION_PLAN.md` /
  `docs/P0_SHADOW_VALIDATION_PLAN.md` — the B1/B2 shadow validation cohort
  completed in September and its checkpoints (2026-10-04 / 10-11) lapsed six
  weeks unexecuted. Kept: `docs/P0_FINAL_REVIEW.md` and the actively
  maintained `docs/P0_TIME_CONTRACT.md` (their dated log lines referencing
  the deleted harness are history, left verbatim), and the local untracked
  `analysis_output/` evidence tree (its review/closeout status is the
  operator's call).
- **A-share native data layer (`a_share_native`):** the optional category, its
  AKShare / BaoStock / Tushare / Eastmoney vendors, the 13 bound analyst tools,
  the `[a-share]` and `[a-share-tushare]` install extras, and the
  `YIALPHA_A_SHARE_NATIVE` / `YIALPHA_A_STOCK` / `YIALPHA_BAOSTOCK_RETRIES` /
  `YIALPHA_BAOSTOCK_TIMEOUT_S` / `TUSHARE_TOKEN` env keys. US / crypto runs
  were never gated on it (default off) and are unchanged.
- **Non-US market support:** the regional alpha-benchmark map, the commodity /
  forex / index-CDN symbol aliases, and the non-US exchange-suffix resolution
  (`.SS` / `.SZ` / `.HK` / `.T` / `.L` / `.NS` / `.TO` / `.AX`) — only
  US-listed equities/ETFs and crypto remain analyzable.
- **NEW hard input gate:** non-US / non-crypto tickers are rejected up front
  with `UnsupportedMarketError` (`yialpha.dataflows.symbol_utils`) at every
  entrance — CLI, batch, web, and `YiAlphaGraph.propagate()` — instead of
  silently resolving to the wrong instrument.

### Fixed

- **NTFS future-mtime served a just-written cache as fresh under
  ``ttl_days=0`` (2026-11-22):** NTFS stores mtimes in 100ns units and the
  write path can round them UP past the immediately following ``time.time()``
  reading, so the freshness check ``now - mtime < ttl * 86_400`` saw a
  NEGATIVE age — with ``ttl_days=0`` ("always refetch") that inequality is
  trivially true and the stale bytes were returned instead of fetching
  (Windows-only, timing-dependent; surfaced as the long-flaky
  ``test_disk_cache_persistent_denial_still_best_effort``). Age is now
  clamped at zero via ``_age_seconds`` in ``dataflows/disk_cache.py``, used
  by both the freshness gate and the stale-serve age report.
- **Round-11 adversarial sweep (2026-09-30: the six blind-spot modules rescanned
  in four read-only shards — dataflows perp chain / dataflows stock+vendor /
  agents+cli / backtest+llm_clients+web — plus the round-10 recorded leftovers;
  1 P1 + 6 P2 + 15 P3 confirmed findings, every one landed with a regression
  pin).**
  - **P1 — weekly resample dropped the last COMPLETED Friday bin on live
    Tue–Thu/Sat runs** (`ohlcv_resample.py`): the still-forming guard asked
    "has today's bar landed" — a question about the NEXT bin (already removed
    by the cutoff filter) — so every live analysis dated Tue–Thu (crypto
    always has today's bar) deleted the closed trailing week and
    weekly pivots / weekly indicators read one week stale, the exact
    "Wednesday analysis returned week-before-last" failure the round-5 fix
    targeted. Criterion is now "the bin's own range extends into today"
    (W-FRI label L covers (L−6, L]; today > L ⇒ closed), with clock-frozen
    deterministic pins for all five calendar cases; round-5's Saturday skip
    stays (it guards a different pathological future-date fixture).
  - **P2 — perp bundle single-leg 60s timeout labeled legal slowness as a
    core failure**: with `YIALPHA_BINANCE_PROACTIVE_BACKOFF` (each acquire may
    sleep ~60s) or `BINANCE_HTTP_RETRIES≥3` backoff stacking, a slow-but-
    succeeding prices leg tripped `future.result(timeout=60)` → core sentinel
    → NO_TRADE. New `YIALPHA_PERP_BUNDLE_LEG_TIMEOUT_S` (default 300); purely
    a labeling threshold — the pool join dominates wall time either way.
  - **P2 — `positioning_split` was an end-to-end dead switch**: the filter
    appended POSITIONING but the interactive selection funnel re-normalized
    over the fixed 4-key order, the batch union intersected over the same
    4-key base, and the live panel/web cards didn't know the section — the
    flag-on perp run silently ran without the analyst. New
    `resolve_run_analysts` re-applies the single-source filter post-selection,
    batch unions filter-appended extras, and the panel (including a latent
    section-title KeyError) + web report cards render `positioning_report`.
  - **P2 — Form 4 PIT leak** (`sec_ownership.py`): a missing/malformed
    `filingDate` skipped BOTH the upper (PIT) and lower (lookback) bounds and
    rendered the filing's trades unfiltered — the same failure class the 13F
    `if fd:` fix closed; Form 4 now drops those filings fail-closed.
  - **P2 — baostock "not installed" entered the transient retry ring**: the
    deterministic import error was raised inside the retried session
    constructor, costing 3 attempts + 6s backoff per call under run_robust's
    default retries; hoisted to the five public entry points (retry docstring
    is now true again).
  - **P2 — akshare's process-global proxy-env pop leaked into concurrent
    vendor requests**: `_direct_connect` popped `HTTP(S)_PROXY` from
    `os.environ` under a lock only its own callers held, so parallel
    proxied requests (FRED/tavily/SEC/AV in the same ToolNode batch) went
    direct during the window. Replaced with a thread-scoped bypass — a
    lazily-installed `Session.merge_environment_settings` patch honoring a
    thread-local depth flag (flag 0 = original method, byte-identical).
  - **P2 — `/api/accuracy` returned 500 on NaN-bearing artifacts**: a NaN
    forward close produced NaN returns that `json.dumps` wrote as bare
    literals and starlette's `allow_nan=False` render then refused. Three
    layers: non-finite forward endpoints are now `unscorable` (excluded from
    hit/miss and disclosed via `excluded_unscorable`, distinct from pending),
    the writer sanitizes non-finite floats and refuses NaN literals, and the
    store strips non-finite values from pre-existing legacy artifacts.
  - **P3 batch**: cross-day basis pairing in perp bundle legs now aligns on
    the common tail date and degrades honestly (regime round-10 precedent);
    OI percentile denominator excludes the latest value itself
    (REGIME_VERSION v4→v5 — regime imports the bundle's `_fetch_open_interest`,
    single source, so the percentile scale enters the id preimage);
    endTime row-level trim on OI/LSR/taker (belt-and-suspenders vs
    endTime-ignoring vendors/mocks); defillama catalog entries without a slug
    no longer crash the cache key; sec_edgar + AV three-table entries gained
    the round-8 `current_pit_end` clamp yfinance already had; yfinance news
    soft-misses no longer log ERROR-with-traceback and the global-news window
    filter runs before the `limit` quota so out-of-window first queries can't
    starve later ones; tavily budget gate reads+charges atomically under its
    lock; `"default"` vendor-chain fallback now orders by the documented
    default (yfinance first, AV last) instead of dict insertion order;
    netretry conditionally retries 5xx `HTTPError`s (never 429/418/<500);
    backtest funding-drag denominator and coverage check bounded to the
    actual charge domain `[first, mark_end)` instead of the fetch window;
    `/api/analyze` validates dates with the same strftime round-trip as
    run_robust (non-padded input → 400 at submit, not a doomed 200);
    sentiment/news lookback helpers normalize `trade_date[:10]` (round-10
    as-of dual-form lesson) before `strptime`; empty `ic_data` under
    `YIALPHA_INDICATOR_IC_CONTEXT` now warns fail-visibly instead of silently
    dropping the advisory.
  - **Telemetry/watchdog (user-approved known leftovers)**: `ToolNode` now
    wraps with a Runnable-aware telemetry wrapper (`.invoke(state, config)`
    forwarding; parameter named `config` so LangGraph passes the real run
    config) — serial-leg tool HTTP walls finally count, closing the A/B
    speedup gate's systematic undercount of the 2.5× criterion; run_robust's
    `_states_root` resolves an `.env`-only `YIALPHA_RESULTS_DIR` and pins it
    into the child env (parent reads where the child writes, by
    construction); `_kill_tree` returns landed-vs-already-gone (taskkill 128 /
    ESRCH) so a natural crash in the poll→kill window reports `exit N`, not
    "killed".
  - **Style consolidations (byte-equivalent except where noted)**: the 9
    script entry bootstrap blocks unified behind `scripts/_bootstrap.py`
    (rank_signals' bare reconfigure gained the defensive form — the only
    behavior hardening); new `atomic_write_text` helper collects the four
    fixed-name tmp-write sites (cli scoreboard gains the Windows
    `replace_retry_on_windows` retry, matching every other publish seam).
  - **Adversarial self-review remediation (ten-angle forked review of the
    round's own diff; every confirmed finding landed with a pin).** The
    review caught three regressions the round's own fixes had introduced:
    the new weekly-bin calendar guard dropped round-8's truncated-cache
    clause (a Saturday run with a stale cache missing the Friday bar served
    a truncated week as complete — both clauses now coexist); the akshare
    thread-scoped proxy bypass silently no-oped for akshare's curl_cffi
    transport (libcurl reads os.environ, never requests' merge hook — the
    bypass is now a proxies-only dict override plus a curl_cffi-conditional
    env-pop window, with a Windows env-key-folding restore bug found and
    fixed along the way); and the hoisted baostock import guard broke
    cache-hit-without-package replays (the guard now sits after the cache
    lookup, before the retry ring). Also landed: duplicate-date defense on
    the new same-day basis pairing (`duplicated(keep="last")`, no more
    `float(Series)` → core sentinel); yfinance global-news `[:10]`
    normalization + restored fault logging + titleless-flat exclusion +
    dead defensive pass removal; `YIALPHA_PERP_BUNDLE_LEG_TIMEOUT_S=0` now
    means explicit off (repo convention) instead of garbage; Form 4 / FTD /
    13F and BaoStock statements gained the `current_pit_end` clamp;
    run_robust anchors relative `.env`-derived results dirs to the project
    root (cwd-independent parent/child agreement); the telemetry wrapper's
    `config` parameter is typed the way langgraph's signature inspection
    actually matches (`Optional[RunnableConfig]` — verified warning-free on
    a real compile); the accuracy/store non-finite dedup collapsed into
    `json.loads(parse_constant=…)` (one pass, zero copies); a shared
    `is_canonical_ymd` validator now backs both run_robust and the web
    submit endpoint; and run_baseline's graph builder routes through the
    single-source analyst filter so flag-on perp backtests include the
    Positioning analyst (the A/B gate already did via
    `batch_selected_analysts`; analyze_window/analyze_nvda are stock-only
    scripts where the filter is the identity).
  - **Docs**: `YIALPHA_SEC_13F_PUB_LAG_DAYS` documented default corrected
    5 → 45 (code was already 45 both places; following the doc would have
    introduced a 40-day lookahead).

- **Round-10 adversarial sweep (2026-09-29, four read-only shards — scripts /
  graph / ledger+regime+instruments / test-gap map — followed by a nine-angle
  adversarial self-review; every confirmed finding and all seven user-approved
  known issues landed, each with a regression pin).**
  - **Shard fixes (the round's original landing):** A/B-gate parallel-leg
    speedup criterion un-deadened (reads the `Analyst Fanout` node wall) and
    its fallback analyst list completed; run_robust kill-race evaluation,
    kill-reason propagation and parameter guards; DEGRADED evidence anchored
    to the child's actual results dir; prune CLI degenerate-arg and
    empty-evidence refusals; export dedup + utf-8 guards; script bootstrap
    hardening; graph-side memory-write fail-soft, the new
    `replace_retry_on_windows` helper for the Windows lockless-reader
    `os.replace` race (memory log ×2 / states log / node perf), finalize
    three-seam fail-soft, fanout `max(1, ·)` clamp, reasoning-token
    double-count fix; ledger-side `ledger-unreadable` labeling, rowid-DESC
    tie-breaks, sub-second EOD, regime_id key guard, empty-as-of evidence
    survival, optional-read degradation for table-less ledgers; listing_age
    anchored to `analysis_as_of`; compact-ISO `available_at` rejection. Test
    gaps closed: the Alpha-Vantage stock path (13 pins) and `fmt` (8 pins)
    are executed by the suite for the first time.
  - **Self-review P2s (nine parallel finder angles over the working diff,
    each verified in code — several empirically — before fixing):** live-run
    full-ISO `analysis_as_of` broke the `%Y-%m-%d` listing-age parse on EVERY
    live stock-perp run — the swallowed ValueError silently nulled
    `listing_age_days`, injected a phantom `onboard_date` missing input into
    the regime id, and split live-vs-replay ids for the same day — fixed by
    date-prefix normalization at the boundary; the three pasted ledger-read
    guards caught only `sqlite3.OperationalError`, but a corrupt/foreign
    ledger file raises the PARENT `sqlite3.DatabaseError` (and two
    `get_connection` calls sat outside the try) — fixed at the chokepoint
    with the new `optional_read_rows` helper in `ledger/sqlite.py`, adopted
    by ten optional readers (three pasted guards deleted, seven missed
    siblings covered); the streamed finish line was unguarded
    (`finalize_streamed_run` / the CLI decision store could kill a
    fully-billed run on an IO hiccup) — fail-soft moved inside `_log_state`
    itself plus a matching CLI-side guard.
  - **Self-review P3s:** truthy-but-unparseable evidence anchors no longer
    drop the run's whole evidence chain (unanchored + WARNING); a blank
    as-of no longer aborts the memory-resolve sweep (per-row `timing_error`
    nomination); `--date` strict zero-padded ISO validation and a
    single-source `_states_root` in run_robust; the A/B serial leg now sums
    the full agent+tool+clear cluster the fanout replaces (was agents-only,
    biasing the 2.5× criterion down) with names derived from
    `ANALYST_NODE_SPECS` — the hand-maintained fallback list that had
    already drifted once is gone; honest residual: serial ToolNode walls are
    still not recorded by telemetry (documented setup behavior), so the
    tool-HTTP share counts zero on the serial surface for now;
    `replace_retry_on_windows` wired into nine more publish seams
    (complete_report/decision, the OHLCV cache, vision manifest/partitions,
    dashboard.html, config snapshots, disk cache, LLM response cache, the
    backtest DecisionCache) with escalating backoff (~0.8s budget) and an
    `-O`-safe exhaustion edge; per-field token reconciliation restores
    input/output when only reasoning rides `usage_metadata`; the web
    positions view reports `available: false` on a degraded read instead of
    a phantom empty book; prune CLI diagnoses all-NaN forward columns
    distinctly from header-only files; fanout workers now record per-analyst
    walls (parallel `--profile` no longer shows 0.0s) and the stream
    wall-time tracker is wired through GraphSetup via a per-invoke provider;
    the debate-stage prompt builders (bull/bear + three risk debators) got
    their first direct pins (23 tests, no production change); prune-CLI
    guard tests run in-process (three interpreter spawns, ~3s, removed from
    every full-suite run); `corporate_actions` derives its available_at
    datetime contract from `_STRICT_ISO_DATE_RE` so the two cannot drift.
  - **User-approved known issues landed:** `REGIME_VERSION` v3→v4 in one
    bump (f4fecf8 precedent — no migration, scoreboard lists versions side
    by side) carrying the frozen-vocabulary re-scope (`overnight_gap_bps` is
    None outside stock_perp — structural absence, not a data gap) and the
    basis tail-date alignment (spot-perp and last-index legs require a bar
    at the common tail date; a misaligned tail degrades to the existing
    missing-input path instead of computing a cross-date ratio that fed the
    id preimage and the ±50 bps stress trigger); MACRO reference-path
    outcomes now `skipped` without writing a row (docstring- and
    legacy-consistent); enforced mode fails closed when no layer certifies
    the book (label space enumerated: plain `ledger-unreadable` and
    unreadable-with-invalid/empty overlays veto, a valid operator overlay
    proceeds with disclosure, shadow stays disclose-only); the three
    "silent serial fallback" doc promises were rewritten to the clamp
    semantics the code actually implements.
- **Round-9 USDT-perp adversarial sweep (2026-09-29, five read-only shards +
  runtime probes + coverage-gap map — 4 P2 and 9 P3 landed, each with a
  regression pin; the rest accepted as documented below).**
  - **Self-review of the landing diff (P2 × 1, P3 × 5 — all processed):**
    the P2 closed a hole in the new vision-store remediation itself — the
    load error's "delete it" advice led to a state where the manifest kept
    claiming days whose partition was gone and `pending` never re-opened
    them (empirically demonstrated by the reviewer); sync now re-opens
    claims for any year whose partition is missing at lock time, making
    delete-then-resync the complete remediation (pinned end-to-end, zero
    network cost via the byte cache), and the report's `already_synced` is
    recomputed after same-call drops so it never counts coverage the call
    itself removed. P3s landed: the ATR regex boundary uses
    `(?<![A-Za-z])` so Chinese-adjacent "当前ATR：$7.29" still parses (a
    bare `\b` killed it — `\w` includes CJK) while NATR stays excluded; the
    blocked-ticket render attributes the refusal to its actual cause
    (normal-regime blocked no longer claims the drawdown breaker fired);
    the `_futures_data_window` boundary date is refused consistently all
    day (was a clear refusal for part of the day, a muddy head-clamped
    empty serve for the rest); one vacuous pin made discriminating
    (`pd.Timestamp` parses slash-form dates — "not-a-date" doesn't).
  - **Test hermeticity (P2):** a dev-box `.env` with
    `YIALPHA_CRYPTO_NATIVE=true` is baked into `DEFAULT_CONFIG` at import, so
    `reset_config()` restored the *dev* defaults — 15 tests pinning the
    flag-off routing contract (crypto runs drop the fundamentals analyst)
    went red locally while CI stayed green, and `test_crypto_native.py`'s
    own `set_config(True)` never restored. New autouse fixture
    `_optional_crypto_categories_off` pins both crypto optional categories to
    their production defaults per test (`tests/conftest.py`); opt-in tests
    still win via `set_config` merge. Pre-fix full suite: 15 failed / 3838
    passed; post-fix 3853 / 0.
  - **Derivatives-stress OI/LSR retention anchor (P2, S2):**
    `derivatives_stress_series` clamped its `/futures/data` head to 30 days
    back from the PAST `end_ms`, but the server rejects a `startTime` older
    than the ABSOLUTE now−30d horizon (HTTP 400 −1130 — the same probe-backed
    semantics `perp_bundle._futures_data_start_ms` recorded): every analysis
    date 1–3 days back silently lost the OI and global-LSR stress legs.
    Head now anchors on now (the server's own horizon), matching the proven
    sibling; the existing pin (which had pinned the buggy anchor) is updated
    to the corrected contract with a frozen clock.
  - **Vision store manifest/partition desync (P2, S3):** an unreadable year
    partition (Windows AV lock, external truncation) could desync the store
    two ways — a rebuild dropped partition rows while `manifest["days"]`
    kept claiming them (silently truncated deep history, QA blind because it
    is manifest-driven), and `load` skipped the year with a warning while
    coverage accounting still counted its days (and never repaired: pending
    was empty). Now a rebuild that finds the partition unreadable RE-OPENS
    that year's old manifest days (keeping this call's freshly merged days;
    re-opened days re-download from the byte-level zip cache with zero
    network cost — pinned end-to-end), `load` fails closed with a typed error
    naming the partition, and EOFError (byte-truncated gzip, which previously
    bricked every later sync against the same dead read) joins the repair
    path.
  - **Indicator-tool empty-`curr_date` escape (P2, S4):**
    `binance_indicator_tools` was missed in round-8's 10-file clamp sweep —
    `pd.Timestamp("")` is NaT and its strftime raised a raw ValueError BEFORE
    the try, escaping the tool with no typed degrade and no ledger event
    (empirically reproduced). Same fallback idiom as the 10 siblings, plus
    the window computation moved inside the try so a degenerate live-mode
    date still degrades typed.
  - **Trade-ticket blocked gate (P2, S4):** the breaker gate keyed only on
    `Drawdown Regime`, but `action=blocked` has two causes that render a
    NORMAL regime (sector cap reached, invalid equity) — a report the
    framework refused to deploy still got a full leveraged entry ticket, and
    `rank_signals` didn't even flag it ⚠️熔断. Both scripts now honor
    `Action: blocked` whatever the regime line says.
  - **P3 batch:** the OI live-snapshot leg catches raw transport errors under
    `retries=0` (one blip no longer discards the already-fetched history
    through the router's generic catch); `_futures_data_window`'s retention
    check compares calendar dates (a window ending exactly 30 days back was
    refused for most of every UTC day); `find_report` discovery skips
    crashed/incomplete runs (`complete_report.md` parity with
    `rank_signals.latest_for_ticker` — a newer broken run no longer shadows
    the last complete report); rank tie-breaks fall to ticker order
    (was filesystem enumeration order); the ATR regexes carry a `\b` left
    boundary ("NATR: 12" no longer feeds 12 in as the ATR); a stop within
    1bp of entry is refused as degenerate (was a ~1e11-USDT notional ticket);
    `liquidation_price` renders only for `crypto_perp` (the Binance USDT-M
    MMR estimate on a margined-equity ticket was a misleading advisory row);
    fear-greed PIT selection sorts by timestamp instead of trusting the
    API's newest-first ordering; the CoinGecko/DefiLlama parse memos never
    memo-hit on the mtime −1.0 sentinel (a persistently failing cache write
    froze the first parse for the process lifetime); CLAUDE.md's stale
    "crypto→SPY proxy" regime-benchmark wording corrected (pure-crypto
    families use the BTCUSDT native benchmark, disclosed in the rendered
    line). Ruff clean restored (16 lint findings the two feature commits had
    introduced, incl. two in this batch).
  - **Coverage-gap map (S5) → new pins:** the perp path is well covered
    overall (perp_bundle the best-pinned module in the repo); the gaps were
    plumbing seams. New `tests/test_binance_perp_tools.py` pins all 10
    `@tool` wrappers' exact vendor-name + positional/keyword mapping and the
    klines `_qualifier` threading (a silently swapped positional served the
    wrong window's crowding data with no error anywhere — only tool NAMES
    were pinned before). Remaining mapped gaps (vision `_fetch` status
    mapping 404-vs-429, accuracy crypto cutoff/venue replay, intraday
    klines interval threading, onchain PIT boundary, direct
    `crypto_native_fundamentals_run` matrix) are recorded below for the next
    pass.
  - **Accepted as documented (no code change):** the live indicator battery
    computing on the still-forming bar (deliberate freshness for the analyst;
    the risk overlay's completed-bar basis is the deterministic seam);
    `spot_perp_basis` structurally degrading for multiplier-scaled pairs
    (dual-leg canonical resolution is a feature, the degradation is an
    honest sentinel); the funding-cadence `fundingInfo` cross-check request
    (load-bearing for the cadence-change disclosure, not waste); the
    vision-sync lock spanning downloads and the per-sync full-year partition
    rewrite (perf, correctness unaffected); unbounded vision cache growth
    (deliberate immutability); OI bundle `limit:30` zero headroom (safe only
    under the current head-clamp invariant — noted fragile in code review);
    bundle text sliding across UTC day boundaries (LLM-cache cost on
    midnight-crossing retries, disclosed span); CoinGecko wrapper-symbol
    preemption and quote-suffix coverage (USDT/USDC only — external catalog
    state, honest sentinels); DefiLlama historical titles carrying today's
    catalog identity (pinned intended-with-caveat); integer protocol slugs
    (per-leg fail-soft); fear-greed applicability for `unknown_perp`
    (advisory noise, one line); rating-less ticket strength disclosure and
    the standalone bracket cap being operator-optional.

- **crypto_native adversarial review batch (2026-09-29, /code-review high —
  14 findings landed, 5 accepted as documented):** two PIT leaks (the DefiLlama
  protocol title rendered the catalog's TODAY TVL into historical runs — now
  omitted with an explicit version-selection caveat; the ETF fund-framing nudge
  could stack on the crypto persona for legacy-crypto symbols whose base
  collides with the ETF set — now suppressed on the crypto branch); three
  crash/coverage bugs (space-named chains like 'zkSync Era' broke the cache
  filename allowlist before any fetch — now folded to a safe key with per-leg
  fail-soft so one broken leg no longer discards the other's rendered
  evidence; malformed LLM-echoed dates like "2026/08/01" crashed
  date.fromisoformat — now degrade in-band; the 1000-prefix guard missed
  7-char bases (1000CAT) and mangled the 1000000x family (1000000MOG →
  000MOG) — one shared multiplier-prefix rule now serves both vendors, and
  legacy dashed symbols (BTC-USD) reduce to their base instead of the
  unresolvable BTCUSD); two rendering bugs (a dead `or` made the price-change
  n/a fallback unreachable; sub-cent meme-coin prices collapsed to 0.00 —
  both now render honestly); and a drift-collapse refactor (flag-aware
  `crypto_native_fundamentals_run` predicate in `graph/routing.py` shared by
  the node gate and the CLI/batch selection layer; mtime-keyed parse memos so
  the LLM tool loop stops re-parsing ~1MB cached JSON per round; unused
  dev/community params dropped from the /coins snapshot; data-driven
  base→chain resolution from the /v2/chains catalog's tokenSymbol so newly
  listed chains auto-extend coverage; shared Markdown-table persona constant;
  test-hygiene: the zero-interval throttle leak). Accepted as documented:
  per-module `_pit_curr_date` / RecordingLLM copies (established repo idiom),
  the stock-bundle skip at the call site (single caller today), protocol
  version selection by today's catalog (disclosed in-band — as-of selection
  would need every candidate's full history), and the pre-existing router
  ValueError on a category-level data_vendors pin naming a vendor a method
  does not have (a_share_native behaves identically).

- **Round-8 known-issues batch landing (2026-09-28, the full audit ledger
  from both waves — every item code-verified, landed at minimum blast
  radius, and pinned across `tests/test_round8_ki_{dataflows,ledger_ab,
  execution,backtest_misc}.py`; adversarial self-review of the diff closed
  before commit).**
  - **Self-review of the landing diff (P1 × 0, P2 × 1, P3 × 7 — all
    processed):** the one P2 closed a systemic PIT hole the new gates
    inherited — `_pit_curr_date`'s clamp passed empty/omitted `curr_date`
    through unchanged, so "or empty for live mode" disarmed the historical
    gates in a pinned backtest; every tool-layer clamp (10 files) now falls
    back to the run's pinned analysis date on empty input. P3s landed:
    the mark_end extension anchors on `execution_lag_bars` (not a
    hardcoded 1), Form 4 discloses budget-capped archive coverage, the
    direction clamp covers -ing forms and no longer strips descriptive
    "long entry" narrative, redaction exact-matches the Binance/Tushare
    credentials, and two test-quality items. Accepted-as-documented: the
    same-day margin page-boundary row gap (self-heals within the TTL) and
    the "rather Sell than Buy" preference shape (heuristic cost, noted in
    code).
  - **Wave-1 vendor PIT set (5 items):** the A-share sector-flow ranking is
    a live-only snapshot and now refuses historical dates with a
    `SECTOR_FLOW_UNAVAILABLE` sentinel (the old caveat note still served
    today's ranking into backtests); Eastmoney margin paginates (400-row
    pages, hard cap 5, per-page cache) so deep look-back windows no longer
    silently lose their older tail; SEC Form 4 pulls the `files[]` archive
    segments when the submissions `recent` head is at its ~1000 cap (the
    honest cap note now fires only when backfill cannot complete); AV
    `INSIDER_TRANSACTIONS` refuses historical as-of dates (mirroring the
    yfinance-side gate); the `get_insider_transactions` tool schema carries
    and PIT-clamps `curr_date` so a future wiring cannot leak today's
    snapshot.
  - **S3 — agents:** parallel analyst fan-out workers each bind a FRESH
    prediction-capture registry before `subgraph.invoke`, killing the
    `__active__` single-slot race that attributed every parallel
    submission to the last writer (serial path untouched); `parse_rating`
    skips rating words preceded by a nearby negator ("argues against Sell"
    no longer parses as Sell); the Positioning Analyst's free-text
    fallback gets a deterministic direction clamp (line-level removal +
    banner) so the direction freeze no longer rests on prompt compliance
    alone.
  - **S5 — backtest:** the A/B gate script excludes propagate-failed runs
    from every distributional statistic (a crashed run's `"?"` rating and
    empty reports used to skew chi-square/TF-IDF and silently flip the
    speedup criterion to N/A), fails a leg that lost all its runs, and
    gains an S3-a race probe criterion (parallel predictions must attribute
    to 2+ distinct analysts when recorded); `DecisionCache` v5 folds a
    decision-config signature (graph `_run_signature` + provider keys)
    into the key so a model/analyst/vendor change never replays another
    configuration's decision (v4/v2 layouts swept on miss); the event-study
    docstrings now state the execution-bar anchor and the offset-0
    fill-bar convention (note-level, no numeric change).
  - **Wave-2 leftover — the final episode's semantic window:** the equity
    marking window starts at the legacy `holding_days + 10` calendar-day
    boundary and EXTENDS (never shortens) to the last decision's execution
    bar + `holding_days` sessions when holidays push that span past the
    legacy boundary — the final episode's `position_return` now covers the
    same window as `raw_return` instead of being force-marked mid-window.
  - **S6 — graph/ledger:** `positioning_report` persists (presence-gated
    log key + `1_analysts/positioning.md`); `instrument_registry` joins
    `_SIGNATURE_CONFIG_KEYS` (flag-on routing re-binds perp tools — a
    resumed checkpoint under the other flag would replay stale reports);
    capture and scoring share ONE `reference_seam_equity` verdict (an
    `unknown_perp` underlying capture used to store the equity source and
    retire as a permanent `reference_source_mismatch`), with the perp
    class frozenset and source labels de-duplicated across
    time_contract / outcome_compute / engine.
  - **S7 — transport/cli:** credential redaction exact-matches the Zhipu
    and Tavily key pools (comma-split env vars) and adds the
    `tvly-<base62>` / `<32-hex>.<alnum>` family shapes ahead of the generic
    16-char shape (a uuid4 hex never matches); dead `cli/utils.py`
    `get_analysis_date` removed (the live `cli.main` twin has the
    future-date guard).
  - **S8 — execution layer (project-grade landing of the 12 audit items,
    default-off, still unwired from the graph):** connection-level submit
    errors (Connection/Proxy/SSL/Chunked family) join the transient set
    and take the query-recovery path; the recovery probe retries
    `-2013` a bounded 3×0.3s before concluding absence (matching-engine
    visibility lag used to turn an accepted order into a false REJECTED);
    hedge-mode closes keep the natural side and set `position_side` to the
    position being closed (the old side-flip turned a drawdown hard-stop
    into a reverse-side OPEN — doubling exposure instead of unwinding);
    the network-edge final guard re-reads BOTH switches and the leverage
    POST moved after it (a rejected order no longer leaves a side-effect
    request); the idempotency id is always gateway-minted (`reference` is
    a label — a bare value could collide with a historical order and
    poison recovery); `YIALPHA_EXECUTION_TIMEOUT_MS` <= 0 clamps to the
    default; the proxy reader accepts lowercase env spellings and warns on
    port-less URLs; unknown Binance statuses map to `NOTTRADED` with a
    warning (+ `EXPIRED_IN_MATCH`); `CancelRequest` gains an optional
    `client_order_id` channel and cancels reconcile via a best-effort GET
    (log-only); `connect()` re-entry closes the previous SDK client.
    KI-12 (client-id collision) audited as already-mitigated (process
    nonce + monotonic tick; math documented in-code).

- **Adversarial bug-hunt round 8 (2026-09-28, two waves; eight review shards
  — S1 dataflows core / S2 vendors (wave 1, commit 347a01a), S3 agents tree /
  S4 risk math / S5 backtest+batch / S6 graph+ledger / S7 cli+scripts+transport
  / S8 execution (audit-only, known-issues ledger, no code); every fix
  code-verified before landing and pinned in
  `tests/test_round8_regressions.py` (12 pins, wave 1) and
  `tests/test_round8_wave2_regressions.py` (26 pins, wave 2); an adversarial
  self-review of the wave-2 diff found 1 P1 + 2 P2 + 5 P3 self-introduced
  defects — all fixed and pinned; full suite 3700 passed).**
  - **Wave 1 (S1+S2, 2 P1 + 2 P2 + 8 P3):** fundamentals/vendor-tool PIT
    clamp coverage across fundamental / A-share / eastmoney / macro /
    SEC-ownership tools; `_as_of_date` fails CLOSED on malformed `curr_date`
    (used to widen the PIT upper bound to today = lookahead); `_latest`
    duration disambiguation; FTD look-back hard cap 730d; `load_ohlcv`
    empty-frame fail-closed + atomic write; weekly-bar guard refinement;
    `atr_stop` forming-bar guard sees the `Date` COLUMN (Range-index frames
    parsed row numbers as 1970 ns stamps, so the guard never fired for
    equities); insider-filing dormancy gate.
  - **Wave 2, P2 — `run_baseline --full` folded baseline + risk-improved
    legs (and multiple tickers) into ONE multi-run distribution table**: two
    results sharing a `run_tag` are replay legs of one realization, not two
    independent draws — `write_report` renders the distribution section only
    when one ticker + all-distinct tags, logging the skip otherwise.
  - **Wave 2, P2 — the fundamentals system message rendered as a Python
    tuple repr**: a trailing comma made the whole instruction block
    (grounding rules + every nudge) render inside `("...",)` on EVERY
    fundamentals run. Now a plain string — a deliberate prompt-bugfix, NOT
    byte-equivalent with the polluted rendering it replaces.
  - **Wave 2, P2 — CVaR tail count had float dust**: `1-0.95` is
    `0.05000…044` in binary, so `ceil(n * that)` counted one extra MILD
    observation on every 20-multiple window at 95% — CVaR biased less
    negative, the −5% breach and its 0.5× de-risk multiplier harder to
    fire. Tail count is epsilon-rounded before the ceil.
  - **Wave 2, P2 — checkpoint run-signature missed three input-affecting
    flags**: `web_search_enabled` / `a_stock` / `regime_context` all change
    node inputs (tool bindings, prompt context) — flipping one across a
    resumed checkpoint replayed the old configuration's analyst reports.
    Added to `_SIGNATURE_CONFIG_KEYS` (existing checkpoints honestly
    invalidate, same as the #1089 precedent).
  - **Wave 2, P2 — legacy outcome legs never retired**: the legacy
    `_compute_one` funding and underlying-diagnostic legs (unlike their
    reference-path twins) did not wrap vendor failures — a delisted
    contract's empty funding window raised into the per-row catcher as
    "failed" WITHOUT retirement, so every later batch re-fetched the same
    dead window forever. Both legs degrade (`LEG_FUNDING` / `LEG_UNDERLYING`
    missing) and the row retires.
  - **Wave 2, P2 — the accuracy scan scored stock decisions on the day's
    still-forming bar**: the crypto leg drops its forming bar via
    `closed_as_of`; the stock leg had no twin, so a boundary-day decision
    run intraday scored a hit/miss label that could flip by the close (and
    memory resolution permanently froze it as the reflection input). Both
    seams (`_dated_close`, `fetch_returns_yf`) now exclude the current
    US-session bar — anchored on the US equity calendar, not the host's
    local date (UTC+8 hosts had a ~5h hole) — and the same rescue applies to
    the memory-resolution cutoff.
  - **Wave 2, P2 — prediction settlement cross-talk**: settling one
    analyst's capture flushed EVERY analyst's in-flight capture for the run,
    marking siblings `submitted` — their later re-files were rejected ("already
    been settled") and the rows silently lost. `flush_predictions` scopes to
    the settling analyst (byte-identical in serial; the parallel `__active__`
    routing race is recorded as a known-issue tied to the A/B gate
    checklist).
  - **Wave 2, P3 selections** (each pinned): `DecisionCache` schema v4
    carries `was_degraded` so replays count forced-Holds as degraded exactly
    like the live run (and dirless-but-enabled caches miss instead of
    raising); event-study bootstrap CI is seeded (replay determinism);
    forward-window buffer converts trading→calendar days with the
    equity/B&H marking window capped at the legacy boundary (forward windows
    complete; metrics byte-identical — pinned); long→short flips split into
    two episodes instead of one merged win/loss sample; `batch.locks`
    releases the process lock when the OS lock fails (was a permanent
    in-process deadlock); `kelly_sizing(return_raw=True)` +
    `RiskDecision.kelly_preclip` make the band clip / floor lift visible;
    the sector cap blocks (instead of bypassing) on invalid equity ticks;
    `DrawdownBreaker.check_exposure` rejects NaN/negative sector values
    (was the only fail-open leg); `atr_stop` returns `(close, None)` — not a
    single-bar range masquerading as ATR, and not raising away the valid
    close — with the stop-computing wrapper still refusing; perp tickets
    honour direction-correct caller stops on SHORT (were silently rebuilt);
    the tradeability gate refuses target/reference ratios outside
    [0.05, 20×] as magnitude garbage (evaluate_tradeability degrades to
    UNEVALUATED where it used to `assert`); corporate-action `event_date`
    is a strict padded `YYYY-MM-DD` (compact AND non-padded forms both
    mis-sort lexicographically); regime's all-inputs-missing guard is
    reachable for stock_perp (a constant `session_state` used to defeat
    it); `PortfolioDecision` rescues case (field-scoped vocabularies —
    cross-vocab wording degrades the field to None) and percent-form
    `expected_return` ("+4.7%" → 0.047); `clear_all_checkpoints` removes
    WAL/SHM sidecars; memory resolution skips (loudly) hand-corrupted
    pending rows lacking a ticker instead of KeyError-ing before any graph
    node runs; config-check flags unknown providers (a typo'd
    `YIALPHA_LLM_PROVIDER` used to get a fake green "no API key required"
    light); `run_robust` clears the DEGRADED marker on a clean retry;
    rank_signals renders missing conviction as "—" (both tables) and
    footnotes the asset column's heuristic; the analyst-parallel A/B gate's
    overlap rescue is small-sample-only (it used to rescue real statistical
    rejections at n=20/leg).
  - **Wave-2 self-review fixes (found by the adversarial re-review of this
    very diff):** the widened fetch buffer originally lengthened the equity
    curve itself (Sharpe/alpha drift on every run — now capped at the legacy
    marking boundary and pinned byte-identical); the direction-rescue
    validator originally matched a cross-field vocabulary union ("Long" on a
    direction field still vetoed the decision — now field-scoped); the
    forming-bar exclusion originally anchored the host's local date (now the
    US equity calendar); strict-ISO validation originally admitted
    non-padded dates via strptime (now regex-shaped first); and the
    atr_stop guard originally threw the valid close away (now returns it
    with `atr=None`).

- **Perp bug-fix sweep round 7 (2026-09-23, USDT-perp focus; five parallel
  review shards — binance+vision core / p0-shadow capture / backtest perp
  economics / risk micro-layer / analysts+tool-binding — plus a main-line
  sweep of the perp bundle, regime and instrument registry; every fix
  verified against the code and pinned in `tests/test_round7_regressions.py`
  (20 pins) or updated dedicated files; full suite 3660 passed / ruff
  clean).**
  - **P2 — regime funding window spanned 8 days** (the THIRD occurrence of
    the off-by-one rounds 1–2 fixed in `_fetch_funding` and
    `_trailing_funding_total`): `compute_regime_state` asked the funding
    seam for `[D-7, D]` = 24 settlements at the 8h cadence, overstating
    `funding_pct` — a frozen RegimeState field feeding the `regime_id` hash
    and the `|funding| >= 1%` stress trigger — by a fixed ~14% on the
    default-ON `regime_state` path.
  - **P2 — the perp bundle's direct `/futures/data` legs 400'd on EVERY
    historical replay**: live probe 2026-09-23 confirmed the OI window
    (`start = D-30`) is always older than the 30-day retention horizon for
    any past D (HTTP 400 -1130), while the tool layer's
    `_futures_data_window` clamp serves the retained days for the same
    date — the bundle bypassed it, so replays lost OI (always) and
    LSR/taker (replays >3 weeks) that were servable, contradicting the
    module's own docstring. The three legs clamp their start into the
    retained tail now (PIT-safe: old rows drop, the end stays at the
    analysis date; live runs byte-identical), the OI percentile render
    states the span actually served (`days` field), and the regime's
    OI/LSR/taker legs inherit the fix as free riders.
  - **P2 — the tradeability gate converted garbage targets into confident
    TRADEABLE verdicts**: `gross_edge` validated only `target is None` and
    `reference <= 0`, so a PM-emitted `price_target: 0` (schema has no
    `gt=0`) on a Sell rating computed `1 - 0/ref = +100%` edge →
    TRADEABLE with "net_edge_above_costs" into the ticket/report, and NaN
    targets/costs defeated every `<=` comparison (NaN funding through
    `series.sum()` made `total_bps` NaN → fell through to TRADEABLE). The
    gate refuses non-finite/non-positive targets (UNEVALUATED,
    `invalid_target_price`) and non-finite costs (NO_TRADE,
    `cost_estimate_not_finite`) — both new stable reason strings.
  - **P2 — `RiskManager.decide` laundered a NaN/negative equity tick past
    the drawdown breaker**: the `equity > 0 else BreakerState(normal)`
    guard fabricated a permissive state on exactly the corrupt ticks the
    breaker's own bad-tick path exists to refuse (NaN `> 0` is False, and
    the injection site's `or 0.0` PRESERVES NaN — NaN is truthy), skipping
    a hard_stop breaker for one decision at full Kelly. `update()` is now
    called unconditionally; its documented path replays the last valid
    regime fail-closed.
  - **P2 — `get_indicators` / `get_verified_market_snapshot` never clamped
    the LLM-supplied date** (the round-5 PIT guard's two missed entries):
    a model echoing the cache's "Data retrieved on" header date served
    post-decision bars through the prompt-mandated "source of truth"
    snapshot on historical stock AND crypto_spot runs. Both tool entries
    clamp to the pinned analysis date now (same guard as the news/weekly/
    price-structure tools).
  - **P2 — vision edge-day disclosure vanished for 24h after the first
    404**: `_coverage_notes` read the call-scoped `report.missing_days`,
    but the sync's 24h retry backoff keeps a previously-404ed edge day out
    of every later call's attempt set — the newest, decision-adjacent day
    silently dropped from the header while the body ended a day early
    (hits run_robust retries). `qa()` now derives `known_missing_days`
    from the manifest and the disclosure unions both sources.
  - **P2 — the three ledger layers disagreed about midnight-cross grace
    rows**: round-6 granted the scorer a one-bar-day `entry_grace`, but
    `_scoring_version` and the shadow audit re-derived the STRICT boundary
    and silently dropped/rejected the same rows (append-only outcomes =
    permanently lost calibration samples, ~0.7% of live captures). The
    grace predicate lives in `time_contract.midnight_cross_grace_day` /
    `entry_boundary_admitted` now, consumed by the scorer, the scoreboard
    and the audit — one definition, no drift.
  - **P2 — schema junk still discarded whole structured decisions (#1058
    class, the remaining strict fields)**: percent-form `confidence` /
    `evidence_coverage` (`85`, `"85%"`) and `OutcomeProbabilities`
    (`61/24/16`) failed the 0-1 bounds; `"1,250.50"` failed float
    coercion; a capitalized Literal `"Medium"` failed the required
    sentiment/positioning confidence — each killing the WHOLE parsed
    object to free-text fallback. Numeric strings now parse with
    comma-grouping stripped, percent forms rescale (one unusable branch
    degrades the OPTIONAL probabilities block, never the decision),
    required Literals rescue case-insensitively and floor out-of-vocab
    junk to "low".
  - **P3 batch (data layer)**: the taker `mean_7d` averaged 8 daily values
    under a 7d label (aligned to the funding seam's 7-day end-inclusive
    convention); `_last_close_hint` ran eagerly on the main thread (now
    inside the worker lambda); the four windowed `/futures/data` tool
    headers claimed the look-back count while serving the window (now
    "windowed request, N rows served"); `derivatives_stress_series` lacked
    the endTime belt-and-suspenders trim its four sibling consumers carry;
    ccxt/TradingView pair forms (`BTC/USDT`, `BTC:USDT`) crashed the
    vision tools with an untyped "unsafe cache filename" ValueError (the
    venue normalizer strips `/` and `:` now, typing the same input the
    REST twin always handled).
  - **P3 batch (risk/bridge)**: the fair-value bridge accepted zero/
    negative underlying targets (finite ≠ plausible — they converted
    cleanly into contract targets that sailed past the
    `contract_target is not None` eligibility veto) and rendered
    micro-price targets as "0.00 USD" (`.2f` class — adaptive `_fmt_price`
    keeps 100-scale renders byte-identical); the bridge's converted
    contract target never fed the tradeability verdict that judged the
    same trade — a USD-quoted PM target's verdict is re-priced against the
    converted target at the record stage (funding threaded through so the
    re-derivation uses the build's own cost inputs; disclosed in the
    bridge block); the class-agnostic constraint sums (`global_gross` /
    `directional` / `cluster`) filtered `others` by symbol only, so a
    same-symbol other-venue row (kept in the post book by the round-6
    replacement key) counted NOWHERE — post gross undercounted by exactly
    that row and the cap could false-PASS (excluded by the replacement key
    now); `_beta` filtered NaNs from each side independently, leaving
    equal lengths over DIFFERENT dates (pairwise-aligned filtering — the
    docstring's own refusal rationale).
  - **P3 batch (analysts/accuracy)**: the news prompt named
    `get_news(query, …)` while the tool's first parameter is `ticker`;
    the IC-context fallback blended cross-venue verdicts (a perp run with
    no tagged verdicts fell back to EVERY file — now untagged-only, the
    docstring's promised venue purity); the spot nudge overclaimed that
    Yahoo-resolving tools "resolve correctly" for alt-coin pairs; the
    sentiment confidence cap regex missed lowercase free-text
    (`re.IGNORECASE`); `accuracy._resolve_asset_type` labelled a USDT-
    suffix override of a logged "stock" tag as "logged", hiding the
    inference caveat; a stale memory_resolution comment claimed the
    legacy crypto umbrella prices as perp (it prices as spot).
  - **Known issues recorded by round 7 (reviewed, deliberately not
    changed)**:
    - The REST paginator's 50k-row safety cap truncates the RECENT tail
      with a log-only disclosure (the LLM never sees it; a 1m-interval
      year-long request serves the oldest ~35 days under a full-window
      header). The pinned warning names the direction honestly; the fix
      direction (an in-band header note threaded out of `_paginate_
      history`, or end-anchoring like `_futures_data_window`) needs a
      design pass.
    - Live-path robust retries double-sample predictions (each attempt
      mints a fresh `run_id`; the B1 cohort runner guards exactly this
      state, the live path does not) — cross-run capture idempotency is a
      design decision recorded for the P0 track.
    - The p0-shadow egress guard's curl layer verifies only the
      first/last effective URL (redirect hops) and `requests.Session.send`
      is unwrapped — hardening gaps in a four-layer guard with no current
      caller on those idioms.
    - The positioning analyst's required `funding_bias` has no
      "unavailable" value while historical runs structurally skip funding
      evidence (flag default off); the shared prediction-capture slot
      mis-routes `submit_prediction` under analyst-parallel + ledger (both
      flags non-default).
    - Backtest perp economics observations (reported only): B&H benchmark
      is frictionless while the strategy pays per-fill costs (conservative
      for the beats-buyhold bar); full-day funding is charged on intrabar
      forced-exit days (conservative); funding/MMR-tier notionals use the
      LAST close where Binance settles on mark; `funding_drag_annualized`
      is diluted by flat days; `alpha_vs_index` counts the horizon per
      leg's own calendar; the gate's beats-buyhold rewards leverage for
      levered runs (B&H pinned at 1x); Sharpe uses ddof=0 while IR/TE use
      ddof=1.
    - Round-5's web-display and weekly-Friday items remain as recorded
      there.
  - **Round-7 closeout (2026-09-23, same day — one user decision + the two
    P3s the wrap-up was interrupted on; every change pinned)**:
    - `unknown_perp` annualizes at 365 (user call on the round-7 known
      issue): `sessions_per_year` gives an unclassified Binance USDT-M
      perp the same 24/7 calendar as classified ones — "unknown" is a
      fundamentals-routing verdict, not a session-count one; equity /
      `None` keep 252. Kills the empty-registry Sharpe understatement
      (×√(252/365) ≈ 0.83) and the cross-machine non-reproducibility;
      pin flipped in `test_backtest_perp_classes`
      (`test_unknown_perp_annualizes_at_365`).
    - The sentiment prompt's best-practice #1 ("read the StockTwits
      Bullish/Bearish ratio") no longer appears unconditionally: on
      square_only runs the StockTwits block is a policy placeholder, so #1
      becomes an explicit no-evidence instruction. The fetch predicate and
      the prompt gate now share `_STOCKTWITS_PROFILES`; the fetched
      (`full` / `square_plus_underlying`) wording is byte-identical.
    - The REST paginator's malformed-tail break (cursor extraction
      failing on the last page was treated as a normal final page,
      silently dropping any remaining recent rows) now emits a
      distinguishing warning naming the truncation direction. The full
      in-band header fix stays with the pagination-cap known issue
      (design pass).

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

- **Round 5 (2026-09-19, same day): known-issues resolved + five review
  shards (execution / orchestration-cache / batch-reporting / edge vendors
  / checkpoint) + the six test-gap files filled.** All fixes carry pins in
  `tests/test_round5_regressions.py` / the new dedicated test files.
  - **trade_ticket spot/perp discrimination landed** (round-4 deferred
    item a): a crypto report whose overlay is present but carries NO
    perp-only bullet (Suggested Leverage / Est. Liquidation Price) is a
    crypto_spot report — the ticket now renders spot semantics (leverage
    hard-capped at 1.0, no liquidation row, no 永续 funding note) instead
    of perp leverage advice on a spot position. Reports with no overlay at
    all keep the perp default (the discriminator has no evidence).
  - **States log gained the venue dimension** (round-3 known issue 1):
    crypto runs write `full_states_log_<date>_perp.json` / `_spot.json`;
    equities and the legacy `crypto` umbrella keep the unsuffixed name —
    a same-date perp AND spot run of one ticker no longer overwrite each
    other. Readers updated: web/store keys runs by the filename stem
    (venue-suffixed date keys list BOTH runs), run_robust resolves the
    quality block by newest-mtime glob, accuracy already read venue from
    the JSON content. node_perf telemetry carries the same suffix.
  - **SHORT funding gate is sign-aware** (round-4 item b): the V2.4
    transitional short heuristic now dampens at the same disclosed
    warn/hard thresholds when annualized funding runs AGAINST the short
    (funding ≤ −0.30 halves, ≤ −0.60 quarters); favourable carry never
    boosts (symmetric with the long gate). Damping is disclosed in the
    shadow render line and recorded (`short_sizing:
    heuristic_funding_damped` + the carry and multiplier read).
  - **Portfolio-control snapshot equity uses BOOK equity** (round-4
    item c): the old `position_value or equity` chain fed advisory dollar
    shocks an account 1/weight the size of the real book; equity now comes
    from the injected portfolio_state (1.0 stays the honest degenerate
    floor for single-instrument runs).
  - **Registry/warm divergence closed at the resolver** (round-3 known
    issue 2): `stock_perp_underlying` consults the persisted instrument
    registry with the SAME precedence routing gives it — a latest
    exchangeInfo row promotes a base the warm/seed set missed AND demotes
    a stale positive the exchange no longer lists as EQUITY; registry-empty
    keeps the warm/seed answer; flag OFF is byte-identical. All 15+
    hot-path consumers (analysts, tools, bundle, regime) and
    `instrument_class` now agree by construction.
  - **Checkpoint resume actually resumes (P1)**: `_run_graph` passed the
    full fresh init-state dict on every invocation — langgraph treats any
    non-None input as a NEW turn from `__start__`, so a checkpointed
    "resume" silently re-billed the whole graph while logging "Resuming
    from step N". A detected checkpoint now threads `graph_input = None`
    through invoke/stream (debug + telemetry paths), resuming at the first
    node after the last checkpoint.
  - **Checkpoint run-signature folds input-affecting config**: the shape
    signature gains a sorted-JSON `config=` segment (model tiers,
    sec_ownership / a_share_native / valuation_tools / indicator_battery /
    indicator_ic_context / market_regime / data_vendors) — restoring a
    checkpoint across a toolset flip no longer replays stale analyst
    reports made by the old tools. Interactive `--checkpoint` now says it
    runs uncheckpointed (streamed path), and a successful interactive run
    clears a leftover crashed checkpoint for its ticker+date+signature.
  - **Orchestration fixes**: run_robust exits non-zero when tickers are
    MISSING after a Ctrl+C abort (the old code compared success against
    the completed subset only — "1/1 成功", exit 0 on a half-finished
    batch); contradictory `--require-data-quality --allow-degraded`
    resolves loudly; the timeout-shim comment stops claiming protection
    the repo does not ship. run_baseline `--smoke` threads
    `--asset-type` into propagate (a perp smoke used to silently run the
    stock pipeline against a perp symbol).
  - **DecisionCache keys the venue** (schema v3): the key folds
    asset_type; a crypto vs perp vs stock backtest of the same
    ticker+date+run_tag no longer replays the other venue's realized
    decision. v2-era entries at the legacy path are removed on sight.
  - **Edge-vendor sweep** (seven files, same depth as the bundle side):
    price-structure + weekly-indicator renders use magnitude-aware price
    formatting (PEPEUSDT-class levels no longer render "0.00" — the
    mandated-citation evidence was destroyed at the render seam while the
    math was correct); fred + the five TA tools clamp the LLM-supplied
    curr_date to the pinned analysis date (the news vendors' guard, now
    at every direct OHLCV/facts entry); ohlcv_resample drops a live
    Friday bin that is still forming (a daily row beyond the label, or a
    historical date, proves completeness); the ETF fund-data failure now
    lands in the quality ledger (direct-connect blind spot — for an ETF
    run that snapshot IS the core evidence); polymarket accepts date-only
    (naive) endDate without degrading the whole category.
  - **Cross-cutting**: run_analyst_parallel_ab's Stop Loss / Entry regexes
    accept exponent notation (the A/B determinism gate could false-PASS on
    micro-price stops by truncating two different values to the same
    mantissa); overlay_fields' target_weight regex keeps a negative sign;
    the report tree (decision.md / complete_report.md) writes atomically
    (same tmp+os.replace discipline as the states log — a watchdog kill
    mid-write used to leave truncated reports that parse as garbage
    levels).
  - **Round-5 self-review (adversarial, on this same diff).** The green
    suite could not see three of the batch's own defects; verified and
    fixed with pins:
    - **Web UI 404'd on every crypto-venue history click** — the store
      now serves venue-suffixed date keys, but `web/app.py:_validate_date`
      still parsed a plain `%Y-%m-%d` ("unconverted data remains: _perp")
      and the post-run `report_url` carried the unsuffixed date; both
      accept/carry the suffix now (auto-detected runs resolve to the
      `crypto` umbrella and keep the unsuffixed file, so both link shapes
      resolve).
    - **The ticket spot discriminator misread DEGENERATE perp reports as
      spot** — a perp run whose price legs failed renders an overlay with
      no perp bullets (the exact signature the flip keyed on). The flip
      now requires a HEALTHY overlay (entry + stop present, which a
      healthy perp overlay always accompanies with its perp bullets); a
      degraded perp report keeps perp semantics and exits
      missing_levels instead of rendering a wrong spot ticket.
    - **True checkpoint resume loses the crashed attempt's quality-ledger
      evidence** — the resumed process rebuilds the accumulator fresh, so
      sentinel events from nodes that never re-ran are gone and
      core_sentinel_count reads 0. The log now carries a
      presence-gated `checkpoint_resumed: true` key so a consumer never
      reads the partial count as "fully evidenced" (re-hydrating evidence
      across processes is not possible — the events died with the crash;
      recorded below as the known limitation).
    - Also: `--asset-type crypto_spot` added to run_baseline's choices
      (the smoke path threads it but the choices list pre-dated spot);
      comment corrections (the SHORT-gate comment wrongly claimed the
      ticket cost leg was sign-blind — it never was; the ETF sentinel
      comment now states its optional tier honestly); web/store's module
      docstring documents the suffixed filename layout.
  - **Test gaps closed** (new files): test_trade_ticket.py (direction
    matrix, level priority, parser suite, end-to-end spot/perp
    discriminator), test_rank_signals.py (score weights / grouping /
    mixed-venue comparability), test_onchain_flows.py (span label,
    structural-gap sentinel, per-chart sentinel, oldest fetched_at),
    test_kelly.py / test_cvar.py (negative-edge clamp, Sell band, tail
    breach), test_corporate_actions.py (split direction, dividend sign,
    PIT), test_batch_runner.py (per-ticker error isolation taxonomy).
    The test-writing pass itself surfaced two more fixed defects:
    trade_ticket's compute path divided by a zero stop distance when
    trader Entry == overlay Stop (degenerate_levels status now, was a
    ZeroDivisionError that killed the whole ticket), and
    `_apply_corporate_actions` applied records whose ex-date postdates the
    window's last bar (a future-dated dividend subtracted from every
    close — unreachable in production via corporate_actions=None, but the
    window guard is unconditional now). `prepare_batch_run([])` raises
    BatchInputError instead of a bare IndexError.

- **Known issues recorded by round 5 (reviewed, deliberately not changed
  here)**:
  - **Execution layer (Track B, default-off/unwired — findings recorded
    for the execution-track project, per the standing agreement not to
    touch it in data-layer batches)**: P1 — hedge-mode CLOSE maps
    `position_side` from the order's direction instead of the opposite of
    the trade side, so the bridge's SHORT+CLOSE risk exit would try to
    close a short that does not exist (REJECTED exactly when
    breaker_close_only makes closes the only allowed orders); P2s —
    free-text `reference` doubles as the Binance client-order-id
    idempotency key (a non-unique tag can REJECTED-spurious or recover a
    PHANTOM fill of an older order after an ambiguous submit);
    one-way-mode positions lose long/short sign in query_position
    (Direction.NET + abs); the kill-switch network edge runs AFTER the
    leverage POST and checks only KillSwitch (not the live-execution
    policy); unknown Binance status maps to SUBMITTING (an active state)
    instead of a fail-closed marker. P3s — stale line-number/"nothing
    calls it" comments, audit gate label always weight_cap, duplicate
    close-offset sets in two modules, browser-broker NaN equity slips the
    notional cap, LIMIT price=0 gap at the bridge, env-false cannot
    override a constructor mainnet=True, isolation-guard regex gaps,
    "buyed/selled" grammar, DRY_RUN_PREVIEW→SUBMITTING sketch in
    gateway.py, and CLAUDE.md's "方向判定逐字对齐" overstates the bridge's
    deliberate bearish gating (doc drift, not code).
  - Checkpoint P3s (accepted): each attempt registers a fresh ledger run
    row (duplicate rows after crash+resume, hygiene); a corrupt checkpoint
    DB fails loudly per-ticker until --clear-checkpoints (fail-closed with
    an escape hatch); has_checkpoint is test-only dead code; a resumed
    run's data_quality covers only the resumed segment (disclosed via
    checkpoint_resumed; cross-process re-hydration is impossible by
    construction); a crash in the report-writing window (between graph
    END and checkpoint clear) leaves an END checkpoint whose rerun resumes
    with zero executed nodes — graceful on the default invoke path, but
    the telemetry stream's zero-chunks guard would raise (stream
    telemetry is default-off).
  - Web display drift from the venue split (display-only): run_count and
    the home timeline dots count venue files (a same-date perp+spot pair
    shows as 2 runs), and within one date the lexicographic key order
    picks `_spot` as "latest" (arbitrary venue preference).
  - The weekly forming-bar guard drops the latest COMPLETED Friday bin on
    weekend/pre-market live EQUITY runs (equity vendors never publish a
    Saturday row, so "a daily row beyond the label" can never prove
    completeness there — stale-by-one-week direction, crypto unaffected).
    The ETF fund-data sentinel stays optional-tier (advisory, never a
    DEGRADED veto); upgrading it to core for ETF runs — where the
    snapshot is the substantive fundamentals evidence — is an open
    product decision.
  - Yahoo-fed live runs still read the forming daily bar in the four
    OHLCV-fed TA modules (current information, not lookahead; closing the
    seam means changing load_ohlcv semantics shared by stockstats — needs
    its own pass); daily-pivot reference session wobbles when Yahoo has
    not yet listed today's row; three-soldiers/crows gap tolerance is
    asymmetric between the mirrors (pattern-definition choice).
  - run_robust --reports-root pointing at a custom root derives the
    states-log path from it while the child keeps writing under the real
    results_dir (opt-in footgun; loud failure); a watchdog-killed attempt
    that already wrote complete_report.md is never credited (conservative
    direction); web/reporting render a MISSING quality block as clean for
    pre-feature logs (run_robust, the strict consumer, already treats
    missing as unknown).
  - kelly / cvar observations (test-suite review, reported only): the
    conviction-band FLOOR overrides the negative-edge clamp for
    Buy/Overweight (a negative-expectancy Buy still sizes at the 0.05 tier
    minimum — documented "regardless of what raw Kelly spits out"
    behaviour, but it contradicts a naive reading of the clamp);
    historical_cvar's tail depth `ceil(n × (1 − confidence))` picks 6 at
    n=100 / 4 at n=60 for confidence=0.95 (float 1−0.95 >
    0.05 — conservative direction, pinned); its tail_count > size clamp is
    unreachable dead code.
  - rank_signals / onchain observations (test-suite review, reported only):
    `render_rank`'s ``tickers`` parameter is dead; ranking sorts on the
    ROUNDED conviction score (78.4 vs 77.6 can tie at 78 and fall back to
    input order, and banker's rounding can make displayed rank contradict
    the displayed score); `--since` silently keeps report dirs without a
    `_YYYYMMDD_HHMMSS` stamp; onchain's oldest-`fetched_at` pick is a
    lexicographic string min (correct for the module's own uniform ISO
    stamps, theoretical for a hand-edited legacy cache in another ISO
    shape); `_points` would raise AttributeError on a non-dict payload
(currently saved by the caller's broad except).
  - trade_ticket residual observations (test-suite review, reported only):
    resolve_levels' truthiness chains treat a parsed 0.0 level as
    "missing" and fall through to the next source (a zero price is
    unreachable through real reports but would silently substitute the
    fallback source); a spot ticket's execution plan still says
    "开 1.0× 杠杆头寸 / 保证金"（numerically right at L=1, wording is
    perp-shaped）; decide_direction silently treats an unknown rating
    string as missing (safe while parsers gate to the 5 tiers). The
    stop==entry ZeroDivisionError and the dead ticker-split line from the
    same review ARE fixed this round (degenerate_levels status + pin).

- **Round 6 (2026-09-22, USDT-perp focus; three parallel review shards —
  LLM client layer / perp risk-ticket micro-layer / graph+ledger — plus a
  main-line sweep of the unswept corners: binance_square, monitoring,
  web, run_batch, backtest attribution, preflight).** All fixes carry pins
  in `tests/test_round6_regressions.py` (26) or updated dedicated files;
  full suite 3636 passed / ruff clean.
  - **P2 — enforced perp tickets pinned leverage to 1.0x with no
    liquidation advisory**: `_sync_candidate_ticket` forwarded a
    side-derived lowercase proxy rating ("buy") while `RATING_STRENGTH` is
    title-case — strength read 0, `L_conv` collapsed to 0.0, so EVERY
    enforced-mode perp ticket advised ≤1.0x and dropped its liquidation
    row. Strength lookup is case-insensitive now
    (`perp_ticket.rating_strength`).
  - **P2 — SHORT tickets could carry the overlay's LONG-form stop**: the
    risk manager computes stops only on positive weights (long-only
    convention), so a report whose markdown rating is bearish while the
    structured rating sized long (rating divergence, or an Underweight
    band with positive Kelly) handed `trade_ticket` a stop BELOW entry
    for a short — `abs()` swallowed the side, the stop sat in the profit
    direction and never fired, and the TP ladder shared its side. Both
    `build_ticket` (long AND short side validation, honest
    `degenerate_levels` + `status_reason`) and `rank_signals`' rr
    denominator guard refuse inverted levels now.
  - **P2 — explicit SHORT with a positive legacy weight bypassed the
    sign-aware funding gate**: round 5's gate lived inside the heuristic
    sizing branch (`proposed <= 0`), so the divergence scenario above
    skipped it entirely and shorted full-size into ≤ −0.60 carry. The
    gate now applies to BOTH short sizing paths, with the
    `legacy_funding_damped` record key and the shadow/enforced render
    lines disclosing it.
  - **P2 — perp backtest `event_study=True` benchmarked against
    SPYUSDT**: `_run_event_study` received the perp-swapped price
    provider, whose venue normalization resolved the benchmark name "SPY"
    into Binance's SPYUSDT tokenized-stock PERP — a silently different
    instrument than the label claimed (and roughly-sane β made it
    invisible). The bench leg goes through the ORIGINAL yfinance provider
    captured before the swap (`bench_provider` param; tests injecting one
    synthetic provider keep the old both-legs behaviour by default).
  - **P2 — entrance identity split for major-coin perps**: the web form
    and the interactive CLI Yahoo-normalized the submitted symbol
    (BTCUSDT → BTC-USD) while `run_robust --tickers BTCUSDT` files the
    contract spelling — the same contract split into TWO ledger
    identities (memory lessons, verify-history grouping, states-log
    dirs), hitting exactly the `_CRYPTO_BASES` majors. Data was never
    wrong (every Binance seam re-normalizes), but the learning loop was
    fragmented. Explicit `crypto_perp`/`crypto_spot` submissions now
    canonicalize through `normalize_symbol_for_venue` (delivery-contract
    inputs 400 loudly); stock/crypto-umbrella paths are unchanged.
  - **P2 — pooled GLM streaming silently zeroed token telemetry**: a
    custom `http_client` (the key pool's) disables langchain-openai's
    automatic `stream_usage`, so streamed pooled calls never asked for a
    usage block — `usage_metadata` read None and CLI stats / perf
    telemetry counted zero tokens on every uncached call. The pooled
    branch sets `stream_usage=True` (kill-switch
    `YIALPHA_LLM_POOL_STREAM_USAGE=false` for endpoints that reject
    `stream_options`).
  - **P2 — `YIALPHA_LLM_TIMEOUT_S=0` did the opposite of its docs**: 0
    fell into the not-positive rejection and BOUNCED to the 120s cloud
    default, so an operator asking for "no ceiling, watchdog owns
    recovery" got every >120s reasoning call timed out and retried.
    0 now resolves to `timeout=None` (explicit disable), matching
    CLAUDE.md / run_robust's documented contract; the old 0→120 pin is
    reversed.
  - **Factor alpha annualized on the wrong calendar**: the intercept is a
    per-joined-observation mean, but annualization used the ASSET's
    `periods_per_year` (365 for perps) while the strategy∩FF-factors join
    is ~252 obs/year — perp factor alpha overstated ~45%. Annualization
    now derives from the joined calendar's observed frequency.
  - **Exchange leverage tiers were fetched and thrown away**: the
    leverageBracket parser discarded `initialLeverage`, so ticket
    advisories never capped against the exchange's own per-tier ceiling —
    low-max contracts could be advised leverage Binance rejects outright.
    `Bracket.max_leverage` parses, `compute_leverage` takes an optional
    `L_bracket` cap (disclosed in the overlay line), and
    `advisory_leverage_cap` fetches it fail-soft (no keys / failure →
    None → byte-identical advice). `perp_ticket_numbers` also gained
    `profile` (runs honor `risk_profile` config; the DEFAULT_PROFILE
    comment claiming the script default matches was false — the script's
    `--risk-profile` is a deliberate execution-time knob, now documented
    truthfully) and `atr_mult` (a configured 3·ATR run no longer renders
    SHORT advice a whole ATR from the stop it places).
  - **Ledger robustness batch**: one reflection-LLM failure during
    pending-resolution no longer kills the whole analysis run before any
    node executes (entry stays pending, earlier updates land);
    `batch_update_with_outcomes` keys its dispatch by (date, ticker,
    venue) so one sweep resolves BOTH same-day perp+spot entries (the
    pair used to collapse to whichever came last) and a legacy None-keyed
    update no longer jumps a venue-tagged block ahead of its own update
    (exact pass first, legacy fallback pass second — the pinned
    legacy-compat path survives); the legacy scoring path's PERMANENT
    vendor failures (a delisted contract's NoMarketDataError) retire the
    row as terminal incomplete instead of re-raising on every compute
    batch forever, while transient transport errors keep the
    failed-and-retried-next-batch semantics (the first cut caught
    Exception wholesale — one network blip would have retired a
    prediction; caught in self-review); the reference-time capture
    anchors its freshness check to
    the REQUESTED day (a UTC-midnight-straddling fetch used to discard a
    perfectly valid snapshot) with the scorer granting a one-bar-day,
    near-midnight grace on both seams (equity weekend-walked, disclosed
    via `entry_grace`) — the forming-bar security property is unchanged
    (pinned: the 09-05 row never leaks); `attach_ticket` retries once on
    a transient DB lock before giving up (a single SQLITE_BUSY used to
    silently strand the whole run unscoreable after grace);
    `parse_timestamp` detects date-only by TIME COMPONENT (a
    space-separated ISO datetime read date_only and bypassed the intraday
    knowability gate); the streamed interactive path now resets the
    ledger run context / PIT date / run scope at finalize (mirroring
    `_run_graph`'s finally) and a failed run-register rolls back the
    already-bound context instead of "recording" into a run row that does
    not exist.
  - **LLM client layer batch (first dedicated sweep — the 4-key pool was
    3 days old)**: 429 cooldowns are no longer re-stamped while already
    cooling (forced retries used to push the soonest-reviving key's
    deadline out by a full cooldown per request — under sustained
    saturation the pool never revived); `has_pool` parses the pool env
    instead of raw truthiness (a `" , "` value engaged the pool path with
    zero keys and the error message pointed at the single-key var — the
    message now names the actual problem); capability lookups are
    case-insensitive on both sides (a naturally-cased custom model ID
    silently lost DeepSeek/MiniMax quirks and sent `tool_choice` to a
    thinking model); the pooled and keepalive clients set
    `follow_redirects=True` to match the openai SDK's own default client
    (a 30x-relaying base_url surfaced as APIStatusError only on the
    custom-client paths); cache writes use per-writer tmp names (two
    processes on one cache key could interleave a half-written payload)
    and `clear()` sweeps tmp husks; cache replays are stamped
    `cached_replay` and token accounting splits live vs replayed tallies
    (a robust retry run booked attempt 1's cached usage as its own
    spend); the dead `state.size != len(keys)` rebuild check is gone
    (it implied mid-process env changes re-arm the pool; they do not);
    the model catalog no longer promises "switchable thinking modes" the
    transport never toggles.
  - **Preflight actually probes the LLM endpoint**: the connectivity
    check only implemented DeepSeek's /v1/models probe — the GLM setups
    this project runs reported a vacuous True ("probe implemented for
    deepseek only"), so a wrong endpoint/key first surfaced mid-run.
    Every OpenAI-compatible provider now probes its resolved base URL
    (`{base}/models`; transport error or 401/403 = red, any other HTTP
    status = reachable — a 404 just means no models-listing route).
  - **Monitoring dashboard writes atomically** (the round-5 tmp+replace
    discipline reached the one writer it missed); assorted DeepSeek→GLM
    doc drift corrected (web app/runner headers, run_batch help).
  - **Known issues recorded by round 6 (reviewed, deliberately not
    changed)**:
    - `resolve_market_benchmark`'s crypto-native branch (`BTCUSDT`)
      needs `asset_type`, but `graph._resolve_benchmark` and two other
      callers drop it at the delegation seam — every crypto run resolves
      SPY there while market_regime's own call site resolves the
      crypto-native benchmark. Whether SPY (a USD market factor for a
      BTC strategy) or BTC (the market itself) is the RIGHT benchmark
      per consumer is a product call; the drift itself is recorded.
    - `DrawdownBreaker` state lives only inside one RiskManager
      instance: live runs (new graph per run) can never accumulate the
      peak/regime progression across runs, so the 20% drawdown breaker
      only bites within a single run's decisions. Cross-run persistence
      needs a state-file design (venue-keyed) — capability gap, not
      touched in a data-layer batch.
    - A version bump of `OUTCOME_COMPUTE_VERSION` would mass-retire
      every stored pending prediction as terminal incomplete in one
      batch (fail-closed by design; the blast radius deserves a line in
      the bump checklist rather than code today).
    - The overlay renders its markdown Stop-Loss bullet in LONG form
      whenever the legacy weight is positive (long-only convention); with
      the script-side side validation now refusing inverted tickets this
      is honest, but a bearish-markdown/positive-weight report yields
      `missing_levels` at ticket time rather than a short ticket —
      closing it needs overlay-side direction-aware rendering (product
      decision on negative-band weights).
    - The rate limiter is per-provider (not per-key) and keyed
      `(provider, rpm)` — with the pool multiplying quota the operator's
      single-key rpm acts as a conservative whole-pool budget, and two
      graphs with different rpm values in one process get two buckets.
      Documented semantics, flagged for a future pass.
    - The entrance unification covers the EXPLICIT venue submissions
      (crypto_perp / crypto_spot) only: a web "auto" submission of a
      crypto symbol still files under the Yahoo form (BTC-USD, the
      historical crypto-umbrella identity) while `run_robust --tickers
      BTCUSDT` without an explicit `--asset-type` files the contract
      spelling for the same umbrella mode. Unifying the UMBRELLA identity
      too would re-home every historical crypto run's ledger identity —
      deliberately not done in this batch.
    - run_count / timeline drift from the venue split and the other
      round-5 web display items remain as recorded there.

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
[unreleased]: https://github.com/Yicheng-Mark/Yialpha/compare/v1.0.0...HEAD
