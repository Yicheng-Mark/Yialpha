# YiAgents Web UI

A small FastAPI app + vanilla HTML/JS frontend that lets you browse past
YiAgents analyses and submit new ones from a browser — no agent / graph /
dataflow code is touched (the project's "铁律").

- **Browse**: ticker grid (with client-side ticker-text + rating-chip filter)
  → per-ticker dates → full report view (rating badge, quantitative risk
  overlay card with keyboard-accessible KPI tooltips, 5 collapsible sections,
  optional node-perf bar chart).
- **Compare**: `#/compare` overlays every ticker's rating history as stepped
  lines on one 5-tier chart (categorical `--cat-*` ramp) plus a ticker×date
  rating matrix; legend toggles series, clicking a point opens that report.
- **Submit**: a form spawns `scripts/run_robust.py` as a subprocess (the same
  watchdog-backed path the CLI uses); the UI polls every 4 s, shows an
  elapsed-based progress bar and links to the finished report. One analysis at
  a time (409 while one is running).
- **i18n**: the 🌐 toggle (top-right) sets BOTH the UI chrome language AND the
  report language. On submit the language is routed to the `run_robust` child
  via `YIAGENTS_OUTPUT_LANGUAGE`, so the existing `get_language_instruction()`
  localizes every agent's output. Static chrome is translated in place; agent
  markdown is rendered verbatim (no in-browser post-translation).
- **Accessibility & theming**: global `:focus-visible` ring, `role="img"` +
  `aria-label` text alternatives on every chart canvas, decorative emoji
  `aria-hidden`, `prefers-reduced-motion` disables animation, skip-to-content
  link, and `color-scheme` per theme so native widgets (date picker / select /
  scrollbar) follow dark/light. Printing (`Ctrl+P`) expands all `<details>`
  sections via `beforeprint` and drops the chrome for ink-on-white output.

## Run it

From the **project root** (the dir containing `.env`, `yiagents/`, `scripts/`):

```bash
pip install -e ".[web]"          # fastapi + uvicorn (one-time)
python web/app.py                # serves http://127.0.0.1:8000
```

> `web/app.py` refuses to start unless `cwd` is the project root. Why:
> `yiagents/__init__.py` loads `.env` via `load_dotenv(usecwd=True)`, so starting
> from elsewhere leaves `DEEPSEEK_API_KEY` and the SOCKS5 proxy unset — the
> spawned `run_robust` subprocess inherits that env and could not reach DeepSeek
> or yfinance. On Windows, also set `PYTHONUTF8=1` if your console is GBK
> (`python web/app.py` already reconfigures stdout/stderr to UTF-8 itself).

## Architecture

```
browser SPA (static/index.html + app.js + charts.js + i18n.js + common.js + theme-init.js + favicon.svg + styles.css + vendored marked/DOMPurify/echarts)
        │  fetch JSON, poll every 4 s
        ▼
FastAPI (app.py) ── mount /static and /reports
  ├─ store.py   read-only scan of ~/.yiagents/logs/* + parse_rating + overlay regex
  ├─ runner.py  asyncio subprocess → scripts/run_robust.py + stdout watcher
  ├─ health.py  thin alias → yiagents.monitoring.preflight (shared with run_baseline.py)
  └─ in-memory TaskRegistry (single-slot busy flag; restart loses history)
        │  subprocess, cwd=project root, env PYTHONUTF8=1
        ▼
scripts/run_robust.py --tickers <T> --date <D> [--asset-type <X>] --workers 1
   (watchdog + taskkill + complete_report.md success check)
        ▼
~/.yiagents/logs/<T>/YiAgentsStrategy_logs/full_states_log_<date>.json   ← UI source of truth
~/.yiagents/logs/reports/<T>_<stamp>/                                    ← download links
```

`common.js` (`window.YiUtil`) is the single home for helpers shared by
`app.js` and `charts.js` (`escapeHTML`, `verdictTilt`, `fmtK`,
`countMatches`, keyword regexes) — previously copy-pasted in both.

### API

| Method / path | Returns |
|---|---|
| `GET /api/tickers` | `{tickers:[{ticker, latest_date, latest_rating, run_count}]}` |
| `GET /api/tickers/{t}/runs` | `{ticker, dates:[…], date_ratings:[…], reports:[{dir, mtime, complete}]}` |
| `GET /api/tickers/{t}/runs/{date}` | rating + overlay + 5 sections + `node_perf?` |
| `GET /api/compare` | `{tickers:[{ticker, date_ratings:[{date, rating}]}]}` (skips all-unreadable tickers) |
| `GET /api/accuracy` | serves the `yiagents verify-history` artifact (`{available:false, hint}` until run) |
| `GET /api/health` | preflight (deps / key / proxy / yfinance / DeepSeek) — shared implementation with `run_baseline.py --preflight` |
| `POST /api/analyze` | `{ticker, date, asset_type, language?}` → `{task_id, started_at}` (409 if busy) |
| `GET /api/tasks/{task_id}` | `{status, elapsed_s, attempt, max_attempts, report_url, …}` |

## Notes / gotchas

- **`trader_investment_decision`**, not `trader_investment_plan` — the JSON write
  key (written in `trading_graph._log_state`). `store.load_run` reads this
  key; any other name leaves the Trader section blank.
- **Report view is derived from the JSON**, not the markdown report tree:
  `reporting.write_report_tree`'s section V uses the pre-overlay
  `risk_debate_state.judge_decision` and does **not** contain
  `final_trade_decision`. The overlay-adjusted final decision lives only in the
  JSON, so the JSON is the single source of truth.
- **Completion = `proc.returncode == 0`**: `run_robust` already requires a fresh
  `complete_report.md` to return 0 (rc 0 with no new report → retry), so rc 0 is
  a reliable signal. The web layer adds **no** outer timeout / kill — that's
  `run_robust`'s watchdog's contract, and a second outer watchdog would race it.
- **`scripts/` is not a package** (no `__init__.py`), so the overlay regex is
  replicated here rather than imported. The preflight checks, by contrast, are
  shared via `yiagents.monitoring.preflight` (single implementation with
  `run_baseline.py`).
- **Marked and DOMPurify are vendored** under `static/vendor/` — no CDN at
  runtime. Model/news Markdown is parsed and then sanitized with an explicit
  HTML/attribute/URL allowlist before it reaches `innerHTML`; if either library
  is unavailable, the renderer falls back to escaped plain text.
