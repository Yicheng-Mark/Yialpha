#!/usr/bin/env python
"""FastAPI web UI for YiAgents: browse past analyses + submit new ones.

MUST run from the project root (the dir containing ``.env``, ``yiagents/``,
``scripts/``). ``yiagents/__init__.py`` loads ``.env`` via
``load_dotenv(usecwd=True)``, so running from elsewhere leaves
``DEEPSEEK_API_KEY`` and the SOCKS5 proxy unset — the spawned ``run_robust``
subprocess inherits this env and would then be unable to reach DeepSeek or
yfinance. A cwd check at import time refuses to start otherwise.

Start (from the project root):

    python web/app.py
    # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import contextlib
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows console is GBK (cp936); printing ✅/❌ would raise UnicodeEncodeError.
# Reconfigure before anything logs.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Make the project root importable when launched as a script
# (``python web/app.py`` puts web/ on sys.path, not its parent). ``web`` is not
# in [tool.setuptools.packages.find], so it is not picked up by an editable
# install — this bootstrap is what makes ``from web import ...`` resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from cli.utils import normalize_ticker_symbol  # noqa: E402
from web import health, runner, store  # noqa: E402
from yiagents.dataflows.utils import safe_ticker_component  # noqa: E402

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _assert_cwd_is_project_root() -> None:
    """Refuse to start unless cwd is the project root (see module docstring)."""
    cwd = Path.cwd()
    missing = []
    if not (cwd / "yiagents").is_dir():
        missing.append("yiagents/")
    if not (cwd / "scripts" / "run_robust.py").is_file():
        missing.append("scripts/run_robust.py")
    if missing:
        sys.stderr.write(
            "\nERROR: web/app.py must run from the YiAgents project root.\n"
            f"  cwd     = {cwd}\n"
            f"  missing = {', '.join(missing)}\n"
            "  yiagents/__init__.py loads .env via load_dotenv(usecwd=True); "
            "running elsewhere leaves DEEPSEEK_API_KEY / proxy unset.\n"
            "  → cd into the project root, then: python web/app.py\n\n"
        )
        sys.exit(2)


_assert_cwd_is_project_root()

app = FastAPI(title="YiAgents Web", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Serve the on-disk report tree read-only so the detail view's "download
# complete_report.md" links resolve. StaticFiles rejects ".." traversal, and
# html=False (default) returns 404 for directory requests — direct file paths
# only, no directory listing. The dir is created eagerly (harmless; run_robust
# creates it too) so mounting never fails on a fresh install with no runs yet.
store.REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(store.REPORTS_ROOT)), name="reports")


def _validate_path_ticker(ticker: str) -> str:
    """Validate a path-parameter ticker; raise 404 on anything unsafe."""
    try:
        return safe_ticker_component(ticker)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"unknown ticker: {ticker!r}") from None


def _validate_date(date: str) -> str:
    """Validate a path-parameter date; raise 404 on a malformed value."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=404, detail=f"bad date: {date!r}") from None
    return date


class AnalyzeReq(BaseModel):
    ticker: str
    date: str
    asset_type: str = "auto"
    # UI language ("en" | "zh"); routed to run_robust via YIAGENTS_OUTPUT_LANGUAGE
    # so the existing get_language_instruction() localizes the agent reports.
    language: str = "en"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/tickers")
def api_tickers():
    return {"tickers": store.list_tickers()}


@app.get("/api/tickers/{ticker}/runs")
def api_runs(ticker: str):
    t = _validate_path_ticker(ticker)
    return store.list_runs(t)


@app.get("/api/tickers/{ticker}/runs/{date}")
def api_run(ticker: str, date: str):
    t = _validate_path_ticker(ticker)
    d = _validate_date(date)
    run = store.load_run(t, d)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run for {t} on {d}")
    return run


@app.get("/api/health")
def api_health():
    # sync def → Starlette threadpools it; the yfinance/DeepSeek probes block
    # for seconds and must not stall the event loop driving the analysis watcher.
    return health.run_health()


@app.post("/api/analyze")
async def api_analyze(req: AnalyzeReq):
    raw = (req.ticker or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="ticker is required")
    # Validate the date inline (400, not _validate_date's 404) — this is a body
    # param submitted by the user, so a malformed value is a 400 bad-request,
    # not a missing resource.
    try:
        datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"bad date: {req.date!r}") from None
    if req.asset_type not in ("auto", "stock", "crypto", "crypto_perp"):
        raise HTTPException(status_code=400, detail=f"bad asset_type: {req.asset_type!r}")
    if req.language not in ("en", "zh"):
        raise HTTPException(status_code=400, detail=f"bad language: {req.language!r}")

    # Canonical Yahoo symbol — the pipeline stores results under this exact
    # spelling (e.g. BTCUSD → BTC-USD), so normalizing here makes the post-run
    # report_url resolve against the on-disk dir.
    ticker = normalize_ticker_symbol(raw)

    if runner.registry.is_busy():
        raise HTTPException(status_code=409, detail="an analysis is already running")

    task_id = await runner.spawn(ticker, req.date, req.asset_type, req.language)
    st = runner.registry.get(task_id)
    return {"task_id": task_id, "started_at": st.started_at}


@app.get("/api/tasks/{task_id}")
async def api_task(task_id: str):
    st = runner.registry.get(task_id)
    if st is None:
        raise HTTPException(status_code=404, detail="unknown task")
    now = st.finished_at if st.finished_at is not None else time.time()
    return {
        "status": st.status,
        "started_at": st.started_at,
        "elapsed_s": round(now - st.started_at, 1),
        "ticker": st.ticker,
        "date": st.date,
        "asset_type": st.asset_type,
        "attempt": st.attempt,
        "max_attempts": st.max_attempts,
        "report_dir": st.report_dir,
        # On success the rendered report lives at the analysis date's JSON,
        # which is exactly what run_robust just (re)wrote.
        "report_url": f"#/t/{st.ticker}/{st.date}" if st.status == "done" else None,
        "log_tail": list(st.log_tail[-20:]),
        "error": st.error,
    }


if __name__ == "__main__":
    import uvicorn

    # Pass the app object (not the "web.app:app" string) so a re-import — which
    # would re-run the cwd assert and re-mount static under a different module
    # identity — never happens. reload stays off; the analysis subprocess is
    # the long-lived thing, not this server.
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
