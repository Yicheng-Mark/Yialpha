"""Structural re-implementation of ``scripts/run_baseline.py::preflight``.

``scripts/`` is not a package (no ``__init__.py``) and ``run_baseline`` pulls
in the full backtest stack (pandas, backtrader, the graph) at import time, so
``preflight`` cannot be imported. This is a faithful copy of the same five
checks, returning structured data instead of printing to stdout.

Run as a sync callable (the FastAPI endpoint registers it as ``def``, so
Starlette runs it in a threadpool) — the yfinance and DeepSeek probes block for
several seconds and must not stall the event loop that drives the analysis
watcher.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import socket


def run_health(ticker: str = "SPY") -> dict:
    """Five preflight checks; returns ``{ok, checks:[{name, ok, hint}]}``."""
    checks: list[dict] = []

    def check(name: str, cond: bool, hint: str = "") -> None:
        checks.append({"name": name, "ok": bool(cond), "hint": hint})

    # 1) Python deps — PySocks is load-bearing for the SOCKS5 proxy that
    #    yfinance / market data / the Binance vendor all traverse.
    for mod in ("socks", "yfinance", "pandas", "httpx", "dotenv"):
        try:
            importlib.import_module(mod)
            check(f"dep {mod}", True)
        except ImportError:
            hint = 'pip install "requests[socks]"' if mod == "socks" else f"pip install {mod}"
            check(f"dep {mod}", False, hint)

    # 2) env / key — .env loads via load_dotenv(usecwd=True), so it is only
    #    present when the server was started from the project root.
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    check(
        "DEEPSEEK_API_KEY set",
        bool(key),
        ".env not loaded — start the server from the project root (dir with .env)",
    )
    check(
        "YIAGENTS_LLM_PROVIDER set",
        bool(os.environ.get("YIAGENTS_LLM_PROVIDER")),
        ".env missing YIAGENTS_LLM_PROVIDER",
    )

    # 3) proxy port reachable (TCP probe of the HTTPS_PROXY host:port)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY", "")
    host, port = "127.0.0.1", 1080
    if "://" in proxy:
        hp = proxy.split("://", 1)[1].split("/", 1)[0]
        if ":" in hp:
            host = hp.split(":", 1)[0]
            with contextlib.suppress(ValueError):
                port = int(hp.rsplit(":", 1)[1])
    try:
        socket.create_connection((host, port), timeout=3).close()
        check(f"proxy {host}:{port} reachable", True)
    except OSError:
        check(f"proxy {host}:{port} reachable", False, "confirm V2Ray/Xray on that port")

    # 4) yfinance real pull — the only check that proves proxy + data path
    #    together end to end.
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        check(
            f"yfinance pulled {ticker}",
            len(df) > 0,
            "proxy / PySocks / Yahoo rate-limit",
        )
    except Exception as e:  # noqa: BLE001 -- surface the raw failure for triage
        check(f"yfinance pulled {ticker}", False, repr(e)[:140])

    # 5) DeepSeek connectivity — free GET /v1/models over the NO_PROXY direct
    #    route, zero LLM (chat-completion) cost.
    if key:
        try:
            import httpx

            r = httpx.get(
                "https://api.deepseek.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=12,
            )
            hint = f"HTTP {r.status_code}"
            if r.status_code in (401, 403):
                hint += " (key invalid / no credit)"
            check("DeepSeek API reachable", r.status_code == 200, hint)
        except Exception as e:  # noqa: BLE001
            check("DeepSeek API reachable", False, repr(e)[:140])
    else:
        check("DeepSeek API reachable", False, "no key (see above)")

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
