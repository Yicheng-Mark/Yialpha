"""Preflight health checks — one implementation, two renderers.

The same five checks were previously duplicated: ``scripts/run_baseline.py``
printed a Chinese stdout report and ``web/health.py`` carried a "faithful
copy" returning structured JSON (scripts/ is not an importable package, so
the web side could not import the script). This module is the single
implementation returning structured data; both renderers format it.

Checks: (1) Python deps incl. PySocks, (2) env/key presence, (3) proxy port
reachable, (4) a real yfinance pull, (5) DeepSeek API probe (free GET
/v1/models over the NO_PROXY direct route — zero LLM cost).

Run as a sync callable when serving HTTP (register the endpoint as ``def``;
the yfinance and DeepSeek probes block for several seconds and must not stall
the event loop driving the analysis watcher).
"""

from __future__ import annotations

import contextlib
import importlib
import os
import socket


def _resolve_proxy_probe() -> tuple[str, int]:
    """Host/port to TCP-probe, resolved from the project's proxy knobs.

    Priority mirrors the project's first-class configuration surface
    (``SOCKS5_PROXY`` in .env.example / config-check; ``ALL_PROXY`` as the
    requests-level fallback in ``proxy_map``): SOCKS5_PROXY, then ALL_PROXY,
    then the generic HTTPS_PROXY / HTTP_PROXY. Only when none is set does the
    probe fall back to the historical default (a local V2Ray/Xray SOCKS
    listener on 127.0.0.1:1080) so an unset knob is still diagnosable.
    """
    proxy = ""
    for var in ("SOCKS5_PROXY", "ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        proxy = os.environ.get(var, "")
        if proxy:
            break
    host, port = "127.0.0.1", 1080  # legacy default; kept as the unset-knob fallback
    if proxy:
        # Strip scheme, path, query, and userinfo (user:pass@) down to host:port.
        host_port = (
            proxy.split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0].split("@")[-1]
        )
        if ":" in host_port:
            host = host_port.rsplit(":", 1)[0] or host
            with contextlib.suppress(ValueError):
                port = int(host_port.rsplit(":", 1)[1])
    return host, port


def run_health(ticker: str = "SPY") -> dict:
    """Five preflight checks; returns ``{ok, checks: [{name, ok, hint}]}``."""
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
    #    present when the process was started from the project root. The key
    #    requirement is satisfied by either the provider's single-key var
    #    (api_key_env) or a populated key pool (GLM Coding Plan: ZHIPU*_API_KEYS).
    from yialpha.llm_clients.api_key_env import get_api_key_env
    from yialpha.llm_clients.key_pool import api_key_pool, has_pool

    provider = os.environ.get("YIALPHA_LLM_PROVIDER", "openai")
    key_env = get_api_key_env(provider) or f"{provider.upper()} API KEY"
    pool = api_key_pool(provider)
    key = os.environ.get(get_api_key_env(provider) or "", "")
    check(
        f"{key_env} set",
        bool(key) or (has_pool(provider) and bool(pool)),
        ".env not loaded — start from the project root (dir with .env)",
    )
    check(
        "YIALPHA_LLM_PROVIDER set",
        bool(os.environ.get("YIALPHA_LLM_PROVIDER")),
        ".env missing YIALPHA_LLM_PROVIDER",
    )

    # 3) proxy port reachable (TCP probe of the resolved proxy host:port —
    #    SOCKS5_PROXY first, then ALL_PROXY/HTTPS_PROXY/HTTP_PROXY)
    host, port = _resolve_proxy_probe()
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

    # 5) Provider connectivity — DeepSeek only: free GET /v1/models over the
    #    NO_PROXY direct route, zero LLM (chat-completion) cost. Other
    #    providers report key presence (check 2) without a probe; a wrong
    #    endpoint/key surfaces on the first real call rather than as a false
    #    preflight red here.
    if provider == "deepseek" and key:
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
        except Exception as e:  # noqa: BLE001 -- surface the raw failure for triage
            check("DeepSeek API reachable", False, repr(e)[:140])
    else:
        check(
            f"{provider} API reachable",
            True,
            "probe implemented for deepseek only; key presence checked above",
        )

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
