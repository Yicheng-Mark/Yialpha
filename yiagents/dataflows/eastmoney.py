"""Eastmoney (东方财富) A-share data vendor — margin trading (融资融券).

A new optional category (``a_stock``) of one China A-share-only, read-only
signal that the default yfinance path cannot supply:

* :func:`get_margin_trading` — 融资融券 (margin trading: 融资余额 / 融券余额 /
  融资买入额 / 融资净买入额), pulled from the Eastmoney datacenter
  ``RPTA_WEB_RZRQ_GGMX`` report. Exchange-disclosed daily series with **full
  history** (back to each name's 两融 list date), so it is clean for long
  backtests.

It reuses a small Eastmoney-specific transport (NOT sec_edgar's): Eastmoney is
a **domestic** source that must be reached **directly, bypassing the SOCKS5/HTTP
proxy** the project's ``.env`` injects for US/quote traffic. See :func:`_session`
for why ``trust_env=False`` (not ``proxies={}``) is the mechanism that guarantees
the bypass.

Point-in-time
-------------
The tool filters rows by ``date <= curr_date`` (and within the look-back window)
so a backtest never sees rows the exchange had not yet published on
``curr_date``. Empty ``curr_date`` means live mode (as-of today). The series is
an exchange/Eastmoney daily disclosure with a 1-trading-day publication cadence
— no publication lag is applied (the row date IS the disclosure date).

China A-share only
------------------
A non-A-share ticker raises :class:`NoMarketDataError` (the router turns that
into the ``NO_DATA_AVAILABLE`` sentinel; the analyst's grounding rule reports
"data not available"). In practice the tool is only advertised to the
fundamentals analyst when ``YIAGENTS_A_STOCK`` is on **and** ``is_a_stock(ticker)``
holds, so this is belt-and-suspenders.

Free, keyless. Eastmoney's public web API does not require an API key; it
requests a browser-like ``User-Agent``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from datetime import date, timedelta

import requests

from .config import get_config
from .errors import NoMarketDataError, VendorRateLimitError

logger = logging.getLogger(__name__)

# Public web endpoint (HTTP, domestic — reached directly, never via the
# SOCKS5 VPN proxy that .env sets for US/quote traffic).
_MARGIN_URL = "http://datacenter-web.eastmoney.com/api/data/v1/get"
# Eastmoney's anonymous web token (a fixed public value; not a secret).
_EM_UT = "b2884a393a59ad64002292a3e90d46a5"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_TIMEOUT = 20
# Small serial throttle so concurrent analyst calls stay polite (mirrors
# sec_edgar's spacing). Eastmoney's public endpoints tolerate bursts, but a
# tiny gap keeps a single analysis well under any implicit ceiling.
_MIN_INTERVAL = 0.12
_last_request = [0.0]
_throttle_lock = threading.Lock()
# Transient connection drops (a bare RemoteDisconnected) are retried a couple of
# times with backoff before degrading. Transport resilience only; it lives
# entirely on the opt-in (a_stock) path so default-off runs are unaffected.
_MAX_RETRIES = 2
_RETRY_BACKOFF = 0.8


def _session() -> requests.Session:
    """A Session that bypasses the environment proxy.

    The project's ``.env`` injects ``HTTP_PROXY=socks5h://127.0.0.1:1080`` (and
    ``HTTPS_PROXY``) for US/quote traffic. Eastmoney is a **domestic** source:
    routing it through that SOCKS5 proxy hangs forever (the VPN does not forward
    domestic traffic). ``requests`` reads env proxies in
    ``merge_environment_settings`` *unless* ``trust_env`` is False. Passing
    ``proxies={}`` to ``requests.get`` is NOT sufficient — the env proxy is
    ``setdefault``-ed back into the empty dict and leaks through (verified: a
    bogus env ``HTTP_PROXY`` still yields a ``ProxyError`` with ``proxies={}``).
    ``trust_env=False`` skips the env merge entirely, which is the only reliable
    bypass. ``proxies={}`` is passed too as a belt-and-suspenders match to the
    documented direct-connect contract.
    """
    s = requests.Session()
    s.trust_env = False
    return s


def _throttle() -> None:
    """Enforce a minimum spacing between Eastmoney requests (thread-safe)."""
    with _throttle_lock:
        elapsed = time.time() - _last_request[0]
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request[0] = time.time()


def _cache_dir() -> str:
    cfg = get_config()
    base = cfg.get("data_cache_dir") or os.path.join(
        os.path.expanduser("~"), ".yiagents", "cache")
    path = os.path.join(base, "eastmoney")
    os.makedirs(path, exist_ok=True)
    return path


def _direct_get(url: str, params: dict | None = None) -> bytes:
    """GET an Eastmoney endpoint directly (no env proxy), with throttle + retry.

    Maps 429 -> :class:`VendorRateLimitError`, any other >=400 or unparseable
    transport failure -> :class:`NoMarketDataError`. Transient connection drops
    (a bare ``RemoteDisconnected``) are retried with backoff before degrading.
    Returns the raw response ``bytes``.
    """
    _throttle()
    headers = {"User-Agent": _UA}
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _session().get(
                url, params=params, headers=headers, proxies={}, timeout=_TIMEOUT,
            )
            break
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
                continue
            raise NoMarketDataError(
                url, detail=f"Eastmoney transport failed after retries: {exc!r}"
            ) from exc
    else:  # pragma: no cover -- the loop either breaks or raises above
        raise NoMarketDataError(url, detail=f"Eastmoney transport failed: {last_exc!r}")

    if resp.status_code == 429:
        raise VendorRateLimitError(f"Eastmoney returned 429 for {url}")
    if resp.status_code >= 400:
        raise NoMarketDataError(url, detail=f"Eastmoney HTTP {resp.status_code}")
    return resp.content


def _cached_or_fetch(path: str, url: str, params: dict | None = None,
                     ttl_days: float = 1.0) -> bytes:
    """Serve from a fresh on-disk cache, else fetch + cache.

    Falls back to a stale cache on a fetch failure (a slightly-old daily series
    beats no data). Daily series, so the default TTL is 1 day.
    """
    stale: bytes | None = None
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                stale = fh.read()
        except OSError:
            stale = None
        if stale is not None and (time.time() - os.path.getmtime(path)) < ttl_days * 86_400.0:
            return stale
    try:
        raw = _direct_get(url, params)
    except NoMarketDataError:
        if stale is not None:
            logger.warning("eastmoney: fetch failed; serving stale cache for %s", path)
            return stale
        raise
    try:
        with open(path, "wb") as fh:
            fh.write(raw)
    except OSError as exc:  # noqa: BLE001 -- caching is best-effort
        logger.warning("eastmoney: could not write cache %s: %s", path, exc)
    return raw


# --------------------------------------------------------------------------- #
# Symbol mapping
# --------------------------------------------------------------------------- #
def _to_em_symbol(ticker: str) -> tuple[str, str]:
    """Map an A-share ticker to (secid, scode).

    ``secid`` is Eastmoney's market-prefixed id (``1.<code>`` Shanghai,
    ``0.<code>`` Shenzhen); ``scode`` is the bare 6-digit code used by the
    datacenter ``SCODE`` filter. Accepts the Yahoo-style suffixes ``.SS``/``.SH``
    (Shanghai) and ``.SZ`` (Shenzhen). A non-A-share ticker raises
    :class:`NoMarketDataError` so the router degrades to the optional-category
    sentinel.
    """
    t = (ticker or "").strip().upper()
    code = None
    if t.endswith(".SS") or t.endswith(".SH"):
        code = t.split(".", 1)[0]
        market = "1"
    elif t.endswith(".SZ"):
        code = t.split(".", 1)[0]
        market = "0"
    if not code or not code.isdigit() or len(code) != 6:
        raise NoMarketDataError(
            ticker, detail="Eastmoney vendor is China A-share only (.SS/.SH/.SZ)")
    return f"{market}.{code}", code


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _yi(v) -> float:
    """A CNY amount in 亿元 (divide the raw 元 value by 1e8). None -> nan."""
    n = _num(v)
    return float("nan") if n is None else n / 1e8


def _fmt_yi(v, sign: bool = True) -> str:
    n = _yi(v)
    if n != n:  # nan
        return "n/a"
    return f"{n:+.2f}" if sign else f"{n:.2f}"


# --------------------------------------------------------------------------- #
# Margin trading (融资融券)
# --------------------------------------------------------------------------- #
def _parse_margin(raw: bytes, scode: str) -> list[dict]:
    """Parse the RPTA_WEB_RZRQ_GGMX payload into per-day margin dicts."""
    try:
        j = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NoMarketDataError(
            "margin", detail=f"could not parse margin JSON: {exc}") from exc
    if not j.get("success"):
        # success=False with empty data is "no margin data for this name" (e.g.
        # it is not on the 两融 list) -> treated as honest-empty by the caller.
        return []
    rows = ((j.get("result") or {}).get("data")) or []
    out: list[dict] = []
    for r in rows:
        d = (r.get("DATE") or "")[:10]
        if not d:
            continue
        out.append({
            "date": d,
            "rzye": r.get("RZYE"),      # 融资余额 (元)
            "rqye": r.get("RQYE"),      # 融券余额 (元)
            "rzrqye": r.get("RZRQYE"),  # 融资融券余额 (元)
            "rzmre": r.get("RZMRE"),    # 融资买入额 (元)
            "rqmcl": r.get("RQMCL"),    # 融券卖出量 (股)
            "rzjme": r.get("RZJME"),    # 融资净买入额 (元)
            "rzyezb": r.get("RZYEZB"),  # 融资余额占比 (fraction)
        })
    return out


def get_margin_trading(
    ticker: str, curr_date: str | None = None, look_back_days: int = 180
) -> str:
    """Recent daily 融资融券 (margin trading) for an A-share ticker, PIT-aware.

    融资余额 (margin balance), 融券余额 (short balance), 融资买入额 (margin buy),
    融券卖出量 (short sell, shares), 融资净买入额 (net margin buy). Pulled from
    the Eastmoney datacenter ``RPTA_WEB_RZRQ_GGMX`` report, which carries
    **full exchange-disclosed history** (back to each name's 两融 list date), so
    it is clean for long backtests. Rows are filtered by ``date <= curr_date``
    (PIT) and within the look-back window.

    Non-A-share ticker -> :class:`NoMarketDataError`. A name not on the margin-
    trading list -> an honest empty string (no fabricated balances).
    """
    _, scode = _to_em_symbol(ticker)
    params = {
        "reportName": "RPTA_WEB_RZRQ_GGMX", "columns": "ALL",
        "filter": f'(SCODE="{scode}")',
        "pageNumber": "1", "pageSize": "400",
        "sortColumns": "DATE", "sortTypes": "-1",
        "source": "WEB", "client": "WEB",
    }
    # pageSize 400 (most-recent-first) comfortably covers a 180d look-back plus
    # headroom; margin reports are daily so 400 rows > 1.5 years.
    cache_path = os.path.join(_cache_dir(), f"margin_{scode}.json")
    raw = _cached_or_fetch(cache_path, _MARGIN_URL, params, ttl_days=1.0)
    rows = _parse_margin(raw, scode)

    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    lower_d = upper_d - timedelta(days=int(look_back_days))

    out = io.StringIO()
    out.write(f"# Margin Trading (融资融券) for {ticker} (last {look_back_days} days, "
              f"as of {curr_date or 'now'})\n")
    out.write("# Source: Eastmoney 融资融券 detail (RPTA_WEB_RZRQ_GGMX; exchange-"
              "disclosed daily, full history). 金额 in 亿元 CNY; 融券卖出 in 万股; "
              "融资占比% (RZYEZB, already in percent = 融资余额/流通市值).\n")

    if not rows:
        out.write(f"\nNo margin-trading (融资融券) data for {ticker} as of "
                  f"{curr_date or 'now'} (the name may not be on the margin-"
                  "trading list). Report 'data not available' and do not estimate.")
        return out.getvalue().rstrip("\n")

    # rows are most-recent-first from the API; PIT + look-back filter.
    kept = [r for r in rows if _in_window(r["date"], lower_d, upper_d, upper)]
    if not kept:
        out.write(f"\nNo margin-trading rows for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}.")
        return out.getvalue().rstrip("\n")

    kept.sort(key=lambda r: r["date"], reverse=True)
    out.write(f"# {len(kept)} trading day(s) in window\n\n")
    out.write("Date       | 融资余额 RZYE | 融券余额 RQYE | 融资买入 RZMRE | "
              "融券卖出 RQMCL | 融资净买 RZJME | 融资占比%\n")
    out.write("-" * 96 + "\n")
    for r in kept:
        rzye = _fmt_yi(r["rzye"], sign=False)
        rqye = _fmt_yi(r["rqye"], sign=False)
        rzmre = _fmt_yi(r["rzmre"])
        rqmcl_v = _num(r["rqmcl"])
        rqmcl_s = "n/a" if rqmcl_v is None else f"{rqmcl_v / 1e4:,.2f}"
        rzjme = _fmt_yi(r["rzjme"])
        # RZYEZB is already in percent units (e.g. 1.22 = 1.22% = 融资余额/流通市值),
        # NOT a 0-1 fraction — do not multiply by 100.
        zb = _num(r["rzyezb"])
        zb_s = "n/a" if zb is None else f"{zb:.2f}"
        out.write(
            f"{r['date']:<10} | {rzye:>12} | {rqye:>12} | {rzmre:>12} | "
            f"{rqmcl_s:>12} | {rzjme:>12} | {zb_s}\n"
        )

    # Summary: 融资余额 latest vs window-start, 融资净买入 window sum, tilt.
    latest = kept[0]
    earliest = kept[-1]
    rzjme_sum = sum((_num(r["rzjme"]) or 0.0) for r in kept)
    rz_start = _num(earliest["rzye"])
    rz_end = _num(latest["rzye"])
    rz_end_s = _fmt_yi(latest["rzye"], sign=False) + "亿"
    delta_s = "n/a"
    if rz_start is not None and rz_end is not None:
        delta_s = f"{(rz_end - rz_start) / 1e8:+,.2f}亿"
    tilt = ("rising margin balance (bullish leverage build-up)" if
            rz_start is not None and rz_end is not None and rz_end > rz_start
            else "flat/falling margin balance")
    rqye_s = _fmt_yi(latest["rqye"], sign=False) + "亿"
    out.write(
        f"\nSummary (last {len(kept)}d): 融资余额 {rz_end_s} "
        f"(window delta {delta_s}, {tilt}); 融资净买入 window sum "
        f"{rzjme_sum / 1e8:+,.2f}亿; 融券余额 {rqye_s}."
    )
    return out.getvalue().rstrip("\n")


# --------------------------------------------------------------------------- #
# PIT date helpers
# --------------------------------------------------------------------------- #
def _d(s: str) -> date:
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return date.min


def _in_window(d_str: str, lower_d: date, upper_d: date, upper_set: bool) -> bool:
    """True when ``d_str`` is within [lower_d, upper_d]; upper bound honoured only
    when ``upper_set`` (live mode uses today as the upper bound)."""
    d = _d(d_str)
    if d == date.min:
        return False
    if upper_set and d > upper_d:
        return False
    if d < lower_d:
        return False
    return True
