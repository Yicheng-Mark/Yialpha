"""BaoStock (证券宝) A-share data vendor — native OHLC + TTM valuation.

A vendor for the optional ``a_share_native`` category, supplying China A-share
data the default yfinance path covers thinly/sparsely: reliable point-in-time
daily OHLC (前复权 qfq) and TTM valuation (peTTM / pbMRQ / psTTM / pcfNcfTTM)
straight from the exchange via BaoStock's TCP API, plus quarterly statement
snapshots (profit / balance / cashflow). This is the substance YiAlpha imports
from the reference CN fork's choice of *endpoints* — not its code: the fork's
``AKShareProvider`` is a 1.6k-line async class welded to MongoDB and a global
``requests`` monkey-patch; this module is a fresh, thin, synchronous YiAlpha
vendor that calls the ``baostock`` library directly.

Transport (why BaoStock is the cleanest A-share source here)
------------------------------------------------------------
BaoStock is reached over a **raw TCP socket to baostock.com:9001**. It does NOT
use HTTP and does NOT read the ``HTTP_PROXY`` / ``HTTPS_PROXY`` env vars the
project's ``.env`` injects for US/quote traffic. So unlike the HTTP-based
domestic vendors (eastmoney / akshare — see :func:`eastmoney._session` for the
proxy-bypass reasoning those need), **no proxy bypass is required here**: the
domestic socket connects directly and cannot hang on the SOCKS5 VPN tunnel the
way an HTTP request to a domestic host would. The requests-level timeout shim
(:mod:`yialpha.dataflows.timeout_shim`) also cannot reach a raw socket — a
hung BaoStock connection is bounded only by run_robust's OS-level watchdog.

Free, keyless (anonymous ``bs.login()``). ``baostock`` is an **optional
dependency**: it is imported lazily inside each public function, so a
default-off run (``YIALPHA_A_SHARE_NATIVE`` unset) never imports it — zero
overhead, byte-equivalent (same opt-in contract as ``binance_gateway``'s SDK
import and ``a_stock`` / ``sec_ownership``).

Point-in-time
-------------
The daily valuation fields (``peTTM`` etc.) are point-in-time by construction
— the value as of that trading day — so no publication lag is applied to the
OHLC / valuation series; rows are simply filtered by ``date <= curr_date``
(within the look-back window). Quarterly statement rows carry a ``pubDate``
and are additionally filtered ``pubDate <= curr_date`` so a backtest never
sees a statement the exchange had not yet published on ``curr_date``.

China A-share only
------------------
A non-A-share ticker raises :class:`NoMarketDataError` (the router turns that
into the ``NO_DATA_AVAILABLE`` sentinel; the analyst's grounding rule reports
"data not available"). Belt-and-suspenders: the category is only advertised
to the fundamentals analyst when ``YIALPHA_A_SHARE_NATIVE`` is on **and**
``is_a_stock(ticker)`` holds, so US / crypto / HK tickers never enter it.
"""

from __future__ import annotations

import io
import json
import logging
import os
import socket
from datetime import date, timedelta
from functools import lru_cache
from typing import NamedTuple

from .baostock_fields import (
    BALANCE_COLUMNS,
    CASH_FLOW_COLUMNS,
    PROFIT_COLUMNS,
    StatementColumn,
)
from .disk_cache import MinIntervalThrottle, cached_or_fetch, vendor_cache_dir
from .errors import NoMarketDataError

logger = logging.getLogger(__name__)

# One ``bs.login()`` opens a TCP socket; keep the connect/disconnect off the
# hot path by caching the full daily series per ticker (PIT-filtered per call),
# mirroring how eastmoney caches the raw JSON once and filters per call.
# Socket timeout (seconds) for the raw TCP session to baostock.com:9001,
# scoped to _BaostockSession's lifetime. BaoStock is not HTTP, so the
# requests-level timeout shim cannot reach it — without this a hung
# connection was bounded only by run_robust's OS-level watchdog.
# YIALPHA_BAOSTOCK_TIMEOUT_S=0 disables it explicitly (0 = no timeout).
_BS_TIMEOUT_ENV = os.environ.get("YIALPHA_BAOSTOCK_TIMEOUT_S")
BS_SOCKET_TIMEOUT: float | None = 30.0
if _BS_TIMEOUT_ENV is not None and _BS_TIMEOUT_ENV != "":
    try:
        _parsed = float(_BS_TIMEOUT_ENV)
        if _parsed > 0:
            BS_SOCKET_TIMEOUT = _parsed
        elif _parsed == 0:
            BS_SOCKET_TIMEOUT = None
        else:
            logger.warning(
                "YIALPHA_BAOSTOCK_TIMEOUT_S=%r is not positive; using the "
                "default 30s instead", _BS_TIMEOUT_ENV,
            )
    except ValueError:
        logger.warning(
            "YIALPHA_BAOSTOCK_TIMEOUT_S=%r is not a number; using the "
            "default 30s instead", _BS_TIMEOUT_ENV,
        )
_CACHE_TTL_S = 86_400.0  # 1 day (historical analysis dates)
# Same-day refresh TTL (seconds): when the analysis date is today (or live
# mode), the series gains the day's bar after the post-close publication, so a
# cache entry written earlier today must not be served all day. Mirrors
# stockstats_utils._needs_same_day_refresh (OHLCV_CACHE_TTL_SECONDS = 900):
# historical dates are immutable and keep the 1-day TTL.
_SAME_DAY_REFRESH_S = 900.0
_login_throttle = MinIntervalThrottle(0.5)

# Valuation/OHLC fields requested in one ``query_history_k_data_plus`` call so
# both the OHLC and fundamentals views are served from a single fetch. adjustflag
# "2" = 前复权 (qfq). peTTM/pbMRQ/psTTM/pcfNcfTTM are already trailing-twelve-
# month / MRQ — they do NOT suffer the single-period inflation the CN fork's
# TTM bugfix doc warns about, because BaoStock computes them server-side.
_DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,"
    "peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)


def _require_baostock():
    """Lazy-import baostock or raise a typed error (optional dependency).

    Raised as :class:`NoMarketDataError` so the optional-category router
    degrades to a sentinel rather than crashing — a machine without the
    optional ``baostock`` extra simply sees "data not available".
    """
    try:
        import baostock as bs  # type: ignore
        return bs
    except ImportError as exc:
        raise NoMarketDataError(
            "baostock",
            detail="baostock package not installed (optional dependency for the "
                   "a_share_native category). Install with the 'a-share' extra.",
        ) from exc


def _to_baostock_code(ticker: str) -> str:
    """Map a YiAlpha A-share ticker to BaoStock's ``sh.600519`` / ``sz.000001``.

    Accepts the Yahoo-style suffixes ``.SS``/``.SH`` (Shanghai) and ``.SZ``
    (Shenzhen) on a 6-digit code. A non-A-share ticker raises
    :class:`NoMarketDataError` so the router degrades to the optional-category
    sentinel.
    """
    t = (ticker or "").strip().upper()
    code = None
    if t.endswith(".SS") or t.endswith(".SH"):
        code = t.split(".", 1)[0]
        prefix = "sh"
    elif t.endswith(".SZ"):
        code = t.split(".", 1)[0]
        prefix = "sz"
    if not code or not code.isdigit() or len(code) != 6:
        raise NoMarketDataError(
            ticker, detail="BaoStock vendor is China A-share only (.SS/.SH/.SZ)")
    return f"{prefix}.{code}"


def _cache_dir() -> str:
    return vendor_cache_dir("baostock")


class _BaostockSession:
    """Context manager: ``bs.login()`` on enter, ``bs.logout()`` on exit.

    Serializes logins (the public service frowns on reconnect storms) and
    reuses a single login for all queries inside the ``with`` block.

    Every network call in this module happens inside this session (login,
    ``query_history_k_data_plus``, the quarterly statement queries, row
    iteration), so the BaoStock socket timeout is scoped to exactly the
    session lifetime: ``socket.setdefaulttimeout`` is set in ``__enter__``
    before login and restored in ``__exit__`` after logout. BaoStock speaks a
    raw TCP protocol to baostock.com:9001 — the requests-level timeout shim
    cannot reach it, so without this a hung connection was bounded only by
    run_robust's OS-level watchdog. Same scoped-setdefault pattern and same
    concurrent-race trade-off as stockstats_utils' ``_scoped_yf_socket_timeout``.
    """

    def __init__(self):
        self.bs = _require_baostock()
        self._prev_timeout: float | None = None

    def __enter__(self):
        # A modest serial spacing between BaoStock logins; per-call queries
        # share one login.
        _login_throttle.wait()
        self._prev_timeout = socket.getdefaulttimeout()
        if BS_SOCKET_TIMEOUT is not None:
            socket.setdefaulttimeout(BS_SOCKET_TIMEOUT)
        try:
            # login() returns a result object with .error_code / .error_msg; a
            # non-zero code means the TCP handshake to baostock.com:9001 failed.
            lg = self.bs.login()
            if getattr(lg, "error_code", "0") != "0":
                raise NoMarketDataError(
                    "baostock",
                    detail=f"login failed: {getattr(lg, 'error_msg', '?')}")
        except BaseException:
            socket.setdefaulttimeout(self._prev_timeout)
            raise
        return self.bs

    def __exit__(self, exc_type, exc, tb):
        try:
            self.bs.logout()
        except Exception:  # noqa: BLE001 -- logout is best-effort
            logger.debug("baostock: logout raised", exc_info=True)
        finally:
            socket.setdefaulttimeout(self._prev_timeout)
        return False


def _query_daily(bs, code: str) -> list[dict]:
    """Fetch the full daily series for ``code`` (qfq, all valuation fields).

    Returns a list of row dicts (string values, as BaoStock yields them). A
    query error raises :class:`NoMarketDataError`; a genuine no-row result
    returns ``[]`` for the caller to render as an honest empty string.
    """
    # Fetch a wide window once per ticker (cheap; one TCP session) and let the
    # PIT filter narrow it per call. ~5y covers any realistic look-back + the
    # full span a backtest curr_date might land in.
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        code, _DAILY_FIELDS, start_date=start, end_date=end,
        frequency="d", adjustflag="2")
    if getattr(rs, "error_code", "0") != "0":
        raise NoMarketDataError(
            code, detail=f"query_history_k_data_plus failed: "
            f"{getattr(rs, 'error_msg', '?')}")
    rows: list[dict] = []
    while (rs.error_code == "0") and rs.next():
        rows.append(dict(zip(rs.fields, rs.get_row_data(), strict=False)))
    return rows


def _daily_cache_ttl_days(curr_date: str | None) -> float:
    """TTL for the per-ticker daily cache, in (fractional) days.

    Historical analysis dates are immutable -> the normal 1-day TTL. When the
    analysis date is today (live mode's default upper bound), the series still
    gains the day's bar after the post-close publication, so the entry is only
    valid for :data:`_SAME_DAY_REFRESH_S` seconds — a run started pre-close
    must not pin the pre-close snapshot for the rest of the day (same contract
    as stockstats_utils' same-day refresh).
    """
    upper = (curr_date or "")[:10]
    try:
        is_today = (not upper) or date.fromisoformat(upper) == date.today()
    except ValueError:
        is_today = False  # malformed date: let the caller's validation handle it
    if is_today:
        return _SAME_DAY_REFRESH_S / 86_400.0
    return _CACHE_TTL_S / 86_400.0


def _cached_daily(code: str, curr_date: str | None = None) -> list[dict]:
    """Serve the daily series from a fresh on-disk cache, else login + fetch.

    ``curr_date`` selects the cache TTL (:func:`_daily_cache_ttl_days`): a
    same-day/live analysis refreshes at most every 15 minutes so the day's
    post-close bar is picked up; a historical analysis keeps the 1-day TTL.

    Thin adapter over the shared :func:`disk_cache.cached_or_fetch` (bytes on
    disk; the row dicts round-trip through JSON). Falls back to a stale cache
    on a fetch failure (a slightly-old daily series beats no data), matching
    every other read-only vendor's stale-on-failure contract.

    .. warning:: Stale-on-failure is a deliberate fail-open trade-off. For
        read-only market data this is reasonable, but a backtest may see a
        slightly-old daily series rather than no data when the BaoStock socket
        is unreachable. The stale serve IS logged at WARNING level (with the
        cache age, by the shared helper) so the staleness is observable;
        callers needing strict freshness can suppress it by ensuring the
        socket is reachable before backtest.
    """

    def _fetch() -> bytes:
        with _BaostockSession() as bs:
            rows = _query_daily(bs, code)
        return json.dumps(rows, ensure_ascii=False).encode("utf-8")

    filename = f"daily_{code.replace('.', '_')}.json"
    raw = cached_or_fetch(
        _cache_dir(), filename, _fetch,
        ttl_days=_daily_cache_ttl_days(curr_date), vendor="baostock",
    )
    assert raw is not None  # fail_open is never set: fetch errors re-raise
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# PIT helpers
# --------------------------------------------------------------------------- #
def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # nan -> None


def _d(s: str) -> date:
    try:
        return date.fromisoformat((s or "")[:10])
    except ValueError:
        return date.min


def _in_window(d_str: str, lower_d: date, upper_d: date, upper_set: bool) -> bool:
    d = _d(d_str)
    if d == date.min:
        return False
    if upper_set and d > upper_d:
        return False
    return not d < lower_d


# --------------------------------------------------------------------------- #
# Public vendor functions
# --------------------------------------------------------------------------- #
def get_a_share_ohlc_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 180
) -> str:
    """Recent daily OHLCV (前复权 qfq) for an A-share ticker, PIT-aware.

    Pulled from BaoStock's exchange daily series (``query_history_k_data_plus``
    with ``adjustflag='2'``); rows are filtered by ``date <= curr_date`` and
    within the look-back window. Non-A-share ticker -> :class:`NoMarketDataError`.
    """
    code = _to_baostock_code(ticker)
    rows = _cached_daily(code, curr_date)

    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    lower_d = upper_d - timedelta(days=int(look_back_days))
    upper_set = bool(upper)

    kept = [r for r in rows if _in_window(r.get("date", ""), lower_d, upper_d, upper_set)]
    kept.sort(key=lambda r: r.get("date", ""), reverse=True)

    out = io.StringIO()
    out.write(f"# A-share OHLC (BaoStock qfq) for {ticker} (last {look_back_days} days, "
              f"as of {curr_date or 'now'})\n")
    out.write("# Source: BaoStock query_history_k_data_plus (前复权). 价格 in CNY; "
              "成交量 in 股; 成交额 in CNY; 换手率%.\n")
    if not kept:
        out.write(f"\nNo daily OHLC rows for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}.")
        return out.getvalue().rstrip("\n")

    out.write(f"# {len(kept)} trading day(s) in window\n\n")
    out.write("Date       | Open    | High    | Low     | Close   | Vol        | "
              "Amount      | Turn%  | Chg%\n")
    out.write("-" * 96 + "\n")
    for r in kept:
        out.write(
            f"{r.get('date',''):<10} | {_f(r.get('open')):>7} | {_f(r.get('high')):>7} | "
            f"{_f(r.get('low')):>7} | {_f(r.get('close')):>7} | "
            f"{(_num(r.get('volume')) or 0):>11,.0f} | "
            f"{(_num(r.get('amount')) or 0):>12,.0f} | "
            f"{_f(r.get('turn')):>6} | {_f(r.get('pctChg')):>5}\n"
        )
    latest = kept[0]
    out.write(
        f"\nSummary (last {len(kept)}d): latest close {_f(latest.get('close'))} CNY "
        f"({(latest.get('pctChg') or 'n/a')}%); latest turn "
        f"{_f(latest.get('turn'))}%; ST flag {latest.get('isST', '?')}."
    )
    return out.getvalue().rstrip("\n")


def get_a_share_fundamentals_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 180
) -> str:
    """Recent daily TTM valuation for an A-share ticker, PIT-aware.

    PE-TTM / PB-MRQ / PS-TTM / PCF-TTM (TTM/MRQ — server-computed, so no
    single-period inflation), plus the ST flag, from BaoStock's daily series
    filtered by ``date <= curr_date``. This is the fundamentals signal the
    default yfinance path supplies sparsely for A-shares; it complements (does
    not replace) the core ``get_fundamentals`` yfinance path. Non-A-share
    ticker -> :class:`NoMarketDataError`.
    """
    code = _to_baostock_code(ticker)
    rows = _cached_daily(code, curr_date)

    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    lower_d = upper_d - timedelta(days=int(look_back_days))
    upper_set = bool(upper)

    kept = [r for r in rows if _in_window(r.get("date", ""), lower_d, upper_d, upper_set)]
    kept.sort(key=lambda r: r.get("date", ""), reverse=True)

    out = io.StringIO()
    out.write(f"# A-share Valuation (BaoStock TTM) for {ticker} (last {look_back_days} days, "
              f"as of {curr_date or 'now'})\n")
    out.write("# Source: BaoStock daily valuation (peTTM / pbMRQ / psTTM / pcfNcfTTM; "
              "server-computed TTM/MRQ). PE/PB/PS/PCF are multiples (x); "
              "turnover in %.\n")
    if not kept:
        out.write(f"\nNo valuation rows for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. Report "
                  "'data not available' and do not estimate multiples.")
        return out.getvalue().rstrip("\n")

    # Sample to at most ~30 rows for readability (quarter-ish cadence over a
    # 180d window), most-recent-first — the LLM does not need all 120 daily
    # rows, just the trend + latest.
    step = max(1, len(kept) // 30)
    shown = kept[::step]
    out.write(f"# {len(kept)} trading day(s) in window; showing {len(shown)} sampled\n\n")
    out.write("Date       | PE-TTM  | PB-MRQ  | PS-TTM  | PCF-TTM | Turn%  | ST\n")
    out.write("-" * 72 + "\n")
    for r in shown:
        out.write(
            f"{r.get('date',''):<10} | {_f(r.get('peTTM')):>7} | {_f(r.get('pbMRQ')):>7} | "
            f"{_f(r.get('psTTM')):>7} | {_f(r.get('pcfNcfTTM')):>7} | "
            f"{_f(r.get('turn')):>6} | {r.get('isST', '?')}\n"
        )

    latest = kept[0]
    earliest = kept[-1]
    pe_now = _num(latest.get("peTTM"))
    pe_then = _num(earliest.get("peTTM"))
    pe_trend = "n/a"
    if pe_now is not None and pe_then is not None and pe_then != 0:
        pe_trend = f"{(pe_now - pe_then):+.1f} over window"
    out.write(
        f"\nSummary (as of {latest.get('date','?')}): PE-TTM {_f(latest.get('peTTM'))}x "
        f"({pe_trend}); PB-MRQ {_f(latest.get('pbMRQ'))}x; PS-TTM "
        f"{_f(latest.get('psTTM'))}x; PCF-TTM {_f(latest.get('pcfNcfTTM'))}x; "
        f"ST flag {latest.get('isST', '?')}. Negative PE = trailing loss."
    )
    return out.getvalue().rstrip("\n")


def _f(v) -> str:
    """Format a numeric field to a fixed-width string; blanks -> 'n/a'."""
    n = _num(v)
    return "n/a" if n is None else f"{n:.2f}"


# --------------------------------------------------------------------------- #
# Quarterly financial statements (利润表 / 资产负债表 / 现金流表)
# --------------------------------------------------------------------------- #
class StatementFetch(NamedTuple):
    """Quarterly statement rows plus the quarters that could NOT be fetched.

    ``failed`` carries ``"YYYYQN"`` labels for per-quarter query errors; the
    renderer must surface them in-band so a truncated revenue/profit trend is
    distinguishable from a genuinely sparse disclosure history.
    """

    rows: list[dict]
    failed: list[str]


def _query_statement(bs, code: str, query_fn_name: str,
                     anchor: date | None = None) -> StatementFetch:
    """Fetch quarterly statement rows for ``code`` via the given BaoStock query fn.

    Each BaoStock statement query returns rows keyed by ``pubDate`` (disclosure
    date) and ``statDate`` (reporting period). The report years are derived
    from ``anchor`` (the PIT analysis date; today in live mode) so the fetched
    window tracks the analysis instead of rotting as calendar years pass —
    rows after ``curr_date`` are pubDate-filtered by the renderer anyway.

    A failure on the FIRST (most recent) quarter raises
    :class:`NoMarketDataError` (fail-closed — the tool's primary row is gone);
    failures on the older trend quarters are logged, collected in
    ``StatementFetch.failed``, and rendered in-band by the caller.
    """
    query_fn = getattr(bs, query_fn_name, None)
    if query_fn is None:
        raise NoMarketDataError(
            code, detail=f"BaoStock {query_fn_name} not available")
    anchor = anchor or date.today()
    years = (str(anchor.year - 2), str(anchor.year - 1), str(anchor.year))

    rows: list[dict] = []
    failed: list[str] = []
    first = True
    # Newest quarter first so the fail-closed gate applies to the row the
    # renderer treats as "latest".
    for year in reversed(years):
        for quarter in ("4", "3", "2", "1"):
            label = f"{year}Q{quarter}"
            try:
                rs = query_fn(code=code, year=year, quarter=quarter)
                if getattr(rs, "error_code", "0") != "0":
                    raise RuntimeError(
                        f"{getattr(rs, 'error_msg', '?')} "
                        f"(error_code {getattr(rs, 'error_code', '?')})")
            except Exception as exc:  # noqa: BLE001 -- one bad quarter must not
                # sink the whole statement, but the gap must stay visible to
                # the agent, not only the log.
                if first:
                    raise NoMarketDataError(
                        code, detail=f"{query_fn_name} failed: {exc}") from exc
                logger.warning(
                    "baostock: %s fetch failed for %s %s: %s",
                    query_fn_name, code, label, exc,
                )
                failed.append(label)
                continue
            first = False
            while (rs.error_code == "0") and rs.next():
                rows.append(dict(zip(rs.fields, rs.get_row_data(), strict=False)))
    # Deduplicate by (pubDate, statDate).
    seen = set()
    unique = []
    for r in rows:
        key = (r.get("pubDate", ""), r.get("statDate", ""))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return StatementFetch(unique, failed)


def _failed_quarters_note(failed: list[str]) -> str:
    """In-band marker for quarters dropped by fetch failures (may be empty)."""
    if not failed:
        return ""
    listed = ", ".join(failed[:8]) + ("…" if len(failed) > 8 else "")
    return (f"\n⚠ {len(failed)} quarter(s) could not be fetched (data source "
            f"error): {listed}. The trend may be incomplete — do not read the "
            "missing quarters as zero or as no disclosure.")


@lru_cache(maxsize=64)
def _statement_rows_cached(code: str, query_fn_name: str,
                           anchor: date) -> StatementFetch:
    """Cached backing for :func:`_statement_rows` (bounded, session-scoped).

    One statement fetch costs a TCP login plus up to 12 per-quarter queries;
    the fundamentals analyst calls the same statement tool several times per
    run, so the rows are cached per (code, fn, analysis-year anchor). Bounded
    at 64 entries so a long multi-ticker backtest cannot grow it unbounded.
    The returned ``StatementFetch`` (and its row dicts) are shared and must
    not be mutated by callers — renderers only read.
    """
    try:
        with _BaostockSession() as bs:
            return _query_statement(bs, code, query_fn_name, anchor)
    except NoMarketDataError:
        raise


def _statement_rows(code: str, query_fn_name: str,
                    anchor: date | None = None) -> StatementFetch:
    """Login + fetch quarterly statement rows (cached per code+fn for the session)."""
    return _statement_rows_cached(code, query_fn_name, anchor or date.today())


def _fmt_cny(v) -> str:
    """Format a CNY amount in 亿 / 万 (for statement line items)."""
    n = _num(v)
    if n is None:
        return "n/a"
    af = abs(n)
    if af >= 1e8:
        return f"{n / 1e8:.2f}亿"
    if af >= 1e4:
        return f"{n / 1e4:.2f}万"
    return f"{n:.2f}"


def _fmt_cell(row: dict, field: str, kind: str) -> str:
    """Format one statement cell per its column spec kind.

    ``pct``: BaoStock ratio fields are decimal fractions (official sample
    ``roeAvg 0.074617``), so multiply by 100 and append "%" (7.46%). ``x``:
    dimensionless multiples, rendered with an "x" suffix. ``cny``: raw-CNY
    amounts / share counts via :func:`_fmt_cny`. ``num``: plain 2-decimal.
    Missing/blank fields (the endpoint leaves not-yet-disclosed cells empty)
    render as "n/a" — an honest gap, never a zero.
    """
    n = _num(row.get(field))
    if n is None:
        return "n/a"
    if kind == "pct":
        return f"{n * 100:.2f}%"
    if kind == "x":
        return f"{n:.2f}x"
    if kind == "cny":
        return _fmt_cny(n)
    return f"{n:.2f}"


def _render_statement_table(kept: list[dict],
                            columns: tuple[StatementColumn, ...]) -> str:
    """Render the PubDate/StatDate + spec-column table for a statement view."""
    header = ("PubDate   | StatDate  | "
              + " | ".join(f"{c.label:>11}" for c in columns) + "\n")
    out = io.StringIO()
    out.write(header)
    out.write("-" * (24 + 14 * len(columns)) + "\n")
    for r in kept[:8]:
        cells = " | ".join(f"{_fmt_cell(r, c.field, c.kind):>11}" for c in columns)
        out.write(
            f"{(r.get('pubDate') or '?')[:10]:<10} | "
            f"{(r.get('statDate') or '?')[:10]:<10} | {cells}\n"
        )
    return out.getvalue()


def get_a_share_income_statement_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 540,
) -> str:
    """Quarterly profitability statement (利润表/盈利能力) for an A-share, PIT-aware.

    Revenue (MBRevenue 主营营业收入), net profit, gross/net margin, ROE, EPS
    (TTM) and share counts from BaoStock's ``query_profit_data`` — the field
    set pinned in ``baostock_fields.PROFIT_DATA_FIELDS`` (ratio fields are
    decimal fractions and are rendered as percentages). Rows are filtered by
    ``pubDate <= curr_date`` (point-in-time: a backtest only sees statements
    the exchange had published by ``curr_date``). Returns the most recent 4-8
    quarters in a readable table. Non-A-share ticker ->
    :class:`NoMarketDataError`.
    """
    code = _to_baostock_code(ticker)
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    rows, failed = _statement_rows(code, "query_profit_data", upper_d)

    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))

    kept = [r for r in rows
            if _in_window(r.get("pubDate", ""), lower_d, upper_d, upper_set)]
    kept.sort(key=lambda r: r.get("pubDate", ""), reverse=True)

    out = io.StringIO()
    out.write(f"# A-share Income Statement (BaoStock 利润表) for {ticker} "
              f"(as of {curr_date or 'now'})\n")
    out.write("# Source: BaoStock query_profit_data (盈利能力: MBRevenue/netProfit/"
              "gpMargin/npMargin/roeAvg/epsTTM/totalShare/liqaShare). PIT: pubDate <= "
              "curr_date. Amounts in CNY (亿/万); margins/ROE in % (source fractions "
              "x100). No operating-profit field is returned by this endpoint.\n")
    if not kept:
        out.write("\nNo income-statement rows published by this date. Report "
                  "'data not available' and do not estimate revenue or profit.")
        out.write(_failed_quarters_note(failed))
        return out.getvalue().rstrip("\n")

    out.write(f"\n# {len(kept)} quarter(s)\n\n")
    out.write(_render_statement_table(kept, PROFIT_COLUMNS))
    latest = kept[0]
    out.write(
        f"\nLatest ({(latest.get('pubDate') or '?')[:10]}, period "
        f"{(latest.get('statDate') or '?')[:10]}): "
        f"revenue {_fmt_cell(latest, 'MBRevenue', 'cny')}, "
        f"net profit {_fmt_cell(latest, 'netProfit', 'cny')}, "
        f"ROE {_fmt_cell(latest, 'roeAvg', 'pct')}, "
        f"EPS(TTM) {_fmt_cell(latest, 'epsTTM', 'num')}."
    )
    out.write(_failed_quarters_note(failed))
    return out.getvalue().rstrip("\n")


def get_a_share_balance_sheet_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 540,
) -> str:
    """Quarterly solvency ratios (偿债能力) for an A-share ticker, PIT-aware.

    BaoStock's ``query_balance_data`` returns **ratios only** — current/quick/
    cash ratio, liability YoY growth, liability-to-asset, asset-to-equity —
    and NO balance-sheet stocks (no totalAssets / totalLiabilities / equity
    levels exist in this endpoint; those were previously fabricated column
    reads that rendered all-n/a). This tool therefore honestly presents the
    solvency-ratio table the endpoint actually provides; do not estimate
    total assets or debt levels from it. Rows filtered by
    ``pubDate <= curr_date``. Non-A-share ticker -> NoMarketDataError.
    """
    code = _to_baostock_code(ticker)
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    rows, failed = _statement_rows(code, "query_balance_data", upper_d)

    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))

    kept = [r for r in rows
            if _in_window(r.get("pubDate", ""), lower_d, upper_d, upper_set)]
    kept.sort(key=lambda r: r.get("pubDate", ""), reverse=True)

    out = io.StringIO()
    out.write(f"# A-share Balance Sheet (BaoStock 偿债能力比率) for {ticker} "
              f"(as of {curr_date or 'now'})\n")
    out.write("# Source: BaoStock query_balance_data (solvency ratios: currentRatio/"
              "quickRatio/cashRatio/YOYLiability/liabilityToAsset/assetToEquity). "
              "PIT: pubDate <= curr_date. This endpoint returns RATIOS ONLY — no "
              "total-assets/liabilities/equity levels; report those as 'data not "
              "available' rather than estimating. Ratios in x; %-fields are source "
              "fractions x100.\n")
    if not kept:
        out.write("\nNo solvency-ratio rows published by this date. Report "
                  "'data not available' and do not estimate assets or debt.")
        out.write(_failed_quarters_note(failed))
        return out.getvalue().rstrip("\n")

    out.write(f"\n# {len(kept)} quarter(s)\n\n")
    out.write(_render_statement_table(kept, BALANCE_COLUMNS))
    latest = kept[0]
    out.write(
        f"\nLatest ({(latest.get('pubDate') or '?')[:10]}): "
        f"current ratio {_fmt_cell(latest, 'currentRatio', 'x')}, "
        f"liability-to-asset {_fmt_cell(latest, 'liabilityToAsset', 'pct')}, "
        f"asset-to-equity {_fmt_cell(latest, 'assetToEquity', 'x')}."
    )
    out.write(_failed_quarters_note(failed))
    return out.getvalue().rstrip("\n")


def get_a_share_cashflow_statement_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 540,
) -> str:
    """Quarterly cash-flow quality ratios (现金流量) for an A-share, PIT-aware.

    BaoStock's ``query_cash_flow_data`` returns **ratios only** —
    CAToAsset/NCAToAsset/tangibleAssetToAsset, ebitToInterest (interest
    coverage), CFOToOR / CFOToNP / CFOToGr — and NO absolute operating/
    investing/financing cash-flow amounts (those were previously fabricated
    column reads that rendered all-n/a). This tool honestly presents the
    quality-ratio table the endpoint provides; do not estimate absolute cash
    flows from it. Rows filtered by ``pubDate <= curr_date``. Non-A-share
    ticker -> :class:`NoMarketDataError`.
    """
    code = _to_baostock_code(ticker)
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    rows, failed = _statement_rows(code, "query_cash_flow_data", upper_d)

    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))

    kept = [r for r in rows
            if _in_window(r.get("pubDate", ""), lower_d, upper_d, upper_set)]
    kept.sort(key=lambda r: r.get("pubDate", ""), reverse=True)

    out = io.StringIO()
    out.write(f"# A-share Cashflow Statement (BaoStock 现金流质量比率) for {ticker} "
              f"(as of {curr_date or 'now'})\n")
    out.write("# Source: BaoStock query_cash_flow_data (quality ratios: CAToAsset/"
              "NCAToAsset/tangibleAssetToAsset/ebitToInterest/CFOToOR/CFOToNP/"
              "CFOToGr). PIT: pubDate <= curr_date. This endpoint returns RATIOS "
              "ONLY — no absolute operating/investing/financing cash amounts; "
              "report those as 'data not available' rather than estimating.\n")
    if not kept:
        out.write("\nNo cashflow-ratio rows published by this date. Report "
                  "'data not available' and do not estimate cash flows.")
        out.write(_failed_quarters_note(failed))
        return out.getvalue().rstrip("\n")

    out.write(f"\n# {len(kept)} quarter(s)\n\n")
    out.write(_render_statement_table(kept, CASH_FLOW_COLUMNS))
    latest = kept[0]
    out.write(
        f"\nLatest ({(latest.get('pubDate') or '?')[:10]}): "
        f"CFO/Revenue {_fmt_cell(latest, 'CFOToOR', 'pct')}, "
        f"CFO/NetProfit {_fmt_cell(latest, 'CFOToNP', 'x')}, "
        f"interest coverage (EBIT/Interest) {_fmt_cell(latest, 'ebitToInterest', 'x')}."
    )
    out.write(_failed_quarters_note(failed))
    return out.getvalue().rstrip("\n")
