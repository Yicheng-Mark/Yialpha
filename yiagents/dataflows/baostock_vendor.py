"""BaoStock (证券宝) A-share data vendor — native OHLC + TTM valuation.

A vendor for the optional ``a_share_native`` category, supplying China A-share
data the default yfinance path covers thinly/sparsely: reliable point-in-time
daily OHLC (前复权 qfq) and TTM valuation (peTTM / pbMRQ / psTTM / pcfNcfTTM)
straight from the exchange via BaoStock's TCP API, plus quarterly statement
snapshots (profit / balance / cashflow). This is the substance YiAgents imports
from the reference CN fork's choice of *endpoints* — not its code: the fork's
``AKShareProvider`` is a 1.6k-line async class welded to MongoDB and a global
``requests`` monkey-patch; this module is a fresh, thin, synchronous YiAgents
vendor that calls the ``baostock`` library directly.

Transport (why BaoStock is the cleanest A-share source here)
------------------------------------------------------------
BaoStock is reached over a **raw TCP socket to baostock.com:9001**. It does NOT
use HTTP and does NOT read the ``HTTP_PROXY`` / ``HTTPS_PROXY`` env vars the
project's ``.env`` injects for US/quote traffic. So unlike the HTTP-based
domestic vendors (eastmoney / akshare — see :func:`eastmoney._session` for the
proxy-bypass reasoning those need), **no proxy bypass is required here**: the
domestic socket connects directly and cannot hang on the SOCKS5 VPN tunnel the
way an HTTP request to a domestic host would.

Free, keyless (anonymous ``bs.login()``). ``baostock`` is an **optional
dependency**: it is imported lazily inside each public function, so a
default-off run (``YIAGENTS_A_SHARE_NATIVE`` unset) never imports it — zero
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
to the fundamentals analyst when ``YIAGENTS_A_SHARE_NATIVE`` is on **and**
``is_a_stock(ticker)`` holds, so US / crypto / HK tickers never enter it.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from datetime import date, timedelta

from .config import get_config
from .errors import NoMarketDataError

logger = logging.getLogger(__name__)

# One ``bs.login()`` opens a TCP socket; keep the connect/disconnect off the
# hot path by caching the full daily series per ticker (PIT-filtered per call),
# mirroring how eastmoney caches the raw JSON once and filters per call.
_CACHE_TTL_S = 86_400.0  # 1 day
_login_lock = threading.Lock()
_last_login = [0.0]
# A modest serial spacing between BaoStock logins (the public service asks
# callers not to reconnect aggressively). Per-call queries share one login.
_MIN_LOGIN_INTERVAL = 0.5

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
    """Map a YiAgents A-share ticker to BaoStock's ``sh.600519`` / ``sz.000001``.

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
    cfg = get_config()
    base = cfg.get("data_cache_dir") or os.path.join(
        os.path.expanduser("~"), ".yiagents", "cache")
    path = os.path.join(base, "baostock")
    os.makedirs(path, exist_ok=True)
    return path


class _BaostockSession:
    """Context manager: ``bs.login()`` on enter, ``bs.logout()`` on exit.

    Serializes logins (the public service frowns on reconnect storms) and
    reuses a single login for all queries inside the ``with`` block.
    """

    def __init__(self):
        self.bs = _require_baostock()

    def __enter__(self):
        with _login_lock:
            elapsed = time.time() - _last_login[0]
            if elapsed < _MIN_LOGIN_INTERVAL:
                time.sleep(_MIN_LOGIN_INTERVAL - elapsed)
            _last_login[0] = time.time()
        # login() returns a result object with .error_code / .error_msg; a
        # non-zero code means the TCP handshake to baostock.com:9001 failed.
        lg = self.bs.login()
        if getattr(lg, "error_code", "0") != "0":
            raise NoMarketDataError(
                "baostock", detail=f"login failed: {getattr(lg, 'error_msg', '?')}")
        return self.bs

    def __exit__(self, exc_type, exc, tb):
        try:
            self.bs.logout()
        except Exception:  # noqa: BLE001 -- logout is best-effort
            logger.debug("baostock: logout raised", exc_info=True)
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


def _cached_daily(code: str) -> list[dict]:
    """Serve the daily series from a fresh on-disk cache, else login + fetch.

    Falls back to a stale cache on a fetch failure (a slightly-old daily
    series beats no data), matching eastmoney's stale-on-failure contract.

    .. warning:: Stale-on-failure is a deliberate fail-open trade-off. For
        read-only market data this is reasonable, but a backtest may see a
        slightly-old daily series rather than no data when the BaoStock socket
        is unreachable. The stale-serve IS logged at WARNING level (below) so
        the staleness is observable; callers needing strict freshness can
        suppress it by ensuring the socket is reachable before backtest.
    """
    cache_path = os.path.join(_cache_dir(), f"daily_{code.replace('.', '_')}.json")
    stale = _read_cache(cache_path)
    if stale is not None and (time.time() - os.path.getmtime(cache_path)) < _CACHE_TTL_S:
        return stale
    try:
        with _BaostockSession() as bs:
            rows = _query_daily(bs, code)
    except NoMarketDataError:
        if stale is not None:
            logger.warning("baostock: fetch failed; serving stale cache for %s", code)
            return stale
        raise
    _write_cache(cache_path, rows)
    return rows


def _read_cache(path: str) -> list[dict] | None:
    import json
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_cache(path: str, rows: list[dict]) -> None:
    import json
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
    except OSError as exc:  # noqa: BLE001 -- caching is best-effort
        logger.warning("baostock: could not write cache %s: %s", path, exc)


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
    rows = _cached_daily(code)

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
    rows = _cached_daily(code)

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
def _query_statement(bs, code: str, query_fn_name: str) -> list[dict]:
    """Fetch quarterly statement rows for ``code`` via the given BaoStock query fn.

    Each BaoStock statement query returns rows keyed by ``pubDate`` (disclosure
    date) and ``statDate`` (reporting period). A query error raises
    :class:`NoMarketDataError`; a genuine no-row result returns ``[]``.
    """
    query_fn = getattr(bs, query_fn_name, None)
    if query_fn is None:
        raise NoMarketDataError(
            code, detail=f"BaoStock {query_fn_name} not available")
    rs = query_fn(code=code, year="2025", quarter="3")
    if getattr(rs, "error_code", "0") != "0":
        raise NoMarketDataError(
            code, detail=f"{query_fn_name} failed: "
            f"{getattr(rs, 'error_msg', '?')}")
    rows: list[dict] = []
    while (rs.error_code == "0") and rs.next():
        rows.append(dict(zip(rs.fields, rs.get_row_data(), strict=False)))
    # Also fetch prior periods for trend.
    for year in ("2025", "2024", "2023"):
        for quarter in ("1", "2", "3", "4"):
            if year == "2025" and quarter == "3":
                continue  # already fetched
            try:
                rs2 = query_fn(code=code, year=year, quarter=quarter)
                if getattr(rs2, "error_code", "0") == "0":
                    while (rs2.error_code == "0") and rs2.next():
                        rows.append(dict(zip(rs2.fields, rs2.get_row_data(),
                                             strict=False)))
            except Exception:  # noqa: BLE001
                pass
    # Deduplicate by (pubDate, statDate).
    seen = set()
    unique = []
    for r in rows:
        key = (r.get("pubDate", ""), r.get("statDate", ""))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _statement_rows(code: str, query_fn_name: str) -> list[dict]:
    """Login + fetch quarterly statement rows (cached per code+fn for the session)."""
    try:
        with _BaostockSession() as bs:
            return _query_statement(bs, code, query_fn_name)
    except NoMarketDataError:
        raise


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


def get_a_share_income_statement_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 540,
) -> str:
    """Quarterly income statement (利润表) for an A-share ticker, PIT-aware.

    Revenue, net profit, operating profit, total profit, EPS, ROE etc. from
    BaoStock's ``query_profit_data``, filtered by ``pubDate <= curr_date``
    (point-in-time: a backtest only sees statements the exchange had published
    by ``curr_date``). Returns the most recent 4-8 quarters in a readable
    table. Non-A-share ticker -> :class:`NoMarketDataError`.
    """
    code = _to_baostock_code(ticker)
    rows = _statement_rows(code, "query_profit_data")

    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))

    kept = [r for r in rows
            if _in_window(r.get("pubDate", ""), lower_d, upper_d, upper_set)]
    kept.sort(key=lambda r: r.get("pubDate", ""), reverse=True)

    out = io.StringIO()
    out.write(f"# A-share Income Statement (BaoStock 利润表) for {ticker} "
              f"(as of {curr_date or 'now'})\n")
    out.write("# Source: BaoStock query_profit_data. PIT: pubDate <= curr_date. "
              "Amounts in CNY (亿/万); ratios in %.\n")
    if not kept:
        out.write("\nNo income-statement rows published by this date. Report "
                  "'data not available' and do not estimate revenue or profit.")
        return out.getvalue().rstrip("\n")

    out.write(f"\n# {len(kept)} quarter(s)\n\n")
    out.write("PubDate   | StatDate  | Revenue  | NetProfit | OpProfit  | "
              "ROE%   | EPS\n")
    out.write("-" * 82 + "\n")
    for r in kept[:8]:
        out.write(
            f"{(r.get('pubDate') or '?')[:10]:<10} | "
            f"{(r.get('statDate') or '?')[:10]:<10} | "
            f"{_fmt_cny(r.get('totalShare')):>8} | "
            f"{_fmt_cny(r.get('npParentCompanyOwners')):>9} | "
            f"{_fmt_cny(r.get('operateProfit')):>9} | "
            f"{_f(r.get('roeAvg')):>6} | "
            f"{_f(r.get('epsTTM')):>5}\n"
        )
    latest = kept[0]
    out.write(
        f"\nLatest ({(latest.get('pubDate') or '?')[:10]}, period "
        f"{(latest.get('statDate') or '?')[:10]}): "
        f"net profit {_fmt_cny(latest.get('npParentCompanyOwners'))}, "
        f"ROE {_f(latest.get('roeAvg'))}%, EPS(TTM) {_f(latest.get('epsTTM'))}."
    )
    return out.getvalue().rstrip("\n")


def get_a_share_balance_sheet_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 540,
) -> str:
    """Quarterly balance sheet (资产负债表) for an A-share ticker, PIT-aware.

    Total assets, total liabilities, equity, debt ratio, current ratio etc.
    from BaoStock's ``query_balance_data``, filtered by ``pubDate <= curr_date``.
    Non-A-share ticker -> :class:`NoMarketDataError`.
    """
    code = _to_baostock_code(ticker)
    rows = _statement_rows(code, "query_balance_data")

    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))

    kept = [r for r in rows
            if _in_window(r.get("pubDate", ""), lower_d, upper_d, upper_set)]
    kept.sort(key=lambda r: r.get("pubDate", ""), reverse=True)

    out = io.StringIO()
    out.write(f"# A-share Balance Sheet (BaoStock 资产负债表) for {ticker} "
              f"(as of {curr_date or 'now'})\n")
    out.write("# Source: BaoStock query_balance_data. PIT: pubDate <= curr_date. "
              "Amounts in CNY (亿/万); ratios in %.\n")
    if not kept:
        out.write("\nNo balance-sheet rows published by this date. Report "
                  "'data not available' and do not estimate assets or debt.")
        return out.getvalue().rstrip("\n")

    out.write(f"\n# {len(kept)} quarter(s)\n\n")
    out.write("PubDate   | StatDate  | TotAssets  | TotLiab    | Equity     | "
              "DebtRatio%\n")
    out.write("-" * 80 + "\n")
    for r in kept[:8]:
        out.write(
            f"{(r.get('pubDate') or '?')[:10]:<10} | "
            f"{(r.get('statDate') or '?')[:10]:<10} | "
            f"{_fmt_cny(r.get('totalAssets')):>10} | "
            f"{_fmt_cny(r.get('totalLiab')):>10} | "
            f"{_fmt_cny(r.get('totalShareholdersEquity')):>10} | "
            f"{_f(r.get('liabilityRate')):>8}\n"
        )
    latest = kept[0]
    out.write(
        f"\nLatest ({(latest.get('pubDate') or '?')[:10]}): "
        f"total assets {_fmt_cny(latest.get('totalAssets'))}, "
        f"total liabilities {_fmt_cny(latest.get('totalLiab'))}, "
        f"debt ratio {_f(latest.get('liabilityRate'))}%."
    )
    return out.getvalue().rstrip("\n")


def get_a_share_cashflow_statement_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 540,
) -> str:
    """Quarterly cashflow statement (现金流表) for an A-share ticker, PIT-aware.

    Operating / investing / financing cash flows from BaoStock's
    ``query_cash_flow_data``, filtered by ``pubDate <= curr_date``.
    Non-A-share ticker -> :class:`NoMarketDataError`.
    """
    code = _to_baostock_code(ticker)
    rows = _statement_rows(code, "query_cash_flow_data")

    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))

    kept = [r for r in rows
            if _in_window(r.get("pubDate", ""), lower_d, upper_d, upper_set)]
    kept.sort(key=lambda r: r.get("pubDate", ""), reverse=True)

    out = io.StringIO()
    out.write(f"# A-share Cashflow Statement (BaoStock 现金流表) for {ticker} "
              f"(as of {curr_date or 'now'})\n")
    out.write("# Source: BaoStock query_cash_flow_data. PIT: pubDate <= curr_date. "
              "Amounts in CNY (亿/万).\n")
    if not kept:
        out.write("\nNo cashflow-statement rows published by this date. Report "
                  "'data not available' and do not estimate cash flows.")
        return out.getvalue().rstrip("\n")

    out.write(f"\n# {len(kept)} quarter(s)\n\n")
    out.write("PubDate   | StatDate  | OpCF      | InvCF     | FinCF     | "
              "FreeCF\n")
    out.write("-" * 74 + "\n")
    for r in kept[:8]:
        opcf = _fmt_cny(r.get("netCFOperate"))
        invcf = _fmt_cny(r.get("netCFInvest"))
        fincf = _fmt_cny(r.get("netCFFinance"))
        # Free CF = OpCF - CapEx (approximated by InvCF if CapEx absent)
        out.write(
            f"{(r.get('pubDate') or '?')[:10]:<10} | "
            f"{(r.get('statDate') or '?')[:10]:<10} | "
            f"{opcf:>9} | {invcf:>9} | {fincf:>9} | n/a\n"
        )
    latest = kept[0]
    out.write(
        f"\nLatest ({(latest.get('pubDate') or '?')[:10]}): "
        f"operating CF {_fmt_cny(latest.get('netCFOperate'))}, "
        f"investing CF {_fmt_cny(latest.get('netCFInvest'))}, "
        f"financing CF {_fmt_cny(latest.get('netCFFinance'))}."
    )
    return out.getvalue().rstrip("\n")
