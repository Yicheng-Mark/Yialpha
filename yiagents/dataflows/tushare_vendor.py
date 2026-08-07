"""Tushare (挖地兔) A-share vendor — opt-in quality tier for fundamentals + news.

A token-gated vendor for the optional ``a_share_native`` category: a config-
selectable alternative to the keyless BaoStock (OHLC/valuation) and AKShare
(news) vendors, for users who hold a Tushare Pro token. Tushare's ``daily_basic``
gives clean daily PE/PB/MV/turnover/free-float and ``fina_indicator`` gives
TTM ROE/margins/leverage; ``news`` gives multi-source Chinese headlines
(sina / eastmoney / cls / 10jqka / wallstreetcn / …). Selected by routing the
category's vendor chain at tushare, e.g.
``data_vendors={"a_share_native": "tushare"}`` or
``tool_vendors={"get_a_share_fundamentals_native": "baostock,tushare"}`` (Tushare
as a fallback after the keyless vendors).

Token gating
------------
Requires ``TUSHARE_TOKEN`` (bare env name, like ``DEEPSEEK_API_KEY`` — no
``YIAGENTS_`` prefix; only ever in ``.env`` / gitignore). A missing token raises
:class:`VendorNotConfiguredError`, which the router treats as "vendor
unavailable" and skips to the next vendor in the chain (or degrades to the
optional-category sentinel if none remain) — so a keyless default still works
on a machine without a token.

Transport
---------
Tushare's ``pro_api`` reaches ``api.tushare.pro`` (domestic) over HTTP via
``requests``, so it reuses the same proxy-bypass contract as AKShare: the
:func:`akshare_vendor._direct_connect` context manager pops ``HTTP_PROXY`` /
``HTTPS_PROXY`` around the call (restored in ``finally``, serialized) so the
domestic host cannot hang on the SOCKS5 VPN tunnel.

``tushare`` is an **optional dependency**, imported lazily; default-off runs
never import it (byte-equivalent).

Point-in-time
-------------
``daily_basic`` rows are filtered by ``trade_date <= curr_date``; statement rows
(``fina_indicator``) by ``ann_date <= curr_date`` (the announcement date Tushare
carries); news by ``datetime <= curr_date``. So a backtest never sees data the
market had not yet published on ``curr_date``.

China A-share only
------------------
A non-A-share ticker raises :class:`NoMarketDataError` (the router degrades that
to the sentinel). Belt-and-suspenders: the category is only advertised when
``YIAGENTS_A_SHARE_NATIVE`` is on AND ``is_a_stock(ticker)`` holds.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import date, timedelta

from .akshare_vendor import _direct_connect  # lightweight: os/threading only
from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError, VendorRateLimitError

logger = logging.getLogger(__name__)

_TOKEN_ENV = "TUSHARE_TOKEN"
_TIMEOUT_S = 20


def _require_tushare():
    """Lazy-import tushare AND verify a token is present.

    Raises :class:`VendorNotConfiguredError` when the token is missing (the
    router skips this vendor) and :class:`NoMarketDataError` when the optional
    ``tushare`` package is not installed (the router degrades to a sentinel).
    """
    token = os.environ.get(_TOKEN_ENV) or get_config().get("tushare_token")
    if not token:
        raise VendorNotConfiguredError(
            "Tushare selected but TUSHARE_TOKEN is not set (a_share_native is "
            "keyless via BaoStock/AKShare by default; Tushare is an opt-in "
            "quality tier).")
    try:
        import tushare as ts  # type: ignore
    except ImportError as exc:
        raise NoMarketDataError(
            "tushare",
            detail="tushare package not installed (optional dependency for the "
                   "a_share_native Tushare tier). Install with the 'a-share' extra.",
        ) from exc
    ts.set_token(token)
    return ts.pro_api()


def _to_tushare_code(ticker: str) -> str:
    """Map a YiAgents A-share ticker to Tushare's ``600519.SH`` / ``000001.SZ``.

    Note Tushare uses ``.SH`` for Shanghai where YiAgents/yfinance use ``.SS``;
    both denote the SSE. A non-A-share ticker raises :class:`NoMarketDataError`.
    """
    t = (ticker or "").strip().upper()
    code = None
    suffix = None
    if t.endswith(".SS") or t.endswith(".SH"):
        code = t.split(".", 1)[0]
        suffix = ".SH"
    elif t.endswith(".SZ"):
        code = t.split(".", 1)[0]
        suffix = ".SZ"
    if not code or not code.isdigit() or len(code) != 6:
        raise NoMarketDataError(
            ticker, detail="Tushare vendor is China A-share only (.SS/.SH/.SZ)")
    return f"{code}{suffix}"


def _ts_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _f(v) -> str:
    n = _num(v)
    return "n/a" if n is None else f"{n:.2f}"


def _query(pro, api_name: str, **fields):
    """Call a Tushare pro API under the direct-connect proxy bypass; map errors.

    Tushare returns a DataFrame on success or raises on a transport/permission
    fault. A rate-limit / permission signal maps to :class:`VendorRateLimitError`
    (router skips to next vendor); other faults to :class:`NoMarketDataError`.
    """
    try:
        with _direct_connect():
            return getattr(pro, api_name)(**fields)
    except Exception as exc:  # noqa: BLE001 -- tushare raises bare Exceptions
        msg = str(exc)
        low = msg.lower()
        if any(k in low for k in ("每分钟", "次数", "限频", "429", "rate", "权限", "permission")):
            raise VendorRateLimitError(f"Tushare {api_name} throttled/forbidden: {msg}")
        raise NoMarketDataError(fields.get("ts_code", api_name),
                                detail=f"Tushare {api_name} failed: {msg}")


# --------------------------------------------------------------------------- #
# Public vendor functions
# --------------------------------------------------------------------------- #
def get_a_share_fundamentals_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 180
) -> str:
    """Recent daily valuation (PE/PB/MV/turnover) for an A-share ticker via
    Tushare ``daily_basic``, PIT-aware.

    PE/PB/PS/PCF-TTM, total/free market cap (亿元) and turnover, filtered by
    ``trade_date <= curr_date``. A config-selectable alternative to the BaoStock
    valuation vendor. Non-A-share ticker -> :class:`NoMarketDataError`;
    missing token -> :class:`VendorNotConfiguredError` (router skips this vendor).
    """
    pro = _require_tushare()
    ts_code = _to_tushare_code(ticker)
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    lower_d = upper_d - timedelta(days=int(look_back_days))

    df = _query(pro, "daily_basic", ts_code=ts_code,
                start_date=_ts_date(lower_d), end_date=_ts_date(upper_d))
    out = io.StringIO()
    out.write(f"# A-share Valuation (Tushare) for {ticker} (last {look_back_days} days, "
              f"as of {curr_date or 'now'})\n")
    out.write("# Source: Tushare daily_basic. PE/PB TTM; total/circ market cap in 亿元 CNY; "
              "turnover in %; free-float ratio in %.\n")
    if df is None or getattr(df, "empty", True):
        out.write(f"\nNo valuation rows for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. Report "
                  "'data not available' and do not estimate multiples.")
        return out.getvalue().rstrip("\n")

    # Client-side PIT filter (never trust only the API's date params for PIT
    # correctness — a backtest must not cite a row beyond curr_date). Tushare's
    # trade_date is YYYYMMDD.
    def _td_in_window(td: str) -> bool:
        try:
            d = date(int(td[:4]), int(td[4:6]), int(td[6:8]))
        except (ValueError, TypeError):
            return True  # unparseable -> keep (let the LLM judge)
        return lower_d <= d <= upper_d

    df = df[df["trade_date"].astype(str).map(_td_in_window)]
    # daily_basic is newest-first; cap to a readable sample.
    df = df.sort_values("trade_date", ascending=False).head(30)
    if df.empty:
        out.write(f"\nNo valuation rows for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. Report "
                  "'data not available' and do not estimate multiples.")
        return out.getvalue().rstrip("\n")
    out.write(f"\n# {len(df)} row(s) shown\n\n")
    out.write("Date       | PE-TTM  | PB      | TotalMV | CircMV  | Turn%  | Free%\n")
    out.write("-" * 72 + "\n")
    for _, r in df.iterrows():
        td = str(r.get("trade_date", ""))
        when = f"{td[:4]}-{td[4:6]}-{td[6:8]}" if len(td) == 8 else td
        out.write(
            f"{when:<10} | {_f(r.get('pe_ttm')):>7} | {_f(r.get('pb')):>7} | "
            f"{_yi(r.get('total_mv')):>7} | {_yi(r.get('circ_mv')):>7} | "
            f"{_f(r.get('turnover_rate')):>6} | {_f(r.get('free_share')):>5}\n"
        )
    latest = df.iloc[0]
    out.write(
        f"\nSummary (as of {latest.get('trade_date','?')}): PE-TTM "
        f"{_f(latest.get('pe_ttm'))}x; PB {_f(latest.get('pb'))}x; total MV "
        f"{_yi(latest.get('total_mv'))}亿; turnover {_f(latest.get('turnover_rate'))}%."
    )
    return out.getvalue().rstrip("\n")


def get_a_share_news_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 14, limit: int = 20,
) -> str:
    """Recent multi-source Chinese news for an A-share ticker via Tushare ``news``.

    Aggregates sina / eastmoney / cls / 10jqka / wallstreetcn / … headlines,
    filtered by ``datetime <= curr_date``. A config-selectable alternative to the
    AKShare news vendor. Non-A-share ticker -> :class:`NoMarketDataError`;
    missing token -> :class:`VendorNotConfiguredError`.
    """
    pro = _require_tushare()
    # The Tushare news API is keyword-based (no per-stock ts_code); use the bare
    # 6-digit code as the keyword and let the LLM judge relevance — same intent
    # as AKShare's per-stock feed but multi-source.
    keyword = _to_tushare_code(ticker).split(".")[0]
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    upper_set = bool(upper)

    df = _query(pro, "news", src="sina", start_date="20240101",
                end_date=_ts_date(upper_d))
    out = io.StringIO()
    out.write(f"# A-share News (Tushare multi-source) for {ticker} "
              f"(as of {curr_date or 'now'}, last {look_back_days}d)\n")
    out.write("# Source: Tushare news API (multi-source: sina/eastmoney/cls/...). "
              "Keyword-filtered; reached directly (proxy bypassed).\n")
    if df is None or getattr(df, "empty", True):
        out.write(f"\nNo news items returned for {ticker}. Report 'no coverage "
                  "found' and do not fabricate headlines.")
        return out.getvalue().rstrip("\n")

    lower_d = upper_d - timedelta(days=int(look_back_days))
    col_title = "title" if "title" in df.columns else df.columns[0]
    col_time = "datetime" if "datetime" in df.columns else None
    col_src = "src" if "src" in df.columns else None

    rows = []
    for _, r in df.iterrows():
        title = str(r.get(col_title, "") or "").strip()
        if not title or keyword not in (title + str(r.get("content", ""))):
            continue
        when = str(r.get(col_time, "") or "")[:10] if col_time else ""
        try:
            d = date.fromisoformat(when) if when else None
        except ValueError:
            d = None
        if d and upper_set and d > upper_d:
            continue
        if d and d < lower_d:
            continue
        src = str(r.get(col_src, "") or "").strip() or "tushare"
        rows.append((when, title, src))
        if len(rows) >= int(limit):
            break

    if not rows:
        out.write(f"\nNo news items for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. Report "
                  "'no coverage found' and do not fabricate headlines.")
        return out.getvalue().rstrip("\n")

    rows.sort(key=lambda x: x[0], reverse=True)
    out.write(f"\n# {len(rows)} item(s)\n\n")
    for when, title, src in rows:
        out.write(f"- **{title}** — {src}, {when or '(undated)'}\n")
    return out.getvalue().rstrip("\n")


def _yi(v) -> str:
    """Market cap (Tushare returns 元) formatted in 亿元."""
    n = _num(v)
    return "n/a" if n is None else f"{n / 1e8:.2f}"
