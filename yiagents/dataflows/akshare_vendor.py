"""AKShare (东财/新浪) A-share vendor — native Chinese-market signals.

A vendor for the optional ``a_share_native`` category, supplying A-share
signals the default path (Reddit / StockTwits / yfinance) covers thinly or not
at all, all reached directly (proxy bypassed) and PIT-filtered:

* per-stock Chinese news via ``stock_news_em`` (东财 per-stock feed);
* per-stock daily capital flow (资金流: 主力/超大单/大单/中单/小单 净流入)
  via ``stock_individual_fund_flow``;
* dragon-tiger board appearances (龙虎榜: 净买额/买卖额/上榜原因)
  via ``stock_lhb_detail_em``;
* northbound (Stock Connect / 沪深港通) individual holding
  via ``stock_hsgt_individual_em``;
* sector/industry fund-flow ranking via ``stock_sector_fund_flow_rank``
  (with the stock's own industry resolved via BaoStock's
  ``query_stock_industry`` — a true per-stock mapping);
* real-time spot quote via ``stock_zh_a_spot_em`` (live mode only);
* market breadth (advance-decline) via ``stock_zh_a_spot`` (live mode only).

Like the BaoStock vendor this is a fresh, thin, synchronous YiAgents module
calling the ``akshare`` library directly — not the reference CN fork's
``AKShareProvider`` class (which is welded to MongoDB + a global
``requests`` monkey-patch + sentiment heuristics; sentiment is left to the LLM
news analyst here).

Transport (the one real risk, handled here)
-------------------------------------------
AKShare reaches Eastmoney / Sina over **HTTP** via ``requests``, which reads
``HTTP_PROXY`` / ``HTTPS_PROXY`` from the environment. The project's ``.env``
injects ``HTTP_PROXY=socks5h://127.0.0.1:1080`` for US/quote traffic; routing a
domestic Eastmoney/Sina request through that SOCKS5 VPN **hangs forever** (the
VPN does not forward domestic traffic) — exactly the trap
:func:`eastmoney._session` documents. Unlike eastmoney.py (which owns its
``requests.Session`` and can set ``trust_env=False``), AKShare owns its sessions
internally, so the bypass is done at the **environment** level: the
:func:`_direct_connect` context manager snapshots and pops the proxy env vars
around the AKShare call and restores them in a ``finally``, serialized by a
module lock so concurrent (analyst-parallel) calls cannot clobber each other's
env state.

Free, keyless. ``akshare`` is an **optional dependency**: imported lazily, so a
default-off run never imports it — zero overhead, byte-equivalent. ``curl_cffi``
is an optional anti-bot enhancement (not required); standard ``requests`` +
browser headers usually suffices, and its absence degrades silently.

Point-in-time
-------------
``stock_news_em`` returns the most recent news published by Eastmoney. Items
are filtered by publish date ``<= curr_date`` so a backtest never cites a
headline the analyst could not have seen on ``curr_date``. For a ``curr_date``
in the past, this endpoint has no historical archive, so it will honestly
return few/no items (the right PIT behaviour — do not fabricate). For live
mode (``curr_date`` empty or recent) it returns the latest headlines.

China A-share only
------------------
A non-A-share ticker raises :class:`NoMarketDataError` (the router turns that
into the ``NO_DATA_AVAILABLE`` sentinel). Belt-and-suspenders: the category is
only advertised to the news analyst when ``YIAGENTS_A_SHARE_NATIVE`` is on
**and** ``is_a_stock(ticker)`` holds.
"""

from __future__ import annotations

import io
import logging
import math
import os
import threading
import time
from datetime import date, timedelta

import pandas as pd

from .errors import NoMarketDataError, VendorRateLimitError
from .utils import is_historical_date

logger = logging.getLogger(__name__)

# AKShare calls are serialized so the proxy-env pop/restore in _direct_connect
# cannot interleave under analyst-parallel mode (one call's "popped" window must
# not overlap another's). Serial mode (the default) is unaffected.
#
# Scope of this lock (kept honest after audit): it serializes every
# _direct_connect user in THIS process — all the AKShare calls above, plus the
# tushare_vendor, which borrows _direct_connect for its own domestic HTTP calls
# and therefore also contends here. It is NOT split per-endpoint: the protected
# critical section is the process-wide os.environ mutation (a few dict ops
# around the network call), whose cost is negligible next to the HTTP round
# trip itself, so per-endpoint locks would add complexity without reducing
# any real contention. Serializing unrelated domestic calls costs at most one
# call's latency under analyst-parallel mode (they are already throttled
# upstream); splitting would only matter with many concurrent domestic vendors,
# which this project does not have.
_call_lock = threading.Lock()
_PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)
_TIMEOUT_S = 20


def _require_akshare():
    """Lazy-import akshare or raise a typed error (optional dependency)."""
    try:
        import akshare as ak  # type: ignore
        return ak
    except ImportError as exc:
        raise NoMarketDataError(
            "akshare",
            detail="akshare package not installed (optional dependency for the "
                   "a_share_native category). Install with the 'a-share' extra.",
        ) from exc


class _direct_connect:
    """Context manager: pop proxy env vars + default timeout around a domestic call.

    AKShare's internal ``requests`` calls read ``HTTP_PROXY``/``HTTPS_PROXY``
    from the environment; if those point at the SOCKS5 VPN, requests to
    domestic Eastmoney/Sina hang forever. Popping them for the duration of
    the call forces a direct connection; they are always restored in ``finally``
    (even on error) and the whole window is serialized by ``_call_lock`` so
    concurrent calls are safe. Mirrors the proxy-bypass *intent* of
    :func:`eastmoney._session` (``trust_env=False``) but at the env level, since
    AKShare owns its sessions internally.

    The same window also applies :data:`_TIMEOUT_S` as a default read timeout:
    AKShare exposes no per-call timeout parameter, so without this a half-open
    domestic socket blocks until run_robust's OS-level watchdog kills the
    process. The shim only fills in timeouts for requests that pass none (see
    :mod:`yiagents.dataflows.timeout_shim`).
    """

    def __init__(self):
        self._saved: dict[str, str] = {}

    def __enter__(self):
        _call_lock.acquire()
        for key in _PROXY_ENV_KEYS:
            if key in os.environ:
                self._saved[key] = os.environ.pop(key)
        # Inner import keeps module import light; the patch window is strictly
        # inside the popped-proxy + call-lock window.
        from .timeout_shim import default_request_timeout

        self._timeout_ctx = default_request_timeout(_TIMEOUT_S)
        self._timeout_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore every popped var (unset ones are left unset, matching the
        # pre-call state). Timeout shim unwinds first so the requests patch
        # window is strictly inside the popped-proxy window.
        try:
            self._timeout_ctx.__exit__(exc_type, exc, tb)
        finally:
            for key, val in self._saved.items():
                os.environ[key] = val
            self._saved.clear()
            _call_lock.release()
        return False


def _to_akshare_code(ticker: str) -> str:
    """Map a YiAgents A-share ticker to AKShare's bare 6-digit code.

    AKShare's ``stock_news_em`` takes a bare 6-digit code (``"600519"``), not the
    Yahoo ``.SS``/``.SZ`` suffix. A non-A-share ticker raises
    :class:`NoMarketDataError` so the router degrades to the optional-category
    sentinel.
    """
    t = (ticker or "").strip().upper()
    code = None
    for suf in (".SS", ".SH", ".SZ"):
        if t.endswith(suf):
            code = t[: -len(suf)]
            break
    if not code or not code.isdigit() or len(code) != 6:
        raise NoMarketDataError(
            ticker, detail="AKShare vendor is China A-share only (.SS/.SH/.SZ)")
    return code


def _parse_news_date(s) -> str:
    """Best-effort normalize an AKShare publish-time string to yyyy-mm-dd."""
    if s is None:
        return ""
    return str(s).strip()[:10]


def _in_window(d_str: str, upper_d: date, upper_set: bool) -> bool:
    """True when d_str <= upper_d (PIT gate); empty/unparseable dates are kept
    (the publish-time field is sometimes missing — dropping a real headline for
    a missing timestamp would lose signal; the LLM can judge its relevance)."""
    if not d_str:
        return True
    try:
        d = date.fromisoformat(d_str[:10])
    except ValueError:
        # Intentional fail-open (docstring above), but observable: in a
        # backtest a malformed date could be future data entering the prompt.
        logger.debug("akshare: unparseable date %r kept by PIT window filter", d_str)
        return True
    return not (upper_set and d > upper_d)


def _market_for(ticker: str) -> str:
    """Derive the AKShare market prefix ('sh'/'sz') from a YiAgents A-share ticker.

    ``stock_individual_fund_flow`` takes a bare 6-digit code **and** a market
    prefix (secid = ``{sh:1, sz:0}.{code}``); the suffix YiAgents carries
    (.SS/.SH -> sh, .SZ -> sz) is the source. North Exchange (.BJ) is not in
    the ``is_a_stock`` gate, so it is not mapped here.
    """
    t = (ticker or "").strip().upper()
    if t.endswith((".SS", ".SH")):
        return "sh"
    if t.endswith(".SZ"):
        return "sz"
    # _to_akshare_code already validated the suffix, so this is unreachable for
    # a well-formed A-share ticker; keep the raise for defence.
    raise NoMarketDataError(
        ticker, detail="AKShare vendor is China A-share only (.SS/.SH/.SZ)")


def _akshare_failure(exc: Exception, ticker: str, context: str) -> Exception:
    """Classify a bare akshare transport error into a typed vendor error.

    Rate-limit / anti-bot signals -> :class:`VendorRateLimitError` (router
    skips to the next vendor); everything else -> :class:`NoMarketDataError`
    (router degrades to the optional-category sentinel). Mirrors the news
    vendor's inline classification, factored for reuse.
    """
    low = str(exc).lower()
    if any(k in low for k in ("429", "rate", "频繁", "拒绝")):
        return VendorRateLimitError(f"AKShare {context} throttled: {exc}")
    return NoMarketDataError(ticker, detail=f"AKShare {context} fetch failed: {exc}")


def _fmt_yuan(v) -> str:
    """Format a raw-CNY-元 amount in the A-share convention (亿 / 万), signed."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(f):
        return "n/a"
    af = abs(f)
    if af >= 1e8:
        return f"{f / 1e8:+.2f}亿"
    if af >= 1e4:
        return f"{f / 1e4:+.2f}万"
    return f"{f:+.0f}"


def _fmt_pct(v) -> str:
    """Format a percentage value (AKShare returns e.g. 5.23 for 5.23%)."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(f):
        return "n/a"
    return f"{f:+.2f}%"


def _cell(r, col):
    """Safe DataFrame row access: None when the column is absent."""
    return r[col] if col else None


# --------------------------------------------------------------------------- #
# Public vendor functions
# --------------------------------------------------------------------------- #
def get_a_share_news_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 14,
    limit: int = 20,
) -> str:
    """Recent per-stock Chinese news (东财) for an A-share ticker, PIT-aware.

    Pulled from AKShare's ``stock_news_em`` (Eastmoney per-stock feed), reached
    **directly** (proxy env popped) so the domestic HTTP host cannot hang on the
    SOCKS5 VPN tunnel. Items are filtered by publish date ``<= curr_date``.
    Returns a markdown list of the most recent headlines (title, date, source,
    short snippet). Non-A-share ticker -> :class:`NoMarketDataError`.

    Note: ``stock_news_em`` has no historical archive — for a ``curr_date`` in
    the distant past this honestly returns few/no items (the right PIT
    behaviour; the analyst reports 'no coverage found' rather than fabricating).
    """
    ak = _require_akshare()
    code = _to_akshare_code(ticker)
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    upper_set = bool(upper)

    try:
        with _direct_connect():
            df = ak.stock_news_em(symbol=code)
    except Exception as exc:  # akshare raises bare exceptions on transport blips
        msg = str(exc)
        # Rate-limit / anti-bot signals -> typed rate-limit so the router can
        # skip to the next vendor; everything else -> no-data sentinel.
        low = msg.lower()
        if any(k in low for k in ("429", "rate", "频繁", "拒绝")):
            raise VendorRateLimitError(f"AKShare stock_news_em throttled: {msg}") from exc
        raise NoMarketDataError(ticker, detail=f"AKShare news fetch failed: {msg}") from exc

    out = io.StringIO()
    out.write(f"# A-share News (AKShare / Eastmoney) for {ticker} "
              f"(as of {curr_date or 'now'}, last {look_back_days}d)\n")
    out.write("# Source: AKShare stock_news_em (东财 per-stock feed). Reached "
              "directly (proxy bypassed). Chinese-language headlines.\n")
    if df is None or getattr(df, "empty", True):
        out.write(f"\nNo news items returned for {ticker}. Report 'no coverage "
                  "found' for this symbol and do not fabricate headlines.")
        return out.getvalue().rstrip("\n")

    # Column names AKShare returns (defensive over variants across versions):
    # 新闻标题 / 新闻内容 / 发布时间 / 文章来源 / 新闻链接.
    col_title = _pick(df.columns, ("新闻标题", "标题", "title"))
    col_content = _pick(df.columns, ("新闻内容", "内容", "摘要", "content"))
    col_time = _pick(df.columns, ("发布时间", "时间", "date", "发布日期"))
    col_source = _pick(df.columns, ("文章来源", "来源", "source"))

    rows = []
    for _, r in df.iterrows():
        d_str = _parse_news_date(r.get(col_time)) if col_time else ""
        if not _in_window(d_str, upper_d, upper_set):
            continue
        title = str(r.get(col_title, "") or "").strip()
        if not title:
            continue
        snippet = str(r.get(col_content, "") or "").strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160].rstrip() + "…"
        src = str(r.get(col_source, "") or "").strip() or "东财"
        rows.append((d_str, title, src, snippet))
        if len(rows) >= int(limit):
            break

    if not rows:
        out.write(f"\nNo news items for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. Report "
                  "'no coverage found' and do not fabricate headlines.")
        return out.getvalue().rstrip("\n")

    rows.sort(key=lambda x: x[0], reverse=True)
    out.write(f"\n# {len(rows)} item(s)\n\n")
    for d_str, title, src, snippet in rows:
        when = d_str or "(undated)"
        out.write(f"- **{title}** — {src}, {when}\n  {snippet}\n")
    return out.getvalue().rstrip("\n")


def _pick(columns, candidates) -> str | None:
    """Return the first candidate column name that exists in ``columns``."""
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def get_a_share_money_flow_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 30,
    limit: int = 30,
) -> str:
    """Recent daily capital flow (资金流) for an A-share ticker, PIT-aware.

    主力 / 超大单 / 大单 / 中单 / 小单 net inflow per day, pulled from AKShare's
    ``stock_individual_fund_flow`` (Eastmoney 个股资金流), reached **directly**
    (proxy env popped) so the domestic host cannot hang on the SOCKS5 tunnel.
    Rows are filtered to ``[curr_date - look_back_days, curr_date]`` (point-in-
    time; the endpoint has no date params and returns full history, so both
    bounds are enforced client-side). Returns a per-day table plus a window
    summary (latest main net, window sum, consecutive main-inflow days). A
    persistent positive 主力净流入 = institutional accumulation (bullish); a
    persistent negative = distribution (bearish). Non-A-share ticker ->
    :class:`NoMarketDataError`.
    """
    ak = _require_akshare()
    code = _to_akshare_code(ticker)
    market = _market_for(ticker)
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))

    try:
        with _direct_connect():
            df = ak.stock_individual_fund_flow(stock=code, market=market)
    except Exception as exc:
        raise _akshare_failure(exc, ticker, "stock_individual_fund_flow") from exc

    out = io.StringIO()
    out.write(f"# A-share Money Flow (AKShare / Eastmoney) for {ticker} "
              f"(as of {curr_date or 'now'}, last {look_back_days}d)\n")
    out.write("# Source: AKShare stock_individual_fund_flow (东财 个股资金流). Reached "
              "directly (proxy bypassed). Amounts in CNY (亿/万); 主力=超大单+大单.\n")
    if df is None or getattr(df, "empty", True):
        out.write(f"\nNo money-flow rows returned for {ticker}. Report 'no coverage "
                  "found' for this symbol and do not fabricate flow data.")
        return out.getvalue().rstrip("\n")

    col_date = _pick(df.columns, ("日期", "date"))
    col_main = _pick(df.columns, ("主力净流入-净额", "主力净流入"))
    col_main_pct = _pick(df.columns, ("主力净流入-净占比",))
    col_el = _pick(df.columns, ("超大单净流入-净额",))
    col_lg = _pick(df.columns, ("大单净流入-净额",))
    col_md = _pick(df.columns, ("中单净流入-净额",))
    col_sm = _pick(df.columns, ("小单净流入-净额",))

    rows = []
    for _, r in df.iterrows():
        d_str = _parse_news_date(r[col_date]) if col_date else ""
        try:
            d = date.fromisoformat(d_str) if d_str else None
        except ValueError:
            d = None
        if d is None:
            continue
        if upper_set and d > upper_d:       # PIT: drop post-curr_date
            continue
        if d < lower_d:                     # window lower bound
            continue
        rows.append((d_str, r))

    if not rows:
        out.write(f"\nNo money-flow rows for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. Report "
                  "'no coverage found' and do not fabricate flow data.")
        return out.getvalue().rstrip("\n")

    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[: int(limit)]

    out.write(f"\n# {len(rows)} day(s) in window\n\n")
    out.write("| 日期 | 主力净额 | 主力占比 | 超大单 | 大单 | 中单 | 小单 |\n")
    out.write("|---|---|---|---|---|---|---|\n")
    for d_str, r in rows:
        out.write(
            f"| {d_str} | {_fmt_yuan(_cell(r, col_main))} | "
            f"{_fmt_pct(_cell(r, col_main_pct))} | "
            f"{_fmt_yuan(_cell(r, col_el))} | {_fmt_yuan(_cell(r, col_lg))} | "
            f"{_fmt_yuan(_cell(r, col_md))} | {_fmt_yuan(_cell(r, col_sm))} |\n"
        )

    # Window summary: latest main net, window sum, consecutive main-inflow days.
    def _main(r):
        try:
            return float(_cell(r, col_main))
        except (TypeError, ValueError):
            # Unparseable cells count as 0.0 in win_sum/streak below — a small
            # distortion, so keep a debug trace rather than nothing.
            logger.debug("akshare: unparseable 主力净额 cell counted as 0.0")
            return 0.0

    latest_d, latest_r = rows[0]
    win_sum = sum(_main(r) for _, r in rows)
    streak = 0
    for _, r in rows:  # rows are desc by date -> count trailing inflow days
        if _main(r) > 0:
            streak += 1
        else:
            break
    out.write("\n## 窗口摘要\n")
    out.write(f"- 最近一日({latest_d})主力净流入: {_fmt_yuan(_cell(latest_r, col_main))} "
              f"(占比 {_fmt_pct(_cell(latest_r, col_main_pct))})\n")
    out.write(f"- 窗口({len(rows)}日)累计主力净流入: {_fmt_yuan(win_sum)}\n")
    out.write(f"- 连续主力净流入日数: {streak} (截至 {latest_d})\n")
    return out.getvalue().rstrip("\n")


def get_a_share_dragon_tiger_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 90,
    limit: int = 10,
) -> str:
    """Recent dragon-tiger board (龙虎榜) appearances for an A-share, PIT-aware.

    Pulled from AKShare's ``stock_lhb_detail_em`` (Eastmoney 龙虎榜详情), reached
    **directly** (proxy env popped). The endpoint returns the whole market's
    appearances for a date range; this filters to this stock's code and to
    ``上榜日 <= curr_date`` (point-in-time). For each appearance: 上榜日, 解读,
    上榜原因, 龙虎榜净买额/买入额/卖出额, 净买额占总成交比. A net institutional
    buy-in (positive 净买额, 机构席位 on the buy side) is a bullish smart-money
    signal; a net sell-out is bearish.

    Note: the endpoint also returns ``上榜后1日/2日/5日/10日`` (post-event forward
    returns) — these are deliberately **not** surfaced: for a backtest
    ``curr_date`` they would leak future returns that had not yet been realized.
    Most stocks do not appear on the board in a given window; an honest empty
    result is normal, not an error. Non-A-share ticker -> NoMarketDataError.
    """
    ak = _require_akshare()
    code = _to_akshare_code(ticker)
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))
    start = lower_d.strftime("%Y%m%d")
    end = upper_d.strftime("%Y%m%d")

    try:
        with _direct_connect():
            df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
    except Exception as exc:
        raise _akshare_failure(exc, ticker, "stock_lhb_detail_em") from exc

    out = io.StringIO()
    out.write(f"# A-share Dragon-Tiger Board (AKShare / Eastmoney) for {ticker} "
              f"(as of {curr_date or 'now'}, last {look_back_days}d)\n")
    out.write("# Source: AKShare stock_lhb_detail_em (东财 龙虎榜). Reached directly "
              "(proxy bypassed). 上榜后N日 forward-return columns omitted (lookahead).\n")
    if df is None or getattr(df, "empty", True):
        out.write(f"\nNo dragon-tiger (龙虎榜) appearances for {ticker} in the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. This is normal "
                  "(most stocks do not appear on the board); report 'no dragon-tiger "
                  "activity' and do not fabricate entries.")
        return out.getvalue().rstrip("\n")

    col_code = _pick(df.columns, ("代码",))
    col_date = _pick(df.columns, ("上榜日", "日期", "date"))
    col_explain = _pick(df.columns, ("解读",))
    col_reason = _pick(df.columns, ("上榜原因",))
    col_net = _pick(df.columns, ("龙虎榜净买额",))
    col_buy = _pick(df.columns, ("龙虎榜买入额",))
    col_sell = _pick(df.columns, ("龙虎榜卖出额",))
    col_net_ratio = _pick(df.columns, ("净买额占总成交比",))

    rows = []
    for _, r in df.iterrows():
        if col_code and str(r[col_code]).strip() != code:
            continue
        d_str = _parse_news_date(r[col_date]) if col_date else ""
        try:
            d = date.fromisoformat(d_str) if d_str else None
        except ValueError:
            d = None
        if d is None:
            continue
        if upper_set and d > upper_d:       # PIT belt (end_date already bounds it)
            continue
        if d < lower_d:
            continue
        rows.append((d_str, r))

    if not rows:
        out.write(f"\nNo dragon-tiger (龙虎榜) appearances for {ticker} in the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. This is normal "
                  "(most stocks do not appear on the board); report 'no dragon-tiger "
                  "activity' and do not fabricate entries.")
        return out.getvalue().rstrip("\n")

    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[: int(limit)]

    out.write(f"\n# {len(rows)} appearance(s)\n")
    for d_str, r in rows:
        explain = str(_cell(r, col_explain) or "").strip()
        reason = str(_cell(r, col_reason) or "").strip()
        out.write(f"\n### {d_str} — 净买额 {_fmt_yuan(_cell(r, col_net))} "
                  f"(占总成交 {_fmt_pct(_cell(r, col_net_ratio))})\n")
        if explain:
            out.write(f"- 解读: {explain}\n")
        out.write(f"- 买入额: {_fmt_yuan(_cell(r, col_buy))} / "
                  f"卖出额: {_fmt_yuan(_cell(r, col_sell))}\n")
        if reason and reason != explain:
            out.write(f"- 上榜原因: {reason}\n")
    return out.getvalue().rstrip("\n")


# --------------------------------------------------------------------------- #
# Northbound capital (北向资金 / Stock Connect)
# --------------------------------------------------------------------------- #
def get_a_share_northbound_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 30,
    limit: int = 30,
) -> str:
    """Northbound (Stock Connect) holding for an A-share ticker, PIT-aware.

    Pulls AKShare's ``stock_hsgt_individual_em`` (Eastmoney 沪深港通持股),
    reached **directly** (proxy env popped) so the domestic host cannot hang on
    the SOCKS5 tunnel. Returns the daily northbound share-holding count and
    holding-market-value for *this* stock, plus day-over-day change, filtered to
    ``[curr_date - look_back_days, curr_date]`` (point-in-time; the endpoint
    returns full history so both bounds are enforced client-side). A persistent
    increase in 北向持股 = foreign-institutional accumulation (a major bullish
    signal in the A-share market); a persistent decrease = foreign distribution.

    Non-A-share ticker -> :class:`NoMarketDataError`.
    """
    ak = _require_akshare()
    code = _to_akshare_code(ticker)
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    upper_set = bool(upper)
    lower_d = upper_d - timedelta(days=int(look_back_days))

    try:
        with _direct_connect():
            df = ak.stock_hsgt_individual_em(symbol=code)
    except Exception as exc:
        raise _akshare_failure(exc, ticker, "stock_hsgt_individual_em") from exc

    out = io.StringIO()
    out.write(f"# A-share Northbound Holding (AKShare / Eastmoney) for {ticker} "
              f"(as of {curr_date or 'now'}, last {look_back_days}d)\n")
    out.write("# Source: AKShare stock_hsgt_individual_em (东财 沪深港通持股). Reached "
              "directly (proxy bypassed). 北向持股 = 沪股通/深股通 foreign holding.\n")
    if df is None or getattr(df, "empty", True):
        out.write(f"\nNo northbound holding rows returned for {ticker}. Report 'no "
                  "coverage found' for this symbol and do not fabricate data.")
        return out.getvalue().rstrip("\n")

    col_date = _pick(df.columns, ("持股日期", "日期", "date"))
    col_shares = _pick(df.columns, ("持股数量", "持股股数"))
    col_mkt_val = _pick(df.columns, ("持股市值", "持股金额"))
    col_shares_pct = _pick(df.columns, ("持股数量占发行股", "持股比例"))
    col_mkt_val_pct = _pick(df.columns, ("持股市值占比",))

    rows = []
    for _, r in df.iterrows():
        d_str = _parse_news_date(r[col_date]) if col_date else ""
        try:
            d = date.fromisoformat(d_str) if d_str else None
        except ValueError:
            d = None
        if d is None:
            continue
        if upper_set and d > upper_d:
            continue
        if d < lower_d:
            continue
        rows.append((d_str, r))

    if not rows:
        out.write(f"\nNo northbound holding rows for {ticker} fall within the last "
                  f"{look_back_days} days as of {curr_date or 'now'}. Report 'no "
                  "coverage found' and do not fabricate data.")
        return out.getvalue().rstrip("\n")

    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[: int(limit)]

    out.write(f"\n# {len(rows)} day(s) in window\n\n")
    out.write("| 日期 | 持股数量 | 持股数量占比 | 持股市值 | 持股市值占比 |\n")
    out.write("|---|---|---|---|---|\n")
    for d_str, r in rows:
        out.write(
            f"| {d_str} | {_fmt_shares(_cell(r, col_shares))} | "
            f"{_fmt_pct(_cell(r, col_shares_pct))} | "
            f"{_fmt_yuan(_cell(r, col_mkt_val))} | "
            f"{_fmt_pct(_cell(r, col_mkt_val_pct))} |\n"
        )

    # Window summary: latest vs first.
    latest_d, latest_r = rows[0]
    first_d, first_r = rows[-1]
    out.write("\n## 窗口摘要\n")
    out.write(f"- 最近一日({latest_d})持股: {_fmt_shares(_cell(latest_r, col_shares))} "
              f"(占比 {_fmt_pct(_cell(latest_r, col_shares_pct))})\n")
    try:
        first_s = float(_cell(first_r, col_shares)) if col_shares else 0.0
        latest_s = float(_cell(latest_r, col_shares)) if col_shares else 0.0
        delta = latest_s - first_s
        pct_chg = (delta / first_s * 100) if first_s > 0 else 0.0
        direction = "增持" if delta > 0 else ("减持" if delta < 0 else "持平")
        out.write(f"- 窗口变化: {direction} {_fmt_shares(abs(delta))} ({pct_chg:+.2f}%) "
                  f"(从 {first_d} 到 {latest_d})\n")
    except (TypeError, ValueError):
        pass
    return out.getvalue().rstrip("\n")


def _fmt_shares(v) -> str:
    """Format a share-count in the A-share convention (亿股 / 万股)."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(f):
        return "n/a"
    af = abs(f)
    if af >= 1e8:
        return f"{f / 1e8:.2f}亿股"
    if af >= 1e4:
        return f"{f / 1e4:.2f}万股"
    return f"{f:.0f}股"


# --------------------------------------------------------------------------- #
# Sector / industry fund-flow (板块/行业资金流)
# --------------------------------------------------------------------------- #
def _baostock_industry(ticker: str) -> str | None:
    """Resolve a stock's industry via BaoStock's per-stock industry mapping.

    ``bs.query_stock_industry(code=...)`` (fields delivered server-side:
    updateDate / code / code_name / industry / industryClassification, per the
    official docs) is a true stock -> industry mapping. The previous attempt
    used ``ak.stock_board_industry_name_ths(symbol=...)`` — that function
    takes **no** parameters and returns the industry-board *catalog*
    (name/code), not a per-stock mapping, so the TypeError was swallowed and
    the stock's industry marker never resolved.

    BaoStock is a plain TCP service (no proxy bypass needed) and is already a
    dependency of the ``a_share_native`` category; the login/throttle
    infrastructure is reused from :mod:`yiagents.dataflows.baostock_vendor`.
    Returns ``None`` on any failure or missing row — the caller must then say
    the industry could not be resolved (fail-honest, never fabricate).
    """
    # Ticker-contract errors (non-A-share input) are raised OUTSIDE the
    # best-effort try: they are the caller's contract. Once the ticker is
    # validated, EVERY failure below — including baostock not being installed
    # (NoMarketDataError from _require_baostock, which must NOT be conflated
    # with the ticker contract) — is an industry-lookup miss -> None.
    from .baostock_vendor import _BaostockSession

    bs_code = f"{_market_for(ticker)}.{_to_akshare_code(ticker)}"
    try:
        with _BaostockSession() as bs:
            rs = bs.query_stock_industry(code=bs_code)
        if getattr(rs, "error_code", "0") != "0":
            logger.warning(
                "akshare: baostock query_stock_industry failed for %s: %s",
                bs_code, getattr(rs, "error_msg", "?"),
            )
            return None
        while (rs.error_code == "0") and rs.next():
            row = dict(zip(rs.fields, rs.get_row_data(), strict=False))
            industry = str(row.get("industry", "") or "").strip()
            return industry or None
    except Exception as exc:  # noqa: BLE001 -- best-effort lookup
        logger.warning("akshare: industry lookup failed for %s: %s", ticker, exc)
        return None
    return None


def _stock_industry(ticker: str) -> str | None:
    """Best-effort resolve the stock's industry name (BaoStock mapping)."""
    return _baostock_industry(ticker)


def get_a_share_sector_flow_native(
    ticker: str, curr_date: str | None = None, look_back_days: int = 1,
    limit: int = 30,
) -> str:
    """Industry/sector fund-flow ranking with this stock's sector highlighted.

    Pulls AKShare's ``stock_sector_fund_flow_rank`` (Eastmoney 行业资金流),
    reached **directly** (proxy env popped), plus BaoStock's
    ``query_stock_industry`` to resolve the stock's own industry (a true
    per-stock mapping — see :func:`_baostock_industry` for why the previous
    AKShare board-catalog call never worked). Shows where capital is rotating
    across industries and whether this stock's sector is gaining or losing
    institutional money. The endpoint returns the latest snapshot (no
    historical date param), so rows are not date-filtered — for a historical
    ``curr_date`` this honestly returns the current snapshot with an explicit
    caveat (the sector-flow signal is a real-time/live-mode indicator, not
    reconstructable for backtests).

    Non-A-share ticker -> :class:`NoMarketDataError`.
    """
    ak = _require_akshare()
    code = _to_akshare_code(ticker)

    try:
        with _direct_connect():
            df = ak.stock_sector_fund_flow_rank(
                indicator="今日", sector_type="行业资金流")
    except Exception as exc:
        raise _akshare_failure(exc, ticker, "stock_sector_fund_flow_rank") from exc

    # Resolve the stock's own industry (best-effort; not critical).
    my_sector = _stock_industry(ticker)

    out = io.StringIO()
    out.write(f"# A-share Sector Fund Flow (AKShare / Eastmoney) — {ticker}'s sector\n")
    out.write(f"(snapshot as of {curr_date or 'now'}, stock code {code})\n")
    if my_sector:
        out.write(f"# Stock's industry (BaoStock query_stock_industry): **{my_sector}**\n")
    else:
        out.write("# Stock's industry: could not be resolved — do not guess it\n")
    out.write("# Source: AKShare stock_sector_fund_flow_rank (东财 行业资金流). Reached "
              "directly (proxy bypassed). Positive 主力净流入 = net institutional inflow.\n")
    if curr_date:
        out.write("# NOTE: sector fund-flow is a live snapshot — it cannot be "
                  "reconstructed for a historical date. Treat this as current "
                  "context, not as a point-in-time backtest signal.\n")
    if df is None or getattr(df, "empty", True):
        out.write("\nNo sector fund-flow data returned. Report 'no coverage found' "
                  "and do not fabricate flows.")
        return out.getvalue().rstrip("\n")

    col_sector = _pick(df.columns, ("名称", "行业", "板块"))
    col_main = _pick(df.columns, ("今日主力净流入-净额", "主力净流入-净额",
                                  "今日主力净流入净额", "主力净流入净额"))
    col_main_pct = _pick(df.columns, ("今日主力净流入-净占比", "主力净流入-净占比"))
    col_super = _pick(df.columns, ("今日超大单净流入-净额", "超大单净流入-净额"))
    col_large = _pick(df.columns, ("今日大单净流入-净额", "大单净流入-净额"))

    rows = []
    for _, r in df.iterrows():
        sector_name = str(_cell(r, col_sector) or "").strip()
        if not sector_name:
            continue
        rows.append((sector_name, r))

    if not rows:
        out.write("\nNo sector rows parsed. Report 'no coverage found' and do not "
                  "fabricate flows.")
        return out.getvalue().rstrip("\n")

    # Sort by main net inflow descending to show capital magnets first.
    def _main_val(r):
        try:
            return float(_cell(r, col_main)) if col_main else 0.0
        except (TypeError, ValueError):
            logger.debug("akshare: unparseable sector 主力净额 cell counted as 0.0")
            return 0.0

    rows.sort(key=lambda x: _main_val(x[1]), reverse=True)
    rows = rows[: int(limit)]

    out.write(f"\n# Top {len(rows)} industries by main net inflow\n\n")
    out.write("| 行业 | 主力净额 | 主力占比 | 超大单 | 大单 |\n")
    out.write("|---|---|---|---|---|\n")
    matched = False
    for sector_name, r in rows:
        marker = ""
        if my_sector and sector_name == my_sector:
            marker = " ← **本股所属**"
            matched = True
        out.write(
            f"| {sector_name}{marker} | {_fmt_yuan(_cell(r, col_main))} | "
            f"{_fmt_pct(_cell(r, col_main_pct))} | "
            f"{_fmt_yuan(_cell(r, col_super))} | "
            f"{_fmt_yuan(_cell(r, col_large))} |\n"
        )
    if my_sector and not matched:
        # The two classifications differ (BaoStock industry vs the table's
        # Eastmoney sectors) — say so instead of silently dropping the marker.
        out.write(
            f"\nNOTE: the stock's industry ({my_sector}) has no identically-named "
            "row in the sector table above (industry classification systems "
            "differ); do not force-match it to one of the listed sectors.\n"
        )
    return out.getvalue().rstrip("\n")


# --------------------------------------------------------------------------- #
# Real-time quote (实时行情)
# --------------------------------------------------------------------------- #
def get_a_share_realtime_quote_native(
    ticker: str, curr_date: str | None = None,
) -> str:
    """Real-time spot quote for an A-share ticker (live mode only).

    Pulls AKShare's ``stock_zh_a_spot_em`` (Eastmoney A股实时行情), reached
    **directly** (proxy env popped), and filters to this stock's row. Returns
    latest price, change %, volume, amount, turnover rate, PE (dynamic), etc.
    This is a live snapshot — for a historical ``curr_date`` the function
    honestly returns a sentinel explaining that real-time data is not available
    for past dates (preventing lookahead bias in backtests). ``curr_date`` =
    today (or empty) IS live mode — the framework gate is
    :func:`utils.is_historical_date` (today = live, any other explicit date =
    backtest), so an explicit today no longer misroutes to the sentinel.

    Non-A-share ticker -> :class:`NoMarketDataError`.
    """
    ak = _require_akshare()
    code = _to_akshare_code(ticker)

    # Historical mode (a past/future analysis date): real-time data would leak
    # the future. Today itself is live and allowed.
    if is_historical_date(curr_date):
        out = io.StringIO()
        out.write(f"# A-share Real-Time Quote for {ticker}\n")
        out.write(f"(requested as of {curr_date})\n\n")
        out.write("REAL_TIME_UNAVAILABLE: This is a historical analysis date. "
                  "Real-time spot quotes are only available in live mode and "
                  "cannot be reconstructed for a past date. Use "
                  "get_a_share_ohlc_native for PIT-correct daily OHLCV instead. "
                  "Do not infer or fabricate a real-time quote for this date.")
        return out.getvalue().rstrip("\n")

    try:
        with _direct_connect():
            df = ak.stock_zh_a_spot_em()
    except Exception as exc:
        raise _akshare_failure(exc, ticker, "stock_zh_a_spot_em") from exc

    out = io.StringIO()
    out.write(f"# A-share Real-Time Quote for {ticker}\n")
    out.write("# Source: AKShare stock_zh_a_spot_em (东财 A股实时行情). Reached "
              "directly (proxy bypassed). Live snapshot — no historical archive.\n")
    if df is None or getattr(df, "empty", True):
        out.write("\nNo real-time data returned. Report 'data not available' and do "
                  "not fabricate a quote.")
        return out.getvalue().rstrip("\n")

    col_code = _pick(df.columns, ("代码",))
    col_name = _pick(df.columns, ("名称",))
    col_price = _pick(df.columns, ("最新价",))
    col_pct = _pick(df.columns, ("涨跌幅",))
    col_chg = _pick(df.columns, ("涨跌额",))
    col_vol = _pick(df.columns, ("成交量",))
    col_amt = _pick(df.columns, ("成交额",))
    col_turn = _pick(df.columns, ("换手率",))
    col_pe = _pick(df.columns, ("市盈率-动态", "市盈率",))
    col_pb = _pick(df.columns, ("市净率",))
    col_high = _pick(df.columns, ("最高",))
    col_low = _pick(df.columns, ("最低",))
    col_open = _pick(df.columns, ("今开",))
    col_prev = _pick(df.columns, ("昨收",))

    row = None
    for _, r in df.iterrows():
        if col_code and str(r[col_code]).strip() == code:
            row = r
            break

    if row is None:
        out.write(f"\nStock {code} not found in the real-time spot table. Report "
                  "'data not available' and do not fabricate a quote.")
        return out.getvalue().rstrip("\n")

    name = str(_cell(row, col_name) or "").strip()
    out.write(f"\n## {name} ({code})\n\n")
    out.write(f"- 最新价: {_fmt_num(_cell(row, col_price))}\n")
    out.write(f"- 涨跌幅: {_fmt_pct(_cell(row, col_pct))} (涨跌额 "
              f"{_fmt_num(_cell(row, col_chg))})\n")
    out.write(f"- 今开 / 最高 / 最低 / 昨收: {_fmt_num(_cell(row, col_open))} / "
              f"{_fmt_num(_cell(row, col_high))} / "
              f"{_fmt_num(_cell(row, col_low))} / "
              f"{_fmt_num(_cell(row, col_prev))}\n")
    out.write(f"- 成交量: {_fmt_shares(_cell(row, col_vol))}\n")
    out.write(f"- 成交额: {_fmt_yuan(_cell(row, col_amt))}\n")
    out.write(f"- 换手率: {_fmt_pct(_cell(row, col_turn))}\n")
    out.write(f"- 市盈率(动): {_fmt_num(_cell(row, col_pe))}  市净率: "
              f"{_fmt_num(_cell(row, col_pb))}\n")
    return out.getvalue().rstrip("\n")


def _fmt_num(v) -> str:
    """Format a numeric value to 2 decimals, or 'n/a'."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(f):
        return "n/a"
    return f"{f:.2f}"


# --------------------------------------------------------------------------- #
# Market breadth (市场宽度 / 涨跌家数)
# --------------------------------------------------------------------------- #
# Daily price-limit (涨停/跌停) thresholds by board, in percent. The old flat
# >=9.9% counting mislabeled every 创业板/科创板 limit move (their limit is
# +/-20%) and every ST move (+/-5%). Kept slightly inside the real limits
# (9.9/19.9/4.9) because the spot 涨跌幅 is rounded to 2 decimals.
_MAIN_LIMIT_PCT = 9.9      # 主板 (Shanghai/Shenzhen main boards)
_GEM_STAR_LIMIT_PCT = 19.9  # 创业板 (300/301) + 科创板 (688/681)
_ST_LIMIT_PCT = 4.9        # ST / *ST special-treatment stocks (name contains "ST")


def _limit_thresholds(code: str | None, name: str | None) -> tuple[float, float]:
    """(limit-up, limit-down) threshold pair in percent for one stock row.

    Tiered by board prefix (300/301 ChiNext, 688/681 STAR -> +/-19.9%) and by
    ST-in-name (+/-4.9%) when the spot table carries a 名称 column; everything
    else defaults to the +/-9.9% main-board tier. Beijing Exchange codes
    (8xx/43x/92x, +/-30%) are not part of the ``is_a_stock`` gate and keep the
    main-board tier — noted in the breadth output's threshold legend.

    ``code`` tolerates both the bare 6-digit form (``"300750"``, Eastmoney
    spot) and the exchange-prefixed form (``"sz300750"`` — Sina's 代码 column
    is a positional rename of ``symbol``); only the trailing 6 digits decide.
    """
    digits = "".join(ch for ch in (code or "") if ch.isdigit())[-6:]
    if digits.startswith(("300", "301", "688", "681")):
        return (_GEM_STAR_LIMIT_PCT, -_GEM_STAR_LIMIT_PCT)
    if "ST" in (name or "").upper():
        return (_ST_LIMIT_PCT, -_ST_LIMIT_PCT)
    return (_MAIN_LIMIT_PCT, -_MAIN_LIMIT_PCT)


def _breadth_counts(df) -> dict | None:
    """Aggregate an AKShare spot table into breadth counts.

    Shared by the formatted breadth tool and the market-regime context line,
    so both report identical numbers from the same fetch. Returns ``None``
    when no row parses (caller decides how to degrade). Vectorized (the
    ~5400-row ``iterrows`` scan dominated the tool's wall time); the
    per-symbol limit thresholds still need the code/name columns, so those
    two run as array lookups rather than a Python row loop.
    """
    col_pct = _pick(df.columns, ("涨跌幅", "changepercent"))
    col_code = _pick(df.columns, ("代码", "code", "symbol"))
    col_name = _pick(df.columns, ("名称", "name"))
    pct = pd.to_numeric(df[col_pct], errors="coerce") if col_pct else pd.Series(dtype=float)
    pct = pct.dropna()
    counted = int(len(pct))
    if counted == 0:
        return None
    adv = int((pct > 0.01).sum())
    dec = int((pct < -0.01).sum())
    flat = counted - adv - dec
    codes = (
        [str(v).strip() for v in df.loc[pct.index, col_code].tolist()]
        if col_code else [""] * counted
    )
    names = (
        [str(v).strip() for v in df.loc[pct.index, col_name].tolist()]
        if col_name else [""] * counted
    )
    pct_values = pct.to_numpy(dtype=float)
    limit_up = limit_down = 0
    for code, name, p in zip(codes, names, pct_values, strict=False):
        up_thr, down_thr = _limit_thresholds(code, name)
        if p >= up_thr:
            limit_up += 1
        elif p <= down_thr:
            limit_down += 1
    return {
        "advancing": adv,
        "declining": dec,
        "flat": flat,
        "limit_up": limit_up,
        "limit_down": limit_down,
        "average_change_pct": float(pct.mean()),
        "counted": counted,
        "has_name_column": col_name is not None,
    }


# ---- breadth TTL cache ------------------------------------------------------
# stock_zh_a_spot() is a whole-market snapshot (~5400 rows). A single live
# run asks for it twice (the breadth tool + the regime context line) and the
# LLM may call the tool repeatedly — each fetch re-pulls the full table for
# numbers that change on minute granularity. A 90s process TTL collapses the
# duplicates without serving meaningfully stale breadth.
_BREADTH_TTL_S = 90.0
_breadth_cache: tuple[float, dict | None] | None = None
_breadth_lock = threading.Lock()


def _cached_breadth_counts() -> dict | None:
    """TTL-cached wrapper around the whole-market spot fetch + aggregation."""
    global _breadth_cache
    with _breadth_lock:
        if (
            _breadth_cache is not None
            and time.monotonic() - _breadth_cache[0] < _BREADTH_TTL_S
        ):
            return _breadth_cache[1]
    result = fetch_a_share_breadth_counts()
    with _breadth_lock:
        _breadth_cache = (time.monotonic(), result)
    return result


def reset_breadth_cache_for_test() -> None:
    """Drop the breadth TTL cache (tests only)."""
    global _breadth_cache
    with _breadth_lock:
        _breadth_cache = None


def fetch_a_share_breadth_counts() -> dict | None:
    """Live whole-market breadth counts, shared by the breadth tool and the
    market-regime context line.

    Raises on vendor failure (the caller decides how to degrade — the
    formatted tool converts via ``_akshare_failure``, the fail-soft regime
    line omits the breadth part). Returns ``None`` when the spot table is
    empty or nothing parses. Live data by construction; callers gate
    historical dates themselves. Callers wanting the deduplicated view use
    :func:`_cached_breadth_counts` (90s TTL) instead of re-fetching.
    """
    ak = _require_akshare()
    with _direct_connect():
        df = ak.stock_zh_a_spot()
    if df is None or getattr(df, "empty", True):
        return None
    return _breadth_counts(df)


def get_a_share_market_breadth_native(
    curr_date: str | None = None,
) -> str:
    """A-share market breadth (advance-decline, live mode only).

    Pulls AKShare's ``stock_zh_a_spot`` (A-share real-time spot for the whole
    market), reached **directly** (proxy env popped), and aggregates: count of
    advancing / declining / flat stocks, average change, limit-up / limit-down
    counts. This is a market-level (not per-stock) signal — the breadth of
    participation behind a move.

    Live mode only (``curr_date`` empty or exactly today, per
    :func:`utils.is_historical_date`): for a historical ``curr_date`` the
    function honestly returns a sentinel explaining that real-time breadth
    cannot be reconstructed for a past date (preventing lookahead bias in
    backtests). Limit-up/down counting is tiered by the stock's board and ST
    status (see :func:`_limit_thresholds`), not a one-size 9.9%.
    """
    if is_historical_date(curr_date):
        out = io.StringIO()
        out.write("# A-share Market Breadth\n")
        out.write(f"(requested as of {curr_date})\n\n")
        out.write("REAL_TIME_UNAVAILABLE: This is a historical analysis date. "
                  "Market breadth (advance-decline counts) is a live-snapshot "
                  "signal that cannot be reconstructed for a past date. Do not "
                  "infer or fabricate breadth data for this date.")
        return out.getvalue().rstrip("\n")

    try:
        counts = _cached_breadth_counts()
    except Exception as exc:
        raise _akshare_failure(exc, "", "stock_zh_a_spot") from exc

    out = io.StringIO()
    out.write("# A-share Market Breadth (AKShare)\n")
    out.write("# Source: AKShare stock_zh_a_spot (whole-market real-time). Reached "
              "directly (proxy bypassed). Advance-decline counts are a live signal.\n")
    if counts is None:
        out.write("\nNo market data returned. Report 'data not available' and do "
                  "not fabricate breadth.")
        return out.getvalue().rstrip("\n")

    adv, dec, flat = counts["advancing"], counts["declining"], counts["flat"]
    limit_up, limit_down = counts["limit_up"], counts["limit_down"]
    counted = counts["counted"]
    avg_pct = counts["average_change_pct"]
    out.write(f"\n# Market Breadth ({counted} stocks)\n\n")
    out.write(f"- 上涨: {adv}  |  下跌: {dec}  |  平盘: {flat}\n")
    out.write(f"- 平均涨跌幅: {avg_pct:+.2f}%\n")
    out.write(f"- 涨停: {limit_up}  |  跌停: {limit_down} "
              f"(tiered thresholds: 主板 ±{_MAIN_LIMIT_PCT}% / 创业板·科创板 "
              f"±{_GEM_STAR_LIMIT_PCT}% / ST ±{_ST_LIMIT_PCT}%; 300/301/688/681 "
              "prefixes and ST-in-name detected per row")
    if not counts["has_name_column"]:
        out.write("; NOTE: no 名称 column returned — the ST tier could not be "
                  "applied, so ST stocks near ±5% are NOT counted as limit moves")
    out.write(")\n")
    ad_ratio = adv / dec if dec > 0 else float("inf")
    if dec > 0:
        out.write(f"- 涨跌比 (A/D ratio): {ad_ratio:.2f}\n")
    else:
        out.write("- 涨跌比 (A/D ratio): ∞ (no declining stocks)\n")
    out.write("- Breadth interpretation: A/D > 2 = broad advance (bullish); "
              "< 0.5 = broad decline (bearish); 0.5–2 = mixed.\n")
    return out.getvalue().rstrip("\n")
