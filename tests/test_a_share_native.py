"""Unit tests for ``yialpha.dataflows.baostock_vendor`` and the
fundamentals-analyst wiring of the ``a_share_native`` category (Track A,
native A-share OHLC + TTM valuation).

Hermetic: no network, no ``baostock`` install required. The vendor paths are
exercised by patching ``baostock_vendor._cached_daily`` to serve synthetic
daily rows. Symbol mapping, PIT filtering (date <= curr_date), the non-A-share
contract, and router integration round out the dataflow coverage.

The wiring tests pin the byte-equivalence contract (the project "iron rule"):
with ``a_share_native`` off (the default) the fundamentals analyst binds exactly
its baseline tools and an unchanged prompt; with it on AND an A-share ticker the
two new tools are appended; with it on AND a non-A-share ticker nothing changes.
Pure mock-LLM, zero network, zero LLM cost.
"""

from __future__ import annotations

import unittest
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from yialpha.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from yialpha.dataflows import akshare_vendor as akv, baostock_vendor as bsv
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError


# --------------------------------------------------------------------------- #
# Synthetic daily series
# --------------------------------------------------------------------------- #
def _row(d: str, close="100.0", pe="25.0", pb="3.0", ps="5.0", pcf="12.0",
         turn="1.0", pct="0.5", vol="1000000", amt="100000000", is_st="0",
         op="99.0", hi="101.0", lo="98.5") -> dict:
    return {
        "date": d, "code": "sh.600519", "open": op, "high": hi, "low": lo,
        "close": close, "preclose": "99.5", "volume": vol, "amount": amt,
        "turn": turn, "pctChg": pct, "peTTM": pe, "pbMRQ": pb, "psTTM": ps,
        "pcfNcfTTM": pcf, "isST": is_st,
    }


# Daily rows 2024-06-03..2024-06-14, plus a future row (2024-12-20) that PIT must
# drop, and an old row (2023-06-02) outside a 180d window from mid-2024
# (2024-06-15 - 180d = 2023-12-18, so 2023-06-02 is well before it).
SYNTH_ROWS = [
    _row("2023-06-02", close="90.0", pe="20.0"),                 # out of window
    _row("2024-06-03", close="95.0", pe="22.0"),
    _row("2024-06-04", close="96.0", pe="22.5"),
    _row("2024-06-05", close="97.0", pe="23.0"),
    _row("2024-06-06", close="98.0", pe="23.5"),
    _row("2024-06-07", close="99.0", pe="24.0"),
    _row("2024-06-10", close="100.0", pe="25.0"),
    _row("2024-06-11", close="101.0", pe="25.5"),
    _row("2024-06-12", close="102.0", pe="26.0"),
    _row("2024-06-13", close="103.0", pe="26.5"),
    _row("2024-06-14", close="104.0", pe="27.0"),
    _row("2024-12-20", close="200.0", pe="60.0"),                # future -> PIT drop
]


def _patch_daily(monkeypatch, rows=SYNTH_ROWS):
    """Serve synthetic rows without touching the network or baostock."""
    monkeypatch.setattr(bsv, "_cached_daily", lambda code, curr_date=None: list(rows))


# --------------------------------------------------------------------------- #
# Symbol mapping
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_symbol_map_shanghai():
    assert bsv._to_baostock_code("600519.SS") == "sh.600519"
    assert bsv._to_baostock_code("600000.SH") == "sh.600000"


@pytest.mark.unit
def test_symbol_map_shenzhen():
    assert bsv._to_baostock_code("000001.SZ") == "sz.000001"
    assert bsv._to_baostock_code("300750.SZ") == "sz.300750"


@pytest.mark.unit
def test_symbol_map_case_and_space_insensitive():
    assert bsv._to_baostock_code(" 600519.ss ") == "sh.600519"


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["AAPL", "0700.HK", "BTCUSDT", "600519", "", "12345.SS"])
def test_non_a_share_raises(bad):
    with pytest.raises(NoMarketDataError):
        bsv._to_baostock_code(bad)


# --------------------------------------------------------------------------- #
# Optional-dependency fail-soft
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_missing_dependency_raises_no_market_data(monkeypatch):
    """A machine without baostock installed gets a typed NoMarketDataError (the
    router degrades that to the optional-category sentinel, not a crash)."""
    import builtins
    real_import = builtins.__import__

    def _block(name, *a, **k):
        if name == "baostock":
            raise ImportError("no baostock")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _block)
    monkeypatch.setattr(bsv, "_cached_daily", lambda code, curr_date=None: [])  # short-circuit cache
    with pytest.raises(NoMarketDataError):
        # Force the lazy import path: clear any cache, call the public function.
        bsv._require_baostock()


# --------------------------------------------------------------------------- #
# PIT filtering
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ohlc_pit_drops_future_and_out_of_window(monkeypatch):
    _patch_daily(monkeypatch)
    out = bsv.get_a_share_ohlc_native("600519.SS", "2024-06-15", 180)
    assert "# A-share OHLC" in out
    assert "2024-06-14" in out          # latest in-window row
    assert "2024-12-20" not in out      # future -> PIT drop
    assert "2023-06-02" not in out      # outside 180d window


@pytest.mark.unit
def test_ohlc_curr_date_boundary_inclusive(monkeypatch):
    """curr_date 2024-06-12 keeps rows up to and including 2024-06-12; the
    2024-06-13/14 rows are post-curr_date -> dropped (no lookahead)."""
    _patch_daily(monkeypatch)
    out = bsv.get_a_share_ohlc_native("600519.SS", "2024-06-12", 180)
    assert "2024-06-12" in out
    assert "2024-06-13" not in out
    assert "2024-06-14" not in out


@pytest.mark.unit
def test_fundamentals_pit_drops_future(monkeypatch):
    _patch_daily(monkeypatch)
    out = bsv.get_a_share_fundamentals_native("600519.SS", "2024-06-15", 180)
    assert "# A-share Valuation" in out
    assert "2024-12-20" not in out
    assert "2024-06-14" in out          # latest in-window summary date
    # Latest snapshot uses the PIT-visible latest (2024-06-14, pe 27.0).
    assert "27.00x" in out


@pytest.mark.unit
def test_fundamentals_live_mode_uses_today(monkeypatch):
    """Empty curr_date = live mode: the upper bound is today, so the future row
    (2024-12-20 relative to a synthetic past) is kept when today >= that date."""
    _patch_daily(monkeypatch)
    out = bsv.get_a_share_fundamentals_native("600519.SS", None, 3650)
    # 2024-12-20 is in the past relative to today (2026), so live mode keeps it.
    assert "2024-12-20" in out


@pytest.mark.unit
def test_ohlc_no_rows_in_window_honest_empty(monkeypatch):
    _patch_daily(monkeypatch)
    out = bsv.get_a_share_ohlc_native("600519.SS", "2020-01-01", 10)
    assert out.startswith("# A-share OHLC")
    assert "No daily OHLC rows" in out


# --------------------------------------------------------------------------- #
# A7 — same-day cache refresh + session-level statement cache
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_daily_cache_ttl_same_day_refresh():
    """curr_date = today (or live mode) must NOT pin the daily cache for a
    full day — the day's post-close bar lands later, so the entry refreshes
    every _SAME_DAY_REFRESH_S (900s, mirroring stockstats_utils). A historical
    date is immutable and keeps the 1-day TTL."""
    today_ttl = bsv._daily_cache_ttl_days(date.today().isoformat())
    live_ttl = bsv._daily_cache_ttl_days(None)
    hist_ttl = bsv._daily_cache_ttl_days("2024-06-15")
    assert today_ttl == pytest.approx(bsv._SAME_DAY_REFRESH_S / 86_400.0)
    assert live_ttl == pytest.approx(bsv._SAME_DAY_REFRESH_S / 86_400.0)
    assert hist_ttl == pytest.approx(bsv._CACHE_TTL_S / 86_400.0)
    assert today_ttl < hist_ttl  # same-day refreshes strictly faster


@pytest.mark.unit
def test_statement_rows_cached_per_code_fn_anchor(monkeypatch):
    """One statement fetch costs a TCP login + up to 12 quarter queries; the
    session cache must collapse repeat calls for the same (code, fn, anchor)
    and key on the anchor (a different analysis year re-fetches)."""
    logins = []

    class _FakeSession:
        def __enter__(self):
            logins.append(1)
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(bsv, "_BaostockSession", _FakeSession)
    monkeypatch.setattr(
        bsv, "_query_statement",
        lambda bs, code, fn, anchor=None: bsv.StatementFetch(_profit_rows(), []))
    bsv._statement_rows_cached.cache_clear()

    r1 = bsv._statement_rows("sh.600519", "query_profit_data",
                             date(2024, 6, 15))
    r2 = bsv._statement_rows("sh.600519", "query_profit_data",
                             date(2024, 6, 15))
    assert r1 is r2                     # served from the cache
    assert len(logins) == 1
    bsv._statement_rows("sh.600519", "query_profit_data", date(2022, 6, 15))
    assert len(logins) == 2              # different anchor -> re-fetch


# --------------------------------------------------------------------------- #
# Router integration
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_router_routes_ohlc_via_baostock(monkeypatch):
    _patch_daily(monkeypatch)
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "baostock"}})
        out = route_to_vendor("get_a_share_ohlc_native", "600519.SS", "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    assert "# A-share OHLC" in out


@pytest.mark.unit
def test_router_optional_category_degrades_to_sentinel(monkeypatch):
    """A non-A-share ticker -> NoMarketDataError -> the router's NO_DATA_AVAILABLE
    sentinel (the typed 'report unavailable' path), not a crash. The optional
    category never re-raises: a missing native-data signal can't abort the run."""
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "baostock"}})
        out = route_to_vendor("get_a_share_ohlc_native", "AAPL", "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


@pytest.mark.unit
def test_router_missing_dependency_degrades_to_sentinel(monkeypatch, tmp_path):
    """baostock not installed -> NoMarketDataError from _require_baostock only
    fires on the login path; an empty cache dir forces that path."""
    import builtins
    real_import = builtins.__import__

    def _block(name, *a, **k):
        if name == "baostock":
            raise ImportError("no baostock")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _block)
    # Point the vendor at an empty cache dir so the shared disk cache misses
    # and _cached_daily actually reaches the (blocked) login path.
    monkeypatch.setattr(bsv, "_cache_dir", lambda: str(tmp_path))
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "baostock"}})
        out = route_to_vendor("get_a_share_ohlc_native", "600519.SS", "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


# --------------------------------------------------------------------------- #
# Wiring byte-equivalence (mock-LLM)
# --------------------------------------------------------------------------- #
class _BoundLLM(Runnable):
    def invoke(self, inp, config=None, **kwargs):  # noqa: D401, ARG002
        return AIMessage(content="MOCK REPORT", tool_calls=[])


class _RecordingLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.bound_tools = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: D401, ARG002
        return AIMessage(content="", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound_tools = list(tools)
        return _BoundLLM()


def _state(ticker="600519.SS"):
    return {
        "trade_date": "2024-06-15",
        "company_of_interest": ticker,
        "asset_type": "stock",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


class FundamentalsAShareWiringTests(unittest.TestCase):
    """a_share_native — default-off byte-equivalence + on-appends-native-tools."""

    def _tool_names(self, config_overrides=None, ticker="600519.SS"):
        from yialpha.dataflows import config as cfgmod
        orig = cfgmod.get_config()
        try:
            if config_overrides:
                cfgmod.set_config({**orig, **config_overrides})
            llm = _RecordingLLM()
            node = create_fundamentals_analyst(llm)
            node(_state(ticker))
            return [t.name for t in llm.bound_tools]
        finally:
            cfgmod.set_config(orig)

    def test_default_off_byte_equivalent_baseline(self):
        names = self._tool_names({"a_share_native": False})
        # Exactly the 4 baseline fundamentals tools; no a_share_native tools.
        self.assertEqual(
            names,
            ["get_fundamentals", "get_balance_sheet", "get_cashflow",
             "get_income_statement"],
        )

    def test_on_with_a_share_appends_native_tools(self):
        names = self._tool_names({"a_share_native": True}, ticker="600519.SS")
        self.assertEqual(
            names,
            ["get_fundamentals", "get_balance_sheet", "get_cashflow",
             "get_income_statement", "get_a_share_fundamentals_native",
             "get_a_share_ohlc_native", "get_a_share_money_flow_native",
             "get_a_share_dragon_tiger_native",
             "get_a_share_income_statement_native",
             "get_a_share_balance_sheet_native",
             "get_a_share_cashflow_statement_native"],
        )

    def test_on_with_non_a_share_is_byte_equivalent(self):
        """Flag on but US ticker -> is_a_stock fails -> baseline only (the
        double gate keeps US/crypto/HK runs byte-for-byte unchanged)."""
        names = self._tool_names({"a_share_native": True}, ticker="AAPL")
        self.assertEqual(
            names,
            ["get_fundamentals", "get_balance_sheet", "get_cashflow",
             "get_income_statement"],
        )

    def test_composes_independently_with_a_stock(self):
        """a_share_native and a_stock are independent flags; both can be on."""
        names = self._tool_names({"a_share_native": True, "a_stock": True},
                                 ticker="000001.SZ")
        self.assertIn("get_margin_trading", names)
        self.assertIn("get_a_share_fundamentals_native", names)
        self.assertIn("get_a_share_ohlc_native", names)
        self.assertIn("get_a_share_money_flow_native", names)
        self.assertIn("get_a_share_dragon_tiger_native", names)


# --------------------------------------------------------------------------- #
# AKShare news vendor (Phase 2)
# --------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402

import pandas as pd  # noqa: E402


def _news_df():
    """Synthetic AKShare stock_news_em DataFrame (CN column names)."""
    return pd.DataFrame([
        {"新闻标题": "茅台三季报超预期", "新闻内容": "净利润同比增长…",  # noqa: RUF001
         "发布时间": "2024-06-14 10:30:00", "文章来源": "东财", "新闻链接": "u1"},
        {"新闻标题": "机构上调评级", "新闻内容": "多家券商上调目标价…",
         "发布时间": "2024-06-10 09:00:00", "文章来源": "新浪", "新闻链接": "u2"},
        {"新闻标题": "未来某事件", "新闻内容": "这是未来新闻",
         "发布时间": "2024-12-20 08:00:00", "文章来源": "东财", "新闻链接": "u3"},
        {"新闻标题": "", "新闻内容": "空标题应被跳过",
         "发布时间": "2024-06-12 08:00:00", "文章来源": "东财", "新闻链接": "u4"},
    ])


def _patch_akshare(monkeypatch, df=None, raises=None):
    """Patch _require_akshare to a fake module; serve df (or raise)."""
    def _stock_news_em(symbol, *a, **k):
        if raises is not None:
            raise raises
        return df
    fake = SimpleNamespace(stock_news_em=_stock_news_em)
    monkeypatch.setattr(akv, "_require_akshare", lambda: fake)


@pytest.mark.unit
def test_akshare_symbol_map():
    assert akv._to_akshare_code("600519.SS") == "600519"
    assert akv._to_akshare_code("000001.SZ") == "000001"
    with pytest.raises(NoMarketDataError):
        akv._to_akshare_code("AAPL")


@pytest.mark.unit
def test_akshare_news_pit_drops_future(monkeypatch):
    _patch_akshare(monkeypatch, _news_df())
    out = akv.get_a_share_news_native("600519.SS", "2024-06-15", 14)
    assert "# A-share News" in out
    assert "茅台三季报超预期" in out      # 2024-06-14 kept
    assert "机构上调评级" in out          # 2024-06-10 kept
    assert "未来某事件" not in out        # 2024-12-20 future -> PIT drop
    assert "空标题应被跳过" not in out    # empty title dropped


@pytest.mark.unit
def test_akshare_news_window_drops_stale(monkeypatch):
    """look_back_days must bound the LOWER edge too, like every sibling tool.

    The parameter used to be echoed in the header ("last Nd") but never
    enforced: arbitrarily old dated headlines survived (and under ``limit``
    could displace recent ones).
    """
    df = pd.DataFrame([
        {"新闻标题": "新鲜新闻", "新闻内容": "c",
         "发布时间": "2024-06-14 10:30:00", "文章来源": "东财", "新闻链接": "u1"},
        {"新闻标题": "陈年旧闻", "新闻内容": "c",
         "发布时间": "2024-01-02 09:00:00", "文章来源": "东财", "新闻链接": "u2"},
    ])
    _patch_akshare(monkeypatch, df)
    out = akv.get_a_share_news_native("600519.SS", "2024-06-15", 14)
    assert "新鲜新闻" in out              # within [2024-06-01, 2024-06-15]
    assert "陈年旧闻" not in out          # 5 months old -> window drop

    # Boundary is inclusive: "last 1d" as of 2024-06-15 = [06-14, 06-15].
    out2 = akv.get_a_share_news_native("600519.SS", "2024-06-15", 1)
    assert "新鲜新闻" in out2

    # A window that excludes every dated item reports honestly.
    out3 = akv.get_a_share_news_native("600519.SS", "2024-03-15", 14)
    assert "No news items" in out3
    assert "新鲜新闻" not in out3
    assert "陈年旧闻" not in out3


@pytest.mark.unit
def test_akshare_news_undated_kept_by_fail_open(monkeypatch):
    """The documented PIT fail-open for missing timestamps still holds."""
    df = pd.DataFrame([
        {"新闻标题": "无日期头条", "新闻内容": "c",
         "发布时间": None, "文章来源": "东财", "新闻链接": "u1"},
    ])
    _patch_akshare(monkeypatch, df)
    out = akv.get_a_share_news_native("600519.SS", "2024-06-15", 14)
    assert "无日期头条" in out


@pytest.mark.unit
def test_akshare_news_empty_df_honest_empty(monkeypatch):
    _patch_akshare(monkeypatch, pd.DataFrame())
    out = akv.get_a_share_news_native("600519.SS", "2024-06-15", 14)
    assert "No news items returned" in out


@pytest.mark.unit
def test_akshare_news_transport_error_degrades(monkeypatch):
    """A generic akshare transport error -> NoMarketDataError (router sentinel)."""
    _patch_akshare(monkeypatch, raises=ConnectionError("boom"))
    with pytest.raises(NoMarketDataError):
        akv.get_a_share_news_native("600519.SS", "2024-06-15", 14)


@pytest.mark.unit
def test_akshare_news_rate_limit_typed(monkeypatch):
    """A 429/频繁 signal -> VendorRateLimitError so the router skips vendors."""
    _patch_akshare(monkeypatch, raises=RuntimeError("请求过于频繁"))
    with pytest.raises(VendorRateLimitError):
        akv.get_a_share_news_native("600519.SS", "2024-06-15", 14)


@pytest.mark.unit
def test_akshare_direct_connect_restores_env(monkeypatch):
    """Proxy env vars popped during the call are always restored after."""
    import os
    monkeypatch.setenv("HTTP_PROXY", "socks5h://127.0.0.1:1080")
    monkeypatch.setenv("HTTPS_PROXY", "socks5h://127.0.0.1:1080")
    _patch_akshare(monkeypatch, _news_df())
    akv.get_a_share_news_native("600519.SS", "2024-06-15", 14)
    assert os.environ["HTTP_PROXY"] == "socks5h://127.0.0.1:1080"
    assert os.environ["HTTPS_PROXY"] == "socks5h://127.0.0.1:1080"


@pytest.mark.unit
def test_router_routes_news_via_akshare(monkeypatch):
    _patch_akshare(monkeypatch, _news_df())
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_news_native", "600519.SS", "2024-06-15", 14)
    finally:
        cfgmod.set_config(orig)
    assert "# A-share News" in out


@pytest.mark.unit
def test_router_news_optional_category_degrades_to_sentinel(monkeypatch):
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_news_native", "AAPL", "2024-06-15", 14)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


# --------------------------------------------------------------------------- #
# AKShare money-flow + dragon-tiger vendors (Phase 4)
# --------------------------------------------------------------------------- #
def _patch_ak(monkeypatch, **fns):
    """Patch _require_akshare to a fake module exposing the given callables.

    Also drops the breadth TTL cache so each test's patched fetch is actually
    served (otherwise a sibling test's 90s-cached counts bleed through)."""
    akv.reset_breadth_cache_for_test()
    monkeypatch.setattr(akv, "_require_akshare", lambda: SimpleNamespace(**fns))


def _boom(exc):
    """A fake akshare fn that always raises ``exc``."""
    def _f(*a, **k):
        raise exc
    return _f


def _fund_flow_df():
    """Synthetic stock_individual_fund_flow DataFrame (CN columns, str dates)."""
    return pd.DataFrame([
        {"日期": "2024-05-01", "收盘价": 90.0, "涨跌幅": 0.3,            # before window
         "主力净流入-净额": 5.0e7, "主力净流入-净占比": 2.0,
         "超大单净流入-净额": 3.0e7, "大单净流入-净额": 2.0e7,
         "中单净流入-净额": -1.0e7, "小单净流入-净额": -4.0e7},
        {"日期": "2024-06-10", "收盘价": 95.0, "涨跌幅": 0.5,
         "主力净流入-净额": 1.2e8, "主力净流入-净占比": 5.0,
         "超大单净流入-净额": 8.0e7, "大单净流入-净额": 4.0e7,
         "中单净流入-净额": -3.0e7, "小单净流入-净额": -9.0e7},
        {"日期": "2024-06-12", "收盘价": 96.0, "涨跌幅": -0.4,
         "主力净流入-净额": -5.0e7, "主力净流入-净占比": -2.0,   # distribution day
         "超大单净流入-净额": -3.0e7, "大单净流入-净额": -2.0e7,
         "中单净流入-净额": 1.0e7, "小单净流入-净额": 4.0e7},
        {"日期": "2024-06-14", "收盘价": 97.0, "涨跌幅": 0.6,
         "主力净流入-净额": 6.0e7, "主力净流入-净占比": 2.5,
         "超大单净流入-净额": 4.0e7, "大单净流入-净额": 2.0e7,
         "中单净流入-净额": -1.5e7, "小单净流入-净额": -4.5e7},
        {"日期": "2024-12-20", "收盘价": 120.0, "涨跌幅": 1.0,            # future
         "主力净流入-净额": 9.9e8, "主力净流入-净占比": 20.0,
         "超大单净流入-净额": 5.0e8, "大单净流入-净额": 4.9e8,
         "中单净流入-净额": -2.0e8, "小单净流入-净额": -7.9e8},
    ])


def _lhb_df():
    """Synthetic stock_lhb_detail_em DataFrame (whole-market, filter by 代码)."""
    return pd.DataFrame([
        {"代码": "600519", "名称": "贵州茅台", "上榜日": "2024-06-12",
         "解读": "日跌幅偏离值达7%", "上榜原因": "跌幅偏离值",
         "龙虎榜净买额": 3.2e8, "龙虎榜买入额": 5.4e8, "龙虎榜卖出额": 2.2e8,
         "净买额占总成交比": 8.1, "上榜后1日": 1.2, "上榜后5日": -3.1},
        {"代码": "000001", "名称": "平安银行", "上榜日": "2024-06-13",   # other stock
         "解读": "涨幅偏离值", "上榜原因": "涨幅",
         "龙虎榜净买额": -1.0e8, "龙虎榜买入额": 2.0e8, "龙虎榜卖出额": 3.0e8,
         "净买额占总成交比": -2.3, "上榜后1日": 0.5, "上榜后5日": 1.0},
        {"代码": "600519", "名称": "贵州茅台", "上榜日": "2024-12-20",   # future
         "解读": "future", "上榜原因": "x",
         "龙虎榜净买额": 9.0e8, "龙虎榜买入额": 9.5e8, "龙虎榜卖出额": 0.5e8,
         "净买额占总成交比": 15.0, "上榜后1日": 2.0, "上榜后5日": 3.0},
    ])


@pytest.mark.unit
def test_akshare_market_for():
    assert akv._market_for("600519.SS") == "sh"
    assert akv._market_for("600000.SH") == "sh"
    assert akv._market_for("000001.SZ") == "sz"
    with pytest.raises(NoMarketDataError):
        akv._market_for("AAPL")


# --- money flow ---
@pytest.mark.unit
def test_money_flow_pit_drops_future_and_window(monkeypatch):
    _patch_ak(monkeypatch, stock_individual_fund_flow=lambda stock, market: _fund_flow_df())
    out = akv.get_a_share_money_flow_native("600519.SS", "2024-06-15", 30)
    assert "# A-share Money Flow" in out
    assert "2024-06-14" in out          # latest in-window
    assert "2024-06-10" in out
    assert "2024-06-12" in out          # distribution day kept
    assert "2024-12-20" not in out      # future -> PIT drop
    assert "2024-05-01" not in out      # outside 30d window
    assert "600519" in out              # secid stock arg is the bare code


@pytest.mark.unit
def test_money_flow_summary(monkeypatch):
    _patch_ak(monkeypatch, stock_individual_fund_flow=lambda stock, market: _fund_flow_df())
    out = akv.get_a_share_money_flow_native("600519.SS", "2024-06-15", 30)
    # window sum = 1.2e8 - 5e7 + 6e7 = 1.3e8 -> +1.30亿
    assert "累计主力净流入: +1.30亿" in out
    # latest (2024-06-14) is inflow, day before (06-12) is outflow -> streak = 1
    assert "连续主力净流入日数: 1" in out
    assert "最近一日(2024-06-14)" in out


@pytest.mark.unit
def test_money_flow_empty_honest(monkeypatch):
    _patch_ak(monkeypatch, stock_individual_fund_flow=lambda stock, market: pd.DataFrame())
    out = akv.get_a_share_money_flow_native("600519.SS", "2024-06-15", 30)
    assert "No money-flow rows returned" in out


@pytest.mark.unit
def test_money_flow_no_rows_in_window(monkeypatch):
    """All rows fall outside the window -> honest 'no coverage found'."""
    _patch_ak(monkeypatch, stock_individual_fund_flow=lambda stock, market: _fund_flow_df())
    out = akv.get_a_share_money_flow_native("600519.SS", "2020-01-01", 10)
    assert out.startswith("# A-share Money Flow")
    assert "no coverage found" in out


@pytest.mark.unit
def test_money_flow_transport_error_degrades(monkeypatch):
    _patch_ak(monkeypatch, stock_individual_fund_flow=_boom(ConnectionError("boom")))
    with pytest.raises(NoMarketDataError):
        akv.get_a_share_money_flow_native("600519.SS", "2024-06-15", 30)


@pytest.mark.unit
def test_money_flow_rate_limit_typed(monkeypatch):
    _patch_ak(monkeypatch, stock_individual_fund_flow=_boom(RuntimeError("请求过于频繁")))
    with pytest.raises(VendorRateLimitError):
        akv.get_a_share_money_flow_native("600519.SS", "2024-06-15", 30)


@pytest.mark.unit
def test_money_flow_direct_connect_restores_env(monkeypatch):
    import os
    monkeypatch.setenv("HTTP_PROXY", "socks5h://127.0.0.1:1080")
    monkeypatch.setenv("HTTPS_PROXY", "socks5h://127.0.0.1:1080")
    _patch_ak(monkeypatch, stock_individual_fund_flow=lambda stock, market: _fund_flow_df())
    akv.get_a_share_money_flow_native("600519.SS", "2024-06-15", 30)
    assert os.environ["HTTP_PROXY"] == "socks5h://127.0.0.1:1080"
    assert os.environ["HTTPS_PROXY"] == "socks5h://127.0.0.1:1080"


@pytest.mark.unit
def test_router_routes_money_flow_via_akshare(monkeypatch):
    _patch_ak(monkeypatch, stock_individual_fund_flow=lambda stock, market: _fund_flow_df())
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_money_flow_native", "600519.SS", "2024-06-15", 30)
    finally:
        cfgmod.set_config(orig)
    assert "# A-share Money Flow" in out


# --- dragon-tiger ---
@pytest.mark.unit
def test_dragon_tiger_filters_code_and_pit(monkeypatch):
    _patch_ak(monkeypatch, stock_lhb_detail_em=lambda start_date, end_date: _lhb_df())
    out = akv.get_a_share_dragon_tiger_native("600519.SS", "2024-06-15", 90)
    assert "# A-share Dragon-Tiger" in out
    assert "2024-06-12" in out          # 600519 appearance kept
    assert "2024-12-20" not in out      # future -> PIT drop
    assert "平安银行" not in out        # other stock (000001) filtered out
    assert "1 appearance" in out


@pytest.mark.unit
def test_dragon_tiger_omits_forward_returns(monkeypatch):
    """上榜后N日 are post-event forward returns -> never surfaced (lookahead)."""
    df = _lhb_df()
    # Tag the 600519 rows' forward-return cell with a unique sentinel; if it ever
    # leaks into the output, the lookahead guard has failed.
    df.loc[df["代码"] == "600519", "上榜后5日"] = 999.5
    _patch_ak(monkeypatch, stock_lhb_detail_em=lambda start_date, end_date: df)
    out = akv.get_a_share_dragon_tiger_native("600519.SS", "2024-06-15", 90)
    assert "999.5" not in out          # forward-return data must not surface
    assert "omitted" in out            # the omission is documented in the header
    assert "+3.20亿" in out            # net buy IS shown


@pytest.mark.unit
def test_dragon_tiger_no_appearance_honest(monkeypatch):
    """Stock not on the board (only other stocks in the df) -> honest empty."""
    df = pd.DataFrame([{"代码": "000001", "名称": "平安银行", "上榜日": "2024-06-13",
                        "解读": "x", "上榜原因": "y", "龙虎榜净买额": 1e8,
                        "龙虎榜买入额": 2e8, "龙虎榜卖出额": 1e8,
                        "净买额占总成交比": 1.0}])
    _patch_ak(monkeypatch, stock_lhb_detail_em=lambda start_date, end_date: df)
    out = akv.get_a_share_dragon_tiger_native("600519.SS", "2024-06-15", 90)
    assert "No dragon-tiger" in out
    assert "normal" in out              # reassure: empty is not bearish


@pytest.mark.unit
def test_dragon_tiger_empty_df_honest(monkeypatch):
    _patch_ak(monkeypatch, stock_lhb_detail_em=lambda start_date, end_date: pd.DataFrame())
    out = akv.get_a_share_dragon_tiger_native("600519.SS", "2024-06-15", 90)
    assert "No dragon-tiger" in out


@pytest.mark.unit
def test_dragon_tiger_transport_error_degrades(monkeypatch):
    _patch_ak(monkeypatch, stock_lhb_detail_em=_boom(ConnectionError("boom")))
    with pytest.raises(NoMarketDataError):
        akv.get_a_share_dragon_tiger_native("600519.SS", "2024-06-15", 90)


@pytest.mark.unit
def test_dragon_tiger_rate_limit_typed(monkeypatch):
    _patch_ak(monkeypatch, stock_lhb_detail_em=_boom(RuntimeError("请求过于频繁")))
    with pytest.raises(VendorRateLimitError):
        akv.get_a_share_dragon_tiger_native("600519.SS", "2024-06-15", 90)


@pytest.mark.unit
def test_dragon_tiger_direct_connect_restores_env(monkeypatch):
    import os
    monkeypatch.setenv("HTTP_PROXY", "socks5h://127.0.0.1:1080")
    monkeypatch.setenv("HTTPS_PROXY", "socks5h://127.0.0.1:1080")
    _patch_ak(monkeypatch, stock_lhb_detail_em=lambda start_date, end_date: _lhb_df())
    akv.get_a_share_dragon_tiger_native("600519.SS", "2024-06-15", 90)
    assert os.environ["HTTP_PROXY"] == "socks5h://127.0.0.1:1080"
    assert os.environ["HTTPS_PROXY"] == "socks5h://127.0.0.1:1080"


@pytest.mark.unit
def test_router_routes_dragon_tiger_via_akshare(monkeypatch):
    _patch_ak(monkeypatch, stock_lhb_detail_em=lambda start_date, end_date: _lhb_df())
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_dragon_tiger_native", "600519.SS", "2024-06-15", 90)
    finally:
        cfgmod.set_config(orig)
    assert "# A-share Dragon-Tiger" in out


@pytest.mark.unit
def test_router_money_flow_non_a_share_degrades_to_sentinel(monkeypatch):
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_money_flow_native", "AAPL", "2024-06-15", 30)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


class NewsAShareWiringTests(unittest.TestCase):
    """a_share_native news — default-off byte-equivalence + on-appends-news-tool.

    web_search_enabled is frozen off in every override below: these tests pin
    the a_share_native gating contract only (the default news toolset,
    web_search included, is pinned in test_tavily_web_search.py).
    """

    def _tool_names(
        self,
        config_overrides=None,
        ticker="600519.SS",
        trade_date=None,
    ):
        from yialpha.agents.analysts.news_analyst import create_news_analyst
        from yialpha.dataflows import config as cfgmod
        orig = cfgmod.get_config()
        frozen = {"web_search_enabled": False}
        if config_overrides:
            frozen.update(config_overrides)
        try:
            cfgmod.set_config({**orig, **frozen})
            llm = _RecordingLLM()
            node = create_news_analyst(llm)
            state = _state(ticker)
            state["trade_date"] = trade_date or date.today().isoformat()
            node(state)
            return [t.name for t in llm.bound_tools]
        finally:
            cfgmod.set_config(orig)

    def test_default_off_byte_equivalent_baseline(self):
        names = self._tool_names({"a_share_native": False})
        self.assertEqual(
            names, ["get_news", "get_global_news", "get_macro_indicators",
                    "get_prediction_markets"],
        )

    def test_on_with_a_share_appends_news_tool(self):
        names = self._tool_names({"a_share_native": True}, ticker="600519.SS")
        self.assertEqual(
            names, ["get_news", "get_global_news", "get_macro_indicators",
                    "get_prediction_markets", "get_a_share_news_native"],
        )

    def test_on_with_non_a_share_is_byte_equivalent(self):
        names = self._tool_names({"a_share_native": True}, ticker="AAPL")
        self.assertEqual(
            names, ["get_news", "get_global_news", "get_macro_indicators",
                    "get_prediction_markets"],
        )

    def test_historical_news_omits_live_prediction_markets(self):
        names = self._tool_names(
            {"a_share_native": False},
            trade_date="2020-01-02",
        )
        self.assertEqual(
            names,
            ["get_news", "get_global_news", "get_macro_indicators"],
        )


# --------------------------------------------------------------------------- #
# Tushare vendor (Phase 3 — opt-in token tier)
# --------------------------------------------------------------------------- #
from yialpha.dataflows import tushare_vendor as tv  # noqa: E402
from yialpha.dataflows.errors import VendorNotConfiguredError  # noqa: E402


def _tushare_pro(df_basic=None, df_news=None, raises=None):
    """Build a fake tushare pro_api object."""
    def _daily_basic(**kw):
        if raises is not None:
            raise raises
        return df_basic
    def _news(**kw):
        if raises is not None:
            raise raises
        return df_news
    return SimpleNamespace(daily_basic=_daily_basic, news=_news)


def _patch_tushare(monkeypatch, pro):
    monkeypatch.setattr(tv, "_require_tushare", lambda: pro)


def _basic_df():
    return pd.DataFrame([
        {"trade_date": "20240614", "pe_ttm": 27.0, "pb": 8.1, "total_mv": 1.64e12,
         "circ_mv": 1.64e12, "turnover_rate": 0.2, "free_share": 0.0},
        {"trade_date": "20240610", "pe_ttm": 26.0, "pb": 7.8, "total_mv": 1.58e12,
         "circ_mv": 1.58e12, "turnover_rate": 0.3, "free_share": 0.0},
        {"trade_date": "20241220", "pe_ttm": 60.0, "pb": 18.0, "total_mv": 3.0e12,
         "circ_mv": 3.0e12, "turnover_rate": 0.5, "free_share": 0.0},  # future
    ])


def _tushare_news_df():
    return pd.DataFrame([
        {"title": "茅台发布年报", "datetime": "2024-06-14 18:00:00", "src": "sina",
         "content": "600519 净利润增长"},
        {"title": "无关新闻", "datetime": "2024-06-13 10:00:00", "src": "cls",
         "content": "某公司动态"},
        {"title": "茅台未来展望", "datetime": "2024-12-20 08:00:00", "src": "eastmoney",
         "content": "600519"},  # future -> PIT drop
    ])


@pytest.mark.unit
def test_tushare_symbol_map_ss_to_sh():
    # YiAlpha .SS (yfinance Shanghai) maps to Tushare's .SH.
    assert tv._to_tushare_code("600519.SS") == "600519.SH"
    assert tv._to_tushare_code("600000.SH") == "600000.SH"


@pytest.mark.unit
def test_tushare_transport_error_is_not_no_market_data():
    """A network outage must surface as VendorError, not NoMarketDataError —
    mapping it to no-data made the router blame the ticker ('symbol may be
    invalid, delisted') during an outage."""
    import requests

    from yialpha.dataflows.errors import NoMarketDataError, VendorError

    pro = _tushare_pro(raises=requests.exceptions.ConnectionError("conn refused"))
    with pytest.raises(VendorError, match="transport failure") as ei:
        tv._query(pro, "daily_basic", ts_code="600519.SH")
    assert not isinstance(ei.value, NoMarketDataError)


@pytest.mark.unit
def test_tushare_symbol_map_sz():
    assert tv._to_tushare_code("000001.SZ") == "000001.SZ"


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["AAPL", "0700.HK", "BTCUSDT", "600519", ""])
def test_tushare_non_a_share_raises(bad):
    with pytest.raises(NoMarketDataError):
        tv._to_tushare_code(bad)


@pytest.mark.unit
def test_tushare_missing_token_raises_not_configured(monkeypatch):
    """No TUSHARE_TOKEN -> VendorNotConfiguredError (router skips this vendor)."""
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    from yialpha.dataflows import config as cfgmod
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "tushare_token": None})
        with pytest.raises(VendorNotConfiguredError):
            tv._require_tushare()
    finally:
        cfgmod.set_config(orig)


@pytest.mark.unit
def test_tushare_fundamentals_pit_drops_future(monkeypatch):
    _patch_tushare(monkeypatch, _tushare_pro(df_basic=_basic_df()))
    out = tv.get_a_share_fundamentals_native("600519.SS", "2024-06-15", 180)
    assert "# A-share Valuation (Tushare)" in out
    assert "2024-06-14" in out          # kept
    assert "2024-12-20" not in out      # future -> PIT drop
    assert "27.00x" in out              # latest PE


@pytest.mark.unit
def test_tushare_news_keyword_filter_and_pit(monkeypatch):
    _patch_tushare(monkeypatch, _tushare_pro(df_news=_tushare_news_df()))
    out = tv.get_a_share_news_native("600519.SS", "2024-06-15", 14)
    assert "茅台发布年报" in out          # has 600519 keyword, in window
    assert "无关新闻" not in out          # no 600519 keyword -> filtered
    assert "茅台未来展望" not in out      # future -> PIT drop


@pytest.mark.unit
def test_tushare_rate_limit_typed(monkeypatch):
    """A 每分钟/权限 signal -> VendorRateLimitError so the router skips vendors."""
    _patch_tushare(monkeypatch, _tushare_pro(raises=RuntimeError("每分钟访问次数超限")))
    with pytest.raises(VendorRateLimitError):
        tv.get_a_share_fundamentals_native("600519.SS", "2024-06-15", 180)


@pytest.mark.unit
def test_router_falls_back_from_tushare_to_baostock(monkeypatch, tmp_path):
    """Tushare configured first but no token -> VendorNotConfiguredError is caught
    by the router and the chain falls through to BaoStock (success). This is the
    key multi-vendor value: a keyless default still works when Tushare is listed
    but unconfigured."""
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    # BaoStock path: serve synthetic daily rows.
    monkeypatch.setattr(bsv, "_cached_daily", lambda code, curr_date=None: list(SYNTH_ROWS))

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "tushare,baostock"}})
        out = route_to_vendor("get_a_share_fundamentals_native", "600519.SS",
                              "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    # Fell through to BaoStock's fundamentals formatter.
    assert "# A-share Valuation (BaoStock" in out


# --------------------------------------------------------------------------- #
# Northbound capital / sector flow / realtime / breadth (Phase 5 — AKShare)
# --------------------------------------------------------------------------- #

def _northbound_df():
    """Synthetic stock_hsgt_individual_em DataFrame."""
    return pd.DataFrame([
        {"持股日期": "2024-05-01", "持股数量": 8.0e7, "持股数量占发行股": 0.64,
         "持股市值": 8.0e9, "持股市值占比": 0.5},
        {"持股日期": "2024-06-10", "持股数量": 1.0e8, "持股数量占发行股": 0.80,
         "持股市值": 9.5e9, "持股市值占比": 0.6},
        {"持股日期": "2024-06-12", "持股数量": 9.0e7, "持股数量占发行股": 0.72,
         "持股市值": 8.6e9, "持股市值占比": 0.55},
        {"持股日期": "2024-06-14", "持股数量": 1.2e8, "持股数量占发行股": 0.96,
         "持股市值": 1.16e10, "持股市值占比": 0.75},
        {"持股日期": "2024-12-20", "持股数量": 2.0e8, "持股数量占发行股": 1.6,
         "持股市值": 2.4e10, "持股市值占比": 1.5},  # future -> PIT drop
    ])


# --- northbound ---
@pytest.mark.unit
def test_northbound_pit_drops_future_and_window(monkeypatch):
    _patch_ak(monkeypatch, stock_hsgt_individual_em=lambda symbol: _northbound_df())
    out = akv.get_a_share_northbound_native("600519.SS", "2024-06-15", 30)
    assert "# A-share Northbound" in out
    assert "2024-06-14" in out
    assert "2024-06-10" in out
    assert "2024-06-12" in out
    assert "2024-12-20" not in out      # future -> PIT drop
    assert "2024-05-01" not in out      # outside 30d window


@pytest.mark.unit
def test_northbound_summary(monkeypatch):
    _patch_ak(monkeypatch, stock_hsgt_individual_em=lambda symbol: _northbound_df())
    out = akv.get_a_share_northbound_native("600519.SS", "2024-06-15", 30)
    # window summary should show 增持 (1.2e8 > 1.0e8 from first in window)
    assert "增持" in out
    assert "窗口变化" in out


@pytest.mark.unit
def test_northbound_empty_honest(monkeypatch):
    _patch_ak(monkeypatch, stock_hsgt_individual_em=lambda symbol: pd.DataFrame())
    out = akv.get_a_share_northbound_native("600519.SS", "2024-06-15", 30)
    assert "No northbound holding rows" in out


@pytest.mark.unit
def test_northbound_transport_error_degrades(monkeypatch):
    _patch_ak(monkeypatch, stock_hsgt_individual_em=_boom(ConnectionError("boom")))
    with pytest.raises(NoMarketDataError):
        akv.get_a_share_northbound_native("600519.SS", "2024-06-15", 30)


@pytest.mark.unit
def test_northbound_rate_limit_typed(monkeypatch):
    _patch_ak(monkeypatch, stock_hsgt_individual_em=_boom(RuntimeError("请求过于频繁")))
    with pytest.raises(VendorRateLimitError):
        akv.get_a_share_northbound_native("600519.SS", "2024-06-15", 30)


@pytest.mark.unit
def test_router_routes_northbound_via_akshare(monkeypatch):
    _patch_ak(monkeypatch, stock_hsgt_individual_em=lambda symbol: _northbound_df())
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_northbound_native", "600519.SS", "2024-06-15", 30)
    finally:
        cfgmod.set_config(orig)
    assert "# A-share Northbound" in out


@pytest.mark.unit
def test_router_northbound_non_a_share_degrades_to_sentinel(monkeypatch):
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_northbound_native", "AAPL", "2024-06-15", 30)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


# --- sector flow ---
def _sector_flow_df():
    """Synthetic stock_sector_fund_flow_rank DataFrame."""
    return pd.DataFrame([
        {"名称": "白酒", "今日主力净流入-净额": 5.0e8, "今日主力净流入-净占比": 3.5,
         "今日超大单净流入-净额": 3.0e8, "今日大单净流入-净额": 2.0e8},
        {"名称": "半导体", "今日主力净流入-净额": 3.0e8, "今日主力净流入-净占比": 2.0,
         "今日超大单净流入-净额": 2.0e8, "今日大单净流入-净额": 1.0e8},
        {"名称": "房地产", "今日主力净流入-净额": -2.0e8, "今日主力净流入-净占比": -1.5,
         "今日超大单净流入-净额": -1.5e8, "今日大单净流入-净额": -5.0e7},
        {"名称": "", "今日主力净流入-净额": 1.0e7, "今日主力净流入-净占比": 0.1,
         "今日超大单净流入-净额": 5.0e6, "今日大单净流入-净额": 5.0e6},
    ])


@pytest.mark.unit
def test_sector_flow_with_industry(monkeypatch):
    """Industry resolved via BaoStock's per-stock mapping (the marker appears).

    The old test mocked ``stock_board_industry_name_ths`` — an interface the
    vendor can no longer call (it takes no kwargs and returns a board catalog,
    not a stock->industry map), so the mock mirrored the very bug being fixed.
    The industry lookup is now mocked at its own seam (``_baostock_industry``).
    """
    _patch_ak(monkeypatch, stock_sector_fund_flow_rank=lambda **kw: _sector_flow_df())
    monkeypatch.setattr(akv, "_baostock_industry", lambda ticker: "白酒")
    out = akv.get_a_share_sector_flow_native("600519.SS", None, 1)
    assert "# A-share Sector Fund Flow" in out
    assert "白酒" in out
    assert "本股所属" in out         # stock's sector highlighted
    assert "房地产" in out           # other sector present
    assert "no identically-named row" not in out  # exact match -> no caveat


@pytest.mark.unit
def test_sector_flow_industry_no_exact_sector_match_notes_it(monkeypatch):
    """A resolved industry with no identically-named sector row (BaoStock's
    classification differs from the Eastmoney sector table) must be disclosed,
    not silently dropped or force-matched."""
    _patch_ak(monkeypatch, stock_sector_fund_flow_rank=lambda **kw: _sector_flow_df())
    monkeypatch.setattr(akv, "_baostock_industry", lambda ticker: "酒、饮料和精制茶制造业")
    out = akv.get_a_share_sector_flow_native("600519.SS", None, 1)
    assert "酒、饮料和精制茶制造业" in out       # resolved industry still shown
    assert "本股所属" not in out                # no fabricated marker
    assert "no identically-named row" in out    # the mismatch is disclosed


@pytest.mark.unit
def test_sector_flow_without_industry(monkeypatch):
    """If industry lookup fails, still returns sector ranking (no marker)."""
    _patch_ak(
        monkeypatch,
        stock_sector_fund_flow_rank=lambda **kw: _sector_flow_df(),
    )
    monkeypatch.setattr(akv, "_baostock_industry", lambda ticker: None)
    out = akv.get_a_share_sector_flow_native("600519.SS", None, 1)
    assert "# A-share Sector Fund Flow" in out
    assert "could not be resolved" in out
    assert "本股所属" not in out


@pytest.mark.unit
def test_sector_flow_empty_honest(monkeypatch):
    _patch_ak(
        monkeypatch,
        stock_sector_fund_flow_rank=lambda **kw: pd.DataFrame(),
    )
    out = akv.get_a_share_sector_flow_native("600519.SS", None, 1)
    assert "No sector fund-flow data" in out


@pytest.mark.unit
def test_sector_flow_transport_error_degrades(monkeypatch):
    _patch_ak(
        monkeypatch,
        stock_sector_fund_flow_rank=_boom(ConnectionError("boom")),
    )
    with pytest.raises(NoMarketDataError):
        akv.get_a_share_sector_flow_native("600519.SS", None, 1)


@pytest.mark.unit
def test_router_routes_sector_flow_via_akshare(monkeypatch):
    _patch_ak(monkeypatch, stock_sector_fund_flow_rank=lambda **kw: _sector_flow_df())
    monkeypatch.setattr(akv, "_baostock_industry", lambda ticker: "白酒")
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_sector_flow_native", "600519.SS", None, 1)
    finally:
        cfgmod.set_config(orig)
    assert "# A-share Sector Fund Flow" in out


# --- realtime quote ---
def _spot_em_df():
    """Synthetic stock_zh_a_spot_em DataFrame (whole market, filter by 代码)."""
    return pd.DataFrame([
        {"代码": "600519", "名称": "贵州茅台", "最新价": 1685.50, "涨跌幅": 1.23,
         "涨跌额": 20.5, "成交量": 1234567, "成交额": 2.0e10, "换手率": 0.15,
         "市盈率-动态": 28.5, "市净率": 9.2, "最高": 1690.0, "最低": 1668.0,
         "今开": 1670.0, "昨收": 1665.0},
        {"代码": "000001", "名称": "平安银行", "最新价": 11.50, "涨跌幅": -0.43,
         "涨跌额": -0.05, "成交量": 99999999, "成交额": 1.1e9, "换手率": 0.52,
         "市盈率-动态": 4.5, "市净率": 0.55, "最高": 11.6, "最低": 11.4,
         "今开": 11.55, "昨收": 11.55},
    ])


@pytest.mark.unit
def test_realtime_quote_live_mode(monkeypatch):
    _patch_ak(monkeypatch, stock_zh_a_spot_em=lambda: _spot_em_df())
    out = akv.get_a_share_realtime_quote_native("600519.SS")
    assert "# A-share Real-Time Quote" in out
    assert "贵州茅台" in out
    assert "1685.50" in out         # latest price
    assert "1.23%" in out           # change %


@pytest.mark.unit
def test_realtime_quote_historical_sentinel(monkeypatch):
    """A historical curr_date -> REAL_TIME_UNAVAILABLE sentinel (no lookahead)."""
    _patch_ak(monkeypatch, stock_zh_a_spot_em=lambda: _spot_em_df())
    out = akv.get_a_share_realtime_quote_native("600519.SS", "2024-06-15")
    assert "REAL_TIME_UNAVAILABLE" in out
    assert "historical" in out.lower()


@pytest.mark.unit
def test_realtime_quote_today_is_live_not_sentinel(monkeypatch):
    """curr_date = exactly today is LIVE mode (utils.is_historical_date) — the
    old gate treated any explicit curr_date as historical and misrouted today's
    live run (the framework default) to the sentinel."""
    _patch_ak(monkeypatch, stock_zh_a_spot_em=lambda: _spot_em_df())
    out = akv.get_a_share_realtime_quote_native("600519.SS",
                                                date.today().isoformat())
    assert "REAL_TIME_UNAVAILABLE" not in out
    assert "贵州茅台" in out


@pytest.mark.unit
def test_realtime_quote_malformed_date_fails_closed(monkeypatch):
    """A non-empty UNPARSEABLE curr_date (slash form an LLM can emit) used to
    pass is_historical_date as "live" and serve today's snapshot into a
    nominally historical run. It now takes the historical branch."""
    from yialpha.dataflows.utils import is_historical_date

    # Unit-level contract: only empty/None is live; garbage is historical.
    assert is_historical_date(None) is False
    assert is_historical_date("") is False
    assert is_historical_date("2026/08/01") is True
    assert is_historical_date("garbage") is True

    # Gate-level: the realtime snapshot is refused, not served.
    _patch_ak(monkeypatch, stock_zh_a_spot_em=lambda: _spot_em_df())
    out = akv.get_a_share_realtime_quote_native("600519.SS", "2024/06/15")
    assert "REAL_TIME_UNAVAILABLE" in out
    assert "贵州茅台" not in out


@pytest.mark.unit
def test_realtime_quote_not_found(monkeypatch):
    _patch_ak(monkeypatch, stock_zh_a_spot_em=lambda: _spot_em_df())
    out = akv.get_a_share_realtime_quote_native("999999.SZ")
    assert "not found" in out


@pytest.mark.unit
def test_realtime_quote_transport_error(monkeypatch):
    _patch_ak(monkeypatch, stock_zh_a_spot_em=_boom(ConnectionError("boom")))
    with pytest.raises(NoMarketDataError):
        akv.get_a_share_realtime_quote_native("600519.SS")


@pytest.mark.unit
def test_router_routes_realtime_via_akshare(monkeypatch):
    _patch_ak(monkeypatch, stock_zh_a_spot_em=lambda: _spot_em_df())
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_realtime_quote_native", "600519.SS", "")
    finally:
        cfgmod.set_config(orig)
    assert "# A-share Real-Time Quote" in out


# --- market breadth ---
def _breadth_df():
    """Synthetic stock_zh_a_spot DataFrame for breadth aggregation."""
    return pd.DataFrame([
        {"涨跌幅": 5.0}, {"涨跌幅": 3.0}, {"涨跌幅": -2.0}, {"涨跌幅": -1.0},
        {"涨跌幅": 10.0},   # limit up (>= 9.9)
        {"涨跌幅": -10.0},  # limit down (<= -9.9)
        {"涨跌幅": 0.0},    # flat
    ])


@pytest.mark.unit
def test_market_breadth_live_mode(monkeypatch):
    _patch_ak(monkeypatch, stock_zh_a_spot=lambda: _breadth_df())
    out = akv.get_a_share_market_breadth_native()
    assert "# A-share Market Breadth" in out
    assert "上涨" in out
    assert "下跌" in out
    assert "涨停" in out
    assert "跌停" in out
    assert "涨跌比" in out


@pytest.mark.unit
def test_market_breadth_historical_sentinel(monkeypatch):
    _patch_ak(monkeypatch, stock_zh_a_spot=lambda: _breadth_df())
    out = akv.get_a_share_market_breadth_native("2024-06-15")
    assert "REAL_TIME_UNAVAILABLE" in out


@pytest.mark.unit
def test_market_breadth_today_is_live_not_sentinel(monkeypatch):
    """curr_date = exactly today is LIVE mode (utils.is_historical_date:
    today = live; any other explicit date = backtest) — the old gate treated
    any explicit curr_date as historical and misrouted today's live run to the
    sentinel."""
    _patch_ak(monkeypatch, stock_zh_a_spot=lambda: _breadth_df())
    out = akv.get_a_share_market_breadth_native(date.today().isoformat())
    assert "REAL_TIME_UNAVAILABLE" not in out
    assert "# A-share Market Breadth" in out


@pytest.mark.unit
def test_limit_thresholds_tiered_by_board_and_st():
    """A7: limit-up/down tiers — 创业板 (300/301) and 科创板 (688/681) move
    +/-20%, ST stocks +/-5%, main board +/-10% (counted at 19.9/4.9/9.9 to
    absorb 2-decimal rounding). The old flat >=9.9 mislabeled every such move."""
    assert akv._limit_thresholds("600519", "贵州茅台") == (9.9, -9.9)
    assert akv._limit_thresholds("000001", "平安银行") == (9.9, -9.9)
    assert akv._limit_thresholds("300750", "宁德时代") == (19.9, -19.9)
    assert akv._limit_thresholds("301001", None) == (19.9, -19.9)
    assert akv._limit_thresholds("688001", None) == (19.9, -19.9)
    assert akv._limit_thresholds("681001", None) == (19.9, -19.9)
    assert akv._limit_thresholds("600000", "ST浦发") == (4.9, -4.9)
    assert akv._limit_thresholds("600000", "*ST新海") == (4.9, -4.9)
    # Sina's positional 代码 (e.g. "sz300750") still tiers by the 6 digits.
    assert akv._limit_thresholds("sz300750", None) == (19.9, -19.9)


@pytest.mark.unit
def test_market_breadth_limit_counts_tiered(monkeypatch):
    """A ChiNext stock at +19.95% and an ST stock at +4.95% are limit-ups under
    their tiers; the old flat 9.9 threshold counted neither."""
    df = pd.DataFrame([
        {"代码": "300750", "名称": "宁德时代", "涨跌幅": 19.95},  # ChiNext limit-up
        {"代码": "600000", "名称": "ST浦发", "涨跌幅": 4.95},     # ST limit-up
        {"代码": "600519", "名称": "贵州茅台", "涨跌幅": 9.95},   # main-board limit-up
        {"代码": "000001", "名称": "平安银行", "涨跌幅": 5.0},     # ordinary advance
    ])
    _patch_ak(monkeypatch, stock_zh_a_spot=lambda: df)
    out = akv.get_a_share_market_breadth_native()
    assert "涨停: 3" in out               # all three tiered limit-ups counted
    assert "跌停: 0" in out


@pytest.mark.unit
def test_market_breadth_transport_error(monkeypatch):
    _patch_ak(monkeypatch, stock_zh_a_spot=_boom(ConnectionError("boom")))
    with pytest.raises(NoMarketDataError):
        akv.get_a_share_market_breadth_native()


@pytest.mark.unit
def test_router_routes_breadth_via_akshare(monkeypatch):
    _patch_ak(monkeypatch, stock_zh_a_spot=lambda: _breadth_df())
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_market_breadth_native", "")
    finally:
        cfgmod.set_config(orig)
    assert "# A-share Market Breadth" in out


# --- market analyst wiring ---
class MarketAnalystAShareWiringTests(unittest.TestCase):
    """a_share_native — default-off byte-equivalence + on-appends-market-tools."""

    def _tool_names(self, config_overrides=None, ticker="600519.SS"):
        from yialpha.agents.analysts.market_analyst import create_market_analyst
        from yialpha.dataflows import config as cfgmod
        orig = cfgmod.get_config()
        try:
            if config_overrides:
                cfgmod.set_config({**orig, **config_overrides})
            llm = _RecordingLLM()
            node = create_market_analyst(llm)
            state = _state(ticker)
            node(state)
            return [t.name for t in llm.bound_tools]
        finally:
            cfgmod.set_config(orig)

    # The market analyst's baseline list carries the 2026-08-15 technical-
    # analysis expansion (weekly context, S/R, volume, patterns, relative
    # strength) on top of the original trio — identical whether the A-share
    # flag is off or the ticker is non-A-share (the double gate).
    _EXPANDED_BASELINE = [
        "get_stock_data", "get_indicators", "get_verified_market_snapshot",
        "get_indicators_weekly", "get_support_resistance",
        "get_volume_features", "get_candlestick_patterns",
        "get_relative_strength",
    ]

    def test_default_off_byte_equivalent_baseline(self):
        names = self._tool_names({"a_share_native": False})
        self.assertEqual(names, self._EXPANDED_BASELINE)

    def test_on_with_a_share_appends_northbound_and_sector(self):
        names = self._tool_names({"a_share_native": True}, ticker="600519.SS")
        # Baseline tools + 4 new A-share market tools.
        self.assertIn("get_stock_data", names)
        self.assertIn("get_indicators", names)
        self.assertIn("get_verified_market_snapshot", names)
        self.assertIn("get_a_share_northbound_native", names)
        self.assertIn("get_a_share_sector_flow_native", names)
        self.assertIn("get_a_share_realtime_quote_native", names)
        self.assertIn("get_a_share_market_breadth_native", names)

    def test_on_with_non_a_share_is_byte_equivalent(self):
        names = self._tool_names({"a_share_native": True}, ticker="AAPL")
        self.assertEqual(names, self._EXPANDED_BASELINE)


# --------------------------------------------------------------------------- #
# BaoStock quarterly statements (Phase 6 — income / balance / cashflow)
# --------------------------------------------------------------------------- #
# Rows use the REAL server-delivered field names (pinned in
# yialpha.dataflows.baostock_fields against the official docs) — the previous
# fixtures invented npParentCompanyOwners/totalAssets/netCFOperate fields the
# endpoints never return, so the mocks mirrored the renderers' bug and the
# all-n/a tables looked "tested". Ratio fields are decimal fractions per the
# official samples (roeAvg 0.074617 == 7.46%).
from yialpha.dataflows import baostock_fields as bsf  # noqa: E402


@pytest.mark.unit
def test_statement_columns_subset_of_official_field_tables():
    """A6 regression: every rendered column field must exist in its endpoint's
    official field table. A renderer reading a field BaoStock never returns
    silently renders all-n/a — the exact bug class behind the three statement
    tables (totalShare-as-Revenue, totalAssets, netCFOperate)."""
    for columns, fields in (
        (bsf.PROFIT_COLUMNS, bsf.PROFIT_DATA_FIELDS),
        (bsf.BALANCE_COLUMNS, bsf.BALANCE_DATA_FIELDS),
        (bsf.CASH_FLOW_COLUMNS, bsf.CASH_FLOW_DATA_FIELDS),
    ):
        for col in columns:
            assert col.field in fields, (
                f"rendered field {col.field!r} is not in the endpoint's "
                f"official field table {fields}")
    # The fabricated fields that caused the all-n/a tables must stay absent.
    for banned in ("totalAssets", "totalLiab", "totalShareholdersEquity",
                   "liabilityRate", "netCFOperate", "netCFInvest", "netCFFinance",
                   "npParentCompanyOwners", "operateProfit"):
        for fields in (bsf.PROFIT_DATA_FIELDS, bsf.BALANCE_DATA_FIELDS,
                       bsf.CASH_FLOW_DATA_FIELDS):
            assert banned not in fields
    # Format kinds are constrained to what _fmt_cell implements.
    for columns in (bsf.PROFIT_COLUMNS, bsf.BALANCE_COLUMNS, bsf.CASH_FLOW_COLUMNS):
        for col in columns:
            assert col.kind in ("cny", "pct", "x", "num"), col


def _profit_rows():
    """Synthetic BaoStock query_profit_data rows (real field names)."""
    return [
        {"code": "sh.600519", "pubDate": "2024-10-30", "statDate": "2024-09-30",
         "roeAvg": 0.152, "npMargin": 0.50, "gpMargin": 0.91,
         "netProfit": 6.08e10, "epsTTM": 48.42, "MBRevenue": 1.21e11,
         "totalShare": 1.256e9, "liqaShare": 1.256e9},
        {"code": "sh.600519", "pubDate": "2024-08-29", "statDate": "2024-06-30",
         "roeAvg": 0.148, "npMargin": 0.49, "gpMargin": 0.91,
         "netProfit": 4.17e10, "epsTTM": 33.19, "MBRevenue": 8.6e10,
         "totalShare": 1.256e9, "liqaShare": 1.256e9},
        {"code": "sh.600519", "pubDate": "2025-04-15", "statDate": "2024-12-31",
         "roeAvg": 0.161, "npMargin": 0.52, "gpMargin": 0.92,
         "netProfit": 8.6e10, "epsTTM": 68.47, "MBRevenue": 1.7e11,
         "totalShare": 1.256e9, "liqaShare": 1.256e9},  # future pubDate
    ]


def _balance_rows():
    """Synthetic BaoStock query_balance_data rows (real field names)."""
    return [
        {"code": "sh.600519", "pubDate": "2024-10-30", "statDate": "2024-09-30",
         "currentRatio": 3.81, "quickRatio": 2.75, "cashRatio": 1.93,
         "YOYLiability": -0.04, "liabilityToAsset": 0.2192,
         "assetToEquity": 1.28},
        {"code": "sh.600519", "pubDate": "2024-08-29", "statDate": "2024-06-30",
         "currentRatio": 3.91, "quickRatio": 2.86, "cashRatio": 2.05,
         "YOYLiability": -0.02, "liabilityToAsset": 0.2121,
         "assetToEquity": 1.27},
    ]


def _cashflow_rows():
    """Synthetic BaoStock query_cash_flow_data rows (real field names)."""
    return [
        {"code": "sh.600519", "pubDate": "2024-10-30", "statDate": "2024-09-30",
         "CAToAsset": 0.72, "NCAToAsset": 0.28, "tangibleAssetToAsset": 0.99,
         "ebitToInterest": 78.5, "CFOToOR": 0.51, "CFOToNP": 1.02,
         "CFOToGr": 0.51},
        {"code": "sh.600519", "pubDate": "2024-08-29", "statDate": "2024-06-30",
         "CAToAsset": 0.73, "NCAToAsset": 0.27, "tangibleAssetToAsset": 0.99,
         "ebitToInterest": 80.1, "CFOToOR": 0.48, "CFOToNP": 0.99,
         "CFOToGr": 0.48},
    ]


@pytest.mark.unit
def test_income_statement_pit_drops_future(monkeypatch):
    monkeypatch.setattr(bsv, "_statement_rows", lambda code, fn, anchor=None: bsv.StatementFetch(_profit_rows(), []))
    out = bsv.get_a_share_income_statement_native("600519.SS", "2024-11-01", 540)
    assert "# A-share Income Statement" in out
    assert "2024-10-30" in out         # published before curr_date -> kept
    assert "2024-08-29" in out         # kept
    assert "2025-04-15" not in out     # future pubDate -> PIT drop
    # Real fields only: revenue=MBRevenue, profit=netProfit; roeAvg is a
    # fraction (0.152) rendered as 15.20%; epsTTM as-is.
    assert "1210.00亿" in out          # MBRevenue 1.21e11 -> 1210.00亿
    assert "608.00亿" in out           # netProfit 6.08e10 -> 608.00亿
    assert "15.20%" in out             # roeAvg 0.152 x100
    assert "48.42" in out              # epsTTM
    assert "OpProfit" not in out       # no operating-profit field exists


@pytest.mark.unit
def test_income_statement_empty_honest(monkeypatch):
    monkeypatch.setattr(bsv, "_statement_rows", lambda code, fn, anchor=None: bsv.StatementFetch([], []))
    out = bsv.get_a_share_income_statement_native("600519.SS", "2024-06-15", 540)
    assert "No income-statement rows" in out


@pytest.mark.unit
def test_income_statement_failed_quarters_visible_in_band(monkeypatch):
    """Quarters dropped by fetch failures must be marked in the output — a
    truncated revenue trend is not a disclosure gap the agent should guess at."""
    monkeypatch.setattr(
        bsv, "_statement_rows",
        lambda code, fn, anchor=None: bsv.StatementFetch(_profit_rows(), ["2023Q4", "2024Q1"]),
    )
    out = bsv.get_a_share_income_statement_native("600519.SS", "2024-11-01", 540)
    assert "⚠ 2 quarter(s) could not be fetched" in out
    assert "2023Q4" in out and "2024Q1" in out
    assert "do not read the missing quarters as zero" in out


@pytest.mark.unit
def test_balance_sheet_pit(monkeypatch):
    monkeypatch.setattr(bsv, "_statement_rows", lambda code, fn, anchor=None: bsv.StatementFetch(_balance_rows(), []))
    out = bsv.get_a_share_balance_sheet_native("600519.SS", "2024-11-01", 540)
    assert "# A-share Balance Sheet" in out
    assert "2024-10-30" in out
    # Real fields only (solvency ratios; the endpoint has NO total-assets
    # levels): liabilityToAsset 0.2192 is a fraction rendered as 21.92%.
    assert "21.92%" in out            # liabilityToAsset 0.2192 x100
    assert "3.81x" in out             # currentRatio as a multiple
    assert "totalAssets" not in out   # fabricated stock levels must be gone
    assert "RATIOS ONLY" in out       # the capability caveat is in-band


@pytest.mark.unit
def test_cashflow_statement_pit(monkeypatch):
    monkeypatch.setattr(bsv, "_statement_rows", lambda code, fn, anchor=None: bsv.StatementFetch(_cashflow_rows(), []))
    out = bsv.get_a_share_cashflow_statement_native("600519.SS", "2024-11-01", 540)
    assert "# A-share Cashflow Statement" in out
    assert "2024-10-30" in out
    # Real fields only (quality ratios; no absolute CFO amounts exist).
    assert "51.00%" in out            # CFOToOR 0.51 x100
    assert "1.02x" in out             # CFOToNP multiple
    assert "netCFOperate" not in out  # fabricated absolute-flow fields gone
    assert "RATIOS ONLY" in out


@pytest.mark.unit
def test_router_routes_income_statement_via_baostock(monkeypatch):
    monkeypatch.setattr(bsv, "_statement_rows", lambda code, fn, anchor=None: bsv.StatementFetch(_profit_rows(), []))
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "baostock"}})
        out = route_to_vendor("get_a_share_income_statement_native", "600519.SS",
                              "2024-11-01", 540)
    finally:
        cfgmod.set_config(orig)
    assert "# A-share Income Statement" in out


@pytest.mark.unit
def test_router_income_statement_non_a_share_degrades(monkeypatch):
    from yialpha.dataflows import config as cfgmod
    from yialpha.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "baostock"}})
        out = route_to_vendor("get_a_share_income_statement_native", "AAPL",
                              "2024-11-01", 540)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


class FundamentalsAShareStatementWiringTests(unittest.TestCase):
    """Verify the 3 quarterly-statement tools are bound to the fundamentals analyst."""

    def _tool_names(self, config_overrides=None, ticker="600519.SS"):
        from yialpha.dataflows import config as cfgmod
        orig = cfgmod.get_config()
        try:
            if config_overrides:
                cfgmod.set_config({**orig, **config_overrides})
            llm = _RecordingLLM()
            node = create_fundamentals_analyst(llm)
            node(_state(ticker))
            return [t.name for t in llm.bound_tools]
        finally:
            cfgmod.set_config(orig)

    def test_on_with_a_share_appends_statement_tools(self):
        names = self._tool_names({"a_share_native": True}, ticker="600519.SS")
        self.assertIn("get_a_share_income_statement_native", names)
        self.assertIn("get_a_share_balance_sheet_native", names)
        self.assertIn("get_a_share_cashflow_statement_native", names)


if __name__ == "__main__":
    unittest.main()
