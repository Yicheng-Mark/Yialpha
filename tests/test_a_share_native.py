"""Unit tests for ``yiagents.dataflows.baostock_vendor`` and the
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

from yiagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from yiagents.dataflows import akshare_vendor as akv, baostock_vendor as bsv
from yiagents.dataflows.errors import NoMarketDataError, VendorRateLimitError


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
    monkeypatch.setattr(bsv, "_cached_daily", lambda code: list(rows))


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
    monkeypatch.setattr(bsv, "_cached_daily", lambda code: [])  # short-circuit cache
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
# Router integration
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_router_routes_ohlc_via_baostock(monkeypatch):
    _patch_daily(monkeypatch)
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

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
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "baostock"}})
        out = route_to_vendor("get_a_share_ohlc_native", "AAPL", "2024-06-15", 180)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


@pytest.mark.unit
def test_router_missing_dependency_degrades_to_sentinel(monkeypatch):
    """baostock not installed -> NoMarketDataError from _require_baostock only
    fires on the login path; _cached_daily is patched to force that path."""
    import builtins
    real_import = builtins.__import__

    def _block(name, *a, **k):
        if name == "baostock":
            raise ImportError("no baostock")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _block)
    # Blow the on-disk cache so _cached_daily actually logs in.
    monkeypatch.setattr(bsv, "_read_cache", lambda p: None)
    monkeypatch.setattr(bsv, "_write_cache", lambda p, r: None)
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

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
        from yiagents.dataflows import config as cfgmod
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
             "get_a_share_dragon_tiger_native"],
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
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

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
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

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
    """Patch _require_akshare to a fake module exposing the given callables."""
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
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor
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
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor
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
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor
    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_share_native": "akshare"}})
        out = route_to_vendor("get_a_share_money_flow_native", "AAPL", "2024-06-15", 30)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


class NewsAShareWiringTests(unittest.TestCase):
    """a_share_native news — default-off byte-equivalence + on-appends-news-tool."""

    def _tool_names(
        self,
        config_overrides=None,
        ticker="600519.SS",
        trade_date=None,
    ):
        from yiagents.agents.analysts.news_analyst import create_news_analyst
        from yiagents.dataflows import config as cfgmod
        orig = cfgmod.get_config()
        try:
            if config_overrides:
                cfgmod.set_config({**orig, **config_overrides})
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
from yiagents.dataflows import tushare_vendor as tv  # noqa: E402
from yiagents.dataflows.errors import VendorNotConfiguredError  # noqa: E402


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
    # YiAgents .SS (yfinance Shanghai) maps to Tushare's .SH.
    assert tv._to_tushare_code("600519.SS") == "600519.SH"
    assert tv._to_tushare_code("600000.SH") == "600000.SH"


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
    from yiagents.dataflows import config as cfgmod
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
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor
    # BaoStock path: serve synthetic daily rows.
    monkeypatch.setattr(bsv, "_cached_daily", lambda code: list(SYNTH_ROWS))

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


if __name__ == "__main__":
    unittest.main()
