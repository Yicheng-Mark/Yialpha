"""Unit tests for ``yiagents.dataflows.eastmoney`` (A-share margin trading),
the router wiring, and the fundamentals-analyst byte-equivalence gate (Track A).

Hermetic: no network. The vendor path is exercised by patching
``eastmoney._cached_or_fetch`` to serve synthetic JSON payloads (built from the
real endpoint shape verified against Eastmoney's live API). Covers: symbol
mapping, margin parsing + PIT + honest-empty degradation, the ``_direct_get``
status/retry behaviour, the direct-connect (``trust_env=False`` +
``proxies={}``) guarantee, router integration (routes via the eastmoney vendor;
optional category degrades to a sentinel on a non-A-share ticker), and the
wiring byte-equivalence contract (the project "iron rule").

The wiring tests pin the double gate (flag AND is_a_stock): with ``a_stock`` off
(the default) the fundamentals analyst binds exactly its 4 baseline tools and an
unchanged prompt regardless of ticker; with it on, the one new tool is appended
only for an A-share ticker. Pure mock-LLM, zero network, zero LLM cost.
"""

from __future__ import annotations

import json
import unittest

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from yiagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from yiagents.dataflows import eastmoney
from yiagents.dataflows.errors import NoMarketDataError, VendorRateLimitError
from yiagents.dataflows.symbol_utils import is_a_stock


# --------------------------------------------------------------------------- #
# Synthetic payloads (shapes verified against Eastmoney's live API)
# --------------------------------------------------------------------------- #
def _margin_json(rows: list[dict], success: bool = True) -> bytes:
    return json.dumps({"version": "v1", "result": {"pages": 1, "data": rows},
                       "success": success, "message": "ok", "code": 200}).encode("utf-8")


def _margin_row(d: str, rzye=1.0e10, rqye=2.0e9, rzmre=5.0e8,
                rqmcl=100000, rzjme=0.0, rzyezb=0.5) -> dict:
    return {"DATE": f"{d} 00:00:00", "SCODE": "600519", "SECNAME": "KweichowMoutai",
            "RZYE": rzye, "RQYE": rqye, "RZRQYE": rzye + rqye, "RZMRE": rzmre,
            "RQMCL": rqmcl, "RQYL": rqmcl * 10, "RZJME": rzjme, "RZYEZB": rzyezb}


_MARGIN_ROWS = [
    _margin_row("2024-07-01", rzye=1.1e10, rzjme=1.0e8),     # > curr -> PIT drop
    _margin_row("2024-06-15", rzye=1.0e10, rzjme=-1.0e8),
    _margin_row("2024-06-10", rzye=9.0e9, rzjme=3.0e8),
]


def _patch_fetch(monkeypatch, tmp_path, payload_for_url):
    """Serve `payload_for_url(url)` from the cache; everything else is a real
    path under tmp_path. `payload_for_url` returns bytes or raises."""
    monkeypatch.setattr(eastmoney, "_cache_dir", lambda: str(tmp_path))

    def fake_fetch(_path, url, params=None, ttl_days=1.0):
        return payload_for_url(url)
    monkeypatch.setattr(eastmoney, "_cached_or_fetch", fake_fetch)


# --------------------------------------------------------------------------- #
# Symbol mapping
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_is_a_stock_helper():
    assert is_a_stock("600519.SS") is True
    assert is_a_stock("600519.SH") is True
    assert is_a_stock("000001.SZ") is True
    assert is_a_stock("600519.ss") is True          # case-insensitive
    assert is_a_stock("AAPL") is False
    assert is_a_stock("0700.HK") is False
    assert is_a_stock("BTCUSDT") is False
    assert is_a_stock("60051.SS") is False          # not 6 digits
    assert is_a_stock("") is False
    assert is_a_stock(None) is False


@pytest.mark.unit
def test_to_em_symbol_shanghai():
    assert eastmoney._to_em_symbol("600519.SS") == ("1.600519", "600519")
    assert eastmoney._to_em_symbol("600519.SH") == ("1.600519", "600519")


@pytest.mark.unit
def test_to_em_symbol_shenzhen():
    assert eastmoney._to_em_symbol("000001.SZ") == ("0.000001", "000001")


@pytest.mark.unit
def test_to_em_symbol_non_a_share_raises():
    for bad in ("AAPL", "0700.HK", "BTCUSDT", "60051.SS", ""):
        with pytest.raises(NoMarketDataError):
            eastmoney._to_em_symbol(bad)


# --------------------------------------------------------------------------- #
# Margin trading (融资融券)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_margin_pit_drops_future_rows(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path, lambda u: _margin_json(_MARGIN_ROWS))
    out = eastmoney.get_margin_trading("600519.SS", "2024-06-20", 180)
    assert "# Margin Trading" in out
    assert "2024-06-15" in out
    assert "2024-06-10" in out
    assert "2024-07-01" not in out      # PIT dropped


@pytest.mark.unit
def test_margin_summary_rising_balance(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path, lambda u: _margin_json(_MARGIN_ROWS))
    out = eastmoney.get_margin_trading("600519.SS", "2024-06-20", 180)
    # end (06-15, RZYE 100亿) > start (06-10, RZYE 90亿) -> rising.
    assert "rising margin balance" in out
    # RZJME window sum = -1e8 + 3e8 = +2e8 = +2.00 亿.
    assert "+2.00" in out
    # RZYEZB is already in percent units (rzyezb=0.5 -> "0.50", NOT "50.00").
    assert "0.50" in out
    assert "50.00" not in out


@pytest.mark.unit
def test_margin_name_not_on_list_honest_empty(monkeypatch, tmp_path):
    # success=True but no rows: the name is not on the 两融 list.
    _patch_fetch(monkeypatch, tmp_path, lambda u: _margin_json([], success=True))
    out = eastmoney.get_margin_trading("600519.SS", "2024-06-20", 180)
    assert out.startswith("# Margin Trading")
    assert "No margin-trading" in out
    assert "do not estimate" in out


@pytest.mark.unit
def test_margin_success_false_honest_empty(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path, lambda u: _margin_json([], success=False))
    out = eastmoney.get_margin_trading("600519.SS", "2024-06-20", 180)
    assert "No margin-trading" in out


@pytest.mark.unit
def test_margin_garbage_json_raises(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path, lambda u: b"<<bad>>")
    with pytest.raises(NoMarketDataError):
        eastmoney.get_margin_trading("600519.SS", "2024-06-20", 180)


# --------------------------------------------------------------------------- #
# _direct_get transport: status mapping, retry, direct-connect guarantee
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status=200, content=b"{}"):
        self.status_code = status
        self.content = content


class _FakeSession:
    """Records the call kwargs (so we can assert proxies={}) and replays a
    scripted sequence of responses/exceptions across retries."""
    def __init__(self, script):
        # script: list of either _FakeResp or an Exception instance to raise.
        self._script = list(script)
        self.calls = []   # list of the proxies kwarg passed per get()

    trust_env = "UNSET"   # _session() sets this to False; recorded for assertion

    def get(self, url, params=None, headers=None, proxies=None, timeout=None):
        self.calls.append(proxies)
        item = self._script.pop(0) if self._script else _FakeResp()
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.unit
def test_session_bypasses_env_proxy():
    """The real _session() must disable trust_env (the only reliable env-proxy
    bypass; proxies={} alone leaks, verified against Eastmoney)."""
    s = eastmoney._session()
    assert s.trust_env is False


@pytest.mark.unit
def test_direct_get_passes_proxies_empty_and_200(monkeypatch):
    monkeypatch.setattr(eastmoney, "_throttle", lambda: None)
    fake = _FakeSession([_FakeResp(200, b'{"ok":1}')])
    monkeypatch.setattr(eastmoney, "_session", lambda: fake)
    raw = eastmoney._direct_get("http://x", {"a": "1"})
    assert raw == b'{"ok":1}'
    assert fake.calls == [{}]            # proxies={} passed every attempt


@pytest.mark.unit
def test_direct_get_429_maps_to_rate_limit(monkeypatch):
    monkeypatch.setattr(eastmoney, "_throttle", lambda: None)
    monkeypatch.setattr(eastmoney, "_session",
                        lambda: _FakeSession([_FakeResp(429)]))
    with pytest.raises(VendorRateLimitError):
        eastmoney._direct_get("http://x")


@pytest.mark.unit
def test_direct_get_500_maps_to_no_data(monkeypatch):
    monkeypatch.setattr(eastmoney, "_throttle", lambda: None)
    monkeypatch.setattr(eastmoney, "_session",
                        lambda: _FakeSession([_FakeResp(500)]))
    with pytest.raises(NoMarketDataError):
        eastmoney._direct_get("http://x")


@pytest.mark.unit
def test_direct_get_retries_transient_then_raises(monkeypatch):
    monkeypatch.setattr(eastmoney, "_throttle", lambda: None)
    monkeypatch.setattr(eastmoney, "_RETRY_BACKOFF", 0)
    import requests as _rq
    # Two transient ConnectionErrors exhaust the retries (1 + 2 = 3 attempts).
    monkeypatch.setattr(eastmoney, "_session", lambda: _FakeSession([
        _rq.exceptions.ConnectionError("drop"),
        _rq.exceptions.ConnectionError("drop"),
    ]))
    with pytest.raises(NoMarketDataError):
        eastmoney._direct_get("http://x")


@pytest.mark.unit
def test_direct_get_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(eastmoney, "_throttle", lambda: None)
    monkeypatch.setattr(eastmoney, "_RETRY_BACKOFF", 0)
    import requests as _rq
    fake = _FakeSession([
        _rq.exceptions.ConnectionError("drop"),
        _FakeResp(200, b'{"ok":2}'),
    ])
    monkeypatch.setattr(eastmoney, "_session", lambda: fake)
    assert eastmoney._direct_get("http://x") == b'{"ok":2}'
    assert len(fake.calls) == 2           # retried once


@pytest.mark.unit
def test_cached_or_fetch_serves_stale_on_failure(monkeypatch, tmp_path):
    """A fetch failure with a stale cache present -> stale bytes, not an error."""
    import os
    monkeypatch.setattr(eastmoney, "_cache_dir", lambda: str(tmp_path))
    p = os.path.join(str(tmp_path), "stale.json")
    with open(p, "wb") as fh:
        fh.write(b'{"stale":true}')

    # Patch _direct_get to fail; the cache file is "stale" (ttl_days=0 forces a
    # fetch attempt) -> fetch fails -> stale bytes served.
    monkeypatch.setattr(eastmoney, "_direct_get",
                        lambda *a, **k: (_ for _ in ()).throw(
                            NoMarketDataError("x", detail="boom")))
    raw = eastmoney._cached_or_fetch(p, "http://x", {"a": 1}, ttl_days=0.0)
    assert raw == b'{"stale":true}'


# --------------------------------------------------------------------------- #
# Router integration
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_router_routes_margin_via_eastmoney(monkeypatch, tmp_path):
    _patch_fetch(monkeypatch, tmp_path, lambda u: _margin_json(_MARGIN_ROWS))
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_stock": "eastmoney"}})
        out = route_to_vendor("get_margin_trading", "600519.SS", "2024-06-20", 180)
    finally:
        cfgmod.set_config(orig)
    assert "Margin Trading" in out


@pytest.mark.unit
def test_router_optional_category_degrades_to_sentinel(monkeypatch, tmp_path):
    """A non-A-share ticker -> NoMarketDataError -> the router's NO_DATA_AVAILABLE
    sentinel (optional category never re-raises)."""
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_stock": "eastmoney"}})
        out = route_to_vendor("get_margin_trading", "AAPL", "2024-06-20", 180)
    finally:
        cfgmod.set_config(orig)
    assert out.startswith("NO_DATA_AVAILABLE")


@pytest.mark.unit
def test_router_eastmoney_host_failure_degrades_to_sentinel(monkeypatch, tmp_path):
    """A transport failure on the optional category -> a sentinel (graceful, no
    crash). _direct_get maps transport errors to NoMarketDataError, so the
    router emits its NO_DATA_AVAILABLE sentinel (the last-no-data path)."""
    monkeypatch.setattr(eastmoney, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(eastmoney, "_direct_get",
                        lambda *a, **k: (_ for _ in ()).throw(
                            NoMarketDataError("x", detail="host down")))
    from yiagents.dataflows import config as cfgmod
    from yiagents.dataflows.interface import route_to_vendor

    orig = cfgmod.get_config()
    try:
        cfgmod.set_config({**orig, "data_vendors": {**orig.get("data_vendors", {}),
                                                    "a_stock": "eastmoney"}})
        out = route_to_vendor("get_margin_trading", "600519.SS", "2024-06-20", 180)
    finally:
        cfgmod.set_config(orig)
    # Graceful degradation: either sentinel, never a raised crash.
    assert out.startswith("NO_DATA_AVAILABLE") or out.startswith("DATA_UNAVAILABLE")
    assert "host down" in out          # the failure reason is surfaced


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


def _state(ticker: str):
    return {
        "trade_date": "2024-06-15",
        "company_of_interest": ticker,
        "asset_type": "stock",
        "instrument_context": f"CTX {ticker}",
        "messages": [HumanMessage(content="analyze")],
    }


class FundamentalsWiringTests(unittest.TestCase):
    """Track A — default-off byte-equivalence + double-gate (flag AND is_a_stock)."""

    def _tool_names(self, config_overrides, ticker):
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

    BASELINE = ["get_fundamentals", "get_balance_sheet", "get_cashflow",
                "get_income_statement"]

    def test_default_off_byte_equivalent_a_share(self):
        # a_stock off -> exactly the 4 baseline tools, even for an A-share ticker.
        names = self._tool_names({"a_stock": False}, "600519.SS")
        self.assertEqual(names, self.BASELINE)

    def test_default_off_byte_equivalent_us(self):
        names = self._tool_names({"a_stock": False}, "AAPL")
        self.assertEqual(names, self.BASELINE)

    def test_on_appends_one_for_a_share(self):
        names = self._tool_names({"a_stock": True}, "600519.SS")
        self.assertEqual(names, self.BASELINE + ["get_margin_trading"])

    def test_on_does_not_append_for_non_a_share(self):
        # Double gate: flag on but ticker is NOT A-share -> baseline unchanged.
        names = self._tool_names({"a_stock": True}, "AAPL")
        self.assertEqual(names, self.BASELINE)

    def test_on_shenzhen_ticker_appends_one(self):
        names = self._tool_names({"a_stock": True}, "000001.SZ")
        self.assertEqual(names, self.BASELINE + ["get_margin_trading"])

    def test_compose_with_sec_ownership_independent(self):
        # a_stock on + sec_ownership on, A-share -> 4 + 3 (sec) + 1 (a_stock) = 8.
        names = self._tool_names(
            {"a_stock": True, "sec_ownership": True}, "600519.SS")
        self.assertEqual(names, self.BASELINE + [
            "get_form4_insider_trading", "get_ftd_data",
            "get_institutional_holdings", "get_margin_trading"])

    def test_compose_sec_ownership_only_for_us(self):
        # a_stock on + sec_ownership on, US ticker -> sec tools appended,
        # a_stock tool NOT (not A-share). Independent flags.
        names = self._tool_names(
            {"a_stock": True, "sec_ownership": True}, "AAPL")
        self.assertEqual(names, self.BASELINE + [
            "get_form4_insider_trading", "get_ftd_data",
            "get_institutional_holdings"])


if __name__ == "__main__":
    unittest.main()
