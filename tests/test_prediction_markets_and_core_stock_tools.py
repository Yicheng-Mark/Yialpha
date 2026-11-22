"""Prediction-markets and core-stock tool wrapper pins (round 13).

``agents/utils/prediction_markets_tools.py`` and
``agents/utils/core_stock_tools.py`` had zero test references despite being
LLM-reachable execution surface (every news analyst run can call
``get_prediction_markets``; every market analyst run calls
``get_stock_data``). Same thin-pass-through shape as the Binance twins; this
file pins, for both modules:

* tool NAME + argument schema (types, defaults, Annotated descriptions) —
  what the LLM actually sees;
* exact vendor-name + positional mapping through ``route_to_vendor``
  (monkeypatched, the ``test_binance_perp_tools.py`` pattern);
* the error contract, which differs BY CATEGORY and is the whole reason the
  two are tested together:
  - ``prediction_markets`` is an OPTIONAL category -> a 429 / network error
    degrades to the ``DATA_UNAVAILABLE`` sentinel, NoMarketDataError to the
    ``NO_DATA_AVAILABLE`` sentinel (flavour data never aborts the run);
  - ``core_stock_apis`` is a CORE category -> the same vendor failure
    RE-RAISES (loud), because a broken primary price feed must not be
    swallowed into a sentinel.
"""

from __future__ import annotations

import pytest

import yialpha.agents.utils.core_stock_tools as cst
import yialpha.agents.utils.prediction_markets_tools as pmt
from yialpha.dataflows import interface
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError


def _recorder(calls: list):
    def fake_route(method, *args, _qualifier="", **kwargs):
        calls.append(
            {"method": method, "args": args, "kwargs": kwargs, "qualifier": _qualifier}
        )
        return "VENDOR_OK"

    return fake_route


@pytest.fixture()
def routed_pm(monkeypatch):
    calls: list = []
    monkeypatch.setattr(pmt, "route_to_vendor", _recorder(calls))
    return calls


@pytest.fixture()
def routed_stock(monkeypatch):
    calls: list = []
    monkeypatch.setattr(cst, "route_to_vendor", _recorder(calls))
    return calls


def _failing_vendor(method, raiser):
    return {**interface.VENDOR_METHODS, method: {"polymarket": raiser}}


# ---------------------------------------------------------------------------
# prediction_markets_tools
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPredictionMarketsSchema:
    def test_tool_name(self):
        assert pmt.get_prediction_markets.name == "get_prediction_markets"

    def test_schema(self):
        args = pmt.get_prediction_markets.args
        assert set(args) == {"topic", "limit"}
        assert args["topic"]["type"] == "string"
        assert args["topic"]["description"].startswith("Event topic/keyword")
        # limit: optional int | None, default None (vendor applies its own 6).
        assert args["limit"]["default"] is None
        assert {"type": "integer"} in args["limit"]["anyOf"]
        assert {"type": "null"} in args["limit"]["anyOf"]


@pytest.mark.unit
class TestPredictionMarketsPassthrough:
    def test_topic_and_limit_positional(self, routed_pm):
        out = pmt.get_prediction_markets.invoke({
            "topic": "Fed rate cut", "limit": 10,
        })
        assert routed_pm == [{
            "method": "get_prediction_markets",
            "args": ("Fed rate cut", 10),
            "kwargs": {},
            "qualifier": "",
        }]
        assert out == "VENDOR_OK"

    def test_default_limit_threads_none(self, routed_pm):
        pmt.get_prediction_markets.invoke({"topic": "recession 2026"})
        assert routed_pm[0]["args"] == ("recession 2026", None)


@pytest.mark.unit
class TestPredictionMarketsErrorContract:
    def test_rate_limit_degrades_to_sentinel(self, monkeypatch):
        def _throttled(*_a, **_k):
            raise VendorRateLimitError("Polymarket HTTP 429")

        monkeypatch.setattr(
            interface, "VENDOR_METHODS",
            _failing_vendor("get_prediction_markets", _throttled),
        )
        out = pmt.get_prediction_markets.invoke({"topic": "election"})
        assert out.startswith("DATA_UNAVAILABLE")
        assert "prediction_markets" in out
        assert "429" in out

    def test_network_error_degrades_to_sentinel(self, monkeypatch):
        def _down(*_a, **_k):
            raise ConnectionError("connection reset")

        monkeypatch.setattr(
            interface, "VENDOR_METHODS",
            _failing_vendor("get_prediction_markets", _down),
        )
        out = pmt.get_prediction_markets.invoke({"topic": "election"})
        assert out.startswith("DATA_UNAVAILABLE")
        assert "prediction_markets" in out

    def test_no_data_degrades_to_no_data_sentinel(self, monkeypatch):
        def _empty(*_a, **_k):
            raise NoMarketDataError("election", "election", "no open markets")

        monkeypatch.setattr(
            interface, "VENDOR_METHODS",
            _failing_vendor("get_prediction_markets", _empty),
        )
        out = pmt.get_prediction_markets.invoke({"topic": "election"})
        assert out.startswith("NO_DATA_AVAILABLE")
        assert "election" in out
        assert "no open markets" in out

    def test_category_is_optional_single_vendor(self):
        assert interface.get_category_for_method("get_prediction_markets") \
            == "prediction_markets"
        assert "prediction_markets" in interface.OPTIONAL_CATEGORIES
        assert list(interface.VENDOR_METHODS["get_prediction_markets"]) \
            == ["polymarket"]


# ---------------------------------------------------------------------------
# core_stock_tools
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCoreStockSchema:
    def test_tool_name(self):
        assert cst.get_stock_data.name == "get_stock_data"

    def test_schema(self):
        args = cst.get_stock_data.args
        assert set(args) == {"symbol", "start_date", "end_date"}
        for name in args:
            assert args[name]["type"] == "string"
            assert args[name]["description"].strip()
        assert args["symbol"]["description"] == "ticker symbol of the company"


@pytest.mark.unit
class TestCoreStockPassthrough:
    def test_three_positionals(self, routed_stock):
        out = cst.get_stock_data.invoke({
            "symbol": "AAPL",
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
        })
        assert routed_stock == [{
            "method": "get_stock_data",
            "args": ("AAPL", "2026-01-01", "2026-01-10"),
            "kwargs": {},
            "qualifier": "",
        }]
        assert out == "VENDOR_OK"


@pytest.mark.unit
class TestCoreStockErrorContract:
    def test_rate_limit_reraises_loudly(self, monkeypatch):
        # core_stock_apis is a CORE category: a throttled primary must
        # surface the typed error (after the yfinance fallback chain fails),
        # NOT degrade to a sentinel — a broken price feed cannot be silently
        # swallowed. Patch BOTH chain vendors so nothing can serve it.
        def _throttled(*_a, **_k):
            raise VendorRateLimitError("HTTP 429")

        monkeypatch.setattr(
            interface, "VENDOR_METHODS",
            {"get_stock_data": {"yfinance": _throttled, "alpha_vantage": _throttled}},
        )
        with pytest.raises(VendorRateLimitError):
            cst.get_stock_data.invoke({
                "symbol": "AAPL",
                "start_date": "2026-01-01",
                "end_date": "2026-01-10",
            })

    def test_no_data_yields_no_data_sentinel(self, monkeypatch):
        # Even a core category degrades (not raises) when the vendor reports
        # a clean "no data": the symbol is genuinely unavailable, and the
        # instructive sentinel tells the LLM not to fabricate prices.
        def _empty(*_a, **_k):
            raise NoMarketDataError("ZZZZ", "ZZZZ", "delisted")

        monkeypatch.setattr(
            interface, "VENDOR_METHODS",
            {"get_stock_data": {"yfinance": _empty, "alpha_vantage": _empty}},
        )
        out = cst.get_stock_data.invoke({
            "symbol": "ZZZZ",
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
        })
        assert out.startswith("NO_DATA_AVAILABLE")
        assert "ZZZZ" in out
        assert "delisted" in out

    def test_category_is_core_with_default_chain(self):
        assert interface.get_category_for_method("get_stock_data") \
            == "core_stock_apis"
        assert "core_stock_apis" not in interface.OPTIONAL_CATEGORIES
        # The documented default chain (yfinance leads, AV trails) is what
        # the unset-category default resolves to.
        assert interface._default_chain_order(
            "core_stock_apis", list(interface.VENDOR_METHODS["get_stock_data"])
        ) == ["yfinance", "alpha_vantage"]
