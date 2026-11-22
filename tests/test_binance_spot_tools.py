"""Binance SPOT tool wrapper pins (round 13).

The three ``@tool`` wrappers in ``agents/utils/binance_spot_tools`` are thin
pass-throughs to ``route_to_vendor`` — the same shape as the perp twins
(``test_binance_perp_tools.py``, whose structure this file mirrors). Until now
they had zero test references: a silently swapped positional would serve the
wrong window's candles with no error anywhere, and the module's headline
promise — the optional-category fallback (429/network -> sentinel) "reused
verbatim" from the perp version — was never verified for the ``binance_spot``
category at all.

Pinned here:
* every wrapper's NAME + argument schema (types, defaults, Annotated
  descriptions) — the LLM sees exactly this schema;
* exact vendor-name + positional mapping through ``route_to_vendor``
  (monkeypatched, same as the perp file);
* vendor success rendered verbatim (no wrapping/reformatting in the tool);
* 429 / network-error degradation through the REAL router (vendor impl
  replaced via ``VENDOR_METHODS`` patching, the ``test_vendor_errors.py``
  pattern): ``binance_spot`` is an OPTIONAL category, so the tool returns the
  ``DATA_UNAVAILABLE`` sentinel instead of raising;
* NoMarketDataError (e.g. a perp with no spot listing on the basis tool) ->
  the ``NO_DATA_AVAILABLE`` sentinel naming the symbol;
* the category contract itself: every spot method maps to category
  ``binance_spot`` (in OPTIONAL_CATEGORIES, single "binance" vendor), which
  is what makes the sentinel behaviour above real. The wrappers themselves
  are ungated pass-throughs — asset-type gating lives in the market analyst
  (pinned in ``test_crypto_spot_mode.py``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import yialpha.agents.utils.binance_spot_tools as bst
from yialpha.dataflows import interface
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError


def _recorder(calls: list):
    def fake_route(method, *args, _qualifier="", **kwargs):
        calls.append(
            {
                "method": method,
                "args": args,
                "kwargs": kwargs,
                "qualifier": _qualifier,
            }
        )
        return "VENDOR_OK"

    return fake_route


@pytest.fixture()
def routed(monkeypatch):
    calls: list = []
    monkeypatch.setattr(bst, "route_to_vendor", _recorder(calls))
    return calls


def _throttled(*_a, **_k):
    raise VendorRateLimitError("Binance HTTP 429")


def _network_down(*_a, **_k):
    raise ConnectionError("network unreachable")


_SPOT_TOOLS = ("klines", "ticker24", "perp_basis")


@pytest.mark.unit
class TestToolSchema:
    def test_tool_names(self):
        assert bst.get_binance_spot_klines.name == "get_binance_spot_klines"
        assert bst.get_binance_spot_ticker24.name == "get_binance_spot_ticker24"
        assert bst.get_binance_spot_perp_basis.name == "get_binance_spot_perp_basis"

    def test_klines_schema(self):
        args = bst.get_binance_spot_klines.args
        assert set(args) == {"symbol", "start_date", "end_date", "interval"}
        assert args["symbol"]["type"] == "string"
        assert args["symbol"]["description"] == "Binance SPOT symbol, e.g. BTCUSDT"
        assert args["interval"]["default"] == "1d"
        # The LLM-facing required set: everything without a default.
        assert not {"symbol", "start_date", "end_date"} & {
            k for k, v in args.items() if "default" in v
        }

    def test_ticker24_schema(self):
        args = bst.get_binance_spot_ticker24.args
        assert set(args) == {"symbol"}
        assert args["symbol"]["description"] == "Binance SPOT symbol, e.g. BTCUSDT"

    def test_basis_schema(self):
        args = bst.get_binance_spot_perp_basis.args
        assert set(args) == {"symbol", "look_back_days"}
        assert args["look_back_days"]["type"] == "integer"
        assert args["look_back_days"]["default"] == 7


@pytest.mark.unit
class TestPassthrough:
    def test_klines_defaults_thread_interval(self, routed):
        out = bst.get_binance_spot_klines.invoke({
            "symbol": "BTCUSDT",
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
        })
        assert routed == [{
            "method": "get_binance_spot_klines",
            "args": ("BTCUSDT", "2026-01-01", "2026-01-10", "1d"),
            "kwargs": {},
            "qualifier": "",
        }]
        assert out == "VENDOR_OK"  # success rendered verbatim, unwrapped

    def test_klines_explicit_interval_positional(self, routed):
        bst.get_binance_spot_klines.invoke({
            "symbol": "ETHUSDT",
            "start_date": "2026-02-01",
            "end_date": "2026-02-02",
            "interval": "1h",
        })
        assert routed[0]["args"] == ("ETHUSDT", "2026-02-01", "2026-02-02", "1h")

    def test_ticker24_single_arg(self, routed):
        out = bst.get_binance_spot_ticker24.invoke({"symbol": "BTCUSDT"})
        assert routed == [{
            "method": "get_binance_spot_ticker24",
            "args": ("BTCUSDT",),
            "kwargs": {},
            "qualifier": "",
        }]
        assert out == "VENDOR_OK"

    def test_basis_default_lookback(self, routed):
        bst.get_binance_spot_perp_basis.invoke({"symbol": "BTCUSDT"})
        assert routed == [{
            "method": "get_binance_spot_perp_basis",
            "args": ("BTCUSDT", 7),
            "kwargs": {},
            "qualifier": "",
        }]

    def test_basis_explicit_lookback(self, routed):
        bst.get_binance_spot_perp_basis.invoke({
            "symbol": "ETHUSDT", "look_back_days": 30,
        })
        assert routed[0]["args"] == ("ETHUSDT", 30)


@pytest.mark.unit
class TestSentinelDegradation:
    """429 / network errors must degrade to the sentinel through the REAL
    router (the perp-fallback promise "reused verbatim"), never raise out of
    the tool: ``binance_spot`` is an OPTIONAL category, so flavour data can't
    abort the run.
    """

    @pytest.mark.parametrize("tool_attr", _SPOT_TOOLS)
    def test_rate_limit_429_degrades_to_sentinel(self, tool_attr, monkeypatch):
        tool = getattr(bst, f"get_binance_spot_{tool_attr}")
        method = tool.name
        monkeypatch.setattr(
            interface, "VENDOR_METHODS",
            {**interface.VENDOR_METHODS, method: {"binance": _throttled}},
        )
        out = tool.invoke({
            "symbol": "BTCUSDT",
            **({"start_date": "2026-01-01", "end_date": "2026-01-10"}
               if tool_attr == "klines" else {}),
        })
        assert out.startswith("DATA_UNAVAILABLE")
        assert "binance_spot" in out
        assert "429" in out

    @pytest.mark.parametrize("tool_attr", _SPOT_TOOLS)
    def test_network_error_degrades_to_sentinel(self, tool_attr, monkeypatch):
        tool = getattr(bst, f"get_binance_spot_{tool_attr}")
        method = tool.name
        monkeypatch.setattr(
            interface, "VENDOR_METHODS",
            {**interface.VENDOR_METHODS, method: {"binance": _network_down}},
        )
        out = tool.invoke({
            "symbol": "BTCUSDT",
            **({"start_date": "2026-01-01", "end_date": "2026-01-10"}
               if tool_attr == "klines" else {}),
        })
        assert out.startswith("DATA_UNAVAILABLE")
        assert "binance_spot" in out
        assert "network unreachable" in out

    def test_no_data_degrades_to_no_data_sentinel(self, monkeypatch):
        # e.g. a TRADIFI perp with no spot listing on the basis tool: the
        # vendor raises NoMarketDataError and the router must return the
        # instructive NO_DATA_AVAILABLE sentinel naming the symbol.
        def _no_spot(*_a, **_k):
            raise NoMarketDataError("XYZUSDT", "XYZUSDT", "no spot listing")

        monkeypatch.setattr(
            interface, "VENDOR_METHODS",
            {**interface.VENDOR_METHODS,
             "get_binance_spot_perp_basis": {"binance": _no_spot}},
        )
        out = bst.get_binance_spot_perp_basis.invoke({"symbol": "XYZUSDT"})
        assert out.startswith("NO_DATA_AVAILABLE")
        assert "XYZUSDT" in out
        assert "no spot listing" in out
        assert "Do not estimate" in out


@pytest.mark.unit
class TestCategoryContract:
    """The wrappers rely on the router-side category wiring for their
    sentinel behaviour; pin it so a category rename/move can't silently turn
    the spot tools into CORE (raise-on-failure) tools.
    """

    @pytest.mark.parametrize("method", [
        "get_binance_spot_klines",
        "get_binance_spot_ticker24",
        "get_binance_spot_perp_basis",
    ])
    def test_methods_live_in_optional_binance_spot_category(self, method):
        assert interface.get_category_for_method(method) == "binance_spot"
        assert "binance_spot" in interface.OPTIONAL_CATEGORIES

    @pytest.mark.parametrize("method", [
        "get_binance_spot_klines",
        "get_binance_spot_ticker24",
        "get_binance_spot_perp_basis",
    ])
    def test_default_chain_is_the_single_binance_vendor(self, method):
        assert list(interface.VENDOR_METHODS[method]) == ["binance"]
        assert interface.get_vendor("binance_spot", method) == "default"

    def test_category_tool_list_matches_wrappers(self):
        # The advertised tool list must not name a wrapper that does not
        # exist in the module (a dead name the LLM could call) — and the
        # three wrappers here must all be advertised.
        advertised = set(interface.TOOLS_CATEGORIES["binance_spot"]["tools"])
        wrappers = {
            "get_binance_spot_klines",
            "get_binance_spot_ticker24",
            "get_binance_spot_perp_basis",
        }
        assert wrappers <= advertised
        # get_binance_spot_indicators is advertised too but lives in
        # binance_indicator_tools (pinned in test_binance_indicators.py).
        assert advertised - wrappers == {"get_binance_spot_indicators"}
