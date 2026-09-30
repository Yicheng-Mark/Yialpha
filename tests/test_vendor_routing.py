"""Vendor router must respect the configured chain and never silently hide a
broken primary.

Regressions for #988 (explicit single-vendor config still fell back to others),
#289 (fallback ran for unchosen vendors), and #989 (serious primary failures
were swallowed without a trace).
"""
import unittest
from unittest import mock

import pytest

from yialpha.dataflows import interface
from yialpha.dataflows.config import reset_config, set_config
from yialpha.dataflows.symbol_utils import NoMarketDataError


def _reset_config():
    # Hard reset: set_config() merges, so empty DEFAULT dicts (e.g. tool_vendors)
    # don't clear keys leaked by other tests. Reset the context outright.
    reset_config()


def _no_data(symbol, *a, **k):
    raise NoMarketDataError(symbol, symbol, "no rows")


def _returns(value):
    def impl(symbol, *a, **k):
        return value
    return impl


def _raises(exc):
    def impl(symbol, *a, **k):
        raise exc
    return impl


@pytest.mark.unit
class VendorRoutingTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _route(self, vendors_for_get_stock_data):
        return mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": vendors_for_get_stock_data},
            clear=False,
        )

    def test_explicit_single_vendor_does_not_fall_back(self):
        # #988: with yfinance pinned, a healthy alpha_vantage must NOT be used.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        av = mock.Mock(side_effect=_returns("AV_DATA"))
        with self._route({"yfinance": _no_data, "alpha_vantage": av}):
            result = interface.route_to_vendor("get_stock_data", "FAKE", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        av.assert_not_called()  # the unchosen vendor was never tried

    def test_explicit_multi_vendor_falls_back_within_chain(self):
        # Listing both vendors opts in to ordered fallback.
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def test_primary_error_is_logged_not_masked(self):
        # #989: primary errors + fallback no-data -> NO_DATA, but the failure
        # must be visible in logs (broken primary not hidden).
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _raises(ValueError("boom")), "alpha_vantage": _no_data}), \
                self.assertLogs("yialpha.dataflows.interface", level="WARNING") as cm:
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        joined = "\n".join(cm.output)
        self.assertIn("boom", joined)            # the real error surfaced in logs
        self.assertIn("yfinance", joined)

    def test_unknown_configured_vendor_raises(self):
        set_config({"data_vendors": {"core_stock_apis": "bogus_vendor"}})
        with self.assertRaises(ValueError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("bogus_vendor", str(ctx.exception))

    def test_default_sentinel_uses_all_vendors(self):
        # No explicit choice ("default") keeps the resilient full-chain behavior.
        set_config({"data_vendors": {"core_stock_apis": "default"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def test_default_fallback_chain_puts_yfinance_first(self):
        # Round-11: the "default" sentinel used VENDOR_METHODS dict insertion
        # order, which put alpha_vantage FIRST for prices/indicators/news —
        # violating the T0-3 chain contract (yfinance, keyless and unthrottled,
        # leads; AV's rate-limited free tier trails). The order now comes from
        # DEFAULT_CONFIG's data_vendors value for the category.
        for method in ("get_stock_data", "get_indicators", "get_news",
                       "get_global_news", "get_insider_transactions"):
            order = []

            def impl(name, order=order):
                def f(symbol, *a, **k):
                    order.append(name)
                    raise NoMarketDataError(symbol, symbol, "no rows")
                return f

            category = interface.get_category_for_method(method)
            set_config({"data_vendors": {category: "default"}})
            with mock.patch.dict(interface.VENDOR_METHODS, {
                method: {
                    "alpha_vantage": impl("alpha_vantage"),
                    "yfinance": impl("yfinance"),
                },
            }, clear=False):
                interface.route_to_vendor(method, "AAPL")
            self.assertEqual(order, ["yfinance", "alpha_vantage"], method)

    def test_default_fallback_fundamentals_chain_sec_first(self):
        order = []

        def impl(name, order=order):
            def f(symbol, *a, **k):
                order.append(name)
                raise NoMarketDataError(symbol, symbol, "no rows")
            return f

        set_config({"data_vendors": {"fundamental_data": "default"}})
        with mock.patch.dict(interface.VENDOR_METHODS, {
            "get_balance_sheet": {
                "alpha_vantage": impl("alpha_vantage"),
                "yfinance": impl("yfinance"),
                "sec_edgar": impl("sec_edgar"),
            },
        }, clear=False):
            interface.route_to_vendor("get_balance_sheet", "AAPL")
        # Default chain for fundamental_data: sec_edgar,yfinance,alpha_vantage.
        self.assertEqual(order, ["sec_edgar", "yfinance", "alpha_vantage"])

    def test_default_chain_order_helper_single_source(self):
        # The helper reads DEFAULT_CONFIG (not the live config — the fallback
        # only fires when the live value IS "default"); vendors not named in
        # the documented chain keep their dict order after it (the
        # fundamentals_overview aggregate), and categories without a
        # documented chain are left untouched.
        self.assertEqual(
            interface._default_chain_order(
                "fundamental_data",
                ["alpha_vantage", "yfinance", "sec_edgar",
                 "fundamentals_overview"]),
            ["sec_edgar", "yfinance", "alpha_vantage",
             "fundamentals_overview"],
        )
        self.assertEqual(
            interface._default_chain_order("a_share_native",
                                           ["baostock", "tushare"]),
            ["baostock", "tushare"],
        )
        self.assertEqual(
            interface._default_chain_order("binance_perp", ["binance"]),
            ["binance"],
        )

    def _route_method(self, method, vendors):
        return mock.patch.dict(interface.VENDOR_METHODS, {method: vendors}, clear=False)

    def test_optional_category_degrades_instead_of_raising(self):
        # An optional enrichment vendor (FRED macro) that raises must NOT abort
        # the run — the router returns a sentinel so the analysis proceeds.
        set_config({"data_vendors": {"macro_data": "fred"}})
        with self._route_method(
            "get_macro_indicators", {"fred": _raises(ValueError("FRED 400: bad series"))}
        ):
            result = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")
        self.assertIn("DATA_UNAVAILABLE", result)
        self.assertIn("macro_data", result)

    def test_core_category_still_raises_on_error(self):
        # A core category (single configured vendor) propagates the error so a
        # broken primary is loud, not silently degraded.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        with self._route({"yfinance": _raises(ValueError("boom"))}), \
                self.assertRaises(ValueError):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")


if __name__ == "__main__":
    unittest.main()
