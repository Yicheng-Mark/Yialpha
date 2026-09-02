"""Config isolation: get/set must not leak nested-dict references."""

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor

import pytest

import yialpha.default_config as default_config
from yialpha.dataflows.config import get_config, set_config, submit_with_context


@pytest.mark.unit
class DataflowsConfigIsolationTests(unittest.TestCase):
    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_get_config_returns_deep_copy(self):
        cfg = get_config()
        cfg["data_vendors"]["core_stock_apis"] = "alpha_vantage"
        cfg["tool_vendors"]["get_stock_data"] = "alpha_vantage"

        fresh = get_config()
        # Default is the multi-vendor fallback chain (T0-3), not bare yfinance.
        self.assertEqual(
            fresh["data_vendors"]["core_stock_apis"], "yfinance,alpha_vantage"
        )
        self.assertNotIn("get_stock_data", fresh["tool_vendors"])

    def test_set_config_does_not_alias_caller_nested_dicts(self):
        custom = copy.deepcopy(default_config.DEFAULT_CONFIG)
        custom["data_vendors"]["core_stock_apis"] = "alpha_vantage"
        custom["tool_vendors"]["get_stock_data"] = "alpha_vantage"

        set_config(custom)

        custom["data_vendors"]["core_stock_apis"] = "yfinance"
        custom["tool_vendors"]["get_stock_data"] = "yfinance"

        fresh = get_config()
        self.assertEqual(fresh["data_vendors"]["core_stock_apis"], "alpha_vantage")
        self.assertEqual(fresh["tool_vendors"]["get_stock_data"], "alpha_vantage")

    def test_partial_nested_update_preserves_existing_defaults(self):
        set_config(
            {
                "data_vendors": {
                    "core_stock_apis": "alpha_vantage",
                }
            }
        )

        fresh = get_config()
        self.assertEqual(fresh["data_vendors"]["core_stock_apis"], "alpha_vantage")
        # Untouched nested keys keep their multi-vendor fallback-chain defaults.
        self.assertEqual(
            fresh["data_vendors"]["technical_indicators"], "yfinance,alpha_vantage"
        )
        self.assertEqual(
            fresh["data_vendors"]["fundamental_data"],
            "sec_edgar,yfinance,alpha_vantage",
        )
        self.assertEqual(
            fresh["data_vendors"]["news_data"], "yfinance,alpha_vantage"
        )

    def test_nested_dict_updates_merge_one_level_deep(self):
        set_config({"tool_vendors": {"get_stock_data": "alpha_vantage"}})
        set_config({"tool_vendors": {"get_news": "alpha_vantage"}})

        fresh = get_config()
        self.assertEqual(fresh["tool_vendors"]["get_stock_data"], "alpha_vantage")
        self.assertEqual(fresh["tool_vendors"]["get_news"], "alpha_vantage")

    def test_overview_defaults_to_aggregate_vendor_statements_stay_sec_first(self):
        # The overview MERGES SEC filing facts + Yahoo valuation; the three
        # statements ride the SEC-first category chain untouched.
        self.assertEqual(
            get_config()["tool_vendors"].get("get_fundamentals"),
            "fundamentals_overview",
        )
        self.assertEqual(
            get_config()["data_vendors"]["fundamental_data"],
            "sec_edgar,yfinance,alpha_vantage",
        )
        from yialpha.dataflows.interface import VENDOR_METHODS, get_vendor

        self.assertEqual(
            get_vendor("fundamental_data", "get_fundamentals"),
            "fundamentals_overview",
        )
        self.assertEqual(
            get_vendor("fundamental_data", "get_income_statement"),
            "sec_edgar,yfinance,alpha_vantage",
        )
        # The aggregate vendor resolves to the merge implementation.
        from yialpha.dataflows.fundamentals_overview import (
            get_fundamentals as aggregate_overview,
        )

        self.assertIs(
            VENDOR_METHODS["get_fundamentals"]["fundamentals_overview"],
            aggregate_overview,
        )

    def test_fundamentals_bundle_defaults_on(self):
        self.assertIs(get_config()["fundamentals_bundle"], True)

    def test_submit_with_context_propagates_config_to_worker_thread(self):
        set_config({"context_probe": 987654})

        with ThreadPoolExecutor(max_workers=1) as executor:
            value = submit_with_context(executor, get_config).result()["context_probe"]

        self.assertEqual(value, 987654)
