"""crypto_native optional category — CoinGecko + DefiLlama fundamentals.

Hermetic (no network): vendor HTTP is monkeypatched at each module's
``_http_get_json``; the LLM is the RecordingLLM Runnable pattern from
test_crypto_perp_mode; conftest isolates config, the vendor cache, and holds
the deterministic bundles / prediction ledger OFF.

Pins the a_share_native-style contract on the crypto side:

* default OFF = the fundamentals analyst still SKIPS pure-crypto instruments
  (skip note, zero LLM/vendor calls) and every other run's tool list is
  byte-identical;
* flag ON + pure crypto (pure_crypto_perp / crypto_spot / legacy crypto) =
  the analyst runs with a REPLACEMENT crypto tool set (market-analyst perp
  precedent) and a crypto-framed prompt;
* flag ON + equity / tokenized-stock perp = unchanged stock path;
* vendor PIT windows, symbol resolution, honest degradation and typed error
  mapping at the router boundary.
"""

import unittest
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from yialpha.agents.analysts.fundamentals_analyst import (
    create_fundamentals_analyst,
)
from yialpha.agents.utils import crypto_native_tools as cnt
from yialpha.dataflows import coingecko_vendor as cg, config as cfgmod, defillama_vendor as dl
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError
from yialpha.dataflows.interface import OPTIONAL_CATEGORIES, route_to_vendor
from yialpha.graph.trading_graph import YiAlphaGraph

_HIST_DATE = "2026-06-30"  # historical relative to any today >= 2026-07-01


class _BoundLLM(Runnable):
    """Records the formatted message list the chain invokes with."""

    def __init__(self, sink):
        self._sink = sink

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        # prompt|llm chains hand the bound LLM a ChatPromptValue; plain
        # message lists arrive on the fallback rounds — normalize both.
        self._sink.append(getattr(inp, "messages", inp))
        return AIMessage(content="MOCK REPORT")


class RecordingLLM(Runnable):
    """Fake LLM that records bind_tools lists + the final invoke input."""

    def __init__(self):
        super().__init__()
        self.bound_tools = None
        self.invocations = []

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        return AIMessage(content="", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound_tools = list(tools)
        return _BoundLLM(self.invocations)


class FundamentalsCryptoNativeWiringTests(unittest.TestCase):
    """Tool-list + prompt gates for the crypto_native branch."""

    @staticmethod
    def _state(ticker, asset_type, trade_date):
        return {
            "trade_date": trade_date,
            "company_of_interest": ticker,
            "asset_type": asset_type,
            "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        }

    @staticmethod
    def _run(ticker, asset_type, *, crypto_native, trade_date=_HIST_DATE):
        cfgmod.set_config({"crypto_native": crypto_native})
        llm = RecordingLLM()
        node = create_fundamentals_analyst(llm)
        out = node(FundamentalsCryptoNativeWiringTests._state(
            ticker, asset_type, trade_date,
        ))
        return llm, out

    def test_default_off_pure_crypto_still_skips(self):
        llm, out = self._run("BTCUSDT", "crypto_perp", crypto_native=False)
        self.assertIsNone(llm.bound_tools, "flag off must make zero LLM calls")
        self.assertIn("skipped — pure-crypto instrument", out["fundamentals_report"])

    def test_flag_on_pure_perp_binds_replacement_crypto_tools(self):
        llm, _ = self._run("BTCUSDT", "crypto_perp", crypto_native=True)
        self.assertEqual(
            sorted(t.name for t in llm.bound_tools),
            ["get_crypto_network_fundamentals", "get_crypto_token_fundamentals"],
        )

    def test_flag_on_crypto_spot_binds_crypto_tools(self):
        llm, _ = self._run("ETHUSDT", "crypto_spot", crypto_native=True)
        self.assertEqual(
            sorted(t.name for t in llm.bound_tools),
            ["get_crypto_network_fundamentals", "get_crypto_token_fundamentals"],
        )

    def test_flag_on_legacy_crypto_binds_crypto_tools(self):
        llm, _ = self._run("BTCUSDT", "crypto", crypto_native=True)
        self.assertEqual(
            sorted(t.name for t in llm.bound_tools),
            ["get_crypto_network_fundamentals", "get_crypto_token_fundamentals"],
        )

    def test_flag_on_live_date_composes_web_search(self):
        llm, _ = self._run(
            "BTCUSDT", "crypto_perp", crypto_native=True,
            trade_date=date.today().isoformat(),
        )
        names = sorted(t.name for t in llm.bound_tools)
        self.assertIn("get_crypto_token_fundamentals", names)
        # the fundamentals-scoped web-search instance shares the tool NAME
        # "web_search" (scoped-budget allocation, see test_market_toolnode)
        self.assertIn("web_search", names)

    def test_flag_on_stock_run_is_byte_equivalent(self):
        _, off = self._run("NVDA", "stock", crypto_native=False)
        cfgmod.set_config({"crypto_native": False})
        llm_off = RecordingLLM()
        create_fundamentals_analyst(llm_off)(self._state("NVDA", "stock", _HIST_DATE))
        cfgmod.set_config({"crypto_native": True})
        llm_on = RecordingLLM()
        create_fundamentals_analyst(llm_on)(self._state("NVDA", "stock", _HIST_DATE))
        self.assertEqual(
            sorted(t.name for t in llm_off.bound_tools),
            sorted(t.name for t in llm_on.bound_tools),
        )
        self.assertEqual(
            ["get_balance_sheet", "get_cashflow", "get_fundamentals",
             "get_income_statement"],
            sorted(t.name for t in llm_on.bound_tools),
        )
        self.assertIn("company", str(llm_on.invocations[0][0].content))

    def test_flag_on_stock_perp_run_is_byte_equivalent(self):
        # MU is in the static equity-perp seed, so stock_perp_underlying
        # resolves without any network warm.
        for flag in (False, True):
            cfgmod.set_config({"crypto_native": flag})
            llm = RecordingLLM()
            create_fundamentals_analyst(llm)(self._state("MUUSDT", "crypto_perp", _HIST_DATE))
            self.assertEqual(
                ["get_balance_sheet", "get_cashflow", "get_fundamentals",
                 "get_income_statement"],
                sorted(t.name for t in llm.bound_tools),
                f"stock-perp path must be unchanged (crypto_native={flag})",
            )

    def test_crypto_prompt_reframes_persona(self):
        llm, _ = self._run("BTCUSDT", "crypto_perp", crypto_native=True)
        system = str(llm.invocations[0][0].content)
        self.assertIn("pure-crypto asset", system)
        self.assertIn("get_crypto_token_fundamentals", system)
        self.assertIn("PURE-CRYPTO asset", system)  # the nudge
        self.assertNotIn("about a company", system)

    def test_stock_prompt_unchanged_when_flag_on(self):
        cfgmod.set_config({"crypto_native": True})
        llm = RecordingLLM()
        create_fundamentals_analyst(llm)(self._state("NVDA", "stock", _HIST_DATE))
        system = str(llm.invocations[0][0].content)
        self.assertIn("about a company", system)
        self.assertNotIn("PURE-CRYPTO asset", system)

    def test_flag_on_legacy_crypto_etf_base_gets_no_fund_nudge(self):
        # GDXUSDT via the legacy auto-detected crypto mode classifies as
        # crypto_spot, and its USDT-stripped base collides with the ETF set —
        # the ETF fund-framing nudge must be suppressed on the crypto branch
        # (contradictory, unfetchable instructions otherwise).
        llm, _ = self._run("GDXUSDT", "crypto", crypto_native=True)
        system = str(llm.invocations[0][0].content)
        self.assertIn("PURE-CRYPTO asset", system)
        self.assertNotIn("EXCHANGE-TRADED FUND", system)


@pytest.mark.unit
def test_fundamentals_toolnode_registers_crypto_native_tools():
    """Wiring-gap guard (test_market_toolnode contract): the two crypto tools
    the analyst binds under YIALPHA_CRYPTO_NATIVE must be executable in the
    fundamentals ToolNode."""
    fake_self = SimpleNamespace(quick_thinking_llm=MagicMock())
    nodes = YiAlphaGraph._create_tool_nodes(fake_self)
    fundamentals_tools = set(nodes["fundamentals"].tools_by_name)
    missing = {
        "get_crypto_token_fundamentals",
        "get_crypto_network_fundamentals",
    } - fundamentals_tools
    assert not missing, f"crypto_native tools bound but not executable: {missing}"


# --------------------------------------------------------------------------- #
# Vendor units — CoinGecko (mocked _http_get_json, tmp cache via conftest)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def no_throttle(monkeypatch):
    """Zero the politeness throttles so mocked tests never sleep 4s."""
    from yialpha.dataflows.disk_cache import MinIntervalThrottle

    monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
    monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))


def _search_payload(coins):
    return {"coins": coins}


class CoinGeckoVendorTests(unittest.TestCase):
    """Resolution, PIT windows, honest degradation, typed errors."""

    def setUp(self):
        from yialpha.dataflows.disk_cache import MinIntervalThrottle

        self._orig_throttle = cg._throttler
        cg._throttler = MinIntervalThrottle(0.0)
        self._orig_http = cg._http_get_json

    def tearDown(self):
        # restore BOTH patched globals — a leaked zero-interval throttle
        # would silently disable politeness for every later test in the
        # session (order-dependent behavior).
        cg._http_get_json = self._orig_http
        cg._throttler = self._orig_throttle

    def test_resolve_picks_exact_symbol_with_lowest_rank(self):
        cg._http_get_json = lambda url, params=None: _search_payload([
            {"id": "bitcoin-pro", "symbol": "btc", "market_cap_rank": 1465},
            {"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1},
            {"id": "bitget-wrapped-btc", "symbol": "btcb", "market_cap_rank": 294},
        ])
        self.assertEqual(cg._resolve_coin_id("BTCUSDT"), ("bitcoin", "BTC"))

    def test_resolve_unranked_matches_sort_last(self):
        cg._http_get_json = lambda url, params=None: _search_payload([
            {"id": "junk-btc", "symbol": "BTC", "market_cap_rank": None},
            {"id": "real", "symbol": "BTC", "market_cap_rank": 42},
        ])
        self.assertEqual(cg._resolve_coin_id("BTCUSDT"), ("real", "BTC"))

    def test_resolve_1000_prefix_falls_back_to_token(self):
        def fake(url, params=None):
            if params and params.get("query") == "1000PEPE":
                return _search_payload([{"id": "x", "symbol": "pepe2", "market_cap_rank": 900}])
            return _search_payload([
                {"id": "pepe", "symbol": "PEPE", "market_cap_rank": 25},
            ])
        cg._http_get_json = fake
        self.assertEqual(cg._resolve_coin_id("1000PEPEUSDT"), ("pepe", "PEPE"))

    def test_multiplier_prefix_candidates_cover_1000cat_and_1000000mog(self):
        # 7-char 1000CAT and the 1000000x family must both probe the real
        # token (a single len>7 "1000" strip mangles 1000000MOG -> 000MOG).
        self.assertEqual(cg._base_candidates("1000CATUSDT"), ["1000CAT", "CAT"])
        self.assertEqual(cg._base_candidates("1000000MOGUSDT"), ["1000000MOG", "MOG"])
        self.assertEqual(cg._base_candidates("1000XUSDT"), ["1000X"])

    def test_legacy_dashed_symbol_resolves_base(self):
        # legacy auto-detected crypto (Yahoo-style BTC-USD) must reduce to
        # BTC, not the unresolvable BTCUSD.
        cg._http_get_json = lambda url, params=None: _search_payload(
            [{"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1}]
        )
        self.assertEqual(cg._resolve_coin_id("BTC-USD"), ("bitcoin", "BTC"))

    def test_malformed_curr_date_degrades_honestly(self):
        # is_historical_date routes "2026/08/01" into the historical branch
        # fail-closed; the vendor must degrade in-band, not crash.
        cg._http_get_json = lambda url, params=None: _search_payload(
            [{"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1}]
        )
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026/08/01")
        self.assertIn("not a parseable yyyy-mm-dd", out)

    def test_unresolvable_symbol_raises_no_market_data(self):
        cg._http_get_json = lambda url, params=None: _search_payload([])
        with self.assertRaises(NoMarketDataError):
            cg.get_crypto_token_fundamentals("ZZZZUSDT", "2026-06-30")

    @staticmethod
    def _ms(d: date, hours: int = 0) -> int:
        return int(datetime(d.year, d.month, d.day, hours, tzinfo=UTC)
                   .timestamp() * 1000)

    def test_historical_pit_filters_and_daily_resamples(self):
        base = date(2026, 6, 26)
        pts = []
        for i in range(5):  # five days, hourly points -> daily dedupe (last wins)
            d = base + timedelta(days=i)
            for h in (1, 13):
                ts = self._ms(d, h)
                pts.append([ts, 100.0 + i * 10 + h])
        caps = [[p[0], p[1] * 1_000_000] for p in pts]
        vols = [[p[0], 5.0e8] for p in pts]

        def fake(url, params=None):
            if "/search" in url:
                return _search_payload(
                    [{"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1}]
                )
            return {"prices": pts, "market_caps": caps, "total_volumes": vols}

        cg._http_get_json = fake
        out = cg.get_crypto_token_fundamentals(
            "BTCUSDT", curr_date="2026-06-28", look_back_days=30,
        )
        self.assertIn("point-in-time series as of 2026-06-28", out)
        # daily resample keeps the LAST point per day (h=13 offset):
        # day i -> 100 + i*10 + 13
        self.assertIn("| 2026-06-26 | 113.00 |", out)
        self.assertIn("| 2026-06-27 | 123.00 |", out)
        self.assertIn("| 2026-06-28 | 133.00 |", out)
        self.assertNotIn("2026-06-29", out)  # PIT: rows after curr_date dropped
        self.assertIn("deliberately EXCLUDED", out)  # leak note

    def test_historical_too_old_degrades_honestly(self):
        def fake(url, params=None):
            if "/search" in url:
                return _search_payload(
                    [{"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1}]
                )
            return {
                "prices": [[self._ms(date(2026, 9, 1)), 1.0]],
                "market_caps": [], "total_volumes": [],
            }

        cg._http_get_json = fake
        out = cg.get_crypto_token_fundamentals(
            "BTCUSDT", curr_date="2024-06-30", look_back_days=30,
        )
        self.assertIn("historical market data is NOT available", out)
        self.assertIn("365 days", out)

    def test_price_change_na_fallback_and_sub_cent_prices(self):
        coin_payload = {
            "id": "pepe", "market_cap_rank": 25, "categories": [],
            "watchlist_portfolio_users": 100,
            "market_data": {
                "current_price": {"usd": 0.0000123},
                "market_cap": {"usd": 4.9e9},
                "total_volume": {"usd": 3.0e8},
                "circulating_supply": 4.0e14,
                "total_supply": 4.2e14, "max_supply": None,
                "ath": {"usd": 0.00008}, "ath_date": {"usd": "2024-05-27T00:00:00Z"},
                "atl": {"usd": 0.0000001}, "atl_date": {"usd": "2023-04-18T00:00:00Z"},
                # NO price_change_percentage_* fields (newly-listed shape)
            },
        }

        def fake(url, params=None):
            if "/search" in url:
                return _search_payload(
                    [{"id": "pepe", "symbol": "pepe", "market_cap_rank": 25}]
                )
            return coin_payload

        cg._http_get_json = fake
        out = cg.get_crypto_token_fundamentals("1000PEPEUSDT", curr_date="")
        self.assertIn("- Price change: n/a", out)  # or-fallback actually fires
        self.assertIn("0.0000123", out)  # sub-cent price not collapsed to 0.00
        self.assertIn("$0.00008", out)  # sub-cent ATH via _fmt_usd

    def test_live_snapshot_renders_usd_fields(self):
        coin_payload = {
            "id": "bitcoin",
            "market_cap_rank": 1,
            "categories": ["Layer 1 (L1)"],
            "genesis_date": "2009-01-03",
            "watchlist_portfolio_users": 2_460_000,
            "market_data": {
                "current_price": {"usd": 83_228.0},
                "market_cap": {"usd": 1.67e12},
                "total_volume": {"usd": 3.836e10},
                "circulating_supply": 20_090_871.0,
                "total_supply": 20_090_896.0,
                "max_supply": 21_000_000.0,
                "ath": {"usd": 126_080.0},
                "ath_date": {"usd": "2025-10-06T10:57:42.000Z"},
                "atl": {"usd": 67.81},
                "atl_date": {"usd": "2013-07-05T00:00:00.000Z"},
                "price_change_percentage_24h": -0.1,
                "price_change_percentage_7d": -2.6,
            },
        }

        def fake(url, params=None):
            if "/search" in url:
                return _search_payload(
                    [{"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1}]
                )
            return coin_payload

        cg._http_get_json = fake
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="")
        self.assertIn("CURRENT SNAPSHOT", out)
        self.assertIn("#1", out)
        self.assertIn("$1.67T", out)
        self.assertIn("ATH: $126.08K on 2025-10-06", out)
        self.assertIn("not served on the free public tier", out)

    def test_429_maps_to_rate_limit_error(self):
        fake = MagicMock()
        fake.status_code = 429
        fake.text = "rate limited"
        from unittest.mock import patch

        import requests as _requests  # noqa: F401 — ensure module import path

        with patch("requests.get", return_value=fake), \
                self.assertRaises(VendorRateLimitError):
            cg._http_get_json("https://api.coingecko.com/api/v3/x")


# --------------------------------------------------------------------------- #
# Vendor units — DefiLlama
# --------------------------------------------------------------------------- #

def _protocols_catalog():
    return [
        {"slug": "uniswap-v3", "name": "Uniswap V3", "symbol": "UNI",
         "tvl": 1.68e9, "chains": ["Ethereum", "Arbitrum"], "gecko_id": "uniswap",
         "category": "Dexs"},
        {"slug": "uniswap-v2", "name": "Uniswap V2", "symbol": "UNI",
         "tvl": 1.05e9, "chains": ["Ethereum"], "gecko_id": "uniswap",
         "category": "Dexs"},
        {"slug": "shibex", "name": "Shibex", "symbol": "SHIB",
         "tvl": 5_994.0, "chains": ["Ethereum"], "gecko_id": None,
         "category": "Other"},
    ]


class DefiLlamaVendorTests(unittest.TestCase):
    def setUp(self):
        from yialpha.dataflows.disk_cache import MinIntervalThrottle

        self._orig_throttle = dl._throttler
        dl._throttler = MinIntervalThrottle(0.0)
        self._orig_http = dl._http_get_json

    def tearDown(self):
        dl._http_get_json = self._orig_http
        dl._throttler = self._orig_throttle

    def test_protocol_resolution_max_tvl_and_floor(self):
        dl._http_get_json = lambda url: _protocols_catalog()
        self.assertEqual(dl._resolve_protocol("UNI")["slug"], "uniswap-v3")
        self.assertIsNone(dl._resolve_protocol("SHIB"))  # below $1M floor
        self.assertIsNone(dl._resolve_protocol("DOGE"))

    @staticmethod
    def _ts(d: date) -> int:
        return int(datetime(d.year, d.month, d.day, 12, tzinfo=UTC)
                   .timestamp())

    def test_network_fundamentals_protocol_chain_and_pit(self):
        base = date(2026, 8, 10)

        def fake(url):
            if url.endswith("/protocols"):
                return _protocols_catalog()
            if "/protocol/uniswap-v3" in url:
                return {"tvl": [
                    {"date": self._ts(base + timedelta(days=i)),
                     "totalLiquidityUSD": 1.0e9 + i * 1.0e7}
                    for i in range(10)
                ]}
            if "/historicalChainTvl/Ethereum" in url:
                return [
                    {"date": self._ts(base + timedelta(days=i)), "tvl": 5.0e10}
                    for i in range(10)
                ]
            if url.endswith("/v2/chains"):
                return [{"name": "Ethereum", "tvl": 5.2e10},
                        {"name": "Solana", "tvl": 6.4e9}]
            raise AssertionError(url)

        dl._http_get_json = fake
        out = dl.get_crypto_network_fundamentals(
            "UNIUSDT", curr_date="2026-08-15", look_back_days=30,
        )
        self.assertIn("slug 'uniswap-v3'", out)  # version disclosed
        self.assertIn("| 2026-08-15 | $1.05B |", out)  # protocol PIT row
        self.assertNotIn("2026-08-16", out)  # rows after curr_date dropped
        self.assertIn("Chain ranking: omitted on historical runs", out)

        live = dl.get_crypto_network_fundamentals("UNIUSDT")
        self.assertIn("ranks #1 of 2 DefiLlama-tracked chains", live)

    def test_no_coverage_raises_no_market_data(self):
        dl._http_get_json = lambda url: _protocols_catalog()
        with self.assertRaises(NoMarketDataError):
            dl.get_crypto_network_fundamentals("DOGEUSDT", "2026-06-30")

    def test_historical_title_omits_current_tvl_and_discloses_caveat(self):
        base = date(2026, 8, 10)

        def fake(url):
            if url.endswith("/protocols"):
                return _protocols_catalog()
            if "/protocol/uniswap-v3" in url:
                return {"tvl": [
                    {"date": self._ts(base + timedelta(days=i)),
                     "totalLiquidityUSD": 1.0e9}
                    for i in range(10)
                ]}
            if "/v2/chains" in url:
                return []
            if "/historicalChainTvl/" in url:
                return []
            raise AssertionError(url)

        dl._http_get_json = fake
        out = dl.get_crypto_network_fundamentals(
            "UNIUSDT", curr_date="2026-08-15", look_back_days=30,
        )
        self.assertNotIn("current TVL", out)  # PIT: no today-snapshot figure
        self.assertIn("picked by TODAY's DefiLlama catalog TVL", out)

    def test_chain_leg_failure_preserves_protocol_section(self):
        base = date(2026, 8, 10)

        def fake(url):
            if url.endswith("/protocols"):
                return _protocols_catalog()
            if "/protocol/uniswap-v3" in url:
                return {"tvl": [
                    {"date": self._ts(base + timedelta(days=i)),
                     "totalLiquidityUSD": 1.0e9 + i * 1.0e7}
                    for i in range(10)
                ]}
            # every other call (chains catalog, chain history) fails
            raise NoMarketDataError("defillama", detail="proxy blip")

        dl._http_get_json = fake
        out = dl.get_crypto_network_fundamentals(
            "UNIUSDT", curr_date="2026-08-15", look_back_days=30,
        )
        self.assertIn("Protocol TVL — Uniswap V3", out)
        self.assertIn("| 2026-08-15 | $1.05B |", out)
        self.assertIn("Unretrievable sections", out)

    def test_space_named_chain_cache_filename_is_sanitized(self):
        # "zkSync Era" contains a space — the disk-cache filename allowlist
        # would otherwise raise ValueError before any fetch.
        def fake(url):
            if url.endswith("/protocols"):
                return []
            if "/v2/historicalChainTvl/" in url:
                return [{"date": self._ts(date(2026, 9, 1)), "tvl": 4.4e9}]
            raise AssertionError(url)

        dl._http_get_json = fake
        out = dl.get_crypto_network_fundamentals("ZKUSDT", curr_date="2026-09-02")
        self.assertIn("Chain TVL — zkSync Era", out)
        self.assertNotIn("data not available for this item", out[:200])

    def test_malformed_curr_date_degrades_honestly(self):
        dl._http_get_json = lambda url: _protocols_catalog()
        out = dl.get_crypto_network_fundamentals("UNIUSDT", curr_date="2026/08/01")
        self.assertIn("not a parseable yyyy-mm-dd", out)

    def test_chain_only_path_for_unmapped_protocol_token(self):
        def fake(url):
            if url.endswith("/protocols"):
                return _protocols_catalog()
            if "/historicalChainTvl/" in url:
                return [{"date": self._ts(date(2026, 9, 1)), "tvl": 4.4e9}]
            raise AssertionError(url)

        dl._http_get_json = fake
        out = dl.get_crypto_network_fundamentals("BTCUSDT", curr_date="2026-09-02")
        self.assertIn("Protocol TVL: no credible DefiLlama protocol", out)
        self.assertIn("Chain TVL — Bitcoin", out)


# --------------------------------------------------------------------------- #
# Router boundary + analyst selection + tool-layer PIT clamp
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_crypto_native_is_optional_category():
    assert "crypto_native" in OPTIONAL_CATEGORIES


class RouterBoundaryTests(unittest.TestCase):
    def setUp(self):
        from yialpha.dataflows.disk_cache import MinIntervalThrottle

        self._orig_throttle = cg._throttler
        cg._throttler = MinIntervalThrottle(0.0)
        self._orig_http = cg._http_get_json

    def tearDown(self):
        cg._http_get_json = self._orig_http
        cg._throttler = self._orig_throttle

    def test_unresolved_symbol_degrades_to_no_data_sentinel(self):
        cg._http_get_json = lambda url, params=None: _search_payload([])
        out = route_to_vendor(
            "get_crypto_token_fundamentals", "ZZZZUSDT", _HIST_DATE,
        )
        self.assertTrue(out.startswith("NO_DATA_AVAILABLE:"), out[:120])

    def test_transport_failure_degrades_to_data_unavailable_sentinel(self):
        def boom(url, params=None):
            raise RuntimeError("proxy exploded")

        cg._http_get_json = boom
        out = route_to_vendor(
            "get_crypto_token_fundamentals", "BTCUSDT", _HIST_DATE,
        )
        self.assertTrue(out.startswith("DATA_UNAVAILABLE:"), out[:120])
        self.assertIn("crypto_native", out)


@pytest.mark.unit
def test_filter_keeps_fundamentals_for_pure_crypto_only_when_flag_on():
    from yialpha.cli.models import AnalystType, AssetType
    from yialpha.cli.utils import filter_analysts_for_asset_type

    base = [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS,
            AnalystType.FUNDAMENTALS]

    cfgmod.set_config({"crypto_native": False})
    kept_off = filter_analysts_for_asset_type(base, AssetType.CRYPTO_PERP, "BTCUSDT")
    assert AnalystType.FUNDAMENTALS not in kept_off

    cfgmod.set_config({"crypto_native": True})
    kept_on = filter_analysts_for_asset_type(base, AssetType.CRYPTO_PERP, "BTCUSDT")
    assert AnalystType.FUNDAMENTALS in kept_on
    # stock perps keep fundamentals regardless of the flag
    kept_mu = filter_analysts_for_asset_type(base, AssetType.CRYPTO_PERP, "MUUSDT")
    assert AnalystType.FUNDAMENTALS in kept_mu


@pytest.mark.unit
def test_batch_union_keeps_fundamentals_when_flag_on(monkeypatch):
    monkeypatch.setattr(
        "yialpha.dataflows.binance.warm_equity_perp_bases",
        lambda: frozenset({"MU"}),  # BTC stays pure-crypto; MU seed-like
    )
    from yialpha.batch.runner import batch_selected_analysts

    cfgmod.set_config({"crypto_native": False})
    assert "fundamentals" not in batch_selected_analysts(
        "crypto_perp", ["BTCUSDT"],
    )
    cfgmod.set_config({"crypto_native": True})
    assert "fundamentals" in batch_selected_analysts(
        "crypto_perp", ["BTCUSDT"],
    )


@pytest.mark.unit
def test_tool_layer_clamps_echoed_curr_date(monkeypatch):
    """An empty LLM-echoed curr_date resolves to the run's pinned analysis
    date (round-8 clamp contract, same as the A-share thin tools)."""
    captured = {}

    def fake_route(method, *args, **kwargs):
        captured["method"] = method
        captured["args"] = args
        return "ok"

    monkeypatch.setattr(cnt, "route_to_vendor", fake_route)
    monkeypatch.setattr(cnt, "get_analysis_date", lambda: "2026-06-30")
    result = cnt.get_crypto_token_fundamentals.invoke(
        {"ticker": "BTCUSDT", "curr_date": "", "look_back_days": 90},
    )
    assert result == "ok"
    assert captured["method"] == "get_crypto_token_fundamentals"
    assert captured["args"][0] == "BTCUSDT"
    assert captured["args"][1] == "2026-06-30"
