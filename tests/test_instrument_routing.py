"""Asset-aware analyst routing — the ONE classification every entrance shares.

Pins ``yialpha.graph.routing`` (instrument_class / fundamentals_applicable)
and the fundamentals NODE's runtime skip: a pure-crypto instrument must
produce a fundamentals report with ZERO LLM and ZERO vendor calls on EVERY
entrance — direct ``YiAlphaGraph()`` construction (default analyst tuple
includes fundamentals), the batch union over a mixed [BTCUSDT, MUUSDT]
batch, scripts, web and backtest all funnel into this node. The CLI-level
filter stays the belt-and-suspenders layer on top.

Zero network: the equity-perp universe resolves cache-or-seed (``MU`` is in
the static seed, ``BTC`` is not).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from yialpha.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from yialpha.graph.routing import (
    CRYPTO_FAMILY,
    fundamentals_applicable,
    instrument_class,
)


@pytest.mark.unit
class TestInstrumentClass:
    def test_truth_table(self):
        # (asset_type, ticker) -> class
        assert instrument_class("stock", "AAPL") == "equity"
        assert instrument_class(None, "AAPL") == "equity"
        assert instrument_class("crypto", "BTC-USD") == "crypto_spot"
        assert instrument_class("crypto_spot", "BTCUSDT") == "crypto_spot"
        assert instrument_class("crypto_perp", "BTCUSDT") == "pure_crypto_perp"
        assert instrument_class("crypto_perp", "MUUSDT") == "stock_perp"
        # ETH (like BTC) is not in the EQUITY-perp seed universe.
        assert instrument_class("crypto_perp", "ETHUSDT") == "pure_crypto_perp"

    def test_fundamentals_applicable_matches_class(self):
        assert fundamentals_applicable("stock", "AAPL") is True
        assert fundamentals_applicable("crypto_perp", "MUUSDT") is True
        assert fundamentals_applicable("crypto_perp", "BTCUSDT") is False
        assert fundamentals_applicable("crypto", "BTC-USD") is False
        assert fundamentals_applicable("crypto_spot", "BTCUSDT") is False

    def test_crypto_family_values(self):
        assert {"crypto", "crypto_spot", "crypto_perp"} == CRYPTO_FAMILY


class _SpyLLM(Runnable):
    """LLM stub that records every invoke — the skip path must record none."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.calls += 1
        return MagicMock(content="MOCK FUNDAMENTALS REPORT")

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def _state(ticker, asset_type):
    return {
        "trade_date": "2026-08-17",
        "company_of_interest": ticker,
        "asset_type": asset_type,
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


@pytest.mark.unit
class TestFundamentalsNodeRuntimeSkip:
    def _run(self, monkeypatch, ticker, asset_type):
        import yialpha.dataflows.interface as iface

        vendor_calls: list[str] = []
        real_route = iface.route_to_vendor

        def counting_route(method, *args, **kwargs):
            vendor_calls.append(method)
            return real_route(method, *args, **kwargs)

        monkeypatch.setattr(iface, "route_to_vendor", counting_route)
        llm = _SpyLLM()
        out = create_fundamentals_analyst(llm)(_state(ticker, asset_type))
        return llm, vendor_calls, out

    def test_pure_crypto_perp_skips_with_zero_calls(self, monkeypatch):
        llm, vendor_calls, out = self._run(monkeypatch, "BTCUSDT", "crypto_perp")
        assert llm.calls == 0, "skip path must not bill an LLM turn"
        assert vendor_calls == [], "skip path must make zero vendor calls"
        assert "skipped" in out["fundamentals_report"]
        assert "BTCUSDT" in out["fundamentals_report"]
        # The skip note must not fabricate a messages update.
        assert "messages" not in out

    def test_legacy_crypto_mode_skips_too(self, monkeypatch):
        llm, _vendor_calls, out = self._run(monkeypatch, "BTC-USD", "crypto")
        assert llm.calls == 0
        assert "skipped" in out["fundamentals_report"]

    def test_stock_perp_runs_the_real_analyst(self, monkeypatch):
        from yialpha.dataflows import config as cfgmod

        orig = cfgmod.get_config()
        try:
            cfgmod.set_config({**orig, "web_search_enabled": False})
            llm, _vendor_calls, out = self._run(monkeypatch, "MUUSDT", "crypto_perp")
            assert llm.calls == 1, "a tokenized-stock perp gets the real analyst"
            assert "skipped" not in out["fundamentals_report"]
        finally:
            cfgmod.set_config(orig)

    def test_plain_stock_unchanged(self, monkeypatch):
        llm, _vendor_calls, out = self._run(monkeypatch, "AAPL", "stock")
        assert llm.calls == 1
        assert "skipped" not in out["fundamentals_report"]


@pytest.mark.unit
class TestCliFilterDelegates:
    """The CLI filter and the node skip must agree — one predicate, two layers."""

    def test_filter_matches_predicate(self):
        from yialpha.cli.models import AnalystType, AssetType
        from yialpha.cli.utils import filter_analysts_for_asset_type

        analysts = [
            AnalystType.MARKET,
            AnalystType.SOCIAL,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ]
        for asset_type in (AssetType.STOCK, AssetType.CRYPTO,
                           AssetType.CRYPTO_SPOT, AssetType.CRYPTO_PERP):
            for ticker in ("AAPL", "BTCUSDT", "MUUSDT"):
                kept = filter_analysts_for_asset_type(analysts, asset_type, ticker)
                want_fundamentals = fundamentals_applicable(asset_type.value, ticker)
                got_fundamentals = AnalystType.FUNDAMENTALS in kept
                assert want_fundamentals == got_fundamentals, (
                    asset_type, ticker, kept,
                )


@pytest.mark.unit
class TestInstrumentDescriptor:
    """describe_instrument: routing facts + contract metadata, fetch-free."""

    def setup_method(self):
        # Never depend on a prior test's warmed snapshot.
        from yialpha.dataflows.binance import refresh_equity_perp_bases

        refresh_equity_perp_bases()

    def test_equity_descriptor(self):
        from yialpha.graph.routing import describe_instrument

        d = describe_instrument("stock", "AAPL")
        assert d.instrument_class == "equity"
        assert d.is_etf is False
        assert d.contract_kind == "security"
        assert d.vendor_symbols == {"yfinance": "AAPL", "sec_edgar": "AAPL"}
        assert d.fundamentals_symbol == "AAPL"
        # No PIT listing source for plain equities: unknown, not false.
        assert d.listed_asof is None
        assert d.as_disclosure() == ""  # nothing to disclose without as_of
        with_asof = describe_instrument("stock", "AAPL", as_of="2026-01-15")
        assert with_asof.listed_asof is None
        assert "unknown" in with_asof.as_disclosure()

    def test_stock_perp_descriptor_vendor_aliases_and_session(self):
        from yialpha.graph.routing import (
            SESSION_BINANCE_TRADFI,
            describe_instrument,
        )

        d = describe_instrument("crypto_perp", "MUUSDT")
        assert d.instrument_class == "stock_perp"
        assert d.contract_kind == "perpetual"
        assert d.delivery_date is None
        assert d.session_calendar == SESSION_BINANCE_TRADFI
        assert d.vendor_symbols == {
            "binance": "MUUSDT", "yfinance": "MU", "sec_edgar": "MU",
        }
        assert d.fundamentals_symbol == "MU"

    def test_pure_crypto_descriptor_continuous_session(self):
        from yialpha.graph.routing import SESSION_CONTINUOUS, describe_instrument

        d = describe_instrument("crypto_perp", "BTCUSDT")
        assert d.instrument_class == "pure_crypto_perp"
        assert d.session_calendar == SESSION_CONTINUOUS
        assert d.underlying_equity is None
        assert d.vendor_symbols == {"binance": "BTCUSDT"}

    def test_etf_flag_for_plain_and_perp_forms(self):
        from yialpha.graph.routing import describe_instrument

        assert describe_instrument("stock", "SPY").is_etf is True
        assert describe_instrument("crypto_perp", "SPYUSDT").is_etf is True
        assert describe_instrument("stock", "MU").is_etf is False

    def test_asof_listing_classification_uses_onboard_date(self, monkeypatch):
        import yialpha.dataflows.binance as bnb
        from yialpha.graph.routing import describe_instrument

        monkeypatch.setattr(
            bnb, "_EQUITY_PERP_LISTING_CACHE",
            {"MU": {"onboard_date": "2026-03-10", "status": "TRADING"}},
        )
        before = describe_instrument("crypto_perp", "MUUSDT", as_of="2026-01-15")
        after = describe_instrument("crypto_perp", "MUUSDT", as_of="2026-06-01")
        assert before.listed_asof is False
        assert after.listed_asof is True
        assert after.onboard_date == "2026-03-10"
        assert "NOT yet listed" in before.as_disclosure()
        assert "listed" in after.as_disclosure()

    def test_asof_unknown_without_listing_evidence(self):
        from yialpha.graph.routing import describe_instrument

        # Seed mode (no warmed listing snapshot): as-of verdict is unknown —
        # never silently False.
        d = describe_instrument("crypto_perp", "MUUSDT", as_of="2026-01-15")
        assert d.listed_asof is None
        assert d.onboard_date is None
        assert "unknown" in d.as_disclosure()


@pytest.mark.unit
class TestWarmEquityPerpBasesRefetch:
    """A failed listing warm must not poison the process cache — the next
    perp-run start retries the live exchangeInfo instead of serving a
    transient outage's seed forever."""

    def _exchange_info(self, onboard_ms=1_770_000_000_000):
        return {
            "symbols": [
                {
                    "symbol": "MUUSDT", "quoteAsset": "USDT",
                    "underlyingType": "EQUITY", "status": "TRADING",
                    "onboardDate": onboard_ms,
                },
            ],
        }

    def test_failure_then_success_rewarming(self, monkeypatch):
        import yialpha.dataflows.binance as bnb

        bnb.refresh_equity_perp_bases()
        calls = {"n": 0}

        def flaky(path, params, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient outage")
            return self._exchange_info()

        monkeypatch.setattr(bnb, "_http_get", flaky)

        first = bnb.warm_equity_perp_bases()
        assert first == bnb._EQUITY_PERP_SEED_BASES
        # The failure was NOT cached: the second warm re-fetches and wins.
        second = bnb.warm_equity_perp_bases()
        assert calls["n"] == 2
        assert "MU" in second

        listing = bnb.equity_perp_listing_info()
        assert listing["MU"]["onboard_date"] == "2026-02-02"
        assert listing["MU"]["status"] == "TRADING"
        bnb.refresh_equity_perp_bases()

    def test_success_caches_across_calls(self, monkeypatch):
        import yialpha.dataflows.binance as bnb

        bnb.refresh_equity_perp_bases()
        calls = {"n": 0}

        def steady(path, params, **kwargs):
            calls["n"] += 1
            return self._exchange_info()

        monkeypatch.setattr(bnb, "_http_get", steady)
        assert bnb.warm_equity_perp_bases() == bnb.warm_equity_perp_bases()
        assert calls["n"] == 1
        bnb.refresh_equity_perp_bases()

    def test_success_re_fetches_after_ttl(self, monkeypatch):
        # A successful warm is honoured for the TTL only: a long-lived
        # process (web subprocess) must pick up newly listed equity perps
        # at the next perp-run start once the snapshot has aged out, instead
        # of serving its first warm forever.
        import yialpha.dataflows.binance as bnb

        bnb.refresh_equity_perp_bases()
        calls = {"n": 0}

        def steady(path, params, **kwargs):
            calls["n"] += 1
            return self._exchange_info()

        monkeypatch.setattr(bnb, "_http_get", steady)
        bnb.warm_equity_perp_bases()
        assert calls["n"] == 1
        # Age the snapshot past the TTL (monotonic clock moves forward only,
        # so rewind the recorded warm time instead).
        recorded = bnb._EQUITY_PERP_WARMED_AT_MONO
        try:
            bnb._EQUITY_PERP_WARMED_AT_MONO = (
                recorded - bnb._EQUITY_PERP_WARM_TTL_S - 1.0
            )
            bnb.warm_equity_perp_bases()
            assert calls["n"] == 2  # stale snapshot re-fetched
        finally:
            bnb._EQUITY_PERP_WARMED_AT_MONO = recorded
            bnb.refresh_equity_perp_bases()

    def test_stale_refetch_failure_serves_previous_snapshot(self, monkeypatch):
        # A re-fetch failure after TTL must NOT downgrade to the static seed:
        # the previous (stale but real) exchangeInfo snapshot is better
        # evidence than the seed and keeps serving until a re-warm succeeds.
        import yialpha.dataflows.binance as bnb

        bnb.refresh_equity_perp_bases()
        calls = {"n": 0}

        def flaky(path, params, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return self._exchange_info()
            raise RuntimeError("outage after TTL")

        monkeypatch.setattr(bnb, "_http_get", flaky)
        first = bnb.warm_equity_perp_bases()
        assert "MU" in first
        bnb._EQUITY_PERP_WARMED_AT_MONO = (
            bnb._EQUITY_PERP_WARMED_AT_MONO - bnb._EQUITY_PERP_WARM_TTL_S - 1.0
        )
        second = bnb.warm_equity_perp_bases()
        assert second == first  # previous snapshot, not the seed
        assert "MU" in second
        bnb.refresh_equity_perp_bases()
