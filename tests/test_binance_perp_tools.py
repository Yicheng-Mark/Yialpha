"""Perp tool wrapper pass-through pins (round 9).

The 10 ``@tool`` wrappers in ``agents/utils/binance_perp_tools`` are thin
pass-throughs to ``route_to_vendor``; until now only their NAMES were pinned
(test_crypto_perp_mode: live 12 / historical 5). A silently swapped positional
would serve the wrong window's crowding data with no error anywhere — the LLM
asks for 5 days of OI, the vendor serves 5 days of something else. These
tests pin the exact vendor-name + positional/keyword mapping for every
wrapper, plus the klines ``_qualifier`` threading (an INDEX-kline miss must
grade auxiliary, last/mark stays the critical price book — see
``quality._is_critical_failure``). The vendor itself is fully covered by the
binance vendor test files; only the plumbing is pinned here.
"""

from __future__ import annotations

import pytest

import yialpha.agents.utils.binance_perp_tools as bpt


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
    monkeypatch.setattr(bpt, "route_to_vendor", _recorder(calls))
    return calls


@pytest.mark.unit
class TestKlinesPassthrough:
    def test_defaults_thread_interval_and_price_type(self, routed):
        bpt.get_binance_klines.invoke({
            "symbol": "BTCUSDT",
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
        })
        assert routed == [{
            "method": "get_binance_klines",
            "args": ("BTCUSDT", "2026-01-01", "2026-01-10", "1d", "last"),
            "kwargs": {},
            "qualifier": "last",
        }]

    def test_intraday_mark_threaded_positionally(self, routed):
        bpt.get_binance_klines.invoke({
            "symbol": "ETHUSDT",
            "start_date": "2026-02-01",
            "end_date": "2026-02-02",
            "interval": "1h",
            "price_type": "mark",
        })
        assert routed[0]["args"] == (
            "ETHUSDT", "2026-02-01", "2026-02-02", "1h", "mark",
        )
        assert routed[0]["qualifier"] == "mark"

    def test_index_qualifier_labels_auxiliary_basis(self, routed):
        # The liquidation anchor is last/mark; the settlement reference is
        # enrichment. A regression that drops _qualifier (or forwards the
        # wrong field) would flip sentinel grading for every klines miss.
        bpt.get_binance_klines.invoke({
            "symbol": "BTCUSDT",
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
            "price_type": "index",
        })
        assert routed[0]["qualifier"] == "index"
        assert routed[0]["args"][4] == "index"


@pytest.mark.unit
class TestDateRangePassthrough:
    def test_funding_rate_three_positionals(self, routed):
        bpt.get_binance_funding_rate.invoke({
            "symbol": "BTCUSDT",
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
        })
        assert routed == [{
            "method": "get_binance_funding_rate",
            "args": ("BTCUSDT", "2026-01-01", "2026-01-31"),
            "kwargs": {},
            "qualifier": "",
        }]

    def test_vision_metrics_interval_positional(self, routed):
        bpt.get_binance_vision_metrics.invoke({
            "symbol": "BTCUSDT",
            "start_date": "2025-06-01",
            "end_date": "2025-06-30",
            "interval": "4h",
        })
        assert routed[0]["args"] == (
            "BTCUSDT", "2025-06-01", "2025-06-30", "4h",
        )

    def test_vision_book_depth_default_interval(self, routed):
        bpt.get_binance_vision_book_depth.invoke({
            "symbol": "1000PEPEUSDT",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        })
        assert routed[0]["method"] == "get_binance_vision_book_depth"
        assert routed[0]["args"] == (
            "1000PEPEUSDT", "2025-01-01", "2025-01-31", "1d",
        )


@pytest.mark.unit
class TestLookbackPassthrough:
    # OI / LSR / taker / basis share one shape: (symbol, look_back_days)
    # positional + period keyword. A swapped (days, period) would ask the
    # vendor for e.g. period="7", days="1h" — nonsense the vendor layer
    # might silently coerce instead of refusing.

    def test_open_interest(self, routed):
        bpt.get_binance_open_interest.invoke({
            "symbol": "BTCUSDT", "look_back_days": 5, "period": "1h",
        })
        assert routed == [{
            "method": "get_binance_open_interest",
            "args": ("BTCUSDT", 5),
            "kwargs": {"period": "1h"},
            "qualifier": "",
        }]

    def test_open_interest_defaults(self, routed):
        bpt.get_binance_open_interest.invoke({"symbol": "SOLUSDT"})
        assert routed[0]["args"] == ("SOLUSDT", 7)
        assert routed[0]["kwargs"] == {"period": "1d"}

    def test_long_short_ratio(self, routed):
        bpt.get_binance_long_short_ratio.invoke({
            "symbol": "ETHUSDT", "look_back_days": 14, "period": "5m",
        })
        assert routed[0]["method"] == "get_binance_long_short_ratio"
        assert routed[0]["args"] == ("ETHUSDT", 14)
        assert routed[0]["kwargs"] == {"period": "5m"}

    def test_taker_buy_sell(self, routed):
        bpt.get_binance_taker_buy_sell.invoke({
            "symbol": "DOGEUSDT", "look_back_days": 3, "period": "15m",
        })
        assert routed[0]["method"] == "get_binance_taker_buy_sell"
        assert routed[0]["args"] == ("DOGEUSDT", 3)
        assert routed[0]["kwargs"] == {"period": "15m"}

    def test_basis(self, routed):
        bpt.get_binance_basis.invoke({
            "symbol": "BTCUSDT", "look_back_days": 30, "period": "1h",
        })
        assert routed[0]["method"] == "get_binance_basis"
        assert routed[0]["args"] == ("BTCUSDT", 30)
        assert routed[0]["kwargs"] == {"period": "1h"}


@pytest.mark.unit
class TestSnapshotPassthrough:
    def test_premium_index_single_arg(self, routed):
        bpt.get_binance_premium_index.invoke({"symbol": "BTCUSDT"})
        assert routed == [{
            "method": "get_binance_premium_index",
            "args": ("BTCUSDT",),
            "kwargs": {},
            "qualifier": "",
        }]

    def test_depth_snapshot_limit_positional(self, routed):
        bpt.get_binance_depth_snapshot.invoke({
            "symbol": "BTCUSDT", "limit": 500,
        })
        assert routed == [{
            "method": "get_binance_depth_snapshot",
            "args": ("BTCUSDT", 500),
            "kwargs": {},
            "qualifier": "",
        }]
