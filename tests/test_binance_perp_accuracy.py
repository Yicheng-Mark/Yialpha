"""Perp-accuracy expansion tests (2026-08-16 audit round).

Pins the Binance data-layer contract fixes: un-rounded OHLC for low-price
contracts (P0), mark-price kline routing, funding-cadence inference, the
premiumIndex snapshot tool, the live-window staleness guard, the
``_futures_data_window`` start-only PIT clamp, the spot 6000/min weight
budget, and the exchangeInfo filter/quantization helpers.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from yiagents.dataflows import binance as bn, binance_filters as bf
from yiagents.dataflows.binance_rate_limiter import (
    get_binance_weight_limiter,
    reset_for_test as reset_limiters,
)
from yiagents.dataflows.errors import NoMarketDataError
from yiagents.dataflows.utils import set_analysis_date

_DAY_MS = 86_400_000


def _kline(open_ms: int, o: float, h: float, lo: float, c: float) -> list:
    return [open_ms, str(o), str(h), str(lo), str(c), "100.0",
            open_ms + _DAY_MS, "1.0", 1, "50.0", "50.0", "0"]


def _today_ms() -> int:
    return int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                                  microsecond=0).timestamp() * 1000)


def _klines_server(all_klines):
    def _mock(path, params, symbol, canonical, **kwargs):  # noqa: ARG001
        start, end, limit = params["startTime"], params["endTime"], params["limit"]
        return [k for k in all_klines if start <= k[0] <= end][:limit]
    return _mock


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class LowPricePrecisionTests(unittest.TestCase):
    """P0: sub-cent contracts must survive the klines frame un-rounded."""

    def test_low_price_klines_not_rounded(self):
        base = _today_ms() - 20 * _DAY_MS
        rows = [_kline(base + i * _DAY_MS, 1.2e-5, 1.25e-5, 1.15e-5, 1.22e-5)
                for i in range(21)]
        with mock.patch.object(bn, "_http_get", _klines_server(rows)):
            df = bn.binance_klines_frame("PEPEUSDT", "2026-01-01", _today_iso())
            self.assertTrue((df["Close"] > 1e-6).all())
            self.assertAlmostEqual(float(df["Close"].iloc[0]), 1.22e-5, places=10)

            out = bn.get_binance_klines("PEPEUSDT", "2026-01-01", _today_iso())
            self.assertIn("1.22e-05", out)  # pandas repr keeps full precision

    def test_btc_scale_klines_unchanged(self):
        base = _today_ms() - 5 * _DAY_MS
        rows = [_kline(base + i * _DAY_MS, 50000.0, 50100.0, 49900.0, 50050.0)
                for i in range(6)]
        with mock.patch.object(bn, "_http_get", _klines_server(rows)):
            df = bn.binance_klines_frame("BTCUSDT", "2026-01-01", _today_iso())
        self.assertAlmostEqual(float(df["Close"].iloc[0]), 50050.0, places=6)


class MarkPriceKlinesTests(unittest.TestCase):
    def _frame_with_recorder(self, rows, **frame_kwargs):
        seen = {}

        def recorder(path, params, symbol, canonical, **kwargs):
            seen["path"] = path
            return _klines_server(rows)(path, params, symbol, canonical, **kwargs)

        with mock.patch.object(bn, "_http_get", recorder):
            bn.binance_klines_frame(
                "BTCUSDT", "2026-01-01", _today_iso(), **frame_kwargs
            )
        return seen

    def test_mark_price_routes_to_mark_endpoint(self):
        base = _today_ms() - 5 * _DAY_MS
        rows = [_kline(base + i * _DAY_MS, 100.0, 101.0, 99.0, 100.5)
                for i in range(6)]
        seen = self._frame_with_recorder(rows, price_type="mark")
        self.assertEqual(seen["path"], "/fapi/v1/markPriceKlines")

    def test_last_price_still_default(self):
        base = _today_ms() - 5 * _DAY_MS
        rows = [_kline(base + i * _DAY_MS, 100.0, 101.0, 99.0, 100.5)
                for i in range(6)]
        seen = self._frame_with_recorder(rows)
        self.assertEqual(seen["path"], "/fapi/v1/klines")

    def test_spot_mark_combination_rejected(self):
        with self.assertRaises(ValueError):
            bn.binance_klines_frame(
                "BTCUSDT", "2026-01-01", "2026-12-31",
                venue="binance_spot", price_type="mark",
            )


class FundingCadenceTests(unittest.TestCase):
    def test_4h_cadence_stated_in_header(self):
        base = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp() * 1000)
        rows = [
            {"fundingTime": base + i * 4 * 3_600_000, "fundingRate": "0.0001",
             "symbol": "XYZUSDT"}
            for i in range(12)
        ]
        with mock.patch.object(
            bn, "_http_get",
            lambda *a, **k: rows if "fundingRate" in a[0] else [],
        ):
            out = bn.get_binance_funding_rate("XYZUSDT", "2026-08-01", "2026-08-10")
        self.assertIn("~4h", out)
        self.assertIn("6.0", out)  # 24/4 settlements per day in the annualise hint

    def test_8h_default_cadence_still_stated(self):
        base = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp() * 1000)
        rows = [
            {"fundingTime": base + i * 8 * 3_600_000, "fundingRate": "0.0001",
             "symbol": "BTCUSDT"}
            for i in range(6)
        ]
        with mock.patch.object(
            bn, "_http_get",
            lambda *a, **k: rows if "fundingRate" in a[0] else [],
        ):
            out = bn.get_binance_funding_rate("BTCUSDT", "2026-08-01", "2026-08-10")
        self.assertIn("~8h", out)


class PremiumIndexTests(unittest.TestCase):
    _SNAP = {
        "symbol": "BTCUSDT", "markPrice": "50100.5", "indexPrice": "50000.0",
        "lastFundingRate": "0.00010000",
        "nextFundingTime": 1767225600000, "time": 1767222000000,
    }

    def test_snapshot_renders_mark_vs_index(self):
        with mock.patch.object(bn, "_http_get", lambda *a, **k: dict(self._SNAP)):
            out = bn.get_binance_premium_index("BTCUSDT")
        self.assertIn("markPrice", out)
        self.assertIn("0.201", out)  # round((50100.5/50000 - 1)*100, 4)
        self.assertIn("liquidations trigger on markPrice", out)

    def test_missing_mark_price_degrades(self):
        bad = {"symbol": "X", "indexPrice": "1"}
        with (
            mock.patch.object(bn, "_http_get", lambda *a, **k: bad),
            self.assertRaises(NoMarketDataError),
        ):
            bn.get_binance_premium_index("XUSDT")


class StalenessGuardTests(unittest.TestCase):
    def test_live_window_stale_frame_rejected(self):
        # Rows end 30 days ago but the caller asked through today: a
        # delisted contract's stale candles must raise, not feed the analyst.
        last = _today_ms() - 30 * _DAY_MS
        rows = [_kline(last - (20 - i) * _DAY_MS, 100, 101, 99, 100.5)
                for i in range(21)]
        with (
            mock.patch.object(bn, "_http_get", _klines_server(rows)),
            self.assertRaises(NoMarketDataError) as ctx,
        ):
            bn.binance_klines_frame("DEADUSDT", "2026-01-01", "2026-12-31")
        self.assertIn("stale", str(ctx.exception))

    def test_historical_window_early_end_allowed(self):
        # A backtest window is exempt: an early-ending series is a
        # legitimate input, not a freshness lie.
        now = datetime.now(timezone.utc)
        end = (now - timedelta(days=200)).strftime("%Y-%m-%d")
        start = (now - timedelta(days=260)).strftime("%Y-%m-%d")
        base = int(
            (now - timedelta(days=260)).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp() * 1000
        )
        rows = [_kline(base + i * _DAY_MS, 100, 101, 99, 100.5) for i in range(50)]
        with mock.patch.object(bn, "_http_get", _klines_server(rows)):
            df = bn.binance_klines_frame("OLDUSDT", start, end)
        self.assertEqual(len(df), 50)


class FuturesDataWindowPITTests(unittest.TestCase):
    def test_start_only_clamps_to_pinned_analysis_date(self):
        set_analysis_date("2026-08-01")
        try:
            extra, end_iso, reaches_now = bn._futures_data_window(
                "BTCUSDT", "BTCUSDT", 7, "2026-07-25", None,
            )
        finally:
            set_analysis_date(None)
        self.assertEqual(end_iso, "2026-08-01")
        self.assertFalse(reaches_now)
        # endTime must be within the pinned day (2026-08-01 end-of-day UTC).
        end_dt = datetime.fromtimestamp(extra["endTime"] / 1000, tz=timezone.utc)
        self.assertEqual(end_dt.date().isoformat(), "2026-08-01")

    def test_live_start_only_reaches_now(self):
        extra, end_iso, reaches_now = bn._futures_data_window(
            "BTCUSDT", "BTCUSDT", 7, None, None,
        )
        self.assertEqual(extra, {})
        self.assertIsNone(end_iso)
        self.assertTrue(reaches_now)


class WeightBudgetTests(unittest.TestCase):
    def test_per_product_documented_budgets(self):
        reset_limiters()
        self.assertEqual(get_binance_weight_limiter("fapi").weight_limit, 2400)
        self.assertEqual(get_binance_weight_limiter("spot").weight_limit, 6000)
        reset_limiters()


class ExchangeInfoFilterTests(unittest.TestCase):
    _INFO = {
        "symbols": [{
            "symbol": "BTCUSDT", "status": "TRADING",
            "pricePrecision": 2, "quantityPrecision": 3,
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10",
                 "minPrice": "0.01", "maxPrice": "1000000"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001",
                 "minQty": "0.001", "maxQty": "1000"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        }]
    }

    def setUp(self):
        bf.reset_for_test()
        self.calls = []

        def fake(path, params, symbol, canonical, **kwargs):  # noqa: ARG001
            self.calls.append(path)
            return dict(self._INFO)

        self._fake = fake

    def test_filters_parsed_and_cached(self):
        with mock.patch.object(bf, "_http_get", self._fake):
            f1 = bf.get_symbol_filters("BTCUSDT")
            f2 = bf.get_symbol_filters("btc-usdt")
        self.assertEqual(len(self.calls), 1)  # TTL cache collapses 2nd fetch
        self.assertEqual(f2, f1)              # same cached object for either alias
        self.assertEqual(str(f1.tick_size), "0.10")
        self.assertEqual(str(f1.step_size), "0.001")
        self.assertEqual(str(f1.min_notional), "5")
        self.assertEqual(f1.status, "TRADING")

    def test_quantize_floors_quantity_to_step(self):
        f = bf.SymbolFilters(
            symbol="BTCUSDT", status="TRADING",
            tick_size=bf.Decimal("0.10"), step_size=bf.Decimal("0.001"),
            min_qty=bf.Decimal("0.001"), max_qty=bf.Decimal("1000"),
            min_notional=bf.Decimal("5"), price_precision=2, quantity_precision=3,
        )
        out = bf.quantize_order(50000.37, 0.30718, f)
        self.assertEqual(out["quantity"], 0.307)
        self.assertEqual(out["price"], 50000.4)
        self.assertFalse(out["below_min_qty"])
        self.assertFalse(out["below_min_notional"])

    def test_quantize_flags_min_notional(self):
        f = bf.SymbolFilters(
            symbol="BTCUSDT", status="TRADING",
            tick_size=bf.Decimal("0.10"), step_size=bf.Decimal("0.001"),
            min_qty=bf.Decimal("0.001"), max_qty=bf.Decimal("1000"),
            min_notional=bf.Decimal("5"), price_precision=2, quantity_precision=3,
        )
        out = bf.quantize_order(2.0, 1.0, f)  # notional 2 < 5
        self.assertTrue(out["below_min_notional"])

    def test_delisted_symbol_degrades(self):
        with mock.patch.object(
            bf, "_http_get", lambda *a, **k: {"symbols": []}
        ), self.assertRaises(NoMarketDataError):
            bf.get_symbol_filters("GONEUSDT")


if __name__ == "__main__":
    unittest.main()
