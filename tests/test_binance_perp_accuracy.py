"""Perp-accuracy expansion tests (2026-08-16 audit round).

Pins the Binance data-layer contract fixes: un-rounded OHLC for low-price
contracts (P0), mark-price kline routing, funding-cadence inference, the
premiumIndex snapshot tool, the live-window staleness guard, the
``_futures_data_window`` start-only PIT clamp, the spot 6000/min weight
budget, and the exchangeInfo filter/quantization helpers.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
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
    return int(datetime.now(UTC).replace(hour=0, minute=0, second=0,
                                                  microsecond=0).timestamp() * 1000)


def _klines_server(all_klines):
    def _mock(path, params, symbol, canonical, **kwargs):  # noqa: ARG001
        start, end, limit = params["startTime"], params["endTime"], params["limit"]
        return [k for k in all_klines if start <= k[0] <= end][:limit]
    return _mock


def _today_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


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
        base = int(datetime(2026, 8, 1, tzinfo=UTC).timestamp() * 1000)
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
        base = int(datetime(2026, 8, 1, tzinfo=UTC).timestamp() * 1000)
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

    def test_fundinginfo_interval_wins_over_spacing(self):
        """The authoritative /fapi/v1/fundingInfo interval beats inference.

        A 4h-spaced series with fundingInfo declaring 4h and a mismatched
        spacing must resolve to the DECLARED value, and the header must say
        where the number came from.
        """
        base = int(datetime(2026, 8, 1, tzinfo=UTC).timestamp() * 1000)
        # Spacing says 8h, fundingInfo says 4h — the endpoint is authoritative.
        rows = [
            {"fundingTime": base + i * 8 * 3_600_000, "fundingRate": "0.0001",
             "symbol": "XYZUSDT"}
            for i in range(6)
        ]

        def fake_get(path, *a, **k):
            if "fundingInfo" in path:
                return [{"symbol": "XYZUSDT", "fundingIntervalHours": 4}]
            return rows

        with mock.patch.object(bn, "_http_get", fake_get):
            out = bn.get_binance_funding_rate("XYZUSDT", "2026-08-01", "2026-08-10")
        self.assertIn("~4h", out)
        self.assertIn("fundingInfo endpoint", out)

    def test_fundinginfo_failure_falls_back_to_inference(self):
        base = int(datetime(2026, 8, 1, tzinfo=UTC).timestamp() * 1000)
        rows = [
            {"fundingTime": base + i * 4 * 3_600_000, "fundingRate": "0.0001",
             "symbol": "XYZUSDT"}
            for i in range(12)
        ]

        def fake_get(path, *a, **k):
            if "fundingInfo" in path:
                raise RuntimeError("transport blip")
            return rows

        with mock.patch.object(bn, "_http_get", fake_get):
            out = bn.get_binance_funding_rate("XYZUSDT", "2026-08-01", "2026-08-10")
        self.assertIn("~4h", out)
        self.assertIn("inferred from settlement spacing", out)


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
        now = datetime.now(UTC)
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
            extra, end_iso, reaches_now, end_ms, coverage_note = bn._futures_data_window(
                "BTCUSDT", "BTCUSDT", 7, "2026-07-25", None,
            )
        finally:
            set_analysis_date(None)
        self.assertEqual(end_iso, "2026-08-01")
        self.assertFalse(reaches_now)
        # 8 daily rows fit the 500-row cap: no truncation, no note.
        self.assertEqual(coverage_note, "")
        # endTime must be within the pinned day (2026-08-01 end-of-day UTC).
        end_dt = datetime.fromtimestamp(extra["endTime"] / 1000, tz=UTC)
        self.assertEqual(end_dt.date().isoformat(), "2026-08-01")
        # end_ms (the ms-based trim anchor) is the same end-of-day instant.
        self.assertEqual(end_ms, extra["endTime"])

    def test_live_start_only_reaches_now(self):
        extra, end_iso, reaches_now, end_ms, coverage_note = bn._futures_data_window(
            "BTCUSDT", "BTCUSDT", 7, None, None,
        )
        self.assertEqual(extra, {})
        self.assertIsNone(end_iso)
        self.assertTrue(reaches_now)
        self.assertIsNone(end_ms)
        self.assertEqual(coverage_note, "")


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

    def test_market_order_ref_price_prechecks_notional(self):
        """Round-5: MARKET orders DO get a -4014 at submit, not at fill."""
        f = bf.SymbolFilters(
            symbol="BTCUSDT", status="TRADING",
            tick_size=bf.Decimal("0.10"), step_size=bf.Decimal("1"),
            min_qty=bf.Decimal("1"), max_qty=bf.Decimal("0"),
            min_notional=bf.Decimal("5"), price_precision=2, quantity_precision=0,
        )
        out = bf.quantize_order(None, 2.0, f, ref_price=2.0)  # 4 < 5
        self.assertTrue(out["below_min_notional"])
        out_ok = bf.quantize_order(None, 3.0, f, ref_price=2.0)  # 6 >= 5
        self.assertFalse(out_ok["below_min_notional"])

    def test_market_order_without_any_price_leaves_floor_undecided(self):
        f = bf.SymbolFilters(
            symbol="BTCUSDT", status="TRADING",
            tick_size=bf.Decimal("0.10"), step_size=bf.Decimal("1"),
            min_qty=bf.Decimal("1"), max_qty=bf.Decimal("0"),
            min_notional=bf.Decimal("5"), price_precision=2, quantity_precision=0,
        )
        out = bf.quantize_order(None, 2.0, f)  # no price, no ref
        self.assertFalse(out["below_min_notional"])  # undecided, exchange decides

    def test_notional_exactly_at_minimum_not_float_shaved(self):
        """Decimal-exact floor: an order landing exactly on the min passes."""
        f = bf.SymbolFilters(
            symbol="BTCUSDT", status="TRADING",
            tick_size=bf.Decimal("0.10"), step_size=bf.Decimal("1"),
            min_qty=bf.Decimal("1"), max_qty=bf.Decimal("0"),
            min_notional=bf.Decimal("0.3"), price_precision=2, quantity_precision=0,
        )
        out = bf.quantize_order(0.3, 1.0, f)  # exactly 0.3 (0.1 float is < 0.3)
        self.assertEqual(out["price"], 0.3)
        self.assertFalse(out["below_min_notional"])


class BasisPrecisionTests(unittest.TestCase):
    """Round-5: round(x, 6) collapsed a sub-cent basis to exactly 0.0."""

    def test_low_price_basis_keeps_precision(self):
        base = _today_ms() - 8 * _DAY_MS
        perp_rows = [
            _kline(base + i * _DAY_MS, 1.23456e-5, 1.24e-5, 1.23e-5, 1.23456e-5)
            for i in range(9)
        ]
        spot_rows = [
            _kline(base + i * _DAY_MS, 1.21e-5, 1.22e-5, 1.20e-5, 1.21e-5)
            for i in range(9)
        ]

        def server(path, params, symbol, canonical, **kwargs):
            rows = perp_rows if path.startswith("/fapi") else spot_rows
            return _klines_server(rows)(path, params, symbol, canonical, **kwargs)

        with mock.patch.object(bn, "_http_get", server):
            out = bn.get_binance_spot_perp_basis("PEPEUSDT", look_back_days=7)
        # The true per-venue basis is 2.456e-07; round(..., 6) rendered it 0.0
        # while basisRate said ~2%. Six significant digits keep it visible.
        self.assertIn("2.456e-07", out)
        # Closes keep their significant digits too (1.23456e-5, not 1.2e-05).
        self.assertIn("1.23456e-05", out)


if __name__ == "__main__":
    unittest.main()
