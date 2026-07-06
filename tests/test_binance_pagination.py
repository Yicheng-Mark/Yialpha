"""Binance klines/fundingRate pagination: long ranges must not be truncated.

fapi caps /klines at 1500 and /fundingRate at 1000 rows per request and returns
them oldest-first. Without paging, a range longer than the cap silently dropped
the most recent, decision-critical rows. These tests mock the HTTP layer with a
fake Binance that honors startTime/endTime/limit exactly, then assert the
public functions page through the full range with no gaps and no truncation.
"""
import unittest
from datetime import datetime, timezone
from unittest import mock

from yiagents.dataflows import binance

_DAY_MS = 86_400_000
_FUND_MS = 28_800_000  # 8h funding interval


def _fake_klines_server(all_klines):
    """A mock ``_http_get`` that pages /fapi/v1/klines like Binance."""
    def _mock(path, params, symbol, canonical, **kwargs):  # noqa: ARG001
        start, end, limit = params["startTime"], params["endTime"], params["limit"]
        return [k for k in all_klines if start <= k[0] <= end][:limit]
    return _mock


def _fake_funding_server(all_rows):
    """A mock ``_http_get`` that pages /fapi/v1/fundingRate like Binance."""
    def _mock(path, params, symbol, canonical, **kwargs):  # noqa: ARG001
        start, end, limit = params["startTime"], params["endTime"], params["limit"]
        return [r for r in all_rows if start <= r["fundingTime"] <= end][:limit]
    return _mock


class TestKlinesPagination(unittest.TestCase):
    def test_long_range_is_not_truncated(self):
        # 2000 daily bars: the old default-limit (500) and even one 1500-page
        # would drop recent rows. Paging must return all 2000, oldest-first.
        base = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        all_klines = [
            [base + i * _DAY_MS, "1.0", "2.0", "0.5", "1.5", "100.0"]
            for i in range(2000)
        ]
        with mock.patch.object(binance, "_http_get", _fake_klines_server(all_klines)):
            out = binance.get_binance_klines("BTCUSDT", "2020-01-01", "2025-12-31")
        self.assertIn("# Total records: 2000", out)
        # CSV body: one row per bar (exclude header comment lines + csv header).
        data_lines = [
            ln for ln in out.splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("Date")
        ]
        self.assertEqual(len(data_lines), 2000)

    def test_long_range_is_monotonic_no_gaps(self):
        base = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        all_klines = [
            [base + i * _DAY_MS, "1.0", "2.0", "0.5", "1.5", "100.0"]
            for i in range(1800)
        ]
        with mock.patch.object(binance, "_http_get", _fake_klines_server(all_klines)):
            out = binance.get_binance_klines("BTCUSDT", "2020-01-01", "2025-12-31")
        data_lines = [
            ln for ln in out.splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("Date")
        ]
        dates = [ln.split(",")[0] for ln in data_lines]
        # Strictly increasing, one calendar day apart, no duplicates/gaps.
        self.assertEqual(len(set(dates)), len(dates))
        self.assertEqual(dates[0], "2020-01-01")
        d_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        for prev, cur in zip(d_objs, d_objs[1:]):
            self.assertEqual((cur - prev).days, 1)

    def test_short_range_single_page_unchanged(self):
        # A range under one page must return exactly its bars (no over-fetch).
        base = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        all_klines = [
            [base + i * _DAY_MS, "1.0", "2.0", "0.5", "1.5", "100.0"]
            for i in range(10)
        ]
        with mock.patch.object(binance, "_http_get", _fake_klines_server(all_klines)):
            out = binance.get_binance_klines("BTCUSDT", "2020-01-01", "2020-01-31")
        self.assertIn("# Total records: 10", out)


class TestFundingPagination(unittest.TestCase):
    def test_long_range_is_not_truncated(self):
        # 2500 funding rows (8h cadence): the old limit=1000 would drop ~1500.
        base = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        all_rows = [
            {"fundingTime": base + i * _FUND_MS, "fundingRate": "0.0001",
             "symbol": "BTCUSDT"}
            for i in range(2500)
        ]
        with mock.patch.object(binance, "_http_get", _fake_funding_server(all_rows)):
            out = binance.get_binance_funding_rate("BTCUSDT", "2020-01-01", "2025-12-31")
        self.assertIn("# Total records: 2500", out)
        data_lines = [
            ln for ln in out.splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("fundingTime")
        ]
        self.assertEqual(len(data_lines), 2500)

    def test_long_range_monotonic(self):
        base = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        all_rows = [
            {"fundingTime": base + i * _FUND_MS, "fundingRate": "0.0001",
             "symbol": "BTCUSDT"}
            for i in range(1200)
        ]
        with mock.patch.object(binance, "_http_get", _fake_funding_server(all_rows)):
            out = binance.get_binance_funding_rate("BTCUSDT", "2020-01-01", "2025-12-31")
        times = [
            ln.split(",")[0]
            for ln in out.splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("fundingTime")
        ]
        # No duplicates; ascending (string compare works on "%Y-%m-%d %H:%M:%S").
        self.assertEqual(len(set(times)), len(times))
        self.assertEqual(times, sorted(times))


class TestNoDataStillRaises(unittest.TestCase):
    def test_empty_klines_raises_no_market_data(self):
        from yiagents.dataflows.errors import NoMarketDataError
        with mock.patch.object(binance, "_http_get", _fake_klines_server([])):
            with self.assertRaises(NoMarketDataError):
                binance.get_binance_klines("BTCUSDT", "2020-01-01", "2020-01-31")


if __name__ == "__main__":
    unittest.main()
