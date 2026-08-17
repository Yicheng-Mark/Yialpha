"""Binance klines/fundingRate pagination: long ranges must not be truncated.

fapi caps /klines at 1500 and /fundingRate at 1000 rows per request and returns
them oldest-first. Without paging, a range longer than the cap silently dropped
the most recent, decision-critical rows. These tests mock the HTTP layer with a
fake Binance that honors startTime/endTime/limit exactly, then assert the
public functions page through the full range with no gaps and no truncation.
"""
import unittest
from datetime import UTC, datetime
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


class TestClosedWindowMemo(unittest.TestCase):
    """2026-08-16: identical fully-closed windows are fetched once per process.

    The memo kills the 2-3x duplicate full-history pulls one analysis run used
    to make (kline tool + indicator battery + basis + backtest propagation),
    which all counted against the shared per-IP weight budget.
    """

    def _rows(self, n=30):
        base = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
        return [
            {"fundingTime": base + i * _FUND_MS, "fundingRate": "0.0001",
             "symbol": "BTCUSDT"}
            for i in range(n)
        ]

    def test_second_identical_closed_window_hits_memo(self):
        calls = []
        rows = self._rows()

        def counting_server(path, params, symbol, canonical, **kwargs):
            calls.append(params["startTime"])
            return _fake_funding_server(rows)(path, params, symbol, canonical, **kwargs)

        with mock.patch.object(binance, "_http_get", counting_server):
            binance._paginate_history(
                "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000,
                lambda r: r["fundingTime"],
                int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000),
                int(datetime(2024, 1, 15, tzinfo=UTC).timestamp() * 1000),
                "BTCUSDT", "BTCUSDT",
            )
            binance._paginate_history(
                "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000,
                lambda r: r["fundingTime"],
                int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000),
                int(datetime(2024, 1, 15, tzinfo=UTC).timestamp() * 1000),
                "BTCUSDT", "BTCUSDT",
            )
        self.assertEqual(len(calls), 1, "closed window must be served from the memo")

    def test_window_touching_present_is_not_cached(self):
        calls = []
        rows = self._rows()
        now_ms = int(datetime.now(UTC).timestamp() * 1000)

        def counting_server(path, params, symbol, canonical, **kwargs):
            calls.append(params["startTime"])
            return _fake_funding_server(rows)(path, params, symbol, canonical, **kwargs)

        for _ in range(2):
            with mock.patch.object(binance, "_http_get", counting_server):
                binance._paginate_history(
                    "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000,
                    lambda r: r["fundingTime"], now_ms - _DAY_MS, now_ms,
                    "BTCUSDT", "BTCUSDT",
                )
        self.assertEqual(len(calls), 2, "a still-open window must never be memoized")

    def test_different_pit_end_is_a_different_key(self):
        calls = []
        rows = self._rows()

        def counting_server(path, params, symbol, canonical, **kwargs):
            calls.append(params["endTime"])
            return _fake_funding_server(rows)(path, params, symbol, canonical, **kwargs)

        base = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
        for end in (base + 10 * _DAY_MS, base + 12 * _DAY_MS):
            with mock.patch.object(binance, "_http_get", counting_server):
                binance._paginate_history(
                    "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000,
                    lambda r: r["fundingTime"], base, end, "BTCUSDT", "BTCUSDT",
                )
        self.assertEqual(len(calls), 2, "a clamped window must not alias a wider one")

    def test_memo_returns_a_copy(self):
        rows = self._rows()
        base = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
        end = base + 5 * _DAY_MS
        args = (
            "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000,
            lambda r: r["fundingTime"], base, end, "BTCUSDT", "BTCUSDT",
        )
        with mock.patch.object(binance, "_http_get", _fake_funding_server(rows)):
            first = binance._paginate_history(*args)
        first.append({"poison": True})  # caller mutation must not leak
        with mock.patch.object(
            binance, "_http_get",
            lambda *a, **k: self.fail("memo hit must serve the copy"),
        ):
            second = binance._paginate_history(*args)
        self.assertNotIn({"poison": True}, second)


class TestKlinesPagination(unittest.TestCase):
    def test_long_range_is_not_truncated(self):
        # 2000 daily bars: the old default-limit (500) and even one 1500-page
        # would drop recent rows. Paging must return all 2000, oldest-first.
        base = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
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
        base = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
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
        for prev, cur in zip(d_objs, d_objs[1:], strict=False):
            self.assertEqual((cur - prev).days, 1)

    def test_short_range_single_page_unchanged(self):
        # A range under one page must return exactly its bars (no over-fetch).
        base = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
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
        base = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
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
        base = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
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
        with mock.patch.object(binance, "_http_get", _fake_klines_server([])), \
                self.assertRaises(NoMarketDataError):
            binance.get_binance_klines("BTCUSDT", "2020-01-01", "2020-01-31")


if __name__ == "__main__":
    unittest.main()
