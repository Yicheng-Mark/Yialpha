"""Round-17 shard ``perp_bundle``: contract pins for the perp market bundle.

Scope: the modules this shard owns — the deterministic Binance USDT-M
perpetual bundle (``yialpha/dataflows/perp_bundle.py``), the Binance Square
feed, the perp/spot transport helpers and the Binance indicator tools. Every
test is hermetic: vendor seams are mocked, clocks are injected, and no test
waits on a real timeout or sleeps through a backoff.
"""
from __future__ import annotations

import importlib
import json
import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from yialpha.agents.utils import binance_indicator_tools as ind
from yialpha.dataflows import (
    binance as bnc,
    binance_brackets as bb,
    binance_square as sq,
    perp_bundle as pb,
    quality,
)

# ---- perp_bundle: pure helpers ---------------------------------------------


class TestFmt:
    def test_r17_fmt_adaptive_magnitude_buckets(self) -> None:
        assert pb._fmt(123456.789) == "123,457"
        assert pb._fmt(-250.0) == "-250.00"
        assert pb._fmt(1.23456789) == "1.2346"
        assert pb._fmt(1.23456789, 2) == "1.23"
        assert pb._fmt(0.00041234) == "0.00041234"
        assert pb._fmt(1e-6) == "0.0000010000"
        assert pb._fmt(0) == "0.0000000000"
        # Bucket edges are inclusive.
        assert pb._fmt(1000) == "1,000"
        assert pb._fmt(100) == "100.00"
        assert pb._fmt(1) == "1.0000"
        assert pb._fmt(1e-4) == "0.00010000"

    def test_r17_fmt_bad_input_is_na_not_zero(self) -> None:
        assert pb._fmt(None) == "n/a"
        assert pb._fmt("not-a-number") == "n/a"
        assert pb._fmt(object()) == "n/a"
        # A NaN reading must not be rendered as a real quote.
        assert pb._fmt(math.nan) == "nan"


class TestLevels:
    def test_r17_levels_skips_malformed_rows(self) -> None:
        rows: list[Any] = [
            ["67000.10", "1.5"],
            [67001, 2],
            [67002],  # too short
            "67003,3",  # not a list
            ["abc", 1],  # non-numeric price
            ["67004", None],  # non-numeric qty
            None,
            [1, 2, 3, 4],  # extra fields are tolerated
        ]
        assert pb._levels(rows) == [(67000.10, 1.5), (67001.0, 2.0), (1.0, 2.0)]
        assert pb._levels([]) == []


class TestLastCloseHint:
    def test_r17_last_close_hint_pairs_tail_date_with_last_close(self, monkeypatch) -> None:
        calls: list[tuple[Any, ...]] = []
        index = pd.to_datetime(["2024-03-06", "2024-03-07", "2024-03-08"])
        frame = pd.DataFrame(
            {pb.KLINE_CLOSE_COLUMN: [100.0, float("nan"), 102.5]}, index=index
        )

        def fake_frame(symbol, start, end, interval, vendor, field, *a, **kw):
            calls.append((symbol, start, end, interval, vendor, field))
            return frame

        monkeypatch.setattr(pb, "binance_klines_frame", fake_frame)
        tail, close = pb._last_close_hint("BTCUSDT", "2024-03-10")
        # The NaN bar is dropped BEFORE the tail is read: the date and the
        # close come from the same (last valid) bar.
        assert (tail, close) == (index[2], 102.5)
        assert calls == [("BTCUSDT", calls[0][1], "2024-03-10", "1d", "binance_perp", "last")]
        # The kline window never reaches past the bundle's end date.
        assert str(calls[0][1]) < "2024-03-10"

    def test_r17_last_close_hint_none_on_empty_or_failure(self, monkeypatch) -> None:
        empty = pd.DataFrame({pb.KLINE_CLOSE_COLUMN: pd.Series(dtype=float)})
        monkeypatch.setattr(pb, "binance_klines_frame", lambda *a, **kw: empty)
        assert pb._last_close_hint("BTCUSDT", "2024-03-10") is None

        def boom(*a: Any, **kw: Any) -> Any:
            raise RuntimeError("kline leg down")

        monkeypatch.setattr(pb, "binance_klines_frame", boom)
        assert pb._last_close_hint("BTCUSDT", "2024-03-10") is None


# ---- binance_square: pure helpers ------------------------------------------


class TestSquareCoercions:
    def test_r17_post_ts_seconds_or_none(self) -> None:
        assert sq._post_ts({"date": 1700000000.75}) == 1700000000.75
        assert sq._post_ts({"date": "1700000000"}) == 1700000000.0
        assert sq._post_ts({"date": None}) is None
        assert sq._post_ts({}) is None
        assert sq._post_ts({"date": "soon"}) is None
        # A non-positive or NaN stamp is "unknown", not epoch 1970.
        assert sq._post_ts({"date": 0}) is None
        assert sq._post_ts({"date": -5}) is None
        assert sq._post_ts({"date": math.nan}) is None

    def test_r17_int_or_zero_coerces_json_counts(self) -> None:
        assert sq._int_or_zero(7) == 7
        assert sq._int_or_zero("12") == 12
        assert sq._int_or_zero(None) == 0
        assert sq._int_or_zero("many") == 0
        assert sq._int_or_zero(3.9) == 3


# ---- binance_indicator_tools: pure helpers ---------------------------------


class TestFmtVal:
    def test_r17_fmt_val_adaptive_magnitude_buckets(self) -> None:
        assert ind._fmt_val(123456.789) == "123,456.79"
        assert ind._fmt_val(-250.0) == "-250.00"
        assert ind._fmt_val(2.5) == "2.5000"
        assert ind._fmt_val(0.5) == "0.500000"
        assert ind._fmt_val(0.00001234) == "0.00001234"
        assert ind._fmt_val(1e-7) == "0.0000001000"
        assert ind._fmt_val(0) == "0.0000000000"
        # Bucket edges are inclusive.
        assert ind._fmt_val(100) == "100.00"
        assert ind._fmt_val(1) == "1.0000"
        assert ind._fmt_val(0.01) == "0.010000"
        assert ind._fmt_val(1e-6) == "0.00000100"

    def test_r17_fmt_val_bad_input_is_na(self) -> None:
        assert ind._fmt_val(None) == "N/A"
        assert ind._fmt_val("not-a-number") == "N/A"
        assert ind._fmt_val(math.nan) == "N/A"
        assert ind._fmt_val(pd.NA) == "N/A"


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        (1700000000, "2023-11-14 22:13"),
        (1700000000.99, "2023-11-14 22:13"),  # float seconds truncate
        ("1700000000", "2023-11-14 22:13"),
    ],
)
def test_r17_utc_date_formats_in_utc(ts: object, expected: str) -> None:
    assert sq._utc_date(ts) == expected


# ===========================================================================
# perp_bundle — the legs (each one fails soft on its own)
# ===========================================================================
_DAY_MS = 86_400_000


def _ms(day: str, clock: str = "00:00:00") -> int:
    """Epoch ms of a UTC instant."""
    moment = datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return int(moment.timestamp() * 1000)


def _fake_http(monkeypatch: pytest.MonkeyPatch, routes: dict) -> list[tuple[str, dict]]:
    """Replace ``perp_bundle._http_get``.

    ``routes`` maps an endpoint path to its payload, to an exception to raise,
    or to a callable returning either. Returns the call log ``[(path, params)]``.
    """
    calls: list[tuple[str, dict]] = []

    def fake(path, params, symbol, canonical, **kwargs):
        calls.append((path, dict(params)))
        if path not in routes:
            raise RuntimeError(f"unexpected request to {path}")
        answer = routes[path]
        if callable(answer) and not isinstance(answer, BaseException):
            answer = answer()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(pb, "_http_get", fake)
    return calls


def _live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every analysis date read as a live run for the bundle."""
    monkeypatch.setattr(pb, "is_historical_date", lambda date: False)


def _days(n: int, start: str = "2024-03-01") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, periods=n, freq="D")]


def _closes_frame(closes: list[float], days: list[str]) -> pd.DataFrame:
    """The kline shape ``binance_klines_frame`` ships: a Date-named index + OHLCV."""
    return pd.DataFrame(
        {
            "Open": [c * 0.999 for c in closes],
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": [float(c) for c in closes],
            "Volume": [1_000.0] * len(closes),
        },
        index=pd.DatetimeIndex(pd.to_datetime(days), name="Date"),
    )


_LSR_PATHS = dict(pb._LSR_SERIES)
#: ``yialpha.risk`` re-exports a FUNCTION named ``atr_stop``; the module is
#: fetched by name so the helper the bundle imports lazily can be patched at source.
_ATR_STOP = importlib.import_module("yialpha.risk.atr_stop")


class TestPerpBundleLegs:
    """One class per leg would repeat the fixture four times over; what every
    test here pins is the same contract: a leg answers with its own status and
    never takes its siblings down."""

    def test_r17_prices_atr_failure_keeps_the_close_and_records_the_error(self, monkeypatch):
        """ATR is an enrichment of the ``last`` leg: when its engine raises, the
        close, the changes and the core verdict survive and the error text rides
        along in the leg instead of being swallowed."""
        frame = _closes_frame([100.0 + i for i in range(10)], _days(10))
        monkeypatch.setattr(pb, "binance_klines_frame", lambda *a, **k: frame)

        def no_ranges(ohlcv: pd.DataFrame) -> tuple[float, float]:
            raise RuntimeError("no ranges")

        monkeypatch.setattr(_ATR_STOP, "latest_atr_from_frame", no_ranges)
        out = pb._fetch_prices("BTCUSDT", "2024-03-10")
        assert out["status"] == "ok" and out["core_complete"] is True
        assert out["last"] == {
            "status": "ok",
            "close": 109.0,
            "chg_1d": 109.0 / 108.0 - 1.0,
            "chg_7d": 109.0 / 102.0 - 1.0,
            "atr_14": None,
            "atr_error": "RuntimeError: no ranges",
        }
        assert out["coverage"] == {"rows": 10, "start": "2024-03-01", "end": "2024-03-10"}
        block = pb.render_perp_bundle_block(
            {"symbol": "BTCUSDT", "as_of": "2024-03-10", "live_run": True, "prices": out}
        )
        assert "ATR14" not in block  # nothing is invented for it ...
        assert "Component availability" not in block  # ... and nothing is said about it

    def test_r17_funding_without_settlements_is_unavailable_and_names_the_window(
        self, monkeypatch
    ):
        _live(monkeypatch)
        calls = _fake_http(monkeypatch, {"/fapi/v1/fundingRate": []})
        out = pb._fetch_funding("BTC-USDT", "2024-03-10")
        assert out == {"status": "unavailable", "reason": "no settlements in window"}
        # Seven calendar days, end-inclusive: 03-04 00:00:00 .. 03-10 23:59:59 UTC.
        assert calls == [
            (
                "/fapi/v1/fundingRate",
                {
                    "symbol": "BTCUSDT",
                    "startTime": _ms("2024-03-04"),
                    "endTime": _ms("2024-03-10", "23:59:59"),
                    "limit": 1000,
                },
            )
        ]

    def test_r17_funding_fetch_error_is_unavailable_with_its_cause(self, monkeypatch):
        _live(monkeypatch)
        _fake_http(monkeypatch, {"/fapi/v1/fundingRate": RuntimeError("fapi unreachable")})
        assert pb._fetch_funding("BTCUSDT", "2024-03-10") == {
            "status": "unavailable", "reason": "RuntimeError: fapi unreachable",
        }

    def test_r17_funding_one_unreadable_rate_fails_the_leg_instead_of_shrinking_the_sum(
        self, monkeypatch
    ):
        """A settlement whose rate cannot be read makes the 7d sum unknown: the
        leg degrades as a whole and never reports the readable subset as the sum."""
        _live(monkeypatch)
        _fake_http(
            monkeypatch,
            {"/fapi/v1/fundingRate": [
                {"fundingRate": "0.0001"}, {"fundingRate": "not-a-rate"}, {"fundingRate": "-0.0002"},
            ]},
        )
        out = pb._fetch_funding("BTCUSDT", "2024-03-10")
        assert out["status"] == "unavailable"
        assert out["reason"].startswith("ValueError: could not convert string to float")

    def test_r17_oi_series_drops_unusable_rows_sorts_them_and_trims_the_future(self, monkeypatch):
        rows = [
            {"timestamp": _ms("2024-03-08"), "sumOpenInterest": "3.0"},
            "not a dict",
            {"sumOpenInterest": "9.0"},
            {"timestamp": _ms("2024-03-06"), "sumOpenInterest": "many"},
            {"timestamp": "soon", "sumOpenInterest": "1.0"},
            {"timestamp": _ms("2024-03-07"), "sumOpenInterest": "2.0"},
            {"timestamp": _ms("2024-03-09"), "sumOpenInterest": "99.0"},
            None,
        ]
        calls = _fake_http(monkeypatch, {"/futures/data/openInterestHist": rows})
        # The retention clamp reads the wall clock: pin it so startTime is exact.
        monkeypatch.setattr(bnc, "_now_ms", lambda: _ms("2024-03-08", "12:00:00"))
        series = pb._oi_series("BTCUSDT", "2024-03-08", 3)
        # 03-09 is after the analysis date: kept out by the PIT trim, not by luck.
        assert series == [(_ms("2024-03-07"), 2.0), (_ms("2024-03-08"), 3.0)]
        ((path, params),) = calls
        assert path == "/futures/data/openInterestHist"
        assert (params["symbol"], params["period"], params["limit"]) == ("BTCUSDT", "1d", 30)
        assert params["endTime"] == _ms("2024-03-08", "23:59:59")
        # Three days back is inside the endpoint's 30-day retention: the head is served as asked.
        assert params["startTime"] == _ms("2024-03-05")
        # A replay older than the retention gets its head clamped into the
        # retained tail (never a startTime the server would reject) while the
        # end stays at the analysis date.
        calls.clear()
        pb._oi_series("BTCUSDT", "2024-01-10", 30)
        ((path, old_params),) = calls
        assert old_params["startTime"] == (
            _ms("2024-03-08", "12:00:00") - bnc._FUTURES_DATA_RETENTION_DAYS * _DAY_MS
        )
        assert old_params["endTime"] == _ms("2024-01-10", "23:59:59")

    def test_r17_open_interest_fetch_error_and_empty_window_are_unavailable(self, monkeypatch):
        path = "/futures/data/openInterestHist"
        _fake_http(monkeypatch, {path: RuntimeError("HTTP 400 -1130")})
        assert pb._fetch_open_interest("BTCUSDT", "2024-03-08") == {
            "status": "unavailable", "reason": "RuntimeError: HTTP 400 -1130",
        }
        _fake_http(monkeypatch, {path: "not-a-list"})
        assert pb._fetch_open_interest("BTCUSDT", "2024-03-08") == {
            "status": "unavailable", "reason": "no rows in window",
        }

    def test_r17_lsr_seven_day_change_reads_eight_rows_back_and_trims_the_future(self, monkeypatch):
        days = _days(10, "2024-02-28")
        rows = [{"timestamp": _ms(d), "longShortRatio": f"{i + 1}.5"} for i, d in enumerate(days)]
        rows.append({"timestamp": _ms("2024-03-09"), "longShortRatio": "99"})
        _fake_http(monkeypatch, dict.fromkeys(_LSR_PATHS.values(), rows))
        out = pb._fetch_lsr("BTCUSDT", "2024-03-08")
        assert out["status"] == "ok"
        for key in _LSR_PATHS:
            assert out[key] == {"status": "ok", "latest": 10.5, "chg_7d": 10.5 - 3.5}
        assert out["cross_vantage_spread"] == 0.0

    def test_r17_lsr_each_vantage_fails_soft_on_its_own(self, monkeypatch):
        _fake_http(
            monkeypatch,
            {
                _LSR_PATHS["top_account"]: [
                    {"timestamp": _ms("2024-03-08"), "longShortRatio": "2.5"}
                ],
                _LSR_PATHS["top_position"]: [],
                _LSR_PATHS["global_account"]: RuntimeError("HTTP 500"),
            },
        )
        out = pb._fetch_lsr("BTCUSDT", "2024-03-08")
        assert out == {
            "status": "ok",  # one vantage answered: the component is partly served
            "top_account": {"status": "ok", "latest": 2.5},
            "top_position": {"status": "unavailable", "reason": "no rows"},
            "global_account": {"status": "unavailable", "reason": "RuntimeError: HTTP 500"},
        }

    def test_r17_lsr_all_vantages_failed_marks_the_component_unavailable(self, monkeypatch):
        _fake_http(monkeypatch, dict.fromkeys(_LSR_PATHS.values(), []))
        out = pb._fetch_lsr("BTCUSDT", "2024-03-08")
        assert out["status"] == "unavailable"
        assert pb._component_footer({"long_short": out}) == [
            "long_short: unavailable",
            "long_short.top_account: unavailable (no rows)",
            "long_short.top_position: unavailable (no rows)",
            "long_short.global_account: unavailable (no rows)",
        ]

    def test_r17_taker_row_after_the_analysis_date_is_not_served(self, monkeypatch):
        path = "/futures/data/takerlongshortRatio"
        _fake_http(
            monkeypatch, {path: [{"timestamp": _ms("2024-03-09"), "buySellRatio": "1.9"}]}
        )
        assert pb._fetch_taker("BTCUSDT", "2024-03-08") == {
            "status": "unavailable", "reason": "no rows in window",
        }

    def test_r17_taker_fetch_error_is_unavailable_with_its_cause(self, monkeypatch):
        path = "/futures/data/takerlongshortRatio"
        _fake_http(monkeypatch, {path: RuntimeError("HTTP 429")})
        assert pb._fetch_taker("BTCUSDT", "2024-03-08") == {
            "status": "unavailable", "reason": "RuntimeError: HTTP 429",
        }

    def test_r17_premium_snapshot_without_a_mark_price_is_unavailable(self, monkeypatch):
        _fake_http(monkeypatch, {"/fapi/v1/premiumIndex": {"symbol": "BTCUSDT", "indexPrice": "100"}})
        assert pb._fetch_premium_snapshot("BTCUSDT") == {
            "status": "unavailable", "reason": "ValueError: premiumIndex returned no markPrice",
        }

    def test_r17_premium_snapshot_transport_error_is_unavailable(self, monkeypatch):
        _fake_http(monkeypatch, {"/fapi/v1/premiumIndex": RuntimeError("fapi down")})
        assert pb._fetch_premium_snapshot("BTCUSDT") == {
            "status": "unavailable", "reason": "RuntimeError: fapi down",
        }

    @pytest.mark.parametrize(
        "payload",
        [
            {"bids": [["100", "1"]], "asks": []},
            {"bids": [], "asks": [["101", "1"]]},
            "not-a-dict",
        ],
        ids=["no-asks", "no-bids", "not-a-dict"],
    )
    def test_r17_depth_without_both_sides_is_unavailable(self, monkeypatch, payload):
        _fake_http(monkeypatch, {"/fapi/v1/depth": payload})
        assert pb._fetch_depth_bands("BTCUSDT") == {
            "status": "unavailable", "reason": "ValueError: depth returned no levels",
        }

    def test_r17_depth_transport_error_is_unavailable(self, monkeypatch):
        _fake_http(monkeypatch, {"/fapi/v1/depth": RuntimeError("depth down")})
        assert pb._fetch_depth_bands("BTCUSDT") == {
            "status": "unavailable", "reason": "RuntimeError: depth down",
        }

    def test_r17_adl_without_quantiles_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            bb, "signed_fapi_get",
            lambda path, params, symbol, canonical: [{"symbol": "BTCUSDT"}],
        )
        assert pb._fetch_adl("BTCUSDT") == {
            "status": "unavailable", "reason": "ValueError: adlQuantile returned no quantiles",
        }

    @pytest.mark.parametrize(
        ("quantiles", "expected"),
        [
            ({"LONG": "2", "SHORT": "n/a"}, {"status": "ok", "long": 2, "short": None}),
            ({"LONG": None}, {"status": "ok", "long": None, "short": None}),
            ({}, {"status": "ok", "long": None, "short": None}),
        ],
        ids=["short-unreadable", "long-null", "empty"],
    )
    def test_r17_adl_unreadable_side_is_none_never_zero(self, monkeypatch, quantiles, expected):
        """0 is a real quantile ("far from auto-deleveraging"): an unreadable
        side must not collapse into it."""
        monkeypatch.setattr(
            bb, "signed_fapi_get",
            lambda path, params, symbol, canonical: [
                {"symbol": "BTCUSDT", "adlQuantile": quantiles}
            ],
        )
        assert pb._fetch_adl("BTCUSDT") == expected

    @pytest.mark.parametrize("hint", [None, (pd.Timestamp("2024-03-09"), 0.0), ("2024-03-09", -5.0)])
    def test_r17_spot_basis_without_a_usable_perp_close(self, monkeypatch, hint):
        monkeypatch.setattr(pb, "stock_perp_underlying", lambda symbol: None)
        assert pb._fetch_spot_basis("BTCUSDT", "2024-03-10", hint) == {
            "status": "unavailable", "reason": "no perp close to compare",
        }

    @pytest.mark.parametrize("spot_close", [0.0, -1.0])
    def test_r17_spot_basis_non_positive_spot_close_is_unavailable(
        self, monkeypatch, spot_close
    ):
        monkeypatch.setattr(pb, "stock_perp_underlying", lambda symbol: None)
        frame = _closes_frame([100.0, spot_close], ["2024-03-09", "2024-03-10"])
        monkeypatch.setattr(pb, "binance_klines_frame", lambda *a, **k: frame)
        out = pb._fetch_spot_basis("BTCUSDT", "2024-03-10", (pd.Timestamp("2024-03-10"), 105.0))
        assert out == {"status": "unavailable", "reason": "ValueError: spot close non-positive"}

    def test_r17_spot_basis_nan_spot_close_degrades_instead_of_pairing(self, monkeypatch):
        """A NaN spot close at the perp's tail date is no bar to pair with: the
        basis degrades with the lag reason rather than dividing by NaN."""
        monkeypatch.setattr(pb, "stock_perp_underlying", lambda symbol: None)
        frame = _closes_frame([100.0, float("nan")], ["2024-03-09", "2024-03-10"])
        monkeypatch.setattr(pb, "binance_klines_frame", lambda *a, **k: frame)
        out = pb._fetch_spot_basis("BTCUSDT", "2024-03-10", (pd.Timestamp("2024-03-10"), 105.0))
        assert out == {
            "status": "unavailable",
            "reason": "ValueError: spot tail lags perp tail: no spot bar at 2024-03-10 00:00:00",
        }


class TestPerpBundleFooter:
    def test_r17_footer_names_failed_nested_legs_even_under_an_ok_component(self):
        """The footer exists so that "a missing enrichment never masquerades
        as a calm market". A nested leg that failed is such a missing
        enrichment whether or not its parent still answers: one LSR vantage
        served and two failed, or a price book whose index leg is gone.

        The nested notes used to be appended only AFTER a note for the parent,
        i.e. only when the parent itself was not ok -- and ``long_short`` is
        "ok" whenever a single vantage answers, ``prices`` whenever last and
        mark do. The block then showed the surviving reading and said nothing
        about the two that were missing."""
        bundle = {
            "prices": {
                "status": pb.STATUS_OK,
                "core_complete": True,
                "last": {"status": pb.STATUS_OK, "close": 100.0},
                "mark": {"status": pb.STATUS_OK, "close": 100.0},
                "index": {"status": pb.STATUS_UNAVAILABLE, "reason": "ValueError: no rows"},
            },
            "long_short": {
                "status": pb.STATUS_OK,
                "top_account": {"status": pb.STATUS_OK, "latest": 2.5},
                "top_position": {"status": pb.STATUS_UNAVAILABLE, "reason": "no rows"},
                "global_account": {
                    "status": pb.STATUS_UNAVAILABLE, "reason": "RuntimeError: HTTP 500",
                },
            },
        }
        assert pb._component_footer(bundle) == [
            "prices.index: unavailable (ValueError: no rows)",
            "long_short.top_position: unavailable (no rows)",
            "long_short.global_account: unavailable (RuntimeError: HTTP 500)",
        ]

    def test_r17_lsr_block_discloses_the_vantages_that_did_not_answer(self, monkeypatch):
        _fake_http(
            monkeypatch,
            {
                _LSR_PATHS["top_account"]: [
                    {"timestamp": _ms("2024-03-08"), "longShortRatio": "2.5"}
                ],
                _LSR_PATHS["top_position"]: [],
                _LSR_PATHS["global_account"]: RuntimeError("HTTP 500"),
            },
        )
        long_short = pb._fetch_lsr("BTCUSDT", "2024-03-08")
        block = pb.render_perp_bundle_block(
            {"symbol": "BTCUSDT", "as_of": "2024-03-08", "live_run": True, "long_short": long_short}
        )
        lines = block.splitlines()
        assert (
            "- **Long/short (acct ratio)**: topAcct 2.50 (>1 = that crowd is long-heavy)"
            in lines
        )
        assert (
            "- **Component availability**: long_short.top_position: unavailable (no rows); "
            "long_short.global_account: unavailable (RuntimeError: HTTP 500)" in lines
        )

    def test_r17_footer_keys_filter_keeps_the_other_halfs_failures_out(self):
        bundle = {
            "adl": {"status": pb.STATUS_UNAVAILABLE, "reason": "auth required"},
            "funding": {"status": pb.STATUS_SKIPPED_LIVE_ONLY},
        }
        assert pb._component_footer(bundle, pb._PRICE_SECTION_KEYS) == []
        assert pb._component_footer(bundle, pb._POSITIONING_SECTION_KEYS) == [
            "adl: unavailable (auth required)",  # in bundle order
            "funding: skipped_live_only",
        ]

    def test_r17_footer_lines_name_the_historical_policy_after_the_availability_notes(self):
        bundle = {
            "live_run": False,
            "funding": {"status": pb.STATUS_SKIPPED_LIVE_ONLY},
            "adl": {"status": pb.STATUS_UNAVAILABLE, "reason": "auth"},
        }
        assert pb._footer_lines(bundle, None) == [
            "- **Component availability**: funding: skipped_live_only; adl: unavailable (auth)",
            "- (Historical run: live-only components — funding/premium/depth/ADL — are skipped "
            "by PIT policy; REST positioning retains 30 days.)",
        ]
        assert pb._footer_lines({"live_run": True}, None) == []


class TestRenderPriceHalf:
    def test_r17_render_split_block_carries_only_the_price_half(self):
        bundle = {
            "symbol": "BTCUSDT",
            "as_of": "2024-03-10",
            "live_run": True,
            "prices": {
                "status": pb.STATUS_OK,
                "core_complete": True,
                "last": {"status": pb.STATUS_OK, "close": 109.0, "chg_1d": 0.0093, "chg_7d": 0.0686},
                "mark": {"status": pb.STATUS_OK, "close": 108.0},
                "index": {"status": pb.STATUS_UNAVAILABLE, "reason": "ValueError: no rows"},
            },
            "open_interest": {
                "status": pb.STATUS_OK, "latest": 1000.0, "percentile": 90.0, "days": 30,
            },
            "adl": {"status": pb.STATUS_UNAVAILABLE, "reason": "auth required"},
        }
        lines = pb.render_perp_bundle_block(bundle, positioning_split=True).splitlines()
        assert lines == [
            "### Perp Market Bundle — BTCUSDT (deterministic prefetch, as of 2024-03-10)",
            "- **Price bases**: last 109.00 (1d +0.93%, 7d +6.86%); mark 108.00; index n/a",
            "- (Positioning split: funding / open interest / long-short / taker / depth / ADL / "
            "spot-perp basis are rendered by the Positioning analyst's block, not here.)",
            "- **Component availability**: prices.index: unavailable (ValueError: no rows)",
        ]


# ===========================================================================
# binance_square — parsers and renderer
# ===========================================================================
class TestSquareParsers:
    def test_r17_validated_feed_url_passes_public_hosts_through(self):
        for url in (
            "https://www.binance.com/bapi/x?y=1",
            "http://example.org./feed",
            "https://8.8.8.8/feed",
        ):
            assert sq._validated_feed_url(url) == url

    def test_r17_posts_from_response_keeps_post_cards_only(self):
        data = {
            "success": True,
            "data": {"vos": [
                {"cardType": "BUZZ_SHORT", "id": 1},
                {"cardType": "AD", "id": 2},
                "junk",
                {"id": 3},
                None,
                {"cardType": "BUZZ_LONG", "id": 4},
            ]},
        }
        posts = [{"cardType": "BUZZ_SHORT", "id": 1}, {"cardType": "BUZZ_LONG", "id": 4}]
        assert sq._posts_from_response(data) == posts
        assert sq._parse_posts(json.dumps(data).encode("utf-8")) == posts

    def test_r17_cached_scene_hands_back_posts_and_the_stored_fetch_time(self, monkeypatch):
        wrapper = {
            "fetched_at_ms": 1_710_036_000_000,
            "response": {"success": True, "data": {"vos": [{"cardType": "BUZZ_SHORT", "id": 7}]}},
        }
        monkeypatch.setattr(sq, "cached_or_fetch", lambda *a, **k: json.dumps(wrapper))
        assert sq._cached_scene("web-trending") == (
            [{"cardType": "BUZZ_SHORT", "id": 7}], 1_710_036_000_000.0,
        )

    def test_r17_cached_scene_rejects_a_wrapper_without_a_response(self, monkeypatch):
        monkeypatch.setattr(sq, "cached_or_fetch", lambda *a, **k: json.dumps({"fetched_at_ms": 1.0}))
        with pytest.raises(ValueError, match="cache wrapper has no response payload"):
            sq._cached_scene("web-trending")

    @pytest.mark.parametrize(
        ("post", "expected"),
        [
            ({"content": "  body  ", "title": "t", "subTitle": "s"}, "body"),
            ({"title": " Title ", "subTitle": " Sub "}, "Title — Sub"),
            ({"title": "Only title"}, "Only title"),
            ({"subTitle": "Only subtitle"}, "Only subtitle"),
            ({"content": "   ", "title": "", "subTitle": None}, ""),
            ({}, ""),
        ],
    )
    def test_r17_post_body_falls_back_to_title_then_subtitle(self, post, expected):
        assert sq._post_body(post) == expected

    @pytest.mark.parametrize(
        "bad",
        [None, "soon", float("nan"), float("inf"), 10 ** 18, -(10 ** 18), {}],
        ids=["none", "text", "nan", "inf", "far-future", "far-past", "dict"],
    )
    def test_r17_utc_date_unreadable_stamps_render_as_a_question_mark(self, bad):
        assert sq._utc_date(bad) == "?"

    def test_r17_hot_coins_line_without_structured_tags_says_so(self):
        assert sq._hot_coins_line([]) == "Feed-wide hot coins: none tagged in the scanned posts."
        # A plain-text cashtag is matched for the target, but it is not a
        # structured tag the feed-wide tally counts.
        assert sq._hot_coins_line([{"content": "$BTC to the moon"}]) == (
            "Feed-wide hot coins: none tagged in the scanned posts."
        )


def _square_post(content: str, date_s: float, code: str = "BTC") -> dict:
    return {
        "cardType": "BUZZ_SHORT", "content": content, "date": date_s, "viewCount": 5,
        "likeCount": 1, "authorName": "ali", "tradingPairsV2": [{"code": code}],
    }


class TestSquareRenderBlock:
    _NOW = datetime(2024, 3, 10, 2, 0, tzinfo=UTC)

    def _now(self, monkeypatch: pytest.MonkeyPatch) -> float:
        now_s = self._NOW.timestamp()
        monkeypatch.setattr(sq.time, "time", lambda: now_s)
        return now_s

    def test_r17_render_block_marks_a_wholly_stale_snapshot_with_its_age(self, monkeypatch):
        """Freshness is measured against the fetch time stored with the payload;
        a snapshot older than the 5-minute TTL with no per-scene times to
        blame is labelled STALE, never fresh."""
        now_s = self._now(monkeypatch)
        post = _square_post("BTC looks strong", now_s - 3_600)
        block = sq._render_block(
            [post], ("BTC",), "BTC", (now_s - 600.0) * 1000.0, "2024-03-10", []
        )
        assert block.splitlines()[0] == (
            "Binance Square recommended feed (global crypto social) — 1 posts scanned "
            "(web-homepage, web-trending), fetched 2024-03-10 01:50 UTC "
            "(STALE — live fetch failed; snapshot is 10 min old). "
            "Posts mentioning BTC: 1 (1 of 1 within the last 3 days (as of 2024-03-10); "
            "matching posts span 2024-03-10 01:00 → 2024-03-10 01:00 UTC)."
        )

    def test_r17_render_block_truncates_a_long_body_and_dates_the_entry(self, monkeypatch):
        now_s = self._now(monkeypatch)
        post = _square_post("x" * 300, now_s - 3_600)
        lines = sq._render_block(
            [post], ("BTC",), "BTC", (now_s - 60.0) * 1000.0, "2024-03-10", []
        ).splitlines()
        assert "(fresh)" in lines[0]
        assert lines[2] == f"- [2024-03-10 01:00 UTC · ali · views 5 · likes 1] {'x' * 280}…"


# ===========================================================================
# binance — the pagination guard and Retry-After parsing
# ===========================================================================
class TestPaginateHistory:
    def test_r17_pagination_stops_when_a_page_makes_no_forward_progress(self, monkeypatch):
        """A FULL page whose last row sits at or before the cursor would be
        requested again forever if the walk trusted it: the cursor has to
        advance, or the walk ends."""
        monkeypatch.setattr(bnc, "_HISTORY_MEMO", {})
        start = _ms("2024-01-02")
        end = _ms("2024-01-31", "23:59:59")
        page = [{"fundingTime": start - i, "fundingRate": "0.0001"} for i in range(1000)]
        asked: list[int] = []

        def stuck_server(path, params, symbol, canonical, **kwargs):
            asked.append(params["startTime"])
            return list(page)

        monkeypatch.setattr(bnc, "_http_get", stuck_server)
        out = bnc._paginate_history(
            "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000,
            lambda r: r["fundingTime"], start, end, "BTCUSDT", "BTCUSDT",
        )
        assert len(out) == 1000 and out[0]["fundingTime"] == start
        assert asked == [start]  # one request, then the guard ended the walk


class TestParseRetryAfter:
    @pytest.mark.parametrize("raw", ["Wed, 21 Oct 2015 07:28:00 GMT", "1.5", "soon"])
    def test_r17_retry_after_unparseable_is_none(self, raw):
        """Only a whole number of seconds is honoured: an HTTP-date or a
        fraction falls back to the caller's immediate-raise path."""
        assert bnc._parse_retry_after({"Retry-After": raw}) is None

    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            (None, None),
            ({}, None),
            ({"Retry-After": ""}, None),
            ({"Retry-After": "60"}, 60),
            ({"retry-after": "12"}, 12),
            ({"Retry-After": "0"}, 0),
            ({"Retry-After": "-5"}, 0),
            ({"Retry-After": 7}, 7),
        ],
    )
    def test_r17_retry_after_parses_whole_seconds_never_negative(self, headers, expected):
        assert bnc._parse_retry_after(headers) == expected


# ===========================================================================
# binance_indicator_tools — _indicators_core
# ===========================================================================
class TestIndicatorsCore:
    @pytest.fixture()
    def klines(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
        """60 daily bars served to the indicator tool; returns the call log."""
        frame = _closes_frame([100.0 + 0.5 * i for i in range(60)], _days(60))
        calls: list[tuple] = []

        def fetch(symbol, start, end, interval="1d", venue="binance_perp"):
            calls.append((symbol, start, end, interval, venue))
            return frame

        monkeypatch.setattr(ind, "binance_klines_frame", fetch)
        monkeypatch.setattr(ind, "stock_perp_underlying", lambda symbol: None)
        return calls

    def test_r17_indicators_core_rejects_an_unknown_venue_before_any_fetch(self, klines):
        out = ind._indicators_core("BTCUSDT", "2024-03-10", 5, "futures", "rsi")
        assert out == "ERROR: venue must be 'perp' or 'spot'."
        assert klines == []

    @pytest.mark.parametrize("broken", ["rvol_20", "rsi"])
    def test_r17_indicators_core_skips_and_names_an_indicator_it_cannot_compute(
        self, klines, monkeypatch, broken
    ):
        """One indicator failing never takes the table down or gets zero-filled:
        it is left out of the columns and named on a Skipped line."""
        if broken == "rvol_20":
            monkeypatch.setattr(ind, "compute_derived", lambda data, name, periods_per_year=252: None)
        else:

            def cannot(sdf, name: str):
                raise RuntimeError("boom")

            monkeypatch.setattr(ind, "compute_indicator", cannot)
        kept = "rsi" if broken == "rvol_20" else "rvol_20"
        lines = ind._indicators_core("BTCUSDT", "2024-03-10", 3, "perp", "rsi,rvol_20").splitlines()
        assert lines[2] == f"| Date | Close | {kept} |"
        assert lines[-2] == f"(Skipped — not computable on this window: {broken}.)"
        assert broken not in lines[2]
        assert quality.snapshot_quality() == []  # the other indicator computed

    def test_r17_indicators_core_records_the_ledger_when_nothing_computes(self, klines, monkeypatch):
        def cannot(sdf, name: str):
            raise RuntimeError("boom")

        monkeypatch.setattr(ind, "compute_indicator", cannot)
        out = ind._indicators_core(
            "BTCUSDT", "2024-03-10", 5, "spot", "rsi", method_name="get_binance_spot_indicators"
        )
        assert out == (
            "DATA_UNAVAILABLE: none of ['rsi'] could be computed on Binance klines "
            "for 'BTCUSDT'."
        )
        # The sentinel carries the identity of the tool that was actually invoked.
        assert quality.snapshot_quality() == [
            {
                "method": "get_binance_spot_indicators",
                "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
                "detail": "BTCUSDT (spot): no requested indicator computable",
                "qualifier": "",
            }
        ]
        warm_up = int(400 * ind._FETCH_MARGIN)
        expected_start = (pd.Timestamp("2024-03-10") - pd.Timedelta(days=warm_up)).strftime("%Y-%m-%d")
        assert klines == [("BTCUSDT", expected_start, "2024-03-10", "1d", "binance_spot")]

    def test_r17_indicators_core_names_an_unknown_indicator_before_any_fetch(self, klines):
        out = ind._indicators_core("BTCUSDT", "2024-03-10", 5, "perp", "rsi,nopenope")
        assert out.startswith("ERROR: unknown indicator name(s) ['nopenope'].")
        assert klines == []


# ---------------------------------------------------------------------------
# perp_bundle — the core price legs and an unusable tail close
# ---------------------------------------------------------------------------
def _price_frames(last: list[float], days: list[str]) -> dict[str, pd.DataFrame]:
    """One kline frame per price basis: ``last`` as given, ``mark`` / ``index``
    a clean 100..109 ramp."""
    clean = [100.0 + i for i in range(10)]
    return {
        "last": _closes_frame(last, days),
        "mark": _closes_frame(clean, days),
        "index": _closes_frame(clean, days),
    }


def _serve_frames(frames: dict[str, pd.DataFrame]):
    def fetch(symbol, start, end, interval, venue, price_type="last", **kwargs):
        return frames[price_type]

    return fetch


class TestCoreTailClose:
    @pytest.mark.parametrize(
        "bad", [float("nan"), 0.0, float("inf"), -5.0], ids=["nan", "zero", "inf", "negative"]
    )
    def test_r17_prices_unusable_tail_close_fails_the_leg_instead_of_becoming_the_quote(
        self, monkeypatch, bad
    ):
        """``last`` and ``mark`` are the CORE legs: a close that is not a finite
        positive number is no quote at all. It used to be taken as-is -- ``0``
        rendered as "last 0.0000000000 (1d -100.00%, 7d -100.00%)" and a
        -10,000 bps mark basis, NaN / inf as "nan" / "inf" -- with
        ``core_complete=True``, so no core sentinel was recorded and a perp run
        without a usable price book still passed as GOOD-tier.
        """
        days = _days(10)
        frames = _price_frames([100.0 + i for i in range(9)] + [bad], days)
        monkeypatch.setattr(pb, "binance_klines_frame", _serve_frames(frames))
        out = pb._fetch_prices("BTCUSDT", "2024-03-10")
        assert out["last"] == {
            "status": "unavailable",
            "reason": (
                "ValueError: last kline tail close is not a finite positive number: "
                f"{bad!r}"
            ),
        }
        assert out["status"] == "unavailable" and out["core_complete"] is False
        assert "last_vs_mark_bps" not in out and "coverage" not in out
        assert out["mark"] == {"status": "ok", "close": 109.0}
        assert out["index"] == {"status": "ok", "close": 109.0}
        lines = pb.render_perp_bundle_block(
            {"symbol": "BTCUSDT", "as_of": "2024-03-10", "live_run": True, "prices": out}
        ).splitlines()
        # The quote line shows the gap; the footer names the failed leg and why.
        assert lines[1] == "- **Price bases**: last n/a (1d n/a, 7d n/a); mark 109.00; index 109.00"
        assert lines[2].startswith(
            "- **Component availability**: prices: unavailable; prices.last: unavailable "
            "(ValueError: last kline tail close"
        )

    def test_r17_prices_unusable_mark_tail_close_fails_the_mark_leg(self, monkeypatch):
        days = _days(10)
        clean = _closes_frame([100.0 + i for i in range(10)], days)
        frames = {
            "last": clean,
            "mark": _closes_frame([100.0 + i for i in range(9)] + [float("nan")], days),
            "index": clean,
        }
        monkeypatch.setattr(pb, "binance_klines_frame", _serve_frames(frames))
        out = pb._fetch_prices("BTCUSDT", "2024-03-10")
        assert out["mark"]["status"] == "unavailable" and out["core_complete"] is False
        assert (out["last"]["status"], out["last"]["close"]) == ("ok", 109.0)
        assert out["last_vs_index_bps"] == 0.0 and "last_vs_mark_bps" not in out

    def test_r17_bundle_unusable_core_tail_close_records_the_core_sentinel(self, monkeypatch):
        """End to end: the assembler's core verdict reads the leg status, so the
        quality chain sees exactly what it sees when the kline fetch failed."""
        frames = _price_frames([100.0 + i for i in range(9)] + [0.0], _days(10))
        monkeypatch.setattr(pb, "binance_klines_frame", _serve_frames(frames))
        _fake_http(monkeypatch, {})  # every REST leg fails soft; nothing reaches the network
        bundle = pb.fetch_perp_market_bundle("BTCUSDT", "2020-01-01")
        assert bundle["prices"]["core_complete"] is False
        assert [e for e in quality.snapshot_quality() if e["kind"] == quality.KIND_CORE_ERROR] == [
            {
                "method": "get_binance_klines",
                "kind": quality.KIND_CORE_ERROR,
                "detail": "BTCUSDT: perp bundle core price legs incomplete: "
                          "last=missing, mark=ok",
                "qualifier": "",
            }
        ]

    def test_r17_prices_clean_tail_is_unchanged_by_the_close_gate(self, monkeypatch):
        days = _days(10)
        monkeypatch.setattr(
            pb, "binance_klines_frame", _serve_frames(_price_frames([100.0 + i for i in range(10)], days))
        )
        out = pb._fetch_prices("BTCUSDT", "2024-03-10")
        assert out["status"] == "ok" and out["core_complete"] is True
        assert (out["last"]["close"], out["mark"]["close"], out["index"]["close"]) == (
            109.0, 109.0, 109.0,
        )


# ---------------------------------------------------------------------------
# binance — the closed-window memo
# ---------------------------------------------------------------------------
class TestPaginateHistoryMemo:
    def test_r17_pagination_memo_evicts_the_oldest_entry_once_full(self, monkeypatch):
        monkeypatch.setattr(
            bnc, "_HISTORY_MEMO",
            {f"stale-{i}": (0.0, []) for i in range(bnc._HISTORY_MEMO_MAX)},
        )
        rows = [
            {"fundingTime": _ms("2024-01-02") + i * 3_600_000, "fundingRate": "0.0001"}
            for i in range(5)
        ]
        monkeypatch.setattr(
            bnc, "_http_get", lambda path, params, symbol, canonical, **kw: list(rows)
        )
        start, end = _ms("2024-01-02"), _ms("2024-01-02", "23:59:59")
        out = bnc._paginate_history(
            "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"}, 1000,
            lambda r: r["fundingTime"], start, end, "BTCUSDT", "BTCUSDT",
        )
        assert out == rows
        memo = bnc._HISTORY_MEMO
        assert len(memo) == bnc._HISTORY_MEMO_MAX
        assert "stale-0" not in memo and "stale-1" in memo  # first in, first out
        (key,) = [k for k in memo if isinstance(k, tuple)]
        assert key == (
            "/fapi/v1/fundingRate", bnc._FAPI_BASE, (("symbol", "BTCUSDT"),), start, end,
        )
        assert memo[key][1] == rows
        out.append({"fundingTime": 0})  # the caller holds a copy, not the cached rows
        assert len(memo[key][1]) == 5


# ---------------------------------------------------------------------------
# binance_square — response shape and an unparseable as-of day
# ---------------------------------------------------------------------------
class TestSquareResponseShape:
    @pytest.mark.parametrize(
        "data",
        [
            {"success": False, "data": {}},
            {"success": True, "data": {"vos": "not-a-list"}},
            {"success": True, "data": []},
            {"success": True},
            [],
        ],
        ids=["not-success", "vos-not-a-list", "data-not-a-dict", "no-data", "not-a-dict"],
    )
    def test_r17_posts_from_response_rejects_every_other_shape(self, data):
        reason = (
            "feed response not success: " if not (isinstance(data, dict) and data.get("success") is True)
            else "feed response has no data.vos list"
        )
        with pytest.raises(ValueError, match=reason):
            sq._posts_from_response(data)


class TestSquareRenderAsOf:
    def test_r17_render_block_falls_back_to_today_for_an_unparseable_as_of(self, monkeypatch):
        """``as_of`` that is not ``YYYY-MM-DD`` cannot anchor the recency window:
        the render falls back to today (UTC midnight) instead of crashing, and
        the note says which day it used."""
        fixed = datetime(2024, 3, 10, 2, 0, tzinfo=UTC)

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=UTC):
                return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

        monkeypatch.setattr(sq, "datetime", _Frozen)
        monkeypatch.setattr(sq.time, "time", lambda: fixed.timestamp())
        post = _square_post("BTC looks strong", fixed.timestamp() - 3_600)
        block = sq._render_block(
            [post], ("BTC",), "BTC", (fixed.timestamp() - 60.0) * 1000.0, "2024/03/10", []
        )
        head = block.splitlines()[0]
        assert head.endswith(
            "Posts mentioning BTC: 1 (1 of 1 within the last 3 days (as of 2024-03-10); "
            "matching posts span 2024-03-10 01:00 → 2024-03-10 01:00 UTC)."
        )
        assert "fetched 2024-03-10 01:59 UTC (fresh)" in head
