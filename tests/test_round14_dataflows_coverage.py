"""Round-14 coverage hardening for three weakly-covered dataflow modules.

Scope (the only file this round adds; no existing test is modified):

* ``etf_fund_data`` — the never-tested ``_top_holdings`` scrape branch
  (yfinance monkeypatched at the ``yfinance.Ticker`` seam, both the REAL
  1.7.0 frame shape — Symbol index + Name + Holding Percent — and the
  legacy "Holding Name"/"% Assets" variant), the format-helper degrade
  branches, and the "info came back but no fund field" refusal.
* ``fear_greed`` — the HTTP layer (``requests.get`` monkeypatched: success
  shape, transport failure, 429, >=400), malformed-body parsing (including
  non-object / non-list JSON bodies, which used to escape the fail-soft
  contract — regression-pinned here), PIT cutoff boundary semantics
  (end-of-day visibility, next-midnight exclusion, dual-form dates), and
  the real ``cached_or_fetch`` disk-cache wiring against ``tmp_path``
  (single fetch per TTL, expiry refetch, stale-serve on failure).
* ``support_resistance`` — the honest-absence degradation branches of every
  level family (short history, non-numeric H/L/C, empty frames, window
  gating, unusable volume rows, flat typical prices), exact rolling-level
  distance math, and the value-area DOWN-expansion branch.

All ``@pytest.mark.unit``; fully hermetic (no network — every vendor seam
is monkeypatched; the shared module-level F&G throttle is neutralized).
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import yialpha.dataflows.etf_fund_data as efd
import yialpha.dataflows.fear_greed as fg
import yialpha.dataflows.support_resistance as sr
from yialpha.dataflows.disk_cache import MinIntervalThrottle
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError


@pytest.fixture(autouse=True)
def _fast_fng_throttle(monkeypatch):
    """Neutralize the module-level 1s politeness throttle for every test."""
    monkeypatch.setattr(fg, "_throttler", MinIntervalThrottle(0.0))


def _bars(rows, start="2025-01-01"):
    """rows: iterable of (open, high, low, close, volume) with a Date col."""
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])
    frame.insert(0, "Date", pd.bdate_range(start, periods=len(frame)))
    return frame


# ---------------------------------------------------------------------------
# etf_fund_data
# ---------------------------------------------------------------------------


def _patch_yf_ticker(monkeypatch, holdings=None, error=None):
    """Patch ``yfinance.Ticker`` so ``funds_data.top_holdings`` is hermetic.

    ``_top_holdings`` imports yfinance INSIDE the function, so patching the
    module attribute is the seam it actually resolves at call time.
    """
    import yfinance as yf

    class _FundsData:
        @property
        def top_holdings(self):
            if error is not None:
                raise error
            return holdings

    class _Ticker:
        def __init__(self, ticker):
            self.ticker = ticker
            self.funds_data = _FundsData()

    monkeypatch.setattr(yf, "Ticker", _Ticker)


def _real_shape_holdings(n=2):
    """Frame in the shape yfinance 1.7.0 actually serves (scrapers/funds.py):
    ``set_index("Symbol")`` with columns ``Name`` / ``Holding Percent``."""
    return pd.DataFrame(
        {
            "Symbol": [f"S{i}" for i in range(n)],
            "Name": [f"Company {i} Inc." for i in range(n)],
            "Holding Percent": [round(6.5 - 0.3 * i, 2) for i in range(n)],
        }
    ).set_index("Symbol")


@pytest.mark.unit
class TestEtfFormatHelpers:
    def test_fmt_usd_formats_and_degrades(self):
        # .0f rounds half-to-even (1234.5 -> 1,234) — pin the actual format.
        assert efd._fmt_usd(1234.5) == "$1,234"
        assert efd._fmt_usd(1235.5) == "$1,236"
        assert efd._fmt_usd(600_000_000_000) == "$600,000,000,000"
        # Non-numeric vendor values degrade to the raw string, never crash.
        assert efd._fmt_usd("N/A") == "N/A"
        assert efd._fmt_usd(None) == "None"

    def test_fmt_pct_formats_and_degrades(self):
        assert efd._fmt_pct(0.0009) == "0.0900%"
        assert efd._fmt_pct(1.2) == "120.0000%"
        assert efd._fmt_pct("junk") == "junk"
        assert efd._fmt_pct(None) == "None"


@pytest.mark.unit
class TestEtfTopHoldings:
    def test_real_yfinance_shape_names_and_weights(self, monkeypatch):
        # Round-14 regression: with the shape yfinance 1.7.0 actually serves,
        # the percent used to render in holding_name and weight_pct was
        # silently dropped ("AAPL,6.52,"). Names and weights must both land.
        holdings = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT"],
                "Name": ["Apple Inc.", "Microsoft Corporation"],
                "Holding Percent": [6.52, 5.8],
            }
        ).set_index("Symbol")
        _patch_yf_ticker(monkeypatch, holdings=holdings)

        assert efd._top_holdings("SPY") == (
            "symbol,holding_name,weight_pct\n"
            "AAPL,Apple Inc.,6.52\n"
            "MSFT,Microsoft Corporation,5.8"
        )

    def test_legacy_holding_name_variant(self, monkeypatch):
        # The column variant the original detection was written for must keep
        # working: "Holding Name" + "% Assets".
        holdings = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT"],
                "Holding Name": ["Apple Inc.", "Microsoft Corporation"],
                "% Assets": [6.5, 5.8],
            }
        )
        _patch_yf_ticker(monkeypatch, holdings=holdings)

        assert efd._top_holdings("SPY") == (
            "symbol,holding_name,weight_pct\n"
            "AAPL,Apple Inc.,6.5\n"
            "MSFT,Microsoft Corporation,5.8"
        )

    def test_caps_at_ten_rows(self, monkeypatch):
        _patch_yf_ticker(monkeypatch, holdings=_real_shape_holdings(15))

        out = efd._top_holdings("SPY")
        lines = out.splitlines()

        assert lines[0] == "symbol,holding_name,weight_pct"
        assert len(lines) - 1 == 10
        assert lines[1].startswith("S0,Company 0 Inc.")
        assert "S9," in out  # the tenth row is kept
        assert "S10," not in out and "S14," not in out  # rows past ten dropped

    def test_empty_or_none_holdings_return_none(self, monkeypatch):
        _patch_yf_ticker(monkeypatch, holdings=pd.DataFrame())
        assert efd._top_holdings("SPY") is None

        _patch_yf_ticker(monkeypatch, holdings=None)
        assert efd._top_holdings("SPY") is None

    def test_vendor_scrape_failure_returns_none(self, monkeypatch):
        # The fund page scrape is best-effort by contract: any failure must
        # return None (disclosed "unavailable"), never propagate.
        _patch_yf_ticker(monkeypatch, error=RuntimeError("scrape blocked"))
        assert efd._top_holdings("SPY") is None

    def test_unrecognized_columns_render_blank_fields(self, monkeypatch):
        holdings = pd.DataFrame({"Foo": ["x"], "Bar": ["y"]})
        _patch_yf_ticker(monkeypatch, holdings=holdings)

        assert efd._top_holdings("SPY") == "symbol,holding_name,weight_pct\n,y,"


@pytest.mark.unit
class TestGetEtfFundData:
    def test_no_fund_fields_raise(self, monkeypatch):
        # Truthy info with no fund field is an honest refusal, not a calm
        # fund: unknown keys only ...
        monkeypatch.setattr(
            efd, "_cached_ticker_info", lambda t, c: {"quoteType": "ETF"}
        )
        with pytest.raises(NoMarketDataError) as ei:
            efd.get_etf_fund_data("SPY", None)
        assert "no fund fields" in ei.value.detail

        # ... and all-None fund fields (field present, value missing).
        monkeypatch.setattr(
            efd,
            "_cached_ticker_info",
            lambda t, c: {"longName": None, "navPrice": None, "totalAssets": None},
        )
        with pytest.raises(NoMarketDataError, match="no fund fields"):
            efd.get_etf_fund_data("SPY", None)

    def test_unformattable_values_render_raw(self, monkeypatch):
        monkeypatch.setattr(
            efd,
            "_cached_ticker_info",
            lambda t, c: {
                "totalAssets": "N/A",
                "yield": "not-a-number",
                "navPrice": 555.0,
            },
        )
        monkeypatch.setattr(efd, "_top_holdings", lambda t: None)

        out = efd.get_etf_fund_data("SPY", None)

        assert "Total Assets (AUM): N/A" in out
        assert "Yield: not-a-number" in out
        assert "$" not in out  # no bogus USD render of an unformattable value
        assert "NAV (per share): 555.0" in out

    def test_market_price_fallback_and_falsy_current_price(self, monkeypatch):
        # No currentPrice -> regularMarketPrice drives the premium; a FALSY
        # currentPrice (0) must also fall through, not zero the premium.
        monkeypatch.setattr(
            efd,
            "_cached_ticker_info",
            lambda t, c: {
                "navPrice": 100.0,
                "regularMarketPrice": 101.0,
                "currentPrice": 0,
            },
        )
        monkeypatch.setattr(efd, "_top_holdings", lambda t: None)

        out = efd.get_etf_fund_data("SPY", None)

        assert "Market Price: 101.0" in out
        assert "Premium/Discount vs NAV: +100.0 bps" in out

    def test_zero_nav_omits_premium(self, monkeypatch):
        # navPrice 0 (Yahoo's unknown) cannot produce a meaningful bps figure.
        monkeypatch.setattr(
            efd,
            "_cached_ticker_info",
            lambda t, c: {"navPrice": 0, "currentPrice": 100.0},
        )
        monkeypatch.setattr(efd, "_top_holdings", lambda t: None)

        out = efd.get_etf_fund_data("SPY", None)

        assert "NAV (per share): 0" in out
        assert "Premium/Discount" not in out
        assert "Market Price" not in out

    def test_full_render_with_real_top_holdings(self, monkeypatch):
        info = {
            "longName": "SPDR S&P 500 ETF Trust",
            "category": "Large Blend",
            "navPrice": 555.0,
            "previousNav": 554.2,
            "totalAssets": 600_000_000_000,
            "annualReportExpenseRatio": 0.0009,
            "yield": 1.2,
            "currentPrice": 556.1,
        }
        monkeypatch.setattr(efd, "_cached_ticker_info", lambda t, c: info)
        _patch_yf_ticker(monkeypatch, holdings=_real_shape_holdings(2))

        out = efd.get_etf_fund_data("spy", None)  # lower-case in, canonical out

        assert out.startswith("# ETF Fund Data for SPY")
        assert "# Data retrieved on: " in out
        assert "live snapshot; NAV/AUM/expense are today's values" in out
        assert "NAV (per share): 555.0" in out
        assert "Previous NAV: 554.2" in out
        assert "Total Assets (AUM): $600,000,000,000" in out
        assert "Annual Expense Ratio: 0.0900%" in out
        assert "Yield: 120.0000%" in out
        assert "Premium/Discount vs NAV: +19.8 bps" in out
        assert "## Top holdings (Yahoo fund page, best-effort)" in out
        assert "S0,Company 0 Inc.,6.5" in out
        assert "S1,Company 1 Inc.,6.2" in out

    def test_future_date_refused_before_fetch(self, monkeypatch):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()

        def no_fetch(t, c):
            raise AssertionError("fund info fetched for a future date")

        monkeypatch.setattr(efd, "_cached_ticker_info", no_fetch)
        with pytest.raises(NoMarketDataError) as ei:
            efd.get_etf_fund_data("SPY", tomorrow)
        assert "point-in-time" in ei.value.detail
        assert tomorrow in ei.value.detail


# ---------------------------------------------------------------------------
# fear_greed — HTTP layer, parsing, PIT boundaries, cache wiring
# ---------------------------------------------------------------------------


def _ts(iso_date: str) -> str:
    dt = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=UTC)
    return str(int(dt.timestamp()))


def _series_bytes(entries) -> bytes:
    payload = {
        "name": "Fear and Greed Index",
        "data": [
            {"value": str(v), "value_classification": label, "timestamp": _ts(day)}
            for day, v, label in entries
        ],
    }
    return json.dumps(payload).encode()


class _FakeResponse:
    def __init__(self, status_code=200, content=b"", text=""):
        self.status_code = status_code
        self.content = content
        self.text = text


@pytest.mark.unit
class TestFetchSeriesBytes:
    def test_success_returns_body_and_request_shape(self, monkeypatch):
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _FakeResponse(status_code=200, content=b'{"data": []}')

        monkeypatch.setattr("requests.get", fake_get)

        assert fg._fetch_series_bytes() == b'{"data": []}'
        assert captured["url"] == fg._BASE_URL
        assert captured["kwargs"]["params"] == {"limit": fg._LIMIT}
        assert captured["kwargs"]["timeout"] == fg._TIMEOUT
        assert captured["kwargs"]["proxies"] == fg.proxy_map()

    def test_network_failure_maps_to_no_market_data(self, monkeypatch):
        def boom(url, **kwargs):
            raise ConnectionError("dns broke")

        monkeypatch.setattr("requests.get", boom)

        with pytest.raises(NoMarketDataError) as ei:
            fg._fetch_series_bytes()
        assert "request failed" in ei.value.detail
        assert "dns broke" in ei.value.detail
        assert isinstance(ei.value.__cause__, ConnectionError)

    def test_429_maps_to_rate_limit(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get", lambda u, **k: _FakeResponse(status_code=429)
        )
        with pytest.raises(VendorRateLimitError):
            fg._fetch_series_bytes()

    def test_http_error_maps_to_no_market_data_with_excerpt(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda u, **k: _FakeResponse(
                status_code=503, text="Server Error\nMaintenance window"
            ),
        )
        with pytest.raises(NoMarketDataError) as ei:
            fg._fetch_series_bytes()
        assert "Fear&Greed HTTP 503" in ei.value.detail
        # Newlines in the excerpt are flattened so the line stays one line.
        assert "Server Error Maintenance window" in ei.value.detail


@pytest.mark.unit
class TestParseSeries:
    def test_invalid_json_returns_empty(self):
        assert fg._parse_series(b"not-json{") == []
        assert fg._parse_series(b"") == []

    def test_blank_label_skipped(self):
        payload = json.dumps(
            {
                "data": [
                    {"value": "73", "value_classification": "   ",
                     "timestamp": _ts("2026-09-29")},
                    {"value": "71", "value_classification": "Greed",
                     "timestamp": _ts("2026-09-28")},
                ]
            }
        ).encode()
        assert fg._parse_series(payload) == [
            {"ts": int(_ts("2026-09-28")), "value": 71, "label": "Greed"}
        ]

    @pytest.mark.parametrize(
        "raw", [b"[]", b'"error"', b"null", b"123"]
    )
    def test_non_dict_body_returns_empty(self, raw):
        # Round-14 regression: a 200 body that is valid JSON but not an
        # object used to raise AttributeError out of the fail-soft contract.
        assert fg._parse_series(raw) == []

    @pytest.mark.parametrize("data", [42, {"a": 1}, "x"])
    def test_non_list_data_returns_empty(self, data):
        assert fg._parse_series(json.dumps({"data": data}).encode()) == []


@pytest.mark.unit
class TestFormatLineFailSoftRegression:
    @pytest.mark.parametrize(
        "raw",
        [b"[]", b'"error"', b"null", json.dumps({"data": 42}).encode()],
    )
    def test_malformed_body_shapes_return_none(self, monkeypatch, raw):
        monkeypatch.setattr(fg, "cached_or_fetch", lambda *a, **k: raw)
        assert fg.format_fear_greed_line("2026-09-29") is None


@pytest.mark.unit
class TestSelectPitBoundaries:
    def test_end_of_day_visible_next_midnight_not(self):
        # The day-D entry is visible through 23:59:59 UTC of D; an entry
        # timestamped at D+1 00:00:00 UTC must never serve for date D.
        eod = int(_ts("2026-09-29")) + 86_399  # 2026-09-29 23:59:59 UTC
        nxt = eod + 1  # 2026-09-30 00:00:00 UTC
        entries = [
            {"ts": nxt, "value": 90, "label": "Extreme Greed"},
            {"ts": eod, "value": 80, "label": "Greed"},
        ]

        cur, prev = fg._select_pit(entries, "2026-09-29")
        assert cur["value"] == 80
        assert prev is None  # nothing older is visible

        cur, prev = fg._select_pit(entries, "2026-09-30")
        assert cur["value"] == 90
        assert prev["value"] == 80

    def test_dual_form_curr_date_uses_date_part(self, monkeypatch):
        # Round-12 shape: live programmatic entries carry
        # "YYYY-MM-DD HH:MM:SS" — the date part must drive the cutoff.
        raw = _series_bytes(
            [("2026-09-29", 73, "Greed"), ("2026-09-28", 74, "Greed")]
        )
        monkeypatch.setattr(fg, "cached_or_fetch", lambda *a, **k: raw)

        line = fg.format_fear_greed_line("2026-09-29 14:30:00")

        assert line == fg.format_fear_greed_line("2026-09-29")
        assert "as of 2026-09-29: 73 (Greed; prior day 74 (Greed))" in line


@pytest.mark.unit
class TestFearGreedCacheWiring:
    def test_disk_cache_round_trip_single_fetch(self, monkeypatch, tmp_path):
        cache = tmp_path / "fear_greed"
        cache.mkdir()
        monkeypatch.setattr(fg, "_cache_dir", lambda: str(cache))
        payload = _series_bytes(
            [("2026-09-28", 74, "Greed"), ("2026-09-27", 71, "Greed")]
        )
        calls = []

        def fake_get(*a, **k):
            calls.append((a, k))
            return _FakeResponse(content=payload)

        monkeypatch.setattr("requests.get", fake_get)

        line1 = fg.format_fear_greed_line("2026-09-28")
        line2 = fg.format_fear_greed_line("2026-09-28")

        assert len(calls) == 1  # second call served by the fresh cache
        expected = (
            "alternative.me crypto Fear & Greed, as of 2026-09-28: "
            "74 (Greed; prior day 71 (Greed))"
        )
        assert line1 == expected
        assert line2 == expected
        assert (cache / "fng_series.json").read_bytes() == payload

    def test_expired_ttl_refetches(self, monkeypatch, tmp_path):
        cache = tmp_path / "fear_greed"
        cache.mkdir()
        monkeypatch.setattr(fg, "_cache_dir", lambda: str(cache))
        old = _series_bytes([("2026-09-27", 71, "Greed")])
        cache_file = cache / "fng_series.json"
        cache_file.write_bytes(old)
        backdated = time.time() - 2 * 86_400  # beyond the 1-day TTL
        os.utime(cache_file, (backdated, backdated))
        fresh = _series_bytes(
            [("2026-09-28", 74, "Greed"), ("2026-09-27", 71, "Greed")]
        )
        calls = []

        def fake_get(*a, **k):
            calls.append((a, k))
            return _FakeResponse(content=fresh)

        monkeypatch.setattr("requests.get", fake_get)

        line = fg.format_fear_greed_line("2026-09-28")

        assert len(calls) == 1  # stale-by-TTL cache did NOT satisfy the call
        assert "as of 2026-09-28: 74 (Greed; prior day 71 (Greed))" in line
        assert cache_file.read_bytes() == fresh  # refetched bytes republished

    def test_stale_served_when_fetch_fails(self, monkeypatch, tmp_path):
        # The inherited stale-serve contract: fetch failure + a cache inside
        # the staleness cap serves the old bytes (advisory cue degrades, it
        # does not vanish).
        cache = tmp_path / "fear_greed"
        cache.mkdir()
        monkeypatch.setattr(fg, "_cache_dir", lambda: str(cache))
        old = _series_bytes([("2026-09-27", 71, "Greed")])
        cache_file = cache / "fng_series.json"
        cache_file.write_bytes(old)
        backdated = time.time() - 2 * 86_400  # 2 days < the 30-day cap
        os.utime(cache_file, (backdated, backdated))

        def boom(*a, **k):
            raise ConnectionError("network down")

        monkeypatch.setattr("requests.get", boom)

        line = fg.format_fear_greed_line("2026-09-27")

        assert line is not None
        assert "as of 2026-09-27: 71 (Greed)" in line


# ---------------------------------------------------------------------------
# support_resistance — degradation branches and value-area direction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDailyPivotsDegradation:
    def test_short_history_is_none(self):
        assert sr.daily_pivots(None) is None
        assert sr.daily_pivots(_bars([(100, 101, 99, 100, 1)])) is None

    def test_non_numeric_prev_session_is_none(self):
        # Object-dtype H/L/C on the previous session: float() refuses, the
        # family degrades to None instead of raising or fabricating.
        frame = pd.DataFrame(
            [
                ["2025-01-01", 10, 12.0, 9.0, 11.0, 100],
                ["2025-01-02", 11, "n/a", "n/a", "n/a", 100],
                ["2025-01-03", 12, 14.0, 11.0, 13.0, 100],
            ],
            columns=["Date", "Open", "High", "Low", "Close", "Volume"],
        )
        assert sr.daily_pivots(frame) is None


@pytest.mark.unit
class TestWeeklyPivotsDegradation:
    def test_empty_frame_is_none(self):
        assert sr.weekly_pivots(pd.DataFrame(), "2025-06-02") is None
        assert sr.weekly_pivots(None, "2025-06-02") is None

    def test_non_numeric_weekly_bar_is_none(self, monkeypatch):
        bad = pd.DataFrame(
            {"Open": [1.0], "High": ["n/a"], "Low": ["n/a"],
             "Close": ["n/a"], "Volume": [1.0]}
        )
        monkeypatch.setattr(sr, "resample_weekly", lambda df, d: bad)
        assert sr.weekly_pivots(_bars([(100, 101, 99, 100, 1)] * 3), "2025-06-02") is None


@pytest.mark.unit
class TestRollingLevels:
    def test_none_or_empty_returns_empty_dict(self):
        assert sr.rolling_levels(None) == {}
        assert sr.rolling_levels(pd.DataFrame()) == {}

    def test_window_gated_by_history_length(self):
        frame = _bars([(100, 101, 99, 100, 1)] * 30)

        # 30 rows: only the 20-day window has enough history.
        assert set(sr.rolling_levels(frame)) == {20}
        # Custom windows: 5 fits, 50 does not.
        assert set(sr.rolling_levels(frame, windows=(5, 50))) == {5}

    def test_distance_values_are_exact(self):
        rows = [
            (10, 12, 9, 11, 100),
            (11, 13, 10, 12, 100),
            (12, 15, 8, 14, 100),     # window high 15 / low 8 live here
            (14, 14.5, 13, 14, 100),
            (14, 14.2, 13.5, 13.8, 100),  # latest close 13.8
        ]
        lvl = sr.rolling_levels(_bars(rows), windows=(3,))[3]

        assert lvl["high"] == 15.0
        assert lvl["low"] == 8.0
        assert lvl["high_dist_pct"] == pytest.approx((15 - 13.8) / 13.8 * 100)
        assert lvl["low_dist_pct"] == pytest.approx((8 - 13.8) / 13.8 * 100)


@pytest.mark.unit
class TestVolumeProfileDegradation:
    def test_mostly_unusable_rows_is_none(self):
        # Enough rows for the length gate, but almost all carry zero volume
        # (unusable): fewer usable rows than bins//2 must refuse, not bin
        # the handful of survivors into a fabricated profile.
        rows = []
        for i in range(60):
            price = 100.0 + 0.1 * i
            vol = 10.0 if i < 5 else 0.0
            rows.append((price, price, price, price, vol))

        assert sr.volume_profile(_bars(rows)) is None

    def test_flat_typical_prices_is_none(self):
        # Every typical price identical -> zero-width price axis -> None.
        rows = [(100.0, 100.0, 100.0, 100.0, 5.0)] * 60
        assert sr.volume_profile(_bars(rows)) is None


@pytest.mark.unit
class TestVolumeProfileValueArea:
    def test_expands_down_when_lower_bin_heavier(self):
        # Value area must take the heavier NEIGHBOUR each step, not always
        # the upper one: with the second-heaviest cluster BELOW the POC and
        # nothing above it, the only expansion is downward — pinned by
        # value_area_low hitting the axis floor while value_area_high stays
        # at the POC bin's own upper edge (an upward step would move it).
        rows = []
        rows += [(100.0, 100.0, 100.0, 100.0, 30.0)] * 30      # bin 0: 900
        rows += [(100.55, 100.55, 100.55, 100.55, 33.0)] * 30  # bin 1: 990 = POC
        rows += [(103.9, 103.9, 103.9, 103.9, 1.0)] * 30       # bin 9: 30

        prof = sr.volume_profile(_bars(rows), lookback=120, bins=10)

        edges = np.linspace(100.0, 103.9, 11)
        assert prof is not None
        assert prof["value_area_low"] == pytest.approx(float(edges[0]))
        assert prof["value_area_high"] == pytest.approx(float(edges[2]))
        assert prof["poc"] == pytest.approx(float((edges[1] + edges[2]) / 2.0))
        assert prof["lookback_rows"] == 90
        assert prof["bins"] == 10


@pytest.mark.unit
class TestBuildSupportResistanceAbsence:
    def test_all_absent_report_on_none_frame(self):
        report = sr.build_support_resistance(None, "2025-06-02")

        assert report["daily_pivots"] is None
        assert report["weekly_pivots"] is None
        assert report["rolling_levels"] == {}
        assert report["volume_profile"] is None
        assert report["latest_close"] is None
        assert report["latest_date"] is None
