"""Deterministic perp market bundle (yialpha.dataflows.perp_bundle).

Pins the PR3 contract: the three price bases (last/mark/index) with their
derived changes/ATR, funding carry, OI change+percentile, 3-vantage LSR with
cross-vantage spread, taker flow, the fixed-bps depth bands with VWAP
slippage estimates, ADL parsing, spot-perp basis (capability_absent for
tokenized-stock perps), the CORE-completeness quality sentinel, and the
market-analyst prompt injection. Hermetic: every network seam is
monkeypatched.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yialpha.dataflows.perp_bundle as pb
from yialpha.dataflows import quality


def _kline_frame(closes, start="2026-07-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [10.0] * len(closes),
        },
        index=idx,
    )


def _ts(days_ago: int) -> int:
    d = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((d.timestamp() - days_ago * 86_400) * 1000)


# ---- price bases ------------------------------------------------------------


@pytest.mark.unit
def test_prices_three_bases_changes_atr_and_bps(monkeypatch):
    closes = [100.0 + i * 0.5 for i in range(20)]  # last close 109.5

    def fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                    price_type="last"):
        if price_type == "last":
            return _kline_frame(closes)
        if price_type == "mark":
            return _kline_frame([c * 0.999 for c in closes])
        return _kline_frame([c * 0.995 for c in closes])

    monkeypatch.setattr(pb, "binance_klines_frame", fake_klines)
    out = pb._fetch_prices("BTCUSDT", "2026-08-18")

    assert out["core_complete"] is True
    assert out["last"]["close"] == pytest.approx(109.5)
    assert out["last"]["chg_1d"] == pytest.approx(109.5 / 109.0 - 1.0)
    assert out["last"]["chg_7d"] == pytest.approx(109.5 / 106.0 - 1.0)
    assert out["last"]["atr_14"] > 0
    assert out["last"]["atr_pct"] == pytest.approx(
        out["last"]["atr_14"] / 109.5
    )
    assert out["mark"]["close"] == pytest.approx(109.5 * 0.999)
    assert out["index"]["close"] == pytest.approx(109.5 * 0.995)
    # last−mark ≈ +100.1 bps; last−index ≈ +502.5 bps
    assert out["last_vs_mark_bps"] == pytest.approx((1 / 0.999 - 1) * 1e4, abs=1.0)
    assert out["last_vs_index_bps"] == pytest.approx((1 / 0.995 - 1) * 1e4, abs=1.0)
    assert out["coverage"]["rows"] == 20


@pytest.mark.unit
def test_prices_core_incomplete_when_mark_leg_fails(monkeypatch):
    def fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                    price_type="last"):
        if price_type == "mark":
            raise RuntimeError("mark endpoint down")
        return _kline_frame([100.0 + i for i in range(12)])

    monkeypatch.setattr(pb, "binance_klines_frame", fake_klines)
    out = pb._fetch_prices("BTCUSDT", "2026-08-18")
    assert out["core_complete"] is False
    assert out["last"]["close"] == pytest.approx(111.0)
    assert out["mark"]["status"] == pb.STATUS_UNAVAILABLE


# ---- funding / OI / LSR / taker ----------------------------------------------


@pytest.mark.unit
def test_funding_trailing_sum_live_only(monkeypatch):
    monkeypatch.setattr(pb, "is_historical_date", lambda d: True)
    assert pb._fetch_funding("BTCUSDT", "2020-01-10")["status"] == (
        pb.STATUS_SKIPPED_LIVE_ONLY
    )

    monkeypatch.setattr(pb, "is_historical_date", lambda d: False)
    monkeypatch.setattr(
        pb, "_http_get",
        lambda path, params, *a, **k: [
            {"fundingTime": _ts(1), "fundingRate": "0.0001"},
            {"fundingTime": _ts(1), "fundingRate": "0.0001"},
            {"fundingTime": _ts(0), "fundingRate": "-0.0002"},
        ],
    )
    out = pb._fetch_funding("BTCUSDT", date.today().isoformat())
    assert out["status"] == pb.STATUS_OK
    assert out["sum_7d"] == pytest.approx(0.0)
    assert out["settlements"] == 3
    assert out["annualized"] == pytest.approx(0.0)


@pytest.mark.unit
def test_open_interest_change_and_percentile(monkeypatch):
    monkeypatch.setattr(
        pb, "_http_get",
        lambda path, params, *a, **k: [
            {"timestamp": _ts(2), "sumOpenInterest": "100"},
            {"timestamp": _ts(1), "sumOpenInterest": "110"},
            {"timestamp": _ts(0), "sumOpenInterest": "121"},
        ],
    )
    out = pb._fetch_open_interest("BTCUSDT", date.today().isoformat())
    assert out["status"] == pb.STATUS_OK
    assert out["latest"] == pytest.approx(121.0)
    assert out["chg_1d"] == pytest.approx(121.0 / 110.0 - 1.0)
    # Round-11: the latest ranks against the PRIOR window only (self
    # excluded from the denominator) — 121 tops both prior values, so the
    # window max reads a full 100th percentile, not the old 2/3 cap.
    assert out["percentile"] == pytest.approx(100.0)
    assert "chg_7d" not in out  # window too short — not fabricated


@pytest.mark.unit
def test_lsr_three_vantages_and_spread(monkeypatch):
    def fake_http(path, params, *a, **k):
        return [
            {"timestamp": _ts(1), "longShortRatio": "1.7"},
            {"timestamp": _ts(0), "longShortRatio": "1.8"},
        ]

    monkeypatch.setattr(pb, "_http_get", fake_http)
    out = pb._fetch_lsr("BTCUSDT", date.today().isoformat())
    assert out["status"] == pb.STATUS_OK
    assert out["top_account"]["latest"] == pytest.approx(1.8)
    assert out["top_position"]["latest"] == pytest.approx(1.8)
    assert out["global_account"]["latest"] == pytest.approx(1.8)
    assert out["cross_vantage_spread"] == pytest.approx(0.0)


@pytest.mark.unit
def test_taker_latest_and_mean(monkeypatch):
    # Rows oldest-first (endpoint order): d=7 (oldest) .. d=0 (today).
    monkeypatch.setattr(
        pb, "_http_get",
        lambda path, params, *a, **k: [
            {"timestamp": _ts(d), "buySellRatio": f"{1.0 + (7 - d) * 0.05:.2f}"}
            for d in range(7, -1, -1)
        ],
    )
    out = pb._fetch_taker("BTCUSDT", date.today().isoformat())
    assert out["status"] == pb.STATUS_OK
    assert out["latest"] == pytest.approx(1.35)
    # Round-7: "7d" means 7 daily values END-INCLUSIVE ([D-6..D]) — the same
    # seam convention the funding fix settled; the old mean over all 8 served
    # rows averaged 8 days under a 7d label.
    assert out["mean_7d"] == pytest.approx(
        sum(1.0 + (7 - d) * 0.05 for d in range(6, -1, -1)) / 7
    )


# ---- live microstructure ------------------------------------------------------


@pytest.mark.unit
def test_premium_snapshot_parse(monkeypatch):
    nft = (datetime.now(UTC).timestamp() + 3600) * 1000
    monkeypatch.setattr(
        pb, "_http_get",
        lambda path, params, *a, **k: {
            "symbol": "BTCUSDT", "markPrice": "100.5", "indexPrice": "100.0",
            "lastFundingRate": "0.0001", "nextFundingTime": nft,
        },
    )
    out = pb._fetch_premium_snapshot("BTCUSDT")
    assert out["status"] == pb.STATUS_OK
    assert out["mark"] == pytest.approx(100.5)
    assert out["mark_vs_index_bps"] == pytest.approx(50.0, abs=0.5)
    assert out["next_funding_rate"] == pytest.approx(0.0001)
    assert out["next_funding_time_utc"] is not None


@pytest.mark.unit
def test_depth_bands_imbalance_and_slippage(monkeypatch):
    # Book thick enough for the 10k reference order, too thin for 50k —
    # a reference order the book cannot absorb has NO honest VWAP estimate.
    bids = [["100.0", "10"], ["99.9", "10"], ["99.5", "200"]]
    asks = [["100.2", "10"], ["100.3", "10"], ["100.5", "200"]]
    monkeypatch.setattr(
        pb, "_http_get",
        lambda path, params, *a, **k: {"bids": bids, "asks": asks},
    )
    out = pb._fetch_depth_bands("BTCUSDT")
    assert out["status"] == pb.STATUS_OK
    mid = 100.1
    assert out["mid"] == pytest.approx(mid)
    assert out["spread_bps"] == pytest.approx((100.2 / 100.0 - 1.0) * 1e4)

    band20 = out["bands"]["20"]
    assert band20["bid_notional"] == pytest.approx(100.0 * 10 + 99.9 * 10)
    assert band20["ask_notional"] == pytest.approx(100.2 * 10 + 100.3 * 10)
    expected_imb = (
        (100.0 * 10 + 99.9 * 10) - (100.2 * 10 + 100.3 * 10)
    ) / (100.0 * 10 + 99.9 * 10 + 100.2 * 10 + 100.3 * 10)
    assert band20["imbalance"] == pytest.approx(expected_imb)

    # Independent re-derivation of the buy-side 10k VWAP impact: walk asks
    # best-first until 10,000 notional is spent.
    buy_qty = 10 + 10 + (10_000 - (100.2 * 10 + 100.3 * 10)) / 100.5
    buy_vwap = 10_000 / buy_qty
    assert out["slippage_bps"]["buy_10000"] == pytest.approx(
        (buy_vwap / mid - 1.0) * 1e4, rel=1e-6,
    )
    sell_qty = 10 + 10 + (10_000 - (100.0 * 10 + 99.9 * 10)) / 99.5
    sell_vwap = 10_000 / sell_qty
    assert out["slippage_bps"]["sell_10000"] == pytest.approx(
        (sell_vwap / mid - 1.0) * 1e4, rel=1e-6,
    )
    # The 50k reference order exceeds the visible book on both sides — no
    # estimate is fabricated.
    assert out["slippage_bps"]["buy_50000"] is None
    assert out["slippage_bps"]["sell_50000"] is None


@pytest.mark.unit
def test_adl_signed_when_keys_configured(monkeypatch):
    # /fapi/v1/adlQuantile is a SIGNED endpoint: with operator keys the
    # request carries timestamp/recvWindow + HMAC signature and the
    # X-MBX-APIKEY header — no doomed unsigned call.
    import yialpha.dataflows.binance_brackets as bb

    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    seen = {}

    def fake_signed_get(path, params, symbol, canonical):
        seen.update(path=path, params=dict(params), symbol=symbol)
        return [
            {"symbol": "BTCUSDT", "adlQuantile": {"LONG": "1", "SHORT": 2}},
        ]

    monkeypatch.setattr(bb, "signed_fapi_get", fake_signed_get)
    out = pb._fetch_adl("BTCUSDT")
    assert out["status"] == pb.STATUS_OK
    assert out["long"] == 1 and out["short"] == 2
    assert seen["path"] == "/fapi/v1/adlQuantile"
    assert seen["symbol"] == "BTCUSDT"  # params carry the canonical symbol


@pytest.mark.unit
def test_adl_signed_request_shape(monkeypatch):
    import hashlib
    import hmac as hmac_mod

    import yialpha.dataflows.binance_brackets as bb

    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    seen = {}

    def fake_http(path, params, symbol, canonical, headers=None):
        seen.update(path=path, params=dict(params), headers=dict(headers or {}))
        return [{"symbol": canonical, "adlQuantile": {"LONG": 0, "SHORT": 0}}]

    monkeypatch.setattr(bb, "_http_get", fake_http)
    out = pb._fetch_adl("BTCUSDT")
    assert out["status"] == pb.STATUS_OK
    assert seen["path"] == "/fapi/v1/adlQuantile"
    assert seen["headers"]["X-MBX-APIKEY"] == "test-key"
    assert "timestamp" in seen["params"] and "recvWindow" in seen["params"]
    # The signature is the HMAC-SHA256 of the exact urlencode'd query.
    import urllib.parse

    expected = hmac_mod.new(
        b"test-secret",
        urllib.parse.urlencode(
            {k: v for k, v in seen["params"].items() if k != "signature"}
        ).encode(),
        hashlib.sha256,
    ).hexdigest()
    assert seen["params"]["signature"] == expected


@pytest.mark.unit
def test_adl_without_keys_degrades_without_request(monkeypatch):
    import yialpha.dataflows.binance_brackets as bb

    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    def no_call(*_a, **_k):  # noqa: ARG001
        raise AssertionError("HTTP call made without keys")

    # Patch BELOW the key check: signed_fapi_get must refuse before any
    # request fires — a doomed unsigned call at a signed endpoint.
    monkeypatch.setattr(bb, "_http_get", no_call)
    degraded = pb._fetch_adl("BTCUSDT")
    assert degraded["status"] == pb.STATUS_UNAVAILABLE
    assert "signed endpoint" in degraded["reason"]
    assert "BINANCE_API_KEY" in degraded["reason"]


@pytest.mark.unit
def test_adl_transport_error_degrades(monkeypatch):
    import yialpha.dataflows.binance_brackets as bb

    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")

    def boom(*_a, **_k):
        raise RuntimeError("transport down")

    monkeypatch.setattr(bb, "signed_fapi_get", boom)
    degraded = pb._fetch_adl("BTCUSDT")
    assert degraded["status"] == pb.STATUS_UNAVAILABLE
    assert "transport down" in degraded["reason"]


# ---- spot basis ---------------------------------------------------------------


@pytest.mark.unit
def test_spot_basis_capability_absent_for_stock_perp(monkeypatch):
    monkeypatch.setattr(pb, "stock_perp_underlying", lambda t: "MU")
    out = pb._fetch_spot_basis(
        "MUUSDT", "2026-08-18", (pd.Timestamp("2026-08-18"), 100.0),
    )
    assert out["status"] == pb.STATUS_CAPABILITY_ABSENT
    assert "no Binance spot leg" in out["reason"]


@pytest.mark.unit
def test_spot_basis_bps_for_pure_crypto(monkeypatch):
    monkeypatch.setattr(pb, "stock_perp_underlying", lambda t: None)

    def fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                    price_type="last"):
        assert venue == "binance_spot"
        return _kline_frame([99.0 + i * 0.1 for i in range(8)])

    monkeypatch.setattr(pb, "binance_klines_frame", fake_klines)
    # (tail date, close) hint: the spot frame runs 2026-07-01..08, so the
    # perp tail 2026-07-08 pairs the spot bar AT that date.
    out = pb._fetch_spot_basis(
        "BTCUSDT", "2026-08-18", (pd.Timestamp("2026-07-08"), 100.0),
    )
    assert out["status"] == pb.STATUS_OK
    assert out["spot_close"] == pytest.approx(99.7)
    assert out["basis_bps"] == pytest.approx((100.0 / 99.7 - 1.0) * 1e4)


# ---- round-11: same-date basis pairing (no cross-day ratios) -------------------


@pytest.mark.unit
def test_prices_basis_degrades_when_mark_tail_lags(monkeypatch):
    """Round-11: the last−mark basis is a SAME-DATE ratio. A mark feed whose
    tail lags one day (mark frame ends D-1 while last ends D) used to pair
    each leg's positional ``iloc[-1]`` — fabricating last@D / mark@D-1
    displacement. The metric must degrade (key absent), never cross-date."""
    last_closes = [100.0 + i for i in range(10)]      # tail 2026-07-10
    mark_closes = [90.0 + i for i in range(9)]        # tail 2026-07-09 (lags)

    def fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                    price_type="last"):
        if price_type == "mark":
            return _kline_frame(mark_closes)
        return _kline_frame(last_closes)

    monkeypatch.setattr(pb, "binance_klines_frame", fake_klines)
    out = pb._fetch_prices("BTCUSDT", "2026-08-18")
    assert out["core_complete"] is True
    assert "last_vs_mark_bps" not in out  # degraded, not cross-dated
    # index aligned with last -> its basis still pairs same-date.
    assert out["last_vs_index_bps"] == pytest.approx(0.0, abs=1e-6)
    # The per-leg readings themselves are untouched by the pairing rule.
    assert out["mark"]["close"] == pytest.approx(98.0)


@pytest.mark.unit
def test_spot_basis_degrades_when_spot_tail_lags(monkeypatch):
    """Round-11 (regime round-10 seam): the spot leg must carry a bar at the
    perp tail's own date. A spot frame ending D-1 degrades to unavailable —
    the old positional pairing fabricated perp@D / spot@D-1 basis."""
    monkeypatch.setattr(pb, "stock_perp_underlying", lambda t: None)

    def fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                    price_type="last"):
        assert venue == "binance_spot"
        return _kline_frame([99.0 + i * 0.1 for i in range(7)])  # ends 07-07

    monkeypatch.setattr(pb, "binance_klines_frame", fake_klines)
    out = pb._fetch_spot_basis(
        "BTCUSDT", "2026-08-18", (pd.Timestamp("2026-07-08"), 100.0),
    )
    assert out["status"] == pb.STATUS_UNAVAILABLE
    assert "lags" in out["reason"]
    assert "basis_bps" not in out


# ---- round-11 remediation: duplicated tail dates must not TypeError -------------


@pytest.mark.unit
def test_prices_pairing_survives_duplicated_tail_date(monkeypatch):
    """A paginated unordered kline frame can carry the tail date twice —
    ``Series.loc[tail]`` then returns a SERIES and ``float(Series)`` raises
    TypeError, failing the whole prices leg into a core NO_TRADE sentinel.
    keep="last" resolves the overlap (the last page's row wins) and the
    same-date basis still pairs."""
    base = [100.0 + i for i in range(9)]  # 2026-07-01..09
    # The last leg's feed delivers 07-10 twice: page A @111, page B @199
    # (unordered pagination backfill). keep="last" -> 199.
    dup_last = pd.concat([
        _kline_frame(base),
        _kline_frame([111.0], start="2026-07-10"),
        _kline_frame([199.0], start="2026-07-10"),
    ])
    assert dup_last.index.duplicated().sum() == 1  # fixture really is dup
    clean_tail = _kline_frame(base + [199.0])  # mark/index: no dup

    def fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                    price_type="last"):
        if price_type == "last":
            return dup_last
        return clean_tail

    monkeypatch.setattr(pb, "binance_klines_frame", fake_klines)
    out = pb._fetch_prices("BTCUSDT", "2026-08-18")
    assert out["core_complete"] is True
    # keep="last": the tail close is 199.0 (not 111.0, not a TypeError).
    assert out["last"]["close"] == pytest.approx(199.0)
    # The basis pairs same-date on the deduplicated values: 199 vs 199 -> 0.
    assert out["last_vs_mark_bps"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_spot_basis_survives_duplicated_tail_date(monkeypatch):
    """Same seam on the spot leg: a duplicated perp-tail date in the spot
    frame must resolve keep="last", not raise float(Series) TypeError."""
    monkeypatch.setattr(pb, "stock_perp_underlying", lambda t: None)

    spot = pd.concat([
        _kline_frame([99.0 + i * 0.1 for i in range(7)]),  # ..2026-07-07
        _kline_frame([99.9], start="2026-07-08"),          # page A @99.9
        _kline_frame([100.0], start="2026-07-08"),         # page B @100.0
    ])
    assert spot.index.duplicated().sum() == 1

    monkeypatch.setattr(pb, "binance_klines_frame", lambda *a, **k: spot)
    out = pb._fetch_spot_basis(
        "BTCUSDT", "2026-08-18", (pd.Timestamp("2026-07-08"), 101.0),
    )
    assert out["status"] == pb.STATUS_OK
    assert out["spot_close"] == pytest.approx(100.0)  # keep="last"
    assert out["basis_bps"] == pytest.approx((101.0 / 100.0 - 1.0) * 1e4)


# ---- round-11: OI percentile ranks against the PRIOR window --------------------


@pytest.mark.unit
def test_open_interest_monotonic_new_high_reads_100th(monkeypatch):
    """Round-11: the old denominator counted the latest as its own
    comparison base, so a monotonic new high on a full window could never
    exceed (n-1)/n — a 30-row window capped at ~97th pct. The latest must
    rank against the PRIOR window only: a new high reads a full 100.0."""
    rows = [
        {"timestamp": _ts(29 - i), "sumOpenInterest": str(100.0 + i)}
        for i in range(30)
    ]
    monkeypatch.setattr(pb, "_http_get", lambda path, params, *a, **k: rows)
    out = pb._fetch_open_interest("BTCUSDT", date.today().isoformat())
    assert out["status"] == pb.STATUS_OK
    assert out["days"] == 30
    assert out["percentile"] == pytest.approx(100.0)
    # Mid-window value: 4 of 5 prior values below it -> 80th, not 4/6~67th.
    rows_mid = [
        {"timestamp": _ts(5 - i), "sumOpenInterest": str(v)}
        for i, v in enumerate([100.0, 90.0, 125.0, 110.0, 105.0, 120.0])
    ]
    monkeypatch.setattr(pb, "_http_get", lambda path, params, *a, **k: rows_mid)
    out_mid = pb._fetch_open_interest("BTCUSDT", date.today().isoformat())
    assert out_mid["percentile"] == pytest.approx(4 / 5 * 100.0)


@pytest.mark.unit
def test_open_interest_single_row_window_is_disclosed_vacuous(monkeypatch):
    """n == 1 (retention clamp / brand-new contract) has no prior base: the
    latest is trivially the window's max -> 100.0, with the vacuous span
    disclosed through ``days`` so the rendered label self-identifies."""
    monkeypatch.setattr(
        pb, "_http_get",
        lambda path, params, *a, **k: [
            {"timestamp": _ts(0), "sumOpenInterest": "121"},
        ],
    )
    out = pb._fetch_open_interest("BTCUSDT", date.today().isoformat())
    assert out["status"] == pb.STATUS_OK
    assert out["days"] == 1
    assert out["percentile"] == pytest.approx(100.0)


# ---- round-11: endTime belt-and-suspenders row trim ----------------------------


@pytest.mark.unit
def test_oi_series_trims_rows_past_window_end(monkeypatch):
    """A vendor (or mock) that ignores the endTime must never leak rows past
    the requested window end into the percentile base (the tool layer's
    binance.py seam, applied to the bundle's direct ``_http_get`` calls).
    Row timestamps are built RELATIVE to the requested endTime so the pin
    holds on any host/UTC date skew."""
    day = 86_400_000

    def fake_http(path, params, *a, **k):
        end = params["endTime"]
        return [
            {"timestamp": end - 2 * day, "sumOpenInterest": "100"},
            {"timestamp": end - day, "sumOpenInterest": "110"},
            {"timestamp": end + 60_000, "sumOpenInterest": "999"},  # beyond
        ]

    monkeypatch.setattr(pb, "_http_get", fake_http)
    rows = pb._oi_series("BTCUSDT", date.today().isoformat(), 30)
    assert [v for _ts, v in rows] == [100.0, 110.0]


@pytest.mark.unit
def test_lsr_and_taker_trim_rows_past_window_end(monkeypatch):
    day = 86_400_000

    def fake_http(path, params, *a, **k):
        end = params["endTime"]
        return [
            {"timestamp": end - day, "longShortRatio": "1.5",
             "buySellRatio": "1.2"},
            {"timestamp": end + 60_000, "longShortRatio": "9.9",
             "buySellRatio": "9.9"},  # beyond end — must drop
        ]

    monkeypatch.setattr(pb, "_http_get", fake_http)
    lsr = pb._fetch_lsr("BTCUSDT", date.today().isoformat())
    for key in ("top_account", "top_position", "global_account"):
        assert lsr[key]["latest"] == pytest.approx(1.5)
    taker = pb._fetch_taker("BTCUSDT", date.today().isoformat())
    assert taker["latest"] == pytest.approx(1.2)


# ---- round-11: per-leg labeling timeout ----------------------------------------


@pytest.mark.unit
def test_leg_timeout_default_and_env_parsing(monkeypatch):
    """``YIALPHA_PERP_BUNDLE_LEG_TIMEOUT_S`` defaults to 300s — ABOVE the
    legal slow paths (proactive-backoff limiter waits out a ~60s weight
    window per request; BINANCE_HTTP_RETRIES >= 3 adds 62s of backoff) that
    the old fixed 60s mislabeled as TimeoutError -> core sentinel ->
    NO_TRADE. ``0`` is EXPLICIT OFF (repo convention, round-11 remediation):
    ``None`` makes ``future.result`` wait forever, identical to the pool
    join's own semantics. Negative / non-numeric values fall back to 300."""
    import importlib

    assert pb.LEG_TIMEOUT_S == 300.0
    for raw, expected in (("900", 900.0), ("0", None), ("-5", 300.0),
                          ("garbage", 300.0)):
        monkeypatch.setenv("YIALPHA_PERP_BUNDLE_LEG_TIMEOUT_S", raw)
        importlib.reload(pb)
        assert pb.LEG_TIMEOUT_S == expected, raw
    monkeypatch.delenv("YIALPHA_PERP_BUNDLE_LEG_TIMEOUT_S", raising=False)
    importlib.reload(pb)
    assert pb.LEG_TIMEOUT_S == 300.0


@pytest.mark.unit
def test_bundle_leg_timeout_reads_module_constant(monkeypatch):
    """The result-wait threshold must read ``LEG_TIMEOUT_S`` (not a
    re-hardcoded 60): a leg over the threshold is labeled
    TimeoutError-unavailable, a leg under it lands normally. Together with
    the 300s default this pins the round-11 fix — a legal ~62s slow leg
    lands in the bundle instead of escalating to a core sentinel."""
    import time as time_mod

    monkeypatch.setattr(pb, "is_historical_date", lambda d: True)
    monkeypatch.setattr(pb, "LEG_TIMEOUT_S", 0.05)

    def slow_taker(symbol, end_date):
        time_mod.sleep(0.1)  # over the patched threshold
        return {"status": pb.STATUS_OK}

    def fast_oi(symbol, end_date):
        return {"status": pb.STATUS_OK}

    monkeypatch.setattr(
        pb, "_fetch_prices",
        lambda s, d: {"status": pb.STATUS_OK, "core_complete": True},
    )
    monkeypatch.setattr(pb, "_fetch_open_interest", fast_oi)
    monkeypatch.setattr(pb, "_fetch_lsr", fast_oi)
    monkeypatch.setattr(pb, "_fetch_taker", slow_taker)
    monkeypatch.setattr(pb, "_fetch_spot_basis",
                        lambda s, d, c: {"status": pb.STATUS_CAPABILITY_ABSENT})
    monkeypatch.setattr(pb, "_last_close_hint", lambda s, d: None)

    quality.ensure_run_context()
    try:
        bundle = pb.fetch_perp_market_bundle("BTCUSDT", "2020-01-10")
    finally:
        quality.reset_quality()

    assert "TimeoutError" in bundle["taker"]["reason"]
    assert bundle["taker"]["status"] == pb.STATUS_UNAVAILABLE
    assert bundle["prices"]["status"] == pb.STATUS_OK  # under threshold lands


# ---- assembler: CORE sentinel --------------------------------------------------


@pytest.mark.unit
def test_core_price_failure_records_quality_sentinel(monkeypatch):
    monkeypatch.setattr(
        pb, "_fetch_prices",
        lambda s, d: {"status": pb.STATUS_UNAVAILABLE, "core_complete": False},
    )
    monkeypatch.setattr(
        pb, "_fetch_open_interest",
        lambda s, d: {"status": pb.STATUS_UNAVAILABLE, "reason": "retention"},
    )
    monkeypatch.setattr(pb, "_fetch_lsr", lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_taker",
                        lambda s, d: {"status": pb.STATUS_UNAVAILABLE})
    monkeypatch.setattr(pb, "_fetch_spot_basis",
                        lambda s, d, c: {"status": pb.STATUS_CAPABILITY_ABSENT})
    monkeypatch.setattr(pb, "_last_close_hint", lambda s, d: None)

    quality.ensure_run_context()
    try:
        bundle = pb.fetch_perp_market_bundle("BTCUSDT", "2020-01-10")
        events = quality.snapshot_quality()
    finally:
        quality.reset_quality()

    assert bundle["prices"]["core_complete"] is False
    core = [e for e in events if e["method"] == "get_binance_klines"]
    assert core, "a core price failure must reach the quality chain"
    # KIND_CORE_ERROR (2026-09-19 round 2): same outage, same kind as the
    # overlay's price failures — run_robust's DEGRADED counter (which only
    # counts the core kinds) must see the bundle's miss too.
    assert core[0]["kind"] == quality.KIND_CORE_ERROR
    # ...and per PR1's method-level matrix that sentinel is CRITICAL. With
    # ZERO successes the vacuum belt-and-suspenders grades INVALID (the
    # default reject policy refuses such runs anyway); with a qualified
    # router success present the unqualified bundle sentinel stays
    # unrecovered → DEGRADED_CRITICAL (ticket NO_TRADE).
    from yialpha.dataflows.quality import classify_quality

    assert classify_quality(events, set())["tier"] == "INVALID"
    assert classify_quality(events, {"get_binance_klines[last]"})["tier"] == (
        "DEGRADED_CRITICAL"
    )


@pytest.mark.unit
def test_historical_run_skips_live_components(monkeypatch):
    monkeypatch.setattr(pb, "is_historical_date", lambda d: True)
    monkeypatch.setattr(
        pb, "_fetch_prices",
        lambda s, d: {
            "status": pb.STATUS_OK, "core_complete": True,
            "last": {"status": pb.STATUS_OK, "close": 100.0},
            "mark": {"status": pb.STATUS_OK, "close": 99.9},
        },
    )
    for name in ("_fetch_open_interest", "_fetch_lsr", "_fetch_taker"):
        monkeypatch.setattr(
            pb, name, lambda s, d, _n=name: {"status": pb.STATUS_UNAVAILABLE},
        )
    monkeypatch.setattr(pb, "_fetch_spot_basis",
                        lambda s, d, c: {"status": pb.STATUS_CAPABILITY_ABSENT})
    monkeypatch.setattr(pb, "_last_close_hint", lambda s, d: None)

    quality.ensure_run_context()
    try:
        bundle = pb.fetch_perp_market_bundle("BTCUSDT", "2020-01-10")
    finally:
        quality.reset_quality()

    assert bundle["live_run"] is False
    for key in ("funding", "premium_snapshot", "depth_bands", "adl"):
        assert key not in bundle, f"{key} is live-only"


@pytest.mark.unit
def test_auxiliary_failures_enter_the_ledger(monkeypatch):
    """Aux outages must reach the quality chain as optional-unavailable —
    before this, a bundle whose enrichment failed across the board still
    classified GOOD (nothing router-recorded was ever attempted). Structural
    states (capability_absent / skipped_live_only) never record."""
    monkeypatch.setattr(pb, "is_historical_date", lambda d: True)
    monkeypatch.setattr(
        pb, "_fetch_prices",
        lambda s, d: {
            "status": pb.STATUS_OK, "core_complete": True,
            "last": {"status": pb.STATUS_OK, "close": 100.0},
            "mark": {"status": pb.STATUS_OK, "close": 99.9},
        },
    )
    for name in ("_fetch_open_interest", "_fetch_lsr", "_fetch_taker"):
        monkeypatch.setattr(
            pb, name, lambda s, d, _n=name: {"status": pb.STATUS_UNAVAILABLE},
        )
    monkeypatch.setattr(pb, "_fetch_spot_basis",
                        lambda s, d, c: {"status": pb.STATUS_CAPABILITY_ABSENT})
    monkeypatch.setattr(pb, "_last_close_hint", lambda s, d: None)

    quality.ensure_run_context()
    try:
        pb.fetch_perp_market_bundle("BTCUSDT", "2020-01-10")
        events = quality.snapshot_quality()
    finally:
        quality.reset_quality()

    recorded = {e["method"] for e in events}
    assert recorded == {
        "get_binance_open_interest",
        "get_binance_long_short_ratio",
        "get_binance_taker_buy_sell",
    }
    assert all(
        e["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE for e in events
    )
    # spot_basis said capability_absent (a structural state) and the price
    # core was complete, so neither a basis nor a klines sentinel may appear.
    from yialpha.dataflows.quality import classify_quality

    verdict = classify_quality(events, {"get_binance_klines"})
    assert verdict["tier"] == "DEGRADED_AUXILIARY"
    assert verdict["critical_missing"] == []


@pytest.mark.unit
def test_adl_unavailable_records_bundle_scoped_sentinel(monkeypatch):
    monkeypatch.setattr(pb, "is_historical_date", lambda d: False)
    monkeypatch.setattr(
        pb, "_fetch_prices",
        lambda s, d: {
            "status": pb.STATUS_OK, "core_complete": True,
            "last": {"status": pb.STATUS_OK, "close": 100.0},
            "mark": {"status": pb.STATUS_OK, "close": 99.9},
        },
    )
    monkeypatch.setattr(pb, "_fetch_funding",
                        lambda s, d: {"status": pb.STATUS_OK, "sum_7d": 0.0})
    monkeypatch.setattr(pb, "_fetch_premium_snapshot",
                        lambda s: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_depth_bands",
                        lambda s: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_adl",
                        lambda s: {"status": pb.STATUS_UNAVAILABLE,
                                   "reason": "signed request required"})
    monkeypatch.setattr(pb, "_fetch_open_interest",
                        lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_lsr", lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_taker",
                        lambda s, d: {"status": pb.STATUS_OK})
    monkeypatch.setattr(pb, "_fetch_spot_basis",
                        lambda s, d, c: {"status": pb.STATUS_OK, "basis_bps": 1.0})
    monkeypatch.setattr(pb, "_last_close_hint", lambda s, d: 100.0)

    quality.ensure_run_context()
    try:
        pb.fetch_perp_market_bundle("BTCUSDT", date.today().isoformat())
        events = quality.snapshot_quality()
    finally:
        quality.reset_quality()

    adl = [e for e in events if e["method"] == "perp_bundle_adl"]
    assert adl and adl[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE


# ---- renderer -----------------------------------------------------------------


def _full_bundle():
    return {
        "symbol": "BTCUSDT",
        "as_of": "2026-08-18",
        "live_run": True,
        "prices": {
            "status": pb.STATUS_OK,
            "core_complete": True,
            "last": {
                "status": pb.STATUS_OK, "close": 109.5,
                "chg_1d": 0.0046, "chg_7d": 0.033,
                "atr_14": 1.9, "atr_pct": 0.017,
            },
            "mark": {"status": pb.STATUS_OK, "close": 109.4},
            "index": {"status": pb.STATUS_OK, "close": 108.9},
            "last_vs_mark_bps": 9.1,
            "last_vs_index_bps": 55.1,
            "coverage": {"rows": 35, "start": "2026-07-15", "end": "2026-08-18"},
        },
        "funding": {"status": pb.STATUS_OK, "sum_7d": 0.0007,
                    "annualized": 0.0365, "settlements": 21},
        "premium_snapshot": {
            "status": pb.STATUS_OK, "mark_vs_index_bps": 45.0,
            "next_funding_rate": 0.0001, "next_funding_time_utc": "2026-08-19 00:00",
        },
        "open_interest": {"status": pb.STATUS_OK, "latest": 85_000.0,
                          "chg_1d": 0.012, "chg_7d": -0.03, "percentile": 88.0},
        "long_short": {
            "status": pb.STATUS_OK,
            "top_account": {"status": pb.STATUS_OK, "latest": 1.8},
            "top_position": {"status": pb.STATUS_OK, "latest": 1.2},
            "global_account": {"status": pb.STATUS_OK, "latest": 1.5},
            "cross_vantage_spread": 0.6,
        },
        "taker": {"status": pb.STATUS_OK, "latest": 1.21, "mean_7d": 1.05},
        "depth_bands": {
            "status": pb.STATUS_OK, "mid": 109.5, "spread_bps": 1.0,
            "bands": {
                "20": {"bid_notional": 200_000.0, "ask_notional": 150_000.0,
                       "imbalance": 0.1428},
                "50": {"bid_notional": 500_000.0, "ask_notional": 400_000.0,
                       "imbalance": 0.1111},
                "100": {"bid_notional": 900_000.0, "ask_notional": 800_000.0,
                        "imbalance": 0.0588},
                "200": {"bid_notional": 1_500_000.0, "ask_notional": 1_400_000.0,
                        "imbalance": 0.0345},
                "500": {"bid_notional": 2_500_000.0, "ask_notional": 2_400_000.0,
                        "imbalance": 0.0204},
            },
            "slippage_bps": {"buy_10000": 2.1, "sell_10000": -2.3,
                             "buy_50000": 8.4, "sell_50000": -9.1},
        },
        "adl": {"status": pb.STATUS_OK, "long": 1, "short": 2},
        "spot_basis": {
            "status": pb.STATUS_OK, "spot_close": 109.2, "basis_bps": 27.5,
        },
    }


@pytest.mark.unit
def test_render_full_bundle_carries_every_section():
    block = pb.render_perp_bundle_block(_full_bundle())
    for fragment in (
        "Perp Market Bundle — BTCUSDT",
        "Price bases",
        "Mark/last basis",
        "ATR14",
        "Funding (7d)",
        "Premium snapshot (live)",
        "Open interest",
        "Long/short (acct ratio)",
        "Taker flow",
        "Depth (bid/ask notional, live)",
        "Est. slippage (VWAP vs mid)",
        "ADL quantile",
        "Spot-perp basis",
    ):
        assert fragment in block, fragment
    # No footer when everything is ok — availability notes only appear when
    # something is actually degraded.
    assert "Component availability" not in block


@pytest.mark.unit
def test_render_discloses_unavailable_and_capability_absent():
    bundle = _full_bundle()
    bundle["adl"] = {"status": pb.STATUS_UNAVAILABLE, "reason": "auth required"}
    bundle["spot_basis"] = {
        "status": pb.STATUS_CAPABILITY_ABSENT,
        "reason": "tokenized-stock perp (MU underlying) has no Binance spot leg",
    }
    block = pb.render_perp_bundle_block(bundle)
    assert "capability absent" in block
    footer = [ln for ln in block.splitlines() if "Component availability" in ln]
    assert footer and "adl: unavailable" in footer[0]


# ---- market-analyst prompt injection ------------------------------------------


class _PromptCaptureLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.prompt = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        return AIMessage(content="MOCK REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def _market_state():
    return {
        "trade_date": date.today().isoformat(),
        "company_of_interest": "BTCUSDT",
        "asset_type": "crypto_perp",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


@pytest.mark.unit
def test_market_analyst_injects_bundle_when_enabled(monkeypatch):
    from yialpha.agents.analysts.market_analyst import create_market_analyst
    from yialpha.dataflows.config import set_config

    set_config({"perp_market_bundle": True})
    monkeypatch.setattr(
        pb, "fetch_perp_market_bundle", lambda s, d: _full_bundle(),
    )
    monkeypatch.setattr(pb, "render_perp_bundle_block", lambda b: "SENTINEL_BLOCK")

    llm = _PromptCaptureLLM()
    create_market_analyst(llm)(_market_state())
    rendered = str(llm.prompt)
    assert "SENTINEL_BLOCK" in rendered
    assert "deterministic prefetch" in rendered


@pytest.mark.unit
def test_market_analyst_bundle_off_by_default():
    from yialpha.agents.analysts.market_analyst import create_market_analyst

    # conftest's _perp_bundle_off_by_default fixture holds the config off;
    # the prompt must not carry a bundle section.
    llm = _PromptCaptureLLM()
    create_market_analyst(llm)(_market_state())
    assert "Perp Market Bundle" not in str(llm.prompt)
