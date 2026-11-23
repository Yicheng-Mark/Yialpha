"""Round-15 branch hardening: yialpha.dataflows.binance + binance_vision.

Targets the residual uncovered blocks from the fresh coverage report (these
are money-path market-data modules: perp pricing, funding cadence, stress
series, vision archive). Every test pins BEHAVIOR — exact rendered numbers,
exact request params, exact error semantics — not just line execution.

Hermetic: no network, no sleeps. The HTTP seam is monkeypatched at the same
levels the existing suite uses:

* ``binance._http_get`` for vendor-function tests (the test_binance_perp_
  rest_extras.py pattern);
* ``binance._do_request`` for _http_get's own body parsing (the in-200
  error-body pattern);
* ``binance.requests.get`` / ``binance_vision.requests.get`` only where the
  function under test lives below _http_get (the resilience-test pattern);
* ``binance._paginate_history`` is left REAL — pagination itself is covered
  by test_binance_pagination.py.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from yialpha.dataflows import binance as bn, binance_vision as bnv
from yialpha.dataflows.config import set_config
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError

_DAY_MS = 86_400_000


def _days_ago(n: int) -> str:
    return (datetime.now(UTC) - timedelta(days=n)).strftime("%Y-%m-%d")


def _today_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _ts_at(days_ago: int, hour: int) -> int:
    dt = datetime.now(UTC) - timedelta(days=days_ago)
    return int(dt.replace(hour=hour, minute=0, second=0, microsecond=0).timestamp() * 1000)


def _kline(open_ms: int, close: str = "100.5") -> list:
    return [
        open_ms, "100", "101", "99", close, "12.5",
        open_ms + _DAY_MS - 1, "0", 0, "0", "0", "0",
    ]


def _kline_rows(count: int, close: str = "100.5") -> list[list]:
    base = _ts_at(count, 0)
    return [_kline(base + i * _DAY_MS, close) for i in range(count)]


def _csv_body(out: str) -> str:
    """The CSV payload under the '# ...' header block."""
    return out.split("\n\n", 1)[1]


class _HttpRecorder:
    """Fake ``_http_get`` recording (path, params, kwargs); answers per-path."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, path, params, symbol, canonical, **kwargs):
        self.calls.append((path, dict(params), dict(kwargs)))
        answer = self.routes.get(path, [])
        if callable(answer):
            return answer(path, params)
        return answer

    def paths(self) -> list[str]:
        return [c[0] for c in self.calls]


class _FakeResp:
    """Minimal requests.Response stand-in (status/text/headers/json)."""

    def __init__(self, status_code: int, text: str, headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers if headers is not None else {}

    def json(self):
        return json.loads(self.text)


# ===========================================================================
# binance._validate_outbound_url (missing 300-317)
# ===========================================================================


@pytest.mark.unit
def test_round15_validate_outbound_url_refuses_bad_schemes_hosts_and_private_ips():
    """The literal SSRF guard: every refusal branch raises ValueError naming
    the refusal class, while public DNS names and public IPs stay allowed."""
    refusals = {
        # 300-301: urlsplit itself raises (invalid IPv6 literal).
        "https://[::1": "unparseable outbound URL",
        # 303: non-HTTP(S) scheme.
        "ftp://fapi.binance.com/x": "non-HTTP(S) outbound URL refused",
        # 306: no host at all.
        "https://user@/x": "without a host refused",
        # 308: localhost / *.localhost / *.internal names.
        "https://localhost/x": "non-public outbound host refused",
        "https://api.localhost/x": "non-public outbound host refused",
        "https://box.internal/x": "non-public outbound host refused",
        # 313-317: literal private / loopback / link-local / unspecified IPs.
        "https://192.168.1.5/x.zip": "non-public outbound host refused",
        "https://10.0.0.1/x.zip": "non-public outbound host refused",
        "https://172.16.0.1/x.zip": "non-public outbound host refused",
        "https://169.254.1.1/x.zip": "non-public outbound host refused",
        "https://0.0.0.0/x.zip": "non-public outbound host refused",
        "https://[::1]/x.zip": "non-public outbound host refused",
    }
    for url, reason in refusals.items():
        with pytest.raises(ValueError, match=re.escape(reason)):
            bn._validate_outbound_url(url)
    # The two allowed shapes: ordinary DNS name and a public literal IP.
    bn._validate_outbound_url("https://fapi.binance.com/fapi/v1/klines")
    bn._validate_outbound_url("https://8.8.8.8/x")


# ===========================================================================
# binance._http_get body parsing (missing 515, 519-521)
# ===========================================================================


@pytest.mark.unit
def test_round15_http_get_empty_body_raises_no_data(monkeypatch):
    monkeypatch.setattr(
        bn, "_do_request", lambda *_a, **_k: _FakeResp(200, "   "),
    )
    with pytest.raises(NoMarketDataError, match="empty response body"):
        bn._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")


@pytest.mark.unit
def test_round15_http_get_non_json_body_raises_no_data(monkeypatch):
    monkeypatch.setattr(
        bn, "_do_request", lambda *_a, **_k: _FakeResp(200, "<html>not json</html>"),
    )
    with pytest.raises(NoMarketDataError, match="non-JSON response"):
        bn._http_get("/fapi/v1/klines", {"symbol": "BTCUSDT"}, "BTCUSDT", "BTCUSDT")


# ===========================================================================
# binance_klines_frame malformed rows + spot venue branch (592, 648, 674)
# ===========================================================================


@pytest.mark.unit
def test_round15_klines_frame_skips_malformed_rows(monkeypatch):
    """Non-list rows, dicts and short arrays are dropped; real klines keep
    their exact exchange-precision values."""
    good = _kline_rows(3)
    rows = ["not-a-kline", {"open": 1}, [_ts_at(5, 0), "1"], *good]
    monkeypatch.setattr(bn, "_http_get", lambda *a, **k: rows)
    df = bn.binance_klines_frame("BTCUSDT", _days_ago(6), _today_iso())
    assert len(df) == 3
    assert list(df["Close"]) == [100.5] * 3
    assert list(df["Volume"]) == [12.5] * 3


@pytest.mark.unit
def test_round15_klines_frame_all_rows_malformed_raises(monkeypatch):
    monkeypatch.setattr(
        bn, "_http_get", lambda *a, **k: ["junk", {"a": 1}, [_ts_at(5, 0)]],
    )
    with pytest.raises(NoMarketDataError, match="no parseable klines"):
        bn.binance_klines_frame("BTCUSDT", _days_ago(6), _today_iso())


# ===========================================================================
# get_binance_spot_klines (592 via binance_klines_frame + 1753-1762)
# ===========================================================================


@pytest.mark.unit
def test_round15_spot_klines_vendor_serves_csv_on_spot_host(monkeypatch):
    rec = _HttpRecorder({"/api/v3/klines": _kline_rows(6)})
    monkeypatch.setattr(bn, "_http_get", rec)
    out = bn.get_binance_spot_klines("BTCUSDT", _days_ago(6), _today_iso())
    # Routed to the SPOT host and the "spot" weight budget, path /api/v3.
    assert rec.paths() == ["/api/v3/klines"]
    _path, params, kwargs = rec.calls[0]
    assert kwargs.get("base") == "https://api.binance.com"
    assert kwargs.get("weight_key") == "spot"
    assert params["symbol"] == "BTCUSDT" and params["interval"] == "1d"
    # Rendered artifact: spot header + untouched exchange-precision prices.
    assert out.startswith("# Spot USDT klines for BTCUSDT")
    assert "# Total records: 6\n" in out
    assert "Date,Open,High,Low,Close,Adj Close,Volume" in out
    assert ",100.0,101.0,99.0,100.5,100.5,12.5" in out


# ===========================================================================
# Funding: cadence helpers + window errors (756, 780-781, 823, 851-852, 867)
# ===========================================================================


@pytest.mark.unit
def test_round15_funding_cadence_hours_none_below_two_settlements():
    # 756: fewer than two settlements to diff -> None (no invented cadence).
    assert bn._funding_cadence_hours([]) is None
    assert bn._funding_cadence_hours([{"fundingTime": 1_700_000_000_000}]) is None
    # Sanity: two settlements 4h apart still infer 4h.
    t0 = 1_700_000_000_000
    assert bn._funding_cadence_hours(
        [{"fundingTime": t0}, {"fundingTime": t0 + 4 * 3_600_000}]
    ) == 4


@pytest.mark.unit
def test_round15_funding_interval_hours_authoritative_bad_value_is_none(monkeypatch):
    """780-781: an unparseable fundingIntervalHours degrades to None (spacing
    inference stays the fallback) instead of crashing the funding tool."""
    monkeypatch.setattr(
        bn, "_http_get",
        lambda *a, **k: [{"symbol": "XYZUSDT", "fundingIntervalHours": "four"}],
    )
    assert bn._funding_interval_hours_authoritative("XYZUSDT") is None


@pytest.mark.unit
def test_round15_funding_rate_empty_window_raises(monkeypatch):
    # 823: the exchange served no rows for the window.
    monkeypatch.setattr(
        bn, "_http_get", lambda *a, **k: [],
    )
    with pytest.raises(NoMarketDataError, match="no funding rates between"):
        bn.get_binance_funding_rate("XYZUSDT", "2026-08-01", "2026-08-02")


@pytest.mark.unit
def test_round15_funding_rate_single_settlement_uses_fundinginfo_cadence(monkeypatch):
    """851-852: with only ONE settlement the spacing is unmeasurable, so the
    cadence line falls back to the fundingInfo endpoint — stated as such, with
    the annualisation multiplier derived from it (4h -> 6.0 settlements/day)."""
    t0 = int(datetime(2026, 8, 1, tzinfo=UTC).timestamp() * 1000)
    rows = [{"fundingTime": t0, "fundingRate": "0.0001", "symbol": "XYZUSDT"}]

    def fake(path, params, symbol, canonical, **kwargs):
        if path == "/fapi/v1/fundingInfo":
            return [{"symbol": "XYZUSDT", "fundingIntervalHours": 4}]
        return rows

    monkeypatch.setattr(bn, "_http_get", fake)
    out = bn.get_binance_funding_rate("XYZUSDT", "2026-08-01", "2026-08-02")
    assert "settles every ~4h on this contract (fundingInfo endpoint" in out
    assert "annualised carry = mean rate x 6.0 x 365" in out
    assert "fundingTime,fundingRate,symbol" in _csv_body(out)
    assert "2026-08-01 00:00:00,0.0001,XYZUSDT" in _csv_body(out)


@pytest.mark.unit
def test_round15_funding_rate_no_parseable_rows_raises(monkeypatch):
    """867: rows came back but none carries a usable fundingTime. The null
    fundingTime on the page tail also exercises the pagination guard: the
    walk must stop on the malformed cursor (typed no-data downstream), never
    crash with a raw TypeError."""
    monkeypatch.setattr(
        bn, "_http_get",
        lambda *a, **k: [{"fundingRate": "0.1"}, "junk", {"fundingTime": None}],
    )
    with pytest.raises(NoMarketDataError, match="no parseable funding rates"):
        bn.get_binance_funding_rate("XYZUSDT", "2026-08-01", "2026-08-02")


# ===========================================================================
# Open interest row hygiene (942)
# ===========================================================================


@pytest.mark.unit
def test_round15_open_interest_skips_malformed_history_rows(monkeypatch):
    """942: non-dict rows and rows without a timestamp are skipped silently;
    the live 'latest' row is still appended for a daily window reaching now."""
    valid = {
        "timestamp": _ts_at(1, 0),
        "sumOpenInterest": "111",
        "sumOpenInterestValue": "222",
    }
    rows = [valid, {"no_timestamp": 1}, "junk", {"timestamp": None, "sumOpenInterest": "9"}]

    def fake(path, params, symbol, canonical, **kwargs):
        if path == "/fapi/v1/openInterest":
            return {"openInterest": "1234"}
        return rows

    monkeypatch.setattr(bn, "_http_get", fake)
    out = bn.get_binance_open_interest("BTCUSDT", 5)
    body = _csv_body(out)
    lines = [ln for ln in body.splitlines() if ln]
    assert lines[0] == "time,openInterest,openInterestValue"
    assert len(lines) == 3  # CSV header + the valid history row + live snapshot
    assert lines[1] == f"{_days_ago(1)},111,222"
    assert lines[2] == "latest,1234,"


# ===========================================================================
# Long/short ratio: series hygiene + retention disclosure (1281, 1284, 1324)
# ===========================================================================


@pytest.mark.unit
def test_round15_lsr_skips_non_list_series_and_malformed_rows(monkeypatch):
    """1281 + 1284: a series answering a non-list body contributes nothing
    (and is NOT reported as unavailable — it did not fail); junk rows inside
    a list are skipped; healthy series still serve their values."""
    valid_ts = _ts_at(1, 0)

    def fake(path, params, symbol, canonical, **kwargs):
        if path == "/futures/data/topLongShortAccountRatio":
            return {"unexpected": "dict body"}  # 1281
        if path == "/futures/data/topLongShortPositionRatio":
            return [  # 1284: junk rows around one valid row
                "junk", {"no_ts": 1}, {"timestamp": None},
                {
                    "timestamp": valid_ts, "longAccount": "0.6",
                    "longShortRatio": "1.5", "shortAccount": "0.4",
                },
            ]
        return [
            {
                "timestamp": valid_ts, "longAccount": "0.55",
                "longShortRatio": "1.22", "shortAccount": "0.45",
            },
        ]

    monkeypatch.setattr(bn, "_http_get", fake)
    out = bn.get_binance_long_short_ratio("BTCUSDT", 5)
    assert "unavailable series" not in out  # nothing raised, nothing failed
    df = pd.read_csv(io.StringIO(_csv_body(out)))
    assert sorted(set(df["series"])) == ["global_account", "top_position"]
    top = df[df["series"] == "top_position"].iloc[0]
    assert (top["longAccount"], top["longShortRatio"], top["shortAccount"]) == (
        0.6, 1.5, 0.4,
    )


@pytest.mark.unit
def test_round15_lsr_windowed_header_discloses_retention_clamp(monkeypatch):
    """1324: a window whose head predates the 30-day retention horizon is
    clamped AND disclosed in the header — the dropped head must never read as
    continuous coverage."""
    captured: list[dict] = []

    def fake(path, params, symbol, canonical, **kwargs):
        if path == "/futures/data/topLongShortAccountRatio":
            captured.append(dict(params))
        return [
            {
                "timestamp": _ts_at(3, 0), "longAccount": "0.6",
                "longShortRatio": "1.5", "shortAccount": "0.4",
            },
        ]

    monkeypatch.setattr(bn, "_http_get", fake)
    out = bn.get_binance_long_short_ratio(
        "BTCUSDT", 7, start_date=_days_ago(45), end_date=_days_ago(2),
    )
    assert "windowed request" in out
    assert (
        f"/futures/data retains only the last {bn._FUTURES_DATA_RETENTION_DAYS} "
        "days: coverage starts" in out
    )
    assert "day(s) dropped from the head" in out
    # The request head was actually clamped to the retained tail.
    assert captured and captured[0]["startTime"] >= bn._now_ms() - 31 * _DAY_MS


# ===========================================================================
# Taker buy/sell volume (1365-1426 — the whole tool was execution-only via
# the invalid-period test before)
# ===========================================================================


def _taker_row(days_ago: int, hour: int, ratio: str = "1.2") -> dict:
    return {
        "timestamp": _ts_at(days_ago, hour),
        "buySellRatio": ratio, "buyVol": "10", "sellVol": "8",
    }


@pytest.mark.unit
def test_round15_taker_buy_sell_daily_rows_pinned(monkeypatch):
    monkeypatch.setattr(
        bn, "_http_get",
        lambda *a, **k: [_taker_row(1, 0), _taker_row(2, 0, "0.9")],
    )
    out = bn.get_binance_taker_buy_sell("BTCUSDT", 3)
    assert "(last 3 1d rows)" in out
    df = pd.read_csv(io.StringIO(_csv_body(out)))
    assert list(df.columns) == ["time", "buySellRatio", "buyVol", "sellVol"]
    # Rows keep the exchange's own order (oldest-first page), values verbatim.
    assert list(df["time"]) == [_days_ago(1), _days_ago(2)]
    assert list(df["buySellRatio"]) == [1.2, 0.9]
    assert list(df["buyVol"]) == [10, 10]
    assert list(df["sellVol"]) == [8, 8]


@pytest.mark.unit
def test_round15_taker_buy_sell_intraday_windowed_trims_and_labels(monkeypatch):
    """Intraday period scales the limit (look_back x rows-per-day), the
    windowed header names the served window, timestamps render at HH:MM
    grain and rows past the window end are trimmed client-side."""

    def fake(path, params, symbol, canonical, **kwargs):
        return [
            _taker_row(3, 0, "1.1"),
            _taker_row(0, 12, "9.9"),   # past the window end -> trimmed
            "junk-row",                  # skipped by the record parser
        ]

    monkeypatch.setattr(bn, "_http_get", fake)
    out = bn.get_binance_taker_buy_sell(
        "BTCUSDT", 2, start_date=_days_ago(45), end_date=_days_ago(2), period="1h",
    )
    assert "windowed request, 1 rows served" in out
    assert f"window through {_days_ago(2)}" in out
    # 44 days x 24 hourly rows > the 500-row endpoint ceiling: the request is
    # end-anchored and the dropped head is DISCLOSED in the header.
    assert "truncated to the last 500 rows" in out
    assert "day(s) dropped from the head" in out
    df = pd.read_csv(io.StringIO(_csv_body(out)))
    assert len(df) == 1
    assert df["time"].iloc[0] == f"{_days_ago(3)} 00:00"
    assert df["buySellRatio"].iloc[0] == 1.1
    # Unwindowed intraday call: limit = look_back_days * rows-per-day (2 x 24).
    unwindowed_params: list[dict] = []

    def fake_empty(path, params, symbol, canonical, **kwargs):
        unwindowed_params.append(dict(params))
        return []

    monkeypatch.setattr(bn, "_http_get", fake_empty)
    with pytest.raises(NoMarketDataError, match="no taker buy/sell volume"):
        bn.get_binance_taker_buy_sell("BTCUSDT", 2, period="1h")
    assert unwindowed_params[0]["limit"] == 48
    assert unwindowed_params[0]["period"] == "1h"


@pytest.mark.unit
def test_round15_taker_buy_sell_empty_raises(monkeypatch):
    monkeypatch.setattr(bn, "_http_get", lambda *a, **k: [])
    with pytest.raises(NoMarketDataError, match="no taker buy/sell volume"):
        bn.get_binance_taker_buy_sell("BTCUSDT", 7)


# ===========================================================================
# Basis (1467 intraday limit + 1493-1541 success path)
# ===========================================================================


def _basis_row(days_ago: int) -> dict:
    return {
        "timestamp": _ts_at(days_ago, 0),
        "basis": "5", "futuresPrice": "100",
        "indexPrice": "95", "basisRate": "0.05",
    }


@pytest.mark.unit
def test_round15_basis_daily_rows_pinned(monkeypatch):
    """1493-1541: values flow verbatim (5 / 100 / 95 / 0.05) into the CSV,
    junk rows are skipped, BOTH header forms are honest about what was
    served, and a retention-clamped head is disclosed in the header."""
    monkeypatch.setattr(
        bn, "_http_get",
        lambda *a, **k: ["junk", {"no_ts": 1}, _basis_row(2), _basis_row(3)],
    )
    # Windowed form whose head predates the 30-day retention horizon: the
    # start is clamped AND the dropped head is disclosed.
    out = bn.get_binance_basis(
        "BTCUSDT", 3, start_date=_days_ago(45), end_date=_days_ago(2),
    )
    assert "windowed request, 2 rows served" in out
    assert f"window through {_days_ago(2)}" in out
    assert "retains only the last 30 days: coverage starts" in out
    assert "day(s) dropped from the head" in out
    df = pd.read_csv(io.StringIO(_csv_body(out)), dtype=str)
    assert list(df["time"]) == [_days_ago(2), _days_ago(3)]
    assert list(df["basis"]) == ["5", "5"]
    assert list(df["futuresPrice"]) == ["100", "100"]
    assert list(df["indexPrice"]) == ["95", "95"]
    assert list(df["basisRate"]) == ["0.05", "0.05"]
    # Unwindowed form: "(last N 1d rows)" with the capped daily limit.
    out2 = bn.get_binance_basis("BTCUSDT", 3)
    assert "(last 3 1d rows)" in out2


@pytest.mark.unit
def test_round15_basis_intraday_limit_scales_with_period(monkeypatch):
    captured: list[dict] = []

    def fake(path, params, symbol, canonical, **kwargs):
        captured.append(dict(params))
        return [_basis_row(1)]

    monkeypatch.setattr(bn, "_http_get", fake)
    out = bn.get_binance_basis("BTCUSDT", 2, period="1h")
    assert "period: 1h" not in out  # basis has no period header line
    assert captured[0]["period"] == "1h"
    assert captured[0]["limit"] == 2 * 24  # 1467: rows-per-day scaled


@pytest.mark.unit
def test_round15_basis_empty_rows_raises(monkeypatch):
    monkeypatch.setattr(bn, "_http_get", lambda *a, **k: [])
    with pytest.raises(NoMarketDataError, match=r"no basis \(look_back_days=7\)"):
        bn.get_binance_basis("BTCUSDT", 7)


# ===========================================================================
# Premium index numeric edges (1575, 1578-1579)
# ===========================================================================


@pytest.mark.unit
def test_round15_premium_index_missing_and_unparseable_fields_render_none(monkeypatch):
    """A missing indexPrice (None) and an unparseable markPrice both degrade
    to None fields — markVsIndexPct is only computed from real numbers."""
    monkeypatch.setattr(
        bn, "_http_get", lambda *a, **k: {"symbol": "XUSDT", "markPrice": "50100.5"},
    )
    out = bn.get_binance_premium_index("XUSDT")
    df = pd.read_csv(io.StringIO(_csv_body(out)), dtype=str)
    assert df["markPrice"].iloc[0] == "50100.5"
    assert pd.isna(df["markVsIndexPct"].iloc[0])
    assert pd.isna(df["indexPrice"].iloc[0])
    assert pd.isna(df["nextFundingTime"].iloc[0])

    monkeypatch.setattr(
        bn, "_http_get",
        lambda *a, **k: {"symbol": "YUSDT", "markPrice": "abc", "indexPrice": "100"},
    )
    out2 = bn.get_binance_premium_index("YUSDT")
    df2 = pd.read_csv(io.StringIO(_csv_body(out2)), dtype=str)
    assert df2["markPrice"].iloc[0] == "abc"
    assert pd.isna(df2["markVsIndexPct"].iloc[0])


# ===========================================================================
# Depth snapshot level hygiene (1665, 1668-1669, 1675)
# ===========================================================================


@pytest.mark.unit
def test_round15_depth_snapshot_skips_malformed_levels(monkeypatch):
    payload = {
        "bids": [["100.0", "2.0"], ["100.5"], ["bad", "qty"], "junk", ["99.5", "3.0"]],
        "asks": [["100.2", "1.0"], ["100.3", "4.0"]],
    }
    monkeypatch.setattr(bn, "_http_get", lambda *a, **k: payload)
    out = bn.get_binance_depth_snapshot("BTCUSDT", 5)
    data = [ln for ln in _csv_body(out).splitlines() if ln]
    assert data[0] == "level,bid_price,bid_qty,ask_price,ask_qty"
    # Only the parseable levels made the ladder: 2 bids x 2 asks.
    assert data[1:] == ["1,100.0,2.0,100.2,1.0", "2,99.5,3.0,100.3,4.0"]
    assert "mid=100.1" in out


@pytest.mark.unit
def test_round15_depth_snapshot_all_levels_unparseable_raises(monkeypatch):
    monkeypatch.setattr(
        bn, "_http_get",
        lambda *a, **k: {"bids": [["bad", "qty"]], "asks": [["x", "y"]]},
    )
    with pytest.raises(NoMarketDataError, match="depth levels unparseable"):
        bn.get_binance_depth_snapshot("BTCUSDT", 5)


# ===========================================================================
# get_binance_spot_ticker24 (1774-1803)
# ===========================================================================


@pytest.mark.unit
def test_round15_spot_ticker24_vendor_row_pinned(monkeypatch):
    """The 24h spot ticker serves exactly the six documented fields verbatim,
    on the spot host and the "spot" weight budget."""
    rec = _HttpRecorder({
        "/api/v3/ticker/24hr": {
            "lastPrice": "42000.10",
            "priceChangePercent": "1.23",
            "highPrice": "43000.00",
            "lowPrice": "41000.00",
            "volume": "1234.50",
            "quoteVolume": "51000000.00",
        },
    })
    monkeypatch.setattr(bn, "_http_get", rec)
    out = bn.get_binance_spot_ticker24("BTCUSDT")
    assert rec.paths() == ["/api/v3/ticker/24hr"]
    _path, params, kwargs = rec.calls[0]
    assert kwargs.get("base") == "https://api.binance.com"
    assert kwargs.get("weight_key") == "spot"
    assert params == {"symbol": "BTCUSDT"}
    assert out.startswith("# Spot USDT 24h ticker for BTCUSDT")
    df = pd.read_csv(io.StringIO(_csv_body(out)), dtype=str)
    assert list(df.columns) == [
        "lastPrice", "priceChangePercent", "highPrice", "lowPrice",
        "volume", "quoteVolume",
    ]
    assert df.iloc[0].tolist() == [
        "42000.10", "1.23", "43000.00", "41000.00", "1234.50", "51000000.00",
    ]


@pytest.mark.unit
def test_round15_spot_ticker24_non_object_body_raises(monkeypatch):
    monkeypatch.setattr(bn, "_http_get", lambda *a, **k: ["not", "a", "dict"])
    with pytest.raises(NoMarketDataError, match="non-object body"):
        bn.get_binance_spot_ticker24("BTCUSDT")


# ===========================================================================
# Spot-perp basis: explicit start + malformed close rows (1865, 1906, 1910-1911)
# ===========================================================================


@pytest.mark.unit
def test_round15_spot_perp_basis_explicit_start_skips_malformed_rows(monkeypatch):
    perp_close, spot_close = "100.5", "100.0"

    def fake(path, params, symbol, canonical, **kwargs):
        rows = []
        for d in (3, 2, 1):
            open_ms = _ts_at(d, 0)
            rows.append(_kline(open_ms, perp_close if "fapi" in path else spot_close))
        rows.append([_ts_at(0, 0)])                      # 1906: too short
        rows.append(_kline(_ts_at(0, 0), "not-a-number"))  # 1910-1911: bad float
        return rows

    monkeypatch.setattr(bn, "_http_get", fake)
    out = bn.get_binance_spot_perp_basis(
        "BTCUSDT", 7, start_date=_days_ago(3), end_date=_days_ago(1),
    )
    df = pd.read_csv(io.StringIO(_csv_body(out)))
    # 1865: the explicit start anchored the window — only those 3 dates join.
    assert list(df["date"]) == [_days_ago(3), _days_ago(2), _days_ago(1)]
    # Pinned money-path numbers: basis 0.5, basisRate 0.005 at 6 sig digits.
    assert list(df["perpClose"]) == [100.5] * 3
    assert list(df["spotClose"]) == [100.0] * 3
    assert list(df["basis"]) == [0.5] * 3
    assert list(df["basisRate"]) == [0.005] * 3


# ===========================================================================
# warm_equity_perp_bases listing details (2166, 2175-2176)
# ===========================================================================


@pytest.mark.unit
def test_round15_warm_listing_first_row_wins_and_overflow_onboard_is_none(monkeypatch):
    """The USDT/USDC twin of a base contributes ONE listing entry (first row
    wins), and an out-of-range onboardDate degrades to None rather than
    poisoning the warm."""
    onboard_ms = int(datetime(2026, 1, 15, tzinfo=UTC).timestamp() * 1000)
    twin_ms = int(datetime(2025, 6, 1, tzinfo=UTC).timestamp() * 1000)
    payload = {"symbols": [
        {  # first AAPL row: USDT twin, onboard 2026-01-15
            "symbol": "AAPLUSDT", "quoteAsset": "USDT",
            "underlyingType": "EQUITY", "status": "TRADING",
            "onboardDate": onboard_ms,
        },
        {  # 2166: USDC twin skipped — the first listing row per base wins.
            "symbol": "AAPLUSDC", "quoteAsset": "USDC",
            "underlyingType": "EQUITY", "status": "TRADING",
            "onboardDate": twin_ms,
        },
        {  # 2175-2176: onboardDate beyond datetime range -> OSError -> None.
            "symbol": "TSLAUSDT", "quoteAsset": "USDT",
            "underlyingType": "EQUITY", "status": "TRADING",
            "onboardDate": 253402300800000,
        },
    ]}
    bn.refresh_equity_perp_bases()
    try:
        monkeypatch.setattr(bn, "_http_get", lambda *a, **k: payload)
        bases = bn.warm_equity_perp_bases()
        assert bases == frozenset({"AAPL", "TSLA"})
        listing = bn.equity_perp_listing_info()
        assert set(listing) == {"AAPL", "TSLA"}  # the twin added no second row
        assert listing["AAPL"]["onboard_date"] == "2026-01-15"
        assert listing["TSLA"]["onboard_date"] is None
        assert listing["AAPL"]["status"] == "TRADING"
    finally:
        bn.refresh_equity_perp_bases()


# ===========================================================================
# stock_perp_underlying: registry-evidence branches (2237-2248)
# ===========================================================================


@pytest.fixture()
def _registry_mode():
    set_config({"instrument_registry": True})
    bn.refresh_equity_perp_bases()  # deterministic seed snapshot while warm
    import yialpha.instruments.registry as reg

    yield reg
    bn.refresh_equity_perp_bases()


def _record(underlying: str | None, klass: str = "stock_perp") -> SimpleNamespace:
    return SimpleNamespace(
        classification_source="binance_exchangeinfo",
        instrument_class=klass,
        underlying_symbol=underlying,
    )


@pytest.mark.unit
def test_round15_stock_perp_underlying_registry_row_maps_yahoo_alias(
    _registry_mode, monkeypatch,
):
    """2240-2242: a registry exchangeInfo row is authoritative and maps its
    base through the single Yahoo alias seam (BRKB -> BRK-B), overriding the
    warm/seed answer."""
    monkeypatch.setattr(_registry_mode, "classify_perp", lambda *a, **k: _record("BRKB"))
    assert bn.stock_perp_underlying("AAPLUSDT") == "BRK-B"


@pytest.mark.unit
def test_round15_stock_perp_underlying_registry_derives_base_from_suffix(
    _registry_mode, monkeypatch,
):
    """2237: a stock_perp row whose underlying_symbol is empty still resolves
    by stripping the quote suffix — promoting a base the seed never listed.
    A quote-only ticker ("USDT" itself) has no derivable base and keeps the
    warm answer."""
    monkeypatch.setattr(_registry_mode, "classify_perp", lambda *a, **k: _record(""))
    assert bn.stock_perp_underlying("ZZZZUSDT") == "ZZZZ"
    # len("USDT") == 4: the suffix-strip guard refuses an empty base.
    assert bn.stock_perp_underlying("USDT") is None


@pytest.mark.unit
def test_round15_stock_perp_underlying_registry_failure_serves_warm(
    _registry_mode, monkeypatch,
):
    """2243-2248: a broken registry consult can never break the hot path —
    the warm/seed answer is served (AAPL is in the static seed)."""
    def boom(*_a, **_k):
        raise RuntimeError("registry down")

    monkeypatch.setattr(_registry_mode, "classify_perp", boom)
    assert bn.stock_perp_underlying("AAPLUSDT") == "AAPL"


# ===========================================================================
# derivatives stress series hygiene (2272, 2275-2276, 2370-2372)
# ===========================================================================


@pytest.mark.unit
def test_round15_stress_series_from_records_drops_unusable_rows():
    rows = [
        {"fundingTime": 1_700_000_000_000, "fundingRate": "0.0001"},
        {"fundingTime": None, "fundingRate": "0.0002"},   # 2272: no timestamp
        {"fundingTime": 1_700_000_000_500, "fundingRate": None},  # 2272: no value
        {"fundingTime": 1_700_000_001_000, "fundingRate": "abc"},  # 2275-2276
    ]
    s = bn._stress_series_from_records(rows, "fundingTime", "fundingRate")
    assert len(s) == 1
    assert s.iloc[0] == 0.0001


@pytest.mark.unit
def test_round15_derivatives_stress_trims_future_rows_keeps_unparseable(monkeypatch):
    """The belt-and-suspenders endTime trim (nested _trim_to_end): a vendor
    that ignores endTime must not leak future rows past the analysis date,
    while rows with unparseable timestamps pass through to the record parser
    (which drops them by shape)."""
    def fake_http(path, params, symbol, canonical, **kwargs):
        if path == "/fapi/v1/fundingRate":
            return [
                {"fundingTime": _ts_at(3, 0), "fundingRate": "0.0001"},
                {"fundingTime": _ts_at(2, 12), "fundingRate": "0.0002"},
            ]
        if path == "/futures/data/openInterestHist":
            return [
                {"timestamp": _ts_at(3, 0), "sumOpenInterest": "1000"},
                {"timestamp": _ts_at(0, 12), "sumOpenInterest": "9999"},  # future
                "junk-row",  # unparseable timestamp -> kept by trim, dropped later
            ]
        if path == "/futures/data/globalLongShortAccountRatio":
            return [
                {"timestamp": _ts_at(3, 0), "longShortRatio": "1.5"},
                {"timestamp": _ts_at(0, 12), "longShortRatio": "9.9"},
            ]
        if path == "/futures/data/takerlongshortRatio":
            return [
                {"timestamp": _ts_at(2, 0), "buySellRatio": "1.25"},
                {"timestamp": _ts_at(0, 12), "buySellRatio": "8.8"},
            ]
        raise AssertionError(f"unexpected path {path}")

    idx = pd.DatetimeIndex(
        [pd.Timestamp(_ts_at(3, 0), unit="ms", tz="UTC"),
         pd.Timestamp(_ts_at(2, 0), unit="ms", tz="UTC")],
    )
    perp_df = pd.DataFrame({"Close": [101.0, 102.0]}, index=idx)
    index_df = pd.DataFrame({"Close": [100.0, 100.0]}, index=idx)

    def fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                    price_type="last", **_k):
        return perp_df if price_type == "last" else index_df

    monkeypatch.setattr(bn, "_http_get", fake_http)
    monkeypatch.setattr(bn, "binance_klines_frame", fake_klines)
    out = bn.derivatives_stress_series("BTCUSDT", _days_ago(2), window_days=30)
    # Future rows trimmed: each /futures/data series carries exactly one row.
    assert out["open_interest"] is not None and len(out["open_interest"]) == 1
    assert out["open_interest"].iloc[0] == 1000.0
    assert out["global_lsr"] is not None and len(out["global_lsr"]) == 1
    assert out["global_lsr"].iloc[0] == 1.5
    assert out["taker_ratio"] == 1.25
    # Funding (full-history endpoint) keeps both settlements.
    assert len(out["funding"]) == 2
    # Basis from the two pinned frames: 101/100-1, 102/100-1.
    assert list(out["basis"]) == pytest.approx([0.01, 0.02])


# ===========================================================================
# binance_vision: URL guard + _fetch status mapping (150-151, 168-181)
# ===========================================================================


@pytest.mark.unit
def test_round15_vision_url_unparseable_rejected():
    """150-151: a URL urlsplit itself cannot parse is refused before any
    request — with the vision-specific message."""
    with pytest.raises(ValueError, match="unparseable outbound URL"):
        bnv._validate_vision_url("https://[::1")


class _VisionResp:
    def __init__(self, status_code: int, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content


@pytest.mark.unit
def test_round15_vision_fetch_maps_statuses_and_refuses_redirects(monkeypatch):
    """_fetch is the archive transport floor: 200 serves bytes, 404 raises the
    missing-day signal carrying the file name, 429/418 raise the vendor
    rate-limit error, any other non-200 raises the typed archive error, and
    redirects are structurally disabled on the request."""
    seen_kwargs: dict = {}

    def fake_get(url, **kwargs):
        seen_kwargs.update(kwargs)
        table = {
            "https://data.binance.vision/ok.zip": _VisionResp(200, content=b"zipbytes"),
            "https://data.binance.vision/missing.zip": _VisionResp(404),
            "https://data.binance.vision/banned.zip": _VisionResp(429),
            "https://data.binance.vision/ipbanned.zip": _VisionResp(418),
            "https://data.binance.vision/boom.zip": _VisionResp(503, text="overloaded"),
        }
        return table[url]

    monkeypatch.setattr(bnv.requests, "get", fake_get)
    good = "https://data.binance.vision/ok.zip"
    assert bnv._fetch(good) == b"zipbytes"
    assert seen_kwargs["allow_redirects"] is False
    assert seen_kwargs["timeout"] == bnv._TIMEOUT
    with pytest.raises(bnv._ArchiveMissingError, match="missing.zip"):
        bnv._fetch("https://data.binance.vision/missing.zip")
    with pytest.raises(VendorRateLimitError, match="HTTP 429"):
        bnv._fetch("https://data.binance.vision/banned.zip")
    with pytest.raises(VendorRateLimitError, match="HTTP 418"):
        bnv._fetch("https://data.binance.vision/ipbanned.zip")
    with pytest.raises(bnv._ArchiveHttpError, match="HTTP 503: overloaded"):
        bnv._fetch("https://data.binance.vision/boom.zip")

    # A non-archive URL is refused BEFORE any request fires.
    def no_call(*_a, **_k):
        raise AssertionError("request must not fire for a refused URL")

    monkeypatch.setattr(bnv.requests, "get", no_call)
    with pytest.raises(ValueError, match="refusing non-archive outbound URL"):
        bnv._fetch("https://evil.example.com/ok.zip")


# ===========================================================================
# binance_vision: cache + zip member hygiene (244, 254, 256-257)
# ===========================================================================


@pytest.mark.unit
def test_round15_vision_load_zip_none_from_cache_raises(monkeypatch):
    monkeypatch.setattr(bnv, "cached_or_fetch", lambda *a, **k: None)
    with pytest.raises(bnv._ArchiveHttpError, match="cache returned no bytes"):
        bnv._load_zip("https://data.binance.vision/data/x/BTCUSDT-metrics-2024-05-01.zip")


@pytest.mark.unit
def test_round15_vision_zip_csv_dataframe_rejects_memberless_and_bad_zips():
    # 254: a zip with no .csv member.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no csv here")
    with pytest.raises(bnv._ArchiveHttpError, match="no CSV member"):
        bnv._zip_csv_dataframe(buf.getvalue(), "metrics")
    # 256-257: not a zip at all.
    with pytest.raises(bnv._ArchiveHttpError, match="not a readable zip"):
        bnv._zip_csv_dataframe(b"definitely not a zip", "metrics")
    # 256-257 via the UnicodeDecodeError twin: valid zip, non-UTF-8 CSV bytes.
    buf2 = io.BytesIO()
    with zipfile.ZipFile(buf2, "w") as zf:
        zf.writestr("data.csv", b"\xff\xfe\x00binary-garbage")
    with pytest.raises(bnv._ArchiveHttpError, match="not a readable zip"):
        bnv._zip_csv_dataframe(buf2.getvalue(), "metrics")


# ===========================================================================
# binance_vision: shaping refusals + summary edges (414, 462, 520, 601)
# ===========================================================================


def _ts_series(*iso: str) -> pd.Series:
    return pd.Series([pd.Timestamp(s, tz="UTC") for s in iso])


@pytest.mark.unit
def test_round15_vision_shape_metrics_missing_columns_raises():
    df = pd.DataFrame({"foo": [1], "_ts": _ts_series("2024-05-01T00:00:00")})
    with pytest.raises(NoMarketDataError, match="none of the expected columns"):
        bnv._shape_metrics(df, "1d", "BTCUSDT", "BTCUSDT")


@pytest.mark.unit
def test_round15_vision_shape_depth_missing_value_columns_raises():
    df = pd.DataFrame(
        {"percentage": [1.0], "_ts": _ts_series("2024-05-01T00:00:00")},
    )
    with pytest.raises(NoMarketDataError, match="neither depth nor notional"):
        bnv._shape_book_depth(df, "1d", "BTCUSDT", "BTCUSDT")


@pytest.mark.unit
def test_round15_vision_coverage_notes_disclose_missing_edges():
    """A known-missing START or END day is disclosed as an edge (not an
    interior hole) — the backoff map keeps it out of the fresh attempt set,
    so the call-scoped report alone would go quiet about it."""
    report = SimpleNamespace(
        missing_days=None, unsynced=0, synced_days_count=2, window_days=3, budget=400,
    )
    qa = {"known_missing_days": ["2024-05-01", "2024-05-03"]}
    start = datetime(2024, 5, 1, tzinfo=UTC)
    end = datetime(2024, 5, 3, 23, 59, 59, tzinfo=UTC)
    note = bnv._coverage_notes(report, qa, start, end)
    assert "window start 2024-05-01 and end 2024-05-03 day(s) not published yet" in note
    # The start-only variant appends just the start edge.
    qa_start_only = {"known_missing_days": ["2024-05-01"]}
    note2 = bnv._coverage_notes(report, qa_start_only, start, end)
    assert "window start 2024-05-01 day(s) not published yet" in note2
    assert "end 2024-05-03" not in note2


@pytest.mark.unit
def test_round15_vision_depth_summary_skips_all_nan_notional_band():
    """A signed band whose notional is entirely NaN is skipped in the per-band
    distribution (it has nothing to summarize) while healthy bands render."""
    shaped = pd.DataFrame({
        "time": ["2024-05-01"] * 2 + ["2024-05-02"] * 2,
        "percentage": [-1.0, -1.0, 1.0, 1.0],
        "depth": [10.0, 11.0, 20.0, 21.0],
        "notional": [None, None, 1000.0, 3000.0],
    })
    out = bnv._render_depth_summary(shaped)
    assert "## Per-band distribution over the full window" in out
    lines = out.splitlines()
    start = lines.index("## Per-band distribution over the full window (daily means)")
    block: list[str] = []
    for ln in lines[start + 1:]:
        if ln.startswith("##"):
            break
        block.append(ln)
    assert len(block) == 2  # CSV header + the one healthy band only
    assert block[1].startswith("1.0,")
    assert not any(row.startswith("-1.0,") for row in block)


# ===========================================================================
# binance_vision: tool-level no-rows / wrap paths (724, 823, 828)
# ===========================================================================


def _patch_loader(monkeypatch, df=None, exc: Exception | None = None):
    def fake_loader(dataset, symbol, start_dt, end_dt, symbol_for_error, canonical):
        if exc is not None:
            raise exc
        report = SimpleNamespace(
            missing_days=None, unsynced=0, synced_days_count=1,
            window_days=1, budget=400,
        )
        return df, report, {}

    monkeypatch.setattr(bnv, "_load_window_from_store", fake_loader)


@pytest.mark.unit
def test_round15_vision_metrics_all_nan_rows_raise_no_rows(monkeypatch):
    """Synced rows whose metric values are ALL missing shape down to an empty
    frame — that must read as the instructive no-rows error, never as an
    empty artifact."""
    df = pd.DataFrame({
        "create_time": ["2024-05-01 00:00:00"],
        "symbol": ["BTCUSDT"],
        "sum_open_interest": [None],
        "sum_open_interest_value": [None],
        "count_long_short_ratio": [None],
        "sum_taker_long_short_vol_ratio": [None],
        "_ts": _ts_series("2024-05-01T00:00:00"),
    })
    _patch_loader(monkeypatch, df=df)
    with pytest.raises(NoMarketDataError, match="no metrics rows in"):
        bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-02")


@pytest.mark.unit
def test_round15_vision_book_depth_archive_error_wrapped(monkeypatch):
    _patch_loader(monkeypatch, exc=bnv._ArchiveHttpError("HTTP 503: overloaded"))
    with pytest.raises(NoMarketDataError, match="bookDepth unavailable: HTTP 503"):
        bnv.get_binance_vision_book_depth("BTCUSDT", "2024-05-01", "2024-05-02")


@pytest.mark.unit
def test_round15_vision_book_depth_all_bands_dropped_raise_no_rows(monkeypatch):
    """Rows whose signed percentage is unparseable are dropped by the shaper;
    a fully-dropped window raises the no-rows error instead of serving an
    empty CSV."""
    df = pd.DataFrame({
        "timestamp": ["2024-05-01 00:00:00"],
        "percentage": [None],
        "depth": ["10"],
        "notional": ["100"],
        "_ts": _ts_series("2024-05-01T00:00:00"),
    })
    _patch_loader(monkeypatch, df=df)
    with pytest.raises(NoMarketDataError, match="no bookDepth rows in"):
        bnv.get_binance_vision_book_depth("BTCUSDT", "2024-05-01", "2024-05-02")
