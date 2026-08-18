"""data.binance.vision archive tools (deep-history perp metrics/depth).

Hermetic: no network — ``_fetch`` is monkeypatched with an in-memory archive
server (zip + sha256 CHECKSUM), and the disk cache is redirected to a tmp
dir. These tests pin the contract that matters for trust:

* every served zip is sha256-verified — mismatched/missing/malformed
  checksums fail CLOSED (NoMarketDataError, never unverified bytes);
* the file plan is one daily zip per UTC day (these datasets publish no
  monthly aggregates — verified via S3 listing 2026-08-17);
* a 404 inside the window (pre-listing / unpublished day) is disclosed in
  the header — an interior hole must not silently render as a continuous
  series;
* PIT correctness: the window is clamped to the pinned analysis date and to
  the last published archive day (publication lags ~1 day);
* downloaded files are cached — a second call issues no new fetches;
* output shaping: day-close open interest vs day-mean ratios, per-band depth
  means, row caps that refuse instead of silently truncating;
* vendor registration + route_to_vendor pass-through.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from yiagents.dataflows import binance_vision as bnv
from yiagents.dataflows.errors import NoMarketDataError
from yiagents.dataflows.utils import set_analysis_date

# Extended fixture schema: the LIVE metrics schema (verified 2026-08-17) has
# 8 columns and NO sum_long_short_ratio — the global-ratio column here
# intentionally exercises the optional older/future column mapping;
# test_verified_real_schema_without_global_ratio pins the real 8-column one.
_METRICS_HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_long_short_ratio,"
    "count_taker_long_short_vol_ratio,sum_taker_long_short_vol_ratio"
)


def _metrics_csv(rows: list[dict]) -> str:
    lines = [_METRICS_HEADER]
    for r in rows:
        lines.append(
            ",".join(
                str(r.get(k, "")) for k in (
                    "create_time", "symbol", "sum_open_interest",
                    "sum_open_interest_value", "count_toptrader_long_short_ratio",
                    "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                    "sum_long_short_ratio", "count_taker_long_short_vol_ratio",
                    "sum_taker_long_short_vol_ratio",
                )
            )
        )
    return "\n".join(lines) + "\n"


def _metrics_row(day: str, hhmm: str, oi: float, top: float, glob: float, taker: float):
    return {
        "create_time": f"{day} {hhmm}:00", "symbol": "BTCUSDT",
        "sum_open_interest": oi, "sum_open_interest_value": oi * 100.0,
        "count_toptrader_long_short_ratio": 1, "sum_toptrader_long_short_ratio": top,
        "count_long_short_ratio": 1, "sum_long_short_ratio": glob,
        "count_taker_long_short_vol_ratio": 1,
        "sum_taker_long_short_vol_ratio": taker,
    }


def _depth_csv(rows: list[dict]) -> str:
    header = "timestamp,percentage,depth,notional"
    lines = [header]
    for r in rows:
        lines.append(f"{r['timestamp']},{r['percentage']},{r['depth']},{r['notional']}")
    return "\n".join(lines) + "\n"


def _zip_bytes(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", csv_text)
    return buf.getvalue()


class _FakeArchive:
    """In-memory data.binance.vision: URL -> payload, auto-CHECKSUM, 404s."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.fetches: list[str] = []
        self.corrupt_checksums: set[str] = set()
        self.drop_checksums: set[str] = set()

    def add(self, url: str, payload: bytes) -> None:
        self.files[url] = payload
        name = url.rsplit("/", 1)[-1]
        self.files[url + ".CHECKSUM"] = (
            f"{hashlib.sha256(payload).hexdigest()}  {name}\n".encode()
        )

    def __call__(self, url: str) -> bytes:
        self.fetches.append(url)
        payload = self.files.get(url)
        if payload is None:
            raise bnv._ArchiveMissingError(url.rsplit("/", 1)[-1])
        if url.endswith(".CHECKSUM"):
            fname = url[: -len(".CHECKSUM")].rsplit("/", 1)[-1]
            if url in self.corrupt_checksums:
                return b"0" * 64 + f"  {fname}\n".encode()
            if url in self.drop_checksums:
                raise bnv._ArchiveMissingError(fname)
        return payload

    def zip_fetches(self) -> list[str]:
        return [u for u in self.fetches if not u.endswith(".CHECKSUM")]


@pytest.fixture()
def archive(monkeypatch, tmp_path):
    server = _FakeArchive()
    cache_dir = tmp_path / "binance_vision"
    cache_dir.mkdir()
    monkeypatch.setattr(bnv, "_fetch", server)
    monkeypatch.setattr(bnv, "vendor_cache_dir", lambda name: str(cache_dir))
    return server


def _daily_url(dataset: str, symbol: str, day: str) -> str:
    # Dash naming, verified against the live S3 bucket 2026-08-17
    # (the binance-public-data docs' underscore names do not match reality).
    return (
        f"https://data.binance.vision/data/futures/um/daily/{dataset}/{symbol}/"
        f"{symbol}-{dataset}-{day}.zip"
    )


def _add_metrics_day(server, symbol: str, day: str, rows: list[dict]) -> None:
    server.add(_daily_url("metrics", symbol, day), _zip_bytes(_metrics_csv(rows)))


def _add_depth_day(server, symbol: str, day: str, rows: list[dict]) -> None:
    server.add(_daily_url("bookDepth", symbol, day), _zip_bytes(_depth_csv(rows)))


# ---- file plan ---------------------------------------------------------------


@pytest.mark.unit
def test_archive_plan_is_one_daily_file_per_day():
    """These datasets publish daily files only (no monthly aggregates exist
    — verified via S3 listing 2026-08-17), so the plan is exactly one dash-
    named zip per UTC day, inclusive of both window edges."""
    start = datetime(2024, 1, 3, tzinfo=UTC)
    end = datetime(2024, 3, 10, 23, 59, 59, tzinfo=UTC)
    plan = [url for url, _fname in bnv._archive_plan("metrics", "BTCUSDT", start, end)]
    assert _daily_url("metrics", "BTCUSDT", "2024-01-03") in plan
    assert _daily_url("metrics", "BTCUSDT", "2024-03-10") in plan
    assert _daily_url("metrics", "BTCUSDT", "2024-01-02") not in plan
    assert _daily_url("metrics", "BTCUSDT", "2024-03-11") not in plan
    assert len(plan) == 29 + 29 + 10  # Jan 3-31 + Feb 2024 (leap) + Mar 1-10


@pytest.mark.unit
def test_validate_vision_url_rejects_non_archive_targets():
    for bad in (
        "http://data.binance.vision/data/x.zip",
        "https://evil.example.com/BTCUSDT_metrics_2024-01.zip",
        "https://127.0.0.1/BTCUSDT_metrics_2024-01.zip",
        "https://localhost/data/x.zip",
        "file:///etc/passwd",
    ):
        with pytest.raises(ValueError):
            bnv._validate_vision_url(bad)
    bnv._validate_vision_url(_daily_url("metrics", "BTCUSDT", "2024-01-01"))


# ---- checksum fail-closed -----------------------------------------------------


@pytest.mark.unit
def test_checksum_mismatch_never_serves(archive):
    day = "2024-05-01"
    _add_metrics_day(archive, "BTCUSDT", day, [_metrics_row(day, "00:00", 1, 1, 1, 1)])
    url = _daily_url("metrics", "BTCUSDT", day)
    archive.corrupt_checksums.add(url + ".CHECKSUM")
    with pytest.raises(NoMarketDataError, match="metrics unavailable.*mismatch|sha256"):
        bnv.get_binance_vision_metrics("BTCUSDT", day, day)


@pytest.mark.unit
def test_missing_checksum_file_never_serves(archive):
    day = "2024-05-01"
    _add_metrics_day(archive, "BTCUSDT", day, [_metrics_row(day, "00:00", 1, 1, 1, 1)])
    url = _daily_url("metrics", "BTCUSDT", day)
    archive.drop_checksums.add(url + ".CHECKSUM")
    with pytest.raises(NoMarketDataError, match="CHECKSUM"):
        bnv.get_binance_vision_metrics("BTCUSDT", day, day)


@pytest.mark.unit
def test_malformed_checksum_rejected():
    with pytest.raises(bnv._ChecksumError):
        bnv._expected_sha256(b"not-a-hash  file.zip", "file.zip")
    with pytest.raises(bnv._ChecksumError):
        bnv._expected_sha256(b"", "file.zip")
    good = hashlib.sha256(b"x").hexdigest()
    assert bnv._expected_sha256(f"{good}  file.zip\n".encode(), "file.zip") == good


# ---- metrics shaping ----------------------------------------------------------


@pytest.mark.unit
def test_metrics_daily_resample_close_oi_mean_ratios(archive):
    for day, oi_seq in (
        ("2024-05-01", (100.0, 200.0, 400.0)),
        ("2024-05-02", (500.0, 700.0, 900.0)),
    ):
        rows = [
            _metrics_row(day, hhmm, oi, top, glob, taker)
            for hhmm, oi, top, glob, taker in (
                ("00:00", oi_seq[0], 1.1, 2.0, 0.9),
                ("08:00", oi_seq[1], 1.3, 2.2, 1.1),
                ("16:00", oi_seq[2], 1.5, 2.4, 1.3),
            )
        ]
        _add_metrics_day(archive, "BTCUSDT", day, rows)
    out = bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-02")
    df = pd.read_csv(io.StringIO(out.split("\n\n", 1)[1]))
    assert list(df["time"]) == ["2024-05-01", "2024-05-02"]
    # OI is a LEVEL: the daily row is the day CLOSE (last 5m snapshot).
    assert list(df["open_interest"]) == [400.0, 900.0]
    # Ratios are INTENSITIES: the daily row is the day MEAN.
    assert list(df["top_trader_long_short_ratio"]) == pytest.approx([1.3, 1.3])
    assert list(df["global_long_short_ratio"]) == pytest.approx([2.2, 2.2])
    assert list(df["taker_buy_sell_ratio"]) == pytest.approx([1.1, 1.1])
    assert "data.binance.vision" in out


@pytest.mark.unit
def test_metrics_raw_5m_rows_preserved(archive):
    day = "2024-05-01"
    _add_metrics_day(
        archive, "BTCUSDT", day,
        [_metrics_row(day, hh, i, 1.0, 1.0, 1.0) for i, hh in enumerate(("00:00", "00:05"))],
    )
    out = bnv.get_binance_vision_metrics("BTCUSDT", day, day, interval="5m")
    df = pd.read_csv(io.StringIO(out.split("\n\n", 1)[1]))
    assert list(df["time"]) == ["2024-05-01 00:00", "2024-05-01 00:05"]


@pytest.mark.unit
def test_verified_real_schema_without_global_ratio(archive):
    """The LIVE schema (verified 2026-08-17) has 8 columns and NO
    sum_long_short_ratio — the global ratio value is not in the archive. The
    shaper must serve the columns that exist without failing."""
    day = "2024-05-01"
    csv_text = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,"
        "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
        "count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2024-05-01 00:00:00,BTCUSDT,100.0,500000.0,2.4,1.39,2.6,1.05\n"
        "2024-05-01 23:55:00,BTCUSDT,300.0,900000.0,2.2,1.31,2.5,0.68\n"
    )
    archive.add(_daily_url("metrics", "BTCUSDT", day), _zip_bytes(csv_text))
    out = bnv.get_binance_vision_metrics("BTCUSDT", day, day)
    df = pd.read_csv(io.StringIO(out.split("\n\n", 1)[1]))
    assert list(df.columns) == [
        "open_interest", "open_interest_value",
        "top_trader_long_short_ratio", "taker_buy_sell_ratio", "time",
    ]
    assert df["open_interest"].iloc[0] == 300.0  # day CLOSE
    assert df["taker_buy_sell_ratio"].iloc[0] == pytest.approx((1.05 + 0.68) / 2)


@pytest.mark.unit
def test_all_missing_raises_instructive_no_data(archive):
    with pytest.raises(NoMarketDataError, match="no metrics archive files"):
        bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-02")


@pytest.mark.unit
def test_row_caps_refuse_instead_of_truncating(archive, monkeypatch):
    monkeypatch.setattr(bnv, "_MAX_OUTPUT_ROWS", 1)
    for day in ("2024-05-01", "2024-05-02"):
        _add_metrics_day(archive, "BTCUSDT", day, [_metrics_row(day, "00:00", 1, 1, 1, 1)])
    with pytest.raises(NoMarketDataError, match="cap"):
        bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-02")
    monkeypatch.setattr(bnv, "_MAX_RAW_ROWS", 1)
    # A fresh day (not the cached ones above) so the raw-cap path actually
    # downloads: the first call's zips are already in the disk cache.
    _add_metrics_day(
        archive, "BTCUSDT", "2024-06-01",
        [_metrics_row("2024-06-01", hh, 1, 1, 1, 1) for hh in ("00:00", "00:05")],
    )
    with pytest.raises(NoMarketDataError, match="cap"):
        bnv.get_binance_vision_metrics(
            "BTCUSDT", "2024-06-01", "2024-06-01", interval="5m",
        )


@pytest.mark.unit
def test_archive_file_count_cap(archive, monkeypatch):
    monkeypatch.setattr(bnv, "_MAX_ARCHIVE_FILES", 1)
    for day in ("2024-05-01", "2024-05-02"):
        _add_metrics_day(archive, "BTCUSDT", day, [_metrics_row(day, "00:00", 1, 1, 1, 1)])
    with pytest.raises(NoMarketDataError, match="narrow the date range"):
        bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-02")


@pytest.mark.unit
def test_invalid_interval_rejected(archive):
    with pytest.raises(ValueError, match="interval"):
        bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-02", interval="2h")
    with pytest.raises(ValueError, match="interval"):
        bnv.get_binance_vision_book_depth("BTCUSDT", "2024-05-01", "2024-05-02", interval="4h")


# ---- bookDepth shaping --------------------------------------------------------


@pytest.mark.unit
def test_book_depth_daily_per_band_means(archive):
    """Verified schema: timestamp, percentage (SIGNED: - = bid side, + = ask
    side), depth, notional. Daily output = mean per (day, band)."""
    for day in ("2024-05-01", "2024-05-02"):
        rows = []
        for hhmm, mult in (("00:00", 1.0), ("08:00", 3.0)):
            for pct in (-5.0, -1.0, 1.0, 5.0):
                rows.append({
                    "timestamp": f"{day} {hhmm}:00",
                    "percentage": pct,
                    "depth": 10.0 * mult,
                    "notional": 1000.0 * mult,
                })
        _add_depth_day(archive, "BTCUSDT", day, rows)
    out = bnv.get_binance_vision_book_depth("BTCUSDT", "2024-05-01", "2024-05-02")
    df = pd.read_csv(io.StringIO(out.split("\n\n", 1)[1]))
    # Two days x four signed bands, mean of the two ~30s snapshots (1 and 3 -> 2).
    assert len(df) == 8
    day1 = df[df["time"] == "2024-05-01"]
    assert sorted(day1["percentage"]) == [-5.0, -1.0, 1.0, 5.0]
    assert list(day1["depth"]) == pytest.approx([20.0] * 4)
    assert list(day1["notional"]) == pytest.approx([2000.0] * 4)
    # Bid-side bands (negative) sort before ask-side within each day.
    assert list(day1["percentage"]) == sorted(day1["percentage"])


@pytest.mark.unit
def test_book_depth_missing_percentage_column_raises(archive):
    day = "2024-05-01"
    bad_csv = "timestamp,symbol,bid_depth\n2024-05-01 00:00:00,BTCUSDT,1\n"
    archive.add(_daily_url("bookDepth", "BTCUSDT", day), _zip_bytes(bad_csv))
    with pytest.raises(NoMarketDataError, match="percentage"):
        bnv.get_binance_vision_book_depth("BTCUSDT", day, day)


# ---- PIT + publication lag ----------------------------------------------------


@pytest.mark.unit
def test_end_date_clamped_to_pinned_analysis_date(archive):
    set_analysis_date("2024-05-01")
    try:
        _add_metrics_day(
            archive, "BTCUSDT", "2024-05-01", [_metrics_row("2024-05-01", "00:00", 7, 1, 1, 1)],
        )
        # June requested but must never be fetched under a May-1 analysis date.
        _add_metrics_day(
            archive, "BTCUSDT", "2024-06-01", [_metrics_row("2024-06-01", "00:00", 9, 1, 1, 1)],
        )
        out = bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-06-30")
    finally:
        set_analysis_date(None)
    assert "2024-06-01" not in out
    assert all("2024-06" not in u for u in archive.zip_fetches())
    assert "to 2024-05-01" in out


@pytest.mark.unit
def test_window_entirely_unpublished_raises():
    today = datetime.now(UTC).date().isoformat()
    with pytest.raises(NoMarketDataError, match="no published archive days"):
        bnv.get_binance_vision_metrics("BTCUSDT", today, today)


@pytest.mark.unit
def test_window_reaching_into_unpublished_days_notes_truncation(archive):
    yesterday = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    _add_metrics_day(
        archive, "BTCUSDT", yesterday, [_metrics_row(yesterday, "00:00", 5, 1, 1, 1)],
    )
    today = datetime.now(UTC).date().isoformat()
    out = bnv.get_binance_vision_metrics("BTCUSDT", yesterday, today)
    assert "publication lags" in out
    assert today not in out.split("\n\n", 1)[1]


@pytest.mark.unit
def test_interior_missing_day_disclosed_in_header(archive):
    """A 404 in the MIDDLE of the window must be disclosed: the resampling
    shaper would otherwise render the hole as a continuous series."""
    for day in ("2024-05-01", "2024-05-03"):  # 2024-05-02 deliberately unpublished
        _add_metrics_day(
            archive, "BTCUSDT", day, [_metrics_row(day, "00:00", 1, 1, 1, 1)],
        )
    out = bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-03")
    assert "1 archive day(s) missing inside the window" in out
    df = pd.read_csv(io.StringIO(out.split("\n\n", 1)[1]))
    assert list(df["time"]) == ["2024-05-01", "2024-05-03"]


# ---- cache --------------------------------------------------------------------


@pytest.mark.unit
def test_second_call_served_from_cache_without_refetch(archive):
    for day in ("2024-05-01", "2024-05-02"):
        _add_metrics_day(archive, "BTCUSDT", day, [_metrics_row(day, "00:00", 1, 1, 1, 1)])
    bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-02")
    n_fetches = len(archive.zip_fetches())
    assert n_fetches > 0
    out2 = bnv.get_binance_vision_metrics("BTCUSDT", "2024-05-01", "2024-05-02")
    assert len(archive.zip_fetches()) == n_fetches  # cache hit: zero new fetches
    assert "2024-05-01" in out2


# ---- registration / router ----------------------------------------------------


@pytest.mark.unit
def test_vendor_registration_and_optional_category():
    from yiagents.dataflows.interface import (
        OPTIONAL_CATEGORIES,
        TOOLS_CATEGORIES,
        VENDOR_METHODS,
    )

    for method, impl in (
        ("get_binance_vision_metrics", bnv.get_binance_vision_metrics),
        ("get_binance_vision_book_depth", bnv.get_binance_vision_book_depth),
    ):
        assert method in VENDOR_METHODS
        assert VENDOR_METHODS[method]["binance"] is impl
        assert method in TOOLS_CATEGORIES["binance_perp"]["tools"]
    assert "binance_perp" in OPTIONAL_CATEGORIES


@pytest.mark.unit
def test_router_degrades_vision_no_data_to_sentinel(archive, monkeypatch):
    """Optional-category contract: a NoMarketDataError from the archive tool
    must surface as the instructive NO_DATA_AVAILABLE sentinel, not an abort."""
    from yiagents.dataflows.interface import route_to_vendor

    quality_calls: list[tuple] = []

    class _Q:
        @staticmethod
        def record_sentinel(*args, **kwargs):
            quality_calls.append((args, kwargs))

    import yiagents.dataflows.interface as iface

    monkeypatch.setattr(iface.quality, "record_sentinel", _Q.record_sentinel)
    out = route_to_vendor("get_binance_vision_metrics", "BTCUSDT", "2024-05-01", "2024-05-02")
    assert out.startswith("NO_DATA_AVAILABLE")
    assert quality_calls  # evidence recorded for the data_quality block
