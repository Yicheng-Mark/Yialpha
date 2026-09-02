"""V2.1-D — USDT/USD quote FX leg (live-only, fail-soft, depeg-guarded).

Pins the frozen contract: the rate is the INVERTED latest Binance SPOT
USDCUSDT daily close (reusing the existing spot-kline seam — no new HTTP
code), failures degrade to None instead of raising, a depegged print is
refused, historical ``as_of`` is unavailable (no PIT archive, and the rate
is never defaulted to 1.0), and the payload lands in the isolated vendor
cache under ``quote_fx/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

import yialpha.perp.quote_fx as qfx
from yialpha.perp.quote_fx import (
    QUOTE_FX_SOURCE,
    QuoteFxResult,
    fetch_usdt_usd,
    render_fx_line,
    usdt_usd_as_of,
)


def _today_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _fake_frame(closes: dict[str, float]) -> pd.DataFrame:
    """Date-indexed daily frame in the seam's own schema (Close column)."""
    dates = sorted(closes)
    return pd.DataFrame(
        {"Close": [closes[d] for d in dates]}, index=pd.to_datetime(dates)
    )


def _recent_closes(last_close: float) -> dict[str, float]:
    """Three daily closes ending on today's (forming) candle."""
    today = datetime.now(UTC).date()
    return {
        (today - timedelta(days=2)).strftime("%Y-%m-%d"): 0.9990,
        (today - timedelta(days=1)).strftime("%Y-%m-%d"): 0.9991,
        today.strftime("%Y-%m-%d"): last_close,
    }


def _clear_quote_fx_cache() -> None:
    """Drop the day's cached payload between in-test scenarios.

    ``cached_or_fetch`` writes a successful fetch BEFORE the band validation
    runs, so within one test a second scenario would be served the first
    scenario's (possibly refused) close from the 1-hour-TTL cache. Each test
    gets a fresh tmp cache from conftest; multi-scenario tests clear it.
    """
    import shutil

    from yialpha.dataflows.disk_cache import vendor_cache_dir

    shutil.rmtree(vendor_cache_dir("quote_fx"), ignore_errors=True)


def _seam_with_close(close: float):
    """Seam stub pinning one last close (binds now — loop-variable safe)."""

    def _seam(*args, **kwargs):
        return _fake_frame(_recent_closes(close))

    return _seam


@pytest.mark.unit
def test_rate_is_inverse_of_latest_close_on_the_spot_seam(monkeypatch):
    calls: list[tuple] = []

    def _fake_seam(symbol, start_date, end_date, interval="1d", venue="binance_perp",
                   price_type="last"):
        calls.append((symbol, start_date, end_date, interval, venue))
        return _fake_frame(_recent_closes(0.98))

    monkeypatch.setattr(qfx, "binance_klines_frame", _fake_seam)
    result = fetch_usdt_usd()
    assert result is not None
    # rate = 1 / close, pinned (close 0.98 -> ~1.020408 USD per USDT).
    assert result.rate == pytest.approx(1.0 / 0.98)
    assert result.source == QUOTE_FX_SOURCE == "binance_spot:USDCUSDT_inverse"
    assert result.replayability == "LIVE_ONLY"
    assert result.available_at == _today_utc()
    assert result.fetched_at  # ISO timestamp travels with the payload
    # The EXISTING spot kline seam is reused: USDCUSDT daily candles on the
    # spot venue, over the last ~3 days ending today.
    assert len(calls) == 1
    symbol, start_date, end_date, interval, venue = calls[0]
    assert symbol == "USDCUSDT"
    assert venue == "binance_spot"
    assert interval == "1d"
    assert end_date == _today_utc()
    assert start_date == (
        datetime.now(UTC).date() - timedelta(days=3)
    ).strftime("%Y-%m-%d")


@pytest.mark.unit
def test_depegged_close_is_refused(monkeypatch):
    for bad in (0.94, 1.06):  # outside [0.95, 1.05]: feed broken, no rate
        _clear_quote_fx_cache()
        monkeypatch.setattr(qfx, "binance_klines_frame", _seam_with_close(bad))
        assert fetch_usdt_usd() is None


@pytest.mark.unit
def test_band_edges_still_convert(monkeypatch):
    # Inclusive band: a 0.95 / 1.05 print is ugly but not a depeg signal.
    for edge in (0.95, 1.05):
        _clear_quote_fx_cache()
        monkeypatch.setattr(qfx, "binance_klines_frame", _seam_with_close(edge))
        result = fetch_usdt_usd()
        assert result is not None
        assert result.rate == pytest.approx(1.0 / edge)


@pytest.mark.unit
def test_unusable_close_returns_none(monkeypatch):
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        _clear_quote_fx_cache()
        monkeypatch.setattr(qfx, "binance_klines_frame", _seam_with_close(bad))
        assert fetch_usdt_usd() is None


@pytest.mark.unit
def test_seam_failure_degrades_to_none(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("spot endpoint down")

    monkeypatch.setattr(qfx, "binance_klines_frame", _boom)
    assert fetch_usdt_usd() is None  # fail-soft: never raises into a run


@pytest.mark.unit
def test_as_of_past_date_is_unavailable_without_a_fetch(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("live-only feed must not be fetched for a past date")

    monkeypatch.setattr(qfx, "binance_klines_frame", _boom)
    assert usdt_usd_as_of("2024-01-01") is None  # no PIT archive -> no rate
    future = (datetime.now(UTC).date() + timedelta(days=30)).strftime("%Y-%m-%d")
    assert usdt_usd_as_of(future) is None       # no observation can exist yet
    assert usdt_usd_as_of("not-a-date") is None  # unparseable degrades, no raise


@pytest.mark.unit
def test_as_of_today_delegates_to_fetch(monkeypatch):
    sentinel = QuoteFxResult(
        rate=0.9993, available_at=_today_utc(), fetched_at="2026-09-03T00:00:00+00:00"
    )
    monkeypatch.setattr(qfx, "fetch_usdt_usd", lambda: sentinel)
    assert usdt_usd_as_of(_today_utc()) is sentinel


@pytest.mark.unit
def test_payload_lands_in_isolated_vendor_cache_and_ttl_serves_it(monkeypatch,
                                                                  tmp_path):
    calls: list[int] = []

    def _fake_seam(*a, **k):
        calls.append(1)
        return _fake_frame(_recent_closes(0.9993))

    monkeypatch.setattr(qfx, "binance_klines_frame", _fake_seam)
    first = fetch_usdt_usd()
    assert first is not None
    # conftest isolates data_cache_dir to tmp; the dated payload must land
    # under quote_fx/, never in the developer's real ~/.yialpha/cache.
    cache_file = tmp_path / "vendor-cache" / "quote_fx" / f"usdcusdt_{_today_utc()}.json"
    assert cache_file.exists()
    wrapper = json.loads(cache_file.read_text(encoding="utf-8"))
    assert wrapper["close"] == pytest.approx(0.9993)
    assert wrapper["available_at"] == _today_utc()
    # Second call inside the 1-hour TTL serves the cache: no second fetch.
    second = fetch_usdt_usd()
    assert second is not None
    assert second.rate == first.rate
    assert second.fetched_at == first.fetched_at  # fetch clock travels with bytes
    assert len(calls) == 1


@pytest.mark.unit
def test_render_fx_line_both_branches():
    assert render_fx_line(None) == (
        "USDT/USD fx: UNAVAILABLE (live-only feed; not PIT-replayable)"
    )
    result = QuoteFxResult(
        rate=0.9993, available_at="2026-09-03", fetched_at="2026-09-03T09:00:00+00:00"
    )
    assert render_fx_line(result) == (
        f"USDT/USD fx: 0.999300 ({QUOTE_FX_SOURCE}, live-only)"
    )
