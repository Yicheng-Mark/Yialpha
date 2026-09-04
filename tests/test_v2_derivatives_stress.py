"""V2.0 P0.4 — Derivatives Stress Score (pure compute + fail-open fetcher)."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from yialpha.dataflows import binance as bn
from yialpha.risk.derivatives_stress import (
    MIN_WINDOW,
    DerivativesStressReport,
    compute_stress,
    render_stress_line,
)


def _rising(n: int = 60, start: float = 1.0, step: float = 0.01) -> pd.Series:
    return pd.Series([start + step * i for i in range(n)])


def _flat(n: int = 60, value: float = 1.0) -> pd.Series:
    return pd.Series([value] * n)


def _spike(n: int = 60, base: float = 100.0, spike: float = 200.0) -> pd.Series:
    return pd.Series([base] * (n - 1) + [spike])


@pytest.mark.unit
def test_rising_components_score_extreme_long_crowd():
    report = compute_stress(
        funding=_rising(),
        global_lsr=_rising(),
        basis=_rising(),
        open_interest=_spike(),
        taker_ratio=1.4,
    )
    assert report.crowding_score >= 95
    assert report.state["funding"] == "extreme_long"
    assert report.state["basis"] == "elevated_premium"
    assert report.state["taker"] == "buy"
    assert "crowded_long" in report.risk_flags
    assert "funding_expensive" in report.risk_flags
    assert "oi_elevated" in report.risk_flags
    assert report.missing == []


@pytest.mark.unit
def test_falling_components_score_extreme_short_crowd():
    report = compute_stress(
        funding=_rising().iloc[::-1].reset_index(drop=True),
        global_lsr=_rising().iloc[::-1].reset_index(drop=True),
        basis=_rising().iloc[::-1].reset_index(drop=True),
        taker_ratio=0.7,
    )
    assert report.crowding_score <= 5
    assert report.state["funding"] == "extreme_short"
    assert report.state["taker"] == "sell"
    assert "crowded_short" in report.risk_flags
    assert "funding_favorable_shorts" in report.risk_flags


@pytest.mark.unit
def test_flat_series_is_neutral():
    # n=90 (full window, no thin_history flag); mid-rank percentile of a
    # perfectly flat window is the neutral 50, not a degenerate 100.
    report = compute_stress(
        funding=_flat(n=90, value=0.0001),
        global_lsr=_flat(n=90, value=1.0),
        basis=_flat(n=90, value=0.0),
        open_interest=_flat(n=90, value=100.0),
        taker_ratio=1.0,
    )
    assert 40 <= report.crowding_score <= 60
    assert report.state["funding"] == "neutral"
    assert report.state["taker"] == "balanced"
    assert not report.risk_flags


@pytest.mark.unit
def test_short_window_drops_component_instead_of_scoring_noise():
    report = compute_stress(funding=_rising(n=MIN_WINDOW - 1))
    assert "funding_pct" in report.missing
    assert "funding_pct" not in report.components
    # No direction components left -> honest neutral + explicit flag.
    assert report.crowding_score == 50
    assert "insufficient_history" in report.risk_flags


@pytest.mark.unit
def test_nothing_available_is_neutral_with_disclosure():
    report = compute_stress()
    assert isinstance(report, DerivativesStressReport)
    assert report.crowding_score == 50
    assert "insufficient_history" in report.risk_flags
    assert set(report.missing) >= {"funding_pct", "lsr_pct", "basis_pct", "taker_ratio"}
    assert report.state == {
        "funding": "unknown", "oi": "unknown", "basis": "unknown",
        "taker": "unknown",
    }


@pytest.mark.unit
def test_thin_window_flags():
    # 60 points < TARGET_WINDOW(90) but >= MIN_WINDOW(30): scored + flagged.
    report = compute_stress(funding=_rising(n=60))
    assert "funding_pct" in report.components
    assert "thin_history" in report.risk_flags


@pytest.mark.unit
def test_partial_components_still_score():
    # funding + LSR only (basis unavailable): weighted over what exists.
    report = compute_stress(funding=_rising(), global_lsr=_rising())
    assert report.crowding_score >= 95
    assert "basis_pct" in report.missing
    assert "basis_pct" not in report.components
    assert report.state["basis"] == "unknown"


@pytest.mark.unit
def test_oi_zscore_zero_std_does_not_crash():
    report = compute_stress(open_interest=_flat(value=100.0))
    # Constant series: std == 0 -> component dropped, no crash.
    assert "oi_zscore" in report.missing


@pytest.mark.unit
def test_render_line_carries_score_states_and_flags():
    report = compute_stress(
        funding=_rising(), global_lsr=_rising(), basis=_rising(),
        open_interest=_rising(), taker_ratio=1.3,
    )
    line = render_stress_line(report)
    assert line.startswith("- **Derivatives Stress**: crowding ")
    assert "/100" in line
    assert "funding extreme_long" in line
    assert "taker buy" in line
    assert "crowded_long" in line
    assert line.endswith("\n")


@pytest.mark.unit
def test_fetcher_fail_open_on_total_transport_failure(monkeypatch):
    from yialpha.dataflows import binance as bn

    def _explode(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(bn, "_http_get", _explode)
    monkeypatch.setattr(bn, "binance_klines_frame", _explode)
    out = bn.derivatives_stress_series("BTCUSDT", "2026-08-18")
    # Every component None — the report degrades honestly, never raises.
    assert set(out) == {"funding", "open_interest", "global_lsr", "basis", "taker_ratio"}
    assert all(v is None for v in out.values())


@pytest.mark.unit
def test_fetcher_builds_series_from_records(monkeypatch):
    from yialpha.dataflows import binance as bn

    def _fake_http(path, params, *args, **kwargs):  # noqa: ANN001
        if path == "/fapi/v1/fundingRate":
            return [
                {"fundingTime": 1_700_000_000_000, "fundingRate": "0.0001"},
                {"fundingTime": 1_700_288_000_000, "fundingRate": "0.0002"},
            ]
        if path == "/futures/data/openInterestHist":
            return [
                {"timestamp": 1_700_000_000_000, "sumOpenInterest": "100"},
                {"timestamp": 1_700_288_000_000, "sumOpenInterest": "110"},
            ]
        if path == "/futures/data/globalLongShortAccountRatio":
            return [
                {"timestamp": 1_700_000_000_000, "longShortRatio": "1.5"},
                {"timestamp": 1_700_288_000_000, "longShortRatio": "1.6"},
            ]
        if path == "/futures/data/takerlongshortRatio":
            return [
                {"timestamp": 1_700_288_000_000, "buySellRatio": "1.2"},
            ]
        raise AssertionError(f"unexpected path {path}")

    def _fake_klines(symbol, start, end, interval="1d", venue="binance_perp",
                     price_type="last"):
        n = 40
        base = 100.0 if price_type == "last" else 99.0
        idx = pd.date_range("2026-06-01", periods=n, freq="D")
        # REAL vendor schema: binance_klines_frame returns the capitalised
        # KLINE_CLOSE_COLUMN ("Close"). A lowercase mock matched the old
        # perp["close"] KeyError bug and hid it for the component's whole
        # life — the schema lock below is the regression pin.
        return pd.DataFrame(
            {bn.KLINE_CLOSE_COLUMN: [base + i * 0.1 for i in range(n)]},
            index=idx,
        )

    monkeypatch.setattr(bn, "_http_get", _fake_http)
    monkeypatch.setattr(bn, "binance_klines_frame", _fake_klines)
    out = bn.derivatives_stress_series("BTCUSDT", "2026-08-18")
    assert len(out["funding"]) == 2
    assert out["funding"].iloc[-1] == pytest.approx(0.0002)
    assert out["open_interest"].iloc[-1] == pytest.approx(110.0)
    assert out["global_lsr"].iloc[-1] == pytest.approx(1.6)
    assert out["taker_ratio"] == pytest.approx(1.2)
    # basis = perp/index − 1 > 0 on these synthetic frames.
    assert (out["basis"] > 0).all()


_FUTURES_DATA_STRESS_PATHS = (
    "/futures/data/openInterestHist",
    "/futures/data/globalLongShortAccountRatio",
)


class _Capture:
    """Fake ``_http_get`` recording every ``(path, params)`` verbatim.

    The ``_fake_http`` stub above is params-blind — that is exactly how the
    live ``-1130`` and the 29-row off-by-one below both slipped through: a
    stub that ignores its arguments cannot show the wrong window was asked
    for. This one records, and the tests pin the exact values.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(
        self,
        path: str,
        params: dict,
        symbol_for_error: str,
        canonical: str,
        base: str = bn._FAPI_BASE,
        weight_key: str = "fapi",
        headers: dict | None = None,
    ) -> object:
        self.calls.append((path, dict(params)))
        return []

    def params_for(self, path: str) -> dict:
        return next(p for pth, p in self.calls if pth == path)


class _KlinesCapture:
    """Fake ``binance_klines_frame`` recording its window; serves no candles."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        venue: str = "binance_perp",
        price_type: str = "last",
        *,
        closed_as_of: int | None = None,
    ) -> pd.DataFrame:
        self.calls.append((start_date, end_date))
        return pd.DataFrame()


def _end_of_day_ms(end_date: str) -> int:
    """Mirror the fetcher's own ``end_ms``: UTC end-of-day, inclusive."""
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    return int((end_dt.timestamp() + 86399) * 1000)


@pytest.mark.unit
def test_futures_data_start_clamped_to_retention_window(monkeypatch):
    """Regression pin for the live -1130 ``parameter 'startTime' is invalid``.

    ``/futures/data/*`` retains only the last 30 days AND rejects a wider
    startTime/endTime span with HTTP 400 — it does NOT silently truncate. The
    caller's default window is 90 days, so the data layer must clamp the two
    ``/futures/data/*`` stress requests into the retention window, while the
    full-history funding endpoint keeps the whole 90 days.

    Doubles as the PIT pin: ``end_date`` here is in the PAST, so ``end_ms`` is
    already below now and the now-clamp must be a no-op — a replay window is
    never dragged forward to the real present.
    """
    cap = _Capture()
    klines = _KlinesCapture()
    monkeypatch.setattr(bn, "_http_get", cap)
    monkeypatch.setattr(bn, "binance_klines_frame", klines)

    # Historical replay: 5 days back from the real clock, computed at run time.
    end_dt = (datetime.now(UTC) - timedelta(days=5)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    end_date = end_dt.strftime("%Y-%m-%d")
    end_ms = _end_of_day_ms(end_date)
    start_ms = int((end_dt - timedelta(days=90)).timestamp() * 1000)
    assert end_ms < int(time.time() * 1000)  # a replay window is strictly past

    bn.derivatives_stress_series("BTCUSDT", end_date)

    retention_ms = bn._FUTURES_DATA_RETENTION_DAYS * 86_400_000
    for path in _FUTURES_DATA_STRESS_PATHS:
        params = cap.params_for(path)
        assert params["endTime"] == end_ms  # NOT dragged to now — PIT intact
        assert params["startTime"] == end_ms - retention_ms
        # The endpoint hard-rejects (400 -1130) a span beyond retention.
        assert params["endTime"] - params["startTime"] <= retention_ms
        assert params["period"] == "1d"
        assert params["limit"] == 30

    # Untouched siblings: funding is the full-history /fapi endpoint (90 days
    # stays legal there), taker only looks back 2 days, basis goes via klines.
    funding = cap.params_for("/fapi/v1/fundingRate")
    assert funding["startTime"] == start_ms
    assert funding["endTime"] == end_ms
    assert funding["endTime"] - funding["startTime"] > retention_ms

    taker = cap.params_for("/futures/data/takerlongshortRatio")
    assert taker["startTime"] == end_ms - 2 * 86_400_000
    assert taker["endTime"] == end_ms
    assert taker["limit"] == 2

    window_start = (end_dt - timedelta(days=90)).strftime("%Y-%m-%d")
    assert klines.calls == [(window_start, end_date)] * 2


@pytest.mark.unit
def test_futures_data_end_clamped_to_now_when_local_date_leads_utc(monkeypatch):
    """The off-by-one that made live OI/LSR arrive as 29 rows (< MIN_WINDOW 30).

    ``end_ms`` is the LOCAL end-of-day. Past local midnight in UTC+8 the local
    date is already tomorrow while UTC is still today, so ``end_ms`` lands hours
    in the FUTURE; measuring the retention window back from that future instant
    pushes ``startTime`` past a daily row that already exists, and the server
    returns 29 of the 30 rows the score needs. Clamping ``endTime`` to now
    re-anchors the window on the real present and the 30th row comes back — and
    it keeps the repo's PIT discipline of never asking for future data.
    """
    cap = _Capture()
    klines = _KlinesCapture()
    monkeypatch.setattr(bn, "_http_get", cap)
    monkeypatch.setattr(bn, "binance_klines_frame", klines)

    end_date = date.today().isoformat()
    end_ms = _end_of_day_ms(end_date)
    # Reproduce the observed live condition: the requested window ends 30h
    # after the real present (local date one day ahead of the UTC date).
    now_ms = end_ms - 30 * 3_600_000
    monkeypatch.setattr(bn, "_now_ms", lambda: now_ms)

    bn.derivatives_stress_series("BTCUSDT", end_date)

    retention_ms = bn._FUTURES_DATA_RETENTION_DAYS * 86_400_000
    for path in _FUTURES_DATA_STRESS_PATHS:
        params = cap.params_for(path)
        assert params["endTime"] == now_ms  # clamped off the future end-of-day
        assert params["endTime"] < end_ms
        assert params["startTime"] == now_ms - retention_ms
        # Exactly the retention horizon: wide enough for 30 daily rows, never
        # wide enough to trip the 400 -1130 span rejection.
        assert params["endTime"] - params["startTime"] == retention_ms

    # The clamp is scoped to the two retention-limited series: funding and
    # taker keep the PIT ``end_ms`` anchor they had before.
    funding = cap.params_for("/fapi/v1/fundingRate")
    assert funding["endTime"] == end_ms
    assert funding["startTime"] == end_ms - (90 * 86_400_000 + 86_399_000)
    taker = cap.params_for("/futures/data/takerlongshortRatio")
    assert taker["endTime"] == end_ms
    assert taker["startTime"] == end_ms - 2 * 86_400_000
    assert klines.calls[0][1] == end_date
