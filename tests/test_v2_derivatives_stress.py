"""V2.0 P0.4 — Derivatives Stress Score (pure compute + fail-open fetcher)."""

from __future__ import annotations

import pandas as pd
import pytest

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
        return pd.DataFrame(
            {"close": [base + i * 0.1 for i in range(n)]}, index=idx
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
