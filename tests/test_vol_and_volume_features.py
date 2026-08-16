"""Vol estimators (WP4) + volume-price features (WP7): formula-level tests.

Hand-computed reference values for Parkinson/GK/close-to-close on constant
bars, property tests for Yang-Zhang/EWMA, OBV/relative-volume construction,
and the deterministic divergence detector — plus the derived-feature
dispatch through every runtime consumer.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from yiagents.dataflows.vol_estimators import (
    CRYPTO_TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    close_to_close_vol,
    ewma_vol,
    garman_klass_vol,
    parkinson_vol,
    periods_per_year_for,
    yang_zhang_vol,
)
from yiagents.dataflows.volume_features import (
    obv,
    relative_volume,
    volume_divergence,
)


def _frame(closes, highs=None, lows=None, opens=None, volumes=None, n=None):
    n = n or len(closes)
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "Date": pd.bdate_range("2025-01-01", periods=n),
        "Open": closes if opens is None else pd.Series(opens, dtype=float),
        "High": closes * 1.01 if highs is None else pd.Series(highs, dtype=float),
        "Low": closes * 0.99 if lows is None else pd.Series(lows, dtype=float),
        "Close": closes,
        "Volume": [1_000.0] * n if volumes is None else pd.Series(volumes, dtype=float),
    })


@pytest.mark.unit
class TestCloseToCloseVol:
    def test_alternating_returns_hand_value(self):
        # closes alternate 100, 101 -> log returns +/-a with mean 0; the
        # sample std (ddof=1) of that window is a*sqrt(n/(n-1)).
        closes = [100.0, 101.0] * 12  # 24 rows
        vol = close_to_close_vol(_frame(closes, n=24), window=20)
        a = math.log(1.01)
        expected = a * math.sqrt(20.0 / 19.0) * math.sqrt(TRADING_DAYS_PER_YEAR)
        assert vol.iloc[-1] == pytest.approx(expected, rel=1e-9)
        assert int(vol.notna().sum()) == 4  # rows 20..23 after 1 warm-up + 20 window

    def test_constant_closes_are_zero_vol(self):
        vol = close_to_close_vol(_frame([100.0] * 40), window=20)
        assert vol.iloc[-1] == pytest.approx(0.0)


@pytest.mark.unit
class TestParkinsonVol:
    def test_constant_range_hand_value(self):
        # H/L = 1.01/0.99 every day: var = ln(1.01/0.99)^2 / (4 ln 2)
        vol = parkinson_vol(_frame([100.0] * 40), window=20)
        hl = math.log(1.01 / 0.99)
        expected = math.sqrt(hl * hl / (4.0 * math.log(2.0)) * TRADING_DAYS_PER_YEAR)
        assert vol.iloc[-1] == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
class TestGarmanKlassVol:
    def test_zero_body_hand_value(self):
        # O = C -> ln(C/O) = 0, so var = 0.5 * ln(H/L)^2.
        closes = np.full(40, 100.0)
        opens = closes.copy()
        highs = closes * 1.02
        lows = closes * 0.98
        vol = garman_klass_vol(
            _frame(closes, opens=opens, highs=highs, lows=lows), window=20
        )
        hl = math.log(1.02 / 0.98)
        expected = math.sqrt(0.5 * hl * hl * TRADING_DAYS_PER_YEAR)
        assert vol.iloc[-1] == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
class TestYangZhangVol:
    def test_nonnegative_and_finite_on_random_data(self):
        rng = np.random.default_rng(1)
        n = 300
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        frame = _frame(close, n=n)
        vol = yang_zhang_vol(frame, window=20)
        tail = vol.dropna()
        assert len(tail) > 250
        assert (tail >= 0).all() and np.isfinite(tail).all()
        # Same order of magnitude as close-to-close on this data.
        c2c = close_to_close_vol(frame, window=20).dropna()
        assert 0.3 * c2c.mean() < tail.mean() < 3.0 * c2c.mean()

    def test_too_short_window_is_all_nan(self):
        out = yang_zhang_vol(_frame([100.0] * 3), window=20)
        assert out.dropna().empty  # not enough rows: honest NaN, no values


@pytest.mark.unit
class TestEwmaVol:
    def test_shock_makes_ewma_jump_above_flat_realized(self):
        rng = np.random.default_rng(2)
        n = 200
        quiet = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
        closes = np.append(quiet, quiet[-1] * 0.90)  # one -10% shock day
        frame = _frame(closes, n=n + 1)
        ewma = ewma_vol(frame)
        rvol = close_to_close_vol(frame, window=20)
        # Right after the shock the EWMA estimate is dominated by it.
        assert ewma.iloc[-1] > rvol.iloc[-1]

    def test_recursion_matches_manual_seeding(self):
        closes = np.array([100.0, 101.0, 100.5, 102.0, 101.5])
        vol = ewma_vol(_frame(closes, n=5))
        lam = 0.94
        rets = np.log(closes[1:] / closes[:-1])
        var = rets[0] ** 2  # seed
        for r in rets[1:]:
            var = lam * var + (1 - lam) * r * r
        assert vol.iloc[-1] == pytest.approx(math.sqrt(var * TRADING_DAYS_PER_YEAR), rel=1e-9)


@pytest.mark.unit
class TestAnnualizationFactor:
    """Crypto (24/7) daily series annualize with 365, equities with 252.

    Pins the 2026-08-16 fix: the Binance indicator tools computed rvol_20 /
    ewma_vol on crypto candles through the 252 default, understating every
    vol reading by sqrt(252/365) ≈ 0.83.
    """

    def test_periods_per_year_for_asset_type(self):
        assert periods_per_year_for("crypto") == CRYPTO_TRADING_DAYS_PER_YEAR
        assert periods_per_year_for("crypto_perp") == CRYPTO_TRADING_DAYS_PER_YEAR
        assert periods_per_year_for("crypto_spot") == CRYPTO_TRADING_DAYS_PER_YEAR
        assert periods_per_year_for("stock") == TRADING_DAYS_PER_YEAR
        assert periods_per_year_for(None) == TRADING_DAYS_PER_YEAR
        assert periods_per_year_for("") == TRADING_DAYS_PER_YEAR

    def test_close_to_close_365_scales_by_sqrt_ratio(self):
        rng = np.random.default_rng(11)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
        frame = _frame(closes, n=80)
        vol252 = close_to_close_vol(frame, window=20)
        vol365 = close_to_close_vol(
            frame, window=20, periods_per_year=CRYPTO_TRADING_DAYS_PER_YEAR
        )
        ratio = float(vol365.iloc[-1] / vol252.iloc[-1])
        assert ratio == pytest.approx(math.sqrt(365.0 / 252.0), rel=1e-12)
        # Default remains the equity convention byte-for-byte.
        assert vol252.iloc[-1] == pytest.approx(
            close_to_close_vol(frame, window=20,
                               periods_per_year=TRADING_DAYS_PER_YEAR).iloc[-1],
            rel=1e-15,
        )

    def test_ewma_vol_accepts_periods_per_year(self):
        closes = [100.0, 101.0, 100.5, 102.0, 101.5]
        frame = _frame(closes, n=5)
        a = ewma_vol(frame).iloc[-1]
        b = ewma_vol(frame, periods_per_year=365.0).iloc[-1]
        assert float(b / a) == pytest.approx(math.sqrt(365.0 / 252.0), rel=1e-12)

    def test_registry_threads_factor_to_vol_features_only(self):
        from yiagents.dataflows.feature_registry import compute_derived

        rng = np.random.default_rng(12)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
        frame = _frame(closes, n=80)
        base = float(compute_derived(frame, "rvol_20").iloc[-1])
        crypto = float(
            compute_derived(frame, "rvol_20",
                            periods_per_year=CRYPTO_TRADING_DAYS_PER_YEAR).iloc[-1]
        )
        assert crypto / base == pytest.approx(math.sqrt(365.0 / 252.0), rel=1e-12)
        # Volume features are unit-free ratios: the factor must not change them.
        r1 = compute_derived(frame, "rel_vol_20")
        r2 = compute_derived(frame, "rel_vol_20", periods_per_year=365.0)
        assert float(r1.iloc[-1]) == pytest.approx(float(r2.iloc[-1]), rel=1e-15)


@pytest.mark.unit
class TestObv:
    def test_signs_and_cumulation(self):
        closes = [100.0, 101.0, 100.5, 102.0, 101.0]
        volumes = [10.0, 20.0, 30.0, 40.0, 50.0]
        series = obv(_frame(closes, volumes=volumes, n=5))
        # directions: NaN->0, +1, -1, +1, -1
        assert list(series) == pytest.approx([0.0, 20.0, -10.0, 30.0, -20.0])

    def test_flat_close_contributes_nothing(self):
        series = obv(_frame([100.0] * 5, volumes=[5.0] * 5))
        assert list(series) == [0.0] * 5


@pytest.mark.unit
class TestRelativeVolume:
    def test_latest_value_is_ratio_to_prior_mean(self):
        volumes = [100.0] * 25 + [300.0]
        rel = relative_volume(_frame([100.0] * 26, volumes=volumes), window=20)
        # Prior 20 sessions (today excluded) all 100 -> 300/100 = 3.0.
        assert rel.iloc[-1] == pytest.approx(300.0 / 100.0)
        assert rel.iloc[19] != rel.iloc[19]  # warm-up NaN (needs 20 PRIOR rows)
        assert rel.iloc[20] == pytest.approx(1.0)


@pytest.mark.unit
class TestVolumeDivergence:
    @staticmethod
    def _waves(direction: str):
        """Two-wave construction: second wave makes the new extreme on
        evaporating volume, so OBV fails to confirm it."""
        if direction == "bear":
            first = list(np.linspace(100.0, 110.0, 10))   # wave 1 up, heavy volume
            pull = list(np.linspace(110.0, 105.0, 5))     # pullback
            second = list(np.linspace(105.0, 110.5, 10))  # wave 2 up, thin volume
        else:
            first = list(np.linspace(120.0, 110.0, 10))   # wave 1 down, heavy volume
            pull = list(np.linspace(110.0, 115.0, 5))
            second = list(np.linspace(115.0, 109.5, 10))  # wave 2 down, thin volume
        closes = first + pull + second
        volumes = [1000.0] * 15 + [100.0] * 10
        return closes, volumes

    def test_bearish_divergence_detected(self):
        closes, volumes = self._waves("bear")
        result = volume_divergence(_frame(closes, volumes=volumes, n=25), window=20)
        assert result is not None
        assert result["bearish_divergence"] is True
        assert result["bullish_divergence"] is False

    def test_bullish_divergence_detected(self):
        closes, volumes = self._waves("bull")
        result = volume_divergence(_frame(closes, volumes=volumes, n=25), window=20)
        assert result is not None
        assert result["bullish_divergence"] is True
        assert result["bearish_divergence"] is False

    def test_confirmed_high_is_not_divergence(self):
        # Rising closes with the new high on the HEAVIEST volume bar: OBV
        # confirms (it also prints its window high).
        closes = list(np.linspace(100.0, 120.0, 26))
        volumes = [1000.0] * 25 + [5000.0]
        result = volume_divergence(_frame(closes, volumes=volumes), window=20)
        assert result is not None
        assert result["bearish_divergence"] is False

    def test_too_few_rows_returns_none(self):
        assert volume_divergence(_frame([100.0] * 10), window=20) is None


@pytest.mark.unit
class TestDerivedDispatchThroughConsumers:
    def test_yfinance_bulk_computes_derived_features(self, monkeypatch):
        import yiagents.dataflows.y_finance as yfin

        frame = _frame(list(np.linspace(100, 130, 60)))
        monkeypatch.setattr(yfin, "load_ohlcv", lambda s, d: frame)
        result = yfin._get_stock_stats_bulk("TEST", "rvol_20", "2025-04-01")
        values = [v for v in result.values() if v != "N/A"]
        assert len(values) > 30  # warm-up NaN then finite
        assert all(float(v) > 0 for v in values)

    def test_validator_snapshot_includes_derived(self, monkeypatch):
        import yiagents.dataflows.market_data_validator as mdv

        frame = _frame(list(np.linspace(100, 130, 60)))
        monkeypatch.setattr(mdv, "load_ohlcv", lambda s, d: frame)
        out = mdv.build_verified_market_snapshot(
            "TEST", "2025-04-01", indicators=("rvol_20", "obv", "rsi")
        )
        assert "| rvol_20 |" in out
        assert "N/A (" not in out  # all three computed

    def test_exporter_handles_derived_features(self, monkeypatch, tmp_path):
        import importlib.util
        from pathlib import Path

        script = Path("scripts/export_ic_dataset.py")
        spec = importlib.util.spec_from_file_location("exporter_derived", script)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        frame = _frame(list(np.linspace(100, 130, 60)))
        monkeypatch.setattr(exporter, "load_ohlcv", lambda t, d: frame)
        result = exporter.main([
            "TEST", "--indicators", "rvol_20", "obv", "rsi",
            "--output-dir", str(tmp_path),
        ])
        assert result == 0
        df = pd.read_csv(tmp_path / "TEST_5d.csv")
        assert {"rvol_20", "obv", "rsi"} <= set(df.columns)
        assert int(df["rvol_20"].notna().sum()) > 30
