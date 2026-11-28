"""Round-17 regression + coverage tests for the ``dataflows_vendors`` shard.

Owned modules: everything under ``yialpha/dataflows/`` except ``perp_bundle``,
``binance_square``, ``binance`` and ``polymarket`` (other shards own those four).
The file is merged from five groups: numeric helpers (volume features,
volatility estimators, market regime, support/resistance), social/news vendors,
equity vendors, crypto-native vendors, cache + router.

Hermetic: every vendor call is mocked at the project's own seam (patched in
the namespace where the name is *used*), no sleeps, dates frozen by
``curr_date`` injection or by patched loaders.
"""

from __future__ import annotations

# ---- imports used by the merged groups only ----
import base64
import hashlib
import http.client
import io
import json
import logging
import math
import ntpath
import os
import statistics
import time
import types
import zipfile
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest
import requests

import yialpha.agents.utils.price_structure_tools as pst
import yialpha.dataflows.market_regime as mr
import yialpha.dataflows.support_resistance as sr
import yialpha.dataflows.vol_estimators as ve
import yialpha.dataflows.volume_features as vf
from yialpha.agents.utils import binance_indicator_tools as bit
from yialpha.dataflows import (
    alpha_vantage_common as avc,
    alpha_vantage_fundamentals as avf,
    alpha_vantage_indicator as avi,
    binance_vision as bnv,
    binance_vision_store as bvs,
    coingecko_vendor as cg,
    defillama_vendor as dl,
    disk_cache as dc,
    disk_cache as disk_cache_mod,
    etf_fund_data as efd,
    fred,
    fundamentals_bundle as fb,
    fundamentals_overview as fo,
    interface,
    market_data_validator as mdv,
    netretry,
    onchain_flows as ocf,
    quality,
    reddit,
    sec_edgar,
    stockstats_utils,
    stocktwits,
    tavily,
    utils as df_utils,
    y_finance,
    yfinance_news,
)
from yialpha.dataflows.config import get_config, set_config
from yialpha.dataflows.disk_cache import MinIntervalThrottle, vendor_cache_dir
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError
from yialpha.dataflows.interface import route_to_vendor
from yialpha.dataflows.utils import pinned_analysis_date


def _g0_bars(rows) -> pd.DataFrame:
    """Capitalised OHLCV frame; ``rows`` are ``(open, high, low, close)``."""
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])
    frame["Volume"] = 1000.0
    frame.insert(0, "Date", pd.bdate_range("2025-01-02", periods=len(frame)))
    return frame


def _g0_ohlcv(closes, volumes=None) -> pd.DataFrame:
    """OHLCV frame in the shape ``load_ohlcv`` returns, around ``closes``."""
    n = len(closes)
    return pd.DataFrame({
        "Date": pd.bdate_range("2025-01-02", periods=n),
        "Open": [c * 0.999 for c in closes],
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Close": list(closes),
        "Volume": [1000.0] * n if volumes is None else list(volumes),
    })


def _g0_walk(n: int, seed: int = 1, sd: float = 0.01) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(100.0 * np.exp(np.cumsum(rng.normal(0.0, sd, n))))


class TestR17VolumeFeaturesInputs:
    def test_r17_volume_features_empty_inputs_return_an_empty_series_or_none(self):
        empty = pd.DataFrame({"Close": [1.0], "Volume": [1.0]}).iloc[:0]
        for out in (vf.obv(None), vf.obv(empty)):
            assert out.empty and out.dtype == float
        for out in (vf.relative_volume(None), vf.relative_volume(empty)):
            assert out.empty and out.dtype == float
        assert vf.volume_divergence(None) is None
        assert vf.volume_divergence(empty) is None

    @pytest.mark.parametrize("window", [1, 0, -5])
    def test_r17_volume_features_degenerate_window_is_empty_or_none(self, window):
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [10, 20, 30]})
        assert vf.relative_volume(df, window=window).empty
        assert vf.volume_divergence(df, window=window) is None


class TestR17RelativeVolume:
    def test_r17_relative_volume_hand_values_and_today_is_excluded_from_the_base(self):
        df = pd.DataFrame({"Volume": [50, 60, 70, 80, 90, 100]})
        out = vf.relative_volume(df, window=3)
        # Row 5 divides by mean(70, 80, 90) = 80, not by a base holding today's 100.
        assert out.iloc[:3].isna().all()
        assert out.iloc[3:].tolist() == pytest.approx([80 / 60, 90 / 70, 100 / 80])

    def test_r17_relative_volume_zero_prior_mean_is_nan_not_inf(self):
        # No prints in the prior window: "unmeasurable", not "infinitely unusual".
        out = vf.relative_volume(pd.DataFrame({"Volume": [0, 0, 0, 5]}), window=3)
        assert out.isna().all()
        assert not np.isinf(out).any()

    def test_r17_relative_volume_depends_only_on_prior_rows(self):
        rng = np.random.default_rng(11)
        df = pd.DataFrame({"Volume": rng.uniform(1.0, 500.0, 60)})
        full = vf.relative_volume(df, window=5)
        for i in range(6, len(df)):
            prefix = vf.relative_volume(df.iloc[: i + 1], window=5)
            assert prefix.iloc[-1] == pytest.approx(full.iloc[i])

    def test_r17_relative_volume_unreadable_volume_leaves_the_row_unmeasurable(self):
        # The row itself cannot be read ...
        df = pd.DataFrame({"Volume": ["10", "20", "30", "n/a"]})
        assert math.isnan(vf.relative_volume(df, window=3).iloc[-1])
        # ... and neither can a row whose base window holds an unreadable value.
        df = pd.DataFrame({"Volume": ["10", "20", "n/a", "40"]})
        assert math.isnan(vf.relative_volume(df, window=3).iloc[-1])


class TestR17Obv:
    def test_r17_obv_skips_unreadable_closes_and_signs_across_the_gap(self):
        df = pd.DataFrame({"Close": [1.0, 2.0, np.nan, 3.0, 1.0], "Volume": [10, 10, 10, 10, 10]})
        # Row 2 is dropped before the direction is taken: 3.0 is signed against 2.0.
        assert vf.obv(df).to_dict() == {0: 0.0, 1: 10.0, 3: 20.0, 4: 10.0}

    def test_r17_obv_unreadable_volume_counts_as_zero_and_the_input_is_not_mutated(self):
        df = pd.DataFrame({"Close": [1.0, 2.0, 2.0, 1.0], "Volume": [np.nan, 10, "n/a", 10]})
        before = df.copy()
        assert vf.obv(df).tolist() == [0.0, 10.0, 10.0, 0.0]
        pd.testing.assert_frame_equal(df, before)
        blank = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [None, None, None]})
        assert vf.obv(blank).tolist() == [0.0, 0.0, 0.0]


# 20 sessions: a volume-heavy advance to 16, a heavy decline to 9, then a quiet
# advance to a new 20-session high of 16.5.
_G0_BEAR_CLOSES = [10, 11, 12, 13, 14, 15, 16, 14, 12, 10, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 16.5]
_G0_BEAR_VOLUMES = [10] * 7 + [15] * 4 + [1] * 9
# The mirror image: a new 20-session low of 3.5 on light volume.
_G0_BULL_CLOSES = [10, 9, 8, 7, 6, 5, 4, 6, 8, 10, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.5, 7, 3.5]


class TestR17VolumeDivergence:
    def test_r17_volume_divergence_bearish_divergence_reading_hand_values(self):
        # OBV = [0, 10, ..., 60, 45, 30, 15, 0, 1, ..., 9]: price at its window high,
        # OBV (9) below its own (60).
        df = pd.DataFrame({"Close": _G0_BEAR_CLOSES, "Volume": _G0_BEAR_VOLUMES})
        assert vf.volume_divergence(df) == {
            "bearish_divergence": True,
            "bullish_divergence": False,
            "price_window_high": 16.5,
            "price_window_low": 9.0,
            "obv_window_high": 60.0,
            "obv_window_low": 0.0,
            "window": 20,
            "as_of_row": 19,
        }

    def test_r17_volume_divergence_bullish_divergence_reading_hand_values(self):
        # OBV = [0, -10, ..., -60, -45, -30, -15, 0, -1, ..., -9]: price at its window
        # low, OBV (-9) above its own (-60).
        df = pd.DataFrame({"Close": _G0_BULL_CLOSES, "Volume": _G0_BEAR_VOLUMES})
        assert vf.volume_divergence(df) == {
            "bearish_divergence": False,
            "bullish_divergence": True,
            "price_window_high": 11.0,
            "price_window_low": 3.5,
            "obv_window_high": 0.0,
            "obv_window_low": -60.0,
            "window": 20,
            "as_of_row": 19,
        }

    def test_r17_volume_divergence_window_boundary_rows(self):
        df = pd.DataFrame({"Close": _G0_BEAR_CLOSES, "Volume": _G0_BEAR_VOLUMES})
        assert vf.volume_divergence(df, window=20)["as_of_row"] == 19
        assert vf.volume_divergence(df.iloc[1:], window=20) is None   # 19 readable rows
        assert vf.volume_divergence(df, window=21) is None

    def test_r17_volume_divergence_constant_prices_confirm_both_extremes(self):
        # Close is at both the window high and the window low; a flat OBV confirms
        # both. Neither flag may fire.
        df = pd.DataFrame({"Close": [10.0] * 25, "Volume": [5] * 25})
        out = vf.volume_divergence(df, window=20)
        assert out["bearish_divergence"] is False
        assert out["bullish_divergence"] is False
        assert out["price_window_high"] == out["price_window_low"] == 10.0
        assert out["obv_window_high"] == out["obv_window_low"] == 0.0

    def test_r17_volume_divergence_obv_equal_to_its_window_extreme_counts_as_confirmed(self):
        # OBV = [0, 1, 0, 1]: the last value TIES its window high. Only a strictly
        # weaker OBV diverges, so this is a confirmed high.
        df = pd.DataFrame({"Close": [10.0, 11.0, 10.0, 11.5], "Volume": [1, 1, 1, 1]})
        assert vf.volume_divergence(df, window=4) == {
            "bearish_divergence": False,
            "bullish_divergence": False,
            "price_window_high": 11.5,
            "price_window_low": 10.0,
            "obv_window_high": 1.0,
            "obv_window_low": 0.0,
            "window": 4,
            "as_of_row": 3,
        }

    def test_r17_volume_divergence_unreadable_closes_do_not_count_towards_the_window(self):
        df = pd.DataFrame({"Close": [math.nan] + _G0_BEAR_CLOSES, "Volume": [1] + _G0_BEAR_VOLUMES})
        assert vf.volume_divergence(df, window=21) is None
        assert vf.volume_divergence(df, window=20) == vf.volume_divergence(df.iloc[1:], window=20)


class TestR17CleanOhlcv:
    def test_r17_clean_ohlcv_none_or_empty_is_an_empty_frame(self):
        assert ve._clean_ohlcv(None).empty
        assert ve._clean_ohlcv(pd.DataFrame()).empty

    def test_r17_clean_ohlcv_drops_unreadable_and_non_positive_rows_but_keeps_infinite(self):
        # ``inf`` passes the ``> 0`` filter, so the module docstring's "rows with a
        # non-finite Close are dropped first" holds for NaN only. Pinned as is.
        df = _g0_ohlcv([1.5, 2.0, 3.0, 4.0, 5.0])
        df.loc[1, ["Open", "High", "Low", "Close"]] = np.nan
        df.loc[2, "Low"] = 0.0
        df.loc[3, "Close"] = -1.0
        df.loc[4, "Close"] = np.inf
        out = ve._clean_ohlcv(df)
        assert out.index.tolist() == [0, 4]
        assert out["Close"].iloc[0] == 1.5 and math.isinf(out["Close"].iloc[1])

    def test_r17_clean_ohlcv_numeric_strings_are_coerced_and_the_input_is_not_mutated(self):
        df = pd.DataFrame({
            "Date": pd.bdate_range("2025-01-02", periods=2),
            "Open": ["100", "101"], "High": ["102", "103"], "Low": ["99", "100"],
            "Close": ["101", "102"], "Volume": ["1000", "1100"],
        })
        before = df.copy()
        out = ve._clean_ohlcv(df)
        assert out["Close"].tolist() == [101.0, 102.0]
        pd.testing.assert_frame_equal(df, before)

    def test_r17_clean_ohlcv_all_rows_bad_is_an_empty_frame(self):
        out = ve._clean_ohlcv(_g0_ohlcv([-1.0, 0.0, math.nan]))
        assert out.empty and out.columns.empty

    def test_r17_clean_ohlcv_all_four_price_columns_are_required(self):
        # Close alone is not enough, even for the close-to-close estimator.
        with pytest.raises(KeyError):
            ve._clean_ohlcv(pd.DataFrame({"Close": [1.0, 2.0]}))


@pytest.mark.parametrize(
    ("estimator", "degenerate_window"),
    [(ve.close_to_close_vol, 1), (ve.garman_klass_vol, 1), (ve.yang_zhang_vol, 2)],
)
class TestR17VolEstimatorInputs:
    def test_r17_vol_estimator_none_empty_and_all_bad_frames_give_an_empty_series(self, estimator, degenerate_window):
        all_bad = _g0_ohlcv([0.0, -1.0, math.nan, 0.0])
        for frame in (None, pd.DataFrame(), all_bad):
            out = estimator(frame, window=5)
            assert out.empty and out.dtype == float

    def test_r17_vol_estimator_degenerate_window_gives_an_empty_series(self, estimator, degenerate_window):
        frame = _g0_ohlcv(_g0_walk(40))
        assert estimator(frame, window=degenerate_window).empty
        assert estimator(frame, window=degenerate_window - 1).empty


class TestR17VolEstimatorValues:
    def test_r17_vol_estimator_constant_prices_are_exactly_zero_not_nan(self):
        # A flat O=H=L=C series has zero return variance, zero range and zero body.
        flat = _g0_bars([(100.0, 100.0, 100.0, 100.0)] * 30)
        for out, warm_up in (
            (ve.close_to_close_vol(flat, window=5), 5),
            (ve.garman_klass_vol(flat, window=5), 4),
            (ve.yang_zhang_vol(flat, window=5), 5),
        ):
            assert int(out.isna().sum()) == warm_up
            assert out.iloc[warm_up:].tolist() == [0.0] * (30 - warm_up)

    def test_r17_vol_estimator_too_short_series_is_all_nan_not_empty(self):
        short = _g0_ohlcv(_g0_walk(5))
        for estimator in (ve.close_to_close_vol, ve.garman_klass_vol, ve.yang_zhang_vol):
            out = estimator(short, window=20)
            assert len(out) == 5 and out.isna().all()

    def test_r17_vol_estimator_garman_klass_negative_terms_clip_to_zero_at_the_window(self):
        # Close above High makes ln(C/O)^2 outweigh the range term: every per-row
        # term is negative, so the window mean is clipped to 0 -> 0.0, not NaN.
        odd = _g0_bars([(100.0, 101.0, 99.0, 110.0)] * 4)
        out = ve.garman_klass_vol(odd, window=3)
        assert out.iloc[:2].isna().all()
        assert out.iloc[2:].tolist() == [0.0, 0.0]
        # One odd bar among normal ones leaves the window mean positive: finite.
        mixed = _g0_bars(
            [(100.0, 103.0, 97.0, 101.0)] * 2
            + [(100.0, 101.0, 99.0, 110.0)]
            + [(100.0, 103.0, 97.0, 99.0)] * 2
        )
        out = ve.garman_klass_vol(mixed, window=3)
        assert out.iloc[2:].notna().all()
        assert (out.iloc[2:] > 0).all()

    def test_r17_vol_estimator_yang_zhang_hand_value(self):
        bars = [
            (100.0, 102.0, 99.0, 101.0), (101.0, 103.0, 100.0, 102.0),
            (102.0, 104.0, 101.0, 100.5), (100.4, 101.5, 99.0, 100.0),
            (100.0, 103.0, 98.0, 102.5), (102.6, 104.0, 102.0, 103.0),
        ]
        out = ve.yang_zhang_vol(_g0_bars(bars), window=3)
        # The overnight term needs the previous close: the first window rows are NaN.
        assert out.iloc[:3].isna().all()
        k = 0.34 / (1.34 + 4.0 / 2.0)
        for i in (3, 4, 5):
            rows = bars[i - 2 : i + 1]
            overnight = [
                math.log(o / prev_close)
                for (_, _, _, prev_close), (o, _, _, _) in zip(bars[i - 3 : i], rows, strict=True)
            ]
            open_to_close = [math.log(c / o) for (o, _, _, c) in rows]
            rogers_satchell = [
                math.log(h / o) * math.log(h / c) + math.log(lo / o) * math.log(lo / c)
                for (o, h, lo, c) in rows
            ]
            var = (
                statistics.variance(overnight)
                + k * statistics.variance(open_to_close)
                + (1 - k) * statistics.fmean(rogers_satchell)
            )
            assert out.iloc[i] == pytest.approx(math.sqrt(var * 252.0), rel=1e-9)

    def test_r17_vol_estimator_yang_zhang_smallest_window_is_two(self):
        out = ve.yang_zhang_vol(_g0_ohlcv(_g0_walk(8)), window=2)
        assert out.iloc[2:].notna().all()


# --------------------------------------------------------------------------- #
# market_regime: benchmark turbulence
# --------------------------------------------------------------------------- #


def _g0_closes_from_returns(returns, start=100.0) -> list[float]:
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return closes


def _g0_squared_z(returns) -> float:
    """Squared z-score of the last return against the sample prior to it."""
    hist, curr = returns[:-1], returns[-1]
    return (curr - statistics.fmean(hist)) ** 2 / statistics.variance(hist)


def _g0_regime_frame(closes) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-02", periods=len(closes)), "Close": closes
    })


def _g0_percentile_of_latest(rvol: pd.Series, i: int, base: int = 252) -> float:
    """Share of the ``base`` values ending at ``i`` that sit below ``rvol[i]``."""
    window = rvol.iloc[max(0, i - base + 1) : i + 1]
    return float((window < rvol.iloc[i]).mean() * 100.0)


def _g0_fake_wrap(close, sma50, sma200, adx):
    def wrap_one_row(_df):
        return pd.DataFrame({
            "close": [close], "close_50_sma": [sma50],
            "close_200_sma": [sma200], "adx": [adx],
        })

    return wrap_one_row


def _g0_fake_rvol(n_below: int, n: int = 40, latest: float = 0.5) -> pd.Series:
    """``n_below`` of ``n`` trailing rvol values sit below the latest one."""
    values = [0.10 + 0.001 * i for i in range(n_below)]
    values += [1.00 + 0.01 * i for i in range(n - n_below - 1)]
    return pd.Series(values + [latest])


# ((i * 7) % 23 - 11) permutes -11..11, so every value is equally common and the
# oldest return (-0.011) is an extreme one: dropping it moves the z-score by ~5%.
_G0_RETURNS = [((i * 7) % 23 - 11) * 0.001 for i in range(300)]


class TestR17ComputeTurbulence:
    def test_r17_turbulence_short_history_falls_back_to_every_prior_return(self, monkeypatch):
        # Below the 252-return window but above min_periods: "whatever history
        # exists" is every return before the current one. Before the fix the slice
        # ``iloc[-(len - 1):-1]`` also dropped the OLDEST return (98 of 99 here).
        returns = _G0_RETURNS[:100]
        frame = _g0_regime_frame(_g0_closes_from_returns(returns))
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: frame)
        out = mr.compute_turbulence("SYN", "2024-12-31", window=252, min_periods=60)
        assert out == pytest.approx(_g0_squared_z(returns), rel=1e-9)

    def test_r17_turbulence_history_that_just_clears_min_periods_is_enough(self, monkeypatch):
        returns = _G0_RETURNS[:61]       # the current return plus min_periods prior ones
        frame = _g0_regime_frame(_g0_closes_from_returns(returns))
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: frame)
        out = mr.compute_turbulence("SYN", "2024-12-31", window=252, min_periods=60)
        assert out == pytest.approx(_g0_squared_z(returns), rel=1e-9)
        # One return fewer is below min_periods: no reading.
        returns = _G0_RETURNS[:60]
        frame = _g0_regime_frame(_g0_closes_from_returns(returns))
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: frame)
        assert mr.compute_turbulence("SYN", "2024-12-31", window=252, min_periods=60) is None

    def test_r17_turbulence_window_smaller_than_min_periods_is_none(self, monkeypatch):
        returns = _G0_RETURNS
        frame = _g0_regime_frame(_g0_closes_from_returns(returns))
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: frame)
        # The full-history branch holds exactly `window` prior returns: with fewer
        # of them than min_periods the estimate is still too noisy to trust.
        assert mr.compute_turbulence("SYN", "2024-12-31", window=30, min_periods=60) is None
        # Exactly min_periods of them is enough.
        out = mr.compute_turbulence("SYN", "2024-12-31", window=60, min_periods=60)
        assert out == pytest.approx(_g0_squared_z(returns[-61:]), rel=1e-9)

    @pytest.mark.parametrize("n_returns", [1, 2])
    def test_r17_turbulence_zero_min_periods_cannot_rescue_a_tiny_history(self, monkeypatch, n_returns):
        closes = [100.0 * 1.01 ** i for i in range(n_returns + 1)]
        frame = _g0_regime_frame(closes)
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: frame)
        # One prior return has no sample variance; none has neither mean nor variance.
        assert mr.compute_turbulence("SYN", "2024-12-31", window=252, min_periods=0) is None

    def test_r17_turbulence_latest_return_is_taken_by_date_not_by_row_order(self, monkeypatch):
        frame = _g0_regime_frame(_g0_closes_from_returns(_G0_RETURNS))

        def value(served):
            monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: served)
            return mr.compute_turbulence("SYN", "2024-12-31")

        expected = value(frame)
        assert expected == pytest.approx(_g0_squared_z(_G0_RETURNS[-253:]), rel=1e-9)
        assert value(frame.iloc[::-1]) == pytest.approx(expected)

    @pytest.mark.parametrize("bad_close", [math.nan, 0.0, -5.0])
    def test_r17_turbulence_unusable_latest_close_reads_as_missing_not_as_a_crash_day(self, monkeypatch, bad_close):
        # A zero close is a bad tick, not a -100% session. Before the fix the
        # latest return became -1.0 (or -1.05) and the squared z-score was in the
        # hundreds -- an "elevated" stress reading manufactured out of one bad bar.
        closes = _g0_closes_from_returns(_G0_RETURNS)
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: _g0_regime_frame(closes))
        expected = mr.compute_turbulence("SYN", "2024-12-31")
        served = _g0_regime_frame(closes + [bad_close])
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: served)
        out = mr.compute_turbulence("SYN", "2024-12-31")
        assert out == pytest.approx(expected)
        assert math.sqrt(out) < 2.5

    @pytest.mark.parametrize("bad_close", [0.0, -5.0])
    def test_r17_turbulence_unusable_close_inside_the_history_reads_as_missing(self, monkeypatch, bad_close):
        # Same rule as for a NaN close: the row is skipped and the return spans
        # the gap. Before the fix the zero produced a -100% return and an
        # infinite one next to it.
        closes = _g0_closes_from_returns(_G0_RETURNS)
        with_nan, with_bad = list(closes), list(closes)
        with_nan[150] = math.nan
        with_bad[150] = bad_close
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: _g0_regime_frame(with_nan))
        expected = mr.compute_turbulence("SYN", "2024-12-31")
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: _g0_regime_frame(with_bad))
        assert mr.compute_turbulence("SYN", "2024-12-31") == pytest.approx(expected)


class TestR17ClassifyTrendState:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ((110.0, 105.0, 100.0, 30.0), ("uptrend", "bull_stack", "strong")),
            ((90.0, 95.0, 100.0, 30.0), ("downtrend", "bear_stack", "strong")),
            # close above sma50 but sma50 below sma200: no clean order either way
            ((110.0, 95.0, 100.0, 22.5), ("mixed", "bear_stack", "moderate")),
            ((90.0, 105.0, 100.0, 22.5), ("mixed", "bull_stack", "moderate")),
            # a close exactly on sma50 is neither above nor below it
            ((105.0, 105.0, 100.0, 22.5), ("mixed", "bull_stack", "moderate")),
            # equal averages are not a bull stack (strict >)
            ((110.0, 100.0, 100.0, 30.0), ("mixed", "bear_stack", "strong")),
            # strength thresholds are inclusive: <=20 weak, >=25 strong
            ((110.0, 105.0, 100.0, 25.0), ("uptrend", "bull_stack", "strong")),
            ((110.0, 105.0, 100.0, 24.9), ("uptrend", "bull_stack", "moderate")),
            ((110.0, 105.0, 100.0, 20.1), ("uptrend", "bull_stack", "moderate")),
            ((110.0, 105.0, 100.0, 20.0), ("uptrend", "bull_stack", "weak")),
            ((110.0, 105.0, 100.0, 0.0), ("uptrend", "bull_stack", "weak")),
        ],
    )
    def test_r17_trend_state_classification_matrix(self, monkeypatch, values, expected):
        monkeypatch.setattr(mr, "wrap", _g0_fake_wrap(*values))
        out = mr.classify_trend_state(_g0_ohlcv(_g0_walk(200)))
        assert (out["trend"], out["stack"], out["adx_strength"]) == expected
        assert (out["close"], out["sma50"], out["sma200"], out["adx"]) == values

    @pytest.mark.parametrize("bad_slot", [0, 1, 2, 3])
    @pytest.mark.parametrize("bad_value", [math.nan, math.inf])
    def test_r17_trend_state_any_non_finite_component_is_omitted(self, monkeypatch, bad_slot, bad_value):
        values = [110.0, 105.0, 100.0, 30.0]
        values[bad_slot] = bad_value
        monkeypatch.setattr(mr, "wrap", _g0_fake_wrap(*values))
        assert mr.classify_trend_state(_g0_ohlcv(_g0_walk(200))) is None

    def test_r17_trend_state_real_frame_values_match_independent_averages(self):
        closes = [100.0 + 0.5 * i for i in range(400)]
        out = mr.classify_trend_state(_g0_ohlcv(closes))
        series = pd.Series(closes)
        assert out["close"] == closes[-1]
        assert out["sma50"] == pytest.approx(series.tail(50).mean())
        assert out["sma200"] == pytest.approx(series.tail(200).mean())
        assert (out["trend"], out["stack"], out["adx_strength"]) == ("uptrend", "bull_stack", "strong")
        # One row short of what the 200-session average needs.
        assert mr.classify_trend_state(_g0_ohlcv(closes[:199])) is None
        assert mr.classify_trend_state(None) is None

    def test_r17_trend_state_frame_stockstats_cannot_read_is_omitted(self):
        frame = _g0_ohlcv(_g0_walk(200))
        unreadable = frame.copy()
        unreadable["Close"] = unreadable["Close"].astype(str)
        assert mr.classify_trend_state(unreadable) is None
        assert mr.classify_trend_state(frame.drop(columns=["Close"])) is None

    @pytest.mark.parametrize("bad_close", [math.nan, math.inf, 0.0, -0.0])
    def test_r17_trend_state_unusable_latest_close_is_omitted_not_read_as_a_trend(self, bad_close):
        # Before the fix a zero close compared below both averages and rendered
        # "trend=downtrend (bear_stack)" -- a trend manufactured out of one bad
        # bar, on the default-on regime line of the market analyst's prompt.
        frame = _g0_ohlcv(_g0_walk(300) + [_g0_walk(300)[-1]])
        frame.loc[frame.index[-1], "Close"] = bad_close
        assert mr.classify_trend_state(frame) is None

    def test_r17_trend_state_negative_close_is_still_read_as_a_price(self):
        # Observation, pinned as is: only a zero close is refused. A negative one
        # is still compared (a downtrend here). The existing downtrend fixture in
        # tests/test_price_structure_and_regime.py prints negative closes for its
        # last 200 rows and relies on this.
        frame = _g0_ohlcv(_g0_walk(300) + [_g0_walk(300)[-1]])
        frame.loc[frame.index[-1], "Close"] = -1.0
        out = mr.classify_trend_state(frame)
        assert out is not None
        assert (out["trend"], out["close"]) == ("downtrend", -1.0)


class TestR17ClassifyVolState:
    @pytest.mark.parametrize(
        ("n_below", "label"),
        [
            (0, "low"), (10, "low"), (11, "normal"), (20, "normal"), (29, "normal"),
            (30, "high"), (35, "high"), (36, "extreme"), (39, "extreme"),
        ],
    )
    def test_r17_vol_state_labels_follow_the_percentile_with_inclusive_thresholds(self, monkeypatch, n_below, label):
        # 40 values: 10 below = 25.0th pct (low), 30 below = 75.0th (high),
        # 36 below = 90.0th (extreme).
        monkeypatch.setattr(
            mr, "close_to_close_vol",
            lambda df, window, periods_per_year: _g0_fake_rvol(n_below),
        )
        out = mr.classify_vol_state(_g0_ohlcv(_g0_walk(40)))
        assert out == {
            "rvol_20": 0.5,
            "percentile": pytest.approx(n_below / 40 * 100),
            "label": label,
        }

    def test_r17_vol_state_non_finite_latest_rvol_is_omitted(self, monkeypatch):
        monkeypatch.setattr(
            mr, "close_to_close_vol",
            lambda df, window, periods_per_year: pd.Series([0.3] * 39 + [math.inf]),
        )
        assert mr.classify_vol_state(_g0_ohlcv(_g0_walk(40))) is None

    def test_r17_vol_state_needs_thirty_readable_rvol_values(self):
        assert mr.classify_vol_state(_g0_ohlcv(_g0_walk(49, seed=3))) is None
        frame = _g0_ohlcv(_g0_walk(50, seed=3))
        rvol = ve.close_to_close_vol(frame).dropna()
        assert len(rvol) == 30
        out = mr.classify_vol_state(frame)
        assert out["rvol_20"] == pytest.approx(rvol.iloc[-1])
        assert out["percentile"] == pytest.approx(_g0_percentile_of_latest(rvol, 29))

    def test_r17_vol_state_percentile_ranks_the_latest_against_its_own_trailing_year_only(self):
        frame = _g0_ohlcv(_g0_walk(120, seed=5))
        full = ve.close_to_close_vol(frame).dropna()
        for rows in (70, 95, 120):
            out = mr.classify_vol_state(frame.iloc[:rows])
            i = rows - 21            # last row with an rvol: 20 warm-up rows
            assert out["rvol_20"] == pytest.approx(full.iloc[i])
            assert out["percentile"] == pytest.approx(_g0_percentile_of_latest(full, i))

    def test_r17_vol_state_level_follows_the_asset_calendar_and_the_percentile_does_not(self):
        frame = _g0_ohlcv(_g0_walk(120, seed=5))
        equity = mr.classify_vol_state(frame)
        crypto = mr.classify_vol_state(frame, asset_type="crypto_perp")
        perp = mr.classify_vol_state(
            frame, asset_type="crypto_perp", instrument_class="stock_perp"
        )
        assert crypto["rvol_20"] == pytest.approx(equity["rvol_20"] * math.sqrt(365.0 / 252.0))
        assert perp["rvol_20"] == pytest.approx(crypto["rvol_20"])
        assert equity["percentile"] == crypto["percentile"] == perp["percentile"]
        assert equity["label"] == crypto["label"] == perp["label"]

    @pytest.mark.parametrize("bad", [{"Close": 0.0}, {"Close": math.nan}, {"Open": math.nan}])
    def test_r17_vol_state_unusable_latest_bar_reads_the_previous_bars_state(self, bad):
        # Observation, pinned as is: when the newest bar cannot yield an rvol the
        # reading is the previous bar's and the line does not say so -- the same
        # convention the Yahoo loader applies to a missing bar (it drops the row).
        frame = _g0_ohlcv(_g0_walk(300))
        for col, value in bad.items():
            frame.loc[frame.index[-1], col] = value
        assert mr.classify_vol_state(frame) == mr.classify_vol_state(frame.iloc[:-1])


class TestR17RegimeContextLine:
    def test_r17_regime_line_unusable_latest_bar_omits_the_trend_and_the_fabricated_stress(self, monkeypatch):
        # End to end down the default-on regime line. Before the fix the line read
        # "trend=downtrend (bear_stack, ADX 11 weak) | ... | turbulence(SPY)=elevated".
        closes = _g0_walk(300)
        no_trade_bar = _g0_ohlcv(closes + [closes[-1]])
        no_trade_bar.loc[no_trade_bar.index[-1], ["Open", "High", "Low", "Close"]] = 0.0
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: no_trade_bar)
        line = mr.format_regime_context("TEST", "2026-05-01")
        assert line is not None
        assert line.startswith("Market regime (TEST, 2026-05-01): ")
        assert "trend=" not in line
        assert "vol=" in line
        assert "turbulence(SPY)=normal" in line
        monkeypatch.setattr(mr, "_load_regime_ohlcv", lambda s, d, a=None: _g0_ohlcv(closes))
        clean = mr.format_regime_context("TEST", "2026-05-01")
        assert clean is not None
        # Everything except the trend component is the same reading.
        prefix = "Market regime (TEST, 2026-05-01): "
        assert line == prefix + " | ".join(
            part for part in clean[len(prefix):].split(" | ") if not part.startswith("trend=")
        )


# --------------------------------------------------------------------------- #
# support_resistance: the latest close drives every "distance" (observation)
# --------------------------------------------------------------------------- #


class TestR17SupportResistanceLatestClose:
    def test_r17_support_resistance_zero_latest_close_raises_out_of_the_tool(self, monkeypatch):
        # Observation, pinned as is (the fix belongs to the tool, which is not
        # this card's): ``rolling_levels`` divides by the latest close, and
        # ``get_support_resistance`` calls ``build_support_resistance`` without a
        # guard, so one zero close surfaces as a ZeroDivisionError in the node
        # -- against the tool's own "typed degrade, never raise into the node".
        # Only a direct caller can reach it: ``load_ohlcv`` drops NaN closes and
        # the Yahoo path does not print zero ones.
        frame = _g0_ohlcv(_g0_walk(300) + [0.0])
        with pytest.raises(ZeroDivisionError):
            sr.rolling_levels(frame)
        monkeypatch.setattr(pst, "load_ohlcv", lambda symbol, curr_date: frame)
        with pytest.raises(ZeroDivisionError):
            pst.get_support_resistance.invoke({"symbol": "TEST", "curr_date": "2026-05-01"})

    def test_r17_support_resistance_unreadable_latest_close_renders_not_available(
        self, monkeypatch
    ):
        # A NaN close cannot come out of ``load_ohlcv``; with one, the levels are
        # real and the distances are NaN, which the tool renders as "N/A".
        closes = _g0_walk(300)
        frame = _g0_ohlcv(closes + [math.nan])
        levels = sr.rolling_levels(frame, windows=(20,))
        assert levels[20]["high"] == pytest.approx(max(closes[-20:]) * 1.01)
        assert levels[20]["low"] == pytest.approx(min(closes[-20:]) * 0.99)
        assert math.isnan(levels[20]["high_dist_pct"])
        assert math.isnan(levels[20]["low_dist_pct"])
        monkeypatch.setattr(pst, "load_ohlcv", lambda symbol, curr_date: frame)
        out = pst.get_support_resistance.invoke({"symbol": "TEST", "curr_date": "2026-05-01"})
        assert "Latest close: N/A on" in out
        assert "N/A% |" in out


def _g1_resp(read_fn):
    """A minimal context-manager response whose ``read()`` runs ``read_fn``."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            return read_fn()
    return _Resp()


def _g1_json_resp(obj):
    return _g1_resp(lambda: json.dumps(obj).encode("utf-8"))


def _g1_raise(exc):
    def _r():
        raise exc
    return _g1_resp(_r)


def _g1_arm_oauth(monkeypatch, client_id="cid", secret="sec"):
    """Configure Reddit OAuth creds and start from an empty token cache."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", client_id)
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", secret)
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    monkeypatch.setattr(reddit, "_token_cache", None)


class TestR17G1RedditRetryAfter:
    """``_retry_after_seconds``: the parsed forms, the 30s cap, unusable values."""

    def test_r17_reddit_retry_after_seconds_numeric_forms(self):
        ra = reddit._retry_after_seconds
        assert ra(SimpleNamespace(headers={"Retry-After": "12"})) == 12.0
        assert ra(SimpleNamespace(headers={"Retry-After": "12.5"})) == 12.5
        assert ra(SimpleNamespace(headers={"Retry-After": " 7 "})) == 7.0
        assert ra(SimpleNamespace(headers={"Retry-After": "1e1"})) == 10.0
        assert ra(SimpleNamespace(headers={"Retry-After": "0"})) == 0.0

    def test_r17_reddit_retry_after_seconds_capped_at_30(self):
        ra = reddit._retry_after_seconds
        assert ra(SimpleNamespace(headers={"Retry-After": "30"})) == 30.0
        assert ra(SimpleNamespace(headers={"Retry-After": "31"})) == 30.0
        assert ra(SimpleNamespace(headers={"Retry-After": "99999999999"})) == 30.0
        # float("1e400") is inf: the cap still binds.
        assert ra(SimpleNamespace(headers={"Retry-After": "1e400"})) == 30.0

    def test_r17_reddit_retry_after_seconds_unusable_header_is_none(self):
        ra = reddit._retry_after_seconds
        # The except branch: an HTTP-date, free text, a suffixed number.
        for value in ("Wed, 21 Oct 2015 07:28:00 GMT", "soon", "12s"):
            assert ra(SimpleNamespace(headers={"Retry-After": value})) is None
        # A headers object with no ``.get``.
        assert ra(SimpleNamespace(headers=[("Retry-After", "12")])) is None
        # Absent header, empty value, ``None`` value, ``None``/missing headers.
        assert ra(SimpleNamespace(headers={})) is None
        assert ra(SimpleNamespace(headers={"Retry-After": ""})) is None
        assert ra(SimpleNamespace(headers={"Retry-After": None})) is None
        assert ra(SimpleNamespace(headers=None)) is None
        assert ra(SimpleNamespace()) is None


class TestR17G1RedditOauthToken:
    """The token endpoint's payload contract (``_request_oauth_token``)."""

    def test_r17_reddit_token_payload_without_access_token_is_rejected(self):
        payloads = (
            {"token_type": "bearer", "expires_in": 3600},
            {"access_token": ""},
            {"access_token": None},
        )
        for payload in payloads:
            with patch.object(reddit, "urlopen", return_value=_g1_json_resp(payload)), \
                 pytest.raises(
                     ValueError, match="token endpoint returned no access_token",
                 ):
                reddit._request_oauth_token(("cid", "sec"), 5.0)

    def test_r17_reddit_token_expires_in_default_and_string_coercion(self):
        for payload, expected in (
            ({"access_token": "tok"}, ("tok", 3600)),
            ({"access_token": "tok", "expires_in": "7200"}, ("tok", 7200)),
        ):
            with patch.object(reddit, "urlopen", return_value=_g1_json_resp(payload)):
                assert reddit._request_oauth_token(("cid", "sec"), 5.0) == expected


class TestR17G1RedditSearchChildren:
    """Malformed children in an OAuth search listing are skipped, not rendered."""

    def test_r17_reddit_search_listing_skips_malformed_children(self, monkeypatch):
        _g1_arm_oauth(monkeypatch)
        listing = {"data": {"children": [
            None,
            "t3",
            7,
            {"kind": "t3", "data": "not a dict"},
            {"kind": "t3", "data": None},
            {"kind": "t3", "data": {"title": "OK", "score": 1, "num_comments": 2,
                                    "created_utc": 1.5, "selftext": "body"}},
        ]}}
        with patch.object(
            reddit, "urlopen",
            side_effect=[_g1_json_resp({"access_token": "abc", "expires_in": 3600}),
                         _g1_json_resp(listing)],
        ):
            posts = reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        assert posts == [{"title": "OK", "score": 1, "num_comments": 2,
                          "created_utc": 1.5, "selftext": "body", "source": "oauth"}]


class TestR17G1RedditFormatter:
    def test_r17_reddit_formatter_truncates_long_selftext(self):
        long_post = [{"title": "Long", "score": 3, "num_comments": 1,
                      "created_utc": 1716215400.0, "selftext": "y" * 300,
                      "source": "oauth"}]
        exact_post = [{"title": "Exact", "score": 3, "num_comments": 1,
                       "created_utc": 1716215400.0, "selftext": "z" * 240,
                       "source": "oauth"}]
        with patch.object(reddit, "_fetch_subreddit", return_value=long_post):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        with patch.object(reddit, "_fetch_subreddit", return_value=exact_post):
            exact = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert f"body excerpt: {'y' * 240}…" in out
        assert "y" * 241 not in out
        assert f"body excerpt: {'z' * 240}" in exact
        assert "…" not in exact


class TestR17G1StocktwitsMessages:
    def test_r17_stocktwits_no_messages_placeholder_is_not_a_degradation(self, caplog):
        # Two tickers: each payload gets its own cache entry, so both reach the transport.
        payloads = {
            "NVDA": {"messages": []},
            "TSLA": {"response": {"status": 200}, "symbol": "TSLA"},
        }
        outs = {}
        for ticker, payload in payloads.items():
            raw = json.dumps(payload).encode("utf-8")
            with patch.object(
                stocktwits, "urlopen", return_value=_g1_resp(lambda body=raw: body),
            ) as op:
                outs[ticker] = stocktwits.fetch_stocktwits_messages(ticker)
            assert op.call_count == 1
        assert outs == {
            "NVDA": "<no StockTwits messages found for $NVDA>",
            "TSLA": "<no StockTwits messages found for $TSLA>",
        }
        # Honest no-data: nothing in the ledger, nothing in the log.
        assert quality.snapshot_quality() == []
        assert not [r for r in caplog.records if "stocktwits" in r.name]

    def test_r17_stocktwits_truncates_long_body(self):
        payload = json.dumps({"messages": [{
            "created_at": "2026-08-15T10:00:00Z",
            "user": {"username": "u"},
            "entities": {"sentiment": {"basic": "Bullish"}},
            "body": "x" * 300,
        }]}).encode("utf-8")
        with patch.object(stocktwits, "urlopen", return_value=_g1_resp(lambda: payload)):
            out = stocktwits.fetch_stocktwits_messages("nvda")
        assert out == (
            "Bullish: 1 (100%) · Bearish: 0 (0%) · Unlabeled: 0 · "
            "Total: 1 most-recent messages\n\n"
            "[2026-08-15T10:00:00Z · @u · Bullish] " + "x" * 280 + "…"
        )


_g1_atom_one = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns="http://www.w3.org/2005/Atom">'
    "<entry><title>NVDA pops</title>"
    "<published>2024-05-20T14:30:00Z</published>"
    '<content type="html">&lt;p&gt;Great quarter&lt;/p&gt;</content>'
    "</entry>"
    "</feed>"
)

_g1_atom_three = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns="http://www.w3.org/2005/Atom">'
    "<entry><title>One</title><published>2024-05-20T14:30:00Z</published>"
    '<content type="html">&lt;p&gt;First&lt;/p&gt;</content></entry>'
    "<entry><title>Two</title></entry>"
    "<entry><title>Three</title><published>not-a-date</published></entry>"
    "</feed>"
)

_g1_atom_empty = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
)

_g1_rss_block = (
    "r/stocks — 1 recent posts mentioning NVDA "
    "(via RSS feed; scores/comments unavailable):\n"
    "  [2024-05-20] NVDA pops\n"
    "    body excerpt: Great quarter"
)


def _g1_http_error(code, msg, hdrs=None):
    return HTTPError("https://vendor.invalid/path", code, msg, hdrs or {}, None)


def _g1_clear_oauth(monkeypatch):
    """No Reddit creds and an empty token cache: the keyless RSS default."""
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(reddit, "_token_cache", None)


def _g1_freeze_clock(monkeypatch, start):
    """Freeze ``time.time`` for the Reddit code; returns the movable ``[now]``."""
    now = [float(start)]
    monkeypatch.setattr(reddit.time, "time", lambda: now[0])
    return now


class TestR17G1RedditRss429:
    def test_r17_reddit_rss_429_backoff_default_cap_and_zero(
        self, monkeypatch, caplog,
    ):
        # No header -> 5s default; a huge header -> the 30s cap; "0" is falsy
        # for the caller's ``or 5.0`` and so also backs off 5s.
        _g1_clear_oauth(monkeypatch)
        waits = []
        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.reddit"):
            for hdrs, expected in (({}, 5.0), ({"Retry-After": "600"}, 30.0),
                                   ({"Retry-After": "0"}, 5.0)):
                rate_limited = _g1_http_error(429, "Too Many Requests", hdrs)
                with patch.object(
                    reddit, "urlopen",
                    side_effect=[rate_limited, _g1_resp(lambda: _g1_atom_one.encode())],
                ) as op, patch.object(reddit.time, "sleep") as slept:
                    posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
                assert op.call_count == 2
                assert slept.call_args_list == [call(expected)]
                assert len(posts) == 1
                waits.append(expected)
        backed_off = [r.getMessage() for r in caplog.records]
        assert backed_off == [
            "Reddit RSS 429 for r/stocks · NVDA — backing off 5.0s then retrying once",
            "Reddit RSS 429 for r/stocks · NVDA — backing off 30.0s then retrying once",
            "Reddit RSS 429 for r/stocks · NVDA — backing off 5.0s then retrying once",
        ]
        assert waits == [5.0, 30.0, 5.0]

    def test_r17_reddit_rss_request_shape_and_timeout_survive_the_retry(
        self, monkeypatch,
    ):
        _g1_clear_oauth(monkeypatch)
        seen = []
        expected_request = (
            "https://www.reddit.com/r/stocks/search.rss"
            "?q=NVDA&restrict_sr=on&sort=new&t=week&limit=7",
            "GET",
            reddit._UA,
            None,
            3.5,
        )

        def _fake_urlopen(req, timeout):
            seen.append((
                req.full_url, req.get_method(), req.get_header("User-agent"),
                req.get_header("Authorization"), timeout,
            ))
            if len(seen) == 1:
                raise _g1_http_error(429, "Too Many Requests", {"Retry-After": "2"})
            return _g1_resp(lambda: _g1_atom_empty.encode())

        with patch.object(reddit, "urlopen", side_effect=_fake_urlopen), \
             patch.object(reddit.time, "sleep") as slept:
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 7, 3.5)
        assert posts == []
        assert slept.call_args_list == [call(2.0)]
        assert seen == [expected_request, expected_request]
        assert quality.snapshot_quality() == []


class TestR17G1RedditRetryAfterMalformed:
    """Pinned and reported in round 17 (not fixed): the cap binds from above
    only. A negative or NaN ``Retry-After`` is returned uncapped and the
    backoff ``time.sleep`` then raises instead of waiting, so the public fetch
    raises (the sentiment analyst has no handler around it). Reddit does not
    send such headers; the fix would be a lower bound in the helper."""

    def test_r17_reddit_retry_after_seconds_negative_and_nan_are_not_capped(self):
        ra = reddit._retry_after_seconds
        assert ra(SimpleNamespace(headers={"Retry-After": "-5"})) == -5.0
        assert ra(SimpleNamespace(headers={"Retry-After": "-1e400"})) == float("-inf")
        assert math.isnan(ra(SimpleNamespace(headers={"Retry-After": "nan"})))

    @pytest.mark.parametrize(
        "header, reason",
        [("-5", "sleep length must be non-negative"), ("nan", "Invalid value NaN")],
    )
    def test_r17_reddit_negative_or_nan_retry_after_raises_out_of_the_public_fetch(
        self, monkeypatch, header, reason,
    ):
        _g1_clear_oauth(monkeypatch)
        rate_limited = _g1_http_error(429, "Too Many Requests", {"Retry-After": header})
        with patch.object(reddit, "urlopen", side_effect=[rate_limited]) as op, \
             pytest.raises(ValueError, match=reason):
            reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        # No second request (the backoff itself raised) and no ledger evidence.
        assert op.call_count == 1
        assert quality.snapshot_quality() == []


class TestR17G1RedditOauthLifecycle:
    def test_r17_reddit_token_failure_negative_cache_expires_after_ttl(
        self, monkeypatch,
    ):
        _g1_arm_oauth(monkeypatch)
        t0 = 1_800_000_000.0
        now = _g1_freeze_clock(monkeypatch, t0)
        unauthorized = _g1_http_error(401, "Unauthorized")
        token_ok = _g1_json_resp({"access_token": "abc", "expires_in": 3600})
        with patch.object(
            reddit, "urlopen", side_effect=[unauthorized, token_ok, token_ok],
        ) as op:
            assert reddit._get_oauth_token(5.0) is None
            assert reddit._token_cache == {
                "token": None, "expires_at": 0.0, "neg_until": t0 + 300,
            }
            now[0] = t0 + 299.0
            assert reddit._get_oauth_token(5.0) is None
            assert op.call_count == 1            # coasting on the negative cache
            now[0] = t0 + 301.0
            assert reddit._get_oauth_token(5.0) == "abc"
        assert op.call_count == 2                # retried once the window passed
        assert reddit._token_cache == {
            "token": "abc", "expires_at": t0 + 301.0 + 3600, "neg_until": 0.0,
        }

    def test_r17_reddit_live_token_is_reused_until_the_safety_margin(
        self, monkeypatch,
    ):
        _g1_arm_oauth(monkeypatch)
        t0 = 1_800_000_000.0
        now = _g1_freeze_clock(monkeypatch, t0)
        first = _g1_json_resp({"access_token": "abc", "expires_in": 3600})
        second = _g1_json_resp({"access_token": "abc2", "expires_in": 3600})
        with patch.object(reddit, "urlopen", side_effect=[first, second, second]) as op:
            assert reddit._get_oauth_token(5.0) == "abc"
            now[0] = t0 + 3299.0
            assert reddit._get_oauth_token(5.0) == "abc"
            assert op.call_count == 1            # 301s of life left: served from memory
            now[0] = t0 + 3300.0                 # exactly the 300s margin before expiry
            assert reddit._get_oauth_token(5.0) == "abc2"
        assert op.call_count == 2
        assert reddit._token_cache == {
            "token": "abc2", "expires_at": t0 + 3300.0 + 3600, "neg_until": 0.0,
        }

    @pytest.mark.parametrize(
        "response, fragment",
        [
            (_g1_raise(HTTPError("u", 500, "Server Error", {}, None)),
             "HTTP Error 500: Server Error"),
            (_g1_raise(http.client.IncompleteRead(b"")), "IncompleteRead"),
            (_g1_resp(lambda: b"<html>blocked</html>"), "Expecting value"),
            (_g1_json_resp({"token_type": "bearer"}),
             "token endpoint returned no access_token"),
        ],
    )
    def test_r17_reddit_token_endpoint_failures_are_logged_and_negative_cached(
        self, monkeypatch, caplog, response, fragment,
    ):
        """One member of each exception family the token path handles."""
        _g1_arm_oauth(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.reddit"), \
             patch.object(reddit, "urlopen", return_value=response) as op:
            assert reddit._get_oauth_token(5.0) is None
            assert reddit._get_oauth_token(5.0) is None
        assert op.call_count == 1                # a failed token request is not retried per call
        assert reddit._token_cache["token"] is None
        assert reddit._token_cache["neg_until"] > 0
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert messages[0].startswith("Reddit OAuth token fetch failed: ")
        assert fragment in messages[0]
        assert messages[0].endswith(" — RSS fallback for 300s")
        # The token helper only logs; the ledger event is the caller's job.
        assert quality.snapshot_quality() == []

    def test_r17_reddit_lock_recheck_reuses_a_token_another_thread_fetched(
        self, monkeypatch,
    ):
        _g1_arm_oauth(monkeypatch)
        theirs = {"token": "theirs", "expires_at": time.time() + 7200, "neg_until": 0.0}

        @contextmanager
        def _other_thread_won():
            # Standing in for the lock: another thread populated the cache
            # while this one was waiting for it.
            monkeypatch.setattr(reddit, "_token_cache", theirs)
            yield

        monkeypatch.setattr(reddit, "_token_lock", _other_thread_won())
        with patch.object(
            reddit, "_request_oauth_token", side_effect=AssertionError("second fetch"),
        ) as request:
            assert reddit._get_oauth_token(5.0) == "theirs"
        request.assert_not_called()

    def test_r17_reddit_lock_recheck_honours_a_negative_cache_set_while_waiting(
        self, monkeypatch,
    ):
        _g1_arm_oauth(monkeypatch)
        theirs = {"token": None, "expires_at": 0.0, "neg_until": time.time() + 300}

        @contextmanager
        def _other_thread_won():
            monkeypatch.setattr(reddit, "_token_cache", theirs)
            yield

        monkeypatch.setattr(reddit, "_token_lock", _other_thread_won())
        with patch.object(
            reddit, "_request_oauth_token", side_effect=AssertionError("second fetch"),
        ) as request:
            assert reddit._get_oauth_token(5.0) is None
        request.assert_not_called()


class TestR17G1RedditLedger:
    """What the Reddit source does and does not put in the quality ledger."""

    def test_r17_reddit_keyless_rss_success_records_nothing(self, monkeypatch):
        _g1_clear_oauth(monkeypatch)
        with patch.object(
            reddit, "urlopen", return_value=_g1_resp(lambda: _g1_atom_one.encode()),
        ):
            out = reddit.fetch_reddit_posts(
                "nvda", subreddits=("stocks", "wsb"), inter_request_delay=0,
            )
        assert quality.snapshot_quality() == []
        assert out == _g1_rss_block + "\n\n" + _g1_rss_block.replace("r/stocks", "r/wsb")

    def test_r17_reddit_rss_failure_on_one_sub_is_recorded_and_rendered_as_no_posts(
        self, monkeypatch,
    ):
        """A failed subreddit renders the same '<no posts found>' line as an
        honest empty feed; only the ledger tells the two apart (pinned)."""
        _g1_clear_oauth(monkeypatch)
        blocked = _g1_http_error(403, "Blocked")
        with patch.object(
            reddit, "urlopen",
            side_effect=[_g1_resp(lambda: _g1_atom_one.encode()), blocked],
        ):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks", "investing"))
        assert quality.snapshot_quality() == [{
            "method": "fetch_reddit_posts",
            "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
            "detail": "r/investing RSS failed for NVDA: HTTP Error 403: Blocked",
            "qualifier": "",
        }]
        assert out == (
            _g1_rss_block + "\n\n"
            "r/investing: <no posts found mentioning NVDA in the past 7 days>"
        )

    def test_r17_reddit_token_outage_records_one_event_per_subreddit(self, monkeypatch):
        _g1_arm_oauth(monkeypatch)
        down = _g1_raise(_g1_http_error(500, "Server Error"))
        rss_posts = [{"title": "x", "score": None, "num_comments": None,
                      "created_utc": None, "selftext": "", "source": "rss"}]
        with patch.object(reddit, "urlopen", return_value=down) as op, \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=rss_posts) as rss:
            reddit.fetch_reddit_posts(
                "NVDA", subreddits=("stocks", "wsb"), inter_request_delay=0,
            )
        assert op.call_count == 1                # one token request, then negative cache
        assert rss.call_count == 2
        assert quality.snapshot_quality() == [
            {"method": "fetch_reddit_posts",
             "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
             "detail": "r/stocks OAuth token unavailable for NVDA; using RSS",
             "qualifier": ""},
            {"method": "fetch_reddit_posts",
             "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
             "detail": "r/wsb OAuth token unavailable for NVDA; using RSS",
             "qualifier": ""},
        ]

    def test_r17_reddit_search_failure_then_rss_failure_is_recorded_at_each_step(
        self, monkeypatch,
    ):
        _g1_arm_oauth(monkeypatch)
        unauthorized = _g1_http_error(401, "Unauthorized")
        token_ok = _g1_json_resp({"access_token": "abc", "expires_in": 3600})
        with patch.object(
            reddit, "urlopen", side_effect=[token_ok, unauthorized, unauthorized],
        ) as op:
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert op.call_count == 3                # token, OAuth search, RSS fallback
        events = quality.snapshot_quality()
        assert [e["detail"] for e in events] == [
            "r/stocks OAuth JSON failed for NVDA: HTTP Error 401: Unauthorized; using RSS",
            "r/stocks RSS failed for NVDA: HTTP Error 401: Unauthorized",
            "zero Reddit posts for NVDA across r/stocks",
        ]
        assert {e["kind"] for e in events} == {quality.KIND_OPTIONAL_UNAVAILABLE}
        assert out == "<no Reddit posts found mentioning NVDA across r/stocks in the past 7 days>"

    def test_r17_reddit_all_subs_failing_record_each_failure_plus_the_zero_total(
        self, monkeypatch,
    ):
        _g1_clear_oauth(monkeypatch)
        blocked = _g1_http_error(403, "Blocked")
        with patch.object(reddit, "urlopen", side_effect=[blocked, blocked]):
            out = reddit.fetch_reddit_posts(
                "NVDA", subreddits=("stocks", "wsb"), inter_request_delay=0,
            )
        assert [e["detail"] for e in quality.snapshot_quality()] == [
            "r/stocks RSS failed for NVDA: HTTP Error 403: Blocked",
            "r/wsb RSS failed for NVDA: HTTP Error 403: Blocked",
            "zero Reddit posts for NVDA across r/stocks, r/wsb",
        ]
        assert out == (
            "<no Reddit posts found mentioning NVDA across r/stocks, r/wsb "
            "in the past 7 days>"
        )

    def test_r17_reddit_zero_posts_records_one_event_for_the_whole_call(self, monkeypatch):
        _g1_clear_oauth(monkeypatch)
        with patch.object(
            reddit, "urlopen", return_value=_g1_resp(lambda: _g1_atom_empty.encode()),
        ):
            out = reddit.fetch_reddit_posts(
                "NVDA", subreddits=("stocks", "wsb"), inter_request_delay=0,
            )
        assert out == (
            "<no Reddit posts found mentioning NVDA across r/stocks, r/wsb "
            "in the past 7 days>"
        )
        assert quality.snapshot_quality() == [{
            "method": "fetch_reddit_posts",
            "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
            "detail": "zero Reddit posts for NVDA across r/stocks, r/wsb",
            "qualifier": "",
        }]

    def test_r17_reddit_one_sub_with_posts_keeps_the_ledger_clean(self, monkeypatch):
        _g1_clear_oauth(monkeypatch)
        empty_then_posts = [
            _g1_resp(lambda: _g1_atom_empty.encode()),
            _g1_resp(lambda: _g1_atom_one.encode()),
        ]
        with patch.object(reddit, "urlopen", side_effect=empty_then_posts):
            out = reddit.fetch_reddit_posts(
                "NVDA", subreddits=("stocks", "wsb"), inter_request_delay=0,
            )
        assert quality.snapshot_quality() == []
        assert out == (
            "r/stocks: <no posts found mentioning NVDA in the past 7 days>\n\n"
            + _g1_rss_block.replace("r/stocks", "r/wsb")
        )

    def test_r17_reddit_iterator_subreddits_render_an_empty_list_in_the_zero_placeholder(
        self, monkeypatch,
    ):
        """Pinned and reported: ``subreddits`` is typed ``Iterable`` but the
        zero-posts path iterates it a second time, after the fetch loop
        exhausted it. Every production caller passes the default tuple."""
        _g1_clear_oauth(monkeypatch)
        with patch.object(
            reddit, "urlopen", return_value=_g1_resp(lambda: _g1_atom_empty.encode()),
        ):
            out = reddit.fetch_reddit_posts(
                "NVDA", subreddits=iter(["stocks", "wsb"]), inter_request_delay=0,
            )
        assert out == "<no Reddit posts found mentioning NVDA across  in the past 7 days>"
        assert [e["detail"] for e in quality.snapshot_quality()] == [
            "zero Reddit posts for NVDA across ",
        ]

    def test_r17_reddit_paces_only_between_subreddits(self, monkeypatch):
        _g1_clear_oauth(monkeypatch)
        with patch.object(
            reddit, "urlopen", return_value=_g1_resp(lambda: _g1_atom_empty.encode()),
        ), patch.object(reddit.time, "sleep") as slept:
            reddit.fetch_reddit_posts(
                "NVDA", subreddits=("stocks", "wsb", "investing"),
                inter_request_delay=0.25,
            )
        assert slept.call_args_list == [call(0.25), call(0.25)]


class TestR17G1RedditRssBody:
    def test_r17_reddit_rss_doctype_body_degrades_like_other_unparseable_feeds(
        self, monkeypatch, caplog,
    ):
        """Round-17 product fix. ``safe_xml_root`` (utils.py) refuses a
        DTD/ENTITY-bearing body with ``ValueError``, but this path caught only
        ``ET.ParseError`` — so an HTML interstitial (every real HTML page
        starts with ``<!DOCTYPE``) served with HTTP 200 raised out of
        ``fetch_reddit_posts`` and aborted the sentiment node, instead of
        degrading to the documented empty result + ledger event like every
        other malformed body. The sibling consumer (sec_ownership) already
        catches that ``ValueError``."""
        _g1_clear_oauth(monkeypatch)
        waf_page = b"<!DOCTYPE html><html><body>blocked</body></html>"
        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.reddit"), \
             patch.object(reddit, "urlopen", return_value=_g1_resp(lambda: waf_page)):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert out == (
            "<no Reddit posts found mentioning NVDA across r/stocks in the past 7 days>"
        )
        refused = (
            "reddit_rss: refused XML containing a DTD/ENTITY declaration "
            "(entity-expansion guard); not parsed"
        )
        assert [e["detail"] for e in quality.snapshot_quality()] == [
            f"r/stocks RSS failed for NVDA: {refused}",
            "zero Reddit posts for NVDA across r/stocks",
        ]
        assert [r.getMessage() for r in caplog.records] == [
            f"Reddit RSS fetch failed for r/stocks · NVDA: {refused}",
        ]

    @pytest.mark.parametrize(
        "body, fragment",
        [
            (b"", "no element found: line 1, column 0"),
            (b"<html><body>blocked & retry</body></html>",
             "not well-formed (invalid token): line 1, column 21"),
        ],
    )
    def test_r17_reddit_rss_malformed_xml_degrades_with_evidence(
        self, monkeypatch, caplog, body, fragment,
    ):
        _g1_clear_oauth(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.reddit"), \
             patch.object(reddit, "urlopen", return_value=_g1_resp(lambda: body)):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) == []
        assert [r.getMessage() for r in caplog.records] == [
            f"Reddit RSS fetch failed for r/stocks · NVDA: {fragment}",
        ]
        assert [e["detail"] for e in quality.snapshot_quality()] == [
            f"r/stocks RSS failed for NVDA: {fragment}",
        ]

    def test_r17_reddit_rss_well_formed_html_body_parses_to_zero_entries(self, monkeypatch):
        """Pinned and reported: an HTML page that happens to be well-formed
        XML is indistinguishable from an honest empty feed — no log line, no
        ledger event."""
        _g1_clear_oauth(monkeypatch)
        with patch.object(
            reddit, "urlopen", return_value=_g1_resp(lambda: b"<html><body>x</body></html>"),
        ):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) == []
        assert quality.snapshot_quality() == []

    def test_r17_reddit_rss_entry_limit_and_missing_fields(self, monkeypatch):
        _g1_clear_oauth(monkeypatch)
        feed = _g1_resp(lambda: _g1_atom_three.encode())
        with patch.object(reddit, "urlopen", return_value=feed):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 2, 5.0) == [
                {"title": "One", "score": None, "num_comments": None,
                 "created_utc": 1716215400.0, "selftext": "First", "source": "rss"},
                {"title": "Two", "score": None, "num_comments": None,
                 "created_utc": None, "selftext": "", "source": "rss"},
            ]
        with patch.object(reddit, "urlopen", return_value=feed):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        # A missing and an unparseable <published> both render as "[?]".
        assert out == (
            "r/stocks — 3 recent posts mentioning NVDA "
            "(via RSS feed; scores/comments unavailable):\n"
            "  [2024-05-20] One\n"
            "    body excerpt: First\n"
            "  [?] Two\n"
            "  [?] Three"
        )


class TestR17G1RedditCreds:
    @pytest.mark.parametrize(
        "client_id, secret", [(" ", "sec"), ("cid", "  "), ("", ""), ("cid", "")],
    )
    def test_r17_reddit_blank_or_one_sided_creds_count_as_unset(
        self, monkeypatch, client_id, secret,
    ):
        monkeypatch.setenv("REDDIT_CLIENT_ID", client_id)
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", secret)
        assert reddit._reddit_oauth_creds() is None

    def test_r17_reddit_creds_are_stripped_and_user_agent_defaults(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", " cid ")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", " sec\n")
        assert reddit._reddit_oauth_creds() == ("cid", "sec")
        monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
        assert reddit._oauth_user_agent() == reddit._UA
        monkeypatch.setenv("REDDIT_USER_AGENT", "   ")
        assert reddit._oauth_user_agent() == reddit._UA
        monkeypatch.setenv("REDDIT_USER_AGENT", " my-app/1.0 ")
        assert reddit._oauth_user_agent() == "my-app/1.0"

    def test_r17_reddit_blank_creds_keep_the_rss_dispatch(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", " ")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "")
        monkeypatch.setattr(reddit, "_token_cache", None)
        with patch.object(
            reddit, "_fetch_subreddit_json",
            side_effect=AssertionError("OAuth must not engage on blank creds"),
        ) as oauth, patch.object(reddit, "_fetch_subreddit_rss", return_value=[]) as rss:
            assert reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0) == []
        rss.assert_called_once_with("NVDA", "stocks", 5, 5.0)
        oauth.assert_not_called()


class TestR17G1RedditRequestShape:
    def test_r17_reddit_oauth_requests_carry_token_and_timeout(self, monkeypatch):
        _g1_arm_oauth(monkeypatch)
        monkeypatch.setenv("REDDIT_USER_AGENT", "my-app/1.0")
        seen = []
        responses = iter([
            _g1_json_resp({"access_token": "abc", "expires_in": 3600}),
            _g1_json_resp({"data": {"children": []}}),
        ])

        def _fake_urlopen(req, timeout):
            seen.append((
                req.full_url, req.get_method(), req.get_header("User-agent"),
                req.get_header("Authorization"), req.get_header("Accept"), timeout,
            ))
            return next(responses)

        with patch.object(reddit, "urlopen", side_effect=_fake_urlopen):
            assert reddit._fetch_subreddit_json("NVDA", "stocks", 5, 2.25) == []
        assert seen == [
            ("https://www.reddit.com/api/v1/access_token", "POST", "my-app/1.0",
             "Basic " + base64.b64encode(b"cid:sec").decode(), None, 2.25),
            ("https://oauth.reddit.com/r/stocks/search"
             "?q=NVDA&restrict_sr=on&sort=new&t=week&limit=5",
             "GET", "my-app/1.0", "Bearer abc", "application/json", 2.25),
        ]
        # An empty OAuth listing is an honest miss: no RSS fallback, no event.
        assert quality.snapshot_quality() == []


class TestR17G1InterruptPassThrough:
    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_r17_reddit_and_stocktwits_do_not_swallow_interrupts(
        self, monkeypatch, interrupt,
    ):
        _g1_clear_oauth(monkeypatch)
        with patch.object(reddit, "urlopen", side_effect=interrupt("stop")), \
             pytest.raises(interrupt):
            reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        with patch.object(stocktwits, "urlopen", return_value=_g1_raise(interrupt("stop"))), \
             pytest.raises(interrupt):
            stocktwits.fetch_stocktwits_messages("NVDA")
        assert quality.snapshot_quality() == []


def _g1_st_message(body, sentiment=None, user="u", created_at="2026-08-15T10:00:00Z"):
    message = {"created_at": created_at, "user": {"username": user}, "body": body}
    message["entities"] = (
        {"sentiment": {"basic": sentiment}} if sentiment is not None else {}
    )
    return message


def _g1_st_payload(*messages):
    return json.dumps({"messages": list(messages)}).encode("utf-8")


def _g1_age_stocktwits_cache(tmp_path, payload, age_days):
    """Seed the StockTwits stream cache through one real call, then back-date it."""
    with patch.object(stocktwits, "urlopen", return_value=_g1_resp(lambda: payload)):
        stocktwits.fetch_stocktwits_messages("NVDA")
    cache_file = tmp_path / "vendor-cache" / "stocktwits" / "stream_NVDA.json"
    assert cache_file.is_file()
    backdated = time.time() - age_days * 86_400
    os.utime(cache_file, (backdated, backdated))
    return cache_file


def _g1_epoch(date_str, hour=0, minute=0, second=0):
    """UTC epoch for a calendar instant (the flat-article parser reads epochs as UTC)."""
    moment = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, minute=minute, second=second, tzinfo=UTC,
    )
    return int(moment.timestamp())


def _g1_flat(title, epoch, summary="", link="https://example.com/x", publisher="P"):
    return {
        "title": title, "publisher": publisher, "link": link,
        "summary": summary, "providerPublishTime": epoch,
    }


def _g1_nested(title, pub_date, summary="", link="https://example.com/n", publisher="PN"):
    return {"content": {
        "title": title, "summary": summary,
        "provider": {"displayName": publisher},
        "canonicalUrl": {"url": link}, "pubDate": pub_date,
    }}


def _g1_undated(title):
    return {"title": title, "publisher": "P", "link": "https://example.com/x", "summary": ""}


def _g1_fake_ticker(monkeypatch, articles):
    """``yf.Ticker`` whose ``get_news`` returns ``articles``; returns the call log."""
    seen = []

    class _Ticker:
        def __init__(self_inner, symbol):
            seen.append(symbol)

        def get_news(self_inner, count):
            seen.append(count)
            return articles

    monkeypatch.setattr(yfinance_news.yf, "Ticker", _Ticker)
    monkeypatch.setattr(yfinance_news, "yf_retry", lambda fn, **kw: fn())
    return seen


def _g1_fake_search(monkeypatch, news_for):
    """``yf.Search`` serving ``news_for(query)``; returns the (query, count, fuzzy) log."""
    seen = []

    class _Search:
        def __init__(self_inner, query, news_count=None, enable_fuzzy_query=None):
            seen.append((query, news_count, enable_fuzzy_query))
            self_inner.news = news_for(query)

    monkeypatch.setattr(yfinance_news.yf, "Search", _Search)
    monkeypatch.setattr(yfinance_news, "yf_retry", lambda fn, **kw: fn())
    return seen


class TestR17G1RedditWrongShapePins:
    """Pinned and reported in round 17 (not fixed): the OAuth path degrades to
    RSS on every *failure* (no token, HTTP error, transport, invalid JSON),
    but a 200 body that is valid JSON of the wrong shape raises out of the
    public fetch — the sentiment analyst has no handler around it. Reddit's
    endpoints do not answer that way, so this needs the API to break its
    contract."""

    @pytest.mark.parametrize("body", [b"[]", b'"unauthorized"', b"null"])
    def test_r17_reddit_non_object_token_payload_raises_out_of_the_public_fetch(
        self, monkeypatch, caplog, body,
    ):
        _g1_arm_oauth(monkeypatch)
        with patch.object(reddit, "urlopen", return_value=_g1_resp(lambda: body)), \
             pytest.raises(AttributeError), caplog.at_level(logging.WARNING):
            reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert caplog.records == []
        assert quality.snapshot_quality() == []

    def test_r17_reddit_null_expires_in_raises_out_of_the_public_fetch(self, monkeypatch):
        _g1_arm_oauth(monkeypatch)
        token = _g1_json_resp({"access_token": "abc", "expires_in": None})
        with patch.object(reddit, "urlopen", return_value=token), \
             pytest.raises(TypeError):
            reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))

    def test_r17_reddit_non_object_search_listing_raises_out_of_the_public_fetch(
        self, monkeypatch,
    ):
        _g1_arm_oauth(monkeypatch)
        token = _g1_json_resp({"access_token": "abc", "expires_in": 3600})
        with patch.object(
            reddit, "urlopen", side_effect=[token, _g1_resp(lambda: b"[]")],
        ), pytest.raises(AttributeError):
            reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))


class TestR17G1StocktwitsFormat:
    def test_r17_stocktwits_summary_and_lines_exact(self):
        raw = _g1_st_payload(
            _g1_st_message("Bull case\nintact", "Bullish", user="alice"),
            _g1_st_message("Bear case", "Bearish", user="bob",
                           created_at="2026-08-15T09:00:00Z"),
            _g1_st_message("no view", created_at="2026-08-15T08:00:00Z"),
        )
        with patch.object(stocktwits, "urlopen", return_value=_g1_resp(lambda: raw)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out == (
            "Bullish: 1 (33%) · Bearish: 1 (33%) · Unlabeled: 1 · "
            "Total: 3 most-recent messages\n\n"
            "[2026-08-15T10:00:00Z · @alice · Bullish] Bull case intact\n"
            "[2026-08-15T09:00:00Z · @bob · Bearish] Bear case\n"
            "[2026-08-15T08:00:00Z · @u · no-label] no view"
        )

    def test_r17_stocktwits_label_match_is_case_sensitive_and_parts_default(self):
        raw = _g1_st_payload(
            {"created_at": "2026-08-15T10:00:00Z", "body": "lower",
             "entities": {"sentiment": {"basic": "bullish"}}},
            {"created_at": "2026-08-15T09:00:00Z", "body": "no user no entities"},
            {"created_at": "2026-08-15T08:00:00Z", "user": {"username": "dave"},
             "body": "sentiment as text", "entities": {"sentiment": "Bullish"}},
            {"created_at": "2026-08-15T07:00:00Z", "user": None,
             "entities": None, "body": None},
        )
        with patch.object(stocktwits, "urlopen", return_value=_g1_resp(lambda: raw)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out == (
            "Bullish: 0 (0%) · Bearish: 0 (0%) · Unlabeled: 4 · "
            "Total: 4 most-recent messages\n\n"
            "[2026-08-15T10:00:00Z · @? · no-label] lower\n"
            "[2026-08-15T09:00:00Z · @? · no-label] no user no entities\n"
            "[2026-08-15T08:00:00Z · @dave · no-label] sentiment as text\n"
            "[2026-08-15T07:00:00Z · @? · no-label] "
        )

    @pytest.mark.parametrize("bullish, pct", [(1, 12), (3, 38), (5, 62), (7, 88)])
    def test_r17_stocktwits_percentages_round_half_to_even(self, bullish, pct):
        # 1/8, 3/8, 5/8 and 7/8 of eight messages are exact .5 percentages.
        raw = _g1_st_payload(*(
            _g1_st_message("m", "Bullish" if i < bullish else None) for i in range(8)
        ))
        with patch.object(stocktwits, "urlopen", return_value=_g1_resp(lambda: raw)):
            out = stocktwits.fetch_stocktwits_messages(f"T{bullish}")
        assert out.splitlines()[0] == (
            f"Bullish: {bullish} ({pct}%) · Bearish: 0 (0%) · "
            f"Unlabeled: {8 - bullish} · Total: 8 most-recent messages"
        )

    def test_r17_stocktwits_limit_slices_the_stream_and_is_not_in_the_cache_key(self):
        raw = _g1_st_payload(*(_g1_st_message(f"m{i}", user=f"u{i}") for i in range(5)))
        with patch.object(stocktwits, "urlopen", return_value=_g1_resp(lambda: raw)) as op:
            two = stocktwits.fetch_stocktwits_messages("NVDA", limit=2)
            one = stocktwits.fetch_stocktwits_messages("NVDA", limit=1)
        assert op.call_count == 1            # the raw stream is cached; limit applies on read
        assert two == (
            "Bullish: 0 (0%) · Bearish: 0 (0%) · Unlabeled: 2 · "
            "Total: 2 most-recent messages\n\n"
            "[2026-08-15T10:00:00Z · @u0 · no-label] m0\n"
            "[2026-08-15T10:00:00Z · @u1 · no-label] m1"
        )
        assert one == (
            "Bullish: 0 (0%) · Bearish: 0 (0%) · Unlabeled: 1 · "
            "Total: 1 most-recent messages\n\n"
            "[2026-08-15T10:00:00Z · @u0 · no-label] m0"
        )

    def test_r17_stocktwits_request_shape_timeout_and_cache_file(self, monkeypatch):
        seen = {}

        def _fake_urlopen(req, timeout):
            seen.update(
                url=req.full_url, method=req.get_method(),
                ua=req.get_header("User-agent"), accept=req.get_header("Accept"),
                timeout=timeout,
            )
            return _g1_resp(lambda: _g1_st_payload(_g1_st_message("hi", "Bearish")))

        with patch.object(stocktwits, "urlopen", side_effect=_fake_urlopen):
            out = stocktwits.fetch_stocktwits_messages("brk.b", limit=3, timeout=4.5)
        assert seen == {
            "url": "https://api.stocktwits.com/api/2/streams/symbol/BRK.B.json",
            "method": "GET",
            "ua": stocktwits._UA,
            "accept": "application/json",
            "timeout": 4.5,
        }
        assert out == (
            "Bullish: 0 (0%) · Bearish: 1 (100%) · Unlabeled: 0 · "
            "Total: 1 most-recent messages\n\n"
            "[2026-08-15T10:00:00Z · @u · Bearish] hi"
        )
        assert (Path(get_config()["data_cache_dir"]) / "stocktwits"
                / "stream_BRK.B.json").is_file()

    def test_r17_stocktwits_non_object_json_bodies_read_as_no_messages(self, caplog):
        bodies = [b"[]", b'"ok"', b"123", b'{"messages": null}']
        with caplog.at_level(logging.WARNING):
            for i, body in enumerate(bodies):
                with patch.object(
                    stocktwits, "urlopen", return_value=_g1_resp(lambda b=body: b),
                ) as op:
                    out = stocktwits.fetch_stocktwits_messages(f"T{i}")
                assert op.call_count == 1
                assert out == f"<no StockTwits messages found for $T{i}>"
        assert caplog.records == []
        assert quality.snapshot_quality() == []

    def test_r17_stocktwits_non_list_messages_crash_out_of_the_tool(self, caplog):
        """Pinned and reported (not fixed): the docstring promises a
        placeholder for an unexpected shape, but only a non-object body and a
        missing or null ``messages`` degrade. A non-list ``messages`` (or a
        non-dict message) raises out of the fetch, unlogged and unrecorded;
        StockTwits' stream shape is stable, so this needs the API to break
        its contract."""
        with patch.object(
            stocktwits, "urlopen",
            return_value=_g1_resp(lambda: b'{"messages": "abc"}'),
        ), pytest.raises(AttributeError, match="no attribute 'get'"), \
             caplog.at_level(logging.WARNING):
            stocktwits.fetch_stocktwits_messages("NVDA")
        assert caplog.records == []
        assert quality.snapshot_quality() == []


class TestR17G1StocktwitsFailure:
    @pytest.mark.parametrize("exc", [
        _g1_http_error(503, "down"), TimeoutError("slow"),
        http.client.IncompleteRead(b""),
    ])
    def test_r17_stocktwits_transport_failure_placeholder_log_and_sentinel(
        self, caplog, exc,
    ):
        name = type(exc).__name__
        with caplog.at_level(logging.WARNING), \
             patch.object(stocktwits, "urlopen", return_value=_g1_raise(exc)) as op:
            first = stocktwits.fetch_stocktwits_messages("NVDA")
            second = stocktwits.fetch_stocktwits_messages("NVDA")
        assert op.call_count == 2            # a failure is never cached
        assert first == second == f"<stocktwits unavailable: {name}>"
        expected_event = {
            "method": "fetch_stocktwits_messages",
            "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
            "detail": f"transport/parse failure: {name}: {exc}",
            "qualifier": "",
        }
        assert [(r.name, r.getMessage()) for r in caplog.records] == [
            ("yialpha.dataflows.stocktwits", f"StockTwits fetch failed for NVDA: {exc}"),
        ] * 2
        assert quality.snapshot_quality() == [expected_event, expected_event]


class TestR17G1StocktwitsStaleCache:
    """Pinned and reported (not fixed): StockTwits passes no ``stale_cap_days``
    to the shared disk cache, so the GLOBAL cap (30 days) governs a
    5-minute-TTL social feed. Binance Square, the other social feed, caps its
    stale serves at 15 minutes and renders a STALE marker; StockTwits renders
    old messages under "most-recent messages" with only the ledger and the
    log to say so."""

    _STALE_RENDER = (
        "Bullish: 1 (100%) · Bearish: 0 (0%) · Unlabeled: 0 · "
        "Total: 1 most-recent messages\n\n"
        "[2026-08-15T10:00:00Z · @then · Bullish] old chatter"
    )

    def test_r17_stocktwits_day_old_stream_is_served_when_the_fetch_fails(
        self, tmp_path, caplog,
    ):
        set_config({"data_cache_max_stale_days": 30})
        payload = _g1_st_payload(_g1_st_message("old chatter", "Bullish", user="then"))
        cache_file = _g1_age_stocktwits_cache(tmp_path, payload, age_days=1)
        rate_limited = _g1_http_error(429, "Too Many Requests")
        with caplog.at_level(logging.WARNING), \
             patch.object(stocktwits, "urlopen", return_value=_g1_raise(rate_limited)) as op:
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert op.call_count == 1
        assert out == self._STALE_RENDER
        assert "STALE" not in out
        assert [(r.name, r.getMessage()) for r in caplog.records] == [(
            "yialpha.dataflows.disk_cache",
            f"stocktwits: fetch failed; serving STALE cache for {cache_file} (age 1.0 days)",
        )]
        assert quality.snapshot_quality() == [{
            "method": "stocktwits/stream_NVDA.json",
            "kind": quality.KIND_STALE_CACHE,
            "detail": "served stale cache (age 1.0 days) after fetch failure",
            "qualifier": "",
        }]

    def test_r17_stocktwits_stream_past_the_global_cap_is_refused(self, tmp_path, caplog):
        set_config({"data_cache_max_stale_days": 30})
        payload = _g1_st_payload(_g1_st_message("old chatter", "Bullish", user="then"))
        cache_file = _g1_age_stocktwits_cache(tmp_path, payload, age_days=31)
        rate_limited = _g1_http_error(429, "Too Many Requests")
        with caplog.at_level(logging.WARNING), \
             patch.object(stocktwits, "urlopen", return_value=_g1_raise(rate_limited)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out == "<stocktwits unavailable: HTTPError>"
        assert sorted((r.name, r.getMessage()) for r in caplog.records) == sorted([
            ("yialpha.dataflows.disk_cache",
             f"stocktwits: fetch failed and cache {cache_file} is 31.0 days old "
             "(over the 30.0000-day cap); refusing to serve stale data"),
            ("yialpha.dataflows.stocktwits",
             "StockTwits fetch failed for NVDA: HTTP Error 429: Too Many Requests"),
        ])
        assert quality.snapshot_quality() == [{
            "method": "fetch_stocktwits_messages",
            "kind": quality.KIND_OPTIONAL_UNAVAILABLE,
            "detail": "transport/parse failure: HTTPError: HTTP Error 429: Too Many Requests",
            "qualifier": "",
        }]


class TestR17G1YfTickerNews:
    def test_r17_yfinance_news_ticker_renders_summary_and_link_for_flat_and_nested(self, monkeypatch):
        articles = [
            _g1_flat("Flat headline", _g1_epoch("2025-05-05"), summary="Flat summary."),
            _g1_nested("Nested headline", "2025-05-05T10:00:00Z", summary="Nested summary."),
            _g1_flat("Bare headline", _g1_epoch("2025-05-06"), link=""),
            {"publisher": "P", "link": "",
             "providerPublishTime": _g1_epoch("2025-05-06")},
        ]
        seen = _g1_fake_ticker(monkeypatch, articles)
        out = yfinance_news.get_news_yfinance("AAPL", "2025-05-01", "2025-05-09")
        assert seen == ["AAPL", 20]             # news_article_limit from the default config
        assert out == (
            "## AAPL News, from 2025-05-01 to 2025-05-09:\n\n"
            "### Flat headline (source: P)\nFlat summary.\nLink: https://example.com/x\n\n"
            "### Nested headline (source: PN)\nNested summary.\nLink: https://example.com/n\n\n"
            "### Bare headline (source: P)\n\n"
            "### No title (source: P)\n\n"
        )

    def test_r17_yfinance_news_ticker_nothing_in_window_is_a_typed_unlogged_soft_miss(
        self, monkeypatch, caplog,
    ):
        articles = [
            _g1_flat("Too old", _g1_epoch("2025-04-30", 23, 59, 59)),
            _g1_flat("Future", _g1_epoch("2025-05-10", 0, 0, 1)),
            _g1_undated("Undated"),
            {"content": {"title": "Nested undated"}},
        ]
        _g1_fake_ticker(monkeypatch, articles)
        with caplog.at_level(logging.WARNING), \
             pytest.raises(NoMarketDataError) as missed:
            yfinance_news.get_news_yfinance("AAPL", "2025-05-01", "2025-05-09")
        assert str(missed.value) == (
            "No market data for 'AAPL': no yfinance news within 2025-05-01..2025-05-09"
        )
        assert missed.value.detail == "no yfinance news within 2025-05-01..2025-05-09"
        assert caplog.records == []             # the designed soft-miss is not a fault

    def test_r17_yfinance_news_ticker_empty_feed_is_a_typed_soft_miss_naming_both_symbols(
        self, monkeypatch,
    ):
        monkeypatch.setattr(yfinance_news, "normalize_symbol", lambda ticker: "CANON")
        _g1_fake_ticker(monkeypatch, [])
        with pytest.raises(NoMarketDataError) as missed:
            yfinance_news.get_news_yfinance("ALIAS", "2025-05-01", "2025-05-09")
        assert str(missed.value) == (
            "No market data for 'ALIAS' (queried as 'CANON'): "
            "yfinance returned no news items"
        )

    def test_r17_yfinance_news_ticker_alias_is_resolved_and_disclosed(self, monkeypatch):
        monkeypatch.setattr(yfinance_news, "normalize_symbol", lambda ticker: "CANON")
        seen = _g1_fake_ticker(monkeypatch, [_g1_flat("H", _g1_epoch("2025-05-05"), link="")])
        out = yfinance_news.get_news_yfinance("ALIAS", "2025-05-01", "2025-05-09")
        assert seen == ["CANON", 20]
        assert out == (
            "## ALIAS (resolved to CANON) News, from 2025-05-01 to 2025-05-09:\n\n"
            "### H (source: P)\n\n"
        )

    def test_r17_yfinance_news_ticker_datetime_form_bounds(self, monkeypatch, caplog):
        """The vendor does not normalise a timestamp-form bound: unpinned it is
        a logged fault (the LLM-facing tool layer cuts it to ``YYYY-MM-DD``
        before the router); a pinned analysis date clamps it, which
        canonicalises it as a side effect."""
        seen = _g1_fake_ticker(monkeypatch, [_g1_flat("H", _g1_epoch("2025-05-05"))])
        with caplog.at_level(logging.ERROR), pytest.raises(ValueError):
            yfinance_news.get_news_yfinance("AAPL", "2025-05-01", "2025-05-09 16:00:00")
        assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
            ("ERROR", "news retrieval failed for AAPL"),
        ]
        assert seen == ["AAPL", 20]             # the fetch runs before the bounds are parsed
        with pinned_analysis_date("2025-05-05"):
            out = yfinance_news.get_news_yfinance(
                "AAPL", "2025-05-01", "2025-05-09 16:00:00",
            )
        assert out == (
            "## AAPL News, from 2025-05-01 to 2025-05-05:\n\n"
            "### H (source: P)\nLink: https://example.com/x\n\n"
        )


class TestR17G1YfNewsWindow:
    def test_r17_yfinance_news_window_boundaries(self):
        start, end = datetime(2025, 5, 1), datetime(2025, 5, 9)
        in_window = yfinance_news._in_news_window
        assert in_window(datetime(2025, 5, 1, 0, 0, 0, tzinfo=UTC), start, end) is True
        assert in_window(datetime(2025, 4, 30, 23, 59, 59, tzinfo=UTC), start, end) is False
        assert in_window(datetime(2025, 5, 9, 23, 59, 59, tzinfo=UTC), start, end) is True
        # The upper bound is end_date + 1 day compared with ``<=``: the
        # midnight instant AFTER the end date is still inside the window.
        assert in_window(datetime(2025, 5, 10, 0, 0, 0, tzinfo=UTC), start, end) is True
        assert in_window(datetime(2025, 5, 10, 0, 0, 1, tzinfo=UTC), start, end) is False

    def test_r17_yfinance_news_midnight_after_the_end_date_is_rendered(self, monkeypatch):
        """Pinned and reported (not fixed): one instant of the day AFTER
        ``end_date`` — exactly 00:00:00 — is rendered inside a report whose
        header says the window ends on ``end_date``."""
        articles = [
            _g1_flat("At midnight", _g1_epoch("2025-05-10")),
            _g1_flat("One second later", _g1_epoch("2025-05-10", 0, 0, 1)),
        ]
        _g1_fake_ticker(monkeypatch, articles)
        out = yfinance_news.get_news_yfinance("AAPL", "2025-05-01", "2025-05-09")
        assert out == (
            "## AAPL News, from 2025-05-01 to 2025-05-09:\n\n"
            "### At midnight (source: P)\nLink: https://example.com/x\n\n"
        )

    def test_r17_yfinance_news_undated_articles_need_a_window_reaching_today(self, monkeypatch):
        """With "today" frozen: an undated article is kept when the window's
        end date is today and dropped when it is yesterday — the documented
        live workflow analyses on the previous close, so undated headlines
        vanish from those runs."""
        class _Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 11, 25, 10, 0, 0)

        monkeypatch.setattr(yfinance_news, "datetime", _Clock)
        _g1_fake_ticker(monkeypatch, [_g1_undated("Undated")])
        with pytest.raises(NoMarketDataError):
            yfinance_news.get_news_yfinance("AAPL", "2026-11-24", "2026-11-24")
        kept = yfinance_news.get_news_yfinance("AAPL", "2026-11-25", "2026-11-25")
        assert kept == (
            "## AAPL News, from 2026-11-25 to 2026-11-25:\n\n"
            "### Undated (source: P)\nLink: https://example.com/x\n\n"
        )

    def test_r17_yfinance_news_unparseable_publish_time_fails_closed_in_a_historical_window(
        self, monkeypatch,
    ):
        articles = [
            {"title": "String epoch", "publisher": "P", "link": "l", "summary": "",
             "providerPublishTime": "1714521600"},
            {"title": "Zero", "publisher": "P", "link": "l", "summary": "",
             "providerPublishTime": 0},
        ]
        assert [yfinance_news._extract_article_data(a)["pub_date"] for a in articles] == [None, None]
        _g1_fake_ticker(monkeypatch, articles)
        with pytest.raises(NoMarketDataError):
            yfinance_news.get_news_yfinance("AAPL", "2025-05-01", "2025-05-09")


class TestR17G1YfGlobalNews:
    def test_r17_yfinance_news_global_defaults_come_from_config(self, monkeypatch):
        config = get_config()
        queries = config["global_news_queries"]
        assert (config["global_news_lookback_days"], config["global_news_article_limit"]) == (7, 10)
        assert len(queries) == 5
        seen = _g1_fake_search(
            monkeypatch,
            lambda q: [_g1_flat(f"HEAD {q}", _g1_epoch("2025-05-07"),
                               summary=f"About {q}", link="")],
        )
        out = yfinance_news.get_global_news_yfinance("2025-05-09")
        assert seen == [(query, 10, True) for query in queries]
        assert out == (
            "## Global Market News, from 2025-05-02 to 2025-05-09:\n\n"
            + "".join(
                f"### HEAD {query} (source: P)\nAbout {query}\n\n" for query in queries
            )
        )

    def test_r17_yfinance_news_global_explicit_args_override_the_config(self, monkeypatch):
        seen = _g1_fake_search(monkeypatch, lambda q: [])
        with pytest.raises(
            NoMarketDataError, match="no yfinance global news within 2025-05-06..2025-05-09",
        ):
            yfinance_news.get_global_news_yfinance("2025-05-09", look_back_days=3, limit=2)
        assert [s[1] for s in seen] == [2] * 5   # every query asked, two articles each

    def test_r17_yfinance_news_search_cache_is_keyed_by_query_and_count(self, monkeypatch):
        seen = _g1_fake_search(
            monkeypatch, lambda q: [_g1_flat(f"HEAD {q}", _g1_epoch("2025-05-07"), link="")],
        )
        first = yfinance_news.get_global_news_yfinance("2025-05-09", limit=3)
        assert [count for _, count, _ in seen] == [3, 3, 3]   # limit reached after 3 queries
        again = yfinance_news.get_global_news_yfinance("2025-05-09", limit=3)
        assert len(seen) == 3                               # served from disk
        assert first == again and again.count("### HEAD ") == 3
        wider = yfinance_news.get_global_news_yfinance("2025-05-09", limit=4)
        assert [count for _, count, _ in seen] == [3, 3, 3, 4, 4, 4, 4]   # count is in the key
        assert wider.count("### HEAD ") == 4
        entries = list((Path(get_config()["data_cache_dir"]) / "yfnews").iterdir())
        assert len(entries) == 7                            # 3 + 4 distinct (query, count) pairs
        assert all(p.name.startswith("search_") and len(p.stem) == 19 for p in entries)

    def test_r17_yfinance_news_cached_search_results_are_window_filtered_on_every_call(
        self, monkeypatch,
    ):
        seen = _g1_fake_search(
            monkeypatch, lambda q: [_g1_flat("MAY STORY", _g1_epoch("2025-05-07"), link="")],
        )
        assert "MAY STORY" in yfinance_news.get_global_news_yfinance("2025-05-09", limit=1)
        assert len(seen) == 1
        with pytest.raises(
            NoMarketDataError, match="no yfinance global news within 2025-04-01..2025-04-08",
        ):
            yfinance_news.get_global_news_yfinance("2025-04-08", limit=1)
        # The first query was a cache hit (not fetched again); the other four
        # were misses. The cached May headline was dropped by the window filter.
        assert [query for query, _, _ in seen] == get_config()["global_news_queries"]

    def test_r17_yfinance_news_global_dedups_titles_across_queries(self, monkeypatch):
        queries = get_config()["global_news_queries"]
        shelves = {
            queries[0]: [
                _g1_nested("A", "2025-05-08T10:00:00Z", summary="first copy"),
                _g1_nested("A", "2025-05-08T11:00:00Z", summary="duplicate copy"),
                _g1_nested("B", "2025-05-08T09:00:00Z"),
            ],
            queries[1]: [
                _g1_nested("A", "2025-05-08T12:00:00Z"),
                _g1_nested("C", "2025-05-08T08:00:00Z"),
            ],
        }
        seen = _g1_fake_search(monkeypatch, lambda q: shelves.get(q, []))
        out = yfinance_news.get_global_news_yfinance("2025-05-09", limit=10)
        assert len(seen) == 5
        heads = [line for line in out.splitlines() if line.startswith("### ")]
        assert heads == ["### A (source: PN)", "### B (source: PN)", "### C (source: PN)"]
        assert "first copy" in out and "duplicate copy" not in out
        limited = yfinance_news.get_global_news_yfinance("2025-05-09", limit=2)
        assert seen[5:] == [(queries[0], 2, True)]          # quota met by the first query
        assert [line for line in limited.splitlines() if line.startswith("### ")] == [
            "### A (source: PN)", "### B (source: PN)",
        ]

    def test_r17_yfinance_news_poisoned_cache_entry_fails_visibly_without_a_refetch(
        self, monkeypatch, caplog,
    ):
        """A fresh-but-corrupt cache entry is served as written: the call
        fails with a logged ERROR (the router sees the fault) and nothing is
        re-fetched until the 30-minute TTL expires."""
        query = get_config()["global_news_queries"][0]
        key_blob = json.dumps({"q": query, "n": 10}, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(key_blob).hexdigest()[:12]
        cache_file = Path(get_config()["data_cache_dir"]) / "yfnews" / f"search_{digest}.json"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_bytes(b"<html>not json</html>")
        seen = _g1_fake_search(monkeypatch, lambda q: [])
        with caplog.at_level(logging.ERROR), pytest.raises(json.JSONDecodeError):
            yfinance_news.get_global_news_yfinance("2025-05-09")
        assert seen == []
        assert [r.getMessage() for r in caplog.records] == ["global news retrieval failed"]

    def test_r17_yfinance_news_global_interrupt_is_not_swallowed(self, monkeypatch, caplog):
        def _boom(*args, **kwargs):
            raise KeyboardInterrupt("stop")

        monkeypatch.setattr(yfinance_news.yf, "Search", _boom)
        monkeypatch.setattr(yfinance_news, "yf_retry", lambda fn, **kw: fn())
        with caplog.at_level(logging.ERROR), pytest.raises(KeyboardInterrupt):
            yfinance_news.get_global_news_yfinance("2025-05-09")
        assert caplog.records == []


_g1_no_json = object()
_g1_tavily_key_env = "TAVILY" + "_API_KEY"
_g1_tavily_keys_env = "TAVILY_API" + "_KEYS"
_g1_tavily_max_env = "YIALPHA_TAVILY_MAX_CALLS" + "_PER_RUN"
_g1_tavily_split_env = "YIALPHA_TAVILY_BUDGET" + "_SPLIT"
_g1_tavily_timeout_env = "YIALPHA_TAVILY_TIMEOUT" + "_S"
_g1_tavily_alpha = "tvly-dev-" + "r17-alpha"
_g1_tavily_beta = "tvly-dev-" + "r17-beta"
_g1_tavily_tail = (
    " Do not fabricate web findings; state plainly that open-web context is "
    "unavailable and continue with the structured data tools."
)
_g1_tavily_no_results = (
    "Web search for 'q': no results found. Do not speculate about what such "
    "results might have said."
)
_g1_tavily_footer = (
    "_Numbers appearing above are unverified snippet text — cite as "
    "qualitative context only; data values must come from the structured "
    "data tools._"
)


def _g1_tavily_response(payload=None, status=200):
    """A ``requests.Response`` stand-in; ``payload=_g1_no_json`` makes ``json()`` raise."""
    class _Response:
        def __init__(self_inner):
            self_inner.status_code = status

        def raise_for_status(self_inner):
            if status >= 400:
                raise requests.exceptions.HTTPError(f"{status} error", response=self_inner)

        def json(self_inner):
            if payload is _g1_no_json:
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return payload
    return _Response()


def _g1_tavily_post(responder):
    """``requests.post`` fake with the vendor's keyword signature; returns the call log."""
    seen = []

    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        seen.append({
            "url": url, "json": json, "headers": headers, "timeout": timeout,
        })
        return responder(headers, timeout)
    return seen, _post


def _g1_tavily_fresh(monkeypatch, *, keys=(_g1_tavily_alpha,), max_calls=None, split=None):
    """A bound run context, a reset budget and a controlled key / cap / split picture."""
    quality.ensure_run_context()
    tavily.reset_run_budget()
    for var in (_g1_tavily_key_env, _g1_tavily_keys_env, _g1_tavily_max_env,
                _g1_tavily_split_env, _g1_tavily_timeout_env):
        monkeypatch.delenv(var, raising=False)
    if keys:
        monkeypatch.setenv(_g1_tavily_keys_env, ",".join(keys))
    if max_calls is not None:
        monkeypatch.setenv(_g1_tavily_max_env, str(max_calls))
    if split is not None:
        monkeypatch.setenv(_g1_tavily_split_env, split)


class TestR17G1TavilyBudgetSplit:
    def test_r17_tavily_parse_budget_split_tolerant_syntax(self):
        parse = tavily.parse_budget_split
        assert parse(" news : 3 , market: 2") == {"news": 3, "market": 2}
        # int() syntax: a leading plus, digit underscores and Unicode digits.
        assert parse("news:+3,market:1_0,fundamentals:１０") == {
            "news": 3, "market": 10, "fundamentals": 10,
        }
        assert parse("news:1,news:2") == {"news": 2}        # a repeated scope: last wins
        assert parse("News:3") == {"News": 3}               # scope names are case-sensitive
        assert parse("anything:0") == {"anything": 0}

    @pytest.mark.parametrize("raw", [
        "news:nan", "news:inf", "news:1.5", "news:1e3", "news: 0x10", "news:-1",
        "news:1,,market:2", "news:1,market:2,", ":3", ",", "news:", "",
    ])
    def test_r17_tavily_parse_budget_split_rejects_malformed_values(self, caplog, raw):
        # Any bad part discards the WHOLE split (never a partial dictionary).
        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.tavily"):
            assert tavily.parse_budget_split(raw) is None
        assert [r.getMessage() for r in caplog.records] == [
            f"Ignoring malformed YIALPHA_TAVILY_BUDGET_SPLIT={raw!r}; expected "
            "'scope:count' pairs, e.g. 'news:3,market:2,fundamentals:1'."
        ]

    def test_r17_tavily_all_zero_explicit_split_disables_every_scope(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch, split="news:0,market:0,fundamentals:0")
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response({"results": []}))
        with patch.object(tavily.requests, "post", post):
            out = tavily.get_web_search("q", scope="market")
        assert seen == []                               # refused before any HTTP
        assert out == (
            "WEB_SEARCH_UNAVAILABLE: per-run search budget exhausted for the market "
            "analyst (0/0 calls used; raise YIALPHA_TAVILY_BUDGET_SPLIT or "
            "YIALPHA_TAVILY_MAX_CALLS_PER_RUN if more angles are needed)."
            + _g1_tavily_tail
        )
        assert [e["detail"] for e in quality.snapshot_quality()] == [
            "per-run search budget exhausted for the market analyst (0/0 calls used; raise "
            "YIALPHA_TAVILY_BUDGET_SPLIT or YIALPHA_TAVILY_MAX_CALLS_PER_RUN if more angles "
            "are needed)."
        ]

    def test_r17_tavily_scaled_default_split_matches_an_independent_largest_remainder(self):
        from fractions import Fraction

        weights = tavily.DEFAULT_BUDGET_SPLIT
        denom = sum(weights.values())
        scaled = tavily._scaled_default_split
        for total in range(1, 61):
            got = scaled(total)
            assert sum(got.values()) == total            # the parts always make the whole
            assert all(part >= 0 for part in got.values())
            # Independent reference: exact quotas, floor, then largest
            # fractional remainder with ties broken toward the heavier scope.
            quota = {scope: Fraction(total * weight, denom) for scope, weight in weights.items()}
            expected = {scope: q.numerator // q.denominator for scope, q in quota.items()}
            order = sorted(weights, key=lambda s: (-(quota[s] - expected[s]), -weights[s]))
            for scope in order[: total - sum(expected.values())]:
                expected[scope] += 1
            assert got == expected
        zero = dict.fromkeys(weights, 0)
        assert scaled(0) == zero and scaled(-5) == zero
        assert scaled(denom) == dict(weights)
        assert scaled(1) == {"news": 1, "market": 0, "fundamentals": 0}

    def test_r17_tavily_explicit_split_below_the_total_binds_first(self, monkeypatch):
        """An explicit split replaces the default wholesale and is never
        rescaled to the total: its parts may sum below it (the scope caps bind
        first) or above it (the total binds first, pinned already)."""
        _g1_tavily_fresh(monkeypatch, max_calls=10, split="news:2,market:1")
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response({"results": []}))
        with patch.object(tavily.requests, "post", post):
            tavily.get_web_search("n0")
            tavily.get_web_search("n1")
            third_news = tavily.get_web_search("n2")
            tavily.get_web_search("m0", scope="market")
            second_market = tavily.get_web_search("m1", scope="market")
            unallocated = tavily.get_web_search("f0", scope="fundamentals")
        assert len(seen) == 3                            # 3 of the 10-call total were ever spent
        assert third_news == (
            "WEB_SEARCH_UNAVAILABLE: per-run search budget exhausted for the news analyst "
            "(2/2 calls used; raise YIALPHA_TAVILY_BUDGET_SPLIT or "
            "YIALPHA_TAVILY_MAX_CALLS_PER_RUN if more angles are needed)." + _g1_tavily_tail
        )
        assert second_market.startswith(
            "WEB_SEARCH_UNAVAILABLE: per-run search budget exhausted for the market analyst "
            "(1/1 calls used; "
        )
        assert unallocated == (
            "WEB_SEARCH_UNAVAILABLE: scope 'fundamentals' has no budget allocation in "
            "YIALPHA_TAVILY_BUDGET_SPLIT (allocated scopes: market, news); ask the operator "
            "to allocate it." + _g1_tavily_tail
        )

    def test_r17_tavily_unknown_scope_is_denied_and_not_charged(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch)
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response({"results": []}))
        with patch.object(tavily.requests, "post", post):
            out = tavily.get_web_search("q", scope="macro")
        assert out == (
            "WEB_SEARCH_UNAVAILABLE: scope 'macro' has no budget allocation in "
            "YIALPHA_TAVILY_BUDGET_SPLIT (allocated scopes: fundamentals, market, news); "
            "ask the operator to allocate it." + _g1_tavily_tail
        )
        assert seen == []
        assert sum(tavily.run_usage().values()) == 0

    @pytest.mark.parametrize("raw, expected", [
        ("7", 7), (" 5 ", 5), ("+2", 2), ("1_0", 10), ("0", 0), ("-4", 0),
        ("1e3", 15), ("3.0", 15), ("inf", 15), ("nan", 15),
    ])
    def test_r17_tavily_max_calls_env(self, monkeypatch, caplog, raw, expected):
        _g1_tavily_fresh(monkeypatch)
        monkeypatch.setenv(_g1_tavily_max_env, raw)
        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.tavily"):
            assert tavily._max_calls_per_run() == expected
        if raw in ("1e3", "3.0", "inf", "nan"):
            assert [r.getMessage() for r in caplog.records] == [
                f"Ignoring non-numeric YIALPHA_TAVILY_MAX_CALLS_PER_RUN={raw!r}"
            ]
        else:
            assert caplog.records == []


class TestR17G1TavilyKeyFailover:
    def test_r17_tavily_dead_pool_from_an_earlier_call_skips_http(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch, keys=[_g1_tavily_alpha, _g1_tavily_beta])
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response(status=401))
        with patch.object(tavily.requests, "post", post):
            first = tavily.get_web_search("q1")
            second = tavily.get_web_search("q2")
        assert len(seen) == 2                # both keys tried once, in the first call only
        rotated = "returned HTTP 401; removed from this run's pool and rotating to the next key"
        assert first == (
            "WEB_SEARCH_UNAVAILABLE: all 2 Tavily keys are unavailable this run (last: HTTP 401). "
            "Get fresh keys at https://app.tavily.com and update TAVILY_API_KEYS." + _g1_tavily_tail
        )
        # The second call never reaches Tavily, and the earlier failures' status
        # is not carried over: the sentinel reports "HTTP ?".
        assert second == first.replace("(last: HTTP 401)", "(last: HTTP ?)")
        assert tavily.run_usage() == {"news": 2, "market": 0, "fundamentals": 0}
        assert [e["detail"] for e in quality.snapshot_quality()] == [
            f"key #1/2 {rotated}",
            f"key #2/2 {rotated}",
            "all 2 Tavily keys are unavailable this run (last: HTTP 401). Get fresh keys at "
            "https://app.tavily.com and update TAVILY_API_KEYS.",
            "all 2 Tavily keys are unavailable this run (last: HTTP ?). Get fresh keys at "
            "https://app.tavily.com and update TAVILY_API_KEYS.",
        ]

    def test_r17_tavily_single_key_429_is_not_marked_dead(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch)
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response(status=429))
        with patch.object(tavily.requests, "post", post):
            first = tavily.get_web_search("q1")
            second = tavily.get_web_search("q2")
        assert len(seen) == 2                # the only key is tried again on the next call
        expected = "WEB_SEARCH_UNAVAILABLE: Tavily API returned HTTP 429." + _g1_tavily_tail
        assert first == second == expected
        assert [e["detail"] for e in quality.snapshot_quality()] == [
            "Tavily API returned HTTP 429.",
        ] * 2

    def test_r17_tavily_only_key_errors_rotate(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch, keys=[_g1_tavily_alpha, _g1_tavily_beta])
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response(status=503))
        with patch.object(tavily.requests, "post", post):
            server_error = tavily.get_web_search("q")
        assert len(seen) == 1                # a 5xx is final: no retry, no rotation
        assert server_error == "WEB_SEARCH_UNAVAILABLE: Tavily API returned HTTP 503." + _g1_tavily_tail
        assert tavily._dead_keys() == set()
        assert [e["detail"] for e in quality.snapshot_quality()] == ["Tavily API returned HTTP 503."]

        _g1_tavily_fresh(monkeypatch, keys=[_g1_tavily_alpha, _g1_tavily_beta])

        def _first_key_forbidden(headers, timeout):
            if headers["Authorization"] == f"Bearer {_g1_tavily_alpha}":
                return _g1_tavily_response(status=403)
            return _g1_tavily_response({"results": []})

        seen, post = _g1_tavily_post(_first_key_forbidden)
        with patch.object(tavily.requests, "post", post):
            rotated = tavily.get_web_search("q")
        assert [s["headers"]["Authorization"] for s in seen] == [
            f"Bearer {_g1_tavily_alpha}", f"Bearer {_g1_tavily_beta}",
        ]
        assert rotated == _g1_tavily_no_results
        assert [e["detail"] for e in quality.snapshot_quality()] == [
            "key #1/2 returned HTTP 403; removed from this run's pool and rotating to the next key",
        ]

    def test_r17_tavily_http_error_without_a_response_reports_unknown_status(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch)

        def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
            raise requests.exceptions.HTTPError("no response attached")

        with patch.object(tavily.requests, "post", _post):
            out = tavily.get_web_search("q")
        assert out == "WEB_SEARCH_UNAVAILABLE: Tavily API returned HTTP ?." + _g1_tavily_tail


class TestR17G1TavilyTransport:
    def test_r17_tavily_timeout_degrades_after_one_transient_retry(self, monkeypatch, caplog):
        _g1_tavily_fresh(monkeypatch)
        monkeypatch.setattr(netretry.time, "sleep", lambda _s: None)
        slow = requests.exceptions.ReadTimeout("slow read")
        seen, post = _g1_tavily_post(lambda headers, timeout: (_ for _ in ()).throw(slow))
        with caplog.at_level(logging.WARNING), patch.object(tavily.requests, "post", post):
            out = tavily.get_web_search("q")
        assert len(seen) == 2
        assert out == "WEB_SEARCH_UNAVAILABLE: Tavily unreachable (ReadTimeout)." + _g1_tavily_tail
        assert sorted((r.name, r.getMessage()) for r in caplog.records) == sorted([
            ("yialpha.dataflows.netretry",
             f"tavily: transient transport failure ({slow!r}); retrying (1/1)"),
            ("yialpha.dataflows.tavily", f"tavily: transport failure ({slow!r})"),
        ])

    def test_r17_tavily_interrupt_is_not_swallowed(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch)
        with patch.object(tavily.requests, "post", side_effect=KeyboardInterrupt("stop")), \
             pytest.raises(KeyboardInterrupt):
            tavily.get_web_search("q")
        assert quality.snapshot_quality() == []

    @pytest.mark.parametrize("raw, expected", [
        ("nan", 1.0), ("inf", float("inf")), ("1e400", float("inf")),
        ("-5", 1.0), ("0", 1.0), (" 2.5 ", 2.5), ("not-a-number", 20.0),
    ])
    def test_r17_tavily_timeout_env(self, monkeypatch, caplog, raw, expected):
        """Pinned and reported (not fixed): only a non-numeric value is logged
        and replaced. ``nan`` silently becomes the 1s floor; ``inf`` (also
        ``1e400``) has no ceiling and is passed to ``requests.post`` as the
        timeout, where ``socket.settimeout`` raises ``OverflowError`` — an
        exception the tool's handlers do not cover."""
        _g1_tavily_fresh(monkeypatch)
        monkeypatch.setenv(_g1_tavily_timeout_env, raw)
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response({"results": []}))
        with caplog.at_level(logging.WARNING), patch.object(tavily.requests, "post", post):
            tavily.get_web_search("q")
        assert seen[0]["timeout"] == expected
        if raw == "not-a-number":
            assert [r.getMessage() for r in caplog.records] == [
                "Ignoring non-numeric YIALPHA_TAVILY_TIMEOUT_S='not-a-number'"
            ]
        else:
            assert caplog.records == []


class TestR17G1TavilyBodies:
    @pytest.mark.parametrize("payload, expected, events", [
        pytest.param(_g1_no_json, "WEB_SEARCH_UNAVAILABLE: Tavily returned a non-JSON body."
                     + _g1_tavily_tail, 1, id="non-json"),
        pytest.param(None, _g1_tavily_no_results, 0, id="json-null"),
        pytest.param({}, _g1_tavily_no_results, 0, id="empty-object"),
        pytest.param({"results": None}, _g1_tavily_no_results, 0, id="null-results"),
    ])
    def test_r17_tavily_empty_shaped_bodies(self, monkeypatch, payload, expected, events):
        """A non-JSON body is a degradation; a JSON ``null`` body, an empty
        object and ``results: null`` all read as an honest "no results" with
        no ledger event (pinned)."""
        _g1_tavily_fresh(monkeypatch)
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response(payload))
        with patch.object(tavily.requests, "post", post):
            out = tavily.get_web_search("q")
        assert out == expected
        assert len(seen) == 1
        assert tavily.run_usage()["news"] == 1
        assert len(quality.snapshot_quality()) == events

    @pytest.mark.parametrize("payload", [
        [{"title": "x"}], "ok", 5, {"results": "abc"}, {"results": [1]},
        {"results": [{"title": "a"}, None]},
    ])
    def test_r17_tavily_wrong_shape_json_body_raises_out_of_the_tool(
        self, monkeypatch, caplog, payload,
    ):
        """Pinned and reported (not fixed): the docstring promises the tool
        "never raises into the agent loop", but only a non-JSON body degrades.
        Valid JSON of the wrong shape raises ``AttributeError`` — unlogged and
        unrecorded — after the call was already charged to the budget. Tavily
        always answers with an object, so this needs the API to break its
        contract."""
        _g1_tavily_fresh(monkeypatch)
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response(payload))
        with caplog.at_level(logging.WARNING), patch.object(tavily.requests, "post", post), \
             pytest.raises(AttributeError):
            tavily.get_web_search("q")
        assert len(seen) == 1
        assert tavily.run_usage()["news"] == 1
        assert caplog.records == []
        assert quality.snapshot_quality() == []


class TestR17G1TavilyRender:
    def test_r17_tavily_result_fields_are_coerced_and_omitted(self, monkeypatch):
        payload = {"results": [
            {"title": 123, "url": "  https://example.com/a  ", "content": "  spaced  ",
             "published_date": 20261101},
            {"content": "no title, no url, no date"},
            {"title": None, "url": None, "content": None, "published_date": None},
        ]}
        _g1_tavily_fresh(monkeypatch)
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response(payload))
        with patch.object(tavily.requests, "post", post):
            out = tavily.get_web_search("  q  ")
        assert seen[0]["json"]["query"] == "q"
        assert out == "\n".join([
            "## Web search: q",
            "1. **123**",
            "   Source: https://example.com/a",
            "   Published: 20261101",
            "   spaced",
            "2. **(untitled)**",
            "   no title, no url, no date",
            "3. **(untitled)**",
            _g1_tavily_footer,
        ])

    @pytest.mark.parametrize("content, rendered", [
        ("a" * 900, "a" * 900),
        ("b" * 899 + " tail words", "b" * 899 + " [...]"),
        # No space to break on inside the first 900 characters: the whole
        # prefix is kept rather than dropped.
        ("c" * 901, "c" * 900 + " [...]"),
    ])
    def test_r17_tavily_snippet_cap_boundary(self, monkeypatch, content, rendered):
        _g1_tavily_fresh(monkeypatch)
        listing = {"results": [{"title": "T", "content": content}]}
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response(listing))
        with patch.object(tavily.requests, "post", post):
            out = tavily.get_web_search("q")
        assert out.splitlines()[-2] == "   " + rendered

    def test_r17_tavily_request_payload_and_default_timeout(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch)
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response({"results": []}))
        with patch.object(tavily.requests, "post", post):
            tavily.get_web_search("  NVDA export news  ", max_results=50, days=7.9)
        assert seen == [{
            "url": "https://api.tavily.com/search",
            "json": {
                "query": "NVDA export news",
                "max_results": 10,          # the ceiling
                "search_depth": "basic",
                "topic": "news",            # the recency window rides the news topic
                "days": 7,
            },
            "headers": {"Authorization": f"Bearer {_g1_tavily_alpha}"},
            "timeout": 20.0,
        }]


class TestR17G1YfNewsUnnamedRetryTarget:
    """Pinned and reported (not fixed): both ``yf_retry`` calls in
    yfinance_news omit ``symbol`` / ``canonical``, so a Yahoo transport
    failure or an exhausted rate limit raises ``NoMarketDataError('?', ...)``
    — stockstats_utils.yf_retry's placeholder — and the router's sentinel, the
    text the model reads, names the target '?'. The vendor function has the
    ticker in scope and names it in its own soft-misses. The one-line fix
    (pass ``symbol=ticker, canonical=canonical``) is left to the coordinator:
    it breaks two existing pins that stub ``yf_retry`` as ``lambda fn: fn()``.
    """

    def test_r17_yfinance_news_ticker_transport_failure_sentinel_names_a_question_mark(
        self, monkeypatch,
    ):
        set_config({
            "data_vendors": {"news_data": "yfinance"},
            "tool_vendors": {"get_news": "yfinance"},
        })

        def _unreachable(count):
            raise OSError("dns boom")

        monkeypatch.setattr(
            yfinance_news.yf, "Ticker", lambda symbol: SimpleNamespace(get_news=_unreachable),
        )
        out = route_to_vendor("get_news", "NVDA", "2025-05-01", "2025-05-09")
        assert out == (
            "NO_DATA_AVAILABLE: No usable market data for '?' from any configured vendor "
            "(Yahoo unreachable (OSError)). The symbol may be invalid, delisted, not covered, "
            "or the vendor returned stale data. Do not estimate or fabricate values — report "
            "that data is unavailable for this symbol."
        )
        # A Yahoo outage is classified as clean "no data" (yf_retry's contract),
        # so it never counts toward the data-vacuum verdict.
        assert quality.snapshot_quality() == [{
            "method": "get_news", "kind": quality.KIND_NO_DATA,
            "detail": "Yahoo unreachable (OSError)", "qualifier": "",
        }]

    def test_r17_yfinance_news_rate_limit_sentinel_names_a_question_mark(self, monkeypatch):
        set_config({
            "data_vendors": {"news_data": "yfinance"},
            "tool_vendors": {"get_news": "yfinance"},
        })
        attempts = []

        def _limited(count):
            attempts.append(count)
            raise stockstats_utils.YFRateLimitError()

        monkeypatch.setattr(
            yfinance_news.yf, "Ticker", lambda symbol: SimpleNamespace(get_news=_limited),
        )
        monkeypatch.setattr(stockstats_utils.time, "sleep", lambda _s: None)
        out = route_to_vendor("get_news", "NVDA", "2025-05-01", "2025-05-09")
        assert len(attempts) == 4             # the call plus three rate-limit retries
        assert out == (
            "NO_DATA_AVAILABLE: No usable market data for '?' from any configured vendor "
            "(Yahoo Finance rate-limited after retries). The symbol may be invalid, "
            "delisted, not covered, or the vendor returned stale data. Do not estimate or "
            "fabricate values — report that data is unavailable for this symbol."
        )

    def test_r17_yfinance_news_global_transport_failure_sentinel_names_a_question_mark(
        self, monkeypatch,
    ):
        set_config({
            "data_vendors": {"news_data": "yfinance"},
            "tool_vendors": {"get_global_news": "yfinance"},
        })

        def _unreachable(*args, **kwargs):
            raise OSError("dns boom")

        monkeypatch.setattr(yfinance_news.yf, "Search", _unreachable)
        out = route_to_vendor("get_global_news", "2025-05-09")
        assert out == (
            "NO_DATA_AVAILABLE: No usable market data for '?' from any configured vendor "
            "(Yahoo unreachable (OSError)). The symbol may be invalid, delisted, not covered, "
            "or the vendor returned stale data. Do not estimate or fabricate values — report "
            "that data is unavailable for this symbol."
        )
        assert [e["method"] for e in quality.snapshot_quality()] == ["get_global_news"]


class TestR17G1SocialPitBoundary:
    """The three live-only sources carry no as-of boundary of their own. PIT
    is the caller's gate (the sentiment analyst never calls Reddit or
    StockTwits for a historical date; no analyst binds web_search for one),
    so whatever date a feed item carries is rendered verbatim."""

    def test_r17_reddit_renders_a_future_dated_post_with_its_date(self, monkeypatch):
        _g1_clear_oauth(monkeypatch)
        future = [{"title": "From the future", "score": None, "num_comments": None,
                   "created_utc": 4_102_444_800.0, "selftext": "", "source": "oauth"}]
        with patch.object(reddit, "_fetch_subreddit", return_value=future):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert out == "r/stocks — 1 recent posts mentioning NVDA:\n  [2100-01-01] From the future"

    def test_r17_stocktwits_renders_a_future_dated_message_with_its_timestamp(self):
        payload = _g1_st_payload(_g1_st_message(
            "tomorrow's news", "Bullish", created_at="2100-01-01T09:30:00Z",
        ))
        with patch.object(stocktwits, "urlopen", return_value=_g1_resp(lambda: payload)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out == (
            "Bullish: 1 (100%) · Bearish: 0 (0%) · Unlabeled: 0 · "
            "Total: 1 most-recent messages\n\n"
            "[2100-01-01T09:30:00Z · @u · Bullish] tomorrow's news"
        )

    def test_r17_tavily_recency_window_is_request_side_only(self, monkeypatch):
        _g1_tavily_fresh(monkeypatch)
        stale = {"results": [{
            "title": "Old but returned", "url": "https://example.com/old",
            "content": "c", "published_date": "2019-01-01",
        }]}
        seen, post = _g1_tavily_post(lambda headers, timeout: _g1_tavily_response(stale))
        with patch.object(tavily.requests, "post", post):
            out = tavily.get_web_search("q", days=3)
        assert (seen[0]["json"]["topic"], seen[0]["json"]["days"]) == ("news", 3)
        # Nothing is filtered client-side; the Published line is the only guard.
        assert "   Published: 2019-01-01" in out


class TestR17G1YfNewsStaleSearchCache:
    def test_r17_yfinance_news_stale_search_cache_is_served_when_yahoo_fails(
        self, monkeypatch, caplog,
    ):
        """Pinned and reported (not fixed): like StockTwits, the global-news
        Search cache passes no ``stale_cap_days``, so a fetch failure serves a
        stale search payload under the global 30-day cap. The window filter
        still runs on it: an in-window stale headline is rendered as current
        global news, with only the ledger and the log to say it is stale."""
        set_config({"data_cache_max_stale_days": 30})
        headline = [_g1_flat("YESTERDAY'S STORY", _g1_epoch("2025-05-08"), link="")]
        _g1_fake_search(monkeypatch, lambda q: headline)
        first = yfinance_news.get_global_news_yfinance("2025-05-09", limit=1)
        assert first == (
            "## Global Market News, from 2025-05-02 to 2025-05-09:\n\n"
            "### YESTERDAY'S STORY (source: P)\n\n"
        )
        entries = list((Path(get_config()["data_cache_dir"]) / "yfnews").iterdir())
        assert len(entries) == 1
        backdated = time.time() - 86_400
        os.utime(entries[0], (backdated, backdated))

        def _down(*args, **kwargs):
            raise OSError("yahoo down")

        monkeypatch.setattr(yfinance_news.yf, "Search", _down)
        with caplog.at_level(logging.WARNING):
            again = yfinance_news.get_global_news_yfinance("2025-05-09", limit=1)
        assert again == first
        assert [(r.name, r.getMessage()) for r in caplog.records] == [(
            "yialpha.dataflows.disk_cache",
            f"yfnews: fetch failed; serving STALE cache for {entries[0]} (age 1.0 days)",
        )]
        assert quality.snapshot_quality() == [{
            "method": f"yfnews/{entries[0].name}",
            "kind": quality.KIND_STALE_CACHE,
            "detail": "served stale cache (age 1.0 days) after fetch failure",
            "qualifier": "",
        }]


def _g2_ohlcv(rows: int = 80, end: str = "2026-05-13", flat: bool = False) -> pd.DataFrame:
    """Synthetic daily OHLCV (business days ending at ``end``) shaped like
    ``load_ohlcv``'s cleaned output: a datetime ``Date`` column and capitalised
    columns. Close runs 100, 101, ... unless ``flat``."""
    step = np.arange(rows, dtype=float)
    close = np.full(rows, 100.0) if flat else 100.0 + step
    return pd.DataFrame({
        "Date": pd.bdate_range(end=end, periods=rows),
        "Open": close, "High": close + 1.0, "Low": close - 1.0,
        "Close": close, "Volume": 1_000_000 + step * 100,
    })


# ---------------------------------------------------------------------------
# market_data_validator: _fmt (gaps 67, 71) and the snapshot (gaps 124-125)
# ---------------------------------------------------------------------------


class TestR17G2ValidatorFmt:
    def test_r17_market_data_validator_fmt_missing_values_render_na(self):
        """Gap 67: every missing-value form renders the explicit 'N/A' marker."""
        assert mdv._fmt(None) == "N/A"
        assert mdv._fmt(float("nan")) == "N/A"
        assert mdv._fmt(pd.NaT) == "N/A"
        assert mdv._fmt(pd.NA) == "N/A"
        # a float column holding NaN arrives as numpy float64
        assert mdv._fmt(np.float64("nan")) == "N/A"

    def test_r17_market_data_validator_fmt_bool_renders_python_text(self):
        """Gap 71: a bool renders 'True'/'False' — it must not drop through to
        the float branch and print as 1.00 / 0.00."""
        assert mdv._fmt(True) == "True"
        assert mdv._fmt(False) == "False"
        assert mdv._fmt(np.bool_(True)) == "True"
        assert mdv._fmt(np.bool_(False)) == "False"

    def test_r17_market_data_validator_fmt_numeric_shapes(self):
        """Floats are fixed at 2 decimals, ints print bare, a Timestamp prints
        its date. inf / -0.0 / sub-cent magnitudes keep Python's own text
        (pinned as-is: cosmetic, and the inputs are degenerate)."""
        assert mdv._fmt(123.4567) == "123.46"
        assert mdv._fmt(np.float64(12.5)) == "12.50"
        assert mdv._fmt(7) == "7"
        assert mdv._fmt(np.int64(1234567)) == "1234567"
        assert mdv._fmt(pd.Timestamp("2026-05-13 16:30")) == "2026-05-13"
        assert mdv._fmt(float("inf")) == "inf"
        assert mdv._fmt(-0.0) == "-0.00"
        assert mdv._fmt(1e-9) == "0.00"
        assert mdv._fmt(1e21) == "1000000000000000000000.00"


class TestR17G2ValidatorSnapshot:
    def test_r17_market_data_validator_snapshot_one_bad_indicator_is_disclosed(self, monkeypatch):
        """Gaps 124-125: an indicator whose computation raises gets its own row
        'N/A (<ExceptionType>)'; the OHLCV rows and every other indicator are
        unaffected."""
        frame = _g2_ohlcv(rows=80, end="2026-05-13")
        monkeypatch.setattr(mdv, "load_ohlcv", lambda symbol, curr_date: frame.copy())
        real_compute = mdv.compute_indicator

        def flaky(stock_df, name):
            if name == "adx":
                raise RuntimeError("stockstats refused")
            return real_compute(stock_df, name)

        monkeypatch.setattr(mdv, "compute_indicator", flaky)
        snap = mdv.build_verified_market_snapshot("COF", "2026-05-13")
        assert "| adx | N/A (RuntimeError) |" in snap
        assert "stockstats refused" not in snap  # only the type name is disclosed
        # hand checks on the untouched parts: closes are 100..179
        assert "| Close | 179.00 |" in snap
        assert "| close_50_sma | 154.50 |" in snap  # mean(130..179)
        assert "| Volume | 1007900.00 |" in snap  # float volume takes the float branch


# ---------------------------------------------------------------------------
# y_finance statements
# ---------------------------------------------------------------------------


class TestR17G2YFinanceStatements:
    def test_r17_y_finance_cashflow_annual_reads_the_annual_statement(self, monkeypatch):
        """Gap 572: any freq but 'quarterly' reads ``Ticker.cashflow`` — yfinance
        1.8's yearly frame (Timestamp period columns, newest first) — and
        caches it under the ``_annual`` key."""
        frame = pd.DataFrame(
            [[120.5, 100.25]],
            columns=pd.to_datetime(["2025-09-27", "2024-09-28"]),
            index=["Operating Cash Flow"],
        )
        constructed = []

        class _Ticker:
            cashflow = frame

            @property
            def quarterly_cashflow(self):
                raise AssertionError("annual run must not read the quarterly statement")

        def make(symbol):
            constructed.append(symbol)
            return _Ticker()

        monkeypatch.setattr(y_finance.yf, "Ticker", make)
        monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
        report = y_finance.get_cashflow("AAPL", freq="Annual")
        assert constructed == ["AAPL"]
        assert report.startswith("# Cash Flow data for AAPL (Annual)\n")
        # the raw vendor frame is cached (CSV round trip), then served as CSV
        assert report.split("\n\n", 1)[1].splitlines() == [
            ",2025-09-27,2024-09-28",
            "Operating Cash Flow,120.5,100.25",
        ]
        cached = Path(vendor_cache_dir("yfinance")) / "stmt_AAPL_cashflow_annual.csv"
        assert cached.read_text(encoding="utf-8").splitlines() == [
            ",2025-09-27,2024-09-28",
            "Operating Cash Flow,120.5,100.25",
        ]

    @pytest.mark.parametrize(("getter", "attribute", "cache_file", "message"), [
        ("get_cashflow", "quarterly_cashflow", "stmt_AAPL_cashflow_quarterly.csv",
         "cash flow retrieval failed for AAPL"),
        ("get_income_statement", "quarterly_income_stmt", "stmt_AAPL_income_statement_quarterly.csv",
         "income statement retrieval failed for AAPL"),
    ])
    def test_r17_y_finance_statement_vendor_failure_is_logged_and_reraised(
        self, monkeypatch, caplog, getter, attribute, cache_file, message,
    ):
        """Gaps 597-599 / 639-641: a vendor failure that is not a typed
        NoMarketDataError is logged with the ticker (with traceback) and
        re-raised — never returned as an 'Error retrieving' string, and nothing
        is written to the statements cache."""
        def _explode(self):
            raise RuntimeError("yfinance exploded")

        ticker_type = type("_Ticker", (), {attribute: property(_explode)})
        monkeypatch.setattr(y_finance.yf, "Ticker", lambda symbol: ticker_type())
        monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
        with (
            caplog.at_level(logging.ERROR, logger="yialpha.dataflows.y_finance"),
            pytest.raises(RuntimeError, match="yfinance exploded"),
        ):
            getattr(y_finance, getter)("AAPL", curr_date="2026-05-13")
        failed = [
            record for record in caplog.records
            if record.getMessage() == message
        ]
        assert len(failed) == 1
        assert failed[0].exc_info is not None
        assert not (Path(vendor_cache_dir("yfinance")) / cache_file).exists()


# ---------------------------------------------------------------------------
# alpha_vantage_fundamentals
# ---------------------------------------------------------------------------


class TestR17G2AvFundamentals:
    def test_r17_alpha_vantage_fundamentals_filter_json_array_body_is_returned_parsed(self, monkeypatch):
        """Gap 38: a valid-JSON body with no report lists is not statement-shaped
        — it comes back parsed and untouched (there is nothing to PIT-filter)."""
        body = '[{"fiscalDateEnding": "2999-12-31", "totalAssets": 1}]'
        monkeypatch.setattr(avf, "_make_api_request", lambda function, params: body)
        assert avf.get_balance_sheet("AAPL", curr_date="2024-01-01") == [
            {"fiscalDateEnding": "2999-12-31", "totalAssets": 1},
        ]


# ---------------------------------------------------------------------------
# fundamentals_bundle
# ---------------------------------------------------------------------------


class TestR17G2FundamentalsBundle:
    def test_r17_fundamentals_bundle_section_blank_ok_payload_renders_nothing(self):
        """Gap 216: an ok component whose payload is blank renders no section
        and — because its status is ok — no footer note either (pinned; every
        vendor behind the bundle raises instead of returning blank text)."""
        assert fb._section("Overview", {"status": fb.STATUS_OK, "payload": " \n\t "}) == []
        assert fb._section("Overview", {"status": fb.STATUS_OK}) == []
        block = fb.render_fundamentals_bundle_block({
            "symbol": "MU", "overview": {"status": fb.STATUS_OK, "payload": "  "},
        })
        assert "#### Overview" not in block
        assert "Component availability" not in block


# ---------------------------------------------------------------------------
# y_finance: the stockstats branch of _get_stock_stats_bulk (gaps 400-415) and
# the indicator window that renders it
# ---------------------------------------------------------------------------


class TestR17G2YFinanceIndicators:
    def test_r17_y_finance_bulk_stockstats_values_match_hand_computed_sma(self, monkeypatch):
        """Gaps 400-401, 406, 410-411, 415: the stockstats branch of the bulk
        calculation keys every row by its 'YYYY-MM-DD' date and renders the
        value with str() — checked against means computed from plain lists."""
        frame = _g2_ohlcv(rows=60, end="2026-05-13")  # closes 100..159
        closes = [100.0 + i for i in range(60)]
        monkeypatch.setattr(y_finance, "load_ohlcv", lambda symbol, curr_date: frame.copy())

        out = y_finance._get_stock_stats_bulk("AAPL", "close_50_sma", "2026-05-13")
        assert list(out) == [day.strftime("%Y-%m-%d") for day in frame["Date"]]
        assert out["2026-05-13"] == str(sum(closes[10:]) / 50) == "134.5"
        assert out["2026-05-12"] == str(sum(closes[9:59]) / 50) == "133.5"
        assert out["2026-02-19"] == "100.0"  # first row: a one-sample (partial) window
        assert "N/A" not in out.values()

        # A NaN from stockstats renders as "N/A": the 20-session standard
        # deviation is undefined on a one-sample window under every version.
        mstd = y_finance._get_stock_stats_bulk("AAPL", "close_20_mstd", "2026-05-13")
        assert list(mstd) == list(out)
        assert mstd["2026-02-19"] == "N/A"
        assert float(mstd["2026-02-20"]) == pytest.approx(
            statistics.stdev([100.0, 101.0]), rel=1e-12,
        )
        # The band itself is the vendor's formula (stockstats 0.6.8, the locked
        # version, uses the sample standard deviation; the release CI resolves
        # uses the population one, so a one-sample window is 100.0, not N/A).
        # Only the plumbing is pinned: one rendered value per row, none missing
        # after the warm-up row.
        bands = y_finance._get_stock_stats_bulk("AAPL", "boll_ub", "2026-05-13")
        assert list(bands) == list(out)
        assert "N/A" not in list(bands.values())[1:]

    def test_r17_y_finance_bulk_short_history_gives_a_partial_window_sma(self, monkeypatch):
        """Pinned vendor semantics (reported, not changed): stockstats expands the
        window on a short history, so 'close_200_sma' over 60 rows is the 60-row
        mean rather than N/A — a recently listed symbol gets a '200 SMA' computed
        from far fewer bars."""
        frame = _g2_ohlcv(rows=60, end="2026-05-13")
        monkeypatch.setattr(y_finance, "load_ohlcv", lambda symbol, curr_date: frame.copy())
        out = y_finance._get_stock_stats_bulk("AAPL", "close_200_sma", "2026-05-13")
        assert out["2026-05-13"] == str(sum(100.0 + i for i in range(60)) / 60) == "129.5"

    def test_r17_y_finance_indicators_window_renders_the_bulk_series(self, monkeypatch):
        """The public indicator-window tool over a real stockstats computation:
        one line per calendar day, newest first, non-trading days spelled out."""
        frame = _g2_ohlcv(rows=60, end="2026-05-13")
        monkeypatch.setattr(y_finance, "load_ohlcv", lambda symbol, curr_date: frame.copy())
        assert y_finance.get_stock_stats_indicators_window("AAPL", "close_50_sma", "2026-05-13", 5) == (
            "## close_50_sma values from 2026-05-08 to 2026-05-13:\n\n"
            "2026-05-13: 134.5\n"
            "2026-05-12: 133.5\n"
            "2026-05-11: 132.5\n"
            "2026-05-10: N/A: Not a trading day (weekend or holiday)\n"
            "2026-05-09: N/A: Not a trading day (weekend or holiday)\n"
            "2026-05-08: 131.5\n"
            "\n\n"
            + y_finance.best_ind_params["close_50_sma"]
        )

    def test_r17_y_finance_indicators_window_datetime_form_curr_date_uses_its_date_part(self, monkeypatch):
        """Fix: a dual-form curr_date ("2026-05-13 16:00:00" — the form an LLM
        echoes from a vendor report's '# Data retrieved on: <timestamp>' header)
        hit the bare ``strptime(curr_date, "%Y-%m-%d")`` before the try block;
        the tool layer caught the ValueError and handed the model
        'unconverted data remains:  16:00:00' in place of every indicator value.
        The news vendors got the [:10] normalisation in rounds 11 / 15; the two
        indicator vendors had not."""
        frame = _g2_ohlcv(rows=60, end="2026-05-13")
        monkeypatch.setattr(y_finance, "load_ohlcv", lambda symbol, curr_date: frame.copy())
        plain = y_finance.get_stock_stats_indicators_window("AAPL", "close_50_sma", "2026-05-13", 5)
        stamped = y_finance.get_stock_stats_indicators_window(
            "AAPL", "close_50_sma", "2026-05-13 16:00:00", 5,
        )
        assert stamped == plain
        assert "to 2026-05-13:\n" in stamped


class TestR17G2YFinanceStatementPit:
    def test_r17_y_finance_statement_pit_filter_applies_after_a_live_cache_fill(self, monkeypatch):
        """The cache holds the RAW vendor frame and the filing-lag filter runs per
        call: a live call that caches every quarter must not let a later
        historical call see a quarter that was not public on its date."""
        frame = pd.DataFrame(
            [[7.0, 6.0, 5.0]],
            columns=pd.to_datetime(["2025-09-27", "2025-06-28", "2025-03-29"]),
            index=["Net Income"],
        )
        constructed = []

        def make(symbol):
            constructed.append(symbol)
            return types.SimpleNamespace(quarterly_income_stmt=frame)

        monkeypatch.setattr(y_finance.yf, "Ticker", make)
        monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
        live = y_finance.get_income_statement("AAPL")
        # 2025-03-29 + the 45-day filing lag = 2025-05-13: that quarter alone is public
        replay = y_finance.get_income_statement("AAPL", curr_date="2025-05-13")
        assert len(constructed) == 1  # the replay was served from the raw cache
        assert live.split("\n\n", 1)[1].splitlines() == [
            ",2025-09-27,2025-06-28,2025-03-29", "Net Income,7.0,6.0,5.0",
        ]
        assert replay.split("\n\n", 1)[1].splitlines() == [",2025-03-29", "Net Income,5.0"]

    def test_r17_y_finance_cashflow_filing_lag_boundary_day(self, monkeypatch):
        """The 45-day lag is inclusive on its boundary day and applied exactly
        once: 2024-09-28 + 45 days = 2024-11-12 is public ON 2024-11-12 and not
        the day before (with every column gone the vendor raises)."""
        frame = pd.DataFrame(
            [[100.25]], columns=pd.to_datetime(["2024-09-28"]), index=["Operating Cash Flow"],
        )
        monkeypatch.setattr(y_finance.yf, "Ticker", lambda symbol: types.SimpleNamespace(cashflow=frame))
        monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
        public = y_finance.get_cashflow("AAPL", freq="annual", curr_date="2024-11-12")
        assert public.split("\n\n", 1)[1].splitlines() == [
            ",2024-09-28", "Operating Cash Flow,100.25",
        ]
        with pytest.raises(NoMarketDataError, match="no cash flow data"):
            y_finance.get_cashflow("AAPL", freq="annual", curr_date="2024-11-11")

    def test_r17_y_finance_statement_nan_and_bool_cells_survive_the_cache_round_trip(self, monkeypatch):
        """A boolean cell stays 'True' and a missing cell stays blank through the
        CSV cache round trip and the PIT filter — neither turns into 1.0 / 0.0 /
        'nan' in the statement the analyst reads."""
        frame = pd.DataFrame(
            [[True], [np.nan], [2.5]],
            columns=pd.to_datetime(["2024-09-28"]),
            index=["Flag", "Blank", "Total"],
        )
        monkeypatch.setattr(y_finance.yf, "Ticker", lambda symbol: types.SimpleNamespace(cashflow=frame))
        monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
        report = y_finance.get_cashflow("AAPL", freq="annual", curr_date="2025-01-01")
        assert report.split("\n\n", 1)[1].splitlines() == [
            ",2024-09-28", "Flag,True", "Blank,", "Total,2.5",
        ]

    def test_r17_y_finance_statement_stale_cache_is_ledgered_and_still_pit_filtered(self, monkeypatch, caplog):
        """A fetch failure with a 2-day-old statements cache: the stale raw frame
        is served, the serve lands in the quality ledger as a stale_cache event,
        and the filing-lag filter still runs on it (a stale frame can only MISS
        a newer filing, never leak one)."""
        raw = pd.DataFrame(
            [[7.0, 5.0]],
            columns=pd.to_datetime(["2025-09-27", "2025-03-29"]), index=["Net Income"],
        ).to_csv().encode("utf-8")
        cache_file = Path(vendor_cache_dir("yfinance")) / "stmt_AAPL_income_statement_quarterly.csv"
        cache_file.write_bytes(raw)
        two_days_ago = time.time() - 2 * 86400
        os.utime(cache_file, (two_days_ago, two_days_ago))

        def _down(self):
            raise RuntimeError("yahoo down")

        ticker_type = type("_Ticker", (), {"quarterly_income_stmt": property(_down)})
        monkeypatch.setattr(y_finance.yf, "Ticker", lambda symbol: ticker_type())
        monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.disk_cache"):
            replay = y_finance.get_income_statement("AAPL", curr_date="2025-05-13")
        assert "serving STALE cache" in caplog.text
        assert replay.split("\n\n", 1)[1].splitlines() == [",2025-03-29", "Net Income,5.0"]
        events = quality.snapshot_quality()
        assert [(event["method"], event["kind"]) for event in events] == [
            ("yfinance/stmt_AAPL_income_statement_quarterly.csv", quality.KIND_STALE_CACHE),
        ]
        assert "age 2.0 days" in events[0]["detail"]


class TestR17G2YFinanceInfoCache:
    def test_r17_y_finance_cached_ticker_info_non_object_cache_raises_typed_error(self):
        """Gap 121: a fresh cache whose payload is valid JSON but not an object
        (list / null / number / string) raises the typed NoMarketDataError — it
        is never handed on as fundamentals."""
        cache_file = Path(vendor_cache_dir("yfinance")) / "stmt_AAPL_fundamentals.json"
        for raw in ('["longName"]', "null", "42", '"Apple Inc."'):
            cache_file.write_text(raw, encoding="utf-8")
            with pytest.raises(NoMarketDataError, match="cached fundamentals payload for AAPL is not an object$"):
                y_finance.get_fundamentals("AAPL")

    def test_r17_y_finance_info_cache_within_ttl_is_served_without_age_disclosure(self, monkeypatch):
        """Pin (by design: _STATEMENT_CACHE_TTL_DAYS = 1 day). A .info cache up to
        24h old is served without a vendor call, and the report header carries the
        SERVE time — a live 'overview' can be a day old and reads as current."""
        cache_file = Path(vendor_cache_dir("yfinance")) / "stmt_AAPL_fundamentals.json"
        cache_file.write_text(json.dumps({"longName": "Apple Inc.", "marketCap": 5}), encoding="utf-8")
        fetched_23h_ago = time.time() - 23 * 3600
        os.utime(cache_file, (fetched_23h_ago, fetched_23h_ago))

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 11, 25, 9, 30, 0, tzinfo=tz)

        def _no_vendor(symbol):
            raise AssertionError("a cache inside the TTL is served without a vendor call")

        monkeypatch.setattr(y_finance, "datetime", _Frozen)
        monkeypatch.setattr(y_finance.yf, "Ticker", _no_vendor)
        assert y_finance.get_fundamentals("AAPL") == (
            "# Company Fundamentals for AAPL\n"
            "# Data retrieved on: 2026-11-25 09:30:00\n\n"
            "Name: Apple Inc.\nMarket Cap: 5"
        )


def _g2_av_200(body: str):
    """A ``requests`` response stand-in: HTTP 200 carrying ``body`` as text."""
    return types.SimpleNamespace(text=body, raise_for_status=lambda: None)


# ---------------------------------------------------------------------------
# alpha_vantage_indicator (gaps 147, 180, 194, 206-207) + cache key + throttle
# ---------------------------------------------------------------------------


class TestR17G2AvIndicator:
    def test_r17_alpha_vantage_indicator_datetime_form_curr_date_uses_its_date_part(self, monkeypatch):
        """Fix: a dual-form curr_date ("2025-03-01 16:00:00" — the form an LLM
        echoes from a vendor report's '# Data retrieved on: <timestamp>' header)
        hit the bare ``strptime(curr_date, "%Y-%m-%d")`` before the try block;
        the tool layer caught the ValueError and handed the model
        'unconverted data remains:  16:00:00' in place of every indicator value.
        The news vendors got the [:10] normalisation in rounds 11 / 15; the two
        indicator vendors had not."""
        csv = "time,RSI\n2025-02-28,61.0\n2025-03-01,62.0\n2025-03-02,63.0\n"
        monkeypatch.setattr(avi, "_make_api_request", lambda function, params: csv)
        plain = avi.get_indicator("AAPL", "rsi", "2025-03-01", 3)
        assert avi.get_indicator("AAPL", "rsi", "2025-03-01 16:00:00", 3) == plain
        assert plain == (
            "## RSI values from 2025-02-26 to 2025-03-01:\n\n"
            "2025-02-28: 61.0\n2025-03-01: 62.0\n\n\n"
            + avi.indicator_descriptions["rsi"]
        )

    def test_r17_alpha_vantage_indicator_gated_but_unimplemented_name_raises_typed_no_data(self, monkeypatch):
        """Gap 147: a name that passes the gate but has no endpoint branch is a
        typed NoMarketDataError (the router moves on to the next vendor), sent
        without any request."""
        def _unexpected(function, params):
            raise AssertionError("no request may be sent for an unimplemented indicator")

        monkeypatch.setattr(avi, "_make_api_request", _unexpected)
        monkeypatch.setattr(
            avi, "supported_indicators", {**avi.supported_indicators, "zzz": ("Z", "close")},
        )
        with pytest.raises(NoMarketDataError, match="indicator zzz not implemented yet"):
            avi.get_indicator("AAPL", "zzz", "2025-03-01", 30)

    def test_r17_alpha_vantage_indicator_without_a_column_mapping_reads_the_second_column(self, monkeypatch):
        """Gap 180: no catalog column for the indicator → the value comes from
        the second CSV column."""
        monkeypatch.setattr(
            avi, "_make_api_request", lambda function, params: "time,Whatever\n2025-02-20,12.5\n",
        )
        monkeypatch.setattr(avi, "col_name_map", {})
        out = avi.get_indicator("AAPL", "close_50_sma", "2025-03-01", 30)
        assert out.startswith(
            "## CLOSE_50_SMA values from 2025-01-30 to 2025-03-01:\n\n2025-02-20: 12.5\n",
        )

    def test_r17_alpha_vantage_indicator_skips_blank_short_and_intraday_rows(self, monkeypatch):
        """Gaps 194, 206-207: a blank line, a row without a value column and a
        row whose time is not a plain date are skipped; a CR line ending, a
        non-padded date and a surplus column are tolerated."""
        csv = (
            "time,SMA\n"
            "2025-02-10 16:00:00,101.0\n"  # intraday timestamp: not a daily bar
            "\n"                            # blank line
            "2025-2-11,102.0\r\n"           # CR line ending + non-padded month/day
            "2025-02-14\n"                  # no value column
            "2025-02-15,106.0,extra\n"      # surplus column ignored
        )
        monkeypatch.setattr(avi, "_make_api_request", lambda function, params: csv)
        assert avi.get_indicator("AAPL", "close_50_sma", "2025-02-15", 10) == (
            "## CLOSE_50_SMA values from 2025-02-05 to 2025-02-15:\n\n"
            "2025-02-11: 102.0\n2025-02-15: 106.0\n\n\n"
            + avi.indicator_descriptions["close_50_sma"]
        )

    def test_r17_alpha_vantage_indicator_intraday_bars_never_match_the_daily_window(self, monkeypatch):
        """Pinned (reported, not fixed): the row parser only accepts plain
        'YYYY-MM-DD' dates, so a non-daily interval — whose rows carry a time —
        yields the no-data prose although the vendor answered. Unreachable from
        the tool layer today (``route_to_vendor`` never passes ``interval``)."""
        captured = {}

        def fake(function, params):
            captured.update(params)
            return "time,SMA\n2025-02-10 16:00:00,101.0\n2025-02-11 16:00:00,102.0\n"

        monkeypatch.setattr(avi, "_make_api_request", fake)
        out = avi.get_indicator("AAPL", "close_50_sma", "2025-02-15", 10, interval="60min")
        assert captured["interval"] == "60min"  # forwarded to the request, so it is in the key
        assert out == (
            "## CLOSE_50_SMA values from 2025-02-05 to 2025-02-15:\n\n"
            "No data available for the specified date range.\n\n\n"
            + avi.indicator_descriptions["close_50_sma"]
        )

    def test_r17_alpha_vantage_indicator_cache_key_covers_period_and_interval(self, monkeypatch):
        """Every parameter that changes the result is part of the 15-minute cache
        key: an exact repeat is served from disk, while another ``time_period``
        or ``interval`` is a new request and never sees the cached neighbour."""
        raw_calls = []

        def fake_raw(function_name, params):
            raw_calls.append((function_name, dict(params)))
            return f"time,SMA\n2025-02-20,{params['time_period']}.{len(params['interval'])}\n"

        monkeypatch.setattr(avc, "_raw_api_request", fake_raw)
        fifty = avi.get_indicator("AAPL", "close_50_sma", "2025-03-01", 30)
        again = avi.get_indicator("AAPL", "close_50_sma", "2025-03-01", 30)
        two_hundred = avi.get_indicator("AAPL", "close_200_sma", "2025-03-01", 30)
        weekly = avi.get_indicator("AAPL", "close_50_sma", "2025-03-01", 30, interval="weekly")
        assert len(raw_calls) == 3
        assert [params["time_period"] for _, params in raw_calls] == ["50", "200", "50"]
        assert [params["interval"] for _, params in raw_calls] == ["daily", "daily", "weekly"]
        assert "2025-02-20: 50.5" in fifty
        assert again == fifty
        assert "2025-02-20: 200.5" in two_hundred and "50.5" not in two_hundred
        assert "2025-02-20: 50.6" in weekly and "50.5" not in weekly

    def test_r17_alpha_vantage_indicator_rate_limit_note_propagates_typed(self, monkeypatch):
        """An in-body rate-limit note leaves the vendor as the typed
        VendorRateLimitError (the router's 'skip to the next vendor' signal)
        without any sleep."""
        slept = []
        monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))
        monkeypatch.setattr(
            avc.requests, "get",
            lambda url, params=None, timeout=None: _g2_av_200(json.dumps({
                "Note": "Our standard API call frequency is 5 calls per minute.",
            })),
        )
        with pytest.raises(VendorRateLimitError, match="rate limit exceeded"):
            avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)
        assert slept == []

    @pytest.mark.parametrize("body", [
        json.dumps({"Information": "Thank you for using Alpha Vantage!"}),
        "",
        " \n ",
        "time,RSI\n",
    ])
    def test_r17_alpha_vantage_indicator_notice_or_empty_body_raises_not_an_empty_frame(
        self, monkeypatch, body,
    ):
        """A notice body or an empty payload is a typed no-data error — the
        vendor never returns an empty frame that reads as 'no values exist'."""
        monkeypatch.setattr(
            avc.requests, "get", lambda url, params=None, timeout=None: _g2_av_200(body),
        )
        with pytest.raises(NoMarketDataError, match="no data rows returned"):
            avi.get_indicator("AAPL", "rsi", "2025-03-01", 30)


# ---------------------------------------------------------------------------
# alpha_vantage_fundamentals: PIT matrix of _filter_reports_by_date
# ---------------------------------------------------------------------------


class TestR17G2AvFundamentalsPit:
    @pytest.mark.parametrize("getter", [
        "get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement",
    ])
    def test_r17_alpha_vantage_fundamentals_blank_body_raises_typed_no_data(self, monkeypatch, getter):
        """Fix: an empty 200 body used to travel on as a SUCCESSFUL payload —
        ``_make_api_request`` returns whatever text arrived and every getter
        passed it through — so the router recorded a success for the statement
        and the tool handed the model an empty string: a fundamentals-blind read
        that nothing flagged. (Live mode: the overview's PIT gate refuses a past
        date before any request is made.)"""
        monkeypatch.setattr(
            avc.requests, "get", lambda url, params=None, timeout=None: _g2_av_200(""),
        )
        with pytest.raises(NoMarketDataError, match="Alpha Vantage returned an empty"):
            getattr(avf, getter)("AAPL")

    @pytest.mark.parametrize(("fiscal_date", "curr_date", "public"), [
        ("2024-09-30", "2024-11-14", True),   # period end + 45 days: public ON that day
        ("2024-09-30", "2024-11-13", False),
        ("2024-9-30", "2024-11-14", True),    # non-padded parses; never compared lexically
        ("2024-9-30", "2024-11-13", False),
        ("2024-09-30T21:00:00Z", "2024-11-14", True),  # timestamp form → its date part
        ("2024/09/30", "2026-01-01", False),  # unparseable forms are dropped, however old
        ("20240930", "2026-01-01", False),
        ("", "2026-01-01", False),
        (None, "2026-01-01", False),
        ("__missing__", "2026-01-01", False),
        ("2024-09-30", "2026/11/14", False),  # unparseable curr_date: everything is dropped
    ])
    def test_r17_alpha_vantage_fundamentals_filing_gate_matrix(self, monkeypatch, fiscal_date, curr_date, public):
        """The Alpha Vantage PIT gate, report by report: a period is visible
        from ``period end + 45 days`` inclusive, the comparison is on parsed
        dates (a non-padded month must not be ordered as text), and any date the
        gate cannot prove public is dropped — a missing, empty, null or
        unparseable ``fiscalDateEnding``, or an unparseable ``curr_date``."""
        report = {} if fiscal_date == "__missing__" else {
            "fiscalDateEnding": fiscal_date, "totalAssets": 1,
        }
        body = json.dumps({"annualReports": [report], "quarterlyReports": [report]})
        monkeypatch.setattr(avf, "_make_api_request", lambda function, params: body)
        out = avf.get_balance_sheet("AAPL", curr_date=curr_date)
        assert out == {
            "annualReports": [report] if public else [],
            "quarterlyReports": [report] if public else [],
        }

    def test_r17_alpha_vantage_fundamentals_live_mode_keeps_every_report_unfiltered(self, monkeypatch):
        """Live mode (no as-of date) returns the response text untouched and
        unparsed — malformed fiscal dates included; there is no date to leak
        against. (A dated call is what parses and filters it into a dict.)"""
        body = json.dumps({
            "symbol": "AAPL",
            "annualReports": [{"fiscalDateEnding": "2024/09/30"}, {"fiscalDateEnding": None}, {}],
        })
        monkeypatch.setattr(avf, "_make_api_request", lambda function, params: body)
        assert avf.get_cashflow("AAPL") == body
        assert isinstance(avf.get_cashflow("AAPL"), str)

    def test_r17_alpha_vantage_fundamentals_no_public_report_comes_back_as_empty_lists(self, monkeypatch):
        """Pin (reported): with every report still inside its filing window the
        vendor returns well-formed EMPTY lists as a success — unlike the
        yfinance statements, which raise once the filter leaves no column — so
        the router records a success and the model sees 'no reports'."""
        body = json.dumps({
            "symbol": "AAPL",
            "annualReports": [{"fiscalDateEnding": "2024-09-30"}],
            "quarterlyReports": [{"fiscalDateEnding": "2024-06-30"}],
        })
        monkeypatch.setattr(avf, "_make_api_request", lambda function, params: body)
        out = avf.get_income_statement("AAPL", curr_date="2024-07-01")
        assert out == {"symbol": "AAPL", "annualReports": [], "quarterlyReports": []}


def _g2_live_today() -> str:
    """The host-local calendar date: never a historical date for ``is_historical_date``."""
    return df_utils.live_anchor_dates()[0].isoformat()


# ---------------------------------------------------------------------------
# market_data_validator: snapshot robustness
# ---------------------------------------------------------------------------


class TestR17G2ValidatorSnapshotRobustness:
    def test_r17_market_data_validator_snapshot_flat_series_renders_finite_values(self, monkeypatch):
        """A constant series — every division-by-zero trap in the indicator set —
        still renders finite, exact values; no cell reads 'inf' / 'nan'."""
        monkeypatch.setattr(
            mdv, "load_ohlcv", lambda symbol, curr_date: _g2_ohlcv(rows=80, flat=True),
        )
        snap = mdv.build_verified_market_snapshot("COF", "2026-05-13")
        for row in (
            "| rsi | 50.00 |", "| macd | 0.00 |", "| macdh | 0.00 |",
            "| boll_ub | 100.00 |", "| boll_lb | 100.00 |", "| adx | 0.00 |",
            "| kdjk | 50.00 |", "| atr | 2.00 |",
        ):
            assert row in snap
        assert "| inf |" not in snap and "| nan |" not in snap and "| N/A |" not in snap

    def test_r17_market_data_validator_snapshot_derived_indicator_renders_its_last_row(self, monkeypatch):
        """A derived (registry) indicator renders its LAST row through _fmt: a
        warm-up NaN is 'N/A', a number is fixed at 2 decimals, and stockstats is
        never consulted for it."""
        monkeypatch.setattr(
            mdv, "load_ohlcv", lambda symbol, curr_date: _g2_ohlcv(rows=80, end="2026-05-13"),
        )

        def fake_derived(data, name):
            assert len(data) == 80
            if name == "rvol_20":
                return pd.Series([np.nan] * 79 + [0.123456])
            return pd.Series([0.5] * 79 + [np.nan])

        def _unexpected(stock_df, name):
            raise AssertionError("a derived indicator must not go through stockstats")

        monkeypatch.setattr(mdv, "compute_derived", fake_derived)
        monkeypatch.setattr(mdv, "compute_indicator", _unexpected)
        snap = mdv.build_verified_market_snapshot("COF", "2026-05-13", indicators=("rvol_20", "ewma_vol"))
        assert "| rvol_20 | 0.12 |" in snap
        assert "| ewma_vol | N/A |" in snap

    def test_r17_market_data_validator_snapshot_future_rows_cannot_move_an_indicator(self, monkeypatch):
        """Look-ahead check: the snapshot built from a frame that also carries
        rows AFTER the analysis date is byte-identical to the one built from the
        frame cut at that date — no OHLCV cell, moving average or indicator can
        see a future bar."""
        base = _g2_ohlcv(rows=80, end="2026-05-13")
        future = pd.DataFrame({
            "Date": pd.bdate_range("2026-05-14", periods=5),
            "Open": [9_999.0] * 5, "High": [12_345.0] * 5, "Low": [1.0] * 5,
            "Close": [9_999.0] * 5, "Volume": [1e12] * 5,
        })
        with_future = pd.concat([base, future], ignore_index=True)
        monkeypatch.setattr(mdv, "load_ohlcv", lambda symbol, curr_date: base.copy())
        expected = mdv.build_verified_market_snapshot("COF", "2026-05-13")
        monkeypatch.setattr(mdv, "load_ohlcv", lambda symbol, curr_date: with_future.copy())
        assert mdv.build_verified_market_snapshot("COF", "2026-05-13") == expected

    def test_r17_market_data_validator_snapshot_dual_form_date_and_window_floor(self, monkeypatch):
        """A curr_date carrying a time is accepted (its rows all sit on that
        date), and a look_back_days below 1 still shows the latest row."""
        monkeypatch.setattr(
            mdv, "load_ohlcv", lambda symbol, curr_date: _g2_ohlcv(rows=80, end="2026-05-13"),
        )
        snap = mdv.build_verified_market_snapshot("COF", "2026-05-13 16:00:00", look_back_days=0)
        assert "- Requested analysis date: 2026-05-13 16:00:00" in snap
        assert "- Latest trading row used: 2026-05-13" in snap
        assert "### Recent verified closes (last 1 rows)" in snap
        assert "| 2026-05-13 | 179.00 |" in snap
        assert "| 2026-05-12 |" not in snap


# ---------------------------------------------------------------------------
# fundamentals_overview: the Yahoo half's two typed refusals
# ---------------------------------------------------------------------------


class TestR17G2FundamentalsOverview:
    @pytest.mark.parametrize(("info", "reason"), [
        ({}, "no Yahoo info returned"),
        ({"trailingPegRatio": None}, "no Yahoo overview fields returned"),
        ({"someUnknownField": 1}, "no Yahoo overview fields returned"),
    ])
    def test_r17_fundamentals_overview_stub_yahoo_info_degrades_to_sec_only(
        self, monkeypatch, caplog, info, reason,
    ):
        """Gaps 49, 52: an empty ``.info``, or a stub with no overview field,
        raises the typed NoMarketDataError in ``_yahoo_valuation_lines``; the
        aggregate overview then serves the SEC facts alone, names the reason in
        its Yahoo line and logs a warning — no fabricated valuation."""
        monkeypatch.setattr(fo, "_sec_fundamentals", lambda ticker, curr_date: "# hdr\nName: Micron\n")
        monkeypatch.setattr(fo, "_cached_ticker_info", lambda ticker, canonical: dict(info))
        with pytest.raises(NoMarketDataError, match=reason):
            fo._yahoo_valuation_lines("MU", "MU")

        with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.fundamentals_overview"):
            out = fo.get_fundamentals("MU")
        assert "## SEC filing facts (as-reported, point-in-time)\nName: Micron" in out
        assert out.endswith(
            "## Real-time valuation: real-time valuation unavailable "
            f"(NoMarketDataError: No market data for 'MU': {reason}).",
        )
        assert "Real-time valuation & snapshot (Yahoo Finance, live)" not in out
        assert "aggregate overview: Yahoo half unavailable for MU" in caplog.text


# ---------------------------------------------------------------------------
# fundamentals_bundle: per-leg fail-soft and its disclosure
# ---------------------------------------------------------------------------


class TestR17G2FundamentalsBundleLegs:
    def test_r17_fundamentals_bundle_etf_leg_sentinel_payload_is_unavailable(self, monkeypatch):
        """Gaps 101-102: a fund snapshot that comes back as a sentinel string is
        'unavailable' with the sentinel's first line as the reason (300-char
        cap) — the sentinel text is never rendered as a fund snapshot."""
        long_line = "NO_DATA_AVAILABLE: " + "x" * 400
        monkeypatch.setattr(efd, "get_etf_fund_data", lambda ticker, curr_date: f"{long_line}\nmore")
        leg = fb._fetch_etf_fund_data("SPY", _g2_live_today())
        assert leg == {"status": fb.STATUS_UNAVAILABLE, "reason": long_line[:300]}
        assert leg["reason"] == "NO_DATA_AVAILABLE: " + "x" * 281  # capped at 300 chars

    def test_r17_fundamentals_bundle_etf_leg_served_snapshot_rides_as_payload(self, monkeypatch):
        """Gap 106: a served fund snapshot rides unchanged under 'payload'."""
        snapshot = "# ETF Fund Data for SPY\nNAV (per share): 555.0"
        monkeypatch.setattr(efd, "get_etf_fund_data", lambda ticker, curr_date: snapshot)
        assert fb._fetch_etf_fund_data("SPY", _g2_live_today()) == {
            "status": fb.STATUS_OK, "payload": snapshot,
        }

    @pytest.mark.parametrize(("error", "reason"), [
        (TimeoutError(), "TimeoutError: "),
        (RuntimeError("boom"), "RuntimeError: boom"),
    ])
    def test_r17_fundamentals_bundle_component_future_failure_is_isolated_and_disclosed(
        self, monkeypatch, error, reason,
    ):
        """Gaps 202-203: when one component's future raises (``result(timeout=90)``
        timing out is the production case) that component is 'unavailable' with
        the exception's type as its reason, the other legs are unaffected, and
        the rendered block says so in its footer."""
        submitted = []

        class _Future:
            def __init__(self, fn, failing):
                self._fn = fn
                self._failing = failing

            def result(self, timeout=None):
                assert timeout == 90
                if self._failing:
                    raise error
                return self._fn()

        def fake_submit(pool, fn):
            submitted.append(fn)
            return _Future(fn, failing=len(submitted) == 3)  # the third job: balance_sheet

        monkeypatch.setattr(fb, "submit_with_context", fake_submit)
        monkeypatch.setattr(fb, "route_to_vendor", lambda method, *args: f"row for {method}")
        bundle = fb.fetch_fundamentals_bundle("stock", "AAPL", "2026-05-13")
        assert bundle["balance_sheet"] == {"status": fb.STATUS_UNAVAILABLE, "reason": reason}
        assert [bundle[key]["status"] for key in ("overview", "income_statement", "cashflow")] == ["ok"] * 3
        block = fb.render_fundamentals_bundle_block(bundle)
        assert "#### Balance sheet (quarterly)" not in block
        assert f"- **Component availability**: balance_sheet: unavailable ({reason})" in block
        assert "#### Income statement (quarterly)\n\nrow for get_income_statement" in block

    def test_r17_fundamentals_bundle_every_failing_leg_is_named_in_the_footer(self, monkeypatch):
        """Per-leg disclosure: when the router raises for all four core
        components the block carries no section and names every leg with its
        reason — a fundamentals-blind run never reads as a fed one."""
        def _down(method, *args):
            raise RuntimeError(f"{method} unreachable")

        monkeypatch.setattr(fb, "route_to_vendor", _down)
        bundle = fb.fetch_fundamentals_bundle("stock", "AAPL", "2026-05-13")
        block = fb.render_fundamentals_bundle_block(bundle)
        assert "####" not in block
        assert (
            "- **Component availability**: overview: unavailable "
            "(RuntimeError: get_fundamentals unreachable); income_statement: "
            "unavailable (RuntimeError: get_income_statement unreachable); "
            "balance_sheet: unavailable (RuntimeError: get_balance_sheet "
            "unreachable); cashflow: unavailable (RuntimeError: get_cashflow "
            "unreachable)"
        ) in block

    def test_r17_fundamentals_bundle_etf_snapshot_failure_and_skip_are_disclosed(self, monkeypatch):
        """ETF path: a failed fund snapshot — the ETF run's core evidence —
        leaves no section but a footer line naming the leg and its reason, and
        lands in the quality ledger; on a historical date the PIT skip is named
        in the footer too."""
        monkeypatch.setattr(fb, "route_to_vendor", lambda method, *args: "overview row")

        def _down(ticker, curr_date):
            raise RuntimeError("fund page down")

        monkeypatch.setattr(efd, "get_etf_fund_data", _down)
        quality.ensure_run_context()  # bind the run ledger the worker contexts inherit
        live = fb.fetch_fundamentals_bundle("stock", "SPY", _g2_live_today())
        assert live["etf_fund_data"] == {
            "status": fb.STATUS_UNAVAILABLE, "reason": "RuntimeError: fund page down",
        }
        assert live["overview"] == {"status": fb.STATUS_OK, "payload": "overview row"}
        assert "income_statement" not in live  # ETF REPLACE routing: no statement legs
        block = fb.render_fundamentals_bundle_block(live)
        assert "ETF fund data (NAV/AUM/expense/holdings)" not in block
        assert (
            "- **Component availability**: etf_fund_data: unavailable (RuntimeError: fund page down)"
        ) in block
        events = quality.snapshot_quality()
        assert [(event["method"], event["kind"]) for event in events] == [
            ("get_etf_fund_data", quality.KIND_OPTIONAL_UNAVAILABLE),
        ]
        assert "SPY: fund snapshot unavailable: RuntimeError: fund page down" in events[0]["detail"]

        replay = fb.fetch_fundamentals_bundle("stock", "SPY", "2026-01-15")
        skip = "fund snapshot is a today-only page (no as-of boundary)"
        assert replay["etf_fund_data"] == {"status": fb.STATUS_SKIPPED_LIVE_ONLY, "reason": skip}
        assert f"etf_fund_data: skipped_live_only ({skip})" in fb.render_fundamentals_bundle_block(replay)


# ---------------------------------------------------------------------------
# y_finance: the today-only overview gate vs the cache, and a transport failure
# ---------------------------------------------------------------------------


class TestR17G2YFinanceOverviewGate:
    def test_r17_y_finance_overview_refuses_a_past_date_before_touching_the_cache(self, monkeypatch):
        """By design (pinned, not changed): the ``.info`` snapshot is today-only.
        On a past date the vendor refuses before it reads the statements cache
        or builds a Ticker — a snapshot cached today is never served into a
        historical run, however fresh — and live mode still serves that cache."""
        cache_file = Path(vendor_cache_dir("yfinance")) / "stmt_AAPL_fundamentals.json"
        cache_file.write_text(json.dumps({"longName": "Apple Inc.", "marketCap": 5}), encoding="utf-8")

        def _no_vendor(symbol):
            raise AssertionError("a historical run must not reach the vendor")

        monkeypatch.setattr(y_finance.yf, "Ticker", _no_vendor)
        with pytest.raises(NoMarketDataError, match=r"\(today only\); not valid as of 2026-01-15$"):
            y_finance.get_fundamentals("AAPL", curr_date="2026-01-15")
        with pytest.raises(NoMarketDataError, match="not valid as of 2026-01-15 16:00:00$"):
            y_finance.get_fundamentals("AAPL", curr_date="2026-01-15 16:00:00")
        assert "Market Cap: 5" in y_finance.get_fundamentals("AAPL")

    def test_r17_y_finance_statement_transport_failure_is_a_typed_no_data(self, monkeypatch):
        """What the caller sees when Yahoo's transport fails: the REAL yf_retry
        turns the OSError into a typed NoMarketDataError the router degrades to
        its sentinel, and nothing is cached. Pinned (reported): that error carries
        yf_retry's '?' placeholder for the instrument, because
        ``_cached_yf_frame`` calls it without symbol / canonical — so the
        model-facing sentinel blames a symbol called '?'."""
        def _reset(self):
            raise ConnectionResetError("reset by peer")

        ticker_type = type("_Ticker", (), {"quarterly_cashflow": property(_reset)})
        monkeypatch.setattr(y_finance.yf, "Ticker", lambda symbol: ticker_type())
        with pytest.raises(NoMarketDataError) as caught:
            y_finance.get_cashflow("AAPL")
        assert caught.value.detail == "Yahoo unreachable (ConnectionResetError)"
        assert (caught.value.symbol, caught.value.canonical) == ("?", "?")
        assert not (Path(vendor_cache_dir("yfinance")) / "stmt_AAPL_cashflow_quarterly.csv").exists()

        set_config({"data_vendors": {"fundamental_data": "yfinance"}})
        sentinel = route_to_vendor("get_cashflow", "AAPL", "quarterly")
        assert sentinel.startswith(
            "NO_DATA_AVAILABLE: No usable market data for '?' from any configured vendor "
            "(Yahoo unreachable (ConnectionResetError)).",
        )


# ---------------------------------------------------------------------------
# alpha_vantage_indicator: PIT filter on a cached raw response
# ---------------------------------------------------------------------------


class TestR17G2AvIndicatorCachePit:
    def test_r17_alpha_vantage_indicator_cached_response_is_filtered_per_call(self, monkeypatch):
        """The 15-minute cache holds the vendor's RAW csv (rows after the
        analysis date included); the date window is applied per call, so an
        earlier analysis date served from that cache cannot see a later row."""
        raw_calls = []

        def fake_raw(function_name, params):
            raw_calls.append(function_name)
            return "time,SMA\n2025-02-28,61.0\n2025-02-20,60.0\n2025-02-10,59.0\n"

        monkeypatch.setattr(avc, "_raw_api_request", fake_raw)
        late = avi.get_indicator("AAPL", "close_50_sma", "2025-03-01", 30)
        early = avi.get_indicator("AAPL", "close_50_sma", "2025-02-15", 10)
        assert raw_calls == ["SMA"]  # the second call was served from the cache
        assert "2025-02-28: 61.0" in late and "2025-02-10: 59.0" in late
        assert early == (
            "## CLOSE_50_SMA values from 2025-02-05 to 2025-02-15:\n\n"
            "2025-02-10: 59.0\n\n\n"
            + avi.indicator_descriptions["close_50_sma"]
        )


# ---------------------------------------------------------------------------
# PIT: a vendor call without a date while an analysis date is pinned
# ---------------------------------------------------------------------------


def _g2_pin_vendor_seams(monkeypatch) -> None:
    """Vendor seams for the pinned-run tests: three quarters (only 2025-03-29 is
    public on 2025-05-13), fresh ``.info`` caches, one not-yet-public AV report
    and a stub SEC half."""
    frame = pd.DataFrame(
        [[7.0, 6.0, 5.0]],
        columns=pd.to_datetime(["2025-09-27", "2025-06-28", "2025-03-29"]),
        index=["Net Income"],
    )
    monkeypatch.setattr(
        y_finance.yf, "Ticker", lambda symbol: types.SimpleNamespace(quarterly_income_stmt=frame),
    )
    monkeypatch.setattr(y_finance, "yf_retry", lambda fn, **kw: fn())
    for symbol in ("AAPL", "MU"):  # fresh .info caches: no vendor call in live mode
        (Path(vendor_cache_dir("yfinance")) / f"stmt_{symbol}_fundamentals.json").write_text(
            json.dumps({"longName": symbol, "marketCap": 5}), encoding="utf-8",
        )
    reports = json.dumps({"quarterlyReports": [
        {"fiscalDateEnding": "2025-06-30", "totalAssets": 2},
    ]})
    monkeypatch.setattr(avf, "_make_api_request", lambda function, params: reports)
    monkeypatch.setattr(fo, "_sec_fundamentals", lambda ticker, curr_date: "# hdr\nName: Micron\n")


_G2_PINNED = "2025-05-13"


class TestR17G2VendorWithoutDateUnderPinnedRun:
    @pytest.mark.parametrize(("vendor_call", "post_decision_marker"), [
        (lambda: y_finance.get_income_statement("AAPL"), "2025-09-27,2025-06-28,2025-03-29"),
        (lambda: y_finance.get_fundamentals("AAPL"), "Market Cap: 5"),
        (lambda: avf.get_balance_sheet("AAPL"), "2025-06-30"),
        (lambda: fo.get_fundamentals("MU"), "(Yahoo Finance, live)"),
    ])
    def test_r17_vendor_pit_call_without_a_date_is_live_mode_under_a_pinned_run(
        self, monkeypatch, vendor_call, post_decision_marker,
    ):
        """Pinned (reported, not fixed): the vendor-level PIT clamps only narrow
        a date that is GIVEN; a direct call with ``curr_date=None`` is live mode
        even while an analysis date is pinned, so it serves quarters that were
        not yet filed and today's overview snapshot into a historical run.
        Production cannot reach this — the tool layer (``_pit_curr_date``) and
        the fundamentals bundle always pass the run's date — hence pinned, so a
        future direct caller that bypasses them is visible."""
        _g2_pin_vendor_seams(monkeypatch)
        with df_utils.pinned_analysis_date(_G2_PINNED):
            out = str(vendor_call())
        assert post_decision_marker in out

    def test_r17_vendor_pit_same_calls_with_the_pinned_date_stay_point_in_time(self, monkeypatch):
        """The contrast that makes the pin meaningful: handed the run's date,
        every one of those vendors keeps only what was public on it."""
        _g2_pin_vendor_seams(monkeypatch)
        with df_utils.pinned_analysis_date(_G2_PINNED):
            statement = y_finance.get_income_statement("AAPL", curr_date=_G2_PINNED)
            with pytest.raises(NoMarketDataError, match="overview snapshot is point-in-time"):
                y_finance.get_fundamentals("AAPL", curr_date=_G2_PINNED)
            av_out = avf.get_balance_sheet("AAPL", curr_date=_G2_PINNED)
            aggregate = fo.get_fundamentals("MU", curr_date=_G2_PINNED)
        assert statement.split("\n\n", 1)[1].splitlines() == [",2025-03-29", "Net Income,5.0"]
        assert av_out == {"quarterlyReports": []}
        assert "(Yahoo Finance, live)" not in aggregate
        assert "point-in-time guard: Yahoo snapshot is today-only" in aggregate


class TestR17G3CoinGeckoFormat:
    def test_r17_coingecko_vendor_fmt_usd_non_numeric_is_na(self) -> None:
        # gap _fmt_usd 240-241: TypeError / ValueError -> "n/a", never "$0.0000"
        assert cg._fmt_usd(None) == "n/a"
        assert cg._fmt_usd("not-a-number") == "n/a"
        assert cg._fmt_usd([1.0]) == "n/a"
        assert cg._fmt_usd({}) == "n/a"
        # numeric strings are numbers
        assert cg._fmt_usd("4.9e9") == "$4.90B"

    def test_r17_coingecko_vendor_fmt_usd_magnitude_ladder(self) -> None:
        assert cg._fmt_usd(0) == "$0.0000"
        assert cg._fmt_usd(-12.5) == "$-12.5000"
        assert cg._fmt_usd(0.01) == "$0.0100"
        assert cg._fmt_usd(999.9999) == "$999.9999"
        assert cg._fmt_usd(1000) == "$1.00K"
        assert cg._fmt_usd(1_500_000) == "$1.50M"
        assert cg._fmt_usd(2.5e9) == "$2.50B"
        assert cg._fmt_usd(1.67e12) == "$1.67T"
        assert cg._fmt_usd(1e15) == "$1000.00T"  # no unit above T

    def test_r17_coingecko_vendor_fmt_usd_sub_cent(self) -> None:
        assert cg._fmt_usd(0.005) == "$0.005"
        assert cg._fmt_usd(0.0010) == "$0.001"
        assert cg._fmt_usd(0.00001234) == "$0.00001234"
        assert cg._fmt_usd(1e-8) == "$0.00000001"

    def test_r17_coingecko_vendor_fmt_usd_degenerate_numbers(self) -> None:
        """Pinned by design (reported, not fixed): NaN and inf pass float()
        and skip every ``>=`` bucket, a JSON ``true`` reads as 1.0, and a
        value below the ``:.10f`` resolution collapses to "$0.0"."""
        assert cg._fmt_usd(math.nan) == "$nan"
        assert cg._fmt_usd(math.inf) == "$infT"
        assert cg._fmt_usd(-math.inf) == "$-inf"
        assert cg._fmt_usd(True) == "$1.0000"
        assert cg._fmt_usd(1e-12) == "$0.0"

    def test_r17_coingecko_vendor_fmt_price_ladder(self) -> None:
        # gap _fmt_price 262-263 (n/a) and 267 (0.01 <= v < 1)
        assert cg._fmt_price(None) == "n/a"
        assert cg._fmt_price("not-a-number") == "n/a"
        assert cg._fmt_price(object()) == "n/a"
        assert cg._fmt_price(83_228) == "83,228.00"
        assert cg._fmt_price(1) == "1.00"
        assert cg._fmt_price(0.01) == "0.0100"
        assert cg._fmt_price(0.5) == "0.5000"
        assert cg._fmt_price(0.9999) == "0.9999"
        assert cg._fmt_price(0.005123) == "0.005123"
        assert cg._fmt_price(0.0000123) == "0.0000123"
        assert cg._fmt_price(1e-8) == "0.00000001"

    def test_r17_coingecko_vendor_fmt_price_zero_negative_and_degenerate(self) -> None:
        """Gap line 271 (v <= 0) plus the pinned-by-design degenerate numbers:
        negative values lose the thousands separator because only the ``v >= 1``
        branch applies it."""
        assert cg._fmt_price(0) == "0.0"
        assert cg._fmt_price(-0.5) == "-0.5"
        assert cg._fmt_price(-1_234.5) == "-1234.5"
        assert cg._fmt_price(math.nan) == "nan"
        assert cg._fmt_price(math.inf) == "inf"
        assert cg._fmt_price(-math.inf) == "-inf"
        assert cg._fmt_price(True) == "1.00"
        assert cg._fmt_price(1e-12) == "0.0"

    def test_r17_coingecko_vendor_usd_field(self) -> None:
        # gap _usd_field 402: non-dict passthrough; a dict yields its USD leg only
        assert cg._usd_field({"usd": 126_080.0, "btc": 1.2}) == 126_080.0
        assert cg._usd_field({"btc": 1.2}) is None
        assert cg._usd_field({"usd": None}) is None
        assert cg._usd_field(67.81) == 67.81
        assert cg._usd_field(None) is None
        assert cg._usd_field("2025-10-06T10:57:42.000Z") == "2025-10-06T10:57:42.000Z"
        assert cg._usd_field([1.0]) == [1.0]

    def test_r17_coingecko_vendor_live_render_missing_usd_leg_reads_na(self) -> None:
        # _usd_field end to end: a missing USD leg renders n/a / "?", never 0
        payload = {
            "id": "x",
            "market_data": {
                "current_price": {"btc": 0.5},
                "market_cap": {"usd": 1.0e9},
                "total_volume": {"usd": 1.0e7},
                "ath": {"btc": 0.9},
                "ath_date": {"btc": "2024-01-01"},
                "atl": {"usd": 0.5},
                "atl_date": {"usd": "2020-03-13T00:00:00Z"},
            },
        }
        out = cg._render_live("x", "XYZ", payload)
        assert "- Price: n/a" in out
        assert "- ATH: n/a on ?" in out
        assert "ATL: $0.5000 on 2020-03-13" in out  # same line, after the ATH field
        assert "- Market cap: $1.00B" in out
        assert "- Market cap rank: n/a" in out
        assert "- Price change: n/a" in out
        assert "- FDV (derived price x max/total supply): n/a" in out


class TestR17G3CoinGeckoSymbols:
    def test_r17_coingecko_vendor_base_symbol_bare_symbol_falls_through(self) -> None:
        # gap perp_base_symbol 176: no separator, no quote suffix -> the symbol itself
        assert cg.perp_base_symbol("BTC") == "BTC"
        assert cg.perp_base_symbol("  eth ") == "ETH"
        assert cg.perp_base_symbol("") == ""
        # the quote asset alone is a whole symbol, not an empty base
        assert cg.perp_base_symbol("USDT") == "USDT"
        assert cg.perp_base_symbol("USDC") == "USDC"
        # a leading separator leaves an empty base segment: not split, not stripped
        assert cg.perp_base_symbol("-BTC") == "-BTC"
        assert cg.perp_base_symbol("/BTC") == "/BTC"

    def test_r17_coingecko_vendor_base_symbol_ladder_collisions(self) -> None:
        # separator ladder first
        assert cg.perp_base_symbol("BTC-USD") == "BTC"
        assert cg.perp_base_symbol("BTC-USDT") == "BTC"
        assert cg.perp_base_symbol("BTC/USDT") == "BTC"
        assert cg.perp_base_symbol("ETH-BTC") == "ETH"
        # compact contracts
        assert cg.perp_base_symbol("BTCUSDT") == "BTC"
        assert cg.perp_base_symbol("ETHUSDC") == "ETH"
        assert cg.perp_base_symbol("BTCUSDC") == "BTC"
        # USDT is tested before USDC
        assert cg.perp_base_symbol("USDCUSDT") == "USDC"
        # multiplier wrappers keep their prefix: 1000SATS and SATS stay distinct bases
        assert cg.perp_base_symbol("1000SATSUSDT") == "1000SATS"
        assert cg.perp_base_symbol("SATSUSDT") == "SATS"
        assert cg.perp_base_symbol("1000CATUSDT") == "1000CAT"
        assert cg.perp_base_symbol("1000000MOGUSDT") == "1000000MOG"


# ---------------------------------------------------------------------------
# CoinGecko market_chart: _daily resampling and the historical PIT window
# ---------------------------------------------------------------------------


def _g3_ms(y: int, m: int, d: int, h: int = 0, minute: int = 0, second: int = 0) -> int:
    """Millisecond epoch of a UTC instant (the market_chart wire format)."""
    return int(datetime(y, m, d, h, minute, second, tzinfo=UTC).timestamp() * 1000)


def _g3_cg_fake(chart: object, coin: object | None = None):
    """URL-dispatching ``_http_get_json`` fake: ``/search`` resolves BTC to
    ``bitcoin``, ``market_chart`` answers ``chart`` and ``/coins/bitcoin``
    answers ``coin``."""

    def fake(url, params=None):
        if url.endswith("/search"):
            return {"coins": [{"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1}]}
        if "/market_chart/" in url:
            return chart
        if url.endswith("/coins/bitcoin"):
            return coin
        raise AssertionError(url)

    return fake


def _g3_cg_chart(points: list) -> dict:
    """market_chart payload: the price ``points`` plus cap = price x 1e6 and
    a flat 5e8 volume, so every row of the rendered table is predictable."""
    return {
        "prices": points,
        "market_caps": [[p[0], p[1] * 1_000_000] for p in points],
        "total_volumes": [[p[0], 5.0e8] for p in points],
    }


@pytest.fixture()
def _g3_cg_passthrough(monkeypatch):
    """Zero the CoinGecko politeness throttle and bypass the disk cache and
    the parse memo: ``_cached_json`` runs the fetch directly."""
    monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
    monkeypatch.setattr(cg, "_cached_json", lambda filename, fetch, ttl_days: fetch())


#: Five daily price points around the as-of day 2026-06-27 (UTC).
_g3_pit_points = [
    [_g3_ms(2026, 6, 25), 100.0],
    [_g3_ms(2026, 6, 26), 110.0],
    [_g3_ms(2026, 6, 27, 23, 59, 59), 120.0],  # last second of the as-of day
    [_g3_ms(2026, 6, 28), 130.0],  # first second of the next day
    [_g3_ms(2026, 6, 29), 140.0],
]


class TestR17G3CoinGeckoDaily:
    def test_r17_coingecko_vendor_daily_non_list_is_empty(self, monkeypatch) -> None:
        # gap _daily 308: every non-list series is an empty daily series
        monkeypatch.setattr(
            cg,
            "_cached_json",
            lambda filename, fetch, ttl_days: {
                "prices": None, "market_caps": "x", "total_volumes": {"a": 1}, "ignored": 5,
            },
        )
        assert cg._market_chart_series("bitcoin") == {
            "prices": [], "market_caps": [], "total_volumes": [],
        }

    def test_r17_coingecko_vendor_daily_skips_malformed_points(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        # gaps _daily 312 (not a 2-item pair) and 315-316 (non-numeric ts / value)
        day = _g3_ms(2026, 6, 27)
        points = [
            "junk",  # not a pair
            [day],  # too short
            [day, 1.0, 2.0],  # too long
            None,
            42,
            (day, 7.5),  # a tuple is a pair too
            ["not-a-ts", 1.0],  # int() ValueError
            [None, 1.0],  # int() TypeError
            [day + 3_600_000, None],  # float() TypeError
            [day + 7_200_000, "abc"],  # float() ValueError
            [day + 7_200_000, "12.5"],  # numeric strings are numbers
        ]
        monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake({"prices": points}))
        assert cg._market_chart_series("bitcoin") == {
            "prices": [("2026-06-27", 12.5)], "market_caps": [], "total_volumes": [],
        }

    def test_r17_coingecko_vendor_daily_utc_bucket_last_listed_wins_sorted(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        points = [
            [_g3_ms(2026, 6, 28, 0, 1), 200.0],  # listed first, sorts last
            [_g3_ms(2026, 6, 27, 12), 130.0],
            [int(datetime(2026, 6, 26, 12, tzinfo=UTC).timestamp()), 100.0],  # seconds
            [_g3_ms(2026, 6, 27, 23, 59), 150.0],  # same UTC day: later-listed wins
        ]
        monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake({"prices": points}))
        assert cg._market_chart_series("bitcoin")["prices"] == [
            ("2026-06-26", 100.0), ("2026-06-27", 150.0), ("2026-06-28", 200.0),
        ]

    def test_r17_coingecko_vendor_daily_buckets_by_utc_day_under_non_utc_tz(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        """CI runs with TZ=UTC and the dev box is UTC+8; the daily bucket must
        follow the UTC calendar day either way (16:00Z is already the next
        calendar day at UTC+8)."""
        original = os.environ.get("TZ")
        monkeypatch.setenv("TZ", "Asia/Shanghai")
        if hasattr(time, "tzset"):
            time.tzset()
        try:
            points = [
                [_g3_ms(2026, 6, 27, 15, 59), 1.0],
                [_g3_ms(2026, 6, 27, 16, 0), 2.0],  # 00:00 on the 28th at UTC+8
                [_g3_ms(2026, 6, 27, 23, 30), 3.0],  # 07:30 on the 28th at UTC+8
                [_g3_ms(2026, 6, 28, 0, 0), 4.0],
            ]
            monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake({"prices": points}))
            assert cg._market_chart_series("bitcoin")["prices"] == [
                ("2026-06-27", 3.0), ("2026-06-28", 4.0),
            ]
        finally:
            if hasattr(time, "tzset"):
                if original is None:
                    os.environ.pop("TZ", None)
                else:
                    os.environ["TZ"] = original
                time.tzset()


class TestR17G3CoinGeckoHistoricalPit:
    def test_r17_coingecko_vendor_historical_keeps_as_of_day_drops_next_day(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake(_g3_cg_chart(_g3_pit_points)))
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026-06-27", look_back_days=2)
        assert (
            "point-in-time series as of 2026-06-27 (rows filtered to date <= 2026-06-27; "
            "window start 2026-06-25)"
        ) in out
        assert "| 2026-06-25 | 100.00 | $100.00M | $500.00M | 1.00M |" in out
        assert "| 2026-06-26 | 110.00 | $110.00M | $500.00M | 1.00M |" in out
        # the as-of day's own last observation is inside the window
        assert "| 2026-06-27 | 120.00 | $120.00M | $500.00M | 1.00M |" in out
        assert "2026-06-28" not in out
        assert "2026-06-29" not in out
        assert "130.00" not in out
        assert "140.00" not in out
        assert "price 100.00 -> 120.00 (+20.0% over 3 days)" in out
        assert "market cap $100.00M -> $120.00M (+20.0%)" in out

    def test_r17_coingecko_vendor_historical_row_on_trailing_limit_is_kept(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake(_g3_cg_chart(_g3_pit_points)))
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026-06-27", look_back_days=1)
        assert "window start 2026-06-26)" in out
        assert "| 2026-06-26 | 110.00 |" in out  # exactly on the trailing limit
        assert "2026-06-25" not in out  # one day before it
        assert "price 110.00 -> 120.00 (+9.1% over 2 days)" in out

    def test_r17_coingecko_vendor_historical_before_series_start_has_no_fabricated_rows(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        # the cached free-tier series starts 2025-07-01: a look-back reaching
        # before it renders only the real rows and names the free-tier window
        points = [[_g3_ms(2025, 7, d), 100.0 + d] for d in (1, 2, 3)]
        monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake(_g3_cg_chart(points)))
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2025-07-03", look_back_days=180)
        rows = [line for line in out.splitlines() if line.startswith("| 2025-")]
        assert rows == [
            "| 2025-07-01 | 101.00 | $101.00M | $500.00M | 1.00M |",
            "| 2025-07-02 | 102.00 | $102.00M | $500.00M | 1.00M |",
            "| 2025-07-03 | 103.00 | $103.00M | $500.00M | 1.00M |",
        ]
        assert "window start 2025-01-04" in out
        assert "Free-tier window: trailing ~365 days only." in out
        assert "over 3 days" in out
        # one day before the first real row: honest not-available, no table
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2025-06-30", look_back_days=180)
        assert "historical market data is NOT available" in out
        assert "| date |" not in out
        assert "Report 'data not available' for this period; do not estimate" in out

    def test_r17_coingecko_vendor_historical_missing_cap_and_volume_read_na(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        # cap / volume exist for the first day only: no forward fill, no zero fill
        payload = {
            "prices": [[_g3_ms(2026, 6, 26), 10.0], [_g3_ms(2026, 6, 27), 20.0]],
            "market_caps": [[_g3_ms(2026, 6, 26), 3.0e7]],
            "total_volumes": [[_g3_ms(2026, 6, 26), 0.0]],
        }
        monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake(payload))
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026-06-27", look_back_days=10)
        assert "| 2026-06-26 | 10.00 | $30.00M | $0.0000 | 3.00M |" in out
        assert "| 2026-06-27 | 20.00 | n/a | n/a | n/a |" in out
        assert "price 10.00 -> 20.00 (+100.0% over 2 days)" in out
        # the cap change needs both ends: nothing is summarised for it
        assert "market cap" not in out.split("Window summary", 1)[1]

    def test_r17_coingecko_vendor_historical_zero_first_price_skips_ratio(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        # a zero first price would divide by zero: the change is omitted, not inf
        payload = {
            "prices": [[_g3_ms(2026, 6, 26), 0.0], [_g3_ms(2026, 6, 27), 20.0]],
            "market_caps": [[_g3_ms(2026, 6, 26), 5.0e7]],
            "total_volumes": [],
        }
        monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake(payload))
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026-06-27", look_back_days=10)
        assert "| 2026-06-26 | 0.0 | $50.00M | n/a | n/a |" in out  # no supply from a 0 price
        assert "| 2026-06-27 | 20.00 | n/a | n/a | n/a |" in out
        assert "Window summary: 2 daily rows." in out
        assert "%)" not in out

    def test_r17_coingecko_vendor_historical_never_fetches_or_renders_the_current_snapshot(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        # current-only fields live in /coins/{id}: a historical run must not
        # even request it, so none of its figures can reach the PIT view
        coin = {
            "id": "bitcoin", "market_cap_rank": 424242, "categories": ["ZZZ"],
            "genesis_date": "2009-01-03", "watchlist_portfolio_users": 7777777,
            "market_data": {
                "current_price": {"usd": 777_777.0}, "market_cap": {"usd": 7.7e12},
                "total_volume": {"usd": 7.7e10}, "circulating_supply": 1.0e7,
                "max_supply": 2.1e7, "ath": {"usd": 999_999.0},
                "ath_date": {"usd": "2026-11-01T00:00:00Z"}, "atl": {"usd": 0.01},
                "atl_date": {"usd": "2013-07-05T00:00:00Z"},
                "price_change_percentage_24h": 55.5,
            },
        }
        urls = []
        dispatch = _g3_cg_fake(_g3_cg_chart(_g3_pit_points), coin)

        def spy(url, params=None):
            urls.append(url)
            return dispatch(url, params)

        monkeypatch.setattr(cg, "_http_get_json", spy)
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026-06-27", look_back_days=10)
        assert urls and all(u.endswith("/search") or "/market_chart/" in u for u in urls), urls
        for current_only in (
            "CURRENT SNAPSHOT", "Market cap rank", "#424242", "777,777.00", "$7.70T",
            "ATH:", "ATL:", "Genesis date", "ZZZ", "+55.5%", "999",
        ):
            assert current_only not in out, current_only
        assert "| 2026-06-27 | 120.00 |" in out  # the PIT series did render

    def test_r17_coingecko_vendor_historical_malformed_date_degrades_before_series_fetch(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        urls = []
        dispatch = _g3_cg_fake(_g3_cg_chart(_g3_pit_points))

        def spy(url, params=None):
            urls.append(url)
            return dispatch(url, params)

        monkeypatch.setattr(cg, "_http_get_json", spy)
        for bad in ("2026-13-45", "2026/06/27", "2026-06-31", "2026-178"):
            out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date=bad)
            assert f"the passed curr_date '{bad}' is not a parseable yyyy-mm-dd date" in out
            assert "do not estimate" in out
        # the symbol resolved, but the series endpoint was never reached
        assert urls and all(u.endswith("/search") for u in urls), urls

    def test_r17_coingecko_vendor_live_unusable_payload_raises_no_market_data(
        self, _g3_cg_passthrough, monkeypatch,
    ) -> None:
        # gap get_crypto_token_fundamentals 539: a 200 whose body is not a
        # coin object raises the typed error instead of a snapshot of n/a
        for payload in (
            {"status": {"error_code": 1, "error_message": "object not found"}},
            [],
            {"name": "no id"},
            "",
        ):
            monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake({}, payload))
            with pytest.raises(NoMarketDataError, match="unusable payload for 'bitcoin'") as excinfo:
                cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="")  # live route
            assert excinfo.value.symbol == "BTCUSDT"




class TestR17G3CoinGeckoCompactDate:
    @pytest.mark.parametrize(
        ("curr_date", "as_of", "lower", "kept", "first_dropped"),
        [
            ("20260627", "2026-06-27", "2026-06-25", "120.00", "130.00"),
            ("2026-W26-5", "2026-06-26", "2026-06-24", "110.00", "120.00"),
            ("2026W265", "2026-06-26", "2026-06-24", "110.00", "120.00"),
        ],
    )
    def test_r17_coingecko_vendor_historical_compact_iso_date_is_normalized(
        self, _g3_cg_passthrough, monkeypatch, curr_date, as_of, lower, kept, first_dropped,
    ) -> None:
        """Round-17 fix (PIT leak). ``date.fromisoformat`` also accepts the
        basic (``20260627``) and ISO week (``2026-W26-5`` / ``2026W265``)
        forms on Python >= 3.11, so the malformed-date guard let such a label
        through unchanged. ``_render_historical`` compares the series'
        ``YYYY-MM-DD`` keys against that label as plain STRINGS, and
        ``"2026-06-28" <= "20260627"`` holds because ``"-"`` sorts before
        ``"0"``: every row of the same year passed the point-in-time filter —
        rows dated AFTER curr_date were rendered as part of the as-of view."""
        monkeypatch.setattr(cg, "_http_get_json", _g3_cg_fake(_g3_cg_chart(_g3_pit_points)))
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date=curr_date, look_back_days=2)
        # the leak first: the first row after the as-of day must be gone
        assert first_dropped not in out, out
        assert "140.00" not in out
        assert (
            f"point-in-time series as of {as_of} (rows filtered to date <= {as_of}; "
            f"window start {lower})"
        ) in out
        assert kept in out
        assert curr_date not in out  # the label is the canonical dashed day




# ---------------------------------------------------------------------------
# onchain_flows — chart points, the disk-cache wrapper and the PIT filter
# ---------------------------------------------------------------------------


class TestR17G3OnchainPoints:
    def test_r17_onchain_flows_points_skips_malformed_rows(self) -> None:
        # gaps _points 115 (non-dict row / missing x) and 119-120 (bad x / bad y)
        payload = {
            "values": [
                "junk",  # not a dict
                {"y": 5.0},  # no x
                {"x": None, "y": 5.0},  # x is None
                {"x": "abc", "y": 5.0},  # float() ValueError
                {"x": [1], "y": 5.0},  # float() TypeError
                {"x": 1e18, "y": 5.0},  # year out of range
                {"x": 1_700_017_200, "y": "nan"},  # float('nan') IS a number
                {"x": 1_700_008_600, "y": None},  # float() TypeError
                {"x": 1_700_008_600, "y": "abc"},  # float() ValueError
                {"x": "1700000000", "y": "2"},  # numeric strings are numbers
            ],
        }
        points = ocf._points(payload)
        assert len(points) == 2
        assert points[0] == (datetime.fromtimestamp(1_700_000_000, tz=UTC), 2.0)
        # float('nan') IS a number: the row is kept, as NaN
        assert points[1][0] == datetime.fromtimestamp(1_700_017_200, tz=UTC)
        assert math.isnan(points[1][1])

    def test_r17_onchain_flows_points_sorts_and_keeps_equal_instants(self) -> None:
        # newest-first input comes out oldest-first; equal instants are both kept
        payload = {
            "values": [
                {"x": 300, "y": 3.0},
                {"x": 100, "y": 1.0},
                {"x": 300, "y": 30.0},
                {"x": 200, "y": 2.0},
            ],
        }
        points = ocf._points(payload)
        assert [int(p[0].timestamp()) for p in points] == [100, 200, 300, 300]
        assert [p[1] for p in points] == [1.0, 2.0, 3.0, 30.0]
        assert ocf._points(None) == []
        assert ocf._points({}) == []
        assert ocf._points({"values": []}) == []
        assert ocf._points({"values": None}) == []

    def test_r17_onchain_flows_points_skips_row_missing_y(self) -> None:
        """Round-17 fix: the docstring promises "malformed rows skipped", but a
        row carrying an x and NO y raised ``KeyError('y')`` — outside the
        caught tuple — so one bad point discarded the whole chart (the
        caller's ``except Exception`` turned it into "no usable points" plus a
        sentinel)."""
        points = ocf._points(
            {
                "values": [
                    {"x": 1_700_000_000, "y": 2.0},
                    {"x": 1_700_008_600},  # malformed: no y
                    {"x": 1_700_017_200, "y": 4.0},
                ],
            },
        )
        assert points == [
            (datetime.fromtimestamp(1_700_000_000, tz=UTC), 2.0),
            (datetime.fromtimestamp(1_700_017_200, tz=UTC), 4.0),
        ]


class TestR17G3OnchainPit:
    def test_r17_onchain_flows_pit_keeps_as_of_second_drops_next_day(self, monkeypatch) -> None:
        day0 = datetime(2026, 11, 18, tzinfo=UTC)
        offsets = [
            timedelta(0),
            timedelta(days=1),
            timedelta(days=2, hours=23, minutes=59, seconds=59),  # last second of as-of
            timedelta(days=3),  # first second of the next day
            timedelta(days=3, hours=12),
        ]
        payload = {
            "values": [
                {"x": int((day0 + off).timestamp()), "y": 10.0 + i}
                for i, off in enumerate(offsets)
            ],
        }
        monkeypatch.setattr(
            ocf, "_fetch_chart_cached", lambda chart: (payload, "2026-11-25T00:00:00+00:00"),
        )
        flows = ocf.fetch_onchain_flows("2026-11-20")
        assert flows is not None
        for entry in flows.series.values():
            instant, latest, mean, span_days = entry
            assert instant == datetime(2026, 11, 20, 23, 59, 59, tzinfo=UTC)
            # mean over the three surviving points (10, 11, 12) only
            assert (latest, mean, span_days) == (12.0, 11.0, 2)
        assert flows.available_at == "2026-11-20T23:59:59+00:00"
        assert flows.fetched_at == "2026-11-25T00:00:00+00:00"  # the wrapper's own stamp
        assert quality.snapshot_quality() == []

    def test_r17_onchain_flows_one_row_missing_y_keeps_the_chart(self, monkeypatch) -> None:
        """End to end for the round-17 fix: a single y-less row no longer
        turns the chart into "unavailable" + a ledger sentinel."""
        start = datetime(2026, 11, 18, tzinfo=UTC)
        payload = {
            "values": [
                {"x": int(start.timestamp()), "y": 1.0},
                {"x": int((start + timedelta(days=1)).timestamp())},  # no y
                {"x": int((start + timedelta(days=2)).timestamp()), "y": 3.0},
            ],
        }
        monkeypatch.setattr(
            ocf, "_fetch_chart_cached", lambda chart: (payload, "2026-11-25T00:00:00+00:00"),
        )
        flows = ocf.fetch_onchain_flows("2026-11-25")
        assert flows is not None
        assert {entry[1] for entry in flows.series.values()} == {3.0}
        assert {entry[2] for entry in flows.series.values()} == {2.0}  # mean of the two good rows
        assert {entry[3] for entry in flows.series.values()} == {2}
        assert quality.snapshot_quality() == []
        block = ocf.render_onchain_block(flows)
        assert block.count("3 latest (2026-11-20; 2d mean 2;") == 2
        assert "unavailable" not in block

    def test_r17_onchain_flows_span_floor_and_truncated_days(self, monkeypatch) -> None:
        """Pinned by design (reported, not fixed): the span label is
        ``timedelta.days`` — truncated, and floored at 1 — so a span under 24h
        and a span of 1 day 23h both read "1d"."""
        day0 = datetime(2026, 11, 19, tzinfo=UTC)

        def _span(hours):
            payload = {
                "values": [
                    {"x": int(day0.timestamp()), "y": 4.0},
                    {"x": int((day0 + timedelta(hours=hours)).timestamp()), "y": 8.0},
                ],
            }
            monkeypatch.setattr(ocf, "_fetch_chart_cached", lambda chart: (payload, ""))
            flows = ocf.fetch_onchain_flows("2026-11-25")
            assert flows is not None
            return {entry[3] for entry in flows.series.values()}, {entry[2] for entry in flows.series.values()}

        assert _span(23) == ({1}, {6.0})  # floored
        assert _span(47) == ({1}, {6.0})  # 1 day 23 hours, truncated
        assert _span(48) == ({2}, {6.0})

        single = {"values": [{"x": int(day0.timestamp()), "y": 9.0}]}
        monkeypatch.setattr(ocf, "_fetch_chart_cached", lambda chart: (single, ""))
        flows = ocf.fetch_onchain_flows("2026-11-25")
        assert flows is not None
        # one point: mean is the point itself (no division by zero), span floored
        assert {entry[1:] for entry in flows.series.values()} == {(9.0, 9.0, 1)}
        assert "9 latest (2026-11-19; 1d mean 9;" in ocf.render_onchain_block(flows)


class TestR17G3OnchainCache:
    def test_r17_onchain_flows_fetch_sends_timeout_and_window_params(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return SimpleNamespace(
                text='{"values": [{"x": 1, "y": 2.0}]}',
                raise_for_status=lambda: None,
            )

        monkeypatch.setattr(ocf, "requests_get", fake_get)
        payload = ocf.fetch_chart_json("n-unique-addresses")
        assert payload == {"values": [{"x": 1, "y": 2.0}]}
        assert captured["url"] == f"{ocf.CHARTS_BASE_URL}/n-unique-addresses"
        assert captured["params"] == {"timespan": "30days", "format": "json", "sampled": "true"}
        assert captured["timeout"] == 15

    def test_r17_onchain_flows_fetch_chart_json_error_paths(self, monkeypatch) -> None:
        def _raise_http():
            raise requests.exceptions.HTTPError("504 Server Error")

        monkeypatch.setattr(
            ocf, "requests_get", lambda url, **kw: SimpleNamespace(text="Gateway Timeout", raise_for_status=_raise_http),
        )
        with pytest.raises(requests.exceptions.HTTPError, match="504"):
            ocf.fetch_chart_json("n-unique-addresses")

        monkeypatch.setattr(
            ocf,
            "requests_get",
            lambda url, **kw: SimpleNamespace(text="<html>maintenance</html>", raise_for_status=lambda: None),
        )
        with pytest.raises(ValueError, match="Expecting value"):
            ocf.fetch_chart_json("n-unique-addresses")

    @pytest.mark.parametrize("raw", [b"\xff\xfe\x00garbage", b"{not json", b"[1, 2]", b"null"])
    def test_r17_onchain_flows_cached_unusable_wrapper_fails_soft(self, raw, monkeypatch) -> None:
        """Gaps _fetch_chart_cached 187-188 (bytes that are not UTF-8 JSON)
        and 190 (a JSON document that is not an object). Pinned by design
        (reported, not fixed): a fresh-but-unusable cache file is served as-is
        for the rest of its TTL — the fetch is NOT retried."""
        (Path(vendor_cache_dir("onchain")) / "n-unique-addresses_30days.json").write_bytes(raw)

        def never(chart, timespan=ocf._TIMESPAN):
            raise AssertionError("a fresh cache file must not be refetched")

        monkeypatch.setattr(ocf, "fetch_chart_json", never)
        assert ocf._fetch_chart_cached("n-unique-addresses") is None

    def test_r17_onchain_flows_cached_round_trip_fresh_hit_and_ttl_expiry(self, monkeypatch) -> None:
        calls: list = []

        def fake_fetch(chart, timespan=ocf._TIMESPAN):
            calls.append(chart)
            return {"values": [{"x": len(calls), "y": float(len(calls))}]}

        monkeypatch.setattr(ocf, "fetch_chart_json", fake_fetch)
        before = datetime.now(UTC)
        first = ocf._fetch_chart_cached("n-unique-addresses")
        second = ocf._fetch_chart_cached("n-unique-addresses")
        assert calls == ["n-unique-addresses"]  # the second call is a fresh disk hit
        assert first[0] == second[0] == {"values": [{"x": 1, "y": 1.0}]}
        # the wrapper's own stamp travels out with the bytes
        assert first[1] == second[1]
        assert before.replace(microsecond=0) <= datetime.fromisoformat(first[1]) <= datetime.now(UTC)

        path = Path(vendor_cache_dir("onchain")) / "n-unique-addresses_30days.json"
        old = time.time() - 7 * 3600  # past the 6-hour TTL
        os.utime(path, (old, old))
        third = ocf._fetch_chart_cached("n-unique-addresses")
        assert calls == ["n-unique-addresses"] * 2
        assert third[0] == {"values": [{"x": 2, "y": 2.0}]}

    def test_r17_onchain_flows_fetch_failure_serves_stale_chart_ledgered(self, monkeypatch) -> None:
        set_config({"data_cache_max_stale_days": 30})
        monkeypatch.setattr(
            ocf, "fetch_chart_json", lambda chart, timespan=ocf._TIMESPAN: {"values": [{"x": 1, "y": 7.0}]},
        )
        first = ocf._fetch_chart_cached("n-unique-addresses")
        path = Path(vendor_cache_dir("onchain")) / "n-unique-addresses_30days.json"
        old = time.time() - 7 * 3600  # past the TTL, far inside the 30-day stale cap
        os.utime(path, (old, old))

        def down(chart, timespan=ocf._TIMESPAN):
            raise requests.exceptions.HTTPError("503 Server Error")

        monkeypatch.setattr(ocf, "fetch_chart_json", down)
        stale = ocf._fetch_chart_cached("n-unique-addresses")
        assert stale == first  # the old payload AND its original fetched_at
        assert [(e["method"], e["kind"]) for e in quality.snapshot_quality()] == [
            ("onchain/n-unique-addresses_30days.json", quality.KIND_STALE_CACHE),
        ]

        ancient = time.time() - 40 * 86_400  # beyond the stale cap
        os.utime(path, (ancient, ancient))
        assert ocf._fetch_chart_cached("n-unique-addresses") is None  # fail_open, no stale serve
        assert len(quality.snapshot_quality()) == 1

    def test_r17_onchain_flows_transport_failure_writes_no_cache(self, monkeypatch) -> None:
        def boom(url, **kwargs):
            raise requests.exceptions.ConnectTimeout("timed out")

        monkeypatch.setattr(ocf, "requests_get", boom)
        assert ocf._fetch_chart_cached("estimated-transaction-volume-usd") is None
        # nothing written: the next call refetches instead of replaying the outage
        assert list(Path(vendor_cache_dir("onchain")).iterdir()) == []




# ---------------------------------------------------------------------------
# DefiLlama — parse-memo fallback, symbol matching, chain ladder, PIT window
# ---------------------------------------------------------------------------


def _g3_dl_ts(y: int, m: int, d: int, h: int = 0, minute: int = 0, second: int = 0) -> int:
    """Second epoch of a UTC instant (DefiLlama's wire format)."""
    return int(datetime(y, m, d, h, minute, second, tzinfo=UTC).timestamp())


def _g3_dl_catalog() -> list:
    """Raw /protocols entries covering the projection's edge shapes."""
    return [
        {"slug": "uniswap-v3", "name": "Uniswap V3", "symbol": "UNI", "tvl": 1.68e9,
         "chains": ["Ethereum", "Arbitrum"], "category": "Dexs"},
        {"slug": "uniswap-v2", "name": "Uniswap V2", "symbol": "UNI", "tvl": 1.05e9,
         "chains": ["Ethereum"], "category": "Dexs"},
        {"slug": "uniswap", "name": "Uniswap (new)", "symbol": "uni", "tvl": 1.7e9,
         "chains": ["Ethereum"], "category": "Dexs"},
        {"slug": "shibex", "name": "Shibex", "symbol": "SHIB", "tvl": 999_999.99,
         "chains": ["Ethereum"], "category": "Other"},
        {"id": 555, "name": "Edge Protocol", "symbol": "EDGE", "tvl": 1_000_000,
         "chains": ["Ethereum"]},  # no slug: numeric id; exactly on the floor
        {"slug": "str-tvl", "name": "Str", "symbol": "STR", "tvl": "42000000",
         "chains": ["Ethereum"]},  # tvl arrives as a numeric string
        "not-a-dict",  # dropped by the projection
    ]


#: Four chain-TVL points around the as-of day 2026-09-01 with a 2-day look-back.
_g3_dl_points = [
    {"date": _g3_dl_ts(2026, 8, 30) - 1, "tvl": 1.0},  # one second before the trailing limit
    {"date": _g3_dl_ts(2026, 8, 30), "tvl": 2.0},  # exactly on the trailing limit
    {"date": _g3_dl_ts(2026, 9, 1, 23, 59, 59), "tvl": 3.0},  # last second of the as-of day
    {"date": _g3_dl_ts(2026, 9, 2), "tvl": 4.0},  # first second of the next day
]


@pytest.fixture()
def _g3_dl_passthrough(monkeypatch):
    """Zero the DefiLlama politeness throttle and bypass the disk cache and
    the parse memo: ``_cached_json`` runs the fetch directly."""
    monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))
    monkeypatch.setattr(dl, "_cached_json", lambda filename, fetch, ttl_days: fetch())


class TestR17G3DefillamaResolution:
    def test_r17_defillama_vendor_cached_json_none_from_helper_raises(self, monkeypatch) -> None:
        # gap defillama _cached_json 134: the helper's fail-open None (never
        # requested by this vendor) must not flow on as a payload
        monkeypatch.setattr(dl, "cached_or_fetch", lambda *args, **kwargs: None)
        with pytest.raises(NoMarketDataError, match="cache helper returned None") as excinfo:
            dl._cached_json("chains.json", lambda: {}, 1.0)
        assert excinfo.value.symbol == "chains.json"

    def test_r17_defillama_vendor_projection_normalizes_edge_shapes(self, _g3_dl_passthrough, monkeypatch) -> None:
        monkeypatch.setattr(dl, "_http_get_json", lambda url: _g3_dl_catalog())
        by_slug = {p["slug"]: p for p in dl._protocols_projection()}
        assert set(by_slug) == {"uniswap-v3", "uniswap-v2", "uniswap", "shibex", "555", "str-tvl"}
        assert isinstance(by_slug["555"]["slug"], str)  # numeric id -> str at the source
        assert by_slug["uniswap"]["symbol"] == "UNI"  # upper-cased at the source
        assert by_slug["uniswap-v3"]["chains"] == ["Ethereum", "Arbitrum"]
        assert by_slug["555"]["category"] is None

    def test_r17_defillama_vendor_protocol_highest_tvl_wins_and_floor(
        self, _g3_dl_passthrough, monkeypatch,
    ) -> None:
        monkeypatch.setattr(dl, "_http_get_json", lambda url: _g3_dl_catalog())
        assert dl._resolve_protocol("UNI")["slug"] == "uniswap"  # 1.7e9 beats v3's 1.68e9
        assert dl._resolve_protocol("uni")["slug"] == "uniswap"  # base symbol is case-insensitive
        assert dl._resolve_protocol("EDGE")["slug"] == "555"  # exactly on the $1M floor clears it
        assert dl._resolve_protocol("STR")["slug"] == "str-tvl"  # numeric-string tvl compares numerically
        assert dl._resolve_protocol("SHIB") is None  # 999,999.99: just under the floor
        assert dl._resolve_protocol("DOGE") is None

    def test_r17_defillama_vendor_protocol_tie_keeps_catalog_order(self, monkeypatch) -> None:
        # pinned by design (reported, not fixed): equal TVL -> first catalog entry
        projection = [
            {"slug": "b-protocol", "name": "B", "symbol": "TIE", "tvl": 5.0e8},
            {"slug": "a-protocol", "name": "A", "symbol": "TIE", "tvl": 5.0e8},
        ]
        monkeypatch.setattr(dl, "_protocols_projection", lambda: projection)
        assert dl._resolve_protocol("TIE")["slug"] == "b-protocol"

    def test_r17_defillama_vendor_highest_tvl_version_is_disclosed(self, monkeypatch) -> None:
        def fake(url):
            if url.endswith("/protocols"):
                return _g3_dl_catalog()
            if url == f"{dl._BASE_URL}/protocol/uniswap":  # any other slug trips the AssertionError
                return {"tvl": [
                    {"date": _g3_dl_ts(2026, 8, 30) + i * 86_400, "totalLiquidityUSD": 1.7e9 - i * 1e6}
                    for i in range(5)
                ]}
            if url.endswith("/v2/chains"):
                return [{"name": "Ethereum", "tvl": 5.2e10}]
            if "/historicalChainTvl/" in url:
                return []
            raise AssertionError(url)

        monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))
        monkeypatch.setattr(dl, "_cached_json", lambda filename, fetch, ttl_days: fetch())
        monkeypatch.setattr(dl, "_http_get_json", fake)
        out = dl.get_crypto_network_fundamentals("UNIUSDT", curr_date="2026-09-01", look_back_days=30)
        assert (
            "Protocol TVL — Uniswap (new) (slug 'uniswap', category Dexs; "
            "chosen as the highest-TVL deployment carrying symbol UNI"
        ) in out
        assert "| 2026-09-01 | $1.70B |" in out
        assert "2026-09-02" not in out  # PIT: rows after curr_date dropped
        assert "uniswap-v3" not in out and "uniswap-v2" not in out
        # current-only figures stay out of the historical view
        assert "current TVL" not in out and "ranks #" not in out

    def test_r17_defillama_vendor_chain_ladder_static_then_catalog_then_none(self, monkeypatch) -> None:
        urls: list = []
        catalog = [
            {"name": "Junk", "tokenSymbol": "ASTR", "tvl": 1_000_000},  # on the floor, lowest
            {"name": "Astra", "tokenSymbol": "astr", "tvl": 6.0e8},  # lower-case tokenSymbol
            {"name": "Astra Clone", "tokenSymbol": "ASTR", "tvl": 6.0e8},  # tie: catalog order
            {"name": "Bool", "tokenSymbol": "ASTR", "tvl": True},  # a JSON true is an int of 1
        ]

        def fake(url):
            urls.append(url)
            return catalog

        monkeypatch.setattr(dl, "_cached_json", lambda filename, fetch, ttl_days: fetch())
        monkeypatch.setattr(dl, "_http_get_json", fake)
        # rung 1: the static table wins without touching the catalog
        assert dl._chain_for_base("BTC") == "Bitcoin"
        assert dl._chain_for_base("ZK") == "zkSync Era"  # the chain NAME keeps its own case
        assert urls == []
        # rung 2: catalog tokenSymbol, highest TVL at/above the floor
        assert dl._chain_for_base("ASTR") == "Astra"
        assert urls == [f"{dl._BASE_URL}/v2/chains"]
        # no rung matches: honest None
        assert dl._chain_for_base("NOPE") is None

        def down(url):
            raise NoMarketDataError(url, detail="DefiLlama HTTP 503")

        monkeypatch.setattr(dl, "_http_get_json", down)
        assert dl._chain_for_base("ASTR") is None  # typed catalog failure -> no fabricated chain
        assert dl._chain_for_base("BTC") == "Bitcoin"  # the static table still answers


class TestR17G3DefillamaPit:
    def test_r17_defillama_vendor_pit_window_inclusive_at_both_utc_day_edges(self) -> None:
        assert dl._pit_window(_g3_dl_points, "date", "tvl", "2026-09-01", 2) == [
            ("2026-08-30", 2.0), ("2026-09-01", 3.0),
        ]

    def test_r17_defillama_vendor_pit_window_compact_label_filters_numerically(self) -> None:
        # DefiLlama compares epoch seconds, so a basic-ISO label (which
        # date.fromisoformat accepts) filters correctly — unlike CoinGecko's
        # string comparison (see the compact-date PIT test above)
        assert dl._pit_window(_g3_dl_points, "date", "tvl", "20260901", 2) == [
            ("2026-08-30", 2.0), ("2026-09-01", 3.0),
        ]

    def test_r17_defillama_vendor_render_series_zero_first_tvl_and_sampling(self) -> None:
        # a zero first TVL would divide by zero: the change is omitted, not inf
        rows = [("2026-01-01", 0.0), ("2026-01-02", 5.0e8), ("2026-01-03", 2.5e8)]
        out = dl._render_series("Chain TVL — X", "TVL", rows)
        assert "Window summary: $0.0000 -> $250.00M; window peak $500.00M on 2026-01-02." in out
        assert "%)" not in out
        # the table samples the most recent 30 rows and says so
        many = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(35)]
        out = dl._render_series("Chain TVL — X", "TVL", [(d, 1.0e9) for d in many])
        assert "; 35 rows in window, table shows last 30):" in out
        table_days = [line.split("|")[1].strip() for line in out.splitlines() if line.startswith("| 2026-")]
        assert table_days == many[-30:]

    def test_r17_defillama_vendor_compact_curr_date_renders_correct_pit_rows(self, monkeypatch) -> None:
        def fake(url):
            if url.endswith("/protocols"):
                return []
            if "/historicalChainTvl/Bitcoin" in url:
                return _g3_dl_points
            raise AssertionError(url)

        monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))
        monkeypatch.setattr(dl, "_cached_json", lambda filename, fetch, ttl_days: fetch())
        monkeypatch.setattr(dl, "_http_get_json", fake)
        out = dl.get_crypto_network_fundamentals("BTCUSDT", curr_date="20260901", look_back_days=2)
        assert "| 2026-08-30 | $2.0000 |" in out
        assert "| 2026-09-01 | $3.0000 |" in out
        assert "2026-09-02" not in out
        # the label is echoed raw (cosmetic; the rows are what matter)
        assert "Crypto network fundamentals — BTC (as-of 20260901)." in out


class TestR17G3DefillamaVerdict:
    def test_r17_defillama_vendor_catalog_failure_is_not_reported_as_no_coverage(self, monkeypatch) -> None:
        """Round-17 fix: a /protocols failure left ``protocol = None`` with the
        exception recorded in ``failures``; when the chain rung then found no
        chain either, the verdict tested ``protocol is None and chain is None``
        FIRST and raised "no DefiLlama coverage for base symbol 'UNI' (not a
        known protocol governance token ...)" — a vendor outage shown as a
        permanent property of the symbol, and the recorded failure dropped."""
        monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))

        def fake(url):
            if url.endswith("/protocols"):
                raise NoMarketDataError(url, detail="DefiLlama HTTP 503: upstream down")
            if url.endswith("/v2/chains"):
                return []  # reachable, simply no chain carries the token
            raise AssertionError(url)

        monkeypatch.setattr(dl, "_http_get_json", fake)
        with pytest.raises(NoMarketDataError, match="HTTP 503") as excinfo:
            dl.get_crypto_network_fundamentals("UNIUSDT", curr_date="2026-09-02", look_back_days=30)
        assert "no DefiLlama coverage" not in str(excinfo.value)
        assert excinfo.value.symbol == f"{dl._BASE_URL}/protocols"

    def test_r17_defillama_vendor_catalog_rate_limit_stays_typed(self, monkeypatch) -> None:
        # same masking: the 429 must reach the router as VendorRateLimitError
        monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))

        def fake(url):
            if url.endswith("/protocols"):
                raise VendorRateLimitError("DefiLlama returned 429 for protocols")
            if url.endswith("/v2/chains"):
                return []
            raise AssertionError(url)

        monkeypatch.setattr(dl, "_http_get_json", fake)
        with pytest.raises(VendorRateLimitError, match="429"):
            dl.get_crypto_network_fundamentals("UNIUSDT", curr_date="2026-09-02", look_back_days=30)

    def test_r17_defillama_vendor_genuine_no_coverage_still_raises_no_coverage(self, monkeypatch) -> None:
        # no failure recorded anywhere: the typed no-coverage error is the verdict
        monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))

        def fake(url):
            if url.endswith("/protocols"):
                return _g3_dl_catalog()
            if url.endswith("/v2/chains"):
                return []
            raise AssertionError(url)

        monkeypatch.setattr(dl, "_cached_json", lambda filename, fetch, ttl_days: fetch())
        monkeypatch.setattr(dl, "_http_get_json", fake)
        with pytest.raises(NoMarketDataError, match="no DefiLlama coverage for base symbol 'DOGE'"):
            dl.get_crypto_network_fundamentals("DOGEUSDT", curr_date="2026-09-02", look_back_days=30)

    def test_r17_defillama_vendor_unexpected_chain_resolution_error_is_logged(self, monkeypatch, caplog) -> None:
        """Round-17 fix: the chain leg's catch-all discarded an unexpected
        resolution error with no log line (its sibling, the protocol-resolution
        handler, logs), so a corrupt chains cache or an unwritable cache
        directory left no trace at all."""
        monkeypatch.setattr(dl, "_http_get_json", lambda url: [])

        def boom(base):
            raise ValueError("cache blob is not JSON")

        monkeypatch.setattr(dl, "_chain_for_base", boom)
        with (
            caplog.at_level(logging.WARNING, logger="yialpha.dataflows.defillama_vendor"),
            pytest.raises(NoMarketDataError, match="no DefiLlama coverage"),
        ):
            dl.get_crypto_network_fundamentals("DOGEUSDT", curr_date="2026-09-02", look_back_days=30)
        chain_logs = [r.getMessage() for r in caplog.records if "chain resolution" in r.getMessage()]
        assert chain_logs == ["DefiLlama chain resolution failed for DOGE: cache blob is not JSON"]

    def test_r17_defillama_vendor_chain_catalog_down_reads_as_coverage_gap(self, monkeypatch) -> None:
        """Pinned by design (reported, not fixed): ``_chain_for_base`` returns
        None on a typed catalog failure ("resolver unavailable") with no log
        line and no failure note, so a /v2/chains outage for a base outside the
        static table reads as "no coverage" — the same wording as a genuine
        miss."""
        monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))

        def fake(url):
            if url.endswith("/protocols"):
                return []
            raise NoMarketDataError(url, detail="DefiLlama HTTP 503")

        monkeypatch.setattr(dl, "_http_get_json", fake)
        with pytest.raises(NoMarketDataError, match="no DefiLlama coverage for base symbol 'PEPE'"):
            dl.get_crypto_network_fundamentals("PEPEUSDT", curr_date="2026-09-02", look_back_days=30)




# ---------------------------------------------------------------------------
# CoinGecko HTTP seam, throttles, the disk-cache adapter and the live gate
# ---------------------------------------------------------------------------


def _g3_http_response(status_code: int, body: bytes) -> requests.Response:
    """A real ``requests.Response`` carrying ``body`` (no network), so the
    vendor's ``.text`` / ``.json()`` paths run against the library itself."""
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = body
    resp.encoding = "utf-8"
    return resp


def _g3_measure_throttle(mod, interval: float, monkeypatch) -> list:
    """Drive ``mod._throttle()`` on a fresh throttle against a fake monotonic
    clock; return the sleep durations it asked for."""
    clock = {"t": 5000.0}
    sleeps: list = []
    monkeypatch.setattr(mod, "_throttler", MinIntervalThrottle(interval))
    monkeypatch.setattr(disk_cache_mod.time, "monotonic", lambda: clock["t"])

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(disk_cache_mod.time, "sleep", fake_sleep)
    mod._throttle()  # first call on a fresh throttle: nothing to wait for
    clock["t"] += interval / 4
    mod._throttle()
    clock["t"] += 10 * interval
    mod._throttle()  # long idle: nothing to wait for again
    return sleeps


class _g3_FrozenDatetime(datetime):
    """datetime subclass whose now() is pinned — the wall-clock seam behind
    ``utils.is_historical_date``'s two live anchors (host-local and UTC)."""

    pinned = datetime(2026, 11, 25, 3, 0, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.pinned if tz is not None else cls.pinned.replace(tzinfo=None)


class TestR17G3Throttle:
    def test_r17_coingecko_vendor_throttle_sleep_is_bounded_by_interval(self, monkeypatch) -> None:
        # the documented MinIntervalThrottle(4.0): one sleep, never longer
        # than the interval, none when the previous call is long past
        assert cg._throttler._min_interval == 4.0
        assert _g3_measure_throttle(cg, 4.0, monkeypatch) == [pytest.approx(3.0)]

    def test_r17_defillama_vendor_throttle_sleep_is_bounded_by_interval(self, monkeypatch) -> None:
        # the documented MinIntervalThrottle(1.0)
        assert dl._throttler._min_interval == 1.0
        assert _g3_measure_throttle(dl, 1.0, monkeypatch) == [pytest.approx(0.75)]


class TestR17G3CoinGeckoHttp:
    def test_r17_coingecko_vendor_get_sends_browser_ua_timeout_params_and_proxy(self, monkeypatch) -> None:
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
        for var in ("HTTP_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "socks5h://127.0.0.1:1080")
        captured: dict = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _g3_http_response(200, b'{"coins": []}')

        monkeypatch.setattr("requests.get", fake_get)
        url = f"{cg._BASE_URL}/search"
        assert cg._http_get_json(url, params={"query": "BTC"}) == {"coins": []}
        assert captured["url"] == url
        assert captured["params"] == {"query": "BTC"}
        assert captured["timeout"] == cg._TIMEOUT == 30
        # the edge 403s the default python UA: a browser UA goes out on every call
        assert captured["headers"]["User-Agent"] == cg._UA
        assert cg._UA.startswith("Mozilla/5.0") and "python" not in cg._UA.lower()
        assert captured["headers"]["Accept"] == "application/json"
        # the foreign host rides the SOCKS5 proxy exactly as proxy_map() resolves it
        assert captured["proxies"] == {"https": "socks5h://127.0.0.1:1080"}

    def test_r17_coingecko_vendor_transport_failure_maps_to_no_market_data(self, monkeypatch) -> None:
        # gap _http_get_json 97-98
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))

        def boom(url, **kwargs):
            raise requests.exceptions.ConnectTimeout("timed out after 30s")

        monkeypatch.setattr("requests.get", boom)
        url = f"{cg._BASE_URL}/search"
        with pytest.raises(NoMarketDataError, match="CoinGecko request failed: ConnectTimeout") as excinfo:
            cg._http_get_json(url, params={"query": "BTC"})
        assert excinfo.value.symbol == url
        assert isinstance(excinfo.value.__cause__, requests.exceptions.ConnectTimeout)

    def test_r17_coingecko_vendor_403_is_typed_no_data_and_still_sends_browser_ua(self, monkeypatch) -> None:
        # gap _http_get_json 101-103
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
        captured: dict = {}

        def fake_get(url, **kwargs):
            captured.update(kwargs)
            return _g3_http_response(403, b'{"status":{"error_code":1002,"error_message":"Forbidden"}}')

        monkeypatch.setattr("requests.get", fake_get)
        with pytest.raises(NoMarketDataError, match=r"CoinGecko HTTP 403: .*Forbidden") as excinfo:
            cg._http_get_json(f"{cg._BASE_URL}/coins/bitcoin")
        assert excinfo.value.symbol == f"{cg._BASE_URL}/coins/bitcoin"
        assert captured["headers"]["User-Agent"] == cg._UA

    def test_r17_coingecko_vendor_5xx_body_is_a_160_char_single_line_excerpt(self, monkeypatch) -> None:
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
        body = "<html>\n" + "x" * 300 + "\n</html>"
        monkeypatch.setattr("requests.get", lambda url, **kw: _g3_http_response(503, body.encode()))
        with pytest.raises(NoMarketDataError) as excinfo:
            cg._http_get_json(f"{cg._BASE_URL}/coins/bitcoin")
        detail = excinfo.value.detail
        assert detail.startswith("CoinGecko HTTP 503: <html> ")
        assert "\n" not in detail
        assert len(detail.removeprefix("CoinGecko HTTP 503: ")) == 160

    def test_r17_coingecko_vendor_non_json_body_maps_to_no_market_data(self, monkeypatch) -> None:
        # gap _http_get_json 106-109: a real Response whose body is HTML — the
        # library's own JSONDecodeError (a ValueError) is what gets caught
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
        monkeypatch.setattr(
            "requests.get", lambda url, **kw: _g3_http_response(200, b"<html>cloudflare</html>"),
        )
        with pytest.raises(NoMarketDataError, match="CoinGecko returned non-JSON: Expecting value") as excinfo:
            cg._http_get_json(f"{cg._BASE_URL}/search")
        assert type(excinfo.value.__cause__).__name__ == "JSONDecodeError"
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_r17_coingecko_vendor_429_with_retry_after_raises_without_sleeping(self, monkeypatch) -> None:
        """Pinned by design (reported, not fixed): a 429 maps to
        VendorRateLimitError immediately — Retry-After is not read and nothing
        sleeps; the router degrades and MinIntervalThrottle spaces the calls."""
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(4.0))
        sleeps: list = []
        monkeypatch.setattr(cg.time, "sleep", sleeps.append)
        resp = _g3_http_response(429, b'{"status":{"error_code":429}}')
        resp.headers["Retry-After"] = "37"
        monkeypatch.setattr("requests.get", lambda url, **kw: resp)
        url = f"{cg._BASE_URL}/search"
        with pytest.raises(VendorRateLimitError) as excinfo:
            cg._http_get_json(url)
        assert excinfo.value.args == (f"CoinGecko returned 429 for {url}",)
        assert sleeps == []

    @pytest.mark.parametrize("base_exc", [KeyboardInterrupt, SystemExit])
    def test_r17_coingecko_vendor_base_exceptions_are_not_wrapped(self, monkeypatch, base_exc) -> None:
        # the transport handler is `except Exception`: BaseException passes through
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))

        def boom(url, **kwargs):
            raise base_exc

        monkeypatch.setattr("requests.get", boom)
        with pytest.raises(base_exc):
            cg._http_get_json(f"{cg._BASE_URL}/search")


class TestR17G3CoinGeckoCache:
    def test_r17_coingecko_vendor_cached_json_none_from_helper_raises(self, monkeypatch) -> None:
        # gap _cached_json 126: the helper's fail-open None (never requested by
        # this vendor) must not flow on as a payload
        monkeypatch.setattr(cg, "cached_or_fetch", lambda *args, **kwargs: None)
        with pytest.raises(NoMarketDataError, match="cache helper returned None") as excinfo:
            cg._cached_json("coin_bitcoin.json", lambda: {}, 1.0)
        assert excinfo.value.symbol == "coin_bitcoin.json"

    def test_r17_coingecko_vendor_cached_json_mtime_unavailable_reparses_disk(self, monkeypatch) -> None:
        # gap _cached_json 130-131: getmtime OSError -> the -1.0 sentinel. The
        # disk bytes are still parsed, the fetch is NOT re-run, and the memo
        # never serves on the sentinel (every call re-parses).
        calls: list = []

        def fetch():
            calls.append(1)
            return {"v": 7}

        real_getmtime = os.path.getmtime

        def raising_getmtime(path):
            if os.path.basename(os.fspath(path)) == "r17_mtime_oserror.json":
                raise OSError("stat unavailable")
            return real_getmtime(path)

        monkeypatch.setattr(os.path, "getmtime", raising_getmtime)
        first = cg._cached_json("r17_mtime_oserror.json", fetch, ttl_days=1.0)
        second = cg._cached_json("r17_mtime_oserror.json", fetch, ttl_days=1.0)
        assert first == second == {"v": 7}
        assert len(calls) == 1
        assert first is not second

    def test_r17_coingecko_vendor_cached_json_memo_hit_and_reparse_on_rewrite(self, monkeypatch) -> None:
        loads: list = []
        real_loads = json.loads

        def counting_loads(raw, *args, **kwargs):
            loads.append(raw)
            return real_loads(raw, *args, **kwargs)

        monkeypatch.setattr(cg.json, "loads", counting_loads)
        first = cg._cached_json("r17_memo.json", lambda: {"v": 1}, ttl_days=1.0)
        second = cg._cached_json("r17_memo.json", lambda: {"v": 1}, ttl_days=1.0)
        assert second is first  # same bytes, same mtime: served from the memo
        assert len(loads) == 1

        path = Path(cg._cache_dir()) / "r17_memo.json"
        path.write_bytes(b'{"v": 2}')
        future = time.time() + 10  # fresh by age, new mtime
        os.utime(path, (future, future))
        third = cg._cached_json("r17_memo.json", lambda: {"v": 1}, ttl_days=1.0)
        assert third == {"v": 2} and third is not first
        assert len(loads) == 2

    @pytest.mark.parametrize("mod", [cg, dl], ids=["coingecko", "defillama"])
    def test_r17_vendor_cached_json_ttl_fresh_hit_and_expiry(self, mod, monkeypatch) -> None:
        calls: list = []

        def fetch():
            calls.append(1)
            return {"n": len(calls)}

        assert mod._cached_json("r17_ttl.json", fetch, ttl_days=1.0) == {"n": 1}
        # Give the fresh entry a distinctive mtime BEFORE the memo can key on
        # it: two writes inside one Windows clock tick share a timestamp, so a
        # same-tick rewrite would otherwise be masked by the mtime-keyed memo
        # (reported; unreachable in production where rewrites are a TTL apart).
        path = Path(mod._cache_dir()) / "r17_ttl.json"
        distinctive = time.time() - 10  # ten seconds old: fresh for a one-day TTL
        os.utime(path, (distinctive, distinctive))
        assert mod._cached_json("r17_ttl.json", fetch, ttl_days=1.0) == {"n": 1}
        assert len(calls) == 1  # fresh disk hit: no second fetch

        old = time.time() - 2 * 86_400  # two days old, past the one-day TTL
        os.utime(path, (old, old))
        assert mod._cached_json("r17_ttl.json", fetch, ttl_days=1.0) == {"n": 2}
        assert len(calls) == 2

    @pytest.mark.parametrize("mod", [cg, dl], ids=["coingecko", "defillama"])
    def test_r17_vendor_cached_json_corrupt_fresh_file_is_not_refetched(self, mod, monkeypatch) -> None:
        """Pinned by design (reported, not fixed): a fresh cache file that is
        not JSON raises a bare JSONDecodeError (a ValueError, not the vendor's
        NoMarketDataError) and is NOT refetched — the entry stays poisoned
        until its TTL expires. The product's own writes cannot produce such a
        file (atomic replace of json.dumps output); it takes outside damage."""
        (Path(mod._cache_dir()) / "r17_corrupt.json").write_bytes(b'{"prices": [1, 2')

        def never():
            raise AssertionError("a fresh cache file must not be refetched")

        with pytest.raises(ValueError, match="Expecting ',' delimiter"):
            mod._cached_json("r17_corrupt.json", never, ttl_days=1.0)

    def test_r17_coingecko_vendor_fetch_failure_serves_stale_cache_ledgered(self, monkeypatch) -> None:
        set_config({"data_cache_max_stale_days": 30})
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
        monkeypatch.setattr(cg, "_http_get_json", lambda url, params=None: {"v": "fresh"})
        assert cg._cached_json("coin_x.json", lambda: cg._http_get_json("u"), ttl_days=1.0) == {"v": "fresh"}
        path = Path(cg._cache_dir()) / "coin_x.json"
        old = time.time() - 2 * 86_400  # past the TTL, far inside the 30-day cap
        os.utime(path, (old, old))

        def down(url, params=None):
            raise NoMarketDataError(url, detail="proxy blip")

        monkeypatch.setattr(cg, "_http_get_json", down)
        assert cg._cached_json("coin_x.json", lambda: cg._http_get_json("u"), ttl_days=1.0) == {"v": "fresh"}
        # the only "stale" label is the ledger sentinel the cache helper records
        assert [(e["method"], e["kind"]) for e in quality.snapshot_quality()] == [
            ("coingecko/coin_x.json", quality.KIND_STALE_CACHE),
        ]

        ancient = time.time() - 40 * 86_400  # beyond the stale cap
        os.utime(path, (ancient, ancient))
        with pytest.raises(NoMarketDataError, match="proxy blip"):
            cg._cached_json("coin_x.json", lambda: cg._http_get_json("u"), ttl_days=1.0)

    def test_r17_coingecko_vendor_cache_keys_carry_endpoint_and_symbol(self, monkeypatch) -> None:
        # one file per endpoint and per symbol: the snapshot, the chart and
        # each /search query never share an entry
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
        queries: list = []
        dispatch = _g3_cg_fake(_g3_cg_chart(_g3_pit_points), {"id": "bitcoin"})

        def fake(url, params=None):
            if "/search" in url:
                queries.append(params["query"])
            return dispatch(url, params)

        monkeypatch.setattr(cg, "_http_get_json", fake)

        def names() -> set:
            return {p.name for p in Path(cg._cache_dir()).iterdir()}

        cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026-06-27")  # historical
        assert names() == {"search_btc.json", "mchart_bitcoin.json"}
        cg.get_crypto_token_fundamentals("BTC-USD", curr_date="")  # live, same base
        assert names() == {"search_btc.json", "mchart_bitcoin.json", "coin_bitcoin.json"}
        assert queries == ["BTC"]  # the shared base reuses its one search entry
        # the wrapper ladder: each candidate query is its own cache entry
        assert cg._resolve_coin_id("1000SATSUSDT") is None
        assert queries == ["BTC", "1000SATS", "SATS"]
        assert names() == {
            "search_btc.json", "mchart_bitcoin.json", "coin_bitcoin.json",
            "search_1000sats.json", "search_sats.json",
        }


class TestR17G3CoinGeckoLiveGate:
    @pytest.mark.parametrize(
        ("label", "snapshot", "chart"),
        [
            ("2026-11-25", True, False),  # today (both live anchors): the snapshot
            ("2026-11-24", False, True),  # yesterday: the PIT daily series
            ("", True, False),  # no as-of constraint: live
        ],
    )
    def test_r17_coingecko_vendor_live_versus_historical_endpoint_choice(
        self, monkeypatch, label, snapshot, chart,
    ) -> None:
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
        monkeypatch.setattr(cg, "_cached_json", lambda filename, fetch, ttl_days: fetch())
        urls: list = []
        dispatch = _g3_cg_fake(_g3_cg_chart(_g3_pit_points), {"id": "bitcoin", "market_cap_rank": 1})

        def spy(url, params=None):
            urls.append(url)
            return dispatch(url, params)

        monkeypatch.setattr(cg, "_http_get_json", spy)
        monkeypatch.setattr(df_utils, "datetime", _g3_FrozenDatetime)
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date=label)
        assert any("/market_chart/" in u for u in urls) is chart
        assert any(u.endswith("/coins/bitcoin") for u in urls) is snapshot
        assert ("CURRENT SNAPSHOT" in out) is snapshot

    def test_r17_coingecko_vendor_live_fdv_fallbacks_and_boolean_change(self) -> None:
        """Pinned by design (reported, not fixed): FDV falls back max -> total
        -> circulating supply while the label names only "max/total" (with
        circulating alone it equals the market cap), and a JSON ``true`` in a
        price_change field is an int that renders as +1.0%."""
        market_data = {
            "current_price": {"usd": 2.0}, "market_cap": {"usd": 2.0e6}, "total_volume": {"usd": 1.0},
            "circulating_supply": 1.0e6, "total_supply": None, "max_supply": None,
            "price_change_percentage_24h": True, "price_change_percentage_7d": -2.0,
        }
        out = cg._render_live("x", "XYZ", {"id": "x", "market_data": market_data})
        assert "- Market cap: $2.00M" in out
        assert "- FDV (derived price x max/total supply): $2.00M" in out
        assert "- Price change: 24h: +1.0% | 7d: -2.0%" in out

        market_data["max_supply"] = 1.0e7
        out = cg._render_live("x", "XYZ", {"id": "x", "market_data": market_data})
        assert "- FDV (derived price x max/total supply): $20.00M" in out




class TestR17G3DefillamaFailSoft:
    def test_r17_defillama_vendor_untyped_protocol_cache_error_keeps_chain_leg(self, monkeypatch) -> None:
        # per-leg fail-soft with an UNTYPED error: a corrupt protocols cache
        # (a bare JSONDecodeError out of _cached_json) must not discard the
        # chain leg's evidence, and the failure is named in-band
        (Path(dl._cache_dir()) / "protocols_projection.json").write_bytes(b"{corrupt")

        def fake(url):
            if "/historicalChainTvl/Bitcoin" in url:
                return _g3_dl_points
            raise AssertionError(url)

        monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))
        monkeypatch.setattr(dl, "_http_get_json", fake)
        out = dl.get_crypto_network_fundamentals("BTCUSDT", curr_date="2026-09-01", look_back_days=2)
        assert "Chain TVL — Bitcoin" in out
        assert "| 2026-09-01 | $3.0000 |" in out
        assert "Unretrievable sections" in out
        assert "JSONDecodeError" in out

    def test_r17_defillama_vendor_render_series_negative_change_sign(self) -> None:
        rows = [("2026-01-01", 4.0e8), ("2026-01-02", 2.0e8), ("2026-01-03", 1.0e8)]
        out = dl._render_series("Chain TVL — X", "TVL", rows)
        assert "Window summary: $400.00M -> $100.00M (-75.0%); window peak $400.00M on 2026-01-01." in out


class TestR17G3CoinGeckoFutureLabel:
    def test_r17_coingecko_vendor_future_label_gets_neither_snapshot_nor_rows(self, monkeypatch) -> None:
        # a future label is "historical" for the gate: the current snapshot is
        # never requested, and only rows at or before the label can render
        monkeypatch.setattr(cg, "_throttler", MinIntervalThrottle(0.0))
        monkeypatch.setattr(cg, "_cached_json", lambda filename, fetch, ttl_days: fetch())
        urls: list = []
        dispatch = _g3_cg_fake(_g3_cg_chart(_g3_pit_points), {"id": "bitcoin", "market_cap_rank": 1})

        def spy(url, params=None):
            urls.append(url)
            return dispatch(url, params)

        monkeypatch.setattr(cg, "_http_get_json", spy)
        monkeypatch.setattr(df_utils, "datetime", _g3_FrozenDatetime)
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026-12-01", look_back_days=10)
        assert not any(u.endswith("/coins/bitcoin") for u in urls)
        assert "historical market data is NOT available" in out  # no cached row is that recent
        assert "CURRENT SNAPSHOT" not in out
        # a look-back that reaches the cached rows does show them — they are
        # the latest rows at or before the label — and still no snapshot
        out = cg.get_crypto_token_fundamentals("BTCUSDT", curr_date="2026-12-01", look_back_days=180)
        assert "| 2026-06-27 | 120.00 |" in out
        assert "point-in-time series as of 2026-12-01" in out
        assert "CURRENT SNAPSHOT" not in out
        assert not any(u.endswith("/coins/bitcoin") for u in urls)


class TestR17G3OnchainContract:
    def test_r17_onchain_flows_nan_point_is_kept_and_poisons_the_window_mean(self, monkeypatch) -> None:
        """Pinned by design (reported, not fixed): ``float('nan')`` passes as
        a number, so a NaN chart point is kept — the latest value survives but
        the window mean becomes NaN and renders as "nan"."""
        day0 = datetime(2026, 11, 18, tzinfo=UTC)
        payload = {
            "values": [
                {"x": int((day0 + timedelta(days=i)).timestamp()), "y": v}
                for i, v in enumerate((1.0, float("nan"), 3.0))
            ],
        }
        monkeypatch.setattr(
            ocf, "_fetch_chart_cached", lambda chart: (payload, "2026-11-25T00:00:00+00:00"),
        )
        flows = ocf.fetch_onchain_flows("2026-11-25")
        assert flows is not None
        for entry in flows.series.values():
            assert entry[1] == 3.0 and math.isnan(entry[2]) and entry[3] == 2
        block = ocf.render_onchain_block(flows)
        assert block.count("3 latest (2026-11-20; 2d mean nan;") == 2

    def test_r17_onchain_flows_dual_form_as_of_raises_for_direct_callers(self, monkeypatch) -> None:
        """Pinned by design (reported, not fixed): the as-of parse is a bare
        ``strptime("%Y-%m-%d")``. A time-stamped trade date
        ("YYYY-MM-DD HH:MM:SS", the live programmatic entry form) raises
        ValueError before any chart is read; the positioning analyst's
        advisory ``except`` then renders "capability absent ... vendor
        unreachable". Unreachable through the graph since round-13 normalizes
        the date once at ``propagate()`` — a direct caller still meets it."""
        monkeypatch.setattr(
            ocf, "_fetch_chart_cached", lambda chart: pytest.fail("as-of parsed after a chart read"),
        )
        with pytest.raises(ValueError, match="unconverted data remains"):
            ocf.fetch_onchain_flows("2026-11-25 07:30:00")




class TestR17G3MultiplierCollisions:
    def test_r17_coingecko_vendor_resolve_bare_symbol_wins_over_unwrapped(self, monkeypatch) -> None:
        # a token whose own symbol starts with the wrapper prefix: the bare
        # symbol is probed first and wins when CoinGecko lists it
        queries: list = []

        def fake(url, params=None):
            queries.append(params["query"])
            return {"coins": [
                {"id": "abc", "symbol": "ABC", "market_cap_rank": 50},
                {"id": "wrapped", "symbol": "1000ABC", "market_cap_rank": 500},
            ]}

        monkeypatch.setattr(cg, "_cached_json", lambda filename, fetch, ttl_days: fetch())
        monkeypatch.setattr(cg, "_http_get_json", fake)
        assert cg._resolve_coin_id("1000ABCUSDT") == ("wrapped", "1000ABC")
        assert queries == ["1000ABC"]  # the un-prefixed candidate is never asked

    def test_r17_coingecko_vendor_resolve_1000sats_falls_back_to_sats(self, monkeypatch) -> None:
        queries: list = []

        def fake(url, params=None):
            queries.append(params["query"])
            # the listing is "sats"; nothing carries the wrapper symbol
            if params["query"] == "SATS":
                return {"coins": [{"id": "sats", "symbol": "sats", "market_cap_rank": 300}]}
            return {"coins": []}

        monkeypatch.setattr(cg, "_cached_json", lambda filename, fetch, ttl_days: fetch())
        monkeypatch.setattr(cg, "_http_get_json", fake)
        assert cg._resolve_coin_id("1000SATSUSDT") == ("sats", "SATS")
        assert queries == ["1000SATS", "SATS"]  # wrapper first, then the bare token
        assert cg._resolve_coin_id("SATSUSDT") == ("sats", "SATS")
        assert queries == ["1000SATS", "SATS", "SATS"]

    def test_r17_defillama_vendor_wrapper_prefix_stripped_before_symbol_match(self, monkeypatch) -> None:
        # pinned by design: DefiLlama has no bare-first ladder — the wrapper
        # prefix always comes off when >= 3 characters remain, so a catalog
        # entry carrying the wrapper symbol itself can never be matched there
        bases: list = []
        monkeypatch.setattr(dl, "_resolve_protocol", lambda base: bases.append(base) and None)
        monkeypatch.setattr(dl, "_chain_for_base", lambda base: None)
        for ticker in ("1000ABCUSDT", "1000SATSUSDT", "1000XUSDT"):
            with pytest.raises(NoMarketDataError, match="no DefiLlama coverage"):
                dl.get_crypto_network_fundamentals(ticker, curr_date="2026-09-02")
        assert bases == ["ABC", "SATS", "1000X"]


class TestR17G3DefillamaLiveGate:
    def test_r17_defillama_vendor_todays_label_is_live_and_yesterday_is_historical(self, monkeypatch) -> None:
        # the current-only chain ranking follows the live gate: today's explicit
        # label keeps it, yesterday's label (a PIT view) drops it
        monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))
        monkeypatch.setattr(dl, "_cached_json", lambda filename, fetch, ttl_days: fetch())

        def fake(url):
            if url.endswith("/protocols"):
                return []
            if url.endswith("/v2/chains"):
                return [{"name": "Bitcoin", "tvl": 5.0e10}, {"name": "Ethereum", "tvl": 6.0e10}]
            if "/historicalChainTvl/Bitcoin" in url:
                return _g3_dl_points
            raise AssertionError(url)

        monkeypatch.setattr(dl, "_http_get_json", fake)
        monkeypatch.setattr(df_utils, "datetime", _g3_FrozenDatetime)
        live = dl.get_crypto_network_fundamentals("BTCUSDT", curr_date="2026-11-25")
        assert "Crypto network fundamentals — BTC (as-of 2026-11-25)." in live
        assert "ranks #2 of 2 DefiLlama-tracked chains with $50.00B TVL." in live
        assert "Chain ranking: omitted" not in live
        hist = dl.get_crypto_network_fundamentals("BTCUSDT", curr_date="2026-11-24")
        assert "Chain ranking: omitted on historical runs" in hist
        assert "ranks #" not in hist
        assert "| 2026-09-01 | $3.0000 |" in hist


# --------------------------------------------------------------------------- #
# sec_edgar._cached_or_fetch
# --------------------------------------------------------------------------- #
def test_r17_sec_edgar_cached_or_fetch_none_from_helper_is_typed_error(tmp_path, monkeypatch):
    """The shared cache helper only returns ``None`` under ``fail_open``, which
    sec_edgar never sets; if that contract ever broke the adapter must surface
    the vendor's typed ``NoMarketDataError`` for the URL — never hand ``None``
    to ``json.loads`` (a ``TypeError`` the router could not classify)."""
    seen: list[tuple] = []
    fetched: list[str] = []
    monkeypatch.setattr(
        sec_edgar, "cached_or_fetch",
        lambda *args, **kwargs: seen.append((args, kwargs)) or None,
    )
    monkeypatch.setattr(sec_edgar, "_sec_get", lambda url: fetched.append(url) or b"{}")
    path = tmp_path / "sec" / "tickers.json"
    with pytest.raises(NoMarketDataError) as excinfo:
        sec_edgar._cached_or_fetch(str(path), sec_edgar._TICKERS_URL, ttl_days=7.0)
    assert excinfo.value.detail == "cache helper returned None"
    assert excinfo.value.symbol == sec_edgar._TICKERS_URL
    # The adapter splits the full path into base dir + filename for the
    # containment-checked shared helper and never asks it to fail open.
    ((args, kwargs),) = seen
    assert args[:2] == (str(tmp_path / "sec"), "tickers.json")
    args[2]()  # the adapter's fetch thunk hits the SEC GET with the same URL
    assert fetched == [sec_edgar._TICKERS_URL]
    assert kwargs == {"ttl_days": 7.0, "vendor": "sec_edgar"}


# --------------------------------------------------------------------------- #
# fred._request
# --------------------------------------------------------------------------- #
def test_r17_fred_request_none_from_helper_is_runtime_error(tmp_path, monkeypatch):
    """``fail_open`` is never set for fred either; a ``None`` from the helper
    is a broken contract and becomes a loud ``RuntimeError`` instead of a
    ``TypeError`` deep inside ``json.loads``."""
    monkeypatch.setenv("FRED_API_KEY", "dummy-fred-key")
    _g4_fred_cache_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(fred, "cached_or_fetch", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match=r"fred: cache helper returned None"):
        fred._request("series", {"series_id": "CPIAUCSL"})


def test_r17_fred_request_cache_key_covers_params_and_ignores_api_key(tmp_path, monkeypatch):
    """The cache entry is a digest of endpoint + params WITHOUT the API key:
    a different window is a different entry, and rotating the key does not
    orphan or cross-serve a cached response."""
    calls: list[tuple] = []

    def fake_cached_or_fetch(base, filename, fetch, *, ttl_days, vendor):
        calls.append((base, filename, ttl_days, vendor))
        return b'{"seriess": []}'

    monkeypatch.setattr(fred, "cached_or_fetch", fake_cached_or_fetch)
    _g4_fred_cache_dir(tmp_path, monkeypatch)
    window_jan = {
        "series_id": "CPIAUCSL", "observation_start": "2025-01-01",
        "observation_end": "2025-06-30", "sort_order": "asc",
    }
    window_feb = {**window_jan, "observation_start": "2025-02-01"}
    monkeypatch.setenv("FRED_API_KEY", "key-one")
    fred._request("series/observations", window_jan)
    fred._request("series/observations", window_feb)
    monkeypatch.setenv("FRED_API_KEY", "key-two")
    fred._request("series/observations", window_feb)
    assert calls[0][0] == str(tmp_path / "fred")
    assert [c[2:] for c in calls] == [(1.0, "fred")] * 3
    assert calls[0][1] != calls[1][1]  # a different window is a different entry
    assert calls[2][1] == calls[1][1]  # the key never enters the filename
    blob = json.dumps(
        {"path": "series/observations", "params": window_jan},
        sort_keys=True, default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:12]
    assert calls[0][1] == f"series_observations_{digest}.json"


# --------------------------------------------------------------------------- #
# interface.route_to_vendor
# --------------------------------------------------------------------------- #
def test_r17_interface_route_category_tool_without_vendor_entry_is_refused():
    """``get_binance_indicators`` is listed under ``binance_perp`` but has no
    ``VENDOR_METHODS`` entry (it is bound directly, never routed): the router
    refuses it by name before any vendor runs."""
    with pytest.raises(ValueError, match="Method 'get_binance_indicators' not supported"):
        interface.route_to_vendor("get_binance_indicators", "BTCUSDT", "2026-01-01", "2026-01-02")


def test_r17_interface_route_empty_vendor_table_is_a_runtime_error(monkeypatch):
    """A method whose vendor table is empty has an empty chain: no vendor ran,
    so neither the no-data nor the error branch applies — the loop's exit is
    the explicit ``RuntimeError`` naming the method."""
    monkeypatch.setattr(interface, "get_vendor", lambda category, method=None: "default")
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_macro_indicators", {})
    with pytest.raises(RuntimeError, match="No available vendor for 'get_macro_indicators'"):
        interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")


# --------------------------------------------------------------------------- #
# binance_vision_store.VisionStore
# --------------------------------------------------------------------------- #
def test_r17_binance_vision_store_read_manifest_wrong_shape_starts_fresh(tmp_path):
    """A manifest that is not an object, whose ``days`` / ``missing`` are not
    objects, or that was truncated mid-write is treated as absent: the store
    starts fresh rather than indexing into a list or iterating a string."""
    store = bvs.VisionStore(tmp_path)
    path = store._manifest_path("metrics", "BTCUSDT")
    path.parent.mkdir(parents=True)
    fresh = {"days": {}, "missing": {}, "updated": None}
    cases = [
        ("[]", "a JSON list"),
        ('{"days": [], "missing": {}}', "days is a list"),
        ('{"days": {"2024-01-01": 288}, "missing": "2024-01-02"}', "missing is a string"),
        ('{"days": {"2024-01-01": 288', "a truncated write"),
    ]
    for raw, why in cases:
        path.write_text(raw, encoding="utf-8")
        assert store._read_manifest("metrics", "BTCUSDT") == fresh, why
    # A well-formed manifest round-trips exactly (counts, missing map, stamp).
    good = {
        "days": {"2024-01-01": 288},
        "missing": {"2024-01-02": "2026-01-02T00:00:00+00:00"},
        "updated": "2026-01-03T00:00:00+00:00",
    }
    path.write_text(json.dumps(good), encoding="utf-8")
    assert store._read_manifest("metrics", "BTCUSDT") == good


def test_r17_binance_vision_store_normalize_without_time_column_is_typed_error():
    """An archive CSV with none of the known time columns is a typed
    ``NoMarketDataError`` naming the archive and the columns it did have —
    never an empty frame that would read as a quiet day."""
    frame = pd.DataFrame({"symbol": ["BTCUSDT"], "depth": [1.5]})
    with pytest.raises(NoMarketDataError) as excinfo:
        bvs.VisionStore._normalize(frame, "bookDepth-2024-01-01.zip")
    assert excinfo.value.detail == (
        "bookDepth-2024-01-01.zip archive CSV has no time column "
        "(columns: ['symbol', 'depth'])"
    )


def test_r17_binance_vision_store_merge_year_nothing_to_merge_writes_nothing(tmp_path):
    """With no partition on disk and no new frames the merge is a no-op: no
    partition, no directory, and the manifest object is left untouched."""
    store = bvs.VisionStore(tmp_path)
    manifest = {"days": {}, "missing": {}}
    store._merge_year("metrics", "BTCUSDT", 2024, [], manifest, [])
    assert not store._dir("metrics", "BTCUSDT").exists()
    assert manifest == {"days": {}, "missing": {}}


# --------------------------------------------------------------------------- #
# disk_cache.cache_file_path — the containment branches, with no symlink privilege
# --------------------------------------------------------------------------- #
class _G4WindowsPaths:
    """``os.path`` with Windows drive semantics on every OS.

    ``join`` / ``commonpath`` are the real ``ntpath`` functions; ``realpath``
    resolves the ``{link: target}`` table it was built with (what a symlink
    resolves to) and otherwise only normalizes.
    """

    join = staticmethod(ntpath.join)
    commonpath = staticmethod(ntpath.commonpath)

    def __init__(self, links: dict[str, str]) -> None:
        self._links = {ntpath.normpath(k): ntpath.normpath(v) for k, v in links.items()}

    def realpath(self, path: str) -> str:
        return self._links.get(ntpath.normpath(path), ntpath.normpath(path))


def test_r17_disk_cache_path_escaping_to_another_drive_is_refused(monkeypatch):
    """The branch the round-16 symlink test can never reach: a cache entry
    whose realpath lands on ANOTHER drive makes ``os.path.commonpath`` raise
    ValueError. The handler must treat that as an escape and fail closed,
    exactly like the same-drive case. Driven through an ``ntpath``-backed
    ``os`` stand-in, so it runs on POSIX CI and needs no symlink privilege."""
    base = "C:/cache"
    links = {"C:/cache/escape.json": "D:/vault/secret.json"}
    monkeypatch.setattr(dc, "os", types.SimpleNamespace(path=_G4WindowsPaths(links)))
    # The precondition, measured: the real ntpath function refuses a drive mix.
    with pytest.raises(ValueError, match="same drive"):
        ntpath.commonpath(["D:\\vault\\secret.json", "C:\\cache"])
    with pytest.raises(ValueError, match="resolves outside the cache directory") as excinfo:
        dc.cache_file_path(base, "escape.json")
    assert str(excinfo.value) == (
        "cache path 'escape.json' resolves outside the cache directory 'C:\\\\cache'"
    )
    # An entry that stays on the drive and inside the directory still passes.
    assert dc.cache_file_path(base, "ok.json") == Path(base) / "ok.json"


def test_r17_disk_cache_same_drive_escape_refuses_before_any_fetch(monkeypatch):
    """An entry resolving outside the base directory on the SAME drive fails
    the containment check too, and ``cached_or_fetch`` refuses it before any
    fetch runs; a link that resolves inside the directory is a valid entry."""
    base = "C:/cache"
    links = {
        "C:/cache/escape.json": "C:/vault/secret.json",
        "C:/cache/inside.json": "C:/cache/real.json",
    }
    monkeypatch.setattr(dc, "os", types.SimpleNamespace(path=_G4WindowsPaths(links)))
    fetch_calls: list[int] = []
    with pytest.raises(ValueError, match="cache path 'escape.json' resolves outside"):
        dc.cached_or_fetch(
            base, "escape.json", lambda: fetch_calls.append(1) or b"x",
            ttl_days=1.0, vendor="t",
        )
    assert fetch_calls == []
    assert dc.cache_file_path(base, "inside.json") == Path(base) / "inside.json"


# --------------------------------------------------------------------------- #
# utils.replace_retry_on_windows
# --------------------------------------------------------------------------- #
def test_r17_utils_replace_retry_budget_is_half_spent_on_attempts(tmp_path, monkeypatch):
    """Pins the exhaustion schedule (report-only). The five attempts fall at
    0 / 0.05 / 0.10 / 0.20 / 0.40 s, then the loop sleeps its last 0.4 s step
    with NO attempt after it. The round-10 comment budgets "~0.8s total" for a
    holder of the target, but a reader holding it longer than 0.40 s defeats
    the writer, which then spends as long again sleeping for nothing."""
    events: list[tuple[str, float]] = []
    clock = {"t": 0.0}

    def fake_sleep(seconds: float) -> None:
        events.append(("sleep", seconds))
        clock["t"] += seconds

    def denied(_src, _dst) -> None:
        events.append(("try", clock["t"]))
        raise PermissionError(32, "target open in a reader")

    monkeypatch.setattr(df_utils.time, "sleep", fake_sleep)
    monkeypatch.setattr(df_utils.os, "replace", denied)
    (tmp_path / "a.json").write_text("old", encoding="utf-8")
    with pytest.raises(PermissionError, match="target open in a reader"):
        df_utils.replace_retry_on_windows(tmp_path / "a.tmp", tmp_path / "a.json")
    assert events == [
        ("try", 0.0), ("sleep", 0.05), ("try", 0.05), ("sleep", 0.05),
        ("try", 0.1), ("sleep", 0.1), ("try", 0.2), ("sleep", 0.2),
        ("try", 0.4), ("sleep", 0.4),
    ]
    assert clock["t"] == pytest.approx(0.8)
    assert events[-1][0] == "sleep"
    assert (tmp_path / "a.json").read_text(encoding="utf-8") == "old"


def test_r17_utils_replace_retry_cross_device_oserror_is_not_retried(tmp_path, monkeypatch):
    """Only the Windows reader race (PermissionError) is worth a retry. A
    cross-volume replace fails deterministically with a plain OSError (errno
    18 on POSIX, winerror 17 on Windows): it must surface at once, not burn
    the 0.8 s budget first."""
    sleeps: list[float] = []
    targets: list[Path] = []

    def wrong_volume(_src, dst):
        targets.append(dst)
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(df_utils.os, "replace", wrong_volume)
    monkeypatch.setattr(df_utils.time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(OSError, match="cross-device") as excinfo:
        df_utils.replace_retry_on_windows(tmp_path / "a.tmp", tmp_path / "a.json")
    assert type(excinfo.value) is OSError
    assert targets == [tmp_path / "a.json"]
    assert sleeps == []


# --------------------------------------------------------------------------- #
# utils.clamp_end_date
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("end_date", "expected"),
    [
        ("2026-06-27", "2026-06-27"),  # on the analysis date: inclusive
        ("2026-06-27 15:30:00", "2026-06-27 15:30:00"),  # datetime label on the pin
        ("2026-06-28", "2026-06-27"),  # the day after: clamped
        # An end date the clamp cannot read is handed back EXACTLY as asked: the
        # documented fail-open (the vendor's own parser decides), pinned by
        # tests/test_market_data_date_clamp.py.
        ("20260628", "20260628"),  # basic ISO, the day after the pin
        ("20261125", "20261125"),  # basic ISO, months later
        ("2026/06/28", "2026/06/28"),  # slash separator, later
        ("latest", "latest"),  # a keyword echo
        ("garbage", "garbage"),
    ],
)
def test_r17_utils_clamp_end_date_with_pinned_analysis_date(end_date, expected):
    """With an analysis date pinned, an end date the clamp can read never ends
    after it. One it cannot read passes through unchanged, so the third PIT
    guard holds for well-formed input only -- see
    ``test_r17_utils_compact_echo_reaches_the_vendor_unchanged`` for what a
    lenient parser then makes of such text. Pinned as is: capping unreadable
    text flips an explicit pin, which is a design decision (round-17 ledger)."""
    assert df_utils.clamp_end_date(end_date, "2026-06-27") == expected


def test_r17_utils_clamp_unreadable_analysis_date_and_live_mode_still_pass_through():
    """The documented fail-open cases: live mode (no analysis date) and a
    pinned analysis date that is not a ``YYYY-MM-DD`` calendar date (there is
    no as-of date to clamp to). The analysis date itself is read as
    ``YYYY-MM-DD`` only: its basic-ISO and slash forms leave the clamp a no-op."""
    assert df_utils.clamp_end_date("garbage", None) == "garbage"
    assert df_utils.clamp_end_date("20261125", "") == "20261125"
    assert df_utils.clamp_end_date("2030-01-01", "not-a-date") == "2030-01-01"
    assert df_utils.clamp_end_date("2030-01-01", "20200102") == "2030-01-01"
    assert df_utils.clamp_end_date("2019-12-31", "2020/01/02") == "2019-12-31"
    assert df_utils.clamp_end_date("2030-01-01", "2020/01/02") == "2030-01-01"


def test_r17_utils_compact_echo_reaches_the_vendor_unchanged(monkeypatch):
    """Observation, pinned as is. The run is pinned to 2026-06-01 and the LLM
    echoes the window end as ``20260602``. The clamp cannot read that form, so
    ``get_binance_indicators`` hands it to ``binance_klines_frame`` as the
    window end: the guard does not apply at all. That layer parses strictly
    and refuses the text (a lost call, no leak). A lenient parser such as
    ``pd.Timestamp`` reads the same text as 2026-06-02, one day past the
    analysis date -- the leak the CoinGecko vendor had before it canonicalised
    its own label."""
    seen: dict[str, str] = {}

    def fake_frame(symbol, start, end, interval="1d", venue="binance_perp"):
        seen.update(start=start, end=end)
        raise RuntimeError("spy: not fetched in this test")

    monkeypatch.setattr(bit, "binance_klines_frame", fake_frame)
    with df_utils.pinned_analysis_date("2026-06-01"):
        out = bit._indicators_core("BTCUSDT", "20260602", 30, "perp", "")
    # Measured: a lenient parser reads the basic form one day past the pin.
    assert pd.Timestamp("20260602") == pd.Timestamp("2026-06-02")
    assert seen["end"] == "20260602"
    assert out.startswith(
        "DATA_UNAVAILABLE: Binance klines for 'BTCUSDT' (perp) up to 20260602"
    )


# --------------------------------------------------------------------------- #
# binance_vision_store.VisionStore — manifest, merge, normalize
# --------------------------------------------------------------------------- #
def _g4_metrics_zip(day: str) -> bytes:
    """A two-row 5m ``metrics`` archive (zip with one CSV member) for ``day``."""
    csv_text = (
        "create_time,symbol,sum_open_interest\n"
        f"{day} 00:05:00,BTCUSDT,1\n"
        f"{day} 00:10:00,BTCUSDT,2\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", csv_text)
    return buf.getvalue()


def _g4_day_of(url: str) -> str:
    """The ``YYYY-MM-DD`` day of a ``.../<symbol>-<dataset>-<day>.zip`` URL."""
    return url[:-4][-10:]


def test_r17_binance_vision_store_manifest_read_rides_through_sharing_violation(
    tmp_path, monkeypatch
):
    """``load()`` / ``qa()`` / ``window_row_estimate()`` read the manifest with
    no lock while another worker's sync replaces it; on Windows the reader's
    open then raises PermissionError for a moment (measured on NTFS: ~3% of
    reads under a hot replace loop). That is neither "missing" nor "corrupt",
    but the reader started fresh on it: qa() reported zero synced days and the
    bookDepth row estimate came back None, which skips the memory guard.
    Retry once, like ``_read_partition_with_retry`` does for partitions."""
    store = bvs.VisionStore(tmp_path)
    path = store._manifest_path("metrics", "BTCUSDT")
    path.parent.mkdir(parents=True)
    good = {
        "days": {"2024-01-01": 288},
        "missing": {"2024-01-02": "2026-01-02T00:00:00+00:00"},
        "updated": "2026-01-03T00:00:00+00:00",
    }
    path.write_text(json.dumps(good), encoding="utf-8")
    real_read_text = Path.read_text
    hits = {"n": 0}

    def denied_once(self, *args, **kwargs):
        if self == path:
            hits["n"] += 1
            if hits["n"] == 1:
                raise PermissionError(13, "Access is denied (replace in flight)")
        return real_read_text(self, *args, **kwargs)

    sleeps: list[float] = []
    monkeypatch.setattr(Path, "read_text", denied_once)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    assert store._read_manifest("metrics", "BTCUSDT") == good
    assert (hits["n"], sleeps) == (2, [0.2])
    # The manifest-driven row estimate (the bookDepth memory guard's input)
    # is back instead of a free pass.
    estimate = store.window_row_estimate(
        "metrics", "BTCUSDT", datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, 23, 59, 59, tzinfo=UTC),
    )
    assert estimate == 288 + 288  # one known day + one unknown at the median


def test_r17_binance_vision_store_manifest_denied_twice_still_starts_fresh(
    tmp_path, monkeypatch
):
    """A manifest that stays unreadable keeps the old contract: the store
    starts fresh rather than raising out of a lockless read."""
    store = bvs.VisionStore(tmp_path)
    path = store._manifest_path("metrics", "BTCUSDT")
    path.parent.mkdir(parents=True)
    path.write_text('{"days": {"2024-01-01": 288}, "missing": {}}', encoding="utf-8")

    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Access is denied")

    sleeps: list[float] = []
    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    assert store._read_manifest("metrics", "BTCUSDT") == {
        "days": {}, "missing": {}, "updated": None,
    }
    assert sleeps == [0.2]


def test_r17_binance_vision_store_resync_after_manifest_loss_is_idempotent(tmp_path, monkeypatch):
    """A lost or corrupt manifest makes every window day pending again: the
    days are re-parsed from the archive cache and merged OVER partitions that
    already hold them. The merge must stay idempotent (same rows, no
    duplicates) and leave no temporary file behind."""
    loads: list[str] = []
    monkeypatch.setattr(bnv, "_load_zip", lambda url: loads.append(url) or _g4_metrics_zip(_g4_day_of(url)))
    store = bvs.VisionStore(tmp_path)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, 23, 59, 59, tzinfo=UTC)
    first = store.sync("metrics", "BTCUSDT", start, end, budget=10)
    assert (first.downloaded, first.already_synced, first.missing, first.unsynced) == (3, 0, 0, 0)
    rows_first = len(store.load("metrics", "BTCUSDT", start, end))
    assert rows_first == 6
    # The manifest never landed (a replace that stayed denied, a crash, a
    # cleanup): the partitions survive and the next sync re-derives the days.
    store._manifest_path("metrics", "BTCUSDT").unlink()
    loads.clear()
    second = store.sync("metrics", "BTCUSDT", start, end, budget=10)
    assert (second.downloaded, second.already_synced) == (3, 0)
    assert sorted(loads) == sorted(bnv._archive_url("metrics", "BTCUSDT", d) for d in ("2024-01-01", "2024-01-02", "2024-01-03"))
    frame = store.load("metrics", "BTCUSDT", start, end)
    assert len(frame) == rows_first
    assert frame["_ts"].is_unique
    assert store._read_manifest("metrics", "BTCUSDT")["days"] == {
        "2024-01-01": 2, "2024-01-02": 2, "2024-01-03": 2,
    }
    # The OS lock file's fate belongs to the installed ``filelock`` (3.x removes
    # it on release, 4.x keeps ``sync.lock.lock``), not to this module: skip it.
    left = sorted(
        p.name for p in store._dir("metrics", "BTCUSDT").iterdir()
        if not p.name.startswith("sync.lock")
    )
    assert left == ["data-2024.csv.gz", "manifest.json"]  # no temporary file behind


def test_r17_binance_vision_store_manifest_publish_denied_fails_closed_and_self_heals(
    tmp_path, monkeypatch
):
    """A reader that holds ``manifest.json`` past the retry budget makes the
    sync raise (fail-closed — the router turns it into the optional-category
    sentinel). The day IS in the partition and is NOT claimed by the
    manifest; the next sync re-derives it from the archive cache and the
    merge absorbs the duplicate. The temp manifest is left behind until then
    (no cleanup on this seam, unlike disk_cache)."""
    monkeypatch.setattr(bnv, "_load_zip", lambda url: _g4_metrics_zip(_g4_day_of(url)))
    store = bvs.VisionStore(tmp_path)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, 23, 59, 59, tzinfo=UTC)
    real_replace = os.replace
    sleeps: list[float] = []

    def deny_manifest(src, dst):
        if Path(dst).name == "manifest.json":
            raise PermissionError(32, "manifest open in a reader")
        real_replace(src, dst)

    monkeypatch.setattr(df_utils.os, "replace", deny_manifest)
    monkeypatch.setattr(df_utils.time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(PermissionError, match="manifest open in a reader"):
        store.sync("metrics", "BTCUSDT", start, end, budget=5)
    assert sleeps == [0.05, 0.05, 0.1, 0.2, 0.4]
    assert len(store.load("metrics", "BTCUSDT", start, end)) == 2
    assert store._read_manifest("metrics", "BTCUSDT")["days"] == {}
    assert (store._dir("metrics", "BTCUSDT") / "manifest.json.tmp").exists()

    monkeypatch.setattr(df_utils.os, "replace", real_replace)
    loads: list[str] = []
    monkeypatch.setattr(bnv, "_load_zip", lambda url: loads.append(url) or _g4_metrics_zip(_g4_day_of(url)))
    report = store.sync("metrics", "BTCUSDT", start, end, budget=5)
    assert (report.downloaded, len(loads)) == (1, 1)
    frame = store.load("metrics", "BTCUSDT", start, end)
    assert (len(frame), frame["_ts"].is_unique) == (2, True)
    assert store._read_manifest("metrics", "BTCUSDT")["days"] == {"2024-01-01": 2}
    # The OS lock file's fate belongs to the installed ``filelock`` (3.x removes
    # it on release, 4.x keeps ``sync.lock.lock``), not to this module: skip it.
    left = sorted(
        p.name for p in store._dir("metrics", "BTCUSDT").iterdir()
        if not p.name.startswith("sync.lock")
    )
    assert left == ["data-2024.csv.gz", "manifest.json"]  # no temporary file behind


def test_r17_binance_vision_store_normalize_drops_unreadable_times_and_dedupes_bands():
    """A row whose timestamp cannot be parsed is dropped (never kept with a
    NaT time), and of two rows at the same timestamp and band — a re-published
    archive — the later one wins."""
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00", "not-a-time",
                "2024-01-01 00:00:00", "2024-01-01 00:30:00",
            ],
            "percentage": [1.0, 1.0, 1.0, -1.0],
            "depth": [10.0, 99.0, 11.0, 12.0],
        }
    )
    out = bvs.VisionStore._normalize(frame, "bookDepth-2024-01-01.zip")
    assert out["_ts"].tolist() == ["2024-01-01 00:00:00", "2024-01-01 00:30:00"]
    assert out["depth"].tolist() == [11.0, 12.0]
    assert out.index.tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# sec_edgar / fred — TTL, rate limit, timeout, stale serve
# --------------------------------------------------------------------------- #
def test_r17_sec_edgar_ticker_table_ttl_is_seven_days(tmp_path, monkeypatch):
    """``tickers.json`` is served from disk for seven days and refetched on
    day eight: no SEC request inside the TTL, exactly one after it."""
    monkeypatch.setattr(sec_edgar, "_cache_dir", lambda: str(tmp_path))
    table = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}).encode()
    path = tmp_path / "tickers.json"
    path.write_bytes(table)
    requested: list[str] = []
    monkeypatch.setattr(sec_edgar, "_sec_get", lambda url: requested.append(url) or table)
    six_days_ago = time.time() - 6 * 86_400
    os.utime(path, (six_days_ago, six_days_ago))
    assert sec_edgar._cik_for_ticker("AAPL") == 320193
    assert requested == []
    eight_days_ago = time.time() - 8 * 86_400
    os.utime(path, (eight_days_ago, eight_days_ago))
    assert sec_edgar._cik_for_ticker("aapl") == 320193
    assert requested == [sec_edgar._TICKERS_URL]


def _g4_fred_cache_dir(tmp_path, monkeypatch) -> Path:
    """Point fred's cache at ``tmp_path / "fred"``, existing before a
    write the way the real resolver makes it."""
    root = tmp_path / "fred"
    root.mkdir()
    monkeypatch.setattr(fred, "vendor_cache_dir", lambda name: str(root))
    return root


def _g4_http_response(status: int, body: bytes) -> requests.Response:
    """A real ``requests.Response`` built in memory (no request was made)."""
    response = requests.Response()
    response.status_code = status
    response.reason = "Too Many Requests" if status == 429 else "OK"
    response.url = f"{fred.FRED_API_BASE}/series"
    response._content = body
    return response


def test_r17_fred_request_rate_limit_is_not_retried_and_leaves_no_cache(tmp_path, monkeypatch):
    """A 429 is deterministic for a plain re-request: one GET with a timeout,
    no retry sleep, the HTTPError surfaces (the shared cache helper finds no
    cache and re-raises) and nothing is written."""
    monkeypatch.setenv("FRED_API_KEY", "k")
    root = _g4_fred_cache_dir(tmp_path, monkeypatch)
    calls: list[tuple] = []
    sleeps: list[float] = []

    def rate_limited(url, params=None, timeout=None):
        calls.append((params["api_key"], timeout))
        return _g4_http_response(429, b'{"error_message": "Too Many Requests"}')

    monkeypatch.setattr(fred.requests, "get", rate_limited)
    monkeypatch.setattr(netretry.time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(requests.exceptions.HTTPError, match="429"):
        fred._request("series", {"series_id": "CPIAUCSL"})
    assert calls == [("k", 30)]
    assert sleeps == []
    assert list(root.iterdir()) == []


def test_r17_fred_request_timeout_is_retried_once_then_cached(tmp_path, monkeypatch):
    """A timeout gets exactly one retry after the 2 s back-off; the result is
    cached and the next call inside the one-day TTL makes no request."""
    monkeypatch.setenv("FRED_API_KEY", "k")
    root = _g4_fred_cache_dir(tmp_path, monkeypatch)
    payload = json.dumps({"seriess": [{"id": "CPIAUCSL", "title": "Consumer Price Index"}]}).encode()
    attempts: list[str] = []
    sleeps: list[float] = []

    def flaky(url, params=None, timeout=None):
        attempts.append(url)
        if len(attempts) == 1:
            raise requests.exceptions.Timeout("read timed out")
        return _g4_http_response(200, payload)

    monkeypatch.setattr(fred.requests, "get", flaky)
    monkeypatch.setattr(netretry.time, "sleep", lambda s: sleeps.append(s))
    expected = {"seriess": [{"id": "CPIAUCSL", "title": "Consumer Price Index"}]}
    assert fred._request("series", {"series_id": "CPIAUCSL"}) == expected
    assert (len(attempts), sleeps) == (2, [2.0])
    (cached,) = root.glob("*.json")
    assert cached.read_bytes() == payload
    assert fred._request("series", {"series_id": "CPIAUCSL"}) == expected
    assert len(attempts) == 2


def test_r17_fred_request_failure_inside_cap_serves_stale_cache_labelled(
    tmp_path, monkeypatch, caplog
):
    """When the vendor is down and a cache older than the TTL exists, it is
    served with a WARNING naming the vendor, the file and its age, and a
    ``stale_cache`` sentinel — never passed off as fresh."""
    monkeypatch.setenv("FRED_API_KEY", "k")
    root = _g4_fred_cache_dir(tmp_path, monkeypatch)
    payload = b'{"seriess": [{"id": "CPIAUCSL"}]}'
    monkeypatch.setattr(
        fred.requests, "get", lambda url, params=None, timeout=None: _g4_http_response(200, payload)
    )
    params = {"series_id": "CPIAUCSL"}
    assert fred._request("series", params) == {"seriess": [{"id": "CPIAUCSL"}]}
    (cached,) = root.glob("*.json")
    two_days_ago = time.time() - 2 * 86_400
    os.utime(cached, (two_days_ago, two_days_ago))

    def dead(url, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("proxy down")

    monkeypatch.setattr(fred.requests, "get", dead)
    monkeypatch.setattr(netretry.time, "sleep", lambda s: None)
    with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.disk_cache"):
        assert fred._request("series", params) == {"seriess": [{"id": "CPIAUCSL"}]}
    (warning,) = [r for r in caplog.records if "STALE" in r.getMessage()]
    assert warning.getMessage() == (
        f"fred: fetch failed; serving STALE cache for {cached} (age 2.0 days)"
    )
    assert "stale_cache" in str(quality.snapshot_quality())


def test_r17_fred_request_without_key_raises_before_any_http(monkeypatch, _dummy_api_keys):
    """``FRED_API_KEY`` unset is the designed non-fatal degradation: the typed
    ``FredNotConfiguredError`` (a ValueError) is raised before any request or
    cache directory is touched."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    calls: list[tuple] = []
    monkeypatch.setattr(fred.requests, "get", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(fred.FredNotConfiguredError, match="FRED_API_KEY"):
        fred._request("series", {"series_id": "CPIAUCSL"})
    assert calls == []


# --------------------------------------------------------------------------- #
# netretry.with_transient_retry
# --------------------------------------------------------------------------- #
def test_r17_netretry_backoff_is_linear_and_bounded(monkeypatch, caplog):
    """Every failed attempt up to ``retries`` sleeps ``backoff_s * (attempt +
    1)`` — a straight line with a fixed ceiling — and the last failure raises
    without a further sleep."""
    sleeps: list[float] = []
    calls = {"n": 0}

    def always_down():
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("blip")

    monkeypatch.setattr(netretry.time, "sleep", lambda s: sleeps.append(s))
    with (
        caplog.at_level(logging.WARNING, logger="yialpha.dataflows.netretry"),
        pytest.raises(requests.exceptions.ConnectionError, match="blip") as excinfo,
    ):
        netretry.with_transient_retry(
            always_down, vendor="t",
            retry_on=(requests.exceptions.ConnectionError,), retries=3, backoff_s=2.0,
        )
    assert calls["n"] == 4  # one attempt + three retries
    assert sleeps == [2.0, 4.0, 6.0]
    assert type(excinfo.value) is requests.exceptions.ConnectionError
    assert [r.getMessage().rsplit("retrying ", 1)[1] for r in caplog.records] == [
        "(1/3)", "(2/3)", "(3/3)",
    ]


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit])
def test_r17_netretry_never_swallows_base_exception(monkeypatch, stop):
    """A broad ``retry_on`` still catches only ``Exception`` subclasses: a
    KeyboardInterrupt / SystemExit raised by the fetch propagates at once —
    no sleep, no retry, the very same object."""
    sleeps: list[float] = []
    monkeypatch.setattr(netretry.time, "sleep", lambda s: sleeps.append(s))
    raised = stop()
    attempts = {"n": 0}

    def interrupted():
        attempts["n"] += 1
        raise raised

    with pytest.raises(stop) as excinfo:
        netretry.with_transient_retry(
            interrupted, vendor="t", retry_on=(Exception,), retries=5,
        )
    assert excinfo.value is raised
    assert (attempts["n"], sleeps) == (1, [])


# --------------------------------------------------------------------------- #
# interface.route_to_vendor
# --------------------------------------------------------------------------- #
def test_r17_interface_route_keyboard_interrupt_propagates_without_fallback(monkeypatch):
    """The router's catch-all is ``except Exception``: a KeyboardInterrupt from
    a vendor is neither recorded as a vendor failure nor handed to the next
    vendor in the chain — it propagates untouched."""
    fallback_calls: list[tuple] = []

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    def fallback(*args, **kwargs):
        fallback_calls.append(args)
        return "data"

    monkeypatch.setattr(interface, "get_vendor", lambda category, method=None: "fred,polymarket")
    monkeypatch.setitem(
        interface.VENDOR_METHODS, "get_macro_indicators",
        {"fred": interrupted, "polymarket": fallback},
    )
    with pytest.raises(KeyboardInterrupt):
        interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")
    assert fallback_calls == []
    assert quality.snapshot_quality() == []
