"""Tests for the quantitative risk-control layer (Phase 1).

Covers ``tradingagents.risk.{kelly, atr_stop, breaker, cvar}``. All tests
are unit-level: no network, no global config, deterministic inputs.

The ATR path that fetches via the network loader (``latest_atr(symbol,
date)``) is intentionally skipped here — it is exercised by integration
tests against the real dataflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.risk.atr_stop import atr_stop, latest_atr_from_frame
from tradingagents.risk.breaker import DrawdownBreaker
from tradingagents.risk.cvar import cvar_position_multiplier, historical_cvar
from tradingagents.risk.kelly import (
    RATING_TO_BAND,
    bayesian_win_rate,
    kelly_fraction,
    kelly_sizing,
)


# ---------------------------------------------------------------------------
# kelly
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKelly:
    def test_sell_is_zero(self):
        assert kelly_sizing("Sell", win_rate=0.99) == 0.0
        assert kelly_sizing("Sell", wins=99, losses=1) == 0.0

    def test_buy_clips_to_band_with_strong_history(self):
        low, high = RATING_TO_BAND["Buy"]
        w = kelly_sizing("Buy", win_rate=0.99, avg_win=3.0, avg_loss=1.0)
        assert low <= w <= high

    def test_buy_lower_bound_enforced_when_kelly_tiny(self):
        # A weak edge would normally give Kelly near 0; the Buy band floors it.
        w = kelly_sizing("Buy", win_rate=0.51, avg_win=1.0, avg_loss=1.0)
        assert w == pytest.approx(RATING_TO_BAND["Buy"][0])

    def test_quarter_kelly_reduces_full_by_four(self):
        full = kelly_fraction(0.6, avg_win=2.0, avg_loss=1.0)
        quarter = full * 0.25
        assert quarter == pytest.approx(full / 4.0)
        # And kelly_sizing with default mult should track quarter-Kelly (modulo band clip).
        # Use Hold (0..0.03) so the clip is loose enough to see the quarter effect.
        ks = kelly_sizing("Hold", win_rate=0.9, avg_win=5.0, avg_loss=1.0)
        expected_raw = kelly_fraction(0.9 * 0.55, 5.0, 1.0) * 0.25
        assert ks == pytest.approx(min(max(expected_raw, 0.0), 0.03))

    def test_bayesian_win_rate_no_data_is_half(self):
        assert bayesian_win_rate(0, 0) == pytest.approx(0.5)

    def test_bayesian_win_rate_smooths_small_samples(self):
        # 1 win / 0 losses should not jump to 1.0 — the prior pulls it toward 0.5.
        p = bayesian_win_rate(1, 0)
        assert 0.5 < p < 1.0
        # Symmetric prior -> 2 wins / 0 losses is higher than 1 win / 0 losses.
        assert bayesian_win_rate(2, 0) > p

    def test_unknown_rating_does_not_raise(self):
        w = kelly_sizing("Strong Buy", win_rate=0.6)
        assert isinstance(w, float)
        assert 0.0 <= w <= 1.0

    def test_kelly_fraction_negative_edge_clamped_to_zero(self):
        # p too low for the payoff -> no edge -> 0.
        assert kelly_fraction(0.1, avg_win=1.0, avg_loss=1.0) == 0.0

    def test_kelly_fraction_degenerate_loss_treated_as_symmetric(self):
        # avg_loss <= 0 -> b=1.
        f = kelly_fraction(0.7, avg_win=2.0, avg_loss=0.0)
        # symmetric kelly: f = p - q = 2p - 1
        assert f == pytest.approx(2 * 0.7 - 1)


# ---------------------------------------------------------------------------
# atr_stop
# ---------------------------------------------------------------------------


def _synthetic_ohlcv(n: int = 30, vol: float = 4.0) -> pd.DataFrame:
    """A small OHLCV frame with a controllable high-low spread (volatility)."""
    dates = pd.bdate_range("2026-01-01", periods=n)
    base = 100.0
    rng = np.arange(n, dtype=float)
    close = base + rng  # gentle uptrend, deterministic
    high = close + vol
    low = close - vol
    return pd.DataFrame({
        "Date": dates,
        "Open": close - vol / 2,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": [1_000_000] * n,
    })


@pytest.mark.unit
class TestAtrStop:
    def test_stop_below_last_close(self):
        df = _synthetic_ohlcv()
        stop = atr_stop(df, mult=2.0)
        last_close = float(df["Close"].iloc[-1])
        assert stop < last_close

    def test_stop_decreases_as_volatility_rises(self):
        calm = _synthetic_ohlcv(vol=2.0)
        wild = _synthetic_ohlcv(vol=10.0)
        assert atr_stop(wild) < atr_stop(calm)

    def test_stop_widens_with_higher_multiplier(self):
        df = _synthetic_ohlcv()
        assert atr_stop(df, mult=3.0) < atr_stop(df, mult=1.0)

    def test_latest_atr_from_frame_returns_close_and_atr(self):
        df = _synthetic_ohlcv()
        last_close, last_atr = latest_atr_from_frame(df)
        assert last_close == pytest.approx(float(df["Close"].iloc[-1]))
        assert last_atr > 0.0

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError):
            latest_atr_from_frame(pd.DataFrame(columns=["Close"]))

    def test_missing_close_raises(self):
        bad = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5]})
        with pytest.raises(ValueError):
            latest_atr_from_frame(bad)

    @pytest.mark.skip(
        reason="latest_atr(symbol, date) hits the network loader; covered by integration tests."
    )
    def test_latest_atr_network_path(self):
        from tradingagents.risk.atr_stop import latest_atr

        latest_atr("AAPL", "2026-06-01")


# ---------------------------------------------------------------------------
# breaker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDrawdownBreaker:
    def test_starts_normal(self):
        b = DrawdownBreaker()
        st = b.update(100.0)
        assert st.regime == "normal"
        assert st.can_open_new is True
        assert st.position_multiplier == 1.0
        assert st.current_drawdown == 0.0

    def test_warn_regime(self):
        b = DrawdownBreaker()
        b.update(100.0)
        st = b.update(96.0)  # -4%
        assert st.regime == "normal"
        st = b.update(94.0)  # -6% -> caution
        assert st.regime == "caution"
        assert st.can_open_new is True

    def test_no_new_regime_blocks_opening(self):
        b = DrawdownBreaker()
        b.update(100.0)
        st = b.update(89.0)  # -11%
        assert st.regime == "no_new"
        assert st.can_open_new is False
        assert st.position_multiplier == 0.5

    def test_hard_stop_flattens(self):
        b = DrawdownBreaker()
        b.update(100.0)
        st = b.update(80.0)  # -20% > 15%
        assert st.regime == "hard_stop"
        assert st.can_open_new is False
        assert st.position_multiplier == 0.0

    def test_cooldown_blocks_reopen_until_recovered(self):
        b = DrawdownBreaker(cooldown_steps=3)
        b.update(100.0)
        b.update(80.0)  # hard stop, cooldown set to 3
        # Recover fully (new high resets cooldown), then dip again gently.
        st = b.update(101.0)  # new peak -> cooldown cleared
        assert st.regime == "normal"
        assert st.can_open_new is True

    def test_cooldown_counts_down_below_no_new(self):
        b = DrawdownBreaker(cooldown_steps=2)
        b.update(100.0)
        b.update(80.0)  # hard stop
        # Climb back above the no_new threshold but still below peak.
        b.update(92.0)  # -8% from peak -> caution, cooldown 2->1
        b.update(92.0)  # cooldown 1->0
        st = b.update(92.0)  # caution, cooldown elapsed
        assert st.regime == "caution"
        assert st.can_open_new is True

    def test_check_exposure_rejects_oversized_single(self):
        b = DrawdownBreaker(max_single_position=0.20)
        allowed, reason = b.check_exposure(25.0, equity_value=100.0)
        assert allowed is False
        assert "single position" in reason.lower()

    def test_check_exposure_allows_within_cap(self):
        b = DrawdownBreaker()
        allowed, _ = b.check_exposure(15.0, equity_value=100.0)
        assert allowed is True

    def test_check_exposure_rejects_oversized_sector(self):
        b = DrawdownBreaker(max_single_sector=0.30)
        allowed, reason = b.check_exposure(
            position_value=10.0,
            equity_value=100.0,
            sector_value=40.0,
        )
        assert allowed is False
        assert "sector" in reason.lower()

    def test_reset_clears_state(self):
        b = DrawdownBreaker()
        b.update(100.0)
        b.update(80.0)  # hard stop
        b.reset()
        assert b.peak is None
        assert b.cooldown_remaining == 0
        st = b.update(100.0)
        assert st.regime == "normal"


# ---------------------------------------------------------------------------
# cvar
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCvar:
    def test_fat_negative_tail_more_negative_than_mean(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0.001, 0.01, size=500)
        # Inject a fat negative tail.
        fat_tail = normal.copy()
        fat_tail[:50] = -0.20
        assert historical_cvar(fat_tail) < historical_cvar(normal)
        assert historical_cvar(fat_tail) < np.mean(fat_tail)

    def test_cvar_is_negative_for_lossy_series(self):
        losses = np.full(100, -0.10)
        assert historical_cvar(losses) < 0.0

    def test_empty_returns_zero(self):
        assert historical_cvar([]) == 0.0

    def test_position_multiplier_de_risks_on_breach(self):
        # Worst 5% are deeply negative -> CVaR breaches -0.05 default.
        returns = np.concatenate([
            np.full(95, 0.0),
            np.full(5, -0.30),
        ])
        assert cvar_position_multiplier(returns) == 0.5

    def test_position_multiplier_normal_when_no_breach(self):
        # Calm series: CVaR comfortably above -0.05.
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.005, size=500)
        assert cvar_position_multiplier(returns) == 1.0

    def test_position_multiplier_normal_on_thin_data(self):
        # < 30 observations -> fail safe, do not block.
        assert cvar_position_multiplier([-0.50, -0.50, -0.50]) == 1.0

    def test_position_multiplier_custom_threshold(self):
        returns = np.full(100, -0.02)  # mild losses
        # Default threshold -0.05 -> not breached.
        assert cvar_position_multiplier(returns) == 1.0
        # Tighter threshold -> breached.
        assert cvar_position_multiplier(returns, breach_threshold=-0.01) == 0.5
