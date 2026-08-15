"""C3 regressions for ``yiagents.risk.breaker``.

Two audited defects:

1. ``update`` / ``check_exposure`` called ``float(equity_value)`` BEFORE the
   ``None`` check, so a ``None`` mark raised ``TypeError`` instead of taking
   the documented defensive path.
2. A NaN tick after a deep-drawdown regime returned ``_state(0.0, "normal")``
   — one corrupt data point whitewashed ``no_new`` / ``hard_stop`` back to
   normal and re-enabled opening positions for a beat. A bad tick must now
   keep the last valid drawdown + regime and block new positions.

Hermetic: pure state machine, no I/O.
"""

from __future__ import annotations

import math

import pytest

from yiagents.risk.breaker import DrawdownBreaker


@pytest.mark.unit
class TestBadInputHandling:
    def test_none_input_does_not_raise_and_keeps_regime(self):
        b = DrawdownBreaker()
        b.update(100.0)
        hard = b.update(80.0)  # -20% -> hard_stop
        assert hard.regime == "hard_stop"

        st = b.update(None)  # previously TypeError
        assert st.regime == "hard_stop"
        assert st.can_open_new is False
        assert st.position_multiplier == 0.0
        assert st.current_drawdown == pytest.approx(-0.20)

    def test_nan_input_keeps_deep_drawdown_regime(self):
        b = DrawdownBreaker()
        b.update(100.0)
        no_new = b.update(89.0)  # -11% -> no_new
        assert no_new.regime == "no_new"
        assert no_new.can_open_new is False

        st = b.update(float("nan"))
        # The regime is NOT whitewashed to normal; new positions stay blocked.
        assert st.regime == "no_new"
        assert st.can_open_new is False
        assert st.position_multiplier == 0.5
        assert st.current_drawdown == pytest.approx(-0.11)

    def test_nan_after_hard_stop_keeps_cooldown_state(self):
        b = DrawdownBreaker(cooldown_steps=3)
        b.update(100.0)
        b.update(70.0)  # -30% -> hard_stop, cooldown armed

        st = b.update(float("nan"))
        assert st.regime == "hard_stop"
        assert st.can_open_new is False
        # Internal state untouched: peak + cooldown survive the bad tick.
        assert b.peak == 100.0
        assert b.cooldown_remaining == 3
        # A subsequent GOOD tick resumes normal accounting from that state.
        recovered = b.update(99.0)  # -1% -> normal, cooldown ticks down
        assert recovered.regime == "normal"

    def test_inf_and_non_positive_inputs_fail_closed(self):
        b = DrawdownBreaker()
        b.update(100.0)
        b.update(80.0)  # hard_stop
        for bad in (math.inf, -math.inf, 0.0, -5.0, "not-a-number"):
            st = b.update(bad)
            assert st.regime == "hard_stop", bad
            assert st.can_open_new is False, bad

    def test_bad_first_tick_without_history_returns_neutral(self):
        b = DrawdownBreaker()
        st = b.update(float("nan"))
        assert st.regime == "normal"
        assert st.can_open_new is True
        # Nothing was remembered.
        assert b.peak is None

    def test_check_exposure_none_equity_rejects_without_raising(self):
        b = DrawdownBreaker()
        allowed, reason = b.check_exposure(10.0, equity_value=None)
        assert allowed is False
        assert "positive finite" in reason

    def test_reset_clears_cached_regime(self):
        b = DrawdownBreaker()
        b.update(100.0)
        b.update(80.0)  # hard_stop cached
        b.reset()
        st = b.update(float("nan"))
        # No cached regime -> neutral start state, not hard_stop.
        assert st.regime == "normal"
