"""Portfolio drawdown breaker and exposure caps.

A stateful circuit breaker fed the live portfolio equity value on each
rebalance. As drawdown deepens it scales new-position size down, blocks
opening new positions, and finally flattens. Exposure caps enforce single
position / single sector concentration limits on top of the regime.

Pure: no global state, no config reads. The caller owns the equity stream.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BreakerState:
    """Snapshot of the breaker after an :meth:`DrawdownBreaker.update` call."""

    can_open_new: bool
    position_multiplier: float  # 1.0 normal, scaled down in caution / hard stop
    current_drawdown: float  # negative or 0
    regime: str  # "normal" | "caution" | "no_new" | "hard_stop"


class DrawdownBreaker:
    """Track running peak equity and translate drawdown into a regime.

    Regime -> position multiplier mapping:

    * ``normal``   — 1.0 (full size)
    * ``caution``  — 1.0 (warn but keep sizing; the *signal* is the regime)
    * ``no_new``   — 0.5 (trim existing, do not add)
    * ``hard_stop``— 0.0 (flatten and cool off)

    On entering ``hard_stop`` a cooldown counter starts. ``can_open_new``
    stays ``False`` until the cooldown elapses **and** the drawdown has
    recovered above the ``no_new`` threshold.
    """

    def __init__(
        self,
        warn_drawdown: float = 0.05,  # 5%  -> caution
        no_new_drawdown: float = 0.10,  # 10% -> stop opening new positions
        hard_stop_drawdown: float = 0.15,  # 15% -> hard stop: flatten & cool off
        max_single_position: float = 0.20,  # one ticker <= 20% of equity
        max_single_sector: float = 0.30,  # one sector <= 30%
        cooldown_steps: int = 5,  # ~ one week of daily rebalances
    ) -> None:
        if not (0.0 <= warn_drawdown <= no_new_drawdown <= hard_stop_drawdown):
            raise ValueError(
                "DrawdownBreaker: thresholds must satisfy "
                "0 <= warn <= no_new <= hard_stop"
            )
        if max_single_position <= 0.0 or max_single_sector <= 0.0:
            raise ValueError("DrawdownBreaker: exposure caps must be positive")
        if cooldown_steps < 0:
            raise ValueError("DrawdownBreaker: cooldown_steps must be non-negative")

        self.warn_drawdown = warn_drawdown
        self.no_new_drawdown = no_new_drawdown
        self.hard_stop_drawdown = hard_stop_drawdown
        self.max_single_position = max_single_position
        self.max_single_sector = max_single_sector
        self.cooldown_steps = cooldown_steps

        self.peak: float | None = None
        self.cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    # Regime logic
    # ------------------------------------------------------------------
    def update(self, equity_value: float) -> BreakerState:
        """Advance the breaker with a new equity value, returning the state.

        ``equity_value`` must be a positive finite number; non-finite or
        non-positive input leaves the prior state untouched (defensive —
        the live equity feed should never produce NaN, but a stray NaN
        must not corrupt the breaker).
        """
        ev = float(equity_value)
        if equity_value is None or ev <= 0.0 or ev != ev:  # NaN check
            # NaN-safe early out: preserve last state shape.
            if self.peak is None:
                return BreakerState(
                    can_open_new=True,
                    position_multiplier=1.0,
                    current_drawdown=0.0,
                    regime="normal",
                )
            return self._state(0.0, "normal")

        if self.peak is None or ev > self.peak:
            self.peak = ev
            # A new high means the cycle reset; clear any pending cooldown.
            self.cooldown_remaining = 0

        dd = (ev - self.peak) / self.peak  # negative or 0
        drawdown = -dd  # positive magnitude, 0 at peak

        # Decide regime.
        if drawdown >= self.hard_stop_drawdown:
            regime = "hard_stop"
            # Start (do not shorten) the cooldown each step we stay stopped.
            if self.cooldown_remaining < self.cooldown_steps:
                self.cooldown_remaining = self.cooldown_steps
        elif drawdown >= self.no_new_drawdown:
            regime = "no_new"
        elif drawdown >= self.warn_drawdown:
            regime = "caution"
        else:
            regime = "normal"

        # Tick the cooldown down when we are below the hard-stop threshold
        # and recovering toward normal.
        if regime != "hard_stop" and self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        return self._state(drawdown, regime)

    def _state(self, drawdown: float, regime: str) -> BreakerState:
        multiplier = {
            "normal": 1.0,
            "caution": 1.0,
            "no_new": 0.5,
            "hard_stop": 0.0,
        }[regime]

        # can_open_new rules:
        # - hard_stop / no_new: never.
        # - caution: allowed but the caller is warned via the regime.
        # - normal: blocked only while a cooldown from a prior hard stop
        #   is still counting down.
        if regime in ("hard_stop", "no_new") or self.cooldown_remaining > 0:
            can_open_new = False
        else:
            can_open_new = True

        return BreakerState(
            can_open_new=can_open_new,
            position_multiplier=multiplier,
            current_drawdown=-drawdown,  # report as negative or 0
            regime=regime,
        )

    # ------------------------------------------------------------------
    # Exposure caps
    # ------------------------------------------------------------------
    def check_exposure(
        self,
        position_value: float,
        equity_value: float,
        sector_value: float | None = None,
    ) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for a proposed single position.

        Rejects when the position (or, if given, the sector aggregate)
        exceeds its cap as a fraction of equity.
        """
        eq = float(equity_value)
        if equity_value is None or eq <= 0.0 or eq != eq:
            return False, "equity_value must be a positive finite number"

        pv = float(position_value) if position_value is not None else 0.0
        single_frac = pv / eq
        if single_frac > self.max_single_position + 1e-12:
            return (
                False,
                f"single position {single_frac:.1%} exceeds cap "
                f"{self.max_single_position:.1%}",
            )

        if sector_value is not None:
            sv = float(sector_value)
            if sv == sv:  # not NaN
                sector_frac = sv / eq
                if sector_frac > self.max_single_sector + 1e-12:
                    return (
                        False,
                        f"sector {sector_frac:.1%} exceeds cap "
                        f"{self.max_single_sector:.1%}",
                    )

        return True, "ok"

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget the running peak and any cooldown, returning to neutral."""
        self.peak = None
        self.cooldown_remaining = 0
