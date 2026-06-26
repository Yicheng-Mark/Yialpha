"""RiskManager: the deterministic overlay that turns a rating into a sized, risk-checked position.

This is the heart of Phase 1. The agent graph keeps producing a 5-tier rating and
a qualitative thesis; this module owns everything quantitative:

* **Size** -- quarter-Kelly scaled by the rating's conviction, blended with the
  historical win rate when a trade ledger is available.
* **Stop** -- ATR-based stop loss (reuses the indicator the market analyst
  already computes).
* **Circuit breaker** -- a portfolio drawdown breaker that warns, blocks new
  entries, and hard-stops as losses deepen, plus per-ticker / per-sector
  exposure caps.
* **CVaR de-risking** -- scales size down when the tail-loss estimate breaches.

The same :class:`RiskManager` powers two callers:

* The Phase-0 backtest engine, via a ``weight_fn`` built by
  :func:`build_backtest_weight_fn`, so a strategy can be re-priced under risk
  rules using identical realized decisions (the A/B comparison every later phase
  runs against).
* The live flow, after the Portfolio Manager emits its rating, as a
  deterministic overlay that overrides the LLM's free-text ``position_sizing``
  and ``stop_loss``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from yiagents.risk.breaker import BreakerState, DrawdownBreaker
from yiagents.risk.cvar import cvar_position_multiplier
from yiagents.risk.kelly import RATING_TO_CONFIDENCE, kelly_sizing

logger = logging.getLogger(__name__)

_BULLISH = ("Buy", "Overweight")
_BEARISH = ("Sell", "Underweight")


@dataclass
class PortfolioState:
    """Snapshot of the portfolio the risk manager sizes against.

    Carried between ``decide`` calls. The breaker's drawdown tracking and
    Kelly's win-rate both read accumulated fields here, so the caller updates
    this object as the portfolio evolves.
    """

    cash: float = 0.0
    equity: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)   # ticker -> market value
    sectors: dict[str, float] = field(default_factory=dict)     # sector -> market value
    returns_history: list[float] = field(default_factory=list)
    # Closed-trade ledger for Kelly win-rate: each {"ticker", "rating", "return"}.
    trade_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def invested(self) -> float:
        return float(sum(self.positions.values()))


@dataclass
class RiskDecision:
    """The deterministic overlay a RiskManager returns for one decision."""

    rating: str
    action: str                      # enter | add | hold | reduce | exit | blocked
    target_weight: float             # final weight after Kelly x breaker x CVaR x caps
    position_value: float            # target_weight * equity
    stop_loss: float | None
    entry_price: float | None
    kelly_raw: float                 # pre-clip Kelly weight (transparency)
    breaker: BreakerState
    cvar_multiplier: float
    exposure_ok: bool
    exposure_reason: str
    rationale: str


def _trade_stats(history: list[dict[str, Any]]) -> tuple[int, int, float, float]:
    """Return (wins, losses, avg_win, avg_loss) from a closed-trade ledger."""
    wins = losses = 0
    win_vals: list[float] = []
    loss_vals: list[float] = []
    for t in history:
        ret = t.get("return")
        if ret is None or not isinstance(ret, (int, float)):
            continue
        if ret > 0:
            wins += 1
            win_vals.append(float(ret))
        else:
            losses += 1
            loss_vals.append(abs(float(ret)))
    avg_win = sum(win_vals) / len(win_vals) if win_vals else 1.0
    avg_loss = sum(loss_vals) / len(loss_vals) if loss_vals else 1.0
    return wins, losses, avg_win, avg_loss


def _action_for(rating: str, is_new: bool) -> str:
    if rating in _BULLISH:
        return "enter" if is_new else "add"
    if rating in _BEARISH:
        return "exit" if rating == "Sell" else "reduce"
    return "hold"


class RiskManager:
    """Stateful orchestrator over Kelly + ATR + drawdown breaker + CVaR."""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_single_position: float = 0.20,
        max_single_sector: float = 0.30,
        warn_drawdown: float = 0.05,
        no_new_drawdown: float = 0.10,
        hard_stop_drawdown: float = 0.15,
        atr_mult: float = 2.0,
        cvar_confidence: float = 0.95,
        cvar_breach: float = -0.05,
        use_atr_stop: bool = True,
        breaker: DrawdownBreaker | None = None,
    ):
        self.kelly_fraction = kelly_fraction
        self.atr_mult = atr_mult
        self.cvar_confidence = cvar_confidence
        self.cvar_breach = cvar_breach
        self.use_atr_stop = use_atr_stop
        self.breaker = breaker or DrawdownBreaker(
            warn_drawdown=warn_drawdown,
            no_new_drawdown=no_new_drawdown,
            hard_stop_drawdown=hard_stop_drawdown,
            max_single_position=max_single_position,
            max_single_sector=max_single_sector,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RiskManager":
        """Build a RiskManager from a DEFAULT_CONFIG-style mapping.

        Reading is defensive: missing keys fall back to the constructor defaults
        so a partial / older config never breaks construction.
        """
        hard_stop = float(config.get("max_drawdown_hard_stop", 0.15))
        return cls(
            kelly_fraction=float(config.get("kelly_fraction", 0.25)),
            max_single_position=float(config.get("max_single_position", 0.20)),
            max_single_sector=float(config.get("max_single_sector", 0.30)),
            # Stagger the intermediate thresholds off the hard-stop so a single
            # config knob scales the whole breaker ladder proportionally.
            warn_drawdown=round(min(hard_stop / 3.0, 0.05), 4),
            no_new_drawdown=round(min(hard_stop * 2.0 / 3.0, 0.10), 4),
            hard_stop_drawdown=hard_stop,
            atr_mult=float(config.get("atr_stop_mult", 2.0)),
        )

    def decide(
        self,
        ticker: str,
        rating: str,
        state: PortfolioState,
        price: float | None = None,
        atr: float | None = None,
        sector: str | None = None,
        date: str | None = None,
    ) -> RiskDecision:
        """Produce the risk overlay for one decision.

        ``price``/``atr`` are optional; when either is missing the ATR stop is
        skipped (the size/breaker logic still runs). ``sector`` enables the
        sector exposure cap.
        """
        rating = rating if rating in RATING_TO_CONFIDENCE else "Hold"

        # 1. Update the drawdown breaker with the latest equity mark.
        breaker_state = self.breaker.update(state.equity) if state.equity > 0 else BreakerState(
            can_open_new=True, position_multiplier=1.0, current_drawdown=0.0, regime="normal",
        )

        # 2. Kelly sizing (rating conviction x optional historical win rate).
        wins, losses, avg_win, avg_loss = _trade_stats(state.trade_history)
        kelly_raw = kelly_sizing(
            rating,
            wins=wins if wins or losses else None,
            losses=losses if wins or losses else None,
            avg_win=avg_win, avg_loss=avg_loss,
            kelly_fraction_mult=self.kelly_fraction,
        )

        # 3. CVaR de-risking from the returns history.
        cvar_mult = cvar_position_multiplier(
            state.returns_history,
            confidence=self.cvar_confidence,
            breach_threshold=self.cvar_breach,
        )

        # 4. Existing exposure to this ticker / sector.
        existing_value = float(state.positions.get(ticker, 0.0))
        is_new = existing_value <= 0.0

        # 5. Combine: Kelly x breaker x CVaR, then clip to exposure cap.
        target_weight = kelly_raw * breaker_state.position_multiplier * cvar_mult
        target_weight = max(0.0, min(target_weight, self.breaker.max_single_position))

        # 6. Breaker can block NEW entries outright (existing positions are not
        #    force-added but may be held; hard-stop flattens via multiplier 0).
        action = _action_for(rating, is_new)
        exposure_ok = True
        exposure_reason = "ok"
        if rating in _BULLISH and is_new and not breaker_state.can_open_new:
            target_weight = 0.0
            action = "blocked"
            exposure_ok = False
            exposure_reason = f"drawdown breaker blocks new entries (regime={breaker_state.regime})"

        # Sector exposure cap (advisory: flags but does not zero the position
        # because sector data may be partial).
        if sector is not None and target_weight > 0:
            sector_weight = state.sectors.get(sector, 0.0) / state.equity if state.equity > 0 else 0.0
            if sector_weight + target_weight > self.breaker.max_single_sector:
                headroom = max(0.0, self.breaker.max_single_sector - sector_weight)
                if headroom <= 0.0:
                    target_weight = 0.0
                    action = "blocked"
                    exposure_ok = False
                    exposure_reason = f"sector cap reached for {sector}"
                else:
                    target_weight = headroom
                    exposure_reason = f"trimmed to sector cap for {sector}"

        # 7. ATR stop (only when we have a price; long-only convention here).
        stop_loss: float | None = None
        entry_price = float(price) if price is not None else None
        if self.use_atr_stop and entry_price is not None and target_weight > 0.0:
            try:
                if atr is not None:
                    stop_loss = atr_stop_from_values(entry_price, float(atr), self.atr_mult)
                # No ATR supplied -> no stop; caller can backfill via latest_atr.
            except (ValueError, TypeError):
                stop_loss = None

        position_value = target_weight * state.equity if state.equity > 0 else 0.0

        rationale = self._rationale(
            rating, action, kelly_raw, target_weight, breaker_state, cvar_mult, exposure_reason,
        )

        return RiskDecision(
            rating=rating,
            action=action,
            target_weight=target_weight,
            position_value=position_value,
            stop_loss=stop_loss,
            entry_price=entry_price,
            kelly_raw=kelly_raw,
            breaker=breaker_state,
            cvar_multiplier=cvar_mult,
            exposure_ok=exposure_ok,
            exposure_reason=exposure_reason,
            rationale=rationale,
        )

    @staticmethod
    def _rationale(rating, action, kelly_raw, target_weight, breaker, cvar_mult, exposure_reason):
        parts = [
            f"Rating {rating} -> action {action}.",
            f"Kelly raw {kelly_raw:.3f} x breaker {breaker.position_multiplier:.2f}"
            f" ({breaker.regime}) x CVaR {cvar_mult:.2f} = target weight {target_weight:.3f}.",
        ]
        if exposure_reason != "ok":
            parts.append(f"Exposure: {exposure_reason}.")
        if breaker.regime != "normal":
            parts.append(f"Drawdown {breaker.current_drawdown:.1%}; regime {breaker.regime}.")
        return " ".join(parts)


def atr_stop_from_values(close: float, atr: float, mult: float) -> float:
    """Long stop = close - mult*ATR. Raises ValueError on non-positive inputs."""
    if close <= 0 or atr <= 0:
        raise ValueError(f"close and atr must be positive (close={close}, atr={atr})")
    return close - mult * atr


def build_backtest_weight_fn(
    risk_manager: RiskManager,
    ticker: str,
    sector: str | None = None,
    price_lookup: Callable[[str], float | None] | None = None,
    atr_lookup: Callable[[str], float | None] | None = None,
) -> Callable[[str, str, dict[str, Any]], float]:
    """Adapt a RiskManager into the backtest engine's ``weight_fn`` signature.

    The engine threads a ``ctx`` dict (``equity_history``, ``returns_history``,
    ``holding_days``). This closure maintains a :class:`PortfolioState` across
    the backtest so the breaker's drawdown and Kelly's ledger accumulate
    correctly. Realized trade returns are fed back into the ledger from
    ``ctx['realized_returns']`` if the caller populates it.
    """
    state = PortfolioState()

    def _fn(rating: str, date: str, ctx: dict[str, Any]) -> float:
        equity_series = ctx.get("equity_history") or [0.0]
        state.equity = float(equity_series[-1]) if equity_series else 0.0
        state.returns_history = list(ctx.get("returns_history") or [])
        # Fold in realized closed-trade returns for Kelly win-rate (if provided).
        for ret in (ctx.get("realized_returns") or []):
            state.trade_history.append({"return": float(ret)})

        price = price_lookup(date) if price_lookup else None
        atr = atr_lookup(date) if atr_lookup else None
        decision = risk_manager.decide(
            ticker, rating, state, price=price, atr=atr, sector=sector, date=date,
        )
        # Reflect the chosen weight into the state's position map so the next
        # call sees the (approximate) resulting exposure for is_new checks.
        state.positions[ticker] = decision.target_weight * state.equity
        return decision.target_weight

    return _fn
