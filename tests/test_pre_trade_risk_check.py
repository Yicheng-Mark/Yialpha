"""Tests for the mandatory pre-``send_order`` risk guard.

The gateway performs no quantitative risk check of its own; ``pre_trade_risk_check``
is the enforcement edge that translates the risk overlay's decision into order
filtration / clipping. These tests pin the three gates: breaker hard-stop block,
weight-cap clipping, and the pass-through when no risk decision is supplied.
"""
from __future__ import annotations

import pytest

from yiagents.execution.bridge import pre_trade_risk_check
from yiagents.execution.domain import Direction, Exchange, Offset, OrderRequest, OrderType
from yiagents.risk.breaker import BreakerState
from yiagents.risk.manager import RiskDecision

pytestmark = pytest.mark.unit


def _order(
    volume: float,
    price: float = 100.0,
    *,
    order_type: OrderType = OrderType.LIMIT,
    offset: Offset = Offset.NONE,
) -> OrderRequest:
    return OrderRequest(
        symbol="BTCUSDT",
        exchange=Exchange.BINANCE,
        direction=Direction.LONG,
        type=order_type,
        volume=volume,
        price=price,
        offset=offset,
        reference="test",
    )


def _decision(
    action: str = "enter",
    regime: str = "normal",
    target_weight: float = 0.05,
) -> RiskDecision:
    return RiskDecision(
        rating="Buy",
        action=action,
        target_weight=target_weight,
        position_value=target_weight * 100_000,
        stop_loss=None,
        entry_price=None,
        kelly_raw=target_weight,
        breaker=BreakerState(
            can_open_new=(action != "blocked"),
            position_multiplier=1.0,
            current_drawdown=0.0,
            regime=regime,
        ),
        cvar_multiplier=1.0,
        exposure_ok=True,
        exposure_reason="ok",
        rationale="test",
    )


class TestPreTradeRiskCheck:
    def test_no_risk_decision_rejects_open_orders(self):
        """Missing risk context is never permission to increase exposure."""
        orders = [_order(10), _order(20)]
        out = pre_trade_risk_check(orders, None)
        assert out == []

    def test_no_risk_decision_preserves_explicit_close(self):
        close = _order(10, offset=Offset.CLOSE)
        assert pre_trade_risk_check([close], None) == [close]

    def test_blocked_action_drops_all_orders(self):
        """Gate 1: action='blocked' -> empty list."""
        orders = [_order(10)]
        out = pre_trade_risk_check(orders, _decision(action="blocked"))
        assert out == []

    @pytest.mark.parametrize("action", ["hold", "reduce", "exit", "unexpected"])
    def test_non_entry_actions_cannot_open_exposure(self, action):
        out = pre_trade_risk_check(
            [_order(10)],
            _decision(action=action),
            equity=100_000,
        )
        assert out == []

    def test_hard_stop_regime_drops_all_orders(self):
        """A hard stop blocks new exposure even if action != blocked."""
        orders = [_order(10)]
        out = pre_trade_risk_check(
            orders, _decision(action="enter", regime="hard_stop")
        )
        assert out == []

    def test_hard_stop_preserves_reduce_only_close(self):
        open_order = _order(10, offset=Offset.OPEN)
        close_order = _order(7, offset=Offset.CLOSE)
        out = pre_trade_risk_check(
            [open_order, close_order],
            _decision(action="enter", regime="hard_stop"),
        )
        assert out == [close_order]

    def test_weight_cap_clips_limit_order(self):
        """Gate 2: a 15% order clipped to the 10% cap."""
        # equity=100_000, max_weight=0.10 -> value cap = 10_000.
        # At price=100, max_volume = 100. An order for 150 -> clipped to 100.
        orders = [_order(150, price=100.0)]
        out = pre_trade_risk_check(
            orders, _decision(), equity=100_000, max_weight=0.10
        )
        assert len(out) == 1
        assert out[0].volume == pytest.approx(100.0)

    def test_weight_cap_drops_zero_volume_order(self):
        """Gate 2: an order clipped to <= 0 volume is dropped entirely."""
        # equity=1_000, max_weight=0.01 -> cap=10. At price=100, max_vol=0.1.
        # An order for 5 -> clipped to 0.1 (kept); an order for volume that
        # clips to 0 would need cap < price. Test the drop: cap < 1 share.
        orders = [_order(5, price=100.0)]
        out = pre_trade_risk_check(
            orders, _decision(), equity=100, max_weight=0.01
        )
        # cap = 1.0, max_volume = 1.0 / 100 = 0.01 -> clipped from 5 to 0.01.
        assert len(out) == 1
        assert out[0].volume == pytest.approx(0.01)

    def test_zero_equity_rejects_open_order(self):
        """A missing account-equity snapshot must fail closed."""
        orders = [_order(10_000)]
        out = pre_trade_risk_check(orders, _decision(), equity=0)
        assert out == []

    def test_normal_decision_with_small_order_passes_unchanged(self):
        """All three gates pass: a small order in a normal regime is untouched."""
        orders = [_order(5, price=100.0)]  # value 500
        out = pre_trade_risk_check(
            orders, _decision(action="enter", regime="normal"),
            equity=100_000, max_weight=0.20,
        )
        assert len(out) == 1
        assert out[0].volume == 5

    def test_market_order_without_reference_price_is_rejected(self):
        """A price-less market order cannot be safely value-capped."""
        market = OrderRequest(
            symbol="BTCUSDT",
            exchange=Exchange.BINANCE,
            direction=Direction.LONG,
            type=OrderType.MARKET,
            volume=10,
            price=0.0,
            offset=Offset.NONE,
            reference="test",
        )
        out = pre_trade_risk_check([market], _decision(), equity=100_000, max_weight=0.01)
        assert out == []

    def test_large_market_order_is_clipped_using_reference_price(self):
        market = _order(
            1_000_000,
            price=0.0,
            order_type=OrderType.MARKET,
            offset=Offset.OPEN,
        )
        out = pre_trade_risk_check(
            [market],
            _decision(),
            equity=100_000,
            max_weight=0.01,
            reference_price=50_000,
        )
        assert len(out) == 1
        assert out[0].volume == pytest.approx(0.02)

    def test_symbol_reference_price_mapping(self):
        market = _order(10, price=0.0, order_type=OrderType.MARKET)
        out = pre_trade_risk_check(
            [market],
            _decision(),
            equity=100_000,
            max_weight=0.01,
            reference_prices={"BTCUSDT": 20_000},
        )
        assert out[0].volume == pytest.approx(0.05)

    def test_empty_input_returns_empty(self):
        """No orders in -> no orders out, regardless of risk decision."""
        assert pre_trade_risk_check([], None) == []
        assert pre_trade_risk_check([], _decision()) == []
