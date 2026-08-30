"""Unit tests for the decision -> OrderRequest bridge (yialpha.execution.bridge).

Covers the direction truth table (rating priority, Trader-action fallback,
Hold -> no order), the volume guard, and function purity. Unlike the manual
trade-ticket renderer, this execution bridge fails closed on bearish signals:
selling to close is explicit, and opening a short needs a double opt-in.
"""

from __future__ import annotations

import pytest

from yialpha.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)
from yialpha.execution.bridge import decision_to_order_requests
from yialpha.execution.domain import (
    Direction,
    Exchange,
    Offset,
    OrderRequest,
    OrderType,
)


def _decision(rating: PortfolioRating | None) -> PortfolioDecision | None:
    if rating is None:
        return None
    return PortfolioDecision(
        rating=rating,
        executive_summary="s",
        investment_thesis="t",
    )


def _proposal(action: TraderAction | None) -> TraderProposal | None:
    if action is None:
        return None
    return TraderProposal(action=action, reasoning="r")


@pytest.mark.unit
class TestDirectionTruthTable:
    """Mirrors scripts/trade_ticket.py decide_direction rating/action priority."""

    @pytest.mark.parametrize(
        "rating, action, expected",
        [
            # Rating wins.
            (PortfolioRating.BUY, None, Direction.LONG),
            (PortfolioRating.OVERWEIGHT, None, Direction.LONG),
            (PortfolioRating.SELL, None, None),
            (PortfolioRating.UNDERWEIGHT, None, None),
            (PortfolioRating.HOLD, None, None),
            # Hold rating does NOT fall back to the Trader action.
            (PortfolioRating.HOLD, TraderAction.BUY, None),
            (PortfolioRating.HOLD, TraderAction.SELL, None),
            # Rating missing entirely -> Trader action decides.
            (None, TraderAction.BUY, Direction.LONG),
            (None, TraderAction.SELL, None),
            (None, TraderAction.HOLD, None),
            # Nothing committed.
            (None, None, None),
        ],
    )
    def test_direction(self, rating, action, expected):
        reqs = decision_to_order_requests(
            _decision(rating),
            _proposal(action),
            symbol="AAPL",
            exchange=Exchange.SMART,
            volume=10,
        )
        if expected is None:
            assert reqs == []
        else:
            assert len(reqs) == 1
            assert reqs[0].direction is expected
            assert reqs[0].symbol == "AAPL"
            assert reqs[0].exchange is Exchange.SMART
            assert reqs[0].volume == 10


@pytest.mark.unit
class TestOrderShape:
    def test_defaults_to_market_order(self):
        reqs = decision_to_order_requests(
            _decision(PortfolioRating.BUY),
            None,
            symbol="AAPL",
            exchange=Exchange.SMART,
            volume=10,
        )
        assert reqs[0].type is OrderType.MARKET
        assert reqs[0].price == 0.0
        assert reqs[0].offset is Offset.NONE

    def test_limit_order_with_price_and_offset(self):
        reqs = decision_to_order_requests(
            _decision(PortfolioRating.SELL),
            None,
            symbol="CL",
            exchange=Exchange.GLOBAL,
            volume=2,
            order_type=OrderType.LIMIT,
            price=78.5,
            offset=Offset.OPEN,
            reference="hedge",
            allow_short_open=True,
        )
        r = reqs[0]
        assert r.type is OrderType.LIMIT and r.price == 78.5
        assert r.offset is Offset.OPEN
        assert r.reference == "hedge"
        assert r.direction is Direction.SHORT

    def test_bearish_signal_can_explicitly_close_long(self):
        reqs = decision_to_order_requests(
            _decision(PortfolioRating.SELL),
            None,
            symbol="BTCUSDT",
            exchange=Exchange.BINANCE,
            volume=0.5,
            offset=Offset.CLOSE,
        )
        assert len(reqs) == 1
        assert reqs[0].direction is Direction.SHORT
        assert reqs[0].offset is Offset.CLOSE

    def test_offset_open_alone_does_not_authorize_short(self):
        reqs = decision_to_order_requests(
            _decision(PortfolioRating.SELL),
            None,
            symbol="BTCUSDT",
            exchange=Exchange.BINANCE,
            volume=0.5,
            offset=Offset.OPEN,
        )
        assert reqs == []

    def test_allow_short_alone_does_not_authorize_implicit_offset(self):
        reqs = decision_to_order_requests(
            _decision(PortfolioRating.SELL),
            None,
            symbol="BTCUSDT",
            exchange=Exchange.BINANCE,
            volume=0.5,
            allow_short_open=True,
        )
        assert reqs == []


@pytest.mark.unit
class TestGuards:
    def test_non_positive_volume_yields_nothing(self):
        for vol in (0, -5):
            reqs = decision_to_order_requests(
                _decision(PortfolioRating.BUY),
                None,
                symbol="AAPL",
                exchange=Exchange.SMART,
                volume=vol,
            )
            assert reqs == []

    def test_hold_is_empty_even_with_volume(self):
        reqs = decision_to_order_requests(
            _decision(PortfolioRating.HOLD),
            _proposal(TraderAction.BUY),
            symbol="AAPL",
            exchange=Exchange.SMART,
            volume=10,
        )
        assert reqs == []


@pytest.mark.unit
class TestPurity:
    def test_same_inputs_same_output(self):
        kwargs = {
            "symbol": "AAPL",
            "exchange": Exchange.SMART,
            "volume": 7,
        }
        a = decision_to_order_requests(_decision(PortfolioRating.BUY), None, **kwargs)
        b = decision_to_order_requests(_decision(PortfolioRating.BUY), None, **kwargs)
        assert len(a) == len(b) == 1
        # Same fields -> same derived request.
        assert a[0].direction is b[0].direction
        assert a[0].volume == b[0].volume
        assert isinstance(a[0], OrderRequest)

    def test_no_mutation_of_inputs(self):
        d = _decision(PortfolioRating.BUY)
        p = _proposal(TraderAction.BUY)
        before_rating = d.rating
        decision_to_order_requests(
            d, p, symbol="AAPL", exchange=Exchange.SMART, volume=3
        )
        assert d.rating is before_rating
        assert p.action is TraderAction.BUY
