"""Unit tests for the execution domain model (yiagents.execution.domain).

Adapted-from-vnpy dataclasses + enums: construction, the ``vt_*`` naming
convention, the ``create_order_data`` factory, and ``is_active``. No network,
no LLM. All ``@pytest.mark.unit``.
"""

from __future__ import annotations

import pytest

from yiagents.execution.domain import (
    ACTIVE_STATUSES,
    AccountData,
    BaseData,
    CancelRequest,
    Direction,
    Exchange,
    Offset,
    OrderData,
    OrderRequest,
    OrderType,
    PositionData,
    Product,
    Status,
    TradeData,
)


@pytest.mark.unit
class TestEnums:
    def test_direction_round_trip(self):
        assert Direction("long") is Direction.LONG
        assert Direction.LONG.value == "long"

    def test_status_round_trip(self):
        for member in Status:
            assert Status(member.value) is member

    def test_active_statuses_membership(self):
        # The three "can still change" statuses; finals are absent.
        assert {
            Status.SUBMITTING,
            Status.NOTTRADED,
            Status.PARTTRADED,
        } == ACTIVE_STATUSES
        assert Status.ALLTRADED not in ACTIVE_STATUSES
        assert Status.CANCELLED not in ACTIVE_STATUSES
        assert Status.REJECTED not in ACTIVE_STATUSES

    def test_exchange_has_binance(self):
        # Synthetic crypto exchange added for YiAgents; the rest are pruned.
        assert Exchange("BINANCE") is Exchange.BINANCE
        assert Exchange("SMART") is Exchange.SMART
        assert Exchange("SSE") is Exchange.SSE

    def test_product_pruned(self):
        # WARRANT / SPREAD / FUND / BOND dropped; OPTION dropped (no options yet).
        values = {p.value for p in Product}
        assert "equity" in values and "futures" in values
        assert "warrant" not in values and "bond" not in values
        assert not hasattr(Product, "OPTION")


@pytest.mark.unit
class TestOrderData:
    def _order(self, **kw):
        base = {
            "gateway_name": "binance",
            "symbol": "BTCUSDT",
            "exchange": Exchange.BINANCE,
            "orderid": "o1",
        }
        base.update(kw)
        return OrderData(**base)

    def test_vt_naming_convention(self):
        o = self._order(direction=Direction.LONG)
        assert o.vt_symbol == "BTCUSDT.BINANCE"
        assert o.vt_orderid == "binance.o1"

    def test_defaults(self):
        o = self._order()
        assert o.type is OrderType.LIMIT
        assert o.offset is Offset.NONE
        assert o.status is Status.SUBMITTING
        assert o.traded == 0
        assert o.reference == ""

    def test_is_active_truth_table(self):
        assert self._order(status=Status.SUBMITTING).is_active()
        assert self._order(status=Status.NOTTRADED).is_active()
        assert self._order(status=Status.PARTTRADED).is_active()
        assert not self._order(status=Status.ALLTRADED).is_active()
        assert not self._order(status=Status.CANCELLED).is_active()
        assert not self._order(status=Status.REJECTED).is_active()

    def test_create_cancel_request(self):
        o = self._order(direction=Direction.LONG)
        c = o.create_cancel_request()
        assert isinstance(c, CancelRequest)
        assert c.orderid == "o1"
        assert c.symbol == "BTCUSDT"
        assert c.exchange is Exchange.BINANCE
        assert c.vt_symbol == "BTCUSDT.BINANCE"


@pytest.mark.unit
class TestTradeData:
    def test_vt_naming_convention(self):
        t = TradeData(
            gateway_name="binance",
            symbol="BTCUSDT",
            exchange=Exchange.BINANCE,
            orderid="o1",
            tradeid="t1",
            direction=Direction.LONG,
        )
        assert t.vt_symbol == "BTCUSDT.BINANCE"
        assert t.vt_orderid == "binance.o1"
        assert t.vt_tradeid == "binance.t1"


@pytest.mark.unit
class TestPositionData:
    def test_vt_positionid(self):
        p = PositionData(
            gateway_name="ib",
            symbol="AAPL",
            exchange=Exchange.SMART,
            direction=Direction.NET,
            volume=100,
        )
        assert p.vt_symbol == "AAPL.SMART"
        assert p.vt_positionid == "ib.AAPL.SMART.net"


@pytest.mark.unit
class TestAccountData:
    def test_available_derived(self):
        a = AccountData(gateway_name="ib", accountid="acc1", balance=10000, frozen=2500)
        assert a.available == 7500
        assert a.vt_accountid == "ib.acc1"

    def test_available_default_zero(self):
        a = AccountData(gateway_name="ib", accountid="acc1")
        assert a.balance == 0 and a.frozen == 0 and a.available == 0


@pytest.mark.unit
class TestOrderRequest:
    def test_vt_symbol(self):
        r = OrderRequest(
            symbol="AAPL",
            exchange=Exchange.SMART,
            direction=Direction.LONG,
            type=OrderType.MARKET,
            volume=10,
        )
        assert r.vt_symbol == "AAPL.SMART"

    def test_create_order_data_round_trip(self):
        r = OrderRequest(
            symbol="AAPL",
            exchange=Exchange.NASDAQ,
            direction=Direction.SHORT,
            type=OrderType.LIMIT,
            volume=5,
            price=189.5,
            offset=Offset.OPEN,
            reference="pm-decision",
        )
        o = r.create_order_data(orderid="42", gateway_name="ib")
        assert isinstance(o, OrderData)
        assert (o.symbol, o.exchange, o.orderid) == ("AAPL", Exchange.NASDAQ, "42")
        assert o.direction is Direction.SHORT
        assert o.type is OrderType.LIMIT
        assert o.volume == 5 and o.price == 189.5
        assert o.offset is Offset.OPEN
        assert o.reference == "pm-decision"
        assert o.gateway_name == "ib"
        # Derived fields propagate.
        assert o.vt_symbol == "AAPL.NASDAQ"
        assert o.vt_orderid == "ib.42"


@pytest.mark.unit
class TestCancelRequest:
    def test_vt_symbol(self):
        c = CancelRequest(orderid="o9", symbol="AAPL", exchange=Exchange.SMART)
        assert c.vt_symbol == "AAPL.SMART"


@pytest.mark.unit
class TestBaseDataSurface:
    def test_base_data_only_gateway_name(self):
        # vnpy's `extra` passthrough field was dropped.
        import dataclasses

        fields = {f.name for f in dataclasses.fields(BaseData)}
        assert fields == {"gateway_name"}
