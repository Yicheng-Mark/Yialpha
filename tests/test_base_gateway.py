"""Unit tests for the synchronous gateway base (yiagents.execution.gateway).

Verifies the ABC contract (the six abstract methods are all required), the
synchronous ``send_order -> OrderData`` seam, the inert no-op callbacks (which
must not require an EventEngine), and the reserved ``proxies`` setting. All
``@pytest.mark.unit``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from yiagents.execution.domain import (
    AccountData,
    CancelRequest,
    Direction,
    Exchange,
    OrderData,
    OrderRequest,
    OrderType,
    PositionData,
    Status,
)
from yiagents.execution.gateway import BaseGateway


def _full_kwargs():
    """The six abstract-method names that every concrete gateway overrides."""
    return {
        "connect",
        "close",
        "send_order",
        "cancel_order",
        "query_account",
        "query_position",
    }


class _MinimalGateway(BaseGateway):
    """A concrete gateway implementing exactly the abstract surface."""

    default_name = "minimal"

    def connect(self, setting: dict) -> None:
        self.setting = setting

    def close(self) -> None:
        pass

    def send_order(self, req: OrderRequest) -> OrderData:
        return req.create_order_data(orderid="1", gateway_name=self.gateway_name)

    def cancel_order(self, req: CancelRequest) -> None:
        pass

    def query_account(self) -> AccountData:
        return AccountData(gateway_name=self.gateway_name, accountid="a1")

    def query_position(self) -> list[PositionData]:
        return []


@pytest.mark.unit
class TestAbcContract:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            BaseGateway("x")  # noqa: abstract methods present

    @pytest.mark.parametrize("missing", sorted(_full_kwargs()))
    def test_missing_one_abstract_still_abstract(self, missing):
        # Drop exactly one method -> the subclass is still abstract.
        attrs = {k: v for k, v in vars(_MinimalGateway).items()}
        attrs.pop(missing, None)

        cls = type("Partial", (BaseGateway,), attrs)
        with pytest.raises(TypeError):
            cls("x")

    def test_full_subclass_instantiates(self):
        gw = _MinimalGateway("min")
        assert gw.gateway_name == "min"
        assert gw.setting == {}


@pytest.mark.unit
class TestSynchronousSeam:
    def test_send_order_returns_order_data(self):
        gw = _MinimalGateway("min")
        req = OrderRequest(
            symbol="AAPL",
            exchange=Exchange.SMART,
            direction=Direction.LONG,
            type=OrderType.MARKET,
            volume=10,
        )
        out = gw.send_order(req)
        assert isinstance(out, OrderData)
        assert out.volume == 10
        assert out.vt_orderid == "min.1"

    def test_query_methods_return_types(self):
        gw = _MinimalGateway("min")
        assert isinstance(gw.query_account(), AccountData)
        assert isinstance(gw.query_position(), list)


@pytest.mark.unit
class TestCallbacksAreNoOps:
    def test_on_callbacks_exist_and_do_nothing(self):
        gw = _MinimalGateway("min")
        # Each callback must be callable without an EventEngine and return None.
        order = OrderData(
            gateway_name="min", symbol="AAPL", exchange=Exchange.SMART, orderid="1"
        )
        from yiagents.execution.domain import TradeData

        trade = TradeData(
            gateway_name="min",
            symbol="AAPL",
            exchange=Exchange.SMART,
            orderid="1",
            tradeid="t1",
        )
        assert gw.on_order(order) is None
        assert gw.on_trade(trade) is None
        assert gw.on_position(
            PositionData(
                gateway_name="min",
                symbol="AAPL",
                exchange=Exchange.SMART,
                direction=Direction.NET,
            )
        ) is None
        assert gw.on_account(
            AccountData(gateway_name="min", accountid="a1")
        ) is None

    def test_no_event_engine_attribute_required(self):
        # The base must not demand an event_engine (vnpy did).
        gw = _MinimalGateway("min")
        assert not hasattr(gw, "event_engine")


@pytest.mark.unit
class TestSetting:
    def test_default_setting_reserves_proxies(self):
        assert "proxies" in BaseGateway.default_setting
        # Base is network-agnostic: no `requests` import forced on consumers.
        assert BaseGateway.default_setting["proxies"] is None

    def test_setting_passed_through(self):
        gw = _MinimalGateway("min", setting={"proxies": {"http": "socks5h://..."}})
        assert gw.setting["proxies"]["http"].startswith("socks5h://")

    def test_get_default_setting(self):
        gw = _MinimalGateway("min")
        assert gw.get_default_setting() is BaseGateway.default_setting
