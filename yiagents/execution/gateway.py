"""Synchronous broker gateway abstraction for Track B execution.

Adapted from VeighNa (vnpy), Copyright (c) 2015-present Xiaoyou Chen,
distributed under the MIT License
(https://github.com/vnpy/vnpy/blob/master/LICENSE). Adapted for YiAgents:

* **Fully synchronous.** vnpy's ``BaseGateway`` takes an ``EventEngine`` and
  pushes every state change through it via the ``on_*`` callbacks (which run
  on the engine's background threads). YiAgents is a single-process,
  synchronous, LangGraph-driven batch flow with a strict "the execution layer
  introduces no new concurrency" rule, so this base carries **no
  EventEngine, no threads, no queues**. ``on_*`` survive as inert no-op hooks
  so a future ``StreamingGateway`` mixin can wire them up without changing
  this synchronous contract.
* **``send_order`` returns ``OrderData`` synchronously** instead of a
  ``vt_orderid`` string with the fill arriving later via ``on_order``. This
  matches the existing ``browser_broker.place_order`` contract (fire one
  order, get one result), which is the only execution shape YiAgents needs.
* **Dropped** subscribe / quote / history / write_log (no tick stream, no
  two-sided quotes, YiAgents has its own market-data layer, and logging uses
  the stdlib ``logging`` module).

A concrete gateway (Track B) subclasses this, implements the abstract methods,
and fills in the class attributes. The class attribute ``default_setting``
reserves a ``proxies`` key — concrete HTTP gateways should populate it from
``yiagents.dataflows.binance.proxy_map()`` (the project-wide SOCKS5h pattern)
rather than hard-coding a proxy here, so the base stays network-agnostic and
reusable by non-HTTP gateways (e.g. a local matching simulator).

Track B adapter sketch (NOT implemented this round) — a ``BrowserBrokerGateway``
will wrap the existing ``yiagents.execution.browser_broker.BrowserBroker``
behind this contract, forwarding its kill-switch / dry-run / validator gates
unchanged and mapping ``OrderResult.status`` -> vnpy ``Status``
(SUBMITTED->ALLTRADED, DRY_RUN_PREVIEW->SUBMITTING, BLOCKED_*->REJECTED).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .domain import (
    AccountData,
    CancelRequest,
    OrderData,
    OrderRequest,
    PositionData,
    TradeData,
)

logger = logging.getLogger(__name__)


class BaseGateway(ABC):
    """Abstract base for a broker / exchange gateway.

    Subclasses implement the abstract methods below. The ``on_*`` callbacks
    are **not** abstract and default to no-ops: in the synchronous execution
    model the return values of ``send_order`` / ``query_*`` carry the state, so
    the callbacks exist only as forward-compatible seams for a future
    event-driven streaming layer.
    """

    # Human-readable name (filled in by concrete gateways).
    default_name: str = ""

    # Connection settings a concrete gateway expects. ``proxies`` is reserved
    # so HTTP gateways can receive a SOCKS5 proxy map without the base
    # importing ``requests`` or hard-coding an endpoint.
    default_setting: dict = {
        "proxies": None,  # dict[str, str] | None; future gateways pass to requests(proxies=...)
    }

    # Exchanges this gateway can route to.
    exchanges: list = []

    def __init__(self, gateway_name: str, *, setting: dict | None = None) -> None:
        self.gateway_name: str = gateway_name
        self.setting: dict = dict(setting) if setting else {}

    # ------------------------------------------------------------------
    # Abstract — a concrete gateway MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self, setting: dict) -> None:
        """Establish the gateway connection (and prime any session state)."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release the connection / underlying resources."""
        raise NotImplementedError

    @abstractmethod
    def send_order(self, req: OrderRequest) -> OrderData:
        """Submit ``req`` and return the resulting ``OrderData`` synchronously.

        The returned ``OrderData`` carries whatever status the broker
        reported immediately (typically ``ALLTRADED`` for a market fill or
        ``NOTTRADED`` / ``SUBMITTING`` for a resting limit order; ``REJECTED``
        if the broker turned it down). The caller reads the result directly —
        no ``on_order`` callback is required for the synchronous flow.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> None:
        """Cancel the order identified by ``req``."""
        raise NotImplementedError

    @abstractmethod
    def query_account(self) -> AccountData:
        """Return the current account balance snapshot."""
        raise NotImplementedError

    @abstractmethod
    def query_position(self) -> list[PositionData]:
        """Return the current list of open positions."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional no-op callbacks — seams for a future streaming layer
    # ------------------------------------------------------------------
    # These intentionally do nothing today. They take the vnpy callback
    # signatures so that a later ``StreamingGateway`` (mixing in an
    # EventEngine) can override them without altering this synchronous base.

    def on_order(self, order: OrderData) -> None:
        pass

    def on_trade(self, trade: TradeData) -> None:
        pass

    def on_position(self, position: PositionData) -> None:
        pass

    def on_account(self, account: AccountData) -> None:
        pass

    def get_default_setting(self) -> dict:
        """Return the default connection setting dict."""
        return self.default_setting
