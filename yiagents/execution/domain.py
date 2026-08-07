"""Execution domain model: the dataclasses and enums an order travels as.

Adapted from VeighNa (vnpy), Copyright (c) 2015-present Xiaoyou Chen,
distributed under the MIT License
(https://github.com/vnpy/vnpy/blob/master/LICENSE). Adapted for YiAgents:

* **Pruned** to the subset needed for single-symbol batch order execution —
  tick / bar / quote / contract / history / log data are removed (YiAgents has
  its own read-only market-data layer in ``yiagents.dataflows`` and never
  consumes a tick stream). Only the order lifecycle (request -> order/trade),
  account, and position objects survive.
* **Made fully synchronous** — see ``gateway.py``; this file carries no
  threading/event dependency.
* **Dropped the gettext-based i18n wrappers** (``from .locale import _``);
  enum values are plain lowercase ASCII strings, which also sidesteps the
  Windows GBK-console encoding pitfall the rest of the project guards against.

The ``vt_symbol`` / ``vt_orderid`` / ``vt_tradeid`` naming convention is
preserved verbatim — concrete gateways (Track B) rely on it for id scoping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime as Datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enums (constant.py, pruned + de-i18n'd)
# ---------------------------------------------------------------------------


class Direction(Enum):
    """Direction of an order / trade / position."""

    LONG = "long"
    SHORT = "short"
    NET = "net"


class Offset(Enum):
    """Offset of an order / trade (mainly relevant for futures)."""

    NONE = ""
    OPEN = "open"
    CLOSE = "close"
    CLOSETODAY = "close_today"
    CLOSEYESTERDAY = "close_yesterday"


class Status(Enum):
    """Order status."""

    SUBMITTING = "submitting"
    NOTTRADED = "not_traded"
    PARTTRADED = "part_traded"
    ALLTRADED = "all_traded"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Product(Enum):
    """Product class — trimmed to the asset types YiAgents trades."""

    EQUITY = "equity"
    FUTURES = "futures"
    INDEX = "index"
    ETF = "etf"
    SPOT = "spot"
    CFD = "cfd"
    SWAP = "swap"
    FOREX = "forex"


class OrderType(Enum):
    """Order type."""

    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    FAK = "fak"
    FOK = "fok"


class Exchange(Enum):
    """Exchange — trimmed to YiAgents' markets.

    ``BINANCE`` is a synthetic entry for Binance crypto (perp + spot); add
    more members as concrete gateways land (this is backward-compatible — new
    enum members never break already-serialized data).
    """

    # Chinese equities
    SSE = "SSE"
    SZSE = "SZSE"
    BSE = "BSE"

    # Hong Kong
    SEHK = "SEHK"
    HKFE = "HKFE"

    # US
    SMART = "SMART"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"

    # Crypto (synthetic)
    BINANCE = "BINANCE"

    # Special
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"


# ---------------------------------------------------------------------------
# Active-status set
# ---------------------------------------------------------------------------

ACTIVE_STATUSES = {Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class BaseData:
    """Any data object carries ``gateway_name`` as its source.

    (vnpy's ``extra`` passthrough field is dropped — YiAgents has no use for it.)
    """

    gateway_name: str


# ---------------------------------------------------------------------------
# Order lifecycle
# ---------------------------------------------------------------------------


@dataclass
class OrderData(BaseData):
    """Tracks the latest status of a specific order."""

    symbol: str
    exchange: Exchange
    orderid: str

    type: OrderType = OrderType.LIMIT
    direction: Direction | None = None
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: Status = Status.SUBMITTING
    datetime: Datetime | None = None
    reference: str = ""

    def __post_init__(self) -> None:
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"

    def is_active(self) -> bool:
        """True if the order can still change (not filled / not final)."""
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> CancelRequest:
        """Build the cancel request targeting this order."""
        return CancelRequest(
            orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
        )


@dataclass
class TradeData(BaseData):
    """A single fill of an order (one order may have several trades)."""

    symbol: str
    exchange: Exchange
    orderid: str
    tradeid: str

    direction: Direction | None = None
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    datetime: Datetime | None = None

    def __post_init__(self) -> None:
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"
        self.vt_tradeid: str = f"{self.gateway_name}.{self.tradeid}"


# ---------------------------------------------------------------------------
# Account / position queries
# ---------------------------------------------------------------------------


@dataclass
class PositionData(BaseData):
    """A single position holding."""

    symbol: str
    exchange: Exchange
    direction: Direction

    volume: float = 0
    frozen: float = 0
    price: float = 0
    pnl: float = 0
    yd_volume: float = 0  # yesterday's volume (T+1 markets)

    def __post_init__(self) -> None:
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_positionid: str = (
            f"{self.gateway_name}.{self.vt_symbol}.{self.direction.value}"
        )


@dataclass
class AccountData(BaseData):
    """Account balance / frozen / available."""

    accountid: str

    balance: float = 0
    frozen: float = 0

    def __post_init__(self) -> None:
        self.available: float = self.balance - self.frozen
        self.vt_accountid: str = f"{self.gateway_name}.{self.accountid}"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


@dataclass
class OrderRequest:
    """Request to create a new order."""

    symbol: str
    exchange: Exchange
    direction: Direction
    type: OrderType
    volume: float

    price: float = 0
    offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self) -> None:
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_order_data(self, orderid: str, gateway_name: str) -> OrderData:
        """Instantiate the ``OrderData`` this request becomes once accepted."""
        return OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=orderid,
            type=self.type,
            direction=self.direction,
            offset=self.offset,
            price=self.price,
            volume=self.volume,
            reference=self.reference,
            gateway_name=gateway_name,
        )


@dataclass
class CancelRequest:
    """Request to cancel an existing order."""

    orderid: str
    symbol: str
    exchange: Exchange

    def __post_init__(self) -> None:
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
