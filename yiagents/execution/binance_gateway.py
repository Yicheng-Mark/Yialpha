"""BinanceGateway — concrete Track B execution gateway (official Binance SDK).

This is the first concrete gateway to land on the ``BaseGateway`` skeleton
(``yiagents/execution/gateway.py``). It drives Binance **order placement** via
the official modular SDK (``binance-sdk-spot`` / ``binance-sdk-derivatives-
trading-usds-futures``, both MIT, ``binance-common`` v4.1.0+). It is the
analysis layer's counterpart for turning a ``PortfolioDecision`` into a real
(or testnet) order.

Why now: the prior 2026-07-08 verdict ("SDK net-negative — zero SOCKS5
support") was overturned once ``binance-common`` v4.1.0 added first-class
proxy support (``ConfigurationRestAPI(proxy=...)`` -> ``utils.parse_proxies``
emits a requests-compatible dict; ``protocol`` is a free string so
``socks5h://`` works). The SDK is therefore usable behind this project's
1080 SOCKS5 VPN for the **execution** track (Track A analysis stays on the
hand-written read-only REST path — different concern).

Design invariants (every path preserves these):

* **Default-off == byte-equivalent.** ``YIAGENTS_EXECUTION_ENABLED`` defaults
  unset/off. When off, ``connect`` builds no client and ``send_order``
  returns ``REJECTED`` without touching the SDK. The module imports no SDK
  symbol at top level (lazy import inside ``connect``), so merely importing
  this module changes nothing — no SDK install required, no graph topology
  change, no agent-input change. The execution layer stays decoupled from
  ``yiagents/agents`` and ``yiagents/graph`` (guarded by
  ``tests/test_execution_isolation.py``).
* **Fail-closed.** Any uncertainty — switch off, missing keys, not
  connected, unsupported order type, ambiguous submit — yields a
  non-submitted ``OrderData`` (``Status.REJECTED``) or a raised
  ``VendorNotConfiguredError``. The *only* path to ``ALLTRADED`` /
  ``NOTTRADED`` is: switch on AND keys present AND client built AND
  ``new_order`` returned a non-error response.
* **Never blindly re-submit an order.** The SDK only auto-retries
  GET/DELETE transport errors; a ``new_order`` POST gets zero retry on any
  HTTP error. We do **not** paper over that: on an ambiguous submit failure
  (429 / 5xx / network — the order may or may not have reached the book) we
  query the order by its client id; if found, return the recovered status;
  if not found, return ``REJECTED`` rather than risk a duplicate fill. An IP
  ban (418) is never retried. Idempotency relies on ``new_client_order_id``.
* **Testnet-first.** ``base_path`` defaults to the SDK testnet URL constant;
  real mainnet requires ``YIAGENTS_EXECUTION_MAINNET=true``.
* **Keys never touch git.** ``BINANCE_API_KEY`` / ``BINANCE_API_SECRET`` are
  read straight from the environment (no ``YIAGENTS_`` prefix — matches the
  bare ``DEEPSEEK_API_KEY`` secret convention). ``.env`` is gitignored; only
  stub comment lines are added to ``.env.example``.

Scope (MVP): LIMIT / MARKET **entry** orders, query / cancel, account,
position. Out of scope (later): wiring into the LangGraph graph; user-data
stream live callbacks (``on_*``); futures conditional orders
(``STOP_MARKET`` / ``TRAILING_STOP_MARKET`` via ``new_algo_order``);
order-count budget throttling; ED25519 key auth.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from .browser_broker import _coerce_bool_env
from .domain import (
    AccountData,
    CancelRequest,
    Direction,
    Exchange,
    Offset,
    OrderData,
    OrderRequest,
    OrderType,
    PositionData,
    Status,
)
from .gateway import BaseGateway
from ..dataflows.errors import VendorNotConfiguredError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment knobs (read directly at call time — mirrors KillSwitch)
# ---------------------------------------------------------------------------

_ENV_ENABLED = "YIAGENTS_EXECUTION_ENABLED"
_ENV_MAINNET = "YIAGENTS_EXECUTION_MAINNET"
_ENV_KEY = "BINANCE_API_KEY"
_ENV_SECRET = "BINANCE_API_SECRET"
_ENV_TIMEOUT_MS = "YIAGENTS_EXECUTION_TIMEOUT_MS"

_DEFAULT_TIMEOUT_MS = 10000

# Binance order status string -> vnpy Status.
_BINANCE_STATUS_MAP = {
    "NEW": Status.NOTTRADED,
    "PARTIALLY_FILLED": Status.PARTTRADED,
    "FILLED": Status.ALLTRADED,
    "CANCELED": Status.CANCELLED,
    "CANCELLED": Status.CANCELLED,
    "PENDING_NEW": Status.SUBMITTING,
    "PENDING_CANCEL": Status.PARTTRADED,  # still working until confirmed
    "REJECTED": Status.REJECTED,
    "EXPIRED": Status.REJECTED,
}


class ExecutionEnableSwitch:
    """Reads ``YIAGENTS_EXECUTION_ENABLED`` straight from the environment.

    Mirrors ``browser_broker.KillSwitch`` but with the **opposite** fail-safe
    polarity: a *kill* switch fail-closed means malformed -> HALTED, whereas an
    *enable* switch fail-closed means malformed -> DISABLED. The default
    (unset / empty) is OFF — the execution track is opt-in.
    """

    _ENV_VAR = _ENV_ENABLED

    @staticmethod
    def is_enabled() -> bool:
        raw = os.environ.get(_ENV_ENABLED)
        try:
            return _coerce_bool_env(raw)
        except ValueError:
            # Malformed must NOT silently enable real trading.
            return False

    @staticmethod
    def reason() -> str:
        raw = os.environ.get(_ENV_ENABLED)
        if raw is None or raw.strip() == "":
            return f"{_ENV_ENABLED} is unset (execution disabled)."
        try:
            on = _coerce_bool_env(raw)
        except ValueError:
            return (
                f"{_ENV_ENABLED}={raw!r} is not a recognized boolean; "
                "treated as disabled (fail-closed)."
            )
        return f"{_ENV_ENABLED}={raw!r} -> execution {'ENABLED' if on else 'disabled'}."


# ---------------------------------------------------------------------------
# Response-normalisation helpers
# ---------------------------------------------------------------------------


def _as_dict(data):
    """Normalise an SDK response object to a plain (camelCase-keyed) dict.

    The SDK's generated pydantic models expose ``to_dict()`` which emits the
    JSON contract keys (camelCase) — those are the stable field names we read
    (``status``, ``executedQty``, ``orderId``, ``avgPrice`` ...). Falls back to
    ``model_dump(by_alias=True)`` or ``dict()``. Lists are mapped element-wise.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return [_as_dict(x) for x in data]
    if isinstance(data, dict):
        return data
    if hasattr(data, "to_dict"):
        try:
            return data.to_dict() or {}
        except Exception:  # noqa: BLE001 - best-effort coercion
            pass
    if hasattr(data, "model_dump"):
        try:
            return data.model_dump(by_alias=True) or {}
        except Exception:  # noqa: BLE001
            pass
    try:
        return dict(data)
    except Exception:  # noqa: BLE001
        return {}


def _to_float(value, default: float = 0.0) -> float:
    """Parse a Binance numeric field (stringly-typed) to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resp_data(resp):
    """Pull the parsed payload from an SDK response object (``.data()`` is a method)."""
    if resp is None:
        return None
    data_fn = getattr(resp, "data", None)
    if callable(data_fn):
        try:
            return data_fn()
        except Exception:  # noqa: BLE001
            return None
    return resp


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class BinanceGateway(BaseGateway):
    """Concrete Track B gateway placing Binance orders via the official SDK.

    Handles both ``crypto_perp`` (USDT-M futures) and ``crypto_spot`` product
    lines, selected by the ``product`` setting. Construction and the enable
    switch are cheap and side-effect-free; the SDK is imported and the network
    session built only inside :meth:`connect` once the switch is on and keys
    are present.
    """

    default_name = "BINANCE"
    exchanges = [Exchange.BINANCE]
    default_setting = {
        "product": "perp",  # "perp" | "spot"
        "proxies": None,  # reserved (BaseGateway); we read HTTPS_PROXY env instead
        "mainnet": False,  # default testnet; mainnet requires explicit opt-in
    }

    def __init__(self, gateway_name: str = default_name, *, setting: dict | None = None) -> None:
        super().__init__(gateway_name, setting=setting)
        self._product: str = str(self.setting.get("product", "perp")).lower()
        self._mainnet: bool = bool(self.setting.get("mainnet", False))
        self._client = None  # SDK facade; built in connect()
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @staticmethod
    def _proxy_dict() -> dict | None:
        """Build the SDK ``proxy={protocol,host,port}`` from the env SOCKS5 URL.

        The project carries the VPN as ``HTTPS_PROXY=socks5h://127.0.0.1:1080``
        in ``.env``; we split it back into the SDK's dict shape so the gateway
        is consistent with the analysis layer's transport without hard-coding
        an endpoint here.
        """
        url = os.environ.get("HTTPS_PROXY") or os.environ.get("ALL_PROXY")
        if not url:
            return None
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return None
        proxy: dict = {"protocol": parsed.scheme or "socks5h", "host": host}
        if parsed.port:
            proxy["port"] = parsed.port
        return proxy

    def connect(self, setting: dict | None = None) -> None:
        """Build the SDK client, gated by the enable switch and trading keys.

        Fail-closed: switch off -> inert (no client); keys missing -> raise
        ``VendorNotConfiguredError`` so a misconfiguration is loud.
        """
        if setting:
            self.setting.update(setting)
            self._product = str(self.setting.get("product", self._product)).lower()
            self._mainnet = bool(self.setting.get("mainnet", self._mainnet))

        if not ExecutionEnableSwitch.is_enabled():
            logger.warning(
                "BinanceGateway inert: %s off (fail-closed). %s",
                _ENV_ENABLED,
                ExecutionEnableSwitch.reason(),
            )
            self._client = None
            self._connected = False
            return

        api_key = os.environ.get(_ENV_KEY)
        api_secret = os.environ.get(_ENV_SECRET)
        if not (api_key and api_secret):
            raise VendorNotConfiguredError(
                f"{_ENV_ENABLED} is on but {_ENV_KEY}/{_ENV_SECRET} are not set; "
                "cannot build the Binance execution client."
            )

        self._client = self._build_client(api_key, api_secret)
        self._connected = True
        logger.info(
            "BinanceGateway connected: product=%s, %s",
            self._product,
            "MAINNET" if self._mainnet else "testnet",
        )

    def _build_client(self, api_key: str, api_secret: str):
        """Lazily import the SDK and construct the per-product facade.

        Imported here (not at module top) so the module is importable without
        the SDK installed and so unit tests can inject a mock client without
        any SDK on disk.
        """
        try:
            timeout_ms = int(os.environ.get(_ENV_TIMEOUT_MS, _DEFAULT_TIMEOUT_MS))
        except ValueError:
            timeout_ms = _DEFAULT_TIMEOUT_MS
        proxy = self._proxy_dict()
        mainnet = self._mainnet or _coerce_bool_env(os.environ.get(_ENV_MAINNET)) is True

        if self._product == "spot":
            from binance_common.configuration import ConfigurationRestAPI  # type: ignore
            from binance_common.constants import (  # type: ignore
                SPOT_REST_API_PROD_URL,
                SPOT_REST_API_TESTNET_URL,
            )
            from binance_sdk_spot.spot import Spot  # type: ignore

            base = SPOT_REST_API_PROD_URL if mainnet else SPOT_REST_API_TESTNET_URL
            cfg = ConfigurationRestAPI(
                api_key=api_key,
                api_secret=api_secret,
                base_path=base,
                proxy=proxy,
                timeout=timeout_ms,
            )
            return Spot(config_rest_api=cfg)

        if self._product == "perp":
            from binance_common.configuration import ConfigurationRestAPI  # type: ignore
            from binance_common.constants import (  # type: ignore
                DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
                DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
            )
            from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (  # type: ignore
                DerivativesTradingUsdsFutures,
            )

            base = (
                DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
                if mainnet
                else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
            )
            cfg = ConfigurationRestAPI(
                api_key=api_key,
                api_secret=api_secret,
                base_path=base,
                proxy=proxy,
                timeout=timeout_ms,
            )
            return DerivativesTradingUsdsFutures(config_rest_api=cfg)

        raise ValueError(
            f"Unsupported Binance product {self._product!r}; expected 'spot' or 'perp'."
        )

    def close(self) -> None:
        """Release the SDK session. Safe to call when never connected."""
        self._client = None
        self._connected = False

    def _require_client(self):
        """Return the client or raise — used by explicit query methods."""
        if self._client is None:
            raise VendorNotConfiguredError(
                "BinanceGateway has no client (disabled or connect() not called)."
            )
        return self._client

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def send_order(self, req: OrderRequest) -> OrderData:
        """Submit ``req`` and return the resulting ``OrderData`` synchronously.

        Fail-closed: not connected / unsupported type -> ``REJECTED``. On an
        ambiguous submit error (the order may have reached the book) we
        **query** rather than re-submit, to avoid duplicate fills.
        """
        client = self._client
        if client is None:
            logger.warning(
                "BinanceGateway.send_order(%s): not connected -> REJECTED (fail-closed).",
                req.symbol,
            )
            return self._rejected(req, reason="gateway not connected")

        try:
            side, sdk_type, extra = self._map_order(req)
        except ValueError as exc:
            logger.warning("BinanceGateway.send_order(%s): %s", req.symbol, exc)
            return self._rejected(req, reason=str(exc))

        client_order_id = req.reference or self._gen_client_order_id()
        kwargs = dict(
            symbol=req.symbol,
            side=side,
            type=sdk_type,
            new_client_order_id=client_order_id,
            new_order_resp_type="RESULT",
        )
        kwargs.update(extra)

        try:
            resp = client.rest_api.new_order(**kwargs)
        except Exception as exc:  # noqa: BLE001 - all submit errors funnel here
            return self._handle_submit_error(req, client_order_id, exc)

        data = _as_dict(_resp_data(resp)) or {}
        return self._order_from_response(req, data, client_order_id)

    @staticmethod
    def _gen_client_order_id() -> str:
        return f"yiagents-{int(time.time() * 1000)}-{os.getpid()}"

    def _map_order(self, req: OrderRequest) -> tuple[str, str, dict]:
        """Translate an :class:`OrderRequest` to SDK ``new_order`` arguments.

        Returns ``(side, sdk_type, extra_kwargs)``. LIMIT/MARKET only; STOP
        and friends raise (conditional orders need ``new_algo_order`` on
        futures — out of MVP scope).
        """
        if req.direction == Direction.LONG:
            side = "BUY"
        elif req.direction == Direction.SHORT:
            side = "SELL"
        else:
            raise ValueError(f"direction {req.direction!r} is not placeable on Binance")

        extra: dict = {}
        if self._product == "perp":
            # One-way (hedge-off) mode. Closing a position uses reduce_only.
            extra["position_side"] = "BOTH"
            extra["reduce_only"] = "true" if req.offset == Offset.CLOSE else "false"

        if req.type == OrderType.MARKET:
            sdk_type = "MARKET"
            extra["quantity"] = req.volume
        elif req.type == OrderType.LIMIT:
            sdk_type = "LIMIT"
            extra["time_in_force"] = "GTC"
            extra["quantity"] = req.volume
            extra["price"] = req.price
        else:
            raise ValueError(
                f"order type {req.type!r} unsupported in MVP "
                "(LIMIT/MARKET only; conditional stops need new_algo_order)"
            )
        return side, sdk_type, extra

    def _handle_submit_error(
        self, req: OrderRequest, client_order_id: str, exc: BaseException
    ) -> OrderData:
        """Recover from a ``new_order`` exception WITHOUT a blind re-submit.

        * IP ban (418 / ``RateLimitBanError``): never retry -> REJECTED.
        * Transient (429 / 5xx / network): the order may already be live, so
          query by client id; found -> return recovered status, not found ->
          REJECTED (no duplicate). We never re-POST.
        * Anything else (bad request / auth): -> REJECTED.
        """
        name = type(exc).__name__
        status_code = getattr(exc, "status_code", "")
        body = getattr(exc, "error_message", "") or str(exc)

        if "RateLimitBan" in name or str(status_code) == "418":
            logger.error(
                "send_order(%s): Binance IP ban (418) -> REJECTED, not retried. %s",
                req.symbol,
                exc,
            )
            return self._rejected(req, reason=f"IP ban 418: {body}")

        transient = name in {"TooManyRequestsError", "ServerError", "NetworkError"} or (
            isinstance(status_code, int) and status_code >= 500
        )
        if transient:
            recovered = self._safe_query_by_client_id(req, client_order_id)
            if recovered is not None:
                logger.warning(
                    "send_order(%s): submit raised %s; recovered via query as %s.",
                    req.symbol,
                    name,
                    recovered.status.name,
                )
                return recovered
            logger.error(
                "send_order(%s): submit raised %s and query could not confirm -> "
                "REJECTED (no blind re-submit). %s",
                req.symbol,
                name,
                exc,
            )
            return self._rejected(req, reason=f"ambiguous submit ({name}); not resubmitted")

        logger.warning("send_order(%s): %s -> REJECTED. %s", req.symbol, name, exc)
        return self._rejected(req, reason=f"{name}: {body}")

    def _safe_query_by_client_id(
        self, req: OrderRequest, client_order_id: str
    ) -> OrderData | None:
        """Best-effort order lookup by client id; None on any failure/absence."""
        client = self._client
        if client is None:
            return None
        try:
            if self._product == "perp":
                resp = client.rest_api.query_order(
                    symbol=req.symbol, orig_client_order_id=client_order_id
                )
            else:
                resp = client.rest_api.get_order(
                    symbol=req.symbol, orig_client_order_id=client_order_id
                )
            data = _as_dict(_resp_data(resp)) or {}
            orderid = data.get("orderId")
            if orderid is None and not data:
                return None
            return self._order_from_response(req, data, client_order_id)
        except Exception:  # noqa: BLE001 - recovery must never raise
            return None

    def _order_from_response(
        self, req: OrderRequest, data: dict, client_order_id: str
    ) -> OrderData:
        """Build the ``OrderData`` from a normalised (camelCase) response dict."""
        binance_status = str(data.get("status", "")).upper()
        orderid = str(data.get("orderId") or client_order_id)
        status = _BINANCE_STATUS_MAP.get(binance_status, Status.SUBMITTING)
        traded = _to_float(data.get("executedQty") or data.get("cumQty"))
        fill_price = _to_float(data.get("avgPrice") or data.get("price"))

        order = req.create_order_data(orderid=orderid, gateway_name=self.gateway_name)
        order.status = status
        order.traded = traded
        if fill_price:
            order.price = fill_price
        update_time = data.get("updateTime")
        if update_time:
            try:
                order.datetime = datetime.fromtimestamp(int(update_time) / 1000, tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                pass
        return order

    def _rejected(self, req: OrderRequest, *, orderid: str = "", reason: str = "") -> OrderData:
        """Build a ``REJECTED`` OrderData, logging the reason (OrderData has no msg field)."""
        logger.warning("BinanceGateway reject [%s]: %s", req.symbol, reason)
        order = req.create_order_data(
            orderid=orderid or f"rej-{int(time.time() * 1000)}",
            gateway_name=self.gateway_name,
        )
        order.status = Status.REJECTED
        return order

    # ------------------------------------------------------------------
    # Cancel / query
    # ------------------------------------------------------------------

    def cancel_order(self, req: CancelRequest) -> None:
        """Cancel by broker order id; raises on failure (explicit action)."""
        client = self._require_client()
        try:
            if self._product == "perp":
                client.rest_api.cancel_order(symbol=req.symbol, order_id=_as_order_id(req.orderid))
            else:
                client.rest_api.delete_order(
                    symbol=req.symbol, order_id=_as_order_id(req.orderid)
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel_order(%s %s) failed: %s", req.symbol, req.orderid, exc)
            raise

    def query_account(self) -> AccountData:
        """Return the USDT balance snapshot (single-asset AccountData)."""
        client = self._require_client()
        if self._product == "perp":
            resp = client.rest_api.futures_account_balance_v2()
            items = _as_dict(_resp_data(resp)) or []
            for item in items:
                d = _as_dict(item)
                if str(d.get("asset", "")).upper() == "USDT":
                    balance = _to_float(d.get("balance") or d.get("walletBalance"))
                    available = _to_float(d.get("availableBalance") or d.get("maxWithdrawAmount"))
                    return AccountData(
                        gateway_name=self.gateway_name,
                        accountid="USDT",
                        balance=balance,
                        frozen=max(balance - available, 0.0),
                    )
            return AccountData(gateway_name=self.gateway_name, accountid="USDT")
        # spot
        resp = client.rest_api.get_account()
        data = _as_dict(_resp_data(resp)) or {}
        for b in data.get("balances") or []:
            d = _as_dict(b)
            if str(d.get("asset", "")).upper() == "USDT":
                free = _to_float(d.get("free"))
                locked = _to_float(d.get("locked"))
                return AccountData(
                    gateway_name=self.gateway_name,
                    accountid="USDT",
                    balance=free + locked,
                    frozen=locked,
                )
        return AccountData(gateway_name=self.gateway_name, accountid="USDT")

    def query_position(self) -> list[PositionData]:
        """Return open positions. Spot has no position concept -> ``[]``."""
        client = self._require_client()
        if self._product != "perp":
            return []
        resp = client.rest_api.position_information_v3()
        items = _as_dict(_resp_data(resp)) or []
        out: list[PositionData] = []
        for item in items:
            d = _as_dict(item)
            amount = _to_float(d.get("positionAmt"))
            if abs(amount) < 1e-12:
                continue  # skip flat rows
            pside = str(d.get("positionSide", "")).upper()
            if pside == "LONG":
                direction = Direction.LONG
            elif pside == "SHORT":
                direction = Direction.SHORT
            else:  # BOTH — one-way mode
                direction = Direction.NET
            out.append(
                PositionData(
                    gateway_name=self.gateway_name,
                    symbol=str(d.get("symbol", "")),
                    exchange=Exchange.BINANCE,
                    direction=direction,
                    volume=abs(amount),
                    price=_to_float(d.get("entryPrice")),
                    pnl=_to_float(d.get("unRealizedProfit")),
                )
            )
        return out


def _as_order_id(orderid: str):
    """Coerce a broker order id to int when possible (SDK futures expects int)."""
    try:
        return int(orderid)
    except (TypeError, ValueError):
        return orderid
