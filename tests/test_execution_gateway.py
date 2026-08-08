"""Unit tests for ``yiagents.execution.binance_gateway``.

All tests are network-free: the official Binance SDK is never imported (the
gateway imports it lazily inside ``connect``), and a mock client is injected
directly onto the gateway instance. These tests pin the fail-closed contract,
the OrderRequest -> SDK mapping, the Binance-status -> vnpy-Status table, and
the no-blind-resubmit safety invariant.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

from yiagents.dataflows.errors import VendorNotConfiguredError
from yiagents.execution.binance_gateway import (
    BinanceGateway,
    ExecutionEnableSwitch,
    _as_order_id,
)
from yiagents.execution.domain import (
    Direction,
    Exchange,
    Offset,
    OrderRequest,
    OrderType,
    Status,
)

pytestmark = pytest.mark.unit

_ENV_ENABLED = "YIAGENTS_EXECUTION_ENABLED"
_ENV_KEY = "BINANCE_API_KEY"
_ENV_SECRET = "BINANCE_API_SECRET"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResp:
    """Stand-in for the SDK ``ApiResponse``: ``.data()`` is a method."""

    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


# Exception classes whose ``__name__`` matches the SDK's so the gateway's
# type-name routing in ``_handle_submit_error`` exercises the real branches.
class NetworkError(Exception):
    def __init__(self, msg="net", status_code=None, error_message=""):
        super().__init__(msg)
        self.status_code = status_code
        self.error_message = error_message


class ServerError(Exception):
    def __init__(self, msg="srv", status_code=500, error_message=""):
        super().__init__(msg)
        self.status_code = status_code
        self.error_message = error_message


class TooManyRequestsError(Exception):
    def __init__(self, msg="429", status_code=429, error_message=""):
        super().__init__(msg)
        self.status_code = status_code
        self.error_message = error_message


class RateLimitBanError(Exception):
    def __init__(self, msg="418", status_code=418, error_message=""):
        super().__init__(msg)
        self.status_code = status_code
        self.error_message = error_message


class BadRequestError(Exception):
    def __init__(self, msg="bad", status_code=400, error_message="bad"):
        super().__init__(msg)
        self.status_code = status_code
        self.error_message = error_message


def _req(
    symbol="BTCUSDT",
    direction=Direction.LONG,
    otype=OrderType.MARKET,
    volume=0.5,
    price=0.0,
    offset=Offset.NONE,
    reference="",
):
    return OrderRequest(
        symbol=symbol,
        exchange=Exchange.BINANCE,
        direction=direction,
        type=otype,
        volume=volume,
        price=price,
        offset=offset,
        reference=reference,
    )


def _perp_gw_with_client():
    """A perp gateway with a mock SDK client injected (bypasses connect)."""
    gw = BinanceGateway()
    gw._client = MagicMock()
    return gw


# ---------------------------------------------------------------------------
# Enable switch
# ---------------------------------------------------------------------------


class TestExecutionEnableSwitch:
    @pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "Yes"])
    def test_truthy_enables(self, monkeypatch, val):
        monkeypatch.setenv(_ENV_ENABLED, val)
        assert ExecutionEnableSwitch.is_enabled() is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "off"])
    def test_falsy_disables(self, monkeypatch, val):
        monkeypatch.setenv(_ENV_ENABLED, val)
        assert ExecutionEnableSwitch.is_enabled() is False

    def test_unset_disables(self, monkeypatch):
        monkeypatch.delenv(_ENV_ENABLED, raising=False)
        assert ExecutionEnableSwitch.is_enabled() is False

    def test_malformed_disables_fail_closed(self, monkeypatch):
        # Opposite polarity to KillSwitch: a typo must NOT enable trading.
        monkeypatch.setenv(_ENV_ENABLED, "treu")
        assert ExecutionEnableSwitch.is_enabled() is False

    def test_reason_mentions_state(self, monkeypatch):
        monkeypatch.delenv(_ENV_ENABLED, raising=False)
        assert "disabled" in ExecutionEnableSwitch.reason().lower()


# ---------------------------------------------------------------------------
# Construction & proxy
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        gw = BinanceGateway()
        assert gw.default_name == "BINANCE"
        assert gw.exchanges == [Exchange.BINANCE]
        assert gw._product == "perp"
        assert gw._mainnet is False
        assert gw._client is None
        assert gw._connected is False

    def test_setting_overrides(self):
        gw = BinanceGateway(setting={"product": "spot", "mainnet": True})
        assert gw._product == "spot"
        assert gw._mainnet is True

    @pytest.mark.parametrize(
        "url,protocol,host,port",
        [
            ("socks5h://127.0.0.1:1080", "socks5h", "127.0.0.1", 1080),
            ("socks5://10.0.0.1:7890", "socks5", "10.0.0.1", 7890),
            ("http://proxy.example.com:8080", "http", "proxy.example.com", 8080),
        ],
    )
    def test_proxy_dict_parses_env(self, monkeypatch, url, protocol, host, port):
        monkeypatch.setenv("HTTPS_PROXY", url)
        d = BinanceGateway._proxy_dict()
        assert d == {"protocol": protocol, "host": host, "port": port}

    def test_proxy_dict_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("ALL_PROXY", raising=False)
        assert BinanceGateway._proxy_dict() is None


# ---------------------------------------------------------------------------
# connect() fail-closed
# ---------------------------------------------------------------------------


class TestConnect:
    def test_switch_off_leaves_inert(self, monkeypatch):
        monkeypatch.delenv(_ENV_ENABLED, raising=False)
        monkeypatch.setenv(_ENV_KEY, "k")
        monkeypatch.setenv(_ENV_SECRET, "s")
        gw = BinanceGateway()
        gw.connect()  # no raise, no client
        assert gw._client is None
        assert gw._connected is False

    def test_missing_keys_raises(self, monkeypatch):
        monkeypatch.setenv(_ENV_ENABLED, "true")
        monkeypatch.delenv(_ENV_KEY, raising=False)
        monkeypatch.delenv(_ENV_SECRET, raising=False)
        gw = BinanceGateway()
        with pytest.raises(VendorNotConfiguredError):
            gw.connect()
        assert gw._client is None

    def test_on_with_keys_builds_client(self, monkeypatch):
        monkeypatch.setenv(_ENV_ENABLED, "true")
        monkeypatch.setenv(_ENV_KEY, "k")
        monkeypatch.setenv(_ENV_SECRET, "s")
        gw = BinanceGateway()
        gw._build_client = MagicMock(return_value="FAKE_CLIENT")  # avoid SDK import
        gw.connect()
        assert gw._client == "FAKE_CLIENT"
        assert gw._connected is True
        gw._build_client.assert_called_once()

    def test_close_releases(self, monkeypatch):
        gw = BinanceGateway()
        gw._client = MagicMock()
        gw._connected = True
        gw.close()
        assert gw._client is None
        assert gw._connected is False


# ---------------------------------------------------------------------------
# send_order mapping
# ---------------------------------------------------------------------------


class TestSendOrderMapping:
    def test_limit_long_perp(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.return_value = _FakeResp(
            {"status": "NEW", "orderId": 100, "executedQty": "0", "updateTime": 1700000000000}
        )
        order = gw.send_order(_req(otype=OrderType.LIMIT, price=60000.0, reference="coid-1"))
        kw = gw._client.rest_api.new_order.call_args.kwargs
        assert kw["symbol"] == "BTCUSDT"
        assert kw["side"] == "BUY"
        assert kw["type"] == "LIMIT"
        assert kw["time_in_force"] == "GTC"
        assert kw["quantity"] == 0.5
        assert kw["price"] == 60000.0
        assert kw["position_side"] == "BOTH"
        assert kw["reduce_only"] == "false"
        assert kw["new_client_order_id"] == "coid-1"
        assert kw["new_order_resp_type"] == "RESULT"
        # status mapping + derived fields
        assert order.status is Status.NOTTRADED
        assert order.orderid == "100"
        assert order.vt_orderid == "BINANCE.100"
        assert order.vt_symbol == "BTCUSDT.BINANCE"

    def test_market_short_close_reduce_only(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.return_value = _FakeResp(
            {"status": "FILLED", "orderId": 101, "executedQty": "0.5", "avgPrice": "61000"}
        )
        order = gw.send_order(
            _req(direction=Direction.SHORT, otype=OrderType.MARKET, offset=Offset.CLOSE)
        )
        kw = gw._client.rest_api.new_order.call_args.kwargs
        assert kw["side"] == "SELL"
        assert kw["type"] == "MARKET"
        assert "time_in_force" not in kw
        assert "price" not in kw
        assert kw["reduce_only"] == "true"
        assert order.status is Status.ALLTRADED
        assert order.traded == pytest.approx(0.5)
        assert order.price == pytest.approx(61000.0)

    def test_spot_has_no_position_side(self):
        gw = _perp_gw_with_client()
        gw._product = "spot"
        gw._client.rest_api.new_order.return_value = _FakeResp(
            {"status": "FILLED", "orderId": 102, "executedQty": "1", "avgPrice": "300"}
        )
        gw.send_order(_req(symbol="BNBUSDT", otype=OrderType.MARKET))
        kw = gw._client.rest_api.new_order.call_args.kwargs
        assert "position_side" not in kw
        assert "reduce_only" not in kw

    def test_response_data_is_method(self):
        # Ensure we call .data() rather than reading .data as an attribute.
        gw = _perp_gw_with_client()
        resp = MagicMock()
        resp.data.return_value = {"status": "FILLED", "orderId": 1, "executedQty": "0.5"}
        gw._client.rest_api.new_order.return_value = resp
        order = gw.send_order(_req())
        assert order.status is Status.ALLTRADED
        resp.data.assert_called_once()


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------


class TestStatusMapping:
    @pytest.mark.parametrize(
        "binance,expected",
        [
            ("NEW", Status.NOTTRADED),
            ("PARTIALLY_FILLED", Status.PARTTRADED),
            ("FILLED", Status.ALLTRADED),
            ("CANCELED", Status.CANCELLED),
            ("CANCELLED", Status.CANCELLED),
            ("REJECTED", Status.REJECTED),
            ("EXPIRED", Status.REJECTED),
        ],
    )
    def test_mapping(self, binance, expected):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.return_value = _FakeResp(
            {"status": binance, "orderId": 9, "executedQty": "0"}
        )
        assert gw.send_order(_req()).status is expected


# ---------------------------------------------------------------------------
# send_order fail-closed
# ---------------------------------------------------------------------------


class TestSendOrderFailClosed:
    def test_not_connected_rejected(self):
        gw = BinanceGateway()  # no client
        order = gw.send_order(_req())
        assert order.status is Status.REJECTED
        assert order.orderid.startswith("rej-")

    def test_unsupported_type_rejected(self):
        gw = _perp_gw_with_client()
        order = gw.send_order(_req(otype=OrderType.STOP))
        assert order.status is Status.REJECTED
        gw._client.rest_api.new_order.assert_not_called()

    def test_net_direction_rejected(self):
        gw = _perp_gw_with_client()
        order = gw.send_order(_req(direction=Direction.NET))
        assert order.status is Status.REJECTED
        gw._client.rest_api.new_order.assert_not_called()


# ---------------------------------------------------------------------------
# No-blind-resubmit safety (the core real-money invariant)
# ---------------------------------------------------------------------------


class TestNoBlindResubmit:
    def test_transient_recovers_via_query(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = NetworkError("timeout")
        gw._client.rest_api.query_order.return_value = _FakeResp(
            {"status": "NEW", "orderId": 555, "executedQty": "0", "updateTime": 1700000000000}
        )
        order = gw.send_order(_req(reference="coid-x"))
        # new_order attempted exactly once — never re-POSTed.
        assert gw._client.rest_api.new_order.call_count == 1
        gw._client.rest_api.query_order.assert_called_once_with(
            symbol="BTCUSDT", orig_client_order_id="coid-x"
        )
        assert order.status is Status.NOTTRADED  # recovered, not rejected

    def test_transient_unconfirmed_rejected(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = ServerError("500")
        gw._client.rest_api.query_order.side_effect = Exception("not found")
        order = gw.send_order(_req(reference="coid-y"))
        assert gw._client.rest_api.new_order.call_count == 1
        assert order.status is Status.REJECTED  # NOT resubmitted

    def test_ip_ban_never_qued_or_retried(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = RateLimitBanError("418", status_code=418)
        order = gw.send_order(_req())
        assert gw._client.rest_api.new_order.call_count == 1
        gw._client.rest_api.query_order.assert_not_called()
        assert order.status is Status.REJECTED

    def test_bad_request_rejected_no_query(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = BadRequestError("bad params")
        order = gw.send_order(_req())
        gw._client.rest_api.query_order.assert_not_called()
        assert order.status is Status.REJECTED


# ---------------------------------------------------------------------------
# query_account / query_position / cancel
# ---------------------------------------------------------------------------


class TestQueries:
    def test_not_connected_query_raises(self):
        gw = BinanceGateway()
        with pytest.raises(VendorNotConfiguredError):
            gw.query_account()
        with pytest.raises(VendorNotConfiguredError):
            gw.query_position()

    def test_perp_account_usdt(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.futures_account_balance_v2.return_value = _FakeResp(
            [{"asset": "USDT", "balance": "1000", "availableBalance": "800"}]
        )
        acct = gw.query_account()
        assert acct.balance == pytest.approx(1000.0)
        assert acct.frozen == pytest.approx(200.0)
        assert acct.available == pytest.approx(800.0)
        assert acct.vt_accountid == "BINANCE.USDT"

    def test_spot_account_usdt(self):
        gw = _perp_gw_with_client()
        gw._product = "spot"
        gw._client.rest_api.get_account.return_value = _FakeResp(
            {"balances": [{"asset": "USDT", "free": "500", "locked": "50"}]}
        )
        acct = gw.query_account()
        assert acct.balance == pytest.approx(550.0)
        assert acct.frozen == pytest.approx(50.0)

    def test_perp_position_skips_flat_maps_directions(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.position_information_v3.return_value = _FakeResp(
            [
                {"symbol": "BTCUSDT", "positionAmt": "0.5", "positionSide": "LONG",
                 "entryPrice": "60000", "unRealizedProfit": "10"},
                {"symbol": "ETHUSDT", "positionAmt": "-2", "positionSide": "SHORT",
                 "entryPrice": "3000", "unRealizedProfit": "-5"},
                {"symbol": "SOLUSDT", "positionAmt": "0", "positionSide": "BOTH",
                 "entryPrice": "0", "unRealizedProfit": "0"},  # flat -> skipped
            ]
        )
        positions = gw.query_position()
        assert len(positions) == 2
        btc = next(p for p in positions if p.symbol == "BTCUSDT")
        assert btc.direction is Direction.LONG
        assert btc.volume == pytest.approx(0.5)
        assert btc.price == pytest.approx(60000.0)
        eth = next(p for p in positions if p.symbol == "ETHUSDT")
        assert eth.direction is Direction.SHORT
        assert eth.vt_positionid == "BINANCE.ETHUSDT.BINANCE.short"

    def test_spot_position_empty(self):
        gw = _perp_gw_with_client()
        gw._product = "spot"
        assert gw.query_position() == []


class TestCancel:
    def test_perp_cancel_coerces_int(self):
        gw = _perp_gw_with_client()
        from yiagents.execution.domain import CancelRequest

        gw.cancel_order(CancelRequest(orderid="777", symbol="BTCUSDT", exchange=Exchange.BINANCE))
        gw._client.rest_api.cancel_order.assert_called_once_with(symbol="BTCUSDT", order_id=777)

    def test_spot_cancel_uses_delete(self):
        gw = _perp_gw_with_client()
        gw._product = "spot"
        from yiagents.execution.domain import CancelRequest

        gw.cancel_order(CancelRequest(orderid="777", symbol="BTCUSDT", exchange=Exchange.BINANCE))
        gw._client.rest_api.delete_order.assert_called_once_with(symbol="BTCUSDT", order_id=777)

    def test_as_order_id_fallback(self):
        assert _as_order_id("123") == 123
        assert _as_order_id("rej-1") == "rej-1"


# ---------------------------------------------------------------------------
# Byte-equivalence invariant: no SDK import at module load
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_module_imports_without_sdk(self):
        # Importing the gateway module must NOT pull the (uninstalled) SDK —
        # that is what makes default-off == byte-equivalent.
        for mod in (
            "binance_sdk_spot",
            "binance_sdk_derivatives_trading_usds_futures",
            "binance_common",
        ):
            sys.modules.pop(mod, None)
        import importlib

        import yiagents.execution.binance_gateway as bg

        importlib.reload(bg)
        assert "binance_sdk_spot" not in sys.modules
        assert "binance_sdk_derivatives_trading_usds_futures" not in sys.modules
        assert "binance_common" not in sys.modules


# ---------------------------------------------------------------------------
# Regression tests for the 2026-08-08 hardening pass.
#
# Each test pins a fix whose pre-fix behaviour was wrong (the assertion would
# fail against the old code). They live here so the fail-closed contract stays
# auditable in one place.
# ---------------------------------------------------------------------------


class TestRateLimit429Detection:
    """A 429 with an SDK exception class name we do not hardcode must still be
    treated as transient (-> query recovery), not as a deterministic REJECT.

    Pre-fix, only ``status_code >= 500`` and an exact class-name set were
    recognised, so a 429 raised as e.g. ``BinanceAPIException`` bypassed the
    query path and was logged as REJECTED — risking a lost fill.
    """

    def test_429_with_unrecognized_class_name_recovers_via_query(self):
        # An exception type whose name is NOT in the hardcoded set and whose
        # status_code is 429 (not >= 500).
        class BinanceAPIException(Exception):
            def __init__(self, msg="rate limited", status_code=429, error_message=""):
                super().__init__(msg)
                self.status_code = status_code
                self.error_message = error_message

        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = BinanceAPIException()
        gw._client.rest_api.query_order.return_value = _FakeResp(
            {"status": "NEW", "orderId": 888, "executedQty": "0", "updateTime": 1700000000000}
        )
        order = gw.send_order(_req(reference="coid-429"))
        assert gw._client.rest_api.new_order.call_count == 1
        gw._client.rest_api.query_order.assert_called_once_with(
            symbol="BTCUSDT", orig_client_order_id="coid-429"
        )
        assert order.status is Status.NOTTRADED  # recovered, not rejected

    def test_429_with_unrecognized_class_name_unconfirmed_rejected(self):
        class BinanceAPIException(Exception):
            def __init__(self, msg="rate limited", status_code=429, error_message=""):
                super().__init__(msg)
                self.status_code = status_code
                self.error_message = error_message

        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = BinanceAPIException()
        gw._client.rest_api.query_order.return_value = _FakeResp(
            {"code": -2013, "msg": "Order does not exist."}
        )
        order = gw.send_order(_req(reference="coid-429b"))
        assert order.status is Status.REJECTED  # not resubmitted
        assert gw._client.rest_api.new_order.call_count == 1

    def test_rate_limit_named_exception_detected_via_substring(self):
        # A rate-limit exception whose class name contains "rate"/"limit"
        # but is NOT exactly "TooManyRequestsError" and carries no status_code.
        class RateLimitError(Exception):
            def __init__(self, msg="rl"):
                super().__init__(msg)
                self.status_code = ""
                self.error_message = ""

        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = RateLimitError()
        gw._client.rest_api.query_order.side_effect = Exception("not found")
        order = gw.send_order(_req(reference="coid-rl"))
        # Must have queried (transient path), not treated as a hard reject.
        gw._client.rest_api.query_order.assert_called_once()
        assert order.status is Status.REJECTED  # unconfirmed -> no resubmit


class TestSafeQueryErrorPayload:
    """An error payload (non-empty, no ``orderId``) must return None, not fall
    through to ``_order_from_response`` which defaults unknown status to
    SUBMITTING.

    Pre-fix, the guard was ``orderid is None and not data``, so a body like
    ``{"code": -2013, "msg": "Order does not exist."}`` (non-empty but no
    orderId) slipped through and was reported as SUBMITTING — a live state.
    """

    def test_error_payload_returns_none(self):
        gw = _perp_gw_with_client()
        # Drive through _safe_query_by_client_id via a transient submit error.
        gw._client.rest_api.new_order.side_effect = NetworkError("net", status_code=500)
        gw._client.rest_api.query_order.return_value = _FakeResp(
            {"code": -2013, "msg": "Order does not exist."}
        )
        order = gw.send_order(_req(reference="coid-err"))
        # Query returned an error payload -> treated as "not found" -> REJECTED.
        assert order.status is Status.REJECTED

    def test_empty_payload_returns_none(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = NetworkError("net", status_code=500)
        gw._client.rest_api.query_order.return_value = _FakeResp({})
        order = gw.send_order(_req(reference="coid-empty"))
        assert order.status is Status.REJECTED


class TestGenClientOrderIdUniqueness:
    """``_gen_client_order_id`` must not collide within a single process/millisecond."""

    def test_unique_under_burst(self):
        ids = {BinanceGateway._gen_client_order_id() for _ in range(1000)}
        assert len(ids) == 1000  # no collisions

    def test_within_binance_length_limit(self):
        # Binance new_client_order_id cap is 36 characters.
        for _ in range(100):
            assert len(BinanceGateway._gen_client_order_id()) <= 36

    def test_starts_with_namespace(self):
        assert BinanceGateway._gen_client_order_id().startswith("yiagents-")


class TestMalformedMainnetEnvFailsClosed:
    """A malformed ``YIAGENTS_EXECUTION_MAINNET`` must not raise from ``connect``."""

    def test_garbage_mainnet_env_does_not_raise(self, monkeypatch):
        monkeypatch.setenv(_ENV_ENABLED, "true")
        monkeypatch.setenv(_ENV_KEY, "k")
        monkeypatch.setenv(_ENV_SECRET, "s")
        monkeypatch.setenv("YIAGENTS_EXECUTION_MAINNET", "garbage")
        gw = BinanceGateway()
        gw._build_client = MagicMock(return_value="FAKE_CLIENT")  # avoid SDK import
        # Pre-fix this raised ValueError from _coerce_bool_env.
        gw.connect()
        # Constructor default (False -> testnet) preserved.
        assert gw._mainnet is False

    def test_garbage_mainnet_env_with_constructor_true_keeps_it(self, monkeypatch):
        monkeypatch.setenv(_ENV_ENABLED, "true")
        monkeypatch.setenv(_ENV_KEY, "k")
        monkeypatch.setenv(_ENV_SECRET, "s")
        monkeypatch.setenv("YIAGENTS_EXECUTION_MAINNET", "not-a-bool")
        gw = BinanceGateway(setting={"mainnet": True})
        gw._build_client = MagicMock(return_value="FAKE_CLIENT")
        gw.connect()
        # Constructor mainnet=True survives the malformed env (fail-closed to
        # the explicit constructor value, not to False).
        assert gw._mainnet is True
