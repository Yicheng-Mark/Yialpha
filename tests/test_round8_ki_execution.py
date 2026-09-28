"""Round-8 S8 known-issue landing pins (execution layer).

Each test pins one landed KI fix (2026-09-28): KI-1 connection-level
transient classification, KI-5 reference-as-label idempotency ids, KI-6
timeout clamping, KI-7 proxy env spellings/port-less URLs, KI-10 connect
re-entry teardown, and the testnet/mainnet ``base_path`` selection line.
Network-free: the SDK is never installed here — ``_build_client`` paths run
against fake modules injected into ``sys.modules``.
"""

from __future__ import annotations

import logging
import sys
import types
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import yialpha.execution.binance_gateway as gw_mod
from yialpha.execution.binance_gateway import (
    BinanceGateway,
    _resolve_timeout_ms,
)
from yialpha.execution.domain import (
    Direction,
    Exchange,
    Offset,
    OrderRequest,
    OrderType,
    Status,
)
from yialpha.risk.breaker import BreakerState
from yialpha.risk.manager import RiskDecision

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _arm_test_execution_policy(monkeypatch):
    monkeypatch.setenv("YIALPHA_ANALYSIS_ONLY", "false")
    monkeypatch.setenv("YIALPHA_LIVE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("YIALPHA_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("YIALPHA_KILL_SWITCH", "false")
    monkeypatch.delenv("YIALPHA_EXECUTION_LEVERAGE", raising=False)

    from yialpha.dataflows.binance_filters import SymbolFilters

    def _permissive_filters(symbol, venue):  # noqa: ARG001
        return SymbolFilters(
            symbol=symbol, status="TRADING",
            tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
            min_qty=Decimal("0.001"), max_qty=Decimal("0"),
            min_notional=Decimal("0"), price_precision=2, quantity_precision=3,
        )

    monkeypatch.setattr(gw_mod, "get_symbol_filters", _permissive_filters)


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


def _req(symbol="BTCUSDT", direction=Direction.LONG, offset=Offset.NONE,
         reference=""):
    return OrderRequest(
        symbol=symbol,
        exchange=Exchange.BINANCE,
        direction=direction,
        type=OrderType.MARKET,
        volume=0.5,
        price=0.0,
        offset=offset,
        reference=reference,
    )


def _risk_decision() -> RiskDecision:
    return RiskDecision(
        rating="Buy",
        action="enter",
        target_weight=0.05,
        position_value=50_000,
        stop_loss=None,
        entry_price=None,
        kelly_raw=0.05,
        breaker=BreakerState(
            can_open_new=True,
            position_multiplier=1.0,
            current_drawdown=0.0,
            regime="normal",
        ),
        cvar_multiplier=1.0,
        exposure_ok=True,
        exposure_reason="ok",
        rationale="test",
    )


def _perp_gw_with_client():
    gw = BinanceGateway()
    gw._client = MagicMock()
    raw_send = gw.send_order

    def _checked_send(req, **kwargs):
        kwargs.setdefault("risk_decision", _risk_decision())
        kwargs.setdefault("equity", 1_000_000)
        kwargs.setdefault("reference_price", 60_000)
        return raw_send(req, **kwargs)

    gw.send_order = _checked_send
    return gw


# ---------------------------------------------------------------------------
# KI-1: connection-level exception families take the query-recovery path
# ---------------------------------------------------------------------------


class TestConnectionLevelTransients:
    @pytest.mark.parametrize(
        "exc_factory",
        [
            lambda: ConnectionError("reset mid-response"),
            lambda: type("ProxyError", (Exception,), {})("proxy refused"),
            lambda: type("SSLError", (Exception,), {})("tls handshake"),
            lambda: type("ChunkedEncodingError", (Exception,), {})("broken stream"),
        ],
        ids=["connection-error", "proxy-error", "ssl-error", "chunked-encoding"],
    )
    def test_connection_family_queries_then_recovers(self, monkeypatch, exc_factory):
        monkeypatch.setattr("time.sleep", lambda _s: None)
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = exc_factory()
        gw._client.rest_api.query_order.return_value = _FakeResp(
            {"status": "NEW", "orderId": 42, "executedQty": "0"}
        )
        order = gw.send_order(_req())
        # The submit reached the book despite the connection-level error:
        # recovery QUERIED (never re-POSTed) and returned the live order.
        assert gw._client.rest_api.new_order.call_count == 1
        assert gw._client.rest_api.query_order.call_count == 1
        assert order.status is Status.NOTTRADED

    def test_connection_family_unconfirmed_rejected(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _s: None)
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.side_effect = ConnectionError("reset")
        gw._client.rest_api.query_order.return_value = _FakeResp(
            {"code": -2013, "msg": "Order does not exist."}
        )
        order = gw.send_order(_req())
        assert order.status is Status.REJECTED
        assert gw._client.rest_api.new_order.call_count == 1


# ---------------------------------------------------------------------------
# KI-5: the reference is a label; the idempotency id is always minted
# ---------------------------------------------------------------------------


class TestReferenceIsLabel:
    def test_repeated_reference_gets_distinct_ids(self):
        gw = _perp_gw_with_client()
        gw._client.rest_api.new_order.return_value = _FakeResp(
            {"status": "NEW", "orderId": 7, "executedQty": "0"}
        )
        gw.send_order(_req(reference="same-ref"))
        gw.send_order(_req(reference="same-ref"))
        ids = [
            c.kwargs["new_client_order_id"]
            for c in gw._client.rest_api.new_order.call_args_list
        ]
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert all(i.startswith("yialpha-") for i in ids)
        assert all(len(i) <= 36 for i in ids)


# ---------------------------------------------------------------------------
# KI-6: timeout parsing / clamping (pure function, no SDK needed)
# ---------------------------------------------------------------------------


class TestResolveTimeoutMs:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, 10000),
            ("10000", 10000),
            ("5000", 5000),
            ("20000", 20000),
            ("abc", 10000),   # historical non-int fallback
            ("0", 10000),     # KI-6: zero is undefined, clamped
            ("-500", 10000),  # KI-6: negative is undefined, clamped
        ],
    )
    def test_table(self, raw, expected):
        assert _resolve_timeout_ms(raw) == expected

    def test_non_positive_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="yialpha.execution.binance_gateway"):
            assert _resolve_timeout_ms("-500") == 10000
        assert any("<= 0" in r.message for r in caplog.records)

    def test_valid_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="yialpha.execution.binance_gateway"):
            assert _resolve_timeout_ms("5000") == 5000
        assert not caplog.records


# ---------------------------------------------------------------------------
# KI-7: proxy env spellings and port-less URLs
# ---------------------------------------------------------------------------


class TestProxyDict:
    @pytest.fixture(autouse=True)
    def _clean_proxy_env(self, monkeypatch):
        for name in (
            "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_lowercase_https_proxy_is_read(self):
        import os

        os.environ["https_proxy"] = "socks5h://127.0.0.1:1080"
        try:
            proxy = BinanceGateway._proxy_dict()
        finally:
            del os.environ["https_proxy"]
        assert proxy == {
            "protocol": "socks5h", "host": "127.0.0.1", "port": 1080,
        }

    # NOTE: no uppercase-vs-lowercase precedence test exists — on Windows
    # ``os.environ`` is case-insensitive, so both spellings are the SAME
    # key and precedence is a non-concept there. POSIX precedence is
    # covered by the read-chain order in the implementation.

    def test_all_proxy_fallback_lowercase(self, monkeypatch):
        monkeypatch.setenv("all_proxy", "socks5h://127.0.0.3:1080")
        assert BinanceGateway._proxy_dict()["host"] == "127.0.0.3"

    def test_unset_returns_none(self):
        assert BinanceGateway._proxy_dict() is None

    def test_portless_url_warns_and_omits_port(self, monkeypatch, caplog):
        monkeypatch.setenv("HTTPS_PROXY", "socks5h://127.0.0.1")
        with caplog.at_level(logging.WARNING, logger="yialpha.execution.binance_gateway"):
            proxy = BinanceGateway._proxy_dict()
        assert proxy == {"protocol": "socks5h", "host": "127.0.0.1"}
        assert any("no port" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# base_path: testnet by default, mainnet only on explicit opt-in
# ---------------------------------------------------------------------------


class _CfgRecorder:
    """Records ConfigurationRestAPI kwargs (facade constructors get their
    own recorder class so they cannot clobber these)."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


class _FacadeRecorder:
    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


def _install_fake_sdk(monkeypatch):
    """Inject fake SDK modules so ``_build_client`` runs without the SDK."""
    common = types.ModuleType("binance_common")
    cfg_mod = types.ModuleType("binance_common.configuration")
    const_mod = types.ModuleType("binance_common.constants")
    const_mod.SPOT_REST_API_PROD_URL = "SPOT_PROD"
    const_mod.SPOT_REST_API_TESTNET_URL = "SPOT_TESTNET"
    const_mod.DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL = "FUT_PROD"
    const_mod.DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL = "FUT_TESTNET"
    cfg_mod.ConfigurationRestAPI = _CfgRecorder
    common.configuration = cfg_mod
    common.constants = const_mod

    spot_pkg = types.ModuleType("binance_sdk_spot")
    spot_mod = types.ModuleType("binance_sdk_spot.spot")
    spot_mod.Spot = _FacadeRecorder
    spot_pkg.spot = spot_mod

    fut_pkg = types.ModuleType("binance_sdk_derivatives_trading_usds_futures")
    fut_mod = types.ModuleType(
        "binance_sdk_derivatives_trading_usds_futures."
        "derivatives_trading_usds_futures"
    )
    fut_mod.DerivativesTradingUsdsFutures = _FacadeRecorder
    fut_pkg.derivatives_trading_usds_futures = fut_mod

    for name, mod in {
        "binance_common": common,
        "binance_common.configuration": cfg_mod,
        "binance_common.constants": const_mod,
        "binance_sdk_spot": spot_pkg,
        "binance_sdk_spot.spot": spot_mod,
        "binance_sdk_derivatives_trading_usds_futures": fut_pkg,
        "binance_sdk_derivatives_trading_usds_futures."
        "derivatives_trading_usds_futures": fut_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


class TestBasePathSelection:
    def test_spot_defaults_to_testnet(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        monkeypatch.delenv("YIALPHA_EXECUTION_MAINNET", raising=False)
        gw = BinanceGateway(setting={"product": "spot"})
        gw._build_client("k", "s")
        assert _CfgRecorder.last_kwargs["base_path"] == "SPOT_TESTNET"

    def test_spot_mainnet_opt_in(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        monkeypatch.setenv("YIALPHA_EXECUTION_MAINNET", "true")
        gw = BinanceGateway(setting={"product": "spot"})
        gw._build_client("k", "s")
        assert _CfgRecorder.last_kwargs["base_path"] == "SPOT_PROD"

    def test_perp_defaults_to_testnet(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        monkeypatch.delenv("YIALPHA_EXECUTION_MAINNET", raising=False)
        gw = BinanceGateway(setting={"product": "perp"})
        gw._build_client("k", "s")
        assert _CfgRecorder.last_kwargs["base_path"] == "FUT_TESTNET"

    def test_timeout_is_passed_through(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        monkeypatch.setenv("YIALPHA_EXECUTION_TIMEOUT_MS", "7000")
        gw = BinanceGateway(setting={"product": "perp"})
        gw._build_client("k", "s")
        assert _CfgRecorder.last_kwargs["timeout"] == 7000


# ---------------------------------------------------------------------------
# KI-10: connect() re-entry tears the previous client down
# ---------------------------------------------------------------------------


class TestConnectReentry:
    def test_reconnect_closes_old_client(self, monkeypatch):
        gw = BinanceGateway(setting={"product": "perp"})
        old_client = MagicMock()
        gw._client = old_client
        gw._connected = True
        new_client = MagicMock()
        monkeypatch.setattr(gw, "_build_client", lambda k, s: new_client)
        monkeypatch.setattr(gw, "_query_hedge_mode", lambda: False)
        monkeypatch.setenv("BINANCE_API_KEY", "k")
        monkeypatch.setenv("BINANCE_API_SECRET", "s")
        gw.connect()
        old_client.close.assert_called_once()
        assert gw._client is new_client
        assert gw._connected is True
