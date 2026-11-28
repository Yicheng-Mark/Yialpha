"""The US+crypto-only market gate.

``is_supported_market`` / ``require_supported_market`` in
``yialpha.dataflows.symbol_utils`` are the single allowlist predicate every
entrance funnels through (interactive CLI prompt, ``prepare_batch_run`` —
which covers the Typer batch command, run_batch.py, run_robust.py and the
web analyze subprocess —, the web API's fast 400, ``run_baseline.py`` and
``YiAlphaGraph.propagate`` as the programmatic backstop). These tests pin
the predicate's truth table and the batch/graph choke points; purely
syntactic — zero network, zero LLM.
"""

from __future__ import annotations

import pytest

from yialpha.dataflows.errors import UnsupportedMarketError
from yialpha.dataflows.symbol_utils import (
    is_supported_market,
    require_supported_market,
)

# --- truth table ------------------------------------------------------------

US_EQUITIES = [
    "AAPL", "SPY", "NVDA", "MSFT", "QQQ",
    "BRK.B", "BRK-B", "BF.B",  # single-letter class shares after . or -
    "aapl", "  spy  ",         # cleaned to upper before the check
]

CRYPTO = [
    "BTC-USD", "ETH-USD",                 # Yahoo dashed pairs
    "BTCUSD", "BTCUSDT", "BTCUSDC",       # compact venue forms
    "ETH-USDT", "SOL-USDC",
    "MUUSDT", "SPCXUSDT",                 # tokenized-stock perps (US equity underlyings)
    "btc-usd", "btcusdt",                 # cleaned to upper
    "BTC-ETH",                            # dashed crypto-crypto quote
]

REJECTED = [
    # non-US exchange suffixes
    "0700.HK", "600519.SS", "000001.SZ", "600519.SH", "7203.T", "AZN.L",
    "RELIANCE.NS", "RELIANCE.BO", "SHOP.TO", "BHP.AX",
    # forex-shaped pairs (both halves ISO codes)
    "EURUSD", "USDJPY", "GBPJPY",
    # former commodity/index CFD aliases
    "XAUUSD", "XAU", "GOLD", "XAGUSD", "WTICOUSD", "USOIL", "NATGAS",
    "SPX500", "US500", "GER40", "UK100", "JP225", "HK50",
    # Yahoo-native symbols outside the two supported asset classes
    "^GSPC", "GC=F", "EURUSD=X",
]


@pytest.mark.unit
@pytest.mark.parametrize("ticker", US_EQUITIES + CRYPTO)
def test_supported(ticker):
    assert is_supported_market(ticker)


@pytest.mark.unit
@pytest.mark.parametrize("ticker", REJECTED)
def test_rejected(ticker):
    assert not is_supported_market(ticker)


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["", "   ", None])
def test_empty_is_not_supported(ticker):
    # The interactive prompt special-cases empty (defaults to SPY downstream);
    # the predicate itself must not bless it.
    assert not is_supported_market(ticker)


# --- require_supported_market ------------------------------------------------

@pytest.mark.unit
def test_require_returns_cleaned_symbol():
    assert require_supported_market("  aapl+ ") == "AAPL"
    assert require_supported_market("btc-usdt") == "BTC-USDT"


@pytest.mark.unit
@pytest.mark.parametrize("ticker", REJECTED[:11])  # suffixes + forex samples
def test_require_raises_with_ticker_attribute(ticker):
    with pytest.raises(UnsupportedMarketError) as excinfo:
        require_supported_market(ticker)
    assert excinfo.value.ticker == ticker.upper()
    assert ticker.upper() in str(excinfo.value)


@pytest.mark.unit
def test_forex_reason_is_specific():
    with pytest.raises(UnsupportedMarketError) as excinfo:
        require_supported_market("EURUSD")
    assert "forex" in str(excinfo.value).lower()


@pytest.mark.unit
def test_cfd_alias_reason_is_specific():
    with pytest.raises(UnsupportedMarketError) as excinfo:
        require_supported_market("XAUUSD")
    assert "CFD" in str(excinfo.value) or "alias" in str(excinfo.value).lower()


# --- batch choke point --------------------------------------------------------

@pytest.mark.unit
def test_prepare_batch_run_rejects_unsupported_market():
    from yialpha.batch.runner import BatchInputError, prepare_batch_run

    with pytest.raises(BatchInputError) as excinfo:
        prepare_batch_run(["AAPL", "0700.HK"], "2026-06-30")
    assert "0700.HK" in str(excinfo.value)


@pytest.mark.unit
def test_prepare_batch_run_accepts_us_batch():
    from yialpha.batch.runner import prepare_batch_run
    from yialpha.default_config import DEFAULT_CONFIG

    cfg, asset_type = prepare_batch_run(["AAPL", "NVDA"], "2026-06-30")
    assert asset_type == "stock"  # validation passed and the class was detected
    assert cfg is not DEFAULT_CONFIG and cfg == DEFAULT_CONFIG  # a per-run copy


@pytest.mark.unit
def test_prepare_batch_run_accepts_crypto_batch():
    from yialpha.batch.runner import prepare_batch_run

    _, asset_type = prepare_batch_run(["BTC-USD", "ETH-USD"], "2026-06-30")
    assert asset_type == "crypto"


# --- propagate() backstop -----------------------------------------------------

@pytest.mark.unit
def test_propagate_rejects_unsupported_market_before_any_work():
    """The graph-level backstop: a non-US ticker raises before any
    analyst/LLM/vendor work — no network, no graph kickoff."""
    from yialpha.graph.trading_graph import YiAlphaGraph

    graph = YiAlphaGraph(debug=True)
    with pytest.raises(UnsupportedMarketError):
        graph.propagate("0700.HK", "2026-06-30")


# --- web fast-400 --------------------------------------------------------------

@pytest.mark.unit
def test_web_analyze_rejects_unsupported_market():
    from fastapi.testclient import TestClient

    from web.app import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/analyze",
            json={"ticker": "0700.HK", "date": "2026-06-30"},
        )
        assert resp.status_code == 400
        assert "0700.HK" in resp.json().get("detail", "")
