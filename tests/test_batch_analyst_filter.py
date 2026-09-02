"""batch_selected_analysts: interactive-CLI analyst parity for batch frontends.

Pure-function tests: the per-ticker crypto rule (Fundamentals dropped), the
tokenized-stock-perp keep (union semantics for mixed batches), the stock
no-op (byte-equivalence with the pre-helper behavior), the unknown
asset-type fallback, and the crypto_perp warm-before-filter ordering
contract mirrored from the interactive CLI.
"""

from __future__ import annotations

import pytest

from yialpha.batch.runner import batch_selected_analysts

_BASE = ("market", "social", "news", "fundamentals")
_THREE = ("market", "social", "news")


@pytest.fixture()
def _no_network(monkeypatch):
    """Keep the helper hermetic and record warm/filter ordering.

    The per-ticker applicability predicate lives in yialpha.graph.routing
    (shared with the fundamentals node's runtime skip) and holds its OWN
    binding of ``stock_perp_underlying`` — patch it there, not on the CLI
    utils module that now merely delegates.
    """
    import yialpha.dataflows.binance as bn
    import yialpha.graph.routing as routing

    order: list[str] = []
    monkeypatch.setattr(
        bn, "warm_equity_perp_bases",
        lambda: order.append("warm") or frozenset(),
    )
    monkeypatch.setattr(
        routing, "stock_perp_underlying",
        lambda t: order.append("filter") or None,
    )
    return order


@pytest.mark.unit
def test_stock_batch_returns_base_unchanged(_no_network):
    assert batch_selected_analysts("stock", ["AAPL", "SPY"]) == _BASE
    # Warming the EQUITY-perp listing is perp-only, like the interactive CLI.
    assert _no_network == []


@pytest.mark.unit
def test_unknown_asset_type_returns_base_unchanged(_no_network):
    assert batch_selected_analysts("nonsense", ["AAPL"]) == _BASE
    assert _no_network == []


@pytest.mark.unit
@pytest.mark.parametrize("asset_type", ["crypto", "crypto_spot", "crypto_perp"])
def test_pure_crypto_batch_drops_fundamentals(_no_network, asset_type):
    assert batch_selected_analysts(asset_type, ["BTCUSDT", "ETHUSDT"]) == _THREE


@pytest.mark.unit
def test_tokenized_stock_perp_keeps_fundamentals(monkeypatch):
    import yialpha.dataflows.binance as bn
    import yialpha.graph.routing as routing

    monkeypatch.setattr(bn, "warm_equity_perp_bases", lambda: frozenset())
    monkeypatch.setattr(
        routing, "stock_perp_underlying",
        lambda t: "MU" if t == "MUUSDT" else None,
    )
    # Mixed batch: union keeps Fundamentals for the whole batch (a shared
    # graph pool serves every ticker); a lone equity perp keeps it too.
    assert batch_selected_analysts("crypto_perp", ["BTCUSDT", "MUUSDT"]) == _BASE
    assert batch_selected_analysts("crypto_perp", ["MUUSDT"]) == _BASE


@pytest.mark.unit
def test_crypto_perp_warms_before_filtering(_no_network):
    batch_selected_analysts("crypto_perp", ["BTCUSDT"])
    # Ordering contract: the live EQUITY-perp listing is warm BEFORE the
    # per-ticker filter consults it (fresh listings are recognized).
    assert _no_network == ["warm", "filter"]
