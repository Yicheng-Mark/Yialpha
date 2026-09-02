"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Test markers (unit / integration) are declared in pyproject.toml's
# [tool.pytest.ini_options] alongside --strict-markers, so they don't need to
# be re-registered here.

_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # `or` not a .get default: an env var present but empty (e.g. a key left
        # blank in a .env copied from .env.example) must still get the placeholder.
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


@pytest.fixture(autouse=True)
def _isolate_config():
    """Reset the dataflows config before and after each test.

    ``set_config`` merges (it never clears keys absent from the override), so a
    test that sets e.g. ``tool_vendors`` would otherwise leak into later tests
    and make routing behavior order-dependent. Reset the context outright so
    every test starts from a clean DEFAULT_CONFIG.
    """
    from yialpha.dataflows.config import reset_config

    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _isolated_vendor_cache(tmp_path):
    """Point ``data_cache_dir`` at a per-test tmp directory.

    Vendor disk caches (OHLCV per-symbol CSVs, eastmoney/sec/baostock/fred/
    alphavantage/yfnews) must never read from — or write into — the
    developer's real ``~/.yialpha/cache`` during tests: a fresh real-world
    cache file would bypass a test's mocked network path, and a mocked
    response would pollute the real cache. Tests that manage their own
    cache dir call ``set_config`` in the test body, which runs after this
    fixture and therefore wins.
    """
    from yialpha.dataflows.config import set_config

    set_config({"data_cache_dir": str(tmp_path / "vendor-cache")})


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch(
        "yialpha.llm_clients.factory.create_llm_client",
        return_value=client,
    ):
        yield client


@pytest.fixture(autouse=True)
def _isolated_binance_history_memo():
    """Drop the in-process Binance closed-window memo around every test.

    Without the reset, a mocked page fetched by an earlier test would serve
    every later test whose fetch lands on the same (path, params, window)
    key, and that test's own mock would never fire.
    """
    from yialpha.dataflows.binance import reset_history_memo_for_test

    reset_history_memo_for_test()
    yield
    reset_history_memo_for_test()


@pytest.fixture(autouse=True)
def _perp_bundle_off_by_default():
    """Hermetic tests: the deterministic perp market bundle (default ON in
    production for crypto_perp market analysts) issues real network
    prefetches. Keep it OFF here unless a test explicitly opts in with
    mocked fetchers (``set_config({"perp_market_bundle": True})`` in the
    test body — set_config merges, so the opt-in wins over this fixture).
    """
    from yialpha.dataflows.config import set_config

    set_config({"perp_market_bundle": False})
    yield


@pytest.fixture(autouse=True)
def _runtime_prefetch_bundles_off():
    """Hermetic tests: the deterministic fundamentals bundle (default ON in
    production) issues real vendor prefetches through the router. Keep it
    OFF here unless a test explicitly opts in with mocked fetchers
    (``set_config({"fundamentals_bundle": True})`` in the test body —
    set_config merges, so the opt-in wins over this fixture).

    Named to sort AFTER ``_isolate_config`` (pytest runs same-scope autouse
    conftest fixtures alphabetically): the reset there restores the
    production default True, so this fixture must run later to hold it off.
    """
    from yialpha.dataflows.config import set_config

    set_config({"fundamentals_bundle": False})
    yield


@pytest.fixture(autouse=True)
def _runtime_ledger_isolated(tmp_path):
    """Point the V2 ledger DB at a per-test tmp file and hold the V2.1/V2.2
    record/shadow flags OFF (production defaults are ON).

    Same contract as ``_perp_bundle_off_by_default`` /
    ``_runtime_prefetch_bundles_off``: tests opt in with ``set_config`` in
    the test body (set_config merges, so the opt-in wins). Named to sort
    AFTER ``_isolate_config`` (pytest runs same-scope autouse conftest
    fixtures alphabetically): the reset there restores the production
    defaults, so this fixture must run later to hold them off. The ledger
    connection cache is dropped around each test so a connection opened for
    the previous test's tmp path never serves the current test.
    """
    from yialpha.dataflows.config import set_config
    from yialpha.ledger.sqlite import reset_ledger_state_for_test

    set_config(
        {
            "ledger_db_path": str(tmp_path / "ledger" / "portfolio.db"),
            "instrument_registry": False,
            "prediction_ledger": False,
            "stock_perp_fair_value": False,
            "regime_state": False,
        }
    )
    reset_ledger_state_for_test()
    yield
    reset_ledger_state_for_test()


@pytest.fixture(autouse=True)
def _hermetic_perp_overlay_mark(monkeypatch):
    """The perp risk overlay's Mark Reference bullet fetches mark klines
    (network). Stub it to None (the production fail-soft degradation) so any
    test running the perp overlay stays hermetic; a test asserting the mark
    bullet monkeypatches the method with a value, which shadows this
    class-level stub.
    """
    from yialpha.graph.trading_graph import YiAlphaGraph

    monkeypatch.setattr(
        YiAlphaGraph, "_latest_mark_close", lambda self, t, d: None,
    )
