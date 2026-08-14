"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Test markers (unit / integration / smoke) are declared in pyproject.toml's
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
    from yiagents.dataflows.config import reset_config

    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _isolated_vendor_cache(tmp_path):
    """Point ``data_cache_dir`` at a per-test tmp directory.

    Vendor disk caches (OHLCV per-symbol CSVs, eastmoney/sec/baostock/fred/
    alphavantage/yfnews) must never read from — or write into — the
    developer's real ``~/.yiagents/cache`` during tests: a fresh real-world
    cache file would bypass a test's mocked network path, and a mocked
    response would pollute the real cache. Tests that manage their own
    cache dir call ``set_config`` in the test body, which runs after this
    fixture and therefore wins.
    """
    from yiagents.dataflows.config import set_config

    set_config({"data_cache_dir": str(tmp_path / "vendor-cache")})


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch(
        "yiagents.llm_clients.factory.create_llm_client",
        return_value=client,
    ):
        yield client
