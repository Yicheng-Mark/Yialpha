"""Guard against .env.example drift.

Every key in ``default_config._ENV_OVERRIDES`` must be documented in
``.env.example`` (commented or not) — a config knob an operator cannot
discover from the example file is an operational hazard (the risk/execution
switches especially). Conversely, every ``YIAGENTS_*`` name in the example
must actually be read by something: either it maps through ``_ENV_OVERRIDES``
or it is one of the known directly-read variables (allowlisted below with its
reader). Anything else is dead documentation.

Both directions fail loudly so new keys cannot silently fall out of sync.
"""

from __future__ import annotations

import re
from pathlib import Path

from yiagents.default_config import _ENV_OVERRIDES

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / ".env.example"

_KEY_RE = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)\s*=", re.MULTILINE)


def _example_keys() -> set[str]:
    text = EXAMPLE.read_text(encoding="utf-8")
    return set(_KEY_RE.findall(text))


# YIAGENTS_* variables read directly by their consumers (not via the
# _ENV_OVERRIDES map in default_config.py). Keep the reader location next to
# each entry so a removed reader can be pruned here in the same commit.
_DIRECT_READ_KEYS = {
    "YIAGENTS_RESULTS_DIR",          # default_config.py DEFAULT_CONFIG
    "YIAGENTS_CACHE_DIR",            # default_config.py DEFAULT_CONFIG
    "YIAGENTS_MEMORY_LOG_PATH",      # default_config.py DEFAULT_CONFIG
    "YIAGENTS_LOG_LEVEL",            # logging_config.py
    "YIAGENTS_SEC_USER_AGENT",       # dataflows/sec_edgar.py:_user_agent
    "YIAGENTS_FUNDAMENTALS_FILING_LAG_DAYS",  # dataflows/utils.py
    "YIAGENTS_HTTP_TIMEOUT_S",       # dataflows/stockstats_utils.py
    "YIAGENTS_BAOSTOCK_TIMEOUT_S",   # dataflows/baostock_vendor.py
    "YIAGENTS_LLM_TIMEOUT_S",        # llm_clients/_timeout.py
    "YIAGENTS_SENTIMENT_PARALLEL_FETCH",  # agents/analysts/sentiment_analyst.py
    "YIAGENTS_ROBUST_CHILD_SCRIPT",  # scripts/run_robust.py
    "YIAGENTS_TIMEOUT_SHIM_DIR",     # scripts/run_robust.py
    "YIAGENTS_EXECUTION_ENABLED",    # execution/binance_gateway.py (legacy gate)
    "YIAGENTS_EXECUTION_MAINNET",    # execution/binance_gateway.py
    "YIAGENTS_EXECUTION_TIMEOUT_MS",  # execution/binance_gateway.py
    "YIAGENTS_EXECUTION_LEVERAGE",   # execution/binance_gateway.py (_ensure_leverage)
}


def test_every_env_override_key_is_documented():
    """No _ENV_OVERRIDES key may be missing from .env.example."""
    missing = set(_ENV_OVERRIDES) - _example_keys()
    assert not missing, (
        "These _ENV_OVERRIDES keys are not documented in .env.example — "
        f"add them (with a one-line comment): {sorted(missing)}"
    )


def test_every_yiagents_key_in_example_is_real():
    """Every YIAGENTS_* entry in .env.example must be read by something.

    Either it maps through _ENV_OVERRIDES or it is a known direct-read var.
    Otherwise it is stale documentation that misleads operators.
    """
    example_yiagents = {
        k for k in _example_keys() if k.startswith("YIAGENTS_")
    }
    dead = example_yiagents - set(_ENV_OVERRIDES) - _DIRECT_READ_KEYS
    assert not dead, (
        "These YIAGENTS_* keys appear in .env.example but are read by "
        "nothing — remove them or add them to _ENV_OVERRIDES/_DIRECT_READ_KEYS "
        f"(with their reader): {sorted(dead)}"
    )


def test_direct_read_allowlist_still_exists_in_example():
    """The allowlist itself must not outlive the docs it describes.

    If a direct-read key was dropped from .env.example, prune it here in the
    same change; a stale allowlist entry would silently exempt a future
    same-named key from the coverage check above.
    """
    example = _example_keys()
    stale = _DIRECT_READ_KEYS - example
    assert not stale, (
        f"These allowlisted direct-read keys are no longer in .env.example — "
        f"remove them from _DIRECT_READ_KEYS: {sorted(stale)}"
    )
