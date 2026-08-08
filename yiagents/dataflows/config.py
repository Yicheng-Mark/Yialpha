"""Thread-safe configuration store for the dataflows layer.

Previously this was a mutable module-global dict (``_config``) mutated by
:func:`set_config`. That made the batch runner's K worker threads share one
process-wide dict, forcing :class:`~yiagents.batch.runner.BatchRunner` to add
``_assert_uniform`` — a guard that *rejects* divergent worker configs because
they would clobber each other through the global.

This module now uses a :class:`~contextvars.ContextVar` instead. The API
(``get_config`` / ``set_config`` / ``initialize_config``) is unchanged, but the
state is now per-context (thread-safe, no lock needed):

* Single-ticker runs: identical behaviour — one context, one config.
* Batch runs with K identical workers: identical behaviour — all workers
  inherit the config set during pool construction. ``_assert_uniform`` is no
  longer structurally necessary (though it remains as a belt-and-suspenders
  guard until callers are audited).
* Batch runs with divergent worker configs: **now possible** — each worker
  can ``set_config`` in its own context without clobbering siblings. This was
  previously impossible (the guard would raise).

``get_config`` still returns a :func:`~copy.deepcopy` so callers can never
mutate the stored config in place.
"""

from contextvars import ContextVar
from copy import deepcopy
from typing import Any

import yiagents.default_config as default_config

# ContextVar replaces the module-global _config. Default is None until
# initialize_config() runs (at import time, below). Each context (thread or
# async task) gets its own slot, so set_config in one worker never affects
# another — the fragile process-wide mutation is gone.
_config_var: ContextVar[dict[str, Any] | None] = ContextVar("_config_var", default=None)


def initialize_config() -> None:
    """Initialize the configuration with default values (if not yet set)."""
    if _config_var.get() is None:
        _config_var.set(deepcopy(default_config.DEFAULT_CONFIG))


def set_config(config: dict[str, Any]) -> None:
    """Update the configuration with custom values.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.

    This sets the config in the **current context** (thread/task). In a batch
    run, each worker thread inherits the config from the context that submitted
    it, so a worker calling ``set_config`` does not clobber siblings.
    """
    current = _config_var.get()
    if current is None:
        current = deepcopy(default_config.DEFAULT_CONFIG)
    else:
        # Copy so we never mutate the inherited dict in place — a child context
        # that diverges should get its own copy, not silently rewrite parent
        # state that other workers might still read.
        current = deepcopy(current)
    incoming = deepcopy(config)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key].update(value)
        else:
            current[key] = value
    _config_var.set(current)


def get_config() -> dict[str, Any]:
    """Get the current context's configuration (a deepcopy, never the stored dict)."""
    if _config_var.get() is None:
        initialize_config()
    cfg = _config_var.get()
    assert cfg is not None  # set by initialize_config()
    return deepcopy(cfg)


def reset_config() -> None:
    """Reset the current context's config to a fresh copy of ``DEFAULT_CONFIG``.

    Unlike :func:`set_config`, this **replaces** the config entirely (no
    one-level merge), so stale keys from a previous partial update are cleared.
    Tests that mutate nested config keys (e.g. ``tool_vendors``) should call
    this in ``tearDown`` to avoid cross-test pollution through the shared
    context.
    """
    _config_var.set(deepcopy(default_config.DEFAULT_CONFIG))


# Initialize with default config at import time (same as before).
initialize_config()
