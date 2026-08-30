"""Tests for yialpha/logging_config.setup_logging.

These verify:
- A RichHandler is installed on the root logger
- The function is idempotent (no duplicate handlers)
- YIALPHA_LOG_LEVEL env var is respected
- Third-party loggers are silenced to WARNING
- Invalid level names raise ValueError
"""

from __future__ import annotations

import logging

import pytest

from yialpha.logging_config import setup_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Snapshot, clear, and restore root logger state around every test.

    Other entry points (cli/main.py, web/app.py) call setup_logging() at import
    time, so a RichHandler may already be on the root logger when these tests
    collect. We clear handlers before each test so the test starts from a clean
    state, then restore the original configuration afterward.
    """
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_levels = {
        name: logging.getLogger(name).level for name in ("httpx", "openai", "urllib3")
    }
    # Start each test from a clean root logger (no handlers).
    root.handlers = []
    yield
    root.setLevel(saved_level)
    root.handlers = saved_handlers
    for name, lvl in saved_levels.items():
        logging.getLogger(name).setLevel(lvl)


def _count_rich_handlers() -> int:
    from rich.logging import RichHandler

    return sum(
        1 for h in logging.getLogger().handlers if isinstance(h, RichHandler)
    )


def test_installs_rich_handler():
    """setup_logging() should add exactly one RichHandler to the root logger."""
    assert _count_rich_handlers() == 0
    setup_logging()
    assert _count_rich_handlers() == 1


def test_default_level_is_info():
    """Without any override, the root level should be INFO."""
    setup_logging()
    assert logging.getLogger().level == logging.INFO


def test_explicit_level():
    """An explicit level argument should be applied."""
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_env_var_override(monkeypatch):
    """YIALPHA_LOG_LEVEL should be read when no explicit level is given."""
    monkeypatch.setenv("YIALPHA_LOG_LEVEL", "DEBUG")
    setup_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_env_var_warning_level(monkeypatch):
    monkeypatch.setenv("YIALPHA_LOG_LEVEL", "WARNING")
    setup_logging()
    assert logging.getLogger().level == logging.WARNING


def test_idempotent():
    """Calling setup_logging twice must not duplicate handlers."""
    setup_logging()
    assert _count_rich_handlers() == 1
    setup_logging()
    assert _count_rich_handlers() == 1


def test_idempotent_updates_level():
    """A second call with a different level should update, not duplicate."""
    setup_logging("INFO")
    assert logging.getLogger().level == logging.INFO
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    assert _count_rich_handlers() == 1


def test_noisy_loggers_silenced():
    """Third-party HTTP loggers should be pinned to WARNING."""
    setup_logging("DEBUG")
    for name in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
        assert logging.getLogger(name).level == logging.WARNING


def test_invalid_level_raises():
    """A misspelled level should fail loudly."""
    with pytest.raises(ValueError, match="Invalid YIALPHA_LOG_LEVEL"):
        setup_logging("VERBSE")


def test_invalid_env_level_raises(monkeypatch):
    monkeypatch.setenv("YIALPHA_LOG_LEVEL", "trce")
    with pytest.raises(ValueError, match="Invalid YIALPHA_LOG_LEVEL"):
        setup_logging()


def test_explicit_overrides_env(monkeypatch):
    """Explicit level arg wins over YIALPHA_LOG_LEVEL."""
    monkeypatch.setenv("YIALPHA_LOG_LEVEL", "WARNING")
    setup_logging("ERROR")
    assert logging.getLogger().level == logging.ERROR
