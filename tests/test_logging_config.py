"""Tests for yialpha/logging_config.setup_logging.

These verify:
- A RichHandler is installed on the root logger
- The function is idempotent (no duplicate handlers)
- YIALPHA_LOG_LEVEL env var is respected
- Third-party loggers are silenced to WARNING
- Invalid level names raise ValueError
- Credential redaction (R6): every record is scrubbed by a handler-level
  fallback filter so a key concatenated anywhere upstream never reaches a sink
"""

from __future__ import annotations

import io
import logging

import pytest
from rich.console import Console

from yialpha.logging_config import (
    CredentialRedactionFilter,
    redact_secrets,
    setup_logging,
)


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


# ---------------------------------------------------------------------------
# Credential redaction (R6): the handler-level fallback scrubber. All keys
# below are synthetic; nothing here touches the network or a real credential.
# ---------------------------------------------------------------------------

_FAKE_KEY = "SYNTHETIC_CONFIG_KEY99"


def _capture_output() -> io.StringIO:
    """Route every installed RichHandler's console into a StringIO sink."""
    stream = io.StringIO()
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "console"):
            handler.console = Console(file=stream, width=300, no_color=True)
    return stream


def test_redaction_filter_attached_to_handler():
    """setup_logging() must attach exactly one CredentialRedactionFilter."""
    setup_logging()
    assert sum(
        1
        for h in logging.getLogger().handlers
        for f in h.filters
        if isinstance(f, CredentialRedactionFilter)
    ) == 1


def test_redaction_filter_not_duplicated_when_idempotent():
    setup_logging()
    setup_logging()
    assert sum(
        1
        for h in logging.getLogger().handlers
        for f in h.filters
        if isinstance(f, CredentialRedactionFilter)
    ) == 1


def test_filter_redacts_configured_api_key(monkeypatch):
    """A key pasted verbatim into a future log call is scrubbed before it
    reaches the sink — the belt-and-braces behind the exception-side fix."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", _FAKE_KEY)
    setup_logging("INFO")
    stream = _capture_output()
    logging.getLogger("yialpha.test").warning(
        "vendor notice: your key %s is throttled", _FAKE_KEY
    )
    output = stream.getvalue()
    assert _FAKE_KEY not in output
    assert "[REDACTED]" in output
    assert "vendor notice" in output and "throttled" in output  # semantics kept


def test_filter_redacts_apikey_query_parameter(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    setup_logging("INFO")
    stream = _capture_output()
    logging.getLogger("yialpha.test").warning(
        "request failed: https://example.com/query?function=X&apikey=secret123&limit=5"
    )
    output = stream.getvalue()
    assert "secret123" not in output
    assert "apikey=[REDACTED]" in output
    assert "function=X" in output  # unrelated query parts preserved


def test_filter_redacts_key_shaped_token_but_not_benign_text(monkeypatch):
    """A standalone 16-char letter+digit token (AV key shape) is scrubbed;
    words, 12-char digests, and pure numbers must survive untouched."""
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    setup_logging("INFO")
    stream = _capture_output()
    logging.getLogger("yialpha.test").warning(
        "shape probe key=AB12CD34EF56GH78 digest=a1b2c3d4e5f6 "
        "word=SECTION508ONLY count=2026090314320000"
    )
    output = stream.getvalue()
    assert "AB12CD34EF56GH78" not in output
    assert "a1b2c3d4e5f6" in output
    assert "SECTION508ONLY" in output
    assert "2026090314320000" in output


def test_filter_preserves_error_semantics(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", _FAKE_KEY)
    setup_logging("INFO")
    stream = _capture_output()
    logging.getLogger("yialpha.test").warning(
        "Alpha Vantage rate limit exceeded: your API key %s is limited to "
        "25 requests per day.",
        _FAKE_KEY,
    )
    output = stream.getvalue()
    assert _FAKE_KEY not in output
    assert "rate limit exceeded" in output
    assert "limited to 25 requests per day" in output
    assert "[REDACTED]" in output


def test_redact_secrets_is_idempotent():
    once = redact_secrets(f"boom {_FAKE_KEY}", extra_secrets=(_FAKE_KEY,))
    twice = redact_secrets(once)
    assert _FAKE_KEY not in once
    assert once == twice


def test_redact_secrets_shape_layer_works_without_env(monkeypatch):
    """Even with no key configured, a key-shaped 16-char token is caught."""
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    out = redact_secrets("key DEADBEEF1234CAFE used")
    assert "DEADBEEF1234CAFE" not in out
    assert "[REDACTED]" in out
