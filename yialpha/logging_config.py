"""Centralised logging configuration for YiAlpha.

Every module in the codebase uses ``logging.getLogger(__name__)`` but nothing
ever called ``basicConfig`` / ``dictConfig``, so all ``INFO`` / ``DEBUG``
records were silently swallowed by Python's default WARNING-only root logger.

:func:`setup_logging` installs a single :class:`logging.StreamHandler` backed
by :class:`rich.logging.RichHandler` on the root logger so that the structured
``INFO``/``DEBUG`` messages the agents, dataflows, and execution layer emit
become visible at runtime.

Usage::

    from yialpha.logging_config import setup_logging
    setup_logging()                       # respects YIALPHA_LOG_LEVEL (default INFO)
    setup_logging("DEBUG")                # explicit override

The function is idempotent — calling it more than once (e.g. once from a
script bootstrap and again from the CLI entry point) will not duplicate
handlers. It is also compatible with
:class:`yialpha.batch.runner._TickerLogFilter`, which attaches a *filter*
(not a handler) to the root logger's handlers; a logger-level filter would
only see records emitted directly on the root logger and never tag the
``yialpha.*`` child-logger records that propagate up to it. Because the
filter rides on handlers, ``setup_logging`` must run before a
``BatchRunner`` is constructed (every entry point does, at import time).
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable

__all__ = [
    "ALPHA_VANTAGE_KEY_ENV",
    "CredentialRedactionFilter",
    "redact_secrets",
    "setup_logging",
]

# ---------------------------------------------------------------------------
# Credential redaction (R6).
#
# ``redact_secrets`` is the single canonical scrubber: it is used at exception
# construction sites (e.g. dataflows/alpha_vantage_common.py embeds vendor
# notice text in error messages) AND as a handler-level fallback via
# :class:`CredentialRedactionFilter`, so a key concatenated anywhere upstream
# can never reach a sink even if a future call site forgets to redact.
#
# Redaction layers, cheapest/most-precise first:
#   1. the active Alpha Vantage key value (env) — exact string match;
#   2. any caller-supplied secret values (e.g. the key a request actually
#      used, which may come from a patched source rather than the env);
#   3. ``apikey=<token>`` / ``apikey: <token>`` query-parameter forms;
#   4. key-shaped tokens: a 16-char alphanumeric run containing both a letter
#      and a digit (Alpha Vantage keys are 16-char mixed alphanumerics). The
#      letter+digit requirement keeps ordinary words and pure numbers intact.
#
# Redaction preserves surrounding error semantics ("rate limit exceeded",
# "Your API key [REDACTED] ...") — the message is scrubbed, never swallowed.
# ---------------------------------------------------------------------------

REDACTED_PLACEHOLDER = "[REDACTED]"

ALPHA_VANTAGE_KEY_ENV = "ALPHA_VANTAGE_API_KEY"

# ``apikey=SECRET`` / ``apikey: SECRET`` (also inside URLs: stops at '&').
_APIKEY_PARAM_RE = re.compile(r"(?i)(apikey\s*[=:]\s*)([^\s,'\"&]+)")

# A standalone 16-char alphanumeric run (Alpha Vantage key shape) holding at
# least one letter and one digit. The lookaheads pin the run so words, plain
# numbers, and longer identifiers (hashes, order IDs) are left untouched.
_KEY_SHAPE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?=[A-Za-z0-9]{16}(?:[^A-Za-z0-9]|$))"
    r"(?=[A-Za-z0-9]*\d)"
    r"(?=[A-Za-z0-9]*[A-Za-z])"
    r"[A-Za-z0-9]{16}"
    r"(?![A-Za-z0-9])"
)


def redact_secrets(text: str, extra_secrets: Iterable[str] | None = None) -> str:
    """Scrub credential material from *text*, preserving error semantics.

    Removes the configured Alpha Vantage key (``ALPHA_VANTAGE_API_KEY``), any
    caller-supplied *extra_secrets* (e.g. the key a specific request used),
    ``apikey=...``/``apikey: ...`` parameter values, and key-shaped 16-char
    alphanumeric tokens. Idempotent: already-redacted text is returned
    unchanged.
    """
    if not text:
        return text
    out = str(text)
    for secret in _active_api_keys() + tuple(extra_secrets or ()):
        if secret and secret in out:
            out = out.replace(secret, REDACTED_PLACEHOLDER)
    out = _APIKEY_PARAM_RE.sub(r"\g<1>" + REDACTED_PLACEHOLDER, out)
    out = _KEY_SHAPE_RE.sub(REDACTED_PLACEHOLDER, out)
    return out


def _active_api_keys() -> tuple[str, ...]:
    """The configured API key, read per call so tests/rotations stay current."""
    key = os.getenv(ALPHA_VANTAGE_KEY_ENV)
    return (key,) if key else ()


class CredentialRedactionFilter(logging.Filter):
    """Handler-level fallback: scrub credentials from every record.

    Attached by :func:`setup_logging` to the root handler so that even if a
    future call site concatenates a key into a log message (or a %s arg), it
    is replaced before the record reaches any sink (console, file, collector).
    Runs *after* level filtering and before emit, so it costs nothing for
    suppressed records and composes with other handler filters (e.g. the
    batch runner's ticker tagger).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed %-args must never break logging
            message = str(record.msg)
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = None  # already interpolated; don't re-format
        if record.exc_text:
            scrubbed = redact_secrets(record.exc_text)
            if scrubbed != record.exc_text:
                record.exc_text = scrubbed
        return True


# Loggers that are noisy at INFO level and add little signal to YiAlpha'
# own output. They are pinned to WARNING regardless of the root level so
# that DEBUG / INFO mode surfaces our own messages, not HTTP wire chatter.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "anthropic",
    "websockets",
    "watchfiles",
    "asyncio",
)

_VALID_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger with a RichHandler.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"DEBUG"``). When *None*, the value is read
        from the ``YIALPHA_LOG_LEVEL`` environment variable; if that is also
        unset, ``"INFO"`` is used. Invalid names raise ``ValueError`` so a
        misspelled level fails loudly at startup rather than silently
        degrading to WARNING.
    """
    resolved = (level or os.environ.get("YIALPHA_LOG_LEVEL") or "INFO").upper()
    if resolved not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid YIALPHA_LOG_LEVEL {resolved!r}; "
            f"expected one of {'/'.join(sorted(_VALID_LEVELS))}"
        )

    root = logging.getLogger()
    root.setLevel(resolved)

    # Idempotent: if a RichHandler (or any handler we recognise) is already
    # attached, just update the level and bail — no duplicate output. Make
    # sure the credential-redaction fallback filter is present even on a
    # handler installed by an earlier call (e.g. a CLI entry point's import).
    _RichHandler = _get_rich_handler_cls()
    existing = next(
        (h for h in root.handlers if isinstance(h, _RichHandler)), None
    )
    if existing is not None:
        existing.setLevel(resolved)
        if not any(
            isinstance(f, CredentialRedactionFilter) for f in existing.filters
        ):
            existing.addFilter(CredentialRedactionFilter())
        return

    # Replace any pre-existing handlers (e.g. a stray basicConfig handler
    # from a library) so we own the output format.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = _RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    handler.setLevel(resolved)
    handler.addFilter(CredentialRedactionFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)

    # Quiet the chatty third-party loggers so our own INFO/DEBUG is legible.
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _get_rich_handler_cls():
    """Import RichHandler lazily so this module has no hard import-time deps."""
    from rich.logging import RichHandler

    return RichHandler
