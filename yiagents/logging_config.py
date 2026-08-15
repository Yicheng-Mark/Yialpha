"""Centralised logging configuration for YiAgents.

Every module in the codebase uses ``logging.getLogger(__name__)`` but nothing
ever called ``basicConfig`` / ``dictConfig``, so all ``INFO`` / ``DEBUG``
records were silently swallowed by Python's default WARNING-only root logger.

:func:`setup_logging` installs a single :class:`logging.StreamHandler` backed
by :class:`rich.logging.RichHandler` on the root logger so that the structured
``INFO``/``DEBUG`` messages the agents, dataflows, and execution layer emit
become visible at runtime.

Usage::

    from yiagents.logging_config import setup_logging
    setup_logging()                       # respects YIAGENTS_LOG_LEVEL (default INFO)
    setup_logging("DEBUG")                # explicit override

The function is idempotent — calling it more than once (e.g. once from a
script bootstrap and again from the CLI entry point) will not duplicate
handlers. It is also compatible with
:class:`yiagents.batch.runner._TickerLogFilter`, which attaches a *filter*
(not a handler) to the root logger's handlers; a logger-level filter would
only see records emitted directly on the root logger and never tag the
``yiagents.*`` child-logger records that propagate up to it. Because the
filter rides on handlers, ``setup_logging`` must run before a
``BatchRunner`` is constructed (every entry point does, at import time).
"""

from __future__ import annotations

import logging
import os

__all__ = ["setup_logging"]

# Loggers that are noisy at INFO level and add little signal to YiAgents'
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
        from the ``YIAGENTS_LOG_LEVEL`` environment variable; if that is also
        unset, ``"INFO"`` is used. Invalid names raise ``ValueError`` so a
        misspelled level fails loudly at startup rather than silently
        degrading to WARNING.
    """
    resolved = (level or os.environ.get("YIAGENTS_LOG_LEVEL") or "INFO").upper()
    if resolved not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid YIAGENTS_LOG_LEVEL {resolved!r}; "
            f"expected one of {'/'.join(sorted(_VALID_LEVELS))}"
        )

    root = logging.getLogger()
    root.setLevel(resolved)

    # Idempotent: if a RichHandler (or any handler we recognise) is already
    # attached, just update the level and bail — no duplicate output.
    _RichHandler = _get_rich_handler_cls()
    existing = next(
        (h for h in root.handlers if isinstance(h, _RichHandler)), None
    )
    if existing is not None:
        existing.setLevel(resolved)
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
