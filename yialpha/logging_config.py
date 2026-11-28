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
#   1. every configured credential value (Alpha Vantage + the Zhipu / Tavily
#      key pools, the Binance keys and every LLM provider key,
#      comma-separated env vars split) — exact string match;
#   2. any caller-supplied secret values (e.g. the key a request actually
#      used, which may come from a patched source rather than the env);
#   3. ``apikey=<token>`` / ``apikey: <token>`` query-parameter forms,
#      quoted or not;
#   4. key-shaped tokens, family by family: a 16-char alphanumeric run
#      containing both a letter and a digit (Alpha Vantage shape — the
#      letter+digit requirement keeps ordinary words and pure numbers
#      intact); ``tvly-<base62>`` (Tavily shape, prefix is distinctive);
#      ``<32-hex>.<alnum>`` (Zhipu/BigModel shape — uuid4().hex carries no
#      dot, so thread/request ids never match); ``sk-<32-hex>`` (DeepSeek
#      shape).
#
# Redaction preserves surrounding error semantics ("rate limit exceeded",
# "Your API key [REDACTED] ...") — the message is scrubbed, never swallowed.
# ---------------------------------------------------------------------------

REDACTED_PLACEHOLDER = "[REDACTED]"

ALPHA_VANTAGE_KEY_ENV = "ALPHA_VANTAGE_API_KEY"

# Multi-key env vars (comma-separated pools) plus the single-key forms: every
# configured value is exact-match redacted first — zero false positives, the
# primary coverage for the providers this deployment actually uses. This is
# the credential SET this deployment configures; a value NOT in an env var
# relies on the shape fallbacks below.
#
# Round-17: the LLM provider keys follow. The list was written for the Zhipu
# era and never followed the 2026-09-30 switch to DeepSeek, so
# ``DEEPSEEK_API_KEY`` -- the one credential every run uses -- sat in no layer
# at all. The names are the ones in
# ``yialpha.llm_clients.api_key_env.PROVIDER_API_KEY_ENV``, spelled out here
# so this module stays import-light; ``tests/test_round17_ops.py`` pins the
# two lists against each other so the next provider switch cannot reopen the
# gap.
_KEYRING_ENVS = (
    ALPHA_VANTAGE_KEY_ENV,
    "ZHIPU_CN_API_KEYS",
    "ZHIPU_CN_API_KEY",
    "TAVILY_API_KEYS",
    "TAVILY_API_KEY",
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "MISTRAL_API_KEY",
    "MOONSHOT_API_KEY",
    "GROQ_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
)

#: A configured value shorter than this cannot be a credential (the shortest
#: real one, an Alpha Vantage key, is 16 characters). Redacting it would
#: erase ordinary text: a 1-character "key" takes every "1" out of every
#: line. Caller-supplied ``extra_secrets`` are never shortened away -- the
#: caller named that value for a reason.
_MIN_CONFIGURED_SECRET_CHARS = 6

# ``apikey=SECRET`` / ``apikey: SECRET`` (also inside URLs: stops at '&').
# Round-17: the name may be quoted and the value may open with a quote, the
# way a JSON request body is logged (``"apikey": "SECRET"``) -- a credential
# named by position is one however the name is spelled.
_APIKEY_PARAM_RE = re.compile(r"(?i)(apikey[\"']?\s*[=:]\s*[\"']?)([^\s,'\"&]+)")

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

# Tavily keys ("tvly-" + long base62 body): the distinctive prefix makes a
# shape-level fallback safe even when the exact value is not loaded in env
# (e.g. rotated out mid-process).
_TAVILY_KEY_SHAPE_RE = re.compile(r"tvly-[0-9A-Za-z]{20,}")

# BigModel/Zhipu keys ("<32-hex id>.<alnum secret>"): the dot-joined shape is
# distinctive enough that uuid4().hex (32 hex, NO dot) never matches, so
# thread ids / request ids in logs survive the fallback untouched.
_ZHIPU_KEY_SHAPE_RE = re.compile(r"[0-9a-f]{32}\.[0-9A-Za-z]{8,}")

# DeepSeek-style keys ("sk-" + 32 lower-case hex characters). "sk-" has to
# stand at a token boundary and be followed by exactly 32 hex characters, so
# a word ending in "risk-…", a longer "sk-" key of another provider, and
# upper-case hex all stay untouched.
_SK_HEX_KEY_SHAPE_RE = re.compile(r"(?<![A-Za-z0-9])sk-[0-9a-f]{32}(?![A-Za-z0-9])")


def redact_secrets(text: str, extra_secrets: Iterable[str] | None = None) -> str:
    """Scrub credential material from *text*, preserving error semantics.

    Removes every configured credential value (Alpha Vantage, the Zhipu
    key pool ``ZHIPU_CN_API_KEYS``/``ZHIPU_CN_API_KEY``, the Tavily pool
    ``TAVILY_API_KEYS``/``TAVILY_API_KEY``, the Binance keys and every LLM
    provider key in :data:`_KEYRING_ENVS` — comma-separated pools are
    split), any caller-supplied *extra_secrets* (e.g. the key a specific
    request used), ``apikey=...``/``apikey: ...`` parameter values (quoted
    or not), and key-shaped tokens (16-char AV shape, ``tvly-`` prefixed
    Tavily shape, ``<32-hex>.<alnum>`` Zhipu shape, ``sk-<32-hex>`` DeepSeek
    shape). Idempotent: already-redacted text is returned unchanged.
    """
    if not text:
        return text
    out = str(text)
    for secret in _active_api_keys() + tuple(extra_secrets or ()):
        if secret and secret in out:
            out = out.replace(secret, REDACTED_PLACEHOLDER)
    out = _APIKEY_PARAM_RE.sub(r"\g<1>" + REDACTED_PLACEHOLDER, out)
    # Family-specific shapes FIRST (a "tvly-…" or "<32hex>.<alnum>" run is a
    # more precise claim than the generic 16-char shape, which would
    # otherwise bite off just the trailing half of a longer family key).
    out = _TAVILY_KEY_SHAPE_RE.sub(REDACTED_PLACEHOLDER, out)
    out = _ZHIPU_KEY_SHAPE_RE.sub(REDACTED_PLACEHOLDER, out)
    out = _SK_HEX_KEY_SHAPE_RE.sub(REDACTED_PLACEHOLDER, out)
    out = _KEY_SHAPE_RE.sub(REDACTED_PLACEHOLDER, out)
    return out


def _active_api_keys() -> tuple[str, ...]:
    """Every configured credential value, read per call so tests/rotations
    stay current. Values too short to be a credential are left out
    (:data:`_MIN_CONFIGURED_SECRET_CHARS`)."""
    keys: list[str] = []
    for name in _KEYRING_ENVS:
        raw = os.getenv(name)
        if not raw:
            continue
        keys.extend(
            part.strip()
            for part in raw.split(",")
            if len(part.strip()) >= _MIN_CONFIGURED_SECRET_CHARS
        )
    return tuple(keys)


class CredentialRedactionFilter(logging.Filter):
    """Handler-level fallback: scrub credentials from every record.

    Attached by :func:`setup_logging` to the root handler so that even if a
    future call site concatenates a key into a log message (or a %s arg), it
    is replaced before the record reaches any sink (console, file, collector).
    Runs *after* level filtering and before emit, so it costs nothing for
    suppressed records and composes with other handler filters (e.g. the
    batch runner's ticker tagger). The exception text of ``exc_info`` records
    is scrubbed as well -- before the handler renders it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 -- malformed %-args must never break logging
            # The record cannot be formatted. Logging then reports that from
            # the handler -- with ``record.msg`` and ``record.args`` printed
            # verbatim -- so when the arguments carry credential material,
            # scrub both and leave nothing raw behind for that report.
            raw = f"{record.msg} {record.args}"
            scrubbed = redact_secrets(raw)
            if scrubbed != raw:
                record.msg, record.args = scrubbed, None
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = None  # already interpolated; don't re-format
        if record.exc_info and not record.exc_text:
            # Round-17: a fresh record carries the live exception, not its
            # text. The traceback is rendered by the HANDLER after this
            # filter has run (the default formatter through ``exc_text``,
            # ``RichHandler`` straight off the exception object), and only
            # ``record.exc_text`` was ever looked at -- still None here. So a
            # key inside an exception's message (``requests`` puts the whole
            # URL into its ``HTTPError``) reached the sink unredacted.
            try:
                rendered = logging.Formatter().formatException(record.exc_info)
            except Exception:  # noqa: BLE001 -- never break logging
                rendered = ""
            scrubbed = redact_secrets(rendered)
            if rendered and scrubbed != rendered:
                # Hand the handler the scrubbed text and drop the live
                # exception, which it would otherwise render unredacted.
                record.exc_text = scrubbed
                record.exc_info = None
        if record.exc_text:
            scrubbed = redact_secrets(record.exc_text)
            if scrubbed != record.exc_text:
                record.exc_text = scrubbed
                record.exc_info = None  # same reason: never re-render it
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
