"""Shared read-timeout resolution for all LLM clients.

Every LangChain chat model (``ChatOpenAI``/``AzureChatOpenAI``,
``ChatAnthropic``, ``ChatGoogleGenerativeAI``, ``ChatBedrockConverse``) accepts
a ``timeout`` init kwarg — it aliases onto each provider's native timeout field
(``request_timeout``, ``default_request_timeout``, or botocore's
connect/read timeout). None of them ship a *default* read timeout, so without
this a half-open socket (server accepts the connection but never responds)
blocks forever and hangs the whole batch.

``resolve_timeout`` applies a single precedence chain uniformly across all
clients so the cloud-default safety net is not an OpenAI-family-only feature.
The chain (first wins):

  1. caller-supplied ``timeout`` kwarg      (explicit per-call override)
  2. ``YIAGENTS_LLM_TIMEOUT_S`` env var     (operator-wide override)
  3. local provider (Ollama / generic)      -> no timeout (slow local
     generation would be killed by a ceiling; this is the expected
     behaviour for self-hosted servers)
  4. cloud provider fallback                -> _DEFAULT_CLOUD_TIMEOUT
     (120s; logs once so operators see the applied default)

A non-numeric ``YIAGENTS_LLM_TIMEOUT_S`` (e.g. ``120s`` with a unit suffix) is
NOT silently swallowed: it logs a WARNING naming the bad value and the provider
it affected, then falls through to the local-exempt / cloud-default branch
(items 3/4). Previously it was suppressed and left the call with no timeout at
all — a silent-degradation fail-open.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Default read-timeout (seconds) applied to CLOUD providers when the operator
# has not set ``YIAGENTS_LLM_TIMEOUT_S``. Real cloud calls land in <10s; 120s
# covers the heaviest reasoning call while still recovering from rare stalls.
# Local providers (Ollama, openai_compatible) are exempt — slow local
# generation would be killed by a 120s ceiling.
_DEFAULT_CLOUD_TIMEOUT = 120.0

# One-shot guard so the cloud-default info message fires at most once per
# process (the value is the same every call, so repeating it only adds noise).
_timeout_warned = False
_timeout_warn_lock = threading.Lock()


def resolve_timeout(
    llm_kwargs: dict[str, Any],
    *,
    is_local: bool = False,
    provider_name: str = "",
) -> None:
    """Apply the read-timeout precedence chain to ``llm_kwargs`` in place.

    Sets ``llm_kwargs["timeout"]`` when a cloud default applies; leaves it
    unset for local providers unless an explicit override (caller kwarg or env)
    is present.

    Args:
        llm_kwargs: the kwargs dict that will be passed to the chat-model
            constructor. Mutated in place (``timeout`` added when applicable).
        is_local: True for self-hosted model servers (Ollama, the generic
            ``openai_compatible`` endpoint) that are exempt from the cloud
            ceiling. Native providers (Anthropic / Google / Azure / Bedrock)
            are always cloud and pass ``False``.
        provider_name: provider label for the non-numeric-env warning, purely
            diagnostic (e.g. "deepseek", "anthropic").
    """
    # 1. Explicit per-call override wins outright.
    if "timeout" in llm_kwargs:
        return

    raw = os.environ.get("YIAGENTS_LLM_TIMEOUT_S")
    if raw:
        try:
            llm_kwargs["timeout"] = float(raw)
        except ValueError:
            # 2b. Non-numeric env value — must not be silent. The old code did
            # ``contextlib.suppress(ValueError)`` here, which dropped the bad
            # value AND then skipped the cloud-default branch (because the
            # truthy env "looked configured"), leaving the call with no timeout
            # and no warning. Log it and fall through to the local/cloud logic.
            logger.warning(
                "YIAGENTS_LLM_TIMEOUT_S=%r is not a number; ignored for provider "
                "%r. Falling back to %s.",
                raw,
                provider_name or "(unknown)",
                "no timeout (local provider)" if is_local
                else f"{_DEFAULT_CLOUD_TIMEOUT}s cloud default",
            )
        else:
            return

    # 3. Local model server: no ceiling (intentional — slow local generation).
    if is_local:
        return

    # 4. Cloud default.
    llm_kwargs["timeout"] = _DEFAULT_CLOUD_TIMEOUT
    global _timeout_warned
    with _timeout_warn_lock:
        if not _timeout_warned:
            _timeout_warned = True
            logger.info(
                "YIAGENTS_LLM_TIMEOUT_S unset: cloud LLM calls use the built-in "
                "%ss read-timeout default. Set YIAGENTS_LLM_TIMEOUT_S to "
                "override; local providers (Ollama, openai_compatible) are "
                "always exempt.",
                _DEFAULT_CLOUD_TIMEOUT,
            )
