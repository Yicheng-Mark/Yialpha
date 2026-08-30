"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar, overload

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone.
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


class _StructuredFallbackSentinel:
    """Marker returned on the free-text-fallback path of ``extract``.

    This lets downstream code distinguish **three** states for an extracted
    structured field (e.g. the PM's ``rating``):

    * A real value (e.g. ``"Buy"``) — structured call succeeded.
    * ``None`` — the structured call succeeded but the field was genuinely
      absent (rare; e.g. the schema allows optional fields).
    * :data:`STRUCTURED_FALLBACK` — the structured call **failed** and the
      agent fell back to free-text generation. The markdown is the LLM's raw
      prose and the extracted field does not exist. This is the silent-
      degradation path that previously returned a bare ``None``, making it
      indistinguishable from a legitimate absence.

    The sentinel is falsy so ``pm_rating or ""`` and similar truthiness guards
    keep working unchanged, but ``is STRUCTURED_FALLBACK`` lets the risk overlay
    and any audit layer flag the degraded path explicitly.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "STRUCTURED_FALLBACK"


#: Singleton instance. Use identity comparison (``is STRUCTURED_FALLBACK``).
STRUCTURED_FALLBACK: Any = _StructuredFallbackSentinel()


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


@overload
def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    *,
    extract: None = None,
) -> str: ...


@overload
def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    *,
    extract: Callable[[T], Any],
) -> tuple[str, Any]: ...


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    *,
    extract: Callable[[T], Any] | None = None,
) -> str | tuple[str, Any]:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.

    When ``extract`` is provided, the structured-success path additionally
    applies it to the parsed result and returns ``(rendered_markdown,
    extracted_value)`` instead of just the markdown string. The free-text
    fallback returns ``(markdown, STRUCTURED_FALLBACK)`` — a falsy sentinel
    that is distinguishable from a genuine ``None`` via identity comparison
    (``is STRUCTURED_FALLBACK``). Callers that don't pass ``extract`` get the
    original ``str``-only return (Trader, Research Manager), so this is fully
    backward-compatible.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                # A thinking model can answer in plain text instead of calling
                # the tool, leaving the parser with nothing to return. Treat it
                # as a structured miss and fall back, with a clear reason.
                raise ValueError("structured output returned no parsed result")
            md = render(result)
            if extract is not None:
                return md, extract(result)
            return md
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    if extract is not None:
        return response.content, STRUCTURED_FALLBACK
    return response.content
