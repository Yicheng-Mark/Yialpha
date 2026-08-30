import warnings
from abc import ABC, abstractmethod
from typing import Any


def normalize_content(response: Any) -> Any:
    """Normalize LLM response content to a plain string.

    Multiple providers (OpenAI Responses API, Google Gemini 3) return content
    as a list of typed blocks, e.g. [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    Downstream agents expect response.content to be a string. This extracts
    and joins the text blocks, discarding reasoning/metadata blocks.
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response


class _NormalizedInvokeMixin:
    """Invoke wrapper that joins typed content blocks into a plain string.

    Defined in a class body so zero-arg ``super()`` resolves through the real
    MRO (a function attached via ``type(...)`` has no ``__class__`` cell).
    """

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        # ``super()`` resolves to the chat class this mixin is combined with
        # at runtime (see make_normalized_chat_class); mypy cannot see that
        # MRO from the mixin alone, so the call carries a targeted ignore.
        return normalize_content(
            super().invoke(input, config, **kwargs)  # type: ignore[misc]
        )


def make_normalized_chat_class(chat_cls: type, class_name: str, doc: str) -> type:
    """``chat_cls`` subclass whose ``invoke`` normalizes content to a string.

    The one boilerplate every cloud client (anthropic / azure / bedrock /
    google / openai variants) used to duplicate: providers that emit typed
    content blocks get them joined into the plain string downstream agents
    expect. ``class_name``/``doc`` keep the dynamic subclass readable in
    tracebacks instead of surfacing as an anonymous closure class.
    """
    return type(class_name, (_NormalizedInvokeMixin, chat_cls), {"__doc__": doc})


def apply_passthrough_kwargs(
    llm_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    keys: tuple[str, ...],
    timeout_provider: str,
) -> None:
    """Copy the provider's accepted kwargs through, then set the read timeout.

    The other half of the per-client boilerplate: whitelist-copy the
    caller-supplied overrides that this provider's chat class understands,
    and apply the shared read-timeout safety net (cloud providers have no
    default read timeout; a half-open socket would hang the batch).
    Mutates ``llm_kwargs`` in place. A key may be pre-filtered by the caller
    (e.g. Anthropic drops ``effort`` on models that 400 on it) simply by not
    including it in ``keys``.
    """
    from ._timeout import resolve_timeout

    for key in keys:
        if key in kwargs:
            llm_kwargs[key] = kwargs[key]
    resolve_timeout(llm_kwargs, is_local=False, provider_name=timeout_provider)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def get_provider_name(self) -> str:
        """Return the provider name used in warning messages."""
        provider = getattr(self, "provider", None)
        if provider:
            return str(provider)
        return self.__class__.__name__.removesuffix("Client").lower()

    def warn_if_unknown_model(self) -> None:
        """Warn when the model is outside the known list for the provider."""
        if self.validate_model():
            return

        warnings.warn(
            (
                f"Model '{self.model}' is not in the known model list for "
                f"provider '{self.get_provider_name()}'. Continuing anyway."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is supported by this client."""
        pass
