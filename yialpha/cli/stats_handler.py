import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.llm_calls_replayed = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_in_replayed = 0
        self.tokens_out_replayed = 0

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response.

        A generation whose ``generation_info`` carries ``cached_replay`` was
        served by the LLM response cache (DiskLLMCache stamps it): its usage
        block belongs to the ORIGINAL call, so this run must not count it as
        its own spend (round-6 — a robust retry replaying attempt 1's nodes
        used to book their cached tokens as fresh billing).
        """
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        replayed = bool(
            (getattr(generation, "generation_info", None) or {}).get("cached_replay")
        )
        if usage_metadata:
            tokens_in = usage_metadata.get("input_tokens", 0)
            tokens_out = usage_metadata.get("output_tokens", 0)
            with self._lock:
                if replayed:
                    # The start callback already counted this call in
                    # llm_calls (langchain fires on_chat_model_start above
                    # the cache layer); take it back out so llm_calls stays
                    # LIVE-only. A replay without usage_metadata is
                    # undetectable here and stays counted — disclosed
                    # limitation, not a silent spend.
                    if self.llm_calls > 0:
                        self.llm_calls -= 1
                    self.llm_calls_replayed += 1
                    self.tokens_in_replayed += tokens_in
                    self.tokens_out_replayed += tokens_out
                else:
                    self.tokens_in += tokens_in
                    self.tokens_out += tokens_out

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> dict[str, Any]:
        """Return current statistics (live calls only; replays tallied separately)."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                # Cache-replayed generations: usage shown for attribution,
                # never mixed into the live spend above.
                "llm_calls_replayed": self.llm_calls_replayed,
                "tokens_in_replayed": self.tokens_in_replayed,
                "tokens_out_replayed": self.tokens_out_replayed,
            }
