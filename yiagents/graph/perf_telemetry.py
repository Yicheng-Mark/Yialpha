"""Per-node performance telemetry for the YiAgents graph.

This module is opt-in. Nothing here auto-wraps any node or auto-installs any
callback; the caller (the graph builder / CLI) decides when to wire it up.
When telemetry is off, node handlers pass through unwrapped and byte-identical.

Design goals
------------
* Cover ALL graph nodes (not just analysts — generalizes
  ``AnalystWallTimeTracker`` in ``analyst_execution.py``).
* Thread-safe: ``record*`` methods hold a lock because the tracker is shared
  across worker threads when T2 parallel analysts run.
* Per-thread "active node" context via ``threading.local()`` so LLM tokens
  captured by the LangChain callback can be attributed to the node that
  triggered the call. Worker threads get their own thread-local storage
  (new threads start empty), so each parallel analyst sub-graph must be
  wrapped to set its own active node — exactly the pattern the parent uses.

Public API
----------
* ``NodePerfTracker``               — thread-safe accumulator.
* ``wrap_node(handler, name, tk)``  — transparent timing wrapper.
* ``NodePerfTokenCallback``         — LangChain callback for token attribution.
* ``dump_perf_report(tk, path)``    — write ``serialize()`` to JSON.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

__all__ = [
    "NodePerfTracker",
    "wrap_node",
    "NodePerfTokenCallback",
    "dump_perf_report",
]

# Sentinel node name for tokens captured with no active-node context
# (e.g. an LLM call fired outside any wrapped node, or in a worker thread
# whose thread-local active_node was never set).
UNATTRIBUTED = "_unattributed_"


class NodePerfTracker:
    """Thread-safe accumulator of per-node wall time and token usage."""

    def __init__(self) -> None:
        # Guards the shared per-node dicts and totals. Lock is NOT reentrant;
        # no method that holds it calls another method that also acquires it.
        self._lock = threading.Lock()
        # Per-node accumulators. Lazily populated. Each value is a flat dict
        # so ``serialize()`` is a cheap shallow copy away.
        self._nodes: dict[str, dict[str, Any]] = {}
        # Per-thread "current node" context. threading.local gives each thread
        # its own slot, so worker threads don't race on a shared value.
        self._thread_local = threading.local()

    # ------------------------------------------------------------------ #
    # Active-node context (thread-local, no lock needed)
    # ------------------------------------------------------------------ #
    def get_active_node(self) -> str | None:
        """Return the active node for the *current thread*, or ``None``."""
        return getattr(self._thread_local, "active_node", None)

    def set_active_node(self, name: str | None) -> None:
        """Set the active node for the *current thread* (nesting-aware:
        callers restore the previous value in a ``finally`` block)."""
        self._thread_local.active_node = name

    # ------------------------------------------------------------------ #
    # Internal helper
    # ------------------------------------------------------------------ #
    def _node_slot(self, name: str) -> dict[str, Any]:
        slot = self._nodes.get(name)
        if slot is None:
            slot = {
                "wall_seconds": 0.0,
                "calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "tokens_reasoning": 0,
            }
            self._nodes[name] = slot
        return slot

    # ------------------------------------------------------------------ #
    # Recording (thread-safe)
    # ------------------------------------------------------------------ #
    def record(self, node_name: str, duration_seconds: float) -> None:
        """Add a wall-time sample for ``node_name`` and increment its call
        count. Negative durations are clamped to 0 (defensive — should never
        happen with ``time.monotonic()`` but guards against clock oddities)."""
        if not node_name:
            return
        dur = float(duration_seconds)
        if dur < 0.0:
            dur = 0.0
        with self._lock:
            slot = self._node_slot(node_name)
            slot["wall_seconds"] += dur
            slot["calls"] += 1

    def record_tokens(
        self,
        node_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        """Accumulate token counts attributed to ``node_name``."""
        if not node_name:
            node_name = UNATTRIBUTED
        with self._lock:
            slot = self._node_slot(node_name)
            slot["tokens_in"] += int(input_tokens)
            slot["tokens_out"] += int(output_tokens)
            slot["tokens_reasoning"] += int(reasoning_tokens)

    def reset(self) -> None:
        """Clear all accumulated samples so the next ``serialize()`` reflects
        only the run that follows. The graph instance is reused across
        ``propagate()`` calls (batch mode reuses K graphs), so the parent calls
        ``reset()`` at the start of each run to keep per-run telemetry disjoint
        — without this, a reused graph's ``node_perf_<date>.json`` would merge
        every ticker it ever processed."""
        with self._lock:
            self._nodes.clear()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def serialize(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all recorded data.

        Shape::

            {
              "nodes": { "<name>": { wall_seconds, calls, tokens_in,
                                     tokens_out, tokens_reasoning }, ... },
              "totals": { wall_seconds, tokens_in, tokens_out,
                          tokens_reasoning },
            }
        """
        with self._lock:
            nodes_snapshot = {
                name: dict(slot) for name, slot in self._nodes.items()
            }
        totals = {
            "wall_seconds": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "tokens_reasoning": 0,
        }
        for slot in nodes_snapshot.values():
            totals["wall_seconds"] += slot["wall_seconds"]
            totals["tokens_in"] += slot["tokens_in"]
            totals["tokens_out"] += slot["tokens_out"]
            totals["tokens_reasoning"] += slot["tokens_reasoning"]
        # Coerce the running float sum back to a plain float for JSON.
        totals["wall_seconds"] = float(totals["wall_seconds"])
        return {"nodes": nodes_snapshot, "totals": totals}


def wrap_node(
    handler: Callable[..., Any],
    node_name: str,
    tracker: NodePerfTracker,
) -> Callable[..., Any]:
    """Return a new callable that wraps ``handler`` with timing + active-node
    bookkeeping. The wrapper is signature-agnostic (forwards ``*args,
    **kwargs``) so it works for plain ``fn(state)`` node handlers AND callable
    ``ToolNode`` instances (LangGraph invokes both as ``fn(state)``).

    Guarantees:
    * Same return value as ``handler``.
    * Same exceptions propagate (only timing side-effects are added).
    * Active node is restored to its previous value in ``finally`` — supports
      nesting when one wrapped node internally drives another.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        previous = tracker.get_active_node()
        tracker.set_active_node(node_name)
        t0 = time.monotonic()
        try:
            return handler(*args, **kwargs)
        finally:
            # ``finally`` (not ``except``) so exceptions still propagate and
            # we still book the time spent before the failure.
            tracker.record(node_name, time.monotonic() - t0)
            tracker.set_active_node(previous)

    return wrapper


class NodePerfTokenCallback(BaseCallbackHandler):
    """LangChain callback that captures LLM token usage and attributes it to
    the currently-active node on ``tracker`` (falling back to
    ``_unattributed_`` when no node is active on this thread).

    Construct as ``NodePerfTokenCallback(tracker)``.
    """

    def __init__(self, tracker: NodePerfTracker) -> None:
        super().__init__()
        self._tracker = tracker

    # ------------------------------------------------------------------ #
    # Token capture
    # ------------------------------------------------------------------ #
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Sum input/output/reasoning tokens across the response and attribute
        them to the active node.

        Reads ``AIMessage.usage_metadata`` from each generation (the same
        pattern as ``cli/stats_handler.py``), with a fallback to
        ``response.llm_output`` for providers that populate that instead.
        """
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0

        generations = getattr(response, "generations", None) or []
        for batch in generations:
            if not batch:
                continue
            for generation in batch:
                message = getattr(generation, "message", None)
                if not isinstance(message, AIMessage):
                    continue
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    continue
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
                details = usage.get("output_token_details") or {}
                reasoning_tokens += int(details.get("reasoning", 0) or 0)

        # Fallback: some providers put usage on llm_output rather than the
        # AIMessage. Only consult it if we didn't already pick anything up
        # from generations, to avoid double-counting.
        if input_tokens == 0 and output_tokens == 0:
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage") or llm_output.get(
                "usage_metadata"
            ) or {}
            if token_usage:
                input_tokens += int(token_usage.get("input_tokens", 0) or 0)
                output_tokens += int(token_usage.get("output_tokens", 0) or 0)
                # OpenAI-style nested reasoning if present.
                details = token_usage.get("output_token_details") or {}
                reasoning_tokens += int(details.get("reasoning", 0) or 0)

        if input_tokens == 0 and output_tokens == 0 and reasoning_tokens == 0:
            return

        active = self._tracker.get_active_node()
        self._tracker.record_tokens(
            active if active else UNATTRIBUTED,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    # ------------------------------------------------------------------ #
    # Explicit no-ops (BaseCallbackHandler defaults are already no-ops, but
    # being explicit avoids any drift across langchain-core versions).
    # ------------------------------------------------------------------ #
    def on_llm_start(
        self,
        serialized: dict[str, Any] | list[Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        return None

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | list[Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        return None

    def on_chain_start(
        self,
        serialized: dict[str, Any] | list[Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        return None

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        return None

    def on_chain_error(
        self, error: BaseException, **kwargs: Any
    ) -> None:
        return None

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        return None

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        return None

    def on_tool_error(
        self, error: BaseException, **kwargs: Any
    ) -> None:
        return None

    def on_llm_error(
        self, error: BaseException, **kwargs: Any
    ) -> None:
        return None


def dump_perf_report(
    tracker: NodePerfTracker,
    path: str | os.PathLike[str],
) -> None:
    """Write ``tracker.serialize()`` as indented JSON to ``path``, creating
    parent directories as needed. Used by the parent to emit
    ``node_perf_<date>.json``."""
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = tracker.serialize()
    # Atomic write: dump to a sibling temp file then os.replace() onto the
    # final path (same-volume rename is atomic on both Windows and POSIX).
    # node_perf_<date>.json is written next to full_states_log and can be
    # interrupted by run_robust's taskkill /F /T; a half-written file would
    # crash the next json.load. Mirrors the atomic pattern in memory.py.
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
