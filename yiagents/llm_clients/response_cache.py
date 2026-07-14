"""Disk-backed per-call LLM response cache (langchain ``BaseCache``).

Default-OFF dev/iteration accelerator. When ``YIAGENTS_LLM_CACHE`` is set,
``YiAgentsGraph.__init__`` installs one ``DiskLLMCache`` via langchain's global
``set_llm_cache``. On an identical ``(model + prompt + temperature + bound
tools / structured-output schema)`` the cached ``ChatGeneration`` is replayed
instead of re-calling the model, so re-running the same ``--smoke`` (ticker,
date) while you iterate on prompts / risk code costs ~zero tokens the second
time. This *complements* — does not duplicate — the decision-level
``yiagents.backtest.cache.DecisionCache``, which already makes whole-backtest
re-runs free.

DISTRIBUTION SAFETY (read this). At temperature > 0 the same prompt is *meant*
to yield different responses across calls; that run-to-run variability is what
the analyst A/B gate (``scripts/run_analyst_parallel_ab.py`` chi-square test)
and the DSR ``n_trials`` distribution in ``run_baseline --full`` measure. A
per-call cache collapses that variability to a single realization, so:

    **Do NOT enable this cache while running the analyst A/B gate or a
    multi-trial DSR backtest.** Both run with it off by default — simply do not
    set ``YIAGENTS_LLM_CACHE=true`` for those workflows.

Byte-equivalent when OFF (default): no file I/O, no ``set_llm_cache`` call,
langchain behaves exactly as today.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, Generation

logger = logging.getLogger(__name__)

# Bump to invalidate the whole cache if the on-disk Generation schema changes
# (e.g. a langchain_core upgrade renames/restructures message fields). A
# mismatch reads as a miss, so a stale cache can never corrupt a run.
_CACHE_FORMAT_VERSION = 1

# Dispatch the on-disk message dict to its concrete class. ``ChatGeneration``'s
# ``message`` field is typed as the base ``BaseMessage``, so a naive
# ``model_validate`` rebuilds a plain BaseMessage and silently drops
# subclass-specific fields (``AIMessage.tool_calls`` / ``usage_metadata``).
# Promoting on the ``type`` tag preserves them. ``convert_to_messages`` would
# also work, but explicit dispatch is transparent and dependency-free.
_MESSAGE_TYPES = {
    "ai": AIMessage,
    "human": HumanMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
    "function": FunctionMessage,
}


def _key(prompt: str, llm_string: str) -> str:
    """Stable, filename-safe hash of the cache lookup tuple.

    ``llm_string`` (built by langchain) already encodes model + temperature +
    bound tools / response_format, and ``prompt`` is the serialized prompt, so
    this pair uniquely identifies one logical LLM call — structured-output
    calls hash apart from plain ``invoke`` calls automatically.
    """
    return hashlib.sha1(f"{llm_string}\x00{prompt}".encode("utf-8")).hexdigest()


def _serialize(return_val: Any) -> dict[str, Any]:
    # Dump the message object DIRECTLY, not via ChatGeneration.model_dump():
    # ``ChatGeneration.message`` is typed as the base ``BaseMessage``, so a
    # generation-level model_dump downcasts the message to that schema and
    # silently strips subclass fields (``AIMessage.tool_calls`` /
    # ``invalid_tool_calls`` / ``usage_metadata``). Serializing the concrete
    # message preserves them.
    gens = []
    for g in return_val:
        is_chat = isinstance(g, ChatGeneration)
        entry: dict[str, Any] = {
            "is_chat": is_chat,
            "generation_info": getattr(g, "generation_info", None),
        }
        if is_chat:
            entry["message"] = g.message.model_dump()
        else:
            entry["text"] = getattr(g, "text", "")
        gens.append(entry)
    return {"v": _CACHE_FORMAT_VERSION, "gens": gens}


def _deserialize(data: dict[str, Any]) -> list[Generation]:
    """Reconstruct a ``Sequence[Generation]`` from the on-disk payload.

    Chat generations are rebuilt with the message promoted to its concrete
    subclass (via the ``type`` tag) so ``AIMessage.tool_calls`` /
    ``usage_metadata`` survive the round-trip. Non-chat generations fall back
    to the plain text ``Generation``. An unrecoverable payload raises, which
    the caller turns into a cache miss.
    """
    gens: list[Generation] = []
    for d in data.get("gens", []):
        gen_info = d.get("generation_info")
        if d.get("is_chat"):
            msg_dict = d.get("message") or {}
            cls = _MESSAGE_TYPES.get(msg_dict.get("type")) if isinstance(msg_dict, dict) else None
            if cls is None:
                raise ValueError(f"unknown message type: {msg_dict.get('type')!r}")
            msg = cls.model_validate(msg_dict)
            gens.append(ChatGeneration(message=msg, generation_info=gen_info))
        else:
            gens.append(Generation(text=d.get("text", ""), generation_info=gen_info))
    return gens


class DiskLLMCache(BaseCache):
    """File-per-key disk cache implementing langchain's ``BaseCache``.

    One JSON file under ``cache_dir`` per ``(prompt, llm_string)``. Writes are
    atomic (tmp + ``os.replace``); corrupt or version-mismatched reads are
    treated as cache misses and the offending file evicted, so a bad cache can
    never break a run. An instance-level lock serializes same-key writes so
    concurrent analyst threads don't clobber each other's tmp file.
    """

    def __init__(self, cache_dir: str | os.PathLike[str], enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def lookup(self, prompt: str, llm_string: str):
        if not self.enabled:
            return None
        path = self.cache_dir / f"{_key(prompt, llm_string)}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("v") != _CACHE_FORMAT_VERSION:
                raise ValueError("cache format version mismatch")
            return _deserialize(data)
        except Exception as exc:  # noqa: BLE001 -- any failure -> miss + evict
            logger.warning("LLM cache: corrupt entry %s (%s); removing", path, exc)
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            return None

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        if not self.enabled:
            return
        path = self.cache_dir / f"{_key(prompt, llm_string)}.json"
        tmp = path.with_name(path.name + ".tmp")
        try:
            payload = _serialize(return_val)
            with self._lock:
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 -- caching must never break a run
            logger.warning("LLM cache: could not write %s (%s)", path, exc)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def clear(self, **kwargs: Any) -> None:
        if not self.enabled or not self.cache_dir.exists():
            return
        for entry in self.cache_dir.glob("*.json"):
            with contextlib.suppress(OSError):
                entry.unlink(missing_ok=True)
