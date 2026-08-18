"""Tavily web-search vendor — general open-web retrieval for the news analyst.

The news pipeline's sources are vendor-shaped (yfinance/AV headlines, FRED
series, prediction markets): there is no way to ask "what happened around
this company/industry this week" of the open web. This module adds exactly
that, via Tavily's agent-oriented search API (free tier: 1,000 credits/month,
https://app.tavily.com; key in ``TAVILY_API_KEY``).

Positioning (enforced by the news analyst's prompt, mirrored here for the
next reader): search results are *qualitative context*, never *data values*.
Any price/percentage appearing in a snippet is unverified second-hand text;
numbers must come from the structured data tools. The tool formats results
with their source URLs so claims stay citable.

Unlike router-routed vendors this is a direct tool (same contract as the
price-structure tools), so degradation handling lives here and MUST keep
recording evidence — key missing / budget exhausted / timeout / API error all
emit a ``WEB_SEARCH_UNAVAILABLE`` sentinel and a ``record_sentinel`` event
(optional-category kind: never aborts the run, but stays visible in
``data_quality``). No disk cache: queries are LLM-generated and news-fresh,
so replaying a stale response would defeat the purpose.

A per-run call budget (ContextVar, like the quality ledger, so concurrent
batch workers keep separate counters) caps credits: the LLM tool loop cannot
burn the free tier from inside one analysis. The budget is split by *scope*
(one per analyst that binds a web_search instance — news / market /
fundamentals) so a search-happy analyst cannot starve the others; the split
defaults to news:8 / market:5 / fundamentals:2 (sum = the total cap) and is
tunable via ``YIAGENTS_TAVILY_BUDGET_SPLIT``.

Key pool: ``TAVILY_API_KEYS`` (comma-separated) plus the single
``TAVILY_API_KEY`` form a round-robin pool — multiple free-tier keys
multiply the monthly credit allowance. A key answering 401/403/429 is
removed from the pool for the rest of the run (revived on the next run) and
the call immediately retries on the next key; only when every key is
unavailable does the sentinel fire. Key rotation is recorded in the quality
ledger (capacity degradation must stay visible, never silent).
"""

from __future__ import annotations

import logging
import os
import threading
from contextvars import ContextVar

import requests

from .netretry import with_transient_retry
from .quality import KIND_OPTIONAL_UNAVAILABLE, record_sentinel

logger = logging.getLogger(__name__)

TAVILY_API_BASE = "https://api.tavily.com"

#: Default network timeout (seconds); override with YIAGENTS_TAVILY_TIMEOUT_S.
DEFAULT_TIMEOUT_S = 20
#: Default per-run search-call cap (TOTAL across scopes); override with
#: YIAGENTS_TAVILY_MAX_CALLS_PER_RUN.
#: 15 assumes the multi-key pool (TAVILY_API_KEYS: ~5,000 free credits/
#: month); a single-key setup may want 6 — the cap exists so a looping model
#: cannot drain the monthly credits from inside one analysis.
DEFAULT_MAX_CALLS_PER_RUN = 15
#: Default per-scope split of the total cap (news / market / fundamentals
#: analysts, one web_search instance each). Sum == DEFAULT_MAX_CALLS_PER_RUN;
#: overriding only the total rescales this split proportionally, while
#: YIAGENTS_TAVILY_BUDGET_SPLIT replaces it wholesale (scopes omitted get 0).
DEFAULT_BUDGET_SPLIT: dict[str, int] = {"news": 8, "market": 5, "fundamentals": 2}
#: Comma-separated extra keys pooled with the single TAVILY_API_KEY.
KEYS_POOL_ENV = "TAVILY_API_KEYS"
#: Cap on Tavily's max_results (API allows more; more would flood context).
MAX_RESULTS_CEILING = 10
#: Per-result content snippet cap — snippets arrive pre-chunked by Tavily but
#: can still run long; keep each result tight so 5 results fit the analyst
#: context comfortably.
_SNIPPET_CHAR_CAP = 900

#: Router-style method tag used in quality events for every sentinel below.
_METHOD = "web_search"

#: Per-scope charged-call counts for the current run. Default None (mutable
#: ContextVar defaults are a shared-state hazard — B039); initialized to a
#: per-context dict on first use.
#:
#: MUTATION CONTRACT (2026-08-17 live-run regression): langchain propagates a
#: COPIED context into ToolNode's executor threads. A copied context shares the
#: bound *object* but not later ``set()`` calls — so in-place mutation
#: (``calls[scope] = calls.get(scope, 0) + 1``) is visible to the run root,
#: while the copy-on-write rebinding this module used before made every real
#: run record ``web_search_usage: 0`` while the model actually searched (and
#: let the budget gate, which reads the same counters, be bypassed — 12 calls
#: under an 8-call cap in the first live run). Same pattern as the quality
#: ledger's shared list. Run isolation comes from ``reset_run_budget()``
#: binding fresh objects in the runner's root context at run start.
_calls_var: ContextVar[dict[str, int] | None] = ContextVar(
    "yiagents_tavily_calls", default=None
)

#: Serializes counter/cursor/dead-set updates: parallel tool calls in one
#: ToolNode batch each hold a context copy sharing these objects, and
#: read-modify-write on a dict is not atomic under the GIL.
_state_lock = threading.Lock()


def _calls_map() -> dict[str, int]:
    calls = _calls_var.get()
    if calls is None:
        calls = {}
        _calls_var.set(calls)
    return calls


#: Round-robin cursor and the set of keys removed from this run's pool
#: (401/403/429). Both reset with the budget at run start — a quota-dead key
#: may recover by the next run, and each run should start its rotation
#: deterministically. Same mutation contract as ``_calls_var``: boxed in
#: mutable containers because ints/frozensets can only be rebound, and a
#: rebind inside a ToolNode worker's context copy never reaches the root.
_key_cursor_var: ContextVar[dict[str, int] | None] = ContextVar(
    "yiagents_tavily_key_cursor", default=None
)
_dead_keys_var: ContextVar[set[int] | None] = ContextVar(
    "yiagents_tavily_dead_keys", default=None
)


def _cursor_box() -> dict[str, int]:
    box = _key_cursor_var.get()
    if box is None:
        box = {"i": 0}
        _key_cursor_var.set(box)
    return box


def _dead_keys() -> set[int]:
    dead = _dead_keys_var.get()
    if dead is None:
        dead = set()
        _dead_keys_var.set(dead)
    return dead


def api_key_pool() -> list[str]:
    """Active key pool: ``TAVILY_API_KEYS`` entries + ``TAVILY_API_KEY``.

    Order-stable, exact-duplicate-removed. Public so ``config-check`` can
    show the pool size (never the values) and tests can pin the parse.
    """
    keys: list[str] = []
    raw_pool = os.getenv(KEYS_POOL_ENV)
    if raw_pool:
        keys.extend(k.strip() for k in raw_pool.split(",") if k.strip())
    single = (os.getenv("TAVILY_API_KEY") or "").strip()
    if single and single not in keys:
        keys.append(single)
    return keys


def _mark_key_dead(key_idx: int, pool_size: int, status: object) -> None:
    with _state_lock:
        _dead_keys().add(key_idx)
    detail = (
        f"key #{key_idx + 1}/{pool_size} returned HTTP {status}; "
        "removed from this run's pool and rotating to the next key"
    )
    # Capacity degradation is still degradation — keep it in the ledger so a
    # shrunken pool is visible in data_quality, never silent.
    record_sentinel(_METHOD, KIND_OPTIONAL_UNAVAILABLE, detail)
    logger.warning("tavily: %s", detail)


def _pick_key(keys: list[str]) -> tuple[str, int] | None:
    """Round-robin pick among the pool, skipping keys dead this run."""
    n = len(keys)
    for _ in range(n):
        with _state_lock:
            box = _cursor_box()
            idx = box["i"] % n
            box["i"] += 1
            dead = _dead_keys()
        if idx not in dead:
            return keys[idx], idx
    return None


def reset_run_budget() -> None:
    """Zero the per-run per-scope call counters and revive the key pool.

    Called wherever ``quality.ensure_run_context()`` is bound (graph run +
    CLI), so a long-lived worker process cannot carry one run's budget into
    the next and starve it of searches, and a key that went quota-dead mid-run
    gets retried on the next run (free-tier quotas recover monthly, and 401s
    may be transient key-management fixes).
    """
    with _state_lock:
        _calls_var.set({})
        _key_cursor_var.set({"i": 0})
        _dead_keys_var.set(set())


def run_usage() -> dict[str, int]:
    """Per-scope charged-call counts for the current run (0-filled).

    Read by the graph's state logger so ``full_states_log_*.json`` answers
    "did web_search actually fire, and for which analyst" without replaying
    message logs. Counts calls that passed the budget gate — including ones
    that later degraded on transport/API errors (those are visible in the
    quality ledger instead).
    """
    calls = _calls_var.get() or {}
    return {scope: calls.get(scope, 0) for scope in DEFAULT_BUDGET_SPLIT}


def _charge(scope: str) -> None:
    # In-place mutation only — never rebind. A rebind inside a ToolNode
    # worker's context copy is invisible to the run root (see the
    # MUTATION CONTRACT note at ``_calls_var``).
    with _state_lock:
        calls = _calls_map()
        calls[scope] = calls.get(scope, 0) + 1


def parse_budget_split(raw: str) -> dict[str, int] | None:
    """Parse a ``"scope:count,scope:count"`` budget-split string.

    Returns ``None`` (after a warning) when malformed — callers fall back to
    the default split, mirroring the non-numeric env fallbacks elsewhere.
    Public so ``yiagents config-check`` validates the same parse the runtime
    uses.
    """
    split: dict[str, int] = {}
    try:
        for part in raw.split(","):
            name, sep, count = part.partition(":")
            name = name.strip()
            if not name or not sep:
                raise ValueError(part)
            n = int(count.strip())
            if n < 0:
                raise ValueError(part)
            split[name] = n
        if not split:
            raise ValueError(raw)
        return split
    except ValueError:
        logger.warning(
            "Ignoring malformed YIAGENTS_TAVILY_BUDGET_SPLIT=%r; expected "
            "'scope:count' pairs, e.g. 'news:3,market:2,fundamentals:1'.",
            raw,
        )
        return None


def _scaled_default_split(total: int) -> dict[str, int]:
    """Scale DEFAULT_BUDGET_SPLIT so the per-scope caps sum to ``total``.

    Largest-remainder rounding; ties break toward the heavier scope, so the
    sum always equals the total cap exactly.
    """
    if total <= 0:
        return dict.fromkeys(DEFAULT_BUDGET_SPLIT, 0)
    denom = sum(DEFAULT_BUDGET_SPLIT.values())
    split = {
        scope: total * weight // denom
        for scope, weight in DEFAULT_BUDGET_SPLIT.items()
    }
    leftover = total - sum(split.values())
    by_shortfall = sorted(
        DEFAULT_BUDGET_SPLIT,
        key=lambda s: (
            -(total * DEFAULT_BUDGET_SPLIT[s] % denom),
            -DEFAULT_BUDGET_SPLIT[s],
        ),
    )
    for scope in by_shortfall[:leftover]:
        split[scope] += 1
    return split


def _budget_split(total: int) -> dict[str, int]:
    """Effective per-scope caps: explicit env split, else scaled default."""
    raw = os.getenv("YIAGENTS_TAVILY_BUDGET_SPLIT")
    if raw is not None:
        explicit = parse_budget_split(raw)
        if explicit is not None:
            return explicit
    return _scaled_default_split(total)


def _max_calls_per_run() -> int:
    raw = os.getenv("YIAGENTS_TAVILY_MAX_CALLS_PER_RUN")
    if raw is None:
        return DEFAULT_MAX_CALLS_PER_RUN
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring non-numeric YIAGENTS_TAVILY_MAX_CALLS_PER_RUN=%r", raw,
        )
        return DEFAULT_MAX_CALLS_PER_RUN


def _timeout_s() -> float:
    raw = os.getenv("YIAGENTS_TAVILY_TIMEOUT_S")
    if raw is None:
        return float(DEFAULT_TIMEOUT_S)
    try:
        return max(1.0, float(raw))
    except ValueError:
        logger.warning("Ignoring non-numeric YIAGENTS_TAVILY_TIMEOUT_S=%r", raw)
        return float(DEFAULT_TIMEOUT_S)


def _sentinel(detail: str) -> str:
    """Emit the degradation sentinel + quality evidence for the current run."""
    record_sentinel(_METHOD, KIND_OPTIONAL_UNAVAILABLE, detail)
    return (
        f"WEB_SEARCH_UNAVAILABLE: {detail} Do not fabricate web findings; "
        "state plainly that open-web context is unavailable and continue "
        "with the structured data tools."
    )


def get_web_search(
    query: str, max_results: int = 5, scope: str = "news"
) -> str:
    """Search the open web via Tavily and format results as a cited digest.

    Args:
        query: Free-text search query (company developments, industry events,
            analyst commentary — anything the vendor news tools miss).
        max_results: Number of results to request (clamped to 1..10).
        scope: Budget scope charging this call — one of the analysts that
            binds a web_search instance ("news" / "market" / "fundamentals").
            Default "news" keeps pre-scoping callers unchanged.

    Returns:
        A markdown digest: one numbered entry per result with title, source
        URL, and a content snippet. On any degradation (missing key, exhausted
        per-run budget — this scope's cap or the run total —, timeout, API
        error) returns a ``WEB_SEARCH_UNAVAILABLE`` sentinel string and
        records an optional-category quality event — this tool never raises
        into the agent loop.
    """
    query = (query or "").strip()
    if not query:
        return _sentinel("empty query.")

    keys = api_key_pool()
    if not keys:
        # Static precondition, checked BEFORE the budget: a key-less run must
        # burn nothing and the message must point at the actual fix
        # (configure a key), not at raising the budget caps.
        return _sentinel(
            "Neither TAVILY_API_KEYS nor TAVILY_API_KEY is set. Get a free "
            "key at https://app.tavily.com and put it in .env."
        )

    total_cap = _max_calls_per_run()
    calls = _calls_map()
    split = _budget_split(total_cap)
    scope_cap = split.get(scope)
    if scope_cap is None:
        return _sentinel(
            f"scope '{scope}' has no budget allocation in "
            f"YIAGENTS_TAVILY_BUDGET_SPLIT (allocated scopes: "
            f"{', '.join(sorted(split))}); ask the operator to allocate it."
        )
    scope_used = calls.get(scope, 0)
    if scope_used >= scope_cap:
        return _sentinel(
            f"per-run search budget exhausted for the {scope} analyst "
            f"({scope_used}/{scope_cap} calls used; raise "
            "YIAGENTS_TAVILY_BUDGET_SPLIT or YIAGENTS_TAVILY_MAX_CALLS_PER_RUN "
            "if more angles are needed)."
        )
    total_used = sum(calls.values())
    if total_used >= total_cap:
        return _sentinel(
            f"per-run total search budget exhausted ({total_used}/{total_cap} "
            "calls used across analysts; raise "
            "YIAGENTS_TAVILY_MAX_CALLS_PER_RUN if more angles are needed)."
        )
    _charge(scope)

    n_results = max(1, min(int(max_results), MAX_RESULTS_CEILING))
    payload = {
        "query": query,
        "max_results": n_results,
        "search_depth": "basic",  # 1 credit/call; "advanced" costs 2 and is
        # no better for the analyst's headline-angle use case.
    }

    response = None
    last_key_status: object = "?"
    for _ in range(len(keys)):
        picked = _pick_key(keys)
        if picked is None:
            break  # every key is dead this run
        api_key, key_idx = picked

        def _http(api_key: str = api_key) -> requests.Response:
            # Default-arg binding: this closure is created inside a loop.
            # Plain requests.post -> fresh Session with trust_env=True, so
            # the shared HTTP(S)_PROXY SOCKS5 route applies (US-hosted API).
            response_ = requests.post(
                f"{TAVILY_API_BASE}/search",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_timeout_s(),
            )
            response_.raise_for_status()
            return response_

        try:
            response = with_transient_retry(
                _http, vendor="tavily",
                retry_on=(
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ),
            )
            break
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            last_key_status = status
            logger.warning("tavily: HTTP %s for query %r", status, query[:80])
            if status in (401, 403, 429) and len(keys) > 1:
                # Key/quota problem on THIS key — drop it for the rest of
                # the run and retry the same call on the next key.
                _mark_key_dead(key_idx, len(keys), status)
                continue
            return _sentinel(f"Tavily API returned HTTP {status}.")
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            # Transport-level (proxy down): rotating keys cannot help.
            logger.warning("tavily: transport failure (%r)", exc)
            return _sentinel(f"Tavily unreachable ({exc.__class__.__name__}).")

    if response is None:
        return _sentinel(
            f"all {len(keys)} Tavily keys are unavailable this run "
            f"(last: HTTP {last_key_status}). Get fresh keys at "
            "https://app.tavily.com and update TAVILY_API_KEYS."
        )

    try:
        results = (response.json() or {}).get("results") or []
    except ValueError:
        return _sentinel("Tavily returned a non-JSON body.")

    if not results:
        return (
            f"Web search for '{query}': no results found. Do not speculate "
            "about what such results might have said."
        )

    lines = [f"## Web search: {query}"]
    for i, r in enumerate(results, start=1):
        title = str(r.get("title") or "(untitled)").strip()
        url = str(r.get("url") or "").strip()
        content = str(r.get("content") or "").strip()
        if len(content) > _SNIPPET_CHAR_CAP:
            content = content[:_SNIPPET_CHAR_CAP].rsplit(" ", 1)[0] + " [...]"
        lines.append(f"{i}. **{title}**")
        if url:
            lines.append(f"   Source: {url}")
        if content:
            lines.append(f"   {content}")
    lines.append(
        "_Numbers appearing above are unverified snippet text — cite as "
        "qualitative context only; data values must come from the structured "
        "data tools._"
    )
    return "\n".join(lines)
