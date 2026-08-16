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
burn the free tier from inside one analysis.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar

import requests

from .netretry import with_transient_retry
from .quality import KIND_OPTIONAL_UNAVAILABLE, record_sentinel

logger = logging.getLogger(__name__)

TAVILY_API_BASE = "https://api.tavily.com"

#: Default network timeout (seconds); override with YIAGENTS_TAVILY_TIMEOUT_S.
DEFAULT_TIMEOUT_S = 20
#: Default per-run search-call cap; override with YIAGENTS_TAVILY_MAX_CALLS_PER_RUN.
#: 6 covers a news analyst doing 2-3 angles plus a follow-up without letting
#: a looping model drain the monthly free credits in one batch.
DEFAULT_MAX_CALLS_PER_RUN = 6
#: Cap on Tavily's max_results (API allows more; more would flood context).
MAX_RESULTS_CEILING = 10
#: Per-result content snippet cap — snippets arrive pre-chunked by Tavily but
#: can still run long; keep each result tight so 5 results fit the analyst
#: context comfortably.
_SNIPPET_CHAR_CAP = 900

#: Router-style method tag used in quality events for every sentinel below.
_METHOD = "web_search"

_calls_var: ContextVar[int] = ContextVar("yiagents_tavily_calls", default=0)


def reset_run_budget() -> None:
    """Zero the per-run call counter.

    Called wherever ``quality.ensure_run_context()`` is bound (graph run +
    CLI), so a long-lived worker process cannot carry one run's budget into
    the next and starve it of searches.
    """
    _calls_var.set(0)


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


def get_web_search(query: str, max_results: int = 5) -> str:
    """Search the open web via Tavily and format results as a cited digest.

    Args:
        query: Free-text search query (company developments, industry events,
            analyst commentary — anything the vendor news tools miss).
        max_results: Number of results to request (clamped to 1..10).

    Returns:
        A markdown digest: one numbered entry per result with title, source
        URL, and a content snippet. On any degradation (missing key, exhausted
        per-run budget, timeout, API error) returns a ``WEB_SEARCH_UNAVAILABLE``
        sentinel string and records an optional-category quality event — this
        tool never raises into the agent loop.
    """
    query = (query or "").strip()
    if not query:
        return _sentinel("empty query.")

    used = _calls_var.get()
    cap = _max_calls_per_run()
    if used >= cap:
        return _sentinel(
            f"per-run search budget exhausted ({used}/{cap} calls used; "
            "raise YIAGENTS_TAVILY_MAX_CALLS_PER_RUN if more angles are needed)."
        )
    _calls_var.set(used + 1)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return _sentinel(
            "TAVILY_API_KEY is not set. Get a free key at "
            "https://app.tavily.com and put it in .env."
        )

    n_results = max(1, min(int(max_results), MAX_RESULTS_CEILING))
    payload = {
        "query": query,
        "max_results": n_results,
        "search_depth": "basic",  # 1 credit/call; "advanced" costs 2 and is
        # no better for the analyst's headline-angle use case.
    }

    def _http() -> requests.Response:
        # Plain requests.post -> fresh Session with trust_env=True, so the
        # shared HTTP(S)_PROXY SOCKS5 route applies (US-hosted API).
        response = requests.post(
            f"{TAVILY_API_BASE}/search",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_timeout_s(),
        )
        response.raise_for_status()
        return response

    try:
        response = with_transient_retry(
            _http, vendor="tavily",
            retry_on=(requests.exceptions.ConnectionError, requests.exceptions.Timeout),
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        logger.warning("tavily: HTTP %s for query %r", status, query[:80])
        return _sentinel(f"Tavily API returned HTTP {status}.")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        logger.warning("tavily: transport failure (%r)", exc)
        return _sentinel(f"Tavily unreachable ({exc.__class__.__name__}).")

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
