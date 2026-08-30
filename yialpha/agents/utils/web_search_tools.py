"""Scoped web_search tool instances — one per analyst with a tool loop.

All instances share the name ``web_search`` and the same anti-fabrication
grounding contract; each charges its own Tavily budget *scope* so the per-run
call budget is split across analysts (news / market / fundamentals) instead
of one shared counter a search-happy first-mover could drain. Each analyst's
ToolNode registers its own instance, so the same-name instances never collide
in a registry. The scope only changes the budget bucket and the one-line
use-case hint steering the LLM toward searches its structured tools cannot
surface.
"""

from typing import Annotated

from langchain_core.tools import BaseTool, tool

from yialpha.dataflows.tavily import get_web_search

# Per-scope use-case hint embedded in the query-arg description: recent
# developments the analyst's structured tools may not cover.
_SCOPE_HINTS = {
    "news": (
        "regulatory actions, industry events, analyst commentary, "
        "earnings-call color"
    ),
    "market": (
        "macro and sector regime shifts, regulatory actions moving whole "
        "sectors, index/ETF flow narratives"
    ),
    "fundamentals": (
        "earnings-call color, guidance revisions, M&A and buyback "
        "announcements, industry supply/demand narrative"
    ),
}


def make_web_search_tool(scope: str = "news") -> BaseTool:
    """Build a ``web_search`` tool instance that charges the ``scope`` budget."""

    hint = _SCOPE_HINTS[scope]

    @tool
    def web_search(
        query: Annotated[
            str,
            "Free-text web search query, e.g. 'NVDA export control news this "
            "week' or 'semiconductor capex outlook 2026'. Use for recent "
            f"developments the structured data tools may not cover: {hint}.",
        ],
        max_results: Annotated[
            int, "Number of results to return (1-10); omit for 5"
        ] = 5,
    ) -> str:
        """
        Search the open web (Tavily) for qualitative context the structured
        data tools miss. Returns a cited digest: numbered results with title,
        source URL, and content snippet.

        Grounding rules: cite the source URL for every claim drawn from these
        results; treat snippets as qualitative context ONLY — any prices or
        figures in them are unverified text, never data values (numbers come
        exclusively from the structured data tools); if the tool reports
        WEB_SEARCH_UNAVAILABLE or no results, say so plainly and continue.
        """
        return get_web_search(query=query, max_results=max_results, scope=scope)

    return web_search


# Module-level singletons: the legacy name stays the news instance (existing
# imports keep working); the market/fundamentals instances are bound by their
# analysts and registered in their ToolNodes.
web_search = make_web_search_tool("news")
web_search_market = make_web_search_tool("market")
web_search_fundamentals = make_web_search_tool("fundamentals")
