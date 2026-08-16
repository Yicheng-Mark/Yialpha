from typing import Annotated

from langchain_core.tools import tool

from yiagents.dataflows.tavily import get_web_search


@tool
def web_search(
    query: Annotated[
        str,
        "Free-text web search query, e.g. 'NVDA export control news this week' "
        "or 'semiconductor capex outlook 2026'. Use for recent developments "
        "the news vendors may not cover: regulatory actions, industry events, "
        "analyst commentary, earnings-call color.",
    ],
    max_results: Annotated[
        int, "Number of results to return (1-10); omit for 5"
    ] = 5,
) -> str:
    """
    Search the open web (Tavily) for qualitative context the vendor news
    tools miss. Returns a cited digest: numbered results with title, source
    URL, and content snippet.

    Grounding rules: cite the source URL for every claim drawn from these
    results; treat snippets as qualitative context ONLY — any prices or
    figures in them are unverified text, never data values (numbers come
    exclusively from the structured data tools); if the tool reports
    WEB_SEARCH_UNAVAILABLE or no results, say so plainly and continue.
    """
    return get_web_search(query=query, max_results=max_results)
