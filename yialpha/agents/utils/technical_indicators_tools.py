from typing import Annotated

from langchain_core.tools import tool

from yialpha.dataflows.interface import route_to_vendor


@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve technical indicator readings for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): One or more comma-separated indicator names, e.g.
            'rsi' or 'macd,rsi,close_50_sma'. Prefer ONE call with several
            names over one call per indicator — the window is identical and
            it saves round-trips.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # PIT clamp: curr_date comes from the LLM; the pinned analysis date is
    # the authority (round-7 — same guard as the news vendors / weekly /
    # price-structure tools; the indicator-window vendors and the AV
    # indicator fallback behind this route never clamped on their own, so a
    # model echoing the cache's "Data retrieved on" header date served
    # post-decision rows on a historical run).
    from yialpha.dataflows.utils import current_pit_end, get_analysis_date

    curr_date = str(
        current_pit_end(curr_date) or get_analysis_date() or curr_date)
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)
