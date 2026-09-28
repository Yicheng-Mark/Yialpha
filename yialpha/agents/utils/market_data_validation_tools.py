from typing import Annotated

from langchain_core.tools import tool

from yialpha.dataflows.market_data_validator import build_verified_market_snapshot


@tool
def get_verified_market_snapshot(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[
        int, "number of recent trading rows to include for sanity-checking"
    ] = 30,
) -> str:
    """Deterministic verification snapshot for exact market-data claims.

    Returns the latest OHLCV row on or before curr_date, common technical
    indicators, and recent closes. Call this before making exact claims about
    price levels, Bollinger bands, RSI, MACD, moving averages, support /
    resistance, or historical comparisons, and treat it as the source of truth.
    """
    # PIT clamp: curr_date comes from the LLM; the pinned analysis date is
    # the authority (round-7 — the snapshot is the prompt-mandated "source
    # of truth for exact OHLCV claims", exactly the wrong place for an
    # unclamped date to serve post-decision bars on a historical run).
    from yialpha.dataflows.utils import current_pit_end, get_analysis_date

    curr_date = str(
        current_pit_end(curr_date) or get_analysis_date() or curr_date)
    return build_verified_market_snapshot(symbol, curr_date, look_back_days)
