"""LangChain ``@tool`` wrappers for the crypto-native fundamentals vendors.

A thin pass-through to ``route_to_vendor`` — same shape as
``a_share_native_tools`` / ``binance_perp_tools`` — so the optional-category
fallback (unresolved symbol / vendor block / beyond-window historical date ->
sentinel) is reused verbatim. Only the fundamentals analyst advertises these,
and only when ``YIALPHA_CRYPTO_NATIVE`` is on **and** the instrument
classifies as pure crypto (``pure_crypto_perp`` / ``crypto_spot``); the
fundamentals ``ToolNode`` carries them too so dispatched tool_calls resolve,
but for a default run (or any equity / tokenized-stock-perp ticker) the LLM
never names them so they stay dormant.

* ``get_crypto_token_fundamentals`` — token-level fundamentals from CoinGecko
  (market cap / rank, supply schedule, FDV, ATH/ATL, price change; historical
  runs get the PIT daily series, free tier caps history at trailing ~365 days).
* ``get_crypto_network_fundamentals`` — network/protocol fundamentals from
  DefiLlama (chain TVL history, protocol TVL history for governance tokens,
  live chain ranking; full-history daily series, PIT-filtered).
"""

from typing import Annotated

from langchain_core.tools import tool

from yialpha.dataflows.interface import route_to_vendor
from yialpha.dataflows.utils import current_pit_end, get_analysis_date


def _pit_curr_date(curr_date):
    """Clamp an LLM-supplied curr_date to the run's pinned analysis date.

    Same guard as the other thin tool layers: these vendors PIT-filter rows
    by the passed ``curr_date`` verbatim, so an LLM echoing a cached header's
    today-date would admit post-decision rows into a backtest. No-op in live
    mode (no analysis date pinned).
    """
    return current_pit_end(curr_date) or get_analysis_date() or curr_date


@tool
def get_crypto_token_fundamentals(
    ticker: Annotated[str, "pure-crypto contract symbol, e.g. BTCUSDT / ETHUSDT / 1000PEPEUSDT"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "historical table window in days (default 180)"] = 180,
) -> str:
    """Retrieve token-level fundamentals for a pure-crypto asset (CoinGecko).

    Live runs: current snapshot — price, market cap + rank, FDV, circulating /
    total / max supply, ATH/ATL with dates, multi-horizon price change,
    watchlist count. Historical runs: point-in-time daily series (price /
    market cap / volume / derived supply, rows filtered to date <= curr_date)
    from the free tier's trailing ~365-day window — older as-of dates
    honestly report 'data not available'. Supply schedule vs circulating
    supply indicates unlock-dilution risk; FDV >> market cap = unvested
    overhang; volume / market cap is a liquidity proxy. If a field reads
    n/a or the tool reports no data, write 'data not available' — do not
    estimate.
    Args:
        ticker: Pure-crypto contract symbol, e.g. BTCUSDT.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Historical table window in days (default 180).
    Returns:
        str: Markdown snapshot/table, or an explicit not-available message.
    """
    return route_to_vendor(
        "get_crypto_token_fundamentals",
        ticker,
        _pit_curr_date(curr_date),
        look_back_days,
    )


@tool
def get_crypto_network_fundamentals(
    ticker: Annotated[str, "pure-crypto contract symbol, e.g. BTCUSDT / UNIUSDT"],
    curr_date: Annotated[str, "current/as-of date in yyyy-mm-dd (the trade date)"],
    look_back_days: Annotated[int, "series window in days (default 180)"] = 180,
) -> str:
    """Retrieve network & protocol fundamentals for a pure-crypto asset (DefiLlama).

    Two evidence legs, whichever resolve: protocol TVL history when the base
    symbol is a DeFi protocol's governance token (highest-TVL version,
    disclosed), and chain TVL history for the base's chain. Live runs add the
    chain's current TVL ranking; historical runs omit it (current-only
    snapshot). Full-history daily series, point-in-time filtered to
    date <= curr_date. Rising TVL signals growing committed capital; TVL
    divergence from price flags multiple expansion/compression. If a section
    reports no rows or no coverage, write 'data not available' for it — do
    not estimate.
    Args:
        ticker: Pure-crypto contract symbol, e.g. UNIUSDT.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Series window in days (default 180).
    Returns:
        str: Markdown sections with honest omission notes, or an explicit
        not-available message.
    """
    return route_to_vendor(
        "get_crypto_network_fundamentals",
        ticker,
        _pit_curr_date(curr_date),
        look_back_days,
    )
