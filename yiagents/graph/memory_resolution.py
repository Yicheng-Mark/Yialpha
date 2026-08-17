"""Standalone memory-log pending-entry resolution.

Extracted from ``YiAgentsGraph._resolve_pending_entries`` (2026-08-16, T2
batch) so the same resolution runs in two places without drifting apart:

* the graph — at the start of every same-ticker run (unchanged behaviour);
* ``yiagents memory-resolve`` — on demand, for ALL tickers, without running
  a full analysis (previously a pending entry for a ticker you stopped
  analyzing stayed pending forever, and the reflection layer never saw it).

The resolution needs an LLM (one reflection call per resolved entry) — the
caller supplies a configured ``Reflector``; everything else is deterministic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

#: A pending memory-log entry older than this (in days) that still cannot be
#: resolved is logged at WARNING so it does not silently accumulate forever.
STALE_PENDING_DAYS = 7


def _entry_precedes_cutoff(entry: dict[str, Any], cutoff: datetime) -> bool:
    """Fail-closed date gate for pending reflection outcomes."""
    try:
        return datetime.strptime(str(entry["date"])[:10], "%Y-%m-%d") < cutoff
    except (KeyError, TypeError, ValueError):
        return False


def _warn_if_stale_pending(ticker: str, trade_date: str, now: datetime) -> None:
    """Log a WARNING when a pending entry is old enough to be concerning.

    Pending entries that can never resolve (delisted ticker, permanent data
    gap) previously accumulated indefinitely with no signal. This makes the
    silent-stuck state observable without changing the log format.
    """
    try:
        entry_date = datetime.strptime(trade_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return
    age_days = (now - entry_date).days
    if age_days > STALE_PENDING_DAYS:
        logger.warning(
            "Pending memory-log entry for %s @ %s is %d days old and still "
            "unresolved (price data unavailable). It may be stuck due to "
            "delisting or a permanent data gap.",
            ticker, trade_date, age_days,
        )


def resolve_pending_entries(
    memory_log: Any,
    reflector: Any,
    ticker: str,
    *,
    benchmark: str = "SPY",
    as_of_date: str | None = None,
    holding_days: int = 5,
) -> int:
    """Resolve pending entries for ``ticker``; returns how many were resolved.

    Fetches returns for each same-ticker pending entry via
    :func:`yiagents.accuracy.fetch_returns_yf`, generates reflections, then
    writes all updates in a single atomic batch write. Entries whose price
    data is not yet available (too recent or delisted) are skipped — and
    logged at WARNING when old enough to look permanently stuck.
    """
    from yiagents.accuracy import fetch_returns_yf

    try:
        cutoff = (
            datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d")
            if as_of_date is not None
            else datetime.now()
        )
    except (TypeError, ValueError):
        logger.warning("Invalid memory as-of date %r; skipping reflection", as_of_date)
        return 0

    cutoff_str = cutoff.strftime("%Y-%m-%d")
    pending = [
        entry
        for entry in memory_log.get_pending_entries()
        if entry["ticker"] == ticker
        and _entry_precedes_cutoff(entry, cutoff)
    ]
    if not pending:
        return 0

    updates = []
    for entry in pending:
        raw, alpha, days = fetch_returns_yf(
            ticker,
            entry["date"],
            benchmark=benchmark,
            holding_days=holding_days,
            as_of_date=cutoff_str,
        )
        if raw is None:
            # Price not available yet — but if this entry is old, flag it so
            # it does not silently accumulate forever (delisted / bad data).
            _warn_if_stale_pending(ticker, entry["date"], cutoff)
            continue  # price not available yet — try again next run
        reflection = reflector.reflect_on_final_decision(
            final_decision=entry.get("decision", ""),
            raw_return=raw,
            alpha_return=alpha if alpha is not None else 0.0,
            benchmark_name=benchmark,
        )
        updates.append({
            "ticker": ticker,
            "trade_date": entry["date"],
            "raw_return": raw,
            "alpha_return": alpha,
            "holding_days": days,
            "reflection": reflection,
            "available_date": cutoff_str,
        })

    if updates:
        memory_log.batch_update_with_outcomes(updates)
    return len(updates)
