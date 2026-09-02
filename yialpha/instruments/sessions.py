"""Canonical session-calendar constants and session counting.

The three calendar identifiers were previously defined inline in
:mod:`yialpha.graph.routing`; V2.1 moves them here (the instruments package
is their canonical home) and routing re-exports the SAME names, so every
existing importer keeps working unchanged.

Caveat (known open item): Binance's exact TradFi perp session calendar — the
published trading hours and holiday schedule for tokenized-stock perpetuals —
is NOT yet machine-verified. The weekday approximation in
:func:`trading_sessions_between` is reserved for annualization work in V2.4
and must be disclosed as an approximation, never presented as the exchange's
official calendar.
"""
from __future__ import annotations

from datetime import date, timedelta

#: Pure-crypto contracts trade continuously (24/7) — no session boundaries.
SESSION_CONTINUOUS = "continuous_24_7"

#: A tokenized-stock perp follows Binance's PUBLISHED TradFi sessions (PR5
#: corrected the old "trades 24/7" assumption).
SESSION_BINANCE_TRADFI = "binance_published_tradfi_sessions"

#: Plain equities follow their listing exchange's session calendar.
SESSION_EXCHANGE = "listing_exchange_sessions"

#: Calendars whose sessions are approximated by NYSE-style weekdays.
_TRADFI_LIKE_CALENDARS = frozenset({SESSION_BINANCE_TRADFI, SESSION_EXCHANGE})


def trading_sessions_between(calendar: str, start_date: str, end_date: str) -> int:
    """Count trading sessions in the HALF-OPEN interval ``[start_date, end_date)``.

    Convention: ``start_date`` INCLUSIVE, ``end_date`` EXCLUSIVE — pairing with
    a daily returns index, the count equals the number of session boundaries
    crossed moving from ``start_date`` up to (but not including) ``end_date``.
    An empty window (``start_date == end_date``) is 0; an inverted window
    (``end_date < start_date``) is a caller bug and raises ``ValueError``.

    * ``SESSION_CONTINUOUS``: the plain calendar-day delta (every day trades).
    * ``SESSION_BINANCE_TRADFI`` / ``SESSION_EXCHANGE``: an NYSE-style weekday
      count (Mon-Fri). Weekend-only closure is an APPROXIMATION — Binance's
      exact TradFi perp calendar (hours + holidays) is not yet
      machine-verified; the count is reserved for annualization in V2.4.

    Raises ``ValueError`` for an unknown calendar, malformed ISO dates, or an
    inverted window.
    """
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as exc:
        raise ValueError(
            f"trading_sessions_between: malformed ISO date "
            f"({start_date!r}, {end_date!r})"
        ) from exc
    if end < start:
        raise ValueError(
            f"trading_sessions_between: end_date {end_date!r} precedes "
            f"start_date {start_date!r}"
        )
    days = (end - start).days
    if calendar == SESSION_CONTINUOUS:
        return days
    if calendar in _TRADFI_LIKE_CALENDARS:
        # Weekdays in [start, start + days): whole weeks contribute 5 each,
        # then the partial week is counted day by day.
        full_weeks, remainder = divmod(days, 7)
        count = full_weeks * 5
        for offset in range(remainder):
            if (start + timedelta(days=offset)).weekday() < 5:
                count += 1
        return count
    raise ValueError(f"trading_sessions_between: unknown session calendar {calendar!r}")
