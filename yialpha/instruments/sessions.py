"""Canonical session-calendar constants and session counting.

The three calendar identifiers were previously defined inline in
:mod:`yialpha.graph.routing`; V2.1 moves them here (the instruments package
is their canonical home) and routing re-exports the SAME names, so every
existing importer keeps working unchanged.

Verification (2026-09-04): Binance tokenized-stock perpetuals trade 24/7 —
klines machine evidence (MUUSDT, listed 2026-04-07) shows they traded WITH
VOLUME through every full US-market holiday (2026-05-25 / 06-19 / 07-03) and
every weekend (14/14 weekend bars carried volume, ~1/4–1/3 of a normal day;
zero-volume days = 0). Annualization therefore uses the continuous 365
factor and does NOT consume the weekday approximation in
:func:`trading_sessions_between` anymore; that approximation survives only
as the date-only ET session bucket / listing-exchange convention, never
presented as the exchange's official calendar.
"""
from __future__ import annotations

from datetime import date, timedelta

#: Pure-crypto contracts trade continuously (24/7) — no session boundaries.
SESSION_CONTINUOUS = "continuous_24_7"

#: A tokenized-stock perp's registry/routing calendar label (V2.2 frozen
#: value). Klines machine evidence (2026-09-04) later VERIFIED the original
#: "trades 24/7" reality — full US-market holidays and weekends carry
#: volume — so annualization uses the continuous 365 factor; the label
#: itself must not change (it feeds deterministic classifications).
SESSION_BINANCE_TRADFI = "binance_published_tradfi_sessions"

#: Plain equities follow their listing exchange's session calendar.
SESSION_EXCHANGE = "listing_exchange_sessions"

#: CALENDAR DISCLOSURE (verified 2026-09-04; supersedes the 2026-09-03
#: weekday-count assumption): one real klines probe (MUUSDT, listed
#: 2026-04-07) shows Binance stock perps trade 24/7 — all full US-market
#: holidays in the window (2026-05-25 / 06-19 / 07-03) and all 14 weekend
#: bars carried volume (~1/4–1/3 of a normal day; zero-volume days = 0),
#: so both 261 (2025 working days) and 252 (NYSE) are falsified and the
#: sessions/year factor is the continuous 365. The backtest engine records
#: this disclosure in ``config_summary["session_calendar_assumption"]``
#: and renders it in the report. Verification did NOT change the API —
#: only the counts and this constant's wording.
SESSION_CALENDAR_CAVEAT = (
    "klines-verified 24/7 calendar (2026-09-04 probe: full US-market "
    "holidays and weekends traded with volume); sessions/year factor 365"
)

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
      count (Mon-Fri) — a date-only approximation. Klines evidence
      (2026-09-04) verified stock perps trade 24/7, so annualization no
      longer consumes this count (factor 365); it remains for the ET
      session bucket / listing-exchange disclosure only.

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
