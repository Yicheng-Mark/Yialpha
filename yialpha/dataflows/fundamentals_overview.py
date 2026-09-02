"""Aggregate fundamentals overview — SEC filing facts + Yahoo real-time valuation.

PR5 made the whole ``fundamental_data`` chain SEC-first, which is right for
the STATEMENTS (SEC ``filed`` dates are the PIT ground truth) but wrong for
the OVERVIEW as a side effect: the SEC overview is seven as-reported filing
facts and carries none of the real-time valuation fields (market cap, PE,
forward PE, beta, 52-week range, margins) that Yahoo ``.info`` supplies — and
because SEC succeeds first, the chain never read them again. Live US-equity
runs silently lost half their overview.

This module is the fix's overview half: a dedicated ``get_fundamentals``
vendor that MERGES the two sources instead of chaining them —

* the SEC half is the statement-grade factual anchor (as-reported XBRL
  facts, PIT by ``filed <= curr_date``), served on every date;
* the Yahoo half is appended ONLY when legitimately current (live runs);
  on a historical replay date Yahoo is PIT-refused exactly as before and
  the overview degrades to SEC-only with an explicit note.

Either half failing degrades to the other half with a disclosure line; only
when BOTH fail does the vendor raise ``NoMarketDataError`` so the router
emits its sentinel and tries the rest of the configured chain.

Statements stay on the plain SEC-first chain (``tool_vendors`` overrides only
``get_fundamentals`` — see default_config).
"""
from __future__ import annotations

import logging
from datetime import datetime

from .errors import NoMarketDataError
from .sec_edgar import get_fundamentals as _sec_fundamentals
from .symbol_utils import normalize_symbol
from .y_finance import (
    _cached_ticker_info,
    _overview_lines,
    overview_would_leak_future,
)

logger = logging.getLogger(__name__)


def _yahoo_valuation_lines(ticker: str, canonical: str) -> list[str]:
    """Labeled real-time valuation lines from the cached ``.info`` snapshot."""
    info = _cached_ticker_info(ticker, canonical)
    if not info:
        raise NoMarketDataError(ticker, canonical, "no Yahoo info returned")
    lines = _overview_lines(info)
    if not lines:
        raise NoMarketDataError(ticker, canonical, "no Yahoo overview fields returned")
    return lines


def _strip_vendor_header(payload: str) -> str:
    """Drop the per-vendor ``#`` header lines so the aggregate owns the framing."""
    return "\n".join(
        line for line in str(payload).splitlines() if not line.startswith("#")
    ).strip()


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Merged overview: SEC filing facts (PIT) + Yahoo valuation (live only)."""
    canonical = normalize_symbol(ticker)
    sec_block: str | None = None
    sec_error: str | None = None
    try:
        sec_block = _strip_vendor_header(_sec_fundamentals(ticker, curr_date))
    except Exception as exc:  # noqa: BLE001 — degrade to the other half
        sec_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "aggregate overview: SEC half unavailable for %s (%s)", canonical, exc,
        )

    yahoo_block: str | None = None
    yahoo_note: str | None = None
    if overview_would_leak_future(curr_date):
        # Same PIT contract as the plain yfinance vendor: a past decision
        # date must not see today's snapshot. The aggregate then serves the
        # SEC half alone (it is PIT by construction).
        yahoo_note = (
            f"real-time valuation not served as of {curr_date} "
            "(point-in-time guard: Yahoo snapshot is today-only)"
        )
    else:
        try:
            yahoo_block = "\n".join(
                _yahoo_valuation_lines(ticker, canonical)
            )
        except Exception as exc:  # noqa: BLE001 — degrade to the other half
            yahoo_note = f"real-time valuation unavailable ({type(exc).__name__}: {exc})"
            logger.warning(
                "aggregate overview: Yahoo half unavailable for %s (%s)",
                canonical, exc,
            )

    if sec_block is None and yahoo_block is None:
        raise NoMarketDataError(
            ticker, canonical,
            f"both overview halves failed (sec_edgar: {sec_error}; "
            f"yahoo: {yahoo_note})",
        )

    out: list[str] = [f"# Company Fundamentals for {canonical}"]
    out.append(
        f"# Aggregate overview as of {curr_date or 'now'} "
        f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): SEC EDGAR filing "
        "facts are the statement anchor; the Yahoo section adds REAL-TIME "
        "valuation (market cap, PE, beta) and is live-run-only by PIT policy."
    )
    out.append("")
    if sec_block is not None:
        out.append("## SEC filing facts (as-reported, point-in-time)")
        out.append(sec_block)
    else:
        out.append(
            f"## SEC filing facts: data not available ({sec_error}) — do not "
            "estimate filing figures."
        )
    out.append("")
    if yahoo_block is not None:
        out.append("## Real-time valuation & snapshot (Yahoo Finance, live)")
        out.append(yahoo_block)
    else:
        out.append(f"## Real-time valuation: {yahoo_note}.")
    return "\n".join(out)
