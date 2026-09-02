"""ETF fund data — NAV / AUM / expense ratio / holdings, real data not prompt framing.

PR5 gave high-confidence ETFs a fund-framing prompt nudge (NAV
premium/discount, AUM, expense ratio, leverage decay) but the TOOLS were
still company tools — the analyst was told to reason about NAV with no NAV
in evidence, so every fund field landed as "data not available". This module
is the missing data branch: a Yahoo-sourced fund snapshot (``.info`` fund
fields + best-effort ``funds_data`` top holdings) with the same PIT
discipline as the overview: the snapshot is today-only, so a historical
replay date raises ``NoMarketDataError`` (the deterministic bundle then
records the component as ``skipped_live_only`` instead of fetching).

Reached from the deterministic fundamentals bundle for ETF instruments
(plain ETF runs and ETF-underlying perps, e.g. SPYUSDT); failures degrade
per-component and are disclosed in the bundle footer — a missing fund field
never masquerades as a calm fund.
"""
from __future__ import annotations

import logging
from datetime import datetime

from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol
from .y_finance import _cached_ticker_info, overview_would_leak_future

logger = logging.getLogger(__name__)

#: Labeled ``.info`` fund fields. Values render as-is (Yahoo's own units:
#: expense ratio and yield as decimals, AUM/NAV in USD).
_FUND_FIELDS = [
    ("Fund Name", "longName"),
    ("Category", "category"),
    ("NAV (per share)", "navPrice"),
    ("Previous NAV", "previousNav"),
    ("Total Assets (AUM)", "totalAssets"),
    ("Annual Expense Ratio", "annualReportExpenseRatio"),
    ("Yield", "yield"),
    ("Beta (3y monthly)", "beta"),
    ("50 Day Average", "fiftyDayAverage"),
    ("200 Day Average", "twoHundredDayAverage"),
    ("52 Week High", "fiftyTwoWeekHigh"),
    ("52 Week Low", "fiftyTwoWeekLow"),
]


def _fmt_usd(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.4%}"
    except (TypeError, ValueError):
        return str(value)


def _top_holdings(ticker: str) -> str | None:
    """Top-10 holdings CSV block from yfinance ``funds_data`` (best-effort).

    The Yahoo fund page is scraped, not API'd — any failure returns None and
    the caller discloses "holdings unavailable" rather than blocking the
    fund snapshot fields that DID come back.
    """
    try:
        import yfinance as yf

        holdings = yf.Ticker(ticker).funds_data.top_holdings
        if holdings is None or holdings.empty:
            return None
        # The frame carries an extra symbol/name column pair depending on
        # version; normalize to Symbol/Holding Name/Weight and cap at 10.
        frame = holdings.reset_index().head(10)
        cols = [str(c) for c in frame.columns]
        symbol_col = next((c for c in cols if "symbol" in c.lower()), None)
        name_col = next(
            (c for c in cols if "holding" in c.lower() and "name" in c.lower()),
            cols[-1] if cols else None,
        )
        weight_col = next(
            (c for c in cols if "%" in c or "weight" in c.lower()), None,
        )
        lines = ["symbol,holding_name,weight_pct"]
        for _, row in frame.iterrows():
            symbol = row[symbol_col] if symbol_col else ""
            name = row[name_col] if name_col else ""
            weight = row[weight_col] if weight_col else ""
            lines.append(f"{symbol},{name},{weight}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — scrape is best-effort by contract
        logger.info("ETF top holdings unavailable for %s: %s", ticker, exc)
        return None


def get_etf_fund_data(ticker: str, curr_date: str | None = None) -> str:
    """Fund snapshot for an ETF ticker: NAV/AUM/expense/holdings (live only)."""
    canonical = normalize_symbol(ticker)
    if overview_would_leak_future(curr_date):
        raise NoMarketDataError(
            ticker, canonical,
            f"fund snapshot is point-in-time (today only); not valid as of {curr_date}",
        )
    info = _cached_ticker_info(ticker, canonical)
    if not info:
        raise NoMarketDataError(ticker, canonical, "no fund info returned")

    lines: list[str] = []
    for label, key in _FUND_FIELDS:
        value = info.get(key)
        if value is None:
            continue
        if key in ("totalAssets",):
            value = _fmt_usd(value)
        elif key in ("annualReportExpenseRatio", "yield"):
            value = _fmt_pct(value)
        lines.append(f"{label}: {value}")

    # Market price vs NAV premium/discount when both sides are present —
    # the single most decision-relevant fund number and it needs BOTH fields.
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    nav = info.get("navPrice")
    if price is not None and nav is not None and float(nav) > 0:
        premium_bps = (float(price) / float(nav) - 1.0) * 1e4
        lines.append(f"Market Price: {price}")
        lines.append(f"Premium/Discount vs NAV: {premium_bps:+.1f} bps")

    if not lines:
        raise NoMarketDataError(
            ticker, canonical, "no fund fields returned (not an ETF Yahoo covers?)",
        )

    out = [
        f"# ETF Fund Data for {canonical}",
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        " (live snapshot; NAV/AUM/expense are today's values)",
        "",
        "\n".join(lines),
    ]
    holdings = _top_holdings(ticker)
    if holdings:
        out += ["", "## Top holdings (Yahoo fund page, best-effort)", holdings]
    else:
        out += ["", "## Top holdings: unavailable (Yahoo fund page did not serve them)"]
    return "\n".join(out)
