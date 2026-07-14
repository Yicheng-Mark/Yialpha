"""SEC EDGAR (XBRL companyfacts) fundamentals vendor.

A third vendor for the four existing fundamentals methods
(``get_fundamentals`` / ``get_balance_sheet`` / ``get_cashflow`` /
``get_income_statement``), slotted into the router in :mod:`interface` next to
``yfinance`` and ``alpha_vantage``. Selected via
``data_vendors["fundamental_data"] = "sec_edgar"`` (or a chain like
``"sec_edgar,yfinance"``); the default is unchanged, so every run that does not
opt in is byte-equivalent.

Why SEC XBRL over yfinance ``.info``: the EDGAR ``companyfacts`` payload is
**as-reported**, ~500 GAAP tags, and — critically — each fact carries the
actual ``filed`` date. That makes point-in-time correctness ground-truth rather
than the 45-day heuristic the other vendors are stuck with (a 10-Q filed 38
days after period end is genuinely public at day 38; yfinance/AV can't know
that). It also has no date-less "today snapshot", so unlike ``.info`` it is NOT
gated out of backtests by ``overview_would_leak_future``.

PIT rule here: a period is visible at ``curr_date`` iff its ``filed`` date is
``<= curr_date`` (ground truth), falling back to
:func:`yiagents.dataflows.utils.is_filing_public` only for the rare fact with no
``filed``. ``curr_date`` empty/None means live mode (no as-of constraint).

US-only: a ticker with no CIK mapping raises :class:`NoMarketDataError`, which
the router turns into either a fall-through to the next vendor in the chain or
the ``NO_DATA_AVAILABLE`` sentinel — so non-US tickers degrade cleanly.

Free, keyless. SEC asks for a descriptive ``User-Agent``; set
``YIAGENTS_SEC_USER_AGENT`` (e.g. ``"YiAgents research you@example.com"``) to
identify yourself, otherwise a generic identifier is sent.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from typing import Any

from .config import get_config
from .errors import NoMarketDataError, VendorRateLimitError
from .utils import is_filing_public

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_TIMEOUT = 30
# SEC fair-access guidance is ~10 requests/sec; stay comfortably under it with a
# small serial throttle (+ jitter) so concurrent analyst calls don't trip a 429.
_MIN_INTERVAL = 0.12
_last_request = [0.0]
_throttle_lock = threading.Lock()


def _user_agent() -> str:
    return os.environ.get("YIAGENTS_SEC_USER_AGENT") or "YiAgents research (sec-edgar vendor)"


def _proxies() -> dict[str, str | None]:
    """SOCKS5/HTTP proxy map, mirroring dataflows/binance.py.

    SEC hosts are US-based; reads HTTP_PROXY/HTTPS_PROXY/ALL_PROXY at call time
    so the project's socks5h:// settings (set in .env) are honoured.
    """
    return {
        "http": os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY"),
        "https": os.environ.get("HTTPS_PROXY") or os.environ.get("ALL_PROXY"),
    }


def _cache_dir() -> str:
    cfg = get_config()
    base = cfg.get("data_cache_dir") or os.path.join(os.path.expanduser("~"), ".yiagents", "cache")
    path = os.path.join(base, "sec")
    os.makedirs(path, exist_ok=True)
    return path


def _throttle() -> None:
    """Enforce a minimum spacing between SEC requests (thread-safe)."""
    with _throttle_lock:
        elapsed = time.time() - _last_request[0]
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request[0] = time.time()


def _sec_get(url: str) -> bytes:
    """GET a SEC endpoint with UA + proxy + throttle; map status codes to vendor errors."""
    _throttle()
    import requests

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _user_agent()},
            proxies=_proxies(),
            timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 -- network failure -> NoMarketDataError
        raise NoMarketDataError(url, detail=f"SEC request failed: {exc!r}") from exc
    if resp.status_code == 429 or resp.status_code == 403:
        # 403 from SEC is almost always a missing/abusive User-Agent or a throttle.
        raise VendorRateLimitError(f"SEC returned {resp.status_code} for {url}")
    if resp.status_code == 404:
        raise NoMarketDataError(url, detail="SEC returned 404 (no such filing/entity)")
    if resp.status_code >= 400:
        raise NoMarketDataError(url, detail=f"SEC HTTP {resp.status_code}")
    return resp.content


def _cached_or_fetch(path: str, url: str, ttl_days: float) -> bytes:
    """Serve from a fresh on-disk cache, else fetch + cache. Falls back to a stale
    cache on network failure (a slightly-old filing beats no data)."""
    if os.path.exists(path):
        if (time.time() - os.path.getmtime(path)) < ttl_days * 86_400.0:
            try:
                with open(path, "rb") as fh:
                    return fh.read()
            except OSError:
                pass
    raw = _sec_get(url)
    try:
        with open(path, "wb") as fh:
            fh.write(raw)
    except OSError as exc:  # noqa: BLE001 -- caching is best-effort
        logger.warning("sec_edgar: could not write cache %s: %s", path, exc)
    return raw


# --------------------------------------------------------------------------- #
# CIK resolution
# --------------------------------------------------------------------------- #
def _cik_for_ticker(ticker: str) -> int:
    """Resolve a US ticker to its SEC CIK. Raises NoMarketDataError if unknown."""
    path = os.path.join(_cache_dir(), "tickers.json")
    raw = _cached_or_fetch(path, _TICKERS_URL, ttl_days=7.0)
    try:
        table = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NoMarketDataError(ticker, detail=f"could not parse SEC ticker table: {exc}") from exc
    # company_tickers.json is {"0": {"cik_str":.., "ticker":.., "title":..}, ...}
    for row in table.values():
        if str(row.get("ticker", "")).upper() == ticker.upper():
            return int(row["cik_str"])
    raise NoMarketDataError(
        ticker, detail="SEC EDGAR covers US-listed issuers only (no CIK for this symbol)"
    )


def _fetch_company_facts(cik: int) -> dict[str, Any]:
    """Fetch (cached) the companyfacts JSON for a CIK."""
    path = os.path.join(_cache_dir(), f"companyfacts_{cik}.json")
    raw = _cached_or_fetch(path, _FACTS_URL.format(cik=cik), ttl_days=1.0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NoMarketDataError(str(cik), detail=f"could not parse companyfacts: {exc}") from exc


# --------------------------------------------------------------------------- #
# XBRL concept maps (display name -> ordered candidate GAAP tags -> unit kind)
# --------------------------------------------------------------------------- #
# unit_kind: "money" (USD), "shares", "per_share" (USD/shares)
_INCOME_CONCEPTS: list[tuple[str, list[str], str]] = [
    ("Revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"], "money"),
    ("Cost of Revenue", ["CostOfRevenue", "CostOfGoodsAndServicesSold"], "money"),
    ("Gross Profit", ["GrossProfit"], "money"),
    ("Operating Expenses", ["OperatingExpenses"], "money"),
    ("Operating Income", ["OperatingIncomeLoss"], "money"),
    ("Net Income", ["NetIncomeLoss"], "money"),
    ("R&D Expense", ["ResearchAndDevelopmentExpense"], "money"),
    ("SG&A Expense", ["SellingGeneralAndAdministrativeExpense"], "money"),
    ("Interest Expense", ["InterestExpense"], "money"),
    ("Income Tax", ["IncomeTaxExpenseBenefit"], "money"),
    ("EPS Basic", ["EarningsPerShareBasic"], "per_share"),
    ("EPS Diluted", ["EarningsPerShareDiluted"], "per_share"),
    ("Shares Basic (wtd avg)", ["WeightedAverageNumberOfSharesOutstandingBasic"], "shares"),
    ("Shares Diluted (wtd avg)", ["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares"),
]

_BALANCE_CONCEPTS: list[tuple[str, list[str], str]] = [
    ("Total Assets", ["Assets"], "money"),
    ("Current Assets", ["AssetsCurrent"], "money"),
    ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue"], "money"),
    ("Inventory", ["InventoryNet"], "money"),
    ("Goodwill", ["Goodwill"], "money"),
    ("Total Liabilities", ["Liabilities"], "money"),
    ("Current Liabilities", ["LiabilitiesCurrent"], "money"),
    ("Long-Term Debt", ["LongTermDebt", "LongTermDebtNoncurrent"], "money"),
    ("Total Equity", ["StockholdersEquity"], "money"),
    ("Retained Earnings", ["RetainedEarningsAccumulatedDeficit"], "money"),
]

_CASHFLOW_CONCEPTS: list[tuple[str, list[str], str]] = [
    ("Operating CF", ["NetCashProvidedByUsedInOperatingActivities"], "money"),
    ("Investing CF", ["NetCashProvidedByUsedInInvestingActivities"], "money"),
    ("Financing CF", ["NetCashProvidedByUsedInFinancingActivities"], "money"),
    ("CapEx", ["PaymentsToAcquirePropertyPlantAndEquipment"], "money"),
    ("Dividends Paid", ["PaymentsForDividends"], "money"),
    ("Share Repurchases", ["PaymentsForRepurchaseOfCommonStock", "CommonStockRepurchased"], "money"),
]

_CONCEPT_MAPS = {
    "income": _INCOME_CONCEPTS,
    "balance": _BALANCE_CONCEPTS,
    "cashflow": _CASHFLOW_CONCEPTS,
}


def _concept_records(facts: dict[str, Any], candidates: list[str]) -> list[dict[str, Any]]:
    """Return the period records for the first matching concept, else [].

    Flattens ``facts.us-gaap.<concept>.units.*`` into a list of
    ``{start, end, val, fy, fp, form, filed}`` dicts.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for name in candidates:
        concept = gaap.get(name)
        if not concept:
            continue
        records: list[dict[str, Any]] = []
        for _unit, rows in concept.get("units", {}).items():
            for r in rows:
                if "val" not in r or r.get("val") is None:
                    continue
                records.append(r)
        if records:
            return records
    return []


def _period_public(rec: dict[str, Any], curr_date: str) -> bool:
    """Ground-truth PIT gate: ``filed <= curr_date`` when filed is present, else
    fall back to the project's 45-day ``is_filing_public`` heuristic."""
    if not curr_date:
        return True
    filed = rec.get("filed")
    if filed:
        try:
            return str(filed)[:10] <= str(curr_date)[:10]
        except (ValueError, TypeError):
            pass
    return is_filing_public(rec.get("end"), curr_date)


def _freq_match(fp: str, freq: str) -> bool:
    """ annual -> FY only; quarterly -> Q1..Q4 (the standalone 10-Q periods). """
    fp = (fp or "").upper()
    if freq == "annual":
        return fp == "FY"
    return fp in {"Q1", "Q2", "Q3", "Q4"}


def _fmt(val: Any, unit_kind: str) -> str:
    try:
        f = float(val)
    except (ValueError, TypeError):
        return ""
    if unit_kind == "per_share":
        return f"{f:.4f}"
    if unit_kind == "shares":
        return f"{f:.0f}"
    return f"{f:.0f}"


def _render_statement(
    title: str,
    ticker: str,
    freq: str,
    concept_map: list[tuple[str, list[str], str]],
    facts: dict[str, Any],
    curr_date: str | None,
) -> str:
    """Build a transposed-CSV statement string from companyfacts, PIT-filtered."""
    # Resolve the column set: union of period-end dates across all line items,
    # PIT + freq filtered, most-recent-first, capped to keep the report readable.
    all_ends: set[str] = set()
    per_item: list[tuple[str, str, list[dict[str, Any]]]] = []
    for display, candidates, unit_kind in concept_map:
        recs = _concept_records(facts, candidates)
        recs = [r for r in recs if _freq_match(r.get("fp", ""), freq) and _period_public(r, curr_date)]
        per_item.append((display, unit_kind, recs))
        for r in recs:
            end = r.get("end")
            if end:
                all_ends.add(end)

    if not all_ends:
        raise NoMarketDataError(
            ticker, detail=f"no SEC {freq} filings public as of {curr_date or 'now'}"
        )

    columns = sorted(all_ends, reverse=True)[:8]

    # Most recent filed date for the source line.
    latest_filed = ""
    for _d, _u, recs in per_item:
        for r in recs:
            f = r.get("filed")
            if f and (not latest_filed or f > latest_filed):
                latest_filed = f

    out = io.StringIO()
    out.write(f"# {title} for {ticker} ({freq})\n")
    out.write(f"# Source: SEC EDGAR XBRL (companyfacts"
              + (f", latest filing {latest_filed}" if latest_filed else "") + ")\n\n")
    out.write("," + ",".join(columns) + "\n")
    for display, unit_kind, recs in per_item:
        # Index period-end -> value (last write wins; durations share an end).
        by_end: dict[str, str] = {}
        for r in recs:
            end = r.get("end")
            if end:
                by_end[end] = _fmt(r.get("val"), unit_kind)
        cells = [by_end.get(c, "") for c in columns]
        out.write(display + "," + ",".join(cells) + "\n")
    return out.getvalue().rstrip("\n")


# --------------------------------------------------------------------------- #
# Public vendor API (signatures match the existing fundamentals contract).
# --------------------------------------------------------------------------- #
def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Key-value fundamentals snapshot from the latest PIT-public SEC filing.

    Unlike yfinance ``.info`` (a date-less today snapshot, gated out of
    backtests), this is built from as-reported XBRL facts whose ``filed`` date
    is ``<= curr_date`` — genuinely point-in-time.
    """
    cik = _cik_for_ticker(ticker)
    facts = _fetch_company_facts(cik)
    entity = facts.get("entityName", ticker)

    def _latest(candidates: list[str]) -> tuple[Any, str]:
        recs = _concept_records(facts, candidates)
        recs = [r for r in recs if _period_public(r, curr_date)]
        recs.sort(key=lambda r: r.get("end", ""), reverse=True)
        return (recs[0]["val"] if recs else None), (recs[0].get("end", "") if recs else "")

    out = io.StringIO()
    out.write(f"# Company Fundamentals for {ticker}\n")
    out.write(f"# Source: SEC EDGAR XBRL (as-reported, PIT as of {curr_date or 'now'})\n\n")
    out.write(f"Name: {entity}\n")
    out.write(f"CIK: {cik}\n")

    snapshots = [
        ("Total Assets", ["Assets"]),
        ("Total Equity", ["StockholdersEquity"]),
        ("Total Liabilities", ["Liabilities"]),
        ("Revenue (latest)", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]),
        ("Net Income (latest)", ["NetIncomeLoss"]),
    ]
    for label, cands in snapshots:
        val, end = _latest(cands)
        if val is None:
            out.write(f"{label}: n/a\n")
        else:
            out.write(f"{label}: {float(val):,.0f}  (period ending {end})\n")

    eps, eps_end = _latest(["EarningsPerShareDiluted"])
    out.write(f"EPS Diluted (latest): {f'{float(eps):.4f}' if eps is not None else 'n/a'}"
              + (f"  (period ending {eps_end})\n" if eps is not None else "\n"))
    return out.getvalue().rstrip("\n")


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    cik = _cik_for_ticker(ticker)
    facts = _fetch_company_facts(cik)
    return _render_statement("Balance Sheet", ticker, freq, _BALANCE_CONCEPTS, facts, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    cik = _cik_for_ticker(ticker)
    facts = _fetch_company_facts(cik)
    return _render_statement("Cash Flow Statement", ticker, freq, _CASHFLOW_CONCEPTS, facts, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    cik = _cik_for_ticker(ticker)
    facts = _fetch_company_facts(cik)
    return _render_statement("Income Statement", ticker, freq, _INCOME_CONCEPTS, facts, curr_date)
