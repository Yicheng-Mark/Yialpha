"""SEC EDGAR ownership & short-interest vendor (Form 4 insider trading + FTD).

A new optional category (``sec_ownership``) of two US-only, read-only signals:

* :func:`get_form4_insider_trading` — Form 4 filings (insider buys/sells) for a
  CIK, pulled from the submissions JSON + each filing's primary XML.
* :func:`get_ftd_data` — fails-to-deliver balances from the SEC's semi-monthly
  CNS dissemination files, filtered by ticker.

Both reuse :mod:`yiagents.dataflows.sec_edgar`'s transport layer verbatim —
``_sec_get`` (UA + SOCKS5 + throttle + status mapping), ``_cached_or_fetch``
(disk cache + stale fallback), ``_cik_for_ticker`` (US-only CIK resolution).
``sec_edgar.py`` itself is **not modified**; the private helpers are imported
across modules within the same package.

The XBRL concept parsers in ``sec_edgar`` are *not* reused: Form 4 is XML and
FTD is pipe-delimited text, so each gets a small dedicated parser here.

Point-in-time
-------------
* **Form 4**: an insider transaction is visible at ``curr_date`` iff its
  ``filingDate <= curr_date`` (ground truth — the market learns the trade when
  the Form 4 is filed, ~2 days after the trade date). Empty ``curr_date`` means
  live mode (no as-of constraint, look-back from today).
* **FTD**: a semi-monthly file with cutoff date ``D`` is published several days
  after ``D``; it is treated as visible at ``curr_date`` iff
  ``D + ftd_pub_lag_days <= curr_date`` (conservative; configurable via
  ``YIAGENTS_FTD_PUB_LAG_DAYS``, default 10). Row-level ``Date <= curr_date`` is
  also enforced. Empty ``curr_date`` means live mode.

US-only
-------
* Form 4 needs a CIK, so a non-US ticker raises :class:`NoMarketDataError` (the
  router turns that into the ``NO_DATA_AVAILABLE`` sentinel; the analyst's
  grounding rule reports "data not available").
* FTD is CNS settlement data keyed by ticker/CUSIP, so a non-US ticker simply
  matches no rows and yields an informative "no fails reported" string (it does
  not resolve a CIK). This is a deliberate asymmetry, documented per-tool.

Free, keyless. Like ``sec_edgar``, SEC asks for a descriptive ``User-Agent``
(set ``YIAGENTS_SEC_USER_AGENT``).
"""

from __future__ import annotations

import io
import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import date, timedelta

from .config import get_config
from .errors import NoMarketDataError
# Reuse sec_edgar's transport verbatim (CIK resolution, GET, cache, throttle).
from .sec_edgar import (
    _cache_dir,
    _cached_or_fetch,
    _cik_for_ticker,
    _sec_get,
)

logger = logging.getLogger(__name__)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
_FTD_URL = "https://www.sec.gov/files/data/fails-deliver-data/cnbs{yyyymmdd}.txt"

# Cap Form 4 filings fetched per call (each is one throttled GET). 25 recent
# insider filings is far more than a 180-day window normally accumulates and
# keeps a single analysis under ~30 SEC requests.
_MAX_FORM4 = 25
# SEC publishes FTD data semi-monthly; the two cutoff dates per month.
_FTD_CUTOFF_DAYS = (15,)  # plus the last calendar day, handled separately


def _ftd_pub_lag_days() -> int:
    """Publication-lag (days) before a FTD cutoff file is treated as public."""
    raw = get_config().get("ftd_pub_lag_days", 10)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 10
    return n if n >= 0 else 0


def _fetch_submissions(cik: int) -> dict:
    """Fetch (cached 1 day) the submissions JSON for a CIK."""
    path = os.path.join(_cache_dir(), f"submissions_{cik}.json")
    raw = _cached_or_fetch(path, _SUBMISSIONS_URL.format(cik=cik), ttl_days=1.0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NoMarketDataError(str(cik), detail=f"could not parse submissions: {exc}") from exc


def _fetch_form4_xml(cik: int, accession: str, doc: str) -> bytes:
    """Fetch (cached 90 days; accession docs are immutable) a Form 4 XML."""
    acc_nodash = accession.replace("-", "")
    url = _ARCHIVES_URL.format(cik=cik, acc_nodash=acc_nodash, doc=doc)
    path = os.path.join(_cache_dir(), f"form4_{cik}_{acc_nodash}_{doc}")
    return _cached_or_fetch(path, url, ttl_days=90.0)


def _txt(el) -> str:
    """Return a trimmed element text, or '' for missing/blank."""
    if el is None or el.text is None:
        return ""
    return el.text.strip()


# --------------------------------------------------------------------------- #
# Form 4 parsing
# --------------------------------------------------------------------------- #
def _parse_form4(raw: bytes) -> dict:
    """Parse a Form 4 XML into a compact dict of the non-derivative trades.

    Returns ``{"owner": str, "title": str, "trades": [{"date","code","shares",
    "price","action","post_shares"}]}``. ``action`` is BUY/SELL/GIFT/OTHER from
    the SEC transactionCode (A/S/G/...). Derivative-table transactions (options,
    RSUs) are intentionally skipped — the clean insider signal is the
    non-derivative common-stock trade.
    """
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise NoMarketDataError("form4", detail=f"could not parse Form 4 XML: {exc}") from exc

    owner = _txt(root.find("reportingOwner/reportingOwnerId/rptOwnerName"))
    title = _txt(root.find("reportingOwner/reportingOwnerRelationship/officerTitle"))

    _CODE = {"A": "BUY", "S": "SELL", "G": "GIFT", "M": "OTHER", "P": "BUY",
             "D": "SELL", "F": "GIFT"}
    trades = []
    for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _txt(txn.find("transactionCoding/transactionCode"))
        trades.append({
            "date": _txt(txn.find("transactionDate/value")),
            "code": code,
            "action": _CODE.get(code, "OTHER"),
            "shares": _txt(txn.find("transactionAmounts/transactionShares/value")),
            "price": _txt(txn.find("transactionAmounts/transactionPricePerShare/value")),
            "post_shares": _txt(
                txn.find("postTransactionAmounts/sharesOwnedFollowingTransaction/value")),
        })
    return {"owner": owner, "title": title, "trades": trades}


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def get_form4_insider_trading(
    ticker: str, curr_date: str | None = None, look_back_days: int = 180
) -> str:
    """Recent SEC Form 4 insider transactions for a US ticker, PIT by filingDate.

    Resolves the CIK, enumerates ``form == "4"`` filings from the submissions
    JSON whose ``filingDate <= curr_date`` and within the look-back window, then
    fetches + parses each Form 4's non-derivative trades. Renders a per-trade
    table plus a window summary (buy/sell counts and net dollar flow).

    Non-US / no CIK -> :class:`NoMarketDataError`. No Form 4 in the window ->
    an informative string (honest "no recent insider activity", not an error).
    """
    cik = _cik_for_ticker(ticker)
    subs = _fetch_submissions(cik)

    # Live mode: no as-of upper bound; anchor the look-back window at today.
    upper = (curr_date or "")[:10]
    if upper:
        upper_d = date.fromisoformat(upper)
    else:
        upper_d = date.today()
    lower_d = upper_d - timedelta(days=int(look_back_days))

    recent = (subs.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    primary_docs = recent.get("primaryDocument") or []

    rows = []
    fetched = 0
    # Parallel arrays, most-recent-first in submissions; take the first N that
    # match the PIT window (filingDate within [lower_d, upper_d]).
    for i, form in enumerate(forms):
        if form != "4":
            continue
        fd = (filing_dates[i] if i < len(filing_dates) else "")[:10]
        try:
            fd_d = date.fromisoformat(fd) if fd else None
        except ValueError:
            fd_d = None
        # PIT: filingDate must be public by curr_date (ground truth).
        if upper and fd_d and fd_d > upper_d:
            continue
        # Lower bound only applies when we have a parseable filing date.
        if fd_d and fd_d < lower_d:
            continue
        if fetched >= _MAX_FORM4:
            break
        fetched += 1
        try:
            xml_bytes = _fetch_form4_xml(
                cik, accessions[i], primary_docs[i] if i < len(primary_docs) else ""
            )
            parsed = _parse_form4(xml_bytes)
        except NoMarketDataError as exc:
            logger.debug("sec_ownership: skipping Form 4 %s: %s", accessions[i], exc)
            continue
        for t in parsed["trades"]:
            rows.append({
                "date": t["date"], "owner": parsed["owner"], "title": parsed["title"],
                "action": t["action"], "shares": t["shares"], "price": t["price"],
                "post_shares": t["post_shares"],
            })

    out = io.StringIO()
    out.write(f"# Form 4 Insider Trading for {ticker} (last {look_back_days} days, "
              f"as of {curr_date or 'now'})\n")
    out.write(f"# Source: SEC EDGAR Form 4 (PIT: filingDate <= {curr_date or 'now'}; "
              f"CIK {cik})\n")

    if not rows:
        out.write(f"\nNo Form 4 transactions for {ticker} in the last "
                  f"{look_back_days} days (as of {curr_date or 'now'}).")
        return out.getvalue().rstrip("\n")

    rows.sort(key=lambda r: r["date"], reverse=True)
    out.write(f"# {len(rows)} non-derivative trade(s) across {fetched} filing(s)\n\n")
    out.write("Date       | Insider          | Title        | Action | Shares     "
              "| Price  | Post-Hold\n")
    out.write("-" * 88 + "\n")
    for r in rows:
        out.write(f"{r['date'] or 'n/a':<10} | {r['owner'][:16]:<16} | "
                  f"{r['title'][:11]:<11} | {r['action']:<6} | {r['shares']:>10} | "
                  f"{r['price']:>6} | {r['post_shares']}\n")

    # Window summary: net dollar flow by action.
    buy_usd = sell_usd = 0.0
    n_buy = n_sell = 0
    for r in rows:
        sh = _num(r["shares"])
        px = _num(r["price"])
        if sh is None or px is None:
            continue
        if r["action"] == "BUY":
            buy_usd += sh * px
            n_buy += 1
        elif r["action"] == "SELL":
            sell_usd += sh * px
            n_sell += 1
    net = buy_usd - sell_usd
    tone = "net buyer" if net > 0 else ("net seller" if net < 0 else "balanced")
    out.write(
        f"\nSummary (last {look_back_days} days): {n_buy} buy(s) ${buy_usd:,.0f} / "
        f"{n_sell} sell(s) ${sell_usd:,.0f} -> net {tone} (${net:+,.0f})."
    )
    return out.getvalue().rstrip("\n")


# --------------------------------------------------------------------------- #
# FTD (fails-to-deliver)
# --------------------------------------------------------------------------- #
def _last_day_of_month(y: int, m: int) -> int:
    if m == 12:
        nxt = date(y + 1, 1, 1)
    else:
        nxt = date(y, m + 1, 1)
    return (nxt - timedelta(days=1)).day


def _enumerate_ftd_cutoffs(start_d: date, end_d: date) -> list[str]:
    """Candidate FTD cutoff dates (mid-month 15th + last day) in [start, end],
    returned as ``YYYYMMDD`` strings most-recent-first. The caller tries each;
    a missing file (404) is normal and skipped."""
    out: list[str] = []
    y, m = start_d.year, start_d.month
    while (y, m) <= (end_d.year, end_d.month):
        for d in (_FTD_CUTOFF_DAYS[0], _last_day_of_month(y, m)):
            cand = date(y, m, d)
            if start_d <= cand <= end_d:
                out.append(f"{cand:%Y%m%d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    out.sort(reverse=True)
    return out


def _detect_delimiter(header: str) -> str:
    """Pick the delimiter that splits the header into the most fields."""
    best, n = "|", header.count("|")
    for cand in ("\t", ","):
        c = header.count(cand)
        if c > n:
            best, n = cand, c
    return best if n > 0 else None


def _ftd_column_map(header: str) -> dict[str, int]:
    """Header-driven column index map. Normalizes names so the parser does not
    depend on the SEC's exact column labels/order (Date/CUSIP/Issuer/Symbol/
    Total Fails/Price)."""
    delim = _detect_delimiter(header)
    parts = header.split(delim) if delim else header.split()
    m = {}
    for i, name in enumerate(parts):
        n = name.strip().lower()
        if "date" in n:
            m["date"] = i
        elif "cusip" in n:
            m["cusip"] = i
        elif "symbol" in n or "ticker" in n:
            m["symbol"] = i
        elif "issuer" in n or "company" in n:
            m["issuer"] = i
        elif "fail" in n:
            m["fails"] = i
        elif "price" in n:
            m["price"] = i
    return m


def _parse_ftd_text(raw: bytes, ticker: str) -> list[dict]:
    """Parse a cnbs FTD file, returning rows matching ``ticker`` (case-insensitive
    on the symbol column). Header-driven; tolerates ``|`` / tab / comma."""
    text = raw.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    cols = _ftd_column_map(lines[0])
    sym_i = cols.get("symbol")
    out = []
    for ln in lines[1:]:
        parts = ln.split(_detect_delimiter(ln) or "\t")
        if sym_i is not None and sym_i < len(parts):
            if parts[sym_i].strip().upper() != ticker.upper():
                continue
        else:
            continue
        out.append({
            "date": parts[cols["date"]].strip() if "date" in cols and cols["date"] < len(parts) else "",
            "fails": parts[cols["fails"]].strip() if "fails" in cols and cols["fails"] < len(parts) else "",
            "price": parts[cols["price"]].strip() if "price" in cols and cols["price"] < len(parts) else "",
        })
    return out


def get_ftd_data(
    ticker: str, curr_date: str | None = None, look_back_days: int = 180
) -> str:
    """Recent SEC fails-to-deliver balances for a ticker, PIT-aware.

    Enumerates the semi-monthly FTD cutoff files within the look-back window that
    are already public by ``curr_date`` (cutoff + ``YIAGENTS_FTD_PUB_LAG_DAYS``,
    default 10), fetches each (404 / missing file is normal and skipped), and
    filters rows by ticker + ``Date <= curr_date``. Renders a per-fail-day table
    plus a window summary (peak fail-day, total fail-days).

    FTD is CNS settlement data keyed by ticker; a non-US ticker simply matches
    no rows -> an informative "no fails reported" string (it does NOT resolve a
    CIK, unlike Form 4).
    """
    upper = (curr_date or "")[:10]
    upper_d = date.fromisoformat(upper) if upper else date.today()
    start_d = upper_d - timedelta(days=int(look_back_days))
    lag = _ftd_pub_lag_days()

    # Only files whose cutoff + publication lag has elapsed by curr_date are
    # PIT-visible. The cutoff itself must also be within the look-back window.
    visible_end = upper_d - timedelta(days=lag)
    cutoffs = _enumerate_ftd_cutoffs(max(start_d, date(2009, 9, 1)), visible_end)

    rows: list[dict] = []
    for yyyymmdd in cutoffs:
        url = _FTD_URL.format(yyyymmdd=yyyymmdd)
        path = os.path.join(_cache_dir(), f"ftd_{yyyymmdd}.txt")
        try:
            raw = _cached_or_fetch(path, url, ttl_days=30.0)
        except NoMarketDataError:
            # No file for this cutoff (SEC only publishes periods with fails;
            # and the exact cutoff calendar day can shift) -> normal, skip.
            continue
        for r in _parse_ftd_text(raw, ticker):
            # Row-level PIT: settlement date must be <= curr_date.
            d = r["date"]
            try:
                d_d = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
            except (ValueError, IndexError):
                continue
            if d_d > upper_d:
                continue
            rows.append(r)

    out = io.StringIO()
    out.write(f"# Fails-to-Deliver for {ticker} (last {look_back_days} days, "
              f"as of {curr_date or 'now'})\n")
    out.write(f"# Source: SEC CNS FTD data (PIT: file cutoff + {lag}d pub lag <= "
              f"{curr_date or 'now'})\n")

    if not rows:
        out.write(f"\nNo fails-to-deliver reported for {ticker} in the last "
                  f"{look_back_days} days (as of {curr_date or 'now'}).")
        return out.getvalue().rstrip("\n")

    rows.sort(key=lambda r: r["date"], reverse=True)
    out.write(f"# {len(rows)} fail-day(s)\n\n")
    out.write("Date       | Total Fails   | Price\n")
    out.write("-" * 40 + "\n")
    for r in rows:
        out.write(f"{r['date']:<10} | {r['fails']:>12} | {r['price']}\n")

    # Peak fail-day + count.
    peak = max(rows, key=lambda r: _num(r["fails"]) or 0)
    peak_v = _num(peak["fails"])
    out.write(
        f"\nPeak fail-day: {peak['date']} ({peak_v:,.0f} fails @ ${peak['price']}). "
        f"Total fail-days in window: {len(rows)}."
    )
    return out.getvalue().rstrip("\n")
