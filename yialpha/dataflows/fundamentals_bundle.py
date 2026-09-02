"""Deterministic fundamentals bundle — the core facts are FETCHED, not chatted.

The fundamentals analyst's tool belt is entirely model-chosen: the LLM
decides whether to call ANY statement tool, so a run could reach its
decision without ever reading an income statement — and nothing in the
quality chain would notice, because nothing failed that was never
attempted. This module is the deterministic complement (same contract as
:mod:`yialpha.dataflows.perp_bundle`): ONE parallel prefetch assembles the
core fundamentals evidence — the merged SEC+Yahoo overview and the three
quarterly statements (for ETFs the fund snapshot REPLACES the statements) —
before the analyst's LLM turn, fetched through ``route_to_vendor`` inside
``submit_with_context`` workers so every success/sentinel lands in the
run's quality ledger exactly as if the model had called the tools itself
(a plain ``pool.submit`` would bind fresh ContextVars inside each worker
thread and the events would vanish there). Fail-soft per component with
status disclosed in the rendered block; the tools remain bound for
drill-down.

Instrument routing: a Binance tokenized-stock perp (MUUSDT) reads the
UNDERLYING equity (MU) via the same remap the tool layer applies; a
high-confidence ETF (SPY, incl. SPYUSDT) additionally prefetches the fund
snapshot (NAV/AUM/expense ratio/holdings — live runs only, the fund page
is a today snapshot); an A-share ticker with ``a_share_native`` on also
pulls the native PIT quarterly statements. Pure-crypto instruments never
reach this module (the analyst node skips them first).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import get_config, submit_with_context
from .interface import route_to_vendor
from .utils import is_historical_date

logger = logging.getLogger(__name__)

#: Component statuses (disclosed verbatim in the rendered footer).
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_SKIPPED_LIVE_ONLY = "skipped_live_only"
#: Whole-bundle gate: the replay date precedes the contract's onboard date
#: (onboard-date evidence present), so live fundamentals would be a PIT
#: violation — nothing is fetched at all.
STATUS_SKIPPED_NOT_LISTED = "skipped_not_listed"

#: Prompt-size cap per statement payload; the bound tools serve the full CSV.
_MAX_PAYLOAD_CHARS = 4000

#: Native A-share quarterly statements (optional category; only fetched when
#: the a_share_native flag is on AND the ticker is a China A-share).
_A_SHARE_JOBS = (
    ("a_share_income", "get_a_share_income_statement_native"),
    ("a_share_balance", "get_a_share_balance_sheet_native"),
    ("a_share_cashflow", "get_a_share_cashflow_statement_native"),
)

_SENTINEL_PREFIXES = ("NO_DATA_AVAILABLE", "DATA_UNAVAILABLE")


def _fetch_via_router(method: str, *args: Any) -> dict[str, Any]:
    """One router call wrapped fail-soft: a sentinel string is an honest
    unavailable, an exception is an unavailable, anything else is ok."""
    try:
        payload = route_to_vendor(method, *args)
    except Exception as exc:  # noqa: BLE001 — per-component isolation
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if isinstance(payload, str) and payload.startswith(_SENTINEL_PREFIXES):
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": payload.splitlines()[0][:300],
        }
    return {"status": STATUS_OK, "payload": str(payload)}


def _fetch_etf_fund_data(ticker: str, curr_date: str) -> dict[str, Any]:
    """Fund snapshot (NAV/AUM/expense/holdings) — live runs only (PIT)."""
    if is_historical_date(curr_date):
        return {
            "status": STATUS_SKIPPED_LIVE_ONLY,
            "reason": "fund snapshot is a today-only page (no as-of boundary)",
        }
    from .etf_fund_data import get_etf_fund_data

    try:
        payload = get_etf_fund_data(ticker, curr_date)
    except Exception as exc:  # noqa: BLE001 — enrichment, never a veto
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if isinstance(payload, str) and payload.startswith(_SENTINEL_PREFIXES):
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": payload.splitlines()[0][:300],
        }
    return {"status": STATUS_OK, "payload": payload}


def fetch_fundamentals_bundle(
    asset_type: str | None, ticker: str, curr_date: str,
) -> dict[str, Any]:
    """Assemble the fundamentals bundle in one parallel pass.

    Every component is fail-soft with per-component status in the result;
    core statement/overview calls run through ``route_to_vendor`` and the
    submissions ride ``submit_with_context``, so the run's quality ledger
    records the same evidence a model-issued tool call would — including
    the case where EVERY core component fails (the router's own core
    sentinels then classify the run instead of a clean-but-blind read).
    """
    from yialpha.graph.routing import describe_instrument

    from .symbol_utils import is_a_stock

    # as_of wired: the descriptor can then answer "was this contract live on
    # curr_date?" from onboard-date evidence instead of always "unknown".
    descriptor = describe_instrument(asset_type, ticker, as_of=curr_date)
    symbol = descriptor.fundamentals_symbol

    bundle: dict[str, Any] = {
        "symbol": ticker,
        "fundamentals_symbol": symbol,
        "as_of": curr_date,
        "instrument_class": descriptor.instrument_class,
        "is_etf": descriptor.is_etf,
        "listing_disclosure": descriptor.as_disclosure(),
    }

    # As-of listing gate: a replay dated BEFORE the contract's onboard date
    # must not read the underlying's live fundamentals (a PIT violation
    # dressed as evidence). listed_asof is False only WITH onboard-date
    # evidence; None (unknown — static seed, plain equity) keeps the normal
    # path and the disclosure rides the rendered block.
    if descriptor.listed_asof is False:
        reason = descriptor.as_disclosure()
        for key in (
            "overview", "income_statement", "balance_sheet", "cashflow",
            "etf_fund_data",
        ):
            bundle[key] = {"status": STATUS_SKIPPED_NOT_LISTED, "reason": reason}
        return bundle

    if descriptor.is_etf:
        # ETF REPLACE routing: a fund has no operating-company statements —
        # the fund snapshot (NAV/AUM/expense/holdings) IS the core evidence.
        # Fetching statements anyway would let their expected failures
        # wrongly grade a well-covered ETF run as degraded.
        jobs: list[tuple[str, Any]] = [
            (
                "overview",
                lambda: _fetch_via_router("get_fundamentals", symbol, curr_date),
            ),
            (
                "etf_fund_data",
                lambda: _fetch_etf_fund_data(symbol, curr_date),
            ),
        ]
    else:
        # overview carries (symbol, curr_date); statements
        # (symbol, freq, curr_date).
        jobs = [
            (
                "overview",
                lambda: _fetch_via_router("get_fundamentals", symbol, curr_date),
            ),
            (
                "income_statement",
                lambda: _fetch_via_router(
                    "get_income_statement", symbol, "quarterly", curr_date,
                ),
            ),
            (
                "balance_sheet",
                lambda: _fetch_via_router(
                    "get_balance_sheet", symbol, "quarterly", curr_date,
                ),
            ),
            (
                "cashflow",
                lambda: _fetch_via_router(
                    "get_cashflow", symbol, "quarterly", curr_date,
                ),
            ),
        ]
    if get_config().get("a_share_native") and is_a_stock(symbol):
        jobs += [
            (
                key,
                (lambda m=method: _fetch_via_router(m, symbol, curr_date)),
            )
            for key, method in _A_SHARE_JOBS
        ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        # submit_with_context carries the run's ContextVars (quality ledger,
        # config, PIT analysis date) into each worker: a plain submit would
        # let record_success/record_sentinel bind FRESH worker-local event
        # lists, and the bundle's evidence would silently miss the ledger.
        futures = {key: submit_with_context(pool, fn) for key, fn in jobs}
        for key, future in futures.items():
            try:
                bundle[key] = future.result(timeout=90)
            except Exception as exc:  # noqa: BLE001 — component isolation
                bundle[key] = {
                    "status": STATUS_UNAVAILABLE,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
    return bundle


def _section(title: str, comp: dict[str, Any]) -> list[str]:
    """One rendered section; payloads cap at :data:`_MAX_PAYLOAD_CHARS`."""
    if comp.get("status") != STATUS_OK:
        return []
    payload = str(comp.get("payload", "")).strip()
    if not payload:
        return []
    if len(payload) > _MAX_PAYLOAD_CHARS:
        payload = (
            payload[:_MAX_PAYLOAD_CHARS]
            + "\n…(truncated — call the bound tool for the full statement)"
        )
    return [f"#### {title}", "", payload, ""]


def render_fundamentals_bundle_block(bundle: dict[str, Any]) -> str:
    """Render the bundle as an advisory markdown block.

    Deterministic vendor output only — no model prose. Every non-ok
    component is disclosed in the footer, so a fundamentals-blind run reads
    as one instead of as a calm company."""
    symbol = bundle.get("symbol", "?")
    target = bundle.get("fundamentals_symbol", symbol)
    as_of = bundle.get("as_of", "?")
    lines: list[str] = [
        f"### Fundamentals Bundle — {symbol} (deterministic prefetch, as of {as_of})",
    ]
    if target != symbol:
        lines.append(
            f"- **Underlying equity**: {target} "
            f"({symbol} is a tokenized-stock perpetual; statements price the "
            "underlying US listing)"
        )
    disclosure = str(bundle.get("listing_disclosure") or "")
    if disclosure:
        lines.append(f"- **Listing**: {disclosure}")
    if bundle.get("is_etf"):
        # ETF REPLACE routing: the fund snapshot replaces the company
        # statements — the sections are not rendered (not "missing"), and the
        # footer says why so a thin fund snapshot never reads as a broken
        # company-fundamentals fetch.
        lines += _section(
            "Overview (SEC filing facts + live valuation)",
            bundle.get("overview") or {},
        )
        lines += _section(
            "ETF fund data (NAV/AUM/expense/holdings)",
            bundle.get("etf_fund_data") or {},
        )
        lines.append(
            "- **Company statements**: not applicable (ETF — the fund "
            "snapshot above replaces income/balance/cashflow)"
        )
    else:
        lines += _section("Overview (SEC filing facts + live valuation)", bundle.get("overview") or {})
        lines += _section("Income statement (quarterly)", bundle.get("income_statement") or {})
        lines += _section("Balance sheet (quarterly)", bundle.get("balance_sheet") or {})
        lines += _section("Cash flow statement (quarterly)", bundle.get("cashflow") or {})
    lines += _section("A-share income statement 利润表 (native)", bundle.get("a_share_income") or {})
    lines += _section("A-share balance sheet 资产负债表 (native)", bundle.get("a_share_balance") or {})
    lines += _section("A-share cash flow 现金流表 (native)", bundle.get("a_share_cashflow") or {})

    notes: list[str] = []
    for key in (
        "overview", "income_statement", "balance_sheet", "cashflow",
        "etf_fund_data", "a_share_income", "a_share_balance", "a_share_cashflow",
    ):
        comp = bundle.get(key)
        if isinstance(comp, dict) and comp.get("status") not in (None, STATUS_OK):
            status = comp.get("status", STATUS_UNAVAILABLE)
            reason = comp.get("reason", "")
            notes.append(f"{key}: {status}" + (f" ({reason})" if reason else ""))
    if notes:
        lines.append("- **Component availability**: " + "; ".join(notes))
    lines.append(
        "- (Earnings/statements above are pre-fetched evidence — cite reporting "
        "periods and filing dates from them exactly as you would tool output; "
        "the bound tools remain available for drill-down.)"
    )
    return "\n".join(lines)
