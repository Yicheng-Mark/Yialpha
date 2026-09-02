"""Instrument routing — the ONE classification every entrance shares.

The question "does this instrument have company fundamentals to read?" was
answered in three different places (CLI ``filter_analysts_for_asset_type``,
the fundamentals prompt nudge, the batch runner's union helper), and every
non-CLI entrance (direct ``YiAlphaGraph()`` construction, scripts, the web
subprocess, backtest) missed the filter entirely — a pure-crypto run through
the direct API still executed the Fundamentals analyst. This module is the
single predicate those callers and the fundamentals NODE itself (runtime
skip backstop) all consult, so the invariant "pure-crypto instruments make
zero fundamentals vendor calls" holds on every entrance, including the
batch union over a mixed ``[BTCUSDT, MUUSDT]`` batch.

The PR-next slice grows the predicate into a real
:class:`InstrumentDescriptor`: routing answers PLUS the contract metadata
the fundamentals/market layers need (vendor symbol map for the underlying,
session calendar, perp onboard date with as-of listing classification).
Everything is computed from the warmed exchangeInfo snapshot (or the static
seed) — describe_instrument NEVER fetches, so every entrance stays
network-free and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from yialpha.dataflows.binance import (
    equity_perp_listing_info,
    stock_perp_underlying,
)

#: The crypto asset-type family (CLI ``AssetType`` values as plain strings —
#: the graph layer threads asset_type as str, not the CLI enum).
CRYPTO_FAMILY = frozenset({"crypto", "crypto_spot", "crypto_perp"})

#: Session calendars. Pure-crypto contracts trade continuously; a
#: tokenized-stock perp follows Binance's PUBLISHED TradFi sessions (PR5
#: corrected the old "trades 24/7" assumption), and its filings/earnings land
#: on the US session calendar; plain equities follow their listing exchange.
SESSION_CONTINUOUS = "continuous_24_7"
SESSION_BINANCE_TRADFI = "binance_published_tradfi_sessions"
SESSION_EXCHANGE = "listing_exchange_sessions"


def instrument_class(asset_type: str | None, ticker: str) -> str:
    """Classify an instrument for routing purposes.

    Returns one of ``"equity"`` (plain stock), ``"crypto_spot"`` (spot
    crypto, incl. the legacy auto-detected ``crypto`` mode), ``"stock_perp"``
    (Binance tokenized-stock USDT-M perpetual, e.g. MUUSDT → Micron) or
    ``"pure_crypto_perp"`` (a pure-crypto perpetual, e.g. BTCUSDT).
    ``stock_perp_underlying`` resolves against the warmed exchangeInfo
    EQUITY listing with the static seed fallback — no network here.
    """
    asset = str(asset_type or "stock")
    if asset not in CRYPTO_FAMILY:
        return "equity"
    if asset in ("crypto", "crypto_spot"):
        return "crypto_spot"
    if stock_perp_underlying(ticker):
        return "stock_perp"
    return "pure_crypto_perp"


#: High-confidence ETF bases among the equity universe (tokenized perps and
#: their plain-stock forms). A deliberately PRECISE heuristic floor — an
#: unrecognized ETF simply keeps the company framing, which is the safer
#: wrong answer; extend only with symbols you are sure are funds.
_ETF_BASES = frozenset({
    "BITO", "EWJ", "EWT", "EWY", "EWZ", "GDX", "IWM", "QQQ",
    "SMH", "SOXL", "SOXS", "SPY", "SQQQ", "TBT", "TMF", "TQQQ",
    "TZA", "URNM", "UVXY", "XBI", "XLE",
})


def _etf_base(asset_type: str | None, ticker: str) -> str:
    """Ticker with any USDT/USDC perp suffix stripped (uppercased)."""
    base = str(ticker).upper().strip()
    if asset_type in CRYPTO_FAMILY:
        for suffix in ("USDT", "USDC"):
            if base.endswith(suffix) and len(base) > len(suffix):
                base = base[: -len(suffix)]
                break
    return base


def is_exchange_traded_fund(asset_type: str | None, ticker: str) -> bool:
    """Heuristic ETF detection for fund-vs-company framing.

    Strips a USDT/USDC perp suffix when the run is crypto-family, then
    checks the base against the high-confidence ETF set. Used to switch the
    fundamentals framing from "operating company" (earnings, guidance) to
    "fund" (NAV premium/discount, AUM, expense ratio, tracking/leverage
    decay) — an ETF analyzed with company framing gets nonsense conclusions
    like earnings-gap risk for SPY. Fund-DATA routing (NAV/AUM/expense/
    holdings prefetch) keys off the same predicate.
    """
    return _etf_base(asset_type, ticker) in _ETF_BASES


def fundamentals_applicable(asset_type: str | None, ticker: str) -> bool:
    """True when the instrument has company fundamentals to read.

    Equities always qualify; a crypto-family instrument qualifies ONLY when
    it is a tokenized-stock perpetual (the analyst then reads the UNDERLYING
    US equity through the remap in the tool layer). Pure crypto has no
    company fundamentals — the caller must skip the analyst entirely rather
    than run it into an honest-but-billed no-data LLM turn.
    """
    return instrument_class(asset_type, ticker) in ("equity", "stock_perp")


# ---------------------------------------------------------------------------
# InstrumentDescriptor — routing facts + contract metadata, one object.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentDescriptor:
    """Everything the graph layer knows about one instrument, fetch-free.

    ``vendor_symbols`` maps each vendor namespace to the symbol it actually
    covers — the fundamentals layer's equity remap (``MUUSDT`` → ``MU``) is
    this map, not an ad-hoc call site. ``onboard_date`` (perps, from the
    warmed exchangeInfo snapshot) supports as-of listing classification:
    ``listed_asof`` answers "was this contract live on ``as_of``?" when both
    dates are known, and is ``None`` (unknown, NOT false) when the process
    only ever saw the static seed — the seed carries no onboard dates, and
    equities have no PIT listing source here at all.
    """

    symbol: str
    asset_type: str
    instrument_class: str
    is_etf: bool = False
    #: Perpetuals never deliver; spot has no contract; equities are
    #: securities. (Delivery-future support would land as its own kind.)
    contract_kind: str = "security"
    #: None for every instrument today: the universe is spot + perpetual.
    delivery_date: str | None = None
    session_calendar: str = SESSION_EXCHANGE
    vendor_symbols: dict[str, str] = field(default_factory=dict)
    underlying_equity: str | None = None
    onboard_date: str | None = None
    listing_status: str = "unknown"
    #: Disclosure rendered with as-of verdicts: listing evidence is a
    #: current-state snapshot, not an exchange archive.
    as_of: str | None = None
    listed_asof: bool | None = None

    @property
    def fundamentals_symbol(self) -> str:
        """The symbol the fundamentals vendors cover (underlying for perps)."""
        return self.underlying_equity or self.symbol

    def as_disclosure(self) -> str:
        """One-line human disclosure of the as-of listing verdict."""
        if self.as_of is None:
            return ""
        if self.listed_asof is None:
            return (
                f"listing as of {self.as_of}: unknown (no onboard-date "
                "evidence; classification uses the current listing snapshot)"
            )
        verdict = "listed" if self.listed_asof else "NOT yet listed"
        onboard = f", onboard {self.onboard_date}" if self.onboard_date else ""
        return (
            f"listing as of {self.as_of}: {verdict}{onboard} "
            "(current-snapshot evidence, not an exchange archive)"
        )


def describe_instrument(
    asset_type: str | None, ticker: str, as_of: str | None = None,
) -> InstrumentDescriptor:
    """Build the full descriptor for one instrument. NEVER fetches.

    Perp listing facts come from the warmed exchangeInfo snapshot (empty in
    seed mode); ``as_of`` (``YYYY-MM-DD``) turns the onboard date into a
    historical listing verdict — the classification answers "was this
    contract live then", which a current-state exchangeInfo call alone
    cannot.
    """
    ticker = str(ticker)
    asset = str(asset_type or "stock")
    klass = instrument_class(asset_type, ticker)
    is_etf = is_exchange_traded_fund(asset_type, ticker)
    if klass == "equity":
        return InstrumentDescriptor(
            symbol=ticker,
            asset_type=asset,
            instrument_class=klass,
            is_etf=is_etf,
            contract_kind="security",
            delivery_date=None,
            session_calendar=SESSION_EXCHANGE,
            vendor_symbols={"yfinance": ticker, "sec_edgar": ticker},
            underlying_equity=None,
            as_of=as_of,
            # No PIT listing source for plain equities here (IPO/delisting
            # history is not in any warmed snapshot).
            listed_asof=None,
        )
    if klass == "crypto_spot":
        return InstrumentDescriptor(
            symbol=ticker,
            asset_type=asset,
            instrument_class=klass,
            is_etf=False,
            contract_kind="spot",
            delivery_date=None,
            session_calendar=SESSION_CONTINUOUS,
            vendor_symbols={"binance": ticker.upper()},
            underlying_equity=None,
            as_of=as_of,
            listed_asof=None,
        )
    # Perps (stock_perp / pure_crypto_perp).
    underlying = stock_perp_underlying(ticker)
    compact = ticker.upper()
    listing = equity_perp_listing_info()
    base = _etf_base(asset_type, ticker)
    entry: dict[str, Any] = listing.get(base) or {}
    onboard_date = entry.get("onboard_date")
    listed_asof: bool | None = None
    if as_of is not None and isinstance(onboard_date, str):
        listed_asof = onboard_date <= as_of
    return InstrumentDescriptor(
        symbol=ticker,
        asset_type=asset,
        instrument_class=klass,
        is_etf=is_etf,
        contract_kind="perpetual",
        delivery_date=None,  # a perpetual by construction never delivers
        session_calendar=(
            SESSION_BINANCE_TRADFI if underlying else SESSION_CONTINUOUS
        ),
        vendor_symbols=(
            {"binance": compact, "yfinance": underlying, "sec_edgar": underlying}
            if underlying
            else {"binance": compact}
        ),
        underlying_equity=underlying,
        onboard_date=onboard_date,
        listing_status=entry.get("status") or "unknown",
        as_of=as_of,
        listed_asof=listed_asof,
    )
