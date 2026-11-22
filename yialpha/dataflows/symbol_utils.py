"""Symbol normalization and market-data error types for vendor calls.

Only two markets are supported: US-listed equities/ETFs and crypto
(:func:`is_supported_market` / :func:`require_supported_market` gate every
entry point). The former metals/energy/index-CFD alias table and the spot
forex ``=X`` mapping were removed with the A-share/forex/CFD strip — those
inputs are now rejected at the gate with :class:`UnsupportedMarketError`
instead of being silently remapped, so the only normalization left is the
crypto rule:

    user types             Yahoo wants       why
    ---------------        ---------------   -----------------------------------
    BTCUSD                 BTC-USD           crypto pairs use a ``-`` separator
    BTCUSDT, BTC-USDT      BTC-USD           stablecoin quotes resolve to the
                                             Yahoo ``-USD`` pair (#982)

Everything else (``AAPL``, ``BRK.B``, ``SPY``) passes through upper-cased
unchanged. Passing an unsupported symbol to Yahoo used to return an empty
result, which the agents received as free text and could hallucinate a
price around (see issue #781); centralizing the allowlist here means every
entry point rejects the same set of symbols with the same message.
"""

from __future__ import annotations

import logging
import re

# NoMarketDataError lives in the vendor-error taxonomy (errors.py); re-exported
# here for the many call sites that import it alongside normalize_symbol.
# UnsupportedMarketError is re-exported the same way so the entry-point gates
# can import the whole gate contract (predicate + raiser + error) from here.
from .errors import (
    NoMarketDataError as NoMarketDataError,
    UnsupportedMarketError as UnsupportedMarketError,
)

logger = logging.getLogger(__name__)


# ISO-4217 codes common enough to appear in retail forex pairs. The spot
# forex ``=X`` normalization is gone; the set survives as the deny-check
# vocabulary for the market gate — a bare six-letter symbol whose halves are
# BOTH in this set is a spot forex pair, which is no longer a supported
# market and must be rejected with a specific reason (not fall through to
# the crypto-venue shape check, which a ``...USD`` pair would otherwise
# satisfy).
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)

# Crypto bases that brokers quote against USD without a separator.
_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK"}
)

# Keys of the former metals/energy/index-CFD alias table (kept as a deny
# set): these broker names used to remap to futures/index symbols and would
# now otherwise look like plausible tickers downstream. Rejecting them by
# name gives the user a precise "no longer supported" error instead of a
# confusing empty-data failure from the vendor.
_REMOVED_CFD_ALIASES = frozenset(
    {
        # Precious metals (former spot names -> COMEX/NYMEX futures)
        "XAUUSD", "XAU", "GOLD", "XAGUSD", "XAG", "SILVER",
        "XPTUSD", "XPDUSD",
        # Energy
        "WTICOUSD", "USOIL", "WTI", "BCOUSD", "UKOIL", "BRENT",
        "NATGAS", "XNGUSD", "COPPER", "XCUUSD",
        # Index CFDs (former -> Yahoo index symbols)
        "SPX500", "US500", "SPX", "NAS100", "US100", "USTEC",
        "US30", "DJI30", "WS30", "GER40", "GER30", "DE40",
        "UK100", "JP225", "JPN225", "FRA40", "EU50", "HK50",
    }
)

# Yahoo symbols may contain letters, digits, and these structural characters.
_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")


# Crypto quote currencies that all map to Yahoo's USD pair. Yahoo lists only
# ``<BASE>-USD`` (not the USDT/USDC stablecoin pairs), so a broker symbol quoted
# in any of these resolves to ``-USD`` (#982). Longest first so ``USDT``/``USDC``
# match before the ``USD`` substring.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")

# US-listed equity/ETF ticker shape for the market gate: 1-6 letters plus an
# optional single class-share letter A/B/C after ``.`` or ``-`` (covers AAPL,
# SPY, BRK.B, BRK-B, BF.B). Letters-only means it naturally rejects Yahoo
# index/futures sigils (``^GSPC``, ``GC=F``), non-US exchange suffixes
# (.SS/.SZ/.HK/.T/.L/.NS/.TO/.AX) and the 6-digit A-share codes — no
# per-suffix deny list to maintain.
_US_STOCK_RE = re.compile(r"^[A-Z]{1,6}(?:[.\-][A-C])?$")

# Binance USDT-M lists tokenized US-equity perpetuals (``underlyingType ==
# "EQUITY"`` in exchangeInfo): the contract tracks a real listed company or
# ETF, so the underlying's fundamentals ARE analyzable through the normal
# stock vendors even though the perp itself is a crypto instrument. This seed
# is a verified snapshot (2026-08-17, fapi /exchangeInfo, status TRADING) and
# the static FAIL-OPEN floor: the live resolver in
# :func:`yialpha.dataflows.binance.equity_perp_bases` supersedes it with the
# runtime exchangeInfo listing, so extend this set only to raise the floor
# (new listings do NOT require a code change).
_EQUITY_PERP_SEED_BASES = frozenset(
    {
        "AAOI", "AAPL", "ADBE", "ALAB", "AMAT", "AMD",
        "AMZN", "APP", "ARM", "ASML", "ASTS", "AVGO",
        "AXTI", "BABA", "BBX", "BE", "BITO", "BMNR",
        "BNC", "BOT", "BRKB", "BSP", "BX", "CAT",
        "CBRS", "CIEN", "COHR", "COIN", "COST", "CRCL",
        "CRDO", "CRM", "CRWD", "CRWV", "CSCO", "DELL",
        "DIS", "DKNG", "DRAM", "EBAY", "EWJ", "EWT",
        "EWY", "EWZ", "FLEX", "FLNC", "FWDI", "GDX",
        "GEV", "GLW", "GME", "GOOGL", "GS", "HD",
        "HIMS", "HOOD", "HPE", "IBM", "INTC", "INTW",
        "IREN", "IWM", "JPM", "KLAC", "KO", "KORU",
        "KSTR", "LITE", "LLY", "LRCX", "LYTE", "META",
        "MRVL", "MSFT", "MSTR", "MU", "MUU", "MVLL",
        "NBIS", "NET", "NFLX", "NOK", "NOW", "NVDA",
        "NVO", "ONDS", "ORCL", "PANW", "PAYP", "PENG",
        "PLTR", "PYPL", "QCOM", "QNTX", "QQQ", "RDDT",
        "RIVN", "RKLB", "SHAZ", "SHOP", "SKHY", "SMCI",
        "SMH", "SNDK", "SNOW", "SNXX", "SOFI", "SONY",
        "SOXL", "SOXS", "SPCX", "SPY", "SQQQ", "STRC",
        "STXX", "TBT", "TER", "TMF", "TQQQ", "TSLA",
        "TSM", "TTWO", "TXN", "TZA", "UBER", "URNM",
        "USAR", "UVXY", "V", "VRT", "VST", "WDC",
        "WEN", "WMT", "XBI", "XLE", "ZM",
    }
)

# Binance base symbols whose Yahoo equivalent is not the identity. Binance
# drops the class-share separator; Yahoo wants the dashed form.
_EQUITY_BASE_YAHOO_ALIASES = {
    "BRKB": "BRK-B",
}


def tokenized_stock_perp_underlying(
    ticker: str, equity_perp_bases: frozenset[str] | None = None
) -> str | None:
    """Return the Yahoo-ready US-equity symbol if ``ticker`` is a Binance
    tokenized-stock USDT-M perp, else None.

    Purely syntactic against the supplied base set — no network calls — so it
    is safe in tests and on hot paths. ``equity_perp_bases`` defaults to the
    static seed snapshot; pass the live set from
    :func:`yialpha.dataflows.binance.equity_perp_bases` for current-listing
    freshness. Accepts dashed or compact forms and USDT/USDC quotes
    (``MUUSDT``/``MU-USDT``/``MUUSDC`` -> ``MU``). Bases with a Yahoo spelling
    exception are mapped (``BRKBUSDT`` -> ``BRK-B``). Pure-crypto perps
    (``BTCUSDT``), unlisted alts (``PEPEUSDT``), and anything without a
    stablecoin quote (``MU``, ``AAPL``) return None.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        return None
    compact = ticker.strip().upper().replace("-", "")
    for quote in ("USDT", "USDC"):
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            bases = _EQUITY_PERP_SEED_BASES if equity_perp_bases is None else equity_perp_bases
            if base in bases:
                return yahoo_equity_symbol(base)
            return None
    return None


def yahoo_equity_symbol(base: str) -> str:
    """Yahoo-ready symbol for a Binance equity perp base (``BRKB`` → ``BRK-B``).

    The single alias seam so every path that learns a stock-perp base — the
    warm/seed matcher above AND the instrument-registry resolver in
    :func:`yialpha.dataflows.binance.stock_perp_underlying` (whose rows carry
    the raw exchangeInfo ``baseAsset``) — maps to the identical Yahoo symbol.
    """
    b = str(base).strip().upper()
    return _EQUITY_BASE_YAHOO_ALIASES.get(b, b)


def _normalize_crypto(s: str) -> str | None:
    """Return ``<BASE>-USD`` if ``s`` is a known crypto quoted in USD/USDT/USDC.

    Accepts dashed or undashed forms: ``BTCUSD``, ``BTCUSDT``, ``BTC-USDT``,
    ``BTC-USDC`` all resolve to ``BTC-USD``. Returns None otherwise.
    """
    compact = s.replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            if base in _CRYPTO_BASES:
                return f"{base}-USD"
            break
    return None


def normalize_symbol(raw: str) -> str:
    """Map a user/broker symbol to its canonical Yahoo Finance symbol.

    Resolution order (first match wins):
      1. Crypto rule: a known crypto base quoted in USD/USDT/USDC (dashed or
         not) -> ``BASE-USD``.
      2. Otherwise the upper-cased symbol is returned unchanged (plain
         US-listed equities/ETFs, e.g. ``AAPL``, ``BRK.B``).

    Only US-listed equities and crypto are supported markets — see
    :func:`require_supported_market`; former alias/forex inputs are rejected
    at the entry-point gates, not remapped here. A trailing ``+`` (broker
    CFD marker, e.g. ``BTCUSD+``) is stripped before matching. The function
    is purely syntactic — it performs no network calls — so it is safe to
    apply on every request.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    # Broker CFD/qualifier suffixes Yahoo never uses.
    s = s.rstrip("+")

    canonical = _normalize_crypto(s)
    if canonical is None:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    """True when ``symbol`` only contains characters Yahoo symbols use."""
    return bool(symbol) and _YAHOO_SAFE.fullmatch(symbol) is not None


def is_crypto_symbol(raw: str) -> bool:
    """True when ``raw`` is a known crypto base quoted in USD/USDT/USDC.

    Accepts every spelling :func:`_normalize_crypto` accepts (``BTCUSD``,
    ``BTCUSDT``, ``BTC-USD``, ...). Purely syntactic (no network) — used to
    pick 24/7 annualization (365 periods) for vol estimators on Yahoo-resolved
    crypto series, which otherwise default to the equity 252 convention and
    understate crypto vol by sqrt(252/365).
    """
    if not isinstance(raw, str) or not raw.strip():
        return False
    return _normalize_crypto(raw.strip().upper().rstrip("+")) is not None


def _gate_deny_reason(s: str) -> str | None:
    """Specific unsupported-market reason for a cleaned symbol, else None.

    The forex/CFD denies sit BEFORE the allow checks in
    :func:`is_supported_market` because both shapes would otherwise sneak
    through: a ``...USD`` forex pair satisfies the crypto-venue suffix check,
    and bare alias words like ``GOLD``/``SPX`` match the US-stock regex.
    """
    if len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        return "spot forex pairs are no longer supported"
    if s in _REMOVED_CFD_ALIASES:
        return "former commodity/index CFD alias, no longer supported"
    return None


# One policy sentence shared by every gate rejection (CLI retry hint, batch
# validation error, web 400) so the wording cannot drift per entry point.
_MARKET_POLICY = (
    "only US-listed stocks/ETFs (e.g. AAPL, BRK.B, SPY) and crypto "
    "(e.g. BTC-USD, BTCUSDT) are supported; A-shares, HK and other "
    "exchange listings, forex, commodities and index CFDs were removed"
)


def is_supported_market(raw: str) -> bool:
    """True when ``raw`` is a supported market symbol: US equity or crypto.

    Accepts (in order, first match wins):
      (a) known crypto — a :data:`_CRYPTO_BASES` base quoted in
          USD/USDT/USDC (``BTC-USD``, ``BTCUSDT``), via
          :func:`is_crypto_symbol`;
      (b) crypto-venue-shaped — a compact (dashes removed) symbol ending in
          ``USDT``/``USDC``/``USD`` with a non-empty base, or a dashed
          ``-BTC``/``-ETH`` quote (``ETH-BTC``). These flow down the
          Binance/Yahoo crypto paths whose underlyings are already
          US-equity-or-crypto — Binance rejects non-US-equity perp
          underlyings in the registry — so the shape alone is sufficient;
      (c) US-listed equity/ETF — :data:`_US_STOCK_RE` full match.

    False otherwise, including spot forex pairs (both halves in
    :data:`_FOREX_CURRENCIES`), the removed CFD aliases, ``^``/``=``
    symbols, and non-US exchange suffixes (.SS/.HK/.T/...). Purely
    syntactic — no network calls.
    """
    if not isinstance(raw, str) or not raw.strip():
        return False
    s = raw.strip().upper().rstrip("+")

    if _normalize_crypto(s) is not None:
        return True
    if _gate_deny_reason(s) is not None:
        return False

    # (b) venue shape. Longest quote first so USDT/USDC are not shadowed by
    # the USD substring; require a non-empty base so a bare "USDT"/"USD"
    # does not pass as a pair with no underlying.
    compact = s.replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote) and len(compact) > len(quote):
            return True
    if s.endswith(("-BTC", "-ETH")) and len(s) > len("-BTC"):
        return True

    return _US_STOCK_RE.fullmatch(s) is not None


def require_supported_market(raw: str) -> str:
    """Clean ``raw`` and enforce the US-stocks + crypto allowlist gate.

    Returns the cleaned symbol (strip/upper, trailing ``+`` removed) when
    supported; otherwise raises :class:`UnsupportedMarketError` naming the
    ticker and stating the policy. Forex-shaped and former-CFD-alias inputs
    get their specific reason prepended so the user learns *why*, not just
    *that*. Purely syntactic — the entry points (CLI prompt, batch runner,
    web API) call this so every path rejects the same set with the same
    message.
    """
    s = raw.strip().upper().rstrip("+") if isinstance(raw, str) else ""
    if s and is_supported_market(s):
        return s
    reason = _gate_deny_reason(s) if s else None
    raise UnsupportedMarketError(
        s or str(raw), f"{reason} — {_MARKET_POLICY}" if reason else _MARKET_POLICY
    )


def _plausible_delivery_code(code: str) -> bool:
    """YYMMDD sanity for a quarterly-delivery suffix.

    Binance delivery contracts expire on quarterly/monthly dates through the
    2020s-2040s; requiring a real month/day in that year window keeps numeric
    bases (which start with digits, never end in a date-shaped 6-pack) from
    tripping the delivery guard.
    """
    yy, mm, dd = int(code[:2]), int(code[2:4]), int(code[4:6])
    return 24 <= yy <= 49 and 1 <= mm <= 12 and 1 <= dd <= 31


def normalize_symbol_for_venue(raw: str, venue: str = "binance_perp") -> str:
    """Map a user/broker symbol to a venue's canonical symbol.

    ``venue="yahoo"`` delegates to :func:`normalize_symbol` so the Yahoo path
    is 100% unchanged (same function, same output). For ``"binance_perp"`` and
    ``"binance_spot"``, the symbol is normalized to a Binance USDT pair
    (``<BASE>USDT``) — both the USDT-M perpetual book and the spot USDT book
    trade under the same ``<BASE>USDT`` symbol, so the two venues share one
    normalization:

      - upper-cased; trailing ``+`` (broker CFD marker) stripped
      - dashes removed (``BTC-USDT`` -> ``BTCUSDT``)
      - quote suffix collapsed to ``USDT``: ``USD``/``USDC`` endings become
        ``USDT`` (Binance USDT-M lists against USDT), ``USDT`` kept as-is
      - pairs already ending in ``USDT`` are returned upper-cased and un-dashed

    A degenerate guard rejects a result that does not separate a base from the
    ``USDT`` quote (e.g. an input that normalizes to a bare ``USDTUSDT``),
    which would otherwise price the wrong instrument.

    Purely syntactic — no network calls — so it is safe to apply on every
    request. Does not alter :func:`normalize_symbol` or any Yahoo call site.
    """
    if venue == "yahoo":
        return normalize_symbol(raw)

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Cannot normalize empty symbol for venue {venue!r}")

    # Round-7: also strip "/" and ":" — the ccxt/TradingView pair forms
    # ("BTC/USDT", "BTC:USDT") survived as literal separators and reached
    # the vision store's cache-filename sanitizer as an untyped
    # "unsafe cache filename" ValueError (the REST twin typed the same
    # input through NoMarketDataError with symbol context).
    s = (
        raw.strip().upper().rstrip("+")
        .replace("-", "")
        .replace("/", "")
        .replace(":", "")
    )

    if venue not in ("binance_perp", "binance_spot"):
        # Unknown venue: fall back to the cleaned form rather than guessing.
        return s

    # Delivery/quarterly contract suffix (BTCUSDT_260327, BTCUSDT260326):
    # this data layer serves PERPETUAL-only endpoints (klines/funding hardcode
    # contractType=PERPETUAL). The bare-base fall-through below used to
    # append USDT anyway, pricing a fabricated symbol like
    # "BTCUSDT_260327USDT" — refusing loudly is the honest behaviour.
    m = re.search(r"(\d{6})$", s)
    if m and _plausible_delivery_code(m.group(1)):
        raise ValueError(
            f"Symbol {raw!r} looks like a dated delivery contract, which the "
            "Binance PERPETUAL-only data layer does not serve; pass the perp "
            "form (e.g. 'BTCUSDT') instead."
        )

    # Collapse USD/USDC quote suffixes to USDT (USDT-M book). Order matters:
    # match the longest suffix first so "USDT" is not shadowed by "USD".
    if s.endswith("USDT"):
        base = s[:-4]
    elif s.endswith("USDC"):
        base = s[:-4]
        logger.info("Binance perp symbol %r quoted in USDC; mapping to USDT pair", raw)
    elif s.endswith("USD"):
        base = s[:-3]
        logger.info("Binance perp symbol %r quoted in USD; mapping to USDT pair", raw)
    else:
        # No recognized quote suffix: assume the caller passed a bare base
        # (e.g. "BTC") and append USDT so it prices the USDT-M pair.
        base = s

    if not base or base == "USDT":
        raise ValueError(
            f"Symbol {raw!r} normalizes to a degenerate Binance perp pair "
            f"(base={base!r}); pass an explicit base like 'BTC' or 'BTCUSDT'."
        )

    return base + "USDT"
