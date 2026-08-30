"""Symbol normalization and market-data error types for vendor calls.

Yahoo Finance (the default vendor) uses specific ticker conventions that
differ from the broker / TradingView / MT5 style symbols users often type:

    user types        Yahoo wants       why
    ---------------   ---------------   -----------------------------------
    XAUUSD, XAUUSD+   GC=F              gold has no forex pair on Yahoo;
                                        it is quoted as a COMEX future
    EURUSD            EURUSD=X          spot forex pairs take a ``=X`` suffix
    BTCUSD            BTC-USD           crypto pairs use a ``-`` separator
    SPX500, US500     ^GSPC             index CFDs map to Yahoo index symbols

Passing the raw broker symbol to Yahoo returns an empty result, which the
agents previously received as free text and could hallucinate a price
around (see issue #781). Centralizing the mapping here means every yfinance
entry point resolves symbols the same way, and new instruments are added by
appending a table row rather than editing call sites.
"""

from __future__ import annotations

import logging
import re

# NoMarketDataError lives in the vendor-error taxonomy (errors.py); re-exported
# here for the many call sites that import it alongside normalize_symbol.
from .errors import NoMarketDataError as NoMarketDataError

logger = logging.getLogger(__name__)


# ISO-4217 codes common enough to appear in retail forex pairs. A bare
# six-letter symbol whose halves are BOTH in this set is treated as a spot
# forex pair and given Yahoo's ``=X`` suffix.
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

# Explicit aliases for instruments whose broker symbol does not map to a
# Yahoo symbol by rule. Metals/energy resolve to their front-month future;
# index CFD names resolve to the underlying Yahoo index symbol. Extend by
# adding rows — no call site changes required.
_ALIASES = {
    # Precious metals (spot names -> COMEX/NYMEX futures)
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # Energy
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # Index CFDs -> Yahoo index symbols
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

# Yahoo symbols may contain letters, digits, and these structural characters.
_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")


# Crypto quote currencies that all map to Yahoo's USD pair. Yahoo lists only
# ``<BASE>-USD`` (not the USDT/USDC stablecoin pairs), so a broker symbol quoted
# in any of these resolves to ``-USD`` (#982). Longest first so ``USDT``/``USDC``
# match before the ``USD`` substring.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")

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
    stablecoin quote (``MU``, ``600519.SS``) return None.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        return None
    compact = ticker.strip().upper().replace("-", "")
    for quote in ("USDT", "USDC"):
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            bases = _EQUITY_PERP_SEED_BASES if equity_perp_bases is None else equity_perp_bases
            if base in bases:
                return _EQUITY_BASE_YAHOO_ALIASES.get(base, base)
            return None
    return None


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
      1. Explicit alias table (metals, energy, index CFDs).
      2. Crypto rule: a known crypto base quoted in USD/USDT/USDC (dashed or
         not) -> ``BASE-USD``.
      3. Forex rule: six letters that are two ISO currency codes -> ``PAIR=X``.
      4. Otherwise the upper-cased symbol is returned unchanged (plain
         equities, ETFs, Yahoo-native symbols like ``GC=F`` or ``^GSPC``).

    A trailing ``+`` (broker CFD marker, e.g. ``XAUUSD+``) is stripped before
    matching. The function is purely syntactic — it performs no network
    calls — so it is safe to apply on every request.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    # Broker CFD/qualifier suffixes Yahoo never uses.
    s = s.rstrip("+")

    crypto = _normalize_crypto(s)
    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif crypto is not None:
        canonical = crypto
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    else:
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


def is_a_stock(ticker: str) -> bool:
    """True when ``ticker`` is a China A-share (Shanghai or Shenzhen) symbol.

    Recognizes the Yahoo-style exchange suffixes ``.SS``/``.SH`` (Shanghai SSE)
    and ``.SZ`` (Shenzhen SZSE) on a 6-digit code — the same form yfinance
    accepts for A-shares (e.g. ``600519.SS``, ``000001.SZ``). Purely syntactic
    (no network), so it is safe as a per-call gate. Used to decide whether the
    optional ``a_stock`` Eastmoney tools (margin trading + capital flow) are
    advertised to the fundamentals analyst, so US / crypto / HK tickers never
    enter that branch (byte-equivalent to a default-off run for them).
    """
    if not isinstance(ticker, str):
        return False
    t = ticker.strip().upper()
    for suf in (".SS", ".SH", ".SZ"):
        if t.endswith(suf):
            code = t[: -len(suf)]
            return code.isdigit() and len(code) == 6
    return False


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

    s = raw.strip().upper().rstrip("+").replace("-", "")

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
