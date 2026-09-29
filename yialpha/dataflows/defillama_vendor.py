"""DefiLlama keyless vendor for the optional ``crypto_native`` category.

Supplies network / protocol fundamentals for PURE-crypto instruments: protocol
TVL history (for governance tokens like UNI / AAVE), chain TVL history (for
L1/L2 assets like ETH / SOL / BTC), and — live runs only — the chain's
current TVL ranking. Same optional-category contract shape as
``a_share_native``:

* **Free / keyless** — the public ``api.llama.fi`` endpoints (no key, no
  documented hard quota; verified live 2026-09-29).
* **Foreign host → rides the SOCKS5 proxy** via ``proxy_map()`` (sec_edgar
  idiom). Unlike CoinGecko the edge accepts any UA; a browser-style UA is sent
  anyway so the two vendors share one fingerprint policy.
* **Politeness** — ``MinIntervalThrottle(1.0)`` serial spacing; the /protocols
  catalog (~8.4k entries, several MB) is fetched at most once per day and only
  a compact projection (slug / symbol / tvl / chains / gecko_id) is cached,
  never the raw blob.
* **PIT semantics** — history endpoints are full daily series from genesis
  (no CoinGecko-style 365-day cap); every view filters rows to
  ``date <= curr_date``. The chain RANKING is a current-only snapshot and is
  therefore served to live runs only.
* **Fail-soft via typed errors** — transport failures and 4xx/5xx raise
  ``NoMarketDataError(detail=...)``; 429 → ``VendorRateLimitError``; the
  router degrades this optional category to a sentinel instead of aborting.

Symbol resolution: protocol = the highest-TVL DefiLlama protocol whose
``symbol`` equals the contract's base AND whose TVL clears a $1M credibility
floor (UNI resolves across uniswap-v1..v4 by TVL, with the version disclosed
— a token is one project, its TVL is the flagship deployment's; the floor
blocks junk symbol collisions like SHIB→'shibex' at $6K TVL, verified live
2026-09-29). Chain = the static base→chain map first, then the /v2/chains
catalog's ``tokenSymbol`` (highest TVL, same $1M floor) so newly listed
chains auto-extend coverage; a protocol's primary chain is the last
fallback. Wrapper-multiplier contracts (1000PEPE / 1000000MOG) reduce to
the un-prefixed token (shared prefix rule with the CoinGecko vendor).
Neither leg resolving is an honest NoMarketDataError; a leg that resolves
but fails to FETCH degrades in-band (per-leg fail-soft — a broken leg never
discards the other leg's rendered evidence).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .coingecko_vendor import (
    _MULTIPLIER_PREFIXES,
    _end_of_day_ts,
    _fmt_num,
    _fmt_usd,
    perp_base_symbol,
)
from .disk_cache import MinIntervalThrottle, cached_or_fetch, vendor_cache_dir
from .errors import NoMarketDataError, VendorRateLimitError
from .utils import is_historical_date, proxy_map

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.llama.fi"
_TIMEOUT = 30
_throttler = MinIntervalThrottle(1.0)
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: Base symbol → DefiLlama chain name for chains that (a) carry Binance
#: perps and (b) have meaningful chain-level TVL. Misses are normal and
#: honest (e.g. DOGE has no DefiLlama chain TVL) — the protocol leg or the
#: NoMarketDataError path then decides coverage.
_BASE_CHAIN: dict[str, str] = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "BNB": "BSC",
    "TRX": "Tron", "AVAX": "Avalanche", "DOT": "Polkadot", "NEAR": "Near",
    "APT": "Aptos", "SUI": "Sui", "TON": "Ton", "ARB": "Arbitrum",
    "OP": "Optimism", "BASE": "Base", "MATIC": "Polygon", "POL": "Polygon",
    "ADA": "Cardano", "ATOM": "Cosmos", "ALGO": "Algorand", "FTM": "Fantom",
    "CRO": "Cronos", "STRK": "Starknet", "ZK": "zkSync Era", "ETHFI": "Ethereum",
    "SEI": "Sei", "INJ": "Injective", "RUNE": "Thorchain", "KAVA": "Kava",
    "EGLD": "Elrond", "ONE": "Harmony", "CELO": "Celo", "GLMR": "Moonbeam",
    "AZERO": "Aleph Zero", "TAO": "Bittensor",
}


def _cache_dir() -> str:
    return vendor_cache_dir("defillama")


def _throttle() -> None:
    _throttler.wait()


def _http_get_json(url: str) -> Any:
    """GET a DefiLlama endpoint (throttle + UA + proxy) and parse JSON.

    Same status→error mapping as the CoinGecko vendor / sec_edgar idiom.
    """
    import requests

    _throttle()
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _UA, "Accept": "application/json"},
            proxies=proxy_map(),
            timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 -- network failure -> typed error
        raise NoMarketDataError(url, detail=f"DefiLlama request failed: {exc!r}") from exc
    if resp.status_code == 429:
        raise VendorRateLimitError(f"DefiLlama returned 429 for {url}")
    if resp.status_code >= 400:
        body = resp.text[:160].replace("\n", " ")
        raise NoMarketDataError(
            url, detail=f"DefiLlama HTTP {resp.status_code}: {body}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise NoMarketDataError(url, detail=f"DefiLlama returned non-JSON: {exc}") from exc


def _cached_json(filename: str, fetch, ttl_days: float) -> Any:
    """cached_or_fetch adapter returning parsed JSON, with an mtime-keyed
    parse memo (same contract as the CoinGecko vendor's — the analyst LLM
    tool loop re-calls these tools and re-parsing the ~1MB projection per
    round is pure waste)."""
    raw = cached_or_fetch(
        _cache_dir(), filename, lambda: json.dumps(fetch()).encode("utf-8"),
        ttl_days=ttl_days, vendor="defillama",
    )
    if raw is None:  # unreachable: fail_open is never set for this vendor
        raise NoMarketDataError(filename, detail="cache helper returned None")
    key = (_cache_dir(), filename)
    try:
        mtime = os.path.getmtime(os.path.join(key[0], filename))
    except OSError:
        mtime = -1.0
    hit = _PARSE_MEMO.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    parsed = json.loads(raw)
    _PARSE_MEMO[key] = (mtime, parsed)
    return parsed


#: (cache dir, filename) -> (mtime, parsed payload); see _cached_json.
_PARSE_MEMO: dict[tuple[str, str], tuple[float, Any]] = {}


def _protocols_projection() -> list[dict[str, Any]]:
    """Compact projection of the /protocols catalog, cached once per day.

    The raw catalog is several MB and 95% fields this vendor never reads; the
    cached artifact is the projection only, so the on-disk footprint stays
    ~1 MB and JSON parse cost per run stays trivial.
    """
    payload = _cached_json(
        "protocols_projection.json",
        lambda: [
            {
                "slug": p.get("slug") or p.get("id"),
                "name": p.get("name"),
                "symbol": str(p.get("symbol") or "").upper(),
                "tvl": p.get("tvl") or 0.0,
                "chains": (p.get("chains") or [])[:6],
                "gecko_id": p.get("gecko_id"),
                "category": p.get("category"),
            }
            for p in _http_get_json(f"{_BASE_URL}/protocols")
            if isinstance(p, dict) and (p.get("slug") or p.get("id"))
        ],
        ttl_days=1.0,
    )
    return payload if isinstance(payload, list) else []


#: Credibility floor for protocol symbol matches: a Binance-perp token's real
#: flagship protocol is well above $1M TVL, while symbol-collision junk sits
#: in the $Ks (shibex vs SHIB). Below-floor matches are treated as no match.
_PROTOCOL_MIN_TVL = 1_000_000.0


def _resolve_protocol(base: str) -> dict[str, Any] | None:
    """Highest-TVL protocol whose governance symbol equals ``base``.

    Versioned deployments (uniswap-v1..v4 all carry symbol UNI) collapse to
    the flagship by TVL; the chosen slug is disclosed in-band so the analyst
    can flag the version choice rather than trust it blindly. Matches below
    ``_PROTOCOL_MIN_TVL`` are ignored (junk-symbol protection).
    """
    matches = [
        p for p in _protocols_projection()
        if p.get("symbol") == base.upper()
        and float(p.get("tvl") or 0.0) >= _PROTOCOL_MIN_TVL
    ]
    if not matches:
        return None
    return max(matches, key=lambda p: float(p.get("tvl") or 0.0))


def _chains_catalog() -> list[dict[str, Any]]:
    """The cached /v2/chains snapshot (name, tokenSymbol, tvl) — the same
    payload the live ranking uses, reused as a data-driven base→chain
    resolver so newly listed chains auto-extend coverage without a code
    edit. Cache-only in effect: a fetch failure raises and callers treat
    the resolver as unavailable (honest omission, not a fabricated chain).
    """
    payload = _cached_json(
        "chains.json", lambda: _http_get_json(f"{_BASE_URL}/v2/chains"),
        ttl_days=1.0,
    )
    return [c for c in payload if isinstance(c, dict)] if isinstance(payload, list) else []


def _chain_cache_key(chain: str) -> str:
    """Filename-safe cache key for a chain name ('zkSync Era' contains a
    space, which the disk-cache allowlist validator rejects — sanitize is a
    validator, not a transformer, so fold disallowed chars here first)."""
    import re

    return re.sub(r"[^a-z0-9._()-]", "_", chain.lower())


def _chain_for_base(base: str) -> str | None:
    """Resolve the base token's own chain: the static map first (curated,
    includes token-not-chain approximations like ETHFI→Ethereum), then the
    /v2/chains catalog's ``tokenSymbol`` (highest TVL wins, $1M floor —
    same junk protection as protocol resolution)."""
    chain = _BASE_CHAIN.get(base)
    if chain:
        return chain
    try:
        ranked = sorted(
            (
                c for c in _chains_catalog()
                if str(c.get("tokenSymbol") or "").upper() == base.upper()
                and isinstance(c.get("tvl"), (int, float))
                and c["tvl"] >= _PROTOCOL_MIN_TVL
            ),
            key=lambda c: c["tvl"], reverse=True,
        )
    except (NoMarketDataError, VendorRateLimitError):
        return None  # resolver unavailable — the caller omits the leg honestly
    if ranked:
        return str(ranked[0].get("name") or "") or None
    return None


def _pit_window(
    points: list[dict[str, Any]], key_ts: str, key_val: str,
    curr_date: str, look_back_days: int,
) -> list[tuple[str, float]]:
    """Filter ``[{key_ts, key_val}, ...]`` series to the PIT window."""
    as_of = str(curr_date)[:10]
    upper = _end_of_day_ts(as_of)
    lower = (date.fromisoformat(as_of) - timedelta(days=look_back_days))
    lower_ts = int(datetime(lower.year, lower.month, lower.day,
                            tzinfo=timezone.utc).timestamp())
    out: list[tuple[str, float]] = []
    for p in points if isinstance(points, list) else []:
        if not isinstance(p, dict):
            continue
        try:
            ts, val = int(p[key_ts]), float(p[key_val])
        except (KeyError, TypeError, ValueError):
            continue
        if lower_ts <= ts <= upper:
            out.append((
                datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                val,
            ))
    return out


def _render_series(
    title: str, unit_word: str, rows: list[tuple[str, float]], sample: int = 30,
) -> str:
    if not rows:
        return (
            f"{title}: no rows fall inside the requested window — report 'data "
            "not available' for this section; do not estimate."
        )
    sampled = rows[-sample:]
    first, last = rows[0][1], rows[-1][1]
    chg = (last / first - 1) * 100 if first else None
    peak_d, peak_v = max(rows, key=lambda r: r[1])
    lines = [
        f"{title} — point-in-time daily {unit_word} (rows filtered to "
        f"date <= as-of; {len(rows)} rows in window, table shows last "
        f"{len(sampled)}):",
        "",
        f"| date | {unit_word} |",
        "|---|---|",
    ]
    lines += [f"| {d} | {_fmt_usd(v)} |" for d, v in sampled]
    summary = f"\nWindow summary: {_fmt_usd(first)} -> {_fmt_usd(last)}"
    if chg is not None:
        summary += f" ({chg:+.1f}%)"
    summary += f"; window peak {_fmt_usd(peak_v)} on {peak_d}."
    return "\n".join(lines) + summary


def _chain_ranking(chain: str) -> str:
    ranked = sorted(
        (c for c in _chains_catalog()
         if isinstance(c.get("tvl"), (int, float))),
        key=lambda c: c["tvl"], reverse=True,
    )
    for i, c in enumerate(ranked, start=1):
        if str(c.get("name", "")).lower() == chain.lower():
            return (
                f"\nChain context (current snapshot): {chain} ranks #{i} of "
                f"{len(ranked)} DefiLlama-tracked chains with {_fmt_usd(c['tvl'])} TVL."
            )
    return ""


def get_crypto_network_fundamentals(
    ticker: str, curr_date: str | None = None, look_back_days: int = 180,
) -> str:
    """Network / protocol fundamentals for a pure-crypto instrument (DefiLlama).

    Two evidence legs, whichever resolve: (1) protocol TVL history when the
    base symbol is a DeFi protocol's governance token (highest-TVL version,
    slug disclosed); (2) chain TVL history for the base's chain (static map,
    falling back to the protocol's primary chain). Live runs additionally get
    the chain's current TVL ranking; historical runs omit it (current-only
    snapshot — serving it would leak future information). Full-history daily
    series, PIT-filtered to date <= curr_date.

    Args:
        ticker: Contract symbol, e.g. BTCUSDT / UNIUSDT / AAVEUSDT.
        curr_date: As-of date yyyy-mm-dd (the trade date being analyzed).
        look_back_days: Series window in days (default 180).

    Returns:
        Markdown sections + honest omission notes. Raises NoMarketDataError
        when neither a protocol nor a chain resolves (the router then degrades
        this optional category to a sentinel).
    """
    base = perp_base_symbol(ticker)
    # Wrapper-multiplier contracts (1000PEPE / 1000CAT / 1000000MOG) track
    # the un-prefixed token — DefiLlama's catalog never lists the wrapper
    # symbol, so strip it (shared prefix set + threshold with the CoinGecko
    # resolution ladder).
    for prefix in _MULTIPLIER_PREFIXES:
        if base.startswith(prefix) and len(base) - len(prefix) >= 3:
            base = base[len(prefix):]
            break

    historical = is_historical_date(curr_date)
    # Malformed non-empty dates route into the historical branch fail-closed
    # (same guard as the CoinGecko vendor); degrade honestly instead of
    # crashing on date.fromisoformat inside the window helpers.
    if historical:
        try:
            date.fromisoformat(str(curr_date)[:10])
        except ValueError:
            return (
                f"Crypto network fundamentals for {base}: the passed "
                f"curr_date '{curr_date}' is not a parseable yyyy-mm-dd "
                "date, so no point-in-time window can be computed. Report "
                "'data not available' for this item; do not estimate."
            )
    # Live runs (curr_date empty/today) anchor the window at today — DefiLlama
    # daily points are end-of-day so today's row is at most the running value.
    as_of = str(
        curr_date or datetime.now(timezone.utc).date().isoformat()
    )[:10]
    as_of_note = f"as-of {as_of}" if curr_date else "live, current data"

    sections: list[str] = []
    failures: list[Exception] = []

    # --- Protocol leg (independent fail-soft: a broken leg must not discard
    # the other leg's rendered evidence — the whole tool only degrades to a
    # sentinel when NOTHING renders). ---
    protocol = None
    try:
        protocol = _resolve_protocol(base)
    except Exception as exc:  # noqa: BLE001 — per-leg degradation
        failures.append(exc)
        logger.warning("DefiLlama protocol resolution failed for %s: %s", base, exc)

    protocol_rendered = False
    if protocol is not None:
        try:
            payload = _cached_json(
                f"protocol_{_chain_cache_key(protocol['slug'])}.json",
                lambda slug=protocol["slug"]: _http_get_json(
                    f"{_BASE_URL}/protocol/{slug}"
                ),
                ttl_days=1.0,
            )
            rows = _pit_window(
                payload.get("tvl") or [], "date", "totalLiquidityUSD",
                as_of, look_back_days,
            )
            # PIT title discipline: the catalog's TVL is a TODAY snapshot —
            # serving it into a historical run would leak a future figure
            # (the chain ranking is omitted for exactly this reason). The
            # version SELECTION also keys on today's catalog; that
            # approximation cannot be repaired without fetching every
            # candidate's history, so it is disclosed instead of hidden.
            tvl_phrase = (
                f", current TVL {_fmt_usd(protocol.get('tvl'))}"
                if not historical else ""
            )
            caveat = (
                " — NOTE: the deployment version was picked by TODAY's "
                "DefiLlama catalog TVL, not the as-of ranking; treat the "
                "version choice as an approximation"
                if historical else ""
            )
            sections.append(
                _render_series(
                    f"Protocol TVL — {protocol.get('name')} (slug "
                    f"'{protocol['slug']}', category "
                    f"{protocol.get('category') or 'n/a'}; chosen as the "
                    f"highest-TVL deployment carrying symbol {base}"
                    f"{tvl_phrase}){caveat}",
                    "TVL", rows,
                )
            )
            protocol_rendered = True
        except Exception as exc:  # noqa: BLE001 — per-leg degradation
            failures.append(exc)
            logger.warning(
                "DefiLlama protocol TVL fetch failed for %s: %s", base, exc,
            )
    else:
        sections.append(
            f"Protocol TVL: no credible DefiLlama protocol lists '{base}' as "
            "its governance symbol (symbol matches must also clear a $1M TVL "
            "credibility floor) — protocol section omitted (this is coverage, "
            "not an error; many L1 payment/store-of-value assets have no "
            "protocol leg)."
        )

    # --- Chain leg (static map -> catalog tokenSymbol -> protocol's primary
    # chain). ---
    chain = None
    try:
        chain = _chain_for_base(base)
    except Exception:  # noqa: BLE001 — _chain_for_base already swallows its own
        chain = None
    if chain is None and protocol is not None:
        chains_list = [
            c for c in (protocol.get("chains") or []) if isinstance(c, str)
        ]
        chain = chains_list[0] if chains_list else None

    chain_rendered = False
    if chain is not None:
        try:
            series = _cached_json(
                f"chain_tvl_{_chain_cache_key(chain)}.json",
                lambda c=chain: _http_get_json(
                    f"{_BASE_URL}/v2/historicalChainTvl/{c}"
                ),
                ttl_days=1.0,
            )
            rows = _pit_window(
                series, "date", "tvl", as_of, look_back_days,
            )
            sections.append(_render_series(f"Chain TVL — {chain}", "TVL", rows))
            chain_rendered = True
            if not historical:
                try:
                    ranking = _chain_ranking(chain)
                except Exception as exc:  # noqa: BLE001 — advisory context
                    logger.warning(
                        "DefiLlama chain ranking unavailable for %s: %s", chain, exc,
                    )
                    ranking = ""
                if ranking:
                    sections.append(ranking.strip())
            else:
                sections.append(
                    "Chain ranking: omitted on historical runs (current-only "
                    "snapshot; serving it would leak future information)."
                )
        except Exception as exc:  # noqa: BLE001 — per-leg degradation
            failures.append(exc)
            logger.warning(
                "DefiLlama chain TVL fetch failed for %s: %s", chain, exc,
            )
    else:
        sections.append(
            f"Chain TVL: base '{base}' has no resolvable DefiLlama chain "
            "(static map miss and no catalog tokenSymbol match) — chain "
            "section omitted."
        )

    # --- Verdict: real coverage that rendered nothing -> surface the first
    # failure to the router (sentinel); genuine no-coverage -> typed error;
    # anything rendered -> in-band failure notes. ---
    if not protocol_rendered and not chain_rendered:
        if protocol is None and chain is None:
            raise NoMarketDataError(
                ticker,
                detail=(
                    f"no DefiLlama coverage for base symbol '{base}' (not a "
                    "known protocol governance token and not a resolvable "
                    "chain)"
                ),
            )
        if failures:
            raise failures[0]
    if failures:
        sections.append(
            "Unretrievable sections (honest failures — write 'data not "
            "available' for them, do not estimate): "
            + "; ".join(f"{type(f).__name__}: {f}" for f in failures)
        )

    header = (
        f"Crypto network fundamentals — {base} ({as_of_note}). Interpretation: "
        "rising TVL signals growing capital committed to the protocol/chain; "
        "TVL divergence from price flags multiple expansion/compression; "
        "window peaks anchor the drawdown context. If a section reports no "
        "rows, write 'data not available' for it — do not estimate.\n\n"
    )
    return header + "\n\n".join(sections)
