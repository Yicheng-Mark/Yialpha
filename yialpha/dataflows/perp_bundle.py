"""Deterministic perp market bundle — the core facts are FETCHED, not chatted.

The market analyst's perp tool belt is broad (klines / funding / OI / LSR /
taker / premium / basis / depth / vision), but WHICH of those the LLM
actually invokes is the model's choice — a run could reach its decision
without ever reading the mark price or the funding rate, and the quality
chain stays clean because nothing failed that was never attempted. This
module is the deterministic complement: ONE parallel prefetch assembles the
decision-critical numbers before the analyst's LLM turn, with every
component fail-soft and its per-field status disclosed in the rendered
block. The tools remain bound for drill-down; the bundle guarantees the
core is present.

Price-basis contract (audit 2026-08, PR3):

* ``last`` — technical trend, 1d/7d change, entry reference, ATR;
* ``mark`` — the price Binance liquidates against (liquidation distance
  claims anchor here);
* ``index`` — the settlement fair-value anchor (last−index premium).

Historical (replay) runs: the three kline legs are PIT-clamped and
memo-cached; the REST positioning family retains only 30 days and degrades
to ``unavailable`` with the retention reason; live-only snapshots (premium,
depth, ADL, trailing funding) are ``skipped_live_only``. Everything is
disclosed per component in the block footer — a missing enrichment never
masquerades as a calm market.

Core completeness wiring: when the last- or mark-kline leg fails, a
``get_binance_klines`` core sentinel is recorded so the four-tier quality
chain (PR1's per-method matrix) turns the ticket NO_TRADE — a perp decision
without its price book is not a GOOD-tier decision.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

from . import quality
from .binance import (
    KLINE_CLOSE_COLUMN,
    _http_get,
    binance_klines_frame,
    stock_perp_underlying,
)
from .utils import is_historical_date

logger = logging.getLogger(__name__)

#: Component statuses (disclosed verbatim in the rendered footer).
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_CAPABILITY_ABSENT = "capability_absent"
STATUS_SKIPPED_LIVE_ONLY = "skipped_live_only"

# /fapi/v1/depth limit must be a documented choice; 500 rows reaches far
# enough for the ±5% band on liquid books at weight 2.
_DEPTH_LIMIT = 500
#: Fixed ±bps bands for cumulative book depth (audit: ±0.2%~5%).
_BANDS_BPS = (20, 50, 100, 200, 500)
#: Reference market-order notionals for the slippage estimate.
_SLIPPAGE_NOTIONALS = (10_000.0, 50_000.0)
#: Kline lookback: 35 calendar days covers the 7d change + ATR14 warm-up.
_KLINE_LOOKBACK_DAYS = 35
#: OI window: the /futures/data family's retention ceiling (percentile base).
_OI_WINDOW_DAYS = 30
#: LSR/taker window: latest value + a 7d change is enough.
_LSR_WINDOW_DAYS = 9


def _fmt(value: float | None, digits: int = 4) -> str:
    """Adaptive-magnitude number formatting (PEPE 1e-5 … BTC 1e5)."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:,.2f}"
    if a >= 1:
        return f"{v:.{digits}f}"
    if a >= 1e-4:
        return f"{v:.8f}"
    return f"{v:.10f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _window(start_days: int, end_date: str) -> tuple[str, str, int, int]:
    """(start_iso, end_iso, start_ms, end_ms) for a trailing window."""
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    start_dt = end_dt - timedelta(days=start_days)
    return (
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
        int(start_dt.timestamp() * 1000),
        int((end_dt.timestamp() + 86399) * 1000),  # end-of-day inclusive
    )


# ---------------------------------------------------------------------------
# Component fetchers — each returns a dict carrying its own "status".
# ---------------------------------------------------------------------------


def _fetch_prices(symbol: str, end_date: str) -> dict[str, Any]:
    """last/mark/index daily closes + last-derived change/ATR, one leg per
    price basis. ``last`` and ``mark`` are CORE (their failure is recorded
    as a core sentinel by the assembler); ``index`` degrades fail-soft."""
    from .symbol_utils import normalize_symbol_for_venue

    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    start, end, _start_ms, _end_ms = _window(_KLINE_LOOKBACK_DAYS, end_date)
    out: dict[str, Any] = {"status": STATUS_OK}
    closes: dict[str, float] = {}
    frames: dict[str, Any] = {}

    for price_type in ("last", "mark", "index"):
        try:
            frame = binance_klines_frame(
                symbol, start, end, "1d", "binance_perp", price_type,
            )
            frames[price_type] = frame
            closes[price_type] = float(frame[KLINE_CLOSE_COLUMN].iloc[-1])
        except Exception as exc:  # noqa: BLE001 — per-leg fail-soft
            out[price_type] = {
                "status": STATUS_UNAVAILABLE,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            logger.info(
                "perp bundle %s kline leg unavailable for %s: %s",
                price_type, canonical, exc,
            )

    if "last" in frames:
        frame = frames["last"]
        col = frame[KLINE_CLOSE_COLUMN]
        out["last"] = {"status": STATUS_OK, "close": closes["last"]}
        n = len(col)
        if n >= 2:
            prev = float(col.iloc[-2])
            if prev > 0:
                out["last"]["chg_1d"] = closes["last"] / prev - 1.0
        if n >= 8:
            prev7 = float(col.iloc[-8])
            if prev7 > 0:
                out["last"]["chg_7d"] = closes["last"] / prev7 - 1.0
        # Same ATR engine the risk overlay uses (completed bars only — a
        # today-dated forming candle never distorts the volatility input).
        try:
            from yialpha.risk.atr_stop import latest_atr_from_frame

            price, atr = latest_atr_from_frame(frame)
            out["last"]["atr_14"] = atr
            out["last"]["atr_pct"] = atr / price if price > 0 else None
        except Exception as exc:  # noqa: BLE001 — ATR degrades, close stays
            out["last"]["atr_14"] = None
            out["last"]["atr_error"] = f"{type(exc).__name__}: {exc}"
        out["coverage"] = {
            "rows": int(n),
            "start": frame.index[0].strftime("%Y-%m-%d"),
            "end": frame.index[-1].strftime("%Y-%m-%d"),
        }
    if "mark" in closes and "last" in closes and closes["mark"] > 0:
        out["last_vs_mark_bps"] = (closes["last"] / closes["mark"] - 1.0) * 1e4
    if "index" in closes and "last" in closes and closes["index"] > 0:
        out["last_vs_index_bps"] = (closes["last"] / closes["index"] - 1.0) * 1e4
    if "mark" in closes:
        out["mark"] = {"status": STATUS_OK, "close": closes["mark"]}
    if "index" in closes:
        out["index"] = {"status": STATUS_OK, "close": closes["index"]}

    # A partial price book is still rendered (each leg's status is honest),
    # but the CORE verdict feeds the quality chain — see the assembler.
    if "last" not in closes or "mark" not in closes:
        out["status"] = STATUS_UNAVAILABLE
        out["core_complete"] = False
    else:
        out["core_complete"] = True
    return out


def _fetch_funding(symbol: str, end_date: str) -> dict[str, Any]:
    """Trailing-7d settlement sum + annualized carry (live runs only — a
    historical replay must not pay a vendor call per decision)."""
    if is_historical_date(end_date):
        return {"status": STATUS_SKIPPED_LIVE_ONLY}
    from .symbol_utils import normalize_symbol_for_venue

    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    start, end, start_ms, end_ms = _window(7, end_date)
    try:
        rows = _http_get(
            "/fapi/v1/fundingRate",
            {"symbol": canonical, "startTime": start_ms, "endTime": end_ms,
             "limit": 1000},
            symbol, canonical,
        )
        rates = [
            float(r["fundingRate"])
            for r in rows if isinstance(r, dict) and r.get("fundingRate") is not None
        ] if isinstance(rows, list) else []
        if not rates:
            return {"status": STATUS_UNAVAILABLE, "reason": "no settlements in window"}
        total = sum(rates)
        return {
            "status": STATUS_OK,
            "sum_7d": total,
            "annualized": total / 7.0 * 365.0,
            "settlements": len(rates),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _oi_series(symbol: str, end_date: str, window_days: int) -> list[tuple[int, float]]:
    """(ts_ms, sumOpenInterest) rows for the trailing window, PIT-clamped."""
    from .symbol_utils import normalize_symbol_for_venue

    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    _start, _end, start_ms, end_ms = _window(window_days, end_date)
    rows = _http_get(
        "/futures/data/openInterestHist",
        {"symbol": canonical, "period": "1d",
         "startTime": start_ms, "endTime": end_ms, "limit": 30},
        symbol, canonical,
    )
    out = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict) or r.get("timestamp") is None:
                continue
            try:
                out.append((int(r["timestamp"]), float(r["sumOpenInterest"])))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def _fetch_open_interest(symbol: str, end_date: str) -> dict[str, Any]:
    try:
        rows = _oi_series(symbol, end_date, _OI_WINDOW_DAYS)
    except Exception as exc:  # noqa: BLE001 — retention/auth errors degrade
        return {
            "status": STATUS_UNAVAILABLE,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not rows:
        return {"status": STATUS_UNAVAILABLE, "reason": "no rows in window"}
    values = [v for _ts, v in rows]
    latest = values[-1]
    out: dict[str, Any] = {
        "status": STATUS_OK,
        "latest": latest,
        "percentile": (
            sum(1 for v in values if v < latest) / len(values) * 100.0
        ),
    }
    if len(values) >= 2 and values[-2] > 0:
        out["chg_1d"] = latest / values[-2] - 1.0
    if len(values) >= 8 and values[-8] > 0:
        out["chg_7d"] = latest / values[-8] - 1.0
    return out


#: (component key, endpoint path, value key) for the three LSR vantage points.
_LSR_SERIES = (
    ("top_account", "/futures/data/topLongShortAccountRatio"),
    ("top_position", "/futures/data/topLongShortPositionRatio"),
    ("global_account", "/futures/data/globalLongShortAccountRatio"),
)


def _fetch_lsr(symbol: str, end_date: str) -> dict[str, Any]:
    """Three-vantage long/short ratio (大户账户 / 大户持仓 / 全账户) plus the
    cross-vantage spread — leveraged-crowd divergence the single global
    number hides."""
    from .symbol_utils import normalize_symbol_for_venue

    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    _start, _end, start_ms, end_ms = _window(_LSR_WINDOW_DAYS, end_date)
    out: dict[str, Any] = {"status": STATUS_OK}
    latest_values: list[float] = []
    for key, path in _LSR_SERIES:
        try:
            rows = _http_get(
                path,
                {"symbol": canonical, "period": "1d",
                 "startTime": start_ms, "endTime": end_ms, "limit": 30},
                symbol, canonical,
            )
            values = [
                float(r["longShortRatio"])
                for r in rows if isinstance(r, dict) and r.get("longShortRatio")
            ] if isinstance(rows, list) else []
            if not values:
                out[key] = {"status": STATUS_UNAVAILABLE, "reason": "no rows"}
                continue
            entry: dict[str, Any] = {"status": STATUS_OK, "latest": values[-1]}
            if len(values) >= 8:
                entry["chg_7d"] = values[-1] - values[-8]
            out[key] = entry
            latest_values.append(values[-1])
        except Exception as exc:  # noqa: BLE001
            out[key] = {"status": STATUS_UNAVAILABLE,
                        "reason": f"{type(exc).__name__}: {exc}"}
    if len(latest_values) >= 2:
        out["cross_vantage_spread"] = max(latest_values) - min(latest_values)
    if all(out.get(k, {}).get("status") != STATUS_OK for k, _p in _LSR_SERIES):
        out["status"] = STATUS_UNAVAILABLE
    return out


def _fetch_taker(symbol: str, end_date: str) -> dict[str, Any]:
    """Latest daily taker buy/sell ratio + its 7d mean (aggression flow)."""
    from .symbol_utils import normalize_symbol_for_venue

    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    _start, _end, start_ms, end_ms = _window(_LSR_WINDOW_DAYS, end_date)
    try:
        rows = _http_get(
            "/futures/data/takerlongshortRatio",
            {"symbol": canonical, "period": "1d",
             "startTime": start_ms, "endTime": end_ms, "limit": 30},
            symbol, canonical,
        )
        values = [
            float(r["buySellRatio"])
            for r in rows if isinstance(r, dict) and r.get("buySellRatio")
        ] if isinstance(rows, list) else []
        if not values:
            return {"status": STATUS_UNAVAILABLE, "reason": "no rows in window"}
        return {
            "status": STATUS_OK,
            "latest": values[-1],
            "mean_7d": (
                sum(values[-8:]) / len(values[-8:]) if len(values) >= 8 else None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_UNAVAILABLE, "reason": f"{type(exc).__name__}: {exc}"}


def _fetch_premium_snapshot(symbol: str) -> dict[str, Any]:
    """Live mark/index snapshot (``/fapi/v1/premiumIndex``): the rate in
    effect for the NEXT settlement + the mark displacement vs index."""
    from .symbol_utils import normalize_symbol_for_venue

    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    try:
        data = _http_get(
            "/fapi/v1/premiumIndex", {"symbol": canonical}, symbol, canonical,
        )
        if not isinstance(data, dict) or data.get("markPrice") is None:
            raise ValueError("premiumIndex returned no markPrice")  # noqa: TRY301
        mark = float(data["markPrice"])
        index = float(data.get("indexPrice") or 0.0) or None
        nft = data.get("nextFundingTime")
        rate = data.get("lastFundingRate")
        return {
            "status": STATUS_OK,
            "mark": mark,
            "index": index,
            "mark_vs_index_bps": (
                (mark / index - 1.0) * 1e4 if index else None
            ),
            "next_funding_rate": float(rate) if rate is not None else None,
            "next_funding_time_utc": (
                datetime.fromtimestamp(int(nft) / 1000, tz=UTC)
                .strftime("%Y-%m-%d %H:%M") if isinstance(nft, (int, float)) and nft > 0
                else None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_UNAVAILABLE, "reason": f"{type(exc).__name__}: {exc}"}


def _levels(rows: list) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lv in rows:
        if not isinstance(lv, list) or len(lv) < 2:
            continue
        try:
            out.append((float(lv[0]), float(lv[1])))
        except (TypeError, ValueError):
            continue
    return out


def _band_notional(levels: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Cumulative resting notional of levels priced within [lo, hi]."""
    return sum(price * qty for price, qty in levels if lo <= price <= hi)


def _walk_impact_bps(
    levels: list[tuple[float, float]], mid: float, notional: float,
) -> float | None:
    """VWAP-vs-mid impact (bps) of filling ``notional`` against ``levels``.

    ``levels`` must be sorted best-first (bids descending, asks ascending).
    Returns ``None`` when the visible book cannot absorb the full reference
    order — a thin book has no honest VWAP, and a partial-fill average would
    understate the very slippage it exists to measure."""
    cost = 0.0
    qty = 0.0
    filled = False
    for price, level_qty in levels:
        take_cost = price * level_qty
        if cost + take_cost >= notional:
            if price > 0:
                take_qty = (notional - cost) / price
                cost += take_qty * price
                qty += take_qty
                filled = True
            break
        cost += take_cost
        qty += level_qty
    if not filled or qty <= 0 or mid <= 0:
        return None
    vwap = cost / qty
    return (vwap / mid - 1.0) * 1e4


def _fetch_depth_bands(symbol: str) -> dict[str, Any]:
    """Fixed ±bps cumulative book depth, per-band imbalance and reference
    market-order slippage (live snapshot)."""
    from .symbol_utils import normalize_symbol_for_venue

    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    try:
        data = _http_get(
            "/fapi/v1/depth", {"symbol": canonical, "limit": _DEPTH_LIMIT},
            symbol, canonical,
        )
        bids = _levels(data.get("bids") or []) if isinstance(data, dict) else []
        asks = _levels(data.get("asks") or []) if isinstance(data, dict) else []
        if not bids or not asks:
            raise ValueError("depth returned no levels")  # noqa: TRY301
        bids.sort(key=lambda lv: lv[0], reverse=True)  # best-first
        asks.sort(key=lambda lv: lv[0])
        mid = (bids[0][0] + asks[0][0]) / 2.0
        out: dict[str, Any] = {
            "status": STATUS_OK,
            "mid": mid,
            "spread_bps": (asks[0][0] / bids[0][0] - 1.0) * 1e4,
            "bands": {},
        }
        for band in _BANDS_BPS:
            edge = band / 1e4
            bid_n = _band_notional(bids, mid * (1.0 - edge), mid)
            ask_n = _band_notional(asks, mid, mid * (1.0 + edge))
            total = bid_n + ask_n
            out["bands"][str(band)] = {
                "bid_notional": bid_n,
                "ask_notional": ask_n,
                "imbalance": (bid_n - ask_n) / total if total > 0 else None,
            }
        out["slippage_bps"] = {}
        for notional in _SLIPPAGE_NOTIONALS:
            out["slippage_bps"][f"buy_{int(notional)}"] = _walk_impact_bps(
                asks, mid, notional,
            )
            out["slippage_bps"][f"sell_{int(notional)}"] = _walk_impact_bps(
                bids, mid, notional,
            )
        return out
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_UNAVAILABLE, "reason": f"{type(exc).__name__}: {exc}"}


def _fetch_adl(symbol: str) -> dict[str, Any]:
    """Position ADL quantile (0..3, higher = closer to auto-deleveraging).

    ``/fapi/v1/adlQuantile`` is a SIGNED USER-DATA endpoint — an unsigned
    request can never succeed, so the fetch is signed via
    :func:`yialpha.dataflows.binance_brackets.signed_fapi_get` when
    BINANCE_API_KEY/SECRET are configured and degrades IMMEDIATELY to
    unavailable (actionable reason, no doomed request) without keys. It is
    enrichment, never a veto."""
    from .binance_brackets import signed_fapi_get
    from .symbol_utils import normalize_symbol_for_venue

    canonical = normalize_symbol_for_venue(symbol, "binance_perp")
    try:
        data = signed_fapi_get("/fapi/v1/adlQuantile", {}, symbol, canonical)
        entry = data[0] if isinstance(data, list) and data else {}
        raw = entry.get("adlQuantile") if isinstance(entry, dict) else None
        if not isinstance(raw, dict):
            raise ValueError("adlQuantile returned no quantiles")  # noqa: TRY301
        out: dict[str, Any] = {"status": STATUS_OK}
        for side in ("LONG", "SHORT"):
            try:
                out[side.lower()] = int(raw[side])
            except (KeyError, TypeError, ValueError):
                out[side.lower()] = None
        return out
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_UNAVAILABLE, "reason": f"{type(exc).__name__}: {exc}"}


def _fetch_spot_basis(symbol: str, end_date: str, perp_close: float | None) -> dict[str, Any]:
    """Perp-vs-spot premium on the SAME window (pure crypto only).

    A tokenized-stock perp (TRADIFI/EQUITY underlying) has no Binance spot
    leg — that is a structural capability_absent, not a fetch failure, and
    must never read as degradation."""
    underlying = stock_perp_underlying(symbol)
    if underlying:
        return {
            "status": STATUS_CAPABILITY_ABSENT,
            "reason": f"tokenized-stock perp ({underlying} underlying) has no Binance spot leg",
        }
    if perp_close is None or perp_close <= 0:
        return {"status": STATUS_UNAVAILABLE, "reason": "no perp close to compare"}
    _start, end, _sms, _ems = _window(8, end_date)
    try:
        spot = binance_klines_frame(symbol, _start, end, "1d", "binance_spot", "last")
        spot_close = float(spot[KLINE_CLOSE_COLUMN].iloc[-1])
        if spot_close <= 0:
            raise ValueError("spot close non-positive")  # noqa: TRY301
        return {
            "status": STATUS_OK,
            "spot_close": spot_close,
            "basis_bps": (perp_close / spot_close - 1.0) * 1e4,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": STATUS_UNAVAILABLE, "reason": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Assembler + renderer
# ---------------------------------------------------------------------------


def fetch_perp_market_bundle(symbol: str, end_date: str) -> dict[str, Any]:
    """Assemble the perp market bundle in one parallel pass.

    Every component is fail-soft; per-component status rides the returned
    dict. A failure of the CORE price legs (last/mark klines) additionally
    records a ``get_binance_klines`` core sentinel so the quality chain
    classifies the run DEGRADED_CRITICAL (ticket NO_TRADE) — the bundle is
    the guarantee that a perp decision without its price book cannot pass
    as a clean run.
    """
    live = not is_historical_date(end_date)
    jobs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("prices", lambda: _fetch_prices(symbol, end_date)),
        ("open_interest", lambda: _fetch_open_interest(symbol, end_date)),
        ("long_short", lambda: _fetch_lsr(symbol, end_date)),
        ("taker", lambda: _fetch_taker(symbol, end_date)),
        ("spot_basis", lambda: _fetch_spot_basis(
            symbol, end_date,
            _last_close_hint(symbol, end_date),
        )),
    ]
    if live:
        jobs += [
            ("funding", lambda: _fetch_funding(symbol, end_date)),
            ("premium_snapshot", lambda: _fetch_premium_snapshot(symbol)),
            ("depth_bands", lambda: _fetch_depth_bands(symbol)),
            ("adl", lambda: _fetch_adl(symbol)),
        ]

    bundle: dict[str, Any] = {
        "symbol": symbol,
        "as_of": end_date,
        "live_run": live,
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(fn) for key, fn in jobs}
        for key, future in futures.items():
            try:
                bundle[key] = future.result(timeout=60)
            except Exception as exc:  # noqa: BLE001 — component isolation
                bundle[key] = {
                    "status": STATUS_UNAVAILABLE,
                    "reason": f"{type(exc).__name__}: {exc}",
                }

    prices = bundle.get("prices") or {}
    if prices.get("core_complete") is not True:
        quality.record_sentinel(
            "get_binance_klines",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"{symbol}: perp bundle core price legs incomplete: "
            f"last={'ok' if 'last' in prices else 'missing'}, "
            f"mark={'ok' if prices.get('mark') else 'missing'}",
        )
    # Auxiliary components enter the ledger too (as optional-unavailable —
    # auxiliary by the per-method matrix, never a veto): before this, a run
    # whose bundle enrichment failed ACROSS THE BOARD still classified GOOD
    # because nothing that "failed" had ever been attempted through a router
    # method. Structural states never record: capability_absent is a missing
    # leg by design, skipped_live_only is PIT policy, not an outage.
    for key, method in _AUX_LEDGER_METHODS.items():
        comp = bundle.get(key)
        if isinstance(comp, dict) and comp.get("status") == STATUS_UNAVAILABLE:
            quality.record_sentinel(
                method,
                quality.KIND_OPTIONAL_UNAVAILABLE,
                f"{symbol}: perp bundle {key} unavailable: "
                f"{str(comp.get('reason', ''))[:200]}",
            )
    adl = bundle.get("adl")
    if isinstance(adl, dict) and adl.get("status") == STATUS_UNAVAILABLE:
        # ADL is a signed USER-DATA endpoint with no router method; the
        # bundle-scoped name records it as evidence and classifies auxiliary
        # (unknown method = fail-open auxiliary by contract).
        quality.record_sentinel(
            "perp_bundle_adl",
            quality.KIND_OPTIONAL_UNAVAILABLE,
            f"{symbol}: {str(adl.get('reason', ''))[:200]}",
        )
    return bundle


#: Bundle component key -> router method name for the quality ledger. The
#: mapping keeps the ledger's method names the same ones the bound tools
#: would record, so a bundle outage and a tool outage read identically.
_AUX_LEDGER_METHODS = {
    "open_interest": "get_binance_open_interest",
    "long_short": "get_binance_long_short_ratio",
    "taker": "get_binance_taker_buy_sell",
    "funding": "get_binance_funding_rate",
    "premium_snapshot": "get_binance_premium_index",
    "depth_bands": "get_binance_depth_snapshot",
    "spot_basis": "get_binance_spot_perp_basis",
}


def _last_close_hint(symbol: str, end_date: str) -> float | None:
    """Best-effort last close for the spot-basis comparison (no extra fetch:
    reuses the memoized kline path; None feeds an honest unavailable)."""
    try:
        start, _end, _sms, _ems = _window(8, end_date)
        frame = binance_klines_frame(symbol, start, end_date, "1d", "binance_perp", "last")
        return float(frame[KLINE_CLOSE_COLUMN].iloc[-1])
    except Exception:  # noqa: BLE001 — hint only
        return None


def _component_footer(
    bundle: dict[str, Any], keys: tuple[str, ...] | None = None
) -> list[str]:
    """One line per non-ok component — a missing enrichment never reads as
    a calm market.

    ``keys`` restricts the footer to the named top-level components (the
    V2.3 positioning split renders only its own components' statuses in each
    half); ``None`` covers every component (the unsplit block's footer).
    """
    notes: list[str] = []
    for key, comp in bundle.items():
        if keys is not None and key not in keys:
            continue
        if not isinstance(comp, dict) or comp.get("status") == STATUS_OK:
            continue
        status = comp.get("status", STATUS_UNAVAILABLE)
        reason = comp.get("reason", "")
        notes.append(f"{key}: {status}" + (f" ({reason})" if reason else ""))
        # Nested vantage points (LSR) get their own note.
        for sub_key, sub in comp.items():
            if isinstance(sub, dict) and sub.get("status") not in (None, STATUS_OK):
                notes.append(
                    f"{key}.{sub_key}: {sub.get('status')}"
                    + (f" ({sub.get('reason', '')})" if sub.get("reason") else "")
                )
    return notes


#: Top-level bundle components rendered by the PRICE half of the split (the
#: market analyst's block under positioning_split): the three kline price
#: bases and their derived change/ATR/coverage lines.
_PRICE_SECTION_KEYS = ("prices",)

#: Top-level bundle components rendered by the POSITIONING half of the split
#: (the positioning analyst's block): carry, positioning, flow, book and
#: basis — everything that describes WHO is in the trade, not WHERE the
#: price is.
_POSITIONING_SECTION_KEYS = (
    "funding",
    "open_interest",
    "long_short",
    "taker",
    "spot_basis",
    "depth_bands",
    "adl",
    "premium_snapshot",
)

#: Both halves together — the unsplit market bundle's full component set.
_ALL_SECTION_KEYS = _PRICE_SECTION_KEYS + _POSITIONING_SECTION_KEYS


def _price_lines(bundle: dict[str, Any]) -> list[str]:
    """Price-basis section lines (last/mark/index + bases + ATR + coverage)."""
    prices = bundle.get("prices") or {}
    last = prices.get("last") or {}
    mark = prices.get("mark") or {}
    index = prices.get("index") or {}
    lines: list[str] = [
        "- **Price bases**: last "
        f"{_fmt(last.get('close'))}"
        + (f" (1d {_pct(last.get('chg_1d'))}, 7d {_pct(last.get('chg_7d'))})" if last else " (unavailable)")
        + f"; mark {_fmt(mark.get('close'))}"
        + f"; index {_fmt(index.get('close'))}",
    ]
    basis_parts = []
    if prices.get("last_vs_mark_bps") is not None:
        basis_parts.append(f"last−mark {prices['last_vs_mark_bps']:+.1f} bps")
    if prices.get("last_vs_index_bps") is not None:
        basis_parts.append(f"last−index {prices['last_vs_index_bps']:+.1f} bps")
    if basis_parts:
        lines.append(
            "- **Mark/last basis**: " + ", ".join(basis_parts)
            + " (liquidation judges on MARK; index is the fair-value anchor)"
        )
    if last.get("atr_14") is not None:
        lines.append(
            f"- **ATR14**: {_fmt(last['atr_14'])} "
            f"({_pct(last.get('atr_pct'))} of price; completed bars only)"
        )
    cov = prices.get("coverage")
    if cov:
        lines.append(
            f"- **Kline coverage**: {cov['rows']} daily rows "
            f"{cov['start']} → {cov['end']}"
        )
    return lines


def _positioning_lines(bundle: dict[str, Any]) -> list[str]:
    """Positioning section lines (funding/OI/LSR/taker/basis/depth/ADL)."""
    lines: list[str] = []
    funding = bundle.get("funding") or {}
    if funding.get("status") == STATUS_OK:
        lines.append(
            f"- **Funding (7d)**: {funding['sum_7d']:+.3%} net "
            f"(annualized {funding['annualized']:+.2%}, "
            f"{funding['settlements']} settlements)"
        )
    premium = bundle.get("premium_snapshot") or {}
    if premium.get("status") == STATUS_OK:
        mvi = premium.get("mark_vs_index_bps")
        nfr = premium.get("next_funding_rate")
        lines.append(
            "- **Premium snapshot (live)**: mark vs index "
            + (f"{mvi:+.1f} bps" if mvi is not None else "n/a")
            + "; next funding "
            + (f"{nfr:+.4%} at {premium.get('next_funding_time_utc')} UTC"
               if nfr is not None else "n/a")
        )

    oi = bundle.get("open_interest") or {}
    if oi.get("status") == STATUS_OK:
        lines.append(
            f"- **Open interest**: {_fmt(oi['latest'])} contracts "
            f"(1d {_pct(oi.get('chg_1d'))}, 7d {_pct(oi.get('chg_7d'))}, "
            f"{oi['percentile']:.0f}th pct of trailing {_OI_WINDOW_DAYS}d)"
        )
    lsr = bundle.get("long_short") or {}
    lsr_parts = []
    for key, label in (
        ("top_account", "topAcct"), ("top_position", "topPos"),
        ("global_account", "globalAcct"),
    ):
        comp = lsr.get(key) or {}
        if comp.get("status") == STATUS_OK:
            lsr_parts.append(f"{label} {comp['latest']:.2f}")
    if lsr_parts:
        spread = lsr.get("cross_vantage_spread")
        lines.append(
            "- **Long/short (acct ratio)**: " + ", ".join(lsr_parts)
            + (f"; cross-vantage spread {spread:.2f}" if spread is not None else "")
            + " (>1 = that crowd is long-heavy)"
        )
    taker = bundle.get("taker") or {}
    if taker.get("status") == STATUS_OK:
        mean7 = taker.get("mean_7d")
        lines.append(
            f"- **Taker flow**: buy/sell {taker['latest']:.2f} latest"
            + (f" (7d mean {mean7:.2f})" if mean7 is not None else "")
            + " (>1 = aggressive buying)"
        )

    depth = bundle.get("depth_bands") or {}
    if depth.get("status") == STATUS_OK:
        band_parts = []
        for band in _BANDS_BPS:
            info = (depth.get("bands") or {}).get(str(band)) or {}
            if info.get("imbalance") is not None:
                band_parts.append(
                    f"±{band}bps {_fmt(info['bid_notional'], 0)}/"
                    f"{_fmt(info['ask_notional'], 0)} ({info['imbalance']:+.0%})"
                )
        slip = depth.get("slippage_bps") or {}
        slip_parts = []
        for notional in _SLIPPAGE_NOTIONALS:
            buy = slip.get(f"buy_{int(notional)}")
            sell = slip.get(f"sell_{int(notional)}")
            if buy is not None and sell is not None:
                slip_parts.append(
                    f"@{notional/1000:.0f}k buy {buy:+.1f}/sell {sell:+.1f} bps"
                )
        if band_parts:
            lines.append(
                f"- **Depth (bid/ask notional, live)**: {'; '.join(band_parts)}"
            )
        if slip_parts:
            lines.append("- **Est. slippage (VWAP vs mid)**: " + "; ".join(slip_parts))

    adl = bundle.get("adl") or {}
    if adl.get("status") == STATUS_OK:
        lines.append(
            f"- **ADL quantile**: long {adl.get('long')} / short {adl.get('short')}"
            " (0..3; higher = closer to auto-deleveraging)"
        )

    basis = bundle.get("spot_basis") or {}
    if basis.get("status") == STATUS_OK:
        lines.append(
            f"- **Spot-perp basis**: {basis['basis_bps']:+.1f} bps "
            f"(spot {_fmt(basis.get('spot_close'))})"
        )
    elif basis.get("status") == STATUS_CAPABILITY_ABSENT:
        lines.append(
            f"- **Spot-perp basis**: capability absent — {basis.get('reason')}"
        )
    return lines


def _footer_lines(
    bundle: dict[str, Any], keys: tuple[str, ...] | None
) -> list[str]:
    """Component-availability + historical-policy footer lines for one half."""
    lines: list[str] = []
    footer = _component_footer(bundle, keys)
    if footer:
        lines.append(
            "- **Component availability**: " + "; ".join(footer)
        )
    if not bundle.get("live_run", True):
        lines.append(
            "- (Historical run: live-only components — funding/premium/"
            "depth/ADL — are skipped by PIT policy; REST positioning "
            "retains 30 days.)"
        )
    return lines


def render_perp_bundle_block(
    bundle: dict[str, Any], *, positioning_split: bool = False
) -> str:
    """Render the bundle as a compact advisory markdown block.

    Deterministic numbers only — no model prose. Prices always carry their
    basis (last/mark/index) and the block footer discloses every component
    that is not ok.

    ``positioning_split=True`` (V2.3, flag-gated at the call site) renders
    ONLY the price half: the positioning sections (funding / OI / LSR /
    taker / depth / ADL / spot-perp basis) are carried by the Positioning
    analyst's own block (:func:`render_positioning_block`) so no data is
    silently lost by the split — the union of the two halves is exactly the
    unsplit section set (pinned by tests). Default False renders the full
    block byte-identically to the pre-split renderer.
    """
    symbol = bundle.get("symbol", "?")
    as_of = bundle.get("as_of", "?")
    lines: list[str] = [
        f"### Perp Market Bundle — {symbol} (deterministic prefetch, as of {as_of})",
    ]
    lines += _price_lines(bundle)
    if positioning_split:
        lines.append(
            "- (Positioning split: funding / open interest / long-short / "
            "taker / depth / ADL / spot-perp basis are rendered by the "
            "Positioning analyst's block, not here.)"
        )
        return "\n".join(
            lines + _footer_lines(bundle, _PRICE_SECTION_KEYS)
        )
    lines += _positioning_lines(bundle)
    return "\n".join(lines + _footer_lines(bundle, None))


def render_positioning_block(bundle: dict[str, Any]) -> str:
    """Render the POSITIONING half of the perp market bundle (V2.3).

    Funding carry, open interest, the three-vantage long/short ratios, taker
    flow, spot-perp basis, depth bands and ADL — the fabric of WHO is in the
    trade. No price-trend/kline sections (those stay with the market
    analyst). The footer discloses the positioning components' own
    availability statuses only.
    """
    symbol = bundle.get("symbol", "?")
    as_of = bundle.get("as_of", "?")
    lines: list[str] = [
        f"### Perp Positioning Bundle — {symbol} "
        f"(deterministic prefetch, as of {as_of})",
    ]
    lines += _positioning_lines(bundle)
    return "\n".join(lines + _footer_lines(bundle, _POSITIONING_SECTION_KEYS))
