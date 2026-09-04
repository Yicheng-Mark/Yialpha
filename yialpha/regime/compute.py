"""Deterministic ``RegimeState`` assembly over the existing PIT data seams.

:func:`compute_regime_state` is the ONE entry point the graph runner calls
(crypto_perp runs only, flag ``regime_state``). It assembles the frozen
:class:`yialpha.regime.state.RegimeState` for a run using ONLY the
date-bounded / point-in-time seams the rest of the framework already owns —
imported at module top so tests monkeypatch THIS module's bindings:

* ``binance_klines_frame`` — perp last/index and Binance-spot daily candles
  (PIT-clamped by the pinned analysis date);
* ``get_binance_funding_rate`` — the funding-rate history CSV (date-bounded;
  the same seam :mod:`yialpha.ledger.outcome_compute` prices carry on);
* ``_fetch_open_interest`` / ``_fetch_lsr`` / ``_fetch_taker`` — the perp
  market bundle's date-bounded positioning components (30-day retention;
  reused, never re-implemented);
* ``_fetch_depth_bands`` — the bundle's LIVE order-book snapshot;
* ``get_YFin_history_cached`` — the underlying-equity history seam;
* ``classify_perp`` — the PIT instrument registry (``onboard_date`` /
  ``underlying_symbol``).

Historical-PIT contract: for a historical ``end_date`` (before today) ONLY
the point-in-time legs are consulted — klines, funding history, the
date-bounded positioning windows, the registry. The live-snapshot legs
(order-book depth) are structurally unavailable for a past instant and are
marked missing (``"depth_bands"`` in ``missing_inputs``), never fetched and
never leaked. Live runs use everything. The positioning endpoints retain
only 30 days, so a replay older than that degrades those legs to missing
too — honestly, per family.

Every component is fail-soft: a failed family leaves its fields ``None``
and appends one ``missing_inputs`` entry. When EVERY numeric/classifier
input for the class is missing the function returns ``None`` — the regime
is uncomputable and callers disclose "regime unavailable" instead of ever
minting a fake regime id over an all-empty record.

``confidence_components`` availability weights (frozen; values in [0, 1]):
``"price_history"`` (1.0 = trend + vol both present, 0.5 = one, 0.0 = none),
``"positioning"`` (pure crypto; share of the five positioning legs present),
``"depth"`` (1.0 iff the live book classified), ``"underlying"`` (stock
perps; 1.0 iff the equity trend computed), ``"calendar"`` (stock perps;
1.0 iff the registry produced a listing age).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from yialpha.dataflows.binance import binance_klines_frame, get_binance_funding_rate
from yialpha.dataflows.perp_bundle import (
    STATUS_OK,
    _fetch_depth_bands,
    _fetch_lsr,
    _fetch_open_interest,
    _fetch_taker,
)
from yialpha.dataflows.utils import is_historical_date
from yialpha.dataflows.y_finance import get_YFin_history_cached
from yialpha.instruments.registry import classify_perp
from yialpha.instruments.sessions import SESSION_CONTINUOUS
from yialpha.ledger.sqlite import utc_now_iso
from yialpha.regime.state import RegimeState, compute_regime_id
from yialpha.versions import REGIME_VERSION

logger = logging.getLogger(__name__)

#: Calendar days of kline lookback: SMA200 on a ~5-sessions-per-week series
#: needs ~280 calendar days; 420 covers it with margin for gaps.
_KLINE_LOOKBACK_DAYS = 420
#: Trailing funding window (settlement sum over 7 days).
_FUNDING_WINDOW_DAYS = 7
#: Realized-vol window: stdev of the last 20 daily returns (21 closes).
_VOL_WINDOW = 20
#: SMA windows for the trend classifier (frozen with REGIME_VERSION).
_SMA_FAST = 50
_SMA_SLOW = 200

#: Honest structural gaps this version (no PIT source wired yet): recorded
#: as missing inputs rather than fabricated values.
_STOCK_PERP_KNOWN_GAPS = ("earnings_calendar", "sector_index")


def _parse_funding_csv(text: str) -> list[tuple[datetime, float]]:
    """(UTC settlement time, rate) rows from the funding seam's CSV."""
    lines = [
        line for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    settlements: list[tuple[datetime, float]] = []
    for row in csv.DictReader(lines):
        time_raw, rate_raw = row.get("fundingTime"), row.get("fundingRate")
        if time_raw is None or rate_raw is None:
            continue
        try:
            moment = datetime.strptime(str(time_raw).strip(), "%Y-%m-%d %H:%M:%S")
            rate = float(rate_raw)
        except ValueError:
            continue
        settlements.append((moment.replace(tzinfo=UTC), rate))
    return sorted(settlements, key=lambda item: item[0])


def _window_start(end_date: str, days: int) -> str:
    """ISO date ``days`` before ``end_date`` (both ``YYYY-MM-DD``)."""
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return (end - timedelta(days=days)).strftime("%Y-%m-%d")


def _sma(values: list[float], window: int) -> float | None:
    """Simple moving average of the LAST ``window`` values, or None."""
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _trend_from_closes(closes: list[float]) -> str | None:
    """Frozen trend classifier: close vs SMA50/SMA200 (needs 200 closes)."""
    close = closes[-1]
    sma_fast = _sma(closes, _SMA_FAST)
    sma_slow = _sma(closes, _SMA_SLOW)
    if sma_fast is None or sma_slow is None:
        return None
    if close > sma_fast and close > sma_slow:
        return "up"
    if close < sma_fast and close < sma_slow:
        return "down"
    return "range"


def _realized_vol_pct(closes: list[float]) -> float | None:
    """Frozen vol classifier: stdev (ddof=1) of the last 20 daily returns, in
    percent (NOT annualized — see the state module docstring)."""
    if len(closes) < _VOL_WINDOW + 1:
        return None
    returns = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(len(closes) - _VOL_WINDOW, len(closes))
        if closes[i - 1] > 0
    ]
    if len(returns) < _VOL_WINDOW:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return var**0.5 * 100.0


def _is_edt(moment_utc: datetime) -> bool:
    """US daylight-saving window (2nd-Sunday-March .. 1st-Sunday-November).

    Boundaries taken at their UTC instants (02:00 local == 07:00/06:00 UTC).
    A hand-rolled rule on purpose: it keeps the ET session bucket
    deterministic on hosts without the ``tzdata`` package and documents
    itself as the approximation it is (see :func:`_et_session_state`).
    """
    year = moment_utc.year
    march1 = datetime(year, 3, 1)
    dst_start = march1 + timedelta(days=(6 - march1.weekday()) % 7 + 7)
    dst_start = dst_start.replace(hour=7)
    nov1 = datetime(year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    dst_end = dst_end.replace(hour=6)
    return dst_start <= moment_utc < dst_end


def _et_session_state(analysis_as_of: str) -> str:
    """DORMANT NYSE-equivalent ET session bucket (pre-v3 stock_perp rule).

    Klines machine evidence (2026-09-04) verified that Binance stock perps
    trade 24/7 — full US-market holidays and weekends all carried volume —
    so stock_perp now takes the continuous bucket in :func:`_session_state`
    and this ET machinery is dead code for it. It is deliberately RETAINED
    (not deleted): the hand-rolled DST rule (:func:`_is_edt`) keeps the
    bucket deterministic without ``tzdata``, and any future instrument class
    with a genuine session calendar may reuse it as-is.

    Semantics (unchanged from REGIME_VERSION v2): full timestamps are
    shifted to US Eastern and bucketed pre_market 04:00-09:30, regular
    09:30-16:00, post_market 16:00-20:00, closed otherwise (weekends always
    closed); a date-only value reads ``regular`` on weekdays and ``closed``
    on weekends; an unparseable value reads ``closed``.
    """
    raw = str(analysis_as_of).strip()
    date_only = "t" not in raw.lower()
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return "closed"
    if date_only:
        return "closed" if moment.weekday() >= 5 else "regular"
    moment_utc = (
        moment.astimezone(UTC) if moment.tzinfo else moment
    ).replace(tzinfo=None)
    local = moment_utc + timedelta(hours=-4 if _is_edt(moment_utc) else -5)
    if local.weekday() >= 5:
        return "closed"
    minutes = local.hour * 60 + local.minute
    if 240 <= minutes < 570:  # 04:00-09:30
        return "pre_market"
    if 570 <= minutes < 960:  # 09:30-16:00
        return "regular"
    if 960 <= minutes < 1200:  # 16:00-20:00
        return "post_market"
    return "closed"


def _session_state(analysis_as_of: str) -> str:  # noqa: ARG001
    """Continuous 24/7 session bucket for stock_perp (REGIME_VERSION v3).

    Klines machine evidence (2026-09-04, MUUSDT probe): Binance tokenized-
    stock perpetuals trade through every weekend and full US-market holiday
    with volume (zero-volume days = 0), so the V2.2 "date-only weekday =
    regular / weekend = closed" NYSE-equivalent approximation contradicted
    the venue's actual calendar. The bucket is now the continuous one
    (:data:`~yialpha.instruments.sessions.SESSION_CONTINUOUS`,
    ``"continuous_24_7"``) regardless of the as-of value — weekends and
    holidays are never ``closed``. ``analysis_as_of`` is intentionally
    unused: a 24/7 calendar has no as-of-dependent session. The dormant ET
    machinery survives in :func:`_et_session_state`.
    """
    return SESSION_CONTINUOUS


def _depth_regimes(depth: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """(liquidity, spread, contract-liquidity) regimes from the live book.

    Frozen thresholds (state module docstring): spread tight <= 2 bps /
    normal <= 10 / wide above; +-50bps band notional deep >= $2M /
    normal >= $500k / thin below; the combined liquidity summary per the
    documented rule.
    """
    spread_raw = depth.get("spread_bps")
    if not isinstance(spread_raw, (int, float)):
        return None, None, None
    spread = float(spread_raw)
    bands = depth.get("bands") or {}
    band50 = bands.get("50") or {}
    try:
        notional = float(band50.get("bid_notional") or 0.0) + float(
            band50.get("ask_notional") or 0.0
        )
    except (TypeError, ValueError):
        return None, None, None
    spread_regime = "tight" if spread <= 2.0 else "normal" if spread <= 10.0 else "wide"
    contract_liquidity = (
        "deep" if notional >= 2_000_000.0
        else "normal" if notional >= 500_000.0
        else "thin"
    )
    if spread_regime == "tight" and notional >= 500_000.0:
        combined = "ample"
    elif spread_regime == "wide" or notional < 500_000.0:
        combined = "constrained"
    else:
        combined = "normal"
    return combined, spread_regime, contract_liquidity


def _component_ok(component: Any) -> bool:
    """True when a bundle component dict carries ``status == ok``."""
    return isinstance(component, dict) and component.get("status") == STATUS_OK


def _market_stress(
    klass: str,
    *,
    funding_pct: float | None,
    spot_basis_bps: float | None,
    oi_pct: float | None,
    oi_chg_1d: float | None,
    index_basis_bps: float | None,
    overnight_gap_bps: float | None,
) -> bool | None:
    """Frozen stress composite: True iff ANY available trigger fires.

    Pure-crypto triggers: |7d funding| >= 1%; |spot-perp basis| >= 50 bps;
    OI >= 90th pct of trailing 30d with a +10% 1d build. Stock-perp
    triggers: |last-index basis| >= 50 bps; |overnight gap| >= 100 bps.
    ``None`` when no trigger input was computable at all — stress unknown,
    not stress absent.
    """
    triggers: list[bool] = []
    if klass == "pure_crypto_perp":
        if funding_pct is not None:
            triggers.append(abs(funding_pct) >= 0.01)
        if spot_basis_bps is not None:
            triggers.append(abs(spot_basis_bps) >= 50.0)
        if (
            oi_pct is not None
            and oi_pct >= 90.0
            and oi_chg_1d is not None
            and oi_chg_1d > 0.10
        ):
            triggers.append(True)
    elif klass == "stock_perp":
        if index_basis_bps is not None:
            triggers.append(abs(index_basis_bps) >= 50.0)
        if overnight_gap_bps is not None:
            triggers.append(abs(overnight_gap_bps) >= 100.0)
    if not triggers:
        return None
    return any(triggers)


def compute_regime_state(
    ticker: str,
    asset_type: str,
    instrument_class: str | None,
    analysis_as_of: str,
    *,
    end_date: str,
) -> RegimeState | None:
    """Assemble the run's :class:`RegimeState`, or ``None`` when uncomputable.

    Per-class field groups (frozen with REGIME_VERSION v3):
    ``pure_crypto_perp`` → COMMON + PURE-CRYPTO; ``stock_perp`` → COMMON +
    STOCK-PERP; ``unknown_perp`` (or an unset class on a perp run) → COMMON
    only, conservative. Non-perp asset types return ``None`` — the regime
    stage is perp-only this version. Every input family is fail-soft; the
    ALL-inputs-missing case returns ``None`` so no regime id is ever minted
    over an empty record.
    """
    if asset_type != "crypto_perp":
        return None
    klass = instrument_class or "unknown_perp"
    live = not is_historical_date(end_date)
    missing: set[str] = set()
    coverage: dict[str, float] = {}

    # --- COMMON: perp price history (trend / vol / overnight gap) -------------
    trend: str | None = None
    realized_vol: float | None = None
    overnight_gap_bps: float | None = None
    closes: list[float] = []
    try:
        frame = binance_klines_frame(
            ticker, _window_start(end_date, _KLINE_LOOKBACK_DAYS), end_date, "1d",
        )
        # Row-aligned drop: dropping each column independently would misalign
        # the two lists when one column holds an isolated NaN, pairing an
        # open with the WRONG prior close (a gap across trading days).
        oc = frame[["Open", "Close"]].dropna()
        closes = [float(v) for v in oc["Close"].tolist()]
        opens = [float(v) for v in oc["Open"].tolist()]
        trend = _trend_from_closes(closes)
        realized_vol = _realized_vol_pct(closes)
        if len(opens) >= 2 and closes[-2] > 0:
            overnight_gap_bps = (opens[-1] / closes[-2] - 1.0) * 1e4
    except Exception as exc:  # noqa: BLE001 — fail-soft per family
        logger.info("regime klines unavailable for %s: %s", ticker, exc)
    if trend is None and realized_vol is None:
        missing.add("perp_klines")
    coverage["price_history"] = (
        1.0 if trend is not None and realized_vol is not None
        else 0.5 if trend is not None or realized_vol is not None
        else 0.0
    )

    # --- COMMON: live depth legs (skipped on historical runs by policy) -------
    liquidity_regime: str | None = None
    spread_regime: str | None = None
    band_liquidity: str | None = None  # depth-notional regime (class gating below)
    if live:
        try:
            depth = _fetch_depth_bands(ticker)
            if _component_ok(depth):
                liquidity_regime, spread_regime, band_liquidity = _depth_regimes(
                    depth
                )
        except Exception as exc:  # noqa: BLE001
            logger.info("regime depth unavailable for %s: %s", ticker, exc)
        if liquidity_regime is None:
            missing.add("depth_bands")
    else:
        # PIT policy: a past instant has no reconstructable order book — the
        # live legs are marked missing, never fetched, never leaked.
        missing.add("depth_bands")
    coverage["depth"] = 1.0 if liquidity_regime is not None else 0.0
    # contract_liquidity is a STOCK-PERP field by the frozen vocabulary; the
    # underlying band reading stays available internally for the pure-crypto
    # cascade-risk composite.
    contract_liquidity = band_liquidity if klass == "stock_perp" else None

    # --- PURE-CRYPTO positioning family ----------------------------------------
    funding_pct: float | None = None
    oi_pct: float | None = None
    oi_chg_1d: float | None = None
    lsr_crowding: float | None = None
    taker_aggression: float | None = None
    spot_basis_bps: float | None = None
    if klass == "pure_crypto_perp":
        try:
            csv_text = get_binance_funding_rate(
                ticker, _window_start(end_date, _FUNDING_WINDOW_DAYS), end_date,
            )
            settlements = _parse_funding_csv(csv_text)
            if settlements:
                funding_pct = sum(rate for _moment, rate in settlements)
            else:
                missing.add("funding_history")
        except Exception as exc:  # noqa: BLE001
            logger.info("regime funding unavailable for %s: %s", ticker, exc)
            missing.add("funding_history")
        try:
            oi = _fetch_open_interest(ticker, end_date)
            if _component_ok(oi):
                oi_pct = float(oi["percentile"])  # type: ignore[index]
                chg = oi.get("chg_1d")
                oi_chg_1d = float(chg) if isinstance(chg, (int, float)) else None
            else:
                missing.add("oi_history")
        except Exception as exc:  # noqa: BLE001
            logger.info("regime OI unavailable for %s: %s", ticker, exc)
            missing.add("oi_history")
        try:
            lsr = _fetch_lsr(ticker, end_date)
            global_account = lsr.get("global_account") if isinstance(lsr, dict) else None
            if _component_ok(global_account):
                lsr_crowding = float(global_account["latest"])  # type: ignore[index]
            else:
                missing.add("lsr")
        except Exception as exc:  # noqa: BLE001
            logger.info("regime LSR unavailable for %s: %s", ticker, exc)
            missing.add("lsr")
        try:
            taker = _fetch_taker(ticker, end_date)
            if _component_ok(taker):
                taker_aggression = float(taker["latest"])  # type: ignore[index]
            else:
                missing.add("taker_flow")
        except Exception as exc:  # noqa: BLE001
            logger.info("regime taker unavailable for %s: %s", ticker, exc)
            missing.add("taker_flow")
        if closes and closes[-1] > 0:
            try:
                spot = binance_klines_frame(
                    ticker, _window_start(end_date, 8), end_date, "1d", "binance_spot",
                )
                spot_close = float(spot["Close"].dropna().iloc[-1])
                if spot_close > 0:
                    spot_basis_bps = (closes[-1] / spot_close - 1.0) * 1e4
                else:
                    missing.add("spot_klines")
            except Exception as exc:  # noqa: BLE001
                logger.info("regime spot basis unavailable for %s: %s", ticker, exc)
                missing.add("spot_klines")
        elif trend is not None:
            missing.add("spot_klines")
        positioning_legs = [
            funding_pct, oi_pct, lsr_crowding, taker_aggression, spot_basis_bps,
        ]
        coverage["positioning"] = (
            sum(1 for leg in positioning_legs if leg is not None)
            / len(positioning_legs)
        )

    # --- STOCK-PERP family -------------------------------------------------------
    underlying_trend: str | None = None
    index_basis_bps: float | None = None
    session_state: str | None = None
    listing_age_days: int | None = None
    if klass == "stock_perp":
        underlying_symbol: str | None = None
        try:
            record = classify_perp(ticker, analysis_as_of or None)
            underlying_symbol = record.underlying_symbol or None
            if record.onboard_date:
                onboard = datetime.strptime(record.onboard_date, "%Y-%m-%d").date()
                as_of_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                listing_age_days = (as_of_date - onboard).days
            else:
                missing.add("onboard_date")
        except Exception as exc:  # noqa: BLE001
            logger.info("regime registry read failed for %s: %s", ticker, exc)
            missing.add("onboard_date")
        coverage["calendar"] = 1.0 if listing_age_days is not None else 0.0
        session_state = _session_state(analysis_as_of)
        if underlying_symbol:
            try:
                end_exclusive = (
                    datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                history = get_YFin_history_cached(
                    underlying_symbol,
                    _window_start(end_date, _KLINE_LOOKBACK_DAYS),
                    end_exclusive,
                )
                equity_closes = [float(v) for v in history["Close"].dropna().tolist()]
                underlying_trend = _trend_from_closes(equity_closes)
            except Exception as exc:  # noqa: BLE001 — fail-soft per family
                logger.info(
                    "regime underlying history unavailable for %s: %s", ticker, exc
                )
        else:
            # No PIT underlying mapping (registry read failed, or the row
            # carried no underlying): the perp contract code is NOT a stock
            # symbol, so the equity history source must never receive it.
            # The leg is disclosed unavailable instead of silently queried
            # with the contract symbol — the regime degrades honestly.
            logger.info(
                "regime underlying mapping unavailable for %s — equity leg "
                "not queried (contract symbol withheld from the stock source)",
                ticker,
            )
            missing.add("underlying_symbol")
        if underlying_trend is None:
            missing.add("underlying_history")
        coverage["underlying"] = 1.0 if underlying_trend is not None else 0.0
        if closes and closes[-1] > 0:
            try:
                index_frame = binance_klines_frame(
                    ticker, _window_start(end_date, 35), end_date, "1d",
                    "binance_perp", "index",
                )
                index_close = float(index_frame["Close"].dropna().iloc[-1])
                if index_close > 0:
                    index_basis_bps = (closes[-1] / index_close - 1.0) * 1e4
                else:
                    missing.add("index_klines")
            except Exception as exc:  # noqa: BLE001
                logger.info("regime index leg unavailable for %s: %s", ticker, exc)
                missing.add("index_klines")
        elif trend is not None:
            missing.add("index_klines")
        # Honest gaps: no PIT earnings calendar or sector-index source is
        # wired yet — disclosed, never faked.
        missing.update(_STOCK_PERP_KNOWN_GAPS)

    market_stress = _market_stress(
        klass,
        funding_pct=funding_pct,
        spot_basis_bps=spot_basis_bps,
        oi_pct=oi_pct,
        oi_chg_1d=oi_chg_1d,
        index_basis_bps=index_basis_bps,
        overnight_gap_bps=overnight_gap_bps,
    )
    liquidation_cascade_risk: str | None = None
    if klass == "pure_crypto_perp" and market_stress is not None:
        thin_book = spread_regime == "wide" or band_liquidity == "thin"
        liquidation_cascade_risk = (
            "elevated" if market_stress and thin_book
            else "moderate" if market_stress
            else "low"
        )

    # --- All-inputs-missing guard: never mint an id over an empty record -------
    core_inputs: list[Any] = [trend, realized_vol]
    if klass == "pure_crypto_perp":
        core_inputs += [
            funding_pct, oi_pct, lsr_crowding, taker_aggression, spot_basis_bps,
        ]
    elif klass == "stock_perp":
        core_inputs += [
            underlying_trend, overnight_gap_bps, index_basis_bps, session_state,
        ]
    if all(value is None for value in core_inputs):
        return None

    state = RegimeState(
        trend_regime=trend,
        realized_vol_pct=realized_vol,
        liquidity_regime=liquidity_regime,
        spread_depth_regime=spread_regime,
        market_stress=market_stress,
        confidence_components=coverage,
        funding_pct=funding_pct,
        oi_pct=oi_pct,
        lsr_crowding=lsr_crowding,
        taker_aggression=taker_aggression,
        spot_perp_basis_bps=spot_basis_bps,
        liquidation_cascade_risk=liquidation_cascade_risk,
        underlying_trend=underlying_trend,
        session_state=session_state,
        overnight_gap_bps=overnight_gap_bps,
        index_mark_basis_bps=index_basis_bps,
        contract_liquidity=contract_liquidity,
        listing_age_days=listing_age_days,
        regime_version=REGIME_VERSION,
        analysis_as_of=str(analysis_as_of),
        computed_at=utc_now_iso(),
        missing_inputs=tuple(sorted(missing)),
    )
    return replace(state, regime_id=compute_regime_id(state, REGIME_VERSION))
