"""USDT/USD quote FX for the V2.1 Fair Value Bridge (live-only, fail-soft).

Stock perps are quoted in USDT while their equity underlying is priced in
USD, so converting a USD price target into a USDT contract target needs a
USD-per-USDT rate. There is no official USDT/USD fixing, so the frozen RFC
picks the most liquid proxy available on an endpoint this repo already
serves: the Binance SPOT ``USDCUSDT`` daily close (USDC ≈ USD at par,
1:1 redeemable), INVERTED — ``rate = 1.0 / close`` is USD per 1 USDT.

Contract decisions (frozen 2026-09-03, worklog v2-perp-progress.md #5):

* **Live-only.** No point-in-time archive of the pair is kept, so an
  ``as_of`` other than today UTC returns None — and defaulting the rate to
  1.0 to keep the bridge alive is FORBIDDEN: a hidden peg assumption is
  worse than an honest UNAVAILABLE disclosure.
* **Fail-soft.** Any fetch/parse failure, non-positive/non-finite close, or
  a close outside the depeg band [0.95, 1.05] yields None; this module
  never raises into a run (the bridge renders a warning line instead).
* **Depeg guard.** A USDC print far from par means the FEED is broken (or
  USDC itself depegged) — it must not silently drive target conversion, so
  the band check happens before any rate is produced.

Call sites are gated by the ``stock_perp_fair_value`` config flag
(production default on, held off in tests) — this module never reads the
flag itself; it is wired by the overlay in a later batch.

The raw payload goes through :func:`yialpha.dataflows.disk_cache.cached_or_fetch`
(vendor dir ``quote_fx``, one file per UTC day, 1-hour TTL) reusing the
existing Binance SPOT kline seam — no new HTTP code lives here.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from yialpha.dataflows.binance import KLINE_CLOSE_COLUMN, binance_klines_frame
from yialpha.dataflows.disk_cache import cached_or_fetch, vendor_cache_dir

logger = logging.getLogger(__name__)

#: Provenance label carried on every result (rendered into disclosures).
QUOTE_FX_SOURCE = "binance_spot:USDCUSDT_inverse"

#: Spot pair whose close is inverted into USD-per-USDT.
_FX_PAIR = "USDCUSDT"

#: Calendar days of daily closes requested — enough that one or two skipped
#: candles never leave the window empty, small enough to stay "latest".
_LOOK_BACK_DAYS = 3

#: Sanity band on the RAW USDCUSDT close (USDT per 1 USDC). Outside it the
#: feed (or the peg) is broken: refuse rather than convert through it.
_DEPEG_LOW, _DEPEG_HIGH = 0.95, 1.05

#: Cache freshness for a live FX print: one hour.
_CACHE_TTL_DAYS = 1.0 / 24.0

#: Disk-cache vendor / directory name under ``data_cache_dir``.
_VENDOR = "quote_fx"

#: Replayability class of this feed — rendered downstream, pinned by tests.
LIVE_ONLY = "LIVE_ONLY"


@dataclass(frozen=True)
class QuoteFxResult:
    """One USDT/USD observation: ``rate`` is USD per 1 USDT.

    ``available_at`` is the UTC date of the daily close the rate derives
    from (today's forming candle carries the freshest print); ``fetched_at``
    is the ISO-8601 UTC time the payload was actually fetched — it travels
    with the cached bytes so a cache hit never masquerades as a fresh
    fetch. ``replayability`` is always ``LIVE_ONLY``: there is no PIT
    archive, so backtests must expect this to be None.
    """

    rate: float
    available_at: str
    fetched_at: str
    source: str = QUOTE_FX_SOURCE
    replayability: str = LIVE_ONLY


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fetch_payload(end_date: str) -> bytes:
    """Fetch the latest USDCUSDT daily close and wrap it as cache bytes.

    Runs inside :func:`cached_or_fetch`'s ``fetch`` slot, reusing the
    existing Binance SPOT kline seam (``binance_klines_frame`` with
    ``venue="binance_spot"`` — the same /api/v3/klines path the spot OHLCV
    tools use, proxy/throttle/host-mirror config included). The wrapper
    stores the chosen close plus its dates so a cache hit can rebuild the
    full :class:`QuoteFxResult` without touching the frame again.
    """
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
        - timedelta(days=_LOOK_BACK_DAYS)
    ).strftime("%Y-%m-%d")
    frame = binance_klines_frame(
        _FX_PAIR, start_date, end_date, interval="1d", venue="binance_spot"
    )
    # The frame is Date-indexed and sorted, so the last row is the latest
    # print (today's forming candle in live mode).
    available_at = frame.index[-1].strftime("%Y-%m-%d")
    close = float(frame[KLINE_CLOSE_COLUMN].iloc[-1])
    return json.dumps(
        {"fetched_at": _utc_now_iso(), "available_at": available_at, "close": close}
    ).encode("utf-8")


def fetch_usdt_usd() -> QuoteFxResult | None:
    """Latest USDT/USD rate (LIVE ONLY); None on any failure or depeg.

    ``rate = 1.0 / <latest Binance SPOT USDCUSDT daily close over the last
    ~3 days>``. The raw payload fetch is wrapped in the shared disk cache
    (one JSON file per UTC day under the ``quote_fx`` vendor dir, TTL one
    hour, fail-open so an unavailable feed degrades this call to None
    instead of aborting the run). Any exception, a non-positive or
    non-finite close, or a close outside [0.95, 1.05] returns None — never
    raises, never defaults the rate to 1.0.
    """
    end_date = datetime.now(UTC).strftime("%Y-%m-%d")
    try:
        payload = cached_or_fetch(
            vendor_cache_dir(_VENDOR),
            f"usdcusdt_{end_date}.json",
            lambda: _fetch_payload(end_date),
            ttl_days=_CACHE_TTL_DAYS,
            vendor=_VENDOR,
            fail_open=True,
        )
    except Exception:  # cache/parse machinery must never break a run
        logger.warning("quote_fx: USDCUSDT payload fetch failed; fx unavailable")
        return None
    if payload is None:
        return None

    try:
        wrapper = json.loads(payload)
        close = float(wrapper["close"])
        available_at = str(wrapper["available_at"])
        fetched_at = str(wrapper["fetched_at"])
    except (ValueError, TypeError, KeyError):
        logger.warning("quote_fx: USDCUSDT payload unreadable; fx unavailable")
        return None

    if not math.isfinite(close) or close <= 0.0:
        logger.warning("quote_fx: USDCUSDT close %r not usable; fx unavailable", close)
        return None
    if close < _DEPEG_LOW or close > _DEPEG_HIGH:
        logger.warning(
            "quote_fx: USDCUSDT close %.6f outside depeg band [%.2f, %.2f]; "
            "refusing to convert through it",
            close, _DEPEG_LOW, _DEPEG_HIGH,
        )
        return None

    rate = 1.0 / close
    return QuoteFxResult(
        rate=rate, available_at=available_at, fetched_at=fetched_at
    )


def usdt_usd_as_of(as_of: str) -> QuoteFxResult | None:
    """Point-in-time accessor: only ``as_of == today (UTC)`` can return a rate.

    The feed is LIVE-ONLY — no PIT archive of USDCUSDT closes exists — so
    any ``as_of`` other than today UTC (earlier dates strictly, and future
    dates equally, which can have no observation yet) returns None, and an
    unparseable date string degrades to None rather than raising. Per the
    frozen RFC it is FORBIDDEN to default the missing rate to 1.0: the
    caller must disclose the conversion as unavailable. ``as_of == today``
    delegates to :func:`fetch_usdt_usd` (whose print is today's forming
    candle, i.e. the freshest quote available right now).
    """
    try:
        as_of_date = datetime.strptime(as_of, "%Y-%m-%d").replace(tzinfo=UTC).date()
    except ValueError:
        logger.warning("quote_fx: unparseable as_of %r; fx unavailable", as_of)
        return None
    if as_of_date != datetime.now(UTC).date():
        return None
    return fetch_usdt_usd()


def render_fx_line(result: QuoteFxResult | None) -> str:
    """One-line disclosure of the FX leg for decision markdown.

    The available branch shows the rate to 6 decimals with its source and
    the live-only caveat; the None branch states UNAVAILABLE explicitly so
    a report reader can never mistake a missing conversion for a 1.0 peg.
    """
    if result is None:
        return "USDT/USD fx: UNAVAILABLE (live-only feed; not PIT-replayable)"
    return f"USDT/USD fx: {result.rate:.6f} ({result.source}, live-only)"
