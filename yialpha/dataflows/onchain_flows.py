"""On-chain flow context for the V2.3 Positioning analyst (keyless, optional).

Vendor choice (frozen V2.3, worklog v2-perp-progress.md #19): the
**blockchain.com Charts API** — ``https://api.blockchain.info/charts/<name>?timespan=...&format=json``.

Why this ONE keyless source and not CoinGecko:

* **Keyless and free** — no API key, no signup, no budget (the positioning
  analyst runs inside every flag-on perp run; a keyed vendor would add key
  management for an advisory block).
* **HISTORICAL-capable** — every chart point carries its own unix timestamp,
  so a replay date can be served point-in-time-correctly by filtering to
  points at/before the as-of day. CoinGecko's keyless ``/global`` /
  ``/market_chart`` endpoints cap free history (365 days) and are documented
  as current-market views, which would force a LIVE_ONLY tag and a worse
  replay contract for no added signal.
* **Replayability**: because the whole view derives from timestamped chart
  points, the block is tagged ``PIT_REPLAYABLE`` with ``available_at`` set
  from the latest point timestamp actually used — the strictest honest tag
  a vendor block can carry.

Charts fetched (both are BTC network-activity proxies that position the
perp's leveraged crowding against REAL settlement-layer usage):

* ``estimated-transaction-volume-usd`` — USD transaction volume (real flow);
* ``n-unique-addresses`` — active addresses (participation breadth).

Contract:

* **Fail-soft, fail-open.** Any fetch/parse failure returns ``None``; the
  positioning analyst renders a capability-absent disclosure line instead.
  This module never raises into a run and there is no key check (keyless).
* **Disk-cached** via :func:`yialpha.dataflows.disk_cache.cached_or_fetch`
  (vendor dir ``onchain``, one file per chart+timespan, 6-hour TTL) so the
  two chart fetches ride one disk read per window; ``fail_open=True``.
* **Module-top seams** (``fetch_chart_json`` / the raw ``requests_get``
  binding) are monkeypatchable so tests never touch the network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from yialpha.dataflows.disk_cache import cached_or_fetch, vendor_cache_dir

logger = logging.getLogger(__name__)

#: Module-top seam: the raw HTTP GET. Tests monkeypatch THIS binding (the
#: same contract as outcome_compute's vendor seams).
requests_get = requests.get

#: Charts API base (keyless, documented, CORS-open public endpoint).
CHARTS_BASE_URL = "https://api.blockchain.info/charts"

#: Chart name -> (label, higher-is-*) for the fetched pair.
CHARTS: dict[str, tuple[str, str]] = {
    "estimated-transaction-volume-usd": (
        "Est. transaction volume (USD)", "higher = more real settlement flow",
    ),
    "n-unique-addresses": (
        "Active addresses", "higher = broader participation",
    ),
}

#: Lookback window requested from the API. blockchain.info accepts either
#: days ("30days") or "all"; 30 days matches the positioning windows the
#: bundle's OI/LSR legs use, so the on-chain context covers the same regime
#: the perp positioning numbers do.
_TIMESPAN = "30days"

#: Cache freshness: chart data updates daily; 6 hours bounds staleness
#: without re-downloading within one research session.
_CACHE_TTL_DAYS = 0.25

#: Disk-cache vendor / directory name under ``data_cache_dir``.
_VENDOR = "onchain"

#: Replayability tag for the whole block (see module docstring).
PIT_REPLAYABLE = "PIT_REPLAYABLE"

#: Provenance label rendered into the block header.
ONCHAIN_SOURCE = "blockchain.info:charts"


def fetch_chart_json(chart: str, timespan: str = _TIMESPAN) -> dict[str, Any] | None:
    """Fetch ONE chart's JSON (``{"values": [{"x": unix_s, "y": v}, ...]}``).

    Returns ``None`` on any HTTP/parse failure (fail-soft). Runs inside
    ``cached_or_fetch``'s fetch slot, so the bytes it produces are what the
    disk cache stores.
    """
    resp = requests_get(
        f"{CHARTS_BASE_URL}/{chart}",
        params={"timespan": timespan, "format": "json", "sampled": "true"},
        timeout=15,
    )
    resp.raise_for_status()
    payload = json.loads(resp.text)
    return payload if isinstance(payload, dict) else None


def _points(payload: dict[str, Any] | None) -> list[tuple[datetime, float]]:
    """(instant, value) chart points, newest last; malformed rows skipped."""
    if not payload:
        return []
    out: list[tuple[datetime, float]] = []
    for row in payload.get("values") or []:
        if not isinstance(row, dict) or row.get("x") is None:
            continue
        try:
            instant = datetime.fromtimestamp(float(row["x"]), tz=UTC)
            value = float(row["y"])
        except (TypeError, ValueError, OSError, OverflowError):
            continue
        out.append((instant, value))
    return sorted(out, key=lambda item: item[0])


@dataclass(frozen=True)
class OnchainFlows:
    """The assembled on-chain context block inputs.

    ``series`` maps chart name -> (latest_point_instant, latest_value,
    mean_over_window, window_days) with ``None`` for charts that failed
    (their disclosure rides the render). ``window_days`` is the span the
    mean was actually computed over — the vendor serves a trailing
    ``30days`` window anchored at fetch-time NOW, so a PIT-filtered replay
    may see fewer days than the label suggests. ``available_at`` is the
    latest point instant used (the PIT anchor for the evidence row);
    ``fetched_at`` travels with the cache bytes (the OLDEST ingredient's
    timestamp — a merged block never claims to be fresher than its oldest
    chart); ``replayability`` is ``PIT_REPLAYABLE`` only within the
    vendor's trailing window (see :func:`fetch_onchain_flows`).
    """

    series: dict[str, tuple[datetime, float, float, int] | None]
    available_at: str
    fetched_at: str
    source: str = ONCHAIN_SOURCE
    replayability: str = PIT_REPLAYABLE


def _fetch_chart_cached(chart: str) -> tuple[Any, str] | None:
    """One chart through the shared disk cache as ``(payload, fetched_at)``.

    The wrapper's stored ``fetched_at`` travels out with the payload so a
    cache hit can never masquerade as a fresh fetch (regenerating it at
    assembly time claimed NOW for up-to-6h-old bytes).
    """
    raw = cached_or_fetch(
        vendor_cache_dir(_VENDOR),
        f"{chart}_{_TIMESPAN}.json",
        lambda: json.dumps(
            {
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "payload": fetch_chart_json(chart),
            }
        ).encode("utf-8"),
        ttl_days=_CACHE_TTL_DAYS,
        vendor=_VENDOR,
        fail_open=True,
    )
    if raw is None:
        return None
    try:
        wrapper = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(wrapper, dict):
        return None
    return (
        wrapper.get("payload"),
        str(wrapper.get("fetched_at") or ""),
    )


def fetch_onchain_flows(as_of: str) -> OnchainFlows | None:
    """Assemble the on-chain context as of ``as_of`` (a YYYY-MM-DD string).

    Points strictly after the as-of day are dropped (PIT filter), so a
    historical replay sees exactly the chart that existed then. Returns
    ``None`` only when BOTH charts fail — one surviving chart still yields a
    block (the missing one is disclosed in the render).

    Structural caveat: the vendor serves only a trailing ~30-day window
    anchored at fetch-time NOW, so a replay dated more than ~30 days back
    keeps zero points for BOTH charts — that case returns ``None`` too, but
    records a ledger sentinel describing the structural gap (not a vendor
    outage) so the run's data_quality block cannot mistake one for the
    other.
    """
    from . import quality

    cutoff = datetime.strptime(as_of, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=UTC
    )
    # The trailing window starts ~30 days before NOW; an as_of whose whole
    # day predates it can keep no points regardless of vendor health.
    window_start = datetime.now(UTC) - timedelta(days=30)
    structural = cutoff < window_start
    series: dict[str, tuple[datetime, float, float, int] | None] = {}
    latest_used: datetime | None = None
    oldest_fetched: str = ""
    for chart in CHARTS:
        try:
            cached = _fetch_chart_cached(chart)
            points = [p for p in _points(cached[0]) if p[0] <= cutoff] if cached else []
        except Exception:  # noqa: BLE001 — advisory vendor, never blocks
            logger.warning("onchain chart %s failed; skipping", chart, exc_info=True)
            points = []
            cached = None
        if not points:
            series[chart] = None
            quality.record_sentinel(
                "fetch_onchain_flows",
                quality.KIND_OPTIONAL_UNAVAILABLE,
                (
                    f"{chart}: as_of {as_of} predates the vendor's trailing "
                    f"30-day window (structural, not an outage)"
                    if structural
                    else f"{chart}: no usable points (vendor/parse failure)"
                ),
            )
            continue
        values = [value for _instant, value in points]
        span_days = max(1, (points[-1][0] - points[0][0]).days)
        series[chart] = (points[-1][0], points[-1][1], sum(values) / len(values), span_days)
        latest_used = (
            points[-1][0] if latest_used is None else max(latest_used, points[-1][0])
        )
        if cached and cached[1]:
            oldest_fetched = min(oldest_fetched, cached[1]) if oldest_fetched else cached[1]
    if latest_used is None:
        return None
    return OnchainFlows(
        series=series,
        available_at=latest_used.isoformat(timespec="seconds"),
        fetched_at=oldest_fetched or datetime.now(UTC).isoformat(timespec="seconds"),
    )


def render_onchain_block(data: OnchainFlows | None) -> str:
    """Render the on-chain context (or the capability-absent disclosure).

    ``data=None`` (vendor fully unavailable) renders the disclosure line the
    positioning analyst injects INSTEAD of inventing values — the same
    honest-gap posture as every other advisory block.
    """
    if data is None:
        return (
            "- **On-chain flows**: capability absent — the blockchain.info "
            "charts vendor returned no usable points (vendor unreachable, or "
            "the replay date predates the vendor's trailing 30-day window); "
            "no on-chain context is available for this run."
        )
    parts: list[str] = []
    for chart, entry in data.series.items():
        label = CHARTS[chart][0]
        note = CHARTS[chart][1]
        if entry is None:
            parts.append(f"{label}: unavailable ({note})")
            continue
        instant, value, mean, span_days = entry
        # The ACTUAL span the mean covered — the vendor's window is anchored
        # at fetch-time NOW, so a PIT-filtered replay sees fewer days than
        # the nominal "30d" and must not mislabel the statistic.
        parts.append(
            f"{label}: {value:,.0f} latest ({instant.strftime('%Y-%m-%d')}; "
            f"{span_days}d mean {mean:,.0f}; {note})"
        )
    header = (
        f"- **On-chain flows** ({data.source}, PIT chart points; as of "
        f"{data.available_at}): " + "; ".join(parts)
    )
    return header
