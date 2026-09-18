"""Tests for yialpha.dataflows.onchain_flows — the positioning split's
on-chain context vendor (blockchain.info charts).

The module's recent hardening round left four behaviors unpinned; these tests
hold them down (all hermetic — the disk-cache seam ``_fetch_chart_cached`` /
the raw ``fetch_chart_json`` binding are monkeypatched, never the network):

1. **Mean-span label**: the render names the span the mean ACTUALLY covered
   (a PIT-filtered replay sees fewer days than the nominal 30d window), not
   the requested "30days".
2. **Structural-gap sentinel**: an as_of older than the vendor's trailing
   ~30-day window records an ``optional_unavailable`` ledger sentinel whose
   detail names the structural gap — distinguishable from a vendor outage.
3. **Per-chart failure sentinel**: one chart's cached payload being garbage /
   None records a sentinel for THAT chart while the other chart still renders.
4. **fetched_at oldest-ingredient-wins**: the block's ``fetched_at`` is the
   OLDEST chart wrapper's timestamp (or assembly-time NOW when no chart
   carried one) — a merged block never claims to be fresher than its oldest
   ingredient.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

import yialpha.dataflows.onchain_flows as ocf
from yialpha.dataflows import quality
from yialpha.dataflows.disk_cache import vendor_cache_dir

_VOLUME = "estimated-transaction-volume-usd"
_ADDRESSES = "n-unique-addresses"


@pytest.fixture(autouse=True)
def _fresh_quality_context():
    """Bind a fresh quality ledger per test and clear it afterwards so
    recorded sentinels never leak into other test files sharing the context."""
    quality.ensure_run_context()
    yield
    quality.reset_quality()


def _chart_payload(n_points: int = 5, *, end: datetime | None = None,
                   base: float = 100.0) -> dict[str, object]:
    """n_points daily points ending at ``end`` (default: today 00:00 UTC)."""
    if end is None:
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int((end - timedelta(days=n_points - 1)).timestamp())
    return {
        "values": [
            {"x": start_ts + i * 86_400, "y": base + i} for i in range(n_points)
        ]
    }


def _today_end() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _patch_cached(monkeypatch, per_chart: dict[str, tuple[object, str]]) -> None:
    """Monkeypatch the disk-cache seam: chart -> (payload, fetched_at)."""
    def _fake(chart: str):
        return per_chart[chart]

    monkeypatch.setattr(ocf, "_fetch_chart_cached", _fake)


# ---------------------------------------------------------------------------
# 1. mean-span label names the ACTUAL span
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_render_labels_actual_span_not_nominal_30d(monkeypatch):
    # 5 daily points = a 4-day span, well short of the requested 30days
    # window; the mean (102) covers exactly those points.
    _patch_cached(monkeypatch, {
        c: (_chart_payload(5), "2026-09-19T00:00:00") for c in ocf.CHARTS
    })
    flows = ocf.fetch_onchain_flows(date.today().isoformat())

    assert flows is not None
    for entry in flows.series.values():
        assert entry is not None
        _instant, latest, mean, span_days = entry
        assert span_days == 4
        assert latest == pytest.approx(104.0)
        assert mean == pytest.approx(102.0)

    block = ocf.render_onchain_block(flows)
    assert block.count("4d mean 102") == 2  # one per chart
    # The nominal window must NOT be mislabeled onto a shorter statistic.
    assert "30d mean" not in block


# ---------------------------------------------------------------------------
# (mechanism behind #4) the cached wrapper travels fetched_at with the bytes
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_disk_cache_wrapper_carries_fetched_at(monkeypatch):
    payload = _chart_payload(5)
    monkeypatch.setattr(ocf, "fetch_chart_json", lambda chart, **k: payload)
    flows = ocf.fetch_onchain_flows(date.today().isoformat())
    assert flows is not None

    cache_dir = Path(vendor_cache_dir("onchain"))
    wrappers = {}
    for chart in ocf.CHARTS:
        cached = cache_dir / f"{chart}_{ocf._TIMESPAN}.json"
        assert cached.is_file()
        wrapper = json.loads(cached.read_text(encoding="utf-8"))
        assert set(wrapper) == {"fetched_at", "payload"}
        assert wrapper["payload"] == payload
        wrappers[chart] = wrapper["fetched_at"]
    # The block reports the oldest of its ingredients' wrapper stamps.
    assert flows.fetched_at == min(wrappers.values())


# ---------------------------------------------------------------------------
# 2. structural-gap sentinel
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_structural_gap_records_typed_sentinels(monkeypatch):
    # Healthy vendor data (a fresh trailing 30-day window), but the replay
    # date is far older than that window: no point can survive the PIT filter
    # regardless of vendor health — a structural gap, not an outage.
    _patch_cached(monkeypatch, {
        c: (_chart_payload(30), "2026-09-19T00:00:00") for c in ocf.CHARTS
    })
    as_of = "2020-01-01"
    flows = ocf.fetch_onchain_flows(as_of)

    assert flows is None  # both charts empty -> no block at all
    events = quality.snapshot_quality()
    assert len(events) == 2
    assert {e["method"] for e in events} == {"fetch_onchain_flows"}
    for e in events:
        assert e["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
        assert "structural" in e["detail"]
        assert "trailing 30-day window" in e["detail"]
        # Each sentinel names its own chart.
        assert any(chart in e["detail"] for chart in ocf.CHARTS)
    # The capability-absent render discloses the structural caveat.
    assert "capability absent" in ocf.render_onchain_block(None).lower()


# ---------------------------------------------------------------------------
# 3. per-chart failure sentinel (other chart still renders)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_failed_chart_records_sentinel_other_chart_renders(monkeypatch):
    # Volume chart's cached payload is None (fail-open fetch wrote nothing);
    # the addresses chart is healthy.
    end = _today_end()
    _patch_cached(monkeypatch, {
        _VOLUME: (None, "2026-09-19T00:00:00"),
        _ADDRESSES: (_chart_payload(5, end=end), "2026-09-19T01:00:00"),
    })
    flows = ocf.fetch_onchain_flows(date.today().isoformat())

    assert flows is not None
    assert flows.series[_VOLUME] is None
    assert flows.series[_ADDRESSES] is not None
    # Exactly one sentinel, for the failed chart, WITHOUT the structural tag
    # (recent as_of + genuinely broken chart must not read as a replay gap).
    events = quality.snapshot_quality()
    assert len(events) == 1
    e = events[0]
    assert e["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    assert _VOLUME in e["detail"]
    assert "no usable points" in e["detail"]
    assert "structural" not in e["detail"]
    # The block still renders: failed chart disclosed, healthy chart shown.
    block = ocf.render_onchain_block(flows)
    assert "Est. transaction volume (USD): unavailable" in block
    assert "Active addresses: 104 latest" in block
    assert "4d mean 102" in block
    # available_at is the healthy chart's last used point, not assembly NOW.
    end = _today_end()
    assert flows.available_at == end.isoformat(timespec="seconds")


@pytest.mark.unit
def test_garbage_cached_payload_fails_soft(monkeypatch):
    # A cached payload that is not a dict (here: a JSON list) makes _points
    # raise; the advisory loop must swallow it into the same per-chart
    # sentinel instead of raising into the run.
    _patch_cached(monkeypatch, {
        _VOLUME: ([{"x": 1, "y": 2}], "2026-09-19T00:00:00"),
        _ADDRESSES: (_chart_payload(5), "2026-09-19T01:00:00"),
    })
    flows = ocf.fetch_onchain_flows(date.today().isoformat())

    assert flows is not None
    assert flows.series[_VOLUME] is None
    assert flows.series[_ADDRESSES] is not None
    events = quality.snapshot_quality()
    assert [e["kind"] for e in events] == [quality.KIND_OPTIONAL_UNAVAILABLE]
    assert "no usable points" in events[0]["detail"]


# ---------------------------------------------------------------------------
# 4. fetched_at: oldest ingredient wins; NOW only as fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetched_at_is_oldest_ingredient(monkeypatch):
    older, newer = "2026-09-18T00:00:00", "2026-09-19T08:00:00"
    _patch_cached(monkeypatch, {
        _VOLUME: (_chart_payload(5), older),
        _ADDRESSES: (_chart_payload(5), newer),
    })
    flows = ocf.fetch_onchain_flows(date.today().isoformat())
    assert flows is not None
    assert flows.fetched_at == older

    # Order must not matter: whichever chart is older wins.
    _patch_cached(monkeypatch, {
        _VOLUME: (_chart_payload(5), newer),
        _ADDRESSES: (_chart_payload(5), older),
    })
    flows2 = ocf.fetch_onchain_flows(date.today().isoformat())
    assert flows2 is not None
    assert flows2.fetched_at == older


@pytest.mark.unit
def test_fetched_at_falls_back_to_assembly_now(monkeypatch):
    # Neither chart carried a wrapper stamp (legacy/empty fetched_at): the
    # block falls back to assembly-time NOW — bounded, not asserted exact.
    _patch_cached(monkeypatch, {
        c: (_chart_payload(5), "") for c in ocf.CHARTS
    })
    before = datetime.now(UTC)
    flows = ocf.fetch_onchain_flows(date.today().isoformat())
    after = datetime.now(UTC)
    assert flows is not None
    got = datetime.fromisoformat(flows.fetched_at).replace(tzinfo=UTC)
    assert before.replace(microsecond=0) <= got <= after
