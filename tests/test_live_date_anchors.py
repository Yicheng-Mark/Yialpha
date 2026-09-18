"""Live-date anchors (host-local vs UTC) on live perp runs.

Pins the dual-anchor contract introduced after a live BTCUSDT bundle was
misclassified as a historical replay: the host calendar (Asia/Shanghai,
UTC+8) rolls to the next date at 00:00 local while the crypto venue is still
on the previous UTC date, and the pipeline carries BOTH labels —

  - the interactive CLI defaults ``--date``/analysis date to the HOST-LOCAL
    date (``datetime.now()``);
  - the crypto/perp data layer anchors on UTC end-to-end (klines staleness,
    ``_futures_data_window``'s "now", the vision publication cap, and the
    user-supplied ``--date`` of a live perp run which naturally names the
    exchange's own current date).

A single-anchor live check therefore misfires for one 8-hour window a day,
in BOTH directions:

  - ``is_historical_date`` on the host-local ``date.today()`` judged a
    UTC-labelled LIVE perp run "historical" → the perp bundle skipped its
    live-only components (funding / premium / depth / ADL), the market
    analyst dropped the 6 live REST tools + web search, and the historical
    nudge replaced the live one;
  - ``quote_fx.usdt_usd_as_of`` on strict ``now(UTC).date()`` returned None
    for the CLI-labelled live run → the stock-perp fair-value bridge
    disclosed "USDT/USD fx: UNAVAILABLE" for 8h a day.

The fix accepts BOTH anchors as "today" (:func:`live_anchor_dates`); the two
differ by at most one day, so past and future labels stay historical.
Hermetic: every clock seam is monkeypatched.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

import yialpha.dataflows.perp_bundle as pb
import yialpha.dataflows.utils as du
import yialpha.perp.quote_fx as qfx
from yialpha.perp.quote_fx import QuoteFxResult, usdt_usd_as_of


def _freeze_anchors(monkeypatch, local_day: str, utc_day: str) -> None:
    """Freeze the live anchors to a simulated UTC+8 post-midnight window.

    Local date is one day AHEAD of the UTC date (the 00:00–08:00 local
    window on a UTC+8 host). Patching ``utils.live_anchor_dates`` propagates
    to every already-imported ``is_historical_date`` caller (the function
    resolves it through utils' module globals at call time).
    """
    anchors = (
        datetime.strptime(local_day, "%Y-%m-%d").date(),
        datetime.strptime(utc_day, "%Y-%m-%d").date(),
    )
    monkeypatch.setattr(du, "live_anchor_dates", lambda: anchors)
    # quote_fx bound the name at import; patch its copy too.
    monkeypatch.setattr(qfx, "live_anchor_dates", lambda: anchors)


# ---- live_anchor_dates sanity -------------------------------------------------


@pytest.mark.unit
def test_live_anchor_dates_returns_local_and_utc_at_most_one_day_apart():
    local_day, utc_day = du.live_anchor_dates()
    assert isinstance(local_day, date) and isinstance(utc_day, date)
    assert abs((local_day - utc_day).days) <= 1


# ---- is_historical_date: both labels are live in the skew window -------------


@pytest.mark.unit
def test_utc_labelled_live_date_is_not_historical(monkeypatch):
    _freeze_anchors(monkeypatch, local_day="2026-09-19", utc_day="2026-09-18")
    # The exchange's own current date — the label a live perp run carries
    # when the user follows the venue's UTC convention past local midnight.
    assert du.is_historical_date("2026-09-18") is False


@pytest.mark.unit
def test_local_labelled_live_date_is_not_historical(monkeypatch):
    _freeze_anchors(monkeypatch, local_day="2026-09-19", utc_day="2026-09-18")
    # The CLI's own default label (host-local date) stays live as before.
    assert du.is_historical_date("2026-09-19") is False


@pytest.mark.unit
def test_past_and_future_labels_stay_historical_in_the_skew_window(monkeypatch):
    _freeze_anchors(monkeypatch, local_day="2026-09-19", utc_day="2026-09-18")
    assert du.is_historical_date("2026-09-17") is True   # past for both anchors
    assert du.is_historical_date("2026-09-20") is True   # future for both anchors
    # A date one week out is historical from either direction.
    assert du.is_historical_date("2026-09-11") is True
    assert du.is_historical_date("2026-09-26") is True


@pytest.mark.unit
def test_live_and_malformed_semantics_unchanged(monkeypatch):
    _freeze_anchors(monkeypatch, local_day="2026-09-19", utc_day="2026-09-18")
    assert du.is_historical_date(None) is False          # live mode, no label
    assert du.is_historical_date("") is False
    assert du.is_historical_date("2026/08/01") is True   # unparseable → causal branch
    assert du.is_historical_date("garbage") is True


# ---- perp bundle wiring: the UTC-labelled live run keeps live components -----


@pytest.mark.unit
def test_perp_bundle_live_gate_accepts_utc_label(monkeypatch):
    """The observed bug, end to end at the gate: a live run labelled with the
    UTC current date (host calendar already rolled) must keep its live-only
    bundle components instead of skipping them as ``skipped_live_only``."""
    _freeze_anchors(monkeypatch, local_day="2026-09-19", utc_day="2026-09-18")
    monkeypatch.setattr(
        pb, "_http_get",
        lambda *a, **k: [
            {"fundingTime": 1_800_000_000_000, "fundingRate": "0.0001"},
            {"fundingTime": 1_800_088_000_000, "fundingRate": "0.0001"},
        ],
    )
    out = pb._fetch_funding("BTCUSDT", "2026-09-18")
    assert out["status"] == pb.STATUS_OK
    assert out["sum_7d"] == pytest.approx(0.0002)


@pytest.mark.unit
def test_perp_bundle_historical_gate_still_skips_live_legs(monkeypatch):
    _freeze_anchors(monkeypatch, local_day="2026-09-19", utc_day="2026-09-18")
    # A genuinely past label keeps the PIT skip — the widening must not leak
    # live positioning/funding into a replay date.
    assert pb._fetch_funding("BTCUSDT", "2026-09-01") == {
        "status": pb.STATUS_SKIPPED_LIVE_ONLY,
    }


# ---- quote_fx: the CLI-labelled live run keeps its USDT/USD rate --------------


@pytest.mark.unit
def test_quote_fx_accepts_local_labelled_live_date(monkeypatch):
    """The mirror bug: the CLI's host-local default label was rejected by the
    strict UTC gate, so a live stock-perp run lost its FX conversion for the
    whole post-local-midnight window."""
    _freeze_anchors(monkeypatch, local_day="2026-09-19", utc_day="2026-09-18")
    sentinel = QuoteFxResult(
        rate=0.9993, available_at="2026-09-18", fetched_at="2026-09-18T18:00:00+00:00",
    )
    monkeypatch.setattr(qfx, "fetch_usdt_usd", lambda: sentinel)
    assert usdt_usd_as_of("2026-09-19") is sentinel   # CLI default label
    assert usdt_usd_as_of("2026-09-18") is sentinel   # venue UTC label


@pytest.mark.unit
def test_quote_fx_past_and_future_labels_stay_unavailable(monkeypatch):
    _freeze_anchors(monkeypatch, local_day="2026-09-19", utc_day="2026-09-18")

    def _boom(*a, **k):
        raise AssertionError("live-only feed must not be fetched for a non-live label")

    monkeypatch.setattr(qfx, "fetch_usdt_usd", _boom)
    assert usdt_usd_as_of("2026-09-17") is None   # past for both anchors
    assert usdt_usd_as_of("2026-09-20") is None   # future for both anchors
    assert usdt_usd_as_of("not-a-date") is None   # unparseable degrades, no raise


# ---- regression pin: the anchors really are the two clocks --------------------


@pytest.mark.unit
def test_live_anchor_dates_matches_the_two_real_clocks():
    local_day, utc_day = du.live_anchor_dates()
    assert local_day == datetime.now().date()
    assert utc_day == datetime.now(UTC).date()
    assert abs((local_day - utc_day).days) <= 1  # sanity across any host offset
