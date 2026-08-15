"""Polymarket prediction-market vendor: forward-looking filtering, volume
ranking, formatting, graceful degradation, router integration, and the
short-TTL disk cache behind ``_request``.

All API access is mocked, so these run without a network connection.
"""
import json
import os
import time
import unittest
from unittest import mock

import pytest
import requests

from yiagents.dataflows import interface, polymarket
from yiagents.dataflows.config import reset_config, set_config


def _market(question, prob, *, volume, end_date, closed=False, wk=None):
    return {
        "question": question,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": f'["{prob}", "{round(1 - prob, 4)}"]',
        "volumeNum": volume,
        "endDate": end_date,
        "closed": closed,
        "oneWeekPriceChange": wk,
    }


# One event with a mix: a high-volume open market, a closed one, a past-dated
# one, and a lower-volume open one. Far-future / far-past dates keep the test
# independent of the real clock.
_SEARCH = {
    "events": [
        {
            "markets": [
                _market("Open big?", 0.76, volume=5_000_000, end_date="2030-12-31T00:00:00Z", wk=-0.045),
                _market("Resolved already?", 1.0, volume=9_000_000, end_date="2030-12-31T00:00:00Z", closed=True),
                _market("Past event?", 0.5, volume=8_000_000, end_date="2020-01-01T00:00:00Z"),
                _market("Open small?", 0.30, volume=1_000, end_date="2030-06-30T00:00:00Z"),
            ]
        }
    ]
}


@pytest.mark.unit
class PolymarketFilterTests(unittest.TestCase):
    def test_closed_and_past_markets_are_excluded(self):
        with mock.patch.object(polymarket, "_request", return_value=_SEARCH):
            out = polymarket.get_prediction_markets("anything", limit=10)
        self.assertIn("Open big?", out)
        self.assertIn("Open small?", out)
        self.assertNotIn("Resolved already?", out)  # closed
        self.assertNotIn("Past event?", out)         # endDate in the past

    def test_ranked_by_volume(self):
        with mock.patch.object(polymarket, "_request", return_value=_SEARCH):
            out = polymarket.get_prediction_markets("anything", limit=10)
        self.assertLess(out.index("Open big?"), out.index("Open small?"))

    def test_limit_caps_results(self):
        with mock.patch.object(polymarket, "_request", return_value=_SEARCH):
            out = polymarket.get_prediction_markets("anything", limit=1)
        self.assertIn("Open big?", out)
        self.assertNotIn("Open small?", out)


@pytest.mark.unit
class PolymarketFormatTests(unittest.TestCase):
    def test_probability_volume_and_weekly_change_render(self):
        with mock.patch.object(polymarket, "_request", return_value=_SEARCH):
            out = polymarket.get_prediction_markets("anything", limit=10)
        self.assertIn("Yes 76%", out)
        self.assertIn("$5,000,000 volume", out)
        self.assertIn("resolves 2030-12-31", out)
        self.assertIn("1-week -4.5pp", out)  # -0.045 -> -4.5pp

    def test_weekly_change_omitted_when_absent(self):
        # "Open small?" has wk=None -> no 1-week clause on its line.
        with mock.patch.object(polymarket, "_request", return_value=_SEARCH):
            out = polymarket.get_prediction_markets("anything", limit=10)
        small_line = next(ln for ln in out.splitlines() if "Open small?" in ln)
        self.assertNotIn("1-week", small_line)

    def test_no_matches_reports_clearly(self):
        with mock.patch.object(polymarket, "_request", return_value={"events": []}):
            out = polymarket.get_prediction_markets("obscure ticker", limit=6)
        self.assertIn("No open prediction markets", out)


@pytest.mark.unit
class PolymarketResilienceTests(unittest.TestCase):
    def test_network_error_propagates_to_router(self):
        # A transport failure must NOT be swallowed into a returned prose
        # string: the router needs the exception to record the optional-
        # category sentinel and return its DATA_UNAVAILABLE message (fail-
        # closed: the evidence chain must capture the degradation).
        with mock.patch.object(
            polymarket, "_request", side_effect=requests.RequestException("boom")
        ), self.assertRaises(requests.RequestException):
            polymarket.get_prediction_markets("Fed rate cut")

    def test_router_records_optional_sentinel_on_transport_error(self):
        """Through the router, the propagated error becomes the optional-
        category DATA_UNAVAILABLE sentinel + a KIND_OPTIONAL_UNAVAILABLE
        data-quality event (previously the swallowed prose lost the evidence)."""
        from yiagents.dataflows import quality

        quality.ensure_run_context()
        try:
            set_config({"data_vendors": {"prediction_markets": "polymarket"}})
            with mock.patch.object(
                polymarket, "_request",
                side_effect=requests.RequestException("boom"),
            ):
                out = interface.route_to_vendor("get_prediction_markets", "fed", 5)
            self.assertTrue(out.startswith("DATA_UNAVAILABLE"))
            events = quality.snapshot_quality()
            self.assertTrue(
                any(e["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
                    for e in events)
            )
        finally:
            quality.reset_quality()


def _gamma_response(payload: dict) -> mock.Mock:
    """A requests.Response stand-in whose ``.content`` is the JSON payload."""
    resp = mock.Mock()
    resp.content = json.dumps(payload).encode("utf-8")
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture(autouse=True)
def _isolated_polymarket_cache(tmp_path, monkeypatch):
    """Route the Polymarket disk cache behind ``_request`` into a per-test
    tmp dir, so cache tests never touch the real user cache and repeat calls
    in other tests cannot bypass their mocked transports."""
    def _cache_dir(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(polymarket, "vendor_cache_dir", _cache_dir)


@pytest.mark.unit
class TestPolymarketDiskCache:
    def test_repeat_call_within_ttl_hits_transport_once(self):
        resp = _gamma_response(_SEARCH)
        with mock.patch.object(
            polymarket.requests, "get", return_value=resp
        ) as transport:
            first = polymarket.get_prediction_markets("fed rate cut", limit=10)
            second = polymarket.get_prediction_markets("fed rate cut", limit=10)
        assert transport.call_count == 1
        assert first == second
        assert "Open big?" in first

    def test_expired_cache_refetches(self, tmp_path):
        resp = _gamma_response(_SEARCH)
        with mock.patch.object(
            polymarket.requests, "get", return_value=resp
        ) as transport:
            polymarket.get_prediction_markets("fed rate cut", limit=10)
            cache_file = next((tmp_path / "polymarket").glob("search_*.json"))
            stale = time.time() - 3600.0  # 1h old, far past the 10-minute TTL
            os.utime(cache_file, (stale, stale))
            polymarket.get_prediction_markets("fed rate cut", limit=10)
        assert transport.call_count == 2

    def test_poisoned_cache_entry_raises_typed_error(self, tmp_path):
        # A corrupt but fresh cache entry is served without a network call and
        # must fail loudly with the vendor's existing JSON error type — never
        # degrade into prose or return garbage.
        resp = _gamma_response(_SEARCH)
        with mock.patch.object(
            polymarket.requests, "get", return_value=resp
        ) as transport:
            polymarket.get_prediction_markets("fed rate cut", limit=10)
            cache_file = next((tmp_path / "polymarket").glob("search_*.json"))
            cache_file.write_bytes(b"<html>definitely not json</html>")
            with pytest.raises(json.JSONDecodeError):
                polymarket.get_prediction_markets("fed rate cut", limit=10)
        assert transport.call_count == 1  # poisoned bytes came from disk


@pytest.mark.unit
class PolymarketRoutingTests(unittest.TestCase):
    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_category_routes_to_polymarket(self):
        self.assertEqual(
            interface.get_category_for_method("get_prediction_markets"),
            "prediction_markets",
        )
        set_config({"data_vendors": {"prediction_markets": "polymarket"}})
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_prediction_markets": {"polymarket": lambda *a, **k: "POLY_OK"}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_prediction_markets", "fed", 5)
        self.assertEqual(out, "POLY_OK")


if __name__ == "__main__":
    unittest.main()
