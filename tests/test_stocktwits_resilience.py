"""StockTwits fetch: transport-error resilience (the placeholder contract the
sentiment node relies on, including the http.client chunked-transfer exceptions
that are not OSErrors, #1024), the short-TTL disk cache (repeat calls within
the TTL skip the network), the data-quality sentinel recorded on degradation
(the fetcher is NOT routed through the router, so without recording it here
the run's evidence chain loses the failure), and the safe-ticker path
validation before URL interpolation."""

from __future__ import annotations

import http.client
import json
import os
import time
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from yialpha.dataflows import quality, stocktwits


def _raise(exc):
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            raise exc
    return _Resp()


def _ok(payload: bytes):
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            return payload
    return _Resp()


@pytest.fixture(autouse=True)
def _isolated_stocktwits_cache(tmp_path, monkeypatch):
    """Route the StockTwits disk cache into a per-test tmp dir.

    Without this, a repeat call inside the TTL would be served from the real
    user cache and bypass the mocked transport (and a mocked response would
    pollute the real cache).
    """
    def _cache_dir(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(stocktwits, "vendor_cache_dir", _cache_dir)


def _messages_payload(*sentiments: str) -> bytes:
    return json.dumps({
        "messages": [
            {
                "created_at": f"2026-08-15T10:0{i}:00Z",
                "user": {"username": f"user{i}"},
                "entities": {"sentiment": {"basic": s}} if s else {},
                "body": f"message {i} ({s or 'no-label'})",
            }
            for i, s in enumerate(sentiments)
        ]
    }).encode("utf-8")


@pytest.mark.unit
class TestStockTwitsResilience:
    @pytest.mark.parametrize(
        "exc",
        [
            http.client.IncompleteRead(b""),
            HTTPError("url", 503, "down", {}, None),  # type: ignore[arg-type]
            TimeoutError("slow"),
        ],
    )
    def test_transport_errors_return_placeholder(self, exc):
        with patch.object(stocktwits, "urlopen", return_value=_raise(exc)):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "unavailable" in out.lower()
        assert out.startswith("<stocktwits unavailable")

    @pytest.mark.parametrize(
        "exc",
        [
            http.client.IncompleteRead(b""),
            HTTPError("url", 503, "down", {}, None),  # type: ignore[arg-type]
            TimeoutError("slow"),
        ],
    )
    def test_transport_degradation_records_quality_sentinel(self, exc):
        """The placeholder must NOT be the only trace: an optional-unavailable
        sentinel is recorded so the run's data_quality block reflects the
        missing source (this fetcher bypasses route_to_vendor)."""
        quality.ensure_run_context()
        try:
            with patch.object(stocktwits, "urlopen", return_value=_raise(exc)):
                stocktwits.fetch_stocktwits_messages("NVDA")
            events = quality.snapshot_quality()
        finally:
            quality.reset_quality()
        assert any(
            e["method"] == "fetch_stocktwits_messages"
            and e["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
            for e in events
        )


@pytest.mark.unit
class TestStockTwitsTickerValidation:
    @pytest.mark.parametrize("bad", ["../../etc/passwd", "A B", "a|b", "."])
    def test_unsafe_ticker_raises_before_url_build(self, bad):
        """A malformed/attacker-controlled ticker must never be interpolated
        into the URL path (fail-closed ValueError from safe_ticker_component)."""
        with patch.object(stocktwits, "urlopen") as transport:
            transport.side_effect = AssertionError("no request may be attempted")
            with pytest.raises(ValueError):
                stocktwits.fetch_stocktwits_messages(bad)

    def test_valid_tickers_pass_validation(self):
        # Dashed, dotted, caret, equals, plus — the symbol grammar — all pass.
        for ok in ("BRK.B", "BTC-USD", "^GSPC", "GC=F", "XAUUSD+"):
            with patch.object(
                stocktwits, "urlopen", return_value=_ok(_messages_payload("Bullish"))
            ):
                out = stocktwits.fetch_stocktwits_messages(ok)
            assert "Bullish: 1 (100%)" in out


@pytest.mark.unit
class TestStockTwitsCache:
    def test_repeat_call_within_ttl_hits_transport_once(self):
        payload = _messages_payload("Bullish", "Bearish", None)
        with patch.object(stocktwits, "urlopen", return_value=_ok(payload)) as transport:
            first = stocktwits.fetch_stocktwits_messages("NVDA")
            second = stocktwits.fetch_stocktwits_messages("NVDA")
        assert transport.call_count == 1
        assert first == second
        assert "Bullish: 1 (33%)" in first
        assert "message 0 (Bullish)" in first

    def test_distinct_tickers_cached_separately(self):
        nvda = _messages_payload("Bullish")
        tsla = _messages_payload("Bearish")
        with patch.object(
            stocktwits, "urlopen", side_effect=[_ok(nvda), _ok(tsla)]
        ) as transport:
            out_nvda = stocktwits.fetch_stocktwits_messages("NVDA")
            out_tsla = stocktwits.fetch_stocktwits_messages("TSLA")
            # Repeats still come from each ticker's own cache entry.
            stocktwits.fetch_stocktwits_messages("NVDA")
            stocktwits.fetch_stocktwits_messages("TSLA")
        assert transport.call_count == 2
        assert "message 0 (Bullish)" in out_nvda
        assert "message 0 (Bearish)" in out_tsla

    def test_expired_cache_refetches(self, tmp_path):
        payload = _messages_payload("Bullish")
        with patch.object(stocktwits, "urlopen", return_value=_ok(payload)) as transport:
            stocktwits.fetch_stocktwits_messages("NVDA")
            cache_file = tmp_path / "stocktwits" / "stream_NVDA.json"
            stale = time.time() - 3600.0  # 1h old, far past the 5-minute TTL
            os.utime(cache_file, (stale, stale))
            stocktwits.fetch_stocktwits_messages("NVDA")
        assert transport.call_count == 2

    def test_poisoned_cache_entry_follows_existing_error_path(self, tmp_path):
        # A corrupt but fresh cache entry is served without a network call and
        # must fail through the vendor's existing degradation path (the
        # "<stocktwits unavailable>" placeholder) rather than return garbage.
        cache_file = tmp_path / "stocktwits" / "stream_NVDA.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(b"<html>definitely not json</html>")
        with patch.object(stocktwits, "urlopen") as transport:
            transport.side_effect = AssertionError(
                "a fresh cache entry must be served without a network call"
            )
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out == "<stocktwits unavailable: JSONDecodeError>"
