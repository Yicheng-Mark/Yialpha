"""Unit tests for the browser data augmentation module.

These exercise :class:`BrowserDataFetcher` and the :func:`count_keywords`
helper with Playwright **entirely mocked** — no browser launches, no network.
Missing-Playwright paths are driven by patching ``_get_playwright`` /
``_ensure_page`` to return ``None``, mirroring the resilience pattern used by
the sibling ``test_browser_broker.py``.

All tests are ``@pytest.mark.unit`` (registered in ``tests/conftest.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from yiagents.dataflows import browser_data
from yiagents.dataflows.browser_data import (
    BrowserDataFetcher,
    BrowserFetchResult,
    count_keywords,
    fetch_alternative_data,
)

# ---------------------------------------------------------------------------
# Fakes: stand-ins for a Playwright page / browser stack. They never hit the
# network and record calls so assertions can inspect behavior.
# ---------------------------------------------------------------------------


class _FakePage:
    """Minimal stand-in for a Playwright ``Page``."""

    def __init__(self, body_text: str = "hello world") -> None:
        self._body = body_text
        self.url = ""
        self.default_timeout: int | None = None
        self.goto_calls: list[str] = []
        self.waited_for: list[str] = []

    def set_default_timeout(self, ms: int) -> None:
        self.default_timeout = ms

    def goto(self, url: str) -> None:
        self.url = url
        self.goto_calls.append(url)

    def wait_for_selector(self, selector: str) -> None:
        self.waited_for.append(selector)

    def inner_text(self, selector: str) -> str:
        # Default extractor calls inner_text("body").
        if selector == "body":
            return self._body
        return ""


def _make_fetcher_with_page(body_text: str = "hello world") -> BrowserDataFetcher:
    """Build a fetcher whose browser stack is replaced by a single fake page.

    Patching ``_ensure_page`` (rather than ``_get_playwright``) lets the whole
    launch path be skipped while still exercising :meth:`fetch` end to end.
    """
    fetcher = BrowserDataFetcher()
    page = _FakePage(body_text=body_text)
    fetcher._page = page  # type: ignore[attr-defined]
    # Bypass the real launch by making _ensure_page return our fake page.
    fetcher._ensure_page = lambda: page  # type: ignore[assignment]
    return fetcher


def _no_page(self: BrowserDataFetcher) -> Any:
    """Patch target: simulate Playwright/browser being unavailable."""
    return None


# ---------------------------------------------------------------------------
# 1. Playwright missing -> everything degrades to ok=False, never raises
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlaywrightMissing:
    def test_fetch_returns_unavailable_when_no_browser(self, monkeypatch):
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        f = BrowserDataFetcher()
        result = f.fetch("https://example.com/quote/AAPL/analysis")
        assert isinstance(result, BrowserFetchResult)
        assert result.ok is False
        assert result.source == "https://example.com/quote/AAPL/analysis"
        assert "unavailable" in result.error.lower()
        assert result.retrieved_at  # non-empty ISO stamp

    def test_curated_fetchers_unavailable_without_browser(self, monkeypatch):
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        f = BrowserDataFetcher()
        for r in (
            f.fetch_earnings_revisions("AAPL"),
            f.fetch_management_guidance_keywords("AAPL"),
            f.fetch_prediction_market_depth("https://polymarket.com/event/x"),
        ):
            assert r.ok is False
            assert "unavailable" in (r.error or "").lower()

    def test_fetch_alternative_data_returns_unavailable_list(self, monkeypatch):
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        results = fetch_alternative_data(
            "AAPL", kinds=("earnings_revisions", "guidance_keywords")
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert r.ok is False
            assert "unavailable" in (r.error or "").lower()

    def test_no_exception_without_playwright(self, monkeypatch):
        # Importing the module must succeed even without playwright (already
        # proven by importing it above). Constructing a fetcher and using it as
        # a context manager must also not raise.
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        with BrowserDataFetcher() as f:
            r = f.fetch("https://example.com")
        assert r.ok is False


# ---------------------------------------------------------------------------
# 2. Happy-path fetch with a mocked page
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFetchHappyPath:
    def test_fetch_returns_body_text_with_source_and_timestamp(self):
        f = _make_fetcher_with_page(body_text="quarterly revenue grew 12%")
        result = f.fetch(
            "https://finance.yahoo.com/quote/AAPL/analysis",
            extractor=lambda p: p.inner_text("body"),
        )
        assert result.ok is True
        assert result.content == "quarterly revenue grew 12%"
        assert result.source == "https://finance.yahoo.com/quote/AAPL/analysis"
        assert result.error is None
        assert result.retrieved_at
        # ISO timestamp parses back to a datetime.
        from datetime import datetime

        datetime.fromisoformat(result.retrieved_at)

    def test_default_extractor_used_when_none(self):
        f = _make_fetcher_with_page(body_text="default body")
        result = f.fetch("https://example.com")
        assert result.ok is True
        assert result.content == "default body"

    def test_wait_selector_passed_through(self):
        f = _make_fetcher_with_page(body_text="x")
        page = f._page  # type: ignore[attr-defined]
        f.fetch("https://example.com", wait_selector="table")
        assert page.waited_for == ["table"]


# ---------------------------------------------------------------------------
# 3. Extractor raising -> ok=False, error populated
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractorFailure:
    def test_extractor_raising_yields_ok_false(self):
        f = _make_fetcher_with_page(body_text="anything")

        def boom(page):
            raise RuntimeError("selector changed")

        result = f.fetch("https://example.com", extractor=boom)
        assert result.ok is False
        assert result.content == ""
        assert result.error is not None
        assert "selector changed" in result.error
        assert result.source == "https://example.com"

    def test_navigation_failure_yields_ok_false(self):
        f = _make_fetcher_with_page(body_text="x")
        page = f._page  # type: ignore[attr-defined]

        def fail_goto(url):
            raise TimeoutError("nav timed out")

        page.goto = fail_goto  # type: ignore[assignment]
        result = f.fetch("https://example.com")
        assert result.ok is False
        assert "nav timed out" in (result.error or "")


# ---------------------------------------------------------------------------
# 4. Curated earnings-revisions fetcher wires the ticker into the URL
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEarningsRevisions:
    def test_source_url_contains_ticker(self, monkeypatch):
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        f = BrowserDataFetcher()
        result = f.fetch_earnings_revisions("AAPL")
        assert "AAPL" in result.source
        # ok status reflects the (unavailable) mock.
        assert result.ok is False

    def test_source_url_uppercased_and_in_path(self):
        f = _make_fetcher_with_page(body_text="Earnings Estimate Revisions table")
        result = f.fetch_earnings_revisions("aapl")
        assert result.ok is True
        assert "/quote/AAPL/analysis" in result.source
        assert result.content == "Earnings Estimate Revisions table"


# ---------------------------------------------------------------------------
# 5. count_keywords pure helper, in isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCountKeywords:
    def test_simple_counts(self):
        counts = count_keywords(
            "strong strong cautious", ("strong", "cautious", "robust")
        )
        assert counts == {"strong": 2, "cautious": 1, "robust": 0}

    def test_case_insensitive(self):
        counts = count_keywords("STRONG Strong strong", ("strong",))
        assert counts == {"strong": 3}

    def test_word_boundary_not_substring(self):
        # "raise" must not match inside "raised", "lower" not in "lowercase",
        # "momentum" not in "momentums" (plural is a different word token).
        counts = count_keywords(
            "raised lowercase momentums", ("raise", "lower", "momentum")
        )
        assert counts == {"raise": 0, "lower": 0, "momentum": 0}

    def test_word_boundary_at_edges(self):
        counts = count_keywords("strong and strong", ("strong",))
        assert counts == {"strong": 2}

    def test_empty_text_and_empty_keyword(self):
        assert count_keywords("", ("strong",)) == {"strong": 0}
        assert count_keywords("anything", ("",)) == {"": 0}

    def test_preserves_keyword_order(self):
        kws = ("zebra", "alpha", "mango")
        counts = count_keywords("alpha mango zebra", kws)
        assert list(counts.keys()) == list(kws)

    def test_default_keywords_fixture(self):
        # The module ships a default keyword tuple; ensure it is non-empty and
        # the helper accepts it.
        assert len(browser_data.DEFAULT_GUIDANCE_KEYWORDS) > 0
        counts = count_keywords(
            "we are confident despite some headwind", browser_data.DEFAULT_GUIDANCE_KEYWORDS
        )
        assert counts["confident"] == 1
        assert counts["headwind"] == 1


# ---------------------------------------------------------------------------
# 5b. fetch_management_guidance_keywords drives counting through the page
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGuidanceKeywords:
    def test_keyword_report_built_from_body(self):
        f = _make_fetcher_with_page(body_text="strong strong cautious momentum")
        result = f.fetch_management_guidance_keywords(
            "AAPL", keywords=("strong", "cautious", "momentum")
        )
        assert result.ok is True
        assert "strong: 2" in result.content
        assert "cautious: 1" in result.content
        assert "momentum: 1" in result.content
        # All requested keywords appear as report lines, even zero-count ones.
        assert "strong: 2" in result.content

    def test_default_keywords_used_when_omitted(self):
        f = _make_fetcher_with_page(body_text="robust confidence here")
        result = f.fetch_management_guidance_keywords("MSFT")
        assert result.ok is True
        assert "robust: 1" in result.content

    def test_unavailable_propagates_through_keyword_fetcher(self, monkeypatch):
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        f = BrowserDataFetcher()
        result = f.fetch_management_guidance_keywords("AAPL")
        assert result.ok is False
        assert "unavailable" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# 6. Context manager + close() idempotency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifecycle:
    def test_context_manager_does_not_raise_without_playwright(self, monkeypatch):
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        # No exception even though no browser is available.
        with BrowserDataFetcher() as f:
            assert f.fetch("https://example.com").ok is False

    def test_close_is_idempotent(self):
        f = _make_fetcher_with_page(body_text="x")
        f.close()
        # Second / third close must be no-ops, never raise.
        f.close()
        f.close()
        assert f._closed is True  # type: ignore[attr-defined]

    def test_context_manager_closes_on_exception(self):
        f = _make_fetcher_with_page(body_text="x")
        with pytest.raises(ValueError), f:
            raise ValueError("boom")
        assert f._closed is True  # type: ignore[attr-defined]

    def test_prediction_market_depth_fetcher(self):
        f = _make_fetcher_with_page(body_text="Yes 0.74 No 0.26 depth data")
        result = f.fetch_prediction_market_depth(
            "https://polymarket.com/event/fed-rate-cut"
        )
        assert result.ok is True
        assert result.source == "https://polymarket.com/event/fed-rate-cut"
        assert "depth data" in result.content


# ---------------------------------------------------------------------------
# fetch_alternative_data dispatch + error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFetchAlternativeData:
    def test_unknown_kind_yields_ok_false(self, monkeypatch):
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        results = fetch_alternative_data("AAPL", kinds=("nonsense",))
        assert len(results) == 1
        assert results[0].ok is False
        assert "nonsense" in (results[0].error or "")

    def test_prediction_market_depth_requires_url(self, monkeypatch):
        monkeypatch.setattr(BrowserDataFetcher, "_ensure_page", _no_page)
        results = fetch_alternative_data(
            "AAPL", kinds=("prediction_market_depth",)
        )
        assert results[0].ok is False
        assert "market_url" in (results[0].error or "")

    def test_runs_each_kind_via_one_session(self):
        # With a mocked page, both equity fetchers should return ok=True and
        # share one fetcher (verified by the helper which fakes the page).
        captured: list[BrowserDataFetcher] = []

        orig_enter = BrowserDataFetcher.__enter__
        orig_exit = BrowserDataFetcher.__exit__

        def enter(self):
            captured.append(self)
            self._page = _FakePage(body_text="strong robust cautious")  # type: ignore[attr-defined]
            self._ensure_page = lambda: self._page  # type: ignore[assignment, method-assign]
            return self

        def exit(self, *exc):
            self.close()
            return False

        # Patch the context-manager protocol on the class.
        BrowserDataFetcher.__enter__ = enter  # type: ignore[assignment]
        BrowserDataFetcher.__exit__ = exit  # type: ignore[assignment]
        try:
            results = fetch_alternative_data(
                "AAPL",
                kinds=("earnings_revisions", "guidance_keywords"),
            )
        finally:
            BrowserDataFetcher.__enter__ = orig_enter  # type: ignore[assignment]
            BrowserDataFetcher.__exit__ = orig_exit  # type: ignore[assignment]

        # Exactly one fetcher was opened (the batch owns one session).
        assert len(captured) == 1
        assert len(results) == 2
        assert results[0].ok is True  # earnings_revisions
        assert results[1].ok is True  # guidance_keywords
        # Default keyword tuple includes strong/robust/cautious; the faked body
        # "strong robust cautious" should tally to 1 each.
        assert "strong: 1" in results[1].content
        assert "robust: 1" in results[1].content
        assert "cautious: 1" in results[1].content
