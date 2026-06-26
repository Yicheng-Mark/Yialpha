"""Browser-driven alternative-data augmentation (Phase 2d).

The framework's API data sources (yfinance / FRED / Polymarket) give breadth but
miss forward-looking / alternative qualitative signals the analyst agents can
read: earnings-revision momentum, management-guidance keyword tone, and
prediction-market order-book depth. This module fetches those via a headless
Microsoft Edge browser driven by Playwright, then returns timestamped,
source-attributed text so the consuming prompt stays point-in-time honest.

Design invariants (every code path must preserve these):

* **Lazy / optional Playwright.** Playwright is imported *inside* methods, never
  at module top level. Importing this module must succeed even when Playwright
  is not installed, and the rest of the framework must never depend on it. A
  missing Playwright (or a browser-launch failure) never raises; it returns a
  structured ``BrowserFetchResult`` with ``ok=False`` and an explanatory error.
* **Standalone.** Channel, headless flag, and navigation timeout are constructor
  parameters with sensible defaults — this module deliberately does *not* couple
  to the project's global config system, mirroring the other self-contained
  dataflows vendors.
* **PIT-honest.** Every returned result carries an ISO retrieval timestamp and
  the source URL, so the analyst prompt can record *when* and *from where* a
  qualitative reading was taken.
* **Best-effort scraping.** Concrete selectors / page URLs are site-specific and
  may need maintenance when a site changes its DOM; all extraction is wrapped so
  a selector change degrades to ``ok=False`` rather than a crash.

The curated fetchers (``fetch_earnings_revisions``,
``fetch_management_guidance_keywords``, ``fetch_prediction_market_depth``) all
build on the generic :meth:`BrowserDataFetcher.fetch` engine.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Default keywords for management-guidance tone analysis. Tuned for the kind of
# directional / hedging language that moves around earnings calls. Callers may
# pass their own tuple to specialize per-sector vocabulary.
DEFAULT_GUIDANCE_KEYWORDS: tuple[str, ...] = (
    "strong",
    "robust",
    "confident",
    "cautious",
    "headwind",
    "uncertainty",
    "accelerating",
    "momentum",
    "guidance",
    "raise",
    "lower",
)


@dataclass
class BrowserFetchResult:
    """Structured outcome of a single browser fetch.

    Attributes:
        ok: Whether the fetch succeeded and ``content`` is populated.
        source: The URL that was fetched (recorded for PIT honesty).
        retrieved_at: ISO-8601 timestamp of when the page was read (runtime).
        content: Extracted text / table / keyword report.
        error: Human-readable failure reason when ``ok`` is ``False``.
    """

    ok: bool
    source: str
    retrieved_at: str
    content: str
    error: str | None = None


def count_keywords(text: str, keywords: Iterable[str]) -> dict[str, int]:
    """Count whole-word, case-insensitive occurrences of each keyword in ``text``.

    Kept as a pure module-level helper so the counting logic can be unit-tested
    in isolation (driving it through a mocked browser would couple the test to
    the DOM path). Word-boundary matching avoids ``"raise"`` matching inside
    ``"raised"`` twice or ``"lower"`` inside ``"lowercase"``.

    Args:
        text: The body to scan (e.g. an earnings-call transcript summary).
        keywords: The terms to tally.

    Returns:
        ``{keyword: count}`` for every keyword in ``keywords`` (zero when
        absent), preserving the iteration order of ``keywords`` for a stable
        report.
    """
    lowered = (text or "").lower()
    counts: dict[str, int] = {}
    for kw in keywords:
        # Manual word-boundary scan rather than re.findall: avoids needing to
        # escape arbitrary user-supplied keywords into a regex, and is plenty
        # fast for the short bodies this reads.
        needle = kw.lower()
        if not needle:
            counts[kw] = 0
            continue
        total = 0
        start = 0
        while True:
            idx = lowered.find(needle, start)
            if idx < 0:
                break
            before = lowered[idx - 1] if idx > 0 else " "
            after = lowered[idx + len(needle)] if idx + len(needle) < len(lowered) else " "
            if not (before.isalnum()) and not (after.isalnum()):
                total += 1
            start = idx + len(needle)
        counts[kw] = total
    return counts


class BrowserDataFetcher:
    """Owns one Playwright browser session.

    Use as a context manager (``with BrowserDataFetcher() as f: ...``) so the
    browser is always torn down, or call :meth:`close` explicitly. ``close`` is
    idempotent. If Playwright is missing or the browser fails to launch, every
    fetch returns a structured ``ok=False`` result rather than raising.
    """

    def __init__(
        self,
        channel: str = "msedge",
        headless: bool = True,
        nav_timeout_ms: int = 30000,
    ) -> None:
        self.channel = channel
        self.headless = headless
        self.nav_timeout_ms = nav_timeout_ms
        # Lazily populated on first fetch; torn down by close().
        self._pw_handle: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Context-manager + lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> BrowserDataFetcher:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False  # never swallow exceptions

    def close(self) -> None:
        """Tear down the browser session. Idempotent and never raises.

        Errors during teardown are logged and swallowed: a half-dead browser
        must not break the analyst's data-collection step.
        """
        if self._closed:
            return
        self._closed = True
        for attr in ("_page", "_context", "_browser", "_pw_handle"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001 - teardown must be best-effort
                    logger.debug("BrowserDataFetcher close error on %s: %s", attr, exc)
            setattr(self, attr, None)

    # ------------------------------------------------------------------
    # Lazy Playwright plumbing
    # ------------------------------------------------------------------
    def _get_playwright(self) -> Any:
        """Import Playwright lazily and start the sync handle.

        Returns the started ``sync_playwright()`` context object (already
        entered) on success, or ``None`` if Playwright is not installed or the
        handle fails to start. Returning ``None`` is the single degradation
        point every fetcher checks before touching the network.
        """
        if self._pw_handle is not None:
            return self._pw_handle
        try:
            # Lazy import: importing playwright at module top would make this
            # module — and therefore the whole dataflows package — depend on an
            # optional dependency. Keep it inside the method.
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.info(
                "Playwright is not installed; browser data augmentation is "
                "unavailable. Install `playwright` and run `playwright install` "
                "to enable it."
            )
            return None
        try:
            self._pw_handle = sync_playwright().start()
        except Exception as exc:  # noqa: BLE001 - any startup failure -> unavailable
            logger.warning("Playwright failed to start: %s", exc)
            self._pw_handle = None
            return None
        return self._pw_handle

    def _ensure_page(self) -> Any:
        """Lazily launch the Edge browser and return a ready ``Page``.

        Returns ``None`` if Playwright is unavailable or the browser will not
        launch — callers must treat ``None`` as "unavailable".
        """
        if self._page is not None:
            return self._page
        pw = self._get_playwright()
        if pw is None:
            return None
        try:
            # channel="msedge" drives the installed Microsoft Edge binary;
            # headless keeps it off-screen for production / batch use.
            self._browser = pw.chromium.launch(
                channel=self.channel, headless=self.headless
            )
            self._context = self._browser.new_context()
            self._page = self._context.new_page()
            self._page.set_default_timeout(self.nav_timeout_ms)
        except Exception as exc:  # noqa: BLE001 - launch failure -> unavailable
            logger.warning("Browser launch failed: %s", exc)
            self._page = None
            self.close()
            return None
        return self._page

    # ------------------------------------------------------------------
    # Generic fetch engine
    # ------------------------------------------------------------------
    def fetch(
        self,
        url: str,
        extractor: Callable[[Any], str] | None = None,
        wait_selector: str | None = None,
    ) -> BrowserFetchResult:
        """Navigate to ``url`` and extract text.

        Args:
            url: The page to fetch.
            extractor: ``extractor(page) -> str`` defining site-specific
                scraping. Defaults to ``page.inner_text("body")``.
            wait_selector: Optional CSS selector to wait for before extracting,
                so dynamically-rendered content is present.

        Returns:
            A :class:`BrowserFetchResult`. Any failure — Playwright missing,
            browser launch failure, navigation timeout, extractor exception —
            yields ``ok=False`` with ``error`` populated, never an exception.
        """
        retrieved_at = datetime.now().isoformat()
        page = self._ensure_page()
        if page is None:
            return BrowserFetchResult(
                ok=False,
                source=url,
                retrieved_at=retrieved_at,
                content="",
                error="Playwright/browser unavailable",
            )
        try:
            page.goto(url)
            if wait_selector:
                page.wait_for_selector(wait_selector)
            extract = extractor if extractor is not None else _default_extractor
            content = extract(page)
        except Exception as exc:  # noqa: BLE001 - any failure -> structured result
            logger.warning("Browser fetch failed for %s: %s", url, exc)
            return BrowserFetchResult(
                ok=False,
                source=url,
                retrieved_at=retrieved_at,
                content="",
                error=f"{type(exc).__name__}: {exc}",
            )
        return BrowserFetchResult(
            ok=True,
            source=url,
            retrieved_at=retrieved_at,
            content=content,
            error=None,
        )

    # ------------------------------------------------------------------
    # Curated fetchers for the roadmap's named alternative signals
    # ------------------------------------------------------------------
    def fetch_earnings_revisions(self, ticker: str) -> BrowserFetchResult:
        """Analyst estimate-revision page for ``ticker``.

        Targets the Yahoo Finance analysis page
        ``https://finance.yahoo.com/quote/{TICKER}/analysis`` which surfaces
        earnings-estimate revisions (consensus up/down moves). Returns the
        revisions-table text. The source URL is recorded on the result.

        Note: the Yahoo DOM is best-effort; if the page restructures, extraction
        degrades to ``ok=False`` rather than crashing.
        """
        url = (
            f"https://finance.yahoo.com/quote/{ticker.upper()}/analysis"
        )
        return self.fetch(url)

    def fetch_management_guidance_keywords(
        self,
        ticker: str,
        keywords: tuple[str, ...] = DEFAULT_GUIDANCE_KEYWORDS,
    ) -> BrowserFetchResult:
        """Pull a recent earnings-call / transcript summary and tally tone words.

        Fetches a transcript-summary page for ``ticker`` (Motley Fool's
        per-ticker earnings page is a reasonable, openly-readable default), then
        counts whole-word occurrences of each guidance/tone keyword in the body.
        ``content`` is a small ``"keyword: N"`` report — a compact qualitative
        signal the analyst prompt can read directly. The source URL is recorded.

        Args:
            ticker: Equity symbol, e.g. ``"AAPL"``.
            keywords: Tone/direction words to tally. Defaults to
                :data:`DEFAULT_GUIDANCE_KEYWORDS`.
        """
        url = (
            f"https://www.fool.com/quote/{ticker.lower()}/"
        )
        result = self.fetch(url)
        if not result.ok:
            return result
        counts = count_keywords(result.content, keywords)
        report_lines = [f"{kw}: {counts[kw]}" for kw in keywords]
        return BrowserFetchResult(
            ok=True,
            source=result.source,
            retrieved_at=result.retrieved_at,
            content="\n".join(report_lines),
            error=None,
        )

    def fetch_prediction_market_depth(self, market_url: str) -> BrowserFetchResult:
        """Fetch order-book / depth detail for a Polymarket (or similar) market.

        The Polymarket Gamma API gives a top-line implied price; the browser
        path instead reads the human-facing market page, which exposes the
        order-book depth that a thin top-line price hides. ``content`` is the
        page's body text.

        Args:
            market_url: Full URL of the prediction-market page.
        """
        return self.fetch(market_url)


def _default_extractor(page: Any) -> str:
    """Default extraction: the page's visible body text."""
    return page.inner_text("body")


# ---------------------------------------------------------------------------
# Module-level convenience entry point
# ---------------------------------------------------------------------------

#: Map of supported ``kinds`` names to the :class:`BrowserDataFetcher` method
#: that implements them. ``fetch_alternative_data`` dispatches through this so
#: analysts can request signals by name without binding to method objects.
_FETCHERS: dict[str, str] = {
    "earnings_revisions": "fetch_earnings_revisions",
    "guidance_keywords": "fetch_management_guidance_keywords",
    "prediction_market_depth": "fetch_prediction_market_depth",
}


def fetch_alternative_data(
    ticker: str,
    kinds: tuple[str, ...] = ("earnings_revisions", "guidance_keywords"),
    market_url: str | None = None,
    **fetcher_kwargs: Any,
) -> list[BrowserFetchResult]:
    """Open one fetcher, run the named alternative-signal fetchers, close it.

    This is the convenience entry point an analyst integration would call: it
    owns the browser lifecycle for the batch so the caller does not have to. If
    Playwright is unavailable every requested kind yields an ``ok=False``
    result — the function never raises.

    Args:
        ticker: Equity symbol for the equity-style fetchers.
        kinds: Names of fetchers to run, drawn from
            ``("earnings_revisions", "guidance_keywords",
            "prediction_market_depth")``. Unknown names produce a single
            ``ok=False`` result explaining the bad name.
        market_url: Required when ``"prediction_market_depth"`` is requested;
            ignored otherwise.
        **fetcher_kwargs: Forwarded to the :class:`BrowserDataFetcher`
            constructor (e.g. ``headless=False`` for an interactive run).

    Returns:
        One :class:`BrowserFetchResult` per requested kind, in order.
    """
    results: list[BrowserFetchResult] = []
    with BrowserDataFetcher(**fetcher_kwargs) as fetcher:
        for kind in kinds:
            method_name = _FETCHERS.get(kind)
            if method_name is None:
                results.append(
                    BrowserFetchResult(
                        ok=False,
                        source="",
                        retrieved_at=datetime.now().isoformat(),
                        content="",
                        error=(
                            f"Unknown alternative-data kind {kind!r}; "
                            f"expected one of {sorted(_FETCHERS)}"
                        ),
                    )
                )
                continue
            method = getattr(fetcher, method_name)
            if kind == "prediction_market_depth":
                if not market_url:
                    results.append(
                        BrowserFetchResult(
                            ok=False,
                            source="",
                            retrieved_at=datetime.now().isoformat(),
                            content="",
                            error=(
                                "prediction_market_depth requires a market_url "
                                "argument"
                            ),
                        )
                    )
                    continue
                results.append(method(market_url))
            else:
                results.append(method(ticker))
    return results
