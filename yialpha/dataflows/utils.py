import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime, timedelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point-in-time (PIT) guards for fundamental data.
#
# A backtest may only see data that was actually public on the simulated date.
# Two distinct lookahead leaks are guarded here:
#
#   1. Financial statements (balance sheet / cash flow / income statement).
#      yfinance and Alpha Vantage key these by ``fiscalDateEnding`` (the fiscal
#      PERIOD end), but a period ending e.g. Sep 30 is NOT public on Oct 1 --
#      the 10-Q is filed days to weeks later (SEC large accelerated filers:
#      10-K ~60 days, 10-Q ~40 days after period end). Filtering on
#      ``fiscalDateEnding <= curr_date`` lets a backtest read a report the
#      market could not yet have seen. ``is_filing_public`` adds a filing lag
#      so a period is visible only once it was plausibly filed.
#
#   2. Overview snapshots (yfinance ``.info`` / Alpha Vantage ``OVERVIEW``).
#      These are single current-point values (PE, marketCap, EPS, beta) with NO
#      date dimension -- they are always *today's* values. Surfacing them on a
#      past backtest date leaks the future wholesale. ``overview_would_leak_future``
#      flags this so the vendor can refuse (the router turns the resulting
#      NoMarketDataError into the NO_DATA_AVAILABLE sentinel the fundamentals
#      analyst is grounded to handle).
#
# These are correctness fixes, not opt-in enhancements: they apply by default.
# The filing lag is tunable via env for conservatism (set to 0 to revert to the
# old lookahead-leaking behaviour).
# ---------------------------------------------------------------------------
_FILING_LAG_ENV = os.environ.get("YIALPHA_FUNDAMENTALS_FILING_LAG_DAYS")
try:
    FUNDAMENTALS_FILING_LAG_DAYS: int = (
        int(_FILING_LAG_ENV) if _FILING_LAG_ENV not in (None, "") else 45
    )
except ValueError:
    FUNDAMENTALS_FILING_LAG_DAYS = 45


def is_filing_public(
    fiscal_period_end,
    curr_date: str,
    lag_days: int = FUNDAMENTALS_FILING_LAG_DAYS,
) -> bool:
    """True iff a report for the fiscal period ending ``fiscal_period_end`` was
    plausibly public by ``curr_date`` (period end + ``lag_days``).

    Conservative on parse failure: if we cannot prove a report was public, drop
    it rather than risk lookahead. ``curr_date`` empty/None means live mode (no
    as-of constraint) -> keep everything.
    """
    if not curr_date:
        return True
    try:
        period_end = datetime.strptime(str(fiscal_period_end)[:10], "%Y-%m-%d")
        as_of = datetime.strptime(str(curr_date)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return False
    return period_end + timedelta(days=lag_days) <= as_of


def live_anchor_dates() -> tuple[date, date]:
    """The two date anchors a LIVE run's label may legitimately carry.

    The interactive CLI defaults its analysis date to the HOST-LOCAL calendar
    date (``datetime.now()``), while the crypto/perp data layer anchors on UTC
    end-to-end (klines staleness, ``_futures_data_window``'s "now",
    ``quote_fx.usdt_usd_as_of``, the vision publication cap — see
    :func:`yialpha.dataflows.stockstats_utils._utc_today` for the same trap).
    On any host off UTC the two anchors disagree for one calendar-window a
    day (00:00–08:00 on UTC+8), and a single-anchor live check misclassifies
    whichever label the other half of the pipeline produced. The two dates
    differ by at most one day, so accepting BOTH keeps the live set exact.
    """
    return datetime.now().date(), datetime.now(UTC).date()


def is_historical_date(curr_date: str | None) -> bool:
    """Return whether an explicit analysis date is not the live/current date.

    This is the shared gate for sources that only expose a current snapshot and
    have no trustworthy historical/as-of parameter (social feeds, prediction
    markets, rolling 24-hour tickers, and selected positioning endpoints).
    A future label must not receive today's snapshot either, so every valid
    explicit date other than today takes the causal/date-bounded branch.

    "Today" is BOTH live anchors from :func:`live_anchor_dates` (host-local
    AND UTC): a live perp run whose label is the UTC current date must not
    lose its live-only components (funding/premium/depth/ADL bundle legs,
    live REST tools, web search) just because the host calendar already
    rolled to the next date — and vice versa for a local-dated label. Any
    other date — past or future — remains historical.

    Only empty/None means live mode (no as-of constraint). A NON-EMPTY but
    unparseable value (e.g. "2026/08/01" — a form an LLM can emit) cannot be
    proven to be today, so it takes the historical branch: the current-snapshot
    sources degrade rather than leak today's state into a nominal backtest —
    the same fail-open-*only*-for-live policy as :func:`clamp_end_date`. (It
    used to return False here, serving live snapshots to malformed dates that
    no caller validated.)
    """
    if not curr_date:
        return False
    try:
        as_of = datetime.strptime(str(curr_date)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return True
    return as_of not in live_anchor_dates()


def overview_would_leak_future(curr_date: str | None) -> bool:
    """True iff ``curr_date`` is an explicit past date, for which a vendor's
    current-point overview snapshot would be misdated or leak information.

    Such snapshots (yfinance ``.info`` / Alpha Vantage ``OVERVIEW``) carry no
    date dimension, so they are only valid when ``curr_date`` is empty (live,
    no as-of constraint) or exactly today.
    """
    return is_historical_date(curr_date)


# ---------------------------------------------------------------------------
# Analysis-date PIT cutoff for the tool execution layer.
#
# The stock/crypto OHLCV tools (get_stock_data, get_binance_klines, ...) are
# called by the LLM with (symbol, start_date, end_date) and carry NO analysis
# date parameter — the LLM picks end_date from its prompt context. The real
# clock may be ahead of the analysis date (a backtest for 2020 run today), so a
# vendor that honors whatever window the LLM supplies would return rows after
# the analysis date, leaking the future into the backtest.
#
# We pin the analysis date in a ContextVar at the start of each graph run
# (:meth:`YiAlphaGraph._run_graph`). The vendor layer reads it via
# :func:`get_analysis_date` and clamps its fetch window with
# :func:`clamp_end_date`. This is transparent to the tool signatures (no LLM-
# facing parameter changes) and crosses the ThreadPoolExecutor boundary the
# same way the config ContextVar does (the batch runner copies the context into
# each worker via ``submit_with_context``). Live mode (no analysis date pinned)
# is a no-op pass-through, so today's live runs are byte-identical.
# ---------------------------------------------------------------------------
_analysis_date_var: ContextVar[str | None] = ContextVar("yialpha_analysis_date", default=None)


def set_analysis_date(curr_date: str | None) -> None:
    """Pin the analysis date for the current graph run / batch worker thread.

    ``None`` (or an empty string) clears it — live mode, no as-of constraint.
    The value is expected in ``YYYY-MM-DD`` form; non-conforming values are
    stored as-is but :func:`clamp_end_date` treats unparseable analysis dates
    as live (no clamp), failing open only for the live path.
    """
    _analysis_date_var.set(curr_date or None)


def get_analysis_date() -> str | None:
    """The analysis date pinned for this run/thread, or ``None`` for live mode."""
    return _analysis_date_var.get()


@contextmanager
def pinned_analysis_date(curr_date: str | None):
    """Pin the analysis date for the duration of a streamed graph run.

    ``_run_graph`` pins/clears the ContextVar around propagate() manually; the
    interactive CLI streams ``graph.stream`` directly instead, so it enters
    this manager alongside its run-lock to get the identical PIT contract —
    including clearing on an exception, so a crashed run cannot leave the
    clamp pinned into the next one.
    """
    set_analysis_date(str(curr_date) if curr_date else None)
    try:
        yield
    finally:
        set_analysis_date(None)


def clamp_end_date(end_date: str | None, curr_date: str | None = None) -> str | None:
    """Return ``end_date`` capped to the analysis date ``curr_date``.

    The third PIT guard (after :func:`is_filing_public` and
    :func:`overview_would_leak_future`): a vendor fetch window must not extend
    past the analysis date. When ``curr_date`` is empty/``None`` (live mode)
    or either date is unparseable, the value is returned unchanged — fail open
    *only* for live, where there is no as-of constraint to violate. On a
    parseable historical analysis date, an ``end_date`` strictly after it is
    clamped down to it so the vendor never returns rows the backtest could not
    have seen.
    """
    if not curr_date or not end_date:
        return end_date
    try:
        as_of = datetime.strptime(str(curr_date)[:10], "%Y-%m-%d")
        requested = datetime.strptime(str(end_date)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return end_date
    if requested <= as_of:
        return end_date
    logger.debug(
        "PIT clamp: end_date %s past analysis date %s; clamped", end_date, curr_date,
    )
    return as_of.strftime("%Y-%m-%d")


def current_pit_end(end_date: str | None) -> str | None:
    """Convenience: clamp ``end_date`` to the run's pinned analysis date.

    Vendors that fetch by date window call this instead of plumbing a
    ``curr_date`` argument through every layer. Returns ``end_date`` unchanged
    in live mode (no analysis date pinned).
    """
    return clamp_end_date(end_date, get_analysis_date())


# Tickers can contain letters, digits, dot, dash, underscore, caret
# (index symbols like ^GSPC), equals (futures like GC=F), and plus
# (forex/CFD symbols like XAUUSD+). None of these enable directory
# traversal, so the value never escapes a containing directory when
# interpolated into a path. Anything else is rejected.
_TICKER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-\^=+]+$")


def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
    """Validate ``value`` is safe to interpolate into a filesystem path.

    Tickers come from user CLI input or from LLM tool calls, both of which
    can be influenced by attacker-controlled content (e.g. prompt injection
    embedded in fetched news). Without validation, a value like
    ``"../../../etc/foo"`` flows into ``os.path.join`` / ``Path /`` and
    escapes the configured cache, checkpoint, or results directory.

    Returns ``value`` unchanged when it matches the allowed pattern; raises
    ``ValueError`` otherwise.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if not _TICKER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"ticker contains characters not allowed in a filesystem path: {value!r}"
        )
    # The regex above allows '.', so values like '.', '..', '...' would pass,
    # and as a path component they traverse the parent directory. Reject any
    # value that's only dots.
    if set(value) == {"."}:
        raise ValueError(f"ticker cannot consist solely of dots: {value!r}")
    return value


def proxy_map() -> dict[str, str]:
    """requests-style proxy dict for US/quote data sources, read at call time.

    ``http`` <- ``HTTP_PROXY`` else ``ALL_PROXY``; ``https`` <- ``HTTPS_PROXY``
    else ``ALL_PROXY``. The ``ALL_PROXY`` fallback matters: a user who sets
    only ``ALL_PROXY`` (a single config knob) must get proxied traffic on
    *every* US source, not just the ones that happened to read it -- otherwise
    one vendor hangs or leaks the real IP while another is proxied. Entries
    whose resolved value is ``None`` (no relevant env var set) are omitted so
    the dict matches ``requests``' ``proxies: Mapping[str, str]`` contract;
    an empty dict means "no proxy", identical to requests' default behaviour.

    Domestic sources that must bypass the proxy (e.g. Eastmoney) do NOT use
    this -- they disable env merging via ``Session(trust_env=False)`` instead.
    """
    raw = {
        "http": os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY"),
        "https": os.environ.get("HTTPS_PROXY") or os.environ.get("ALL_PROXY"),
    }
    return {k: v for k, v in raw.items() if v is not None}


def safe_xml_root(raw: bytes | str, *, source: str):
    """Parse untrusted XML with DOCTYPE/ENTITY refused before parsing.

    Security-scan acceptance fix (2026-09-03, Mimosa findings on
    reddit.py / sec_ownership.py): stdlib ``xml.etree`` never resolves
    EXTERNAL entities, but it DOES expand internal ones — a crafted feed
    could trigger quadratic/billion-laughs entity expansion. Both expansion
    and external-entity resolution ride on a DTD, so any document whose
    prolog/body contains ``<!DOCTYPE`` or ``<!ENTITY`` (case-insensitive) is
    REFUSED outright rather than parsed. The legitimate feeds this project
    consumes (reddit.com RSS, SEC EDGAR Form 4 XML over HTTPS) carry
    neither; a document that does is either hostile or malformed, and both
    deserve the same refusal.
    """
    import xml.etree.ElementTree as ET

    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    lowered = text.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ValueError(
            f"{source}: refused XML containing a DTD/ENTITY declaration "
            "(entity-expansion guard); not parsed"
        )
    return ET.fromstring(text)
