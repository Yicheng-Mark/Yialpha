"""Per-run ledger context — the ContextVar seam behind record-stage wiring.

Mirrors the contract of :mod:`yialpha.dataflows.quality` /
:mod:`yialpha.dataflows.run_scope`: the graph runner binds ONE
:class:`LedgerRunContext` in the PARENT context around each run
(:func:`set_ledger_run_context` ... :func:`reset_ledger_run_context`);
every LangGraph node task inherits the same frozen value, so analyst-side
record-stage wiring (blind-prediction capture, evidence rows) can answer
"which run am I recording for?" without threading a run id through state.

Record-stage invariant: everything keyed off this module is fail-soft. With
no context bound (unit tests, direct node calls) — or with the
``prediction_ledger`` config flag off — :func:`record_evidence_block`
degrades to a no-op: no ledger write, no exception, and never a change to
report/decision bytes. A ledger-side rejection is disclosed, never silent:
a point-in-time guard refusal (an injection whose ``available_at``
postdates the run's ``analysis_as_of``) logs its own explicit WARNING
naming the two timestamps, and any other ledger failure is logged with the
traceback — for the same reason: evidence recording must never abort the
run it is describing, but a refused injection must be visible.

Evidence mapping table — every ``[EXTERNAL EVIDENCE]`` injection site this
module serves, as (source / category / symbol / scope / replayability).
Replayability is deliberately conservative: a block that MIXES
archive-derivable content (klines, funding history, OI, vision-archive
series, SEC filings) with live-snapshot content (depth, ADL, premium
snapshot, current-feed posts, web digests, live valuation) is tagged
``LIVE_ONLY``, because "replayable" must mean the WHOLE view can be
reconstructed later — one non-replayable leg is enough to break that:

* market_analyst perp market bundle → ``perp_market_bundle`` /
  ``binance_perp`` / contract symbol / ``CONTRACT`` — ``LIVE_ONLY`` on
  live runs (depth bands, ADL and the premium snapshot are live views);
  ``PIT_REPLAYABLE`` on historical replay runs, where the bundle carries
  only the date-bounded kline bases.
* market_analyst regime block (V2.2, perp runs with ``regime_state``) →
  ``regime_state`` / ``binance_perp`` / contract symbol / ``CONTRACT`` —
  ``PIT_REPLAYABLE`` on historical replay runs (only date-bounded legs
  feed the regime there) and ``LIVE_ONLY`` on live runs (the live order
  book feeds the liquidity legs).
* fundamentals_analyst fundamentals bundle → ``fundamentals_bundle`` /
  ``fundamental_data`` / underlying equity symbol (perp runs) or the
  ticker itself / ``UNDERLYING`` — ``LIVE_ONLY`` (SEC statements are
  archival but the merged overview includes live valuation views, so the
  mixed block is conservatively live-only).
* news_analyst company digest (stock-perp dual coverage) →
  ``news_company_digest`` / ``news_data`` / underlying equity symbol /
  ``UNDERLYING`` — ``LIVE_ONLY`` (vendor news search is a
  current-retrieval API, not an archive).
* news_analyst contract digest (exact-symbol open-web search) →
  ``news_contract_digest`` / ``news_data`` / contract symbol /
  ``CONTRACT`` — ``LIVE_ONLY`` (Tavily web digest, no as-of boundary).
* sentiment_analyst news block → ``sentiment_news`` / ``news_data`` /
  underlying equity symbol / ``UNDERLYING`` on stock-perp runs
  (company-news tier keeps its UNDERLYING identity) else the contract
  symbol / ``CONTRACT`` — ``LIVE_ONLY``.
* sentiment_analyst StockTwits block → ``sentiment_stocktwits`` /
  ``social`` / the cashtag symbol (the underlying equity) /
  ``UNDERLYING`` — ``LIVE_ONLY``.
* sentiment_analyst Reddit block → ``sentiment_reddit`` / ``social`` /
  the queried symbol / ``CONTRACT`` on perp runs else ``UNDERLYING`` —
  ``LIVE_ONLY``.
* sentiment_analyst Binance Square block → ``binance_square`` /
  ``social`` / contract symbol / ``CONTRACT`` — ``LIVE_ONLY``
  (current-feed snapshot).
* positioning_analyst positioning bundle (V2.3, perp runs with
  ``positioning_split``) → ``positioning_bundle`` / ``binance_perp`` /
  contract symbol / ``CONTRACT`` — the positioning half of the same
  fetched bundle as ``perp_market_bundle``, so the replayability mapping
  mirrors it exactly: ``LIVE_ONLY`` on live runs (depth/ADL/premium are
  live views), ``PIT_REPLAYABLE`` on historical replays.
* positioning_analyst on-chain flows block (V2.3, ``onchain_evidence``) →
  ``onchain_flows`` / ``onchain`` (NEW category — on-chain network-activity
  charts; no quality-ledger category exists for them, and ``social`` would
  mislabel chart data) / contract symbol / ``CONTRACT`` —
  ``PIT_REPLAYABLE``: the blockchain.info charts endpoint is
  historical-capable and the block's points are PIT-filtered to the run's
  as-of day, with ``available_at`` taken from the latest point timestamp
  actually used.

Categories align with the quality-ledger vocabulary
(:data:`yialpha.dataflows.interface.TOOLS_CATEGORIES`) wherever one exists
(``binance_perp``, ``fundamental_data``, ``news_data``); the social
sources sit outside the router and use ``social``. Macro/global blocks
would use ``MACRO`` — none of today's sites is macro-scoped.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LedgerRunContext:
    """The one run every record-stage write in this context belongs to.

    ``analysis_as_of`` is the PIT anchor: evidence recorded for the run is
    checked against it by the ledger (a date-only value admits any time on
    that UTC day). ``instrument_class`` is the routing verdict
    (``equity`` / ``crypto_spot`` / ``stock_perp`` / ``pure_crypto_perp`` /
    ``unknown_perp``) or ``None`` when the caller did not classify.
    ``regime_id`` (V2.2, additive) names the run's ``RegimeState`` row —
    ``None`` while the regime stage is off, uncomputable, or not yet bound.
    """

    run_id: str
    ticker: str
    asset_type: str
    instrument_class: str | None
    analysis_as_of: str
    regime_id: str | None = None


_RUN_CONTEXT: ContextVar[LedgerRunContext | None] = ContextVar(
    "yialpha_ledger_run_context", default=None
)


def set_ledger_run_context(
    run_id: str,
    ticker: str,
    asset_type: str,
    instrument_class: str | None = None,
    analysis_as_of: str = "",
    regime_id: str | None = None,
) -> LedgerRunContext:
    """Force-bind the run context (graph runner, one call per run).

    Returns the bound value so callers can log it. Batch/run orchestration
    calls this in the PARENT context before the graph runs; node tasks
    inherit the frozen value. The V2.2 regime stage re-binds mid-run with
    ``regime_id`` once the regime is computed — force-set semantics, so the
    re-bind is just the same call with the extra key.
    """
    context = LedgerRunContext(
        run_id=str(run_id),
        ticker=str(ticker),
        asset_type=str(asset_type),
        instrument_class=instrument_class,
        analysis_as_of=analysis_as_of,
        regime_id=regime_id,
    )
    _RUN_CONTEXT.set(context)
    return context


def ensure_ledger_run_context(
    run_id: str,
    ticker: str,
    asset_type: str,
    instrument_class: str | None = None,
    analysis_as_of: str = "",
    regime_id: str | None = None,
) -> None:
    """Bind the run context only when absent (retried pipelines keep the
    original — the run row, and everything anchored to it, is never
    rewritten mid-flight)."""
    if _RUN_CONTEXT.get() is None:
        set_ledger_run_context(
            run_id, ticker, asset_type, instrument_class, analysis_as_of, regime_id
        )


def reset_ledger_run_context() -> None:
    """Drop the bound run context (run end). The next run starts cold."""
    _RUN_CONTEXT.set(None)


def current_ledger_run_context() -> LedgerRunContext | None:
    """The run context bound in the calling context, or ``None``.

    ``None`` is the record-stage "not recording" signal: every consumer
    degrades to a no-op rather than guessing a run to attribute writes to.
    """
    return _RUN_CONTEXT.get()


def record_evidence_block(
    source: str,
    category: str,
    symbol: str | None,
    scope: str,
    payload_text: str,
    *,
    replayability: str,
    event_time: str | None = None,
    quality_status: str | None = None,
    source_url: str | None = None,
    available_at: str | None = None,
) -> None:
    """Append one evidence row for an ``[EXTERNAL EVIDENCE]`` block.

    No-op when no run context is bound or the ``prediction_ledger`` flag is
    off. ``payload_text`` is hashed by the ledger, so re-injecting the same
    block into the same run (tool-loop re-entries, prompt replays) dedupes
    to one row. ``analysis_as_of`` comes from the bound context — the PIT
    guard lives in :func:`yialpha.ledger.evidence.record_evidence` and a
    violation (injection available after the run's as-of instant) is
    explicitly disclosed as a WARNING here — never raised into the data
    path, and never silently swallowed.

    ``available_at``: an explicit value is the caller's real data timestamp
    (external fetched market data keeps its true availability — it is never
    backfilled to the past to slip past the guard; a future-dated external
    row is PIT-rejected and disclosed). ``None`` means the block is
    RUN-DERIVED (sentiment/news digests, analysis output): it became
    available when the run analyzed, so it defaults to the run's anchored
    ``analysis_as_of`` — NOT wall-clock record time, which one instant after
    the anchor on a live run would PIT-reject against the run's own clock
    drift. See the module docstring for the site → source/replayability/
    scope mapping.
    """
    context = current_ledger_run_context()
    if context is None:
        return
    run_id = context.run_id
    # R4 time contract: run-derived evidence anchors to the run's as-of;
    # explicit values pass through untouched.
    available = available_at if available_at is not None else context.analysis_as_of
    try:
        from yialpha.dataflows.config import get_config  # local: avoid import cycle

        if not get_config().get("prediction_ledger"):
            return
        from yialpha.ledger.evidence import record_evidence

        record_evidence(
            run_id=run_id,
            source=source,
            category=category,
            symbol=symbol,
            scope=scope,
            payload=payload_text,
            source_url=source_url,
            event_time=event_time,
            replayability=replayability,
            quality_status=quality_status,
            analysis_as_of=context.analysis_as_of,
            available_at=available,
        )
    except Exception as exc:  # noqa: BLE001 -- record stage must never abort a run
        from yialpha.ledger.evidence import PITEvidenceViolation

        if isinstance(exc, PITEvidenceViolation):
            # Explicit disclosure, never a silent swallow: the point-in-time
            # guard REFUSED this block because its available_at postdates the
            # run's anchored as-of. Evidence recording stays fail-soft (the
            # run it describes must not abort), but the refusal is surfaced.
            logger.warning(
                "record_evidence_block(%s) for run %s: point-in-time guard "
                "rejected the block (available_at %s postdates the run's "
                "analysis_as_of %s); block not recorded",
                source,
                run_id,
                available,
                context.analysis_as_of,
            )
            return
        logger.warning(
            "record_evidence_block(%s) failed for run %s; block not recorded",
            source,
            run_id,
            exc_info=True,
        )
