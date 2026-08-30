"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

For CRYPTO targets (asset_type crypto/crypto_spot/crypto_perp, live dates
only, ``binance_square_enabled``) a fourth block is added:

  4. Binance Square posts — crypto-native social feed filtered for the
                            target coin, ranked by engagement

Stock runs never see the fourth block (the prompt stays byte-identical to
the three-source version); historical replays never see ANY of the three
social feeds (they are current-snapshot APIs with no as-of boundary).

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage

from yialpha.agents.schemas import SentimentReport, render_sentiment_report
from yialpha.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from yialpha.agents.utils.prompt_builder import build_collaborator_prompt
from yialpha.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from yialpha.dataflows.binance_square import fetch_binance_square_block
from yialpha.dataflows.config import get_config, submit_with_context
from yialpha.dataflows.reddit import fetch_reddit_posts
from yialpha.dataflows.stocktwits import fetch_stocktwits_messages
from yialpha.dataflows.utils import is_historical_date

# Opt-in (default OFF = byte-equivalent sequential fetch). Fan out the three
# independent source fetches (Yahoo news / StockTwits / Reddit) on a thread
# pool. Each block is written to a FIXED prompt slot, so completion order does
# not change the assembled prompt -- only wall-clock. Fetchers degrade to a
# string and do not raise, so the parallel path preserves sequential semantics.
_SENTIMENT_PARALLEL_FETCH = os.environ.get(
    "YIALPHA_SENTIMENT_PARALLEL_FETCH", ""
).lower() in ("1", "true", "yes", "on")

_HISTORICAL_STOCKTWITS_UNAVAILABLE = (
    "<unavailable: StockTwits exposes a current feed without a trustworthy "
    "historical as-of boundary; omitted from this historical analysis>"
)
_HISTORICAL_REDDIT_UNAVAILABLE = (
    "<unavailable: Reddit search/RSS exposes current retrieval results without "
    "a trustworthy historical as-of boundary; omitted from this historical analysis>"
)
_HISTORICAL_BINANCE_SQUARE_UNAVAILABLE = (
    "<unavailable: Binance Square exposes a current feed without a trustworthy "
    "historical as-of boundary; omitted from this historical analysis>"
)

#: Asset types that get the Binance Square crypto-sentiment block. Stock runs
#: keep the byte-identical three-source prompt.
_CRYPTO_ASSET_TYPES = frozenset({"crypto", "crypto_spot", "crypto_perp"})


def _binance_square_enabled(asset_type: str | None) -> bool:
    """Whether this run's sentiment prompt should carry the Square block."""
    return asset_type in _CRYPTO_ASSET_TYPES and bool(
        get_config().get("binance_square_enabled", True)
    )


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def _get_news_impl(ticker: str, start_date: str, end_date: str) -> str:
    """Call the underlying function behind the ``get_news`` LangChain tool.

    ``get_news`` is a ``@tool``-decorated ``BaseTool``; its raw callable lives
    under ``.func``. mypy cannot see ``.func`` on the ``BaseTool`` type, so
    this helper centralises the access with a safe ``getattr`` fallback.
    """
    fn = getattr(get_news, "func", None)
    if fn is not None:
        return fn(ticker, start_date, end_date)
    return get_news(ticker, start_date, end_date)  # type: ignore[operator]


def _fetch_sentiment_sources(
    ticker: str,
    start_date: str,
    end_date: str,
    asset_type: str | None = None,
) -> tuple[str, str, str, str | None]:
    """Fetch the sentiment sources, returning (news, stocktwits, reddit, square).

    Sequential by default; fanned out on a thread pool when
    ``YIALPHA_SENTIMENT_PARALLEL_FETCH`` is on. Byte-equivalent -- each block
    is returned to a fixed slot, so completion order does not affect the result.
    Fetchers degrade to a string and do not raise, so the parallel path
    preserves the sequential semantics.

    The fourth slot is the Binance Square crypto-sentiment block: populated
    only for crypto asset types with ``binance_square_enabled`` (live runs),
    an explicit historical-unavailable placeholder for crypto historical
    runs, and ``None`` for stock runs (whose prompt stays byte-identical to
    the three-source version).
    """
    square_enabled = _binance_square_enabled(asset_type)

    if is_historical_date(end_date):
        # News is queried with explicit date bounds.  The social endpoints
        # are latest-feed APIs; filtering their current response after download
        # cannot reconstruct what was discoverable on a past date, so omit them
        # instead of contaminating a backtest prompt with today's narratives.
        return (
            _get_news_impl(ticker, start_date, end_date),
            _HISTORICAL_STOCKTWITS_UNAVAILABLE,
            _HISTORICAL_REDDIT_UNAVAILABLE,
            _HISTORICAL_BINANCE_SQUARE_UNAVAILABLE if square_enabled else None,
        )

    if _SENTIMENT_PARALLEL_FETCH:
        # Lambdas preserve each call's exact form so the result is byte-
        # identical to the sequential path; only fetch order differs.
        with ThreadPoolExecutor(max_workers=4 if square_enabled else 3) as pool:
            fut_news = submit_with_context(
                pool, _get_news_impl, ticker, start_date, end_date
            )
            fut_stocktwits = submit_with_context(
                pool, fetch_stocktwits_messages, ticker, limit=30
            )
            fut_reddit = submit_with_context(pool, fetch_reddit_posts, ticker)
            fut_square = (
                submit_with_context(pool, fetch_binance_square_block, ticker)
                if square_enabled
                else None
            )
            return (
                fut_news.result(),
                fut_stocktwits.result(),
                fut_reddit.result(),
                fut_square.result() if fut_square is not None else None,
            )
    return (
        _get_news_impl(ticker, start_date, end_date),
        fetch_stocktwits_messages(ticker, limit=30),
        fetch_reddit_posts(ticker),
        fetch_binance_square_block(ticker) if square_enabled else None,
    )


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch the sources. Each fetcher degrades gracefully and returns
        # a string (no exceptions surface from here), so the LLM always sees
        # something — either real data or a clear placeholder. The Square
        # block is None unless this is a live crypto run.
        news_block, stocktwits_block, reddit_block, binance_square_block = (
            _fetch_sentiment_sources(
                ticker, start_date, end_date, asset_type=state.get("asset_type")
            )
        )

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            binance_square_block=binance_square_block,
        )

        prompt = build_collaborator_prompt(include_tools=False)

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    binance_square_block: str | None = None,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks.

    ``binance_square_block=None`` (stock runs, disabled config, or a caller
    predating the fourth source) reproduces the three-source prompt
    byte-for-byte: the source-count word, the Square section, and guidance
    item 9 all collapse to the empty string at fixed interpolation points.
    """
    source_count = "four" if binance_square_block is not None else "three"
    square_section = (
        f"""
### Binance Square posts — crypto-native social feed (current snapshot)
Crypto-native retail chatter from Binance Square, filtered for the target asset and ranked by view/like counts, plus feed-wide hot-coin mentions for overall market mood. Posts are opinions (frequently shilling or sarcasm), not data.

<start_of_binance_square>
{binance_square_block}
<end_of_binance_square>
"""
        if binance_square_block is not None
        else ""
    )
    square_guidance = (
        """
9. **Treat Binance Square as crypto-native retail chatter.** Weight posts by their view/like counts (a 200k-view post reflects real attention; a 300-view post is noise), stay alert to shilling and sarcasm, and read it against the news framing — Square posts are opinion, never price data. If the block reports zero posts for the target asset, say so explicitly instead of generalizing from the hot-coin list.
"""
        if binance_square_block is not None
        else ""
    )
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on {source_count} complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>
{square_section}
## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so. **Do not invent, paraphrase, or attribute messages, posts, or headlines that do not appear in the data blocks above** — quote what is actually there, or report the source as empty.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.
{square_guidance}
## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}""" + NO_EXTERNAL_TOOLS
