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

from yiagents.agents.schemas import SentimentReport, render_sentiment_report
from yiagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from yiagents.agents.utils.prompt_builder import build_collaborator_prompt
from yiagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from yiagents.dataflows.config import submit_with_context
from yiagents.dataflows.reddit import fetch_reddit_posts
from yiagents.dataflows.stocktwits import fetch_stocktwits_messages
from yiagents.dataflows.utils import is_historical_date

# Opt-in (default OFF = byte-equivalent sequential fetch). Fan out the three
# independent source fetches (Yahoo news / StockTwits / Reddit) on a thread
# pool. Each block is written to a FIXED prompt slot, so completion order does
# not change the assembled prompt -- only wall-clock. Fetchers degrade to a
# string and do not raise, so the parallel path preserves sequential semantics.
_SENTIMENT_PARALLEL_FETCH = os.environ.get(
    "YIAGENTS_SENTIMENT_PARALLEL_FETCH", ""
).lower() in ("1", "true", "yes", "on")

_HISTORICAL_STOCKTWITS_UNAVAILABLE = (
    "<unavailable: StockTwits exposes a current feed without a trustworthy "
    "historical as-of boundary; omitted from this historical analysis>"
)
_HISTORICAL_REDDIT_UNAVAILABLE = (
    "<unavailable: Reddit search/RSS exposes current retrieval results without "
    "a trustworthy historical as-of boundary; omitted from this historical analysis>"
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


def _fetch_sentiment_sources(ticker: str, start_date: str, end_date: str) -> tuple[str, str, str]:
    """Fetch the three sentiment sources, returning (news, stocktwits, reddit).

    Sequential by default; fanned out on a thread pool when
    ``YIAGENTS_SENTIMENT_PARALLEL_FETCH`` is on. Byte-equivalent -- each block
    is returned to a fixed slot, so completion order does not affect the result.
    Fetchers degrade to a string and do not raise, so the parallel path
    preserves the sequential semantics.
    """
    if is_historical_date(end_date):
        # News is queried with explicit date bounds.  The two social endpoints
        # are latest-feed APIs; filtering their current response after download
        # cannot reconstruct what was discoverable on a past date, so omit them
        # instead of contaminating a backtest prompt with today's narratives.
        return (
            _get_news_impl(ticker, start_date, end_date),
            _HISTORICAL_STOCKTWITS_UNAVAILABLE,
            _HISTORICAL_REDDIT_UNAVAILABLE,
        )

    if _SENTIMENT_PARALLEL_FETCH:
        # Lambdas preserve each call's exact form so the result is byte-
        # identical to the sequential path; only fetch order differs.
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_news = submit_with_context(
                pool, _get_news_impl, ticker, start_date, end_date
            )
            fut_stocktwits = submit_with_context(
                pool, fetch_stocktwits_messages, ticker, limit=30
            )
            fut_reddit = submit_with_context(pool, fetch_reddit_posts, ticker)
            return fut_news.result(), fut_stocktwits.result(), fut_reddit.result()
    return (
        _get_news_impl(ticker, start_date, end_date),
        fetch_stocktwits_messages(ticker, limit=30),
        fetch_reddit_posts(ticker),
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

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block, stocktwits_block, reddit_block = _fetch_sentiment_sources(
            ticker, start_date, end_date
        )

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
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
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

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

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so. **Do not invent, paraphrase, or attribute messages, posts, or headlines that do not appear in the data blocks above** — quote what is actually there, or report the source as empty.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}""" + NO_EXTERNAL_TOOLS
