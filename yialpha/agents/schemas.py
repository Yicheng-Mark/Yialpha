"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(StrEnum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(StrEnum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class OutcomeProbabilities(BaseModel):
    """Probability distribution over the three outcome branches (V2.0).

    Soft contract: the three values should sum to ~1.0. Deliberately NOT
    hard-validated — a strict sum check would make provider structured-output
    calls fail on benign rounding (0.61 + 0.24 + 0.16) and silently degrade
    the PM to free text, which costs more than an imperfect distribution.
    Consumers treat values as approximate weights, never as calibrated odds
    (calibration arrives with the V2.1 outcome ledger).
    """

    bull: float = Field(ge=0.0, le=1.0, description="Probability of a bullish outcome.")
    neutral: float = Field(
        ge=0.0, le=1.0, description="Probability of a sideways / neutral outcome."
    )
    bear: float = Field(ge=0.0, le=1.0, description="Probability of a bearish outcome.")


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.

    V2.0 additions (all optional, all backward compatible): ``confidence``,
    ``probabilities``, ``expected_return``, ``invalidation``,
    ``evidence_coverage``. They feed the ExecutionTicket and the V2.1
    attribution ledger; a model that omits them keeps rendering byte-identical
    to the pre-V2 markdown.

    V2.1 additions (all optional, additive — no version bump): the
    currency/basis qualifiers and the USD underlying target that feed the
    stock-perp Fair Value Bridge (``price_target_currency``,
    ``price_target_basis``, ``underlying_price_target``,
    ``underlying_target_currency``). Renderers ignore them entirely, so a
    decision without them stays byte-identical to the pre-V2.1 markdown.

    V2.3 additions (all optional, additive — no version bump): the
    stock-perp DUAL VIEW (``underlying_direction``, ``contract_direction``,
    ``basis_view``). On a tokenized-stock perp the EQUITY view (USD) and the
    USDT CONTRACT view (premium / funding / liquidity) can legitimately
    disagree — the PM states each separately instead of collapsing them into
    one rating. All three default None; a decision without them renders
    byte-identically to the pre-V2.3 markdown (pinned by tests).
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional conviction in the rating on a 0-1 scale (1 = fully "
            "certain). Judge it against evidence quality and agreement across "
            "analysts, not just rhetorical strength."
        ),
    )
    probabilities: OutcomeProbabilities | None = Field(
        default=None,
        description=(
            "Optional probability split over the three outcome branches "
            "(bull / neutral / bear), each 0-1, summing to about 1.0."
        ),
    )
    expected_return: float | None = Field(
        default=None,
        description=(
            "Optional expected return to the price target over the stated "
            "horizon, as a fraction (0.047 means +4.7%). Omit when the view "
            "has no numeric target."
        ),
    )
    invalidation: list[str] | None = Field(
        default=None,
        description=(
            "Optional conditions that would invalidate this thesis, e.g. "
            "'closes below 180 support', 'funding flips and stays negative'. "
            "One to three concrete, observable triggers."
        ),
    )
    evidence_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional self-assessed share of the analysis that rested on real "
            "retrieved data rather than priors, 0-1. Lower it when reports "
            "carried NO_DATA placeholders."
        ),
    )
    price_target_currency: str | None = Field(
        default=None,
        description=(
            "Optional currency of price_target: exactly one of USD or USDT "
            "(case-insensitive). Only meaningful when the instrument's quote "
            "currency differs from the target's."
        ),
    )
    price_target_basis: str | None = Field(
        default=None,
        description=(
            "Optional price basis of the target: exactly one of 'last' "
            "(last-traded price) or 'mark' (mark price), case-insensitive."
        ),
    )
    underlying_price_target: float | None = Field(
        default=None,
        description=(
            "Optional price target on the UNDERLYING asset, in USD — the leg "
            "the perp Fair Value Bridge converts into a USDT contract target. "
            "Omit for instruments whose target is already single-currency."
        ),
    )
    underlying_target_currency: str | None = Field(
        default=None,
        description=(
            "Optional currency of underlying_price_target: USD only "
            "(case-insensitive). The bridge accepts no other underlying unit."
        ),
    )
    underlying_direction: Literal["bullish", "bearish", "neutral"] | None = Field(
        default=None,
        description=(
            "Optional V2.3 dual view (stock-perp runs): the directional view on "
            "the UNDERLYING equity in USD — exactly one of bullish / bearish / "
            "neutral. Omit on non stock-perp instruments."
        ),
    )
    contract_direction: (
        Literal["bullish", "bearish", "neutral", "avoid_long", "avoid_short", "avoid"]
        | None
    ) = Field(
        default=None,
        description=(
            "Optional V2.3 dual view (stock-perp runs): the view on the USDT "
            "PERP contract itself — bullish / bearish / neutral / avoid_long / "
            "avoid_short / avoid. The premium, funding and liquidity can justify "
            "a contract view that disagrees with underlying_direction (e.g. "
            "underlying bullish + avoid_long when the perp trades at a rich "
            "premium with expensive positive funding). Omit on non stock-perp "
            "instruments."
        ),
    )
    basis_view: str | None = Field(
        default=None,
        description=(
            "Optional V2.3 dual view (stock-perp runs): one short prose "
            "sentence on the contract-vs-underlying basis (premium/discount, "
            "funding carry) that explains why the two views agree or disagree. "
            "Omit when no dual view was filled."
        ),
    )

    @field_validator(
        "price_target", "confidence", "expected_return", "evidence_coverage",
        "underlying_price_target",
        mode="before",
    )
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator("price_target_currency")
    @classmethod
    def _price_target_currency_upper(cls, v: str | None) -> str | None:
        """Normalize case; only USD / USDT (case-insensitive) are accepted."""
        if v is None:
            return v
        normalized = v.strip().upper()
        if normalized not in ("USD", "USDT"):
            raise ValueError(
                f"price_target_currency must be one of USD, USDT "
                f"(case-insensitive), got {v!r}"
            )
        return normalized

    @field_validator("underlying_target_currency")
    @classmethod
    def _underlying_target_currency_upper(cls, v: str | None) -> str | None:
        """Normalize case; the underlying target leg is USD-denominated only."""
        if v is None:
            return v
        normalized = v.strip().upper()
        if normalized != "USD":
            raise ValueError(
                f"underlying_target_currency must be USD (case-insensitive), "
                f"got {v!r}"
            )
        return normalized

    @field_validator("price_target_basis")
    @classmethod
    def _price_target_basis_lower(cls, v: str | None) -> str | None:
        """Normalize case; only last / mark (case-insensitive) are accepted."""
        if v is None:
            return v
        normalized = v.strip().lower()
        if normalized not in ("last", "mark"):
            raise ValueError(
                f"price_target_basis must be 'last' or 'mark' "
                f"(case-insensitive), got {v!r}"
            )
        return normalized


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle. V2.0 optional fields
    append lines ONLY when filled — a decision without them renders
    byte-identically to the pre-V2 shape, so legacy parsers and the
    accuracy-loop rating regex keep working unchanged.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.confidence is not None:
        parts.extend(["", f"**Confidence**: {decision.confidence:.0%}"])
    if decision.probabilities is not None:
        p = decision.probabilities
        parts.extend([
            "",
            f"**Probabilities**: bull {p.bull:.0%} / neutral {p.neutral:.0%}"
            f" / bear {p.bear:.0%}",
        ])
    if decision.expected_return is not None:
        parts.extend(["", f"**Expected Return**: {decision.expected_return:+.1%}"])
    if decision.invalidation:
        parts.extend(["", "**Invalidation**: " + "; ".join(decision.invalidation)])
    if decision.evidence_coverage is not None:
        parts.extend(["", f"**Evidence Coverage**: {decision.evidence_coverage:.0%}"])
    # V2.3 dual view (stock perps): a compact block ONLY when the PM filled
    # at least one dual-view field. All-None renders byte-identically to the
    # pre-V2.3 shape (pinned), so legacy parsers never see the new headers.
    if decision.underlying_direction is not None:
        parts.extend(["", f"**Underlying view**: {decision.underlying_direction}"])
    if decision.contract_direction is not None:
        parts.extend(["", f"**Contract view**: {decision.contract_direction}"])
    if decision.basis_view:
        parts.extend(["", f"**Basis view**: {decision.basis_view}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(StrEnum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' only when multiple independent sources returned substantive "
            "data — never when the prompt states a deterministic source-policy "
            "cap (that ceiling is enforced on the report mechanically)."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Positioning Analyst (V2.3)
# ---------------------------------------------------------------------------


class PositioningReport(BaseModel):
    """Structured positioning report produced by the V2.3 Positioning Analyst.

    The frozen contract: Positioning NEVER outputs a trade direction. There
    is no ``direction``/``bullish``/``bearish`` field and the model config
    forbids extras, so a provider that emits a direction key anyway FAILS
    validation (structured attempt falls back; the freeze is schema-enforced,
    pinned by tests). What the report carries instead is the positioning
    FABRIC: who pays (funding bias), how crowded each side is, and how much
    liquidity stands behind the move.

    ``narrative`` preserves the rich source-by-source analysis;
    :func:`render_positioning_report` prepends a deterministic header.
    """

    #: extra="forbid" IS the direction freeze: any key outside this schema
    #: (notably a trade-direction key) fails validation instead of being
    #: silently dropped.
    model_config = ConfigDict(extra="forbid")

    funding_bias: Literal["long_pays_expensive", "neutral", "long_gets_paid"] = Field(
        description=(
            "Funding carry read. Exactly one of: long_pays_expensive (net "
            "positive cumulative funding — longs pay, the crowded side is "
            "long), neutral (near-zero funding), long_gets_paid (net negative "
            "cumulative funding — shorts pay)."
        ),
    )
    crowding: Literal["crowded_long", "balanced", "crowded_short"] = Field(
        description=(
            "Leveraged-crowd positioning from OI build + long/short ratios + "
            "taker flow. Exactly one of: crowded_long, balanced, crowded_short."
        ),
    )
    liquidity_risk: Literal["thin", "normal", "deep"] = Field(
        description=(
            "Execution/liquidation-cascade risk from book depth, spread and "
            "ADL quantiles. Exactly one of: thin, normal, deep."
        ),
    )
    narrative: str = Field(
        description=(
            "Full positioning report covering, in order: "
            "(1) funding carry (trailing sum, next-rate snapshot); "
            "(2) open-interest build and its percentile; "
            "(3) long/short ratios across the three vantage points with "
            "cross-vantage divergence; "
            "(4) taker order-flow aggression; "
            "(5) book depth bands, slippage and ADL quantiles; "
            "(6) spot-perp / index basis; "
            "(7) on-chain flow context when the onchain block is present. "
            "This is a POSITIONING read: describe crowding, carry and "
            "liquidity — do NOT state a price direction or a trade "
            "recommendation; the directional call belongs to other analysts. "
            "(8) a markdown table summarising the key positioning signals, "
            "their reading, source, and supporting evidence."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the positioning read based on component "
            "availability. Use 'low' when one or more positioning components "
            "carried an unavailable/skipped-live-only status; 'medium' when "
            "the core (funding + OI + LSR) is present but thin; 'high' only "
            "when the full positioning fabric was available."
        ),
    )


def render_positioning_report(report: PositioningReport) -> str:
    """Render a PositioningReport to the markdown shape the pipeline consumes.

    Deterministic header (fabric labels + confidence) prepended to the
    narrative; no direction line exists anywhere in the render.
    """
    return "\n".join([
        f"**Positioning — funding:** {report.funding_bias} | "
        f"**crowding:** {report.crowding} | "
        f"**liquidity:** {report.liquidity_risk}",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
