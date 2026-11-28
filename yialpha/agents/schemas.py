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

import json
import math
import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _strip_grouping(value: str) -> str:
    """Best-effort comma handling for model-emitted numbers (round-7).

    Commas bundling 3-digit groups — with or without a trailing decimal
    point ("1,250", "1,250.50") — are grouping separators; a single comma
    with no dot and a non-3-digit tail is a decimal separator ("0,85").
    Anything else passes through untouched and fails the float parse
    honestly (the caller degrades the optional field)."""
    if "," not in value:
        return value
    _head, sep, tail = value.rpartition(",")
    if not sep:
        return value
    if "." in tail or len(tail) == 3:
        return value.replace(",", "")
    if value.count(",") == 1:
        return value.replace(",", ".")
    return value


def _coerce_optional_float(value):
    """#1058 coercer, round-7 widened: nullish strings → None, numeric
    strings parsed here (comma-grouping stripped) so an unparseable string
    degrades the OPTIONAL field to None instead of raising through pydantic
    and discarding the WHOLE structured decision."""
    # Round-17: two float-parseable forms are not a number for these fields.
    # A JSON boolean (pydantic's lax conversion reads ``true`` as ``1.0``:
    # "100% conviction", a 1.0 price target) and a non-finite value
    # (``"NaN"`` / ``"inf"`` / ``"1e999"`` — no price, and the two coercers
    # below already refuse them). Both degrade the OPTIONAL field to None,
    # exactly like an unparseable string.
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in _NULLISH_FLOAT:
            return None
        try:
            parsed = float(_strip_grouping(stripped))
        except ValueError:
            return None
    else:
        parsed = value
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return None
    return parsed


def _coerce_unit_interval(value):
    """0-1 optional float with percent-form rescue (round-7, #1058 class).

    ``85`` / ``"85%"`` → ``0.85`` — models emit the percent twin often
    enough that a strict ``le=1.0`` silently degraded whole PM decisions to
    regex fallback. Anything still outside [0, 1] after the rescue degrades
    to None: an optional conviction field never vetoes the decision."""
    percent = isinstance(value, str) and "%" in value
    if percent:
        value = value.strip().rstrip("%").strip()
    v = _coerce_optional_float(value)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if percent or v > 1.0:
        v = v / 100.0
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        return None
    return v


def _coerce_signed_percent_float(value):
    """Unbounded optional float with percent-form rescue (round-8, #1058
    class). ``"+4.7%"`` → ``0.047`` — expected_return's own description
    ("0.047 means +4.7%") invites the percent twin, and pre-round-8 it
    degraded to None while sibling 0-1 fields already rescued theirs. No
    unit-interval clamp here: the value is a signed return fraction and can
    legitimately exceed ±1."""
    percent = isinstance(value, str) and "%" in value
    if percent:
        value = value.strip().rstrip("%").strip()
    v = _coerce_optional_float(value)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if percent:
        v = v / 100.0
    return v if math.isfinite(v) else None


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
    desired_side: Literal["LONG", "SHORT", "FLAT"] | None = Field(
        default=None,
        description=(
            "Optional V2.4 signed-position intent. Explicit LONG or FLAT "
            "matches the conservative legacy mapping; SHORT is the ONLY way "
            "this system may open a short position — the legacy rating "
            "vocabulary can never produce one (Underweight/Sell map to "
            "REDUCE/CLOSE, never SHORT). Omit to keep the legacy "
            "rating-driven mapping."
        ),
    )

    @field_validator(
        "price_target", "underlying_price_target",
        mode="before",
    )
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator("expected_return", mode="before")
    @classmethod
    def _expected_return_percent_rescue(cls, v):
        return _coerce_signed_percent_float(v)

    @field_validator("confidence", "evidence_coverage", mode="before")
    @classmethod
    def _unit_interval_rescue(cls, v):
        return _coerce_unit_interval(v)

    @field_validator(
        "underlying_direction", "contract_direction", "desired_side",
        mode="before",
    )
    @classmethod
    def _direction_case_rescue(cls, v, info):
        """Case-insensitive, FIELD-SCOPED rescue (round-8, #1058 class).

        The three fields carry different vocabularies, so a cross-vocabulary
        wording ("Long" on a direction field, "Bullish" on desired_side) is
        OUT of vocab for that field and degrades it to None — an optional
        field never vetoes the decision. Only wording that matches the
        TARGET field's vocabulary (case-insensitively) is rescued.
        """
        if not isinstance(v, str):
            return v
        normalized = v.strip().lower()
        vocab = {
            "underlying_direction": {"bullish", "bearish", "neutral"},
            "contract_direction": {
                "bullish", "bearish", "neutral",
                "avoid_long", "avoid_short", "avoid",
            },
            "desired_side": {"long", "short", "flat"},
        }.get(info.field_name, set())
        if normalized not in vocab:
            return None
        return normalized.upper() if info.field_name == "desired_side" else normalized

    @model_validator(mode="before")
    @classmethod
    def _rescue_percent_probabilities(cls, data):
        """Percent-form probability branches (61/24/16, "70%") used to fail
        ``le=1.0`` and take the WHOLE decision down with them (#1058 class).
        Each present branch is rescued through the same unit-interval
        coercer; one unusable (or missing) branch degrades the OPTIONAL
        ``probabilities`` block to None — the distribution is advisory, the
        rating is the artifact.

        Round-13 shape rescue: a weak model can serialize the container to
        a JSON *string* ('``{"bull": 0.61, ...}``') instead of an object.
        One ``json.loads`` re-parses that form; a load failure (or a payload
        that is not an object) returns the ORIGINAL data so validation fails
        exactly as before — garbage still vetoes honestly. Only the shape is
        rescued, never the values: no branch is invented or defaulted.
        """
        if not isinstance(data, dict) or "probabilities" not in data:
            return data
        raw = data.get("probabilities")
        if raw is None:
            return data
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return data
            if not isinstance(raw, dict):
                return data
        if not isinstance(raw, dict):
            return data
        fixed: dict = {}
        for key in ("bull", "neutral", "bear"):
            fixed[key] = (
                _coerce_unit_interval(raw[key]) if key in raw else None
            )
        if any(v is None for v in fixed.values()):
            return {**data, "probabilities": None}
        return {**data, "probabilities": fixed}

    @field_validator("invalidation", mode="before")
    @classmethod
    def _invalidation_single_str_wrapped(cls, v):
        """Round-13 shape rescue (#1058 class): a weak model can emit the
        1-3 triggers as ONE sentence ("closes below 180 support; funding
        flips negative") instead of a JSON array — a bare ``str`` on
        ``list[str]`` raised ValidationError and vetoed the WHOLE decision,
        degrading every primary output (rating, thesis, targets) to the
        regex free-text fallback. Wrap a single string as ``[s]``; a
        nullish/blank string degrades to None ("not provided") the same way
        the optional floats do. Anything non-str passes through untouched
        and still fails honestly if unusable."""
        if not isinstance(v, str):
            return v
        stripped = v.strip()
        if not stripped or stripped.lower() in _NULLISH_FLOAT:
            return None
        return [v]

    @field_validator("price_target_currency")
    @classmethod
    def _price_target_currency_upper(cls, v: str | None) -> str | None:
        """Normalize case; out-of-vocab values coerce to None, never raise.

        These qualifier fields are OPTIONAL and the json_schema carries no
        enum (only description prose), so the model can legitimately emit a
        placeholder ("EUR", "USDT (Binance)"). A strict raise here discards
        the WHOLE structured decision — the rating degrades to regex
        fallback AND the fair-value bridge loses the target it was about to
        convert — the exact #1058 failure mode ``_coerce_optional_float``
        was fixed for. An unusable qualifier degrades to "not provided"
        instead of vetoing the primary output."""
        if v is None:
            return v
        normalized = v.strip().upper()
        return normalized if normalized in ("USD", "USDT") else None

    @field_validator("underlying_target_currency")
    @classmethod
    def _underlying_target_currency_upper(cls, v: str | None) -> str | None:
        """Normalize case; the underlying leg is USD-only, out-of-vocab → None
        (same coerce-not-raise contract as ``price_target_currency``)."""
        if v is None:
            return v
        normalized = v.strip().upper()
        return normalized if normalized == "USD" else None

    @field_validator("price_target_basis")
    @classmethod
    def _price_target_basis_lower(cls, v: str | None) -> str | None:
        """Normalize case; out-of-vocab bases coerce to None (the downstream
        consumers already treat a missing basis as the disclosed default)."""
        if v is None:
            return v
        normalized = v.strip().lower()
        return normalized if normalized in ("last", "mark") else None


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

    @field_validator("overall_score", mode="before")
    @classmethod
    def _not_a_boolean(cls, v):
        """Round-17: pydantic's lax float conversion reads a JSON ``true`` as
        ``1.0`` — "maximally bearish" on this 0-10 scale. A boolean is not a
        score; it fails the field's validation like any other out-of-contract
        value (``85`` does), so the caller takes its free-text fallback."""
        if isinstance(v, bool):
            raise ValueError("overall_score must be a number, not a boolean")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_case_rescue(cls, v):
        """Case-insensitive rescue (round-7, #1058 class): "Medium" must not
        fail the required Literal and cost the report its deterministic
        band/score header; out-of-vocab junk degrades to the conservative
        floor the field itself defines for weak evidence."""
        if not isinstance(v, str):
            return v
        normalized = v.strip().lower()
        return normalized if normalized in ("low", "medium", "high") else "low"
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

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_case_rescue(cls, v):
        """Case-insensitive rescue (round-7, #1058 class): a capitalized
        variant ("Medium" — the module's own renderer emits ``.capitalize()``
        prose) must not fail the required Literal and cost the report its
        deterministic band/score header. Out-of-vocab junk degrades to the
        field's own conservative floor, never a raise."""
        if not isinstance(v, str):
            return v
        normalized = v.strip().lower()
        return normalized if normalized in ("low", "medium", "high") else "low"


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


# Direction-bearing imperative phrases for the free-text fallback clamp.
# Imperative position phrasing only — verb-led forms cover plain and -ing
# variants ("go/going long", "enter/entering a short", "open/opening");
# "long/short entry" alone is NOT matched (descriptive narrative like "long
# entry levels were observed by takers" is legitimate content); "buy/sell
# flow" likewise survives. A blacklist can never be complete — the clamp is
# a backstop for the prompt-level freeze, observable via its banner.
_DIRECTION_CLAIM_RE = re.compile(
    r"(?im)^[^\n]*\b("
    r"(?:go|going|enter|entering|open|opening)\s+(?:a\s+|the\s+)?(?:long|short)"
    r"|(?:take|taking|consider|considering|place|placing|add|adding)\s+"
    r"(?:a\s+)?(?:long|short)(?:\s+entry)?"
    r"|(?:strong\s+)?(?:bullish|bearish)\s+(?:call|signal|setup)"
    r")\b[^\n]*$"
)


def clamp_direction_language(text: str) -> str:
    """Deterministic positioning direction-freeze for the free-text fallback.

    The structured path enforces the freeze mechanically —
    ``PositioningReport`` has no direction field and ``extra="forbid"``
    rejects any. The free-text fallback has only the prompt's instruction
    left, so strip direction-bearing LINES (whole-line replacement keeps the
    surrounding narrative readable) and append an observable banner. No-op
    for text without direction claims.
    """
    redacted = _DIRECTION_CLAIM_RE.sub(
        "[direction statement removed — positioning freeze]", text)
    if redacted != text:
        redacted += (
            "\n\n> ⚠️ Free-text fallback: direction-bearing lines were "
            "deterministically removed (positioning direction freeze)."
        )
    return redacted
