"""Sentiment source policy — deterministic confidence cap + evidence injection.

PR4 (2026-09) pins:

  * ``_confidence_cap`` / ``_enforce_confidence_cap`` — the deterministic
    ceiling computed from the evidence BEFORE the LLM runs and enforced on
    the rendered report AFTER: single social platform ⇒ Medium max,
    <5 target posts / unavailable feed ⇒ Low, plain stocks ⇒ no cap;
  * the node-level injection contract — fetched third-party blocks ride a
    USER-role evidence message behind an untrusted-content banner, never
    the system prompt.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import yialpha.agents.analysts.sentiment_analyst as sent


# ---------------------------------------------------------------------------
# Confidence cap: computation
# ---------------------------------------------------------------------------
def _square_block(count: int) -> str:
    return (
        "Binance Square recommended feed (global crypto social) — 50 posts "
        "scanned (web-homepage, web-trending), fetched 2026-09-01 08:00 UTC "
        f"(fresh). Posts mentioning BTC: {count} (all within the last 3 days)."
    )


@pytest.mark.unit
def test_cap_truth_table():
    # Plain stocks: no deterministic cap (LLM guidance applies).
    assert sent._confidence_cap("full", None) == (None, "")
    # Pure crypto, healthy sample: single platform -> Medium ceiling.
    cap, reason = sent._confidence_cap("square_only", _square_block(9))
    assert cap == "medium" and "single-platform" in reason
    # Thin sample (<5): Low.
    cap, reason = sent._confidence_cap("square_only", _square_block(3))
    assert cap == "low" and "3 Square post" in reason
    # Zero posts: Low + Neutral guidance in the reason.
    cap, reason = sent._confidence_cap("square_only", _square_block(0))
    assert cap == "low" and "zero Square posts" in reason
    # Feed unavailable/absent: Low.
    cap, reason = sent._confidence_cap("square_only", None)
    assert cap == "low" and "unavailable" in reason
    cap, _ = sent._confidence_cap("square_only", "<binance_square unavailable: X>")
    assert cap == "low"
    # Stock perp: two-platform read still capped at Medium.
    cap, reason = sent._confidence_cap(
        "square_plus_underlying", _square_block(7)
    )
    assert cap == "medium" and "two-platform" in reason
    cap, _ = sent._confidence_cap("square_plus_underlying", _square_block(2))
    assert cap == "low"


@pytest.mark.unit
def test_square_post_count_parses_rendered_header():
    assert sent._square_post_count(_square_block(12)) == 12
    assert sent._square_post_count(_square_block(0)) == 0
    # Non-square blocks are not parseable -> None (weakest-evidence case).
    assert sent._square_post_count(None) is None
    assert sent._square_post_count("Some other text") is None
    assert sent._square_post_count("<binance_square unavailable: X>") is None


def _square_block_recent(total: int, recent: int) -> str:
    return (
        "Binance Square recommended feed (global crypto social) — 50 posts "
        "scanned (web-homepage, web-trending), fetched 2026-09-01 08:00 UTC "
        f"(fresh). Posts mentioning BTC: {total} ({recent} of {total} within "
        "the last 3 days; 4 older matching posts kept for context)."
    )


@pytest.mark.unit
def test_cap_keys_on_recent_count_not_stale_total():
    # 6 mentions but only 2 recent: months-old chatter must not lift the
    # ceiling — the cap uses the RECENT count from the header's recency note.
    assert sent._square_post_count(_square_block_recent(6, 2)) == 2
    cap, reason = sent._confidence_cap("square_only", _square_block_recent(6, 2))
    assert cap == "low" and "2 Square post" in reason
    # 5+ RECENT mentions: the Medium ceiling is legitimately earned.
    cap, reason = sent._confidence_cap("square_only", _square_block_recent(9, 7))
    assert cap == "medium"
    # Legacy header without the recency note: falls back to the raw total.
    assert sent._square_post_count(_square_block(9)) == 9


# ---------------------------------------------------------------------------
# Confidence cap: enforcement on the rendered report
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_over_cap_confidence_is_lowered_with_note():
    report = (
        "**Overall Sentiment:** **Mildly Bullish** (Score: 6.0/10)\n"
        "**Confidence:** High\n\n"
        "Narrative body."
    )
    out = sent._enforce_confidence_cap(report, "medium", "single-platform evidence")
    assert "**Confidence:** Medium (capped by source policy: single-platform evidence)" in out
    assert "**Confidence:** High" not in out
    # Untouched fields survive.
    assert "**Mildly Bullish**" in out and "Narrative body." in out


@pytest.mark.unit
def test_compliant_confidence_passes_through_byte_unchanged():
    report = "**Confidence:** Low\n\nBody."
    assert sent._enforce_confidence_cap(report, "medium", "r") == report
    report2 = "**Confidence:** Medium\n\nBody."
    assert sent._enforce_confidence_cap(report2, "medium", "r") == report2


@pytest.mark.unit
def test_missing_confidence_line_is_noop():
    # Free-text fallback without the header: nothing to rewrite.
    report = "Just prose, no header."
    assert sent._enforce_confidence_cap(report, "low", "r") == report


# ---------------------------------------------------------------------------
# Node-level injection: evidence in user role, cap applied to the report
# ---------------------------------------------------------------------------
class _FreeTextLLM:
    """LLM whose structured binding fails -> free-text path; records input."""

    def __init__(self, content: str):
        self._content = content
        self.seen_messages = None

    def with_structured_output(self, schema):  # noqa: ANN001
        raise NotImplementedError("no structured output")

    def invoke(self, messages):
        self.seen_messages = messages
        return AIMessage(content=self._content)


def _state(ticker="BTCUSDT", asset_type="crypto_perp"):
    return {
        "trade_date": date.today().isoformat(),
        "company_of_interest": ticker,
        "asset_type": asset_type,
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


def _stub_node_sources(monkeypatch, square_block):
    class _NewsStub:
        @property
        def func(self):
            return lambda *a, **k: "NEWS BODY"

    monkeypatch.setattr(sent, "get_news", _NewsStub())
    monkeypatch.setattr(
        sent, "fetch_binance_square_block", lambda t, as_of=None: square_block
    )


@pytest.mark.unit
def test_node_injects_evidence_in_user_role_and_caps_report(monkeypatch):
    square = _square_block(2)  # thin sample -> cap low
    _stub_node_sources(monkeypatch, square)
    llm = _FreeTextLLM(
        "**Overall Sentiment:** **Neutral** (Score: 5.0/10)\n"
        "**Confidence:** High\n\nBody."
    )
    node = sent.create_sentiment_analyst(llm)
    out = node(_state())

    messages = llm.seen_messages
    assert messages is not None
    # The evidence message is LAST and in the USER role; the system message
    # carries instructions only.
    evidence = messages[-1]
    assert isinstance(evidence, HumanMessage)
    assert evidence.content.startswith("[EXTERNAL EVIDENCE")
    assert "NEWS BODY" in evidence.content
    assert square in evidence.content
    system_parts = [
        m for m in messages if getattr(m, "type", "") == "system"
    ] or [m for m in messages if isinstance(getattr(m, "content", ""), str)
          and "sentiment analyst" in str(getattr(m, "content", "")).lower()]
    assert system_parts, "a system message must exist"
    system_text = str(system_parts[0].content if hasattr(system_parts[0], "content") else "")
    assert "<start_of_news>" not in system_text
    assert "EXTERNAL EVIDENCE" in system_text  # instructions reference the role
    # The over-cap High was deterministically lowered in the final report.
    assert "**Confidence:** Low (capped by source policy" in out["sentiment_report"]
    assert "messages" in out and len(out["messages"]) == 1


@pytest.mark.unit
def test_node_uncapped_stock_report_untouched(monkeypatch):
    _stub_node_sources(monkeypatch, square_block="unused")
    # Stock run: square never fetched (config gate off for stocks); no cap.
    llm = _FreeTextLLM("**Confidence:** High\n\nBody.")
    node = sent.create_sentiment_analyst(llm)
    out = node(_state(ticker="AAPL", asset_type="stock"))
    assert out["sentiment_report"] == "**Confidence:** High\n\nBody."
