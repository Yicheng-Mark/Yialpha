"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``YiAlphaGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

import os
from datetime import datetime
from pathlib import Path


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via tmp-file + ``os.replace`` (atomic on Windows/POSIX).

    ``save_reports`` runs inside run_robust's watchdog kill window
    (``taskkill /F /T`` on timeout) exactly like ``_log_state``; a direct
    ``write_text`` can be torn mid-write, leaving a truncated
    ``decision.md`` / ``complete_report.md`` that downstream readers
    (trade_ticket, ``errors="ignore"``) parse as missing levels or
    garbage-derived numbers instead of failing loudly. Same pattern the
    states log and the memory log already use.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _render_data_quality(quality: dict) -> str:
    """Render the run's ``data_quality`` evidence block as a report section.

    A degraded run (some categories served NO_DATA sentinels) must be
    distinguishable from a fully-fed one in the human-facing report — not
    only in ``full_states_log``. Core degradations get a prominent ⚠ banner;
    optional enrichments that were simply absent get a soft note. Mirrors the
    risk overlay's visible-warning convention.
    """
    core = int(quality.get("core_sentinel_count") or 0)
    optional = int(quality.get("optional_sentinel_count") or 0)
    stale = int(quality.get("stale_cache_count") or 0)
    if not (core or optional or stale):
        return ""
    lines = []
    if core:
        lines.append(
            f"⚠ **DEGRADED RUN**: {core} core data categor"
            f"{'y' if core == 1 else 'ies'} returned no usable data — the analysis "
            "decided WITHOUT that data. Weigh the conclusions accordingly."
        )
    if stale:
        lines.append(
            f"⚠ {stale} data request"
            f"{'s' if stale != 1 else ''} served STALE cache after a vendor failure."
        )
    if optional:
        lines.append(
            f"Note: {optional} optional enrichment categor"
            f"{'y' if optional == 1 else 'ies'} unavailable (analysis proceeded without it)."
        )
    for e in quality.get("sentinels") or []:
        lines.append(f"- `{e.get('method')}` ({e.get('kind')}): {e.get('detail')}")
    return "## Data Quality\n\n" + "\n\n".join(lines)


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        _atomic_write_text(analysts_dir / "market.md", final_state["market_report"])
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        _atomic_write_text(analysts_dir / "sentiment.md", final_state["sentiment_report"])
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        _atomic_write_text(analysts_dir / "news.md", final_state["news_report"])
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        _atomic_write_text(analysts_dir / "fundamentals.md", final_state["fundamentals_report"])
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            _atomic_write_text(research_dir / "bull.md", debate["bull_history"])
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            _atomic_write_text(research_dir / "bear.md", debate["bear_history"])
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            _atomic_write_text(research_dir / "manager.md", debate["judge_decision"])
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        _atomic_write_text(trading_dir / "trader.md", final_state["trader_investment_plan"])
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    risk = final_state.get("risk_debate_state") or {}
    if risk:
        risk_dir = save_path / "4_risk"
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            _atomic_write_text(risk_dir / "aggressive.md", risk["aggressive_history"])
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            _atomic_write_text(risk_dir / "conservative.md", risk["conservative_history"])
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            _atomic_write_text(risk_dir / "neutral.md", risk["neutral_history"])
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

    # 5. Final decision. ``final_trade_decision`` is authoritative because the
    # deterministic risk overlay appends its sizing/stop/exposure override there
    # after the Portfolio Manager's prose. Falling back to judge_decision keeps
    # older states readable, but a report must never prefer that pre-overlay
    # text when the final version exists.
    final_decision = final_state.get("final_trade_decision") or risk.get("judge_decision")
    if final_decision:
        portfolio_dir = save_path / "5_portfolio"
        portfolio_dir.mkdir(exist_ok=True)
        _atomic_write_text(portfolio_dir / "decision.md", final_decision)
        sections.append(
            "## V. Final Risk-Adjusted Decision\n\n"
            f"### Final Decision\n{final_decision}"
        )

    # Write consolidated report. The data-quality banner sits directly under
    # the header so a degraded run cannot be mistaken for a fully-fed one.
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    quality_section = _render_data_quality(final_state.get("data_quality") or {})
    body = "\n\n".join([quality_section] + sections) if quality_section else "\n\n".join(sections)
    _atomic_write_text(save_path / "complete_report.md", header + body)
    return save_path / "complete_report.md"
