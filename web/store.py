"""Read-only scans of ``~/.yiagents/logs`` to serve run history to the web UI.

The single source of truth for a completed run is::

    ~/.yiagents/logs/<TICKER>/YiAgentsStrategy_logs/full_states_log_<date>.json

written atomically by ``yiagents.graph.trading_graph._log_state``. A same-date
re-run atomically overwrites (``os.replace``), so one file == one analysis date
and the date in the filename is the analysis date (not the run wall-clock).

Report directories ``~/.yiagents/logs/reports/<TICKER>_<stamp>/`` carry a
wall-clock stamp, NOT the analysis date, and may be multiple per date; they are
listed separately as download links and never force-paired 1:1 with a date.

Why derive everything from the JSON (and not the markdown report tree):
``reporting.write_report_tree``'s section V uses the pre-overlay
``risk_debate_state.judge_decision`` and does NOT contain ``final_trade_decision``
(the quantitative risk overlay is appended only into the JSON's
``final_trade_decision``). The JSON is therefore the only source that has the
final, overlay-adjusted decision.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from yiagents.agents.utils.rating import parse_rating
from yiagents.dataflows.utils import safe_ticker_component

LOGS_ROOT = Path.home() / ".yiagents" / "logs"
REPORTS_ROOT = LOGS_ROOT / "reports"

# Subdirs under LOGS_ROOT that are not per-ticker result dirs.
_NON_TICKER_DIRS = {"reports", "robust"}

_DATE_RE = re.compile(r"full_states_log_(\d{4}-\d{2}-\d{2})\.json$")

# Marker the Portfolio Manager's overlay appends to final_trade_decision. The
# section below it holds the quantitative-risk numbers.
_OVERLAY_MARKER = "## Quantitative Risk Overlay"

# Replicated from scripts/analyze_window.py::_FIELDS. ``scripts/`` is not a
# package (no __init__.py), so it cannot be imported; this is a faithful copy
# of the same regex set, scoped to the overlay block only.
_OVERLAY_FIELDS = {
    "action": r"\*\*Action\*\*:\s*(.+)",
    "target_weight": r"\*\*Target Weight\*\*:\s*([0-9.]+%)",
    "position_value": r"\*\*Target Weight\*\*:.*?\(([-0-9,]+)\)",
    "stop_loss": r"\*\*Stop Loss\*\*:\s*([-0-9.]+)",
    "entry": r"\*\*Entry Reference\*\*:\s*([-0-9.]+)",
    "regime": r"\*\*Drawdown Regime\*\*:\s*(\S+)",
    "rationale": r"\*\*Rationale\*\*:\s*(.+)",
}


def parse_overlay_local(decision_md: str) -> dict | None:
    """Extract the risk-overlay numbers the PM appends to its final decision.

    Returns ``None`` when the overlay section is absent (risk overlay off, or
    the field is empty); a dict of the parsed fields (possibly partial — e.g.
    ``position_value`` is absent when the overlay uses list form without a
    parenthetical dollar amount) when present.
    """
    if not decision_md or _OVERLAY_MARKER not in decision_md:
        return None
    block = decision_md[decision_md.index(_OVERLAY_MARKER) :]
    out: dict[str, str] = {}
    for key, pat in _OVERLAY_FIELDS.items():
        m = re.search(pat, block)
        if m:
            out[key] = m.group(1).strip()
    return out


def _strategy_dir(ticker: str) -> Path:
    """``LOGS_ROOT/<ticker>/YiAgentsStrategy_logs`` (ticker already validated)."""
    return LOGS_ROOT / ticker / "YiAgentsStrategy_logs"


def _dates_for(ticker: str) -> list[str]:
    """Sorted (ascending) analysis dates with a saved full_states_log."""
    sdir = _strategy_dir(ticker)
    if not sdir.is_dir():
        return []
    dates: list[str] = []
    for f in sdir.glob("full_states_log_*.json"):
        m = _DATE_RE.search(f.name)
        if m:
            dates.append(m.group(1))
    return sorted(dates)


def _latest_rating(ticker: str, latest_date: str) -> str | None:
    """Rating of the most recent run, for the home grid + distribution summary.

    Returns ``None`` only when the JSON can't be read; a readable run always
    yields a rating (``parse_rating`` defaults to ``Hold`` when no tier word
    appears, matching ``load_run``'s behavior).
    """
    path = _strategy_dir(ticker) / f"full_states_log_{latest_date}.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return parse_rating(state.get("final_trade_decision", "") or "")


def list_tickers() -> list[dict]:
    """One entry per ticker that has at least one completed run."""
    out: list[dict] = []
    if not LOGS_ROOT.is_dir():
        return out
    for d in sorted(LOGS_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not d.is_dir() or d.name in _NON_TICKER_DIRS:
            continue
        dates = _dates_for(d.name)
        if not dates:
            continue
        out.append(
            {
                "ticker": d.name,
                "latest_date": dates[-1],  # _dates_for is ascending
                "latest_rating": _latest_rating(d.name, dates[-1]),
                "run_count": len(dates),
            }
        )
    return out


def list_reports(ticker: str) -> list[dict]:
    """Wall-clock-stamped report dirs ``<TICKER>_<stamp>/``, newest first."""
    try:
        safe_ticker_component(ticker)
    except ValueError:
        return []
    if not REPORTS_ROOT.is_dir():
        return []
    reports: list[dict] = []
    for d in REPORTS_ROOT.glob(f"{ticker}_*"):
        if not d.is_dir():
            continue
        cr = d / "complete_report.md"
        try:
            mtime = cr.stat().st_mtime if cr.is_file() else d.stat().st_mtime
        except OSError:
            continue
        reports.append(
            {
                "dir": d.name,
                "mtime": mtime,
                "complete": cr.is_file(),
            }
        )
    reports.sort(key=lambda r: r["mtime"], reverse=True)
    return reports


def list_runs(ticker: str) -> dict:
    dates = _dates_for(ticker)
    return {
        "ticker": ticker,
        "dates": dates,
        # Per-date rating lets the detail view show a small badge next to each
        # analysis date (a rating evolution over time). Same light parse as
        # ``_latest_rating``; None only when the JSON is unreadable.
        "date_ratings": [{"date": d, "rating": _latest_rating(ticker, d)} for d in dates],
        "reports": list_reports(ticker),
    }


def load_node_perf(ticker: str, date: str) -> dict | None:
    """Per-node wall-clock + token telemetry, if ``--profile`` wrote it."""
    path = _strategy_dir(ticker) / f"node_perf_{date}.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_run(ticker: str, date: str) -> dict | None:
    """Full report view: rating badge + overlay card + 5 collapsible sections."""
    path = _strategy_dir(ticker) / f"full_states_log_{date}.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    final = state.get("final_trade_decision", "") or ""
    rating = parse_rating(final)
    overlay = parse_overlay_local(final)

    ideb = state.get("investment_debate_state") or {}
    rdeb = state.get("risk_debate_state") or {}

    return {
        "ticker": ticker,
        "trade_date": state.get("trade_date", date),
        "rating": rating,
        "company_of_interest": state.get("company_of_interest", "") or "",
        "sections": {
            "market_report": state.get("market_report", "") or "",
            "sentiment_report": state.get("sentiment_report", "") or "",
            "news_report": state.get("news_report", "") or "",
            # Fundamentals Analyst is dropped for crypto (spot + perp); null hides
            # its section in the UI rather than rendering an empty card.
            "fundamentals_report": state.get("fundamentals_report") or None,
            "investment_debate": {
                "bull_history": ideb.get("bull_history", "") or "",
                "bear_history": ideb.get("bear_history", "") or "",
                "history": ideb.get("history", "") or "",
                "current_response": ideb.get("current_response", "") or "",
                "judge_decision": ideb.get("judge_decision", "") or "",
            },
            # JSON write key is trader_investment_decision (renamed from
            # trader_investment_plan in _log_state L705). Reading any other key
            # leaves this section blank.
            "trader_decision": state.get("trader_investment_decision", "") or "",
            "risk_debate": {
                "aggressive_history": rdeb.get("aggressive_history", "") or "",
                "conservative_history": rdeb.get("conservative_history", "") or "",
                "neutral_history": rdeb.get("neutral_history", "") or "",
                "history": rdeb.get("history", "") or "",
                "judge_decision": rdeb.get("judge_decision", "") or "",
            },
            "investment_plan": state.get("investment_plan", "") or "",
            "final_trade_decision": final,
        },
        "overlay": overlay,
        "node_perf": load_node_perf(ticker, date),
    }
