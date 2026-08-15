"""Report parity: the shared writer produces the report tree for the CLI and the
programmatic API alike (#1037)."""

from types import SimpleNamespace

import pytest

from yiagents.graph.trading_graph import YiAgentsGraph
from yiagents.reporting import write_report_tree


def _state():
    return {
        "market_report": "MKT",
        "news_report": "NEWS",
        "investment_debate_state": {"judge_decision": "RM PLAN"},
        "trader_investment_plan": "TRADE",
        "risk_debate_state": {"judge_decision": "PM DECISION"},
    }


@pytest.mark.unit
def test_write_report_tree_creates_files(tmp_path):
    out = write_report_tree(_state(), "AAPL", tmp_path)
    assert out.name == "complete_report.md"
    assert (tmp_path / "1_analysts" / "market.md").read_text() == "MKT"
    assert (tmp_path / "1_analysts" / "news.md").read_text() == "NEWS"
    assert (tmp_path / "2_research" / "manager.md").read_text() == "RM PLAN"
    assert (tmp_path / "3_trading" / "trader.md").read_text() == "TRADE"
    assert (tmp_path / "5_portfolio" / "decision.md").read_text() == "PM DECISION"
    complete = out.read_text()
    assert "Trading Analysis Report: AAPL" in complete
    assert "MKT" in complete and "PM DECISION" in complete


@pytest.mark.unit
def test_report_prefers_final_risk_adjusted_decision(tmp_path):
    state = _state()
    state["final_trade_decision"] = (
        "**Rating**: Buy\n\n## Quantitative Risk Overlay\n"
        "- **Target Weight**: 5.0%\n- **Stop Loss**: 95.00"
    )
    out = write_report_tree(state, "AAPL", tmp_path)
    portfolio = (tmp_path / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
    assert "Quantitative Risk Overlay" in portfolio
    assert "Stop Loss" in portfolio
    assert portfolio != state["risk_debate_state"]["judge_decision"]
    assert "Final Risk-Adjusted Decision" in out.read_text(encoding="utf-8")


@pytest.mark.unit
def test_save_reports_explicit_path(tmp_path):
    # Unbound: with an explicit save_path, the method doesn't touch self/config.
    out = YiAgentsGraph.save_reports(None, _state(), "AAPL", save_path=tmp_path)
    assert (tmp_path / "complete_report.md").exists()
    assert out == tmp_path / "complete_report.md"


@pytest.mark.unit
def test_save_reports_defaults_under_results_dir(tmp_path):
    mock_self = SimpleNamespace(config={"results_dir": str(tmp_path)})
    out = YiAgentsGraph.save_reports(mock_self, _state(), "AAPL")
    assert out.exists()
    assert out.parent.parent.name == "reports"  # results_dir/reports/AAPL_<stamp>/...
    assert out.parent.name.startswith("AAPL_")


@pytest.mark.unit
def test_degraded_run_renders_data_quality_banner(tmp_path):
    state = _state()
    state["data_quality"] = {
        "core_sentinel_count": 2,
        "optional_sentinel_count": 1,
        "sentinels": [
            {"method": "get_stock_data", "kind": "no_data", "detail": "all vendors timed out"},
            {"method": "get_news", "kind": "no_data", "detail": "yfinance empty"},
            {"method": "get_form4_insider_trading", "kind": "optional_unavailable",
             "detail": "US-listed only"},
        ],
    }
    out = write_report_tree(state, "AAPL", tmp_path)
    complete = out.read_text(encoding="utf-8")
    # Banner sits between header and section I so a degraded run can't be
    # mistaken for a fully-fed one.
    assert complete.index("DEGRADED RUN") < complete.index("## I. Analyst Team Reports")
    assert "2 core data categories" in complete
    assert "1 optional enrichment category unavailable" in complete
    assert "`get_stock_data`" in complete and "all vendors timed out" in complete


@pytest.mark.unit
def test_healthy_run_has_no_data_quality_section(tmp_path):
    out = write_report_tree(_state(), "AAPL", tmp_path)
    complete = out.read_text(encoding="utf-8")
    assert "Data Quality" not in complete
    # An all-zero block (counts present but nothing degraded) is also silent.
    state = _state()
    state["data_quality"] = {"core_sentinel_count": 0, "optional_sentinel_count": 0,
                             "sentinels": []}
    out2 = write_report_tree(state, "AAPL", tmp_path / "r2")
    assert "Data Quality" not in out2.read_text(encoding="utf-8")
