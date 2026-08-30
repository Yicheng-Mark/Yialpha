"""Regression tests for the interactive CLI's evidence chain + input guards.

Covers three 2026-08 audit fixes:

- D1: the interactive ``run_analysis`` path streams ``graph.stream`` directly
  and used to bypass ``YiAlphaGraph._run_graph``'s evidence contract — no
  ``full_states_log_<date>.json`` (invisible in the web history) and no
  ``data_quality`` block (the DEGRADED banner never rendered). The CLI now
  calls ``quality.ensure_run_context()`` before streaming and the new public
  ``YiAlphaGraph.finalize_streamed_run`` after, reusing ``_log_state``.
- D3: pure-dot tickers ("..") passed the charset check and escaped the
  results directory as a path component.
- D4: a cancelled region prompt (Esc/Ctrl-C → None) crashed on tuple unpack.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

# Imported at MODULE scope on purpose: yialpha.cli.main calls setup_logging()
# at import time, which replaces every root handler — including the one the
# caplog fixture installs per test. A lazy in-test import would strip caplog's
# handler right before the warning is emitted (observed: the test passes only
# when an earlier test imported the module first). Collection happens before
# any fixture runs, so the import side effect is long done by then.
from yialpha.cli.main import _store_cli_decision
from yialpha.dataflows import quality


@pytest.fixture(autouse=True)
def _clean_quality():
    quality.reset_quality()
    yield
    quality.reset_quality()


def _full_streamed_state(ticker: str = "NVDA", trade_date: str = "2026-06-10") -> dict:
    """The merged final_state shape the CLI's stream loop produces.

    ``stream_mode="values"`` chunks carry every key of the initial state, so
    the merged state has all keys ``_log_state`` direct-indexes.
    """
    return {
        "company_of_interest": ticker,
        "trade_date": trade_date,
        "market_report": "m",
        "sentiment_report": "s",
        "news_report": "n",
        "fundamentals_report": "f",
        "investment_debate_state": {
            "bull_history": "b", "bear_history": "e", "history": "h",
            "current_response": "c", "judge_decision": "j",
        },
        "trader_investment_plan": "t",
        "risk_debate_state": {
            "aggressive_history": "a", "conservative_history": "c",
            "neutral_history": "n", "history": "h", "judge_decision": "j",
        },
        "investment_plan": "i",
        "final_trade_decision": "Rating: BUY",
        "pm_rating": "Rating: BUY",
    }


def _make_graph(tmp_path):
    """Minimal YiAlphaGraph stand-in exposing finalize_streamed_run."""
    from yialpha.graph.trading_graph import YiAlphaGraph

    graph = YiAlphaGraph.__new__(YiAlphaGraph)  # skip __init__: attrs below suffice
    graph.ticker = None
    graph.log_states_dict = {}
    object.__setattr__(graph, "config", {"results_dir": str(tmp_path / "results")})
    return graph


# --------------------------------------------------------------------------- #
# D1 — the streamed CLI run lands the same evidence as propagate()
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_finalize_streamed_run_writes_full_states_log_and_quality(tmp_path):
    graph = _make_graph(tmp_path)

    # The exact sequence run_analysis now performs around its stream loop.
    quality.ensure_run_context()
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "router degraded")
    final_state = _full_streamed_state()
    final_state = graph.finalize_streamed_run("NVDA", "2026-06-10", final_state)

    log_path = (
        tmp_path / "results" / "NVDA" / "YiAlphaStrategy_logs"
        / "full_states_log_2026-06-10.json"
    )
    assert log_path.exists()
    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert data["data_quality"]["core_sentinel_count"] == 1
    assert data["data_quality"]["sentinels"][0]["method"] == "get_stock_data"
    assert data["final_trade_decision"] == "Rating: BUY"

    # The consumed block is attached to the returned state (report writer /
    # web UI render the DEGRADED banner from it), and the accumulator reset.
    assert final_state["data_quality"]["core_sentinel_count"] == 1
    assert quality.snapshot_quality() == []
    # _log_state keys the log dir off self.ticker; finalize must bind it.
    assert graph.ticker == "NVDA"


@pytest.mark.unit
def test_finalize_streamed_run_without_context_still_writes_empty_quality(tmp_path):
    graph = _make_graph(tmp_path)
    final_state = graph.finalize_streamed_run("AAPL", "2026-01-10", _full_streamed_state("AAPL", "2026-01-10"))
    assert final_state["data_quality"]["core_sentinel_count"] == 0
    assert final_state["data_quality"]["sentinels"] == []


@pytest.mark.unit
def test_store_cli_decision_uses_get_and_warns_when_missing(tmp_path, caplog):
    """A streamed state missing final_trade_decision must not KeyError the UI."""
    calls: list[dict] = []
    graph = SimpleNamespace(
        memory_log=SimpleNamespace(store_decision=lambda **kw: calls.append(kw))
    )

    with caplog.at_level("WARNING", logger="yialpha.cli.main"):
        _store_cli_decision(graph, "NVDA", "2026-06-10", {})

    assert calls == [{
        "ticker": "NVDA",
        "trade_date": "2026-06-10",
        "final_trade_decision": "",
    }]
    assert any("no final_trade_decision" in r.message for r in caplog.records)


@pytest.mark.unit
def test_store_cli_decision_passes_decision_through(caplog):
    calls: list[dict] = []
    graph = SimpleNamespace(
        memory_log=SimpleNamespace(store_decision=lambda **kw: calls.append(kw))
    )

    with caplog.at_level("WARNING", logger="yialpha.cli.main"):
        _store_cli_decision(
            graph, "NVDA", "2026-06-10", {"final_trade_decision": "Rating: BUY"}
        )

    assert calls[0]["final_trade_decision"] == "Rating: BUY"
    assert not any("no final_trade_decision" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# D5 — the streamed CLI run pins the analysis date (PIT clamp) like _run_graph
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestStreamedRunPinsAnalysisDate:
    def test_pins_clears_and_survives_exceptions(self):
        from yialpha.dataflows.utils import (
            get_analysis_date,
            pinned_analysis_date,
            set_analysis_date,
        )

        set_analysis_date(None)
        assert get_analysis_date() is None
        with pinned_analysis_date("2026-06-10"):
            assert get_analysis_date() == "2026-06-10"
            # The pin is what the vendor-layer clamp reads — a window past the
            # analysis date must be clamped INSIDE the block.
            from yialpha.dataflows.utils import clamp_end_date

            assert clamp_end_date("2026-08-01", get_analysis_date()) == "2026-06-10"
        assert get_analysis_date() is None

        with pytest.raises(RuntimeError), pinned_analysis_date("2026-06-10"):
            assert get_analysis_date() == "2026-06-10"
            raise RuntimeError("stream crashed")
        # A crashed run must not leave the clamp pinned into the next one.
        assert get_analysis_date() is None

        # Empty string means live mode — never pinned.
        with pinned_analysis_date(""):
            assert get_analysis_date() is None

    def test_run_analysis_source_pins_before_streaming(self):
        """Wiring guard: the streamed path must enter the pin context.

        ``run_analysis`` is an interactive Live-UI function that cannot be
        invoked headlessly here, so this asserts on its source the same way
        the signature-smoke guards do: the ``with`` statement that hosts the
        stream loop must include ``pinned_analysis_date`` — the exact
        omission the 2026-08-16 audit caught (clamp dead on the CLI path).
        """
        import inspect

        from yialpha.cli import main as cli_main

        src = inspect.getsource(cli_main.run_analysis)
        assert "pinned_analysis_date(selections[\"analysis_date\"])" in src
        # ...and the pin must appear before the stream loop that needs it.
        assert src.index("pinned_analysis_date") < src.index("graph.graph.stream")


# --------------------------------------------------------------------------- #
# D3 — pure-dot tickers are path escapes, reject them
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestPureDotTickerRejected:
    def test_is_valid_ticker_input_rejects_dot_strings(self):
        from yialpha.cli.utils import is_valid_ticker_input

        for bad in ("..", "...", ".", " . ", ".."):
            assert not is_valid_ticker_input(bad), bad

    def test_is_valid_ticker_input_still_accepts_real_symbols(self):
        from yialpha.cli.utils import is_valid_ticker_input

        for good in ("", "SPY", "BRK.B", "0700.HK", "GC=F", "^GSPC", "BTC-USD"):
            assert is_valid_ticker_input(good), good

    def test_get_ticker_exits_cleanly_on_dot_escape(self, monkeypatch):
        """A '..' that somehow reaches get_ticker exits 1, never builds paths."""
        from yialpha.cli import utils

        monkeypatch.setattr(
            utils.questionary, "text",
            lambda *a, **k: SimpleNamespace(ask=lambda: ".."),
        )
        with pytest.raises(SystemExit) as exc:
            utils.get_ticker()
        assert exc.value.code == 1

    def test_get_ticker_accepts_normal_symbol(self, monkeypatch):
        from yialpha.cli import utils

        monkeypatch.setattr(
            utils.questionary, "text",
            lambda *a, **k: SimpleNamespace(ask=lambda: " brk.b "),
        )
        assert utils.get_ticker() == "BRK.B"


# --------------------------------------------------------------------------- #
# D4 — cancelled region prompts exit cleanly (None unpack used to TypeError)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestRegionPromptCancel:
    @staticmethod
    def _mock_select(monkeypatch, ask_result):
        from yialpha.cli import utils

        monkeypatch.setattr(
            utils.questionary, "select",
            lambda *a, **k: SimpleNamespace(ask=lambda: ask_result),
        )

    def test_cancelled_region_prompts_exit_1(self, monkeypatch):
        from yialpha.cli import utils

        for fn in (utils.ask_glm_region, utils.ask_qwen_region, utils.ask_minimax_region):
            self._mock_select(monkeypatch, None)
            with pytest.raises(SystemExit) as exc:
                fn()
            assert exc.value.code == 1

    def test_region_prompts_return_selection(self, monkeypatch):
        from yialpha.cli import utils

        selections = {
            utils.ask_glm_region: ("glm", "https://api.z.ai/api/paas/v4/"),
            utils.ask_qwen_region: ("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            utils.ask_minimax_region: ("minimax", "https://api.minimax.io/v1"),
        }
        for fn, expected in selections.items():
            self._mock_select(monkeypatch, expected)
            assert fn() == expected
