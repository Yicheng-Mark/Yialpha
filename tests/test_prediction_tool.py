"""submit_prediction tool + per-run capture buffer (V2.1 batch C).

Covers: toolkit gating (flag off / non-perp keep the tool list and prompt
surface byte-identical; flag on + perp binds it with the right instrument),
per-entry validation, the buffer lifecycle (begin -> tool append ->
settle/flush), ledger submission wiring (analyst / instrument / scope /
evidence chain), the no-run-context no-op contract, the fail-soft ledger
exception path, the empty-buffer quality sentinel, and ONE full-node
market-analyst integration (capture LLM actually calls the tool across a
simulated tool-loop round trip; predictions AND evidence land under one
run id).

conftest's autouse fixtures hold ``prediction_ledger`` OFF and point the
ledger DB at a per-test tmp file; opt-ins use ``set_config`` in the body.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yialpha.agents.analysts.market_analyst as market_analyst
import yialpha.dataflows.perp_bundle as pb
from yialpha.agents.utils import prediction_tools as pt
from yialpha.dataflows import quality
from yialpha.dataflows.config import set_config
from yialpha.ledger.evidence import evidence_for_run, record_evidence, register_run
from yialpha.ledger.models import REPLAYABILITY_LIVE_ONLY, SCOPE_CONTRACT
from yialpha.ledger.predictions import predictions_for_run
from yialpha.ledger.run_context import (
    reset_ledger_run_context,
    set_ledger_run_context,
)

_TODAY = date.today().isoformat()
_RUN_ID = "run-btc-pred-1"


@pytest.fixture(autouse=True)
def _clean_record_stage():
    """Every test starts/ends with no run context, no capture, no events."""
    reset_ledger_run_context()
    pt.reset_prediction_capture_for_test()
    quality.reset_quality()
    yield
    reset_ledger_run_context()
    pt.reset_prediction_capture_for_test()
    quality.reset_quality()


def _bind_perp_run() -> None:
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY
    )
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)


def _entries() -> list[dict[str, object]]:
    return [
        {
            "horizon_days": 1,
            "direction": "up",
            "prob_up": 0.55,
            "expected_return": 0.01,
            "confidence": 0.6,
        },
        {"horizon_days": 5, "direction": "flat", "prob_up": 0.5},
        {
            "horizon_days": 21,
            "direction": "down",
            "prob_up": 0.35,
            "target_price": 95000.0,
            "target_currency": "USDT",
            "price_basis": "mark",
        },
    ]


class _ToolCaptureLLM(Runnable):
    """Capture LLM (tests/test_perp_bundle.py pattern) that also records
    the bound tool list so toolkit gating is asserted directly."""

    def __init__(self, responses: list[AIMessage] | None = None):
        super().__init__()
        self.prompt = None
        self.bound_tools: list[str] = []
        self.bound: list = []
        self.calls = 0
        self._responses = responses or []

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        self.calls += 1
        if self._responses:
            return self._responses[min(self.calls - 1, len(self._responses) - 1)]
        return AIMessage(content="FINAL REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound = list(tools)
        self.bound_tools = [tool.name for tool in tools]
        return self


def _perp_state(ticker: str = "BTCUSDT") -> dict[str, object]:
    return {
        "trade_date": _TODAY,
        "company_of_interest": ticker,
        "asset_type": "crypto_perp",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


# ---- toolkit gating -----------------------------------------------------------


@pytest.mark.unit
def test_toolkit_flag_off_keeps_tool_list_and_prompt_unchanged():
    # conftest holds prediction_ledger OFF: the perp run must bind the exact
    # baseline tool set — no submit_prediction tool, no prompt mention.
    llm = _ToolCaptureLLM()
    market_analyst.create_market_analyst(llm)(_perp_state())
    assert "submit_prediction" not in llm.bound_tools
    assert "submit_prediction" not in str(llm.prompt)
    # The baseline perp tool list is intact (spot-check the anchors).
    for name in ("get_binance_klines", "get_binance_funding_rate"):
        assert name in llm.bound_tools


@pytest.mark.unit
def test_toolkit_non_perp_never_binds_the_tool():
    set_config({"prediction_ledger": True})
    llm = _ToolCaptureLLM()
    market_analyst.create_market_analyst(llm)(
        {
            "trade_date": _TODAY,
            "company_of_interest": "MU",
            "asset_type": "stock",
            "instrument_context": "CTX",
            "messages": [HumanMessage(content="analyze")],
        }
    )
    assert "submit_prediction" not in llm.bound_tools
    assert "submit_prediction" not in str(llm.prompt)


@pytest.mark.unit
def test_toolkit_flag_on_perp_binds_tool_named_after_contract():
    set_config({"prediction_ledger": True})
    llm = _ToolCaptureLLM()
    market_analyst.create_market_analyst(llm)(_perp_state())
    assert "submit_prediction" in llm.bound_tools
    assert "submit_prediction" in str(llm.prompt)
    bound = [tool for tool in llm.bound if tool.name == "submit_prediction"]
    assert len(bound) == 1
    # The description names the instrument (the only prompt surface).
    assert "BTCUSDT" in (bound[0].description or "")


# ---- tool: per-entry validation ----------------------------------------------


@pytest.mark.unit
def test_tool_accepts_full_three_horizon_call():
    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    result = pt.submit_prediction.invoke({"predictions": _entries()})
    assert pt.prediction_capture_pending() is True
    assert "accepted horizons [1, 5, 21]" in result
    assert "Rejected" not in result


@pytest.mark.unit
def test_tool_rejects_invalid_entries_individually():
    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    result = pt.submit_prediction.invoke(
        {
            "predictions": [
                {"horizon_days": 5, "direction": "flat", "prob_up": 0.5},
                {"horizon_days": 3, "direction": "up"},  # off-ladder horizon
                {"horizon_days": 1, "direction": "sideways"},  # bad direction
                {"horizon_days": 21, "direction": "up", "prob_up": 1.5},  # bad prob
                {"horizon_days": 1, "direction": "up", "price_basis": "vwap"},
            ]
        }
    )
    assert "accepted horizons [5]" in result
    assert "Rejected" in result
    assert "entry 2" in result and "horizon" in result
    assert "entry 3" in result and "sideways" in result
    assert "entry 4" in result and "prob_up" in result
    assert "entry 5" in result and "price_basis" in result
    # Only the valid entry reached the buffer.
    pt.flush_predictions()
    rows = predictions_for_run(_RUN_ID)
    assert [row.horizon_days for row in rows] == [5]


@pytest.mark.unit
def test_tool_without_capture_returns_explanatory_string():
    _bind_perp_run()  # context bound, but begin never ran: no active capture
    result = pt.submit_prediction.invoke({"predictions": _entries()})
    assert result.startswith("submit_prediction ignored")
    assert pt.prediction_capture_pending() is False


@pytest.mark.unit
def test_begin_is_idempotent_across_tool_loop_reentries():
    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    pt.submit_prediction.invoke({"predictions": _entries()[:1]})
    # The node body re-runs per tool round; begin must not drop the entry.
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    pt.submit_prediction.invoke({"predictions": _entries()[1:]})
    pt.flush_predictions()
    assert [row.horizon_days for row in predictions_for_run(_RUN_ID)] == [1, 5, 21]


# ---- flush / settle -----------------------------------------------------------


@pytest.mark.unit
def test_flush_writes_rows_with_run_context_and_evidence_chain():
    _bind_perp_run()
    record_evidence(
        run_id=_RUN_ID,
        source="perp_market_bundle",
        category="binance_perp",
        symbol="BTCUSDT",
        scope=SCOPE_CONTRACT,
        payload="MARKET BLOCK",
        replayability=REPLAYABILITY_LIVE_ONLY,
        analysis_as_of=_TODAY,
    )
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    pt.submit_prediction.invoke({"predictions": _entries()})
    written = pt.flush_predictions()
    assert len(written) == 3
    rows = predictions_for_run(_RUN_ID)
    assert [row.horizon_days for row in rows] == [1, 5, 21]
    for row in rows:
        assert row.run_id == _RUN_ID
        assert row.analyst == "market"
        assert row.instrument_id == "BTCUSDT"
        assert row.prediction_scope == SCOPE_CONTRACT
        assert row.analysis_as_of == _TODAY
        assert list(row.evidence_ids) == [
            ev.evidence_id for ev in evidence_for_run(_RUN_ID)
        ]
    # A repeated settle must not conflict: the flushed capture is retired.
    assert pt.flush_predictions() == []


@pytest.mark.unit
def test_flush_without_run_context_is_a_noop():
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)  # no ctx: no-op
    pt.submit_prediction.invoke({"predictions": _entries()})  # ignored, no raise
    assert pt.flush_predictions() == []
    assert predictions_for_run(_RUN_ID) == []


@pytest.mark.unit
def test_flush_swallows_ledger_exceptions(monkeypatch):
    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    pt.submit_prediction.invoke({"predictions": _entries()})

    def _boom(*_args, **_kwargs):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(pt, "submit_predictions", _boom)
    assert pt.flush_predictions() == []
    assert predictions_for_run(_RUN_ID) == []


@pytest.mark.unit
def test_settle_empty_buffer_records_optional_sentinel():
    quality.ensure_run_context()
    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    pt.settle_prediction_capture("market", "BTCUSDT")
    events = quality.snapshot_quality()
    assert len(events) == 1
    assert events[0]["method"] == "submit_prediction"
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    assert "BTCUSDT" in events[0]["detail"]


@pytest.mark.unit
def test_settle_with_entries_writes_rows_and_no_sentinel():
    quality.ensure_run_context()
    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    pt.submit_prediction.invoke({"predictions": _entries()})
    pt.settle_prediction_capture("market", "BTCUSDT")
    assert quality.snapshot_quality() == []
    assert len(predictions_for_run(_RUN_ID)) == 3


# ---- sentiment: dedicated bound round (structured path has no tool loop) -----


class _FakeSentimentLLM:
    """Structured-output-less LLM whose bound round returns one tool call."""

    def __init__(self, response: AIMessage):
        self._response = response
        self.bound_tools: list[str] = []

    def with_structured_output(self, schema):  # noqa: ARG002
        raise AttributeError("no structured output in tests")

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound_tools = [tool.name for tool in tools]
        return self

    def invoke(self, messages, config=None, **kwargs):  # noqa: ARG002
        return self._response


@pytest.mark.unit
def test_sentiment_prediction_round_captures_and_settles(monkeypatch):
    import yialpha.agents.analysts.sentiment_analyst as sa

    _bind_perp_run()
    set_config({"prediction_ledger": True})
    monkeypatch.setattr(
        sa,
        "_fetch_sentiment_sources",
        lambda *a, **k: ("NEWS", "STOCKTWITS", "REDDIT", "SQUARE"),
    )
    monkeypatch.setattr(
        sa, "invoke_structured_or_freetext", lambda *a, **k: "REPORT"
    )
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "submit_prediction",
                "args": {"predictions": _entries()},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    llm = _FakeSentimentLLM(tool_call_message)
    sa.create_sentiment_analyst(llm)(_perp_state())

    assert llm.bound_tools == ["submit_prediction"]
    rows = predictions_for_run(_RUN_ID)
    assert [row.horizon_days for row in rows] == [1, 5, 21]
    assert all(row.analyst == "sentiment" for row in rows)
    assert all(row.prediction_scope == SCOPE_CONTRACT for row in rows)
    assert quality.snapshot_quality() == []


@pytest.mark.unit
def test_sentiment_flag_off_runs_no_prediction_round(monkeypatch):
    import yialpha.agents.analysts.sentiment_analyst as sa

    monkeypatch.setattr(
        sa,
        "_fetch_sentiment_sources",
        lambda *a, **k: ("NEWS", "STOCKTWITS", "REDDIT", None),
    )
    monkeypatch.setattr(
        sa, "invoke_structured_or_freetext", lambda *a, **k: "REPORT"
    )
    llm = _FakeSentimentLLM(AIMessage(content="x"))
    sa.create_sentiment_analyst(llm)(_perp_state())
    assert llm.bound_tools == []  # bind_tools never called: no extra round
    assert quality.snapshot_quality() == []


# ---- full-node integration: market analyst ------------------------------------


class _ToolCallingLLM(Runnable):
    """Round 1: execute the bound submit_prediction tool (as the ToolNode
    would) and answer with its tool call. Round 2: the final report."""

    def __init__(self, entries: list[dict[str, object]]):
        super().__init__()
        self.entries = entries
        self.prompt = None
        self.bound: list = []
        self.calls = 0

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        self.calls += 1
        if self.calls == 1:
            tool = next(t for t in self.bound if t.name == "submit_prediction")
            tool.invoke({"predictions": self.entries})
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_prediction",
                        "args": {"predictions": self.entries},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="FINAL REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound = list(tools)
        return self


@pytest.mark.unit
def test_market_node_full_integration_predictions_and_evidence(monkeypatch):
    set_config({"prediction_ledger": True, "perp_market_bundle": True})
    _bind_perp_run()
    monkeypatch.setattr(
        pb, "fetch_perp_market_bundle", lambda s, d: {"symbol": s, "as_of": d}
    )
    monkeypatch.setattr(pb, "render_perp_bundle_block", lambda b: "MARKET BLOCK")

    llm = _ToolCallingLLM(_entries())
    node = market_analyst.create_market_analyst(llm)
    state = _perp_state()

    # Round 1: the model files its blind predictions (tool call pending, so
    # no settlement yet — exactly the graph's analyst -> ToolNode -> analyst
    # round trip, here driven by re-invoking the node with the reply).
    result1 = node(state)
    assert result1["market_report"] == ""
    assert predictions_for_run(_RUN_ID) == []  # not flushed mid-loop

    state2 = dict(state)
    state2["messages"] = list(state["messages"]) + [result1["messages"][0]]
    result2 = node(state2)

    assert result2["market_report"] == "FINAL REPORT"
    rows = predictions_for_run(_RUN_ID)
    assert [row.horizon_days for row in rows] == [1, 5, 21]
    assert all(row.analyst == "market" for row in rows)
    assert all(row.instrument_id == "BTCUSDT" for row in rows)
    assert all(row.prediction_scope == SCOPE_CONTRACT for row in rows)

    evidence = evidence_for_run(_RUN_ID)
    assert [ev.source for ev in evidence] == ["perp_market_bundle"]
    evidence_ids = [ev.evidence_id for ev in evidence]
    assert all(list(row.evidence_ids) == evidence_ids for row in rows)
    assert quality.snapshot_quality() == []


# ---- graph ToolNode dispatch wiring (round-3e P0 regression) ------------------
# The integration above lets the mock LLM invoke the bound tool ITSELF ("as the
# ToolNode would"), which hid the real gap: the graph's market/news/fundamentals
# ToolNodes never registered submit_prediction, so live tool calls died with
# "not a valid tool". These tests drive the REAL ToolNodes built by
# YiAlphaGraph._create_tool_nodes.


def _graph_tool_nodes():
    """The graph's real ToolNode map (test_market_toolnode.py stub pattern)."""
    from types import SimpleNamespace

    from yialpha.graph.trading_graph import YiAlphaGraph

    fake_self = SimpleNamespace(quick_thinking_llm=object())  # PoT closure only
    return YiAlphaGraph._create_tool_nodes(fake_self)


def _invoke_tool_node(node, ai_message: AIMessage):
    """Drive a real ToolNode the way the compiled graph does (runtime injected)."""
    from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
    from langgraph.runtime import Runtime

    return node.invoke(
        {"messages": [ai_message]},
        config={CONF: {CONFIG_KEY_RUNTIME: Runtime()}},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("node_name", "analyst"),
    [("market", "market"), ("news", "news"), ("fundamentals", "fundamentals")],
)
def test_graph_tool_node_dispatches_submit_prediction(node_name, analyst):
    _bind_perp_run()
    pt.begin_prediction_capture(analyst, "BTCUSDT", SCOPE_CONTRACT)

    tool_nodes = _graph_tool_nodes()
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "submit_prediction",
                "args": {"predictions": _entries()},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    out = _invoke_tool_node(tool_nodes[node_name], ai_message)

    tool_message = out["messages"][0]
    content = str(tool_message.content)
    assert "not a valid tool" not in content
    assert getattr(tool_message, "status", None) != "error"
    assert "accepted horizons [1, 5, 21]" in content
    # The generic instance routed the entries into THIS analyst's capture via
    # the _ACTIVE_CAPTURE_KEY registry entry (not a ContextVar sibling write).
    assert pt.prediction_capture_pending() is True


# ---- dedicated fallback prediction rounds (market / news / fundamentals) ------


class _FallbackLLM(Runnable):
    """Report round files NOTHING (final report immediately); the dedicated
    fallback round then answers with one submit_prediction tool call."""

    def __init__(self):
        super().__init__()
        self.bound_per_call: list[list[str]] = []
        self.calls = 0

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="FINAL REPORT", tool_calls=[])
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_prediction",
                    "args": {"predictions": _entries()},
                    "id": "call-2",
                    "type": "tool_call",
                }
            ],
        )

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound_per_call.append([tool.name for tool in tools])
        return self


@pytest.mark.unit
def test_market_fallback_round_files_when_tool_loop_did_not(monkeypatch):
    set_config({"prediction_ledger": True, "perp_market_bundle": True})
    quality.ensure_run_context()
    _bind_perp_run()
    monkeypatch.setattr(
        pb, "fetch_perp_market_bundle", lambda s, d: {"symbol": s, "as_of": d}
    )
    monkeypatch.setattr(pb, "render_perp_bundle_block", lambda b: "MARKET BLOCK")

    llm = _FallbackLLM()
    result = market_analyst.create_market_analyst(llm)(_perp_state())

    assert result["market_report"] == "FINAL REPORT"
    # The fallback round bound ONLY the prediction tool.
    assert llm.bound_per_call[-1] == ["submit_prediction"]
    rows = predictions_for_run(_RUN_ID)
    assert [row.horizon_days for row in rows] == [1, 5, 21]
    assert all(row.analyst == "market" for row in rows)
    assert all(row.instrument_id == "BTCUSDT" for row in rows)
    assert all(row.prediction_scope == SCOPE_CONTRACT for row in rows)
    # Filed -> no optional_unavailable sentinel.
    assert quality.snapshot_quality() == []


@pytest.mark.unit
def test_news_fallback_round_files_under_underlying_scope(monkeypatch):
    import yialpha.agents.analysts.news_analyst as na
    from yialpha.ledger.models import SCOPE_UNDERLYING

    set_config({"prediction_ledger": True})
    quality.ensure_run_context()
    run_id = "run-mu-pred-fb"
    set_ledger_run_context(
        run_id, "MUUSDT", "crypto_perp", "stock_perp", _TODAY
    )
    register_run(run_id, "MUUSDT", "crypto_perp", "stock_perp", _TODAY)
    monkeypatch.setattr(na, "_fetch_company_news", lambda u, s, e: "COMPANY BLOCK")
    monkeypatch.setattr(na, "_fetch_perp_contract_news", lambda t, d: "CONTRACT BLOCK")

    llm = _FallbackLLM()
    result = na.create_news_analyst(llm)(_perp_state("MUUSDT"))

    assert result["news_report"] == "FINAL REPORT"
    assert llm.bound_per_call[-1] == ["submit_prediction"]
    rows = predictions_for_run(run_id)
    assert [row.horizon_days for row in rows] == [1, 5, 21]
    assert all(row.analyst == "news" for row in rows)
    # Stock-perp semantics: the forecast is filed for the underlying equity.
    assert all(row.instrument_id == "MU" for row in rows)
    assert all(row.prediction_scope == SCOPE_UNDERLYING for row in rows)
    assert quality.snapshot_quality() == []


@pytest.mark.unit
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "MUUSDT"])
def test_all_horizons_freeze_submission_reference_before_later_market_and_flush(monkeypatch, symbol):
    from datetime import datetime

    import pandas as pd

    from yialpha.ledger import time_contract as tc

    _bind_perp_run()
    instant = [datetime.fromisoformat("2026-09-05T12:00:00+00:00")]
    bars = pd.DataFrame({"Close": [100.0, 999.0]}, index=pd.to_datetime(["2026-09-04", "2026-09-05"]))
    monkeypatch.setattr(tc, "_now_utc", lambda: instant[0])
    monkeypatch.setattr(tc, "binance_klines_frame", lambda *a, **k: bars)
    pt.begin_prediction_capture("market", symbol, SCOPE_CONTRACT)
    pt.submit_prediction.invoke({"predictions": _entries()})
    bars.loc[:, "Close"] = [500.0, 1000.0]
    instant[0] = datetime.fromisoformat("2026-09-06T18:00:00+00:00")
    # An identical retry must keep the accepted snapshot and avoid a re-fetch.
    pt.submit_prediction.invoke({"predictions": _entries()})
    pt.flush_predictions()
    rows = predictions_for_run(_RUN_ID)
    assert [row.horizon_days for row in rows] == [1, 5, 21]
    for row in rows:
        assert row.timing["reference_price"] == 100.0
        assert row.timing["prediction_formed_at"] == "2026-09-05T12:00:00+00:00"
        assert row.analysis_as_of == _TODAY


@pytest.mark.unit
def test_separate_horizon_submissions_retain_distinct_formation_times(monkeypatch):
    from datetime import datetime

    import pandas as pd

    from yialpha.ledger import time_contract as tc

    _bind_perp_run()
    instant = [datetime.fromisoformat("2026-09-05T12:00:00+00:00")]
    monkeypatch.setattr(tc, "_now_utc", lambda: instant[0])
    monkeypatch.setattr(tc, "binance_klines_frame", lambda *a, **k: pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2026-09-04"]),
    ))
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    pt.submit_prediction.invoke({"predictions": _entries()[:1]})
    instant[0] = datetime.fromisoformat("2026-09-05T13:00:00+00:00")
    pt.submit_prediction.invoke({"predictions": _entries()[1:]})
    pt.flush_predictions()
    rows = predictions_for_run(_RUN_ID)
    assert rows[0].timing["prediction_formed_at"] == "2026-09-05T12:00:00+00:00"
    assert all(row.timing["prediction_formed_at"] == "2026-09-05T13:00:00+00:00" for row in rows[1:])


@pytest.mark.unit
def test_tool_rejects_conflicting_replay_and_retains_original():
    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    pt.submit_prediction.invoke({"predictions": _entries()})
    conflict = {**_entries()[0], "direction": "down"}
    result = pt.submit_prediction.invoke({"predictions": [conflict]})
    assert "immutable once accepted" in result
    pt.flush_predictions()
    rows = predictions_for_run(_RUN_ID)
    assert rows[0].direction == "up"
    # Default hermetic seams have no reference: new rows must never silently
    # become legacy forecasts simply because acquisition failed.
    assert all(row.timing["version"] == "close_reference_v1" for row in rows)
    assert all(row.timing["reference_error"] for row in rows)


class _ObservedCaptureLock:
    """Expose contention so races are driven by events, never scheduling sleeps."""

    def __init__(self, contended):
        from threading import Lock

        self._lock = Lock()
        self._contended = contended

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self._contended.set()
            assert self._lock.acquire(timeout=5), "capture lock was never released"
        return self

    def __exit__(self, *exc):
        self._lock.release()


def _concurrent_timing(price=100.0):
    return {
        "version": "close_reference_v1",
        "prediction_formed_at": "2026-09-05T12:00:00+00:00",
        "reference_price": price,
        "reference_price_at": "2026-09-05T00:00:00+00:00",
        "reference_available_at": "2026-09-05T00:00:00+00:00",
        "reference_observed_at": "2026-09-05T12:00:00+00:00",
        "reference_source": "binance_perp:1d:last",
        "reference_error": None,
    }


@pytest.mark.unit
@pytest.mark.parametrize("conflicting", [False, True])
def test_parallel_tool_replays_keep_first_acceptance(monkeypatch, conflicting):
    from itertools import count
    from threading import Event

    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    overlap = Event()
    capture = pt._active_capture()
    capture._lock = _ObservedCaptureLock(overlap)
    fetches = []
    counter = count()

    def fetch(*args):
        number = next(counter)
        fetches.append(number)
        if number:
            # Before the fix, the competing call reaches the vendor instead
            # of contending on the capture lock. Release either interleaving.
            overlap.set()
        assert overlap.wait(timeout=5), "second ToolNode call did not overlap"
        return _concurrent_timing(100.0 + number)

    monkeypatch.setattr(pt, "capture_prediction_timing", fetch)
    first = _entries()[0]
    second = {**first, "direction": "down"} if conflicting else first
    response = _invoke_tool_node(_graph_tool_nodes()["market"], AIMessage(
        content="",
        tool_calls=[
            {"name": "submit_prediction", "args": {"predictions": [entry]},
             "id": f"parallel-{index}", "type": "tool_call"}
            for index, entry in enumerate((first, second))
        ],
    ))
    messages = [str(message.content) for message in response["messages"]]
    assert len(capture.entries) == 1
    assert fetches == [0]
    assert sum("immutable once accepted" in message for message in messages) == int(conflicting)
    assert sum("accepted horizons [1]" in message for message in messages) == 2 - int(conflicting)
    accepted_index = next(index for index, message in enumerate(messages) if "accepted horizons" in message)
    assert len(pt.flush_predictions()) == 1
    rows = predictions_for_run(_RUN_ID)
    assert len(rows) == 1
    assert rows[0].direction == (first, second)[accepted_index]["direction"]
    assert rows[0].timing == _concurrent_timing()
    assert pt.flush_predictions() == []


@pytest.mark.unit
def test_flush_waits_for_acceptance_before_selecting_pending_capture(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from threading import Event

    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    overlap = Event()
    capture = pt._active_capture()
    capture._lock = _ObservedCaptureLock(overlap)
    flushes = []

    def flush():
        try:
            return pt.flush_predictions()
        finally:
            # The old implementation skips the empty, still-fetching capture.
            overlap.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        def fetch(*args):
            flushes.append(executor.submit(copy_context().run, flush))
            assert overlap.wait(timeout=5), "flush neither waited nor completed"
            return _concurrent_timing()

        monkeypatch.setattr(pt, "capture_prediction_timing", fetch)
        pt.submit_prediction.invoke({"predictions": _entries()[:1]})
        assert len(flushes[0].result(timeout=5)) == 1
    assert len(predictions_for_run(_RUN_ID)) == 1
    assert pt.prediction_capture_pending() is False
    assert "already been settled" in pt.submit_prediction.invoke({"predictions": _entries()[1:]})


@pytest.mark.unit
def test_capture_write_failure_can_retry_without_recapturing(monkeypatch):
    _bind_perp_run()
    pt.begin_prediction_capture("market", "BTCUSDT", SCOPE_CONTRACT)
    fetches = []

    def fetch(*args):
        fetches.append(args)
        return _concurrent_timing()

    monkeypatch.setattr(pt, "capture_prediction_timing", fetch)
    pt.submit_prediction.invoke({"predictions": _entries()})
    original_submit = pt.submit_predictions

    def fail(*args, **kwargs):
        raise RuntimeError("temporary ledger failure")

    monkeypatch.setattr(pt, "submit_predictions", fail)
    assert pt.flush_predictions() == []
    assert pt.prediction_capture_pending() is True
    monkeypatch.setattr(pt, "submit_predictions", original_submit)
    pt.submit_prediction.invoke({"predictions": _entries()})
    assert len(pt.flush_predictions()) == 3
    assert len(fetches) == 1
    assert all(row.timing == _concurrent_timing() for row in predictions_for_run(_RUN_ID))
    assert pt.prediction_capture_pending() is False
