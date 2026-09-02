"""Evidence wiring at the [EXTERNAL EVIDENCE] injection sites (V2.1 batch C).

Covers :func:`yialpha.ledger.run_context.record_evidence_block` directly
(flag / run-context gates, dedupe, PIT-guard fail-soft) and every analyst
injection site found by grepping the marker: the perp market bundle
(market analyst), the fundamentals bundle (fundamentals analyst), the
dual-angle company/contract digests (news analyst), and the four sentiment
blocks (news / StockTwits / Reddit / Binance Square). Sites are driven
through the smallest callable unit — the analyst NODE with a capture LLM
and mocked fetchers — asserting one row per block with the mapped source /
category / symbol / scope / replayability. conftest holds the
``prediction_ledger`` flag OFF and the ledger DB per-test tmp; opt-ins use
``set_config`` in the body.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yialpha.agents.analysts.fundamentals_analyst as fa
import yialpha.agents.analysts.market_analyst as ma
import yialpha.agents.analysts.news_analyst as na
import yialpha.agents.analysts.sentiment_analyst as sa
import yialpha.dataflows.fundamentals_bundle as fbundle
import yialpha.dataflows.perp_bundle as pb
from yialpha.agents.utils import prediction_tools as pt
from yialpha.dataflows import quality
from yialpha.dataflows.config import set_config
from yialpha.ledger.evidence import evidence_for_run, register_run
from yialpha.ledger.models import (
    REPLAYABILITY_LIVE_ONLY,
    REPLAYABILITY_PIT_REPLAYABLE,
    SCOPE_CONTRACT,
    SCOPE_UNDERLYING,
)
from yialpha.ledger.run_context import (
    record_evidence_block,
    reset_ledger_run_context,
    set_ledger_run_context,
)

_TODAY = date.today().isoformat()


@pytest.fixture(autouse=True)
def _clean_record_stage():
    reset_ledger_run_context()
    pt.reset_prediction_capture_for_test()
    quality.reset_quality()
    yield
    reset_ledger_run_context()
    pt.reset_prediction_capture_for_test()
    quality.reset_quality()


def _bind_run(
    run_id: str = "run-ev-1",
    ticker: str = "BTCUSDT",
    as_of: str = _TODAY,
) -> str:
    set_ledger_run_context(run_id, ticker, "crypto_perp", "pure_crypto_perp", as_of)
    register_run(run_id, ticker, "crypto_perp", "pure_crypto_perp", as_of)
    return run_id


def _sources(run_id: str) -> dict[str, list[str]]:
    """source -> [replayability, scope, symbol-or-None, category]."""
    mapped: dict[str, list[str]] = {}
    for row in evidence_for_run(run_id):
        mapped[row.source] = [row.replayability, row.scope, row.symbol or "", row.category]
    return mapped


class _CaptureLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.prompt = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        return AIMessage(content="FINAL REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def _perp_state(ticker: str = "BTCUSDT") -> dict[str, object]:
    return {
        "trade_date": _TODAY,
        "company_of_interest": ticker,
        "asset_type": "crypto_perp",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


# ---- record_evidence_block unit contract --------------------------------------


@pytest.mark.unit
def test_record_evidence_block_gates_and_dedupe():
    set_config({"prediction_ledger": True})
    run_id = _bind_run()
    record_evidence_block(
        "s", "news_data", "BTCUSDT", SCOPE_CONTRACT, "PAYLOAD",
        replayability=REPLAYABILITY_LIVE_ONLY,
    )
    rows = evidence_for_run(run_id)
    assert len(rows) == 1
    assert rows[0].source == "s"
    assert rows[0].replayability == REPLAYABILITY_LIVE_ONLY

    # Re-injection of the same payload dedupes to the one row.
    record_evidence_block(
        "s", "news_data", "BTCUSDT", SCOPE_CONTRACT, "PAYLOAD",
        replayability=REPLAYABILITY_LIVE_ONLY,
    )
    assert len(evidence_for_run(run_id)) == 1


@pytest.mark.unit
def test_record_evidence_block_noop_without_run_context():
    set_config({"prediction_ledger": True})
    record_evidence_block(
        "s", "news_data", "BTCUSDT", SCOPE_CONTRACT, "PAYLOAD",
        replayability=REPLAYABILITY_LIVE_ONLY,
    )  # no context bound: must not raise or write anything
    assert evidence_for_run("run-none") == []


@pytest.mark.unit
def test_record_evidence_block_flag_off_writes_nothing():
    # conftest holds prediction_ledger OFF.
    _bind_run()
    record_evidence_block(
        "s", "news_data", "BTCUSDT", SCOPE_CONTRACT, "PAYLOAD",
        replayability=REPLAYABILITY_LIVE_ONLY,
    )
    assert evidence_for_run("run-ev-1") == []


@pytest.mark.unit
def test_record_evidence_block_swallows_pit_violation():
    set_config({"prediction_ledger": True})
    run_id = _bind_run(as_of=(date.today() - timedelta(days=3)).isoformat())
    # available_at defaults to now, which postdates the historical as_of:
    # the ledger raises, the wrapper logs and swallows — never aborts.
    record_evidence_block(
        "s", "news_data", "BTCUSDT", SCOPE_CONTRACT, "PAYLOAD",
        replayability=REPLAYABILITY_PIT_REPLAYABLE,
    )
    assert evidence_for_run(run_id) == []


# ---- market analyst: perp market bundle ----------------------------------------


def _mock_perp_bundle(monkeypatch, block: str = "MARKET BLOCK") -> None:
    monkeypatch.setattr(
        pb, "fetch_perp_market_bundle", lambda s, d: {"symbol": s, "as_of": d}
    )
    monkeypatch.setattr(pb, "render_perp_bundle_block", lambda b: block)


@pytest.mark.unit
def test_market_bundle_records_live_only_contract_row(monkeypatch):
    set_config({"prediction_ledger": True, "perp_market_bundle": True})
    run_id = _bind_run()
    _mock_perp_bundle(monkeypatch)
    llm = _CaptureLLM()
    ma.create_market_analyst(llm)(_perp_state())
    mapped = _sources(run_id)
    assert mapped == {
        "perp_market_bundle": [
            REPLAYABILITY_LIVE_ONLY,
            SCOPE_CONTRACT,
            "BTCUSDT",
            "binance_perp",
        ]
    }


@pytest.mark.unit
def test_market_bundle_reinjection_dedupes(monkeypatch):
    set_config({"prediction_ledger": True, "perp_market_bundle": True})
    run_id = _bind_run()
    _mock_perp_bundle(monkeypatch)
    node = ma.create_market_analyst(_CaptureLLM())
    state = _perp_state()
    node(state)  # tool-loop re-entry: the node body (and injection) re-runs
    node(dict(state, messages=list(state["messages"])))
    assert len(evidence_for_run(run_id)) == 1


@pytest.mark.unit
def test_market_bundle_historical_is_pit_replayable(monkeypatch):
    set_config({"prediction_ledger": True, "perp_market_bundle": True})
    run_id = _bind_run()  # as_of = today: PIT guard admits the row
    _mock_perp_bundle(monkeypatch, "HISTORICAL PRICES ONLY")
    monkeypatch.setattr(ma, "is_historical_date", lambda d: True)
    ma.create_market_analyst(_CaptureLLM())(_perp_state())
    mapped = _sources(run_id)
    assert mapped["perp_market_bundle"][0] == REPLAYABILITY_PIT_REPLAYABLE


@pytest.mark.unit
def test_market_bundle_flag_off_records_nothing(monkeypatch):
    set_config({"perp_market_bundle": True})  # prediction_ledger stays OFF
    run_id = _bind_run()
    _mock_perp_bundle(monkeypatch)
    ma.create_market_analyst(_CaptureLLM())(_perp_state())
    assert evidence_for_run(run_id) == []


@pytest.mark.unit
def test_market_bundle_without_run_context_records_nothing(monkeypatch):
    from yialpha.ledger.sqlite import ledger_exists

    set_config({"prediction_ledger": True, "perp_market_bundle": True})
    _mock_perp_bundle(monkeypatch)
    llm = _CaptureLLM()
    result = ma.create_market_analyst(llm)(_perp_state())  # must not raise
    assert result["market_report"] == "FINAL REPORT"
    # Nothing was ever written: the ledger DB was not even created.
    assert ledger_exists() is False


# ---- fundamentals analyst: fundamentals bundle ---------------------------------


@pytest.mark.unit
def test_fundamentals_bundle_records_underlying_row(monkeypatch):
    set_config({"prediction_ledger": True, "fundamentals_bundle": True})
    run_id = _bind_run(ticker="MUUSDT")
    monkeypatch.setattr(
        fbundle, "fetch_fundamentals_bundle", lambda a, t, d: {"ok": True}
    )
    monkeypatch.setattr(fbundle, "render_fundamentals_bundle_block", lambda b: "FUNDIES")
    fa.create_fundamentals_analyst(_CaptureLLM())(_perp_state("MUUSDT"))
    mapped = _sources(run_id)
    assert mapped == {
        "fundamentals_bundle": [
            REPLAYABILITY_LIVE_ONLY,
            SCOPE_UNDERLYING,
            "MU",  # the stock-perp underlying, not the contract symbol
            "fundamental_data",
        ]
    }


# ---- news analyst: dual-angle company/contract digests -------------------------


@pytest.mark.unit
def test_news_dual_angle_records_two_rows(monkeypatch):
    set_config({"prediction_ledger": True})
    run_id = _bind_run(ticker="MUUSDT")
    monkeypatch.setattr(na, "_fetch_company_news", lambda u, s, e: "COMPANY BLOCK")
    monkeypatch.setattr(na, "_fetch_perp_contract_news", lambda t, d: "CONTRACT BLOCK")
    na.create_news_analyst(_CaptureLLM())(_perp_state("MUUSDT"))
    mapped = _sources(run_id)
    assert mapped == {
        "news_company_digest": [
            REPLAYABILITY_LIVE_ONLY,
            SCOPE_UNDERLYING,
            "MU",
            "news_data",
        ],
        "news_contract_digest": [
            REPLAYABILITY_LIVE_ONLY,
            SCOPE_CONTRACT,
            "MUUSDT",
            "news_data",
        ],
    }


# ---- sentiment analyst: per-block rows -----------------------------------------


class _FlatLLM:
    def with_structured_output(self, schema):  # noqa: ARG002
        raise AttributeError("no structured output in tests")

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    def invoke(self, messages, config=None, **kwargs):  # noqa: ARG002
        return AIMessage(content="ok", tool_calls=[])


@pytest.mark.unit
def test_sentiment_records_one_row_per_block(monkeypatch):
    set_config({"prediction_ledger": True})
    run_id = _bind_run()
    monkeypatch.setattr(
        sa,
        "_fetch_sentiment_sources",
        lambda *a, **k: ("NEWS BLOCK", "STOCKTWITS BLOCK", "REDDIT BLOCK", "SQUARE"),
    )
    monkeypatch.setattr(sa, "invoke_structured_or_freetext", lambda *a, **k: "REPORT")
    sa.create_sentiment_analyst(_FlatLLM())(_perp_state())
    mapped = _sources(run_id)
    assert mapped == {
        "sentiment_news": [REPLAYABILITY_LIVE_ONLY, SCOPE_CONTRACT, "BTCUSDT", "news_data"],
        "sentiment_stocktwits": [
            REPLAYABILITY_LIVE_ONLY, SCOPE_UNDERLYING, "BTCUSDT", "social",
        ],
        "sentiment_reddit": [REPLAYABILITY_LIVE_ONLY, SCOPE_CONTRACT, "BTCUSDT", "social"],
        "binance_square": [REPLAYABILITY_LIVE_ONLY, SCOPE_CONTRACT, "BTCUSDT", "social"],
    }


@pytest.mark.unit
def test_sentiment_square_absent_leaves_no_square_row(monkeypatch):
    set_config({"prediction_ledger": True})
    run_id = _bind_run()
    monkeypatch.setattr(
        sa,
        "_fetch_sentiment_sources",
        lambda *a, **k: ("NEWS BLOCK", "STOCKTWITS BLOCK", "REDDIT BLOCK", None),
    )
    monkeypatch.setattr(sa, "invoke_structured_or_freetext", lambda *a, **k: "REPORT")
    sa.create_sentiment_analyst(_FlatLLM())(_perp_state())
    mapped = _sources(run_id)
    assert "binance_square" not in mapped
    assert set(mapped) == {"sentiment_news", "sentiment_stocktwits", "sentiment_reddit"}
