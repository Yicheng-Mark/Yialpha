"""V2.3 — Positioning analyst: split gating, schema freeze, wiring, on-chain.

The positioning split's three guarantees under test:

1. The report is schema-incapable of stating a trade direction
   (``extra="forbid"`` — a direction key FAILS validation, it is not merely
   dropped).
2. Flag off / non-perp runs never see the analyst (node returns the honest
   skip note with zero LLM/vendor calls; the CLI filter keeps it out of the
   plan; the execution-plan builder pins its position after Market).
3. Flag on + perp: the node fetches the positioning half of the shared
   bundle, records evidence rows, captures POSITIONING-scope blind
   predictions (funding-sign semantics), and optionally appends the
   on-chain block (capability-absent degrades to a disclosure).

conftest holds positioning_split/onchain_evidence OFF; opt-ins set_config.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import yialpha.agents.analysts.positioning_analyst as pa
import yialpha.dataflows.onchain_flows as ocf
import yialpha.dataflows.perp_bundle as pb
from yialpha.agents.schemas import PositioningReport
from yialpha.agents.utils import prediction_tools as pt
from yialpha.cli.models import AnalystType, AssetType
from yialpha.cli.utils import filter_analysts_for_asset_type, resolve_run_analysts
from yialpha.dataflows import quality
from yialpha.dataflows.config import set_config
from yialpha.graph.analyst_execution import build_analyst_execution_plan
from yialpha.ledger.evidence import evidence_for_run, register_run
from yialpha.ledger.models import SCOPE_POSITIONING
from yialpha.ledger.predictions import predictions_for_run
from yialpha.ledger.run_context import (
    reset_ledger_run_context,
    set_ledger_run_context,
)

_TODAY = date.today().isoformat()
_RUN_ID = "run-positioning-1"
_BASE_ANALYSTS = [
    AnalystType.MARKET,
    AnalystType.SOCIAL,
    AnalystType.NEWS,
    AnalystType.FUNDAMENTALS,
]


class _FakeLLM:
    """Structured-output-less LLM whose bound round returns one response."""

    def __init__(self, response: AIMessage | None = None):
        self.response = response or AIMessage(content="x")
        self.calls = 0
        self.bound_tools: list[str] = []

    def with_structured_output(self, schema):  # noqa: ARG002
        raise AttributeError("no structured output in tests")

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        self.bound_tools = [tool.name for tool in tools]
        return self

    def invoke(self, messages, config=None, **kwargs):  # noqa: ARG002
        self.calls += 1
        return self.response


def _perp_state(ticker: str = "BTCUSDT") -> dict[str, object]:
    return {
        "trade_date": _TODAY,
        "company_of_interest": ticker,
        "asset_type": "crypto_perp",
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


def _minimal_bundle() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "as_of": _TODAY,
        "live_run": True,
        "funding": {
            "status": pb.STATUS_OK,
            "sum_7d": 0.0007,
            "annualized": 0.0365,
            "settlements": 21,
        },
        "premium_snapshot": {
            "status": pb.STATUS_OK,
            "mark_vs_index_bps": 45.0,
            "next_funding_rate": 0.0001,
            "next_funding_time_utc": "2026-08-19 00:00",
        },
        "open_interest": {
            "status": pb.STATUS_OK,
            "latest": 85_000.0,
            "chg_1d": 0.012,
            "chg_7d": -0.03,
            "percentile": 88.0,
        },
        "long_short": {
            "status": pb.STATUS_OK,
            "top_account": {"status": pb.STATUS_OK, "latest": 1.8},
            "top_position": {"status": pb.STATUS_OK, "latest": 1.2},
            "global_account": {"status": pb.STATUS_OK, "latest": 1.5},
            "cross_vantage_spread": 0.6,
        },
        "taker": {"status": pb.STATUS_OK, "latest": 1.21, "mean_7d": 1.05},
        "depth_bands": {
            "status": pb.STATUS_OK,
            "mid": 109.5,
            "spread_bps": 1.0,
            "bands": {},
            "slippage_bps": {"buy_10000": 2.1, "sell_10000": -2.3},
        },
        "adl": {"status": pb.STATUS_OK},
    }


def _chart_payload(days: int = 5) -> dict[str, object]:
    start = datetime.now().date() - timedelta(days=days)
    base_epoch = int(
        datetime(start.year, start.month, start.day).timestamp()
    )
    values = [
        {"x": base_epoch + i * 86_400, "y": 100.0 + i} for i in range(days)
    ]
    return {"values": values}


@pytest.fixture(autouse=True)
def _clean_record_stage():
    reset_ledger_run_context()
    pt.reset_prediction_capture_for_test()
    quality.reset_quality()
    yield
    reset_ledger_run_context()
    pt.reset_prediction_capture_for_test()
    quality.reset_quality()


def _bind_run() -> None:
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY
    )


# ---- schema freeze: no direction, ever ---------------------------------------


@pytest.mark.unit
def test_positioning_report_forbids_a_direction_key():
    good = PositioningReport(
        funding_bias="long_pays_expensive",
        crowding="crowded_long",
        liquidity_risk="normal",
        confidence="medium",
        narrative="ok",
    )
    assert good.funding_bias == "long_pays_expensive"
    with pytest.raises(ValidationError):
        # extra="forbid": a provider emitting a trade direction FAILS
        # validation — the freeze is structural, not a prompt request.
        PositioningReport.model_validate(
            {
                **good.model_dump(),
                "direction": "long",  # type: ignore[dict-item]
            }
        )


# ---- gating: flag off / non-perp never see the analyst ----------------------


@pytest.mark.unit
def test_node_skips_non_perp_with_zero_llm_calls(monkeypatch):
    set_config({"positioning_split": True})
    llm = _FakeLLM()
    monkeypatch.setattr(pa, "invoke_structured_or_freetext", lambda *a, **k: ("R", True))
    out = pa.create_positioning_analyst(llm)(
        {**_perp_state(), "asset_type": "stock"}
    )
    assert "skipped" in out["positioning_report"].lower()
    assert llm.calls == 0


@pytest.mark.unit
def test_node_skips_when_flag_off(monkeypatch):
    llm = _FakeLLM()
    monkeypatch.setattr(pa, "invoke_structured_or_freetext", lambda *a, **k: ("R", True))
    out = pa.create_positioning_analyst(llm)(_perp_state())
    assert "skipped" in out["positioning_report"].lower()
    assert llm.calls == 0


@pytest.mark.unit
def test_cli_filter_appends_positioning_only_for_perp_flag_on():
    # Flag off (conftest): pure-crypto perp drops FUNDAMENTALS (existing
    # rule) and gains nothing else — the pre-V2.3 behavior.
    flag_off = filter_analysts_for_asset_type(
        list(_BASE_ANALYSTS), AssetType.CRYPTO_PERP, "BTCUSDT"
    )
    assert flag_off == [
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
    ]
    set_config({"positioning_split": True})
    try:
        perp = filter_analysts_for_asset_type(
            list(_BASE_ANALYSTS), AssetType.CRYPTO_PERP, "BTCUSDT"
        )
        assert perp[-1] == AnalystType.POSITIONING
        assert AnalystType.FUNDAMENTALS not in perp
        # Non-perp asset types never gain it.
        stock = filter_analysts_for_asset_type(
            list(_BASE_ANALYSTS), AssetType.STOCK, "AAPL"
        )
        assert AnalystType.POSITIONING not in stock
        spot = filter_analysts_for_asset_type(
            list(_BASE_ANALYSTS), AssetType.CRYPTO, "BTC-USD"
        )
        assert AnalystType.POSITIONING not in spot
    finally:
        set_config({"positioning_split": False})


@pytest.mark.unit
def test_run_baseline_build_routes_through_batch_filter(monkeypatch):
    """Round-11 C1: run_baseline's graph builder routes its analyst set through
    the single-source batch filter, so a flag-on crypto_perp backtest includes
    the Positioning analyst (it used to hand the bare 4-analyst default)."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "run_baseline.py"
    spec = importlib.util.spec_from_file_location("run_baseline_under_test", script)
    assert spec is not None and spec.loader is not None
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)

    seen: dict = {}

    class _FakeGraph:
        def __init__(self, selected_analysts=None, debug=False, config=None):
            seen["selected"] = tuple(selected_analysts or ())

    monkeypatch.setattr(rb, "YiAlphaGraph", _FakeGraph)
    monkeypatch.setattr(
        "yialpha.batch.runner.batch_selected_analysts",
        lambda at, tk: ("market", "positioning", "social", "news"),
    )
    rb._build_graph(asset_type="crypto_perp", tickers=("BTCUSDT",))
    assert seen["selected"] == ("market", "positioning", "social", "news")
    # No tickers (legacy callers): bare default tuple, no filter call.
    monkeypatch.setattr(
        "yialpha.batch.runner.batch_selected_analysts",
        lambda *a, **k: pytest.fail("filter must not run without tickers"),
    )
    rb._build_graph()
    assert seen["selected"] == ("market", "social", "news", "fundamentals")


@pytest.mark.unit
def test_execution_plan_pins_positioning_right_after_market():
    plan = build_analyst_execution_plan(
        ["market", "news", "positioning", "social"]
    )
    keys = [spec.key for spec in plan.specs]
    assert keys == ["market", "positioning", "news", "social"]
    # Without positioning: byte-identical to the pre-V2.3 plan.
    assert [spec.key for spec in build_analyst_execution_plan(
        ["market", "news", "social"]
    ).specs] == ["market", "news", "social"]


# ---- round-11: the interactive funnel re-applies the filter ------------------
#
# The checkbox's option set is the fixed four-key ANALYST_ORDER, so the
# filter's flag-appended POSITIONING never reaches the user selection;
# run_analysis must go through resolve_run_analysts (normalize, then
# re-apply the single-source filter) or the switch runs dead end to end.


@pytest.mark.unit
def test_resolve_run_analysts_reappends_positioning_for_flag_on_perp():
    set_config({"positioning_split": True})
    try:
        keys = resolve_run_analysts(
            list(_BASE_ANALYSTS), AssetType.CRYPTO_PERP, "BTCUSDT"
        )
        # Set assertion, not order: the execution-plan builder pins the run
        # order (test_execution_plan_pins_positioning_right_after_market).
        assert set(keys) == {"market", "social", "news", "positioning"}
        assert keys[-1] == "positioning"  # appended at the tail
    finally:
        set_config({"positioning_split": False})


@pytest.mark.unit
def test_resolve_run_analysts_flag_off_stays_pre_v23():
    # conftest holds positioning_split OFF: the resolved keys are exactly the
    # historical normalization (pure-crypto perp drops FUNDAMENTALS and gains
    # nothing; stock keeps the full four in order) — byte-equivalent chain.
    perp = resolve_run_analysts(list(_BASE_ANALYSTS), AssetType.CRYPTO_PERP, "BTCUSDT")
    assert perp == ["market", "social", "news"]
    stock = resolve_run_analysts(list(_BASE_ANALYSTS), AssetType.STOCK, "AAPL")
    assert stock == [a.value for a in _BASE_ANALYSTS]


@pytest.mark.unit
def test_resolve_run_analysts_never_readds_unselected(monkeypatch):
    import yialpha.graph.routing as routing

    # Tokenized-stock perp so FUNDAMENTALS stays applicable — the only shape
    # where a user-checked FUNDAMENTALS meets the flag-on append.
    monkeypatch.setattr(routing, "stock_perp_underlying", lambda t: "MU")
    set_config({"positioning_split": True})
    try:
        user_picks = [AnalystType.MARKET, AnalystType.FUNDAMENTALS]
        keys = resolve_run_analysts(user_picks, AssetType.CRYPTO_PERP, "MUUSDT")
        # User selection + the flag-appended POSITIONING, and nothing that
        # the user left unchecked comes back.
        assert set(keys) == {"market", "fundamentals", "positioning"}
    finally:
        set_config({"positioning_split": False})


# ---- flag-on node integration ------------------------------------------------


@pytest.mark.unit
def test_node_runs_bundle_evidence_and_prediction_capture(monkeypatch):
    _bind_run()
    set_config({"positioning_split": True, "prediction_ledger": True})
    try:
        monkeypatch.setattr(
            pb, "fetch_perp_market_bundle", lambda t, d: _minimal_bundle()
        )
        monkeypatch.setattr(pa, "invoke_structured_or_freetext", lambda *a, **k: ("R", True))
        tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_prediction",
                    "args": {
                        "predictions": [
                            {
                                "horizon_days": horizon,
                                "direction": "up",
                                "prob_up": 0.7,
                                "confidence": 0.5,
                            }
                            for horizon in (1, 5, 21)
                        ]
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        llm = _FakeLLM(tool_call)
        out = pa.create_positioning_analyst(llm)(_perp_state())

        assert out["positioning_report"] == "R"
        assert llm.bound_tools == ["submit_prediction"]
        # POSITIONING-scope blind predictions under the run.
        rows = predictions_for_run(_RUN_ID)
        assert [row.horizon_days for row in rows] == [1, 5, 21]
        assert all(row.analyst == "positioning" for row in rows)
        assert all(row.prediction_scope == SCOPE_POSITIONING for row in rows)
        # The injected positioning block became one evidence row.
        sources = {row.source for row in evidence_for_run(_RUN_ID)}
        assert "positioning_bundle" in sources
        assert "onchain_flows" not in sources  # flag off
    finally:
        set_config({"positioning_split": False, "prediction_ledger": False})


@pytest.mark.unit
def test_node_appends_onchain_block_when_enabled(monkeypatch):
    _bind_run()
    set_config(
        {"positioning_split": True, "onchain_evidence": True, "prediction_ledger": True}
    )
    try:
        monkeypatch.setattr(
            pb, "fetch_perp_market_bundle", lambda t, d: _minimal_bundle()
        )
        monkeypatch.setattr(pa, "invoke_structured_or_freetext", lambda *a, **k: ("R", True))
        monkeypatch.setattr(ocf, "fetch_chart_json", lambda chart, **k: _chart_payload())
        llm = _FakeLLM()
        pa.create_positioning_analyst(llm)(_perp_state())
        sources = {row.source for row in evidence_for_run(_RUN_ID)}
        assert {"positioning_bundle", "onchain_flows"} <= sources
        onchain_rows = [r for r in evidence_for_run(_RUN_ID) if r.source == "onchain_flows"]
        assert onchain_rows[0].replayability == "PIT_REPLAYABLE"
    finally:
        set_config(
            {
                "positioning_split": False,
                "onchain_evidence": False,
                "prediction_ledger": False,
            }
        )


@pytest.mark.unit
def test_node_degrades_when_bundle_fetch_fails(monkeypatch):
    set_config({"positioning_split": True})
    try:
        def boom(t, d):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(pb, "fetch_perp_market_bundle", boom)
        # Round-12: the old pin asserted the LLM still ran ("fail-soft:
        # report still runs") — but the structured schema's Literals
        # (funding_bias / crowding / liquidity_risk) have no "unavailable"
        # option, so that call forced fabricated labels rendered as a
        # data-derived header. The bundle-fetch failure path now returns an
        # honest skip note with NO LLM call (same fail-honest rule as the
        # fundamentals grounding note).
        llm = _FakeLLM()
        out = pa.create_positioning_analyst(llm)(_perp_state())
        assert "skipped" in out["positioning_report"]
        assert "unavailable" in out["positioning_report"]
        assert llm.calls == 0  # no billed call on zero evidence
    finally:
        set_config({"positioning_split": False})


# ---- positioning half of the bundle -----------------------------------------


@pytest.mark.unit
def test_render_positioning_block_covers_positioning_sections():
    block = pb.render_positioning_block(_minimal_bundle())
    for marker in ("Funding", "Open interest", "long/short", "Taker"):
        assert marker.lower() in block.lower()
    # The market analyst's own block is UNCHANGED by the split (flag-on only
    # adds the analyst; duplication beats data loss — documented decision).
    market_block = pb.render_perp_bundle_block(_minimal_bundle())
    assert "funding" in market_block.lower()


# ---- on-chain vendor ---------------------------------------------------------


@pytest.mark.unit
def test_onchain_fetch_filters_future_points_and_pit_tags(monkeypatch):
    monkeypatch.setattr(ocf, "fetch_chart_json", lambda chart, **k: _chart_payload())
    today = date.today().isoformat()
    flows = ocf.fetch_onchain_flows(today)
    assert flows is not None
    assert flows.replayability == "PIT_REPLAYABLE"
    assert flows.source == "blockchain.info:charts"
    # All chart points are at/before the as-of day (PIT filter).
    assert flows.available_at[:10] <= today
    assert "blockchain.info" in ocf.render_onchain_block(flows)


@pytest.mark.unit
def test_onchain_capability_absent_renders_disclosure(monkeypatch):
    monkeypatch.setattr(ocf, "fetch_chart_json", lambda chart, **k: None)
    flows = ocf.fetch_onchain_flows(date.today().isoformat())
    assert flows is None
    block = ocf.render_onchain_block(None)
    assert "capability absent" in block.lower()
