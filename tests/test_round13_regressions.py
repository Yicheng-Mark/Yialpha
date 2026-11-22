"""Round-13 regression pins — backtest + misc shards.

Episode accounting for forced exits (stop-out / liquidation): the engine's
decision-driven episode bookkeeping only ran on decision fill bars, so a
mid-bar forced exit left ``opening_trade`` unstamped — a later re-entry
overwrote it and the forced loss vanished from win_rate / num_trades /
profit_factor and from the Kelly ledger fed via ``ctx['realized_returns']``.
Pinned here: a stopped-out leg that re-enters later is TWO counted trades,
the stopped leg is a stamped loss, and the risk layer sees it.

Also: event-study estimation windows must not intersect a prior event's
window; with_structured_output must forward include_raw; the sector cap must
not double-count the resized ticker's own book; DecisionCache.put must
swallow unserializable payloads; the production ticket payload must carry
the bps decomposition the outcome scorer requires.

NOTE: each fix shard owns its own test file this round — concurrent
full-file appends to one shared file truncated it once already. The binance
shard lives in tests/test_round13_binance_regressions.py; the graph/cli
shard in tests/test_round13_graph_cli_regressions.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tests.test_backtest_engine import FakeGraph
from yialpha.backtest.engine import run_backtest


def _rising(ticker, start, end):
    idx = pd.bdate_range(start, end)
    values = [100.0 * (1 + 0.001 * i) for i in range(len(idx))]
    return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)


def _funding(rate: float):
    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        return pd.Series([rate] * len(idx), index=idx.strftime("%Y-%m-%d"),
                         dtype=float)
    return provider


def _extremes(lows, highs, pad_from_start):
    def provider(ticker, start, end):
        idx = pd.bdate_range(start, end)
        n = len(idx)
        # Neutral filler 99.5/200 sits ABOVE the 97 stop (no accidental
        # triggers on unscripted bars) while still being a plausible wick.
        low_vals = [99.5] * pad_from_start + lows + [99.5] * (
            n - pad_from_start - len(lows))
        high_vals = [200.0] * pad_from_start + highs + [200.0] * (
            n - pad_from_start - len(highs))
        return (
            pd.Series(low_vals[:n], index=idx.strftime("%Y-%m-%d"), dtype=float),
            pd.Series(high_vals[:n], index=idx.strftime("%Y-%m-%d"), dtype=float),
        )
    return provider


_KW = {
    "initial_capital": 100_000.0, "holding_days": 5,
    "price_provider": _rising, "compute_index_alpha": False,
    "periods_per_year": 365, "cost_bps": 0.0,
    "funding_provider": _funding(0.0),
}


@pytest.mark.unit
def test_stop_exit_episode_counted_then_reentry_is_second_trade():
    """Stop-out on 01-03, re-entry signal on 01-10 -> two counted episodes.

    Pre-fix: the stopped-out leg's ``position_return`` was never stamped and
    the re-entry overwrote ``opening_trade`` — num_trades=1 and win_rate
    blind to the forced loss.
    """
    # Capture the realized-returns list the risk layer would consume, at
    # each sizing call, to pin the Kelly-ledger half of the defect too.
    snapshots: list[list[float]] = []

    def weight_fn(rating: str, date: str, ctx: dict) -> float | None:
        snapshots.append(list(ctx.get("realized_returns") or []))
        ctx["risk_decision"] = SimpleNamespace(action="Buy", stop_loss=97.0)
        return 1.0

    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy", "2024-01-10": "Buy"}),
        "BTCUSDT", ["2024-01-01", "2024-01-10"],
        asset_type="crypto_perp",
        weight_fn=weight_fn,
        extremes_provider=_extremes([94.0], [101.0], 2),
        **_KW,
    )

    # One stop trigger, on the wick bar.
    assert res.config_summary["perp_stop_trigger_count"] == 1

    # TWO counted trades: the stopped-out leg is a stamped loss, the
    # re-entry leg rides the rising tape to a win.
    assert res.metrics.num_trades == 2
    assert res.metrics.win_rate == pytest.approx(0.5)
    stamped = [t for t in res.trades if t.position_return is not None]
    assert len(stamped) == 2
    assert stamped[0].position_return < 0
    assert stamped[1].position_return > 0

    # The risk layer's ledger saw the forced exit BEFORE the second sizing
    # call (pre-fix the list stayed empty until the very end).
    assert len(snapshots) == 2
    assert len(snapshots[1]) == 1
    assert snapshots[1][0] < 0


@pytest.mark.unit
def test_liquidation_exit_episode_counted_then_reentry():
    """Liquidation on the crash bar, re-entry after -> two counted episodes.

    Pre-fix the liquidated leg's loss vanished the same way the stopped-out
    leg's did: the re-entry overwrote ``opening_trade`` before the episode
    was ever stamped.
    """
    # 100 flat x3 bars, one -60% bar (wick to 20 -> liquidation at the 10x
    # trigger), then flat at 40 for the re-entry leg.
    def flat_then_crash(ticker, start, end):
        idx = pd.bdate_range(start, end)
        values = [100.0] * 3 + [40.0] + [40.0] * (len(idx) - 4)
        return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)

    def crash_lows(ticker, start, end):
        idx = pd.bdate_range(start, end)
        lows = [99.0] * 3 + [20.0] + [39.0] * (len(idx) - 4)
        return (
            pd.Series(lows[: len(idx)], index=idx.strftime("%Y-%m-%d"),
                      dtype=float),
            pd.Series([101.0] * len(idx), index=idx.strftime("%Y-%m-%d"),
                      dtype=float),
        )

    res = run_backtest(
        FakeGraph({"2024-01-01": "Buy", "2024-01-10": "Buy"}),
        "BTCUSDT", ["2024-01-01", "2024-01-10"],
        asset_type="crypto_perp", leverage=10.0,
        funding_provider=_funding(0.0),
        price_provider=flat_then_crash, compute_index_alpha=False,
        periods_per_year=365, initial_capital=100_000.0,
        brackets_provider=lambda t: [(0.0, 0.004)],
        extremes_provider=crash_lows, cost_bps=0.0,
    )
    assert res.config_summary["perp_liquidations"], "10x through -60% must liquidate"
    # The liquidated leg is stamped at its forced exit (a ~-96%-of-margin
    # loss), and the re-entry leg is a second counted trade (flat tape ->
    # exactly 0 net return, finite so it counts).
    assert res.metrics.num_trades == 2
    stamped = [t for t in res.trades if t.position_return is not None]
    assert len(stamped) == 2
    assert stamped[0].position_return < -0.5  # isolated-margin wipeout
    assert stamped[1].position_return == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Event study: estimation window must not eat a prior event's window (P2)
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_event_study_skips_events_whose_estimation_window_touches_prior_event():
    """Round-13: dense events contaminate the market model.

    Event A at return-index 200 with window (0, 10) covers returns
    [200, 210]. Event B 15 bars later at 215 with estimation_size=60 /
    pre_event_gap=5 has estimation window [150, 210] — the intersection
    [200, 210] feeds A's abnormal drift into B's beta. Pre-fix B was fit
    anyway; the guard must drop it while keeping a well-separated event C.
    """
    import numpy as np

    from yialpha.backtest.event_study import event_study

    rng = np.random.default_rng(7)
    bench_ret = rng.normal(0.0004, 0.01, size=700)
    idx = pd.bdate_range("2020-01-02", periods=700).strftime("%Y-%m-%d")

    def _prices(rets):
        return pd.Series(
            100.0 * np.cumprod(1.0 + np.asarray(rets)), dtype=float,
        ).set_axis(idx)

    asset_prices = _prices(bench_ret)
    bench_prices = _prices(bench_ret)

    pos_a, pos_b, pos_c = 200, 215, 400
    dates = [idx[pos_a], idx[pos_b], idx[pos_c]]
    result = event_study(
        asset_prices, bench_prices, dates,
        estimation_size=60, pre_event_gap=5, event_window=(0, 10),
    )
    kept = [e.event_date for e in result.events]
    assert idx[pos_a] in kept          # first event: nothing prior to touch
    assert idx[pos_b] not in kept      # estimation window overlaps A's window
    assert idx[pos_c] in kept          # 185 bars clear of B: cleanly separated
    assert result.n_events == 2


# --------------------------------------------------------------------------- #
# Round-13 misc shard: llm include_raw forwarding, sector re-size math,
# DecisionCache serialization swallow
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_with_structured_output_forwards_include_raw(monkeypatch):
    """Round-13: the override accepted include_raw then silently dropped it —
    a caller asking langchain's diagnostic (raw, parsed, parse_error) triple
    got the bare parsed value instead."""
    from unittest.mock import patch

    captured = {}

    def _fake_super(schema, *, method=None, include_raw=False, **kwargs):
        captured["include_raw"] = include_raw
        return "BOUND"

    with patch(
        "langchain_openai.ChatOpenAI.with_structured_output",
        new=lambda self, schema, **kw: _fake_super(schema, **kw),
    ):
        # Real constructor (pydantic model — __new__ bypass is not viable);
        # the placeholder key conftest injects makes offline construction OK.
        from yialpha.llm_clients.openai_client import OpenAIClient

        llm = OpenAIClient("gpt-4.1", provider="openai").get_llm()
        out = llm.with_structured_output(dict, include_raw=True)
    assert out == "BOUND"
    assert captured["include_raw"] is True


@pytest.mark.unit
def test_sector_cap_does_not_double_count_own_existing_exposure():
    """Round-13: re-sizing a held ticker used sector+target (total) on top of
    a sector figure that already included the ticker — a legitimate
    keep-at-0.20 re-size was trimmed to the 0.05 headroom."""
    from yialpha.risk.manager import PortfolioState, RiskManager

    rm = RiskManager()  # max_single_sector=0.30
    # Baseline: same manager, fresh book -> the untrimmed Kelly weight.
    fresh = PortfolioState(cash=0, equity=100_000)
    baseline_w = rm.decide("AAPL", "Buy", fresh, price=100.0, atr=2.0).target_weight
    assert baseline_w > 0.0

    held = PortfolioState(
        cash=0, equity=100_000,
        positions={"AAPL": 20_000.0}, sectors={"tech": 25_000.0},
    )
    d = rm.decide("AAPL", "Buy", held, price=100.0, atr=2.0, sector="tech")
    # True post-trade sector weight is 0.25 <= 0.30 -> NO trim: the target
    # equals the same untrimmed weight (pre-fix it collapsed to 0.05).
    assert d.target_weight == pytest.approx(baseline_w)
    assert "trimmed" not in (d.exposure_reason or "")

    # A NEW ticker in the same sector still hits the cap (0.25 + w > 0.30).
    d_new = rm.decide("MSFT", "Buy", held, price=100.0, atr=2.0, sector="tech")
    assert d_new.target_weight <= 0.30 - 0.25 + 1e-9


@pytest.mark.unit
def test_decision_cache_put_swallows_unserializable_payload(tmp_path):
    """Round-13: json.dump TypeError escaped put() and killed the backtest —
    violating the module's own "a bad cache can never break a backtest"."""
    from yialpha.backtest.cache import CachedDecision, DecisionCache

    cache = DecisionCache(tmp_path, enabled=True)
    decision = CachedDecision(
        ticker="AAPL", date="2026-01-02", rating=object(),  # type: ignore[arg-type]
        final_decision="**Rating**: Buy", run_tag="t", asset_type="stock",
    )
    cache.put(decision)  # must not raise
    assert not any(p.suffix == ".json" for p in tmp_path.rglob("*") if p.is_file())
    # No .tmp litter either.
    assert not any(p.suffix == ".tmp" for p in tmp_path.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# Ticket cost decomposition rides the production payload (P1)
# --------------------------------------------------------------------------- #

@pytest.mark.unit
def test_production_ticket_payload_satisfies_outcome_cost_decomposition(
    monkeypatch, tmp_path,
):
    """Round-13: build_candidate_ticket's payload (model_dump) must carry the
    bps decomposition the outcome scorer requires.

    Pre-fix, every production ticket carried only the scalar ``estimated_cost``
    — the scorer (require_decomposition=True) refuses that, so ANY timed
    CONTRACT prediction was doomed to terminal incomplete with
    legs_missing=["ticket_cost"]. The existing tests never caught it because
    they hand-wrote payloads with a ``cost_detail`` dict. This pin builds the
    ticket through the REAL production constructor instead."""
    from tests.test_outcome_compute import _NOW, _RUN, _Seams, _seed, _write_ticket
    from yialpha.ledger.outcome_compute import compute_outcomes
    from yialpha.ledger.outcomes import outcomes_for_prediction
    from yialpha.tickets import build_candidate_ticket

    # Ledger isolation comes from the autouse _runtime_ledger_isolated
    # fixture (per-test tmp DB); no manual env juggling here.
    seams = _Seams()
    seams.install(monkeypatch)

    prediction_id = _seed()  # seeds under _RUN
    ticket = build_candidate_ticket(
        symbol="BTCUSDT",
        asset_type="crypto_perp",
        rating="Buy",
        target_weight=0.2,
        entry_price=100.0,
        stop_loss=97.0,
        reference_price=100.0,
        atr=1.5,
        trade_date="2026-08-25",
        horizon_days=5,
    )
    payload = ticket.model_dump()
    # The decomposition the scorer reads must be present and non-degenerate.
    assert payload["cost_detail"]["entry_fee_bps"] > 0
    _write_ticket(_RUN, payload)

    compute_outcomes(_NOW)
    (record,) = outcomes_for_prediction(prediction_id)
    assert record.status == "complete", record.legs_missing
    assert "ticket_cost" not in (record.legs_missing or ())


# --------------------------------------------------------------------------- #
# Round-13 wave 4: baostock quarter gate + prediction fallback prompt parity
# --------------------------------------------------------------------------- #

class _BSResultSet:
    """Minimal BaoStock result-set double: empty success by default."""

    error_code = "0"
    error_msg = ""
    fields = ("pubDate", "statDate")

    def __init__(self, rows: list[list[str]] | None = None):
        self._rows = rows or []
        self._i = 0

    def next(self):
        if self._i < len(self._rows):
            self._i += 1
            return True
        return False

    def get_row_data(self):
        return self._rows[self._i - 1]


class _FakeBS:
    """Session double whose query_profit_data fails on scripted quarters."""

    def __init__(self, fail_labels: set[str]):
        self._fail = fail_labels

    def query_profit_data(self, code, year, quarter):
        label = f"{year}Q{quarter}"
        if label in self._fail:
            raise RuntimeError("network glitch")
        return _BSResultSet()


@pytest.mark.unit
def test_baostock_expected_quarter_deadline_derivation():
    from datetime import date

    from yialpha.dataflows.baostock_vendor import _expected_latest_disclosed_quarter as q

    # Statutory A-share deadlines: Q1 by Apr 30, semi-annual by Aug 31,
    # Q3 by Oct 31, annual (prior Q4) by next Apr 30 — boundary-inclusive.
    assert q(date(2026, 4, 29)) == ("2025", "3")
    assert q(date(2026, 4, 30)) == ("2026", "1")
    assert q(date(2026, 8, 30)) == ("2026", "1")
    assert q(date(2026, 8, 31)) == ("2026", "2")
    assert q(date(2026, 10, 30)) == ("2026", "2")
    assert q(date(2026, 10, 31)) == ("2026", "3")
    assert q(date(2026, 12, 31)) == ("2026", "3")
    assert q(date(2026, 1, 1)) == ("2025", "3")


@pytest.mark.unit
def test_baostock_gate_fails_closed_on_expected_quarter_not_first_queried():
    """Anchor 2026-09-15 -> expected latest-disclosed quarter is 2026Q2 (the
    first QUERIED is 2026Q4, usually not yet published). A fetch error on the
    EXPECTED quarter must fail closed even though it is not the first queried
    (pre-fix only the first-queried quarter could trip the gate, so the real
    primary row's failure degraded to an in-band note)."""
    from datetime import date

    import pytest as _pytest

    from yialpha.dataflows.baostock_vendor import _query_statement
    from yialpha.dataflows.errors import NoMarketDataError

    with _pytest.raises(NoMarketDataError):
        _query_statement(
            _FakeBS({"2026Q2"}), "sh.600519", "query_profit_data",
            anchor=date(2026, 9, 15),
        )


@pytest.mark.unit
def test_baostock_gate_tolerates_undue_newer_quarter_error():
    """The same anchor's anchor-year Q4 is NOT yet due for disclosure: its
    transient error must degrade to an in-band failed note, not kill the
    whole statement (pre-fix the first-queried-quarter gate failed the tool
    here despite full history being available)."""
    from datetime import date

    from yialpha.dataflows.baostock_vendor import _query_statement

    fetch = _query_statement(
        _FakeBS({"2026Q4"}), "sh.600519", "query_profit_data",
        anchor=date(2026, 9, 15),
    )
    assert "2026Q4" in fetch.failed


@pytest.mark.unit
def test_market_fallback_round_advertises_only_the_prediction_tool(monkeypatch):
    """Round-13: the fallback round re-partials tool_names. Pre-fix the base
    prompt advertised the FULL tool list while only submit_prediction was
    bound — the model kept calling advertised-but-unbound tools and dispatch
    silently dropped them, losing the blind-prediction row."""
    import re

    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable

    import yialpha.agents.analysts.market_analyst as market_analyst
    import yialpha.dataflows.perp_bundle as pb
    from tests.test_prediction_tool import (
        _bind_perp_run,
        _entries,
        _perp_state,
    )
    from yialpha.dataflows import quality
    from yialpha.dataflows.config import set_config

    class _CaptureLLM(Runnable):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.fallback_system: str | None = None

        def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="FINAL REPORT", tool_calls=[])
            self.fallback_system = str(inp[0].content)
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "submit_prediction",
                    "args": {"predictions": _entries()},
                    "id": "call-fb",
                    "type": "tool_call",
                }],
            )

        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    set_config({"prediction_ledger": True, "perp_market_bundle": True})
    quality.ensure_run_context()
    _bind_perp_run()
    monkeypatch.setattr(
        pb, "fetch_perp_market_bundle", lambda s, d: {"symbol": s, "as_of": d}
    )
    monkeypatch.setattr(pb, "render_perp_bundle_block", lambda b: "MARKET BLOCK")

    llm = _CaptureLLM()
    market_analyst.create_market_analyst(llm)(_perp_state())

    assert llm.fallback_system is not None, "fallback round must have run"
    m = re.search(
        r"You have access to the following tools: ([^.]+)\.",
        llm.fallback_system,
    )
    assert m, "tools sentence missing from the fallback prompt"
    assert m.group(1).strip() == "submit_prediction"
