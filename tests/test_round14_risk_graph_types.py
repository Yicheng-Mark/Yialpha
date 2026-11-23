"""Round-14 type-fix regressions for ``risk/manager.py`` + ``graph/trading_graph.py``.

Five mypy union errors were fixed on the money path; each fix had to keep
runtime decisions byte-identical, so each behavior exposed by the union
analysis is locked here:

1. ``RiskManager.decide`` unpacks ``kelly_sizing(..., return_raw=True)``.
   The annotation unions the ``return_raw`` modes (``float | tuple``), but
   every ``return_raw=True`` path in kelly.py yields the ``(clipped, raw)``
   tuple — the fix narrowed the type at the call site with a runtime-noop
   ``cast``, so ``decision.kelly_raw`` / ``decision.kelly_preclip`` must stay
   EXACTLY what a direct ``kelly_sizing(return_raw=True)`` call computes.
2. ``_memoized_close_and_atr`` now annotates ``tuple[float, float | None]``:
   the underlying ``latest_atr_from_frame`` returns ``atr=None`` when the
   frame has fewer than 15 completed bars, and the memo (both the stock and
   the crypto_perp venue) must keep passing that None through — the risk
   overlay's documented degradation is "skip the ATR stop", never fabricate.
3. ``_log_state`` / ``_dump_perf`` pass ``self.ticker or ""`` into
   ``safe_ticker_component``: ``self.ticker`` is ``str | None`` until the
   run entry sets it, and a coalesced "" must be rejected by the SAME
   ValueError branch a None would hit (never a silent path escape).
4. ``_dump_perf`` guards ``self.perf_tracker is None`` with an early
   return: telemetry-off must be a total no-op (no file, no directory,
   no ``tracker.serialize()`` AttributeError), while a live tracker keeps
   dumping to the venue-suffixed stem.

Hermetic: loaders / telemetry seams monkeypatched, no network.
"""

from __future__ import annotations

from unittest import mock

import pytest

import yialpha.graph.trading_graph as tg
from yialpha.dataflows.utils import safe_ticker_component
from yialpha.risk.kelly import kelly_sizing
from yialpha.risk.manager import PortfolioState, RiskManager, _trade_stats


@pytest.fixture(autouse=True)
def _clear_memo():
    tg._memoized_close_and_atr.cache_clear()
    yield
    tg._memoized_close_and_atr.cache_clear()


# --------------------------------------------------------------------------- #
# 1. RiskManager.decide: kelly_sizing(return_raw=True) tuple contract
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("rating", ["Buy", "Overweight", "Hold", "Sell"])
@pytest.mark.parametrize("with_history", [False, True])
def test_decide_kelly_fields_match_direct_kelly_call(rating, with_history):
    """decide()'s kelly_raw/kelly_preclip equal a direct kelly_sizing call.

    Pins the cast at manager.py:210 to a runtime no-op: the unpacked values
    are the tuple kelly_sizing actually returns, for the confidence path
    (no ledger), the Bayesian win-rate path (ledger present), and the
    Sell band's early ``(0.0, 0.0)`` return.
    """
    history = (
        [
            {"ticker": "AAPL", "rating": "Buy", "return": 0.02},
            {"ticker": "MSFT", "rating": "Buy", "return": -0.01},
            {"ticker": "NVDA", "rating": "Buy", "return": 0.03},
        ]
        if with_history
        else []
    )
    state = PortfolioState(
        equity=100_000.0,
        cash=50_000.0,
        positions={"AAPL": 5_000.0},
        returns_history=[0.001, -0.002, 0.001],
        trade_history=history,
    )
    decision = RiskManager().decide("AAPL", rating, state, price=100.0, atr=1.5)

    # Mirror the exact kelly inputs decide() resolves from the same ledger
    # (via _trade_stats), then the same kelly_sizing call decide() makes.
    wins, losses, avg_win, avg_loss = _trade_stats(history)
    expected_raw, expected_preclip = kelly_sizing(
        rating,
        wins=wins if wins or losses else None,
        losses=losses if wins or losses else None,
        avg_win=avg_win,
        avg_loss=avg_loss,
        kelly_fraction_mult=0.25,
        return_raw=True,
    )
    assert isinstance(decision.kelly_raw, float)
    assert isinstance(decision.kelly_preclip, float)
    assert decision.kelly_raw == expected_raw
    assert decision.kelly_preclip == expected_preclip
    # Sell never takes a long position: both tuple slots are exactly 0.0.
    if rating == "Sell":
        assert (decision.kelly_raw, decision.kelly_preclip) == (0.0, 0.0)
        assert decision.target_weight == 0.0


@pytest.mark.unit
def test_decide_survives_empty_ledger_and_flat_trades():
    """Degenerate ledgers (empty / flat-zero returns) still produce a full
    RiskDecision — the unpack must not depend on ledger contents."""
    state = PortfolioState(
        equity=10_000.0,
        trade_history=[{"ticker": "X", "rating": "Buy", "return": 0.0}],
    )
    decision = RiskManager().decide("X", "Buy", state, price=50.0, atr=None)
    assert decision.kelly_preclip is not None
    assert 0.0 <= decision.kelly_raw <= 1.0
    # atr=None -> no ATR stop is fabricated.
    assert decision.stop_loss is None


# --------------------------------------------------------------------------- #
# 2. _memoized_close_and_atr: atr=None passes through on both venues
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_memoized_close_and_atr_passes_none_atr_stock_venue():
    with mock.patch(
        "yialpha.dataflows.stockstats_utils.load_ohlcv", return_value=object()
    ), mock.patch(
        "yialpha.risk.atr_stop.latest_atr_from_frame",
        return_value=(104.9, None),
    ):
        out = tg._memoized_close_and_atr("AAPL", "2026-06-01")
    assert out == (104.9, None)


@pytest.mark.unit
def test_memoized_close_and_atr_passes_none_atr_perp_venue():
    with mock.patch(
        "yialpha.dataflows.binance.binance_klines_frame", return_value=object()
    ) as klines, mock.patch(
        "yialpha.risk.atr_stop.latest_atr_from_frame",
        return_value=(65000.5, None),
    ):
        out = tg._memoized_close_and_atr("BTCUSDT", "2026-06-10", "crypto_perp")
    assert out == (65000.5, None)
    # The perp branch must keep loading the perp's own candles (120d lookback).
    (ticker, start, end), _ = klines.call_args
    assert ticker == "BTCUSDT"
    assert end == "2026-06-10"
    assert start == "2026-02-10"


@pytest.mark.unit
def test_latest_close_and_atr_consumer_handles_none_atr():
    """_latest_close_and_atr feeds decide(atr=None) without raising: the
    overlay's degradation is 'no stop', matching the widened signature."""
    with mock.patch(
        "yialpha.dataflows.stockstats_utils.load_ohlcv", return_value=object()
    ), mock.patch(
        "yialpha.risk.atr_stop.latest_atr_from_frame",
        return_value=(100.0, None),
    ):
        close, atr = tg.YiAlphaGraph._latest_close_and_atr(
            object(), "AAPL", "2026-06-01"
        )
    assert (close, atr) == (100.0, None)
    decision = RiskManager().decide(
        "AAPL", "Buy", PortfolioState(equity=10_000.0), price=close, atr=atr
    )
    assert decision.stop_loss is None
    assert decision.target_weight > 0.0


# --------------------------------------------------------------------------- #
# 3. safe_ticker_component: None and "" hit the same ValueError branch
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("bad", [None, ""])
def test_safe_ticker_component_rejects_none_and_empty_alike(bad):
    """``self.ticker or ""`` narrows the type without weakening the guard:
    None (pre-fix, unreachable at both writers) and its coalesced "" are
    rejected by the same first branch with the same exception class."""
    with pytest.raises(ValueError, match="non-empty string"):
        safe_ticker_component(bad)


@pytest.mark.unit
def test_log_state_writes_unchanged_with_valid_tracker_ticker(tmp_path):
    """The ``or ""`` coalescing is invisible on the live path: a valid
    ticker still lands the states log under its sanitized directory."""
    from yialpha.dataflows import quality

    g = tg.YiAlphaGraph.__new__(tg.YiAlphaGraph)
    g.config = {"results_dir": str(tmp_path)}
    g.ticker = "BRK.B"

    final_state = {
        "company_of_interest": "BRK.B",
        "trade_date": "2026-09-19",
        "market_report": "m", "sentiment_report": "s", "news_report": "n",
        "fundamentals_report": "f",
        "investment_debate_state": {
            "bull_history": "b", "bear_history": "r", "history": "h",
            "current_response": "c", "judge_decision": "j",
        },
        "trader_investment_plan": "t",
        "risk_debate_state": {
            "aggressive_history": "a", "conservative_history": "c",
            "neutral_history": "n", "history": "h", "judge_decision": "j",
        },
        "investment_plan": "i",
        "final_trade_decision": "**Rating**: Buy",
        # skip the fallback price loader (network seam)
        "price_at_decision": 65000.0,
        "price_at_decision_basis": "risk_overlay_close",
    }

    quality.ensure_run_context()
    try:
        g._log_state("2026-09-19", final_state, "stock")
    finally:
        quality.reset_quality()

    log = (
        tmp_path / "BRK.B" / "YiAlphaStrategy_logs"
        / "full_states_log_2026-09-19.json"
    )
    assert log.is_file()


# --------------------------------------------------------------------------- #
# 4. _dump_perf: telemetry-off is a total no-op; live tracker still dumps
# --------------------------------------------------------------------------- #


def _bare_graph(tmp_path, ticker="BTCUSDT"):
    g = tg.YiAlphaGraph.__new__(tg.YiAlphaGraph)
    g.config = {"results_dir": str(tmp_path)}
    g.ticker = ticker
    return g


@pytest.mark.unit
def test_dump_perf_noop_when_tracker_is_none(tmp_path, monkeypatch):
    """Pre-fix, a None tracker reached dump_perf_report and died on
    ``tracker.serialize()`` (AttributeError). The guard must make telemetry
    off a no-op: no dump call, no directory side effects."""
    dumped: list[object] = []
    monkeypatch.setattr(
        "yialpha.graph.perf_telemetry.dump_perf_report",
        lambda tracker, path: dumped.append((tracker, path)),
    )
    g = _bare_graph(tmp_path)
    g.perf_tracker = None

    g._dump_perf("2026-09-19", "stock")

    assert dumped == []
    assert list(tmp_path.rglob("node_perf_*")) == []
    assert not (tmp_path / "BTCUSDT").exists()


@pytest.mark.unit
def test_dump_perf_still_dumps_with_live_tracker(tmp_path, monkeypatch):
    """The guard must not touch the live path: a tracker present dumps to
    the same venue-suffixed stem as before."""
    dumped: list[object] = []
    monkeypatch.setattr(
        "yialpha.graph.perf_telemetry.dump_perf_report",
        lambda tracker, path: dumped.append((tracker, path)),
    )
    g = _bare_graph(tmp_path)
    tracker = object()
    g.perf_tracker = tracker

    g._dump_perf("2026-09-19", "crypto_perp")

    assert len(dumped) == 1
    got_tracker, got_path = dumped[0]
    assert got_tracker is tracker
    assert str(got_path).endswith("node_perf_2026-09-19_perp.json")
    assert "BTCUSDT" in str(got_path)
