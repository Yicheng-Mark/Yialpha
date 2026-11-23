"""Round-14 coverage batch for ``graph/trading_graph.py`` + ``risk/manager.py``.

The two files carry the suite's largest absolute uncovered-line count (86% /
93%). This file pins the *behaviorally meaningful* remainder — every test
targets a documented contract, not a line number:

* config wiring: the constructor's opt-in telemetry / response-cache /
  callback plumbing, and the per-provider thinking/effort kwargs;
* fail-soft seams: mark-price close, derivatives stress, trailing funding,
  states-log write, perf dump, portfolio control, ledger record stage —
  each must degrade to its documented disclosure, never break a run;
* propagation entry: perp warm-up, checkpoint-resume branch, stream
  telemetry's unknown-analyst degrade;
* presence-gated log keys (run_id / checkpoint_resumed / portfolio_control /
  regime_id) — key PRESENCE is the downstream "recorded" signal;
* RiskManager: the ``invested`` property, ledger-stat skipping, the two
  sector-cap block branches, and the backtest weight-fn's realized-return
  fold-in mirroring a direct ``kelly_sizing`` call.

Hermetic throughout: no LLM, no network. Vendor seams are monkeypatched;
the two class-level hermetic stubs installed by ``tests/conftest.py``
(``_latest_mark_close`` / ``_render_stress_line``) are shadowed per-test
with the ORIGINAL functions captured at collection time (the shadowing
pattern that fixture's own docstring sanctions).

Deliberately NOT covered here (documented skips):
* full ``__init__`` LLM-client construction matrix beyond the three flags —
  ``test_graph_construction_smoke.py`` owns that seam;
* the portfolio-control SHADOW/ENFORCED success pipelines — owned by
  ``test_v24_portfolio_control.py`` / ``test_v24_acceptance.py``;
* the fair-value bridge HAPPY path (fx present, re-pricing succeeds) —
  owned by ``test_round6_regressions.py``; only the failure/degenerate
  branches are pinned here.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import yialpha.graph.trading_graph as tg
from yialpha.dataflows import quality
from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.risk.kelly import kelly_sizing
from yialpha.risk.manager import (
    PortfolioState,
    RiskManager,
    _trade_stats,
    build_backtest_weight_fn,
)

pytestmark = pytest.mark.unit

# Captured at COLLECTION time, i.e. before any function-scoped conftest
# fixture replaced them: the pristine implementations of the two methods
# tests/conftest.py stubs class-wide for hermeticity. Tests that need the
# real bodies re-install these via monkeypatch, shadowing the stub exactly
# the way the conftest fixture's docstring describes.
_ORIG_LATEST_MARK_CLOSE = YiAlphaGraph._latest_mark_close
_ORIG_RENDER_STRESS_LINE = YiAlphaGraph._render_stress_line

_TG_LOGGER = "yialpha.graph.trading_graph"

# Pristine reference for fixture cleanup: tests may monkeypatch the module
# function itself, and fixture teardown runs BEFORE monkeypatch unwinds
# (conftest autouse fixtures force monkeypatch to set up early), so the
# cleanup must not depend on the patched attribute.
from yialpha.ledger.run_context import (  # noqa: E402
    reset_ledger_run_context as _real_reset_ledger_run_context,
)


@pytest.fixture(autouse=True)
def _clear_module_memos():
    """Both module-level LRU memos start and end each test cold.

    The teardown is defensive: conftest autouse fixtures force ``monkeypatch``
    to set up (hence tear down) around this fixture, so a test that swapped a
    memo for a stub may still have it installed when this runs.
    """
    def _clear_all() -> None:
        for name in ("_memoized_close_and_atr", "_memoized_mark_close"):
            cache_clear = getattr(getattr(tg, name, None), "cache_clear", None)
            if cache_clear is not None:
                cache_clear()

    _clear_all()
    yield
    _clear_all()


@pytest.fixture(autouse=True)
def _reset_ledger_run_context():
    _real_reset_ledger_run_context()
    yield
    _real_reset_ledger_run_context()


def _bare_graph(config: dict[str, Any]) -> YiAlphaGraph:
    """Graph shell with ``__init__`` skipped (established suite pattern)."""
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = config
    return g


# --------------------------------------------------------------------------- #
# __init__ opt-in wiring: perf telemetry, LLM response cache, callbacks
# --------------------------------------------------------------------------- #


def test_init_wires_perf_telemetry_response_cache_and_callbacks(
    mock_llm_client, tmp_path
):
    """The three opt-in constructor branches compose: telemetry creates the
    shared tracker + token callback (which also proves non-empty callbacks
    reach the LLM kwargs), and llm_cache installs the disk cache globally."""
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.globals import get_llm_cache, set_llm_cache

    from yialpha.default_config import DEFAULT_CONFIG
    from yialpha.graph.perf_telemetry import NodePerfTokenCallback, NodePerfTracker
    from yialpha.llm_clients.response_cache import DiskLLMCache

    class _UserCallback(BaseCallbackHandler):
        """A real callback handler: the kwarg is pydantic-validated by the LLM."""

    config = dict(DEFAULT_CONFIG)
    config["data_cache_dir"] = str(tmp_path / "cache")
    config["results_dir"] = str(tmp_path / "results")
    config["node_perf_telemetry"] = True
    config["llm_cache"] = True
    user_callback = _UserCallback()

    try:
        graph = YiAlphaGraph(config=config, callbacks=[user_callback])
        # Telemetry on: tracker built, token callback appended AFTER the
        # user's own callbacks (their order must be preserved).
        assert isinstance(graph.perf_tracker, NodePerfTracker)
        assert graph.callbacks[0] is user_callback
        assert isinstance(graph.callbacks[1], NodePerfTokenCallback)
        # Response cache on: installed as the process-wide langchain cache
        # at the config's cache dir BEFORE any LLM is created.
        installed = get_llm_cache()
        assert isinstance(installed, DiskLLMCache)
    finally:
        set_llm_cache(None)


# --------------------------------------------------------------------------- #
# _get_provider_kwargs: per-provider thinking/effort + retry plumbing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("config", "key", "value"),
    [
        ({"llm_provider": "google", "google_thinking_level": "high"},
         "thinking_level", "high"),
        ({"llm_provider": "Google", "google_thinking_level": "low"},
         "thinking_level", "low"),
        ({"llm_provider": "openai", "openai_reasoning_effort": "low"},
         "reasoning_effort", "low"),
        ({"llm_provider": "anthropic", "anthropic_effort": "medium"},
         "effort", "medium"),
    ],
)
def test_provider_kwargs_forward_thinking_and_effort(config, key, value):
    kwargs = _bare_graph(config)._get_provider_kwargs()
    assert kwargs[key] == value


def test_provider_kwargs_temperature_coerces_env_strings():
    kwargs = _bare_graph(
        {"llm_provider": "openai", "temperature": "0.2", "llm_max_retries": 0}
    )._get_provider_kwargs()
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_retries"] == 0


def test_provider_kwargs_skip_unset_optional_tiers():
    kwargs = _bare_graph({"llm_provider": "openai"})._get_provider_kwargs()
    for absent in ("thinking_level", "reasoning_effort", "effort", "temperature"):
        assert absent not in kwargs
    assert kwargs["max_retries"] == 2


# --------------------------------------------------------------------------- #
# _build_risk_manager: fail-open on build failure, visibly
# --------------------------------------------------------------------------- #


def test_build_risk_manager_degrades_visibly_on_failure(tmp_path, monkeypatch, caplog):
    g = _bare_graph({"risk_enabled": True, "results_dir": str(tmp_path)})
    g._risk_overlay_degraded = False

    def boom(cls, config):  # pragma: no cover - raise site
        raise ValueError("bad risk config")

    monkeypatch.setattr(RiskManager, "from_config", classmethod(boom))
    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        assert g._build_risk_manager() is None
    assert g._risk_overlay_degraded is True
    assert any("running without the risk overlay" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Mark-price legs: module memo + fail-soft wrapper
# --------------------------------------------------------------------------- #


def test_memoized_mark_close_loads_perp_mark_candles(monkeypatch):
    from yialpha.dataflows import binance as binance_module

    calls: list[tuple[Any, ...]] = []

    def fake_frame(symbol, start, end, *args, **kwargs):
        calls.append((symbol, start, end, args, kwargs))
        return pd.DataFrame({"Close": [64_000.0, 65_042.5]})

    monkeypatch.setattr(binance_module, "binance_klines_frame", fake_frame)
    assert tg._memoized_mark_close("BTCUSDT", "2026-06-10") == 65_042.5
    # Mark leg specifics: 10-day lookback, daily perp MARK candles (the
    # caller passes interval/venue/price_type positionally).
    (symbol, start, end, args, kwargs) = calls[0]
    assert symbol == "BTCUSDT"
    assert start == "2026-05-31"
    assert end == "2026-06-10"
    assert args == ("1d", "binance_perp", "mark")
    assert kwargs == {}


def test_latest_mark_close_success_and_core_sentinel_on_failure(
    monkeypatch, caplog
):
    """Success passes the memo value through; a failure returns None AND
    records a KIND_CORE_ERROR sentinel (mark is a core price leg on a perp
    run — the loss must not pass as a clean report)."""
    monkeypatch.setattr(YiAlphaGraph, "_latest_mark_close", _ORIG_LATEST_MARK_CLOSE)
    g = _bare_graph({})
    quality.ensure_run_context()
    try:
        monkeypatch.setattr(
            tg, "_memoized_mark_close", lambda ticker, d: 65_100.0
        )
        assert g._latest_mark_close("BTCUSDT", "2026-06-10") == 65_100.0

        def boom(ticker, d):
            raise RuntimeError("mark klines down")

        monkeypatch.setattr(tg, "_memoized_mark_close", boom)
        with caplog.at_level(logging.INFO, logger=_TG_LOGGER):
            assert g._latest_mark_close("BTCUSDT", "2026-06-10") is None
        events = quality.snapshot_quality()
        assert any(
            e["kind"] == quality.KIND_CORE_ERROR
            and e["method"] == "get_binance_klines"
            and "mark-price close unavailable" in e["detail"]
            for e in events
        )
    finally:
        quality.reset_quality()


# --------------------------------------------------------------------------- #
# _apply_risk_overlay: no portfolio snapshot injected (documented default)
# --------------------------------------------------------------------------- #


def test_apply_risk_overlay_without_portfolio_state(tmp_path, monkeypatch):
    from yialpha.graph.overlay_fields import OVERLAY_MARKER

    g = _bare_graph({"results_dir": str(tmp_path)})
    g._risk_overlay_degraded = False
    g.risk_manager = RiskManager()
    g.signal_processor = SimpleNamespace(process_signal=lambda s: "Hold")
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (100.0, 2.0)
    )
    final_state = {
        "final_trade_decision": "**Rating**: Buy\n\nThesis.",
        "pm_rating": "Buy",
        "pm_decision_fields": {"price_target": 120.0, "confidence": 0.7},
    }

    out = g._apply_risk_overlay("AAPL", "2026-01-05", final_state, None)

    # equity defaults to 0.0 (single-instrument runs): the overlay still
    # applies, prices the decision, and builds the Candidate ticket.
    assert OVERLAY_MARKER in out["final_trade_decision"]
    assert out["price_at_decision"] == 100.0
    assert out["price_at_decision_basis"] == "risk_overlay_close"
    assert "- **Stop Loss**: 96" in out["final_trade_decision"]
    ticket = out["execution_ticket"]
    assert ticket["symbol"] == "AAPL"
    assert ticket["stop_price"] == 96.0


# --------------------------------------------------------------------------- #
# _link_ticket_to_ledger: fail-soft record stage
# --------------------------------------------------------------------------- #


def _ledger_ticket() -> SimpleNamespace:
    return SimpleNamespace(
        ticket_id="T1",
        decision_id="D1",
        ticket_version="v1",
        run_id=None,
        evidence_ids=None,
        prediction_ids=None,
        regime_id=None,
        model_dump=lambda: {"ticket_id": "T1"},
    )


def test_link_ticket_records_run_ids_then_survives_mirror_failure(
    monkeypatch, caplog
):
    from yialpha.ledger import evidence, predictions, tickets_mirror
    from yialpha.ledger.run_context import set_ledger_run_context

    g = _bare_graph({})
    monkeypatch.setattr(
        evidence, "evidence_ids_for_run", lambda run_id: ["ev-1"]
    )
    monkeypatch.setattr(predictions, "predictions_for_run", lambda run_id: [])
    mirrored: list[dict[str, Any]] = []

    def failing_attach(**kwargs):
        mirrored.append(kwargs)
        raise ValueError("mirror broken")

    monkeypatch.setattr(tickets_mirror, "attach_ticket", failing_attach)
    set_ledger_run_context(
        "run-x", "BTCUSDT", "crypto_perp", analysis_as_of="2026-01-05"
    )
    ticket = _ledger_ticket()

    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        block = g._link_ticket_to_ledger(
            ticket, {}, "BTCUSDT", "2026-01-05", "crypto_perp", 100.0
        )
    # Linkage keys land on the ticket BEFORE the mirror write; a mirror
    # failure costs the DB row only, never the run.
    assert block == ""
    assert ticket.run_id == "run-x"
    assert ticket.evidence_ids == ["ev-1"]
    assert mirrored and mirrored[0]["ticket_id"] == "T1"
    assert any(
        "ticket ledger linkage failed" in r.message for r in caplog.records
    )


def test_link_ticket_bridge_wiring_failure_never_breaks_run(
    monkeypatch, caplog
):
    from yialpha.graph import routing

    g = _bare_graph({"stock_perp_fair_value": True})

    def boom(asset_type, ticker):
        raise RuntimeError("routing down")

    monkeypatch.setattr(routing, "instrument_class", boom)
    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        block = g._link_ticket_to_ledger(
            _ledger_ticket(), {}, "MUUSDT", "2026-01-05", "crypto_perp", 100.0
        )
    assert block == ""
    assert any(
        "fair-value bridge wiring failed" in r.message for r in caplog.records
    )


def test_link_ticket_non_numeric_underlying_target_degrades_to_no_bridge(
    monkeypatch,
):
    """A schema-passthrough string target is dropped (never float-converted
    to garbage); with no usable underlying target the bridge is skipped
    entirely instead of converting nothing into something."""
    from yialpha.graph import routing

    g = _bare_graph({"stock_perp_fair_value": True})
    monkeypatch.setattr(routing, "instrument_class", lambda at, t: "stock_perp")
    final_state = {
        "pm_decision_fields": {"underlying_price_target": "not-a-number"}
    }
    ticket = _ledger_ticket()
    block = g._link_ticket_to_ledger(
        ticket, final_state, "MUUSDT", "2026-01-05", "crypto_perp", 100.0
    )
    assert block == ""
    # No attribute the bridge owns was written: nothing was converted, so
    # nothing half-converted leaks onto the ticket.
    assert not hasattr(ticket, "underlying_target")
    assert not hasattr(ticket, "contract_target")


def test_link_ticket_repricing_failure_keeps_converted_target(
    monkeypatch, caplog
):
    """USD-quoted fallback converts the target, then re-prices the verdict
    through the same tradeability gate; a re-pricing failure degrades to a
    warning but KEEPS the contract-currency target already written."""
    from yialpha.graph import routing
    from yialpha.perp import quote_fx as quote_fx_module
    from yialpha.perp.quote_fx import QuoteFxResult
    from yialpha.risk import cost_model

    g = _bare_graph({"stock_perp_fair_value": True})
    monkeypatch.setattr(routing, "instrument_class", lambda at, t: "stock_perp")
    fx = QuoteFxResult(
        rate=1.0,
        available_at="2026-01-05",
        fetched_at="2026-01-05T00:00:00+00:00",
    )
    monkeypatch.setattr(quote_fx_module, "usdt_usd_as_of", lambda as_of: fx)
    monkeypatch.setattr(g, "_latest_mark_close", lambda t, d: None)

    def boom(*args, **kwargs):
        raise RuntimeError("cost model down")

    monkeypatch.setattr(cost_model, "estimate_round_trip_cost", boom)
    ticket = _ledger_ticket()
    ticket.target_price = 150.0
    ticket.tradeability = "TRADEABLE"
    ticket.gross_edge = ticket.estimated_cost = ticket.net_edge = None
    final_state = {
        "pm_decision_fields": {
            "price_target_currency": "USD",
            "price_target": 150.0,
        }
    }

    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        block = g._link_ticket_to_ledger(
            ticket, final_state, "MUUSDT", "2026-01-05", "crypto_perp", 100.0
        )
    # Target re-denominated BEFORE the re-pricing try; verdict untouched by
    # the swallowed exception; the disclosure block still renders.
    assert ticket.underlying_target == 150.0
    assert ticket.target_price == 150.0
    assert ticket.price_target_basis == "last"
    assert ticket.tradeability == "TRADEABLE"
    assert ticket.quote_fx["rate"] == 1.0
    assert block  # bridge disclosure rendered
    assert any(
        "bridge verdict re-pricing failed" in r.message for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# _sync_candidate_ticket: config-default atr multiplier on a bare graph
# --------------------------------------------------------------------------- #


def test_sync_candidate_ticket_uses_config_atr_mult_without_manager(tmp_path):
    """A bare graph (no risk_manager attribute) must arm the synced stop from
    the config's atr_stop_mult — one multiplier for stop AND leverage advice."""
    from yialpha.risk.signed_math import atr_stop
    from yialpha.tickets import TicketSide, build_candidate_ticket

    g = _bare_graph({"atr_stop_mult": 3.0, "results_dir": str(tmp_path)})
    ticket = build_candidate_ticket(
        symbol="BTCUSDT",
        asset_type="crypto_perp",
        rating="Buy",
        target_weight=0.1,
        entry_price=100.0,
        stop_loss=96.0,
        reference_price=100.0,
        quality_events=[],
        core_successes=set(),
        trade_date="2026-01-05",
    )
    decision = SimpleNamespace(entry_price=100.0)

    g._sync_candidate_ticket(
        ticket,
        decision,
        "crypto_perp",
        "SHORT",
        0.05,
        reference_price=100.0,
        atr=2.0,
        funding_annualized=None,
    )
    assert ticket.side is TicketSide.SHORT
    assert ticket.proposed_size == 0.05
    # SHORT mirrors the stop ABOVE entry, using the config's 3.0 multiplier
    # (not the hardcoded 2.0 the legacy long-only stop used).
    assert ticket.stop_price == atr_stop("SHORT", 100.0, 2.0, 3.0)
    assert ticket.stop_price == 106.0
    # The perp leverage advisory follows the synced direction and size.
    assert ticket.leverage is not None and ticket.leverage > 0.0


# --------------------------------------------------------------------------- #
# _apply_portfolio_control: the pipeline must never break a run
# --------------------------------------------------------------------------- #


def test_portfolio_control_failure_returns_empty_and_warns(
    monkeypatch, caplog
):
    from yialpha.ledger import portfolio

    g = _bare_graph({"portfolio_control_mode": "shadow"})
    ticket = SimpleNamespace(tradeability="TRADEABLE", tradeability_reason="")

    def boom():
        raise RuntimeError("ledger read failed")

    monkeypatch.setattr(portfolio, "load_positions_input", boom)
    final_state = {"pm_rating": "Buy", "pm_decision_fields": {"rating": "Buy"}}
    decision = SimpleNamespace(target_weight=0.1)

    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        block = g._apply_portfolio_control(
            ticket,
            decision,
            final_state,
            "BTCUSDT",
            "2026-01-05",
            "crypto_perp",
            equity=10_000.0,
            reference_price=100.0,
        )
    assert block == ""
    assert any(
        "portfolio control failed" in r.message for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# _render_perp_ticket: render-time verification guards
# --------------------------------------------------------------------------- #


def _perp_decision() -> SimpleNamespace:
    return SimpleNamespace(target_weight=0.1, entry_price=100.0, stop_loss=95.0)


def _patch_perp_numbers(monkeypatch, result):
    from yialpha.risk import perp_ticket as pt

    monkeypatch.setattr(pt, "perp_ticket_numbers", lambda *a, **kw: result)
    monkeypatch.setattr(pt, "advisory_leverage_cap", lambda symbol, **kw: None)


def test_render_perp_ticket_degrades_to_empty_when_numbers_unavailable(
    monkeypatch,
):
    """perp_ticket_numbers -> None (bad inputs) must cost the bullet only."""
    _patch_perp_numbers(monkeypatch, None)
    assert (
        YiAlphaGraph._render_perp_ticket(
            "BTCUSDT", "2026-01-05", "Buy", _perp_decision(), 100.0, 2.0
        )
        == ""
    )


def test_render_perp_ticket_bracket_cap_and_stop_first_note(monkeypatch):
    detail = {
        "L_liq": 2.0,
        "L_vol": 3.0,
        "L_conv": 4.0,
        "L_hard": 20.0,
        "L_bracket": 10.0,
        "liq_mmr": 0.005,
        "liq_fee_rate": 0.0005,
        "liq_mmr_basis": "default bracket ladder @ assumed 25,000 USDT",
    }
    _patch_perp_numbers(monkeypatch, (5.0, detail, 90.0, 95.0))
    out = YiAlphaGraph._render_perp_ticket(
        "BTCUSDT", "2026-01-05", "Buy", _perp_decision(), 100.0, 2.0
    )
    # The exchange bracket cap rides the leverage line...
    assert "bracket 10x" in out
    # ...and the stop-first ordering is VERIFIED at render time.
    assert "the stop at 95 fires first by design" in out
    assert "bracket 10x" in out
    assert "Stop Trigger Basis" in out


def test_render_perp_ticket_warns_when_stop_not_beyond_liquidation(monkeypatch):
    """A stop sitting INSIDE the estimated liquidation distance must render
    the explicit NOT-guaranteed warning, never a reassuring default."""
    detail = {"L_liq": 2.0, "L_vol": 3.0, "L_conv": 4.0, "L_hard": 20.0}
    _patch_perp_numbers(monkeypatch, (5.0, detail, 90.0, 85.0))
    out = YiAlphaGraph._render_perp_ticket(
        "BTCUSDT", "2026-01-05", "Buy", _perp_decision(), 100.0, 2.0
    )
    assert "does NOT sit beyond" in out
    assert "NOT guaranteed" in out
    assert "fires first" not in out


# --------------------------------------------------------------------------- #
# Derivatives stress / trailing funding: live-only windows, honest n/a
# --------------------------------------------------------------------------- #


def test_render_stress_line_invalid_and_historical_dates():
    """An unparseable date falls into the historical bucket (no fetch)."""
    out = _bare_graph({})._render_stress_line("BTCUSDT", "2026-13-45")
    assert "n/a (historical run" in out


def test_render_stress_line_live_window_compute_and_fetch_failure(
    monkeypatch, caplog
):
    # staticmethod: re-install wrapped so instance calls do not bind self.
    monkeypatch.setattr(
        YiAlphaGraph, "_render_stress_line",
        staticmethod(_ORIG_RENDER_STRESS_LINE),
    )
    from yialpha.dataflows import binance as binance_module

    live = date.today().isoformat()
    all_missing = {
        "funding": None,
        "global_lsr": None,
        "basis": None,
        "open_interest": None,
        "taker_ratio": None,
    }
    monkeypatch.setattr(
        binance_module, "derivatives_stress_series",
        lambda symbol, end_date, **kw: all_missing,
    )
    g = _bare_graph({})
    out = g._render_stress_line("BTCUSDT", live)
    # Zero usable components -> the honest neutral score, never fabricated.
    assert "Derivatives Stress" in out
    assert "crowding 50/100" in out

    def boom(symbol, end_date, **kw):
        raise RuntimeError("vendor down")

    monkeypatch.setattr(binance_module, "derivatives_stress_series", boom)
    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        out = g._render_stress_line("BTCUSDT", live)
    assert out == "- **Derivatives Stress**: n/a (fetch failed)\n"
    assert any(
        "derivatives stress unavailable" in r.message for r in caplog.records
    )


def test_trailing_funding_total_window_policy(monkeypatch, caplog):
    from yialpha.backtest import engine as backtest_engine

    live = date.today().isoformat()
    # Unparseable / historical dates: no fetch, None, by policy.
    assert YiAlphaGraph._trailing_funding_total("BTCUSDT", "not-a-date") is None
    assert YiAlphaGraph._trailing_funding_total("BTCUSDT", "2020-01-01") is None

    # Live window, provider healthy: the END-INCLUSIVE 7d settlement sum.
    monkeypatch.setattr(
        backtest_engine,
        "_binance_funding_provider",
        lambda ticker, start, end: pd.Series([0.0001, 0.0002]),
    )
    total = YiAlphaGraph._trailing_funding_total("BTCUSDT", live)
    assert total == pytest.approx(0.0003)

    # Live window, empty series: explicit None (nothing to sum).
    monkeypatch.setattr(
        backtest_engine,
        "_binance_funding_provider",
        lambda ticker, start, end: pd.Series(dtype="float64"),
    )
    assert YiAlphaGraph._trailing_funding_total("BTCUSDT", live) is None

    def boom(ticker, start, end):
        raise RuntimeError("funding vendor down")

    monkeypatch.setattr(backtest_engine, "_binance_funding_provider", boom)
    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        assert YiAlphaGraph._trailing_funding_total("BTCUSDT", live) is None
    assert any(
        "funding lookup unavailable" in r.message for r in caplog.records
    )


def test_format_funding_note_distinguishes_none_causes():
    fmt = YiAlphaGraph._format_funding_note
    assert fmt(None, "not-a-date") == "n/a (unreadable date)"
    assert fmt(None, "2020-01-01") == "n/a (historical run; not fetched)"
    assert fmt(None, date.today().isoformat()) == "n/a (fetch failed)"
    assert fmt(0.001, "2026-01-05") == "+0.100% net over 7d (longs pay)"
    assert fmt(-0.001, "2026-01-05") == "-0.100% net over 7d (longs receive)"
    assert fmt(0.0, "2026-01-05") == "+0.000% net over 7d (flat)"


# --------------------------------------------------------------------------- #
# propagate(): perp warm-up + checkpoint resume
# --------------------------------------------------------------------------- #


def test_propagate_perp_run_warms_equity_listing_once(monkeypatch):
    g = _bare_graph({"memory_enabled": False})
    g._checkpointer_ctx = None
    calls: list[Any] = []
    monkeypatch.setattr(tg, "require_supported_market", lambda t: None)
    monkeypatch.setattr(tg, "warm_equity_perp_bases", lambda: calls.append("warm"))
    g._resolve_pending_entries = lambda *a, **kw: calls.append("resolve")

    def fake_run_graph(cn, td, asset_type="stock", portfolio_state=None,
                       _resuming=False):
        calls.append("run")
        return {"final_trade_decision": "**Rating**: Hold"}

    g._run_graph = fake_run_graph

    g.propagate("BTCUSDT", "2026-01-05 06:07:08", asset_type="crypto_perp")
    # Warm happens before any pending-entry resolution / graph run, and the
    # dual-form date was normalized to its date part at the entry.
    assert calls == ["warm", "resolve", "run"]
    assert g.ticker == "BTCUSDT"

    # Stock runs never pay the warm-up fetch.
    calls.clear()
    g.propagate("AAPL", "2026-01-05", asset_type="stock")
    assert calls == ["resolve", "run"]


def test_propagate_checkpoint_resume_branch(tmp_path, monkeypatch, caplog):
    g = _bare_graph(
        {"checkpoint_enabled": True, "data_cache_dir": str(tmp_path),
         "memory_enabled": False, "max_debate_rounds": 1,
         "max_risk_discuss_rounds": 1}
    )
    g.selected_analysts = ("market",)  # _run_signature reads it
    g._checkpointer_ctx = None
    compiles: list[dict[str, Any]] = []

    @contextmanager
    def fake_checkpointer(data_dir, ticker):
        compiles.append({"enter": ticker})
        yield object()

    monkeypatch.setattr(tg, "get_checkpointer", fake_checkpointer)
    monkeypatch.setattr(tg, "checkpoint_step", lambda *a, **kw: 7)
    g.workflow = SimpleNamespace(compile=lambda **kw: compiles.append(kw) or object())
    resumed: list[bool] = []
    g._resolve_pending_entries = lambda *a, **kw: None
    g._run_graph = lambda cn, td, asset_type="stock", portfolio_state=None, \
        _resuming=False: resumed.append(_resuming) or {"final_trade_decision": "x"}

    with caplog.at_level(logging.INFO, logger=_TG_LOGGER):
        g.propagate("AAPL", "2026-01-02", asset_type="stock")

    assert resumed == [True]
    assert any("Resuming from step 7" in r.message for r in caplog.records)
    # The saver context was entered AND exited: compile ran with the saver,
    # then the finally block restored the uncheckpointed graph.
    assert {"enter": "AAPL"} in compiles
    assert any(entry == {} for entry in compiles[1:])
    assert g._checkpointer_ctx is None


# --------------------------------------------------------------------------- #
# _invoke_or_stream: unknown analyst shape degrades to plain streaming
# --------------------------------------------------------------------------- #


def test_stream_telemetry_unknown_analyst_still_streams(monkeypatch):
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {"stream_telemetry": True}
    g.selected_analysts = ("bogus_analyst",)

    seen_trackers: list[Any] = []

    class _Graph:
        def stream(self, state, **args):
            seen_trackers.append(g._active_wall_time_tracker)
            yield {"final": True}

    g.graph = _Graph()
    out = g._invoke_or_stream({}, {})
    # ValueError from the plan builder is swallowed: no tracker, but the run
    # still streams (observation-only telemetry must never block a run), and
    # the published slot is cleared in finally.
    assert out == {"final": True}
    assert seen_trackers == [None]
    assert g._active_wall_time_tracker is None


# --------------------------------------------------------------------------- #
# _bind_run_record_stage: fail-soft record/context stages
# --------------------------------------------------------------------------- #


def test_bind_record_stage_falls_back_to_date_anchor_on_resolution_failure(
    monkeypatch, caplog
):
    from yialpha.dataflows import utils as dataflows_utils
    from yialpha.ledger import evidence
    from yialpha.ledger.run_context import current_ledger_run_context

    g = _bare_graph({"prediction_ledger": True})
    monkeypatch.setattr(
        dataflows_utils, "is_historical_date",
        lambda d: (_ for _ in ()).throw(RuntimeError("clock broken")),
    )
    registered: list[dict[str, Any]] = []
    monkeypatch.setattr(
        evidence, "register_run",
        lambda **kw: registered.append(kw),
    )

    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        ledger_run_id, _regime = g._bind_run_record_stage(
            "AAPL", "2026-01-05", "stock"
        )
    assert ledger_run_id is not None
    # Anchored to the date-only analysis date — never to a fabricated now.
    assert registered[0]["analysis_as_of"] == "2026-01-05"
    assert current_ledger_run_context().analysis_as_of == "2026-01-05"
    assert any(
        "precise run as-of resolution failed" in r.message
        for r in caplog.records
    )


def test_bind_record_stage_register_failure_resets_context(
    monkeypatch, caplog
):
    from yialpha.ledger import evidence, run_context as ledger_run_context
    from yialpha.ledger.run_context import current_ledger_run_context

    g = _bare_graph({"prediction_ledger": True})
    resets: list[int] = []
    real_reset = ledger_run_context.reset_ledger_run_context

    def counting_reset():
        resets.append(1)
        real_reset()

    monkeypatch.setattr(
        ledger_run_context, "reset_ledger_run_context", counting_reset
    )

    def boom(**kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(evidence, "register_run", boom)

    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        assert g._bind_run_record_stage("AAPL", "2026-01-05", "stock") == (
            None, None,
        )
    # The stale context must not survive a failed register (orphan-ticket
    # hazard): recording is honestly off and the context was reset.
    assert any(
        "ledger run-context binding failed" in r.message for r in caplog.records
    )
    assert resets
    assert current_ledger_run_context() is None


def test_bind_record_stage_survives_reset_failure_after_register_failure(
    monkeypatch,
):
    """Belt-and-braces: even when the context RESET itself raises after a
    failed register_run, the record stage still fails soft — the run must
    never break from its own bookkeeping cleanup."""
    from yialpha.ledger import evidence, run_context as ledger_run_context

    g = _bare_graph({"prediction_ledger": True})
    reset_calls: list[int] = []

    def failing_reset():
        reset_calls.append(1)
        raise RuntimeError("reset broken too")

    monkeypatch.setattr(
        ledger_run_context, "reset_ledger_run_context", failing_reset
    )

    def boom(**kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(evidence, "register_run", boom)
    # No exception escapes; recording is off either way.
    assert g._bind_run_record_stage("AAPL", "2026-01-05", "stock") == (None, None)
    assert reset_calls


def test_bind_record_stage_regime_failure_degrades_to_no_regime(
    monkeypatch, caplog
):
    from yialpha.ledger import evidence
    from yialpha.regime import compute as regime_compute

    g = _bare_graph({"prediction_ledger": True, "regime_state": True})
    monkeypatch.setattr(evidence, "register_run", lambda **kw: None)

    def boom(*args, **kwargs):
        raise RuntimeError("no klines for regime")

    monkeypatch.setattr(regime_compute, "compute_regime_state", boom)
    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        ledger_run_id, regime_run_id = g._bind_run_record_stage(
            "BTCUSDT", "2026-01-05", "crypto_perp"
        )
    assert ledger_run_id is not None
    assert regime_run_id is None
    assert any(
        "regime state computation failed" in r.message for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# _run_graph: debug streaming + the fail-soft finalize seams
# --------------------------------------------------------------------------- #


def test_run_graph_debug_stream_merges_chunks_and_finalizes_softly(
    tmp_path, monkeypatch, caplog
):
    printed: list[str] = []

    class _Msg:
        def __init__(self, content: str) -> None:
            self.content = content

        def pretty_print(self) -> None:
            printed.append(self.content)

    msg = _Msg("analyst delta")

    class _DebugGraph:
        @staticmethod
        def stream(state, **args):
            yield {"messages": [msg], "company_of_interest": "AAPL"}
            yield {"messages": [msg], "final_trade_decision": None}

    stored: list[dict[str, Any]] = []
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {"results_dir": str(tmp_path)}
    g.debug = True
    g.perf_tracker = SimpleNamespace(reset=lambda: None)

    def failing_dump(trade_date, asset_type="stock"):
        raise RuntimeError("dump blew up")

    monkeypatch.setattr(g, "_dump_perf", failing_dump)
    g.memory_log = SimpleNamespace(
        get_past_context=lambda *a, **kw: "",
        store_decision=lambda **kw: stored.append(kw),
    )
    g.resolve_instrument_context = lambda *a, **kw: "instrument ctx"
    g.propagator = SimpleNamespace(
        create_initial_state=lambda *a, **kw: {"seed": 1},
        get_graph_args=lambda: {},
    )
    g._bind_run_record_stage = lambda *a, **kw: (None, None)
    g._apply_risk_overlay = (
        lambda cn, td, state, ps, asset_type="stock": state
    )
    g._log_state = lambda td, state, at="stock": {"tier": "ok"}
    g.signal_processor = SimpleNamespace(process_signal=lambda s: "Hold")
    g.graph = _DebugGraph()

    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        final_state, signal = g._run_graph(
            "AAPL", "2026-01-02", asset_type="stock", _resuming=False
        )

    # Debug path: streamed chunks merged exactly like invoke would; the
    # unchanged trailing message printed once, not once per chunk.
    assert printed == ["analyst delta"]
    assert final_state["company_of_interest"] == "AAPL"
    assert final_state["data_quality"] == {"tier": "ok"}
    assert signal == "Hold"
    # Missing decision: warned and stored as "" (parse_rating falls back),
    # never a KeyError masking the condition.
    assert any(
        "lacked 'final_trade_decision'" in r.message for r in caplog.records
    )
    assert stored[0]["final_trade_decision"] == ""
    # Perf dump failure costs telemetry only.
    assert any(
        "node perf telemetry dump failed" in r.message for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# _log_state fail-soft + presence-gated log keys
# --------------------------------------------------------------------------- #


def test_log_state_fail_soft_returns_none_and_warns(tmp_path, monkeypatch, caplog):
    g = _bare_graph({"results_dir": str(tmp_path)})
    g.ticker = "AAPL"

    def boom(trade_date, final_state, asset_type="stock"):
        raise OSError("disk full")

    monkeypatch.setattr(g, "_write_states_log", boom)
    with caplog.at_level(logging.WARNING, logger=_TG_LOGGER):
        assert g._log_state("2026-01-02", {}, "stock") is None
    assert any(
        "full_states_log write failed" in r.message for r in caplog.records
    )


def _full_final_state(**extra: Any) -> dict[str, Any]:
    base = {
        "company_of_interest": "AAPL",
        "trade_date": "2026-01-02",
        "market_report": "m",
        "sentiment_report": "s",
        "news_report": "n",
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
        "price_at_decision": 100.0,
        "price_at_decision_basis": "risk_overlay_close",
        # presence-gated record-stage keys
        "run_id": "run-gated",
        "checkpoint_resumed": True,
        "portfolio_control_shadow": {"mode": "shadow", "ticket_id": "T9"},
        "portfolio_control": {"mode": "enforced", "action": "APPROVED"},
        "regime_id": "regime-7",
        "positioning_report": "positioning",
    }
    base.update(extra)
    return base


def test_write_states_log_persists_presence_gated_keys(tmp_path):
    g = _bare_graph({"results_dir": str(tmp_path)})
    g.ticker = "AAPL"
    g.log_states_dict = {"2025-12-31": {"stale": True}}
    final_state = _full_final_state()

    quality.ensure_run_context()
    try:
        block = g._log_state("2026-01-02", final_state, "crypto_perp")
    finally:
        quality.reset_quality()

    log_path = (
        tmp_path / "AAPL" / "YiAlphaStrategy_logs"
        / "full_states_log_2026-01-02_perp.json"
    )
    entry = json.loads(Path(log_path).read_text(encoding="utf-8"))
    # Key PRESENCE (not null values) is the downstream "recorded" signal.
    assert entry["run_id"] == "run-gated"
    assert entry["checkpoint_resumed"] is True
    assert entry["portfolio_control_shadow"] == {"mode": "shadow",
                                                 "ticket_id": "T9"}
    assert entry["portfolio_control"] == {"mode": "enforced",
                                          "action": "APPROVED"}
    assert entry["regime_id"] == "regime-7"
    assert entry["positioning_report"] == "positioning"
    # The consumed quality block rides back to the caller.
    assert block == entry["data_quality"]
    # Write-and-drop: the in-memory dict holds ONLY this date's entry.
    assert g.log_states_dict == {"2026-01-02": entry}


# --------------------------------------------------------------------------- #
# process_signal delegation
# --------------------------------------------------------------------------- #


def test_process_signal_delegates_to_signal_processor():
    g = _bare_graph({})
    g.signal_processor = SimpleNamespace(process_signal=lambda s: f"parsed:{s}")
    assert g.process_signal("**Rating**: Buy") == "parsed:**Rating**: Buy"


# --------------------------------------------------------------------------- #
# risk/manager.py: invested, ledger-stat skipping, sector caps, weight fn
# --------------------------------------------------------------------------- #


def test_portfolio_state_invested_sums_position_values():
    assert PortfolioState(
        positions={"AAPL": 1_000.0, "MSFT": 250.5}
    ).invested == 1_250.5
    assert PortfolioState().invested == 0.0


def test_trade_stats_skips_missing_and_non_numeric_returns():
    history = [
        {"ticker": "X", "rating": "Buy", "return": None},
        {"ticker": "X", "rating": "Buy", "return": "not-a-number"},
        {"ticker": "X", "rating": "Buy", "return": 0},       # flat: neither
        {"ticker": "X", "rating": "Buy", "return": 0.02},
        {"ticker": "X", "rating": "Buy", "return": -0.03},
    ]
    wins, losses, avg_win, avg_loss = _trade_stats(history)
    assert (wins, losses) == (1, 1)
    assert avg_win == 0.02
    assert avg_loss == 0.03


def test_decide_invalid_equity_blocks_under_sector_cap():
    """A dead equity tick (0 / NaN) must NOT silently bypass the sector cap
    (round-8): a fresh breaker replays its neutral start state, the sector
    gate then sees the invalid equity and blocks."""
    rm = RiskManager()
    state = PortfolioState(equity=0.0)  # invalid: not (0 < equity < inf)
    decision = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0,
                         sector="Tech")
    assert decision.action == "blocked"
    assert decision.target_weight == 0.0
    assert decision.exposure_ok is False
    assert "invalid equity" in decision.exposure_reason
    # Zero weight -> no stop is fabricated.
    assert decision.stop_loss is None


def test_decide_sector_cap_reached_blocks_new_size():
    rm = RiskManager()  # max_single_sector 0.30
    # Sector already 35% net of this ticker's own 5%: no headroom left.
    state = PortfolioState(
        equity=100_000.0,
        positions={"AAPL": 5_000.0},
        sectors={"Tech": 40_000.0},
    )
    decision = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0,
                         sector="Tech")
    assert decision.action == "blocked"
    assert decision.target_weight == 0.0
    assert "sector cap reached" in decision.exposure_reason


def test_decide_sector_cap_trims_to_headroom():
    rm = RiskManager(max_single_sector=0.10)
    # Sector 3% net of own book leaves 7% headroom; Kelly wants ~8%.
    state = PortfolioState(
        equity=100_000.0,
        positions={"AAPL": 5_000.0},
        sectors={"Tech": 8_000.0},
    )
    decision = rm.decide("AAPL", "Buy", state, price=100.0, atr=2.0,
                         sector="Tech")
    assert decision.target_weight == pytest.approx(0.07)
    assert "trimmed to sector cap" in decision.exposure_reason


def test_backtest_weight_fn_folds_realized_returns_into_kelly_ledger():
    """The closure's realized-return fold-in must feed decide EXACTLY the
    ledger a direct kelly_sizing call would see (same mirror contract as
    test_round14_risk_graph_types)."""
    rm = RiskManager()
    weight_fn = build_backtest_weight_fn(rm, "AAPL")

    ctx_first = {
        "equity_history": [100_000.0],
        "returns_history": [],
        "execution_price": 100.0,
    }
    w_first = weight_fn("Buy", "2026-01-05", ctx_first)

    ctx_second = {
        "equity_history": [100_000.0],
        "returns_history": [],
        "execution_price": 100.0,
        "realized_returns": [0.02, -0.01],
    }
    w_second = weight_fn("Buy", "2026-01-06", ctx_second)

    for ctx, with_ledger in ((ctx_first, False), (ctx_second, True)):
        decision = ctx["risk_decision"]
        history = [{"return": 0.02}, {"return": -0.01}] if with_ledger else []
        wins, losses, avg_win, avg_loss = _trade_stats(history)
        # Mirror decide()'s exact ledger gating (0 observations == None).
        expected_raw, expected_preclip = kelly_sizing(
            "Buy",
            wins=wins if wins or losses else None,
            losses=losses if wins or losses else None,
            avg_win=avg_win, avg_loss=avg_loss,
            kelly_fraction_mult=0.25, return_raw=True,
        )
        assert decision.kelly_raw == expected_raw
        assert decision.kelly_preclip == expected_preclip
        assert decision.target_weight == pytest.approx(expected_raw)

    # The ledger grew between calls, so the two sizings differ.
    assert w_first != w_second
    # No atr_lookup -> sized position without a stop is disclosed loudly.
    assert "ATR stop unavailable" in ctx_second["risk_warning"]
