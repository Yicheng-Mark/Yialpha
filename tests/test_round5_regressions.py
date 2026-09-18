"""Round-5 regression pins for the 2026-09-19 fix batch.

Each test pins one verified defect from today's uncommitted diff (working
tree read directly; see the per-test docstrings):

* the states log / node-perf filenames gain a venue suffix so a same-date
  perp AND spot run of one ticker stop overwriting each other;
* web/store keys runs by the suffixed filename stem (three first-class
  runs for one date); run_robust resolves the DEGRADED counter by
  newest-mtime glob instead of an exact unsuffixed name;
* ``stock_perp_underlying`` consults the instrument registry under the
  ``instrument_registry`` flag (promote/demote/alias), byte-identical to
  the legacy warm/seed matcher when off;
* the shadow portfolio-control SHORT path gets a sign-aware funding gate
  (halve/quarter ladder on adverse NEGATIVE carry) and the snapshot rides
  BOOK equity, not weight x equity;
* DecisionCache v3 folds asset_type into the key and deletes v2-era
  legacy-path files; the checkpoint run-signature folds input-affecting
  config flags; a checkpointed RESUME passes ``graph_input=None``;
* reporting writes the report tree atomically (no torn .tmp siblings);
* the A/B script's stop/entry regexes parse exponent notation fully;
* FRED macro clamps the LLM-passed window to the pinned analysis date;
* weekly resampling drops a trailing bin whose label arrived but whose
  daily rows do not extend past it (live only);
* polymarket tolerates date-only (naive) endDate; the ETF fund-snapshot
  failure records a quality sentinel; the price-structure / weekly
  renderers emit significant-digit output for micro-price symbols;
* ``run_baseline.smoke`` threads ``asset_type`` into propagate.

Hermetic: every network / LLM / clock seam is monkeypatched or tmp-pathed.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import yialpha.dataflows.binance as bn
import yialpha.dataflows.fred as fred
import yialpha.dataflows.fundamentals_bundle as fb
import yialpha.dataflows.ohlcv_resample as ors
import yialpha.instruments.registry as registry
from yialpha.backtest.cache import (
    DECISION_CACHE_SCHEMA_VERSION,
    CachedDecision,
    DecisionCache,
    _safe_component,
)
from yialpha.dataflows import quality
from yialpha.dataflows.config import set_config
from yialpha.dataflows.utils import set_analysis_date
from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.instruments.models import InstrumentRecord

_TODAY = date.today().isoformat()


def _load_script(name: str):
    """Import ``scripts/<name>.py`` under a private module name (no side
    effects on sys.modules), the same pattern the trade_ticket pin uses."""
    spec = importlib.util.spec_from_file_location(
        f"{name}_round5_under_test",
        Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Shared graph shells (test_regime_state.py / test_v24_acceptance.py seams)
# --------------------------------------------------------------------------- #
def _graph_shell(tmp_path, **config_over) -> YiAlphaGraph:
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {
        "risk_enabled": True,
        "kelly_fraction": 0.25,
        "max_single_position": 0.20,
        "max_single_sector": 0.30,
        "max_drawdown_hard_stop": 0.15,
        "atr_stop_mult": 2.0,
        "results_dir": str(tmp_path),
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    }
    g.config.update(config_over)
    g._risk_overlay_degraded = False
    g.risk_manager = g._build_risk_manager()
    g.ticker = "BTCUSDT"
    g.debug = False
    g.perf_tracker = None
    return g


def _shadow_graph(tmp_path, **config_over) -> YiAlphaGraph:
    return _graph_shell(tmp_path, portfolio_control_mode="shadow", **config_over)


def _bind_run(run_id: str, instrument_class: str = "pure_crypto_perp") -> None:
    from yialpha.ledger.evidence import register_run
    from yialpha.ledger.run_context import set_ledger_run_context

    register_run(run_id, "BTCUSDT", "crypto_perp", instrument_class, _TODAY)
    set_ledger_run_context(
        run_id, "BTCUSDT", "crypto_perp", instrument_class, _TODAY
    )


def _perp_state(**over) -> dict:
    base = {
        "final_trade_decision": "**Rating**: Buy\n\nThesis.",
        "pm_rating": "Buy",
        "pm_decision_fields": {},
    }
    base.update(over)
    return base


def _stub_overlay_prices(g, monkeypatch, funding_annualized: float | None) -> None:
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (60.0, 1.5)
    )
    # funding_total_7d is the gate's input; annualized = total / 7 * 365.
    monkeypatch.setattr(
        g,
        "_trailing_funding_total",
        lambda t, d: None
        if funding_annualized is None
        else funding_annualized * 7.0 / 365.0,
    )
    # _latest_mark_close / _render_stress_line: conftest autouse stubs.


@pytest.fixture(autouse=True)
def _clean_run_context():
    from yialpha.dataflows.run_scope import reset_run_scope
    from yialpha.ledger.run_context import reset_ledger_run_context

    reset_ledger_run_context()
    quality.reset_quality()
    reset_run_scope()
    yield
    reset_ledger_run_context()
    quality.reset_quality()
    reset_run_scope()


# --------------------------------------------------------------------------- #
# 1. states log / node-perf venue suffix
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_states_log_stem_venue_suffix_mapping():
    """crypto_perp → ``<date>_perp``, crypto_spot → ``<date>_spot``; stock
    and the legacy "crypto" keep the unsuffixed name (equities never had the
    same-ticker-same-date venue collision)."""
    assert (
        YiAlphaGraph._states_log_stem("2026-09-19", "crypto_perp")
        == "full_states_log_2026-09-19_perp"
    )
    assert (
        YiAlphaGraph._states_log_stem("2026-09-19", "crypto_spot")
        == "full_states_log_2026-09-19_spot"
    )
    assert (
        YiAlphaGraph._states_log_stem("2026-09-19", "stock")
        == "full_states_log_2026-09-19"
    )
    assert (
        YiAlphaGraph._states_log_stem("2026-09-19", "crypto")
        == "full_states_log_2026-09-19"
    )
    # Defensive default: a missing asset_type reads as the stock pipeline.
    assert (
        YiAlphaGraph._states_log_stem("2026-09-19", None)
        == "full_states_log_2026-09-19"
    )


def _loggable_state(**over) -> dict:
    state = {
        "company_of_interest": "BTCUSDT",
        "trade_date": "2026-09-19",
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
        # set so _log_state skips the fallback price loader (network seam)
        "price_at_decision": 65000.0,
        "price_at_decision_basis": "risk_overlay_close",
        "asset_type": "crypto_perp",
    }
    state.update(over)
    return state


@pytest.mark.unit
def test_log_state_lands_venue_suffixed_file(tmp_path):
    """The old ticker+date-only filename silently overwrote a same-date perp
    run with its spot twin (or the stock run); perp and spot must land as
    two first-class files while stock keeps the legacy name."""
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {"results_dir": str(tmp_path)}
    g.ticker = "BTCUSDT"

    quality.ensure_run_context()
    try:
        g._log_state("2026-09-19", _loggable_state(), "crypto_perp")
        g._log_state(
            "2026-09-19", _loggable_state(asset_type="crypto_spot"), "crypto_spot"
        )
        g._log_state("2026-09-19", _loggable_state(asset_type="stock"), "stock")
    finally:
        quality.reset_quality()

    sdir = tmp_path / "BTCUSDT" / "YiAlphaStrategy_logs"
    perp_file = sdir / "full_states_log_2026-09-19_perp.json"
    spot_file = sdir / "full_states_log_2026-09-19_spot.json"
    plain_file = sdir / "full_states_log_2026-09-19.json"
    assert perp_file.is_file() and spot_file.is_file() and plain_file.is_file()
    # All three coexist (no overwrite) and carry their venue tag.
    for path, expected in (
        (perp_file, "crypto_perp"), (spot_file, "crypto_spot"),
        (plain_file, "stock"),
    ):
        entry = json.loads(path.read_text(encoding="utf-8"))
        assert entry["asset_type"] == expected
    # The atomic-write sibling tmp is gone after each replace.
    assert not [p for p in sdir.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.unit
def test_dump_perf_uses_same_venue_suffix(tmp_path, monkeypatch):
    """node_perf_<date>.json shares the states-log stem's venue suffix, so a
    same-date perp+spot pair keeps two telemetry files like two logs (and
    web/store.load_node_perf finds the suffixed one)."""
    captured: list[Path] = []
    monkeypatch.setattr(
        "yialpha.graph.perf_telemetry.dump_perf_report",
        lambda tracker, path: captured.append(Path(path)),
    )
    g = YiAlphaGraph.__new__(YiAlphaGraph)
    g.config = {"results_dir": str(tmp_path)}
    g.ticker = "BTCUSDT"
    g.perf_tracker = object()

    g._dump_perf("2026-09-19", "crypto_spot")
    g._dump_perf("2026-09-19", "crypto_perp")
    g._dump_perf("2026-09-19", "stock")

    names = [p.name for p in captured]
    assert names == [
        "node_perf_2026-09-19_spot.json",
        "node_perf_2026-09-19_perp.json",
        "node_perf_2026-09-19.json",
    ]
    # Perf lands next to the states log in the per-ticker directory.
    assert all(p.parent == tmp_path / "BTCUSDT" / "YiAlphaStrategy_logs" for p in captured)


# --------------------------------------------------------------------------- #
# 2. web/store venue-suffixed run keys
# --------------------------------------------------------------------------- #
def _write_states_log(root: Path, stem: str, state: dict) -> None:
    d = root / "BTCUSDT" / "YiAlphaStrategy_logs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"full_states_log_{stem}.json").write_text(
        json.dumps(state), encoding="utf-8"
    )


@pytest.mark.unit
def test_web_store_dates_yield_three_venue_keys(tmp_path, monkeypatch):
    """A strategy dir holding the plain + _perp + _spot files for one date
    yields THREE date keys — the suffix is part of the run identity, and the
    captured regex group IS the filename stem downstream constructors use."""
    from web import store

    monkeypatch.setattr(store, "LOGS_ROOT", tmp_path)
    _write_states_log(
        tmp_path, "2026-09-19",
        {"final_trade_decision": "**Rating**: Buy", "asset_type": "stock"},
    )
    _write_states_log(
        tmp_path, "2026-09-19_perp",
        {"final_trade_decision": "**Rating**: Sell", "asset_type": "crypto_perp"},
    )
    _write_states_log(
        tmp_path, "2026-09-19_spot",
        {"final_trade_decision": "**Rating**: Hold", "asset_type": "crypto_spot"},
    )

    runs = store.list_runs("BTCUSDT")
    assert runs["dates"] == ["2026-09-19", "2026-09-19_perp", "2026-09-19_spot"]
    # list_tickers exposes three runs, latest by lexicographic stem order.
    (ticker_entry,) = [t for t in store.list_tickers() if t["ticker"] == "BTCUSDT"]
    assert ticker_entry["run_count"] == 3
    assert ticker_entry["latest_date"] == "2026-09-19_spot"


@pytest.mark.unit
def test_web_store_load_run_reads_perp_file(tmp_path, monkeypatch):
    """``load_run(ticker, "2026-09-19_perp")`` must read the PERP file (the
    exact-name constructor works unchanged with the suffixed key), serve its
    rating/asset_type, and pair it with the suffixed node_perf file."""
    from web import store

    monkeypatch.setattr(store, "LOGS_ROOT", tmp_path)
    _write_states_log(
        tmp_path, "2026-09-19",
        {"final_trade_decision": "**Rating**: Buy", "asset_type": "stock"},
    )
    _write_states_log(
        tmp_path, "2026-09-19_perp",
        {
            "final_trade_decision": "**Rating**: Sell",
            "asset_type": "crypto_perp",
            "trade_date": "2026-09-19",
        },
    )
    perf_dir = tmp_path / "BTCUSDT" / "YiAlphaStrategy_logs"
    (perf_dir / "node_perf_2026-09-19_perp.json").write_text(
        json.dumps({"nodes": []}), encoding="utf-8"
    )

    run = store.load_run("BTCUSDT", "2026-09-19_perp")
    assert run is not None
    assert run["rating"] == "Sell"
    assert run["asset_type"] == "crypto_perp"
    assert run["node_perf"] == {"nodes": []}
    # The stock twin is a separate, independently loadable run.
    assert store.load_run("BTCUSDT", "2026-09-19")["rating"] == "Buy"


# --------------------------------------------------------------------------- #
# 3. run_robust DEGRADED counter: newest-by-mtime glob
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_core_sentinel_count_newest_venue_file_wins(tmp_path):
    """``_core_sentinel_count`` must resolve the venue-suffixed files by glob
    and take the NEWEST match for the date — reading the constructed
    unsuffixed name would miss the perp/spot run that actually finished, and
    picking the older file would report the wrong sentinel count."""
    rr = _load_script("run_robust")
    sdir = tmp_path / "MU" / "YiAlphaStrategy_logs"
    sdir.mkdir(parents=True)
    plain = sdir / "full_states_log_2026-09-19.json"
    perp = sdir / "full_states_log_2026-09-19_perp.json"
    plain.write_text(
        json.dumps({"data_quality": {"core_sentinel_count": 0}}), encoding="utf-8"
    )
    perp.write_text(
        json.dumps({"data_quality": {"core_sentinel_count": 2}}), encoding="utf-8"
    )
    # Older mtime on the plain (stock) log, newer on the perp log.
    os.utime(plain, (1_000_000, 1_000_000))
    os.utime(perp, (2_000_000, 2_000_000))

    reports_root = tmp_path / "reports"  # .parent is the ticker root
    assert rr._core_sentinel_count(reports_root, "MU", "2026-09-19") == 2
    # Reverse the mtimes → the plain file wins with its own count.
    os.utime(plain, (3_000_000, 3_000_000))
    os.utime(perp, (2_000_000, 2_000_000))
    assert rr._core_sentinel_count(reports_root, "MU", "2026-09-19") == 0


@pytest.mark.unit
def test_core_sentinel_count_none_without_any_log(tmp_path):
    """No candidate files → None (unknown, not zero): a missing log must not
    be laundered into a clean quality verdict."""
    rr = _load_script("run_robust")
    assert rr._core_sentinel_count(tmp_path / "reports", "MU", "2026-09-19") is None


# --------------------------------------------------------------------------- #
# 4. stock_perp_underlying registry consult
# --------------------------------------------------------------------------- #
def _record(**over) -> InstrumentRecord:
    fields = {
        "symbol": "MUUSDT",
        "instrument_class": "stock_perp",
        "classification_source": registry.SOURCE_EXCHANGEINFO,
        "classification_confidence": 1.0,
    }
    fields.update(over)
    return InstrumentRecord(**fields)


@pytest.mark.unit
def test_stock_perp_underlying_flag_off_never_touches_registry(monkeypatch):
    """Flag OFF is byte-identical to the legacy warm/seed matcher: the
    registry must not even be consulted (the fresh-subprocess divergence the
    promote/demote rules exist to fix is opt-in)."""
    set_config({"instrument_registry": False})

    def explode(*args, **kwargs):
        raise AssertionError("registry must not be consulted with the flag off")

    monkeypatch.setattr(registry, "classify_perp", explode)
    monkeypatch.setattr(bn, "equity_perp_bases", lambda: frozenset({"MU", "NVDA"}))
    assert bn.stock_perp_underlying("MUUSDT") == "MU"
    monkeypatch.setattr(bn, "equity_perp_bases", lambda: frozenset())
    assert bn.stock_perp_underlying("MUUSDT") is None


@pytest.mark.unit
def test_stock_perp_underlying_registry_promotes_warm_miss(monkeypatch):
    """Flag ON + an exchangeInfo-backed stock_perp row PROMOTES a base the
    warm/seed set missed — the registry is authoritative in both directions."""
    set_config({"instrument_registry": True})
    monkeypatch.setattr(bn, "equity_perp_bases", lambda: frozenset())
    monkeypatch.setattr(
        registry, "classify_perp",
        lambda symbol, as_of=None: _record(underlying_symbol="MU"),
    )
    assert bn.stock_perp_underlying("MUUSDT") == "MU"


@pytest.mark.unit
def test_stock_perp_underlying_registry_empty_warm_answer_stands(monkeypatch):
    """No persisted evidence (registry_empty) is not a demotion: the
    warm/seed positive stands — positive classifications never regress for
    seed/historical symbols."""
    set_config({"instrument_registry": True})
    monkeypatch.setattr(bn, "equity_perp_bases", lambda: frozenset({"NVDA"}))
    monkeypatch.setattr(
        registry, "classify_perp",
        lambda symbol, as_of=None: _record(
            symbol="NVDAUSDT",
            classification_source=registry.SOURCE_REGISTRY_EMPTY,
            classification_confidence=0.0,
        ),
    )
    assert bn.stock_perp_underlying("NVDAUSDT") == "NVDA"


@pytest.mark.unit
def test_stock_perp_underlying_registry_demotes_pure_crypto(monkeypatch):
    """Flag ON + exchangeInfo says pure_crypto_perp → None even when the
    seed still contains the base (a stale warm positive the exchange no
    longer lists as EQUITY)."""
    set_config({"instrument_registry": True})
    monkeypatch.setattr(bn, "equity_perp_bases", lambda: frozenset({"MU"}))
    monkeypatch.setattr(
        registry, "classify_perp",
        lambda symbol, as_of=None: _record(instrument_class="pure_crypto_perp"),
    )
    assert bn.stock_perp_underlying("MUUSDT") is None


@pytest.mark.unit
def test_stock_perp_underlying_registry_alias_brkb(monkeypatch):
    """A registry-resolved base maps through the SAME Yahoo alias seam as
    the warm/seed matcher: raw exchangeInfo ``BRKB`` → Yahoo ``BRK-B``."""
    from yialpha.dataflows.symbol_utils import yahoo_equity_symbol

    set_config({"instrument_registry": True})
    monkeypatch.setattr(bn, "equity_perp_bases", lambda: frozenset())
    monkeypatch.setattr(
        registry, "classify_perp",
        lambda symbol, as_of=None: _record(
            symbol="BRKBUSDT", underlying_symbol="BRKB"
        ),
    )
    assert bn.stock_perp_underlying("BRKBUSDT") == "BRK-B"
    # The seam itself is the single mapping point.
    assert yahoo_equity_symbol("BRKB") == "BRK-B"
    assert yahoo_equity_symbol("mu") == "MU"


# --------------------------------------------------------------------------- #
# 5. shadow SHORT funding gate
# --------------------------------------------------------------------------- #
def _run_shadow_short(g, monkeypatch, funding_annualized, run_id):
    from yialpha.ledger.run_context import reset_ledger_run_context

    _stub_overlay_prices(g, monkeypatch, funding_annualized)
    _bind_run(run_id)
    try:
        return g._apply_risk_overlay(
            "BTCUSDT", _TODAY,
            _perp_state(
                final_trade_decision="**Rating**: Sell\n\nThesis.",
                pm_rating="Sell",
                pm_decision_fields={"desired_side": "SHORT"},
            ),
            {"equity": 100_000},
            asset_type="crypto_perp",
        )
    finally:
        reset_ledger_run_context()


@pytest.mark.unit
def test_short_funding_gate_quarters_on_hard_adverse_carry(tmp_path, monkeypatch):
    """A SHORT PAYS NEGATIVE funding: carry <= -0.60/yr quarters the
    heuristic proposal (0.25 x kelly x max_single) and the shadow record
    discloses ``heuristic_funding_damped`` + a rendered Funding gate line —
    the legacy sign-blind gate let adverse short carry pass undamped."""
    g = _shadow_graph(tmp_path)
    out = _run_shadow_short(g, monkeypatch, funding_annualized=-0.72, run_id="R5SGQ1")
    record = out["portfolio_control_shadow"]
    base = 0.25 * 0.20  # kelly_fraction x max_single_position
    assert record["short_sizing"] == "heuristic_funding_damped"
    assert record["short_funding_damp"] == 0.25
    assert record["short_funding_annualized"] == pytest.approx(-0.72)
    assert record["proposed_size"] == pytest.approx(base * 0.25)
    assert "Funding gate" in out["final_trade_decision"]
    assert "sign-aware" in out["final_trade_decision"]


@pytest.mark.unit
def test_short_funding_gate_halves_on_warn_carry(tmp_path, monkeypatch):
    g = _shadow_graph(tmp_path)
    out = _run_shadow_short(g, monkeypatch, funding_annualized=-0.36, run_id="R5SGH1")
    record = out["portfolio_control_shadow"]
    assert record["short_funding_damp"] == 0.5
    assert record["proposed_size"] == pytest.approx(0.25 * 0.20 * 0.5)
    assert "Funding gate" in out["final_trade_decision"]


@pytest.mark.unit
def test_short_funding_gate_positive_carry_undamped(tmp_path, monkeypatch):
    """Positive funding is FAVOURABLE to a short: never damps (and never
    boosts — symmetric with the long gate's no-boost rule)."""
    g = _shadow_graph(tmp_path)
    out = _run_shadow_short(g, monkeypatch, funding_annualized=0.40, run_id="R5SGP1")
    record = out["portfolio_control_shadow"]
    assert record["short_sizing"] == "heuristic"
    assert record["short_funding_damp"] == 1.0
    assert record["proposed_size"] == pytest.approx(0.25 * 0.20)
    assert "Funding gate" not in out["final_trade_decision"]


# --------------------------------------------------------------------------- #
# 6. shadow snapshot rides BOOK equity
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_shadow_equity_is_book_equity_not_position_value(tmp_path, monkeypatch):
    """With portfolio_state {"equity": 100000} and a nonzero target_weight
    decision, the snapshot equity is the BOOK (100000) — the old
    ``position_value or equity`` chain fed advisory_metrics an account
    weight-times too small (position_value = weight x equity)."""
    from yialpha.ledger.run_context import reset_ledger_run_context

    g = _shadow_graph(tmp_path)
    _stub_overlay_prices(g, monkeypatch, funding_annualized=None)
    _bind_run("R5SEQ1")
    try:
        out = g._apply_risk_overlay(
            "BTCUSDT", _TODAY, _perp_state(), {"equity": 100_000},
            asset_type="crypto_perp",
        )
    finally:
        reset_ledger_run_context()
    record = out["portfolio_control_shadow"]
    assert record["equity"] == 100_000.0
    # The wrong value (weight x equity) is capped by max_single_position at
    # 0.20 x 100000 — the pin is that equity equals the full book.
    assert record["equity"] > 0.20 * 100_000
    assert record["candidate"]["weight"] > 0.0  # a nonzero-weight LONG decision


# --------------------------------------------------------------------------- #
# 7. DecisionCache v3: asset_type in the key
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_decision_cache_asset_type_key_roundtrip(tmp_path):
    cache = DecisionCache(tmp_path / "dc", enabled=True)
    rec = cache.remember(
        "BTCUSDT", "2026-09-19", "Sell", "**Rating**: Sell",
        run_tag="default", asset_type="crypto_perp",
    )
    assert isinstance(rec, CachedDecision)
    assert rec.asset_type == "crypto_perp"
    assert DECISION_CACHE_SCHEMA_VERSION == 3
    # Persisted payload carries the venue (roundtrip through to_dict/from_dict).
    path = cache._key_path("BTCUSDT", "2026-09-19", "default", "crypto_perp")
    assert path is not None and path.is_file()
    hit = cache.get("BTCUSDT", "2026-09-19", asset_type="crypto_perp")
    assert hit is not None
    assert hit.rating == "Sell"
    assert hit.asset_type == "crypto_perp"
    assert hit.final_decision == "**Rating**: Sell"
    # A different venue is a MISS (never replays the other venue's decision)
    # and does not evict the perp entry.
    assert cache.get("BTCUSDT", "2026-09-19", asset_type="crypto_spot") is None
    assert cache.get("BTCUSDT", "2026-09-19", asset_type="crypto_perp") is not None


@pytest.mark.unit
def test_decision_cache_v2_legacy_file_removed_on_miss(tmp_path):
    """A v2-era file at the legacy 3-component path fails the version check
    (get → None) AND is removed, so the stale entry cannot shadow the
    correctly-keyed v3 entry later."""
    cache = DecisionCache(tmp_path / "dc", enabled=True)
    legacy = cache.cache_dir / (
        f"{_safe_component('BTCUSDT')}_{_safe_component('2026-09-19')}_"
        f"{_safe_component('default')}.json"
    )
    legacy.write_text(
        json.dumps({
            "schema_version": 2,
            "ticker": "BTCUSDT",
            "date": "2026-09-19",
            "run_tag": "default",
            "rating": "Buy",
            "final_decision": "stale",
        }),
        encoding="utf-8",
    )
    assert legacy.is_file()
    assert cache.get("BTCUSDT", "2026-09-19", asset_type="crypto_perp") is None
    assert not legacy.exists()
    # And even the default (stock) venue reads a miss, not the stale row.
    assert cache.get("BTCUSDT", "2026-09-19") is None


@pytest.mark.unit
def test_decision_cache_default_venue_still_stock(tmp_path):
    """The default asset_type stays "stock" on both remember and get — the
    pre-v3 single-venue call sites keep working unchanged."""
    cache = DecisionCache(tmp_path / "dc", enabled=True)
    cache.remember("AAPL", "2026-09-19", "Buy", "md")
    hit = cache.get("AAPL", "2026-09-19")
    assert hit is not None and hit.asset_type == "stock"


# --------------------------------------------------------------------------- #
# 8. checkpoint run-signature config segment
# --------------------------------------------------------------------------- #
def _sig_shell(tmp_path, **config_over) -> YiAlphaGraph:
    g = _graph_shell(tmp_path, **config_over)
    g.selected_analysts = ["market_analyst", "news_analyst"]
    return g


@pytest.mark.unit
def test_run_signature_folds_input_affecting_config(tmp_path):
    """The signature carries a ``config=`` segment of the tool-binding /
    semantic flags, so a checkpoint restored under a different toolset
    starts fresh instead of serving stale analyst reports."""
    g = _sig_shell(tmp_path)
    sig = g._run_signature("stock")
    assert "config=" in sig
    # Same config → stable (sorted JSON: key order never churns it).
    assert g._run_signature("stock") == sig
    # indicator_battery flip changes the signature.
    g.config["indicator_battery"] = ["macd"]
    assert g._run_signature("stock") != sig
    g.config["indicator_battery"] = None
    assert g._run_signature("stock") == sig
    # sec_ownership flip changes the signature.
    g.config["sec_ownership"] = True
    assert g._run_signature("stock") != sig
    # The segment is machine-parseable sorted JSON carrying the keys.
    seg = sig.split("config=", 1)[1]
    payload = json.loads(seg)
    assert payload["sec_ownership"] is None  # unset flag degrades to None
    assert payload["indicator_battery"] is None


# --------------------------------------------------------------------------- #
# 9. reporting atomic write
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_write_report_tree_atomic_no_tmp_leftovers(tmp_path):
    """write_report_tree writes via tmp-file + os.replace: the report tree
    lands complete and NO ``.tmp`` sibling survives anywhere under the
    output dir (run_robust's taskkill window used to risk torn files)."""
    from yialpha.reporting import write_report_tree

    decision = "**Rating**: Buy\n\nFinal thesis."
    state = {
        "market_report": "Market section body.",
        "final_trade_decision": decision,
    }
    out_dir = tmp_path / "rep"
    out_path = write_report_tree(state, "BTCUSDT", out_dir)
    assert out_path == out_dir / "complete_report.md"
    assert (out_dir / "5_portfolio" / "decision.md").read_text(
        encoding="utf-8"
    ) == decision
    assert (out_dir / "1_analysts" / "market.md").read_text(
        encoding="utf-8"
    ) == "Market section body."
    complete = (out_dir / "complete_report.md").read_text(encoding="utf-8")
    assert "Final thesis." in complete
    assert "BTCUSDT" in complete
    leftovers = [p for p in out_dir.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# 10. A/B script exponent regexes
# --------------------------------------------------------------------------- #
_ab = _load_script("run_analyst_parallel_ab")


@pytest.mark.unit
def test_ab_price_regexes_parse_exponent_fully():
    """The overlay renders Stop Loss / Entry Reference with %.6g, which emits
    exponent form below 1e-4; the old ``[\\-0-9.]+`` class stopped at the
    ``e`` and parsed 1.14 out of 1.14e-05 — feeding the determinism compare
    values five orders of magnitude off."""
    m = _ab._STOP_LOSS_RE.search("- **Stop Loss**: 1.14e-05")
    assert m is not None
    assert m.group(1) == "1.14e-05"
    assert float(m.group(1)) == pytest.approx(1.14e-05)
    assert float(m.group(1)) != pytest.approx(1.14)

    m = _ab._ENTRY_REF_RE.search("- **Entry Reference**: $1.02E-05")
    assert m is not None
    assert m.group(1) == "1.02E-05"
    assert float(m.group(1)) == pytest.approx(1.02e-05)

    # Legacy magnitudes keep parsing unchanged (incl. the optional $).
    m = _ab._STOP_LOSS_RE.search("- **Stop Loss**: 88000.25")
    assert m.group(1) == "88000.25"
    m = _ab._ENTRY_REF_RE.search("- **Entry Reference**: $90000.0")
    assert m.group(1) == "90000.0"


# --------------------------------------------------------------------------- #
# 11. FRED PIT clamp
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_fred_macro_window_clamped_to_pinned_analysis_date(monkeypatch):
    """``curr_date`` comes from the LLM; the pinned analysis date is the
    authority a historical replay must not see past. With the pin at
    2026-09-01, an LLM-echoed 2026-12-31 must not widen the observation
    window past the pin."""
    captured: dict[str, str] = {}

    def fake_request(path: str, params: dict) -> dict:
        if path == "series":
            return {"seriess": [{
                "title": "Consumer Price Index",
                "units_short": "Idx",
                "frequency": "Monthly",
                "seasonal_adjustment_short": "SA",
            }]}
        captured.update(params)
        return {"observations": []}

    monkeypatch.setattr(fred, "_request", fake_request)
    try:
        set_analysis_date("2026-09-01")
        out = fred.get_macro_data("cpi", "2026-12-31")
        assert captured["observation_end"] == "2026-09-01"
        assert "2026-09-01" in out  # the rendered window names the clamped end
    finally:
        set_analysis_date(None)
    # Live mode (no pin): the LLM-passed end stands — byte-identical legacy
    # behaviour for a same-day interactive run.
    out = fred.get_macro_data("cpi", "2026-12-31")
    assert captured["observation_end"] == "2026-12-31"
    assert "2026-12-31" in out


# --------------------------------------------------------------------------- #
# 12. weekly resample forming-week guard
# --------------------------------------------------------------------------- #
def _daily_frame(days: list[date], close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime([d.isoformat() for d in days]),
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Volume": 1.0,
    })


def _live_week_days() -> tuple[date, date, date]:
    """(prev_friday, last_friday_on_or_before_today, today).

    ``today`` is the host-local live anchor (is_historical_date → False for
    it by construction); the trailing bin's Friday label is
    ``last_friday_on_or_before_today``."""
    today = date.today()
    last_friday = today - timedelta(days=(today.weekday() - 4) % 7)
    return last_friday - timedelta(days=7), last_friday, today


@pytest.mark.unit
def test_resample_weekly_drops_forming_trailing_week_on_live_date():
    """Live run dated on/after the trailing bin's own Friday label, last
    daily row ON that label → the trailing week is DROPPED: a bin is
    complete only when a daily row exists BEYOND its label."""
    prev_friday, last_friday, today = _live_week_days()
    days = [prev_friday] + [
        last_friday - timedelta(days=k) for k in (4, 3, 2, 1, 0)
    ]
    weekly = ors.resample_weekly(_daily_frame(days), curr_date=today.isoformat())
    labels = [ts.date() for ts in weekly["Date"]]
    # The trailing (still-forming) week is gone; the prior closed week stays.
    assert last_friday not in labels
    assert labels == [prev_friday]


@pytest.mark.unit
def test_resample_weekly_keeps_week_when_row_beyond_label_exists():
    """A Saturday daily row beyond the Friday label proves the bin closed →
    the week is KEPT (this is exactly what load_ohlcv serves on a crypto
    venue after the UTC day rolls)."""
    prev_friday, last_friday, today = _live_week_days()
    days = [prev_friday] + [
        last_friday - timedelta(days=k) for k in (4, 3, 2, 1, 0)
    ] + [last_friday + timedelta(days=1)]  # the Saturday row
    weekly = ors.resample_weekly(_daily_frame(days), curr_date=today.isoformat())
    labels = [ts.date() for ts in weekly["Date"]]
    assert labels == [prev_friday, last_friday]


@pytest.mark.unit
def test_resample_weekly_historical_date_skips_forming_guard():
    """A historical curr_date keeps the label-equal trailing week: replayed
    bars are closed by PIT, so the live forming-guard must not run (same
    frame shape that the live test drops)."""
    prev_friday = date(2026, 1, 2)
    last_friday = date(2026, 1, 9)  # a Friday, firmly in the past
    days = [prev_friday] + [
        last_friday - timedelta(days=k) for k in (4, 3, 2, 1, 0)
    ]
    weekly = ors.resample_weekly(_daily_frame(days), curr_date="2026-01-09")
    labels = [ts.date() for ts in weekly["Date"]]
    assert labels == [prev_friday, last_friday]


# --------------------------------------------------------------------------- #
# 13. polymarket naive endDate
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_polymarket_naive_end_date_does_not_raise():
    """A date-only endDate ("2026-12-31") parses naive; comparing it against
    the tz-aware ``now`` raised TypeError, which the ValueError-only guard
    let escape and degrade the WHOLE optional category for one market's
    date form. Both directions must compare correctly."""
    from yialpha.dataflows.polymarket import _is_forward_looking

    def market(end_date: str) -> dict:
        return {
            "closed": False,
            "endDate": end_date,
            "outcomePrices": '["0.5", "0.5"]',
            "outcomes": '["Yes", "No"]',
        }

    now = datetime.now(UTC)
    future_naive = (date.today() + timedelta(days=30)).isoformat()
    past_naive = (date.today() - timedelta(days=30)).isoformat()
    assert _is_forward_looking(market(future_naive), now) is True
    assert _is_forward_looking(market(past_naive), now) is False
    # The tz-aware Z form keeps its pinned behaviour on both sides too.
    assert _is_forward_looking(market(f"{future_naive}T00:00:00Z"), now) is True
    assert _is_forward_looking(market(f"{past_naive}T00:00:00Z"), now) is False


# --------------------------------------------------------------------------- #
# 14. ETF fund-snapshot failure records a quality sentinel
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_etf_fund_failure_records_quality_sentinel(monkeypatch):
    """``_fetch_etf_fund_data`` is a direct-connect vendor (bypasses the
    router): its failure must land in the data-quality ledger too, not only
    in the in-band STATUS_UNAVAILABLE footer."""
    def boom(ticker, curr_date):
        raise RuntimeError("vendor down")

    monkeypatch.setattr(
        "yialpha.dataflows.etf_fund_data.get_etf_fund_data", boom
    )
    quality.ensure_run_context()
    try:
        out = fb._fetch_etf_fund_data("SPY", _TODAY)
        events = quality.snapshot_quality()
    finally:
        quality.reset_quality()
    assert out["status"] == fb.STATUS_UNAVAILABLE
    assert "vendor down" in out["reason"]
    matches = [
        e for e in events
        if e["method"] == "get_etf_fund_data"
        and e["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE
    ]
    assert matches, f"ETF fund failure must reach the quality ledger: {events}"
    assert "SPY" in matches[0]["detail"]


# --------------------------------------------------------------------------- #
# 15. micro-price renderers
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_fmt_price_magnitude_tiers():
    """price_structure_tools._fmt_price: micro prices render in significant
    digits (PEPEUSDT-class ~1e-5, where .2f printed "0.00" and destroyed the
    mandated-citation evidence); thousands keep separators; mid-range keeps
    <=4 decimals with trailing zeros stripped; None/NaN → N/A."""
    from yialpha.agents.utils.price_structure_tools import _fmt_price

    assert _fmt_price(1.14e-05) == "1.14e-05"
    assert _fmt_price(1234.5) == "1,234.50"
    assert _fmt_price(0.05) == "0.05"
    assert _fmt_price(65000.0) == "65,000.00"
    assert _fmt_price(None) == "N/A"
    assert _fmt_price(float("nan")) == "N/A"
    assert _fmt_price(float("inf")) == "N/A"


@pytest.mark.unit
def test_fmt_num_magnitude_tiers():
    """weekly_indicators_tools._fmt_num: same tiering for the weekly table's
    price-scale cells (Close/MACD on a micro-price spot symbol)."""
    from yialpha.agents.utils.weekly_indicators_tools import _fmt_num

    assert _fmt_num(1.14e-05) == "1.14e-05"
    assert _fmt_num(1234.5) == "1,234.50"
    assert _fmt_num(0.05) == "0.05"
    assert _fmt_num(None) == "N/A"
    assert _fmt_num(float("nan")) == "N/A"
    # RSI-scale values are unchanged by the tiering.
    assert _fmt_num(55.23) == "55.23"


# --------------------------------------------------------------------------- #
# 16. run_baseline smoke threads asset_type
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_smoke_threads_asset_type_into_propagate(monkeypatch):
    """``--smoke --asset-type crypto_perp`` used to run the STOCK pipeline
    against a perp symbol (propagate's default): the perp smoke route is the
    documented way to verify the perp pipeline, so smoke must thread the
    flag into propagate."""
    rb = _load_script("run_baseline")
    captured: dict = {}

    class FakeGraph:
        perf_tracker = None

        def propagate(self, ticker, trade_date, asset_type="stock"):
            captured["propagate"] = (ticker, trade_date, asset_type)
            return {"final_trade_decision": "**Rating**: Buy"}, "Buy"

    monkeypatch.setattr(rb, "_build_graph", lambda **kwargs: FakeGraph())
    rc = rb.smoke("BTCUSDT", "2026-09-19", profile=False, asset_type="crypto_perp")
    assert rc == 0
    assert captured["propagate"] == ("BTCUSDT", "2026-09-19", "crypto_perp")
    # Default stays the stock pipeline (legacy callers unchanged).
    rc = rb.smoke("AAPL", "2026-09-19")
    assert rc == 0
    assert captured["propagate"] == ("AAPL", "2026-09-19", "stock")


# --------------------------------------------------------------------------- #
# 17. checkpoint resume passes graph_input=None
# --------------------------------------------------------------------------- #
def _resumable_shell(tmp_path, **config_over) -> YiAlphaGraph:
    from types import SimpleNamespace

    config = {"checkpoint_enabled": True, "data_cache_dir": str(tmp_path / "ckpt")}
    config.update(config_over)
    g = _graph_shell(tmp_path, **config)
    g.selected_analysts = ["market_analyst"]
    g.memory_log = SimpleNamespace(
        get_past_context=lambda *a, **k: None,
        store_decision=lambda **k: None,
    )
    g.resolve_instrument_context = lambda *a, **k: None
    g.propagator = SimpleNamespace(
        create_initial_state=lambda *a, **k: {"final_trade_decision": "**Rating**: Buy"},
        get_graph_args=lambda: {},
    )
    g._log_state = lambda d, fs, asset_type="stock": {}
    g.process_signal = lambda md: "BUY"
    return g


@pytest.mark.unit
def test_run_graph_resume_passes_none_graph_input(tmp_path, monkeypatch):
    """langgraph re-executes from __start__ whenever a non-None input dict
    is passed: a checkpointed "resume" that fed the fresh init state re-
    billed every LLM call while the log said "Resuming". ``_resuming`` (the
    pre-run checkpoint probe in propagate) must switch the streamed input to
    None; every other path keeps the init state."""
    captured: list = []

    def fake_invoke(state, args):
        captured.append(state)
        return {"final_trade_decision": "**Rating**: Buy"}

    # Resume path: None is the actual resume.
    g = _resumable_shell(tmp_path)
    g._invoke_or_stream = fake_invoke
    g._apply_risk_overlay = lambda c, d, fs, ps, asset_type="stock": fs
    final_state, _signal = g._run_graph(
        "BTCUSDT", _TODAY, asset_type="crypto_perp", _resuming=True
    )
    assert captured == [None]
    assert final_state["final_trade_decision"] == "**Rating**: Buy"

    # Non-resuming path (fresh run on a checkpoint-enabled graph): the init
    # state IS the input.
    captured.clear()
    g._run_graph("BTCUSDT", _TODAY, asset_type="crypto_perp")
    assert len(captured) == 1
    assert captured[0] == {"final_trade_decision": "**Rating**: Buy"}

    # _resuming=True but checkpointing DISABLED: still the init state (there
    # is no thread to resume; propagate only sets _resuming after it found a
    # checkpoint for the exact thread).
    captured.clear()
    g2 = _resumable_shell(tmp_path, checkpoint_enabled=False)
    g2._invoke_or_stream = fake_invoke
    g2._apply_risk_overlay = lambda c, d, fs, ps, asset_type="stock": fs
    g2._run_graph("BTCUSDT", _TODAY, asset_type="crypto_perp", _resuming=True)
    assert captured == [{"final_trade_decision": "**Rating**: Buy"}]
