"""Round-6 regression pins for the 2026-09-22 fix batch (USDT-perp sweep).

Each test pins one verified defect from the sixth adversarial sweep; see the
per-test docstrings for the failure mode that motivated it:

* perp ticket math: case-insensitive rating strength (the enforced-portfolio
  proxy passed "buy"/"sell" and leverage collapsed to 1.0x with no
  liquidation advisory), profile / atr_mult / bracket-max-leverage threading;
* trade_ticket / rank_signals: side validation — a SHORT carrying the
  overlay's LONG-form stop (below entry) must refuse, not invert risk;
* portfolio control: the sign-aware SHORT funding gate now applies to BOTH
  sizing paths (a positive legacy weight no longer bypasses the damp);
* backtest: the event-study benchmark leg uses the ORIGINAL yfinance provider
  (a perp provider silently resolved "SPY" into the SPYUSDT perp), and factor
  alpha annualizes on the JOINED calendar (a 7×24 asset joined onto equity
  factors is ~252 obs/yr regardless of its native 365);
* ledger: a reflection-LLM failure no longer kills the whole analysis run,
  same-day perp+spot batch updates both dispatch, space-separated datetimes
  are not "date_only", legacy vendor raises retire instead of re-raising
  every batch, and the ticket mirror retries once on a transient DB lock;
* llm_clients: pooled streaming asks for a usage block, timeout 0 is the
  documented explicit-disable, 429 cooldowns are not re-stamped while already
  cooling, has_pool parses (not truthiness), capabilities lookups are
  case-insensitive, cache replays are marked and counted separately;
* web entrance: an explicit Binance-venue submission files the CONTRACT
  symbol (BTCUSDT), not the Yahoo-spelled twin (BTC-USD).

Hermetic: every network / LLM / clock seam is monkeypatched or tmp-pathed.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from yialpha.dataflows import quality

_TODAY = date.today().isoformat()


def _load_script(name: str):
    """Import ``scripts/<name>.py`` under a private module name."""
    spec = importlib.util.spec_from_file_location(
        f"{name}_round6_under_test",
        Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
# 1. perp ticket math: rating strength + profile / atr_mult / bracket cap
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_rating_strength_is_case_insensitive():
    """The enforced-portfolio path forwards a side-derived lowercase proxy
    ("buy"/"sell"/"hold"); the strength table is title-case. A miss read 0,
    pinning L_conv to 0 → leverage 1.0x and no liquidation advisory on every
    enforced perp ticket."""
    from yialpha.risk.perp_ticket import perp_ticket_numbers, rating_strength

    assert rating_strength("buy") == 2
    assert rating_strength("SELL") == -2
    assert rating_strength(" Hold ") == 0
    assert rating_strength("Overweight") == 1
    assert rating_strength(None) == 0
    assert rating_strength("garbage") == 0

    # The enforced path's exact call shape: lowercase proxy rating.
    lev, detail, liq, _stop = perp_ticket_numbers(100.0, 2.0, "buy", None, 0.1)
    assert detail["L_conv"] == pytest.approx(5.0)  # conservative |Buy|
    assert lev > 1.0
    assert liq is not None  # L>1 ⇒ liquidation advisory exists
    # Before the fix: strength 0 → l_conv 0.0 → L == 1.0, liq None.
    lev_broken, detail_broken, liq_broken, _ = perp_ticket_numbers(
        100.0, 2.0, "Buy", None, 0.1, profile="nonsense_profile_falls_back"
    )
    assert lev_broken >= 1.0  # never below 1


@pytest.mark.unit
def test_perp_ticket_profile_kwarg_overrides_conviction_cap():
    from yialpha.risk.perp_ticket import perp_ticket_numbers

    _lev, detail_mod, _liq, _stop = perp_ticket_numbers(
        100.0, 2.0, "Buy", None, 0.1, profile="moderate"
    )
    assert detail_mod["L_conv"] == pytest.approx(10.0)
    _lev, detail_cons, _liq, _stop = perp_ticket_numbers(
        100.0, 2.0, "Buy", None, 0.1
    )
    assert detail_cons["L_conv"] == pytest.approx(5.0)


@pytest.mark.unit
def test_perp_ticket_atr_mult_threads_into_constructed_stop():
    """A configured 3·ATR run must derive the SHORT stop at entry+3·ATR —
    the hardcoded 2·ATR rendered advice a whole ATR away from the stop the
    manager actually placed."""
    from yialpha.risk.perp_ticket import perp_ticket_numbers

    _lev, _detail, _liq, stop3 = perp_ticket_numbers(
        100.0, 2.0, "Sell", None, -0.05, atr_mult=3.0
    )
    assert stop3 == pytest.approx(106.0)
    _lev, _detail, _liq, stop2 = perp_ticket_numbers(
        100.0, 2.0, "Sell", None, -0.05
    )
    assert stop2 == pytest.approx(104.0)


@pytest.mark.unit
def test_bracket_max_leverage_caps_compute_leverage():
    from yialpha.dataflows.binance_brackets import (
        Bracket,
        bracket_max_leverage_for,
    )
    from yialpha.risk.perp_ticket import advisory_leverage_cap, compute_leverage

    ladder = [
        Bracket(0.0, 0.004, max_leverage=50.0),
        Bracket(50_000.0, 0.005, max_leverage=5.0),
    ]
    # Tiering mirrors mmr_for_notional (>= floor, ascending).
    assert bracket_max_leverage_for(ladder, 10_000.0) == 50.0
    assert bracket_max_leverage_for(ladder, 200_000.0) == 5.0
    # The unsigned default ladder carries no max_leverage → None → no cap.
    assert bracket_max_leverage_for(
        [Bracket(0.0, 0.004)], 10_000.0
    ) is None

    # Buy on conservative would want min(14.6, 18.3, 5, 20) = 5 — the
    # bracket cap at the assumed 50k notional (5x) does not bind; a 3x tier
    # DOES bind and is disclosed.
    lev, caps = compute_leverage(0.068, 0.02, "crypto_perp", 2)
    assert lev == pytest.approx(5.0)
    assert "L_bracket" not in caps
    lev3, caps3 = compute_leverage(0.068, 0.02, "crypto_perp", 2, bracket_max_leverage=3.0)
    assert lev3 == pytest.approx(3.0)
    assert caps3["L_bracket"] == pytest.approx(3.0)

    # Fail-soft: no keys / transport failure → None → advisory unchanged.
    import yialpha.dataflows.binance_brackets as brackets_mod

    def _boom(symbol, **kwargs):
        raise RuntimeError("no keys")

    orig = brackets_mod.get_leverage_brackets
    brackets_mod.get_leverage_brackets = _boom
    try:
        assert advisory_leverage_cap("BTCUSDT") is None
    finally:
        brackets_mod.get_leverage_brackets = orig


@pytest.mark.unit
def test_signed_brackets_parse_initial_leverage(monkeypatch):
    """The signed leverageBracket response's initialLeverage must survive
    parsing — it was discarded, so no advisory ever capped against the
    exchange's own per-tier leverage ceiling."""
    import yialpha.dataflows.binance_brackets as brackets_mod

    def _fake_signed_get(path, params, symbol, canonical):
        assert path == "/fapi/v1/leverageBracket"
        return [
            {
                "symbol": canonical,
                "brackets": [
                    {
                        "bracket": 1, "initialLeverage": 50, "notionalFloor": 0,
                        "maintMarginRatio": 0.004, "cum": 0.0,
                    },
                    {
                        "bracket": 2, "initialLeverage": 25, "notionalFloor": 50000,
                        "maintMarginRatio": 0.005, "cum": 250.0,
                    },
                ],
            }
        ]

    monkeypatch.setattr(brackets_mod, "signed_fapi_get", _fake_signed_get)
    out = brackets_mod._signed_leverage_brackets("BTCUSDT", "BTCUSDT")
    assert [b.max_leverage for b in out] == [50.0, 25.0]
    assert [b.mmr for b in out] == [0.004, 0.005]


# --------------------------------------------------------------------------- #
# 2. trade_ticket / rank_signals side validation
# --------------------------------------------------------------------------- #
TRADER_MD_B2 = (
    "## Trader Plan\n\n"
    "**Action**: Sell\n\n"
    "**Entry Price**: $64,200\n"
    "**Stop Loss**: $66,300\n"
    "FINAL TRANSACTION PROPOSAL: **SELL**\n"
)
MARKET_MD_B2 = (
    "## BTCUSDT Market Snapshot\n\n"
    "Current Price: $64,000\n"
    "ATR(14): $1,050\n"
    "Support: 62,000\n"
    "Resistance: 66,000\n"
)
#: LONG-form overlay stop (below entry) on a report whose rating says SHORT.
PERP_OVERLAY_B2 = (
    "### Quantitative Risk Overlay\n"
    "- **Action**: enter\n"
    "- **Target Weight**: 8.0% (960)\n"
    "- **Stop Loss**: 62,000.0\n"
    "- **Entry Reference**: 64,000.0\n"
    "- **Drawdown Regime**: normal\n"
    "- **Suggested Leverage**: ≤ 4.0x\n"
    "- **Est. Liquidation Price**: 52,000.0\n"
    "- **Funding (7d)**: +0.021% (longs pay)\n"
)


def _write_report_b2(root: Path, decision_md: str, trader_md: str) -> Path:
    report_dir = root / "BTCUSDT_20260922_120000"
    (report_dir / "3_trading").mkdir(parents=True)
    (report_dir / "5_portfolio").mkdir(parents=True)
    (report_dir / "1_analysts").mkdir(parents=True)
    (report_dir / "3_trading" / "trader.md").write_text(trader_md, encoding="utf-8")
    (report_dir / "5_portfolio" / "decision.md").write_text(decision_md, encoding="utf-8")
    (report_dir / "1_analysts" / "market.md").write_text(MARKET_MD_B2, encoding="utf-8")
    return report_dir


@pytest.mark.unit
def test_short_ticket_with_long_form_stop_refuses(tmp_path):
    """markdown rating Underweight (→ short) + overlay stop BELOW entry: the
    old build_ticket let abs() swallow the side and issued a ticket whose
    stop sat in the profit direction (never triggering). Must degrade
    honestly with a reason."""
    tt = _load_script("trade_ticket")
    decision = "**Rating**: Underweight\n\nDivergent view.\n\n" + PERP_OVERLAY_B2
    report_dir = _write_report_b2(tmp_path, decision, TRADER_MD_B2)

    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")
    assert t["status"] == "degenerate_levels"
    assert "wrong side" in t.get("status_reason", "")
    assert t["direction"] == "short"


@pytest.mark.unit
def test_long_ticket_with_stop_above_entry_refuses(tmp_path):
    """Mirror: a long ticket whose only available stop sits ABOVE entry is
    the same inversion; refuse rather than size off a stop in the loss
    direction masquerading as risk."""
    tt = _load_script("trade_ticket")
    # No trader stop at all → resolve_levels constructs from ATR only when
    # stop is None; give the OVERLAY a stop above entry instead.
    overlay = PERP_OVERLAY_B2.replace("62,000.0", "66,900.0")
    decision = "**Rating**: Buy\n\nMomentum.\n\n" + overlay
    report_dir = _write_report_b2(tmp_path, decision, TRADER_MD_B2)
    t = tt.build_ticket(report_dir, capital=10_000.0, profile="moderate")
    assert t["status"] == "degenerate_levels"


@pytest.mark.unit
def test_rank_signals_rr_none_on_inverted_stop(tmp_path, monkeypatch):
    """rank_signals' conviction rr leg must not divide by a stop on the
    wrong side either — rr stays None instead of ranking off inverted risk."""
    rs = _load_script("rank_signals")
    decision = "**Rating**: Underweight\n\nDivergent view.\n\n" + PERP_OVERLAY_B2
    report_dir = _write_report_b2(tmp_path, decision, TRADER_MD_B2)
    monkeypatch.setattr(
        rs, "latest_for_ticker", lambda ticker: report_dir
    )
    row = rs.analyze_ticker("BTCUSDT")
    assert row["status"] == "ok"
    assert row["direction"] == "short"
    # resolve_target validated the target side long ago; with the stop guard
    # the rr is None (target may also be absent in this synthetic report).
    assert row.get("rr") is None
    # Control: same report with the stop ABOVE entry (short-correct side)
    # still computes an rr when a valid target exists.
    overlay_ok = PERP_OVERLAY_B2.replace("62,000.0", "66,000.0")
    decision_ok = (
        "**Rating**: Underweight\n\nDivergent view.\n\n" + overlay_ok
        + "\n**Price Target**: $60,000\n"
    )
    rd2 = _write_report_b2(tmp_path / "ok", decision_ok, TRADER_MD_B2)
    monkeypatch.setattr(rs, "latest_for_ticker", lambda ticker: rd2)
    row2 = rs.analyze_ticker("BTCUSDT")
    assert row2["status"] == "ok"
    assert row2.get("rr") is not None


# --------------------------------------------------------------------------- #
# 3. portfolio control: funding gate on the legacy-positive SHORT path
# --------------------------------------------------------------------------- #
def _graph_shell(tmp_path, **config_over):
    from yialpha.graph.trading_graph import YiAlphaGraph

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


def _stub_overlay_prices(g, monkeypatch, funding_annualized):
    monkeypatch.setattr(
        g, "_latest_close_and_atr", lambda t, d, at="stock": (60.0, 1.5)
    )
    monkeypatch.setattr(
        g,
        "_trailing_funding_total",
        lambda t, d: None
        if funding_annualized is None
        else funding_annualized * 7.0 / 365.0,
    )


def _bind_run(run_id: str) -> None:
    from yialpha.ledger.evidence import register_run
    from yialpha.ledger.run_context import set_ledger_run_context

    register_run(run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY)
    set_ledger_run_context(
        run_id, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _TODAY
    )


def _run_shadow_divergent_short(g, monkeypatch, funding_annualized, run_id):
    """Structured pm_rating=Buy (legacy weight > 0) + explicit desired_side
    SHORT — the divergence scenario whose positive legacy weight used to
    bypass the sign-aware funding gate entirely."""
    from yialpha.ledger.run_context import reset_ledger_run_context

    _stub_overlay_prices(g, monkeypatch, funding_annualized)
    _bind_run(run_id)
    try:
        return g._apply_risk_overlay(
            "BTCUSDT", _TODAY,
            {
                "final_trade_decision": "**Rating**: Buy\n\nDivergent.",
                "pm_rating": "Buy",
                "pm_decision_fields": {"desired_side": "SHORT"},
            },
            {"equity": 100_000},
            asset_type="crypto_perp",
        )
    finally:
        reset_ledger_run_context()


@pytest.mark.unit
def test_legacy_positive_short_damped_on_adverse_carry(tmp_path, monkeypatch):
    g = _graph_shell(tmp_path, portfolio_control_mode="shadow")
    out = _run_shadow_divergent_short(g, monkeypatch, -0.72, "R6SG1")
    record = out["portfolio_control_shadow"]
    assert record["side"] == "SHORT"
    assert record["short_sizing"] == "legacy_funding_damped"
    assert record["short_funding_damp"] == 0.25
    # The legacy Buy weight (band ceiling 0.08 in this shell) quartered.
    assert record["proposed_size"] == pytest.approx(0.08 * 0.25)
    assert record["short_funding_annualized"] == pytest.approx(-0.72)
    assert "Funding gate" in out["final_trade_decision"]


@pytest.mark.unit
def test_legacy_positive_short_undamped_on_favourable_carry(tmp_path, monkeypatch):
    g = _graph_shell(tmp_path, portfolio_control_mode="shadow")
    out = _run_shadow_divergent_short(g, monkeypatch, 0.40, "R6SG2")
    record = out["portfolio_control_shadow"]
    assert record["short_sizing"] == "legacy_weight"
    assert record["short_funding_damp"] == 1.0
    # +0.40 carry is FAVOURABLE to the short, but the legacy weight was
    # produced by the risk manager's LONG-side gate (funding >= warn halves
    # a long) — 0.08 → 0.04 upstream; the SHORT gate itself never damps.
    assert record["proposed_size"] == pytest.approx(0.04)
    assert "Funding gate" not in out["final_trade_decision"]


# --------------------------------------------------------------------------- #
# 4. backtest: event-study bench provider + factor annualization
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_event_study_benchmark_uses_index_provider(monkeypatch):
    """The perp price provider must NOT serve the benchmark leg: it resolves
    "SPY" into the SPYUSDT tokenized perp. The bench leg goes through the
    ORIGINAL (yfinance) provider captured before the perp swap."""
    from types import SimpleNamespace

    import yialpha.backtest.event_study as es_mod
    from yialpha.backtest.engine import TradeRow, _run_event_study

    calls: dict[str, list] = {}

    def asset_provider(symbol, start, end):
        calls.setdefault("asset", []).append(symbol)
        idx = pd.date_range("2025-01-01", periods=300, freq="D")
        return pd.Series([100.0 + i for i in range(300)], index=idx)

    def bench_provider(symbol, start, end):
        calls.setdefault("bench", []).append(symbol)
        idx = pd.date_range("2025-01-01", periods=300, freq="D")
        return pd.Series([500.0 + i * 0.5 for i in range(300)], index=idx)

    captured: dict[str, object] = {}

    def fake_event_study(asset_prices, benchmark_prices, event_dates, **kw):
        captured["asset_last"] = float(asset_prices.iloc[-1])
        captured["bench_last"] = float(benchmark_prices.iloc[-1])
        captured["bench_name"] = kw.get("benchmark_name")
        return SimpleNamespace(n_events=0)

    monkeypatch.setattr(es_mod, "event_study", fake_event_study)

    trades = [
        TradeRow(date="2025-08-01", rating="Buy", target_weight=0.1,
                 executed_weight=0.1, price=100.0, execution_date="2025-08-02"),
    ]
    metrics = SimpleNamespace()
    _run_event_study(
        metrics, None, "BTCUSDT", asset_provider, trades,
        index_name="SPY", end_date="2025-12-01",
        bench_provider=bench_provider,
    )
    # The bench leg was served by the DEDICATED bench provider (its series),
    # never by the perp provider.
    assert calls == {"asset": ["BTCUSDT"], "bench": ["SPY"]}
    assert captured["bench_last"] == pytest.approx(500.0 + 299 * 0.5)
    assert captured["bench_name"] == "SPY"

    # Default (no bench_provider): both legs share the injected provider —
    # the test-injection contract is unchanged.
    calls.clear()
    captured.clear()
    _run_event_study(
        metrics, None, "BTCUSDT", asset_provider, trades,
        index_name="SPY", end_date="2025-12-01",
    )
    assert calls == {"asset": ["BTCUSDT", "SPY"]}


@pytest.mark.unit
def test_factor_alpha_annualizes_on_joined_calendar():
    """A 7×24 crypto strategy inner-joined onto equity-calendar FF factors
    has ~252 joined obs/year; annualizing the intercept with the asset's 365
    overstated factor alpha by ~45%."""
    import numpy as np

    from yialpha.backtest.factor_model import factor_attribution

    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    strategy = pd.Series(0.001, index=days)  # flat 0.1%/day on the 365 grid
    weekdays = days[days.weekday < 5]
    factors = pd.DataFrame(
        {
            "Mkt-RF": np.zeros(len(weekdays)),
            "SMB": np.zeros(len(weekdays)),
            "HML": np.zeros(len(weekdays)),
            "RF": np.zeros(len(weekdays)),
        },
        index=weekdays,
    )
    res = factor_attribution(strategy, factors, periods_per_year=365, model="FF3")
    assert res is not None
    span_years = (weekdays[-1] - weekdays[0]).days / 365.25
    joined_ppy = len(weekdays) / span_years
    # The join is weekday-only (≈262/yr for this holiday-free synthetic) —
    # materially below the asset's native 365 the old code annualized with.
    assert joined_ppy < 300.0
    # alpha_per == mean excess == 0.001 → annualized on the JOINED frequency.
    assert res.alpha_annual == pytest.approx(0.001 * joined_ppy, rel=1e-6)
    # The old behaviour (× 365) is materially different — pin the gap.
    assert res.alpha_annual < 0.001 * 365.0 * 0.99


# --------------------------------------------------------------------------- #
# 5. web entrance: venue-canonical symbol for explicit crypto submissions
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_web_analyze_perp_submission_files_contract_symbol(monkeypatch):
    """A web-submitted BTCUSDT perp run must spawn run_robust with the
    CONTRACT symbol — the Yahoo twin (BTC-USD) split the same contract into
    two ledger identities across entrances."""
    from fastapi.testclient import TestClient

    from web import app as web_app

    spawned: dict[str, object] = {}

    class _FakeState:
        ticker = "BTCUSDT"
        date = "2026-09-22"
        asset_type = "crypto_perp"
        started_at = 0.0
        status = "running"

    class _FakeRegistry:
        def is_busy(self):
            return False

        def get(self, task_id):
            return _FakeState()

    async def _fake_spawn(ticker, date, asset_type, language="en"):
        spawned.update(
            ticker=ticker, date=date, asset_type=asset_type, language=language
        )
        return "task-1"

    monkeypatch.setattr(web_app.runner, "registry", _FakeRegistry())
    monkeypatch.setattr(web_app.runner, "spawn", _fake_spawn)

    client = TestClient(web_app.app)
    resp = client.post(
        "/api/analyze",
        json={"ticker": "BTCUSDT", "date": "2026-09-22", "asset_type": "crypto_perp"},
    )
    assert resp.status_code == 200
    assert spawned["ticker"] == "BTCUSDT"
    assert spawned["asset_type"] == "crypto_perp"

    # Stock submissions keep the Yahoo canon (unchanged path).
    client.post(
        "/api/analyze",
        json={"ticker": "BTCUSD", "date": "2026-09-22", "asset_type": "stock"},
    )
    assert spawned["ticker"] == "BTC-USD"


@pytest.mark.unit
def test_web_analyze_rejects_delivery_contract_symbol(monkeypatch):
    """A dated delivery contract on the perp entrance is refused loudly by
    normalize_symbol_for_venue → 400, not a fabricated symbol."""
    from fastapi.testclient import TestClient

    from web import app as web_app

    class _FakeRegistry:
        def is_busy(self):
            return False

    async def _fake_spawn(*args, **kwargs):
        raise AssertionError("spawn must not be reached")

    monkeypatch.setattr(web_app.runner, "registry", _FakeRegistry())
    monkeypatch.setattr(web_app.runner, "spawn", _fake_spawn)
    client = TestClient(web_app.app)
    resp = client.post(
        "/api/analyze",
        json={"ticker": "BTCUSDT_260327", "date": "2026-09-22", "asset_type": "crypto_perp"},
    )
    assert resp.status_code == 400
    assert "delivery" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# 6. ledger: memory resolution / batch dispatch / timestamps / mirror retry
# --------------------------------------------------------------------------- #
class _FakeReflector:
    def __init__(self, fail_on: int | None = None):
        self.calls = 0
        self.fail_on = fail_on

    def reflect_on_final_decision(self, **kwargs):
        self.calls += 1
        if self.fail_on is not None and self.calls == self.fail_on:
            raise RuntimeError("LLM 429")
        return "lesson"


def _seed_pending(log, ticker, d, asset=None):
    log.store_decision(ticker, d, "Decision text.", asset_type=asset)


@pytest.mark.unit
def test_memory_resolution_survives_reflection_llm_failure(tmp_path, monkeypatch):
    """One reflection LLM failure must leave the entry pending and resolve
    the others — it used to propagate and kill the whole analysis run
    before any node executed."""
    import yialpha.accuracy as accuracy_mod
    from yialpha.agents.utils.memory import TradingMemoryLog
    from yialpha.graph.memory_resolution import resolve_pending_entries

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    old = (date.today() - timedelta(days=10)).isoformat()
    _seed_pending(log, "NVDA", old)
    _seed_pending(log, "NVDA", (date.today() - timedelta(days=9)).isoformat())

    monkeypatch.setattr(
        accuracy_mod, "fetch_returns_yf",
        lambda *a, **k: (0.05, 0.02, 5),
    )
    reflector = _FakeReflector(fail_on=1)
    resolved = resolve_pending_entries(log, reflector, "NVDA", as_of_date=_TODAY)
    # Entry 1's price resolved but its reflection raised → left pending;
    # entry 2 resolved fully. The sweep itself returned, did not raise.
    assert resolved == 1
    assert reflector.calls == 2
    pending = [e for e in log.get_pending_entries() if e["ticker"] == "NVDA"]
    assert len(pending) == 1


@pytest.mark.unit
def test_batch_update_dispatches_both_same_day_venues(tmp_path):
    """One sweep carrying updates for the same-day perp AND spot entries of
    one ticker must resolve BOTH — the (date, ticker)-keyed dispatch map
    collapsed the pair and stranded one venue for an extra sweep."""
    from yialpha.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    d = (date.today() - timedelta(days=10)).isoformat()
    _seed_pending(log, "BTCUSDT", d, asset="crypto_perp")
    _seed_pending(log, "BTCUSDT", d, asset="crypto_spot")

    log.batch_update_with_outcomes([
        {
            "ticker": "BTCUSDT", "trade_date": d, "raw_return": 0.05,
            "alpha_return": 0.02, "holding_days": 5, "reflection": "perp lesson",
            "asset_type": "crypto_perp",
        },
        {
            "ticker": "BTCUSDT", "trade_date": d, "raw_return": -0.03,
            "alpha_return": -0.01, "holding_days": 5, "reflection": "spot lesson",
            "asset_type": "crypto_spot",
        },
    ])

    text = (tmp_path / "mem.md").read_text(encoding="utf-8")
    assert "perp lesson" in text and "spot lesson" in text
    assert "asset=crypto_perp" in text and "asset=crypto_spot" in text
    assert not [e for e in log.get_pending_entries() if e["ticker"] == "BTCUSDT"]


@pytest.mark.unit
def test_batch_update_legacy_none_update_prefers_untagged_block(tmp_path):
    """A None-asset update lands on the UNTAGGED block in the exact pass,
    not on a venue-tagged twin that shares the (date, ticker) — the legacy
    fallback pass still exists for logs with no untagged block."""
    from yialpha.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    d = (date.today() - timedelta(days=10)).isoformat()
    _seed_pending(log, "BTCUSDT", d)  # untagged legacy entry
    _seed_pending(log, "BTCUSDT", d, asset="crypto_perp")

    log.batch_update_with_outcomes([
        {
            "ticker": "BTCUSDT", "trade_date": d, "raw_return": 0.05,
            "alpha_return": 0.02, "holding_days": 5,
            "reflection": "legacy lesson", "asset_type": None,
        },
        {
            "ticker": "BTCUSDT", "trade_date": d, "raw_return": 0.07,
            "alpha_return": 0.01, "holding_days": 5,
            "reflection": "perp lesson", "asset_type": "crypto_perp",
        },
    ])
    text = (tmp_path / "mem.md").read_text(encoding="utf-8")
    assert "legacy lesson" in text and "perp lesson" in text
    # The legacy lesson resolved the untagged entry (venue tag preserved on
    # the perp twin, not stamped onto the legacy block).
    assert "asset=crypto_perp" in text


@pytest.mark.unit
def test_parse_timestamp_space_separated_datetime_is_not_date_only():
    """A space-separated ISO datetime is an intraday instant — the "t" not
    in ts sniffing called it date_only and bypassed the intraday
    knowability gate."""
    from yialpha.ledger.models import parse_timestamp

    _dt, date_only = parse_timestamp("2026-09-22")
    assert date_only is True
    _dt, date_only = parse_timestamp("2026-09-22T10:00:00+00:00")
    assert date_only is False
    _dt, date_only = parse_timestamp("2026-09-22 10:00:00")
    assert date_only is False
    # An explicit 00:00:00 midnight is component-indistinguishable from a
    # pure date (and names the same instant as the daily bar-close moment)
    # — reading it date_only is safe by construction, not asserted against.


@pytest.mark.unit
def test_attach_ticket_retries_once_on_transient_failure(tmp_path, monkeypatch):
    """A transient SQLITE_BUSY at run end used to silently strand the whole
    run unscoreable (outcome's ticket_cost leg dies after grace). One
    short-backoff retry, then honest give-up."""
    import yialpha.ledger.sqlite as sqlite_mod
    import yialpha.ledger.tickets_mirror as mirror_mod

    # Point the ledger at a tmp DB and create the schema.
    monkeypatch.setenv("YIALPHA_LEDGER_DB", str(tmp_path / "ledger.db"))
    sqlite_mod.get_connection.cache_clear() if hasattr(
        sqlite_mod.get_connection, "cache_clear"
    ) else None
    conn = sqlite_mod.get_connection()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tickets ("
        "ticket_id TEXT PRIMARY KEY, decision_id TEXT, run_id TEXT, "
        "payload TEXT, ticket_version TEXT, written_at TEXT)"
    )
    conn.commit()

    attempts = {"n": 0}

    def _flaky_transaction():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("database is locked")
        ctx = sqlite_mod.ledger_transaction()
        return ctx

    real_transaction = sqlite_mod.ledger_transaction
    monkeypatch.setattr(mirror_mod, "ledger_transaction", _flaky_transaction)

    monkeypatch.setattr("time.sleep", lambda s: None)
    mirror_mod.attach_ticket(
        "T1", "D1", "R1", {"lev": 5.0}, "v2.4"
    )
    assert attempts["n"] == 2
    row = conn.execute("SELECT ticket_id FROM tickets").fetchone()
    assert row is not None and row[0] == "T1"
    del real_transaction


# --------------------------------------------------------------------------- #
# 7. llm_clients: pooled streaming usage / timeout 0 / cooldown / lookup
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pool_stream_usage_default_on_env_off(monkeypatch):
    from yialpha.llm_clients.openai_client import _pool_stream_usage_enabled

    assert _pool_stream_usage_enabled() is True
    monkeypatch.setenv("YIALPHA_LLM_POOL_STREAM_USAGE", "false")
    assert _pool_stream_usage_enabled() is False


@pytest.mark.unit
def test_has_pool_parses_not_truthiness(monkeypatch):
    """A pool env of " , " is NOT a configured pool: the raw-strip check
    engaged the pool path with zero keys and the error message pointed at
    the single-key env var."""
    from yialpha.llm_clients import key_pool

    monkeypatch.delenv("ZHIPU_CN_API_KEYS", raising=False)
    monkeypatch.delenv("ZHIPU_CN_API_KEY", raising=False)
    assert key_pool.has_pool("glm-cn") is False
    monkeypatch.setenv("ZHIPU_CN_API_KEYS", " , ")
    assert key_pool.has_pool("glm-cn") is False
    assert key_pool.get_pool_http_client("glm-cn") is None
    monkeypatch.setenv("ZHIPU_CN_API_KEYS", "k1")
    assert key_pool.has_pool("glm-cn") is True


@pytest.mark.unit
def test_pool_cooldown_deadline_not_restamped_while_cooling():
    from yialpha.llm_clients.key_pool import _PoolState

    state = _PoolState(2, 120.0)
    first = state.mark(0, 429)
    assert "cooling down" in first
    again = state.mark(0, 429)  # a forced retry hitting the same cooling key
    assert "still cooling" in again
    # The deadline did not move: force_pick revival stays deterministic.
    deadline_after_first = state.dead_until[0]
    state.mark(0, 429)
    assert state.dead_until[0] == deadline_after_first


@pytest.mark.unit
def test_capabilities_lookup_is_case_insensitive():
    from yialpha.llm_clients.capabilities import (
        _DEEPSEEK_THINKING,
        _MINIMAX_THINKING,
        get_capabilities,
    )

    assert get_capabilities("DeepSeek-V4-Pro") is _DEEPSEEK_THINKING
    assert get_capabilities("DEEPSEEK-REASONER") is _DEEPSEEK_THINKING
    assert get_capabilities("minimax-m2.7") is _MINIMAX_THINKING
    assert get_capabilities("MiniMax-M2.7") is _MINIMAX_THINKING


@pytest.mark.unit
def test_stats_handler_splits_replayed_usage():
    """Cache-replayed usage belongs to the ORIGINAL call: a robust retry
    run must not book attempt 1's cached tokens as its own spend."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from yialpha.cli.stats_handler import StatsCallbackHandler

    handler = StatsCallbackHandler()

    def _response(replay: bool):
        msg = AIMessage(
            content="x",
            usage_metadata={
                "input_tokens": 10, "output_tokens": 20, "total_tokens": 30
            },
        )
        gen = ChatGeneration(
            message=msg,
            generation_info={"cached_replay": True} if replay else None,
        )
        return LLMResult(generations=[[gen]])

    # Real callback order: on_chat_model_start fires ABOVE the cache layer,
    # then the (possibly replayed) end — llm_calls must stay LIVE-only.
    handler.on_chat_model_start({"id": ["m"]}, [[AIMessage(content="hi")]])
    handler.on_llm_end(_response(replay=False))
    handler.on_chat_model_start({"id": ["m"]}, [[AIMessage(content="hi")]])
    handler.on_llm_end(_response(replay=True))
    stats = handler.get_stats()
    assert stats["llm_calls"] == 1  # the replay's start count was taken back
    assert stats["llm_calls_replayed"] == 1
    assert stats["tokens_in"] == 10
    assert stats["tokens_out"] == 20
    assert stats["tokens_in_replayed"] == 10
    assert stats["tokens_out_replayed"] == 20


@pytest.mark.unit
def test_disk_cache_lookup_stamps_cached_replay(tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration

    from yialpha.llm_clients.response_cache import DiskLLMCache

    cache = DiskLLMCache(tmp_path, enabled=True)
    cache.update(
        "p", "llm",
        [ChatGeneration(message=AIMessage(content="hi"))],
    )
    gens = cache.lookup("p", "llm")
    assert gens is not None
    assert gens[0].generation_info is not None
    assert gens[0].generation_info.get("cached_replay") is True
    # The marker must not leak into the persisted payload.
    payload = json.loads(
        next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    )
    assert "cached_replay" not in json.dumps(payload)


@pytest.mark.unit
def test_shared_http_clients_follow_redirects(monkeypatch):
    """Both custom transports (keepalive singleton + key-pool client) must
    set follow_redirects=True, matching the openai SDK's own default client
    — a 30x-relaying base_url otherwise turns into an APIStatusError only
    on the custom-client paths."""
    from yialpha.llm_clients import http_client as hc, key_pool as kp

    # Both transports are process-wide singletons. Start from none and drop
    # the ones built here again on the way out: the pooled client was built
    # against the test key above, and its proxy configuration is read once.
    hc.reset_for_test()
    kp.reset_for_test()
    try:
        client = hc.get_shared_http_client()
        assert client is not None and client.follow_redirects is True

        monkeypatch.setenv("ZHIPU_CN_API_KEYS", "test-key-1")
        monkeypatch.delenv("ZHIPU_CN_API_KEY", raising=False)
        pooled = kp.get_pool_http_client("glm-cn")
        assert pooled is not None and pooled.follow_redirects is True
    finally:
        hc.reset_for_test()
        kp.reset_for_test()


@pytest.mark.unit
def test_preflight_probe_backend_url_wins_over_provider_env(monkeypatch):
    """The probe must resolve the endpoint with the SAME precedence as the
    client: YIALPHA_LLM_BACKEND_URL first, then the provider's env, then the
    spec default. (Review P2: the first cut had the first two swapped, so an
    operator setting both probed a different host than chat traffic uses.)"""
    from yialpha.monitoring.preflight import _resolve_probe_base

    monkeypatch.setenv("ZHIPU_CN_BASE_URL", "https://provider-env.example/v4/")
    monkeypatch.setenv("YIALPHA_LLM_BACKEND_URL", "https://backend-url.example/v4/")
    assert _resolve_probe_base("glm-cn") == "https://backend-url.example/v4/"

    monkeypatch.delenv("YIALPHA_LLM_BACKEND_URL")
    assert _resolve_probe_base("glm-cn") == "https://provider-env.example/v4/"

    monkeypatch.delenv("ZHIPU_CN_BASE_URL")
    assert _resolve_probe_base("glm-cn") == (
        "https://open.bigmodel.cn/api/paas/v4/"
    )
    assert _resolve_probe_base("not-a-provider") == ""


# --------------------------------------------------------------------------- #
# 8. constraints: post-candidate book keyed by (symbol, instrument_class)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_post_positions_keeps_other_venue_same_symbol():
    from yialpha.risk.constraints import _post_positions

    class _Pos:
        def __init__(self, symbol, instrument_class):
            self.symbol = symbol
            self.instrument_class = instrument_class

    class _Snap:
        positions = (
            _Pos("BTCUSDT", "crypto_perp"),
            _Pos("BTCUSDT", "crypto_spot"),
            _Pos("ETHUSDT", "crypto_perp"),
        )

    candidate = _Pos("BTCUSDT", "crypto_perp")
    book = _post_positions(candidate, _Snap)
    symbols = sorted(
        (p.symbol, p.instrument_class) for p in book
    )
    # The perp row is replaced by the candidate; the spot twin survives.
    assert symbols == [
        ("BTCUSDT", "crypto_perp"),
        ("BTCUSDT", "crypto_spot"),
        ("ETHUSDT", "crypto_perp"),
    ]
