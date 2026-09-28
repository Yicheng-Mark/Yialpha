"""Round-8 known-issue landing pins (backtest / graph / cli / transport).

One file for the smaller landed fixes that each needed a behavior pin:
DecisionCache v5 config fingerprint, the mark_end semantic-window extension,
the checkpoint signature's instrument_registry key, positioning-report
persistence, config-check's unknown-provider gate, run_robust's clean-retry
DEGRADED clearing, credential redaction shapes, the accuracy forming-bar
seam, and the legacy outcome legs' degradation paths. Hermetic throughout.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from yialpha.backtest.cache import DecisionCache
from yialpha.backtest.engine import run_backtest
from yialpha.graph.trading_graph import YiAlphaGraph
from yialpha.logging_config import redact_secrets
from yialpha.reporting import write_report_tree

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# S5-b: DecisionCache v5 config-signature keys
# ---------------------------------------------------------------------------


class TestDecisionCacheConfigSignature:
    def test_same_signature_roundtrip_hit(self, tmp_path):
        cache = DecisionCache(tmp_path, enabled=True)
        cache.remember(
            "AAPL", "2026-01-10", "Buy", "**Rating**: Buy",
            run_tag="r1", config_signature="sig-A",
        )
        hit = cache.get("AAPL", "2026-01-10", "r1", config_signature="sig-A")
        assert hit is not None and hit.rating == "Buy"

    def test_different_signature_is_miss_and_keeps_the_other_entry(self, tmp_path):
        cache = DecisionCache(tmp_path, enabled=True)
        cache.remember(
            "AAPL", "2026-01-10", "Buy", "d",
            run_tag="r1", config_signature="sig-A",
        )
        assert cache.get(
            "AAPL", "2026-01-10", "r1", config_signature="sig-B") is None
        # The miss must not have deleted the differently-signed entry.
        assert cache.get(
            "AAPL", "2026-01-10", "r1", config_signature="sig-A") is not None

    def test_payload_signature_mismatch_is_miss(self, tmp_path):
        cache = DecisionCache(tmp_path, enabled=True)
        path = cache._key_path("AAPL", "2026-01-10", "r1", "stock", "sig-A")
        payload = {
            "schema_version": 5, "ticker": "AAPL", "date": "2026-01-10",
            "run_tag": "r1", "rating": "Buy", "final_decision": "d",
            "asset_type": "stock", "was_degraded": False,
            "config_signature": "sig-OTHER",  # colliding filename, other sig
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert cache.get(
            "AAPL", "2026-01-10", "r1", "stock",
            config_signature="sig-A") is None

    def test_v4_legacy_layout_is_swept_on_miss(self, tmp_path):
        cache = DecisionCache(tmp_path, enabled=True)
        v4_name = cache.cache_dir / (
            f"{cache._key_path('AAPL', '2026-01-10', 'r1', 'stock').name}"
        )
        # Build the exact 4-segment v4 name (same components, no signature).
        from yialpha.backtest.cache import _safe_component

        v4 = cache.cache_dir / (
            f"{_safe_component('AAPL')}_{_safe_component('2026-01-10')}_"
            f"{_safe_component('r1')}_{_safe_component('stock')}.json"
        )
        v4.write_text("{}", encoding="utf-8")
        del v4_name  # the derived 5-segment path above is NOT the v4 name
        assert cache.get("AAPL", "2026-01-10", "r1", "stock") is None
        assert not v4.exists()  # swept so it can never shadow the v5 entry


# ---------------------------------------------------------------------------
# wave-2 leftover: mark_end extends to the last episode's semantic window
# ---------------------------------------------------------------------------


class _BuyGraph:
    """Always-Buy decision graph with a resolvable benchmark."""

    calls = 0

    def propagate(self, company_name, trade_date, asset_type="stock"):
        type(self).calls += 1
        return {"final_trade_decision": "**Rating**: Buy\n\nanalysis"}, "Buy"

    def _resolve_benchmark(self, ticker):
        return "SPY"


def _holiday_gap_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Weekday index with four post-decision holidays punched out, so 25
    trading sessions span MORE than the legacy h+10 = 35 calendar days."""
    idx = pd.bdate_range(start, end)
    drop = pd.to_datetime([
        "2025-11-13", "2025-11-20", "2025-12-01", "2025-12-15",
    ])
    idx = idx[~idx.isin(drop)]
    values = [100.0 * (1 + 0.002 * i) for i in range(len(idx))]
    return pd.Series(values, index=idx.strftime("%Y-%m-%d"), dtype=float)


class TestMarkEndSemanticWindow:
    def test_final_episode_covers_the_full_forward_window(self):
        dates = ["2025-11-03"]
        result = run_backtest(
            _BuyGraph(), "AAPL", dates, holding_days=25,
            price_provider=_holiday_gap_prices,
        )
        last = result.trades[-1]
        # The KI's complaint: position_return was marked on the truncated
        # legacy boundary while raw_return used the full forward window —
        # after the extend-only fix both cover the same exec..exec+25 span.
        assert last.raw_return is not None
        assert last.position_return == pytest.approx(last.raw_return, abs=1e-9)

    def test_small_holding_days_still_complete(self):
        # h=5: the legacy h+10 floor already covers the forward window, so
        # mark_end keeps the legacy boundary (byte-identity vs pre-round-8
        # is pinned in test_round8_wave2_regressions). The marking window
        # legitimately extends past the trade's own horizon, so
        # position_return covers MORE than raw_return here — assert the run
        # completes and marks on the legacy boundary only.
        result = run_backtest(
            _BuyGraph(), "AAPL", ["2025-11-03"], holding_days=5,
            price_provider=_holiday_gap_prices,
        )
        assert result.trades[-1].position_return is not None
        assert result.trades[-1].raw_return is not None


# ---------------------------------------------------------------------------
# S6-c: instrument_registry joins the checkpoint run signature
# ---------------------------------------------------------------------------


def _graph_shell(instrument_registry: bool) -> YiAlphaGraph:
    g = object.__new__(YiAlphaGraph)
    g.selected_analysts = ["market", "news"]
    g.config = {
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "analyst_parallel": False,
        "instrument_registry": instrument_registry,
    }
    return g


class TestCheckpointSignatureInstrumentRegistry:
    def test_flag_flip_changes_the_signature(self):
        off = _graph_shell(False)._run_signature("crypto_perp")
        on = _graph_shell(True)._run_signature("crypto_perp")
        assert off != on
        assert "instrument_registry" in off  # visibly folded into config=


# ---------------------------------------------------------------------------
# S6-a: positioning_report persistence (log key + report tree)
# ---------------------------------------------------------------------------


class TestPositioningPersistence:
    def test_report_tree_writes_positioning_md(self, tmp_path):
        state = {
            "market_report": "market text",
            "positioning_report": "positioning text",
        }
        write_report_tree(state, "BTCUSDT", tmp_path)
        assert (tmp_path / "1_analysts" / "positioning.md").read_text(
            encoding="utf-8") == "positioning text"
        assert "Positioning Analyst" in (
            tmp_path / "complete_report.md").read_text(encoding="utf-8")

    def test_report_tree_without_positioning_writes_no_file(self, tmp_path):
        write_report_tree({"market_report": "m"}, "BTCUSDT", tmp_path)
        assert not (tmp_path / "1_analysts" / "positioning.md").exists()

    def test_log_state_entry_carries_positioning(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            YiAlphaGraph, "_latest_close_and_atr",
            lambda self, t, d, a: (None, None),
        )
        g = object.__new__(YiAlphaGraph)
        g.ticker = "BTCUSDT"
        g.config = {"results_dir": str(tmp_path)}
        debate = {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "",
        }
        risk = {
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "history": "", "judge_decision": "",
        }
        final_state = {
            "company_of_interest": "BTCUSDT",
            "trade_date": "2026-01-10",
            "market_report": "m", "sentiment_report": "s",
            "news_report": "n", "fundamentals_report": "f",
            "investment_debate_state": debate,
            "trader_investment_plan": "t",
            "risk_debate_state": risk,
            "investment_plan": "i",
            "final_trade_decision": "d",
            "asset_type": "crypto_perp",
            "positioning_report": "positioning text",
        }
        g._log_state("2026-01-10", final_state, "crypto_perp")
        log_file = next((tmp_path / "BTCUSDT").glob(
            "YiAlphaStrategy_logs/full_states_log_*.json"))
        entry = json.loads(log_file.read_text(encoding="utf-8"))
        assert entry["positioning_report"] == "positioning text"

    def test_log_state_without_positioning_lacks_the_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            YiAlphaGraph, "_latest_close_and_atr",
            lambda self, t, d, a: (None, None),
        )
        g = object.__new__(YiAlphaGraph)
        g.ticker = "AAPL"
        g.config = {"results_dir": str(tmp_path)}
        final_state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-01-10",
            "market_report": "m", "sentiment_report": "s",
            "news_report": "n", "fundamentals_report": "f",
            "investment_debate_state": {
                "bull_history": "", "bear_history": "", "history": "",
                "current_response": "", "judge_decision": "",
            },
            "trader_investment_plan": "t",
            "risk_debate_state": {
                "aggressive_history": "", "conservative_history": "",
                "neutral_history": "", "history": "", "judge_decision": "",
            },
            "investment_plan": "i",
            "final_trade_decision": "d",
        }
        g._log_state("2026-01-10", final_state, "stock")
        log_file = next((tmp_path / "AAPL").glob(
            "YiAlphaStrategy_logs/full_states_log_*.json"))
        entry = json.loads(log_file.read_text(encoding="utf-8"))
        assert "positioning_report" not in entry  # presence-gated additive


# ---------------------------------------------------------------------------
# config-check: an unknown provider fails the gate (exit 1)
# ---------------------------------------------------------------------------


class TestConfigCheckUnknownProvider:
    def test_unknown_provider_fails(self, monkeypatch):
        from typer.testing import CliRunner

        from yialpha.cli.main import app

        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "typoed-providr")
        result = CliRunner().invoke(app, ["config-check"])
        assert result.exit_code == 1
        assert "unknown provider" in result.output


# ---------------------------------------------------------------------------
# run_robust: a clean retry clears the earlier DEGRADED marker
# ---------------------------------------------------------------------------

_ROBUST_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_robust.py"
)
_rr_spec = importlib.util.spec_from_file_location("run_robust_ki", _ROBUST_PATH)
assert _rr_spec is not None and _rr_spec.loader is not None
rr = importlib.util.module_from_spec(_rr_spec)
_rr_spec.loader.exec_module(rr)


class TestRunRobustDegradedClear:
    def test_clean_retry_pops_degraded(self, tmp_path, monkeypatch):
        child = tmp_path / "exit0.py"
        child.write_text("import sys; sys.exit(0)\n", encoding="utf-8")
        monkeypatch.setenv("YIALPHA_ROBUST_CHILD_SCRIPT", str(child))

        sentinel_reads = {"n": 0}

        def fake_sentinels(reports_root, ticker, date):  # noqa: ARG001
            sentinel_reads["n"] += 1
            return 2 if sentinel_reads["n"] == 1 else 0

        monkeypatch.setattr(rr, "_core_sentinel_count", fake_sentinels)
        monkeypatch.setattr(rr, "_find_new_report", lambda *a: Path("fake.md"))
        monkeypatch.setattr(rr, "_complete_mtime", lambda *a: 0.0)

        import argparse

        opts = argparse.Namespace(
            reports_root=str(tmp_path / "reports"),
            max_attempts=2,
            per_ticker_timeout=60,
            stall_timeout=30,
            stall_cpu=0.0,
            backoff=0.0,
            allow_degraded=False,
            require_data_quality=True,
            no_llm_cache=True,
            asset_type=None,
            date="2026-01-10",
            tickers=["AAPL"],
            workers=1,
        )
        result = rr._run_one_ticker("AAPL", "2026-01-10", opts)
        assert result["attempts"] == 2          # attempt 1 degraded -> retried
        assert result["ok"] is True
        assert "degraded" not in result         # clean retry cleared the marker


# ---------------------------------------------------------------------------
# S7-a: credential redaction covers the Zhipu / Tavily key shapes
# ---------------------------------------------------------------------------


class TestRedactionShapes:
    def test_tavly_shape_redacted(self):
        key = "tvly-AbCdEf1234567890GhIjKlMnOpQrStU"
        assert redact_secrets(f"failed with key {key}") == (
            "failed with key [REDACTED]")

    def test_zhipu_shape_redacted(self):
        key = f"{'a1' * 16}.{'B2' * 8}"  # 32-hex dot 16-alnum
        assert redact_secrets(f"auth error: {key}") == "auth error: [REDACTED]"

    def test_uuid_hex_without_dot_survives(self):
        import uuid

        thread_id = uuid.uuid4().hex
        assert redact_secrets(f"thread {thread_id} resumed") == (
            f"thread {thread_id} resumed")

    def test_pool_env_values_exact_redacted(self, monkeypatch):
        monkeypatch.setenv("ZHIPU_CN_API_KEYS", "alpha123, beta456")
        assert "alpha123" not in redact_secrets("rotated to alpha123 failed")
        assert "beta456" not in redact_secrets("then beta456 also failed")

    def test_alpha_vantage_shape_still_redacted(self):
        assert redact_secrets("av key 9G6OZ7mB2cDe4FgH") == (
            "av key [REDACTED]")


# ---------------------------------------------------------------------------
# accuracy forming-bar seam
# ---------------------------------------------------------------------------


class TestAccuracyFormingBarSeam:
    def test_dated_close_clamps_end_to_equity_today(self, monkeypatch):
        import yialpha.accuracy as acc
        import yialpha.dataflows.y_finance as yfmod

        monkeypatch.setattr(acc, "_us_equity_today", lambda: date(2026, 6, 10))
        seen = {}

        def fake_history(symbol, start, end):
            seen["end"] = end
            return pd.DataFrame(
                {"Close": [100.0, 101.0]},
                index=pd.Index(["2026-06-01", "2026-06-09"]),
            )

        monkeypatch.setattr(yfmod, "get_YFin_history_cached", fake_history)
        series = acc._dated_close("AAPL", "stock", "2026-06-01", "2026-06-20")
        # The fetch's exclusive end never passes the US equity calendar's
        # today — the forming bar cannot enter the series.
        assert seen["end"] == "2026-06-10"
        assert len(series) == 2

    def _patch_frames(self, monkeypatch, stock_rows, bench_rows):
        import yialpha.dataflows.y_finance as yfmod

        frames = {
            "AAPL": pd.DataFrame(
                {"Close": [r[1] for r in stock_rows]},
                index=pd.Index([r[0] for r in stock_rows]),
            ),
            "SPY": pd.DataFrame(
                {"Close": [r[1] for r in bench_rows]},
                index=pd.Index([r[0] for r in bench_rows]),
            ),
        }

        def fake_history(symbol, start, end):  # noqa: ARG001
            return frames[symbol]

        monkeypatch.setattr(yfmod, "get_YFin_history_cached", fake_history)
        return frames

    def test_cutoff_today_trims_forming_bar_to_pending(self, monkeypatch):
        import yialpha.accuracy as acc

        monkeypatch.setattr(acc, "_us_equity_today", lambda: date(2026, 6, 10))
        rows = [
            ("2026-06-01", 100.0), ("2026-06-05", 105.0),
            ("2026-06-09", 110.0), ("2026-06-10", 999.0),  # forming
        ]
        self._patch_frames(monkeypatch, rows, rows)
        raw, alpha, days = acc.fetch_returns_yf(
            "AAPL", "2026-06-01", holding_days=5, as_of_date="2026-06-10")
        # Trimmed to 3 closed bars < holding window: the entry stays pending.
        assert (raw, alpha, days) == (None, None, None)

    def test_cutoff_today_never_uses_forming_close(self, monkeypatch):
        import yialpha.accuracy as acc

        monkeypatch.setattr(acc, "_us_equity_today", lambda: date(2026, 6, 10))
        rows = [
            ("2026-06-01", 100.0), ("2026-06-03", 104.0),
            ("2026-06-05", 110.0), ("2026-06-10", 999.0),  # forming
        ]
        self._patch_frames(monkeypatch, rows, rows)
        raw, _alpha, days = acc.fetch_returns_yf(
            "AAPL", "2026-06-01", holding_days=2, as_of_date="2026-06-10")
        assert days == 2
        assert raw == pytest.approx((110.0 - 100.0) / 100.0)  # not 999


# ---------------------------------------------------------------------------
# legacy outcome legs: funding degradation paths degrade, never crash
# ---------------------------------------------------------------------------


def _legacy_item(scope: str = "POSITIONING") -> dict:
    return {
        "prediction_id": "p-1",
        "run_id": "r-1",
        "horizon_days": 5,
        "prediction_scope": scope,
        "direction": "up",
        "analysis_as_of": "2026-01-05T00:00:00+00:00",
        "instrument_id": "BTCUSDT",
        "instrument_class": "pure_crypto_perp",
        "asset_type": "crypto_perp",
        "timing": None,
        "timing_error": None,
        "regime_id": None,
    }


class TestLegacyLegDegradation:
    def _patch_writes(self, monkeypatch):
        import yialpha.ledger.outcome_compute as oc

        written = []

        def fake_write(prediction_id, run_id, horizon, **kwargs):
            written.append(kwargs)
            return "oid-1"

        monkeypatch.setattr(oc, "write_outcome", fake_write)
        return oc, written

    def test_positioning_funding_no_data_degrades_to_incomplete(self, monkeypatch):
        from yialpha.dataflows.errors import NoMarketDataError

        oc, written = self._patch_writes(monkeypatch)
        monkeypatch.setattr(
            oc, "_funding_window_sum",
            lambda *a: (_ for _ in ()).throw(NoMarketDataError("gone")),
        )
        detail = oc._compute_one(_legacy_item(), "2026-02-01T00:00:00+00:00")
        assert detail.status == "incomplete"
        assert "LEG_FUNDING_WINDOW" in str(detail.legs_missing) or any(
            "funding" in str(leg).lower() for leg in detail.legs_missing)
        assert written and written[0]["status"] == "incomplete"

    def test_positioning_funding_missing_degrades_to_incomplete(self, monkeypatch):
        oc, written = self._patch_writes(monkeypatch)
        monkeypatch.setattr(oc, "_funding_window_sum", lambda *a: (None, None))
        detail = oc._compute_one(_legacy_item(), "2026-02-01T00:00:00+00:00")
        assert detail.status == "incomplete"
        assert written and written[0]["status"] == "incomplete"

    def test_underlying_missing_price_degrades_after_grace(self, monkeypatch):
        oc, written = self._patch_writes(monkeypatch)
        monkeypatch.setattr(
            oc, "_equity_close_series", lambda *a, **k: pd.Series(dtype=float))
        item = _legacy_item(scope="UNDERLYING")
        item["instrument_class"] = "stock_perp"
        item["instrument_id"] = "MUUSDT"
        # now well past horizon_end (2026-01-10) + 7d grace: terminal.
        detail = oc._compute_one(item, "2026-03-01T00:00:00+00:00")
        assert detail.status == "incomplete"
        assert any(
            "price" in str(leg).lower() for leg in detail.legs_missing)
        assert written and written[0]["status"] == "incomplete"
