"""R4 — the unified evidence time contract (run_context / evidence / graph).

Four pinned properties:

1. ONE run as-of: ``_run_graph`` binds ``register_run``, the run context,
   the regime compute and the regime re-bind to the SAME ``analysis_as_of``
   — precise UTC instant on live runs, date-only on historical replays —
   across ALL THREE regime paths (success / uncomputable / off).
2. Run-derived evidence (sentiment/news digests, analysis output) recorded
   without an explicit ``available_at`` anchors to the run's
   ``analysis_as_of`` — NOT wall-clock record time, which one instant after
   the anchor would PIT-reject against the run's own clock drift.
3. External fetched data keeps its real ``available_at``: explicit values
   pass through untouched (never backfilled), and a future-dated row is
   PIT-rejected.
4. PIT rejections are DISCLOSED, never silently swallowed: the typed
   :class:`PITEvidenceViolation` (a ``ValueError``) surfaces as its own
   explicit WARNING naming both timestamps.

The graph-side cases reuse this module's own shell helpers
(``_make_graph_shell`` / ``_stub_pipeline``) the same way the offline
review probe does, with every external I/O seam patched out.
"""

from __future__ import annotations

import io
import logging
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

from yialpha.dataflows import run_scope
from yialpha.dataflows.config import set_config
from yialpha.ledger.evidence import PITEvidenceViolation, evidence_for_run, register_run
from yialpha.ledger.models import REPLAYABILITY_LIVE_ONLY, SCOPE_CONTRACT
from yialpha.ledger.run_context import (
    current_ledger_run_context,
    record_evidence_block,
    reset_ledger_run_context,
    set_ledger_run_context,
)

_ROOT = Path(__file__).resolve().parents[1]
_RUN_ID = "RTIMECONTR001"
_ANCHOR = "2026-09-03T03:00:00+00:00"
_INJECTED_AT = "2026-09-03T03:01:00+00:00"
_DAY = "2026-09-03"


@pytest.fixture(autouse=True)
def _clean_time_contract():
    reset_ledger_run_context()
    run_scope.reset_run_scope()
    yield
    reset_ledger_run_context()
    run_scope.reset_run_scope()


# --------------------------------------------------------------------------- #
# 1. run-derived evidence anchors to the run's as-of, not wall clock
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_run_derived_evidence_defaults_to_the_run_anchor(monkeypatch):
    set_config({"prediction_ledger": True})
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _ANCHOR)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _ANCHOR
    )
    # Wall clock "now" is one minute AFTER the run anchored: under the old
    # wall-clock default this block was PIT-rejected (and the failure
    # swallowed). Run-derived evidence belongs to the run's instant.
    monkeypatch.setattr("yialpha.ledger.evidence.utc_now_iso", lambda: _INJECTED_AT)

    record_evidence_block(
        "news", "news_data", "BTCUSDT", SCOPE_CONTRACT, "PAYLOAD",
        replayability=REPLAYABILITY_LIVE_ONLY,
    )
    rows = evidence_for_run(_RUN_ID)
    assert len(rows) == 1
    assert rows[0].available_at == _ANCHOR


@pytest.mark.unit
def test_explicit_available_at_passes_through_untouched(monkeypatch):
    """External fetched data keeps its REAL availability — never backfilled."""
    set_config({"prediction_ledger": True})
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _ANCHOR)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _ANCHOR
    )
    monkeypatch.setattr("yialpha.ledger.evidence.utc_now_iso", lambda: _INJECTED_AT)

    record_evidence_block(
        "onchain_flows", "onchain", "BTCUSDT", SCOPE_CONTRACT, "PAYLOAD",
        replayability=REPLAYABILITY_LIVE_ONLY,
        available_at="2026-09-03T01:00:00+00:00",
    )
    rows = evidence_for_run(_RUN_ID)
    assert len(rows) == 1
    assert rows[0].available_at == "2026-09-03T01:00:00+00:00"


@pytest.mark.unit
def test_future_available_at_is_pit_rejected_and_disclosed(caplog):
    """available_at > analysis_as_of stays refused — and the refusal is a
    visible WARNING naming both timestamps, not a silent swallow."""
    set_config({"prediction_ledger": True})
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _ANCHOR)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _ANCHOR
    )

    with caplog.at_level(logging.WARNING, logger="yialpha.ledger.run_context"):
        record_evidence_block(
            "news", "news_data", "BTCUSDT", SCOPE_CONTRACT, "PAYLOAD",
            replayability=REPLAYABILITY_LIVE_ONLY,
            available_at="2026-09-03T04:00:00+00:00",
        )
    assert evidence_for_run(_RUN_ID) == []
    assert "point-in-time guard rejected" in caplog.text
    assert "2026-09-03T04:00:00+00:00" in caplog.text
    assert _ANCHOR in caplog.text


@pytest.mark.unit
def test_pit_guard_raises_the_typed_valueerror():
    """The guard's error is typed (a ValueError subclass) so the historical
    ``match='PIT violation'`` contract and fail-soft handling both hold."""
    assert issubclass(PITEvidenceViolation, ValueError)
    register_run(_RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _ANCHOR)
    set_ledger_run_context(
        _RUN_ID, "BTCUSDT", "crypto_perp", "pure_crypto_perp", _ANCHOR
    )
    with pytest.raises(PITEvidenceViolation, match="PIT violation"):
        from yialpha.ledger.evidence import record_evidence

        record_evidence(
            run_id=_RUN_ID,
            source="news",
            category="news_data",
            symbol="BTCUSDT",
            scope=SCOPE_CONTRACT,
            payload="PAYLOAD",
            replayability=REPLAYABILITY_LIVE_ONLY,
            analysis_as_of=_ANCHOR,
            available_at="2026-09-03T04:00:00+00:00",
        )


# --------------------------------------------------------------------------- #
# 2. ONE run as-of across every _run_graph ledger path
# --------------------------------------------------------------------------- #
def _shell():
    helpers = runpy.run_path(str(_ROOT / "tests/test_regime_state.py"))
    return helpers


def _run_shell_mode(mode: str, is_historical: bool, forbidden):
    """Mirror the offline review probe: one shell run per regime path with
    every external seam patched, capturing register_run's as-of, the bound
    context's as-of and the run_context WARNING stream."""
    from yialpha.ledger import run_context as ctx
    from yialpha.regime.state import RegimeState

    helpers = _shell()
    set_config({
        "prediction_ledger": True,
        "regime_state": mode != "off",
        "instrument_registry": False,
    })
    observed: list[dict[str, str]] = []
    messages = io.StringIO()
    handler = logging.StreamHandler(messages)
    logger = logging.getLogger("yialpha.ledger.run_context")
    logger.addHandler(handler)
    graph = helpers["_make_graph_shell"](
        _ROOT, prediction_ledger=True, regime_state=mode != "off"
    )
    helpers["_stub_pipeline"](graph)

    def invoke(state, args):
        context = ctx.current_ledger_run_context()
        observed.append({"context_as_of": context.analysis_as_of})
        if mode == "success":
            ctx.record_evidence_block(
                "news", "news_data", "BTCUSDT", SCOPE_CONTRACT,
                "synthetic nonsecret evidence", replayability=REPLAYABILITY_LIVE_ONLY,
            )
        return dict(state)

    graph._invoke_or_stream = invoke
    regime = (
        RegimeState(regime_id="Gmock", analysis_as_of=_ANCHOR)
        if mode == "success"
        else None
    )
    try:
        with (
            patch(
                "yialpha.ledger.evidence.register_run",
                side_effect=lambda **kw:
                    observed.append({"registered_as_of": kw["analysis_as_of"]}),
            ),
            patch("yialpha.graph.routing.instrument_class",
                  return_value="pure_crypto_perp"),
            patch("yialpha.dataflows.utils.is_historical_date",
                  return_value=is_historical),
            patch("yialpha.regime.compute.compute_regime_state",
                  return_value=regime),
            patch("yialpha.ledger.regime_store.upsert_regime"),
            patch("yialpha.ledger.sqlite.utc_now_iso", return_value=_ANCHOR),
            patch("yialpha.ledger.evidence.utc_now_iso", return_value=_INJECTED_AT),
            patch("yialpha.ledger.evidence.get_connection", side_effect=forbidden),
            patch("sqlite3.connect", side_effect=forbidden),
        ):
            graph._run_graph("BTCUSDT", _DAY, asset_type="crypto_perp")
    finally:
        logger.removeHandler(handler)
        ctx.reset_ledger_run_context()
        run_scope.reset_run_scope()
    return {
        "observed": observed,
        "captured": messages.getvalue(),
    }


@pytest.mark.unit
def test_all_regime_paths_bind_one_precise_live_as_of():
    """Live run: register, context, regime compute and re-bind share the ONE
    precise as-of; run-derived evidence is NOT PIT-rejected by wall clock."""

    def forbidden(*args, **kwargs):
        raise AssertionError("External I/O forbidden in offline probe")

    for mode in ("off", "none", "success"):
        result = _run_shell_mode(mode, is_historical=False, forbidden=forbidden)
        registered = result["observed"][0]["registered_as_of"]
        context = result["observed"][1]["context_as_of"]
        assert registered == context == _ANCHOR, (mode, result["observed"])
        assert "T" in registered, mode
        if mode == "success":
            assert "point-in-time guard rejected" not in result["captured"], (
                mode,
                result["captured"],
            )


@pytest.mark.unit
def test_historical_replay_keeps_the_date_only_as_of():
    """D7 preserved: a historical replay anchors everything to the bare
    analysis date (all three paths agree on the date-only value)."""

    def forbidden(*args, **kwargs):
        raise AssertionError("External I/O forbidden in offline probe")

    for mode in ("off", "none", "success"):
        result = _run_shell_mode(mode, is_historical=True, forbidden=forbidden)
        registered = result["observed"][0]["registered_as_of"]
        context = result["observed"][1]["context_as_of"]
        assert registered == context == _DAY, (mode, result["observed"])
        assert "T" not in registered, mode


# --------------------------------------------------------------------------- #
# 3. run_scope isolation survives a stubbed _log_state
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_run_scope_starts_cold_and_is_dropped_even_with_stubbed_log_state():
    """The reset gate lives in _run_graph (start-cold + finally), NOT only
    inside _log_state — a stubbed _log_state (tests, CLI finalize bypass)
    can no longer leak one run's prefetch snapshot into the next run."""
    set_config({
        "prediction_ledger": False, "regime_state": False,
        "instrument_registry": False,
    })
    helpers = _shell()
    graph = helpers["_make_graph_shell"](_ROOT, prediction_ledger=False)
    helpers["_stub_pipeline"](graph)

    # Stale scope from a "previous run" carries a cached marker.
    run_scope.ensure_run_scope()
    assert run_scope.run_cached(("marker", "key"), lambda: "STALE") == "STALE"

    observed: list[str] = []

    def invoke(state, args):
        observed.append(run_scope.run_cached(("marker", "key"), lambda: "FRESH"))
        return dict(state)

    graph._invoke_or_stream = invoke
    graph._run_graph("BTCUSDT", _DAY, asset_type="crypto_perp")

    # Start-cold reset purged the stale entry before the run's nodes ran...
    assert observed == ["FRESH"]
    # ...and the finally reset dropped the scope even though _log_state is
    # a stub: run_cached no longer caches (every call fetches).
    fetches: list[int] = []

    def _val():
        fetches.append(1)
        return object()

    run_scope.run_cached(("after", "run"), _val)
    run_scope.run_cached(("after", "run"), _val)
    assert len(fetches) == 2


@pytest.mark.unit
def test_run_scope_reset_keeps_run_isolated_within_a_single_run():
    """The isolation CONTRACT is unchanged: within one run, the same key
    fetches exactly once (the reset must not break per-run single-fetch).
    After the run ends the scope is dropped — a post-run call fetches fresh
    by design (that is the next run starting cold)."""
    set_config({
        "prediction_ledger": False, "regime_state": False,
        "instrument_registry": False,
    })
    helpers = _shell()
    graph = helpers["_make_graph_shell"](_ROOT, prediction_ledger=False)
    helpers["_stub_pipeline"](graph)
    fetches: list[str] = []
    observed: list[object] = []

    def invoke(state, args):
        # Two node entries within ONE run share the run's cache dict.
        observed.append(
            run_scope.run_cached(("bundle", "btc"), lambda: fetches.append("x"))
        )
        observed.append(
            run_scope.run_cached(("bundle", "btc"), lambda: fetches.append("x"))
        )
        return dict(state)

    graph._invoke_or_stream = invoke
    graph._run_graph("BTCUSDT", _DAY, asset_type="crypto_perp")
    # Both in-run node entries hit the cache: one fetch total.
    assert fetches == ["x"]
    # Run ended -> scope dropped: each post-run call fetches fresh (next
    # run starts cold), not from the finished run's cache.
    graph._invoke_or_stream({}, {})
    assert fetches == ["x", "x", "x"]
    assert current_ledger_run_context() is None
