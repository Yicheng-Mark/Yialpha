"""Batch-1 data-reliability guards: the data-vacuum gate and its wiring.

Covers the pieces ``test_data_quality_evidence.py`` does not:

* the Trader-node gate wiring in ``GraphSetup`` (the three-wiring rule: a new
  behaviour on a node must be verifiable from the outside, not just assumed);
* run_robust's inverted quality gate (default ON, ``--allow-degraded`` out);
* direct-connect tool + Reddit degrade sentinels (the ledger blind spots);
* the timeout defaults (yfinance 30s, BaoStock scoped 30s) and the default
  multi-vendor fallback chains.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from yialpha.dataflows import quality

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_quality():
    quality.reset_quality()
    yield
    quality.reset_quality()


# --------------------------------------------------------------------------- #
# Graph wiring: the Trader node must be gated
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_trader_node_is_gated():
    """The vacuum gate is a node-level wiring decision — assert it from the
    outside instead of trusting the comment in setup.py (round-5 lesson: a
    prompt/tool wired in one place but not the other ships silently)."""
    import yialpha.graph.setup as setup_mod
    from yialpha.graph.setup import GraphSetup

    real_gate = setup_mod.quality.gate_on_data_vacuum
    wrapped = []

    def spy(handler):
        wrapped.append(handler)
        return real_gate(handler)

    with mock.patch.object(setup_mod.quality, "gate_on_data_vacuum", spy):
        GraphSetup(
            quick_thinking_llm=mock.MagicMock(),
            deep_thinking_llm=mock.MagicMock(),
            debate_llm=mock.MagicMock(),
            tool_nodes={k: mock.MagicMock() for k in ("market", "social", "news", "fundamentals")},
            conditional_logic=mock.MagicMock(),
        ).setup_graph(("market", "social", "news", "fundamentals"))

    assert len(wrapped) == 1, "the vacuum gate must wrap exactly one node (Trader)"


# --------------------------------------------------------------------------- #
# run_robust: quality gate inverted to default-ON
# --------------------------------------------------------------------------- #
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "run_robust.py"
_spec = importlib.util.spec_from_file_location("run_robust_gate_under_test", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


class TestRobustQualityGateDefault(unittest.TestCase):
    def _parse(self, *extra):
        with mock.patch.object(
            sys, "argv", ["run_robust.py", "--tickers", "AAPL", "--date", "2026-01-01", *extra]
        ):
            return rr._parse_args()

    def test_quality_gate_default_on(self):
        opts = self._parse()
        self.assertTrue(opts.require_data_quality)

    def test_allow_degraded_opts_out(self):
        opts = self._parse("--allow-degraded")
        self.assertFalse(opts.require_data_quality)
        self.assertTrue(opts.allow_degraded)

    def test_legacy_require_flag_is_noop_but_default_stays_on(self):
        opts = self._parse("--require-data-quality")
        self.assertTrue(opts.require_data_quality)


# --------------------------------------------------------------------------- #
# Direct-connect tools: every DATA_UNAVAILABLE must leave ledger evidence
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_market_data_validator_degrade_records_sentinel(monkeypatch):
    import yialpha.dataflows.market_data_validator as validator
    from yialpha.dataflows.errors import NoMarketDataError

    def no_data(symbol, curr_date):
        raise NoMarketDataError(symbol, symbol, "no rows")

    monkeypatch.setattr(validator, "load_ohlcv", no_data)
    out = validator.build_verified_market_snapshot("COF", "2026-05-13")
    assert out.startswith("DATA_UNAVAILABLE")
    events = quality.snapshot_quality()
    assert [e["method"] for e in events] == ["build_verified_market_snapshot"]
    assert events[0]["kind"] == quality.KIND_OPTIONAL_UNAVAILABLE


@pytest.mark.unit
def test_price_structure_load_degrade_records_sentinel(monkeypatch):
    import yialpha.agents.utils.price_structure_tools as ps

    def boom(symbol, curr_date):
        raise RuntimeError("socket hang")

    monkeypatch.setattr(ps, "load_ohlcv", boom)
    out = ps._load("COF", "2026-05-13")
    assert isinstance(out, str) and out.startswith("DATA_UNAVAILABLE")
    events = quality.snapshot_quality()
    assert [e["method"] for e in events] == ["price_structure_ohlcv"]


@pytest.mark.unit
def test_price_structure_insufficient_rows_records_sentinel(monkeypatch):
    import pandas as pd

    import yialpha.agents.utils.price_structure_tools as ps

    monkeypatch.setattr(
        ps, "load_ohlcv",
        lambda s, d: pd.DataFrame({"Date": [d], "Close": [1.0]}),  # < 2 rows
    )
    out = ps._load("COF", "2026-05-13")
    assert isinstance(out, str) and out.startswith("DATA_UNAVAILABLE")
    assert quality.snapshot_quality()[0]["method"] == "price_structure_ohlcv"


@pytest.mark.unit
def test_weekly_indicators_degrade_records_sentinel(monkeypatch):
    import yialpha.agents.utils.weekly_indicators_tools as weekly

    def boom(symbol, curr_date):
        raise RuntimeError("resample failed")

    monkeypatch.setattr(weekly, "weekly_ohlcv", boom)
    # Call the raw function: langchain's .invoke() executes the tool body in
    # a copied context, so the sentinel would land outside this test's
    # ContextVar (in production ToolNode nodes inherit the bound ledger —
    # that propagation is covered by test_quality_context_propagation.py).
    out = weekly.get_indicators_weekly.func("COF", "2026-05-13")
    assert out.startswith("DATA_UNAVAILABLE")
    events = quality.snapshot_quality()
    assert [e["method"] for e in events] == ["get_indicators_weekly"]


@pytest.mark.unit
def test_binance_indicators_degrade_records_sentinel(monkeypatch):
    import yialpha.agents.utils.binance_indicator_tools as bind

    def boom(symbol, start, end, **kw):
        raise RuntimeError("429")

    monkeypatch.setattr(bind, "binance_klines_frame", boom)
    out = bind._indicators_core("BTCUSDT", "2026-05-13", 30, "perp", "")
    assert out.startswith("DATA_UNAVAILABLE")
    events = quality.snapshot_quality()
    assert [e["method"] for e in events] == ["get_binance_indicators"]
    assert "perp" in events[0]["detail"]


# --------------------------------------------------------------------------- #
# Reddit: genuine degrades recorded; the keyless RSS default stays silent
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_reddit_rss_failure_records_sentinel(monkeypatch):
    import yialpha.dataflows.reddit as reddit

    def boom(*args, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(reddit, "urlopen", boom)
    posts = reddit._fetch_subreddit_rss("COF", "wallstreetbets", 5, 1.0)
    assert posts == []
    events = quality.snapshot_quality()
    assert [e["method"] for e in events] == ["fetch_reddit_posts"]
    assert "RSS failed" in events[0]["detail"]


@pytest.mark.unit
def test_reddit_oauth_token_unavailable_records_sentinel(monkeypatch):
    import yialpha.dataflows.reddit as reddit

    monkeypatch.setattr(reddit, "_get_oauth_token", lambda timeout: None)
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", lambda *a: [])
    reddit._fetch_subreddit_json("COF", "wallstreetbets", 5, 1.0)
    events = quality.snapshot_quality()
    assert "OAuth token unavailable" in events[0]["detail"]


@pytest.mark.unit
def test_reddit_zero_posts_records_sentinel(monkeypatch):
    import yialpha.dataflows.reddit as reddit

    monkeypatch.setattr(reddit, "_fetch_subreddit", lambda *a: [])
    out = reddit.fetch_reddit_posts("ZZZZ", subreddits=("wallstreetbets",), timeout=1.0)
    assert "no Reddit posts found" in out
    assert quality.snapshot_quality(), "zero posts across all subs is evidence"


@pytest.mark.unit
def test_reddit_keyless_default_path_stays_silent(monkeypatch):
    """No creds configured = RSS is the designed default, not a degrade —
    recording here would fire a sentinel on every keyless run (noise)."""
    import yialpha.dataflows.reddit as reddit

    def rss_with_posts(ticker, sub, limit, timeout, _retry=True):
        return [{"title": "t", "score": None, "num_comments": None,
                 "created_utc": None, "selftext": "", "source": "rss"}]

    monkeypatch.setattr(reddit, "_reddit_oauth_creds", lambda: None)
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss", rss_with_posts)
    reddit.fetch_reddit_posts("COF", subreddits=("wallstreetbets",), timeout=1.0)
    assert quality.snapshot_quality() == []


# --------------------------------------------------------------------------- #
# Timeouts: 30s defaults, explicit 0 = opt-out
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_yf_timeout_default_30():
    import yialpha.dataflows.stockstats_utils as ssu

    assert ssu.YF_HTTP_TIMEOUT == 30.0


@pytest.mark.unit
def test_yf_timeout_env_zero_disables(monkeypatch):
    import importlib

    import yialpha.dataflows.stockstats_utils as ssu

    monkeypatch.setenv("YIALPHA_HTTP_TIMEOUT_S", "0")
    importlib.reload(ssu)
    try:
        assert ssu.YF_HTTP_TIMEOUT is None  # explicit 0 = deliberate opt-out
    finally:
        monkeypatch.delenv("YIALPHA_HTTP_TIMEOUT_S", raising=False)
        importlib.reload(ssu)  # restore the module default for later tests
    assert ssu.YF_HTTP_TIMEOUT == 30.0


@pytest.mark.unit
def test_baostock_timeout_default_and_scope():
    import yialpha.dataflows.baostock_vendor as bs

    assert bs.BS_SOCKET_TIMEOUT == 30.0

    calls = SimpleNamespace(login=lambda: SimpleNamespace(error_code="0", error_msg=""),
                            logout=lambda: None)
    session = bs._BaostockSession.__new__(bs._BaostockSession)
    session.bs = calls
    prev = socket.getdefaulttimeout()
    try:
        with session:
            # Scoped while the session is live: the raw TCP socket to
            # baostock.com:9001 inherits the default timeout.
            assert socket.getdefaulttimeout() == 30.0
        # Restored after exit — the global default must not leak.
        assert socket.getdefaulttimeout() == prev
    finally:
        socket.setdefaulttimeout(prev)


# --------------------------------------------------------------------------- #
# Default config: multi-vendor chains + vacuum policy default
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_default_vendor_chains_have_fallback():
    from yialpha.default_config import DEFAULT_CONFIG

    vendors = DEFAULT_CONFIG["data_vendors"]
    assert vendors["core_stock_apis"] == "yfinance,alpha_vantage"
    assert vendors["technical_indicators"] == "yfinance,alpha_vantage"
    # PR5 (2026-09): fundamentals are SEC-FIRST — EDGAR filings carry the
    # real ``filed`` date (point-in-time ground truth), unlike the
    # period_end+45-day heuristic yfinance/AV apply. yfinance follows as the
    # keyless unlimited supplement; AV stays last.
    assert vendors["fundamental_data"] == "sec_edgar,yfinance,alpha_vantage"
    assert vendors["news_data"] == "yfinance,alpha_vantage"
    # alpha_vantage (rate-limited free tier) must never precede the keyless
    # vendors in any chain.
    for chain in vendors.values():
        parts = chain.split(",")
        if "alpha_vantage" in parts:
            assert parts.index("alpha_vantage") == len(parts) - 1
            assert "yfinance" in parts


@pytest.mark.unit
def test_data_vacuum_policy_default_reject():
    from yialpha.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["data_vacuum_policy"] == "reject"
    # The chain must be registered so an unattended run inherits the default.
    from yialpha.default_config import _ENV_OVERRIDES

    assert _ENV_OVERRIDES["YIALPHA_DATA_VACUUM_POLICY"] == "data_vacuum_policy"
