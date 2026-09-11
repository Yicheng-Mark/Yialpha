"""Offline coverage for the B1 capture / B2 scoring entries and the allowlist
guard layer. No test performs a real vendor call: allowed paths are exercised
through pre-patched transport originals, and runner flows use fixture seams."""

from __future__ import annotations

import json
import os
import socket
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.p0_shadow.guard import VENDOR_ALLOWED_HOSTS, GuardError, isolated_runtime
from scripts.p0_shadow.runner import FORMED, _FixtureMarket, capture_cohort, score_cohort

WEDNESDAY = datetime(2026, 9, 9, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _fresh_yfinance_cache_managers():
    """Re-arm yfinance's process-wide cache singletons before each runtime."""
    from yfinance import cache as yf_cache

    for name in ("_TzDBManager", "_CookieDBManager", "_ISINDBManager"):
        manager = getattr(yf_cache, name, None)
        if manager is not None:
            manager.set_location(manager.get_location())
    yield


class _NoopRecorder:
    """_FixtureMarket records through this; capture files its own evidence."""

    def record_response(self, **kwargs):
        return {"request_id": "fixture-" + uuid.uuid4().hex}


def _fixture_market() -> _FixtureMarket:
    return _FixtureMarket(_NoopRecorder(), "baseline")


def _fixture_patches(market: _FixtureMarket):
    from yialpha.ledger import outcome_compute as oc, time_contract as tc

    def enter(stack: ExitStack):
        for context in (
            patch.object(tc, "_now_utc", lambda: market.now),
            patch.object(tc, "_reference_frame", market.reference),
            patch.object(oc, "binance_klines_frame", market.perp),
            patch.object(oc, "get_YFin_history_cached", market.equity),
            patch.object(oc, "get_binance_funding_rate", market.funding),
        ):
            stack.enter_context(context)

    return enter


# ---------------------------------------------------------------- allowlist ---

def test_allowlist_dns_connect_and_http_layers(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    import curl_cffi
    import requests

    fake_dns = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    fake_response = SimpleNamespace(url="https://fapi.binance.com/fapi/v1/klines", history=[])
    connected = []
    yahoo = "https://query1.finance.yahoo.com/v8/finance/chart"
    with (patch.object(socket, "getaddrinfo", lambda *a, **k: fake_dns),
          patch.object(socket, "create_connection",
                       lambda address, *a, **k: SimpleNamespace(peer=address)),
          patch.object(socket.socket, "connect", lambda self, address: connected.append(address)),
          patch.object(requests.Session, "request",
                       lambda self, method, url, *a, **k: fake_response),
          patch.object(curl_cffi.Curl, "getinfo", lambda self, info: yahoo),
          patch.object(curl_cffi.Curl, "perform", lambda self, *a, **k: None),
          isolated_runtime(root, "allow", network_mode="allowlist") as summary):
        assert socket.getaddrinfo("fapi.binance.com", 443) == fake_dns
        socket.create_connection(("fapi.binance.com", 443))
        sock = socket.socket()
        sock.connect(("93.184.216.34", 443))  # resolved from the allowed host
        sock.close()
        session = requests.Session()
        assert session.request("GET", "https://fapi.binance.com/fapi/v1/klines") is fake_response
        curl = curl_cffi.Curl()
        curl.perform()
        curl.close()
        for denied in (
            lambda: socket.getaddrinfo("evil.example", 443),
            lambda: socket.getaddrinfo(12345, 443),
            lambda: socket.create_connection(("199.9.9.9", 9)),
            lambda: requests.Session().request("GET", "https://evil.example/x"),
            lambda: requests.Session().request("GET", "https://query2.evil.yahoo.com/x"),
        ):
            with pytest.raises(GuardError, match="allowlist"):
                denied()
    assert connected == [("93.184.216.34", 443)]
    assert summary["network_mode"] == "allowlist"
    assert summary["allowed_calls"] == 5
    assert summary["network_attempts"] == 5


def test_allowlist_denies_unconfigured_and_redirecting_curl(tmp_path):
    import curl_cffi

    root = tmp_path / "cohort"
    root.mkdir()
    with isolated_runtime(root, "curl", network_mode="allowlist"):
        curl = curl_cffi.Curl()
        try:
            with pytest.raises(GuardError, match="allowlist"):
                curl.perform()
        finally:
            curl.close()
    redirect = SimpleNamespace(
        url="https://fapi.binance.com/fapi/v1/klines",
        history=[SimpleNamespace(url="https://evil.example/hop", status=302)],
    )
    with (patch.object(curl_cffi.requests.Session, "request",
                       lambda self, method, url, *a, **k: redirect),
          isolated_runtime(root, "hop", network_mode="allowlist"),
          pytest.raises(GuardError, match="allowlist")):
        curl_cffi.requests.Session().request("GET", "https://fapi.binance.com/x")


def test_allowlist_curl_perform_accepts_bytes_effective_url(tmp_path):
    """curl_cffi getinfo returns bytes; the perform gate must decode them.

    Regression: yfinance's first fc.yahoo.com request was denied because the
    pre-perform host check saw bytes as an unknown host (2026-09-11).
    """
    import curl_cffi

    root = tmp_path / "cohort"
    root.mkdir()
    chart = b"https://query1.finance.yahoo.com/v8/finance/chart"
    with (patch.object(curl_cffi.Curl, "getinfo", lambda self, info: chart),
          patch.object(curl_cffi.Curl, "perform", lambda self, *a, **k: None),
          isolated_runtime(root, "bytes", network_mode="allowlist") as summary):
        curl = curl_cffi.Curl()
        curl.perform()
        curl.close()
    assert summary["allowed_calls"] >= 1
    assert {"guce.yahoo.com", "consent.yahoo.com"} <= VENDOR_ALLOWED_HOSTS


def test_allowlist_keeps_configured_proxy_as_transport_only(tmp_path, monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8086")
    monkeypatch.setenv("all_proxy", "http://127.0.0.1:8086")
    root = tmp_path / "cohort"
    root.mkdir()
    connected = []
    with (patch.object(socket, "getaddrinfo",
                       lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                                         ("127.0.0.1", 8086))]),
          patch.object(socket.socket, "connect",
                       lambda self, address: connected.append(address)),
          isolated_runtime(root, "proxy", network_mode="allowlist") as summary):
        # Env stays exactly as configured; the proxy is a permitted transport.
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:8086"
        sock = socket.socket()
        sock.connect(("127.0.0.1", 8086))
        sock.close()
        assert socket.getaddrinfo("127.0.0.1", 8086)
        # Other loopback targets are not configured transports.
        with pytest.raises(GuardError, match="allowlist"):
            socket.socket().connect(("127.0.0.1", 9999))
        with pytest.raises(GuardError, match="allowlist"):
            socket.create_connection(("localhost", 8080))
    assert connected == [("127.0.0.1", 8086)]
    assert summary["proxy_endpoint_hosts"] == ["127.0.0.1"]


def test_unknown_network_mode_still_rejected(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    with pytest.raises(GuardError, match="network_mode"), \
            isolated_runtime(root, "bad", network_mode="live"):
        pytest.fail("live runtime accepted")


def test_offline_mode_blocks_allowlisted_hosts_too(tmp_path):
    root = tmp_path / "cohort"
    root.mkdir()
    with (isolated_runtime(root, "strict", network_mode="offline") as summary,
          pytest.raises(GuardError, match="disabled")):
        socket.getaddrinfo(next(iter(VENDOR_ALLOWED_HOSTS)), 443)
    assert summary["network_mode"] == "offline"
    assert summary["network_attempts"] == 1


# ------------------------------------------------------------------ capture ---

def test_b1_capture_forms_process_samples_and_is_idempotent(tmp_path):
    root = tmp_path / "cohort"
    market = _fixture_market()
    with ExitStack() as stack:
        _fixture_patches(market)(stack)
        first = capture_cohort(root, cohort_id="b1", identity={"head": "fixture"},
                               now_fn=lambda: market.now)
        assert first["sample_kind"] == "process_validation_fixed_input"
        assert first["captured_groups"] == ["C-BTC", "C-ETH", "C-MU"]
        assert first["skipped_groups"] == []
        assert first["network"]["denied_attempts"] == 0

        again = capture_cohort(root, cohort_id="b1", identity={"head": "fixture"},
                               now_fn=lambda: market.now)
        assert again["captured_groups"] == []
        assert again["skipped_groups"] == ["C-BTC", "C-ETH", "C-MU"]

        weekend = capture_cohort(root, cohort_id="b1", groups=("U-MU-WE",),
                                 identity={"head": "fixture"}, now_fn=lambda: market.now)
        assert weekend["captured_groups"] == ["U-MU-WE"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_kind"] == "process_validation_fixed_input"
    assert manifest["llm_used"] is False
    snapshots = sorted((root / "snapshots").glob("*.json"))
    assert len(snapshots) == 12
    capture_reports = sorted((root / "attempts").glob("capture-*/capture.json"))
    assert len(capture_reports) == 3
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in capture_reports]
    productive = [report for report in reports if report["captured"]]
    assert len(productive) == 2  # the first run (3 groups) and the weekend append
    assert all(report["llm_used"] is False for report in reports)
    assert all(report["vendor_calls"] for report in productive)
    assert {call["source"] for report in productive for call in report["vendor_calls"]} <= {
        "binance_perp:1d:last", "yfinance:1d:close", "binance_perp:funding"}
    for group in ("b1-C-BTC", "b1-C-ETH", "b1-C-MU", "b1-U-MU-WE"):
        assert (root / "costs" / f"{group}.json").is_file()
    recorded = json.loads(
        next((root / "raw").glob("*/record.json")).read_text(encoding="utf-8"))
    assert recorded["evidence_level"] == "vendor_library_output"
    assert recorded["sample_kind"] == "process_validation_fixed_input"
    timing = json.loads(snapshots[0].read_text(encoding="utf-8"))["prediction"]["timing"]
    assert timing["reference_price"] == 99.0  # Friday close before Saturday formation


def test_b1_capture_refuses_weekend_group_on_a_weekday(tmp_path):
    with pytest.raises(GuardError, match="weekend"):
        capture_cohort(tmp_path / "cohort", cohort_id="b1", groups=("U-MU-WE",),
                       identity={"head": "fixture"}, now_fn=lambda: WEDNESDAY)
    assert not (tmp_path / "cohort").exists()


def test_b1_capture_rejects_offline_fixture_cohorts(tmp_path):
    from scripts.p0_shadow.runner import run_offline_demo

    run_offline_demo(tmp_path / "demo", cohort_id="demo")
    market = _fixture_market()
    with (ExitStack() as stack,
          pytest.raises(GuardError, match="process-validation")):
        _fixture_patches(market)(stack)
        capture_cohort(tmp_path / "demo", cohort_id="demo", groups=("C-BTC",),
                       identity={"head": "fixture"}, now_fn=lambda: market.now)


# -------------------------------------------------------------------- score ---

def test_b2_score_drains_the_queue_and_writes_full_reports(tmp_path):
    root = tmp_path / "cohort"
    market = _fixture_market()
    with ExitStack() as stack:
        _fixture_patches(market)(stack)
        capture_cohort(root, cohort_id="b2", groups=("C-BTC", "C-ETH", "C-MU", "U-MU-WE"),
                       identity={"head": "fixture"}, now_fn=lambda: market.now)
        matured = FORMED + timedelta(days=28)
        market.now = matured
        result = score_cohort(root, cohort_id="b2", identity={"head": "fixture"},
                              now_fn=lambda: matured)
        assert result["outcomes_by_status"] == {"complete": 11, "incomplete": 1}
        assert result["pending_remaining"] == 0
        assert result["scoring_invoked"] is True

        again = score_cohort(root, cohort_id="b2", identity={"head": "fixture"},
                             now_fn=lambda: matured)
        assert again["outcomes_by_status"] == {"complete": 11, "incomplete": 1}
        assert again["pending_remaining"] == 0
    attempt = result["attempt_id"]
    for name in ("scoring.json", "scoreboard.json", "scoreboard.md"):
        assert (root / "attempts" / attempt / name).is_file()
    scoring = json.loads((root / "attempts" / attempt / "scoring.json").read_text(encoding="utf-8"))
    assert scoring["request_ids"]
    assert scoring["sample_kind"] == "process_validation_fixed_input"
    board_md = (root / "attempts" / attempt / "scoreboard.md").read_text(encoding="utf-8")
    assert board_md.startswith("> 固定输入流程验证，非预测能力评估")
    audit = json.loads((root / "reports" / attempt / "audit.json").read_text(encoding="utf-8"))
    assert audit["sample_kind"] == "process_validation_fixed_input"
    assert audit["data_mode"] == "real_vendor_calls"
    assert (root / "checkpoints" / attempt / "portfolio.db").is_file()
    assert sorted(path.name for path in (root / "attempts").glob("score-*")) == sorted(
        [Path(result["attempt_id"]).name, Path(again["attempt_id"]).name])


def test_b2_score_rejects_offline_fixture_cohorts(tmp_path):
    from scripts.p0_shadow.runner import run_offline_demo

    run_offline_demo(tmp_path / "demo", cohort_id="demo")
    with pytest.raises(GuardError, match="process-validation"):
        score_cohort(tmp_path / "demo", cohort_id="demo", identity={"head": "fixture"})


def test_b2_score_real_clock_is_not_bypassable_from_the_cli():
    """The CLI never exposes now_fn; scoring always uses datetime.now(UTC)."""
    import inspect

    import scripts.p0_shadow_validation as cli

    assert "now_fn" not in inspect.getsource(cli)


# ---------------------------------------------------------------------- CLI ---

def test_cli_dispatches_capture_and_score(tmp_path, capsys, monkeypatch):
    import scripts.p0_shadow.runner as runner_module
    import scripts.p0_shadow_validation as cli

    calls = {}

    def fake_capture(root, *, cohort_id, groups):
        calls["capture"] = (str(root), cohort_id, groups)
        return {"ok": 1}

    def fake_score(root, *, cohort_id):
        calls["score"] = (str(root), cohort_id)
        return {"ok": 2}

    monkeypatch.setattr(runner_module, "capture_cohort", fake_capture)
    monkeypatch.setattr(runner_module, "score_cohort", fake_score)
    assert cli.main(["capture", "--root", str(tmp_path / "x"), "--cohort-id", "c1",
                     "--groups", "C-BTC, C-ETH"]) == 0
    assert cli.main(["score", "--root", str(tmp_path / "x"), "--cohort-id", "c1"]) == 0
    assert calls["capture"][2] == ("C-BTC", "C-ETH")
    assert calls["score"] == (str(tmp_path / "x"), "c1")
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"ok": 2}


def test_cli_rejects_unknown_groups_without_side_effects(tmp_path, capsys):
    import scripts.p0_shadow_validation as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["capture", "--root", str(tmp_path / "never"), "--cohort-id", "c2",
                  "--groups", "NOPE"])
    assert exc.value.code == 2
    assert "P0 guard rejected operation" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()
