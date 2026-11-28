"""Round-16 audit tests for the ``web`` shard (app / runner / store / health).

Every test pins one documented contract of the in-place local Web UI: the
read-only store must degrade gracefully on malformed on-disk logs, path/Host
validation must hold at the HTTP boundary, and the single-slot runner must
never leak its busy slot on an exception path.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

from web import app as app_module, runner, store
from web.app import app

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def logs_root(monkeypatch, tmp_path):
    """Point the store at an empty per-test logs root (never ~/.yialpha)."""
    root = tmp_path / "logs"
    monkeypatch.setattr(store, "LOGS_ROOT", root)
    return root


@pytest.fixture
def client(monkeypatch, logs_root):
    """App client: isolated task registry, isolated logs root, real lifespan."""
    monkeypatch.setattr(runner, "registry", runner.TaskRegistry())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_no_raise(monkeypatch, logs_root):
    """Same as ``client`` but server exceptions surface as 500 responses."""
    monkeypatch.setattr(runner, "registry", runner.TaskRegistry())
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _write_state_log(root, ticker, day, decision="**Rating**: Buy", extra=None, raw=None):
    """Write one ``full_states_log_<day>.json`` (``raw`` bytes win if given)."""
    d = root / ticker / "YiAlphaStrategy_logs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"full_states_log_{day}.json"
    if raw is not None:
        path.write_bytes(raw)
        return path
    state = {"company_of_interest": ticker, "trade_date": day, "final_trade_decision": decision}
    if extra:
        state.update(extra)
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# store: malformed / non-UTF-8 / non-finite on-disk logs must not 500
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b'{"final_trade_decision": "**Rating**: Buy \xe4\xb8\xad\xff"}', id="invalid-utf8"),
        pytest.param(b"\xff\xfe[]", id="binary-junk"),
        pytest.param(b"[]", id="list-root"),
        pytest.param(b"null", id="null-root"),
        pytest.param(b'"a string"', id="string-root"),
        pytest.param(b"42", id="number-root"),
    ],
)
def test_r16_store_unreadable_log_does_not_break_ticker_grid(client_no_raise, logs_root, raw):
    """One malformed states log must read as rating=None, never 500 the whole
    home grid (``/api/tickers`` lists every ticker on one scan)."""
    _write_state_log(logs_root, "AAPL", "2026-06-10", raw=raw)
    response = client_no_raise.get("/api/tickers")
    assert response.status_code == 200, response.text
    [entry] = response.json()["tickers"]
    assert entry["ticker"] == "AAPL"
    assert entry["latest_date"] == "2026-06-10"
    assert entry["latest_rating"] is None
    assert entry["history"] == [{"date": "2026-06-10", "rating": None}]


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b'{"final_trade_decision": "Buy \xff"}', id="invalid-utf8"),
        pytest.param(b"[]", id="list-root"),
        pytest.param(b"null", id="null-root"),
    ],
)
def test_r16_store_unreadable_log_is_skipped_by_compare(client_no_raise, logs_root, raw):
    """``list_compare`` skips a ticker whose only log is unreadable."""
    _write_state_log(logs_root, "AAPL", "2026-06-10", raw=raw)
    _write_state_log(logs_root, "MSFT", "2026-06-10", decision="**Rating**: Sell")
    response = client_no_raise.get("/api/compare")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "tickers": [{"ticker": "MSFT", "date_ratings": [{"date": "2026-06-10", "rating": "Sell"}]}]
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b'{"final_trade_decision": "Buy \xff"}', id="invalid-utf8"),
        pytest.param(b"[]", id="list-root"),
        pytest.param(b"null", id="null-root"),
        pytest.param(b'{"final_trade_decision": "**Rating**: Buy"', id="truncated"),
        pytest.param(b"", id="empty"),
    ],
)
def test_r16_store_unreadable_run_detail_is_404(client_no_raise, logs_root, raw):
    """A malformed run's detail view is an honest 404 (same as a missing
    file), not a 500."""
    _write_state_log(logs_root, "AAPL", "2026-06-10", raw=raw)
    listing = client_no_raise.get("/api/tickers/AAPL/runs")
    assert listing.status_code == 200, listing.text
    assert listing.json()["date_ratings"] == [{"date": "2026-06-10", "rating": None}]
    detail = client_no_raise.get("/api/tickers/AAPL/runs/2026-06-10")
    assert detail.status_code == 404, detail.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "decision",
    [42, 3.5, True, [], ["Buy"], {"rating": "Buy"}],
    ids=["int", "float", "bool", "empty-list", "list", "dict"],
)
def test_r16_store_non_string_decision_field_reads_as_hold(client_no_raise, logs_root, decision):
    """A readable object whose ``final_trade_decision`` is not a string is a
    *readable* run (rating defaults to Hold per the documented contract) with
    an empty decision — ``parse_rating``/``parse_overlay`` must not see the
    raw value and 500."""
    _write_state_log(
        logs_root,
        "AAPL",
        "2026-06-10",
        raw=json.dumps({"final_trade_decision": decision}).encode("utf-8"),
    )
    grid = client_no_raise.get("/api/tickers")
    assert grid.status_code == 200, grid.text
    assert grid.json()["tickers"][0]["latest_rating"] == "Hold"
    detail = client_no_raise.get("/api/tickers/AAPL/runs/2026-06-10")
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert payload["rating"] == "Hold"
    assert payload["overlay"] is None
    assert payload["sections"]["final_trade_decision"] == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    "extra",
    [
        {"investment_debate_state": "text"},
        {"investment_debate_state": ["Buy"]},
        {"risk_debate_state": 7},
        {"risk_debate_state": [None]},
    ],
    ids=["ideb-str", "ideb-list", "rdeb-int", "rdeb-list"],
)
def test_r16_store_non_object_debate_block_reads_as_empty_sections(client_no_raise, logs_root, extra):
    """A malformed debate block reads as empty debate sections, not a 500."""
    _write_state_log(logs_root, "AAPL", "2026-06-10", extra=extra)
    detail = client_no_raise.get("/api/tickers/AAPL/runs/2026-06-10")
    assert detail.status_code == 200, detail.text
    sections = detail.json()["sections"]
    assert sections["investment_debate"] == {
        "bull_history": "",
        "bear_history": "",
        "history": "",
        "current_response": "",
        "judge_decision": "",
    }
    assert sections["risk_debate"] == {
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "history": "",
        "judge_decision": "",
    }


@pytest.mark.unit
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_r16_store_non_finite_literal_in_run_log_is_served_as_null(client_no_raise, logs_root, literal):
    """The graph writes states logs with ``json.dump`` defaults (``allow_nan``
    on), so a NaN/Infinity literal can legitimately reach disk; the read side
    must null it in the parse pass instead of failing the response render."""
    raw = (
        '{"final_trade_decision": "**Rating**: Buy", '
        f'"data_quality": {{"ratio": {literal}, "core_sentinel_count": 1}}, '
        f'"sentiment_report": "s", "regime_state": {{"turbulence": {literal}}}}}'
    ).encode()
    _write_state_log(logs_root, "AAPL", "2026-06-10", raw=raw)
    response = client_no_raise.get("/api/tickers/AAPL/runs/2026-06-10")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["rating"] == "Buy"
    assert payload["data_quality"] == {"ratio": None, "core_sentinel_count": 1}
    assert literal not in response.text


@pytest.mark.unit
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_r16_store_non_finite_literal_in_node_perf_is_served_as_null(client_no_raise, logs_root, literal):
    """``dump_perf_report`` also writes with ``allow_nan`` on; a non-finite
    telemetry value must not 500 the run detail that embeds ``node_perf``."""
    _write_state_log(logs_root, "AAPL", "2026-06-10")
    perf_dir = logs_root / "AAPL" / "YiAlphaStrategy_logs"
    (perf_dir / "node_perf_2026-06-10.json").write_text(
        f'{{"nodes": [{{"name": "market", "tokens_per_s": {literal}}}]}}', encoding="utf-8"
    )
    response = client_no_raise.get("/api/tickers/AAPL/runs/2026-06-10")
    assert response.status_code == 200, response.text
    assert response.json()["node_perf"] == {"nodes": [{"name": "market", "tokens_per_s": None}]}


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b'{"hit_rate": 0.5, "note": "\xff"}', id="invalid-utf8"),
        pytest.param(b"[]", id="list-root"),
        pytest.param(b"null", id="null-root"),
        pytest.param(b'{"hit_rate": 0.5', id="truncated"),
        pytest.param(b"", id="empty"),
    ],
)
def test_r16_store_accuracy_artifact_malformed_is_unavailable(client_no_raise, logs_root, raw):
    """A malformed / non-object accuracy artifact reports ``available=False``
    with the command hint — never 500 and never a fabricated report."""
    acc_dir = logs_root / "accuracy"
    acc_dir.mkdir(parents=True)
    (acc_dir / "accuracy_report.json").write_bytes(raw)
    response = client_no_raise.get("/api/accuracy")
    assert response.status_code == 200, response.text
    assert response.json() == {"available": False, "hint": "yialpha verify-history"}


@pytest.mark.unit
def test_r16_store_accuracy_artifact_non_finite_is_nulled_but_available(client, logs_root):
    """Round-11 contract (must survive the round-16 reader refactor): a
    pre-fix artifact's bare NaN is nulled and the report is still served."""
    acc_dir = logs_root / "accuracy"
    acc_dir.mkdir(parents=True)
    (acc_dir / "accuracy_report.json").write_text(
        '{"directional": {"Buy": {"hit_rate": NaN, "n": 4}}, "mean_hold_return": Infinity}',
        encoding="utf-8",
    )
    response = client.get("/api/accuracy")
    assert response.status_code == 200
    assert response.json() == {
        "available": True,
        "directional": {"Buy": {"hit_rate": None, "n": 4}},
        "mean_hold_return": None,
    }


@pytest.mark.unit
def test_r16_store_accuracy_artifact_absent_is_unavailable(client, logs_root):
    """No artifact on disk yet: the honest answer is ``available=False`` plus
    the command hint, never a fabricated empty report."""
    assert client.get("/api/accuracy").json() == {
        "available": False,
        "hint": "yialpha verify-history",
    }


# --------------------------------------------------------------------------- #
# app: Host-header rejection must still carry the security headers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_r16_app_host_403_response_carries_security_headers(client):
    """The security-headers middleware promises its restrictions on *every*
    response; the Host-rejection 403 must not be the one uncovered path."""
    response = client.get("/api/tickers", headers={"Host": "evil.example.com"})
    assert response.status_code == 403
    assert response.json() == {"detail": "untrusted Host header: 'evil.example.com'"}
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


# --------------------------------------------------------------------------- #
# app: Host header grammar is parsed strictly (malformed == untrusted)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("header", "expected"),
    [
        pytest.param("127.0.0.1:8000", "127.0.0.1", id="ipv4-port"),
        pytest.param("127.0.0.1", "127.0.0.1", id="ipv4-bare"),
        pytest.param("LOCALHOST:8000", "localhost", id="case-folded"),
        pytest.param("  [::1]:8000 ", "::1", id="bracketed-ipv6-with-space"),
        pytest.param("[::1]", "::1", id="bracketed-ipv6-no-port"),
        pytest.param("::1", "::1", id="bare-ipv6"),
        pytest.param("lan.example.com:8443", "lan.example.com", id="name-port"),
        pytest.param("", "", id="empty"),
        pytest.param("[::1]evil.com", "", id="bracket-then-name"),
        pytest.param("[::1]:8000evil.com", "", id="bracket-port-then-name"),
        pytest.param("[::1", "", id="unterminated-bracket"),
        pytest.param("127.0.0.1:80@evil.com", "", id="userinfo-after-port"),
        pytest.param("localhost@evil.com", "", id="at-sign"),
        pytest.param("127.0.0.1 evil.com", "", id="space-joined"),
        pytest.param("127.0.0.1:8a00", "", id="non-numeric-port"),
        pytest.param("[::1]:abc", "", id="bracketed-non-numeric-port"),
        pytest.param("[]", "", id="empty-brackets"),
    ],
)
def test_r16_app_hostname_of_parses_host_grammar_strictly(header, expected):
    """``_hostname_of`` must return a trusted-looking name only for a
    well-formed ``host[:port]`` / ``[v6][:port]`` value; anything else
    (userinfo, trailing garbage, unterminated bracket) yields ``""`` so it
    can never match the allowlist."""
    assert app_module._hostname_of(header) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "host",
    [
        pytest.param("[::1]evil.com", id="bracket-then-name"),
        pytest.param("[::1]:8000evil.com", id="bracket-port-then-name"),
        pytest.param("[::1", id="unterminated-bracket"),
        pytest.param("127.0.0.1:80@evil.com", id="userinfo-after-port"),
        pytest.param("localhost@evil.com", id="at-sign"),
        pytest.param("127.0.0.1 evil.com", id="space-joined"),
        pytest.param("127.0.0.1:8a00", id="non-numeric-port"),
        pytest.param("[::1]:abc", id="bracketed-non-numeric-port"),
    ],
)
def test_r16_app_malformed_host_header_fails_closed(client, host):
    """A malformed Host header that merely *starts* with a loopback alias is
    not a loopback alias: the DNS-rebinding guard must answer 403."""
    response = client.get("/api/tickers", headers={"Host": host})
    assert response.status_code == 403, response.text
    assert response.json()["detail"].startswith("untrusted Host header")


@pytest.mark.unit
@pytest.mark.parametrize(
    "host",
    [
        pytest.param("127.0.0.1:8000", id="ipv4-port"),
        pytest.param("127.0.0.1", id="ipv4-bare"),
        pytest.param("localhost:8000", id="name-port"),
        pytest.param("LOCALHOST", id="upper-bare"),
        pytest.param("[::1]:8000", id="bracketed-ipv6-port"),
        pytest.param("[::1]", id="bracketed-ipv6"),
        pytest.param("::1", id="bare-ipv6"),
    ],
)
def test_r16_app_wellformed_loopback_host_headers_still_served(client, host):
    """Strict parsing must not over-reject: every well-formed loopback
    spelling keeps reaching the app (200 from a JSON endpoint)."""
    response = client.get("/api/tickers", headers={"Host": host})
    assert response.status_code == 200, response.text
    assert response.json() == {"tickers": []}


@pytest.mark.unit
def test_r16_app_allowed_hosts_env_normalizes_entries(monkeypatch):
    """``YIALPHA_WEB_ALLOWED_HOSTS`` entries go through the same strict
    hostname parser (case-folded, port/brackets stripped); empty segments are
    ignored and a malformed entry adds nothing to the allowlist."""
    monkeypatch.setenv(
        "YIALPHA_WEB_ALLOWED_HOSTS",
        " Lan.Example.COM:8443 , [::2]:9000, ,bad host,[::1]evil.com,[",
    )
    allowed = app_module._allowed_hostnames()
    assert allowed == set(app_module._TRUSTED_HOSTNAMES) | {"lan.example.com", "::2"}


@pytest.mark.unit
def test_r16_app_malformed_allowed_hosts_entry_does_not_trust_empty_host(client, monkeypatch):
    """An operator typo in ``YIALPHA_WEB_ALLOWED_HOSTS`` must never widen the
    allowlist to the empty hostname (a ``Host:`` header with no value)."""
    monkeypatch.setenv("YIALPHA_WEB_ALLOWED_HOSTS", "[, @, ]")
    assert "" not in app_module._allowed_hostnames()
    response = client.get("/api/tickers", headers={"Host": ""})
    assert response.status_code == 403, response.text


# --------------------------------------------------------------------------- #
# app: POST /api/analyze validation matrix — reject before spawn
# --------------------------------------------------------------------------- #


def _spawn_must_not_run(monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError(f"invalid request reached process spawn: {args!r} {kwargs!r}")

    monkeypatch.setattr(runner, "spawn", fail_if_called)


def _recording_spawn(monkeypatch, calls):
    """Replace ``runner.spawn`` with one that records its arguments and
    registers the task the way the real ``spawn`` does (never a process)."""

    async def fake_spawn(ticker, analysis_date, asset_type, language="en"):
        calls.append((ticker, analysis_date, asset_type, language))
        task_id = f"task-{len(calls)}"
        runner.registry.tasks[task_id] = runner.TaskState(
            task_id=task_id,
            ticker=ticker,
            date=analysis_date,
            asset_type=asset_type,
            started_at=1_700_000_000.0,
            language=language,
        )
        return task_id

    monkeypatch.setattr(runner, "spawn", fake_spawn)


_PAST_DATE = "2026-06-10"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "asset_type", "detail_start"),
    [
        pytest.param("", "auto", "ticker is required", id="empty-ticker"),
        pytest.param("   \t", "auto", "ticker is required", id="whitespace-ticker"),
        pytest.param("--max-attempts", "auto", "Unsupported market", id="option-like"),
        pytest.param("-x", "auto", "Unsupported market", id="short-option"),
        pytest.param("--tickers=MSFT", "auto", "Unsupported market", id="option-with-value"),
        pytest.param("AAPL MSFT", "auto", "bad ticker", id="space-separated-pair"),
        pytest.param("AAPL,MSFT", "auto", "bad ticker", id="comma-separated-pair"),
        pytest.param("AAPL;MSFT", "auto", "bad ticker", id="semicolon-separated-pair"),
        pytest.param("AAPL\nMSFT", "auto", "bad ticker", id="newline-separated-pair"),
        pytest.param("../MSFT", "auto", "bad ticker", id="dot-dot"),
        pytest.param("..\\MSFT", "auto", "bad ticker", id="dot-dot-backslash"),
        pytest.param("0700.HK", "auto", "Unsupported market", id="hk"),
        pytest.param("600519.SS", "auto", "Unsupported market", id="a-share"),
        pytest.param("EURUSD", "auto", "Unsupported market", id="forex"),
        pytest.param("GC=F", "auto", "Unsupported market", id="commodity-cfd"),
        pytest.param("^GSPC", "auto", "Unsupported market", id="index"),
        pytest.param("BTC USDT", "crypto_perp", "bad ticker: 'BTC USDT'", id="venue-space"),
        pytest.param("BTC,USDT", "crypto_spot", "bad ticker: 'BTC,USDT'", id="venue-comma"),
        pytest.param("BTC/USDT", "auto", "bad ticker: 'BTC/USDT'", id="yahoo-slash"),
        pytest.param("[A-Z]USDT", "crypto_perp", "bad ticker: '[AZ]USDT'", id="glob-metachars"),
        pytest.param("USDT", "crypto_perp", "Symbol 'USDT' normalizes to a degenerate", id="venue-degenerate"),
        pytest.param("USDTUSDT", "crypto_spot", "Symbol 'USDTUSDT' normalizes to a degenerate", id="venue-degenerate-2"),
    ],
)
def test_r16_app_analyze_rejects_bad_ticker_before_spawn(client, monkeypatch, ticker, asset_type, detail_start):
    """Every ticker the child's ``prepare_batch_run`` would exit 2 on (bad
    charset, unsupported market, degenerate venue pair) is refused with 400
    here — the web endpoint must not accept what the subprocess rejects."""
    _spawn_must_not_run(monkeypatch)
    response = client.post(
        "/api/analyze", json={"ticker": ticker, "date": _PAST_DATE, "asset_type": asset_type}
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"].startswith(detail_start)
    assert runner.registry.tasks == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "detail_start"),
    [
        pytest.param("date", "", "bad date", id="empty-date"),
        pytest.param("date", "2026-13-01", "bad date", id="month-13"),
        pytest.param("date", "2026-02-30", "bad date", id="feb-30"),
        pytest.param("date", "0000-00-00", "bad date", id="zeros"),
        pytest.param("date", "2026-6-10", "bad date", id="non-padded-month"),
        pytest.param("date", "2026-06-1", "bad date", id="non-padded-day"),
        pytest.param("date", "20260610", "bad date", id="compact"),
        pytest.param("date", "2026-06-10T00:00", "bad date", id="datetime"),
        pytest.param("date", "2026-06-10 ", "bad date", id="trailing-space"),
        pytest.param("date", "2026-06-10_perp", "bad date", id="venue-suffix"),
        pytest.param("date", "2026-06-10/../../x", "bad date", id="traversal"),
        pytest.param("asset_type", "", "bad asset_type", id="empty-asset-type"),
        pytest.param("asset_type", "STOCK", "bad asset_type", id="upper-asset-type"),
        pytest.param("asset_type", "forex", "bad asset_type", id="unsupported-asset-type"),
        pytest.param("asset_type", "binance_perp", "bad asset_type", id="internal-venue-name"),
        pytest.param("language", "", "bad language", id="empty-language"),
        pytest.param("language", "fr", "bad language", id="unsupported-language"),
        pytest.param("language", "EN", "bad language", id="upper-language"),
        pytest.param("language", "Chinese", "bad language", id="config-value-not-ui-code"),
    ],
)
def test_r16_app_analyze_rejects_bad_fields_before_spawn(client, monkeypatch, field, value, detail_start):
    """Malformed date / asset_type / language are 400s decided inline — never
    a task that dies inside the subprocess, and never a 404 (it is a body
    parameter, not a missing resource)."""
    _spawn_must_not_run(monkeypatch)
    body = {"ticker": "AAPL", "date": _PAST_DATE}
    body[field] = value
    response = client.post("/api/analyze", json=body)
    assert response.status_code == 400, response.text
    assert response.json()["detail"].startswith(detail_start)


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"date": _PAST_DATE}, id="missing-ticker"),
        pytest.param({"ticker": "AAPL"}, id="missing-date"),
        pytest.param({"ticker": 5, "date": _PAST_DATE}, id="int-ticker"),
        pytest.param({"ticker": "AAPL", "date": 20260610}, id="int-date"),
        pytest.param({"ticker": ["AAPL"], "date": _PAST_DATE}, id="list-ticker"),
        pytest.param({"ticker": "AAPL", "date": _PAST_DATE, "asset_type": ["stock"]}, id="list-asset-type"),
        pytest.param({}, id="empty-object"),
        pytest.param([], id="list-body"),
        pytest.param("AAPL", id="string-body"),
    ],
)
def test_r16_app_analyze_malformed_body_is_422(client, monkeypatch, body):
    """Pydantic body validation answers 422 for shape errors (missing /
    wrong-typed fields) without reaching the handler's own checks."""
    _spawn_must_not_run(monkeypatch)
    response = client.post("/api/analyze", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"][:1] == ["body"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"],
)
def test_r16_app_analyze_simple_content_types_are_not_parsed_as_json(client, monkeypatch, content_type):
    """A cross-site "simple" POST (no CORS preflight) must not be able to
    start a paid analysis: non-JSON content types never yield a model."""
    _spawn_must_not_run(monkeypatch)
    response = client.post(
        "/api/analyze",
        content=b'{"ticker": "AAPL", "date": "2026-06-10"}',
        headers={"content-type": content_type},
    )
    assert response.status_code == 422, response.text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "asset_type", "language", "expected_ticker"),
    [
        pytest.param(" aapl ", "auto", "en", "AAPL", id="strip-and-upper"),
        pytest.param("brk.b", "stock", "en", "BRK.B", id="class-share-dot"),
        pytest.param("BF-b", "stock", "en", "BF-B", id="class-share-dash"),
        pytest.param("btc-usdt", "crypto_perp", "zh", "BTCUSDT", id="perp-contract-symbol"),
        pytest.param("BTC/USDT", "crypto_spot", "en", "BTCUSDT", id="ccxt-pair-form"),
        pytest.param("btc:usdc", "crypto_perp", "en", "BTCUSDT", id="tradingview-form"),
        pytest.param("btc-usdt", "crypto", "en", "BTC-USD", id="umbrella-crypto-uses-yahoo-form"),
        pytest.param("1000PEPEUSDT", "auto", "en", "1000PEPEUSDT", id="unlisted-alt-stays-compact"),
    ],
)
def test_r16_app_analyze_accepted_request_spawns_canonical_symbol(
    client, monkeypatch, ticker, asset_type, language, expected_ticker
):
    """A valid request hands the child exactly one canonical ticker, the
    submitted date / asset_type / UI language, and returns the task id."""
    calls = []
    _recording_spawn(monkeypatch, calls)
    response = client.post(
        "/api/analyze",
        json={"ticker": ticker, "date": _PAST_DATE, "asset_type": asset_type, "language": language},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"task_id": "task-1", "started_at": 1_700_000_000.0}
    assert calls == [(expected_ticker, _PAST_DATE, asset_type, language)]
    assert runner.registry.is_busy() is False  # fake task is not "running"


@pytest.mark.unit
def test_r16_app_analyze_defaults_asset_type_and_language(client, monkeypatch):
    """Omitted ``asset_type``/``language`` default to ``auto`` / ``en`` —
    the child's argv then stays byte-identical to a plain CLI run."""
    calls = []
    _recording_spawn(monkeypatch, calls)
    response = client.post("/api/analyze", json={"ticker": "AAPL", "date": _PAST_DATE})
    assert response.status_code == 200, response.text
    assert calls == [("AAPL", _PAST_DATE, "auto", "en")]


@pytest.mark.unit
def test_r16_app_analyze_busy_slot_is_409_and_never_spawns(client, monkeypatch):
    """Single-slot concurrency: while one task is ``running`` a second submit
    is 409 and no second process is launched."""
    _spawn_must_not_run(monkeypatch)
    running = runner.TaskState(
        task_id="running-1", ticker="NVDA", date=_PAST_DATE, asset_type="stock", started_at=1.0
    )
    runner.registry.tasks[running.task_id] = running
    runner.registry._running_id = running.task_id
    response = client.post("/api/analyze", json={"ticker": "AAPL", "date": _PAST_DATE})
    assert response.status_code == 409, response.text
    assert response.json() == {"detail": "an analysis is already running"}


@pytest.mark.unit
def test_r16_app_analyze_stale_slot_owner_is_not_busy(client, monkeypatch):
    """A finished task that still owns the slot id must not block the next
    submit: busy is derived from the owner's *status*, not its presence."""
    calls = []
    _recording_spawn(monkeypatch, calls)
    done = runner.TaskState(
        task_id="done-1", ticker="NVDA", date=_PAST_DATE, asset_type="stock", started_at=1.0
    )
    done.status = "error"
    done.error = "run_robust exited 1"
    runner.registry.tasks[done.task_id] = done
    runner.registry._running_id = done.task_id
    response = client.post("/api/analyze", json={"ticker": "AAPL", "date": _PAST_DATE})
    assert response.status_code == 200, response.text
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# runner: the single busy slot survives every exit path (no real subprocess)
# --------------------------------------------------------------------------- #


class _FakeStdout:
    """Scripted child stdout: queued lines, an optional read error, or a
    ``block`` that never resolves (stands in for a long-running child)."""

    def __init__(self, lines=(), error=None, block=False):
        self._lines = list(lines)
        self._error = error
        self._block = block

    async def readline(self):
        if self._error is not None:
            raise self._error
        if self._block:
            await asyncio.Event().wait()
        return self._lines.pop(0) if self._lines else b""


class _FakeProc:
    def __init__(self, lines=(), rc=0, *, pid=4242, wait_error=None, read_error=None, block=False):
        self.pid = pid
        self._rc = rc
        self._wait_error = wait_error
        self.returncode = None
        self.stdout = _FakeStdout(lines, read_error, block)

    async def wait(self):
        if self._wait_error is not None:
            raise self._wait_error
        self.returncode = self._rc
        return self._rc


@pytest.fixture
def launch(monkeypatch):
    """Replace ``asyncio.create_subprocess_exec`` with a recorder that returns
    a scripted fake process; the real runner code path (slot reservation,
    watcher task, release) runs unchanged around it."""
    recorded = {"calls": []}

    def install(proc=None, error=None):
        async def fake_exec(*cmd, **kwargs):
            recorded["calls"].append(
                {"cmd": list(cmd), "kwargs": kwargs, "busy_at_launch": runner.registry.is_busy()}
            )
            if error is not None:
                raise error
            return proc if proc is not None else _FakeProc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        return recorded

    return install


def _fresh_registry(monkeypatch):
    registry = runner.TaskRegistry()
    monkeypatch.setattr(runner, "registry", registry)
    return registry


def _drive(coro_factory):
    """Run ``coro_factory()`` on a fresh loop, then drain the watcher tasks
    it created so the registry reaches its terminal state."""

    async def main():
        task_id = await coro_factory()
        watchers = list(runner._WATCH_TASKS)
        if watchers:
            await asyncio.wait_for(asyncio.gather(*watchers), timeout=10)
        return task_id

    return asyncio.run(main())


_RUN_DATE = "2026-01-10"


@pytest.mark.unit
def test_r16_runner_spawn_reserves_slot_before_launch_and_watcher_releases_it(monkeypatch, launch):
    """Happy path: the slot is already busy when the child is launched (the
    reservation precedes the first await), a rc-0 child ends ``done`` with the
    attempt banner parsed and the newest report dir attached, and the slot is
    released afterwards."""
    registry = _fresh_registry(monkeypatch)
    recorded = launch(
        _FakeProc(
            lines=[
                b"[AAPL] \xe2\x96\xb6\xef\xb8\x8f attempt 2/3 \xe2\x86\x92 log robust_AAPL_x_a2.log\r\n",
                b"[AAPL] done\n",
            ]
        )
    )
    monkeypatch.setenv("YIALPHA_OUTPUT_LANGUAGE", "Klingon")  # parent value must not leak
    monkeypatch.setenv("YIALPHA_R16_INHERIT_PROBE", "yes")
    monkeypatch.setattr(
        runner.store, "list_reports", lambda t: [{"dir": f"{t}_20260110_120000", "mtime": 1.0, "complete": True}]
    )

    task_id = _drive(lambda: runner.spawn("AAPL", _RUN_DATE, "crypto_perp", "zh"))

    [call] = recorded["calls"]
    assert call["busy_at_launch"] is True
    assert call["cmd"] == [
        sys.executable,
        str(runner._ROBUST),
        "--tickers", "AAPL",
        "--date", _RUN_DATE,
        "--workers", "1",
        "--asset-type", "crypto_perp",
    ]
    kwargs = call["kwargs"]
    assert kwargs["cwd"] == str(runner._PROJECT_ROOT)
    assert kwargs["stdout"] == asyncio.subprocess.PIPE
    assert kwargs["stderr"] == asyncio.subprocess.STDOUT
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert kwargs["env"]["YIALPHA_OUTPUT_LANGUAGE"] == "Chinese"
    assert kwargs["env"]["YIALPHA_R16_INHERIT_PROBE"] == "yes"  # the rest of the env is inherited

    state = registry.get(task_id)
    assert state is not None
    assert state.status == "done"
    assert (state.attempt, state.max_attempts) == (2, 3)
    assert state.pid == 4242
    assert state.report_dir == "AAPL_20260110_120000"
    assert state.error is None
    assert state.started_at <= state.finished_at <= time.time()
    assert state.log_tail == [
        # CRLF stripped, UTF-8 decoded
        "[AAPL] \u25b6\ufe0f attempt 2/3 \u2192 log robust_AAPL_x_a2.log",
        "[AAPL] done",
    ]
    assert registry.is_busy() is False
    assert registry._running_id is None
    assert not runner._WATCH_TASKS  # the done-callback dropped the strong ref


@pytest.mark.unit
def test_r16_runner_log_tail_keeps_last_n_lines_and_replaces_bad_bytes(monkeypatch, launch):
    """The log-tail view is bounded at ``_LOG_TAIL_MAX`` newest lines and a
    non-UTF-8 byte sequence decodes with U+FFFD instead of killing the watcher."""
    registry = _fresh_registry(monkeypatch)
    total = runner._LOG_TAIL_MAX + 10
    lines = [f"line {i}\n".encode() for i in range(1, total + 1)]
    lines[total // 2] = b"\xff\xfe half-gbk \xe4\xb8\xad\n"
    launch(_FakeProc(lines=lines))

    task_id = _drive(lambda: runner.spawn("AAPL", _RUN_DATE, "stock"))
    tail = registry.get(task_id).log_tail
    assert len(tail) == runner._LOG_TAIL_MAX
    assert tail[0] == f"line {total - runner._LOG_TAIL_MAX + 1}"
    assert tail[-1] == f"line {total}"
    assert "\ufffd\ufffd half-gbk 中" in tail


@pytest.mark.unit
@pytest.mark.parametrize("rc", [1, 2, -9])
def test_r16_runner_nonzero_exit_is_error_without_report_lookup(monkeypatch, launch, rc):
    """A failing child is ``error`` with the exit code in the message; the
    report lookup only runs for rc 0, and the slot is released regardless."""
    registry = _fresh_registry(monkeypatch)

    def must_not_lookup(_ticker):
        raise AssertionError("report lookup ran for a failed run")

    monkeypatch.setattr(runner.store, "list_reports", must_not_lookup)
    launch(_FakeProc(lines=[b"[AAPL] attempt 1/3\n"], rc=rc))

    task_id = _drive(lambda: runner.spawn("AAPL", _RUN_DATE, "stock"))
    state = registry.get(task_id)
    assert state.status == "error"
    assert state.error == f"run_robust exited {rc}"
    assert state.report_dir is None
    assert (state.attempt, state.max_attempts) == (1, 3)
    assert state.finished_at is not None
    assert registry.is_busy() is False


@pytest.mark.unit
def test_r16_runner_wait_failure_is_error_and_releases_slot(monkeypatch, launch):
    """``proc.wait()`` raising must not strand the slot: the watcher records a
    ``watch wait error`` task and still releases."""
    registry = _fresh_registry(monkeypatch)
    launch(_FakeProc(wait_error=OSError("wait broken")))

    task_id = _drive(lambda: runner.spawn("AAPL", _RUN_DATE, "stock"))
    state = registry.get(task_id)
    assert state.status == "error"
    assert state.error.startswith("watch wait error: OSError('wait broken')")
    assert registry.is_busy() is False


@pytest.mark.unit
@pytest.mark.parametrize(("rc", "expected_status"), [(0, "done"), (3, "error")])
def test_r16_runner_stdout_read_failure_still_follows_exit_code(monkeypatch, launch, rc, expected_status):
    """A broken stdout pipe is recorded, but the exit code stays the success
    criterion: the watcher still waits for the child and the slot is released."""
    registry = _fresh_registry(monkeypatch)
    monkeypatch.setattr(runner.store, "list_reports", lambda t: [])
    launch(_FakeProc(read_error=ConnectionResetError("pipe closed"), rc=rc))

    task_id = _drive(lambda: runner.spawn("AAPL", _RUN_DATE, "stock"))
    state = registry.get(task_id)
    assert state.status == expected_status
    assert "watch output error" in state.error
    assert registry.is_busy() is False


@pytest.mark.unit
def test_r16_runner_watcher_cancellation_releases_slot(monkeypatch, launch):
    """Event-loop shutdown cancels the watcher mid-read (``CancelledError`` is
    a ``BaseException``, so the ``except Exception`` arms do not run): the
    ``finally`` must still mark the task and free the slot."""
    registry = _fresh_registry(monkeypatch)
    launch(_FakeProc(block=True))

    async def scenario():
        task_id = await runner.spawn("AAPL", _RUN_DATE, "stock")
        (watcher,) = list(runner._WATCH_TASKS)
        await asyncio.sleep(0)  # let the watcher start and block inside readline()
        assert registry.is_busy() is True
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watcher
        return task_id

    task_id = asyncio.run(scenario())
    state = registry.get(task_id)
    assert state.status == "error"
    assert state.error == "run_robust exited None"
    assert state.finished_at is not None
    assert registry.is_busy() is False


@pytest.mark.unit
def test_r16_runner_spawn_cancelled_during_launch_rolls_back_reservation(monkeypatch, launch):
    """``CancelledError`` while the process is being created is a
    ``BaseException``: the reservation made before the await is rolled back
    and the cancellation propagates to the caller."""
    registry = _fresh_registry(monkeypatch)
    launch(error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.spawn("AAPL", _RUN_DATE, "stock"))

    assert registry.is_busy() is False
    [state] = registry.tasks.values()
    assert state.status == "error"
    assert state.finished_at is not None
    assert state.pid is None


@pytest.mark.unit
def test_r16_runner_late_release_from_finished_task_keeps_new_owner_busy(monkeypatch, launch):
    """A second run may start once the first finished; the first task's
    release (or any non-owner's) must not free the slot the second owns."""
    registry = _fresh_registry(monkeypatch)
    recorded = launch(_FakeProc(rc=1))

    async def scenario():
        first = await runner.spawn("NVDA", _RUN_DATE, "stock")
        await asyncio.gather(*list(runner._WATCH_TASKS))
        assert registry.is_busy() is False
        recorded["calls"].clear()
        # second launch blocks (child still running)
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", _blocking_exec(recorded, registry)
        )
        second = await runner.spawn("AAPL", _RUN_DATE, "stock")
        await asyncio.sleep(0)  # the second watcher is now blocked on its child
        assert registry.is_busy() is True
        registry.release(first)  # late / non-owner release
        registry.release("not-a-task")
        assert registry.is_busy() is True
        (watcher,) = list(runner._WATCH_TASKS)
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watcher
        return second

    second = asyncio.run(scenario())
    assert registry._running_id is None
    assert registry.get(second).status == "error"


def _blocking_exec(recorded, registry):
    async def fake_exec(*cmd, **kwargs):
        recorded["calls"].append({"cmd": list(cmd), "busy_at_launch": registry.is_busy()})
        return _FakeProc(block=True)

    return fake_exec


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asset_type", "extra"),
    [
        pytest.param("auto", [], id="auto-omits-flag"),
        pytest.param("", [], id="empty-omits-flag"),
        pytest.param("stock", ["--asset-type", "stock"], id="stock"),
        pytest.param("crypto", ["--asset-type", "crypto"], id="crypto"),
        pytest.param("crypto_perp", ["--asset-type", "crypto_perp"], id="perp"),
        pytest.param("crypto_spot", ["--asset-type", "crypto_spot"], id="spot"),
    ],
)
def test_r16_runner_build_cmd_keeps_each_value_in_its_own_argv_slot(asset_type, extra):
    """``_build_cmd`` passes the ticker and the date as single argv elements
    (no shell, no string concatenation), so a value can never split into
    extra options; ``auto`` omits ``--asset-type`` to stay CLI-identical."""
    cmd = runner._build_cmd("BRK.B", _RUN_DATE, asset_type)
    assert cmd == [
        sys.executable,
        str(runner._ROBUST),
        "--tickers",
        "BRK.B",
        "--date",
        _RUN_DATE,
        "--workers",
        "1",
        *extra,
    ]
    i = cmd.index("--tickers")
    assert cmd[i + 1] == "BRK.B"  # exactly one token follows --tickers
    assert cmd.count("--date") == 1 and cmd[cmd.index("--date") + 1] == _RUN_DATE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("language", "expected"), [("en", "English"), ("zh", "Chinese"), ("zz", "English")]
)
def test_r16_runner_language_routes_through_env_not_argv(monkeypatch, launch, language, expected):
    """The UI language reaches the child only as ``YIALPHA_OUTPUT_LANGUAGE``
    (argv untouched) and an unknown code falls back to English."""
    registry = _fresh_registry(monkeypatch)
    recorded = launch(_FakeProc())

    _drive(lambda: runner.spawn("AAPL", _RUN_DATE, "stock", language))

    [call] = recorded["calls"]
    # argv is the plain CLI spelling: no language flag, no language word
    assert call["cmd"] == [
        sys.executable, str(runner._ROBUST),
        "--tickers", "AAPL", "--date", _RUN_DATE, "--workers", "1",
        "--asset-type", "stock",
    ]
    assert call["kwargs"]["env"]["YIALPHA_OUTPUT_LANGUAGE"] == expected
    assert registry.get(next(iter(registry.tasks))).language == language


# --------------------------------------------------------------------------- #
# app: GET /api/tasks/{task_id} payload
# --------------------------------------------------------------------------- #


def _register(task_id, *, status="running", asset_type="stock", ticker="AAPL", started_at=0.0,
              finished_at=None, tail=0):
    state = runner.TaskState(
        task_id=task_id,
        ticker=ticker,
        date=_PAST_DATE,
        asset_type=asset_type,
        started_at=started_at,
        language="en",
        status=status,
        attempt=2,
        max_attempts=3,
        report_dir=f"{ticker}_20260610_120000" if status == "done" else None,
        error="run_robust exited 1" if status == "error" else None,
        finished_at=finished_at,
        pid=4242 if status != "error" else None,
    )
    state.log_tail.extend(f"line {i}" for i in range(tail))
    runner.registry.tasks[task_id] = state
    return state


@pytest.mark.unit
def test_r16_app_task_running_payload_has_no_report_url(client):
    """A running task reports its progress fields, a live elapsed time and
    ``report_url=None`` (the link only exists for a finished run)."""
    _register("t-run", started_at=time.time() - 5.0)
    response = client.get("/api/tasks/t-run")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["report_url"] is None
    assert payload["error"] is None
    assert 4.9 <= payload["elapsed_s"] <= 7.0
    assert (payload["attempt"], payload["max_attempts"]) == (2, 3)
    assert payload["ticker"] == "AAPL" and payload["date"] == _PAST_DATE
    assert payload["asset_type"] == "stock" and payload["report_dir"] is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asset_type", "expected_url"),
    [
        pytest.param("stock", "#/t/AAPL/2026-06-10", id="stock"),
        pytest.param("auto", "#/t/AAPL/2026-06-10", id="auto"),
        pytest.param("crypto", "#/t/AAPL/2026-06-10", id="umbrella-crypto-no-suffix"),
        pytest.param("crypto_perp", "#/t/BTCUSDT/2026-06-10_perp", id="perp"),
        pytest.param("crypto_spot", "#/t/BTCUSDT/2026-06-10_spot", id="spot"),
    ],
)
def test_r16_app_task_done_report_url_carries_venue_suffix(client, asset_type, expected_url):
    """A finished run links to the states-log key its venue writes: the
    suffixed file for an explicit Binance venue, the plain date otherwise;
    elapsed time freezes at ``finished_at``."""
    ticker = "BTCUSDT" if asset_type.startswith("crypto_") else "AAPL"
    _register("t-done", status="done", asset_type=asset_type, ticker=ticker,
              started_at=1_000.0, finished_at=1_012.34)
    response = client.get("/api/tasks/t-done")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "done"
    assert payload["report_url"] == expected_url
    assert payload["elapsed_s"] == 12.3
    assert payload["report_dir"] == f"{ticker}_20260610_120000"


@pytest.mark.unit
def test_r16_app_task_error_payload_exposes_reason_and_last_20_log_lines(client):
    """A failed task carries its reason and the newest 20 log lines; there is
    never a report link for a run that did not succeed."""
    _register("t-err", status="error", started_at=1_000.0, finished_at=1_060.0, tail=50)
    response = client.get("/api/tasks/t-err")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error"] == "run_robust exited 1"
    assert payload["report_url"] is None
    assert payload["elapsed_s"] == 60.0
    assert payload["log_tail"] == [f"line {i}" for i in range(30, 50)]


@pytest.mark.unit
def test_r16_app_task_unknown_id_is_404(client):
    """An unknown task id is an honest 404 (history is in-memory and lost on
    restart) — not an empty 200 payload."""
    response = client.get("/api/tasks/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "unknown task"}
