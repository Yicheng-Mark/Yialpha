"""Round-14 coverage: monitoring.preflight run_health, reporting sections, dashboard edges.

Hermeticity: every network surface ``run_health`` touches (the TCP proxy probe,
the yfinance pull, the httpx provider probe) is replaced with in-memory stubs,
and an autouse fixture scrubs every env knob preflight reads — the developer's
real ``.env`` is loaded at ``yialpha`` import time (``find_dotenv(usecwd=True)``)
and the shared conftest injects placeholder API keys, so without the scrub a
test's provider/proxy resolution would depend on the machine it runs on.
"""

from __future__ import annotations

import importlib as real_importlib
import sys
import types
from pathlib import Path

import httpx
import pytest

import yialpha.monitoring.preflight as preflight
from yialpha.backtest.engine import BacktestResult
from yialpha.backtest.metrics import BacktestMetrics
from yialpha.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from yialpha.llm_clients.key_pool import PROVIDER_API_KEYS_ENV
from yialpha.monitoring.dashboard import render_dashboard, write_dashboard
from yialpha.monitoring.preflight import (
    _resolve_probe_base,
    _resolve_proxy_probe,
    run_health,
)
from yialpha.reporting import write_report_tree

_PROXY_ENV_VARS = (
    "SOCKS5_PROXY", "ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY",
    "socks5_proxy", "all_proxy", "https_proxy", "http_proxy",
)
_BASE_URL_ENV_VARS = (
    "YIALPHA_LLM_BACKEND_URL", "ZHIPU_BASE_URL", "ZHIPU_CN_BASE_URL", "OLLAMA_BASE_URL",
)
# Not in PROVIDER_API_KEY_ENV's single-key map additions the conftest placeholder
# fixture covers — listed explicitly so key presence is never ambient.
_EXTRA_KEY_ENV_VARS = (
    "GROQ_API_KEY", "MISTRAL_API_KEY", "MOONSHOT_API_KEY", "NVIDIA_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
)


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch):
    """Remove every provider/proxy/key env knob preflight consults.

    Tests then setenv exactly the knobs they mean, so provider resolution is
    fully determined by the test regardless of the machine's real .env or the
    conftest placeholder keys.
    """
    names = (
        set(_PROXY_ENV_VARS)
        | set(_BASE_URL_ENV_VARS)
        | set(_EXTRA_KEY_ENV_VARS)
        | {v for v in PROVIDER_API_KEY_ENV.values() if v}
        | set(PROVIDER_API_KEYS_ENV.values())
        | {"YIALPHA_LLM_PROVIDER"}
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _stub_transport(
    monkeypatch,
    *,
    proxy_ok: bool = True,
    yf_result=(1.0, 2.0),
    http_status: int = 200,
    http_exc: Exception | None = None,
):
    """Replace every network surface of run_health; return the call recorder.

    ``yf_result`` doubles as the raise-source: a BaseException instance makes
    the fake Ticker raise it (the yfinance error path).
    """
    conns: list = []
    closed: list = []
    http_calls: list = []
    tickers: list = []

    def create_connection(addr, timeout=None):  # noqa: ANN001
        conns.append((addr, timeout))
        if not proxy_ok:
            raise OSError(f"connect refused: {addr[0]}:{addr[1]}")

        def _close():
            closed.append(addr)

        return types.SimpleNamespace(close=_close)

    class _FakeTicker:
        def __init__(self, ticker: str):
            self.ticker = ticker
            tickers.append(ticker)

        def history(self, **kwargs):  # noqa: ARG002
            if isinstance(yf_result, BaseException):
                raise yf_result
            return yf_result

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = _FakeTicker

    def fake_get(url, headers=None, timeout=None):  # noqa: ANN001
        http_calls.append({"url": url, "headers": headers, "timeout": timeout})
        if http_exc is not None:
            raise http_exc
        return types.SimpleNamespace(status_code=http_status)

    monkeypatch.setattr(preflight, "socket",
                        types.SimpleNamespace(create_connection=create_connection))
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setattr(httpx, "get", fake_get)
    return types.SimpleNamespace(
        conns=conns, closed=closed, http_calls=http_calls, tickers=tickers
    )


class _ImportLibShim:
    """preflight's ``importlib`` stand-in: ImportError for chosen module names."""

    def __init__(self, missing: frozenset[str]):
        self._missing = missing

    def import_module(self, name: str):
        if name in self._missing:
            raise ImportError(f"No module named {name!r}")
        return real_importlib.import_module(name)


def _by_name(result: dict) -> dict[str, dict]:
    return {c["name"]: c for c in result["checks"]}


# ---------------------------------------------------------------------------
# preflight._resolve_probe_base
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveProbeBase:
    def test_unregistered_provider_returns_empty(self):
        # Native providers (anthropic/google/...) and typos alike: no probeable
        # endpoint — run_health turns that into the vacuous-green check.
        assert _resolve_probe_base("anthropic") == ""
        assert _resolve_probe_base("bedrock") == ""
        assert _resolve_probe_base("not-a-provider") == ""

    def test_openai_has_no_default_base(self):
        # The openai spec relies on the SDK default endpoint, so "" is correct.
        assert _resolve_probe_base("openai") == ""

    def test_spec_default_for_glm(self):
        assert _resolve_probe_base("glm") == "https://api.z.ai/api/paas/v4/"

    def test_provider_env_override(self, monkeypatch):
        monkeypatch.setenv("ZHIPU_BASE_URL", "https://coding.example/v4")
        assert _resolve_probe_base("glm") == "https://coding.example/v4"
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
        assert _resolve_probe_base("ollama") == "http://gpu-box:11434/v1"

    def test_backend_url_wins_over_provider_env(self, monkeypatch):
        monkeypatch.setenv("ZHIPU_BASE_URL", "https://coding.example/v4")
        monkeypatch.setenv("YIALPHA_LLM_BACKEND_URL", "https://relay.example/v1")
        assert _resolve_probe_base("glm") == "https://relay.example/v1"

    def test_blank_backend_url_falls_through(self, monkeypatch):
        # Whitespace-only is not a usable override — resolution continues.
        monkeypatch.setenv("YIALPHA_LLM_BACKEND_URL", "   ")
        assert _resolve_probe_base("glm") == "https://api.z.ai/api/paas/v4/"


# ---------------------------------------------------------------------------
# preflight._resolve_proxy_probe — IPv6 regression (round-14 bug fix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProxyProbeIpv6:
    def test_bracketed_ipv6_literal_stripped(self, monkeypatch):
        # Regression: the brackets are RFC 3986 delimiters, not part of the
        # host — "[::1]" is unresolvable and produced a false "proxy
        # unreachable" against a working IPv6 listener.
        monkeypatch.setenv("SOCKS5_PROXY", "socks5://[::1]:1080")
        assert _resolve_proxy_probe() == ("::1", 1080)

    def test_bracketed_ipv6_with_userinfo(self, monkeypatch):
        monkeypatch.setenv("SOCKS5_PROXY", "socks5h://user@[2001:db8::1]:7890")
        assert _resolve_proxy_probe() == ("2001:db8::1", 7890)

    def test_plain_host_unchanged_by_fix(self, monkeypatch):
        monkeypatch.setenv("SOCKS5_PROXY", "socks5://127.0.0.1:1080")
        assert _resolve_proxy_probe() == ("127.0.0.1", 1080)


# ---------------------------------------------------------------------------
# preflight.run_health — deps block
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunHealthDeps:
    def test_all_green_openai_vacuous_probe(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("SOCKS5_PROXY", "socks5h://proxy.test:1080")
        stub = _stub_transport(monkeypatch)

        result = run_health()

        assert [c["name"] for c in result["checks"]] == [
            "dep socks", "dep yfinance", "dep pandas", "dep httpx", "dep dotenv",
            "OPENAI_API_KEY set", "YIALPHA_LLM_PROVIDER set",
            "proxy proxy.test:1080 reachable", "yfinance pulled SPY",
            "openai API reachable",
        ]
        assert all(c["ok"] for c in result["checks"])
        assert result["ok"] is True
        # The openai spec has no probeable base: vacuous green, zero HTTP calls.
        probe = _by_name(result)["openai API reachable"]
        assert probe["hint"] == (
            "no probeable endpoint/key resolved; key presence checked above"
        )
        assert stub.http_calls == []
        # The TCP probe targeted the resolved SOCKS5_PROXY with timeout 3 and
        # closed the socket afterwards.
        assert stub.conns == [(("proxy.test", 1080), 3)]
        assert stub.closed == [("proxy.test", 1080)]
        assert stub.tickers == ["SPY"]

    def test_missing_deps_report_pip_hints(self, monkeypatch):
        monkeypatch.setattr(preflight, "importlib",
                            _ImportLibShim(frozenset({"socks", "pandas"})))
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _stub_transport(monkeypatch)

        result = run_health()
        checks = _by_name(result)

        assert checks["dep socks"]["ok"] is False
        assert checks["dep socks"]["hint"] == 'pip install "requests[socks]"'
        assert checks["dep pandas"]["ok"] is False
        assert checks["dep pandas"]["hint"] == "pip install pandas"
        assert checks["dep yfinance"]["ok"] is True
        assert checks["dep httpx"]["ok"] is True
        assert checks["dep dotenv"]["ok"] is True
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# preflight.run_health — env/key block
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunHealthEnvKey:
    def test_unset_provider_and_key_are_red(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("YIALPHA_LLM_PROVIDER", raising=False)
        _stub_transport(monkeypatch)

        result = run_health()
        checks = _by_name(result)

        assert checks["OPENAI_API_KEY set"]["ok"] is False
        assert checks["OPENAI_API_KEY set"]["hint"] == (
            ".env not loaded — start from the project root (dir with .env)"
        )
        assert checks["YIALPHA_LLM_PROVIDER set"]["ok"] is False
        assert checks["YIALPHA_LLM_PROVIDER set"]["hint"] == ".env missing YIALPHA_LLM_PROVIDER"
        assert result["ok"] is False

    def test_unknown_provider_falls_back_to_generic_key_env_name(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "mystery")
        _stub_transport(monkeypatch)

        result = run_health()
        checks = _by_name(result)

        # No registered key env and no pool: the generic fallback name is red...
        assert checks["MYSTERY API KEY set"]["ok"] is False
        # ...and the provider check stays key-presence-only (spec is None).
        assert checks["mystery API reachable"]["ok"] is True
        assert checks["mystery API reachable"]["hint"] == (
            "no probeable endpoint/key resolved; key presence checked above"
        )

    def test_key_pool_satisfies_key_check_and_feeds_probe(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "glm")
        monkeypatch.setenv("ZHIPU_API_KEYS", "k1, k2 ,k1")  # dedup + trim
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        stub = _stub_transport(monkeypatch)

        result = run_health()
        checks = _by_name(result)

        # has_pool + non-empty pool satisfy the key requirement without the
        # single-key var (GLM Coding Plan setups).
        assert checks["ZHIPU_API_KEY set"]["ok"] is True
        # The probe runs against the spec default (trailing slash stripped)
        # with the FIRST pooled key.
        assert stub.http_calls == [{
            "url": "https://api.z.ai/api/paas/v4/models",
            "headers": {"Authorization": "Bearer k1"},
            "timeout": 12,
        }]
        assert checks["glm API reachable"]["ok"] is True
        assert checks["glm API reachable"]["hint"] == (
            "HTTP 200 from https://api.z.ai/api/paas/v4/"
        )


# ---------------------------------------------------------------------------
# preflight.run_health — proxy + yfinance blocks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunHealthProxyAndYfinance:
    def test_proxy_unreachable_is_red(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("SOCKS5_PROXY", "socks5://proxy.test:1080")
        stub = _stub_transport(monkeypatch, proxy_ok=False)

        result = run_health()
        check = _by_name(result)["proxy proxy.test:1080 reachable"]

        assert check["ok"] is False
        assert check["hint"] == "confirm V2Ray/Xray on that port"
        assert stub.conns == [(("proxy.test", 1080), 3)]
        assert stub.closed == []  # nothing to close on refusal
        assert result["ok"] is False

    def test_empty_history_is_red_with_hint(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _stub_transport(monkeypatch, yf_result=())

        check = _by_name(run_health())["yfinance pulled SPY"]
        assert check["ok"] is False
        assert check["hint"] == "proxy / PySocks / Yahoo rate-limit"

    def test_history_exception_surfaces_truncated_repr(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _stub_transport(monkeypatch, yf_result=RuntimeError("boom " + "x" * 300))

        check = _by_name(run_health())["yfinance pulled SPY"]
        assert check["ok"] is False
        assert check["hint"].startswith("RuntimeError")
        assert len(check["hint"]) == 140  # repr(e)[:140] truncation contract

    def test_custom_ticker_names_the_check(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        stub = _stub_transport(monkeypatch)

        result = run_health("QQQ")

        assert _by_name(result)["yfinance pulled QQQ"]["ok"] is True
        assert stub.tickers == ["QQQ"]


# ---------------------------------------------------------------------------
# preflight.run_health — DeepSeek branch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunHealthDeepSeek:
    def test_probe_200_is_green(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
        stub = _stub_transport(monkeypatch, http_status=200)

        result = run_health()
        check = _by_name(result)["DeepSeek API reachable"]

        assert check["ok"] is True
        assert check["hint"] == "HTTP 200"
        assert stub.http_calls == [{
            "url": "https://api.deepseek.com/v1/models",
            "headers": {"Authorization": "Bearer sk-ds"},
            "timeout": 12,
        }]

    @pytest.mark.parametrize("status", [401, 403])
    def test_probe_key_rejection_is_red(self, monkeypatch, status):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
        _stub_transport(monkeypatch, http_status=status)

        check = _by_name(run_health())["DeepSeek API reachable"]
        assert check["ok"] is False
        assert check["hint"] == f"HTTP {status} (key invalid / no credit)"

    def test_probe_transport_error_is_red(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
        _stub_transport(monkeypatch, http_exc=httpx.ConnectError("no route to host"))

        check = _by_name(run_health())["DeepSeek API reachable"]
        assert check["ok"] is False
        assert "ConnectError" in check["hint"]

    def test_no_key_skips_probe_entirely(self, monkeypatch):
        # Historical branch shape: deepseek without a key performs NO
        # connectivity check (the missing key is already flagged red above).
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        stub = _stub_transport(monkeypatch)

        result = run_health()

        assert "DeepSeek API reachable" not in _by_name(result)
        assert len(result["checks"]) == 9  # 5 deps + 2 env + proxy + yfinance
        assert stub.http_calls == []


# ---------------------------------------------------------------------------
# preflight.run_health — non-DeepSeek provider probe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunHealthProviderProbe:
    @pytest.mark.parametrize(
        ("status", "expected_ok"),
        [(200, True), (302, True), (404, True), (401, False), (403, False),
         (500, False), (502, False)],
    )
    def test_status_semantics(self, monkeypatch, status, expected_ok):
        # Reachability contract: transport 5xx and key rejections are red;
        # any other answer (incl. a missing /models route) is green.
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gk-1")
        stub = _stub_transport(monkeypatch, http_status=status)

        check = _by_name(run_health())["groq API reachable"]

        assert check["ok"] is expected_ok
        assert stub.http_calls[0]["url"] == "https://api.groq.com/openai/v1/models"
        assert stub.http_calls[0]["headers"] == {"Authorization": "Bearer gk-1"}
        assert stub.http_calls[0]["timeout"] == 12
        if status in (401, 403):
            assert "(key invalid / no credit / wrong endpoint tier)" in check["hint"]
        assert check["hint"].startswith(f"HTTP {status} from ")

    def test_transport_error_is_red(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gk-1")
        _stub_transport(monkeypatch, http_exc=httpx.ConnectError("dns failure"))

        check = _by_name(run_health())["groq API reachable"]
        assert check["ok"] is False
        assert "ConnectError" in check["hint"]

    def test_backend_url_overrides_probe_target(self, monkeypatch):
        # Client precedence: YIALPHA_LLM_BACKEND_URL outranks the provider's
        # own base_url_env even inside run_health's probe.
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "glm")
        monkeypatch.setenv("YIALPHA_LLM_BACKEND_URL", "https://relay.test/v1/")
        monkeypatch.setenv("ZHIPU_BASE_URL", "https://coding.test/v4")
        monkeypatch.setenv("ZHIPU_API_KEYS", "k1")
        stub = _stub_transport(monkeypatch)

        run_health()

        assert stub.http_calls[0]["url"] == "https://relay.test/v1/models"

    def test_missing_probe_key_is_vacuous_green(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "groq")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        stub = _stub_transport(monkeypatch)

        check = _by_name(run_health())["groq API reachable"]

        assert check["ok"] is True
        assert "no probeable endpoint/key resolved" in check["hint"]
        assert stub.http_calls == []

    def test_local_provider_is_vacuous_green(self, monkeypatch):
        monkeypatch.setenv("YIALPHA_LLM_PROVIDER", "ollama")
        stub = _stub_transport(monkeypatch)

        result = run_health()
        checks = _by_name(result)

        # Local runtimes: no key env registered (generic fallback name), no
        # probe even though the spec carries a base_url (is_local).
        assert checks["OLLAMA API KEY set"]["ok"] is False
        assert checks["ollama API reachable"]["ok"] is True
        assert stub.http_calls == []


# ---------------------------------------------------------------------------
# reporting.write_report_tree
# ---------------------------------------------------------------------------


def _full_state() -> dict:
    return {
        "market_report": "MKT", "sentiment_report": "SENT", "news_report": "NEWS",
        "fundamentals_report": "FUND", "positioning_report": "POS",
        "investment_debate_state": {
            "bull_history": "BULL", "bear_history": "BEAR", "judge_decision": "JUDGE",
        },
        "trader_investment_plan": "PLAN",
        "risk_debate_state": {
            "aggressive_history": "AGG", "conservative_history": "CONS",
            "neutral_history": "NEU", "judge_decision": "RISKJUDGE",
        },
        "final_trade_decision": "FINAL",
    }


@pytest.mark.unit
class TestWriteReportTree:
    def test_full_tree_writes_every_section_file(self, tmp_path):
        write_report_tree(_full_state(), "AAPL", tmp_path)

        expected = {
            "1_analysts/market.md": "MKT",
            "1_analysts/sentiment.md": "SENT",
            "1_analysts/news.md": "NEWS",
            "1_analysts/fundamentals.md": "FUND",
            "1_analysts/positioning.md": "POS",
            "2_research/bull.md": "BULL",
            "2_research/bear.md": "BEAR",
            "2_research/manager.md": "JUDGE",
            "3_trading/trader.md": "PLAN",
            "4_risk/aggressive.md": "AGG",
            "4_risk/conservative.md": "CONS",
            "4_risk/neutral.md": "NEU",
            "5_portfolio/decision.md": "FINAL",
        }
        for rel, text in expected.items():
            assert (tmp_path / rel).read_text(encoding="utf-8") == text, rel

        # decision.md prefers final_trade_decision over the pre-overlay judge.
        assert (tmp_path / "5_portfolio" / "decision.md").read_text() != "RISKJUDGE"

    def test_full_tree_section_ordering(self, tmp_path):
        out = write_report_tree(_full_state(), "AAPL", tmp_path)
        complete = out.read_text(encoding="utf-8")

        markers = [
            "## I. Analyst Team Reports",
            "## II. Research Team Decision",
            "## III. Trading Team Plan",
            "## IV. Risk Management Team Decision",
            "## V. Final Risk-Adjusted Decision",
        ]
        positions = [complete.index(m) for m in markers]
        assert positions == sorted(positions)
        for name in ("Sentiment Analyst", "Fundamentals Analyst", "Positioning Analyst",
                     "Bull Researcher", "Bear Researcher", "Research Manager",
                     "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            assert name in complete, name

    def test_empty_state_writes_only_consolidated_report(self, tmp_path):
        out = write_report_tree({}, "SPY", tmp_path)

        assert out == tmp_path / "complete_report.md"
        files = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
        assert files == [Path("complete_report.md")]
        content = out.read_text(encoding="utf-8")
        assert content.startswith("# Trading Analysis Report: SPY\n")
        assert "Generated: " in content
        assert "## I." not in content and "## V." not in content

    def test_falsy_section_values_are_absent(self, tmp_path):
        state = {
            "sentiment_report": "",
            "investment_debate_state": {"bull_history": "", "bear_history": "BEAR"},
            "risk_debate_state": {"aggressive_history": "", "judge_decision": "RISKJUDGE"},
        }
        out = write_report_tree(state, "AAPL", tmp_path)
        complete = out.read_text(encoding="utf-8")

        # Empty-string sections create no files/directories at all.
        assert not (tmp_path / "1_analysts").exists()
        assert not (tmp_path / "2_research" / "bull.md").exists()
        assert (tmp_path / "2_research" / "bear.md").read_text(encoding="utf-8") == "BEAR"
        assert not (tmp_path / "4_risk").exists()
        # ...but the risk judge still lands in section V via the fallback.
        assert (tmp_path / "5_portfolio" / "decision.md").read_text(encoding="utf-8") == (
            "RISKJUDGE"
        )
        assert "## IV." not in complete
        assert "## V. Final Risk-Adjusted Decision" in complete

    @pytest.mark.parametrize(
        ("quality", "needle"),
        [
            ({"stale_cache_count": 1}, "1 data request served STALE cache"),
            ({"stale_cache_count": 3}, "3 data requests served STALE cache"),
            ({"core_sentinel_count": 1}, "1 core data category returned no usable data"),
            ({"optional_sentinel_count": 1}, "1 optional enrichment category unavailable"),
        ],
    )
    def test_data_quality_pluralization(self, tmp_path, quality, needle):
        state = {"market_report": "MKT", "data_quality": quality}
        out = write_report_tree(state, "AAPL", tmp_path)
        assert needle in out.read_text(encoding="utf-8")

    def test_stale_and_core_banners_coexist(self, tmp_path):
        state = {
            "market_report": "MKT",
            "data_quality": {
                "core_sentinel_count": 1,
                "stale_cache_count": 2,
                "sentinels": [{"method": "get_stock_data", "kind": "no_data",
                               "detail": "vendors timed out"}],
            },
        }
        out = write_report_tree(state, "AAPL", tmp_path)
        content = out.read_text(encoding="utf-8")
        assert "DEGRADED RUN" in content
        assert "1 core data category returned no usable data" in content
        assert "2 data requests served STALE cache after a vendor failure." in content
        assert "`get_stock_data` (no_data): vendors timed out" in content

    def test_sentinels_without_counts_are_silent(self, tmp_path):
        # The section is count-driven: sentinel details alone (all counts zero)
        # render nothing — matches the healthy-run contract.
        state = {
            "market_report": "MKT",
            "data_quality": {"sentinels": [{"method": "get_news", "kind": "no_data",
                                            "detail": "empty"}]},
        }
        out = write_report_tree(state, "AAPL", tmp_path)
        assert "Data Quality" not in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# monitoring.dashboard edge branches
# ---------------------------------------------------------------------------


def _mk_result(
    ticker: str = "AAPL", with_metrics: bool = True, equity: list | None = None
) -> BacktestResult:
    metrics = None
    if with_metrics:
        metrics = BacktestMetrics(
            total_return=0.15, cagr=0.15, volatility=0.15, sharpe=1.2, sortino=1.6,
            max_drawdown=-0.08, calmar=1.875, deflated_sharpe=0.6,
            alpha_vs_buyhold=0.05, num_periods=252, periods_per_year=252,
        )
    equity = [100_000, 105_000, 110_000] if equity is None else equity
    dates = ["2024-01-01", "2024-04-01", "2024-07-01"][: len(equity)]
    return BacktestResult(
        ticker=ticker, initial_capital=100_000, holding_days=5,
        equity=equity, equity_dates=dates,
        trades=[], benchmark_equity=[100_000, 105_000], benchmark_name="b&h",
        metrics=metrics,
    )


@pytest.mark.unit
class TestDashboardEdges:
    def test_no_results_renders_placeholder(self):
        page = render_dashboard([])
        assert "No backtest result supplied." in page
        assert "<svg" not in page
        assert "<h2>Metrics</h2>" not in page
        assert "Drawdown alert" not in page  # no primary -> no alert path

    def test_no_equity_and_no_metrics_placeholders(self):
        page = render_dashboard(_mk_result(with_metrics=False, equity=[]))
        assert "No equity data." in page
        assert "No metrics available." in page
        assert "<svg" not in page
        assert "Drawdown alert" not in page  # metrics None -> alert skipped

    @pytest.mark.parametrize(
        "portfolio_state",
        [[], {"positions": {}}, {"positions": {"MSFT": 0}}],
        ids=["empty-list", "empty-positions", "all-zero-positions"],
    )
    def test_positions_table_absent_edges(self, portfolio_state):
        page = render_dashboard(_mk_result(), portfolio_state=portfolio_state)
        assert "Current positions" not in page

    def test_positions_table_positive_control(self):
        page = render_dashboard(_mk_result(), portfolio_state={"positions": {"AAPL": 25000}})
        assert "Current positions" in page
        assert "25,000" in page

    def test_runs_table_skips_metricless_results(self):
        page = render_dashboard([_mk_result("AAPL"), _mk_result("MSFT", with_metrics=False)])
        assert "Runs" in page
        assert "AAPL" in page
        # The metricless secondary run is filtered out of the runs table.
        assert "MSFT" not in page

    def test_write_dashboard_explicit_path_leaves_no_tmp(self, tmp_path):
        target = tmp_path / "nested" / "dash.html"
        out = write_dashboard(_mk_result(), path=target)

        assert out == target
        text = target.read_text(encoding="utf-8")
        assert "<html" in text and "AAPL" in text
        # The pid-suffixed tmp sibling is cleaned up on every path (finally).
        assert [p.name for p in target.parent.iterdir()] == ["dash.html"]
        # An explicit path must not create the results_dir default tree.
        assert not (tmp_path / "monitoring").exists()
