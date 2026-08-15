"""Tests for the preflight proxy probe target resolution.

The probe used to read only HTTPS_PROXY/HTTP_PROXY and fall back to a
hard-coded 127.0.0.1:1080 — but the project's first-class proxy knob is
``SOCKS5_PROXY`` (.env.example, config-check), so a correctly-configured user
saw a false "proxy unreachable" against a port nothing listens on. The
resolution order now mirrors the project's proxy surface: SOCKS5_PROXY, then
ALL_PROXY, then HTTPS_PROXY, then HTTP_PROXY, defaulting to the historical
local listener when nothing is set.
"""

from __future__ import annotations

import pytest

from yiagents.monitoring.preflight import _resolve_proxy_probe

_PROXY_ENV_VARS = ("SOCKS5_PROXY", "ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY")


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch):
    for var in _PROXY_ENV_VARS + ("https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.unit
class TestProxyProbeTarget:
    def test_socks5_proxy_is_first_class(self, monkeypatch):
        monkeypatch.setenv("SOCKS5_PROXY", "socks5h://127.0.0.1:7890")
        assert _resolve_proxy_probe() == ("127.0.0.1", 7890)

    def test_socks5_proxy_wins_over_https_proxy(self, monkeypatch):
        monkeypatch.setenv("SOCKS5_PROXY", "socks5://10.0.0.1:1080")
        monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.2:8080")
        assert _resolve_proxy_probe() == ("10.0.0.1", 1080)

    def test_all_proxy_fallback(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "socks5h://proxy.lan:1080")
        assert _resolve_proxy_probe() == ("proxy.lan", 1080)

    def test_https_proxy_then_http_proxy(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8118")
        assert _resolve_proxy_probe() == ("127.0.0.1", 8118)
        monkeypatch.delenv("HTTPS_PROXY")
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8119")
        assert _resolve_proxy_probe() == ("127.0.0.1", 8119)

    def test_default_when_nothing_set(self):
        # Historical fallback kept: a local V2Ray/Xray SOCKS listener.
        assert _resolve_proxy_probe() == ("127.0.0.1", 1080)

    def test_userinfo_and_scheme_stripped(self, monkeypatch):
        monkeypatch.setenv(
            "SOCKS5_PROXY", "socks5h://user:secret@10.1.2.3:1080?timeout=5"
        )
        assert _resolve_proxy_probe() == ("10.1.2.3", 1080)
