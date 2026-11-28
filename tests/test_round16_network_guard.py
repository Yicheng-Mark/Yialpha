"""The suite's network guard (``tests/conftest.py::NetworkGuard``) is itself tested.

The guard is the only thing standing between "hermetic" and "green because
the dev-box proxy happened to be up", so each of its transport layers gets a
test that fails if that layer stops being covered. Every transport a vendor in
this project can sit on is exercised through the API a caller would really
use, never by poking the patched names.

The classes are taken from the autouse fixture's own instance rather than
imported from ``conftest``: pytest's module name for a conftest file depends
on the import mode, and a second copy of the module would mean a second,
different ``NetworkAccessError``.
"""
from __future__ import annotations

import asyncio
import re
import socket
import sys

import pytest
import requests

# Port 9 (discard). Nothing listens there; the guard must refuse before the
# OS is asked, so a refusal that regresses surfaces as ConnectionRefusedError
# or a timeout — never as a pass.
_NOWHERE = ("127.0.0.1", 9)
# A loopback proxy on a port nothing listens on, standing in for the dev
# box's machine-wide proxy (127.0.0.1:8086).
_PROXY_PORT = 65123
_PROXY_URL = f"http://127.0.0.1:{_PROXY_PORT}"
_GUARD_MESSAGE = "real network I/O"


def _swallow(fn):
    """Call ``fn`` and swallow the guard's error, exactly like a vendor whose
    ``except Exception`` degrades to NO_DATA would."""
    try:
        fn()
    except RuntimeError as exc:
        assert _GUARD_MESSAGE in str(exc), f"not the guard: {exc!r}"


@pytest.mark.unit
class TestGuardRefusesEachTransport:
    def test_r16_plain_socket_connect_and_connect_ex(self, _hermetic_no_network):
        sock = socket.socket()
        try:
            with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
                sock.connect(_NOWHERE)
            # connect_ex must raise too: an errno return would read as a
            # "connection failed" result instead of a hermeticity failure.
            with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
                sock.connect_ex(_NOWHERE)
        finally:
            sock.close()
        assert len(_hermetic_no_network.attempts) == 2
        assert "connect to ('127.0.0.1', 9)" in _hermetic_no_network.attempts[0]
        _hermetic_no_network.attempts.clear()  # this test IS the attempt

    def test_r16_requests_library(self, _hermetic_no_network):
        # trust_env=False pins WHERE the connect is aimed. Left at the default
        # the client consults the machine's proxy settings first, and whether
        # a loopback target is sent to the proxy depends on the host's
        # NO_PROXY / registry override list (green on the dev box, red on a
        # scrubbed CI runner). The proxy case has its own test below.
        with requests.Session() as session:
            session.trust_env = False
            with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
                session.get("http://127.0.0.1:9/", timeout=1)
        assert "connect to ('127.0.0.1', 9)" in _hermetic_no_network.attempts[0]
        _hermetic_no_network.attempts.clear()

    def test_r16_httpx_sync_client(self, _hermetic_no_network):
        httpx = pytest.importorskip("httpx")
        with (
            httpx.Client(trust_env=False, timeout=1) as client,
            pytest.raises(RuntimeError, match=_GUARD_MESSAGE),
        ):
            client.get("http://127.0.0.1:9/")
        [attempt] = _hermetic_no_network.attempts
        assert "connect to ('127.0.0.1', 9)" in attempt
        _hermetic_no_network.attempts.clear()

    def test_r16_asyncio_connect(self, _hermetic_no_network):
        """``asyncio.open_connection`` on Windows' default Proactor loop goes
        through ``ConnectEx`` and never calls ``socket.connect``."""
        async def connect():
            await asyncio.open_connection(*_NOWHERE)

        with (
            asyncio.Runner() as runner,
            pytest.raises(RuntimeError, match=_GUARD_MESSAGE),
        ):
            runner.run(connect())
        [attempt] = _hermetic_no_network.attempts
        assert "('127.0.0.1', 9)" in attempt
        if sys.platform == "win32":
            assert "asyncio connect" in attempt
        _hermetic_no_network.attempts.clear()

    def test_r16_httpx_async_client(self, _hermetic_no_network):
        httpx = pytest.importorskip("httpx")

        async def fetch():
            async with httpx.AsyncClient(trust_env=False, timeout=1) as client:
                await client.get("http://127.0.0.1:9/")

        with asyncio.Runner() as runner, pytest.raises(BaseException) as excinfo:
            runner.run(fetch())
        # anyio reports the failure from inside a TaskGroup, i.e. wrapped in
        # an ExceptionGroup; the guard's error must be one of its leaves.
        assert excinfo.group_contains(RuntimeError, match=_GUARD_MESSAGE)
        [attempt] = _hermetic_no_network.attempts
        assert "connect to ('127.0.0.1', 9)" in attempt
        _hermetic_no_network.attempts.clear()

    @pytest.fixture
    def _loopback_proxy_env(self, monkeypatch):
        """Route every request through a proxy on an unused loopback port,
        with NO_PROXY cleared — deterministic on any machine."""
        monkeypatch.setenv("HTTP_PROXY", _PROXY_URL)
        monkeypatch.setenv("HTTPS_PROXY", _PROXY_URL)
        for var in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(var, raising=False)
        return _PROXY_URL

    def test_r16_loopback_proxy_hop_is_refused_requests(
        self, _hermetic_no_network, _loopback_proxy_env
    ):
        """A machine-wide proxy is usually loopback too (127.0.0.1:8086 on the
        dev box). The hop TO that proxy is real network access and must be
        refused: a guard that whitelisted loopback addresses would wave
        through every proxied request."""
        with (  # requests.Session(): trust_env=True (default)
            requests.Session() as session,
            pytest.raises(RuntimeError, match=_GUARD_MESSAGE),
        ):
            session.get("http://example.invalid/", timeout=1)
        [attempt] = _hermetic_no_network.attempts
        assert attempt.startswith(f"connect to ('127.0.0.1', {_PROXY_PORT})")
        _hermetic_no_network.attempts.clear()

    def test_r16_loopback_proxy_hop_is_refused_httpx(
        self, _hermetic_no_network, _loopback_proxy_env
    ):
        """Same hop for httpx — the transport every LLM SDK in this project
        sits on (openai / anthropic / google-genai all use it)."""
        httpx = pytest.importorskip("httpx")
        with (
            httpx.Client(timeout=1) as client,  # trust_env=True (default)
            pytest.raises(RuntimeError, match=_GUARD_MESSAGE),
        ):
            client.get("https://example.invalid/")
        [attempt] = _hermetic_no_network.attempts
        assert attempt.startswith(f"connect to ('127.0.0.1', {_PROXY_PORT})")
        _hermetic_no_network.attempts.clear()

    def test_r16_curl_cffi_session(self, _hermetic_no_network):
        """libcurl opens its own sockets: the socket layer never sees it."""
        curl_requests = pytest.importorskip("curl_cffi.requests")
        with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
            curl_requests.Session().get("https://example.invalid/", timeout=1)
        [attempt] = _hermetic_no_network.attempts
        assert "curl_cffi GET https://example.invalid/" in attempt
        _hermetic_no_network.attempts.clear()

    def test_r16_yfinance_default_session_is_guarded(self, _hermetic_no_network):
        """The seam that matters: yfinance builds a curl_cffi session whenever
        curl_cffi is installed, and curl_cffi is a hard yfinance dependency."""
        pytest.importorskip("yfinance")
        from yfinance import data as yf_data

        with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
            yf_data.new_session().get(
                "https://query1.finance.yahoo.com/v8/finance/chart/X", timeout=1
            )
        [attempt] = _hermetic_no_network.attempts
        assert "curl_cffi GET " in attempt
        _hermetic_no_network.attempts.clear()


@pytest.mark.unit
class TestGuardPlumbing:
    def test_r16_socketpair_is_not_network_access(self, _hermetic_no_network):
        """anyio's TestClient portal and asyncio's self-pipe rely on
        ``socket.socketpair()``; on Windows that is a loopback ``connect()``."""
        left, right = socket.socketpair()
        try:
            left.sendall(b"ping")
            assert right.recv(4) == b"ping"
        finally:
            left.close()
            right.close()
        assert _hermetic_no_network.attempts == []

    def test_r16_socketpair_failure_leaves_no_blanket_permission(
        self, _hermetic_no_network
    ):
        """A ``socketpair()`` that dies midway must not leave its thread
        whitelisted for real connections (the flag is cleared in ``finally``).

        A second guard is installed on its own MonkeyPatch *after* the flaky
        socketpair is in place, so that guard captures the flaky one; the
        flaky one delegates to the autouse guard's wrapper when healthy."""
        guard = type(_hermetic_no_network)()
        patcher = pytest.MonkeyPatch()
        outer_socketpair = socket.socketpair  # the autouse guard's wrapper
        calls = {"n": 0}

        def _flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("simulated socketpair failure")
            return outer_socketpair(*args, **kwargs)

        patcher.setattr(socket, "socketpair", _flaky)
        guard.install(patcher)
        try:
            with pytest.raises(OSError, match="simulated socketpair failure"):
                socket.socketpair()
            sock = socket.socket()
            try:
                with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
                    sock.connect(_NOWHERE)
            finally:
                sock.close()
            # A healthy socketpair still passes through both guards.
            left, right = socket.socketpair()
            left.close()
            right.close()
        finally:
            patcher.undo()
        assert calls["n"] == 2
        [attempt] = guard.attempts
        assert "connect to" in attempt
        # The outer guard saw no attempt of its own: the nested one refused.
        assert _hermetic_no_network.attempts == []


@pytest.mark.unit
class TestGuardTeardown:
    def test_r16_swallowed_attempt_still_fails_the_test(self, _hermetic_no_network):
        """A second, independent guard on its own MonkeyPatch; the autouse one
        around this test stays installed and must record nothing (the nested
        guard refuses the call before the outer one is reached)."""
        guard = type(_hermetic_no_network)()
        patcher = pytest.MonkeyPatch()
        guard.install(patcher)
        try:
            _swallow(lambda: socket.create_connection(_NOWHERE, timeout=1))
            assert len(guard.attempts) == 1, "the attempt must have been recorded"
            with pytest.raises(pytest.fail.Exception, match="1 real network attempt"):
                guard.assert_offline()
            # The check is on the record, not on what the body caught.
            guard.attempts.clear()
            guard.assert_offline()
        finally:
            patcher.undo()
        assert _hermetic_no_network.attempts == []

    def test_r16_record_names_the_calling_line(self, _hermetic_no_network):
        guard = type(_hermetic_no_network)()
        patcher = pytest.MonkeyPatch()
        guard.install(patcher)
        try:
            _swallow(lambda: socket.create_connection(_NOWHERE, timeout=1))
        finally:
            patcher.undo()
        [attempt] = guard.attempts
        frames = attempt.split(" via ", 1)[1].split(" <- ")
        # Innermost project frame first: the line that made the call.
        assert re.match(
            r"tests[\\/]test_round16_network_guard\.py:\d+ <lambda>$", frames[0]
        ), frames[0]
        # ...and the chain ends at the test itself. pytest / plugin internals
        # are not project frames even when the virtualenv lives INSIDE the
        # repo tree (its site-packages is then "under the project root").
        assert frames[-1].endswith("test_r16_record_names_the_calling_line"), frames[-1]
        assert not any(
            "site-packages" in frame or "dist-packages" in frame for frame in frames
        ), frames

    def test_r16_clean_guard_passes(self, _hermetic_no_network):
        type(_hermetic_no_network)().assert_offline()
