"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import os
import socket
import sys
import threading
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Test markers (unit / integration) are declared in pyproject.toml's
# [tool.pytest.ini_options] alongside --strict-markers, so they don't need to
# be re-registered here.

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1]) + os.sep
_THIS_FILE = os.path.normcase(os.path.abspath(__file__))
_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # `or` not a .get default: an env var present but empty (e.g. a key left
        # blank in a .env copied from .env.example) must still get the placeholder.
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


class NetworkAccessError(RuntimeError):
    """A unit test attempted real network I/O."""


def _is_project_frame(filename: str) -> bool:
    """Is this stack frame the repo's own code rather than a dependency's?

    A virtualenv created inside the repo tree (``.venv/``) puts its
    ``site-packages`` "under the project root", so the prefix alone also
    matches pytest and plugin internals. The guard's own frames are excluded
    for the same reason: they sit between the test and the refused call.
    """
    if not filename.startswith(_PROJECT_ROOT):
        return False
    if os.path.normcase(os.path.abspath(filename)) == _THIS_FILE:
        return False
    parts = filename[len(_PROJECT_ROOT):].replace("\\", "/").split("/")
    return not any(part in ("site-packages", "dist-packages") for part in parts)


class NetworkGuard:
    """Refuse and record real outbound network I/O while installed.

    Mocking a vendor *function* is not the same as being offline. Round-16's
    socket audit found 22 tests whose vendor seam was only partly patched
    (``_cik_for_ticker`` imported into ``sec_ownership``, an unstubbed Tavily
    ``requests.post``, prefetch threads outliving the test). They were green
    on a dev box whose proxy was up — the request really went out — and turned
    red the day the proxy was down.

    Three transport layers are covered, because a vendor only has to sit on
    one that is not covered to walk past the others:

    * ``socket.socket.connect`` / ``connect_ex`` — requests / urllib3 / httpx
      (sync) / PySocks, including the hop to the loopback proxy itself;
    * ``BaseProactorEventLoop.sock_connect`` — Windows' default asyncio loop
      connects through ``ConnectEx`` and never calls ``socket.connect``;
    * ``curl_cffi`` sessions — libcurl opens its own sockets. yfinance builds
      a curl_cffi session whenever the package is installed, and the package
      is a hard dependency of yfinance.

    Two mechanisms, because most vendors catch ``Exception`` and degrade:

    * the refused call raises :class:`NetworkAccessError` where it is made;
    * every attempt is *recorded* and re-checked at teardown
      (:meth:`assert_offline`), so an attempt whose exception the code under
      test swallowed still fails the test.

    Windows has no AF_UNIX socketpair: the stdlib emulates it with a loopback
    listener + ``connect()``. That call is in-process plumbing (anyio's
    TestClient portal, asyncio / multiprocessing self-pipes), so connections
    made from inside ``socket.socketpair()`` are let through — a loopback
    address alone is NOT a pass, because the dev-box proxy is loopback too.

    Limits, by design: in-process only (a child interpreter started by a test
    does not inherit it) and it does not intercept DNS resolution.
    """

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._in_socketpair = threading.local()

    def _refuse(self, what: str) -> NetworkAccessError:
        """Record ``what`` and return the exception for the caller to raise."""
        # Innermost project frames only: they name the vendor seam to mock.
        frames = [
            f"{os.path.relpath(f.filename, _PROJECT_ROOT)}:{f.lineno} {f.name}"
            for f in traceback.extract_stack(limit=80)
            if _is_project_frame(f.filename)
        ]
        self.attempts.append(f"{what} via {' <- '.join(reversed(frames[-6:]))}")
        return NetworkAccessError(
            f"test attempted real network I/O ({what}); mock the vendor seam "
            "instead (see _hermetic_no_network in tests/conftest.py)"
        )

    def install(self, monkeypatch) -> None:
        real_connect = socket.socket.connect
        real_socketpair = socket.socketpair

        def _socketpair(*args, **kwargs):
            self._in_socketpair.active = True
            try:
                return real_socketpair(*args, **kwargs)
            finally:
                self._in_socketpair.active = False

        def _connect(sock, address):
            if getattr(self._in_socketpair, "active", False):
                return real_connect(sock, address)
            raise self._refuse(f"connect to {address!r}")

        def _connect_ex(sock, address):
            _connect(sock, address)  # raises before any errno is computed

        monkeypatch.setattr(socket, "socketpair", _socketpair)
        monkeypatch.setattr(socket.socket, "connect", _connect)
        monkeypatch.setattr(socket.socket, "connect_ex", _connect_ex)

        if sys.platform == "win32":
            from asyncio import proactor_events

            async def _sock_connect(loop, sock, address):
                raise self._refuse(f"asyncio connect to {address!r}")

            monkeypatch.setattr(
                proactor_events.BaseProactorEventLoop, "sock_connect", _sock_connect
            )

        try:
            from curl_cffi import requests as curl_requests
        except ImportError:  # pragma: no cover - depends on the environment
            pass
        else:
            def _curl_request(session, *args, **kwargs):
                method = args[0] if args else kwargs.get("method", "?")
                url = args[1] if len(args) > 1 else kwargs.get("url", "?")
                raise self._refuse(f"curl_cffi {method} {url}")

            for cls in (curl_requests.Session, curl_requests.AsyncSession):
                monkeypatch.setattr(cls, "request", _curl_request)

    def assert_offline(self) -> None:
        if self.attempts:
            pytest.fail(
                f"{len(self.attempts)} real network attempt(s) made during this "
                "test (the error was swallowed, so the body may have passed):\n  "
                + "\n  ".join(self.attempts[:8]),
                pytrace=False,
            )


@pytest.fixture(autouse=True)
def _hermetic_no_network(monkeypatch, request):
    """Keep the suite offline: install :class:`NetworkGuard` around every test.

    Yields the guard, so a test can request this fixture by name to inspect
    ``attempts``. ``integration`` tests (``-m integration``) are the opt-out.
    """
    if request.node.get_closest_marker("integration") is not None:
        yield None
        return
    guard = NetworkGuard()
    guard.install(monkeypatch)
    yield guard
    guard.assert_offline()


@pytest.fixture(autouse=True)
def _isolate_config():
    """Reset the dataflows config before and after each test.

    ``set_config`` merges (it never clears keys absent from the override), so a
    test that sets e.g. ``tool_vendors`` would otherwise leak into later tests
    and make routing behavior order-dependent. Reset the context outright so
    every test starts from a clean DEFAULT_CONFIG.
    """
    from yialpha.dataflows.config import reset_config

    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _isolated_vendor_cache(_isolate_config, tmp_path):
    """Point ``data_cache_dir`` at a per-test tmp directory.

    Vendor disk caches (OHLCV per-symbol CSVs, sec/fred/alphavantage/yfnews)
    must never read from — or write into — the
    developer's real ``~/.yialpha/cache`` during tests: a fresh real-world
    cache file would bypass a test's mocked network path, and a mocked
    response would pollute the real cache. Tests that manage their own
    cache dir call ``set_config`` in the test body, which runs after this
    fixture and therefore wins.

    Takes ``_isolate_config`` as an explicit argument so this fixture's
    ``set_config`` is guaranteed to run after the config reset (previously
    only guaranteed by alphabetical autouse ordering).
    """
    from yialpha.dataflows.config import set_config

    set_config({"data_cache_dir": str(tmp_path / "vendor-cache")})


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch(
        "yialpha.llm_clients.factory.create_llm_client",
        return_value=client,
    ):
        yield client


@pytest.fixture(autouse=True)
def _isolated_binance_history_memo():
    """Drop the in-process Binance closed-window memo around every test.

    Without the reset, a mocked page fetched by an earlier test would serve
    every later test whose fetch lands on the same (path, params, window)
    key, and that test's own mock would never fire.
    """
    from yialpha.dataflows.binance import reset_history_memo_for_test

    reset_history_memo_for_test()
    yield
    reset_history_memo_for_test()


@pytest.fixture(autouse=True)
def _isolated_quality_run_context():
    """Reset the data-quality sentinel ContextVars around every test.

    ``record_sentinel`` lazily *sets* a fresh event list in the calling
    context when none is bound (production always binds one via
    ``ensure_run_context`` at graph submit and clears it via ``reset_quality``
    after the snapshot). A test that exercises a sentinel-recording failure
    path without going through the graph therefore leaves that list bound in
    the shared pytest context, and any LATER test that reads
    ``snapshot_quality()`` — e.g. a direct ``_apply_risk_overlay`` call —
    classifies the stale events as this run's own DEGRADED evidence
    (observed: test_atr_memoization.py → test_candidate_authority.py
    ordering flipped APPROVED tickets to VETOED; each file passes alone).
    """
    from yialpha.dataflows.quality import reset_quality

    reset_quality()
    yield
    reset_quality()


@pytest.fixture(autouse=True)
def _perp_bundle_off_by_default(_isolate_config):
    """Hermetic tests: the deterministic perp market bundle (default ON in
    production for crypto_perp market analysts) issues real network
    prefetches. Keep it OFF here unless a test explicitly opts in with
    mocked fetchers (``set_config({"perp_market_bundle": True})`` in the
    test body — set_config merges, so the opt-in wins over this fixture).

    Depends on ``_isolate_config`` explicitly (its reset restores the
    production default True, so this fixture must run after it — previously
    only guaranteed by alphabetical autouse ordering).
    """
    from yialpha.dataflows.config import set_config

    set_config({"perp_market_bundle": False})
    yield


@pytest.fixture(autouse=True)
def _runtime_prefetch_bundles_off(_isolate_config):
    """Hermetic tests: the deterministic fundamentals bundle (default ON in
    production) issues real vendor prefetches through the router. Keep it
    OFF here unless a test explicitly opts in with mocked fetchers
    (``set_config({"fundamentals_bundle": True})`` in the test body —
    set_config merges, so the opt-in wins over this fixture).

    Takes ``_isolate_config`` as an explicit argument: the reset there
    restores the production default True, so this fixture must run after it
    to hold the flags off. (Previously this ordering was only guaranteed by
    alphabetical autouse ordering — the fixture was *named* to sort after
    ``_isolate_config``; the explicit dependency makes it name-independent.)
    """
    from yialpha.dataflows.config import set_config

    # Advisory regime_context fetches OHLCV independently of regime_state.
    # Tests exercising that feature must opt in with their own mocked data.
    set_config({"fundamentals_bundle": False, "regime_context": False})
    yield


@pytest.fixture(autouse=True)
def _optional_crypto_categories_off(_isolate_config):
    """Hermetic tests: optional crypto categories default OFF in production,
    but a dev-box ``.env`` (``YIALPHA_CRYPTO_NATIVE=true`` /
    ``YIALPHA_CRYPTO_FEAR_GREED=true``) is baked into ``DEFAULT_CONFIG`` at
    import time, so ``reset_config`` restores the *dev* defaults, not the
    production ones — and flag-off contracts (e.g. crypto runs drop the
    fundamentals analyst) go red locally while CI stays green. Pin both to
    the production default; tests opt in with ``set_config`` in the body
    (set_config merges, so the opt-in wins over this fixture). Takes
    ``_isolate_config`` as an explicit argument: the reset there re-bakes
    the .env values, so this fixture must run after it (previously only
    guaranteed by alphabetical autouse ordering).
    """
    from yialpha.dataflows.config import set_config

    set_config({"crypto_native": False, "crypto_fear_greed": False})
    yield


@pytest.fixture(autouse=True)
def _runtime_ledger_isolated(_isolate_config, tmp_path):
    """Point the V2 ledger DB at a per-test tmp file and hold the V2.1/V2.2
    record/shadow flags OFF (production defaults are ON).

    Same contract as ``_perp_bundle_off_by_default`` /
    ``_runtime_prefetch_bundles_off``: tests opt in with ``set_config`` in
    the test body (set_config merges, so the opt-in wins). Takes
    ``_isolate_config`` as an explicit argument: the reset there restores
    the production defaults, so this fixture must run after it to hold them
    off (previously only guaranteed by alphabetical autouse ordering). The
    ledger connection cache is dropped around each test so a connection
    opened for the previous test's tmp path never serves the current test.
    """
    from yialpha.dataflows.config import set_config
    from yialpha.ledger.sqlite import reset_ledger_state_for_test

    set_config(
        {
            "ledger_db_path": str(tmp_path / "ledger" / "portfolio.db"),
            "instrument_registry": False,
            "prediction_ledger": False,
            "stock_perp_fair_value": False,
            "regime_state": False,
            "positioning_split": False,
            "onchain_evidence": False,
            "portfolio_control_mode": "legacy",
            "portfolio_positions_file": "",
        }
    )
    reset_ledger_state_for_test()
    yield
    reset_ledger_state_for_test()


@pytest.fixture(autouse=True)
def _hermetic_perp_overlay_mark(monkeypatch):
    """The perp risk overlay's Mark Reference bullet fetches mark klines
    (network). Stub it to None (the production fail-soft degradation) so any
    test running the perp overlay stays hermetic; a test asserting the mark
    bullet monkeypatches the method with a value, which shadows this
    class-level stub.
    """
    from yialpha.graph.trading_graph import YiAlphaGraph

    monkeypatch.setattr(
        YiAlphaGraph, "_latest_mark_close", lambda self, t, d: None,
    )
    # The live stress advisory has its own five vendor requests. Historical
    # dates (older than the production 3-day window) never fetch, so those
    # keep the real renderer and its disclosure; recent dates get a stub.
    from datetime import date, datetime, timedelta

    _real_stress_line = YiAlphaGraph._render_stress_line

    def _hermetic_stress_line(self, ticker, trade_date):
        try:
            dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
        except ValueError:
            dt = None
        if dt is None or dt < date.today() - timedelta(days=3):
            return _real_stress_line(ticker, trade_date)
        return "- **Derivatives Stress**: n/a (offline test)\n"

    monkeypatch.setattr(YiAlphaGraph, "_render_stress_line", _hermetic_stress_line)


@pytest.fixture(autouse=True)
def _hermetic_prediction_reference(monkeypatch):
    """Forecast acceptance cannot fetch live prices during offline tests.

    Default to explicit unavailable references. Reference-contract tests
    override these module-level vendor seams with their simulated bars.
    """
    import pandas as pd

    from yialpha.ledger import time_contract

    monkeypatch.setattr(time_contract, "binance_klines_frame", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(time_contract, "get_YFin_history_cached", lambda *a, **k: pd.DataFrame())
