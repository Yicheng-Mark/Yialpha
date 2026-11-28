"""Round-16 ``dataflows_core`` shard: regressions + contract tests for the shared
vendor substrate under ``yialpha/dataflows/`` (symbol gate, quality ledger,
disk cache, resampling, estimators).

Everything here is hermetic (no network, no sleep) and deterministic (frozen
dates). Each test pins one contract of an owned module; the file is owned by
this shard alone.
"""

from __future__ import annotations

import contextvars
import copy
import logging
import os
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import yialpha.default_config as default_config
from yialpha.dataflows import (
    config as df_config,
    disk_cache as dc,
    quality,
    symbol_utils,
    utils as df_utils,
)
from yialpha.dataflows.config import set_config
from yialpha.dataflows.errors import UnsupportedMarketError, VendorError
from yialpha.dataflows.symbol_utils import (
    is_supported_market,
    normalize_symbol,
    normalize_symbol_for_venue,
    require_supported_market,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Input gate (symbol_utils) — hard rule: non-US / non-crypto raises
# --------------------------------------------------------------------------- #

_CFD_ALIAS_WITH_QUOTE = [
    f"{alias}{sep}{quote}"
    for alias in ("GOLD", "XAU", "SPX500", "GER40", "US30", "HK50", "WTI", "BRENT")
    for sep in ("", "-")
    for quote in ("USD", "USDT", "USDC")
]


@pytest.mark.parametrize("ticker", _CFD_ALIAS_WITH_QUOTE)
def test_r16_market_gate_cfd_alias_with_quote_raises(ticker):
    """A removed commodity/index-CFD alias carrying a USD/USDT/USDC quote is
    still the removed CFD, not a crypto pair: the gate must raise
    ``UnsupportedMarketError`` with the CFD reason instead of waving it
    through the crypto-venue shape check."""
    assert is_supported_market(ticker) is False
    with pytest.raises(UnsupportedMarketError) as excinfo:
        require_supported_market(ticker)
    assert excinfo.value.ticker == ticker
    assert "former commodity/index CFD alias" in excinfo.value.reason


@pytest.mark.parametrize(
    "ticker",
    [
        "BTCUSDT", "BTC-USD", "ETH-USDC", "SOLUSD",        # crypto
        "MUUSDT", "BRK-BUSDT", "AAPLUSDC",                 # tokenized-stock perps
        "PAXGUSDT", "XAUTUSDT", "GDXUSDT",                 # gold *tokens/ETF* are not CFD aliases
    ],
)
def test_r16_market_gate_alias_quote_deny_is_exact_not_substring(ticker):
    """The alias+quote deny matches the whole base: crypto and equity-perp
    pairs merely *resembling* a commodity (PAXG/XAUT tokenized gold, GDX
    gold-miners ETF) keep passing the gate."""
    assert require_supported_market(ticker) == ticker


@pytest.mark.parametrize("raw", ["GOLDUSD", "XAU-USDT", "GOLD", "SPX500"])
def test_r16_market_gate_cfd_alias_is_never_remapped(raw):
    """An alias is rejected at the gate, never silently remapped:
    ``normalize_symbol`` (the Yahoo path) still passes it through untouched."""
    assert normalize_symbol(raw) == raw


def test_r16_market_gate_error_is_bad_input_not_vendor_failure():
    """``UnsupportedMarketError`` is a ``ValueError`` and NOT a ``VendorError``:
    the router only degrades vendor failures to sentinels, so a refused market
    can never be swallowed into a NO_DATA_AVAILABLE report."""
    err = UnsupportedMarketError("600519.SS", "policy sentence")
    assert isinstance(err, ValueError)
    assert not isinstance(err, VendorError)
    assert err.ticker == "600519.SS"
    assert err.reason == "policy sentence"
    assert str(err) == "Unsupported market for '600519.SS': policy sentence"
    # Default reason: message carries no dangling separator.
    assert str(UnsupportedMarketError("X")) == "Unsupported market for 'X'"


@pytest.mark.parametrize("raw", [None, 123, "", "   "])
def test_r16_market_gate_unusable_input_raises_policy_without_reason(raw):
    """Non-string / blank input is refused with the bare policy sentence (no
    specific reason prefix) and the raw value on ``ticker``."""
    assert is_supported_market(raw) is False
    with pytest.raises(UnsupportedMarketError) as excinfo:
        require_supported_market(raw)
    assert excinfo.value.ticker == str(raw)
    assert excinfo.value.reason == symbol_utils._MARKET_POLICY
    assert "no longer supported —" not in excinfo.value.reason


def test_r16_market_gate_returns_cleaned_symbol():
    """The gate returns the cleaned symbol: stripped, upper-cased, trailing
    broker ``+`` removed — and everything after the gate sees that form."""
    assert require_supported_market("  btc-usd ") == "BTC-USD"
    assert require_supported_market("aapl+") == "AAPL"
    assert require_supported_market(" BTCUSDT++ ") == "BTCUSDT"


# --- venue normalization (Binance) ------------------------------------------ #


@pytest.mark.parametrize("raw", ["", "   ", None, 7])
def test_r16_venue_normalizer_rejects_empty_symbol(raw):
    """Empty / non-string input for a Binance venue raises ``ValueError`` —
    there is no symbol to price."""
    with pytest.raises(ValueError, match="Cannot normalize empty symbol"):
        normalize_symbol_for_venue(raw, "binance_perp")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BTC-USDT", "BTCUSDT"),
        ("BTC/USDT", "BTCUSDT"),
        ("BTC:USDT", "BTCUSDT"),
        (" btc/usdt ", "BTCUSDT"),
        ("ETH-USDC", "ETHUSDT"),
        ("ETH/USDC", "ETHUSDT"),
        ("BTC", "BTCUSDT"),
        (" btc+ ", "BTCUSDT"),
    ],
)
def test_r16_venue_normalizer_collapses_separators_quotes_and_bare_base(raw, expected):
    """Binance normalization: ccxt/TradingView separators stripped, USDC/USD
    quotes collapsed onto the USDT book, a bare base gets the USDT quote."""
    assert normalize_symbol_for_venue(raw, "binance_perp") == expected
    assert normalize_symbol_for_venue(raw, "binance_spot") == expected


@pytest.mark.parametrize("raw", ["USDT", "USDC", "USD", "USDTUSDT", "usdt"])
def test_r16_venue_normalizer_rejects_degenerate_pair(raw):
    """A symbol with no base distinct from its quote must be refused, not
    priced as a fabricated pair."""
    for venue in ("binance_perp", "binance_spot"):
        with pytest.raises(ValueError, match="degenerate Binance perp pair"):
            normalize_symbol_for_venue(raw, venue)


@pytest.mark.parametrize("raw", ["BTCUSDT_260327", "BTCUSDT260930", "BTC-USD_250627"])
def test_r16_venue_normalizer_rejects_dated_delivery_contract(raw):
    """A plausible YYMMDD delivery suffix is refused loudly: the layer serves
    PERPETUAL-only endpoints and must not price the perp instead."""
    with pytest.raises(ValueError, match="dated delivery contract"):
        normalize_symbol_for_venue(raw, "binance_perp")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("BTCUSDT_991231", "BTCUSDT_991231USDT"), ("1000123456", "1000123456USDT")],
)
def test_r16_venue_normalizer_six_digits_are_not_a_delivery_date(raw, expected):
    """Only a real month/day in the 2024-2049 window reads as a delivery code;
    other 6-digit tails are just part of the base (numeric bases start with
    digits, they never end in a date-shaped pack)."""
    assert normalize_symbol_for_venue(raw, "binance_perp") == expected


def test_r16_venue_normalizer_unknown_venue_returns_cleaned_form():
    """An unknown venue gets the cleaned (upper, undashed) symbol back rather
    than a guessed quote — no USDT is appended."""
    assert normalize_symbol_for_venue("btc-usdt", "okx") == "BTCUSDT"
    assert normalize_symbol_for_venue("btc", "okx") == "BTC"


def test_r16_venue_normalizer_yahoo_venue_delegates_to_normalize_symbol():
    """``venue="yahoo"`` is exactly ``normalize_symbol`` (same output), so the
    Yahoo path stays untouched by the Binance rules."""
    assert normalize_symbol_for_venue("BTCUSDT", "yahoo") == normalize_symbol("BTCUSDT") == "BTC-USD"
    assert normalize_symbol_for_venue("BRK.B", "yahoo") == normalize_symbol("BRK.B") == "BRK.B"


def test_r16_tokenized_stock_perp_underlying():
    """The perp -> Yahoo-equity seam: dashed/compact USDT & USDC forms of a
    listed equity perp resolve to the stock ticker, with the BRK class-share
    spelling mapped; crypto / unlisted / non-perp inputs return None."""
    resolve = symbol_utils.tokenized_stock_perp_underlying
    assert resolve("MUUSDT") == "MU"
    assert resolve("mu-usdt") == "MU"
    assert resolve("MU-USDC") == "MU"
    assert resolve("BRKBUSDT") == "BRK-B"
    assert resolve("BRKB-USDT") == "BRK-B"


def test_r16_tokenized_stock_perp_underlying_rejects_non_equity_perps():
    resolve = symbol_utils.tokenized_stock_perp_underlying
    assert resolve("BTCUSDT") is None          # pure-crypto perp
    assert resolve("PEPEUSDT") is None         # unlisted alt
    assert resolve("MU") is None               # no stablecoin quote
    assert resolve("USDT") is None             # no base
    assert resolve("") is None and resolve("   ") is None
    assert resolve(None) is None and resolve(42) is None
    # A caller-supplied live listing supersedes the static seed.
    assert resolve("NEWXUSDT", frozenset({"NEWX"})) == "NEWX"
    assert resolve("MUUSDT", frozenset()) is None


# --------------------------------------------------------------------------- #
# Quality ledger (quality.py) — vacuum verdict + ContextVar isolation
# --------------------------------------------------------------------------- #


class _BrokenLedger:
    """A bound ledger whose mutations blow up (simulates recorder failure)."""

    def append(self, _item):
        raise RuntimeError("ledger on fire")

    def add(self, _item):
        raise RuntimeError("ledger on fire")

    def __iter__(self):
        return iter(())


def test_r16_quality_recorders_never_raise_on_broken_ledger():
    """``record_sentinel`` / ``record_success`` swallow any exception, even
    from an already-bound ledger object: quality evidence can never fail the
    data call it describes, and the broken object is not swapped out."""
    quality._events_var.set(_BrokenLedger())
    quality._success_var.set(_BrokenLedger())
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "x")
    quality.record_success("get_stock_data", "last")
    assert isinstance(quality._events_var.get(), _BrokenLedger)
    assert isinstance(quality._success_var.get(), _BrokenLedger)
    assert quality.snapshot_quality() == []
    assert quality.snapshot_core_successes() == set()


def test_r16_quality_sentinel_event_shape_is_coerced_and_truncated():
    """One sentinel event is exactly ``{method, kind, detail, qualifier}``
    with string-coerced fields and ``detail`` capped at 500 chars."""
    quality.ensure_run_context()
    quality.record_sentinel("get_binance_klines", quality.KIND_NO_DATA, "x" * 900, None)
    (event,) = quality.snapshot_quality()
    assert set(event) == {"method", "kind", "detail", "qualifier"}
    assert event["detail"] == "x" * 500
    assert event["qualifier"] == ""
    assert event["method"] == "get_binance_klines" and event["kind"] == "no_data"
    # snapshot is a copy: mutating it cannot rewrite the run's evidence.
    event["detail"] = "tampered"
    assert quality.snapshot_quality()[0]["detail"] == "x" * 500


def test_r16_quality_success_set_is_keyed_by_method_and_qualifier():
    """Successes are keyed ``method[qualifier]``: an INDEX-klines success and a
    LAST-klines success are different books, and the snapshot is a copy."""
    quality.ensure_run_context()
    quality.record_success("get_binance_klines", "index")
    quality.record_success("get_binance_klines", "last")
    quality.record_success("get_stock_data")
    quality.record_success("get_binance_klines", "last")  # idempotent
    assert quality.snapshot_core_successes() == {
        "get_binance_klines[index]", "get_binance_klines[last]", "get_stock_data",
    }
    quality.snapshot_core_successes().clear()
    assert len(quality.snapshot_core_successes()) == 3


@pytest.mark.parametrize(
    ("kinds", "successes", "vacuum"),
    [
        ([], [], False),                                   # nothing attempted: no verdict
        (["optional_unavailable"], [], False),             # optional only
        (["stale_cache"], [], False),                      # stale cache served = data existed
        (["no_data"], [], True),                           # core tried, zero success
        (["core_error"], [], True),
        (["no_data", "core_error", "optional_unavailable"], [], True),
        (["no_data"], ["get_stock_data"], False),          # any success un-vacuums
        (["core_error"], ["get_binance_klines[last]"], False),
        ([], ["get_stock_data"], False),
    ],
)
def test_r16_quality_vacuum_verdict_matrix(kinds, successes, vacuum):
    """Vacuum = (>=1 core sentinel) AND (success set empty); optional/stale
    evidence never counts as a core attempt, and a success under either
    ``record_success`` granularity un-vacuums."""
    events = [{"method": f"m{i}", "kind": kind, "detail": "d", "qualifier": ""} for i, kind in enumerate(kinds)]
    assert quality.is_data_vacuum(events, set(successes)) is vacuum
    assert quality.summarize_quality(events, set(successes))["data_vacuum"] is vacuum


def test_r16_quality_vacuum_verdict_tolerates_none_inputs():
    """``None`` events / successes read as "no evidence" — not a crash."""
    assert quality.is_data_vacuum(None, None) is False
    assert quality.is_data_vacuum([{"kind": "core_error"}], None) is True


def test_r16_quality_policy_matrix(monkeypatch, caplog):
    """``data_vacuum_policy`` resolution: case/whitespace-insensitive accept;
    blank, None and unknown values fail CLOSED to ``reject`` (a typo must not
    degrade into warn)."""
    with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.quality"):
        for raw, expected in [
            ("reject", "reject"), ("warn", "warn"), (" WARN ", "warn"),
            ("", "reject"), (None, "reject"), ("bogus", "reject"), (0, "reject"),
        ]:
            set_config({"data_vacuum_policy": raw})
            assert quality._vacuum_policy() == expected, raw
    assert "Invalid data_vacuum_policy" in caplog.text


def test_r16_quality_policy_fails_closed_when_config_raises(monkeypatch):
    """If the config lookup itself blows up the policy is ``reject`` — an
    unreadable config can never downgrade the vacuum gate."""

    def _boom():
        raise RuntimeError("config store unavailable")

    monkeypatch.setattr("yialpha.dataflows.config.get_config", _boom)
    assert quality._vacuum_policy() == "reject"
    quality.ensure_run_context()
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "no rows")
    with pytest.raises(quality.DataVacuumError, match="data vacuum"):
        quality.check_data_vacuum()


def test_r16_quality_check_data_vacuum_reject_names_core_evidence():
    """Under ``reject`` the error carries each core method and its detail (the
    operator must see *what* was missing), and the gate wrapper propagates it
    before the handler runs."""
    set_config({"data_vacuum_policy": "reject"})
    quality.ensure_run_context()
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "no rows")
    quality.record_sentinel("get_news", quality.KIND_OPTIONAL_UNAVAILABLE, "ignored detail")
    called = []
    with pytest.raises(quality.DataVacuumError) as excinfo:
        quality.gate_on_data_vacuum(lambda: called.append("ran"))()
    assert called == []
    msg = str(excinfo.value)
    assert "get_stock_data: no rows" in msg and "ignored detail" not in msg
    assert "data_vacuum_policy=warn" in msg


def test_r16_quality_copied_context_shares_the_run_ledger():
    """``ensure_run_context`` binds ONE list/set in the submitting context;
    every copied node context inherits the same objects, so evidence written
    inside ``copy_context().run`` is visible to the parent that snapshots."""
    quality.ensure_run_context()

    def node(tag):
        quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, tag)
        quality.record_success("get_binance_klines", "last")

    for i in range(3):
        contextvars.copy_context().run(node, f"n{i}")
    assert [e["detail"] for e in quality.snapshot_quality()] == ["n0", "n1", "n2"]
    assert quality.snapshot_core_successes() == {"get_binance_klines[last]"}


def test_r16_quality_fresh_context_cannot_see_or_pollute_the_run():
    """A context that did NOT inherit the submitter's binding (an unrelated
    concurrent run) starts blank: it sees none of the run's evidence and its
    own recordings never leak into the run's verdict."""
    quality.ensure_run_context()
    quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, "parent")

    def stranger():
        seen = quality.snapshot_quality()
        quality.record_sentinel("get_stock_data", quality.KIND_CORE_ERROR, "stranger")
        quality.record_success("get_stock_data")
        return seen

    assert contextvars.Context().run(stranger) == []
    assert quality.snapshot_core_successes() == set()
    assert [e["detail"] for e in quality.snapshot_quality()] == ["parent"]
    assert quality.is_data_vacuum(quality.snapshot_quality(), quality.snapshot_core_successes())


def test_r16_quality_thread_only_sees_the_context_it_is_given():
    """Threads do not inherit ContextVars: a worker running under a copied
    context shares the run's ledger; a worker without it starts blank and its
    recording never leaks back into the submitting thread."""
    quality.ensure_run_context()
    seen = {}

    def worker(tag):
        seen[tag] = quality.snapshot_quality()
        quality.record_sentinel("get_stock_data", quality.KIND_NO_DATA, tag)

    ctx = contextvars.copy_context()
    threads = [
        threading.Thread(target=lambda: ctx.run(worker, "inherit")),
        threading.Thread(target=lambda: worker("blank")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert seen == {"inherit": [], "blank": []}
    assert [e["detail"] for e in quality.snapshot_quality()] == ["inherit"]


def test_r16_quality_unbound_context_records_into_its_own_context():
    """Without ``ensure_run_context`` the recorders lazily bind a ledger in
    the *calling* context only: a sibling copied context does not see it
    (which is exactly why every entry point must bind before submitting)."""
    contextvars.copy_context().run(
        quality.record_sentinel, "get_stock_data", quality.KIND_NO_DATA, "orphan"
    )
    assert quality.snapshot_quality() == []
    contextvars.copy_context().run(quality.record_success, "get_stock_data")
    assert quality.snapshot_core_successes() == set()


# --------------------------------------------------------------------------- #
# Disk cache (disk_cache.py) — TTL / stale-on-failure / containment / throttle
# --------------------------------------------------------------------------- #

_DAY = 86_400.0


def _backdate(path: Path, days: float) -> None:
    """Set ``path``'s mtime to ``days`` days ago (negative = in the future)."""
    stamp = time.time() - days * _DAY
    os.utime(path, (stamp, stamp))


def _counting_fetch(payload: bytes = b"fetched"):
    calls = {"n": 0}

    def fetch() -> bytes:
        calls["n"] += 1
        return payload

    return fetch, calls


def _boom() -> bytes:
    raise RuntimeError("network down")


def test_r16_disk_cache_ttl_zero_always_refetches(tmp_path):
    """``ttl_days=0`` means "always refetch": a cache written a moment ago is
    not fresh — and neither is an mtime a hair in the FUTURE (the NTFS
    rounding case ``_age_seconds`` clamps), which must not resurrect it."""
    (tmp_path / "f.json").write_bytes(b"on-disk")
    fetch, calls = _counting_fetch(b"new")
    for _ in range(2):
        _backdate(tmp_path / "f.json", -1 / 24)  # one hour ahead of the clock
        assert dc.cached_or_fetch(str(tmp_path), "f.json", fetch, ttl_days=0, vendor="t") == b"new"
    assert calls["n"] == 2


def test_r16_disk_cache_age_clamp_and_missing_file_age(tmp_path):
    """``_age_seconds`` never goes negative (a future mtime reads as 0) and a
    file that cannot be statted is infinitely old — both feed fail-closed
    freshness/staleness decisions."""
    assert dc._age_seconds(time.time() + 3_600) == 0.0
    assert 0.0 <= dc._age_seconds(time.time() - 120) < 125
    # A path that exists on no platform (a "Z:/..." literal is relative on
    # Linux, where the test then depends on the working directory).
    missing = tmp_path / "definitely" / "missing.json"
    assert dc._mtime_age_days(missing) == float("inf")


def test_r16_disk_cache_freshness_boundary_is_strictly_inside_ttl(tmp_path):
    """Fresh means ``age < ttl``: with a one-day TTL a cache 10 minutes inside
    is served without a fetch and one 10 minutes past is refetched."""
    path = tmp_path / "f.json"
    path.write_bytes(b"cached")
    fetch, calls = _counting_fetch()
    _backdate(path, 1 - 600 / _DAY)
    assert dc.cached_or_fetch(str(tmp_path), "f.json", fetch, ttl_days=1.0, vendor="t") == b"cached"
    assert calls["n"] == 0
    _backdate(path, 1 + 600 / _DAY)
    assert dc.cached_or_fetch(str(tmp_path), "f.json", fetch, ttl_days=1.0, vendor="t") == b"fetched"
    assert calls["n"] == 1


def test_r16_disk_cache_stale_age_is_the_mtime_of_the_bytes_in_hand(tmp_path):
    """On a failed fetch the staleness verdict uses the mtime captured WITH the
    bytes that were read — never a later re-stat: another writer refreshing
    the file mid-fetch must not make the ancient bytes in hand look fresh."""
    path = tmp_path / "f.json"
    path.write_bytes(b"ancient")
    _backdate(path, 40)  # beyond the default 30-day stale cap
    quality.ensure_run_context()

    def failing_fetch() -> bytes:
        path.write_bytes(b"someone-elses-fresh-copy")  # concurrent publish
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        dc.cached_or_fetch(str(tmp_path), "f.json", failing_fetch, ttl_days=1.0, vendor="t")
    # Nothing stale was served, so no stale sentinel; the other writer's file
    # was left intact (the refusal path never writes).
    assert quality.snapshot_quality() == []
    assert path.read_bytes() == b"someone-elses-fresh-copy"


def test_r16_disk_cache_unstattable_cache_counts_as_absent(tmp_path, monkeypatch):
    """If the cache cannot be statted right after being read, its bytes are
    discarded: the fetch runs, and on a failed fetch the unverifiable copy is
    not served as stale."""
    path = tmp_path / "f.json"
    path.write_bytes(b"unverifiable")
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self.name == "f.json":
            raise FileNotFoundError(path)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    fetch, calls = _counting_fetch(b"verified")
    assert dc.cached_or_fetch(str(tmp_path), "f.json", fetch, ttl_days=1.0, vendor="t") == b"verified"
    assert calls["n"] == 1
    with pytest.raises(RuntimeError, match="network down"):
        dc.cached_or_fetch(str(tmp_path), "f.json", _boom, ttl_days=1.0, vendor="t")


def test_r16_disk_cache_over_cap_fail_open_returns_none_and_records_no_sentinel(tmp_path, caplog):
    """``fail_open`` + cache beyond the stale cap: the advisory data is simply
    absent (``None``) — the too-old cache is never served, and because nothing
    was served no stale-cache sentinel is recorded either."""
    path = tmp_path / "f.json"
    path.write_bytes(b"too-old")
    _backdate(path, 45)
    quality.ensure_run_context()
    with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.disk_cache"):
        out = dc.cached_or_fetch(
            str(tmp_path), "f.json", _boom, ttl_days=1.0, vendor="adv", fail_open=True
        )
    assert out is None
    assert quality.snapshot_quality() == []
    assert any("refusing to serve stale" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, 0.0), (0.0, 0.0), (2.5, 2.5), ("7", 7.0), ("0", 0.0),
        ("abc", 30.0), (None, 30.0), ([], 30.0), (-5, 30.0), ("-1", 30.0),
        (float("nan"), 30.0), (float("inf"), 30.0), ("inf", 30.0),
    ],
)
def test_r16_disk_cache_max_stale_days_matrix(raw, expected):
    """``data_cache_max_stale_days``: non-negative finite numbers are honored
    (0 = never serve stale); garbage, negatives and infinities fall back to
    the 30-day default — an infinite cap would silently disable the cap."""
    assert dc.max_stale_days() == 30.0  # default while the key is absent
    set_config({"data_cache_max_stale_days": raw})
    assert dc.max_stale_days() == expected


def test_r16_disk_cache_caller_stale_cap_only_tightens(tmp_path):
    """``stale_cap_days`` may lower the stale ceiling for one caller below the
    global cap, never raise it: minute-scale feeds refuse a 10-day-old cache
    the global 30-day cap would allow, and a caller asking for 365 days is
    still cut off at the global cap."""
    path = tmp_path / "f.json"
    path.write_bytes(b"old")
    _backdate(path, 10)
    with pytest.raises(RuntimeError, match="network down"):
        dc.cached_or_fetch(
            str(tmp_path), "f.json", _boom, ttl_days=1.0, vendor="social", stale_cap_days=1.0
        )
    with pytest.raises(RuntimeError, match="network down"):
        dc.cached_or_fetch(
            str(tmp_path), "f.json", _boom, ttl_days=1.0, vendor="t", stale_cap_days=0.0
        )
    assert dc.cached_or_fetch(
        str(tmp_path), "f.json", _boom, ttl_days=1.0, vendor="t", stale_cap_days=365.0
    ) == b"old"
    _backdate(path, 40)
    with pytest.raises(RuntimeError, match="network down"):
        dc.cached_or_fetch(
            str(tmp_path), "f.json", _boom, ttl_days=1.0, vendor="t", stale_cap_days=365.0
        )


def test_r16_disk_cache_cache_root_falls_back_when_config_blank():
    """A blank ``data_cache_dir`` (an empty ``YIALPHA_CACHE_DIR`` resolves to
    that) falls back to ``~/.yialpha/cache`` instead of a relative '' dir."""
    set_config({"data_cache_dir": ""})
    assert dc.cache_base_dir() == dc._DEFAULT_CACHE_BASE


def test_r16_disk_cache_path_is_built_only_from_validated_parts(tmp_path):
    """The cache path is ``base_dir`` joined with the validated component and
    nothing else: the base need not exist yet and no realpath is returned."""
    base = tmp_path / "not" / "yet" / "created"
    assert dc.cache_file_path(str(base), "daily_BTC-USD.csv") == base / "daily_BTC-USD.csv"
    assert dc.cache_file_path(str(base), "f(1).json") == base / "f(1).json"


@pytest.mark.parametrize(
    "nasty",
    [
        "xslF345X05/wk-form4_20250130.xml", "..\\..\\etc\\passwd", "BTC/USDT", "BTC:USDT",
        "a b", "100%", "a;b", "a$b", "a|b", "..%2f", "con\n", "sp ace/",
    ],
)
def test_r16_disk_cache_normalizer_output_always_passes_the_validator(tmp_path, nasty):
    """Whatever ``safe_cache_component`` flattens an external identifier into
    is always an acceptable single path component (the two allowlists are
    complements), so a cache key can never escape the cache directory."""
    flat = dc.safe_cache_component(nasty)
    assert dc.sanitize_cache_filename(f"doc_{flat}.xml") == f"doc_{flat}.xml"
    assert Path(dc.cache_file_path(str(tmp_path), f"doc_{flat}.xml")).parent == tmp_path


@pytest.mark.parametrize("bad", ["..", ".", "", "a/b", "a\\b", "C:x", "..\\x", "a b", "a%b"])
def test_r16_disk_cache_unsafe_filename_rejected_without_touching_disk(tmp_path, bad):
    """An unsafe component is refused by ``cached_or_fetch`` itself — before
    any fetch runs and before anything is created in the cache dir."""
    fetch, calls = _counting_fetch()
    with pytest.raises(ValueError, match="unsafe cache filename"):
        dc.cached_or_fetch(str(tmp_path), bad, fetch, ttl_days=1.0, vendor="t")
    assert calls["n"] == 0
    assert list(tmp_path.iterdir()) == []


def test_r16_disk_cache_symlink_escape_is_refused(tmp_path):
    """A cache entry that is a symlink to a file OUTSIDE the base dir fails the
    containment check (realpath-resolved); a symlink to a file inside is fine."""
    base = tmp_path / "cache"
    base.mkdir()
    outside = tmp_path / "outside" / "secret.json"
    outside.parent.mkdir()
    outside.write_bytes(b"secret")
    (base / "inside.json").write_bytes(b"inside")
    try:
        os.symlink(outside, base / "escape.json")
        os.symlink(base / "inside.json", base / "ok.json")
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlinks not permitted on this host")
    with pytest.raises(ValueError, match="resolves outside the cache directory"):
        dc.cache_file_path(str(base), "escape.json")
    assert dc.cache_file_path(str(base), "ok.json") == base / "ok.json"


def test_r16_disk_cache_failed_publish_leaves_no_tmp_and_keeps_previous_file(tmp_path, monkeypatch, caplog):
    """When the atomic publish keeps failing, the fetched bytes are still
    returned, the previous cache file is left intact (never truncated) and
    the per-writer temp file is cleaned up."""

    def denied(_tmp, _target):
        raise PermissionError("target open in a reader")

    monkeypatch.setattr(dc, "replace_retry_on_windows", denied)
    path = tmp_path / "f.json"
    path.write_bytes(b"previous")
    _backdate(path, 5)
    with caplog.at_level(logging.WARNING, logger="yialpha.dataflows.disk_cache"):
        out = dc.cached_or_fetch(str(tmp_path), "f.json", _counting_fetch()[0], ttl_days=1.0, vendor="t")
    assert out == b"fetched"
    assert path.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]
    assert any("could not write cache" in r.message for r in caplog.records)


class _FakeClock:
    """Deterministic clock: ``sleep`` only advances it (never blocks)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def time(self) -> float:
        return self.monotonic()

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


def test_r16_disk_cache_throttle_serializes_concurrent_waiters(monkeypatch):
    """``MinIntervalThrottle.wait`` holds its lock across the sleep, so K
    threads racing through it are released exactly ``min_interval`` apart:
    K-1 sleeps, every one for the full interval, nobody released early."""
    clock = _FakeClock()
    monkeypatch.setattr(dc, "time", clock)
    throttle = dc.MinIntervalThrottle(4.0)
    start = threading.Barrier(9)
    released: list[float] = []
    guard = threading.Lock()

    def racer():
        start.wait(timeout=10)
        throttle.wait()
        with guard:
            released.append(clock.monotonic())

    threads = [threading.Thread(target=racer) for _ in range(9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(clock.sleeps) == 8
    assert clock.sleeps == [4.0] * 8
    assert sorted(released) == [1000.0 + 4.0 * i for i in range(9)]


@pytest.mark.parametrize("interval", [0.0, -1.5])
def test_r16_disk_cache_throttle_non_positive_interval_never_sleeps(monkeypatch, interval):
    """A zero/negative interval is a no-op throttle: no sleep, calls it a day."""
    clock = _FakeClock()
    monkeypatch.setattr(dc, "time", clock)
    throttle = dc.MinIntervalThrottle(interval)
    throttle.wait()
    throttle.wait()
    assert clock.sleeps == [] and clock.now == 1000.0


def test_r16_disk_cache_throttle_waits_only_the_remaining_interval(monkeypatch):
    """A caller arriving late sleeps just the remainder of the interval, and a
    caller arriving after it does not sleep at all (monotonic clock)."""
    clock = _FakeClock()
    monkeypatch.setattr(dc, "time", clock)
    throttle = dc.MinIntervalThrottle(10.0)
    throttle.wait()                      # first call: never sleeps
    clock.now += 3.0                     # 3s of real work
    throttle.wait()
    assert clock.sleeps == [7.0]
    clock.now += 60.0
    throttle.wait()
    assert clock.sleeps == [7.0]


# --------------------------------------------------------------------------- #
# Config store (config.py) — merge / reset / per-context isolation
# --------------------------------------------------------------------------- #


def test_r16_config_set_in_an_unbound_context_stays_there():
    """A context that inherited no binding (fresh thread, ``Context()``) gets
    its own default-based store when it calls ``set_config``; the outer
    context's config is untouched."""
    outer_before = df_config.get_config()
    seen = {}

    def child():
        df_config.set_config({"data_cache_max_stale_days": 7, "__r16__": True})
        seen["cfg"] = df_config.get_config()

    contextvars.Context().run(child)
    assert seen["cfg"]["data_cache_max_stale_days"] == 7
    assert seen["cfg"]["__r16__"] is True
    assert df_config.get_config() == outer_before
    assert "__r16__" not in df_config.get_config()


def test_r16_config_copied_context_can_diverge_without_clobbering_parent():
    """In a copied child context ``set_config`` deep-copies before merging, so
    a child override never rewrites the parent's stored dict (or the defaults)."""
    parent_before = df_config.get_config()
    default_vendor = parent_before["data_vendors"]["core_stock_apis"]

    def child():
        # get_config hands out a deep copy: mutating it changes nothing.
        df_config.get_config()["data_vendors"]["core_stock_apis"] = "throwaway"
        assert df_config.get_config()["data_vendors"]["core_stock_apis"] == default_vendor
        df_config.set_config({"data_vendors": {"core_stock_apis": "alpha_vantage"}})
        return df_config.get_config()["data_vendors"]

    child_vendors = contextvars.copy_context().run(child)
    assert child_vendors["core_stock_apis"] == "alpha_vantage"
    for key, value in parent_before["data_vendors"].items():
        if key != "core_stock_apis":
            assert child_vendors[key] == value  # one-level merge keeps siblings
    assert df_config.get_config() == parent_before
    # reset replaces the whole store with DEFAULT_CONFIG (no merge, no leftovers)
    df_config.set_config({"__r16_probe__": 1})
    df_config.reset_config()
    defaults = df_config.get_config()
    assert "__r16_probe__" not in defaults
    assert defaults == copy.deepcopy(default_config.DEFAULT_CONFIG)


# --------------------------------------------------------------------------- #
# PIT guards (utils.py)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("period_end", "curr_date", "lag", "public"),
    [
        ("2024-09-30", "2024-11-14", 45, True),     # period end + 45d == as-of: public
        ("2024-09-30", "2024-11-13", 45, False),    # one day short: still lookahead
        ("2024-09-30", "2024-09-30", 0, True),      # lag 0 = the documented legacy behaviour
        ("2024-09-30", "2024-09-29", 0, False),
        ("2024-09-30 00:00:00", "2024-11-14", 45, True),   # datetime-string forms
        ("2024-09-30", "", 45, True),               # live mode: no as-of constraint
        ("2024-09-30", None, 45, True),
        ("not-a-date", "2025-01-01", 45, False),    # unprovable => dropped (fail-closed)
        (None, "2025-01-01", 45, False),
        ("2024-09-30", "2024/11/14", 45, False),    # unparseable as-of => dropped
    ],
)
def test_r16_pit_is_filing_public_matrix(period_end, curr_date, lag, public):
    """A statement is visible only from ``fiscal period end + lag`` onwards
    (inclusive); anything that cannot be parsed is treated as NOT public."""
    assert df_utils.is_filing_public(period_end, curr_date, lag_days=lag) is public


@pytest.mark.parametrize("period_end", [date(2024, 9, 30), pd.Timestamp("2024-09-30")])
def test_r16_pit_is_filing_public_accepts_date_objects(period_end):
    """``date`` / ``pd.Timestamp`` period ends are read by their first ten
    characters, so vendors can pass the raw frame value through."""
    assert df_utils.is_filing_public(period_end, "2024-11-14", lag_days=45) is True
    assert df_utils.is_filing_public(period_end, "2024-11-13", lag_days=45) is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 45), ("", 45), ("0", 0), ("45", 45), (" 60 ", 60), ("+90", 90),
     ("abc", 45), ("3.5", 45), ("-5", 45), ("-45", 45)],
)
def test_r16_pit_filing_lag_env_matrix(raw, expected):
    """``YIALPHA_FUNDAMENTALS_FILING_LAG_DAYS``: a non-negative integer is
    honored (0 = documented legacy switch); garbage AND NEGATIVE values fall
    back to the conservative default — a negative lag would let a backtest
    read a statement before its period even ended."""
    assert df_utils._parse_filing_lag_days(raw) == expected


def test_r16_pit_filing_lag_module_constant_follows_the_env():
    """The import-time constant is the env parser applied to the env var, and
    it is the default lag of ``is_filing_public``."""
    import inspect

    env = os.environ.get("YIALPHA_FUNDAMENTALS_FILING_LAG_DAYS")
    assert df_utils._parse_filing_lag_days(env) == df_utils.FUNDAMENTALS_FILING_LAG_DAYS
    default = inspect.signature(df_utils.is_filing_public).parameters["lag_days"].default
    assert default == df_utils.FUNDAMENTALS_FILING_LAG_DAYS


def test_r16_pit_pinned_date_restores_outer_pin_and_clears_on_error():
    """``pinned_analysis_date`` restores whatever was pinned before it (None
    for a top-level run — also on an exception) instead of blindly clearing:
    a nested manager must not strip an outer run's PIT clamp."""
    assert df_utils.get_analysis_date() is None
    try:
        with df_utils.pinned_analysis_date("2026-07-01"):
            assert df_utils.current_pit_end("2026-08-15") == "2026-07-01"
            with df_utils.pinned_analysis_date("2026-06-01"):
                assert df_utils.current_pit_end("2026-08-15") == "2026-06-01"
            # inner manager exited: the OUTER clamp is still in force
            assert df_utils.get_analysis_date() == "2026-07-01"
            assert df_utils.current_pit_end("2026-08-15") == "2026-07-01"
            raise RuntimeError("run crashed")
    except RuntimeError as exc:
        assert str(exc) == "run crashed"
    assert df_utils.get_analysis_date() is None
    assert df_utils.current_pit_end("2026-08-15") == "2026-08-15"


def test_r16_pit_pin_is_isolated_per_context():
    """The pinned analysis date is a ContextVar: a copied context inherits the
    pin, an unrelated context never sees it, and a copied context cannot
    drop the submitter's clamp."""
    with df_utils.pinned_analysis_date("2026-07-01"):
        assert contextvars.copy_context().run(df_utils.current_pit_end, "2026-08-15") == "2026-07-01"
        assert contextvars.Context().run(df_utils.get_analysis_date) is None
        contextvars.copy_context().run(df_utils.set_analysis_date, None)
        assert df_utils.get_analysis_date() == "2026-07-01"
    assert df_utils.get_analysis_date() is None


@pytest.mark.parametrize(
    ("curr_date", "historical"),
    [
        ("", False), (None, False), ("2026/07/01", True), ("garbage", True),
    ],
)
def test_r16_pit_malformed_or_blank_analysis_date_classification(curr_date, historical):
    """Blank = live (current-snapshot sources allowed); an unparseable explicit
    date is treated as historical so today's snapshot cannot leak into a
    nominal backtest, and the overview gate follows the same verdict."""
    assert df_utils.is_historical_date(curr_date) is historical
    assert df_utils.overview_would_leak_future(curr_date) is historical


def test_r16_pit_live_anchor_dates_are_the_two_calendars(monkeypatch):
    """Both host-local and UTC today are live anchors, on a frozen clock where
    the two calendars already disagree (local 2026-11-24 00:30 on UTC+8, UTC
    still 2026-11-23): a label equal to EITHER is live; anything else — past
    or future, including local-yesterday while it is still UTC-today — is
    historical, and the overview gate follows the same verdict."""

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return datetime(2026, 11, 24, 0, 30)  # host-local wall clock
            return datetime(2026, 11, 23, 16, 30, tzinfo=tz)  # same instant, UTC

    monkeypatch.setattr(df_utils, "datetime", _FrozenDatetime)
    assert df_utils.live_anchor_dates() == (date(2026, 11, 24), date(2026, 11, 23))
    for live_label in ("2026-11-24", "2026-11-23", "2026-11-23 15:04:05"):
        assert df_utils.is_historical_date(live_label) is False
        assert df_utils.overview_would_leak_future(live_label) is False
    for historical_label in ("2026-11-22", "2026-11-25", "2027-01-01"):
        assert df_utils.is_historical_date(historical_label) is True
        assert df_utils.overview_would_leak_future(historical_label) is True


def test_r16_pit_unparseable_analysis_date_fails_open_only_in_clamp():
    """An unparseable pinned analysis date leaves ``clamp_end_date`` a no-op
    (documented live-only fail-open) while the snapshot gate stays closed."""
    with df_utils.pinned_analysis_date("2026/07/01"):
        assert df_utils.current_pit_end("2026-08-15") == "2026-08-15"
        assert df_utils.is_historical_date(df_utils.get_analysis_date()) is True


def test_r16_utils_replace_retry_zero_attempts_refuses_without_touching_disk(tmp_path):
    """``attempts < 1`` never calls ``os.replace`` and refuses loudly instead
    of silently pretending the publish happened."""
    (tmp_path / "a.tmp").write_text("new")
    (tmp_path / "a.json").write_text("old")
    with pytest.raises(RuntimeError, match="without a single attempt"):
        df_utils.replace_retry_on_windows(tmp_path / "a.tmp", tmp_path / "a.json", attempts=0)
    assert (tmp_path / "a.json").read_text() == "old"
    assert (tmp_path / "a.tmp").read_text() == "new"


def test_r16_utils_replace_retry_escalates_then_succeeds(tmp_path, monkeypatch):
    """A reader holding the target makes ``os.replace`` raise PermissionError;
    the publish retries with escalating back-off and succeeds once the reader
    is gone (clock is fake — no real sleeping)."""
    sleeps: list[float] = []
    attempts = {"n": 0}
    real_replace = os.replace

    def busy_then_free(src, dst):
        attempts["n"] += 1
        if attempts["n"] <= 3:
            raise PermissionError("target open in a reader")
        real_replace(src, dst)

    monkeypatch.setattr(df_utils.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(df_utils.os, "replace", busy_then_free)
    (tmp_path / "a.tmp").write_text("new")
    (tmp_path / "a.json").write_text("old")
    df_utils.replace_retry_on_windows(tmp_path / "a.tmp", tmp_path / "a.json", delay_s=0.05)
    assert attempts["n"] == 4
    assert sleeps == [0.05, 0.05, 0.1]
    assert (tmp_path / "a.json").read_text() == "new"
    assert not (tmp_path / "a.tmp").exists()


def test_r16_utils_replace_retry_exhaustion_reraises_last_error(tmp_path, monkeypatch):
    """When the target stays locked the last PermissionError is re-raised (the
    caller applies its own fail-soft policy) after the full back-off budget."""

    def always_busy(_src, _dst):
        raise PermissionError("target open in a reader")

    sleeps: list[float] = []
    monkeypatch.setattr(df_utils.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(df_utils.os, "replace", always_busy)
    (tmp_path / "a.tmp").write_text("new")
    with pytest.raises(PermissionError, match="target open in a reader"):
        df_utils.replace_retry_on_windows(tmp_path / "a.tmp", tmp_path / "a.json")
    assert sleeps == [0.05, 0.05, 0.1, 0.2, 0.4]
    assert (tmp_path / "a.tmp").read_text() == "new"


def test_r16_utils_atomic_write_text_publishes_whole_file(tmp_path):
    """``atomic_write_text`` writes ``<name>.tmp`` then replaces, leaving only
    the complete target behind."""
    target = tmp_path / "nested" / "note.txt"
    target.parent.mkdir()
    assert df_utils.atomic_write_text(target, "héllo\n") == target
    assert target.read_text(encoding="utf-8") == "héllo\n"
    assert [p.name for p in target.parent.iterdir()] == ["note.txt"]


def test_r16_utils_safe_ticker_component_matrix():
    """Path-safe ticker component: structural ticker characters pass through
    unchanged; separators, drive letters, over-long and all-dot values raise."""
    ok = ["AAPL", "BRK.B", "BTC-USD", "^GSPC", "GC=F", "XAUUSD+", "A" * 32, "a..b"]
    bad = ["../x", "a/b", "a\\b", "C:", "A B", "", ".", "..", "...", "A" * 33, None, 5]
    for value in ok:
        assert df_utils.safe_ticker_component(value) == value
    for value in bad:
        with pytest.raises(ValueError):
            df_utils.safe_ticker_component(value)


def test_r16_utils_safe_xml_refuses_dtd_and_entity(monkeypatch):
    """Untrusted XML carrying a DOCTYPE/ENTITY declaration (case-insensitive,
    bytes or str) is refused before the parser ever sees it."""
    for raw in [
        '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "y">]><a/>',
        b"<feed><!entity x 'y'/></feed>",
        "<!DoCtYpE rss><rss/>",
    ]:
        with pytest.raises(ValueError, match="entity-expansion guard"):
            df_utils.safe_xml_root(raw, source="feed")
    root = df_utils.safe_xml_root(b"<rss><channel><title>t</title></channel></rss>", source="feed")
    assert root.tag == "rss" and root.findtext("channel/title") == "t"


def test_r16_utils_proxy_map_reads_env_at_call_time(monkeypatch):
    """``proxy_map`` is read per call (never cached at import): scheme -> its
    own proxy else ALL_PROXY, unset entries omitted, an empty value treated
    as unset."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(key, raising=False)
    assert df_utils.proxy_map() == {}
    monkeypatch.setenv("ALL_PROXY", "socks5://all:1080")
    assert df_utils.proxy_map() == {"http": "socks5://all:1080", "https": "socks5://all:1080"}
    monkeypatch.setenv("HTTPS_PROXY", "http://h:8080")
    assert df_utils.proxy_map() == {"http": "socks5://all:1080", "https": "http://h:8080"}
    monkeypatch.delenv("ALL_PROXY")
    assert df_utils.proxy_map() == {"https": "http://h:8080"}
    monkeypatch.delenv("HTTPS_PROXY")
    assert df_utils.proxy_map() == {}
