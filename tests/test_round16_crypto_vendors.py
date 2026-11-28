"""Round-16 shard ``crypto_vendors``: contract pins and regressions.

Scope: the ``yialpha/dataflows`` modules this shard owns (Binance perp/spot
transport, weight limiter, exchange filters, leverage brackets, perp bundle,
the crypto vendor adapters and the data.binance.vision archive). Every test
is hermetic — vendor seams (``requests.get``, the project's ``_http_get``)
are mocked, backoff sleeps are intercepted, clocks are injected — and pins
one stated contract per test.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
import urllib.parse
from decimal import Decimal
from types import SimpleNamespace

import pytest

from yialpha.dataflows import (
    binance_brackets as brackets,
    binance_filters as filters,
    binance_http,
    binance_rate_limiter as rl,
)
from yialpha.dataflows.config import set_config
from yialpha.dataflows.errors import NoMarketDataError

# ---- shared fixtures --------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_shard_state():
    """Drop this shard's process-global caches around every test.

    The limiter registry, the exchangeInfo filter cache, the leverage-bracket
    cache and the shared keepalive session all live at module level; without
    the reset a value observed in one test would back off (or serve) the next.
    """
    def _reset() -> None:
        rl.reset_for_test()
        filters.reset_for_test()
        brackets.reset_for_test()
        binance_http.reset_for_test()

    _reset()
    yield
    _reset()


class _Clock:
    """Deterministic stand-in for the ``time`` module inside a vendor module.

    ``monotonic`` is an injected reading (no sleeping, no wall clock) and
    ``sleep`` records the requested delay instead of waiting.
    """

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


@pytest.fixture()
def limiter_clock(monkeypatch) -> _Clock:
    """Inject :class:`_Clock` as the limiter module's ``time``."""
    clock = _Clock()
    monkeypatch.setattr(rl, "time", clock)
    return clock


class _Resp:
    """Minimal ``requests.Response`` stand-in: status/text/json/headers."""

    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data
        self.text = "[]" if json_data is None else str(json_data)
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no body")
        return self._json


# ---- binance_rate_limiter ---------------------------------------------------


@pytest.mark.unit
def test_r16_limiter_observe_ignores_none_and_unparseable_keeps_last_valid_reading(
    limiter_clock,
):
    """Pin: observe(None)/observe(garbage) is a no-op — the last valid server reading AND its timestamp survive, so a single missing header never resets a hot budget to cold."""
    lim = rl.BinanceWeightLimiter(2400)
    lim.observe(2300)
    seen_at = lim._used_ts
    limiter_clock.now += 10.0

    lim.observe(None)
    for bad in ("abc", "", "1,900", object()):
        lim.observe(bad)

    assert lim.used == 2300
    assert lim._used_ts == seen_at
    # A numeric string (how the header arrives) IS accepted.
    lim.observe("1900")
    assert lim.used == 1900


@pytest.mark.unit
def test_r16_limiter_products_have_independent_budgets_and_readings(limiter_clock):
    """Pin: fapi and spot are separate process-wide limiters with their own documented budgets (2400 / 6000) — one product's hot reading never delays the other's acquire."""
    fapi = rl.get_binance_weight_limiter("fapi")
    spot = rl.get_binance_weight_limiter("spot")
    assert fapi is not spot
    assert rl.get_binance_weight_limiter("fapi") is fapi
    assert rl.get_binance_weight_limiter("spot") is spot
    assert (fapi.weight_limit, spot.weight_limit) == (2400, 6000)

    fapi.observe(2300)  # 96% of the fapi budget: hot
    spot.observe(5900)  # 98% of the spot budget: hot on its own ledger
    assert (fapi.used, spot.used) == (2300, 5900)
    fapi.acquire()
    spot.acquire()
    assert limiter_clock.sleeps == [60.0, 60.0]

    limiter_clock.now += 30.0
    fapi.observe(10)  # only fapi's server reading receded
    fapi.acquire()  # fapi budget open again: no wait
    spot.acquire()  # spot still hot: waits out the rest of ITS window
    assert limiter_clock.sleeps == [60.0, 60.0, 30.0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("product", "limit", "trips", "clear"),
    [("fapi", 2400, 1920, 1919), ("spot", 6000, 4800, 4799)],
)
def test_r16_limiter_trip_boundary_is_inclusive_at_80pct_of_own_limit(
    limiter_clock, product, limit, trips, clear,
):
    """Pin: the backoff trips at >= 80% of EACH product's own limit (the boundary reading itself trips); one weight below it does not."""
    lim = rl.get_binance_weight_limiter(product)
    assert lim.weight_limit == limit

    lim.observe(clear)
    lim.acquire()
    assert limiter_clock.sleeps == []

    lim.observe(trips)
    lim.acquire()
    assert limiter_clock.sleeps == [60.0]


@pytest.mark.unit
def test_r16_limiter_wait_is_remaining_window_not_flat_minute(limiter_clock):
    """Pin: acquire sleeps only the un-aged remainder of the 60s rolling window (a hot reading 25s old -> 35s), and a reading aged to the window edge sleeps nothing."""
    lim = rl.BinanceWeightLimiter(2400)
    lim.observe(2300)

    limiter_clock.now += 25.0
    lim.acquire()
    assert limiter_clock.sleeps == [35.0]

    limiter_clock.now += 35.0  # reading is exactly 60s old: rolled out
    lim.acquire()
    assert limiter_clock.sleeps == [35.0]


@pytest.mark.unit
def test_r16_limiter_latest_server_reading_governs_not_high_water_mark(limiter_clock):
    """Pin: the limiter tracks the server's LATEST reading, not a high-water mark — a fresh in-window reading below the trip point reopens the budget immediately."""
    lim = rl.BinanceWeightLimiter(2400)
    lim.observe(2300)
    lim.acquire()
    assert len(limiter_clock.sleeps) == 1

    limiter_clock.now += 5.0  # still inside the first reading's window
    lim.observe(100)
    lim.acquire()
    assert len(limiter_clock.sleeps) == 1


@pytest.mark.unit
def test_r16_limiter_sleeper_does_not_hold_the_lock(monkeypatch):
    """Pin: acquire() sleeps OUTSIDE the lock — a thread backing off must not serialize observe()/used()/acquire() for other threads."""
    entered_sleep = threading.Event()
    release = threading.Event()

    def blocking_sleep(_seconds):
        entered_sleep.set()
        release.wait(timeout=10)

    monkeypatch.setattr(
        rl, "time", SimpleNamespace(monotonic=time.monotonic, sleep=blocking_sleep),
    )
    lim = rl.BinanceWeightLimiter(2400)
    lim.observe(2300)
    sleeper = threading.Thread(target=lim.acquire, daemon=True)
    sleeper.start()
    assert entered_sleep.wait(timeout=10), "sleeper never reached its backoff"

    other_done = threading.Event()

    def other_thread():
        lim.observe(5)  # would deadlock behind the sleeper if the lock were held
        assert lim.used == 5
        lim.acquire()  # fresh cold reading: returns at once
        other_done.set()

    threading.Thread(target=other_thread, daemon=True).start()
    try:
        assert other_done.wait(timeout=10), "observe/acquire blocked behind the sleeper"
    finally:
        release.set()
        sleeper.join(timeout=10)
    assert not sleeper.is_alive()


@pytest.mark.unit
def test_r16_limiter_unknown_product_is_its_own_singleton_with_fapi_default():
    """Pin: an unregistered product key gets a distinct singleton limiter on the fapi default budget — never aliased onto the live fapi/spot instances."""
    other = rl.get_binance_weight_limiter("options")
    assert other.weight_limit == 2400
    assert other is rl.get_binance_weight_limiter("options")
    assert other is not rl.get_binance_weight_limiter("fapi")
    assert other is not rl.get_binance_weight_limiter("spot")


@pytest.mark.unit
@pytest.mark.parametrize("limit", [0, -5])
def test_r16_limiter_weight_limit_floors_at_one(limiter_clock, limit):
    """Pin: a zero/negative limit can never put the trip point at 0 (which would back off on every reading, including weight 0) — the limit floors at 1."""
    lim = rl.BinanceWeightLimiter(limit)
    assert lim.weight_limit == 1

    lim.observe(0)
    lim.acquire()
    assert limiter_clock.sleeps == []


@pytest.mark.unit
def test_r16_limiter_config_threshold_overrides_both_products(limiter_clock):
    """Pin: binance_weight_threshold, when set, replaces BOTH product defaults (shared-IP conservatism) and is read at limiter creation."""
    set_config({"binance_weight_threshold": 1000})
    assert rl.get_binance_weight_limiter("fapi").weight_limit == 1000
    assert rl.get_binance_weight_limiter("spot").weight_limit == 1000
    # Still two independent ledgers under the shared ceiling.
    rl.get_binance_weight_limiter("fapi").observe(999)
    rl.get_binance_weight_limiter("fapi").acquire()
    rl.get_binance_weight_limiter("spot").acquire()
    assert limiter_clock.sleeps == [60.0]


# ---- binance_filters --------------------------------------------------------


def _fx(step="0.001", tick="0.10", min_qty="0.001", max_qty="1000", min_notional="5"):
    """A Binance-like filter block; every field overridable as an exchange string."""
    return filters.SymbolFilters(
        symbol="BTCUSDT", status="TRADING",
        tick_size=Decimal(tick), step_size=Decimal(step),
        min_qty=Decimal(min_qty), max_qty=Decimal(max_qty),
        min_notional=Decimal(min_notional), price_precision=2, quantity_precision=3,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("quantity", "step", "expected"),
    [
        (0.30718, "0.001", 0.307),  # floor, never round up
        (0.3079, "0.001", 0.307),  # 0.9 of a step still floors
        (4.35, "0.01", 4.35),  # on-step float artifact must not shave a step
        (8.20, "0.01", 8.20),
        (0.1 + 0.2, "0.1", 0.3),  # 0.30000000000000004 -> 0.3
        (1.0, "0.001", 1.0),  # exactly on step: unchanged
        (0.0009, "0.001", 0.0),  # sub-step quantity floors to zero
        (2.9999, "1", 2.0),  # integer step
        (123_456.7891, "0.001", 123_456.789),  # large magnitude stays exact
    ],
)
def test_r16_quantize_quantity_floors_to_step_never_up(quantity, step, expected):
    """Pin: quantity floors to a stepSize multiple (never rounds up — that would exceed the intended exposure) using Decimal-exact math, so an exactly-on-step float is not shaved down a step."""
    out = filters.quantize_order(None, quantity, _fx(step=step))
    assert out["quantity"] == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("price", "tick", "expected"),
    [
        (50000.37, "0.10", 50000.4),
        (50000.34, "0.10", 50000.3),
        (0.15, "0.1", 0.2),  # tie -> even neighbour (2 ticks)
        (0.25, "0.1", 0.2),  # tie -> even (2): distinguishes half-even from half-up
        (0.35, "0.1", 0.4),  # tie -> even (4): distinguishes it from half-down/truncation
        (1.005, "0.01", 1.0),
        (1.015, "0.01", 1.02),
        (65432.1, "0.1", 65432.1),  # already on tick: unchanged
        (1.23456, "1e-8", 1.23456),  # sub-cent contract keeps exchange precision
    ],
)
def test_r16_quantize_price_rounds_half_even_to_tick(price, tick, expected):
    """Pin: a limit price rounds HALF_EVEN to the tickSize multiple (direction changes fill priority, not validity) and the result is float-exact."""
    out = filters.quantize_order(price, 1.0, _fx(tick=tick))
    assert out["price"] == expected


@pytest.mark.unit
def test_r16_quantize_flags_are_judged_on_quantized_values():
    """Pin: below_min_qty / over_max_qty / below_min_notional are judged on the QUANTIZED quantity and price (what the exchange will see), never on the raw floats."""
    f = _fx()
    # Raw 0.0019 floors onto minQty 0.001 -> compliant; 0.0009 floors to 0 -> not.
    assert filters.quantize_order(None, 0.0019, f)["below_min_qty"] is False
    assert filters.quantize_order(None, 0.0009, f)["below_min_qty"] is True
    # Raw 1000.0004 floors back under maxQty; 1000.001 does not.
    assert filters.quantize_order(None, 1000.0004, f)["over_max_qty"] is False
    assert filters.quantize_order(None, 1000.001, f)["over_max_qty"] is True
    # maxQty "0" = no upper bound.
    assert filters.quantize_order(None, 1e9, _fx(max_qty="0"))["over_max_qty"] is False
    # Price 2.15 rounds UP to 2.2: raw notional 4.945 (< 5) but the exchange sees 5.06.
    up = filters.quantize_order(2.15, 2.30, f)
    assert (up["price"], up["quantity"]) == (2.2, 2.3)
    assert up["below_min_notional"] is False
    # Price 2.05 rounds DOWN to 2.0: raw notional 5.0225 (>= 5) but the exchange sees 4.90.
    down = filters.quantize_order(2.05, 2.45, f)
    assert (down["price"], down["quantity"]) == (2.0, 2.45)
    assert down["below_min_notional"] is True


@pytest.mark.unit
@pytest.mark.parametrize("bad", [None, "", "abc", "1,000"])
def test_r16_filters_dec_falls_back_on_unparseable_exchange_value(bad):
    """Pin: a missing/garbage exchangeInfo numeric degrades to the caller's default (tick/step 1e-8, quantities 0) instead of raising."""
    assert filters._dec(bad, "1e-8") == Decimal("1e-8")
    assert filters._dec(bad) == Decimal(0)
    assert filters._dec("0.001") == Decimal("0.001")


def _exchange_info(tick_size: str = "0.10", **overrides):
    """A one-symbol fapi exchangeInfo payload.

    ``tick_size`` sets the PRICE_FILTER tick; ``overrides`` replace named
    entry fields (pass ``filters=`` to swap the whole filter block).
    """
    entry = {
        "symbol": "BTCUSDT",
        "status": "TRADING",
        "pricePrecision": 1,
        "quantityPrecision": 3,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "0.10"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
            {"filterType": "MIN_NOTIONAL", "notional": "5", "applyMinToMarket": True},
        ],
    }
    entry.update(overrides)
    for block in entry["filters"]:
        if block.get("filterType") == "PRICE_FILTER":
            block["tickSize"] = tick_size
    return {"symbols": [entry]}


@pytest.mark.unit
def test_r16_filters_extract_prefers_notional_and_defaults_missing_block():
    """Pin: NOTIONAL (spot) wins over MIN_NOTIONAL (fapi) when both are present; a symbol with no notional/lot block yields 0 limits (never raises), and an entry that omits symbol/status falls back to the requested canonical."""
    both = _exchange_info(filters=[
        {"filterType": "MIN_NOTIONAL", "notional": "100"},
        {"filterType": "NOTIONAL", "notional": "5", "minNotional": "5"},
        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
    ])
    f = filters._extract_filters(both["symbols"][0], "BTCUSDT")
    assert (f.min_notional, f.tick_size) == (Decimal(5), Decimal("1e-8"))
    # The entry's declared precisions are carried through.
    assert (f.price_precision, f.quantity_precision) == (1, 3)
    # A spot-shaped entry carries neither field: both fall back to 8.
    spot_entry = {"symbol": "BTCUSDT", "status": "TRADING", "filters": [
        {"filterType": "NOTIONAL", "notional": "5", "minNotional": "5"},
    ]}
    bare_spot = filters._extract_filters(spot_entry, "BTCUSDT")
    assert (bare_spot.price_precision, bare_spot.quantity_precision) == (8, 8)
    assert bare_spot.min_notional == Decimal(5)

    # An entry that omits symbol/status falls back to the requested canonical.
    bare = filters._extract_filters({}, "ETHUSDT")
    assert (bare.symbol, bare.status) == ("ETHUSDT", "")
    assert (bare.min_qty, bare.max_qty, bare.min_notional) == (Decimal(0), Decimal(0), Decimal(0))
    assert (bare.tick_size, bare.step_size) == (Decimal("1e-8"), Decimal("1e-8"))


@pytest.mark.unit
def test_r16_filters_cache_key_separates_venue_and_uses_spot_host_and_budget(monkeypatch):
    """Pin: the exchangeInfo cache is keyed by (venue, canonical) — a spot lookup after a perp lookup for the same pair is a MISS that goes to /api/v3 on the configured spot host against the spot weight budget, and its cached block is per-venue."""
    set_config({"binance_spot_mirror": True})
    calls: list[tuple] = []

    def fake_get(path, params, symbol, canonical, **kwargs):
        calls.append((path, dict(params), kwargs.get("base"), kwargs.get("weight_key")))
        return _exchange_info(tick_size="0.01" if path.startswith("/api/") else "0.10")

    monkeypatch.setattr(filters, "_http_get", fake_get)

    perp = filters.get_symbol_filters("BTCUSDT", "binance_perp")
    spot = filters.get_symbol_filters("BTCUSDT", "binance_spot")
    assert filters.get_symbol_filters("BTCUSDT", "binance_spot") is spot
    assert filters.get_symbol_filters("btc-usdt", "binance_perp") is perp

    assert calls == [
        ("/fapi/v1/exchangeInfo", {"symbol": "BTCUSDT"}, None, None),
        ("/api/v3/exchangeInfo", {"symbol": "BTCUSDT"},
         "https://data-api.binance.vision", "spot"),
    ]
    assert (perp.tick_size, spot.tick_size) == (Decimal("0.10"), Decimal("0.01"))


@pytest.mark.unit
def test_r16_filters_force_refresh_bypasses_ttl_and_replaces_cached_block(monkeypatch):
    """Pin: force_refresh ignores a fresh TTL hit, re-queries, and the new block replaces the cached one for every later reader."""
    payload = {"tick": "0.10"}
    monkeypatch.setattr(
        filters, "_http_get",
        lambda path, params, symbol, canonical, **kw: _exchange_info(
            tick_size=payload["tick"],
        ),
    )
    assert filters.get_symbol_filters("BTCUSDT").tick_size == Decimal("0.10")

    payload["tick"] = "0.50"
    assert filters.get_symbol_filters("BTCUSDT").tick_size == Decimal("0.10")  # TTL hit
    refreshed = filters.get_symbol_filters("BTCUSDT", force_refresh=True)
    assert refreshed.tick_size == Decimal("0.50")
    assert filters.get_symbol_filters("BTCUSDT") is refreshed


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        {"symbols": []},
        {"symbols": None},
        {"symbols": "BTCUSDT"},
        [],
        None,
    ],
)
def test_r16_filters_malformed_exchangeinfo_raises_typed_error_and_is_not_cached(
    monkeypatch, payload,
):
    """Pin: an exchangeInfo reply with no usable symbol list raises NoMarketDataError (caller degrades, never orders on guessed precision) and the failure is NOT cached — a later good reply is served."""
    replies = iter([payload, _exchange_info()])
    monkeypatch.setattr(
        filters, "_http_get", lambda *a, **k: next(replies),
    )
    with pytest.raises(NoMarketDataError):
        filters.get_symbol_filters("BTCUSDT")
    assert filters.get_symbol_filters("BTCUSDT").tick_size == Decimal("0.10")


# ---- binance_brackets -------------------------------------------------------


@pytest.mark.unit
def test_r16_brackets_signed_get_without_keys_raises_before_any_request(monkeypatch):
    """Pin: with no BINANCE_API_KEY/SECRET a signed fetch raises NoMarketDataError (actionable reason) BEFORE any request, and nothing is cached — configuring keys later succeeds."""
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    def must_not_fetch(*args, **kwargs):
        raise AssertionError("a keyless signed endpoint must not be requested")

    monkeypatch.setattr(brackets, "_http_get", must_not_fetch)
    with pytest.raises(NoMarketDataError) as excinfo:
        brackets.get_leverage_brackets("BTCUSDT")
    assert "BINANCE_API_KEY" in excinfo.value.detail

    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setattr(brackets, "_http_get", lambda *a, **k: [_bracket_payload()])
    assert [b.mmr for b in brackets.get_leverage_brackets("BTCUSDT")] == [0.004, 0.005]


def _bracket_payload():
    """A two-tier /fapi/v1/leverageBracket entry as Binance serves it."""
    return {
        "symbol": "BTCUSDT",
        "brackets": [
            {"bracket": 1, "initialLeverage": 125, "notionalFloor": 0,
             "maintMarginRatio": 0.004, "cum": 0.0},
            {"bracket": 2, "initialLeverage": 100, "notionalFloor": 50000,
             "maintMarginRatio": 0.005, "cum": 166.0},
        ],
    }


@pytest.mark.unit
def test_r16_brackets_signature_covers_exactly_the_sent_query(monkeypatch):
    """Pin: the signature is HMAC-SHA256 over the urlencoded query that is actually sent (params + timestamp + recvWindow, in order), the request carries that same dict plus the signature, and the API key travels only in the header."""
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    sent: dict = {}

    def fake_get(path, params, symbol, canonical, headers=None, **kwargs):
        sent.update(path=path, params=params, headers=headers, kwargs=kwargs)
        return []

    monkeypatch.setattr(brackets, "_http_get", fake_get)
    before = int(time.time() * 1000)
    brackets.signed_fapi_get(
        "/fapi/v1/leverageBracket", {"symbol": "BTCUSDT"}, "btcusdt", "BTCUSDT",
    )
    after = int(time.time() * 1000)

    unsigned = {k: v for k, v in sent["params"].items() if k != "signature"}
    assert list(unsigned) == ["symbol", "timestamp", "recvWindow"]
    assert before <= unsigned["timestamp"] <= after
    assert unsigned["recvWindow"] == "5000"
    expected = hmac.new(
        b"test-secret", urllib.parse.urlencode(unsigned).encode(), hashlib.sha256,
    ).hexdigest()
    assert sent["params"]["signature"] == expected
    assert sent["headers"] == {"X-MBX-APIKEY": "test-key"}
    assert sent["path"] == "/fapi/v1/leverageBracket"
    assert sent["kwargs"] == {}  # signed traffic stays on the fapi default host/budget


@pytest.mark.unit
def test_r16_brackets_pair_tuple_ladder_carries_no_leverage_cap():
    """Pin: bracket_max_leverage_for returns None for (floor, mmr) pair ladders — callers keep their static ceiling rather than inventing a cap; Bracket entries in a mixed ladder still tier by floor."""
    pairs = [(0.0, 0.004), (50_000.0, 0.005)]
    assert brackets.bracket_max_leverage_for(pairs, 60_000.0) is None
    assert brackets.bracket_max_leverage_for(pairs, 10.0) is None

    mixed = [(0.0, 0.004), brackets.Bracket(50_000.0, 0.005, 75.0)]
    assert brackets.bracket_max_leverage_for(mixed, 10_000.0) is None
    assert brackets.bracket_max_leverage_for(mixed, 60_000.0) == 75.0
    # A tier whose source carried no initialLeverage caps at None, not 0.
    no_cap = [brackets.Bracket(0.0, 0.004, 125.0), brackets.Bracket(1e6, 0.0125)]
    assert brackets.bracket_max_leverage_for(no_cap, 2e6) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        [{"symbol": "BTCUSDT", "brackets": []}],
        [{"symbol": "BTCUSDT", "brackets": [
            {"notionalFloor": "abc", "maintMarginRatio": 0.004},
            {"notionalFloor": 0},
        ]}],
        [{"symbol": "BTCUSDT", "brackets": None}],
    ],
)
def test_r16_brackets_unparseable_ladder_raises_and_is_not_cached(monkeypatch, payload):
    """Pin: a signed reply with zero parseable tiers raises NoMarketDataError — never an empty ladder a caller would read as "no maintenance margin" — and the failure is not cached."""
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    replies = iter([payload, [_bracket_payload()]])
    monkeypatch.setattr(brackets, "_http_get", lambda *a, **k: next(replies))

    with pytest.raises(NoMarketDataError):
        brackets.get_leverage_brackets("BTCUSDT")
    assert len(brackets.get_leverage_brackets("BTCUSDT")) == 2


@pytest.mark.unit
def test_r16_brackets_parsed_ladder_drops_bad_tiers_sorts_and_caches_by_canonical(monkeypatch):
    """Pin: tiers with missing/non-numeric fields are dropped, survivors are sorted ascending by notional floor (the tiering helpers need ascending order), and the TTL cache is keyed by canonical symbol so aliases share it."""
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    calls: list[str] = []

    def fake_get(path, params, symbol, canonical, headers=None, **kwargs):
        calls.append(canonical)
        return [{
            "symbol": "BTCUSDT",
            "brackets": [
                {"initialLeverage": 100, "notionalFloor": 50000,
                 "maintMarginRatio": "0.005"},
                {"initialLeverage": 125, "notionalFloor": "0",
                 "maintMarginRatio": "0.004"},
                {"initialLeverage": 50, "notionalFloor": "oops",
                 "maintMarginRatio": "0.0125"},
                {"initialLeverage": 75, "notionalFloor": 150000,
                 "maintMarginRatio": None},
                {"initialLeverage": 50, "notionalFloor": 1_000_000,
                 "maintMarginRatio": "0.0125"},
            ],
        }]

    monkeypatch.setattr(brackets, "_http_get", fake_get)
    ladder = brackets.get_leverage_brackets("btc-usdt")
    assert [b.notional_floor for b in ladder] == [0.0, 50000.0, 1_000_000.0]
    assert [b.mmr for b in ladder] == [0.004, 0.005, 0.0125]
    assert [b.max_leverage for b in ladder] == [125.0, 100.0, 50.0]

    assert brackets.get_leverage_brackets("BTCUSDT") is ladder
    assert calls == ["BTCUSDT"]


@pytest.mark.unit
def test_r16_brackets_mmr_tier_is_inclusive_at_floor_and_default_ladder_unchanged():
    """Pin: mmr_for_notional picks the tier whose floor is <= notional (boundary inclusive) and returns 0.0 below a ladder's first floor — the behavior the engine's liquidation math is built on."""
    ladder = brackets.default_brackets()
    assert brackets.mmr_for_notional(ladder, 49_999.99) == 0.004
    assert brackets.mmr_for_notional(ladder, 50_000.0) == 0.005
    assert brackets.mmr_for_notional(ladder, 1e12) == 0.5
    assert brackets.mmr_for_notional([(1_000.0, 0.004)], 500.0) == 0.0
    assert brackets.default_brackets() is not ladder  # a fresh mutable copy
