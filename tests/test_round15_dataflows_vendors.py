"""Round-15 coverage: vendor dataflows render + degrade branches.

Fresh paths for the Alpha Vantage common client / news tools, the FRED macro
vendor, the DefiLlama crypto-native vendor, and the Polymarket prediction-
market vendor: the global-news window builder, entitlement forwarding, dict
legacy payloads, FRED's raw ``_request`` transport, DefiLlama's
``_http_get_json`` error mapping / chain-catalog resolution / per-leg
fail-soft verdicts, and Polymarket's parse + malformed-date guards.

All HTTP seams are mocked (module-level ``_get`` helpers, ``requests.get``,
``_request``); no network, no real LLM, no sleeps.
"""
import json
import os
from datetime import UTC, date, datetime, timedelta

import pytest

import yialpha.dataflows.alpha_vantage_common as av
import yialpha.dataflows.alpha_vantage_news as avn
import yialpha.dataflows.defillama_vendor as dl
from yialpha.dataflows import fred, polymarket
from yialpha.dataflows.disk_cache import MinIntervalThrottle
from yialpha.dataflows.errors import NoMarketDataError, VendorRateLimitError
from yialpha.dataflows.utils import set_analysis_date


# --------------------------------------------------------------------------- #
# Shared fakes
# --------------------------------------------------------------------------- #
class _TextResponse:
    """Minimal requests.Response for the AV client (body + no-op status)."""

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class _FredResponse:
    """requests.Response stand-in for FRED (status, body, JSON, text)."""

    def __init__(self, *, status_code=200, payload=None, text="", json_error=None):
        self.status_code = status_code
        self.content = (
            json.dumps(payload).encode("utf-8") if payload is not None else b"{}"
        )
        self.text = text or self.content.decode("utf-8")
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload

    def raise_for_status(self):
        pass


class _DLResponse:
    """requests.Response stand-in for DefiLlama's _http_get_json."""

    def __init__(self, *, status_code=200, payload=None, text="", json_error=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def _dl_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, 12, tzinfo=UTC).timestamp())


# --------------------------------------------------------------------------- #
# alpha_vantage_common — key guard, date formatting, entitlement, payloads
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_round15_avc_get_api_key_raises_typed_error_when_env_missing(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(av.AlphaVantageNotConfiguredError) as ei:
        av.get_api_key()
    assert "ALPHA_VANTAGE_API_KEY" in str(ei.value)
    # Routing contract: unconfigured surfaces as ValueError so the router's
    # "vendor unavailable" handling keeps working.
    assert isinstance(ei.value, ValueError)


@pytest.mark.unit
def test_round15_avc_format_datetime_passthrough_datetimeobj_and_minute_form():
    # Already in API form (len 13 with T): returned untouched.
    assert av.format_datetime_for_api("20250102T1430") == "20250102T1430"
    # "YYYY-MM-DD HH:MM" parses to the minute-precision form.
    assert av.format_datetime_for_api("2025-01-02 14:30") == "20250102T1430"
    # datetime objects format directly.
    assert av.format_datetime_for_api(datetime(2025, 1, 2, 14, 30)) == "20250102T1430"


@pytest.mark.unit
def test_round15_avc_format_datetime_rejects_bad_string_and_type():
    with pytest.raises(ValueError, match="Unsupported date format"):
        av.format_datetime_for_api("2025/01/02")
    with pytest.raises(ValueError, match="Date must be string or datetime"):
        av.format_datetime_for_api(20250102)


@pytest.mark.unit
def test_round15_avc_raw_request_entitlement_forwarded_and_dropped(monkeypatch):
    captured = []

    def fake_get(url, params=None, **kwargs):
        captured.append(params)
        return _TextResponse("timestamp,close\n2025-01-02,1.0")

    monkeypatch.setattr(av.requests, "get", fake_get)
    # Truthy per-call entitlement (premium tier) reaches the query params.
    av._raw_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL", "entitlement": "realtime"})
    assert captured[-1]["entitlement"] == "realtime"
    # A falsy entitlement entry is removed entirely, not sent as None.
    av._raw_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL", "entitlement": None})
    assert "entitlement" not in captured[-1]
    assert captured[-1]["function"] == "TIME_SERIES_DAILY"
    assert captured[-1]["apikey"]


@pytest.mark.unit
def test_round15_avc_raw_request_returns_plain_json_body(monkeypatch):
    body = '{"Global Quote": {"05. price": "150.00"}}'
    monkeypatch.setattr(
        av.requests, "get", lambda url, params=None, **kw: _TextResponse(body)
    )
    # A JSON body without Information/Note/Error Message is plain data.
    assert av._raw_api_request("GLOBAL_QUOTE", {"symbol": "AAPL"}) == body


@pytest.mark.unit
def test_round15_avc_make_api_request_serializes_legacy_dict_payload(monkeypatch):
    calls = {"n": 0}

    def fake_raw(function_name, params):
        calls["n"] += 1
        return {"Time Series (Daily)": {"2025-01-02": {"4. close": "1.0"}}}

    monkeypatch.setattr(av, "_raw_api_request", fake_raw)
    first = av._make_api_request("TIME_SERIES_DAILY", {"symbol": "LEGACY"})
    second = av._make_api_request("TIME_SERIES_DAILY", {"symbol": "LEGACY"})
    # The unlikely dict return is serialized (bytes in the cache) and the
    # result is stable JSON text; the second call is served from disk.
    assert json.loads(first) == {"Time Series (Daily)": {"2025-01-02": {"4. close": "1.0"}}}
    assert first == second
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# alpha_vantage_news — feed passthrough + the untested get_global_news path
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_round15_avn_get_news_passes_nonempty_feed_and_params(monkeypatch):
    body = '{"feed": [{"title": "Markets calm ahead of Fed"}]}'
    captured = {}

    def fake_api(function_name, params):
        captured["fn"] = function_name
        captured.update(params)
        return body

    monkeypatch.setattr(avn, "_make_api_request", fake_api)
    out = avn.get_news("MU", "2026-03-01", "2026-03-10")
    assert out == body  # non-empty feed passes through untouched
    assert captured["fn"] == "NEWS_SENTIMENT"
    assert captured["tickers"] == "MU"
    assert captured["time_from"] == "20260301T0000"
    assert captured["time_to"] == "20260310T0000"


@pytest.mark.unit
def test_round15_avn_global_news_builds_topic_window_and_limit(monkeypatch):
    captured = {}

    def fake_api(function_name, params):
        captured["fn"] = function_name
        captured.update(params)
        return '{"feed": [{"title": "Macro headline"}]}'

    monkeypatch.setattr(avn, "_make_api_request", fake_api)
    out = avn.get_global_news("2026-03-10", look_back_days=7, limit=5)
    assert '"Macro headline"' in out
    assert captured["fn"] == "NEWS_SENTIMENT"
    assert captured["topics"] == "financial_markets,economy_macro,economy_monetary"
    assert captured["time_from"] == "20260303T0000"  # 7 days back
    assert captured["time_to"] == "20260310T0000"
    assert captured["limit"] == "5"


@pytest.mark.unit
def test_round15_avn_global_news_clamps_to_pinned_analysis_date(monkeypatch):
    captured = {}

    def fake_api(function_name, params):
        captured.update(params)
        return '{"feed": []}'

    monkeypatch.setattr(avn, "_make_api_request", fake_api)
    set_analysis_date("2026-03-05")
    try:
        with pytest.raises(NoMarketDataError):  # clamped window is empty
            avn.get_global_news("2026-03-10", look_back_days=7)
    finally:
        set_analysis_date(None)
    assert captured["time_to"] == "20260305T0000"  # clamped, not 2026-03-10


@pytest.mark.unit
def test_round15_avn_global_news_empty_feed_is_typed_soft_miss(monkeypatch):
    monkeypatch.setattr(avn, "_make_api_request", lambda fn, p: '{"feed": []}')
    with pytest.raises(NoMarketDataError) as ei:
        avn.get_global_news("2026-03-10")
    assert ei.value.symbol == "GLOBAL_NEWS"
    assert "empty news feed" in ei.value.detail


@pytest.mark.unit
def test_round15_avn_global_news_datetime_form_curr_date_normalized(monkeypatch):
    """Regression (round-15): a datetime-form curr_date ("2026-03-10 16:00:00")
    used to hit bare strptime ValueError (uncatched by the router); the sibling
    yfinance tool got the [:10] normalize in round 11 and this one now matches."""
    captured = {}

    def fake_api(function_name, params):
        captured.update(params)
        return '{"feed": []}'

    monkeypatch.setattr(avn, "_make_api_request", fake_api)
    with pytest.raises(NoMarketDataError):  # empty feed, not a ValueError crash
        avn.get_global_news("2026-03-10 16:00:00", look_back_days=7)
    assert captured["time_to"] == "20260310T0000"


# --------------------------------------------------------------------------- #
# fred — get_api_key success + the raw _request transport + render degrade
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_round15_fred_get_api_key_returns_env_value(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "round15-fred-key")
    assert fred.get_api_key() == "round15-fred-key"


@pytest.mark.unit
def test_round15_fred_request_builds_url_params_and_parses_json(monkeypatch):
    captured = {}
    payload = {"observations": [{"date": "2026-01-01", "value": "1.0"}]}

    def fake_get(url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = kwargs.get("timeout")
        return _FredResponse(payload=payload)

    monkeypatch.setenv("FRED_API_KEY", "round15-fred-key")
    monkeypatch.setattr(fred.requests, "get", fake_get)
    out = fred._request("series/observations", {"series_id": "CPIAUCSL"})
    assert out == payload
    assert captured["url"] == f"{fred.FRED_API_BASE}/series/observations"
    assert captured["params"]["series_id"] == "CPIAUCSL"
    assert captured["params"]["api_key"] == "round15-fred-key"
    assert captured["params"]["file_type"] == "json"
    assert captured["timeout"] == fred.REQUEST_TIMEOUT


@pytest.mark.unit
def test_round15_fred_request_400_json_body_raises_actionable_error(monkeypatch):
    resp = _FredResponse(
        status_code=400,
        payload={"error_message": "The series does not exist."},
    )
    monkeypatch.setenv("FRED_API_KEY", "round15-fred-key")
    monkeypatch.setattr(fred.requests, "get", lambda url, **kw: resp)
    with pytest.raises(ValueError, match="series does not exist"):
        fred._request("series/observations", {"series_id": "NOPE"})


@pytest.mark.unit
def test_round15_fred_request_400_non_json_body_falls_back_to_text(monkeypatch):
    resp = _FredResponse(
        status_code=400,
        payload=None,
        text="<html>400 Bad Request</html>",
        json_error=ValueError("no json"),
    )
    monkeypatch.setenv("FRED_API_KEY", "round15-fred-key")
    monkeypatch.setattr(fred.requests, "get", lambda url, **kw: resp)
    with pytest.raises(ValueError, match="400 Bad Request"):
        fred._request("series", {"series_id": "NOPE"})


@pytest.mark.unit
def test_round15_fred_request_identical_calls_served_from_disk_cache(monkeypatch):
    calls = {"n": 0}
    payload = {"observations": [{"date": "2026-01-01", "value": "2.0"}]}

    def fake_get(url, params=None, **kwargs):
        calls["n"] += 1
        return _FredResponse(payload=payload)

    monkeypatch.setenv("FRED_API_KEY", "round15-fred-key")
    monkeypatch.setattr(fred.requests, "get", fake_get)
    first = fred._request("series/observations", {"series_id": "UNRATE"})
    second = fred._request("series/observations", {"series_id": "UNRATE"})
    assert calls["n"] == 1  # 1-day TTL cache collapses the repeat call
    assert first == second == payload


@pytest.mark.unit
def test_round15_fred_nonnumeric_values_drop_change_summary(monkeypatch):
    meta = {
        "seriess": [{
            "title": "Odd Series", "units_short": "idx",
            "frequency": "Monthly", "seasonal_adjustment_short": "NSA",
        }]
    }
    obs = {
        "observations": [
            {"date": "2026-01-01", "value": "4.1"},
            {"date": "2026-02-01", "value": "AB"},  # non-numeric: kept as a row
        ]
    }

    def stub(path, params):
        return meta if path == "series" else obs

    monkeypatch.setattr(fred, "_request", stub)
    out = fred.get_macro_data("unemployment", "2026-02-28", 90)
    assert "**Latest:** AB (2026-02-01)" in out
    assert "Change over window" not in out  # delta uncomputable -> omitted
    assert "| 2026-01-01 | 4.1 |" in out


# --------------------------------------------------------------------------- #
# defillama — _http_get_json error mapping, memo fallback, resolution, verdicts
# --------------------------------------------------------------------------- #
@pytest.fixture()
def _dl_no_throttle(monkeypatch):
    monkeypatch.setattr(dl, "_throttler", MinIntervalThrottle(0.0))


@pytest.mark.unit
def test_round15_dl_http_get_json_success_parses_payload(_dl_no_throttle, monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _DLResponse(payload={"name": "Ethereum", "tvl": 5.2e10})

    monkeypatch.setattr("requests.get", fake_get)
    out = dl._http_get_json("https://api.llama.fi/v2/chains")
    assert out == {"name": "Ethereum", "tvl": 5.2e10}
    assert captured["url"] == "https://api.llama.fi/v2/chains"
    assert captured["headers"]["User-Agent"] == dl._UA
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["timeout"] == dl._TIMEOUT


@pytest.mark.unit
def test_round15_dl_http_get_json_transport_failure_maps_to_no_data(
    _dl_no_throttle, monkeypatch,
):
    import requests

    def fake_get(url, **kwargs):
        raise requests.exceptions.ConnectionError("reset by peer")

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(NoMarketDataError, match="DefiLlama request failed"):
        dl._http_get_json("https://api.llama.fi/protocols")


@pytest.mark.unit
def test_round15_dl_http_get_json_429_maps_to_rate_limit(_dl_no_throttle, monkeypatch):
    monkeypatch.setattr(
        "requests.get",
        lambda url, **kw: _DLResponse(status_code=429, text="too many"),
    )
    with pytest.raises(VendorRateLimitError, match="429"):
        dl._http_get_json("https://api.llama.fi/protocols")


@pytest.mark.unit
def test_round15_dl_http_get_json_5xx_maps_to_no_data_with_excerpt(
    _dl_no_throttle, monkeypatch,
):
    resp = _DLResponse(status_code=503, text="service unavailable\nlong body")
    monkeypatch.setattr("requests.get", lambda url, **kw: resp)
    with pytest.raises(NoMarketDataError, match="DefiLlama HTTP 503"):
        dl._http_get_json("https://api.llama.fi/protocols")


@pytest.mark.unit
def test_round15_dl_http_get_json_non_json_body_maps_to_no_data(
    _dl_no_throttle, monkeypatch,
):
    resp = _DLResponse(text="<html>gateway</html>", json_error=ValueError("no json"))
    monkeypatch.setattr("requests.get", lambda url, **kw: resp)
    with pytest.raises(NoMarketDataError, match="non-JSON"):
        dl._http_get_json("https://api.llama.fi/protocols")


@pytest.mark.unit
def test_round15_dl_cached_json_mtime_unavailable_still_serves(tmp_path, monkeypatch):
    """When the cache file's mtime cannot be stat'd (OSError), the parse memo
    degrades to re-parsing the disk bytes — the fetch itself is NOT re-run."""
    monkeypatch.setattr(dl, "_cache_dir", lambda: str(tmp_path))
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"v": 7}

    real_getmtime = os.path.getmtime

    def raising_getmtime(path):
        if os.path.basename(str(path)) == "round15_memo_oserror.json":
            raise OSError("stat unavailable")
        return real_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", raising_getmtime)
    first = dl._cached_json("round15_memo_oserror.json", fetch, ttl_days=1.0)
    second = dl._cached_json("round15_memo_oserror.json", fetch, ttl_days=1.0)
    assert first == second == {"v": 7}
    assert calls["n"] == 1  # disk cache served both; only the memo was skipped


@pytest.mark.unit
def test_round15_dl_chain_for_base_catalog_token_symbol_resolution(monkeypatch):
    catalog = [
        {"name": "Junk Astra", "tokenSymbol": "ASTR", "tvl": 4_000.0},  # below floor
        {"name": "Astra", "tokenSymbol": "ASTR", "tvl": 6.0e8},
        {"name": "Astra Clone", "tokenSymbol": "ASTR", "tvl": 9.0e8},   # highest wins
        {"name": "NoTvl", "tokenSymbol": "ASTR"},                       # non-numeric tvl
        {"name": "Other", "tokenSymbol": "XYZ", "tvl": 9.0e9},
    ]
    monkeypatch.setattr(dl, "_chains_catalog", lambda: catalog)
    # ASTR is NOT in the static _BASE_CHAIN map, so resolution reaches the
    # catalog's tokenSymbol ladder.
    assert "ASTR" not in dl._BASE_CHAIN
    assert dl._chain_for_base("ASTR") == "Astra Clone"
    # Unmatched base: honest None (no fabricated chain).
    assert dl._chain_for_base("NOPE") is None


@pytest.mark.unit
def test_round15_dl_pit_window_skips_malformed_points():
    points = [
        "not-a-dict",                                   # non-dict entry skipped
        7,
        {"unknown": 1},                                 # missing keys
        {"date": None, "tvl": 1.0},                     # int(None) TypeError
        {"date": _dl_ts(date(2026, 8, 31)), "tvl": "x"},  # float() ValueError
        {"date": _dl_ts(date(2026, 9, 1)), "tvl": 4.4e9},  # valid
    ]
    out = dl._pit_window(points, "date", "tvl", "2026-09-02", 30)
    assert out == [("2026-09-01", 4.4e9)]


def _dl_fakes(protocol_rows=None, chain_rows=None):
    """URL-dispatching _http_get_json fake for the end-to-end verdict tests."""

    def fake(url):
        if url.endswith("/protocols"):
            return [
                {"slug": "uniswap-v3", "name": "Uniswap V3", "symbol": "UNI",
                 "tvl": 1.68e9, "chains": ["Ethereum"], "gecko_id": "uniswap",
                 "category": "Dexs"},
            ]
        if "/protocol/uniswap-v3" in url:
            if protocol_rows is None:
                raise NoMarketDataError("defillama", detail="protocol history down")
            return {"tvl": [
                {"date": _dl_ts(date(2026, 8, 30) + timedelta(days=i)),
                 "totalLiquidityUSD": 1.0e9 + i * 1.0e7}
                for i in range(protocol_rows)
            ]}
        if url.endswith("/v2/chains"):
            return [{"name": "Ethereum", "tokenSymbol": "ETH", "tvl": 5.2e10}]
        if "/historicalChainTvl/" in url:
            if chain_rows is None:
                raise NoMarketDataError("defillama", detail="chain history down")
            return [
                {"date": _dl_ts(date(2026, 8, 30) + timedelta(days=i)), "tvl": 5.0e10}
                for i in range(chain_rows)
            ]
        raise AssertionError(f"unexpected url {url}")

    return fake


@pytest.mark.unit
def test_round15_dl_multiplier_prefix_stripped_for_wrapper_contracts(monkeypatch):
    """1000PEPEUSDT tracks PEPE: the wrapper prefix is stripped before the
    protocol-symbol lookup, so the governance-token leg still resolves."""
    def fake(url):
        if url.endswith("/protocols"):
            return [
                {"slug": "pepe-protocol", "name": "Pepe Protocol", "symbol": "PEPE",
                 "tvl": 5.0e7, "chains": ["Ethereum"], "gecko_id": "pepe",
                 "category": "Meme"},
            ]
        if "/protocol/pepe-protocol" in url:
            return {"tvl": [
                {"date": _dl_ts(date(2026, 8, 30) + timedelta(days=i)),
                 "totalLiquidityUSD": 5.0e7}
                for i in range(4)
            ]}
        if url.endswith("/v2/chains"):
            return []
        if "/historicalChainTvl/" in url:
            return [{"date": _dl_ts(date(2026, 9, 1)), "tvl": 4.4e9}]
        raise AssertionError(url)

    monkeypatch.setattr(dl, "_http_get_json", fake)
    out = dl.get_crypto_network_fundamentals(
        "1000PEPEUSDT", curr_date="2026-09-02", look_back_days=30,
    )
    assert "slug 'pepe-protocol'" in out
    assert "carrying symbol PEPE" in out  # the UN-prefixed base, not 1000PEPE


@pytest.mark.unit
def test_round15_dl_protocol_resolution_failure_degrades_to_note(monkeypatch):
    """A failing /protocols catalog must not abort the chain leg: the run
    renders the chain evidence plus an honest unretrievable-sections note."""
    def fake(url):
        if url.endswith("/protocols"):
            raise NoMarketDataError("defillama", detail="catalog endpoint down")
        return _dl_fakes(chain_rows=4)(url)

    monkeypatch.setattr(dl, "_http_get_json", fake)
    out = dl.get_crypto_network_fundamentals(
        "BTCUSDT", curr_date="2026-09-02", look_back_days=30,
    )
    assert "Chain TVL — Bitcoin" in out
    assert "Unretrievable sections" in out
    assert "catalog endpoint down" in out


@pytest.mark.unit
def test_round15_dl_protocol_fetch_failure_preserves_chain_section(monkeypatch):
    """The per-leg fail-soft contract, mirrored: a broken protocol leg never
    discards the chain leg's rendered evidence (the chain falls back to the
    protocol's primary chain, then fetches its own history)."""
    monkeypatch.setattr(dl, "_http_get_json", _dl_fakes(protocol_rows=None, chain_rows=4))
    out = dl.get_crypto_network_fundamentals(
        "UNIUSDT", curr_date="2026-09-02", look_back_days=30,
    )
    assert "Chain TVL — Ethereum" in out
    assert "Unretrievable sections" in out
    assert "protocol history down" in out


@pytest.mark.unit
def test_round15_dl_chain_resolution_exception_falls_back_to_none(monkeypatch):
    """An unexpected (non-typed) error inside chain resolution is swallowed:
    the tool then honestly reports no resolvable chain instead of crashing."""
    def fake(url):
        if url.endswith("/protocols"):
            return []
        raise AssertionError(url)

    monkeypatch.setattr(dl, "_http_get_json", fake)
    monkeypatch.setattr(
        dl, "_chain_for_base", lambda base: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(NoMarketDataError, match="no DefiLlama coverage"):
        dl.get_crypto_network_fundamentals("BTCUSDT", curr_date="2026-09-02")


@pytest.mark.unit
def test_round15_dl_live_ranking_missing_chain_returns_empty(monkeypatch):
    """Live run whose chain is absent from the chains catalog: the ranking
    quietly renders nothing (no fabricated rank) and the chain leg survives."""
    def fake(url):
        if url.endswith("/protocols"):
            return []
        if url.endswith("/v2/chains"):
            return [{"name": "Ethereum", "tvl": 5.2e10}]  # Bitcoin not listed
        if "/historicalChainTvl/Bitcoin" in url:
            return [{"date": _dl_ts(date(2026, 1, 2)), "tvl": 5.0e10}]
        raise AssertionError(url)

    monkeypatch.setattr(dl, "_http_get_json", fake)
    out = dl.get_crypto_network_fundamentals("BTCUSDT", look_back_days=4000)
    assert "Chain TVL — Bitcoin" in out
    assert "ranks #" not in out  # unmatched chain -> empty ranking string


@pytest.mark.unit
def test_round15_dl_live_chain_ranking_failure_is_advisory_only(monkeypatch):
    """A chains-catalog failure during the LIVE ranking lookup is logged and
    skipped: ranking is advisory context, never a leg failure."""
    def fake(url):
        if url.endswith("/protocols"):
            return []
        if "/historicalChainTvl/Bitcoin" in url:
            return [{"date": _dl_ts(date(2026, 1, 2)), "tvl": 5.0e10}]
        raise AssertionError(url)

    monkeypatch.setattr(dl, "_http_get_json", fake)
    monkeypatch.setattr(
        dl, "_chains_catalog",
        lambda: (_ for _ in ()).throw(NoMarketDataError("defillama", detail="down")),
    )
    out = dl.get_crypto_network_fundamentals("BTCUSDT", look_back_days=4000)
    assert "Chain TVL — Bitcoin" in out
    assert "ranks #" not in out
    assert "Unretrievable sections" not in out  # advisory: not a failure note


@pytest.mark.unit
def test_round15_dl_all_legs_failing_to_render_raises_first_failure(monkeypatch):
    """Both legs RESOLVE but neither FETCHES: the first per-leg failure is
    re-raised so the router records the optional-category sentinel."""
    monkeypatch.setattr(dl, "_http_get_json", _dl_fakes(None, None))
    with pytest.raises(NoMarketDataError, match="protocol history down"):
        dl.get_crypto_network_fundamentals(
            "UNIUSDT", curr_date="2026-09-02", look_back_days=30,
        )


# --------------------------------------------------------------------------- #
# polymarket — JSON-string helpers + malformed endDate/price guards
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_round15_poly_parse_json_list_passthrough_and_invalid():
    assert polymarket._parse_json_list(["Yes", "No"]) == ["Yes", "No"]  # already list
    assert polymarket._parse_json_list('["0.6", "0.4"]') == ["0.6", "0.4"]
    assert polymarket._parse_json_list("not json") == []
    assert polymarket._parse_json_list(None) == []


@pytest.mark.unit
def test_round15_poly_unparseable_end_date_treated_as_forward_looking():
    from datetime import datetime as _dt

    now = _dt(2030, 1, 1, tzinfo=UTC)
    market = {
        "closed": False,
        "endDate": "not-a-date",  # unparseable: kept forward-looking by contract
        "outcomePrices": '["0.6", "0.4"]',
        "outcomes": '["Yes", "No"]',
    }
    assert polymarket._is_forward_looking(market, now) is True
    # Date-only endDate parses naive and is pinned to UTC (never TypeError).
    market["endDate"] = "2030-12-31"
    assert polymarket._is_forward_looking(market, now) is True


@pytest.mark.unit
def test_round15_poly_non_numeric_price_market_skipped_in_render(monkeypatch):
    """A market whose first outcome price is not a float passes the
    forward-looking filter but is skipped at render time — no crash, no line."""
    market = {
        "question": "Unparseable odds?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["oops", "0.4"]',
        "volumeNum": 1234.0,
        "endDate": "2030-12-31T00:00:00Z",
        "closed": False,
        "oneWeekPriceChange": None,
    }
    monkeypatch.setattr(
        polymarket, "_request",
        lambda path, params: {"events": [{"markets": [market]}]},
    )
    out = polymarket.get_prediction_markets("weird topic")
    assert "Unparseable odds?" not in out
    assert out.startswith("## Polymarket prediction markets")  # header survives
