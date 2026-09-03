"""GLM key pool: parsing, per-request rotation, failure policy, get_llm wiring.

Covers yialpha/llm_clients/key_pool.py — the comma-separated ZHIPU*_API_KEYS
pool (merged with the single-key var, deduped, order-stable), the
RotatingKeyAuth transport (next live key per request; 401/403 permanent, 429
cooldown), and the OpenAIClient.get_llm attachment (api_key from the pool,
rotating http_client overriding a forwarded keepalive client, coding-plan
base URL via ZHIPU_CN_BASE_URL).
"""

import httpx
import pytest

from yialpha.llm_clients.key_pool import (
    DEFAULT_COOLDOWN_S,
    PROVIDER_API_KEYS_ENV,
    RotatingKeyAuth,
    _PoolState,
    api_key_pool,
    cooldown_seconds,
    get_pool_http_client,
    has_pool,
    reset_for_test,
)

_K1 = "k1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_K2 = "k2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_K3 = "k3aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.fixture(autouse=True)
def _clean_pool_env(monkeypatch):
    """Isolate every pool var + reset the cached clients/states per test.

    conftest's autouse _dummy_api_keys sets ZHIPU_API_KEY / ZHIPU_CN_API_KEY
    to "placeholder"; delete them here so api_key_pool sees only what each
    test pins explicitly.
    """
    for env_var in ("ZHIPU_API_KEYS", "ZHIPU_CN_API_KEYS", "ZHIPU_CN_BASE_URL",
                    "ZHIPU_BASE_URL", "YIALPHA_LLM_KEY_COOLDOWN_S"):
        monkeypatch.delenv(env_var, raising=False)
    for env_var in ("ZHIPU_API_KEY", "ZHIPU_CN_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    reset_for_test()
    yield
    reset_for_test()


def _set_pool(monkeypatch, *keys, single=None):
    monkeypatch.setenv("ZHIPU_CN_API_KEYS", ",".join(keys))
    if single is not None:
        monkeypatch.setenv("ZHIPU_CN_API_KEY", single)


@pytest.mark.unit
def test_pool_parse_merge_and_dedup(monkeypatch):
    _set_pool(monkeypatch, _K1, _K2, _K1, single=_K3)
    # Order-stable: KEYS-var entries first (duplicates removed), single-key
    # var appended when not already present — the Tavily pool contract.
    assert api_key_pool("glm-cn") == [_K1, _K2, _K3]
    # Single key already in the pool is not duplicated.
    _set_pool(monkeypatch, _K1, single=_K1)
    assert api_key_pool("glm-cn") == [_K1]


@pytest.mark.unit
def test_has_pool_gate(monkeypatch):
    # Only the *_API_KEYS var engages the pool machinery; the single-key var
    # alone keeps the SDK default transport (byte-equivalent single setup).
    monkeypatch.setenv("ZHIPU_CN_API_KEY", _K1)
    assert has_pool("glm-cn") is False
    monkeypatch.setenv("ZHIPU_CN_API_KEYS", "")
    assert has_pool("glm-cn") is False
    monkeypatch.setenv("ZHIPU_CN_API_KEYS", f" {_K1} , ")
    assert has_pool("glm-cn") is True
    # Providers without a registered pool env never engage.
    assert has_pool("deepseek") is False
    assert PROVIDER_API_KEYS_ENV["glm-cn"] == "ZHIPU_CN_API_KEYS"


@pytest.mark.unit
def test_state_round_robin_and_death():
    state = _PoolState(3, cooldown_s=60.0)
    picks = [state.pick(), state.pick(), state.pick(), state.pick()]
    assert picks == [0, 1, 2, 0]  # cycles

    state.mark(0, 429)  # cooling down
    assert state.pick() == 1  # skips key 0
    assert state.pick() == 2

    state.mark(1, 401)  # permanent
    assert state.pick() == 2  # only live key left
    state.mark(2, 403)
    assert state.pick() is None  # nothing live
    # Force-pick prefers the cooling-down key 0 (recovers) over the permanent 1.
    assert state.force_pick() == 0


@pytest.mark.unit
def test_state_cooldown_expiry():
    state = _PoolState(1, cooldown_s=0.0)  # expires immediately
    state.mark(0, 429)
    assert state.pick() == 0  # revived


@pytest.mark.unit
def test_cooldown_env_override(monkeypatch):
    assert cooldown_seconds() == DEFAULT_COOLDOWN_S
    monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", "5")
    assert cooldown_seconds() == 5.0
    monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", "0")
    assert cooldown_seconds() == 1.0  # clamped to a sane floor
    monkeypatch.setenv("YIALPHA_LLM_KEY_COOLDOWN_S", "not-a-number")
    assert cooldown_seconds() == DEFAULT_COOLDOWN_S  # warned fallback


def _mock_client(responses_by_key, keys, state):
    """httpx.Client on a MockTransport that answers per Authorization key."""
    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization", "")
        seen_headers.append(auth)
        key = auth.removeprefix("Bearer ")
        status, body = responses_by_key.get(key, (200, '{"ok": true}'))
        return httpx.Response(status, json=body, request=request)

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=RotatingKeyAuth(keys, state),
    )
    return client, seen_headers


@pytest.mark.unit
def test_auth_rotates_on_401_and_recovers(monkeypatch):
    _set_pool(monkeypatch, _K1, _K2)
    state = _PoolState(2, cooldown_s=60.0)
    responses = {_K1: (401, '{"error": "bad key"}'), _K2: (200, '{"ok": true}')}
    client, seen = _mock_client(responses, [_K1, _K2], state)

    # The 401'd key is retried past within ONE request cycle.
    response = client.post("https://example.test/v1/chat/completions", json={})
    assert response.status_code == 200
    assert seen == [f"Bearer {_K1}", f"Bearer {_K2}"]

    # K1 is permanently dead: the next request goes straight to K2.
    seen.clear()
    response = client.post("https://example.test/v1/chat/completions", json={})
    assert response.status_code == 200
    assert seen == [f"Bearer {_K2}"]


@pytest.mark.unit
def test_auth_429_cools_down_then_returns(monkeypatch):
    _set_pool(monkeypatch, _K1, _K2)
    state = _PoolState(2, cooldown_s=60.0)
    responses = {
        _K1: (429, '{"error": "rate limited"}'),
        _K2: (429, '{"error": "rate limited"}'),
    }
    client, seen = _mock_client(responses, [_K1, _K2], state)

    # Both keys 429: the request cycles through the pool once and the LAST
    # failure propagates (the SDK/langchain retry path owns it from here).
    response = client.post("https://example.test/v1/chat/completions", json={})
    assert response.status_code == 429
    assert len(seen) == 2

    # Both are cooling down, so pick() returns None and the auth force-serves
    # the soonest-to-revive key instead of failing at the auth layer.
    seen.clear()
    response = client.post("https://example.test/v1/chat/completions", json={})
    assert response.status_code == 429
    assert len(seen) == 1


@pytest.mark.unit
def test_auth_non_key_failure_propagates_unrotated():
    # A 500 is not a key problem: no rotation, response returned untouched.
    keys = [_K1, _K2]
    state = _PoolState(2, cooldown_s=60.0)
    client, seen = _mock_client({_K1: (500, '{"error": "boom"}')}, keys, state)
    response = client.post("https://example.test/v1/chat/completions", json={})
    assert response.status_code == 500
    assert seen == [f"Bearer {_K1}"]
    assert state.pick() == 1  # nothing was marked dead


@pytest.mark.unit
def test_get_pool_http_client_cached(monkeypatch):
    _set_pool(monkeypatch, _K1, _K2)
    first = get_pool_http_client("glm-cn")
    assert isinstance(first, httpx.Client)
    assert get_pool_http_client("glm-cn") is first  # one shared client
    # No pool var -> None (SDK default transport preserved).
    monkeypatch.delenv("ZHIPU_CN_API_KEYS")
    assert get_pool_http_client("glm-cn") is None


@pytest.mark.unit
def test_get_llm_attaches_pool(monkeypatch):
    from yialpha.llm_clients.openai_client import OpenAIClient

    _set_pool(monkeypatch, _K1, _K2, _K3)
    monkeypatch.setenv("ZHIPU_CN_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4/")

    llm = OpenAIClient("glm-5.3", provider="glm-cn").get_llm()
    # ChatOpenAI carries a pool key for its own validation; the rotating
    # transport decides the real Authorization header per request.
    assert llm.openai_api_key.get_secret_value() == _K1
    assert llm.http_client is get_pool_http_client("glm-cn")
    # Coding-plan endpoint honored over the pay-as-you-go spec default.
    assert str(llm.openai_api_base) == "https://open.bigmodel.cn/api/coding/paas/v4/"
    # Pool providers default to wire-level streaming: on slow reasoning
    # endpoints the non-streaming read timeout bounds TOTAL generation time
    # (round-3a/3b both died at Market Analyst that way); streaming applies
    # it per chunk while keeping invoke()'s aggregated semantics.
    assert llm.streaming is True

    # A forwarded keepalive http_client (graph's YIALPHA_HTTP_KEEPALIVE path)
    # is superseded by the rotating pool client — losing it would silently
    # pin every call to one key.
    keepalive = httpx.Client()
    llm2 = OpenAIClient(
        "glm-5.3-flash", provider="glm-cn", http_client=keepalive
    ).get_llm()
    assert llm2.http_client is get_pool_http_client("glm-cn")
    assert llm2.http_client is not keepalive


@pytest.mark.unit
def test_get_llm_without_pool_unchanged(monkeypatch):
    from yialpha.llm_clients.openai_client import OpenAIClient

    # Single-key setup (no ZHIPU_CN_API_KEYS): today's behaviour — key from
    # the single env var, SDK default transport, spec default base URL, and
    # no wire-level streaming opt-in.
    monkeypatch.setenv("ZHIPU_CN_API_KEY", _K1)
    llm = OpenAIClient("glm-5.3", provider="glm-cn").get_llm()
    assert llm.openai_api_key.get_secret_value() == _K1
    assert str(llm.openai_api_base) == "https://open.bigmodel.cn/api/paas/v4/"
    assert llm.streaming is False


@pytest.mark.unit
def test_get_llm_pool_streaming_opt_out(monkeypatch):
    from yialpha.llm_clients.openai_client import OpenAIClient

    _set_pool(monkeypatch, _K1, _K2)
    monkeypatch.setenv("ZHIPU_CN_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4/")
    monkeypatch.setenv("YIALPHA_LLM_POOL_STREAMING", "false")
    llm = OpenAIClient("glm-5.3", provider="glm-cn").get_llm()
    # Explicit opt-out restores the exact non-streaming request path.
    assert llm.streaming is False
    assert llm.http_client is get_pool_http_client("glm-cn")


@pytest.mark.unit
def test_ensure_api_key_accepts_pool(monkeypatch):
    from yialpha.cli.utils import ensure_api_key

    _set_pool(monkeypatch, _K1, _K2)
    # No interactive prompt (a prompt would hang the test): the pool alone
    # satisfies the key requirement.
    assert ensure_api_key("glm-cn") == _K1
