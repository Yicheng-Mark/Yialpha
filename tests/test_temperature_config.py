"""Tests for the configurable sampling temperature (#178/#168).

Temperature is a cross-provider knob: when set it must reach the underlying
chat client; when unset the provider keeps its own default.
"""

import importlib

import pytest

from yialpha.llm_clients.factory import create_llm_client

# Model names used below that the catalog (``VALID_MODELS[provider]``) does not
# list; ``_llm_with`` asserts the unknown-model warning for exactly these. OpenAI
# needs one on purpose: its catalog lists GPT-5 models only, and langchain-openai
# drops ``temperature`` for GPT-5 when the chat client is built -- so gpt-4.1 is
# the name on which "temperature reaches the client" can be observed at all.
_OFF_CATALOG = frozenset({"gpt-4.1"})


def _llm_with(provider, model, **client_kwargs):
    """``create_llm_client(...).get_llm()`` for one provider / model pair."""
    client = create_llm_client(provider=provider, model=model, **client_kwargs)
    if model in _OFF_CATALOG:
        with pytest.warns(RuntimeWarning, match="not in the known model list"):
            return client.get_llm()
    return client.get_llm()


@pytest.fixture(autouse=True)
def _undo_default_config_reload():
    """Put ``yialpha.default_config`` back as it was before the test.

    ``TestTemperatureEnvOverlay`` re-evaluates the module with
    ``importlib.reload``, which re-executes it into its existing namespace: after
    a reload under a patched environment the module keeps the ``DEFAULT_CONFIG``
    built from that environment for the rest of the session (``monkeypatch``
    restores ``os.environ`` only). Snapshot the namespace and restore it after.
    """
    import yialpha.default_config as dc

    saved = dict(dc.__dict__)
    yield
    dc.__dict__.clear()
    dc.__dict__.update(saved)


@pytest.mark.unit
class TestTemperatureForwarding:
    @pytest.mark.parametrize(
        "provider,model",
        [
            ("openai", "gpt-4.1"),
            ("anthropic", "claude-sonnet-4-6"),
            ("google", "gemini-3.5-flash"),
            ("deepseek", "deepseek-flash"),
        ],
    )
    def test_temperature_reaches_client_when_set(self, provider, model):
        llm = _llm_with(provider, model, temperature=0.0, api_key="placeholder")
        assert llm.temperature == 0.0

    def test_temperature_omitted_leaves_provider_default(self):
        # Not passing temperature must not force it to a value.
        llm = _llm_with("openai", "gpt-4.1", api_key="placeholder")
        # langchain's default is unset/None, not 0.0
        assert llm.temperature is None


@pytest.mark.unit
class TestTemperatureEnvOverlay:
    def test_env_sets_temperature(self, monkeypatch):
        import yialpha.default_config as dc
        monkeypatch.setenv("YIALPHA_TEMPERATURE", "0.2")
        importlib.reload(dc)
        # Stored on config (string from env is fine; consumed via float()).
        assert dc.DEFAULT_CONFIG["temperature"] in ("0.2", 0.2)
        assert float(dc.DEFAULT_CONFIG["temperature"]) == 0.2

    def test_default_temperature_is_none(self, monkeypatch):
        import yialpha.default_config as dc
        monkeypatch.delenv("YIALPHA_TEMPERATURE", raising=False)
        importlib.reload(dc)
        assert dc.DEFAULT_CONFIG["temperature"] is None


@pytest.mark.unit
class TestProviderKwargsTemperature:
    """_get_provider_kwargs float-coerces and forwards temperature, or omits it."""

    def _kwargs_for(self, temperature):
        from yialpha.graph.trading_graph import YiAlphaGraph
        # Call the method without constructing the full graph.
        graph = YiAlphaGraph.__new__(YiAlphaGraph)
        graph.config = {"llm_provider": "openai", "temperature": temperature}
        return YiAlphaGraph._get_provider_kwargs(graph)

    def test_float_string_coerced(self):
        assert self._kwargs_for("0.3")["temperature"] == 0.3

    def test_float_passthrough(self):
        assert self._kwargs_for(0.0)["temperature"] == 0.0

    def test_none_omitted(self):
        assert "temperature" not in self._kwargs_for(None)

    def test_empty_string_omitted(self):
        assert "temperature" not in self._kwargs_for("")
