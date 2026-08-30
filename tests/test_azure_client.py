"""Tests for the Azure OpenAI client wrapper (yialpha/llm_clients/azure_client.py).

Azure keys on a *deployment name* created in the portal rather than a model id,
and validate_model() always returns True (any deployed name is accepted). These
tests pin that contract plus the passthrough-kwarg wiring, using the same
monkeypatch-kwarg-capture pattern as test_anthropic_effort.py.
"""

import warnings

import pytest

from yialpha.llm_clients import azure_client as mod


def _capture_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        mod, "NormalizedAzureChatOpenAI",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    return captured


@pytest.mark.unit
class TestAzureClient:
    def test_deployment_name_from_env(self, monkeypatch):
        """AZURE_OPENAI_DEPLOYMENT_NAME overrides the model as azure_deployment."""
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "my-gpt5-deploy")
        captured = _capture_kwargs(monkeypatch)
        mod.AzureOpenAIClient(model="gpt-5").get_llm()
        assert captured["kwargs"]["azure_deployment"] == "my-gpt5-deploy"

    def test_deployment_name_falls_back_to_model(self, monkeypatch):
        """No deployment env -> azure_deployment defaults to the model arg."""
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)
        captured = _capture_kwargs(monkeypatch)
        mod.AzureOpenAIClient(model="gpt-5").get_llm()
        assert captured["kwargs"]["azure_deployment"] == "gpt-5"

    def test_passthrough_kwargs_forwarded(self, monkeypatch):
        """timeout/max_retries/api_key/temperature are in _PASSTHROUGH_KWARGS."""
        captured = _capture_kwargs(monkeypatch)
        client = mod.AzureOpenAIClient(
            model="gpt-5",
            timeout=30,
            max_retries=3,
            api_key="placeholder",
            temperature=0.7,
        )
        client.get_llm()
        assert captured["kwargs"]["timeout"] == 30
        assert captured["kwargs"]["max_retries"] == 3
        assert captured["kwargs"]["api_key"] == "placeholder"
        assert captured["kwargs"]["temperature"] == 0.7

    def test_non_passthrough_kwargs_not_forwarded(self, monkeypatch):
        """A kwarg not in _PASSTHROUGH_KWARGS must not leak to the LLM ctor."""
        captured = _capture_kwargs(monkeypatch)
        mod.AzureOpenAIClient(model="gpt-5", unrelated_flag=True).get_llm()
        assert "unrelated_flag" not in captured["kwargs"]

    def test_validate_model_always_true(self):
        """Azure accepts any deployed model name -> no warning, always valid."""
        client = mod.AzureOpenAIClient(model="anything-deployed-here")
        assert client.validate_model() is True
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any RuntimeWarning -> failure
            client.warn_if_unknown_model()
