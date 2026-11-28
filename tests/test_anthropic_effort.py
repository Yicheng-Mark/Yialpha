"""Tests for Anthropic effort-parameter gating (#831).

Haiku (any version) and Sonnet 4.5 reject the ``effort`` parameter with a
400. Only Opus 4.5+ and Sonnet 4.6+ accept it. The gate uses a per-family
minimum version so future ``claude-{opus,sonnet}-X-Y`` releases inherit
support automatically.
"""

import pytest

from yialpha.llm_clients import anthropic_client as mod

# Model names used below that the model catalog (``VALID_MODELS["anthropic"]``)
# does not list. The gate is version arithmetic on the NAME (Haiku never,
# Sonnet >= 4.6, Opus >= 4.5), so it has to be exercised on names the catalog
# does not carry; each of them makes ``get_llm()`` emit the unknown-model
# RuntimeWarning. ``_get_llm_kwargs`` asserts that warning for exactly these
# names, so the file is warning-clean without dropping a gate case.
_OFF_CATALOG = frozenset({
    "claude-haiku-5-0", "claude-haiku-4-7-preview",
    "claude-sonnet-4-5", "claude-sonnet-4-0", "claude-sonnet-5-0",
    "claude-opus-4-5", "claude-opus-5-0",
    "claude-mythos-preview", "claude-experimental-x",
})


def _get_llm_kwargs(monkeypatch, model, **client_kwargs):
    """Build ``AnthropicClient(model, **client_kwargs).get_llm()``; return the kwargs
    the chat class was constructed with."""
    captured: dict = {}
    monkeypatch.setattr(
        mod, "NormalizedChatAnthropic",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    client = mod.AnthropicClient(model=model, **client_kwargs)
    if model in _OFF_CATALOG:
        with pytest.warns(RuntimeWarning, match="not in the known model list"):
            client.get_llm()
    else:
        client.get_llm()
    return captured["kwargs"]


@pytest.mark.unit
class TestEffortGate:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5", "claude-haiku-5-0", "claude-haiku-4-7-preview",
            # Sonnet 4.5 (and earlier) 400 on effort — only Sonnet 4.6+ supports it.
            "claude-sonnet-4-5", "claude-sonnet-4-0",
        ],
    )
    def test_unsupported_models_do_not_receive_effort(self, monkeypatch, model):
        kwargs = _get_llm_kwargs(monkeypatch, model, effort="medium", api_key="x")
        assert "effort" not in kwargs

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7",
            "claude-sonnet-4-6",
        ],
    )
    def test_current_opus_and_sonnet_receive_effort(self, monkeypatch, model):
        kwargs = _get_llm_kwargs(monkeypatch, model, effort="high", api_key="x")
        assert kwargs["effort"] == "high"

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5-0", "claude-opus-4-8", "claude-sonnet-5-0"],
    )
    def test_future_opus_sonnet_inherit_effort_via_pattern(self, monkeypatch, model):
        """Forward-compat: new Opus/Sonnet versions don't need a code change."""
        kwargs = _get_llm_kwargs(monkeypatch, model, effort="low", api_key="x")
        assert kwargs["effort"] == "low"

    def test_mythos_preview_receives_effort(self, monkeypatch):
        kwargs = _get_llm_kwargs(
            monkeypatch, "claude-mythos-preview", effort="medium", api_key="x"
        )
        assert kwargs["effort"] == "medium"

    def test_unknown_anthropic_model_does_not_receive_effort(self, monkeypatch):
        """Default is conservative — unknown models don't get effort to avoid 400s."""
        kwargs = _get_llm_kwargs(
            monkeypatch, "claude-experimental-x", effort="medium", api_key="x"
        )
        assert "effort" not in kwargs

    def test_other_kwargs_still_forwarded_when_effort_skipped(self, monkeypatch):
        """Skipping effort must not break other passthrough kwargs."""
        kwargs = _get_llm_kwargs(
            monkeypatch,
            "claude-haiku-4-5",
            effort="medium",
            api_key="placeholder",
            max_tokens=1024,
            timeout=30,
        )
        assert kwargs["api_key"] == "placeholder"
        assert kwargs["max_tokens"] == 1024
        assert kwargs["timeout"] == 30
        assert "effort" not in kwargs
