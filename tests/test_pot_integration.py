"""Unit tests for the PoT integration helper (LLM codegen -> sandbox)."""

from __future__ import annotations

import pytest

from yiagents.agents.utils.pot_integration import (
    PotAnalyzer,
    extract_code_block,
)


class FakeLLM:
    """Returns a queue of canned responses (strings or AIMessage-like)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return self._responses.pop(0)


class FakeMessage:
    def __init__(self, content):
        self.content = content


@pytest.mark.unit
def test_extract_code_block_python_fence():
    text = "Here:\n```python\nresult = 1+1\n```\nDone."
    assert extract_code_block(text) == "result = 1+1"


@pytest.mark.unit
def test_extract_code_block_plain_fence():
    assert extract_code_block("```\nresult = 5\n```") == "result = 5"


@pytest.mark.unit
def test_extract_code_block_no_fence_returns_raw():
    assert extract_code_block("result = 9") == "result = 9"


@pytest.mark.unit
def test_compute_happy_path_string_response():
    llm = FakeLLM(["```python\nresult = (prices[-1]/prices[0] - 1) * 100\n```"])
    analyzer = PotAnalyzer(llm)
    out = analyzer.compute("What is the total return %?", data={"prices": [100, 110]})
    assert out.ok is True
    assert out.result == pytest.approx(10.0)
    assert out.attempts == 1


@pytest.mark.unit
def test_compute_happy_path_aimessage_response():
    llm = FakeLLM([FakeMessage("```python\nresult = float(np.mean(values))\n```")])
    analyzer = PotAnalyzer(llm)
    out = analyzer.compute("Mean?", data={"values": [2, 4, 6]})
    assert out.ok is True
    assert out.result == pytest.approx(4.0)


@pytest.mark.unit
def test_compute_repairs_after_failure():
    # First response divides by zero; second fixes it.
    llm = FakeLLM([
        "```python\nresult = 10 / 0\n```",
        "```python\nresult = 10 / 2\n```",
    ])
    analyzer = PotAnalyzer(llm, max_retries=1)
    out = analyzer.compute("Compute", data={})
    assert out.ok is True
    assert out.result == pytest.approx(5.0)
    assert out.attempts == 2


@pytest.mark.unit
def test_compute_gives_up_after_retries():
    llm = FakeLLM([
        "```python\nresult = 1/0\n```",
        "```python\nresult = 1/0\n```",
    ])
    analyzer = PotAnalyzer(llm, max_retries=1)
    out = analyzer.compute("Compute", data={})
    assert out.ok is False
    assert out.error is not None
    assert out.attempts == 2


@pytest.mark.unit
def test_compute_handles_no_code_block():
    llm = FakeLLM(["I cannot compute that."])
    analyzer = PotAnalyzer(llm, max_retries=0)
    out = analyzer.compute("Compute", data={})
    assert out.ok is False
    assert "no code block" in out.error


@pytest.mark.unit
def test_compute_llm_invoke_failure_returns_error():
    class BoomLLM:
        def invoke(self, prompt):
            raise RuntimeError("api down")
    analyzer = PotAnalyzer(BoomLLM())
    out = analyzer.compute("Compute", data={})
    assert out.ok is False
    assert "llm invoke failed" in out.error


@pytest.mark.unit
def test_compute_prompt_includes_data_and_question():
    llm = FakeLLM(["```python\nresult = 1\n```"])
    analyzer = PotAnalyzer(llm)
    analyzer.compute("What is the P/E?", data={"price": 100, "eps": 5})
    assert "What is the P/E?" in llm.calls[0]
    assert '"price"' in llm.calls[0]
