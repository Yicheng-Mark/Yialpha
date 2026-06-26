"""Program-of-Thoughts integration: LLM emits code, the sandbox runs the numbers.

The PoT executor (:mod:`yiagents.agents.utils.pot_executor`) runs Python in
a restricted sandbox. This module is the glue that asks an LLM to *produce* that
code for a numerical question, runs it, and -- on failure -- feeds the error
back for one repair pass. The roadmap's goal ("analysts' support/resistance,
percentage moves, valuation ratios must go through PoT, not mental arithmetic")
is realized by routing numeric claims through :class:`PotAnalyzer`.

Keeps the analyst LLMs in charge of *which* numbers matter; this layer only
guarantees the arithmetic is correct by externalizing it to Python.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from yiagents.agents.utils.pot_executor import PoTExecutor, PoTResult

logger = logging.getLogger(__name__)


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class PotAnalysis:
    """Outcome of one PoT compute attempt."""

    question: str
    code: str
    result: Any
    ok: bool
    error: str | None
    attempts: int


_SYSTEM_PROMPT = (
    "You are a numerical-analysis code generator. Given a question and a JSON-like "
    "`data` object, write a short Python snippet that computes the answer. "
    "Rules:\n"
    "1. Use only `np` (numpy) and `pd` (pandas); they are pre-imported in the sandbox. "
    "Do NOT write `import` statements.\n"
    "2. The variables in `data` are already available by name.\n"
    "3. Assign the final numeric answer to a variable named `result`.\n"
    "4. Output ONLY a single ```python``` code block. No prose.\n"
)


def _build_prompt(question: str, data: dict | None, hint: str) -> str:
    import json

    payload = json.dumps(data or {}, default=str, ensure_ascii=False)
    hint_line = f"\nHint: {hint}" if hint else ""
    return (
        f"{_SYSTEM_PROMPT}\n"
        f"Question: {question}\n"
        f"data = {payload}{hint_line}\n"
        f"\nWrite the Python snippet now."
    )


def extract_code_block(text: str) -> str:
    """Pull the first ```python``` block out of an LLM response.

    When there is no fence, only treat the raw text as code if it looks like
    code (contains an assignment or a ``result`` reference); otherwise return
    "" so a prose refusal ("I cannot compute that.") is reported as
    "no code block" rather than executed as a broken program.
    """
    if not text:
        return ""
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    candidate = text.strip()
    if "=" in candidate or "result" in candidate:
        return candidate
    return ""


class PotAnalyzer:
    """Drive LLM code generation + sandboxed execution for numerical questions."""

    def __init__(self, llm: Any, executor: PoTExecutor | None = None, max_retries: int = 1):
        self.llm = llm
        self.executor = executor or PoTExecutor()
        self.max_retries = max(0, int(max_retries))

    def compute(self, question: str, data: dict | None = None, hint: str = "") -> PotAnalysis:
        """Generate code for ``question`` and run it; repair once on failure."""
        prompt = _build_prompt(question, data, hint)
        last: PotAnalysis | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.invoke(prompt)
            except Exception as exc:  # noqa: BLE001 -- LLM failure shouldn't crash the caller
                logger.warning("PoT LLM invoke failed: %s", exc)
                return PotAnalysis(question=question, code="", result=None, ok=False,
                                   error=f"llm invoke failed: {exc}", attempts=attempt + 1)

            code = extract_code_block(_content(response))
            if not code:
                last = PotAnalysis(question=question, code="", result=None, ok=False,
                                   error="no code block in LLM response", attempts=attempt + 1)
                continue

            pot: PoTResult = self.executor.run_sandboxed(code, data=data)
            if pot.ok:
                return PotAnalysis(question=question, code=code, result=pot.result,
                                   ok=True, error=None, attempts=attempt + 1)

            last = PotAnalysis(question=question, code=code, result=None, ok=False,
                               error=pot.error, attempts=attempt + 1)
            # Repair pass: feed the error back so the model can fix it.
            prompt = (
                f"{prompt}\n\n"
                f"Your previous attempt failed:\n```python\n{code}\n```\n"
                f"Error: {pot.error}\nFix it and output the corrected snippet."
            )

        return last  # type: ignore[return-value]


def _content(response: Any) -> str:
    """Normalize an LLM response (AIMessage or str) to its text content."""
    if isinstance(response, str):
        return response
    return getattr(response, "content", "") or ""
