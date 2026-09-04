"""LangChain ``@tool`` wrapper over the LLM-code-generation PoT analyzer.

The fundamentals analyst already has :func:`get_valuation_metrics` for
*deterministic* valuation formulas (Graham, DCF, WACC — pre-coded in
:mod:`yialpha.dataflows.valuation_methods`). But analysts sometimes need to
compute *ad-hoc* numbers that don't have a pre-coded formula:

- "What percentile is this stock's P/E within its sector over the last 5 years?"
- "What is the implied annual growth rate from the current price vs. DCF?"
- "What is the debt-to-equity ratio after adjusting for operating leases?"

For these, :class:`~yialpha.agents.utils.pot_integration.PotAnalyzer` asks the
LLM to write a short Python snippet, runs it in the restricted sandbox
(:mod:`yialpha.agents.utils.pot_executor`), and returns the computed answer.
This is the Program-of-Thoughts pattern: the LLM decides *what* to compute,
Python guarantees the *arithmetic* is correct.

This tool is gated onto the fundamentals analyst by the same
``valuation_tools`` config flag (``YIALPHA_VALUATION_TOOLS``). When off, the
tool list and prompt are byte-for-byte unchanged.

The LLM instance is injected at tool-construction time (the analyst factory
already has the bound ``llm``). The tool's own LLM call is a *separate*
invoke on the same client — it does not participate in the tool-call loop of
the analyst; it is a one-shot code-gen + sandbox-run that returns a result
string.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.tools import tool

from yialpha.agents.utils.pot_integration import PotAnalyzer

logger = logging.getLogger(__name__)


def make_pot_compute_tool(llm: Any):
    """Build a LangChain ``@tool`` that runs ad-hoc numerical computations via PoT.

    The tool closure captures ``llm`` so the PoT code-generation call goes to
    the same provider/model the analyst is using. Returns the tool function
    (to be appended to the analyst's tool list when ``valuation_tools`` is on).
    """

    analyzer = PotAnalyzer(llm)

    @tool
    def pot_compute(
        question: Annotated[str, "the numerical question to compute, e.g. 'What is the debt-to-equity ratio given total_debt=5000 and equity=12000?'"],
        data_json: Annotated[str, "JSON object of named values the computation needs, e.g. '{\"total_debt\": 5000, \"equity\": 12000}'. Keys become variable names in the sandbox."] = "{}",
    ) -> str:
        """Compute an exact numerical answer by writing and running Python code.

        Use this for ad-hoc calculations that don't have a pre-built formula
        tool: ratios, percentiles, growth rates, percentage changes, conversions.
        Pass the raw numbers you've gathered as a JSON object; the tool generates
        Python code to compute the answer and runs it in a sandbox (numpy +
        pandas available as ``np`` and ``pd``). If the computation fails, the
        error is returned so you can retry with corrected inputs.

        Do NOT use this for data retrieval — use the dedicated data tools for
        that. This tool is for *arithmetic on numbers you already have*.
        """
        import json

        try:
            data = json.loads(data_json) if data_json else {}
        except json.JSONDecodeError as exc:
            return f"PoT error: invalid JSON in data_json ({exc})"
        if not isinstance(data, dict):
            return (
                "PoT error: data_json must be a JSON object (dict of named "
                f"values), got {type(data).__name__}."
            )

        analysis = analyzer.compute(question, data=data)
        if analysis.ok:
            return (
                f"PoT result: {analysis.result}\n"
                f"(computed by Python code, {analysis.attempts} attempt(s))\n"
                f"```python\n{analysis.code}\n```"
            )
        return (
            f"PoT failed after {analysis.attempts} attempt(s): "
            f"{analysis.error or 'unknown error'}. "
            f"Falling back — please compute manually or retry with simpler inputs."
        )

    return pot_compute
