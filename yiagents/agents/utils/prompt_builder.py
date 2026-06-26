"""FinCoT-style structured prompt builder (Phase 2b).

The roadmap cites FinCoT as lifting accuracy ~17pp while compressing output
~8.9x. The mechanism is structural, not persona-based: replace the chatty
"You are a ... Analyst" framing with a tight three-part prompt --

  1. Task definition   (what to produce)
  2. Reasoning steps    (a chain of structured analysis steps, optionally a
                         Mermaid workflow so the model follows a fixed order)
  3. Output constraints (format, grounding rules, what NOT to claim)

This module builds those sections deterministically so every analyst can adopt
the same compact shape. Callers keep their domain content (indicator catalogs,
tool lists) and only swap the framing.
"""

from __future__ import annotations

from collections.abc import Sequence


def build_mermaid_workflow(steps: Sequence[str]) -> str:
    """Render ordered reasoning steps as a compact Mermaid flowchart.

    Gives the model a fixed sequence to follow (task -> structured reasoning
    -> output), which is the structure that drives FinCoT's gains. Mermaid is
    rendered as plain text the model reads; it is not executed.
    """
    if not steps:
        return ""
    safe = [str(s).replace('"', "'") for s in steps]
    lines = ["flowchart TD", f'    S([Start]) --> N1["{safe[0]}"]']
    for i in range(1, len(safe)):
        lines.append(f'    N{i}["{safe[i-1]}"] --> N{i+1}["{safe[i]}"]')
    lines.append(f'    N{len(safe)}["{safe[-1]}"] --> E([Output])')
    return "\n".join(lines)


def build_fincot_prompt(
    *,
    task: str,
    reasoning_steps: Sequence[str],
    output_constraints: Sequence[str],
    context: str | None = None,
    include_workflow: bool = True,
) -> str:
    """Compose a de-persona, three-section structured prompt.

    Parameters
    ----------
    task:
        One or two sentences defining what to produce (the "task definition").
    reasoning_steps:
        Ordered analysis steps the model must work through. Rendered both as a
        numbered list and (optionally) a Mermaid workflow.
    output_constraints:
        Hard rules on format, grounding, and what must not be asserted.
    context:
        Optional preamble (e.g. instrument identity, available tools) inserted
        before the task definition.
    include_workflow:
        When True, also render the steps as a Mermaid flowchart for visual
        structure. Disable to save tokens when the steps are very short.
    """
    parts: list[str] = []

    if context:
        parts.append(context.strip())
        parts.append("")

    parts.append("## Task")
    parts.append(task.strip())
    parts.append("")

    parts.append("## Reasoning steps")
    for i, step in enumerate(reasoning_steps, 1):
        parts.append(f"{i}. {step}")
    parts.append("")

    if include_workflow and reasoning_steps:
        parts.append("```mermaid")
        parts.append(build_mermaid_workflow(reasoning_steps))
        parts.append("```")
        parts.append("")

    parts.append("## Output constraints")
    for c in output_constraints:
        parts.append(f"- {c}")

    return "\n".join(parts).strip() + "\n"
