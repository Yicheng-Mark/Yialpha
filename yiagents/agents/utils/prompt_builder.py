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

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_collaborator_prompt(include_tools: bool) -> ChatPromptTemplate:
    """The shared 'collaborating with other assistants' analyst scaffolding.

    Every tool-calling analyst opens with the same collaborator framing:
    multi-assistant handoff, the ``FINAL TRANSACTION PROPOSAL`` stop signal,
    the current-date 'now' anchor, and the ``{system_message}`` body.
    ``include_tools=True`` additionally advertises the tool list (``{tool_names}``)
    and extends the date anchor to 'tool-call date ranges'; the no-tool
    sentiment analyst (``include_tools=False``) omits both and carries no
    ``{tool_names}`` slot.

    Centralising this block keeps a cross-cutting change (e.g. the
    ``NO_EXTERNAL_TOOLS`` rollout) to one edit instead of N near-identical
    copies that silently drift. The returned template still needs the
    ``{system_message}``, ``{current_date}``, ``{instrument_context}`` (and,
    when ``include_tools``, ``{tool_names}``) partials applied by the caller.

    Byte-equivalent to each analyst's prior inline ``ChatPromptTemplate`` -- the
    embedded text is the exact block the analysts already shared.
    """
    if include_tools:
        system = (
            "You are a helpful AI assistant, collaborating with other assistants."
            " Use the provided tools to progress towards answering the question."
            " If you are unable to fully answer, that's OK; another assistant with different tools"
            " will help where you left off. Execute what you can to make progress."
            " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
            " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
            " You have access to the following tools: {tool_names}."
            " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
            "{system_message}"
        )
    else:
        system = (
            "You are a helpful AI assistant, collaborating with other assistants."
            " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
            " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
            " Today's date is {current_date}; treat it as 'now' for all analysis. {instrument_context}"
            "\n{system_message}"
        )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


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
