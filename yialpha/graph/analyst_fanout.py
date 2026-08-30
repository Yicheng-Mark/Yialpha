"""Parallel analyst fan-out for YiAlpha.

Reproduces the per-analyst ``agent <-> tool <-> clear`` cluster that
``setup.py`` wires serially (one cluster per selected analyst), but runs all
clusters concurrently in a ``ThreadPoolExecutor``.

Iron law (HARD): serial-vs-parallel must produce an IDENTICAL decision
distribution. To guarantee that, each analyst's prompt, tool list, and
``messages`` input are constructed byte-identical to today's serial run:

* ``specs[0]`` receives the parent state's ``messages`` verbatim (the initial
  ``[HumanMessage(ticker)]`` reduced from ``propagation.create_initial_state``);
* ``specs[i>0]`` receive ``[placeholder]`` where ``placeholder`` is the exact
  :class:`~langchain_core.messages.HumanMessage` that :func:`create_msg_delete`
  would emit after the previous analyst's ``clear_node`` reduces away its
  messages. The placeholder depends only on ``instrument_context`` +
  ``trade_date``, so it is identical for every ``i>0`` — exactly as serial
  produces the same one each time.

The fan-out node only ever returns ``{spec.report_key: ...}`` for each spec —
it never writes ``messages`` back to the parent state (verified: no downstream
node reads ``state["messages"]``).

This module is plumbing only — it does NOT touch any agent factory,
``create_msg_delete``, ``agent_states.py``, ``propagation.py``, or
``conditional_logic.py``.
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from yialpha.agents.utils.agent_states import AgentState
from yialpha.agents.utils.agent_utils import (
    create_msg_delete,
    get_clear_placeholder_from_state,
)
from yialpha.dataflows.config import submit_with_context

from .analyst_execution import AnalystExecutionPlan, AnalystNodeSpec

logger = logging.getLogger(__name__)

__all__ = [
    "build_analyst_subgraph",
    "create_analyst_fanout_node",
    "derive_subgraph_recursion_limit",
]

#: Floor for the derived per-subgraph recursion budget. Protects against a
#: pathological serial budget (e.g. ``max_recur_limit=4``) producing a
#: per-analyst budget so small the subgraph cannot even finish one
#: agent -> tool -> agent -> clear cycle.
_MIN_SUBGRAPH_RECURSION_LIMIT = 10


def derive_subgraph_recursion_limit(max_recur_limit: int, n_specs: int) -> int:
    """Derive the per-analyst-subgraph recursion budget from the serial budget.

    Accounting note (parallel-vs-serial iron law): in serial mode all analyst
    clusters share ONE whole-graph recursion pool of ``max_recur_limit``
    steps (``propagation.py`` passes ``{"recursion_limit": max_recur_limit}``
    to the parent graph invoke, and every analyst/tool/clear beat consumes
    from it). In parallel mode each analyst subgraph gets its OWN budget, so
    the comparable allocation is ``max_recur_limit // n_specs`` per subgraph:
    the parallel worst case (all N subgraphs exhausting their budget, N *
    (max_recur_limit // N) ≈ max_recur_limit) matches the serial worst case.

    Residual, deliberate difference: the serial pool also pays for the
    non-analyst steps downstream of the analysts (debate, trader, risk
    debate, PM), so the derived parallel budget is if anything slightly MORE
    generous per analyst than serial. Without a per-analyst budget derived
    this way, a deep tool loop could trip ``GraphRecursionError`` in serial
    while succeeding in parallel — breaking the identical-decision-
    distribution guarantee in the opposite direction.

    A floor of :data:`_MIN_SUBGRAPH_RECURSION_LIMIT` guards against tiny
    serial budgets (``max_recur_limit < 10 * n_specs``) producing unusable
    per-analyst budgets.
    """
    if n_specs < 1:
        raise ValueError(f"n_specs must be >= 1, got {n_specs!r}")
    return max(max_recur_limit // n_specs, _MIN_SUBGRAPH_RECURSION_LIMIT)


def build_analyst_subgraph(
    spec: AnalystNodeSpec,
    agent_factory: Callable[[], Callable[..., Any]],
    tool_node: ToolNode | Callable[..., Any],
    conditional_logic_fn: Callable[..., Any],
    recursion_limit: int = _MIN_SUBGRAPH_RECURSION_LIMIT,
):
    """Build and compile a per-analyst ``StateGraph(AgentState)`` cluster.

    The cluster mirrors ``setup.py:82-119`` exactly for ONE analyst:

    * ``START -> spec.agent_node``
    * ``spec.agent_node --(conditional_logic_fn)--> [spec.tool_node, spec.clear_node]``
      (list path-map form, identical to ``setup.py:108-112``)
    * ``spec.tool_node -> spec.agent_node``  (tool loop)
    * ``spec.clear_node -> END``

    Node names and edge semantics are identical to the serial wiring, so each
    analyst's prompt / tool list / ``messages`` input is byte-identical to a
    serial run.

    Args:
        spec: The :class:`AnalystNodeSpec` describing node names + report key.
        agent_factory: Zero-arg factory returning the agent node fn (same shape
            as ``setup.py``'s ``analyst_factories[key]``).
        tool_node: The :class:`~langgraph.prebuilt.ToolNode` (or compatible
            callable) bound to this analyst's tool list.
        conditional_logic_fn: The bound ``should_continue_<key>`` method used as
            the conditional router (e.g. ``conditional_logic.should_continue_market``).
        recursion_limit: Per-subgraph recursion limit. NOTE: this langgraph
            version's ``compile()`` does not accept ``recursion_limit``; the
            effective value is applied at invoke time by
            :func:`create_analyst_fanout_node`, which derives it from the
            serial whole-graph budget via
            :func:`derive_subgraph_recursion_limit`. The parameter is
            retained for API stability + documentation of intent.

    Returns:
        The compiled subgraph (``CompiledStateGraph``).
    """
    workflow = StateGraph(AgentState)

    # Same three nodes + identical names as setup.py:82-85.
    workflow.add_node(spec.agent_node, agent_factory())
    workflow.add_node(spec.tool_node, tool_node)
    workflow.add_node(spec.clear_node, create_msg_delete())

    # Same edges as setup.py:99, 107-113, 119 (the per-analyst slice).
    workflow.add_edge(START, spec.agent_node)
    workflow.add_conditional_edges(
        spec.agent_node,
        conditional_logic_fn,
        [spec.tool_node, spec.clear_node],
    )
    workflow.add_edge(spec.tool_node, spec.agent_node)
    workflow.add_edge(spec.clear_node, END)

    return workflow.compile()


def _invoke_analyst_subgraph(
    subgraph: Any,
    clone: dict[str, Any],
    spec_key: str,
    invoke_config: dict[str, Any],
    wall_time_tracker: Any | None,
    perf_tracker: Any | None = None,
    perf_node_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run one subgraph and feed the optional wall-time tracker.

    Module-level helper so the ``ThreadPoolExecutor`` can ``submit`` it without
    closure-over-loop-variable pitfalls. ``mark_started`` / ``mark_completed``
    are only called when a tracker is supplied (duck-typed; no hard dependency
    on :class:`AnalystWallTimeTracker`).

    When ``perf_tracker`` is supplied (duck-typed
    :class:`~yialpha.graph.perf_telemetry.NodePerfTracker`), the worker
    thread's thread-local ``active_node`` is set to ``perf_node_name`` around
    the invoke and restored afterwards. Without this, every analyst LLM token
    captured inside the worker thread would be attributed to
    ``_unattributed_`` — ``wrap_node`` only wraps PARENT-graph nodes, and the
    worker thread's thread-local slot starts empty. Using the analyst's
    serial node name (``spec.agent_node``) keeps parallel-leg telemetry
    directly comparable with the serial leg.
    """
    if wall_time_tracker is not None:
        wall_time_tracker.mark_started(spec_key)
    previous_active_node = (
        perf_tracker.get_active_node() if perf_tracker is not None else None
    )
    if perf_tracker is not None and perf_node_name is not None:
        perf_tracker.set_active_node(perf_node_name)
    try:
        final_state = subgraph.invoke(clone, config=invoke_config)
        return spec_key, final_state
    finally:
        if perf_tracker is not None and perf_node_name is not None:
            perf_tracker.set_active_node(previous_active_node)
        if wall_time_tracker is not None:
            wall_time_tracker.mark_completed(spec_key)


def create_analyst_fanout_node(
    plan: AnalystExecutionPlan,
    analyst_factories: dict[str, Callable[[], Callable[..., Any]]],
    tool_nodes: dict[str, ToolNode | Callable[..., Any]],
    conditional_logic: Any,
    max_threads: int | None = None,
    wall_time_tracker: Any | None = None,
    recursion_limit: int | None = None,
    perf_tracker: Any | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a LangGraph node fn that runs all analyst subgraphs in parallel.

    The returned ``fanout_node(state)``:

    1. On FIRST call, builds (and caches on the closure) one compiled subgraph
       per spec via :func:`build_analyst_subgraph` — cached so we never rebuild
       on subsequent invokes.
    2. On each call: deep-copies the parent state per spec and applies the
       placeholder rule to set ``clone["messages"]`` (verbatim for ``specs[0]``,
       ``[placeholder]`` for ``specs[i>0]``). The clone is fully independent so
       subgraph mutations never leak into the parent state or sibling clones.
    3. Runs all subgraphs concurrently in a
       :class:`~concurrent.futures.ThreadPoolExecutor`, each
       ``subgraph.invoke(clone, config=...)``.
    4. Fails fast: if any subgraph raises, the remaining pending futures are
       cancelled and the exception is re-raised so the parent graph aborts the
       run — matching today's serial behaviour where one analyst exception
       aborts ``propagate()``. Errors are NEVER swallowed.
    5. Returns ``{spec.report_key: final_state[spec.report_key]}`` for every
       spec. It does NOT include ``messages`` or any other key.

    Args:
        plan: :class:`AnalystExecutionPlan` describing the selected analysts.
        analyst_factories: ``spec.key -> zero-arg factory`` returning the agent
            node fn (same shape as ``setup.py``'s ``analyst_factories``).
        tool_nodes: ``spec.key -> ToolNode`` (or compatible callable).
        conditional_logic: Object exposing ``should_continue_<key>`` methods
            (resolved via ``getattr``).
        max_threads: :class:`~concurrent.futures.ThreadPoolExecutor`
            ``max_workers``. ``None`` (default) means one worker per spec
            (``len(plan.specs)``).
        wall_time_tracker: Optional duck-typed tracker (e.g.
            :class:`~yialpha.graph.analyst_execution.AnalystWallTimeTracker`)
            with ``mark_started(key)`` / ``mark_completed(key)``. If provided,
            each subgraph invoke is timed so ``stream_telemetry=true`` can show
            per-analyst wall time in the parallel leg. Inert when ``None``.
        recursion_limit: Per-subgraph recursion limit, applied at invoke time.
            ``None`` (default) derives it from the thread-local config's
            ``max_recur_limit`` (the same serial whole-graph budget
            ``propagation.py`` uses) via
            :func:`derive_subgraph_recursion_limit`, so the parallel per-
            analyst budget stays comparable with the serial shared pool.
        perf_tracker: Optional duck-typed
            :class:`~yialpha.graph.perf_telemetry.NodePerfTracker`. When
            supplied, each worker thread sets its thread-local ``active_node``
            to the analyst's serial node name for the duration of the subgraph
            invoke, so LLM tokens spent inside the subgraph are attributed to
            that analyst instead of ``_unattributed_``. Inert when ``None``.

    Returns:
        A node fn ``fanout_node(state) -> dict`` suitable for
        ``workflow.add_node``.
    """
    if recursion_limit is None:
        # Read the serial budget from the thread-local config the graph was
        # built under (``YiAlphaGraph.__init__`` calls ``set_config`` before
        # building, so this sees the effective per-run value).
        from yialpha.dataflows.config import get_config

        serial_limit = int(get_config().get("max_recur_limit", 100))
        recursion_limit = derive_subgraph_recursion_limit(serial_limit, len(plan.specs))

    # Cached on the closure: built lazily on the first invoke so we don't pay
    # the build cost (or require factories to be ready) at graph-construction
    # time, and so we never rebuild on subsequent invokes.
    compiled_subgraphs: dict[str, Any] | None = None

    def _ensure_subgraphs() -> dict[str, Any]:
        nonlocal compiled_subgraphs
        if compiled_subgraphs is not None:
            return compiled_subgraphs
        built: dict[str, Any] = {}
        for spec in plan.specs:
            cond_fn = getattr(conditional_logic, f"should_continue_{spec.key}")
            built[spec.key] = build_analyst_subgraph(
                spec,
                analyst_factories[spec.key],
                tool_nodes[spec.key],
                cond_fn,
                recursion_limit=recursion_limit,
            )
        compiled_subgraphs = built
        return built

    def _clone_for_spec(
        spec_index: int,
        parent_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep-copy the parent state and apply the placeholder rule.

        ``copy.deepcopy`` guarantees the clone is independent — subgraph
        mutations (agent appends, ``clear_node`` removes) never leak into the
        parent state or sibling clones. For ``specs[0]`` the deepcopy already
        yields a verbatim, independent copy of ``messages``; for ``specs[i>0]``
        we overwrite ``messages`` with the single placeholder.
        """
        clone = copy.deepcopy(parent_state)
        if spec_index == 0:
            # specs[0] sees the parent's messages verbatim — identical to the
            # serial run, where the first analyst follows START directly and
            # observes the initial [HumanMessage(ticker)].
            return clone

        # specs[i>0] see exactly the placeholder that create_msg_delete would
        # emit after the previous analyst's clear_node. Built with the SAME
        # shared helper (get_clear_placeholder_from_state) so the string is
        # byte-identical to the serial path — the parallel iron law depends on
        # there being a SINGLE placeholder builder. Same placeholder for every
        # i>0 because serial produces the same one each time.
        clone["messages"] = [get_clear_placeholder_from_state(parent_state)]
        return clone

    def fanout_node(state: dict[str, Any]) -> dict[str, Any]:
        subgraphs = _ensure_subgraphs()
        workers = max_threads if max_threads is not None else len(plan.specs)
        invoke_config = {"recursion_limit": recursion_limit}
        # One independent clone per spec, with the placeholder rule applied.
        clones = [
            (spec, _clone_for_spec(index, state))
            for index, spec in enumerate(plan.specs)
        ]

        results: dict[str, dict[str, Any]] = {}
        first_exception: BaseException | None = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_spec_key: dict[concurrent.futures.Future, str] = {}
            for spec, clone in clones:
                future = submit_with_context(
                    executor,
                    _invoke_analyst_subgraph,
                    subgraphs[spec.key],
                    clone,
                    spec.key,
                    invoke_config,
                    wall_time_tracker,
                    perf_tracker,
                    spec.agent_node,
                )
                future_to_spec_key[future] = spec.key

            for future in concurrent.futures.as_completed(future_to_spec_key):
                try:
                    spec_key, final_state = future.result()
                except BaseException as exc:  # noqa: BLE001 — re-raised below
                    # Fail-fast: capture the first exception, cancel every
                    # still-pending future (best-effort; running futures can't
                    # be cancelled but we abort regardless), then stop waiting.
                    first_exception = exc
                    for pending in future_to_spec_key:
                        pending.cancel()
                    break
                results[spec_key] = final_state

        if first_exception is not None:
            # Match serial behaviour exactly: one analyst exception aborts the
            # whole run. Never swallow.
            raise first_exception

        # Return ONLY the report keys. Do NOT write messages (or anything else)
        # back to the parent state — downstream nodes don't read messages, and
        # each report field uses last-write-wins semantics so the merge is
        # independent per analyst.
        return {
            spec.report_key: results[spec.key][spec.report_key]
            for spec in plan.specs
        }

    # Introspection hook: exposes the per-subgraph recursion budget the node
    # applies at invoke time (derived from the serial budget unless given
    # explicitly), so tests / operators can verify the accounting without
    # provoking a GraphRecursionError.
    fanout_node.resolved_recursion_limit = recursion_limit  # type: ignore[attr-defined]
    return fanout_node
