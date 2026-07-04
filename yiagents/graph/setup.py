# yiagents/graph/setup.py

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from yiagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from yiagents.agents.utils.agent_states import AgentState

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        *,
        perf_tracker: Any = None,
        analyst_parallel: bool = False,
        analyst_parallel_max_threads: int = 16,
    ):
        """Initialize with required components.

        The three keyword-only opts are all OFF/disjoint by default so the
        compiled graph is byte-identical to the historical serial one:

        * ``perf_tracker`` — when not None, every node handler is wrapped with
          :func:`yiagents.graph.perf_telemetry.wrap_node` to record per-node
          wall time. ``None`` = pass-through, no behaviour change.
        * ``analyst_parallel`` — when False (default) the 4 analysts are wired
          serially exactly as before. When True they collapse into a single
          ``Analyst Fanout`` node (see ``analyst_fanout.py``).
        * ``analyst_parallel_max_threads`` — cap on the fan-out thread pool.
        """
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.perf_tracker = perf_tracker
        self.analyst_parallel = analyst_parallel
        self.analyst_parallel_max_threads = analyst_parallel_max_threads

    def _wrap_node(self, handler, name: str):
        """Wrap a node handler with perf timing when a tracker is configured,
        otherwise pass it through untouched (byte-identical to no telemetry).

        ``ToolNode`` is deliberately NOT wrapped: it is a Runnable that
        LangGraph invokes via ``.invoke()``, not a plain callable, so wrapping
        it in a function (which then calls ``tool_node(state)``) raises
        ``'ToolNode' object is not callable``. Tool-call wall time is small and
        folds into the surrounding analyst node's flow; the dominant LLM cost
        is captured by the (wrapped) agent nodes. Every other node handler is a
        plain ``def fn(state)`` closure and wraps safely."""
        if self.perf_tracker is None:
            return handler
        if isinstance(handler, ToolNode):
            return handler
        from .perf_telemetry import wrap_node

        return wrap_node(handler, name, self.perf_tracker)

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        # T2: optional parallel analysts. When off (default) the serial chain
        # below is byte-identical to today. When on, a single "Analyst Fanout"
        # node replaces the 12-node analyst chain — each analyst runs in its
        # own sub-graph (independent state), so the shared ``messages`` /
        # ``clear_node`` coupling that assumes serial execution is structurally
        # avoided. Only engages the fan-out; every downstream node (debate,
        # trader, risk debate, PM) stays serial as before.
        fanout_node = None
        if self.analyst_parallel:
            from .analyst_fanout import create_analyst_fanout_node

            fanout_node = create_analyst_fanout_node(
                plan,
                analyst_factories,
                self.tool_nodes,
                self.conditional_logic,
                max_threads=self.analyst_parallel_max_threads,
            )

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Helper: register a node, wrapping it for perf telemetry only when a
        # tracker is configured (pass-through otherwise).
        def _add(name, handler):
            workflow.add_node(name, self._wrap_node(handler, name))

        # Add analyst nodes to the graph.
        if fanout_node is not None:
            # Parallel mode: one wrapper node fanning out to per-analyst
            # sub-graphs. The 12 serial analyst/clear/tool nodes are NOT added
            # at the parent level in this mode.
            _add("Analyst Fanout", fanout_node)
        else:
            # Serial mode (default): identical to the historical wiring.
            for spec in plan.specs:
                _add(spec.agent_node, analyst_factories[spec.key]())
                _add(spec.clear_node, create_msg_delete())
                _add(spec.tool_node, self.tool_nodes[spec.key])

        # Add other nodes (present in both modes)
        _add("Bull Researcher", bull_researcher_node)
        _add("Bear Researcher", bear_researcher_node)
        _add("Research Manager", research_manager_node)
        _add("Trader", trader_node)
        _add("Aggressive Analyst", aggressive_analyst)
        _add("Neutral Analyst", neutral_analyst)
        _add("Conservative Analyst", conservative_analyst)
        _add("Portfolio Manager", portfolio_manager_node)

        # Define edges
        if fanout_node is not None:
            # Parallel mode: START -> fan-out -> Bull Researcher. Downstream
            # edges (Bull Researcher onward) are identical to serial mode.
            workflow.add_edge(START, "Analyst Fanout")
            workflow.add_edge("Analyst Fanout", "Bull Researcher")
        else:
            # Serial mode: chain the analysts, then hand off to Bull Researcher.
            workflow.add_edge(START, plan.specs[0].agent_node)
            for i, spec in enumerate(plan.specs):
                current_analyst = spec.agent_node
                current_tools = spec.tool_node
                current_clear = spec.clear_node

                workflow.add_conditional_edges(
                    current_analyst,
                    getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                    [current_tools, current_clear],
                )
                workflow.add_edge(current_tools, current_analyst)

                if i < len(plan.specs) - 1:
                    workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
                else:
                    workflow.add_edge(current_clear, "Bull Researcher")

        # Remaining edges (identical in both modes — they start at Bull Researcher)
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
