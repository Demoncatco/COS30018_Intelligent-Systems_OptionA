# LangGraph workflow: decide an action, run one node, then decide again
# All tool and synthesis nodes return to decide_action; stop routes to END
# The decision guards bound the loop, while the wrapper records progress and tool failures

import logging
from typing import Any, Callable, Protocol

from langgraph.graph import END, StateGraph

from .llm_client import call_qwen
from .nodes.decide_action import DEFAULT_MAX_STEPS, decide_action
from .nodes.query_doc_store import query_doc_store
from .nodes.synthesize import synthesize
from .nodes.web_search import web_search
from .state import RecommendationState

logger = logging.getLogger("recommendation_agent.graph")

# Record retrieval and search failures so the loop can continue
# Model failures still propagate to the caller
TOOL_NODES = {"query_doc_store", "web_search"}


class GraphNode(Protocol):
    """Describe the named state callback expected by LangGraph without wrapping it at runtime"""
    def __call__(self, state: RecommendationState) -> RecommendationState: ...


def initial_state(
    finding: str,
    user_query: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    impression: str | None = None,
    patient_context: str | None = None,
) -> RecommendationState:
    """Start a fresh loop for one finding with empty history, trace and error lists"""
    state: RecommendationState = {
        "finding": finding,
        "max_steps": max_steps,
        "step_count": 0,
        "actions_taken": [],
        "trace": [],
        "errors": [],
    }
    if user_query:
        state["user_query"] = user_query
    if impression:
        state["impression"] = impression
    if patient_context:
        state["patient_context"] = patient_context
    return state


def _summarize(name: str, state: RecommendationState) -> str:
    """Build the node-specific part of the trace line without changing graph state"""
    if name == "decide_action":
        line = f"action={route_after_decide(state)}"
        if state.get("action_used_fallback"):
            line += " (model reply unparseable, used fallback)"
        note = state.get("decision_note")
        if note:
            line += f" [{note}]"
        return line
    if name == "query_doc_store":
        titles = [c["source_title"] for c in state.get("retrieved_chunks", [])]
        return f"{len(titles)} chunks {titles}"
    if name == "web_search":
        kept = len(state.get("web_results", []))
        dropped = state.get("web_results_filtered_count", 0)
        return f"{kept} trusted results kept, {dropped} untrusted dropped"
    if name == "synthesize":
        preview = state.get("recommendation", "")[:90].replace("\n", " ")
        return f'wrote recommendation (disclaimer_added={state.get("disclaimer_added")}) "{preview}..."'
    return ""


def _instrument(
    name: str,
    fn: GraphNode,
) -> GraphNode:
    """Wrap a node with trace logging, action history and recoverable tool-error handling"""
    def node(state: RecommendationState) -> RecommendationState:
        """Copy the node result before adding bookkeeping so incoming state is not overwritten"""
        errors = list(state.get("errors", []))
        try:
            new_state = fn(state).copy()
        except Exception as exc:
            # Source-tool failures are recoverable; model failures still reach the caller
            if name not in TOOL_NODES:
                raise
            message = f"{name} failed: {exc.__class__.__name__}: {exc}"
            errors.append(message)
            new_state = state.copy()
            logger.warning("[step %s] %s ERROR -> %s", state.get("step_count", 0), name, message)

        if name != "decide_action":
            new_state["actions_taken"] = list(state.get("actions_taken", [])) + [name]
        new_state["errors"] = errors

        summary = _summarize(name, new_state)
        counts = (
            f"chunks={len(new_state.get('retrieved_chunks', []))} "
            f"web={len(new_state.get('web_results', []))} "
            f"recommendation={'yes' if new_state.get('recommendation') else 'no'}"
        )
        line = f"[step {new_state.get('step_count', 0)}] {name:<16} -> {summary}   | {counts}"
        logger.info(line)
        new_state["trace"] = list(state.get("trace", [])) + [line]
        return new_state

    return node


def route_after_decide(state: RecommendationState) -> str:
    """Read the chosen route and preserve a missing-action KeyError"""
    if "action" not in state:
        raise KeyError("action")
    return state["action"]


def _bind(
    fn: Callable[..., RecommendationState], **kwargs: Any
) -> GraphNode:
    """Bind supplied service replacements while preserving node defaults for omitted arguments"""
    given = {k: v for k, v in kwargs.items() if v is not None}
    return lambda state: fn(state, **given)


def build_graph(
    llm: Callable | None = None,
    retrieve_fn: Callable | None = None,
    search_fn: Callable | None = None,
):
    """Wire decide_action to the tools, synthesis and stop routes, then compile the graph
    Service replacements allow the same graph to run with either real integrations or test fakes
    """
    llm = llm or call_qwen

    graph = StateGraph(RecommendationState)
    graph.add_node("decide_action", _instrument("decide_action", _bind(decide_action, llm=llm)))
    graph.add_node("query_doc_store", _instrument("query_doc_store", _bind(query_doc_store, retrieve_fn=retrieve_fn)))
    graph.add_node("web_search", _instrument("web_search", _bind(web_search, search_fn=search_fn)))
    graph.add_node("synthesize", _instrument("synthesize", _bind(synthesize, llm=llm)))

    # Start with a decision; completed actions return here for another decision
    graph.set_entry_point("decide_action")
    # This mapping is the only exit from the loop: stop becomes LangGraph END
    graph.add_conditional_edges(
        "decide_action",
        route_after_decide,
        {
            "query_doc_store": "query_doc_store",
            "web_search": "web_search",
            "synthesize": "synthesize",
            "stop": END,
        },
    )
    for tool in ("query_doc_store", "web_search", "synthesize"):
        graph.add_edge(tool, "decide_action")

    return graph.compile()


def run(finding: str, user_query: str | None = None, max_steps: int = DEFAULT_MAX_STEPS, **graph_kwargs):
    """Build and invoke the graph with a recursion allowance above the decision-step limit"""
    app = build_graph(**graph_kwargs)
    # Each loop is decide + one tool = 2 supersteps; the guards allow up to
    # max_steps + 2 decisions, so leave generous headroom over that
    return app.invoke(
        initial_state(finding, user_query, max_steps),
        config={"recursion_limit": 2 * (max_steps + 2) + 10},
    )
