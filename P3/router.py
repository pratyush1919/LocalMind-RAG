"""
router.py  —  P3-T8
Router node: reads consensus_score and iteration to decide next step.
Also handles agent communication log serialisation.

Concepts covered:
  - Agentic Workflow Routing  (conditional edges based on state)
  - Agent Communication       (agent_messages serialised to JSONL after each run)
  - Workflow Visualization    (graph.get_graph().draw_mermaid() → DAG)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from P3.state import AgentState, make_message

logger = logging.getLogger(__name__)

_LOGS_DIR = Path("./logs")


def router_node(state: AgentState) -> dict:
    """
    Routing logic node. Reads consensus + iteration, writes next route.
    """
    consensus = state.get("consensus_score", 0.0)
    iteration = state.get("iteration", 0)
    is_final = state.get("is_final", False)

    # Increment iteration counter
    new_iteration = iteration + 1

    if is_final or new_iteration > 2:
        route = "end"
    else:
        route = "researcher"  # send back for refinement

    logger.info(f"[Router] consensus={consensus:.2f} iteration={new_iteration} → {route}")

    msg = make_message(
        agent="router",
        type="routing_decision",
        content={"route": route, "consensus": consensus, "iteration": new_iteration},
    )

    return {
        "route": route,
        "iteration": new_iteration,
        "agent_messages": [msg],
    }


def route_after_supervisor(state: AgentState) -> str:
    """Conditional edge function: after Supervisor, where to go?"""
    route = state.get("route", "researcher")
    if route == "planner":
        return "planner"
    elif route == "memory":
        return "generator"   # memory-only: skip retrieval, generate from memory_context
    elif route == "web":
        return "researcher"  # web_needed: researcher will trigger web_search tool
    else:
        return "researcher"  # simple_factual and default


def route_after_router(state: AgentState) -> str:
    """Conditional edge function: after Router, loop or end?"""
    route = state.get("route", "end")
    return route  # "researcher" or "end"


# ─────────────────────────────────────────────────────────────────────────────
# Agent communication log
# ─────────────────────────────────────────────────────────────────────────────

def save_agent_communication_log(state: AgentState, run_id: str) -> Path:
    """
    Serialise the full agent_messages list to JSONL after a run completes.
    Every message between every agent is recorded.

    Concept: Agent Communication (observable, inspectable log)
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOGS_DIR / f"agent_comms_{run_id}.jsonl"

    with open(log_path, "w") as f:
        for msg in state.get("agent_messages", []):
            f.write(json.dumps(msg) + "\n")

    logger.info(f"[Router] Agent communication log saved: {log_path}")
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Full LangGraph compilation
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(retrieval_ctx=None):
    """
    Compile the full 7-agent LangGraph with all nodes, edges, and checkpointer.

    Graph structure:
      START → supervisor → [planner → researcher] | [researcher] | [generator]
      researcher → generator → critic → reviewer → router → [researcher | END]

    Concepts: Agentic Workflow Routing (conditional edges), Workflow DAGs
    """
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver

    from P3.state import AgentState
    from P3.supervisor import supervisor_node
    from P3.planner import planner_node
    from P3.researcher import make_researcher_node
    from P3.generator import generator_node
    from P3.critic import critic_node
    from P3.reviewer import reviewer_node

    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", make_researcher_node(retrieval_ctx))
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("router", router_node)

    # ── Static edges ───────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", "reviewer")
    graph.add_edge("reviewer", "router")

    # ── Conditional edges ──────────────────────────────────────────────────
    # After supervisor: route to planner, researcher, or generator
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "planner": "planner",
            "researcher": "researcher",
            "generator": "generator",
        },
    )

    # After router: loop back to researcher OR end
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "researcher": "researcher",
            "end": END,
        },
    )

    # ── Checkpointer (session memory) ─────────────────────────────────────
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("[Graph] Compiled LangGraph with 7 agents and MemorySaver checkpointer")
    return compiled


def get_workflow_dag(compiled_graph) -> str:
    """
    Export the LangGraph as a Mermaid DAG string.
    Saved to /docs/workflow_dag.md for the README and Gradio UI (P9).

    Concept: Workflow DAGs (P5-T3)
    """
    try:
        mermaid = compiled_graph.get_graph().draw_mermaid()
        dag_path = Path("./graph_store/workflow_dag.md")
        dag_path.parent.mkdir(parents=True, exist_ok=True)
        dag_path.write_text(f"```mermaid\n{mermaid}\n```")
        logger.info(f"[Graph] Workflow DAG saved to {dag_path}")
        return mermaid
    except Exception as e:
        logger.warning(f"[Graph] Could not export DAG: {e}")
        return ""