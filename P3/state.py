"""
state.py
Defines the shared AgentState that flows through every node in the LangGraph.
Every agent reads from and writes to this state object.
The agent_messages field is the inter-agent communication bus —
each agent appends structured messages that downstream agents read.
"""

from __future__ import annotations
from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict
import operator


# Agent message structure
class AgentMessage(TypedDict):
    """
    Structured message passed between agents via agent_messages list.
    Every agent appends its own messages; downstream agents read the full log.
    """
    agent: str          # sender: "supervisor" | "planner" | "researcher" | ...
    type: str           # message type: "routing" | "plan" | "retrieval" | "critique" | ...
    content: Any        # payload — str, dict, list depending on type
    timestamp: str      # ISO timestamp for observability

# Agent state
class AgentState(TypedDict):
    """
    Shared state that persists across all agent nodes in the LangGraph.
    LangGraph passes this dict between nodes. Each node receives the current
    state, does its work, and returns a partial dict of fields to update.
    Fields with Annotated[List, operator.add] are APPEND-only —
    LangGraph merges them by concatenation, not replacement. This is critical
    for agent_messages so no agent's messages are lost.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    query: str                          # raw user query
    session_id: str                     # unique session identifier
    user_role: str                      # RBAC role: "admin" | "analyst" | "viewer"

    # ── Supervisor outputs ─────────────────────────────────────────────────
    query_type: str                     # "simple_factual" | "multi_hop" | "comparative"
                                        # | "conversational" | "web_needed"
    route: str                          # next step: "planner" | "researcher" | "memory" | "web"

    # ── Planner outputs ────────────────────────────────────────────────────
    sub_queries: List[str]              # decomposed sub-questions
    decomposition_rationale: str        # why this decomposition was chosen (ToT)

    # ── Retrieval outputs ──────────────────────────────────────────────────
    retrieved_chunks: List[Dict]        # raw retrieved nodes (serialised)
    reranked_chunks: List[Dict]         # after reranking pipeline
    retrieval_strategy: str             # strategy chosen by planner (keyword/semantic/hybrid)
    tool_calls_log: Annotated[          # every tool call: name + inputs + outputs
        List[Dict], operator.add
    ]

    # ── Generation outputs ─────────────────────────────────────────────────
    answer: str                         # current draft answer
    role_used: str                      # Generator role: "analyst"|"summarizer"|"researcher"|"explainer"
    citations: List[str]                # source citations extracted from answer

    # ── Critic outputs ─────────────────────────────────────────────────────
    critique: Dict                      # {faithfulness, completeness, reasoning_quality, issues}
    refinement_needed: bool             # True if any score < 7

    # ── Reviewer outputs ───────────────────────────────────────────────────
    review_notes: Dict                  # {source_quality, answer_completeness, ...}
    consensus_score: float              # weighted avg of Critic + Reviewer scores
    is_final: bool                      # True when consensus >= 7.5 or max iterations

    # ── Loop control ───────────────────────────────────────────────────────
    iteration: int                      # refinement loop count (max 2)

    # ── Memory context ─────────────────────────────────────────────────────
    memory_context: str                 # injected from long-term memory (P4)

    # ── Agent communication bus ────────────────────────────────────────────
    agent_messages: Annotated[          # append-only inter-agent message log
        List[AgentMessage], operator.add
    ]

    # ── Graph RAG flag ─────────────────────────────────────────────────────
    graph_rag_used: bool                # True if knowledge graph was queried (P5)
    graph_entities: List[str]           # entities found in knowledge graph

    # ── Web Research flag (P11) ────────────────────────────────────────────
    use_web_research: bool              # User capability permission gate: True allows web research
    web_research_used: bool             # True if web search was actually executed and yielded evidence


# Default state factory
def initial_state(
    query: str,
    session_id: str,
    user_role: str = "analyst",
    memory_context: str = "",
    use_web_research: bool = False,
) -> AgentState:
    """
    Create a fresh AgentState for a new query.
    All list/dict fields initialised to empty — agents append to them.
    """
    return AgentState(
        query=query,
        session_id=session_id,
        user_role=user_role,
        query_type="",
        route="supervisor",
        sub_queries=[],
        decomposition_rationale="",
        retrieved_chunks=[],
        reranked_chunks=[],
        retrieval_strategy="hybrid",
        tool_calls_log=[],
        answer="",
        role_used="",
        citations=[],
        critique={},
        refinement_needed=False,
        review_notes={},
        consensus_score=0.0,
        is_final=False,
        iteration=0,
        memory_context=memory_context,
        agent_messages=[],
        graph_rag_used=False,
        graph_entities=[],
        use_web_research=use_web_research,
        web_research_used=False,
    )


# Helper: create a typed AgentMessage
def make_message(agent: str, type: str, content: Any) -> AgentMessage:
    from datetime import datetime, timezone
    return AgentMessage(
        agent=agent,
        type=type,
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )