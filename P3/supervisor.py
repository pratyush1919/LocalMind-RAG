"""
supervisor.py  —  P3-T2
Supervisor Agent: receives the raw user query, classifies it, and decides
which agent path to activate.

Concepts covered:
  - Supervisor Agent        (top-level controller of the agent network)
  - Agentic Workflow Routing (routes different query types to different paths)
  - Agent Communication     (writes routing decision to agent_messages)

Routing logic:
  simple_factual  → researcher directly (skip planner)
  multi_hop       → planner first (needs decomposition)
  comparative     → planner first (needs decomposition into sub-topics)
  conversational  → memory only (no new retrieval)
  web_needed      → browser agent (P11 — falls back to researcher until P11)
"""

import logging
import sys
from pathlib import Path
from app_context import get_agent_llm
sys.path.insert(0, str(Path(__file__).parent.parent))
from P3.state import AgentState, make_message
logger = logging.getLogger(__name__)

_SUPERVISOR_PROMPT = """\
You are a query classification and routing agent.

Classify the following user query into EXACTLY ONE category:

- simple_factual: A straightforward factual question with a single, definitive answer.
  Examples: "What is BERT?", "When was GPT-3 released?"

- multi_hop: Requires connecting information from multiple sources or reasoning steps.
  Examples: "How does BERT differ from GPT in terms of pre-training objectives and downstream tasks?"

- comparative: Asks to compare or contrast two or more things.
  Examples: "Compare RNNs and Transformers for sequence modelling."

- conversational: A follow-up or reference to a previous exchange.
  Examples: "Tell me more about that", "What did you mean earlier?", "Going back to..."

- web_needed: Requires real-time or very recent information not in static documents.
  Examples: "What are the latest AI papers from this week?", "Current stock price of..."

Respond with a JSON object:
{{
  "query_type": "<one of the five types above>",
  "route": "<planner|researcher|memory|web>",
  "rationale": "<one sentence explaining the classification>"
}}

Rules:
  simple_factual  → route: researcher
  multi_hop       → route: planner
  comparative     → route: planner
  conversational  → route: memory
  web_needed      → route: web

Query: {query}

Memory context (recent conversation): {memory_context}
"""


def supervisor_node(state: AgentState) -> dict:
    """
    LangGraph node function for the Supervisor Agent.

    Reads:  state.query, state.memory_context
    Writes: state.query_type, state.route, state.agent_messages
    """
    import json

    llm = get_agent_llm()

    prompt = _SUPERVISOR_PROMPT.format(
        query=state["query"],
        memory_context=state.get("memory_context", "None"),
    )

    # Default fallback
    query_type = "simple_factual"
    route = "researcher"
    rationale = "Default routing — LLM call failed"

    try:
        response = llm.invoke(prompt)
        print("\n=== SUPERVISOR PROMPT ===")
        print(prompt)
        print("=== SUPERVISOR RESPONSE ===")
        print(response.content)
        text = response.content.strip()

        # Strip markdown code fences if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        parsed = json.loads(text)
        query_type = parsed.get("query_type", "simple_factual")
        route = parsed.get("route", "researcher")
        rationale = parsed.get("rationale", "")

        # Safety: validate values
        valid_types = {"simple_factual", "multi_hop", "comparative", "conversational", "web_needed"}
        valid_routes = {"planner", "researcher", "memory", "web"}
        if query_type not in valid_types:
            query_type = "simple_factual"
        if route not in valid_routes:
            route = "researcher"

    except Exception as e:
        logger.warning(f"[Supervisor] LLM classification failed: {e} — using defaults")

    logger.info(f"[Supervisor] type={query_type} route={route}")

    msg = make_message(
        agent="supervisor",
        type="routing",
        content={
            "query_type": query_type,
            "route": route,
            "rationale": rationale,
        },
    )

    return {
        "query_type": query_type,
        "route": route,
        "agent_messages": [msg],
    }