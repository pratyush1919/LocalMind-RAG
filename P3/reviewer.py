"""
reviewer.py  —  P3-T7
Reviewer Agent: independently validates the final answer quality and computes
a consensus score with the Critic to decide whether to finalise.

Concepts covered:
  - Reviewer Agent      (independent from Critic — no shared scoring bias)
  - Consensus Building  (Critic scores + Reviewer scores → weighted consensus)
  - Agent Collaboration (Critic and Reviewer collaborate via shared agent_messages)

Independence is key:
  The Critic sees the answer fresh (no prior context).
  The Reviewer reads the FULL agent_messages log — it can see what tools were used,
  what the Critic said, and whether the Researcher addressed the critique.
  This gives the Reviewer a holistic view the Critic lacks.

Consensus score formula:
  consensus = (
      0.3 * critic_faithfulness_norm +
      0.2 * critic_completeness_norm +
      0.1 * critic_reasoning_norm +
      0.2 * reviewer_source_quality_norm +
      0.1 * reviewer_answer_completeness_norm +
      0.1 * reviewer_citation_accuracy_norm
  ) * 10

  Where each _norm = score / 10 (normalise to [0,1])

  If consensus >= 7.5 → is_final = True
  If consensus < 7.5 AND iteration < 2 → send back to Researcher
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from P3.state import AgentState, make_message

logger = logging.getLogger(__name__)

_REVIEWER_PROMPT = """\
You are an independent quality reviewer for AI-generated research answers.
You have access to the full agent communication log to understand how the answer was produced.

Evaluate the final answer on four dimensions (score 0-10 each):
  - source_quality: Were relevant, high-quality sources used?
  - answer_completeness: Does the answer fully address the original question?
  - logical_consistency: Is the answer internally consistent and logically sound?
  - citation_accuracy: Do citations match the actual source content?

Also check:
  - Did the Research Agent use appropriate tools?
  - Did the Generator follow the assigned role?

Return a JSON object:
{{
  "source_quality": <0-10>,
  "answer_completeness": <0-10>,
  "logical_consistency": <0-10>,
  "citation_accuracy": <0-10>,
  "tool_selection_appropriate": <true|false>,
  "role_followed": <true|false>,
  "reviewer_summary": "<one sentence overall assessment>"
}}

---

Original question: {query}

Final answer:
{answer}

Agent communication log summary:
{agent_log_summary}

Critic scores (for reference):
{critic_scores}
"""


def _summarise_agent_log(agent_messages: List[Dict]) -> str:
    """Create a compact summary of agent_messages for the Reviewer prompt."""
    lines = []
    for msg in agent_messages:
        agent = msg.get("agent", "?")
        msg_type = msg.get("type", "?")
        content = msg.get("content", {})

        if msg_type == "routing":
            lines.append(f"Supervisor → route={content.get('route')} type={content.get('query_type')}")
        elif msg_type == "plan":
            sqs = content.get("sub_queries", [])
            lines.append(f"Planner → {len(sqs)} sub-queries: {sqs}")
        elif msg_type == "retrieval":
            lines.append(
                f"Researcher → tools={content.get('tools_used')} chunks={content.get('chunk_count')}"
            )
        elif msg_type == "generation":
            lines.append(
                f"Generator → role={content.get('role_used')} citations={content.get('citation_count')}"
            )
        elif msg_type == "critique":
            scores = content.get("scores", {})
            lines.append(
                f"Critic → F={scores.get('faithfulness')} C={scores.get('completeness')} "
                f"R={scores.get('reasoning_quality')} issues={content.get('issues')}"
            )

    return "\n".join(lines) if lines else "No agent messages available."


def _compute_consensus(critique: Dict, review: Dict) -> float:
    """
    Compute weighted consensus score from Critic + Reviewer dimensions.
    Returns a score in [0, 10].

    Concept: Consensus Building
    """
    weights = {
        "faithfulness": 0.30,
        "completeness": 0.20,
        "reasoning_quality": 0.10,
        "source_quality": 0.20,
        "answer_completeness": 0.10,
        "citation_accuracy": 0.10,
    }

    total = 0.0
    total += weights["faithfulness"] * critique.get("faithfulness", 8)
    total += weights["completeness"] * critique.get("completeness", 8)
    total += weights["reasoning_quality"] * critique.get("reasoning_quality", 8)
    total += weights["source_quality"] * review.get("source_quality", 8)
    total += weights["answer_completeness"] * review.get("answer_completeness", 8)
    total += weights["citation_accuracy"] * review.get("citation_accuracy", 8)

    return round(total, 2)


def reviewer_node(state: AgentState) -> dict:
    """
    LangGraph node for the Reviewer Agent.

    Reads:  state.answer, state.query, state.critique, state.agent_messages
    Writes: state.review_notes, state.consensus_score, state.is_final,
            state.agent_messages
    """
    from app_context import get_agent_llm

    llm = get_agent_llm(temperature=0.1)

    answer = state.get("answer", "")
    query = state["query"]
    critique = state.get("critique", {})
    agent_messages = state.get("agent_messages", [])
    iteration = state.get("iteration", 0)

    agent_log_summary = _summarise_agent_log(agent_messages)
    critic_scores = {
        k: critique.get(k) for k in ("faithfulness", "completeness", "reasoning_quality")
    }

    # Default review
    review: Dict = {
        "source_quality": 8,
        "answer_completeness": 8,
        "logical_consistency": 8,
        "citation_accuracy": 7,
        "tool_selection_appropriate": True,
        "role_followed": True,
        "reviewer_summary": "Default review — LLM evaluation unavailable",
    }

    try:
        prompt = _REVIEWER_PROMPT.format(
            query=query,
            answer=answer,
            agent_log_summary=agent_log_summary,
            critic_scores=json.dumps(critic_scores, indent=2),
        )
        response = llm.invoke(prompt)
        text = response.content.strip()
        print("\n=== REVIEWER RAW RESPONSE ===\n")
        print(text)
        print("\n=============================\n")
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        import ast
        try:
            parsed = json.loads(text)
        except:
            parsed = ast.literal_eval(text)

        for key in ("source_quality", "answer_completeness", "logical_consistency", "citation_accuracy"):
            parsed[key] = max(0, min(10, int(parsed.get(key, 8))))

        review = parsed

    except Exception as e:
        logger.warning(f"[Reviewer] LLM evaluation failed: {e} — using defaults")

    # Consensus score
    consensus_score = _compute_consensus(critique, review)
    # Apply penalties for procedural failures

    if not review.get("role_followed", True):
        consensus_score *= 0.80

    if not review.get("tool_selection_appropriate", True):
        consensus_score *= 0.90

    consensus_score = round(consensus_score, 2)
    # Finalisation decision
    # is_final if: consensus >= 7.5 OR max iterations reached
    hard_fail = (
        not review.get("role_followed", True)
        or not review.get("tool_selection_appropriate", True)
    )
    logger.info(
    f"[Reviewer] role_followed={review.get('role_followed')} "
    f"tool_selection_appropriate={review.get('tool_selection_appropriate')} "
    f"consensus={consensus_score}"
    )
    is_final = (
        consensus_score >= 7.5
        and not hard_fail
    ) or iteration >= 2

    logger.info(
        f"[Reviewer] consensus={consensus_score:.2f} | is_final={is_final} | "
        f"iteration={iteration} | {review.get('reviewer_summary', '')}"
    )

    msg = make_message(
        agent="reviewer",
        type="review",
        content={
            "review_scores": {
                k: review.get(k) for k in
                ("source_quality", "answer_completeness", "logical_consistency", "citation_accuracy")
            },
            "consensus_score": consensus_score,
            "is_final": is_final,
            "tool_selection_appropriate": review.get("tool_selection_appropriate"),
            "role_followed": review.get("role_followed"),
            "reviewer_summary": review.get("reviewer_summary"),
        },
    )

    return {
        "review_notes": review,
        "consensus_score": consensus_score,
        "is_final": is_final,
        "agent_messages": [msg],
    }