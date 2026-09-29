"""
critic.py  —  P3-T6
Critic Agent: reviews the generated answer for faithfulness and completeness,
then triggers a refinement loop if quality is insufficient.

Concepts covered:
  - Self-Reflection       (agent reflects on its own system's output)
  - Reflection Agents     (dedicated agent whose sole job is critique)
  - Critique-and-Refine   (critique written to state → Researcher reads and refines)
  - Answer Verification   (checks every claim against retrieved context)
  - Iterative Retrieval   (low scores trigger another retrieval round)

Scoring dimensions (each 0-10):
  faithfulness       — every claim supported by retrieved context?
  completeness       — does the answer address all parts of the query?
  reasoning_quality  — is the reasoning chain logical and coherent?

Critique-and-Refine loop (orchestrated by Router in P3-T8):
  Critic scores answer → if any score < 7 → writes specific issues to state
  → Router sends back to Researcher → Researcher reads issues and refines retrieval
  → Generator produces new answer → Critic reviews again
  Maximum 2 refinement iterations (enforced by Router)
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from P3.state import AgentState, make_message

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = """\
You are a critical evaluator of AI-generated answers.

IMPORTANT:
Return ONLY valid JSON.
Do not write explanations.
Do not use markdown.
Do not wrap in code fences.

For every issue:
quote the exact answer text that caused the issue.
If no supporting quote exists,
do not report the issue.

Your task: evaluate the answer below against the provided source context.

Score each dimension from 0 to 10:
  - faithfulness: Are all factual claims in the answer directly supported by the context?
    (10 = fully grounded, 0 = mostly hallucinated)
  - completeness: Does the answer address ALL aspects of the original question?
    (10 = fully complete, 0 = major gaps)
  - reasoning_quality: Is the reasoning logical, coherent, and well-structured?
    (10 = excellent reasoning, 0 = incoherent)

Also list specific issues found (be precise — these will be used to refine retrieval).

Return a JSON object:
{{
  "faithfulness": <0-10>,
  "completeness": <0-10>,
  "reasoning_quality": <0-10>,
  "issues": ["issue 1", "issue 2", ...],
  "summary": "<one sentence overall assessment>"
}}

---

Original question: {query}

Retrieved context (ground truth):
{context}

Generated answer:
{answer}
"""


def _format_context_for_critic(chunks: List[Dict]) -> str:
    if not chunks:
        return "No context available."
    parts = []
    for i, c in enumerate(chunks[:5], 1):
        parts.append(f"[Source {i}]: {c.get('text', '')}")
    return "\n\n".join(parts)


def critic_node(state: AgentState) -> dict:
    """
    LangGraph node for the Critic Agent.

    Reads:  state.answer, state.query, state.reranked_chunks
    Writes: state.critique, state.refinement_needed, state.agent_messages
    """
    from app_context import get_agent_llm

    llm = get_agent_llm(temperature=0.1)

    answer = state.get("answer", "")
    citations = state.get("citations", [])
    query = state["query"]
    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    context = _format_context_for_critic(chunks)

    # Default critique in case LLM fails
    critique: Dict = {
        "faithfulness": 8,
        "completeness": 8,
        "reasoning_quality": 8,
        "issues": [],
        "summary": "Default critique — LLM evaluation unavailable",
    }

    if not answer.strip():
        critique = {
            "faithfulness": 0,
            "completeness": 0,
            "reasoning_quality": 0,
            "issues": ["No answer was generated"],
            "summary": "Empty answer",
        }
    else:
        try:
            prompt = _CRITIC_PROMPT.format(
                query=query,
                context=context,
                answer=answer,
            )
            response = llm.invoke(prompt)
            text = response.content.strip()
            print("\n=== CRITIC RAW RESPONSE ===\n")
            print(text)
            print("\n===========================\n")
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            match = re.search(
                r"\{[\s\S]*?\}",
                text,
                re.DOTALL,
            )

            if not match:
                raise ValueError("No JSON found")

            parsed = json.loads(match.group())
            if not isinstance(parsed.get("issues"), list):
                parsed["issues"] = []

            # Validate and clamp scores
            for key in ("faithfulness", "completeness", "reasoning_quality"):
                parsed[key] = max(0, min(10, int(parsed.get(key, 8))))

            critique = parsed
            if not citations:
                critique["faithfulness"] = min(
                    critique["faithfulness"],
                    5,
                )

                critique.setdefault(
                    "issues",
                    []
                ).append(
                    "Answer contains no citations."
                )

        except Exception as e:
            logger.warning(f"[Critic] LLM evaluation failed: {e} — using default scores")

    # Refinement decision: any score < 7 triggers another loop
    min_score = min(
        critique["faithfulness"],
        critique["completeness"],
        critique["reasoning_quality"],
    )
    
    if not citations:
        min_score = min(min_score, 5)
    refinement_needed = min_score < 7

    logger.info(
        f"[Critic] F={critique['faithfulness']} C={critique['completeness']} "
        f"R={critique['reasoning_quality']} | Refine={refinement_needed} | "
        f"Issues: {critique.get('issues', [])}"
    )

    msg = make_message(
        agent="critic",
        type="critique",
        content={
            "scores": {
                "faithfulness": critique["faithfulness"],
                "completeness": critique["completeness"],
                "reasoning_quality": critique["reasoning_quality"],
            },
            "issues": critique.get("issues", []),
            "summary": critique.get("summary", ""),
            "refinement_needed": refinement_needed,
        },
    )

    return {
        "critique": critique,
        "refinement_needed": refinement_needed,
        "agent_messages": [msg],
    }