"""
planner.py  —  P3-T3
Planner Agent: decomposes complex queries into sub-questions using
Tree of Thoughts (ToT) reasoning.

Concepts covered:
  - Query Planning          (breaking complex questions into solvable parts)
  - Task Decomposition      (structured sub-query generation)
  - Tree of Thoughts        (generate multiple decomposition strategies, score each, pick best)
  - Retrieval Planning      (sub-query metadata guides retrieval strategy per sub-question)
  - Agent Communication     (writes chosen plan + rationale to agent_messages)

Tree of Thoughts applied to query decomposition:
  Standard approach: generate ONE decomposition
  ToT approach:
    1. Generate 3 DIFFERENT decomposition strategies
    2. Score each on: coverage (does it cover all aspects?) +
                      independence (can sub-queries be answered separately?)
    3. Select the highest-scoring strategy
    4. Write the selected decomposition tree to agent_messages

  Why this matters for interviews:
    ToT demonstrates that you understand LLMs can reason better when they
    explore multiple solution paths before committing. The decomposition quality
    directly affects retrieval quality downstream.
"""

import json
import logging
import sys
from pathlib import Path
from app_context import get_agent_llm
sys.path.insert(0, str(Path(__file__).parent.parent))
from P3.state import AgentState, make_message
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
You are a research planning agent. Break the following complex question into
{n} independent sub-questions that together fully answer the original.

Rules:
- Each sub-question must be answerable independently
- Together they must cover ALL aspects of the original question
- Avoid overlapping sub-questions
- Number of sub-questions: 2-4 (choose the minimum needed)

IMPORTANT:
- Do NOT repeat the original question.
- You MUST generate at least 2 sub-questions.
- The output should be a true decomposition, not a rephrasing.

Return a JSON object:
{{
  "sub_queries": ["sub-question 1", "sub-question 2", ...],
  "coverage_score": <0-10, how well sub-queries cover the original>,
  "independence_score": <0-10, how independent the sub-queries are>
}}

Original question: {query}
"""

_TOT_SCORE_PROMPT = """\
You are evaluating three different ways to decompose a research question.
Score each decomposition on a scale of 1-10 for:
  - Coverage: Do the sub-questions together fully answer the original?
  - Independence: Can each sub-question be answered without the others?

Original question: {query}

Decomposition A:
{decomp_a}

Decomposition B:
{decomp_b}

Decomposition C:
{decomp_c}

Return JSON:
{{
  "scores": {{
    "A": {{"coverage": <1-10>, "independence": <1-10>}},
    "B": {{"coverage": <1-10>, "independence": <1-10>}},
    "C": {{"coverage": <1-10>, "independence": <1-10>}}
  }},
  "best": "<A|B|C>",
  "rationale": "<one sentence>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def _generate_one_decomposition(query: str, llm, seed_hint: str = "") -> dict:
    """Generate a single decomposition strategy."""
    prompt = _DECOMPOSE_PROMPT.format(query=query, n="2-4") + (
        f"\n\nHint: {seed_hint}" if seed_hint else ""
    )
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"[Planner] Decomposition failed: {e}")
        return {
            "sub_queries": [query],
            "coverage_score": 5,
            "independence_score": 5,
        }

def _normalize_subqueries(sub_queries):
    """
    Ensure planner always returns list[str].
    """
    normalized = []

    for item in sub_queries:
        if isinstance(item, str):
            normalized.append(item)

        elif isinstance(item, dict):
            normalized.append(
                item.get("question")
                or item.get("query")
                or str(item)
            )

        else:
            normalized.append(str(item))

    return normalized
def _is_valid_decomposition(
    original_query: str,
    sub_queries: list[str],
) -> bool:
    """
    A valid decomposition must:
    - contain at least 2 sub-queries
    - not simply repeat the original query
    """

    if len(sub_queries) < 2:
        return False

    normalized_original = original_query.strip().lower()

    if (
        len(sub_queries) == 1
        and sub_queries[0].strip().lower() == normalized_original
    ):
        return False

    return True
def tree_of_thoughts_decompose(query: str, llm) -> tuple[list, dict]:
    """
    Tree of Thoughts: generate 3 decomposition strategies, score each, return the best.

    Returns:
        (sub_queries: list, metadata: dict with rationale and scores)

    Concept: Tree of Thoughts
    """
    # Generate 3 diverse decompositions with different seed hints
    hints = [
        "Focus on chronological or causal structure.",
        "Focus on entity-centric decomposition (who/what/how).",
        "Focus on topic-based decomposition (separate concerns).",
    ]

    decompositions = {}
    for key, hint in zip(["A", "B", "C"], hints):

        result = _generate_one_decomposition(
            query,
            llm,
            seed_hint=hint,
        )

        result["sub_queries"] = _normalize_subqueries(
            result.get("sub_queries", [])
        )

        if not _is_valid_decomposition(
            query,
            result["sub_queries"],
        ):
            logger.warning(
                f"[Planner-ToT] Invalid decomposition "
                f"for strategy {key}"
            )

            result["coverage_score"] = 0
            result["independence_score"] = 0

        decompositions[key] = result

        logger.debug(
            f"[Planner-ToT] Strategy {key}: "
            f"{result['sub_queries']}"
        )

    valid_strategies = {
            key
            for key, value in decompositions.items()
            if _is_valid_decomposition(
                query,
                value["sub_queries"],
            )
    }

    print("\n" + "=" * 80)
    print("TREE OF THOUGHTS DECOMPOSITIONS")
    print("QUERY:", query)

    print("\nA:")
    print(json.dumps(decompositions["A"], indent=2))

    print("\nB:")
    print(json.dumps(decompositions["B"], indent=2))

    print("\nC:")
    print(json.dumps(decompositions["C"], indent=2))


    # Score all three and pick the best
    try:
        score_prompt = _TOT_SCORE_PROMPT.format(
            query=query,
            decomp_a=json.dumps(decompositions["A"]["sub_queries"], indent=2),
            decomp_b=json.dumps(decompositions["B"]["sub_queries"], indent=2),
            decomp_c=json.dumps(decompositions["C"]["sub_queries"], indent=2),
        )
        response = llm.invoke(score_prompt)
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        scoring = json.loads(text)

        print("\nSCORING RESULT")
        print(json.dumps(scoring, indent=2))
        scores = scoring.get("scores", {})
        candidate_scores = {}

        for strategy in valid_strategies:
            strategy_score = scores.get(strategy, {})

            total_score = (
                strategy_score.get("coverage", 0) * 2
                + strategy_score.get("independence", 0)
            )

            candidate_scores[strategy] = total_score

        print("\nCANDIDATE SCORES:")
        print(candidate_scores)
        if candidate_scores:
            best_key = max(
                candidate_scores,
                key=candidate_scores.get,
            )
        else:
            best_key = "A"

        rationale = scoring.get("rationale", "Best coverage and independence")
        scores = scoring.get("scores", {})


    except Exception as e:
        logger.warning(f"[Planner-ToT] Scoring failed: {e} — picking A by default")
        best_key = "A"
        rationale = "Fallback: scoring LLM call failed"
        scores = {}

    print("\nSELECTED BEST KEY:", best_key)
    best_decomp = decompositions.get(best_key, decompositions["A"])
    print("\nBEST DECOMP:")
    print(json.dumps(best_decomp, indent=2))
    sub_queries = _normalize_subqueries(
        best_decomp.get("sub_queries", [query])
    )

    metadata = {
        "chosen_strategy": best_key,
        "rationale": rationale,
        "all_strategies": {
            k: v.get("sub_queries", []) for k, v in decompositions.items()
        },
        "scores": scores,
    }

    logger.info(
        f"[Planner-ToT] Chose strategy {best_key}: {len(sub_queries)} sub-queries | {rationale}"
    )
    return sub_queries, metadata


def simple_decompose(query: str, llm) -> list:
    result = _generate_one_decomposition(query, llm)

    return _normalize_subqueries(
        result.get("sub_queries", [query])
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    LangGraph node for the Planner Agent.

    Uses Tree of Thoughts for multi_hop and comparative queries.
    Uses simple decomposition for other types (faster).

    Reads:  state.query, state.query_type
    Writes: state.sub_queries, state.decomposition_rationale, state.agent_messages
    """

    llm = get_agent_llm(temperature=0.4)
    query = state["query"]
    query_type = state.get("query_type", "simple_factual")

    # Tree of Thoughts for complex queries
    use_tot = query_type in ("multi_hop", "comparative")

    if use_tot:
        logger.info("[Planner] Using Tree of Thoughts decomposition")
        sub_queries, tot_metadata = tree_of_thoughts_decompose(query, llm)
        rationale = (
            f"ToT strategy {tot_metadata['chosen_strategy']} selected: "
            f"{tot_metadata['rationale']}"
        )
        msg_content = {
            "sub_queries": sub_queries,
            "tot_metadata": tot_metadata,
            "method": "tree_of_thoughts",
        }
    else:
        logger.info("[Planner] Using simple decomposition")
        sub_queries = simple_decompose(query, llm)
        rationale = "Simple decomposition (single-pass)"
        msg_content = {
            "sub_queries": sub_queries,
            "method": "simple",
        }

    # Ensure at least the original query is present
    if not sub_queries:
        sub_queries = [query]

    assert all(
        isinstance(q, str)
        for q in sub_queries
    ), f"Non-string sub-queries detected: {sub_queries}"
    msg = make_message(
        agent="planner",
        type="plan",
        content=msg_content,
    )

    logger.info(f"[Planner] {len(sub_queries)} sub-queries: {sub_queries}")

    return {
        "sub_queries": sub_queries,
        "decomposition_rationale": rationale,
        "agent_messages": [msg],
    }