"""
researcher.py  —  P3-T4 + P5 Graph RAG integration

Research Agent:
    Retrieves information for each sub-query by selecting the
    right tool from its toolbox.

P3 concepts:
  - Tool-Augmented Retrieval
  - Research Agent
  - Agentic Retrieval
  - Iterative Retrieval
  - Agent Communication

P5 addition:
  - Graph RAG
  - Knowledge Graph Retrieval
  - Multi-Hop Retrieval

Toolbox:
  tool_0: graph_rag       — P5 knowledge graph traversal
  tool_1: document_search — P2 hybrid retrieval pipeline
  tool_2: web_search      — Browser Agent (P11) — stubbed
  tool_3: fetch_url       — MCP fetch server (P10) — stubbed

Routing:

  simple_factual
        ↓
  document_search

  comparative / multi_hop
        ↓
  graph_rag
        ↓
  document_search

  web_needed + low coverage
        ↓
  web_search

  URL + low coverage
        ↓
  fetch_url

The Research Agent evaluates coverage after each retrieval
attempt and decides whether another tool should be tried.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from P3.state import AgentState, make_message
from guardrails.rbac import check_tool_access


logger = logging.getLogger(__name__)


# Coverage threshold.
# If retrieval coverage is below this value, the researcher
# may attempt another retrieval tool.
_COVERAGE_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────────────────────────────────────


def _tool_graph_rag(query: str) -> List[Dict]:
    """
    Tool 0: Knowledge Graph Retrieval / Graph RAG.

    Used primarily for comparative and multi-hop queries.

    Flow:

        query
          ↓
        entity matching
          ↓
        graph traversal
          ↓
        1-hop / 2-hop facts
          ↓
        pseudo-chunk for Generator

    The result is represented as a chunk-like dictionary so the
    existing P3 state structure does not need to change.
    """

    try:
        from graph_store.graph_retriever import graph_rag_retrieve

        result = graph_rag_retrieve(
            query,
            max_hops=2,
            max_facts=15,
        )

        if not result.get("graph_used"):
            logger.info(
                f"[Researcher] Graph RAG found no usable facts "
                f"for: '{query[:60]}'"
            )
            return []

        graph_chunk = {
            "node_id": "graph_rag_facts",
            "text": result.get("formatted_text", ""),
            "score": 0.95,
            "metadata": {
                "filename": "knowledge_graph",
                "page_number": 0,
                "doc_category": "graph",
                "chunk_strategy": "graph_rag",
            },
            "source": "graph_rag",
            "entities_found": result.get(
                "entities_found",
                [],
            ),
        }

        logger.info(
            f"[Researcher] Graph RAG returned "
            f"{len(result.get('facts', []))} facts "
            f"from entities={result.get('entities_found', [])}"
        )

        return [graph_chunk]

    except Exception as e:
        logger.debug(
            f"[Researcher] Graph RAG tool failed: {e}"
        )
        return []


def _tool_document_search(
    query: str,
    retrieval_ctx,
) -> List[Dict]:
    """
    Tool 1: Full Phase 2 hybrid retrieval pipeline.

    Returns serialized chunk dictionaries for agent state.
    """

    from P2.retrieval_pipeline import retrieve

    result = retrieve(
        query,
        retrieval_ctx,
    )

    nodes = result.get(
        "nodes",
        [],
    )

    chunks = []

    for n in nodes:

        chunks.append(
            {
                "node_id": n.node.node_id,
                "text": n.node.text,
                "score": float(
                    n.score or 0.0
                ),
                "metadata": n.node.metadata,
                "source": "document_search",
            }
        )

    return chunks


def _tool_web_search(
    query: str,
) -> List[Dict]:
    """
    Tool 2: Web search via Browser Agent (Phase 11).
    """
    logger.info(
        f"[Researcher] web_search called for: "
        f"'{query[:60]}'"
    )
    try:
        from agents.browser_agent import research
        results = research(query)
        logger.info(
            f"[Researcher] web_search returned {len(results)} chunks"
        )
        return results
    except Exception as e:
        logger.warning(
            f"[Researcher] web_search execution failed: {e}"
        )
        return []



def _tool_fetch_url(
    url: str,
) -> List[Dict]:
    """
    Tool 3: Fetch a specific URL via MCP fetch server.

    Stubbed until P10.
    """

    logger.info(
        f"[Researcher] fetch_url stub called for: {url}"
    )

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Coverage evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_coverage(
    chunks: List[Dict],
    query: str,
) -> float:
    """
    Estimate how well retrieved chunks cover the query.

    Heuristic:

      - 0 chunks → 0.0
      - Average score of top chunks
      - Small bonus for multiple source documents

    Returns:
        float in [0, 1]
    """

    if not chunks:
        return 0.0

    scores = [
        max(
            0.0,
            min(
                c.get("score", 0.0),
                1.0,
            ),
        )
        for c in chunks[:5]
    ]

    avg_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    # Diversity bonus.
    sources = {
        c.get(
            "metadata",
            {},
        ).get(
            "filename",
            "?",
        )
        for c in chunks
    }

    diversity_bonus = min(
        len(sources) * 0.05,
        0.15,
    )

    coverage = min(
        avg_score + diversity_bonus,
        1.0,
    )

    print("\n=== COVERAGE DEBUG ===")
    print("scores:", scores)
    print("avg_score:", avg_score)
    print("sources:", sources)
    print("bonus:", diversity_bonus)
    print("coverage:", coverage)
    print("======================\n")

    return coverage


# ─────────────────────────────────────────────────────────────────────────────
# Tool selection logic
# ─────────────────────────────────────────────────────────────────────────────


def _select_tool(
    query: str,
    query_type: str,
    coverage: float,
    tools_tried: List[str],
    use_web_research: bool = False,
) -> str:
    """
    Decide which retrieval tool to use next.

    Graph RAG has priority for comparative / multi-hop queries.
    If Graph RAG returns strong evidence, it is sufficient and
    we avoid polluting the evidence set with unrelated documents.
    """

    # ------------------------------------------------------------
    # P5: Graph RAG
    # ------------------------------------------------------------

    if (
        "graph_rag" not in tools_tried
        and query_type in (
            "comparative",
            "multi_hop",
        )
    ):
        return "graph_rag"

    # ------------------------------------------------------------
    # P5: Strong Graph RAG result
    # ------------------------------------------------------------

    if (
        query_type in (
            "comparative",
            "multi_hop",
        )
        and "graph_rag" in tools_tried
        and coverage >= 0.80
    ):
        return "done"

    # ------------------------------------------------------------
    # P3/P2: Document search
    # ------------------------------------------------------------

    if "document_search" not in tools_tried:
        return "document_search"

    # ------------------------------------------------------------
    # P3/P11: Additional tools when coverage is low
    # ------------------------------------------------------------

    if coverage < _COVERAGE_THRESHOLD:

        # P11: Hard capability boundary. Web search is ONLY accessible if use_web_research is True.
        if (
            use_web_research
            and (query_type == "web_needed" or coverage < 0.60)
            and "web_search" not in tools_tried
        ):
            return "web_search"

        if (
            "http" in query.lower()
            and "fetch_url" not in tools_tried
        ):
            return "fetch_url"

    return "done"
# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────


def researcher_node(
    state: AgentState,
    retrieval_ctx=None,
) -> dict:
    """
    LangGraph node for the Research Agent.

    For each sub-query:

        1. Select tool
        2. Execute tool
        3. Evaluate coverage
        4. Repeat if needed

    Reads:
        state.sub_queries
        state.query_type
        state.critique

    Writes:
        state.retrieved_chunks
        state.reranked_chunks
        state.tool_calls_log
        state.graph_rag_used
        state.graph_entities
        state.agent_messages
    """

    query = state["query"]

    query_type = state.get(
        "query_type",
        "simple_factual",
    )

    sub_queries = (
        state.get("sub_queries")
        or [query]
    )

    iteration = state.get(
        "iteration",
        0,
    )

    # ------------------------------------------------------------
    # Refinement iteration
    # ------------------------------------------------------------

    if (
        iteration > 0
        and state.get("critique", {}).get("issues")
    ):

        issues = state["critique"]["issues"]

        refinement_hints = (
            f" [Also address: "
            f"{'; '.join(issues[:2])}]"
        )

        sub_queries = [
            q + refinement_hints
            for q in sub_queries
        ]

        logger.info(
            f"[Researcher] Refinement iteration "
            f"{iteration} — enriched queries "
            f"with critic feedback"
        )

    # ------------------------------------------------------------
    # Retrieval state
    # ------------------------------------------------------------

    all_chunks: List[Dict] = []

    tool_calls_log: List[Dict] = []

    all_tools_used = set()

    # IMPORTANT:
    # tools_tried belongs to EACH sub-query.
    #
    # This means:
    #
    # sub-query 1:
    #     graph_rag → document_search
    #
    # sub-query 2:
    #     graph_rag → document_search
    #
    # independently.
    #
    for sub_query_index, sub_query in enumerate(
        sub_queries
    ):

        tools_tried: List[str] = []

        logger.info(
            f"[Researcher] Processing sub-query: "
            f"'{sub_query[:80]}'"
        )

        coverage = 0.0

        sub_chunks: List[Dict] = []

        # --------------------------------------------------------
        # Tool selection loop
        # --------------------------------------------------------

        use_web_research = bool(state.get("use_web_research", False))

        for attempt in range(2):

            tool_name = _select_tool(
                sub_query,
                query_type,
                coverage,
                tools_tried,
                use_web_research=use_web_research,
            )
            
            if tool_name == "done":
                break

            if tool_name == "web_search" and not use_web_research:
                logger.warning("[Researcher] web_search blocked because use_web_research is False")
                break
            
            # --------------------------------------------------------
            # P6-T2: RBAC enforcement
            # --------------------------------------------------------
            #
            # IMPORTANT:
            # Check permission BEFORE:
            #   1. marking the tool as tried
            #   2. adding it to tools_used
            #   3. executing the tool
            #
            # This ensures a denied tool is never treated as
            # successfully attempted/executed.
            # --------------------------------------------------------
            
            rbac_result = check_tool_access(
                tool_name=tool_name,
                user_role=state.get(
                    "user_role",
                    "analyst",
                ),
                session_id=state.get(
                    "session_id",
                    "",
                ),
            )
            
            if not rbac_result.allowed:
            
                logger.warning(
                    f"[RBAC] Tool access denied: "
                    f"tool={tool_name} "
                    f"role={rbac_result.role} "
                    f"reason={rbac_result.reason}"
                )
            
                tool_calls_log.append(
                    {
                        "tool": tool_name,
                        "query": sub_query,
                        "result_count": 0,
                        "sub_query_index": sub_query_index,
                        "attempt": attempt,
                        "allowed": False,
                        "rbac_reason": rbac_result.reason,
                    }
                )
            
                # Do not execute the denied tool.
                #
                # We also stop this sub-query rather than allowing
                # the researcher to silently bypass RBAC by selecting
                # another privileged tool.
                break
            
            # --------------------------------------------------------
            # RBAC allowed → tool may execute
            # --------------------------------------------------------
            
            tools_tried.append(
                tool_name
            )
            
            all_tools_used.add(
                tool_name
            )
            
            logger.info(
                f"[Researcher] Using tool: "
                f"{tool_name}"
            )

            # ----------------------------------------------------
            # Execute tool
            # ----------------------------------------------------

            tool_result: List[Dict] = []

            try:

                if tool_name == "graph_rag":

                    tool_result = _tool_graph_rag(
                        sub_query
                    )

                elif tool_name == "document_search":

                    if retrieval_ctx is not None:

                        tool_result = _tool_document_search(
                            sub_query,
                            retrieval_ctx,
                        )

                    else:

                        logger.warning(
                            "[Researcher] retrieval_ctx "
                            "not provided — skipping "
                            "document_search"
                        )

                elif tool_name == "web_search":

                    tool_result = _tool_web_search(
                        sub_query
                    )

                elif tool_name == "fetch_url":

                    tool_result = _tool_fetch_url(
                        sub_query
                    )

            except Exception as e:

                logger.error(
                    f"[Researcher] Tool "
                    f"{tool_name} failed: {e}"
                )

            # ----------------------------------------------------
            # Log tool call
            # ----------------------------------------------------

            tool_calls_log.append(
                {
                    "tool": tool_name,
                    "query": sub_query,
                    "result_count": len(
                        tool_result
                    ),
                    "sub_query_index": sub_query_index,
                    "attempt": attempt,
                }
            )

            # ----------------------------------------------------
            # Add results
            # ----------------------------------------------------

            sub_chunks.extend(
                tool_result
            )

            # ----------------------------------------------------
            # Recalculate coverage
            # ----------------------------------------------------

            coverage = _evaluate_coverage(
                sub_chunks,
                sub_query,
            )

            logger.info(
                f"[Researcher] Coverage after "
                f"{tool_name}: {coverage:.3f}"
            )

        # --------------------------------------------------------
        # Add this sub-query's results
        # --------------------------------------------------------

        all_chunks.extend(
            sub_chunks
        )

    # ============================================================
    # Deduplicate chunks
    # ============================================================

    seen_ids = set()

    deduped_chunks = []

    for chunk in all_chunks:

        cid = chunk.get(
            "node_id",
            chunk.get(
                "text",
                "",
            )[:50],
        )

        if cid not in seen_ids:

            seen_ids.add(cid)

            deduped_chunks.append(
                chunk
            )

    # ============================================================
    # Sort by retrieval score
    # ============================================================

    deduped_chunks.sort(
        key=lambda c: c.get(
            "score",
            0.0,
        ),
        reverse=True,
    )

    # ============================================================
    # P5 Graph RAG observability
    # ============================================================

    graph_rag_used = any(
        tc.get("tool") == "graph_rag"
        and tc.get("result_count", 0) > 0
        for tc in tool_calls_log
    )

    graph_entities = []

    for chunk in deduped_chunks:

        if chunk.get("source") == "graph_rag":

            graph_entities.extend(
                chunk.get(
                    "entities_found",
                    [],
                )
            )

    graph_entities = list(
        set(graph_entities)
    )

    # ============================================================
    # Logging
    # ============================================================

    web_research_used = bool(
        "web_search" in all_tools_used
        and any(c.get("source") == "web_search" for c in deduped_chunks)
    )

    logger.info(
        f"[Researcher] "
        f"{len(sub_queries)} sub-queries → "
        f"{len(deduped_chunks)} unique chunks "
        f"| tools used: {list(all_tools_used)} "
        f"| graph_rag={graph_rag_used} "
        f"| web_research_used={web_research_used}"
    )

    # ============================================================
    # Agent message
    # ============================================================

    msg = make_message(
        agent="researcher",
        type="retrieval",
        content={
            "sub_queries": sub_queries,
            "tools_used": list(
                all_tools_used
            ),
            "chunk_count": len(
                deduped_chunks
            ),
            "tool_calls": tool_calls_log,
            "graph_rag_used": graph_rag_used,
            "graph_entities": graph_entities,
            "web_research_used": web_research_used,
        },
    )

    # ============================================================
    # Return state updates
    # ============================================================

    return {
        "retrieved_chunks": deduped_chunks,

        "reranked_chunks": deduped_chunks[:5],

        "tool_calls_log": tool_calls_log,

        "graph_rag_used": graph_rag_used,

        "graph_entities": graph_entities,

        "web_research_used": web_research_used,

        "agent_messages": [msg],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def make_researcher_node(
    retrieval_ctx,
):
    """
    Returns a LangGraph-compatible researcher node
    with retrieval_ctx pre-bound.

    Usage:

        node_fn = make_researcher_node(ctx)

        graph.add_node(
            "researcher",
            node_fn,
        )
    """

    def _node(
        state: AgentState,
    ) -> dict:

        return researcher_node(
            state,
            retrieval_ctx=retrieval_ctx,
        )

    return _node