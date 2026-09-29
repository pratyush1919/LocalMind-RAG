
"""
agent_pipeline.py  —  Phase 6 integration

Single entry point for running a query through the full multi-agent system.

Pipeline:

    User Query
        ↓
    P6 Input Guardrails
        ↓
    P4 Memory Load
        ↓
    P3/P5 Multi-Agent LangGraph
        ↓
    P6 Output Guardrails + Compliance + Risk Scoring
        ↓
    P4 Memory Save
        ↓
    Final Safe Result

P4 memory integration:
    BEFORE query:
        load_memory_context()
            ↓
        inject memory_context into AgentState
            ↓
        P3/P5 multi-agent pipeline

    AFTER query:
        save_turn()
            ↓
        semantic memory
        memory graph
        user profile
        answer cache
        compression

P6 Trust & Safety integration:
    BEFORE graph:
        prompt_firewall()
            ↓
        reject prompt injection / jailbreak / harmful input

    AFTER graph:
        full_tsl_check()
            ↓
        PII detection/redaction
        data leakage detection
        secure RAG verification
        faithfulness scoring
        compliance checking
        risk scoring

Usage:
    from agent_pipeline import run_query, build_agent_pipeline

    pipeline = build_agent_pipeline(nodes, vector_index)

    result = run_query(
        "How does BERT differ from GPT?",
        pipeline
    )
"""

import logging
import uuid
from typing import Optional


logger = logging.getLogger(__name__)


# ============================================================================
# AgentPipeline
# ============================================================================

class AgentPipeline:
    """
    Holds compiled LangGraph + retrieval context.

    Instantiated once at startup and reused across queries.
    """

    def __init__(self, compiled_graph, retrieval_ctx):
        self.graph = compiled_graph
        self.retrieval_ctx = retrieval_ctx


# ============================================================================
# Build pipeline
# ============================================================================

def build_agent_pipeline(
    nodes,
    vector_index,
    embed_model=None,
    llm=None,
) -> AgentPipeline:
    """
    Build and return a ready-to-use AgentPipeline.

    Args:
        nodes:
            Chunked document nodes from the ingestion pipeline.

        vector_index:
            VectorStoreIndex from the vector store.

        embed_model:
            Pre-loaded embedding model.
            Passing this avoids reloading the model.

        llm:
            Pre-loaded LLM.

    Returns:
        AgentPipeline containing the compiled LangGraph
        and retrieval context.
    """

    from P2.retrieval_pipeline import RetrievalContext
    from P3.router import build_graph

    ctx = RetrievalContext(
        nodes=nodes,
        vector_index=vector_index,
        embed_model=embed_model,
        llm=llm,
    )

    compiled_graph = build_graph(
        retrieval_ctx=ctx
    )

    return AgentPipeline(
        compiled_graph=compiled_graph,
        retrieval_ctx=ctx,
    )


# ============================================================================
# Run query
# ============================================================================

# ============================================================================
# Run query stream (Canonical Execution Engine)
# ============================================================================

def run_query_stream(
    query: str,
    pipeline: AgentPipeline,
    session_id: Optional[str] = None,
    user_role: str = "analyst",
    memory_context: str = "",
    use_memory: bool = True,
    turn_count: int = 1,
    use_web_research: bool = False,
):
    """
    Execute a query through the RAG + Memory + Guardrails pipeline,
    yielding safe real-time progress events as each lifecycle step and
    LangGraph agent completes.

    Yields:
        dict: Event envelopes {"event": "<name>", "data": {<safe_payload>}}
    """
    from P3.state import initial_state
    from P3.router import save_agent_communication_log

    # ========================================================================
    # 1. Create identifiers
    # ========================================================================

    run_id = str(uuid.uuid4())[:8]
    session_id = session_id or run_id

    logger.info(
        f"[Pipeline] run_id={run_id} | "
        f"session_id={session_id} | "
        f"role={user_role} | "
        f"query='{query[:80]}'"
    )

    yield {
        "event": "connected",
        "data": {
            "status": "connected",
            "run_id": run_id,
            "session_id": session_id,
        },
    }

    # ========================================================================
    # 2. P6 — Input Prompt Firewall
    # ========================================================================

    try:
        from guardrails.input_guardrails import prompt_firewall

        firewall_result = prompt_firewall(
            text=query,
            session_id=session_id,
            user_role=user_role,
        )

    except Exception as e:
        logger.error(f"[Pipeline] Input firewall failed: {e}")
        blocked_result = {
            "answer": (
                "Request blocked because the security validation "
                "service could not complete."
            ),
            "consensus_score": 0.0,
            "is_final": False,
            "iterations": 0,
            "run_id": run_id,
            "session_id": session_id,
            "user_role": user_role,
            "guardrail_blocked": True,
            "guardrail_reason": "Input firewall failure",
        }
        yield {
            "event": "firewall",
            "data": {
                "status": "error",
                "layer": "input_firewall",
                "reason": "Security validation failure",
            },
        }
        yield {"event": "final", "data": blocked_result}
        return

    if not firewall_result.allowed:
        logger.warning(
            f"[Pipeline] Query blocked by input firewall | "
            f"run_id={run_id} | "
            f"layer={firewall_result.layer} | "
            f"reason={firewall_result.reason}"
        )
        blocked_result = {
            "answer": (
                "Your request was blocked by the Trust & Safety Layer."
            ),
            "consensus_score": 0.0,
            "is_final": False,
            "iterations": 0,
            "run_id": run_id,
            "session_id": session_id,
            "user_role": user_role,
            "guardrail_blocked": True,
            "guardrail_layer": firewall_result.layer,
            "guardrail_reason": firewall_result.reason,
        }
        yield {
            "event": "firewall",
            "data": {
                "status": "blocked",
                "layer": firewall_result.layer,
                "reason": firewall_result.reason,
            },
        }
        yield {"event": "final", "data": blocked_result}
        return

    # Always use the sanitized version after the firewall.
    query = firewall_result.sanitized_text
    yield {
        "event": "firewall",
        "data": {
            "status": "passed",
            "layer": "clean",
        },
    }

    # ========================================================================
    # 3. Create initial AgentState
    # ========================================================================

    state = initial_state(
        query=query,
        session_id=session_id,
        user_role=user_role,
        memory_context=memory_context,
        use_web_research=use_web_research,
    )

    # ========================================================================
    # 4. P4 — Load long-term memory BEFORE graph execution
    # ========================================================================

    if use_memory and not memory_context:
        try:
            from memory.memory_manager import load_memory_context

            logger.info(
                f"[Pipeline] Loading memory | session_id={session_id}"
            )

            memory_context = load_memory_context(
                query=query,
                session_id=session_id,
                embed_model=(
                    pipeline.retrieval_ctx.embed_model
                    if pipeline.retrieval_ctx
                    else None
                ),
            )
            state["memory_context"] = memory_context
            logger.info(
                f"[Pipeline] Memory loaded | context_length={len(memory_context)}"
            )
            yield {
                "event": "memory",
                "data": {
                    "status": "loaded",
                    "has_context": bool(memory_context),
                },
            }
        except Exception as e:
            logger.warning(f"[Pipeline] Failed to load memory: {e}")
            state["memory_context"] = ""
            yield {
                "event": "memory",
                "data": {
                    "status": "skipped",
                    "has_context": False,
                },
            }
    else:
        yield {
            "event": "memory",
            "data": {
                "status": "skipped" if not use_memory else "loaded",
                "has_context": bool(memory_context),
            },
        }

    # ========================================================================
    # 5. LangGraph thread configuration
    # ========================================================================

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    # ========================================================================
    # 6. Run P3/P5 multi-agent graph with live node streaming
    # ========================================================================

    final_state = dict(state)

    try:
        if hasattr(pipeline.graph, "stream") and callable(pipeline.graph.stream):
            for chunk in pipeline.graph.stream(state, config=config, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    if isinstance(node_update, dict):
                        final_state.update(node_update)

                    # Emit safe public progress events per node
                    if node_name == "supervisor":
                        yield {
                            "event": "supervisor",
                            "data": {
                                "status": "completed",
                                "query_type": node_update.get("query_type", "unknown"),
                                "route": node_update.get("route", ""),
                            },
                        }
                    elif node_name == "planner":
                        sub_queries = node_update.get("sub_queries", [])
                        yield {
                            "event": "planner",
                            "data": {
                                "status": "completed",
                                "sub_queries_count": len(sub_queries),
                            },
                        }
                    elif node_name == "researcher":
                        chunks = node_update.get("reranked_chunks", [])
                        web_used = bool(node_update.get("web_research_used", False))
                        if web_used:
                            web_sources_count = sum(1 for c in chunks if c.get("source") == "web_search")
                            yield {
                                "event": "web_research",
                                "data": {
                                    "status": "completed",
                                    "web_sources_count": web_sources_count,
                                },
                            }
                        yield {
                            "event": "retrieval",
                            "data": {
                                "status": "completed",
                                "sources_count": len(chunks),
                                "graph_rag_used": bool(node_update.get("graph_rag_used", False)),
                                "web_research_used": web_used,
                            },
                        }
                    elif node_name == "generator":
                        citations = node_update.get("citations", [])
                        yield {
                            "event": "generation",
                            "data": {
                                "status": "completed",
                                "role_used": node_update.get("role_used", ""),
                                "citations_count": len(citations),
                            },
                        }
                    elif node_name == "critic":
                        critique = node_update.get("critique", {})
                        yield {
                            "event": "critique",
                            "data": {
                                "status": "completed",
                                "faithfulness": critique.get("faithfulness", 0),
                                "completeness": critique.get("completeness", 0),
                            },
                        }
                    elif node_name == "reviewer":
                        yield {
                            "event": "review",
                            "data": {
                                "status": "completed",
                                "consensus_score": float(node_update.get("consensus_score", 0.0)),
                                "is_final": bool(node_update.get("is_final", False)),
                            },
                        }
                    elif node_name == "router":
                        route = node_update.get("route", "")
                        yield {
                            "event": "refinement",
                            "data": {
                                "status": "looping" if route == "researcher" else "completed",
                                "iteration": int(node_update.get("iteration", 0)),
                            },
                        }
        else:
            invoked_state = pipeline.graph.invoke(state, config=config)
            final_state.update(invoked_state)

    except Exception as e:
        logger.error(f"[Pipeline] Graph execution failed: {e}")
        error_result = {
            "answer": f"An error occurred: {e}",
            "consensus_score": 0.0,
            "is_final": False,
            "iterations": 0,
            "run_id": run_id,
            "session_id": session_id,
            "user_role": user_role,
        }
        yield {
            "event": "error",
            "data": {
                "status": "error",
                "message": "Graph execution failed.",
            },
        }
        yield {"event": "final", "data": error_result}
        return

    # ========================================================================
    # 7. Save agent communication log
    # ========================================================================

    try:
        save_agent_communication_log(
            final_state,
            run_id,
        )
    except Exception as e:
        logger.warning(
            f"[Pipeline] Failed to save agent communication log: {e}"
        )

    # ========================================================================
    # 8. Extract retrieved evidence
    # ========================================================================

    reranked_chunks = final_state.get(
        "reranked_chunks",
        []
    )

    sources = []
    source_details = []
    seen_sources = set()

    for c in reranked_chunks:
        meta = c.get("metadata", {})
        fname = meta.get("filename") or meta.get("title") or "unknown"
        if fname not in seen_sources:
            seen_sources.add(fname)
            sources.append(fname)

        source_details.append({
            "filename": fname,
            "title": meta.get("title", fname),
            "url": meta.get("url"),
            "domain": meta.get("domain"),
            "source_type": meta.get("source_type", "document"),
            "snippet": c.get("text", "")[:300],
            "doc_category": meta.get("doc_category", "general"),
            "strategy": c.get("source", "hybrid"),
        })

    # ========================================================================
    # 9. Extract generated answer
    # ========================================================================

    raw_answer = final_state.get(
        "answer",
        ""
    )

    # ========================================================================
    # 10. P6 — Full Trust & Safety Layer
    # ========================================================================

    try:
        from guardrails.risk_scoring import full_tsl_check

        tsl_result = full_tsl_check(
            query=query,
            answer=raw_answer,
            source_chunks=reranked_chunks,
            session_id=session_id,
            run_id=run_id,
            user_role=user_role,
        )

    except Exception as e:
        logger.error(f"[Pipeline] Output TSL failed: {e}")
        tsl_failed_result = {
            "answer": (
                "The generated response could not be returned "
                "because security validation failed."
            ),
            "consensus_score": final_state.get(
                "consensus_score",
                0.0
            ),
            "is_final": False,
            "iterations": final_state.get(
                "iteration",
                0
            ),
            "query_type": final_state.get(
                "query_type",
                ""
            ),
            "role_used": final_state.get(
                "role_used",
                ""
            ),
            "citations": final_state.get(
                "citations",
                []
            ),
            "sources": sources,
            "source_details": source_details,
            "web_research_used": bool(final_state.get("web_research_used", False)),
            "strategy": final_state.get(
                "retrieval_strategy",
                ""
            ),
            "graph_rag_used": final_state.get(
                "graph_rag_used",
                False
            ),
            "critique": final_state.get(
                "critique",
                {}
            ),
            "agent_messages": final_state.get(
                "agent_messages",
                []
            ),
            "run_id": run_id,
            "session_id": session_id,
            "user_role": user_role,
            "guardrail_blocked": True,
            "guardrail_reason": "Output TSL failure",
        }
        yield {
            "event": "tsl",
            "data": {
                "status": "failed",
                "reason": "Output validation failure",
            },
        }
        yield {"event": "final", "data": tsl_failed_result}
        return

    # ========================================================================
    # 11. TSL-cleaned answer becomes the public answer
    # ========================================================================

    safe_answer = tsl_result.get(
        "answer_clean",
        raw_answer,
    )

    yield {
        "event": "tsl",
        "data": {
            "status": "completed",
            "risk_score": float(tsl_result.get("risk_score", 0.0)),
            "pii_found": bool(tsl_result.get("pii_found", False)),
        },
    }

    # ========================================================================
    # 12. Build final result
    # ========================================================================

    result = {
        "answer": safe_answer,
        "consensus_score": final_state.get(
            "consensus_score",
            0.0
        ),
        "is_final": final_state.get(
            "is_final",
            False
        ),
        "iterations": final_state.get(
            "iteration",
            0
        ),
        "query_type": final_state.get(
            "query_type",
            ""
        ),
        "role_used": final_state.get(
            "role_used",
            ""
        ),
        "citations": final_state.get(
            "citations",
            []
        ),
        "sources": sources,
        "source_details": source_details,
        "web_research_used": bool(final_state.get("web_research_used", False)),
        "strategy": final_state.get(
            "retrieval_strategy",
            ""
        ),
        "graph_rag_used": final_state.get(
            "graph_rag_used",
            False
        ),
        "critique": final_state.get(
            "critique",
            {}
        ),
        "agent_messages": final_state.get(
            "agent_messages",
            []
        ),
        "run_id": run_id,
        "session_id": session_id,
        "user_role": user_role,
        "tsl": tsl_result,
        "guardrail_blocked": False,
        "guardrail_reason": None,
    }

    # ========================================================================
    # 13. P4 — Save completed turn
    # ========================================================================

    if use_memory:
        try:
            from memory.memory_manager import save_turn

            logger.info(
                f"[Pipeline] Saving turn to memory | session_id={session_id}"
            )

            save_turn(
                query=query,
                answer=safe_answer,
                session_id=session_id,
                query_type=final_state.get(
                    "query_type",
                    "unknown"
                ),
                consensus_score=final_state.get(
                    "consensus_score",
                    0.0
                ),
                critique=final_state.get(
                    "critique",
                    {}
                ),
                reranked_chunks=reranked_chunks,
                embed_model=(
                    pipeline.retrieval_ctx.embed_model
                    if pipeline.retrieval_ctx
                    else None
                ),
                turn_count_in_session=turn_count,
            )

            logger.info(
                f"[Pipeline] Turn saved successfully | session_id={session_id}"
            )

        except Exception as e:
            logger.warning(
                f"[Pipeline] Failed to save turn to memory: {e}"
            )

    # ========================================================================
    # 14. Final logging and yield final response event
    # ========================================================================

    logger.info(
        f"[Pipeline] Done | "
        f"run_id={run_id} | "
        f"consensus={result['consensus_score']:.2f} | "
        f"iterations={result['iterations']} | "
        f"is_final={result['is_final']} | "
        f"risk={tsl_result.get('risk_score', 0.0):.3f}"
    )

    yield {
        "event": "final",
        "data": result,
    }


# ============================================================================
# Run query (Synchronous Collector over Canonical Stream)
# ============================================================================

def run_query(
    query: str,
    pipeline: AgentPipeline,
    session_id: Optional[str] = None,
    user_role: str = "analyst",
    memory_context: str = "",
    use_memory: bool = True,
    turn_count: int = 1,
    use_web_research: bool = False,
) -> dict:
    """
    Run a query through the canonical RAG execution engine, collecting
    the final result dictionary.

    This ensures 100% behavioral consistency with run_query_stream().
    """
    final_result = None

    for event in run_query_stream(
        query=query,
        pipeline=pipeline,
        session_id=session_id,
        user_role=user_role,
        memory_context=memory_context,
        use_memory=use_memory,
        turn_count=turn_count,
        use_web_research=use_web_research,
    ):
        if event["event"] == "final":
            final_result = event["data"]

    return final_result or {
        "answer": "An unexpected error occurred during execution.",
        "consensus_score": 0.0,
        "is_final": False,
        "iterations": 0,
        "run_id": "",
        "session_id": session_id or "",
        "user_role": user_role,
    }


# ============================================================================
# Architecture
# ============================================================================

'''
                         START
                           │
                           ▼
                    INPUT FIREWALL
                     P6 — T1
                    /        \
                   /          \
                BLOCK         PASS
                  │             │
                 END            ▼
                         Load Memory
                            P4
                             │
                             ▼
                         Supervisor
                       /     |      \
                      /      |       \
                     ▼       ▼        ▼
                Planner  Researcher  Generator
                    │
                    ▼
                Researcher
                    │
                    ▼
                 Generator
                    │
                    ▼
                  Critic
                    │
                    ▼
                 Reviewer
                    │
                    ▼
                  Router
                 /      \
                /        \
               ▼          ▼
          Researcher     END
              │
              └───────────────┐
                              │
                              ▼
                       OUTPUT TSL
                          P6
                  /        |        \
                 /         |         \
               PII      Leakage   Faithfulness
                 \         |         /
                  \        |        /
                       Compliance
                           │
                           ▼
                      Risk Score
                           │
                           ▼
                     Safe Answer
                           │
                           ▼
                      Save Turn
                         P4
                           │
                           ▼
                          END
'''

