"""
api/routes/health.py
Health check route for LocalMind-RAG.
"""

import os
import logging
from fastapi import APIRouter, Request

from api.schemas.common import (
    GraphStoreHealth,
    HealthResponse,
    LLMHealth,
    VectorStoreHealth,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Perform a health check on the LocalMind-RAG backend.
    Reports vector store counts, pipeline status, LLM configuration, and graph store state.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    
    # 1. Vector store status
    doc_count = 0
    vector_status = "uninitialized"
    try:
        from P1.vector_store import collection_count
        doc_count = collection_count("chunks")
        vector_status = "ready" if doc_count > 0 else "empty"
    except Exception as e:
        logger.warning(f"[Health] Vector store check error: {e}")
        vector_status = f"error: {e}"

    # 2. RAG pipeline status
    if pipeline is not None and vector_status == "ready":
        rag_status = "ready"
    elif pipeline is not None:
        rag_status = "ready_empty_store"
    else:
        rag_status = "uninitialized"

    # 3. LLM backend configuration
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "groq":
        model_name = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        llm_status = "configured (groq cloud)"
    else:
        model_name = os.getenv("OLLAMA_MODEL", "mistral-rag")
        llm_status = "configured (ollama local)"

    # 4. Knowledge Graph stats
    graph_health = None
    try:
        from graph_store.knowledge_graph import get_graph_stats
        stats = get_graph_stats()
        graph_health = GraphStoreHealth(
            status="ready" if stats["nodes"] > 0 else "empty",
            node_count=stats["nodes"],
            edge_count=stats["edges"],
        )
    except Exception as e:
        logger.debug(f"[Health] Graph store check notice: {e}")
        graph_health = GraphStoreHealth(
            status="uninitialized",
            node_count=0,
            edge_count=0,
        )

    # 5. Overall status determination
    if rag_status == "ready" and vector_status == "ready":
        overall_status = "ok"
    elif rag_status in ("ready_empty_store", "uninitialized"):
        overall_status = "degraded"
    else:
        overall_status = "error"

    return HealthResponse(
        status=overall_status,
        service="LocalMind-RAG",
        version="1.0.0",
        rag_pipeline=rag_status,
        vector_store=VectorStoreHealth(
            status=vector_status,
            collection_name="chunks",
            document_count=doc_count,
        ),
        llm=LLMHealth(
            backend=backend,
            model=model_name,
            status=llm_status,
        ),
        graph_store=graph_health,
    )
