"""
api/dependencies.py
FastAPI dependency helpers for accessing shared backend instances.
"""

from typing import Optional
from fastapi import HTTPException, Request, status

from P2.retrieval_pipeline import RetrievalContext
from P3.agent_pipeline import AgentPipeline


def get_pipeline(request: Request) -> AgentPipeline:
    """
    Retrieve the compiled AgentPipeline singleton from app.state.
    Raises 503 if the RAG pipeline is uninitialized (e.g., no documents indexed yet).
    """
    pipeline: Optional[AgentPipeline] = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is uninitialized. Please ensure document nodes are indexed.",
        )
    return pipeline


def get_retrieval_context(request: Request) -> RetrievalContext:
    """
    Retrieve the shared RetrievalContext singleton from app.state.
    Raises 503 if uninitialized.
    """
    ctx: Optional[RetrievalContext] = getattr(request.app.state, "retrieval_ctx", None)
    if ctx is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval context is uninitialized.",
        )
    return ctx
