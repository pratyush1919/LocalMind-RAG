"""
api/schemas/common.py
Common Pydantic schemas for the LocalMind-RAG API.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class VectorStoreHealth(BaseModel):
    """Health information for ChromaDB collections."""
    status: str = Field(..., description="Vector store status, e.g. ready or empty")
    collection_name: str = Field(..., description="Primary collection name")
    document_count: int = Field(..., description="Number of vectorized chunk records")


class LLMHealth(BaseModel):
    """Health information for the configured LLM backend."""
    backend: str = Field(..., description="Configured LLM backend (ollama or groq)")
    model: str = Field(..., description="Model identifier name")
    status: str = Field(..., description="Backend availability or configuration status")


class GraphStoreHealth(BaseModel):
    """Health information for the NetworkX knowledge graph."""
    status: str = Field(..., description="Graph store status")
    node_count: int = Field(..., description="Number of entity nodes in graph")
    edge_count: int = Field(..., description="Number of relationship edges in graph")


class HealthResponse(BaseModel):
    """Standard response model for GET /api/health."""
    status: str = Field(..., description="Overall system health status: ok, degraded, or error")
    service: str = Field("LocalMind-RAG", description="Service identifier name")
    version: str = Field("1.0.0", description="API version")
    rag_pipeline: str = Field(..., description="Status of the compiled multi-agent RAG pipeline")
    vector_store: VectorStoreHealth = Field(..., description="Vector database status")
    llm: LLMHealth = Field(..., description="LLM backend status")
    graph_store: Optional[GraphStoreHealth] = Field(None, description="Knowledge graph status if available")


class ErrorDetail(BaseModel):
    """Structured error detail payload."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error explanation")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Optional diagnostic details")


class ErrorResponse(BaseModel):
    """Standardized API error envelope."""
    error: ErrorDetail
