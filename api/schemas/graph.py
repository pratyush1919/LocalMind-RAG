"""
api/schemas/graph.py
Pydantic schemas for the Knowledge Graph read-only endpoint.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GraphNodeItem(BaseModel):
    """A concept or entity node in the Knowledge Graph."""
    id: str = Field(..., description="Unique lowercase normalized node identifier")
    label: str = Field(..., description="Display label of the entity")
    frequency: int = Field(0, description="Reference frequency count across document chunks")


class GraphEdgeItem(BaseModel):
    """A directed predicate relationship edge between two entity nodes."""
    source: str = Field(..., description="Subject node id")
    target: str = Field(..., description="Object node id")
    predicate: str = Field(..., description="Primary predicate relationship")
    predicates: List[str] = Field(default_factory=list, description="All predicates connecting source and target")
    weight: float = Field(1.0, description="Edge weight / co-occurrence count")
    source_file: Optional[str] = Field(None, description="Primary source document provenance")
    chunk_id: Optional[str] = Field(None, description="Primary chunk id")
    provenance: List[Dict[str, Any]] = Field(default_factory=list, description="Provenance occurrences")


class GraphResponse(BaseModel):
    """Response model for GET /api/graph."""
    total_nodes: int = Field(..., description="Total nodes in the graph")
    total_edges: int = Field(..., description="Total relationships in the graph")
    nodes: List[GraphNodeItem] = Field(default_factory=list, description="Entity nodes")
    edges: List[GraphEdgeItem] = Field(default_factory=list, description="Relationship edges")
