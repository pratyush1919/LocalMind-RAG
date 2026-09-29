"""
api/routes/graph.py
Read-only Knowledge Graph endpoint for LocalMind-RAG.
Exposes real graph nodes, relationships, and provenance without modifying any backend execution code.
"""

import logging
from fastapi import APIRouter, status

from api.schemas.graph import (
    GraphEdgeItem,
    GraphNodeItem,
    GraphResponse,
)
from graph_store.knowledge_graph import load_graph

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/graph",
    response_model=GraphResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Knowledge Graph nodes and edges",
    description="Retrieve the complete concept and entity relationship graph with provenance references.",
)
async def get_graph_endpoint() -> GraphResponse:
    """
    Handle GET /api/graph.
    Returns real persisted graph nodes and directed relationship edges.
    """
    try:
        G = load_graph()

        node_items = [
            GraphNodeItem(
                id=str(node_id),
                label=str(G.nodes[node_id].get("label", node_id)),
                frequency=int(G.nodes[node_id].get("frequency", 0)),
            )
            for node_id in G.nodes()
        ]

        edge_items = []
        for u, v, data in G.edges(data=True):
            edge_items.append(
                GraphEdgeItem(
                    source=str(u),
                    target=str(v),
                    predicate=str(data.get("predicate", "related-to")),
                    predicates=[str(p) for p in data.get("predicates", [])],
                    weight=float(data.get("weight", 1.0)),
                    source_file=data.get("source_file"),
                    chunk_id=data.get("chunk_id"),
                    provenance=data.get("provenance", []),
                )
            )

        return GraphResponse(
            total_nodes=len(node_items),
            total_edges=len(edge_items),
            nodes=node_items,
            edges=edge_items,
        )
    except Exception as e:
        logger.error(f"[GraphRoute] Failed to load knowledge graph: {e}", exc_info=True)
        return GraphResponse(
            total_nodes=0,
            total_edges=0,
            nodes=[],
            edges=[],
        )
