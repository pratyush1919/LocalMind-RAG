"""
graph_store/graph_retriever.py  —  P5-T2
Graph RAG: retrieve structured relationship facts from the knowledge graph
and combine them with standard vector chunks.

Concepts covered:
  - Graph RAG                 (knowledge graph + vector retrieval combined)
  - Knowledge Graph Retrieval (entity lookup + traversal in the graph)
  - Multi-Hop Retrieval       (X → Y → Z path traversal — 2 hops)

Why Graph RAG answers differently than pure vector RAG:
  Query: "How does BERT relate to the attention mechanism?"

  Vector RAG: returns chunks mentioning BERT and chunks mentioning attention
              separately. The LLM must infer the relationship.

  Graph RAG:  finds BERT node → traverses edges → finds "BERT uses attention"
              and "attention is-a mechanism" in 1-2 hops → returns structured
              relationship facts PLUS the vector chunks for full context.

  Result: the Generator gets both structured facts (from graph) and passage
          context (from vectors) — richer, more accurate answers for
          relationship/comparative queries.

Graph retrieval flow:
  1. Extract key entities from the query (simple keyword match against graph nodes)
  2. For each entity: find all 1-hop neighbors (direct relationships)
  3. For each 1-hop neighbor: find all 2-hop neighbors (indirect relationships)
  4. Format graph facts as structured text
  5. Combine with vector chunks via RRF fusion in the agent state
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Entity matching
# ─────────────────────────────────────────────────────────────────────────────

def find_query_entities(query: str, G) -> List[str]:
    """
    Find graph nodes that match terms in the query.

    Strategy: exact substring match (query_term in node_id or vice versa).
    Normalized to lowercase for matching.

    Returns list of matched node IDs.
    """
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    matched = []

    for node_id in G.nodes():
        node_terms = set(node_id.split())
        # Match if any node term appears in query OR query term appears in node
        if any(term in query_lower for term in node_terms) or \
           any(term in node_id for term in query_terms if len(term) > 3):
            matched.append(node_id)

    # Sort by frequency (most referenced entities first)
    matched.sort(
        key=lambda n: G.nodes[n].get("frequency", 0),
        reverse=True,
    )
    return matched[:5]  # cap at 5 seed entities


# ─────────────────────────────────────────────────────────────────────────────
# Graph traversal
# ─────────────────────────────────────────────────────────────────────────────

def traverse_graph(
    G,
    seed_entities: List[str],
    max_hops: int = 2,
    max_facts: int = 20,
) -> List[dict]:
    """
    Traverse the knowledge graph from seed entities up to max_hops.

    For each (subject, predicate, object) triple reachable within max_hops:
      - Include the triple as a structured fact
      - Include provenance: source_file, chunk_id

    Args:
        G:              NetworkX DiGraph.
        seed_entities:  Starting nodes (matched from query).
        max_hops:       Maximum traversal depth (2 = X→Y→Z).
        max_facts:      Maximum facts to return.

    Returns:
        List of fact dicts: { subject, predicate, object, hop, source_file }

    Concepts: Knowledge Graph Retrieval, Multi-Hop Retrieval
    """
    facts = []
    visited_edges = set()

    queue = [(entity, 0) for entity in seed_entities]
    visited_nodes = set(seed_entities)

    while queue and len(facts) < max_facts:
        current_node, hop = queue.pop(0)
        if hop >= max_hops:
            continue

        # Outgoing edges (subject → object)
        for _, neighbor, edge_data in G.out_edges(current_node, data=True):
            edge_key = (current_node, neighbor)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            pred = edge_data.get("predicate", "related-to")
            facts.append({
                "subject": G.nodes[current_node].get("label", current_node),
                "predicate": pred,
                "object": G.nodes[neighbor].get("label", neighbor),
                "hop": hop + 1,
                "source_file": edge_data.get("source_file", "unknown"),
                "chunk_id": edge_data.get("chunk_id", ""),
            })

            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, hop + 1))

        # Incoming edges (object → subject) for bidirectional context
        for predecessor, _, edge_data in G.in_edges(current_node, data=True):
            edge_key = (predecessor, current_node)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            pred = edge_data.get("predicate", "related-to")
            facts.append({
                "subject": G.nodes[predecessor].get("label", predecessor),
                "predicate": pred,
                "object": G.nodes[current_node].get("label", current_node),
                "hop": hop + 1,
                "source_file": edge_data.get("source_file", "unknown"),
                "chunk_id": edge_data.get("chunk_id", ""),
            })

    # Sort by hop (closer = more relevant) then by source confidence
    facts.sort(key=lambda f: f["hop"])
    return facts[:max_facts]


# ─────────────────────────────────────────────────────────────────────────────
# Format graph facts for LLM context
# ─────────────────────────────────────────────────────────────────────────────

def format_graph_facts(facts):
    """
    Format Graph RAG facts with first-class graph citation IDs.

    Graph evidence uses a separate citation namespace:

        [Graph Source 1]
        [Graph Source 2]
        ...

    This prevents graph evidence from being confused with normal
    document citations such as [Source 1].
    """

    if not facts:
        return "[Graph RAG — Knowledge Graph Facts]\n  No graph facts found."

    lines = ["[Graph RAG — Knowledge Graph Facts]"]

    for idx, f in enumerate(facts, start=1):

        hop_label = (
            "direct"
            if f["hop"] == 1
            else f"{f['hop']}-hop"
        )

        source_file = f.get("source_file", "unknown")

        lines.append(
            f"  [Graph Source {idx}] "
            f"{f['subject']}  {f['predicate']}  {f['object']} "
            f"({source_file}, {hop_label})"
        )

    return "\n".join(lines)
# ─────────────────────────────────────────────────────────────────────────────
# Main Graph RAG retrieval function
# ─────────────────────────────────────────────────────────────────────────────

def graph_rag_retrieve(
    query: str,
    max_hops: int = 2,
    max_facts: int = 15,
) -> dict:
    """
    Full Graph RAG retrieval: entity matching → graph traversal → formatted facts.

    Called by the Research Agent node (P3-T4) when Graph RAG is enabled.
    Results are merged with vector chunks in the agent state.

    Args:
        query:      User query or sub-query.
        max_hops:   Traversal depth (2 covers X→Y→Z paths).
        max_facts:  Maximum facts to return.

    Returns:
        dict:
          facts:          List of structured fact dicts
          formatted_text: Ready-to-inject string for LLM context
          entities_found: List of seed entities matched
          graph_used:     bool (False if graph is empty or no entities matched)

    Concepts: Graph RAG, Knowledge Graph Retrieval, Multi-Hop Retrieval
    """
    from graph_store.knowledge_graph import load_graph

    G = load_graph()

    if G.number_of_nodes() == 0:
        logger.info("[GraphRAG] Knowledge graph is empty — skipping graph retrieval")
        return {
            "facts": [],
            "formatted_text": "",
            "entities_found": [],
            "graph_used": False,
        }

    # Step 1: Entity matching
    entities = find_query_entities(query, G)
    if not entities:
        logger.info(f"[GraphRAG] No entities matched for query: '{query[:60]}'")
        return {
            "facts": [],
            "formatted_text": "",
            "entities_found": [],
            "graph_used": False,
        }

    logger.info(f"[GraphRAG] Matched entities: {entities}")

    # Step 2: Graph traversal
    facts = traverse_graph(G, entities, max_hops=max_hops, max_facts=max_facts)
    logger.info(f"[GraphRAG] Traversal found {len(facts)} facts")

    # Step 3: Format for LLM
    formatted = format_graph_facts(facts)

    return {
        "facts": facts,
        "formatted_text": formatted,
        "entities_found": entities,
        "graph_used": len(facts) > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Integration helper: merge graph facts into generator context
# ─────────────────────────────────────────────────────────────────────────────

def augment_chunks_with_graph(
    vector_chunks: List[dict],
    graph_result: dict,
) -> Tuple[List[dict], str]:
    """
    Combine vector retrieval chunks with graph facts for the Generator.

    The graph facts are prepended to the context string as a structured block.
    Vector chunks remain as [Source N] citations.

    Returns:
        (chunks, graph_context_prefix)
        Where graph_context_prefix is prepended to the Generator's context block.

    Concept: Graph RAG (combined vector + graph context)
    """
    if not graph_result.get("graph_used"):
        return vector_chunks, ""

    graph_prefix = graph_result.get("formatted_text", "")
    entity_count = len(graph_result.get("entities_found", []))

    logger.info(
        f"[GraphRAG] Augmenting {len(vector_chunks)} vector chunks with "
        f"{len(graph_result['facts'])} graph facts ({entity_count} entities)"
    )
    return vector_chunks, graph_prefix
