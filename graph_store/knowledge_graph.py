""""
graph_store/knowledge_graph.py  —  P5-T1
Build a persistent knowledge graph from ingested documents.

Concepts covered:
  - Knowledge Graph Construction  (extract entities + relations from text)
  - Knowledge Graph Retrieval     (query the graph at retrieval time — P5-T2)

Pipeline:
  For each document chunk →
    LLM extracts: entities (people, orgs, concepts, dates)
                  relationships (X is-a Y, X caused Y, X related-to Y)
  → Store triples (subject, predicate, object) in NetworkX
  → Persist as /graph_store/knowledge_graph.json
  → Render interactive HTML via pyvis → /docs/knowledge_graph.html

Why Graph RAG matters:
  Standard RAG: "How does X relate to Z?" retrieves chunks mentioning X and Z
                separately — misses the *connection* between them.
  Graph RAG:    Traverses X → Y → Z path in the graph, surfacing the
                relationship chain that pure vector search cannot find.

Triple format:
  (subject, predicate, object, metadata)
  e.g. ("BERT", "is-a", "transformer model", {"source": "bert_paper.pdf", "chunk_id": "n3"})
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_GRAPH_PATH = Path("./graph_store/knowledge_graph.json")
_VIZ_PATH = Path("./docs/knowledge_graph.html")


# ─────────────────────────────────────────────────────────────────────────────
# Entity + relation extraction
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
Extract entities and relationships from the text below.

Entity types to extract: CONCEPT, MODEL, METHOD, ORGANIZATION, PERSON, DATASET, METRIC

For each relationship, output a JSON list of triples:
[
  {{"subject": "entity1", "predicate": "relationship", "object": "entity2"}},
  ...
]

Relationship types: is-a, part-of, uses, produces, improves-on, trained-on,
                    evaluated-on, introduced-by, related-to, contrasts-with

Rules:
- Extract 3-8 triples maximum per chunk (quality over quantity)
- Use short, normalized entity names (e.g. "BERT" not "the BERT model")
- Predicates must be from the list above
- If no clear relationships exist, return an empty list []
- Return ONLY the JSON list, no other text

Text:
{text}
"""


def _parse_triples_json(raw: str) -> List[dict]:
    """
    Robustly extract and validate JSON triples array from LLM response string.
    Handles:
      - <think>...</think> reasoning tags
      - markdown ```json ... ``` code blocks
      - conversational preamble and trailing commentary
      - malformed JSON and predicate normalization
    """
    if not raw or not isinstance(raw, str):
        return []

    # 1. Strip reasoning/think tags (e.g. Qwen, DeepSeek)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()

    # 2. Extract content from markdown code fence if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        candidate = cleaned

    # 3. Parse JSON array
    data = None
    try:
        data = json.loads(candidate)
    except Exception:
        # Fallback: search for outermost [ ... ] array
        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", candidate)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
            except Exception:
                pass

        # Fallback: search for single object { ... } if model emitted single triple
        if data is None:
            dict_match = re.search(r"\{[\s\S]*\}", candidate)
            if dict_match:
                try:
                    single = json.loads(dict_match.group(0))
                    if isinstance(single, dict):
                        data = [single]
                except Exception:
                    pass

    if not isinstance(data, list):
        return []

    valid_predicates = {
        "is-a", "part-of", "uses", "produces", "improves-on",
        "trained-on", "evaluated-on", "introduced-by", "related-to", "contrasts-with",
    }

    valid = []
    for t in data:
        if not isinstance(t, dict):
            continue
        subj = str(t.get("subject", "")).strip()
        pred = str(t.get("predicate", "")).strip().lower()
        obj = str(t.get("object", "")).strip()

        if subj and obj and pred in valid_predicates:
            valid.append({"subject": subj, "predicate": pred, "object": obj})

    return valid[:8]  # max 8 triples per chunk


def extract_triples(text: str, llm=None) -> List[dict]:
    """
    Extract entity-relationship triples from a text chunk using an LLM.

    Returns list of {"subject": str, "predicate": str, "object": str} dicts.

    Concept: Knowledge Graph Construction
    """
    if llm is None:
        from P1.llm_factory import get_langchain_llm
        llm = get_langchain_llm(temperature=0.0)

    prompt = _EXTRACTION_PROMPT.format(text=text[:1200])  # cap input length

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        return _parse_triples_json(raw)
    except Exception as e:
        logger.warning(f"[KG] Triple extraction failed on chunk: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Graph persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    """Load the knowledge graph from disk. Returns empty NetworkX graph if not found."""
    import networkx as nx
    G = nx.DiGraph()  # directed — "BERT uses attention" ≠ "attention uses BERT"
    if _GRAPH_PATH.exists():
        try:
            data = json.loads(_GRAPH_PATH.read_text())
            G = nx.node_link_graph(data, directed=True, multigraph=False)
            logger.debug(
                f"[KG] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )
        except Exception as e:
            logger.warning(f"[KG] Failed to load graph: {e} — starting fresh")
    return G


def save_graph(G) -> None:
    """Persist the NetworkX DiGraph to disk as JSON."""
    _GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = __import__("networkx").node_link_data(G)
    _GRAPH_PATH.write_text(json.dumps(data, indent=2))
    logger.debug(f"[KG] Saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ─────────────────────────────────────────────────────────────────────────────
# Graph population
# ─────────────────────────────────────────────────────────────────────────────

def add_triples_to_graph(
    G,
    triples: List[dict],
    source_file: str = "",
    chunk_id: str = "",
) -> int:
    """
    Add extracted triples to the NetworkX graph.

    Nodes: entity names (normalized to lowercase)
    Edges: directed (subject → object) with predicate + provenance metadata

    Provenance is accumulated across chunks so that the same relationship
    can point back to every source chunk that supported it.

    Returns number of new edges added.
    """
    added = 0

    for triple in triples:
        subj = triple["subject"].strip().lower()
        pred = triple["predicate"].strip()
        obj = triple["object"].strip().lower()

        if not subj or not obj:
            continue

        # ─────────────────────────────────────────────────────────────
        # Add / update nodes
        # ─────────────────────────────────────────────────────────────

        if subj not in G:
            G.add_node(
                subj,
                label=triple["subject"],
                frequency=0,
            )

        if obj not in G:
            G.add_node(
                obj,
                label=triple["object"],
                frequency=0,
            )

        G.nodes[subj]["frequency"] = (
            G.nodes[subj].get("frequency", 0) + 1
        )

        # ─────────────────────────────────────────────────────────────
        # Provenance record for this occurrence
        # ─────────────────────────────────────────────────────────────

        provenance_record = {
            "source_file": source_file,
            "chunk_id": chunk_id,
        }

        # ─────────────────────────────────────────────────────────────
        # Existing edge
        # ─────────────────────────────────────────────────────────────

        if G.has_edge(subj, obj):
            edge = G[subj][obj]

            # Accumulate predicates on the same edge
            existing_predicates = edge.get("predicates", [])

            if pred not in existing_predicates:
                existing_predicates.append(pred)

            edge["predicates"] = existing_predicates

            # Keep existing primary predicate for backward compatibility
            edge.setdefault("predicate", pred)

            # Increment relationship frequency
            edge["weight"] = edge.get("weight", 1) + 1

            # Accumulate provenance without duplicates
            provenance = edge.get("provenance", [])

            if provenance_record not in provenance:
                provenance.append(provenance_record)

            edge["provenance"] = provenance

        # ─────────────────────────────────────────────────────────────
        # New edge
        # ─────────────────────────────────────────────────────────────

        else:
            G.add_edge(
                subj,
                obj,
                predicate=pred,
                predicates=[pred],
                weight=1,

                # Backward-compatible original fields
                source_file=source_file,
                chunk_id=chunk_id,

                # New accumulated provenance
                provenance=[provenance_record],
            )

            added += 1

    return added


def select_representative_nodes(nodes: List, max_chunks: int = 15) -> List:
    """
    Select up to max_chunks representative nodes across the document for KG extraction.
    1. Filters out trivial/boilerplate chunks (< 30 words) unlikely to contain entity relationships.
    2. Deterministically samples evenly distributed chunks across the entire document.
    
    Note: 100% of all document chunks remain indexed in vector and BM25 stores;
    sampling applies strictly to the relation-extraction LLM step.
    """
    if not nodes:
        return []

    # 1. Filter out trivial / boilerplate snippets
    meaningful = [
        n for n in nodes
        if len(getattr(n, "text", "").strip().split()) >= 30
    ]
    if not meaningful:
        meaningful = list(nodes)

    if len(meaningful) <= max_chunks:
        return meaningful

    # 2. Deterministic, document-wide stratification
    m = len(meaningful)
    step = (m - 1) / (max_chunks - 1)
    indices = [round(i * step) for i in range(max_chunks)]

    # Preserve order and eliminate duplicate indices if any
    seen = set()
    selected_indices = []
    for idx in indices:
        if idx not in seen and 0 <= idx < m:
            seen.add(idx)
            selected_indices.append(idx)

    return [meaningful[i] for i in selected_indices]


def build_knowledge_graph(
    nodes: List,
    llm=None,
    batch_size: int = 20,
    max_chunks: int = 15,
    checkpoint_interval: int = 3,
) -> dict:
    """
    Build or update the knowledge graph from a list of document nodes.

    - Reuses a single LLM client instance across the entire extraction.
    - Caps extraction to representative chunks (<= max_chunks) to avoid 10+ min hangs.
    - 100% of document chunks are retained in vector/BM25 retrieval.
    - Checkpoints graph incrementally to disk during extraction.
    - Merges with existing graph without duplicate edges or lost provenance.

    Args:
        nodes:               LlamaIndex BaseNode list (from chunker).
        llm:                 LangChain LLM for triple extraction (reused if provided).
        batch_size:          Log interval for progress reporting.
        max_chunks:          Maximum representative chunks to process via local LLM.
        checkpoint_interval: Save graph to disk every N chunks.

    Returns:
        dict: { nodes, edges, triples_extracted, skipped, chunks_analyzed, total_document_chunks }

    Concept: Knowledge Graph Construction
    """
    # 1. Re-use single LLM client instance
    if llm is None:
        from P1.llm_factory import get_langchain_llm
        llm = get_langchain_llm(temperature=0.0)

    # 2. Load existing graph (persists across document ingestions)
    G = load_graph()
    existing_chunks = set()

    for u, v, edge_data in G.edges(data=True):
        # New provenance-aware format
        for provenance in edge_data.get("provenance", []):
            chunk_id = provenance.get("chunk_id")
            if chunk_id:
                existing_chunks.add(chunk_id)

        # Backward compatibility with graphs created before T4
        legacy_chunk_id = edge_data.get("chunk_id")
        if legacy_chunk_id:
            existing_chunks.add(legacy_chunk_id)

    # 3. Select deterministic representative nodes across the document
    representative_nodes = select_representative_nodes(nodes, max_chunks=max_chunks)
    logger.info(
        f"[KG] Representative extraction: selected {len(representative_nodes)}/{len(nodes)} chunks "
        f"(capped at {max_chunks} for local LLM extraction; 100% of chunks indexed in vector/BM25)"
    )

    total_triples = 0
    skipped = 0

    # 4. Sequential extraction with incremental checkpointing
    for idx, node in enumerate(representative_nodes):
        chunk_id = getattr(node, "node_id", None) or getattr(node, "id_", "unknown")
        if chunk_id in existing_chunks:
            skipped += 1
            continue

        source_file = node.metadata.get("filename", "unknown") if hasattr(node, "metadata") else "unknown"
        triples = extract_triples(node.text, llm=llm)

        if triples:
            added = add_triples_to_graph(G, triples, source_file, chunk_id)
            total_triples += added

        # Incremental checkpointing
        if (idx + 1) % checkpoint_interval == 0:
            save_graph(G)
            logger.debug(
                f"[KG] Checkpointed graph at chunk {idx + 1}/{len(representative_nodes)} "
                f"| edges={G.number_of_edges()} | triples={total_triples}"
            )

        if (idx + 1) % batch_size == 0 or (idx + 1) == len(representative_nodes):
            logger.info(
                f"[KG] Processed {idx + 1}/{len(representative_nodes)} representative chunks "
                f"| edges={G.number_of_edges()} | triples={total_triples}"
            )

    # 5. Final persistence
    save_graph(G)

    result = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "triples_extracted": total_triples,
        "skipped": skipped,
        "chunks_analyzed": len(representative_nodes),
        "total_document_chunks": len(nodes),
    }
    logger.info(f"[KG] Build complete: {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pyvis visualization
# ─────────────────────────────────────────────────────────────────────────────

def render_graph_html(max_nodes: int = 100) -> Optional[str]:
    """
    Render the knowledge graph as an interactive HTML file using pyvis.

    Limits to the top max_nodes by frequency to keep the visualization fast.
    Output saved to /docs/knowledge_graph.html.

    Returns the path string if successful, None if pyvis not installed.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("[KG] pyvis not installed — skipping visualization")
        return None

    G = load_graph()
    if G.number_of_nodes() == 0:
        logger.info("[KG] Graph is empty — no visualization to render")
        return None

    # Select top nodes by frequency
    top_nodes = sorted(
        G.nodes(data=True),
        key=lambda x: x[1].get("frequency", 0),
        reverse=True,
    )[:max_nodes]
    top_node_ids = {n[0] for n in top_nodes}
    subgraph = G.subgraph(top_node_ids)

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#eee",
        directed=True,
    )
    net.barnes_hut(spring_length=200, spring_strength=0.05)

    # Color nodes by frequency tier
    for node_id, data in subgraph.nodes(data=True):
        freq = data.get("frequency", 1)
        color = "#e94560" if freq >= 5 else "#16213e" if freq >= 2 else "#0f3460"
        net.add_node(
            node_id,
            label=data.get("label", node_id),
            title=f"{node_id} (freq={freq})",
            color=color,
            size=10 + min(freq * 2, 20),
        )

    for u, v, data in subgraph.edges(data=True):
        pred = data.get("predicate", "related-to")
        net.add_edge(u, v, title=pred, label=pred, color="#888", arrows="to")

    _VIZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(_VIZ_PATH))
    logger.info(f"[KG] Visualization saved: {_VIZ_PATH}")
    return str(_VIZ_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Graph stats (for Gradio dashboard P9)
# ─────────────────────────────────────────────────────────────────────────────

def get_graph_stats() -> dict:
    """Return summary statistics about the knowledge graph."""
    G = load_graph()
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "top_entities": []}

    top_entities = sorted(
        [(n, d.get("frequency", 0)) for n, d in G.nodes(data=True)],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(__import__("networkx").density(G), 4),
        "top_entities": [{"entity": e, "frequency": f} for e, f in top_entities],
    }
