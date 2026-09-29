"""
ingestion_pipeline.py  —  Phase 1 + P5 integration

Ties together:
    document_loader
        ↓
    structure_parser
        ↓
    chunker
        ↓
    node_store
        ↓
    BM25
        ↓
    dense vector store
        ↓
    multi-vector summary index
        ↓
    optional knowledge graph (P5)

This is the single function the rest of the project calls to ingest documents.

Usage:
    from ingestion_pipeline import ingest

    result = ingest(
        "./data/my_paper.pdf",
        strategy="sentence",
        build_graph=True,
    )
"""

import logging
import time
from pathlib import Path
from typing import Literal, Union

from P1.document_loader import load_document, load_directory
from P1.chunker import chunk_documents, ChunkStrategy
from P1.vector_store import build_index, collection_count
from P2.multi_vector import build_summary_index, SUMMARY_COLLECTION


logger = logging.getLogger(__name__)


def ingest(
    path: Union[str, Path],
    strategy: Literal["sentence", "parent_child", "semantic"] = "sentence",
    collection_name: str = "chunks",
    doc_category: str = None,
    embed_model=None,
    multi_vector_enabled: bool = True,
    summary_collection=SUMMARY_COLLECTION,
    build_graph: bool = False,
) -> dict:
    """
    Full ingestion pipeline.

    Pipeline:

        Load
          ↓
        Structure Parsing
          ↓
        Chunking
          ↓
        Node Store
          ↓
        BM25
          ↓
        Dense Vector Store
          ↓
        Multi-Vector Summary Index
          ↓
        Knowledge Graph (optional)

    Args:
        path:
            File or directory containing documents.

        strategy:
            Chunking strategy:
                - "sentence"
                - "parent_child"
                - "semantic"

        collection_name:
            ChromaDB collection for dense chunk vectors.

        doc_category:
            Optional metadata category attached during document loading.

        embed_model:
            Pre-loaded embedding model.
            Passing one avoids reloading the embedding model.

        multi_vector_enabled:
            Whether to build the summary-vector index.

        summary_collection:
            ChromaDB collection used for summary vectors.

        build_graph:
            Whether to build/update the P5 knowledge graph
            from the newly created chunks.

            False by default so normal ingestion does not
            automatically invoke the LLM for graph extraction.

    Returns:
        Dictionary containing ingestion statistics and,
        when enabled, knowledge graph statistics.
    """

    start = time.time()
    path = Path(path)

    # ============================================================
    # 1. LOAD DOCUMENTS
    # ============================================================

    logger.info(f"[Ingest] Loading: {path}")

    if path.is_dir():
        docs = load_directory(
            path,
            doc_category=doc_category,
        )
    else:
        docs = load_document(
            path,
            doc_category=doc_category,
        )

    if not docs:
        return {
            "status": "error",
            "message": "No documents loaded",
            "docs": 0,
        }

    logger.info(
        f"[Ingest] Loaded {len(docs)} document(s)"
    )

    # ============================================================
    # 2. STRUCTURE PARSING
    # ============================================================

    logger.info(
        f"[Ingest] Parsing document structure "
        f"for {len(docs)} document(s)"
    )

    from P1.structure_parser import (
        split_document_into_section_docs,
    )

    structured_docs = []

    for doc in docs:
        structured_docs.extend(
            split_document_into_section_docs(doc)
        )

    logger.info(
        f"[Ingest] Structure parser created "
        f"{len(structured_docs)} section documents"
    )

    # Debug / observability information
    print("\n===== SECTION SIZE CHECK =====\n")

    for section_doc in structured_docs[:20]:
        print(
            f"-> {len(section_doc.text.split())} words"
        )

    print(
        f"\nTotal sections: {len(structured_docs)}"
    )

    print("\n==============================\n")

    # ============================================================
    # 3. CHUNKING
    # ============================================================

    logger.info(
        f"[Ingest] Chunking {len(structured_docs)} "
        f"section document(s) "
        f"with strategy='{strategy}'"
    )

    chunk_strat = ChunkStrategy(strategy)

    nodes = chunk_documents(
        structured_docs,
        strategy=chunk_strat,
        embed_model=embed_model,
    )

    logger.info(
        f"[Ingest] Produced {len(nodes)} chunks"
    )

    # ============================================================
    # 4. PERSIST SOURCE NODES (Merge with existing nodes if present)
    # ============================================================

    from P1.node_store import load_nodes, save_nodes

    node_path = (
        f"vectorstore/{collection_name}_nodes.pkl"
    )

    all_nodes = list(nodes)
    if Path(node_path).exists():
        try:
            existing_payload = load_nodes(node_path)
            existing_nodes = existing_payload.get("nodes", [])
            new_ids = {getattr(n, "node_id", None) or getattr(n, "id_", None) for n in nodes}
            merged = [
                n for n in existing_nodes
                if (getattr(n, "node_id", None) or getattr(n, "id_", None)) not in new_ids
            ] + nodes
            all_nodes = merged
            logger.info(f"[Ingest] Merged {len(nodes)} new nodes with {len(existing_nodes)} existing nodes (total: {len(all_nodes)})")
        except Exception as e:
            logger.warning(f"[Ingest] Could not merge with existing nodes: {e}")
            all_nodes = nodes

    save_nodes(
        all_nodes,
        strategy=strategy,
        path=node_path,
    )

    logger.info(
        f"[Ingest] Saved {len(all_nodes)} total nodes "
        f"to {node_path}"
    )

    # ============================================================
    # 5. BUILD + PERSIST BM25 INDEX
    # ============================================================

    from P2.sparse_retriever import (
        build_bm25_index,
    )

    build_bm25_index(
        all_nodes,
        similarity_top_k=10,
        persist=True,
    )

    logger.info(
        f"[Ingest] BM25 index persisted over {len(all_nodes)} total nodes"
    )

    # ============================================================
    # 6. BUILD DENSE VECTOR INDEX
    # ============================================================

    logger.info(
        f"[Ingest] Indexing {len(nodes)} chunks "
        f"→ collection '{collection_name}'"
    )

    build_index(
        nodes,
        collection_name=collection_name,
        embed_model=embed_model,
    )

    logger.info(
        "[Ingest] Dense vector index built"
    )

    # ============================================================
    # 7. BUILD MULTI-VECTOR SUMMARY INDEX
    # ============================================================

    if multi_vector_enabled:

        logger.info(
            f"[Ingest] Building summary index "
            f"→ '{summary_collection}'"
        )

        build_summary_index(
            nodes,
            embed_model=embed_model,
            collection_name=summary_collection,
        )

        logger.info(
            f"[Ingest] Summary collection built: "
            f"{summary_collection}"
        )

    # ============================================================
    # 8. OPTIONAL P5 KNOWLEDGE GRAPH
    # ============================================================

    graph_built = False
    graph_stats = None

    if build_graph:

        logger.info(
            f"[Ingest] P5 graph construction enabled. "
            f"Building/updating knowledge graph from "
            f"{len(nodes)} chunks..."
        )

        try:
            from graph_store.knowledge_graph import (
                build_knowledge_graph,
            )

            graph_stats = build_knowledge_graph(
                nodes,
                max_chunks=15,
            )

            graph_built = True

            logger.info(
                f"[Ingest] Knowledge graph built: "
                f"{graph_stats}"
            )

        except Exception as e:

            # Graph construction should not invalidate
            # the already-successful document ingestion.
            logger.exception(
                f"[Ingest] Knowledge graph build failed: {e}"
            )

            graph_built = False

    # ============================================================
    # 9. FINAL STATISTICS
    # ============================================================

    elapsed = round(
        time.time() - start,
        2,
    )

    total_vectors = collection_count(
        collection_name
    )

    result = {
        "status": "success",
        "path": str(path),
        "strategy": strategy,
        "docs_loaded": len(docs),
        "section_docs": len(structured_docs),
        "chunks_created": len(nodes),
        "collection": collection_name,
        "multi_vector_enabled": multi_vector_enabled,
        "summary_collection": summary_collection,
        "total_vectors_in_collection": total_vectors,
        "graph_built": graph_built,
        "elapsed_seconds": elapsed,
    }

    if graph_stats is not None:
        result["graph_stats"] = graph_stats

    logger.info(
        f"[Ingest] Done in {elapsed}s: {result}"
    )

    return result


# ================================================================
# COMMAND-LINE TEST
# ================================================================

if __name__ == "__main__":

    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
    )

    from app_context import EMBED_MODEL

    if len(sys.argv) < 2:
        print(
            "Usage: python -m P1.ingestion_pipeline "
            "<path> [strategy]"
        )
        print(
            "strategy: sentence | parent_child | semantic"
        )
        sys.exit(1)

    path = sys.argv[1]

    strategy = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "sentence"
    )

    result = ingest(
        path,
        strategy=strategy,
        embed_model=EMBED_MODEL,
        multi_vector_enabled=True,

        # P5:
        # Set to True when you explicitly want to
        # build/update the knowledge graph.
        build_graph=True,
    )

    print("\nIngestion complete:")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


"""
Pipeline overview:

                           ┌─────────────────┐
                           │ User Documents  │
                           │ pdf/docx/txt    │
                           └────────┬────────┘
                                    │
                                    ▼
                    ┌─────────────────────────┐
                    │ document_loader.py      │
                    │                         │
                    │ load_document()         │
                    │ load_directory()        │
                    └────────┬────────────────┘
                             │
                             ▼
                    ┌─────────────────────────┐
                    │ structure_parser.py     │
                    │                         │
                    │ split_into_sections()   │
                    │ remove TOC/navigation   │
                    │ extract hierarchy       │
                    └────────┬────────────────┘
                             │
                             ▼
                    ┌─────────────────────────┐
                    │ Section Documents       │
                    │                         │
                    │ heading hierarchy       │
                    │ section_title           │
                    └────────┬────────────────┘
                             │
                             ▼
                    ┌─────────────────────────┐
                    │ chunker.py              │
                    │                         │
                    │ Sentence               │
                    │ Semantic               │
                    │ Parent-Child           │
                    └────────┬────────────────┘
                             │
                             ▼
                    ┌─────────────────────────┐
                    │ Chunk Nodes             │
                    │                         │
                    │ chunk_index             │
                    │ metadata                │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼────────────────────┐
              │              │                    │
              ▼              ▼                    ▼
       ┌────────────┐ ┌──────────────┐   ┌────────────────┐
       │ node_store │ │ BM25 Index   │   │ Dense Vector   │
       │ .pkl       │ │ sparse       │   │ ChromaDB       │
       └────────────┘ └──────────────┘   └───────┬────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │ Multi-Vector     │
                                        │ Summary Index    │
                                        └──────────────────┘

                              build_graph=True
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ knowledge_graph.py     │
                         │                        │
                         │ LLM triple extraction  │
                         │        ↓               │
                         │ Entity + Relations     │
                         │        ↓               │
                         │ NetworkX DiGraph       │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         graph_store/
                             knowledge_graph.json
"""