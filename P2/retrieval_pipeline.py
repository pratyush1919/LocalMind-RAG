"""
The single function Phase 3 agents call to go from query → top-5 reranked chunks.

Ties together:
  sparse_retriever.py    BM25
  hybrid_retriever.py    Ensemble + RRF
  query_expander.py      HyDE + Multi-Query + Planning
  multi_vector.py        Raw + Summary vectors
  reranker.py            6-stage pipeline → top-5
"""

import logging
from typing import List, Optional
from llama_index.core.schema import NodeWithScore
logger = logging.getLogger(__name__)

class RetrievalContext:
    """
    Holds all heavy retrieval objects.

    Startup:
        load nodes.pkl
        load chroma indexes
        load bm25

    Query:
        retrieve(...)
    """

    def __init__(
        self,
        nodes,
        vector_index,
        embed_model=None,
        llm=None,
        authority_docs=None,
    ):
        from P1.vector_store import get_retriever, load_index
        from P2.sparse_retriever import load_bm25_retriever
        from P2.hybrid_retriever import HybridRetriever
        from P2.multi_vector import MultiVectorRetriever

        self.nodes = nodes
        self.vector_index = vector_index

        self.embed_model = embed_model or _default_embed()
        self.llm = llm or _default_llm()

        self.authority_docs = authority_docs or []

        # --------------------------------------------------
        # Dense Retriever
        # --------------------------------------------------

        dense_retriever = get_retriever(
            vector_index,
            similarity_top_k=10,
        )

        # --------------------------------------------------
        # BM25 Retriever
        # --------------------------------------------------

        bm25_retriever = load_bm25_retriever(
            nodes=nodes,
            similarity_top_k=10,
        )

        # --------------------------------------------------
        # Optional Web Retriever
        # --------------------------------------------------

        web_retriever = None

        try:
            web_index = load_index(
                collection_name="web_content",
                embed_model=self.embed_model,
            )

            web_retriever = get_retriever(
                web_index,
                similarity_top_k=10,
            )

            logger.info(
                "[RetrievalContext] Web retriever loaded"
            )

        except Exception:
            logger.info(
                "[RetrievalContext] No web content found"
            )

        # --------------------------------------------------
        # Hybrid Retriever
        # --------------------------------------------------

        self.hybrid = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            web_retriever=web_retriever,
            similarity_top_k=10,
        )

        # --------------------------------------------------
        # Multi Vector Retriever
        # --------------------------------------------------

        self.multi_vector = MultiVectorRetriever(
            raw_retriever=self.hybrid,
            embed_model=self.embed_model,
            top_k=10,
        )

        logger.info(
            "[RetrievalContext] Ready"
        )

def _default_embed():
    from llm_factory import get_embedding_model
    return get_embedding_model()


def _default_llm():
    from llm_factory import get_llama_llm
    return get_llama_llm(temperature=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    ctx: RetrievalContext,
    use_hyde: bool = True,
    use_multi_query: bool = True,
    use_planning: bool = True,
    use_multi_vector: bool = True,
    doc_category_filter: Optional[str] = None,
) -> dict:

    from P2.query_expander import expanded_retrieve
    from P2.reranker import rerank_pipeline
    from P2.hybrid_retriever import reciprocal_rank_fusion
    import time
    start = time.perf_counter()
    # Step 1: Query expansion → up to 20 candidates
    expansion = expanded_retrieve(
        query=query,
        hybrid_retriever=ctx.hybrid,
        vector_index=ctx.vector_index,
        embed_model=ctx.embed_model,
        llm=ctx.llm,
        use_hyde=use_hyde,
        use_multi_query=use_multi_query,
        use_planning=use_planning,
    )
    strategy = expansion["strategy"]
    candidates: List[NodeWithScore] = expansion["results"]

    if strategy == "memory_only":
        return {
            "nodes": [],
            "strategy": strategy,
            "hyde_used": False,
            "paraphrases": [],
            "sources": [],
        }

    # Step 2: Multi-vector — add summary-vector matches
    mv_results = []

    if use_multi_vector:
        mv_results = ctx.multi_vector.retrieve(query)

    if mv_results:
        candidates = reciprocal_rank_fusion(
            [candidates, mv_results],
            top_k=20,
        )

    # Apply doc_category filter if requested (post-retrieval)
    if doc_category_filter:
        candidates = [
            n for n in candidates
            if n.node.metadata.get("doc_category") == doc_category_filter
        ]

    print("\n=== CANDIDATES BEFORE RERANK ===\n")

    for i, n in enumerate(candidates[:10]):
        print(
            f"\n[{i}] "
            f"{n.node.metadata.get('filename')} "
            f"{n.node.metadata.get('chunk_index')} "
            f"score={n.score}"
        )

        print(n.node.text[:300])
        print("-" * 80)

    # Step 3: 6-stage reranking → top-5
    final_nodes = rerank_pipeline(
        query=query,
        nodes=candidates,
        embed_model=ctx.embed_model,
        llm=ctx.llm,
        authority_docs=ctx.authority_docs,
    )

    sources = list({n.node.metadata.get("filename", "unknown") for n in final_nodes})

    logger.info(
        f"[Retrieval] query='{query[:60]}' | strategy={strategy} | "
        f"sources={sources} | chunks={len(final_nodes)}"
    )

    elapsed = time.perf_counter() - start
    logger.info(
        f"[Retrieval] "
        f"strategy={strategy} "
        f"chunks={len(final_nodes)} "
        f"time={elapsed:.2f}s"
    )
    return {
        "nodes": final_nodes,
        "strategy": strategy,
        "hyde_used": expansion["hyde_used"],
        "paraphrases": expansion["paraphrases"],
        "sources": sources,
    }


if __name__ == "__main__":
    import time
    import logging

    from app_context import EMBED_MODEL, LLM

    from P1.node_store import load_nodes
    from P1.vector_store import load_index

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print("STEP 1: Load Persisted Nodes")
    print("=" * 80)

    payload = load_nodes(
        "vectorstore/chunks_nodes.pkl"
    )

    nodes = payload["nodes"]

    print(f"Loaded {len(nodes)} chunks")

    print("\n" + "=" * 80)
    print("STEP 2: Load Existing Vector Index")
    print("=" * 80)

    index = load_index(
        collection_name="chunks",
        embed_model=EMBED_MODEL,
    )

    print("Vector index loaded")

    print("\n" + "=" * 80)
    print("STEP 3: Create Retrieval Context")
    print("=" * 80)

    ctx = RetrievalContext(
        nodes=nodes,
        vector_index=index,
        embed_model=EMBED_MODEL,
        llm=LLM,
    )

    print("Context ready")

    print("\n" + "=" * 80)
    print("STEP 4: Run Retrieval Pipeline")
    print("=" * 80)

    query = "How does Unicode work in Python?"
    start = time.perf_counter()

    result = retrieve(
        query=query,
        ctx=ctx,
        use_hyde=True,
        use_multi_query=True,
        use_planning=True,
        use_multi_vector=True,
    )

    elapsed = time.perf_counter() - start

    print(f"\nQuery: {query}")
    print(f"Strategy: {result['strategy']}")
    print(f"Sources: {result['sources']}")
    print(f"Paraphrases: {len(result['paraphrases'])}")

    print(
        f"\n⏱ Total Retrieval Time: "
        f"{elapsed:.2f} seconds"
    )

    print("\nTop Results:\n")

    for i, node in enumerate(result["nodes"], start=1):

        print(f"[{i}] Score = {node.score:.6f}")

        print(
            f"File = "
            f"{node.node.metadata.get('filename')}"
        )

        print(
            f"Chunk = "
            f"{node.node.metadata.get('chunk_index')}"
        )

        print(node.node.text[:250])

        print("-" * 80)

    print("\n✅ Retrieval pipeline smoke test passed")

'''
                           ┌──────────────────┐
                           │ User Query       │
                           └────────┬─────────┘
                                    │
                                    ▼
                    ┌─────────────────────────┐
                    │ query_expander.py       │
                    │                         │
                    │ Query Planning          │
                    │ HyDE                    │
                    │ Multi Query             │
                    └────────┬────────────────┘
                             │
                             ▼
                    ┌─────────────────────────┐
                    │ Expanded Queries        │
                    └────────┬────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼

┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ BM25           │  │ Dense Vector   │  │ Web Retriever  │
│ sparse index   │  │ chunks         │  │ optional       │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │
        └──────────┬────────┴──────────┬────────┘
                   │                   │
                   ▼                   ▼

           ┌──────────────────────────────┐
           │ hybrid_retriever.py          │
           │                              │
           │ Reciprocal Rank Fusion       │
           │ BM25 + Dense + Web           │
           └──────────────┬───────────────┘
                          │
                          ▼

             ┌────────────────────────┐
             │ Hybrid Candidates      │
             └──────────┬─────────────┘
                        │
                        ▼

          ┌──────────────────────────────┐
          │ multi_vector.py              │
          │                              │
          │ chunks                       │
          │ chunks_summary               │
          │ RRF Fusion                   │
          └──────────────┬───────────────┘
                         │
                         ▼

             ┌────────────────────────┐
             │ 20 Candidate Chunks    │
             └──────────┬─────────────┘
                        │
                        ▼

           ┌────────────────────────────┐
           │ reranker.py                │
           │                            │
           │ Stage 1 Cross Encoder      │
           │ Stage 2 Intent Aware       │
           │ Stage 3 MMR Diversity      │
           │ Stage 4 Metadata Boost     │
           └─────────────┬──────────────┘
                         │
                         ▼

              ┌──────────────────────┐
              │ Top 5 Chunks         │
              │ + Scores             │
              │ + Sources            │
              └──────────────────────┘

RetrievalContext
│
├── nodes.pkl
│
├── BM25 Retriever
│
├── Dense Retriever
│
├── Hybrid Retriever
│
├── MultiVector Retriever
│
├── Embedding Model
│
└── LLM
'''