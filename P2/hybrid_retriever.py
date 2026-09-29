"""
3-source ensemble retriever using Reciprocal Rank Fusion (RRF).
 
Sources:
  1. Dense vector retriever    (ChromaDB 'chunks' collection)
  2. BM25 sparse retriever     (keyword / exact-match)
  3. Web content retriever     (ChromaDB 'web_content' — populated by web scraper)
 
RRF formula:
  score(d) = Σ_r  1 / (k + rank_r(d))
  where k=60 is a constant that dampens the impact of very high ranks.

Why RRF over score averaging?
  Different retrievers produce scores on incompatible scales.
  BM25 scores are unbounded; cosine similarity is [-1, 1].
  RRF uses only RANK positions — scale-invariant and empirically strong.
"""

import logging
logger = logging.getLogger(__name__)

# def reciprocal_rank_fusion(result_lists, k = 60, top_k=10):
#     """
#     Merge multiple ranked result lists using Reciprocal Rank Fusion.
#     """
#     scores = {}
#     nodes = {}
#     for result in result_lists:
#         for rank, node_with_score in enumerate(result, start=1):
#             node_id = node_with_score.node.node_id
#             scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank)
#             if node_id not in nodes or (node_with_score.score or 0) > (nodes[node_id].score or 0):
#                 nodes[node_id] = node_with_score
                
#     # Sort by RRF score descending
#     ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
 
#     fused = []
#     for node_id in ranked_ids:
#         nws = nodes[node_id]
#         nws.score = scores[node_id]   # overwrite with RRF score for transparency
#         fused.append(nws)
 
#     logger.debug(f"[RRF] {sum(len(r) for r in result_lists)} candidates → {len(fused)} fused")
#     return fused

def reciprocal_rank_fusion(result_lists, k=60, top_k=10):
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    Deduplicates documents using metadata identity rather than node_id.
    """

    scores = {}
    nodes = {}

    for result in result_lists:
        for rank, node_with_score in enumerate(result, start=1):

            meta = node_with_score.node.metadata

            # Stable document identity
            doc_id = (
                f"{meta.get('filename','unknown')}::"
                f"{meta.get('chunk_index','unknown')}"
            )

            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

            # Keep the highest-scoring version
            if (
                doc_id not in nodes
                or (node_with_score.score or 0)
                > (nodes[doc_id].score or 0)
            ):
                nodes[doc_id] = node_with_score

    ranked_ids = sorted(
        scores,
        key=scores.__getitem__,
        reverse=True,
    )[:top_k]

    fused = []

    for doc_id in ranked_ids:
        nws = nodes[doc_id]
        nws.score = scores[doc_id]
        fused.append(nws)

    logger.debug(
        f"[RRF] {sum(len(r) for r in result_lists)} candidates → {len(fused)} fused"
    )

    return fused

class HybridRetriever:
    """
    3-source ensemble retriever: dense + BM25 + web_content, fused via RRF.
    This is the central retrieval component for the entire project.
    All downstream phases (agents, reranker, eval) consume results from here.
    """

    def __init__(
        self,
        dense_retriever,           # from vector_store.get_retriever()
        bm25_retriever,            # from sparse_retriever.build_bm25_index()
        web_retriever=None,       
        similarity_top_k: int = 10,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.web_retriever = web_retriever
        self.similarity_top_k = similarity_top_k
        self.rrf_k = rrf_k
 
    def retrieve(self, query):
        """
        Run all retrievers in parallel, fuse with RRF, return top-k results.
        Flow:
          1. Dense search   → semantic similarity over bge-small vectors
          2. BM25 search    → keyword / term-frequency matching
          3. Web search     → cross-source (web_content collection, if populated)
          4. RRF fusion     → merge by rank position
        """
        result_lists = []
 
        # Source 1: Dense retrieval
        try:
            dense_results = self.dense_retriever.retrieve(query)
            result_lists.append(dense_results)
            logger.debug(f"[Hybrid] Dense: {len(dense_results)} results")
        except Exception as e:
            logger.warning(f"[Hybrid] Dense retriever failed: {e}")
 
        # Source 2: BM25 sparse retrieval
        try:
            bm25_results = self.bm25_retriever.retrieve(query)
            result_lists.append(bm25_results)
            logger.debug(f"[Hybrid] BM25: {len(bm25_results)} results")
        except Exception as e:
            logger.warning(f"[Hybrid] BM25 retriever failed: {e}")
 
        # Source 3: Web content (Cross-Source Retrieval — populated by Browser Agent P11)
        if self.web_retriever is not None:
            try:
                web_results = self.web_retriever.retrieve(query)
                result_lists.append(web_results)
                logger.debug(f"[Hybrid] Web: {len(web_results)} results")
            except Exception as e:
                logger.debug(f"[Hybrid] Web retriever unavailable: {e}")
 
        if not result_lists:
            logger.error("[Hybrid] All retrievers failed")
            return []
 
        fused = reciprocal_rank_fusion(
            result_lists,
            k=self.rrf_k,
            top_k=self.similarity_top_k,
        )
        logger.info(f"[Hybrid] Query fused → {len(fused)} results from {len(result_lists)} sources")
        return fused
 
    def retrieve_with_metadata_filter(self,query,doc_category= None,filename= None):
        """
        Retrieve with optional metadata scoping.
        BM25 results are post-filtered (BM25 has no native metadata filtering).
        Dense retriever uses ChromaDB's native MetadataFilters.
        """
        from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterCondition
 
        filters = []
        if doc_category:
            filters.append(MetadataFilter(key="doc_category", value=doc_category))
        if filename:
            filters.append(MetadataFilter(key="filename", value=filename))
 
        # Apply filters to dense retriever on-the-fly
        if filters:
            metadata_filters = MetadataFilters(
                filters=filters,
                condition=FilterCondition.AND,
            )
            self.dense_retriever.filters = metadata_filters
 
        results = self.retrieve(query)
 
        # Reset filter
        if filters:
            self.dense_retriever.filters = None
 
        # Post-filter BM25 results (they're already in the fused list)
        if doc_category or filename:
            results = [
                r for r in results
                if (not doc_category or r.node.metadata.get("doc_category") == doc_category)
                and (not filename or r.node.metadata.get("filename") == filename)
            ]
 
        return results

def build_hybrid_retriever(nodes,vector_index,similarity_top_k=10,embed_model=None):
    """
    Convenience factory: build all 3 retrievers and return a HybridRetriever.
    Returns:
        HybridRetriever ready for .retrieve(query)
    """
    from P1.vector_store import get_retriever, load_index
    from P2.sparse_retriever import build_bm25_index
 
    # Dense retriever (main document chunks)
    dense = get_retriever(vector_index, similarity_top_k=similarity_top_k)
 
    # BM25 sparse retriever
    bm25 = build_bm25_index(nodes, similarity_top_k=similarity_top_k)
 
    # Web content retriever
    web_retriever = None
    try:
        web_index = load_index(collection_name="web_content", embed_model=embed_model)
        web_retriever = get_retriever(web_index, similarity_top_k=similarity_top_k)
        logger.info("[Hybrid] Web content retriever loaded (cross-source enabled)")
    except Exception:
        logger.info("[Hybrid] Web content collection empty — cross-source disabled")
 
    return HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        web_retriever=web_retriever,
        similarity_top_k=similarity_top_k,
    )

#Quick Test: In main folder, run: python -m P2.hybrid_retriever data\Python.docx Encapsulation
if __name__ == "__main__":
    import logging
    import sys
    from app_context import LLM, EMBED_MODEL
    from P1.document_loader import load_document
    from P1.chunker import chunk_documents, ChunkStrategy
    from P1.vector_store import build_index
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage:")
        print("python hybrid_retriever.py <file_path> <query>")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2]

    print(f"\n📄 Loading: {file_path}")

    docs = load_document(file_path)

    nodes = chunk_documents(
        docs,
        strategy=ChunkStrategy.SENTENCE
    )

    print(f"✂️ Chunks created: {len(nodes)}")

    embed_model = EMBED_MODEL

    print("🔢 Building vector index...")
    index = build_index(
        nodes,
        collection_name="hybrid_test",
        embed_model=embed_model
    )

    print("🔍 Building hybrid retriever...")
    hybrid = build_hybrid_retriever(
        nodes,
        index,
        similarity_top_k=5,
        embed_model=embed_model
    )

    print(f"\n❓ Query: {query}\n")

    results = hybrid.retrieve(query)

    for i, r in enumerate(results, start=1):
        print(f"[{i}] RRF Score = {r.score:.6f}")
        print(f"Metadata: {r.node.metadata}")
        print(f"Text: {r.node.text[:200]}")
        print("-" * 80)  