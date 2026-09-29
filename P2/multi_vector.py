"""
Store two vector representations per chunk and search both at retrieval time.
Representations:
  1. Raw text embedding   → ChromaDB 'chunks' collection 
  2. Summary embedding    → ChromaDB 'chunks_summary' collection

Why two vectors per chunk?
  Raw text vectors excel at detail queries: "What is the dropout rate used in the paper?"
  Summary vectors excel at abstract/high-level queries: "What is this paper about?"
  A query like "overview of the training procedure" may rank poorly against raw
  chunks full of implementation details — but ranks highly against a concise LLM
  summary of those chunks.

The framework supports multiple summary generation strategies.
For local execution I used lightweight truncation because
LLM summarization added substantial indexing latency.
"""
import logging
from urllib import response
logger = logging.getLogger(__name__)
SUMMARY_COLLECTION = "chunks_summary"

class MultiVectorRetriever:
    """
    Search both raw chunk vectors and summary vectors,
    then fuse results with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        raw_retriever,
        embed_model=None,
        top_k=10,
    ):
        from P1.vector_store import load_index, get_retriever

        if embed_model is None:
            from P1.llm_factory import get_embedding_model
            embed_model = get_embedding_model()

        self.raw_retriever = raw_retriever
        self.top_k = top_k

        summary_index = load_index(
            collection_name=SUMMARY_COLLECTION,
            embed_model=embed_model,
        )

        self.summary_retriever = get_retriever(
            summary_index,
            similarity_top_k=top_k,
        )

        logger.info(
            f"[MultiVec] Summary retriever loaded from '{SUMMARY_COLLECTION}'"
        )

    def retrieve(self, query):
        from P2.hybrid_retriever import reciprocal_rank_fusion

        result_lists = []

        # Raw vectors
        try:
            raw_results = self.raw_retriever.retrieve(query)
            result_lists.append(raw_results)
        except Exception as e:
            logger.warning(f"[MultiVec] Raw retrieval failed: {e}")

        # Summary vectors
        try:
            summary_results = self.summary_retriever.retrieve(query)
            result_lists.append(summary_results)
        except Exception as e:
            logger.warning(f"[MultiVec] Summary retrieval failed: {e}")

        if not result_lists:
            return []

        fused = reciprocal_rank_fusion(
            result_lists,
            top_k=self.top_k,
        )

        logger.info(
            f"[MultiVec] Fused {len(result_lists)} vector types → {len(fused)} results"
        )

        return fused

def build_multi_vector_retriever(
    raw_retriever,
    embed_model=None,
    top_k=10,
):
    """
    Factory function for MultiVectorRetriever.

    Usage:
        mv = build_multi_vector_retriever(raw_retriever)
        results = mv.retrieve("What is Python?")
    """

    if embed_model is None:
        from P1.llm_factory import get_embedding_model
        embed_model = get_embedding_model()

    return MultiVectorRetriever(
        raw_retriever=raw_retriever,
        embed_model=embed_model,
        top_k=top_k,
    )


def build_summary_index(nodes, embed_model=None,collection_name=SUMMARY_COLLECTION):
    '''
    Build a summary vector index for the given nodes.
    Each node gets a summary vector (currently just a truncated embedding).
    '''  
    def extract_keywords(text: str, max_keywords: int = 40) -> str:
        """
        Convert a chunk into a compact keyword representation.

        Example:
            Input:
                'Strings are immutable sequences. Methods: upper lower split'

            Output:
                'strings immutable sequences methods upper lower split'
        """
        import regex as re
        STOPWORDS = {
        "the", "and", "for", "with", "from", "that", "this",
        "into", "their", "there", "have", "been", "will",
        "would", "about", "which", "what", "when", "where",
        "while", "using", "used", "they", "them", "then",
        "than", "also", "such", "your", "you", "are", "was",
        "were", "has", "had", "not", "can", "could"
        } 
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]+\b", text.lower())

        keywords = []

        seen = set()

        for word in words:
            if word in STOPWORDS:
                continue

            if len(word) < 3:
                continue

            if word not in seen:
                keywords.append(word)
                seen.add(word)

            if len(keywords) >= max_keywords:
                break

        return " ".join(keywords)
    
    from P1.vector_store import build_index
    from llama_index.core.schema import TextNode

    if embed_model is None:
        from P1.llm_factory import get_embedding_model
        embed_model = get_embedding_model()

    summary_nodes = []

    for node in nodes:
        keyword_text = " ".join(
            filter(
                None,
                [
                    node.metadata.get("filename"),
                    node.metadata.get("heading_level_1"),
                    node.metadata.get("heading_level_2"),
                    node.metadata.get("heading_level_3"),
                    extract_keywords(node.text),
                ]
            )
        )

        summary_node = TextNode(
            text=keyword_text,
            node_id=node.node_id,
            metadata={
                **node.metadata,
                "vector_type": "summary",
                "summary_strategy": "keywords",
            },
        )
        summary_nodes.append(summary_node)

    build_index(
        summary_nodes,
        collection_name=collection_name,
        embed_model=embed_model,
    )

    logger.info(
        f"[MultiVec] Summary index built: {len(summary_nodes)} summary vectors"
    )

    return summary_nodes

# Quick Test: In main folder, run: python -m P2.multi_vector
if __name__ == "__main__":
    import logging
    from app_context import LLM, EMBED_MODEL
    from P1.document_loader import load_document
    from P1.chunker import chunk_documents, ChunkStrategy
    from P1.vector_store import build_index, get_retriever
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print("STEP 1: Load Document")
    print("=" * 80)

    docs = load_document("data/Python.docx")
    print(f"Loaded {len(docs)} document(s)")

    print("\n" + "=" * 80)
    print("STEP 2: Chunk Document")
    print("=" * 80)

    nodes = chunk_documents(
        docs,
        strategy=ChunkStrategy.SENTENCE,
    )

    print(f"Created {len(nodes)} chunks")

    print("\n" + "=" * 80)
    print("STEP 3: Build Raw Vector Index")
    print("=" * 80)

    embed_model = EMBED_MODEL

    raw_index = build_index(
        nodes,
        collection_name="multivector_test_raw",
        embed_model=embed_model,
    )

    raw_retriever = get_retriever(
        raw_index,
        similarity_top_k=5,
    )

    print("\n" + "=" * 80)
    print("STEP 4: Build Summary Index")
    print("=" * 80)

    build_summary_index(
        nodes,
        embed_model=EMBED_MODEL,
    )

    print("\n" + "=" * 80)
    print("STEP 5: Build MultiVectorRetriever")
    print("=" * 80)

    mv_retriever = MultiVectorRetriever(
        raw_retriever=raw_retriever,
        embed_model=EMBED_MODEL,
        top_k=5,
    )

    print("Retriever created successfully")

    print("\n" + "=" * 80)
    print("STEP 6: Query")
    print("=" * 80)

    query = "What string functions are available in Python?"

    results = mv_retriever.retrieve(query)
    print(results[0].node.node_id)
    print(results[2].node.node_id)
    print(f"\nQuery: {query}")
    print(f"Returned {len(results)} results\n")

    for i, r in enumerate(results, start=1):
        print(f"[{i}] Score = {r.score:.6f}")
        print(f"Metadata = {r.node.metadata}")
        print(f"Text = {r.node.text[:250]}")
        print("-" * 80)

    from P1.vector_store import delete_collection
    delete_collection("multivector_test_raw")
    delete_collection("chunks_summary")
    print("\n✅ MultiVectorRetriever smoke test passed")