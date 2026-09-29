"""
Sparse Retrieval: BM25 + Keyword Search
Why BM25 matters alongside dense retrieval:
  Dense (vector) search excels at semantic similarity but can MISS exact terms.
  Example: query "BERT" may not return the chunk with "BERT" if the embedding
  space maps it near "language model" — but BM25 nails exact-match every time.
 
  Dense misses:   acronyms, version numbers (GPT-4, v2.3), proper nouns,
                  rare technical terms, typos in source docs
  BM25 misses:    paraphrases, synonyms, conceptual similarity
 
  Together they cover each other's blind spots.
"""
import logging
import sys
from pathlib import Path
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import VECTORSTORE_DIR
BM25_DIR = VECTORSTORE_DIR / "bm25"

def build_bm25_index(
    nodes,
    similarity_top_k=10,
    persist=True,
):
    '''
    Build a BM25 retriever from a list of nodes.
    '''
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    stemmer = Stemmer.Stemmer("english")

    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=stemmer,
        language="english",
    )

    if persist:
        BM25_DIR.mkdir(parents=True, exist_ok=True)

        retriever.persist(str(BM25_DIR))

        logger.info(
            f"[BM25] Persisted index to {BM25_DIR}"
        )

    logger.info(
        f"[BM25] Built index over {len(nodes)} nodes"
    )

    return retriever
def load_bm25_retriever(
    nodes,
    similarity_top_k=10,
):
    '''
    Load a persisted BM25 retriever.
    '''
    from llama_index.retrievers.bm25 import BM25Retriever

    if BM25_DIR.exists():
        try:
            retriever = BM25Retriever.from_persist_dir(
                str(BM25_DIR)
            )

            retriever.similarity_top_k = similarity_top_k

            logger.info(
                f"[BM25] Loaded from {BM25_DIR}"
            )

            return retriever

        except Exception as e:
            logger.warning(
                f"[BM25] Failed to load persisted index: {e}"
            )

    logger.info(
        "[BM25] Rebuilding index"
    )

    return build_bm25_index(
        nodes,
        similarity_top_k=similarity_top_k,
    )

def bm25_retrieve(retriever, query):
    """
    Retrieve relevant nodes using BM25 retriever.
    Returns list of NodeWithScore.
    """
    results = retriever.retrieve(query)

    results = [r for r in results if r.score > 0]

    logger.info(
        f"[BM25] Retrieved {len(results)} nodes for query: '{query}'"
    )
    return results


# Quick Test
if __name__ == "__main__":
    import sys
    import logging
    sys.path.insert(0, ".")
    from P1.document_loader import load_document
    from P1.chunker import chunk_documents, ChunkStrategy
 
    logging.basicConfig(level=logging.INFO)
 
    if len(sys.argv) < 3:
        print("Usage: python sparse_retriever.py <file_path> <query>")
        sys.exit(1)
 
    docs = load_document(sys.argv[1])
    nodes = chunk_documents(docs, strategy=ChunkStrategy.SENTENCE)
    retriever = build_bm25_index(nodes, similarity_top_k=5)
 
    query = sys.argv[2]
    results = bm25_retrieve(retriever, query)
 
    print(f"\n🔍 BM25 results for: '{query}'\n")
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r.score:.4f} | {r.node.text[:150]}...")