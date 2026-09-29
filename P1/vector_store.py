"""
ChromaDB-backed vector store with full metadata filtering support.

Concepts used:
  - Bi-Encoder Retrieval   (embedding model wraps bge-small bi-encoder)
  - Dense Retrieval        (vector similarity search)
  - Vector Search          (ChromaDB ANN index)
  - Metadata Filtering     (MetadataFilters on any metadata field)
  - Vectorized Memory Store (reused in P4 for the 'memory' collection)
 
ChromaDB collections used across the project:
  'chunks'      — vectorized chunks from documents
  'chunks_summary' — LLM-generated summaries of chunks
  'memory'      — vectorized long-term memory
  'episodes'    — full session episodes
  'web_content' — browser agent web results
"""

import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
load_dotenv()
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import VECTORSTORE_DIR
CHROMA_DIR = str(VECTORSTORE_DIR)

def get_chroma_client():
    """Return a persistent ChromaDB client."""
    import chromadb
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)

def build_index(nodes, collection_name="chunks",embed_model=None):
    """Build a ChromaDB-backed vector store index. Embed nodes and upsert into a ChromaDB collection. If collection already exists, it will be reused and new nodes will be added to it."""
    from llama_index.vector_stores.chroma import ChromaVectorStore
 
    if embed_model is None:
        from P1.llm_factory import get_embedding_model
        embed_model = get_embedding_model()
 
    client = get_chroma_client()
    chroma_collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
 
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info(
        f"[VectorStore] Collection '{collection_name}': "
        f"{chroma_collection.count()} vectors stored"
    )
    return index

def load_index(collection_name="chunks",embed_model=None):
    """
    Load an existing ChromaDB collection as a VectorStoreIndex.
    Use this to query without re-indexing.
    """
    from llama_index.vector_stores.chroma import ChromaVectorStore
 
    if embed_model is None:
        from llm_factory import get_embedding_model
        embed_model = get_embedding_model()
 
    client = get_chroma_client()
    chroma_collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
 
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    logger.info(
        f"[VectorStore] Loaded collection '{collection_name}': "
        f"{chroma_collection.count()} vectors"
    )
    return index

def make_filter(field: str, value, operator: FilterOperator = FilterOperator.EQ):
    """
    Create a single metadata filter.
 
    Example:
        make_filter("doc_category", "research")
        → only return chunks where doc_category == 'research'
    """
    return MetadataFilter(key=field, value=value, operator=operator)
 
 
def make_filters(*filters: MetadataFilter, condition: str = "and") -> MetadataFilters:
    """
    Combine multiple filters with AND / OR logic.
 
    Example:
        make_filters(
            make_filter("doc_category", "research"),
            make_filter("filename", "attention_is_all_you_need.pdf"),
        )
    """
    from llama_index.core.vector_stores.types import FilterCondition
    cond = FilterCondition.AND if condition == "and" else FilterCondition.OR
    return MetadataFilters(filters=list(filters), condition=cond)
 
 
def get_retriever(
    index: VectorStoreIndex,
    similarity_top_k = 10,
    filters = None,
):
    """
    Return a vector retriever from the index, optionally with metadata filters.

    Example — retrieve only from research papers:
        f = make_filters(make_filter("doc_category", "research"))
        retriever = get_retriever(index, filters=f)
        nodes = retriever.retrieve("transformer attention mechanism")
    """
    retriever_kwargs = dict(similarity_top_k=similarity_top_k)
    if filters:
        retriever_kwargs["filters"] = filters
    return index.as_retriever(**retriever_kwargs)

def list_collections():
    """Return names of all ChromaDB collections."""
    client = get_chroma_client()
    return [col.name for col in client.list_collections()]
 
 
def collection_count(collection_name):
    """Return number of vectors in a collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(collection_name).count()
 
 
def delete_collection(collection_name):
    """Delete a ChromaDB collection (use with caution)."""
    client = get_chroma_client()
    client.delete_collection(collection_name)
    logger.warning(f"[VectorStore] Deleted collection '{collection_name}'")


'''
Quick Test: python vector_store.py ../data/Python.docx "What are Python string methods?"
Change the chunking strategy in chunk_documents() to test different approaches.
'''
if __name__ == "__main__":
    import sys
    import logging
    from document_loader import load_document
    from chunker import chunk_documents, ChunkStrategy
 
    logging.basicConfig(level=logging.INFO)
 
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <path_to_file> [query]")
        sys.exit(1)
 
    file_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "What is the main topic of this document?"
 
    # Load → chunk → index
    print(f"\n📄 Loading {file_path}...")
    docs = load_document(file_path)
    nodes = chunk_documents(docs, strategy=ChunkStrategy.PARENT_CHILD)  #Modify this line to test different chunking strategies
 
    print(f"\n🔢 Indexing {len(nodes)} chunks into ChromaDB...")
    index = build_index(nodes, collection_name="chunks_test")
 
    # Plain retrieval
    print(f"\n🔍 Querying: '{query}'\n")
    retriever = get_retriever(index, similarity_top_k=3)
    results = retriever.retrieve(query)
 
    for i, r in enumerate(results):
        print(f"--- Result {i+1} (score: {r.score:.4f}) ---")
        print(f"  Metadata: {r.node.metadata}")
        print(f"  Text: {r.node.text[:]}...\n")
 
    # Filtered retrieval demo
    print("\n🏷️  Filtered retrieval (doc_category = 'general'):")
    f = make_filters(make_filter("doc_category", "general"))
    filtered_retriever = get_retriever(index, similarity_top_k=3, filters=f)
    filtered_results = filtered_retriever.retrieve(query)
    print(f"  → {len(filtered_results)} chunks returned with filter\n")
 
    # Cleanup test collection
    delete_collection("chunks_test")
    print("✅ Smoke-test complete. Test collection deleted.")