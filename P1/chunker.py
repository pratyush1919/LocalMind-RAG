"""
Three chunking strategies combined.
Concepts covered:
  - Semantic Chunk Retrieval      (SemanticSplitterNodeParser)
  - Parent-Child Retrieval        (HierarchicalNodeParser)
  - Hierarchical Retrieval        (HierarchicalNodeParser)
  - Metadata Filtering            (metadata tags on every chunk)
"""

import logging
from enum import Enum
from llama_index.core import Document
logger = logging.getLogger(__name__)

class ChunkStrategy(str,Enum):
    """Chunking strategies for document splitting and retrieval."""
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"
    SENTENCE = "sentence"

def stamp_metadata(nodes, strategy):
    """Attach chunking strategy and chunk_index to each node's metadata for later filtering and retrieval."""
    for i, node in enumerate(nodes):
        node.metadata["chunking_strategy"] = strategy
        node.metadata["chunk_index"] = i
        if "doc_category" not in node.metadata:
            node.metadata["doc_category"] = "general"

def chunk_sentence(documents,chunk_size=512,chunk_overlap=64):
    """Chunk a document into sentences."""
    from llama_index.core.node_parser import SentenceSplitter
 
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    stamp_metadata(nodes, strategy="sentence")
    logger.info(f"[Sentence] {len(documents)} docs → {len(nodes)} chunks")
    return nodes

def chunk_parent_child(documents,parent_chunk_size=1024,child_chunk_size=256):
    """
    Create two-level hierarchical chunks.
 
    Parent nodes (1024 tokens) give broad context.
    Child nodes (256 tokens) are what gets retrieved — small = precise match.
    The parent is attached to each child via relationships.
 
    At retrieval time:
      1. Search over child nodes (precise match)
      2. Fetch parent node (broad context for LLM)

    Returns dict with 'all_nodes', 'leaf_nodes', 'parent_nodes' for
    flexible use downstream.
    """
    from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
 
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[parent_chunk_size, child_chunk_size],
    )
    all_nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    leaf_nodes = get_leaf_nodes(all_nodes)
 
    stamp_metadata(all_nodes, strategy="parent_child")
    logger.info(
        f"[Parent-Child] {len(documents)} docs → "
        f"{len(all_nodes)} total nodes ({len(leaf_nodes)} leaf / "
        f"{len(all_nodes)-len(leaf_nodes)} parent)"
    )
    return {
        "all_nodes": all_nodes,
        "leaf_nodes": leaf_nodes,
        "parent_nodes": [n for n in all_nodes if n not in leaf_nodes],
    }

def chunk_semantic(documents,embed_model=None,buffer_size = 1,breakpoint_percentile = 95):
    """
    Split documents at natural topic boundaries using embedding similarity.
 
    How it works:
      - Embed every sentence.
      - Compute cosine similarity between adjacent sentence embeddings.
      - When similarity drops sharply (below breakpoint_percentile), create a new chunk.
      - Chunks are semantically coherent — they cover one topic.
 
    Slower than sentence chunking (requires embeddings) but produces
    higher-quality chunks for complex documents.
 
    """
    from llama_index.core.node_parser import SemanticSplitterNodeParser
 
    if embed_model is None:
        from llm_factory import get_embedding_model
        embed_model = get_embedding_model()
 
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    stamp_metadata(nodes, strategy="semantic")
    logger.info(f"[Semantic] {len(documents)} docs → {len(nodes)} chunks")
    return nodes

def chunk_documents(documents,strategy = ChunkStrategy.SENTENCE,embed_model=None):
    """
    Chunk documents using the specified strategy.
    Returns a flat list of nodes (for PARENT_CHILD, returns leaf nodes).

    Returns:
        List[BaseNode] ready for embedding and vector store ingestion.
    """
    if strategy == ChunkStrategy.SENTENCE:
        return chunk_sentence(documents)
 
    elif strategy == ChunkStrategy.PARENT_CHILD:
        result = chunk_parent_child(documents)
        # Return leaf nodes for indexing; parent nodes stored separately if needed
        return result["leaf_nodes"]
 
    elif strategy == ChunkStrategy.SEMANTIC:
        return chunk_semantic(documents, embed_model=embed_model)
 
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


'''
Quick Test:
python chunker.py ..\data\Python.docx sentence
python chunker.py ..\data\Python.docx parent_child
python chunker.py ..\data\Python.docx semantic
'''
if __name__ == "__main__":
    import sys
    import logging
    from document_loader import load_document
 
    logging.basicConfig(level=logging.INFO)
 
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_file> [sentence|parent_child|semantic]")
        sys.exit(1)
 
    file_path = sys.argv[1]
    strategy_str = sys.argv[2] if len(sys.argv) > 2 else "sentence"
    strategy = ChunkStrategy(strategy_str)
 
    docs = load_document(file_path)
    nodes = chunk_documents(docs, strategy=strategy)
 
    print(f"\nStrategy: {strategy.value}  |  {len(nodes)} chunks produced\n") 
    for i, node in enumerate(nodes[:5]): # Show first 5 chunks as a sample
        print(f"--- Chunk {i+1} ---")
        print(f"  Metadata : {node.metadata}")
        print(f"  Tokens   : ~{len(node.text.split())} words")
        print(f"  Preview  : {node.text[:200].replace(chr(10), ' ')}...")
        print()
 