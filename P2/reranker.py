"""
6-stage reranking pipeline that progressively refines top-20 candidates to top-5.
Stage flow:
  Input: top-20 from hybrid retriever + query expansion
  │
  ├─ Stage 1: Cross-Encoder Rerank (FlashRank)  → top-10
  │
  │
  ├─ Stage 2: Intent-Aware Rerank               → top-8
  │
  │
  ├─ Stage 3: Diversity-Aware Rerank (MMR)       → top-6
  │
  │
  ├─ Stage 4: Metadata-Aware Rerank              → top-5
  │
  │
  ├─ Stage 4b: Personalized Rerank (user profile)→ top-5 reordered
  │
  │
  Output: top-5 context chunks → Generator Agent
 
Why so many stages?
  Each stage catches failures the previous missed:
  - Cross-encoder: better relevance than bi-encoder but ignores intent
  - Intent filter: ensures answer format matches query type
  - MMR: prevents all 5 chunks being from the same 2-page section
  - Metadata: freshness + authority matter for trust
  - Personalized: user history signals what format/depth they prefer
"""
import json
import sys
import logging
import math
from pathlib import Path
from typing import List, Literal, Optional
from llama_index.core.schema import NodeWithScore
logger = logging.getLogger(__name__)
QueryIntent = Literal["factual", "analytical", "comparative", "procedural", "unknown"]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import USER_PROFILE_PATH
USER_Path = USER_PROFILE_PATH

_FLASHRANK = None
def get_flashrank():
    global _FLASHRANK

    if _FLASHRANK is None:

        from flashrank import Ranker

        _FLASHRANK = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir="./flashrank_cache",
        )

        logger.info("[Rerank] FlashRank loaded")

    return _FLASHRANK

def cross_encoder_rerank(query: str, nodes: List[NodeWithScore],top_k = 10) -> List[NodeWithScore]:
    '''
    Cross-encoder (this stage): processes query+doc TOGETHER → sees interactions → more precise.
    FlashRank runs the cross-encoder on CPU in ~50-200ms for 20 candidates.
    '''
    if not nodes:
        return nodes
 
    try:
        from flashrank import RerankRequest
        ranker = get_flashrank()
        passages = [
            {"id": i, "text": n.node.text, "meta": n.node.metadata}
            for i, n in enumerate(nodes)
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = ranker.rerank(rerank_request)
 
        # Map scores back to NodeWithScore list
        id_to_node = {i: n for i, n in enumerate(nodes)}
        result = []
        for item in reranked[:top_k]:
            nws = id_to_node[item["id"]]
            nws.score = float(item["score"])
            result.append(nws)
 
        logger.info(f"[Rerank-S1] Cross-encoder: {len(nodes)} → {len(result)}")
        return result
 
    except ImportError:
        logger.warning("[Rerank-S1] FlashRank not installed — skipping cross-encoder stage")
        return nodes
    except Exception as e:
        logger.warning(f"[Rerank-S1] Cross-encoder failed: {e} — passing through")
        return nodes
    

def classify_query_intent(query:str , llm=None) -> QueryIntent:
    """
    Classify the query intent to guide intent-aware reranking.
    Intents:
      factual     — "What is X?" "When was Y?" → prefer chunks with definitions/facts
      analytical  — "Why does X happen?" "How does Y work?" → prefer explanatory chunks
      comparative — "X vs Y" "difference between" → prefer chunks covering both topics
      procedural  — "How to X" "Steps to Y" → prefer ordered/list chunks
    """
    query_lower = query.lower().strip()
    # Heuristic rules for simple cases
    if query_lower.startswith(("what", "when", "who", "where")):
        return "factual"

    if query_lower.startswith(("how", "steps", "tutorial")):
        return "procedural"

    if " vs " in query_lower:
        return "comparative"

    if "difference between" in query_lower:
        return "comparative"

    if query_lower.startswith(("why", "explain", "analyze", "analyse")):
        return "analytical"
    
    # Fallback to LLM
    if llm is None:
        from P1.llm_factory import get_llama_llm
        llm = get_llama_llm(temperature=0.0)

    prompt = (
        "Classify the following query intent as one of: "
        "factual, analytical, comparative, procedural, unknown\n\n"
        f"Query: {query}\n\nIntent (one word):"
    )
    try:
        intent = str(llm.complete(prompt)).strip().lower()
        valid = {"factual", "analytical", "comparative", "procedural"}
        return intent if intent in valid else "unknown"
    except Exception:
        return "unknown"
 

def intent_aware_rerank(query: str,nodes: List[NodeWithScore],top_k: int = 8,llm=None) -> List[NodeWithScore]:
    """
    Boost nodes whose content type matches the query intent.
    Score adjustment:
      factual    → boost nodes with short, definition-like text
      analytical → boost nodes with longer explanatory content
      comparative → boost nodes whose metadata shows they cover multiple topics
      procedural → boost nodes with numbered lists or step-like structure
    """
    if not nodes:
        return nodes
 
    intent = classify_query_intent(query, llm=llm)
    logger.info(f"[Rerank-S2] Query intent: {intent}")
 
    def intent_boost(node: NodeWithScore) -> float:
        text = node.node.text
        boost = 0.0
        if intent == "factual":
            # Short, definition-like chunks score higher
            if len(text) < 400:
                boost += 0.1
            if any(w in text.lower() for w in ["is defined as", "refers to", "is a", "means"]):
                boost += 0.1
        elif intent == "analytical":
            # Longer explanatory content
            if len(text) > 400:
                boost += 0.1
            if any(w in text.lower() for w in ["because", "therefore", "as a result", "this means"]):
                boost += 0.1
        elif intent == "comparative":
            # Chunks referencing multiple entities
            if any(w in text.lower() for w in ["compared to", "versus", "while", "whereas", "unlike"]):
                boost += 0.15
        elif intent == "procedural":
            # Numbered/bulleted structure
            if any(c in text for c in ["1.", "2.", "step", "first,", "then,"]):
                boost += 0.15
        return boost
 
    for nws in nodes:
        nws.score = (nws.score or 0.0) + intent_boost(nws)
 
    nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
    result = nodes[:top_k]
    logger.info(f"[Rerank-S2] Intent-aware: {len(nodes)} → {len(result)}")
    return result
 
 
def mmr_rerank(
    query: str,
    nodes: List[NodeWithScore],
    embed_model=None,
    top_k=6,
    lambda_param=0.5,
) -> List[NodeWithScore]:
    """
    Maximal Marginal Relevance (MMR) for diversity-aware selection.

    MMR(d) = λ * Sim(d, query) - (1-λ) * max_Sim(d, selected)
    """

    if not nodes or len(nodes) <= top_k:
        return nodes

    if embed_model is None:
        from P1.llm_factory import get_embedding_model
        embed_model = get_embedding_model()

    import numpy as np

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    try:
        # Query embedding
        query_vec = embed_model.get_query_embedding(query)

        # Cache chunk embeddings
        for nws in nodes:
            if "embedding" not in nws.node.metadata:
                nws.node.metadata["embedding"] = (
                    embed_model.get_text_embedding(nws.node.text)
                )

        # Reuse cached embeddings
        doc_vecs = [
            nws.node.metadata["embedding"]
            for nws in nodes
        ]

    except Exception as e:
        logger.warning(
            f"[Rerank-S3] MMR embedding failed: {e} — skipping MMR"
        )
        return nodes[:top_k]

    selected_indices = []
    remaining = list(range(len(nodes)))

    while len(selected_indices) < top_k and remaining:

        best_idx = None
        best_score = float("-inf")

        for i in remaining:

            relevance = cosine(doc_vecs[i], query_vec)

            if selected_indices:
                redundancy = max(
                    cosine(doc_vecs[i], doc_vecs[s])
                    for s in selected_indices
                )
            else:
                redundancy = 0.0

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    result = [nodes[i] for i in selected_indices]

    logger.info(
        f"[Rerank-S3] MMR diversity: {len(nodes)} → {len(result)}"
    )

    return result


def metadata_aware_rerank(nodes: List[NodeWithScore],top_k = 5,recency_boost = 0.05, authority_docs: Optional[List[str]] = None) -> List[NodeWithScore]:
    """
    Apply score boosts based on document metadata signals.
    Boosts applied:
      - Recency: chunks from recently added documents score higher
      - Authority: manually flagged "authoritative" documents score higher
    """
    if not nodes:
        return nodes
 
    authority_docs = authority_docs or []
 
    from datetime import datetime, timezone
 
    now = datetime.now(timezone.utc)
 
    for nws in nodes:
        boost = 0.0
        meta = nws.node.metadata
 
        # Recency boost — newer documents score higher
        created_at = meta.get("created_at")
        if created_at:
            try:
                doc_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                days_old = (now - doc_time).days
                # Linear decay: 0 days old → +0.05, 365 days old → +0
                boost += recency_boost * max(0, 1.0 - days_old / 365)
            except Exception:
                pass
 
        # Authority boost — flag specific filenames as trusted sources
        if meta.get("filename") in authority_docs:
            boost += 0.1
 
        nws.score = (nws.score or 0.0) + boost
 
    nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
    result = nodes[:top_k]
    logger.info(f"[Rerank-S4] Metadata-aware: {len(nodes)} → {len(result)}")
    return result
 
def personalized_rerank(nodes: List[NodeWithScore],top_k = 5) -> List[NodeWithScore]:
    """
    Boost chunks from document categories and topics the user frequently queries.
    Reads user_profile.json which stores:
      - preferred_doc_types: ["research", "technical"]
      - frequently_asked_topics: ["transformers", "attention", "BERT"]
    """
    if not nodes:
        return nodes
 
    # Load user profile — graceful degradation if not yet built (pre-P4)
    profile = {}
    if USER_Path.exists():
        try:
            profile = json.loads(USER_Path.read_text())
        except Exception as e:
            logger.warning(f"[Rerank-S4b] Failed to load user profile: {e}")
 
    preferred_categories = set(profile.get("preferred_doc_types", []))
    frequent_topics = [t.lower() for t in profile.get("frequently_asked_topics", [])]
 
    if not preferred_categories and not frequent_topics:
        logger.debug("[Rerank-S4b] No user profile data — skipping personalization")
        return nodes[:top_k]
 
    for nws in nodes:
        boost = 0.0
        meta = nws.node.metadata
        text_lower = nws.node.text.lower()
 
        # Category preference boost
        if meta.get("doc_category") in preferred_categories:
            boost += 0.08
 
        # Topic frequency boost
        topic_hits = sum(1 for t in frequent_topics if t in text_lower)
        boost += min(topic_hits * 0.03, 0.09)  # cap at 0.09
 
        nws.score = (nws.score or 0.0) + boost
 
    nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
    result = nodes[:top_k]
    logger.info(f"[Rerank-S4b] Personalized: {len(nodes)} → {len(result)}")
    return result

def rerank_pipeline(
    query: str,
    nodes: List[NodeWithScore],
    embed_model=None,
    llm=None,
    authority_docs: Optional[List[str]] = None,
) -> List[NodeWithScore]:
    """
    Run the full 6-stage reranking pipeline.
 
    Input:  up to 20 candidates from expanded_retrieve() 
    Output: top-5 final chunks for the Generator Agent
 
    Returns:
        Top-5 NodeWithScore — the final context fed to the Generator.
    """
    if not nodes:
        return []
 
    logger.info(f"[Reranker] Starting pipeline with {len(nodes)} candidates")
 
    # Stage 1: Cross-encoder (bi-encoder → cross-encoder quality jump)
    nodes = cross_encoder_rerank(query, nodes, top_k=10)
 
    # Stage 2: Intent-aware (align chunk type with query type)
    nodes = intent_aware_rerank(query, nodes, top_k=8, llm=llm)
 
    # Stage 3: Diversity (MMR — prevent redundant chunks)
    nodes = mmr_rerank(query, nodes, embed_model=embed_model, top_k=6)
 
    # Stage 4: Metadata (recency + authority boosts)
    nodes = metadata_aware_rerank(nodes, top_k=5, authority_docs=authority_docs)
 
    # Stage 4b: Personalized (user profile preferences)
    nodes = personalized_rerank(nodes, top_k=5)
 
    logger.info(f"[Reranker] Pipeline complete → {len(nodes)} final chunks")
    return nodes

if __name__ == "__main__":
    import sys
    import logging
    from app_context import LLM, EMBED_MODEL
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    from P1.document_loader import load_document
    from P1.chunker import chunk_documents, ChunkStrategy
    from P1.vector_store import build_index, get_retriever
 
    if len(sys.argv) < 3:
        print("Usage: python reranker.py <file_path> <query>")
        sys.exit(1)
 
    docs = load_document(sys.argv[1])
    nodes = chunk_documents(docs, strategy=ChunkStrategy.SENTENCE)
    index = build_index(nodes, collection_name="rerank_test", embed_model=EMBED_MODEL)
    retriever = get_retriever(index, similarity_top_k=20)
 
    query = sys.argv[2]
    candidates = retriever.retrieve(query)
    embed_model = EMBED_MODEL
    llm = LLM
    print(f"\n📋 {len(candidates)} candidates before reranking")
    final = rerank_pipeline(
        query=query,
        nodes=candidates,
        embed_model=embed_model,
        llm=llm,
    )
    from P1.vector_store import delete_collection
    delete_collection("rerank_test")