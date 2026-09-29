"""
Three complementary query enrichment strategies that improve retrieval recall.

Why each strategy helps:
  HyDE:        Questions and answers live in different parts of embedding space.
               "What is attention?" is semantically far from "Attention is a mechanism that..."
               HyDE bridges this by embedding an ideal hypothetical answer instead.
 
  Multi-Query: Any single phrasing may miss relevant chunks. 3 paraphrases cast
               a wider net and increase recall with minimal latency cost.
 
  Planning:    Not every query needs the full hybrid pipeline. A simple factual
               question ("What year was GPT-3 released?") needs keyword search.
               A conceptual question needs semantic search. Planning routes correctly.
"""
import logging
from typing import Literal
logger = logging.getLogger(__name__)
QueryStrategy = Literal["keyword", "semantic", "hybrid", "memory_only", "web_needed"]

def hyde_embed(query,embed_model,llm=None):
    """
    Generate a hypothetical ideal answer to `query`, embed it, return the vector.
    Instead of embedding the question, we embed a plausible answer.
    This lands the retrieval vector closer to actual answer chunks in the index.
    """
    if llm is None:
        from P1.llm_factory import get_llama_llm
        llm = get_llama_llm(temperature=0.3)
 
    prompt = (
        f"Write a 2-3 sentence factual answer to the following question, "
        f"as if you found it in a technical document. "
        f"Be specific and informative.\n\nQuestion: {query}\n\nAnswer:"
    )
 
    try:
        response = llm.complete(prompt)
        hypothetical_answer = str(response).strip()
        logger.debug(f"[HyDE] Hypothetical answer: {hypothetical_answer[:100]}...")
        hyde_vector = embed_model.get_text_embedding(hypothetical_answer)
        return hyde_vector, hypothetical_answer
    except Exception as e:
        logger.warning(f"[HyDE] Failed to generate hypothetical answer: {e}")
        # Fallback: just embed the original query
        return embed_model.get_query_embedding(query), query
 
 
def hyde_retrieve(query,vector_index,embed_model,llm=None,top_k=10):
    """
    Run HyDE retrieval: generate hypothetical answer → embed → search vector index.
    Returns NodeWithScore list using the hypothetical answer vector for search.
    """
    import numpy as np
    from llama_index.core.schema import QueryBundle
 
    hyde_vector, hypothetical_answer = hyde_embed(query, embed_model, llm)
 
    # Use the hypothetical answer as the retrieval text (not the query)
    retriever = vector_index.as_retriever(similarity_top_k=top_k)
    query_bundle = QueryBundle(
        query_str=hypothetical_answer,
        embedding=hyde_vector,
    )
    results = retriever.retrieve(query_bundle)
    logger.info(f"[HyDE] Retrieved {len(results)} results via hypothetical answer embedding")
    return results

def generate_query_paraphrases(query,llm=None,n = 3):
    """
    Generate n paraphrases of the query for broader retrieval coverage.
 
    Concept: Multi-Query Retrieval
    """
    if llm is None:
        from P1.llm_factory import get_llama_llm
        llm = get_llama_llm(temperature=0.2)
 
    prompt = (
        f"Generate {n} different ways to ask the following question. "
        f"Each paraphrase should use different words but preserve the meaning. "
        f"Return only the paraphrases, one per line, no numbering.\n\n"
        f"Original: {query}\n\nParaphrases:"
    )
 
    try:
        response = str(llm.complete(prompt)).strip()
        paraphrases = [line.strip() for line in response.split("\n") if line.strip()]
        paraphrases = paraphrases[:n]  # cap at n
        logger.debug(f"[MultiQuery] Generated {len(paraphrases)} paraphrases")
        return paraphrases
    except Exception as e:
        logger.warning(f"[MultiQuery] Failed to generate paraphrases: {e}")
        return [query]  # fallback: just the original
 
 
def multi_query_retrieve(query,retriever,llm=None,n_paraphrases = 3,
):
    """
    Retrieve for the original query + n paraphrases, union results, deduplicate.
 
    Flow:
      1. Generate n paraphrases of the query
      2. Retrieve for each paraphrase
      3. Retrieve for the original query
      4. Union all results, deduplicate by node_id
      5. Re-rank by highest score across all queries
 
    """
    from P2.hybrid_retriever import reciprocal_rank_fusion
 
    paraphrases = generate_query_paraphrases(query, llm=llm, n=n_paraphrases)
    all_queries = [query] + paraphrases
 
    all_result_lists = []
    for q in all_queries:
        try:
            results = retriever.retrieve(q)
            all_result_lists.append(results)
            logger.debug(f"[MultiQuery] '{q[:50]}...' → {len(results)} results")
        except Exception as e:
            logger.warning(f"[MultiQuery] Query failed: {e}")
 
    if not all_result_lists:
        return []
 
    # RRF-fuse all result lists for final ranking
    fused = reciprocal_rank_fusion(all_result_lists, top_k=10)
    logger.info(f"[MultiQuery] {len(all_queries)} queries → {len(fused)} fused results")
    return fused

def plan_retrieval_strategy(query, llm=None):
    """
    Ask the LLM to classify the query and recommend the optimal retrieval strategy.
    This is Retrieval Planning: before running any retrieval, decide HOW to retrieve.
    Returns one of:
      'keyword'     — use BM25 only (exact names, version numbers, acronyms)
      'semantic'    — use dense vector only (conceptual, definitional questions)
      'hybrid'      — use full hybrid pipeline (most complex questions)
      'memory_only' — answer from conversation memory without new retrieval
      'web_needed'  — documents lack info, needs Browser Agent (P11)
    """
    if llm is None:
        from P1.llm_factory import get_llama_llm
        llm = get_llama_llm(temperature=0.0)
 
    prompt = (
        "You are a retrieval strategy planner. Given a user query, classify it "
        "into exactly ONE of these retrieval strategies:\n\n"
        "- keyword: The query contains specific terms, acronyms, model names, version "
        "numbers, or proper nouns where exact matching is critical.\n"
        "- semantic: The query is conceptual, definitional, or comparative — meaning "
        "matters more than exact words.\n"
        "- hybrid: The query is complex, multi-part, or needs both exact terms and "
        "conceptual understanding.\n"
        "- memory_only: This is a follow-up or conversational reference to a previous "
        "exchange (e.g. 'as I mentioned', 'going back to', 'what did you say about').\n"
        "- web_needed: The query asks about recent events, current data, or information "
        "unlikely to be in a static document collection.\n\n"
        f"Query: {query}\n\n"
        "Respond with ONLY the strategy name (one word, lowercase):"
    )
 
    try:
        response = str(llm.complete(prompt)).strip().lower()
        valid = {"keyword", "semantic", "hybrid", "memory_only", "web_needed"}
        strategy = response if response in valid else "hybrid"
        logger.info(f"[Plan] Query classified as: '{strategy}'")
        return strategy
    except Exception as e:
        logger.warning(f"[Plan] Strategy planning failed: {e} — defaulting to hybrid")
        return "hybrid"
    
def expanded_retrieve(query,hybrid_retriever,vector_index,embed_model,llm=None,use_hyde= True,use_multi_query = True,use_planning = True):
    """
    Full query expansion pipeline combining planning + HyDE + multi-query.
    Flow:
      1. Retrieval Planning: classify query → choose strategy
      2a. If keyword/memory_only: skip HyDE (would hurt precision)
      2b. If semantic/hybrid: run HyDE for better vector alignment
      3. Multi-Query: always run for broader recall (unless memory_only)
      4. RRF-fuse HyDE results + multi-query results → final ranked list

    Returns:
        dict with keys: results, strategy, paraphrases, hyde_used
    """
    from P2.hybrid_retriever import reciprocal_rank_fusion
 
    # Step 1: Retrieval Planning
    strategy: QueryStrategy = "hybrid"
    if use_planning:
        strategy = plan_retrieval_strategy(query, llm=llm)
 
    if strategy == "memory_only":
        logger.info("[Expanded] Memory-only query — skipping document retrieval")
        return {"results": [], "strategy": strategy, "paraphrases": [], "hyde_used": False}
 
    all_result_lists = []
    hyde_used = False
    paraphrases = []
 
    # Step 2: HyDE (for semantic/hybrid queries)
    if use_hyde and strategy in ("semantic", "hybrid"):
        try:
            hyde_results = hyde_retrieve(query, vector_index, embed_model, llm=llm, top_k=10)
            all_result_lists.append(hyde_results)
            hyde_used = True
        except Exception as e:
            logger.warning(f"[Expanded] HyDE failed: {e}")
 
    # Step 3: Multi-Query retrieval (for all non-memory strategies)
    if use_multi_query and strategy in ("semantic", "hybrid", "keyword"):
        paraphrases = generate_query_paraphrases(query, llm=llm, n=3)
        all_queries = [query] + paraphrases
 
        for q in all_queries:
            try:
                if strategy == "keyword":
                    # For keyword strategy: BM25 only
                    results = hybrid_retriever.bm25_retriever.retrieve(q)
                else:
                    # For semantic/hybrid: full hybrid pipeline
                    results = hybrid_retriever.retrieve(q)
                all_result_lists.append(results)
            except Exception as e:
                logger.warning(f"[Expanded] Query '{q[:40]}' failed: {e}")
    else:
        # Simple single query
        try:
            results = hybrid_retriever.retrieve(query)
            all_result_lists.append(results)
        except Exception as e:
            logger.error(f"[Expanded] Retrieval failed: {e}")
 
    if not all_result_lists:
        return {"results": [], "strategy": strategy, "paraphrases": paraphrases, "hyde_used": hyde_used}
 
    # Step 4: Final fusion
    fused = reciprocal_rank_fusion(all_result_lists, top_k=20)  # top-20 for reranker input (P2-T5)
    logger.info(
        f"[Expanded] Strategy={strategy} | HyDE={hyde_used} | "
        f"Paraphrases={len(paraphrases)} | Final={len(fused)} results"
    )
 
    return {
        "results": fused,
        "strategy": strategy,
        "paraphrases": paraphrases,
        "hyde_used": hyde_used,
    }

#Quick Test: In main folder, run: python -m P2.query_expander
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    from P1.document_loader import load_document
    from P1.chunker import chunk_documents, ChunkStrategy
    from P1.vector_store import build_index
    from P1.llm_factory import get_embedding_model
    from P2.hybrid_retriever import build_hybrid_retriever
    from app_context import EMBED_MODEL, LLM

    TEST_QUERY = "Which string functions are available in Python?"

    print("\n" + "=" * 80)
    print("STEP 1: Load document")
    print("=" * 80)

    docs = load_document("data/Python.docx")
    print(f"Documents loaded: {len(docs)}")

    print("\n" + "=" * 80)
    print("STEP 2: Chunk document")
    print("=" * 80)

    nodes = chunk_documents(
        docs,
        strategy=ChunkStrategy.SENTENCE
    )
    print(f"Chunks created: {len(nodes)}")

    print("\n" + "=" * 80)
    print("STEP 3: Build vector index")
    print("=" * 80)

    index = build_index(
        nodes,
        collection_name="query_expansion_test",
        embed_model=EMBED_MODEL,
    )

    print("\n" + "=" * 80)
    print("STEP 4: Build hybrid retriever")
    print("=" * 80)

    hybrid = build_hybrid_retriever(
        nodes,
        index,
        similarity_top_k=5,
        embed_model=EMBED_MODEL,
    )

    print("\n" + "=" * 80)
    print("STEP 5: Retrieval Planning")
    print("=" * 80)

    strategy = plan_retrieval_strategy(TEST_QUERY, llm=LLM)

    print("Query:")
    print(TEST_QUERY)

    print("\nChosen Strategy:")
    print(strategy)

    print("\n" + "=" * 80)
    print("STEP 6: Multi Query Generation")
    print("=" * 80)

    paraphrases = generate_query_paraphrases(TEST_QUERY,llm = LLM)

    print("Generated Paraphrases:")
    for i, p in enumerate(paraphrases, start=1):
        print(f"{i}. {p}")

    print("\n" + "=" * 80)
    print("STEP 7: HyDE")
    print("=" * 80)

    hyde_vector, hypothetical_answer = hyde_embed(
        TEST_QUERY,
        EMBED_MODEL,
        llm=LLM
    )

    print("Hypothetical Answer:")
    print(hypothetical_answer)

    print("\nEmbedding Dimension:")
    print(len(hyde_vector))

    print("\n" + "=" * 80)
    print("STEP 8: HyDE Retrieval")
    print("=" * 80)

    hyde_results = hyde_retrieve(
        TEST_QUERY,
        index,
        EMBED_MODEL,
        top_k=3,
        llm=LLM
    )

    print(f"Results Returned: {len(hyde_results)}")

    for i, r in enumerate(hyde_results[:3], start=1):
        print(f"\n[{i}]")
        print(r.node.text[:200])

    print("\n" + "=" * 80)
    print("STEP 9: Multi Query Retrieval")
    print("=" * 80)

    multi_results = multi_query_retrieve(
        TEST_QUERY,
        hybrid,
        llm=LLM
    )

    print(f"Results Returned: {len(multi_results)}")

    for i, r in enumerate(multi_results[:3], start=1):
        print(f"\n[{i}] score={r.score:.6f}")
        print(r.node.text[:200])

    print("\n" + "=" * 80)
    print("STEP 10: Full Expanded Retrieval")
    print("=" * 80)

    result = expanded_retrieve(
        query=TEST_QUERY,
        hybrid_retriever=hybrid,
        vector_index=index,
        embed_model=EMBED_MODEL,
        llm=LLM
    )

    print("\nStrategy:")
    print(result["strategy"])

    print("\nHyDE Used:")
    print(result["hyde_used"])

    print("\nParaphrases:")
    for p in result["paraphrases"]:
        print("-", p)

    print("\nTop Results:")

    for i, r in enumerate(result["results"][:5], start=1):
        print(f"\n[{i}] score={r.score:.6f}")
        print(r.node.text[:200])

    print("\n✅ Query Expansion Smoke Test Passed")