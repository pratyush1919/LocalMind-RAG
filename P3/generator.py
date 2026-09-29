
"""
generator.py — P3-T5 + P5 integration

Generator Agent: produces a grounded, cited answer from retrieved context.

P3 responsibilities:
  - Role-Based Generation
  - Grounded Generation
  - Citation extraction
  - Citation fallback
  - Agent communication

P5 addition:
  - Separates Graph RAG facts from normal document/vector chunks.
  - Graph facts are explicitly presented as graph evidence.
  - Graph facts use [Graph Source N] citation namespace.
  - Normal retrieved chunks use [Source N] citation namespace.
  - Graph and document citations are validated independently.

Role selection:
  analyst    → comparative queries
  summarizer → long/report queries
  researcher → factual / multi-hop queries
  explainer  → how-to queries
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from P3.state import AgentState, make_message

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Role prompts
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_SYSTEM_PROMPTS = {
    "analyst": (
        "You are an analytical research assistant. "
        "Structure comparisons clearly and answer using only supplied evidence."
    ),

    "summarizer": (
        "You are a concise summarization assistant. "
        "Focus on the key information supported by the supplied evidence."
    ),

    "researcher": (
        "You are a precise research assistant. "
        "Give factual, evidence-grounded answers and acknowledge uncertainty."
    ),

    "explainer": (
        "You are a clear, patient explainer. "
        "Explain concepts step by step using only the supplied evidence."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────
# Generation prompt
# ─────────────────────────────────────────────────────────────────────────────

_GENERATION_PROMPT = """\
<generation_task>

  <role>
    {system_prompt}
  </role>

  <objective>
    Answer the user's question using ONLY the supplied evidence.
  </objective>

  <grounding_policy>

    <rule id="G1">
      Use only facts explicitly present in the supplied evidence.
    </rule>

    <rule id="G2">
      Do not use outside knowledge.
    </rule>

    <rule id="G3">
      Do not guess, invent, or assume missing information.
    </rule>

    <rule id="G4">
      If the evidence is insufficient, say that the available evidence
      is insufficient instead of guessing.
    </rule>

  </grounding_policy>


  <evidence>

    <graph_evidence>
      <description>
        These facts come from the Knowledge Graph / Graph RAG system.
      </description>

      <citation_namespace>
        [Graph Source N]
      </citation_namespace>

      <citation_rule>
        Every factual claim derived from graph_evidence MUST use the
        exact [Graph Source N] citation associated with that fact.
      </citation_rule>

      {graph_context}

    </graph_evidence>


    <document_evidence>
      <description>
        These facts come from retrieved documents/vector search.
      </description>

      <citation_namespace>
        [Source N]
      </citation_namespace>

      <citation_rule>
        Every factual claim derived from document_evidence MUST use the
        exact [Source N] citation associated with that fact.
      </citation_rule>

      {document_context}

    </document_evidence>

  </evidence>


  <citation_policy>

    <rule id="C1">
      Graph evidence MUST be cited using [Graph Source N].
    </rule>

    <rule id="C2">
      Document evidence MUST be cited using [Source N].
    </rule>

    <rule id="C3">
      NEVER cite a graph fact using [Source N].
    </rule>

    <rule id="C4">
      NEVER cite a document fact using [Graph Source N].
    </rule>

    <rule id="C5">
      Use only citation numbers that actually exist in the supplied evidence.
    </rule>

    <rule id="C6">
      If one statement uses both graph and document evidence, include both
      citation types.
    </rule>

    <rule id="C7">
      Citations must appear immediately after the claim they support.
    </rule>

    <rule id="C8">
      Do not create a separate Source or References section.
    </rule>

  </citation_policy>


  <reasoning_policy>

    <rule id="R1">
      First identify which evidence supports the answer.
    </rule>

    <rule id="R2">
      For relationship, connection, comparison, and multi-hop questions,
      prioritize graph_evidence when it directly supports the relationship.
    </rule>

    <rule id="R3">
      Do not treat the presence of a document as evidence for a graph fact.
    </rule>

    <rule id="R4">
      Do not transfer citations between evidence types.
    </rule>

  </reasoning_policy>


  <answer_policy>

    <rule id="A1">
      Answer the question directly.
    </rule>

    <rule id="A2">
      Synthesize the evidence instead of copying it verbatim.
    </rule>

    <rule id="A3">
      Keep the answer concise and factual.
    </rule>

    <rule id="A4">
      Do not mention filenames, page numbers, metadata, chunk IDs,
      or internal retrieval details.
    </rule>

    <rule id="A5">
      Every factual paragraph must contain an appropriate citation.
    </rule>

  </answer_policy>


  <citation_examples>

    <graph_example>
      RAG uses a Knowledge Graph. [Graph Source 2]
    </graph_example>

    <document_example>
      RAG combines retrieved information with language generation. [Source 1]
    </document_example>

    <mixed_example>
      RAG combines retrieved information with language generation [Source 1]
      and uses a Knowledge Graph [Graph Source 2].
    </mixed_example>

  </citation_examples>


  <question>
    {question}
  </question>


  <output_format>
    Return ONLY the final answer.
    Do not output XML.
    Do not output your reasoning.
    Do not output a separate citation list.
  </output_format>

</generation_task>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Role selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_role(query_type: str, agent_messages: List[Dict]) -> str:
    """
    Select Generator role.

    First checks whether Supervisor explicitly assigned a role
    via agent_messages.

    Falls back to query_type-based heuristic.
    """

    # Check agent_messages for an explicit role assignment.
    for msg in reversed(agent_messages):
        if (
            msg.get("agent") == "supervisor"
            and "role" in (msg.get("content") or {})
        ):
            return msg["content"]["role"]

    # Heuristic fallback.
    mapping = {
        "comparative": "analyst",
        "multi_hop": "researcher",
        "simple_factual": "researcher",
        "conversational": "explainer",
        "web_needed": "researcher",
    }

    return mapping.get(query_type, "researcher")


# ─────────────────────────────────────────────────────────────────────────────
# Context formatting
# ─────────────────────────────────────────────────────────────────────────────

def _format_context(chunks: List[Dict]) -> str:
    """
    Format normal document/vector chunks as numbered sources.

    Graph RAG chunks are NOT handled here. They are separated before
    this function is called.
    """

    if not chunks:
        return "No relevant document context found."

    parts = []

    for i, chunk in enumerate(chunks[:5], start=1):
        meta = chunk.get("metadata", {})

        filename = meta.get("filename", "unknown")
        doc_name = filename.rsplit(".", 1)[0]

        text = chunk.get("text", "").strip()

        parts.append(
            f"[Source {i}] (Document: {doc_name})\n\n{text}"
        )

    return "\n\n".join(parts)


def _format_graph_context(graph_chunks: List[Dict]) -> str:
    """
    Format Graph RAG evidence separately from document chunks.

    Graph RAG results are produced by the researcher agent and marked with:

        source == "graph_rag"

    The graph facts themselves are kept intact because they represent
    structured relationship evidence produced by the Knowledge Graph.

    Graph citation IDs are already embedded by graph_retriever.py as:

        [Graph Source N]
    """

    if not graph_chunks:
        return "No graph context available."

    parts = []

    for chunk in graph_chunks:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        parts.append(text)

    if not parts:
        return "No graph context available."

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Citation handling
# ─────────────────────────────────────────────────────────────────────────────

def _extract_citations(answer: str) -> List[str]:
    """
    Extract both document and Graph RAG citations.

    Supported:

        [Source 1]
        [Source 2]

        [Graph Source 1]
        [Graph Source 2]
    """

    return re.findall(
        r"\[(?:Graph )?Source \d+\]",
        answer,
    )


def _extract_graph_citations(answer: str) -> List[str]:
    """Extract only Graph RAG citations."""

    return re.findall(
        r"\[Graph Source \d+\]",
        answer,
    )


def _extract_document_citations(answer: str) -> List[str]:
    """
    Extract only normal document citations.

    Negative lookbehind prevents:

        [Graph Source 1]

    from being incorrectly classified as:

        [Source 1]
    """

    return re.findall(
        r"(?<!Graph )\[Source \d+\]",
        answer,
    )


def _inject_citations(
    answer: str,
    document_chunks: List[Dict],
    graph_chunks: List[Dict],
) -> str:
    """
    Citation fallback.

    IMPORTANT:
        Graph-enabled answers must NOT receive a fake document citation.

    If graph evidence is present and the LLM failed citation validation,
    we conservatively attach the first graph citation rather than falsely
    attributing graph-derived claims to [Source 1].

    For document-only answers, retain the original [Source 1] fallback.
    """

    paragraphs = [
        p.strip()
        for p in answer.split("\n\n")
        if p.strip()
    ]

    fixed = []

    has_graph_context = bool(graph_chunks)

    for paragraph in paragraphs:

        # Already has either citation type.
        if re.search(
            r"\[(?:Graph )?Source \d+\]",
            paragraph,
        ):
            fixed.append(paragraph)
            continue

        if has_graph_context:

            # Graph citations are embedded in the graph context.
            # Use the first available graph citation as a conservative
            # fallback rather than inventing a document citation.
            graph_citations = []

            for chunk in graph_chunks:
                graph_citations.extend(
                    _extract_graph_citations(
                        chunk.get("text", "")
                    )
                )

            if graph_citations:
                fixed.append(
                    paragraph + f" {graph_citations[0]}"
                )
            else:
                # This should normally never happen because graph
                # formatting creates Graph Source IDs.
                fixed.append(paragraph)

        else:

            # Preserve original P3 document fallback behavior.
            if document_chunks:
                fixed.append(
                    paragraph + " [Source 1]"
                )
            else:
                fixed.append(paragraph)

    return "\n\n".join(fixed)


def _paragraphs_missing_citations(text: str) -> List[str]:
    """
    Return paragraphs that contain neither document nor graph citation.

    Valid citation forms:

        [Source N]
        [Graph Source N]
    """

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    missing = []

    citation_pattern = r"\[(?:Graph )?Source \d+\]"

    for paragraph in paragraphs:

        if not re.search(
            citation_pattern,
            paragraph,
        ):
            missing.append(paragraph)

    return missing


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def generator_node(state: AgentState) -> dict:
    """
    LangGraph node for the Generator Agent.

    Reads:
        state.reranked_chunks
        state.retrieved_chunks
        state.query
        state.query_type
        state.agent_messages

    Writes:
        state.answer
        state.role_used
        state.citations
        state.agent_messages

    P5:
        Separates Graph RAG chunks from normal document chunks before
        constructing the generation context.

        Graph evidence uses [Graph Source N].
        Document evidence uses [Source N].
    """

    from app_context import get_agent_llm

    llm = get_agent_llm(temperature=0.2)

    query = state["query"]
    query_type = state.get(
        "query_type",
        "simple_factual",
    )

    chunks = (
        state.get("reranked_chunks")
        or state.get("retrieved_chunks")
        or []
    )

    agent_messages = state.get(
        "agent_messages",
        [],
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Select role
    # ─────────────────────────────────────────────────────────────────────────

    role = _select_role(
        query_type,
        agent_messages,
    )

    system_prompt = _ROLE_SYSTEM_PROMPTS.get(
        role,
        _ROLE_SYSTEM_PROMPTS["researcher"],
    )

    # ─────────────────────────────────────────────────────────────────────────
    # P5 — Separate Graph RAG from document/vector evidence
    # ─────────────────────────────────────────────────────────────────────────

    graph_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("source") == "graph_rag"
    ]

    document_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("source") != "graph_rag"
    ]

    graph_context = _format_graph_context(
        graph_chunks
    )

    document_context = _format_context(
        document_chunks
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Debug context
    # ─────────────────────────────────────────────────────────────────────────

    print("\n=== GRAPH CONTEXT SENT TO LLM ===\n")
    print(graph_context)

    print("\n=== DOCUMENT CONTEXT SENT TO LLM ===\n")
    print(document_context)

    print("\n==================================\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Build generation prompt
    # ─────────────────────────────────────────────────────────────────────────

    prompt = _GENERATION_PROMPT.format(
        system_prompt=system_prompt,
        graph_context=graph_context,
        document_context=document_context,
        question=query,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Generate answer
    # ─────────────────────────────────────────────────────────────────────────

    answer = (
        "I was unable to generate an answer based on the available context."
    )

    try:

        response = llm.invoke(prompt)

        answer = response.content.strip()

        print("\n=== GENERATED ANSWER ===\n")
        print(answer)
        print("\n========================\n")

    except Exception as e:

        logger.error(
            f"[Generator] LLM call failed: {e}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Citation validation
    # ─────────────────────────────────────────────────────────────────────────

    citations = _extract_citations(answer)

    graph_citations = _extract_graph_citations(
        answer
    )

    document_citations = _extract_document_citations(
        answer
    )

    missing = _paragraphs_missing_citations(
        answer
    )

    if not citations or missing:

        logger.warning(
            "LLM failed citation validation."
        )

        # P5:
        # If graph evidence exists, NEVER inject [Source 1] because
        # that would falsely attribute graph-derived claims to a
        # document source.
        if graph_chunks:

            logger.warning(
                "[Generator] Graph evidence is present. "
                "Using graph-aware citation fallback instead of "
                "document [Source 1] fallback."
            )

        else:

            logger.warning(
                "[Generator] No graph evidence present. "
                "Using normal document citation fallback."
            )

        answer = _inject_citations(
            answer,
            document_chunks,
            graph_chunks,
        )

        citations = _extract_citations(
            answer
        )

        graph_citations = _extract_graph_citations(
            answer
        )

        document_citations = _extract_document_citations(
            answer
        )

    missing = _paragraphs_missing_citations(
        answer
    )

    print(
        f"Citations after injection: {len(citations)}"
    )

    print(
        f"Graph citations: {len(graph_citations)}"
    )

    print(
        f"Document citations: {len(document_citations)}"
    )

    print(
        f"Paragraphs missing citations after injection: {len(missing)}"
    )

    for i, paragraph in enumerate(
        missing,
        start=1,
    ):

        print(
            f"\n--- MISSING PARAGRAPH {i} ---"
        )

        print(paragraph)

    # ─────────────────────────────────────────────────────────────────────────
    # Agent message
    # ─────────────────────────────────────────────────────────────────────────

    msg = make_message(
        agent="generator",
        type="generation",
        content={
            "role_used": role,
            "answer_length": len(answer),
            "citation_count": len(citations),
            "graph_citation_count": len(graph_citations),
            "document_citation_count": len(document_citations),
            "chunk_count": len(chunks),
            "graph_chunk_count": len(graph_chunks),
            "document_chunk_count": len(document_chunks),
        },
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Return updated state
    # ─────────────────────────────────────────────────────────────────────────

    return {
        "answer": answer,
        "role_used": role,
        "citations": citations,
        "agent_messages": [msg],
    }

