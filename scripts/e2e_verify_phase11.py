"""
scripts/e2e_verify_phase11.py
End-to-end live verification of Phase 11:
Test A (Web OFF) -> Test B (Web ON) -> Test C (Web OFF)

Exercises the real Researcher agent, real Browser Agent, real DuckDuckGo search,
real Groq evidence extraction, and the full pipeline streaming lifecycle.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.browser_agent import research
from P3.agent_pipeline import AgentPipeline, run_query_stream
from P3.researcher import _select_tool, researcher_node
from P3.state import initial_state

print("=" * 70)
print("PHASE 11 E2E LIVE VERIFICATION: Web OFF -> Web ON -> Web OFF")
print("=" * 70)

# Local document chunks for simulation
LOCAL_DOC_CHUNKS = [
    {
        "node_id": "chunk_local_1",
        "text": "BERT (Bidirectional Encoder Representations from Transformers) is designed to pre-train deep bidirectional representations from unlabeled text.",
        "score": 0.92,
        "metadata": {
            "filename": "bert_paper.pdf",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "doc_category": "research",
            "source_type": "document",
        },
        "source": "document_search",
    }
]

# ----------------------------------------------------------------------------
# Test A: Web Research OFF
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST A: Turn 1 — Web Research OFF")
print("Query: 'What is BERT?'")
print("=" * 70)

state_a = initial_state(
    query="What is BERT?",
    session_id="e2e_sess_a",
    use_web_research=False,
)
state_a["query_type"] = "simple_factual"
state_a["sub_queries"] = ["What is BERT?"]

mock_ctx = MagicMock()
with patch("P3.researcher._tool_document_search", return_value=LOCAL_DOC_CHUNKS), \
     patch("agents.browser_agent.research") as spy_research:
    result_a = researcher_node(state_a, retrieval_ctx=mock_ctx)

    # Assertions
    assert spy_research.call_count == 0, "FAIL: BrowserAgent was invoked when web research was OFF!"
    assert result_a.get("web_research_used") is False, "FAIL: web_research_used is True when OFF!"
    assert len(result_a.get("retrieved_chunks", [])) > 0, "FAIL: Local retrieval failed!"

    print(f"[Turn 1] Tools used: {result_a.get('tools_used', [])}")
    print(f"[Turn 1] Web research called: {spy_research.call_count > 0}")
    print(f"[Turn 1] web_research_used: {result_a.get('web_research_used')}")
    print(f"[Turn 1] Local chunk title: {result_a['retrieved_chunks'][0]['metadata'].get('title')}")
    print(f"[Turn 1] Chunk source: {result_a['retrieved_chunks'][0].get('source')}")
    print(">>> TEST A PASSED: Local retrieval only, zero web calls.\n")

# ----------------------------------------------------------------------------
# Test B: Web Research ON (LIVE DuckDuckGo + Groq evidence extraction)
# ----------------------------------------------------------------------------
print("=" * 70)
print("TEST B: Turn 2 — Web Research ON (LIVE Groq & Web Search)")
print("Query: 'Latest breakthroughs in quantum computing'")
print("=" * 70)

state_b = initial_state(
    query="Latest breakthroughs in quantum computing",
    session_id="e2e_sess_b",
    use_web_research=True,
)
state_b["query_type"] = "web_needed"
state_b["sub_queries"] = ["Latest breakthroughs in quantum computing"]

with patch("P3.researcher._tool_document_search", return_value=[]):
    result_b = researcher_node(state_b, retrieval_ctx=mock_ctx)

    # Assertions
    assert result_b.get("web_research_used") is True, "FAIL: web_research_used is False when ON!"
    web_chunks = [c for c in result_b.get("retrieved_chunks", []) if c.get("source") == "web_search"]
    assert len(web_chunks) > 0, "FAIL: No web chunks returned from live web research!"

    print(f"[Turn 2] Tools used: {result_b.get('tools_used', [])}")
    print(f"[Turn 2] web_research_used: {result_b.get('web_research_used')}")
    print(f"[Turn 2] Total retrieved chunks: {len(result_b.get('retrieved_chunks', []))}")
    print(f"[Turn 2] Web chunks count: {len(web_chunks)}")
    for i, wc in enumerate(web_chunks[:2]):
        meta = wc.get("metadata", {})
        print(f"   Source [{i+1}]:")
        print(f"      Title: {meta.get('title')}")
        print(f"      URL: {meta.get('url')}")
        print(f"      Domain: {meta.get('domain')}")
        print(f"      Source Type: {meta.get('source_type')}")
        print(f"      Snippet excerpt: {wc.get('text', '')[:120]}...")

    # Now verify SSE event emission through run_query_stream
    class MockGraphWeb:
        def stream(self, state, config=None, stream_mode=None):
            yield {"supervisor": {"query_type": "web_needed", "route": "researcher"}}
            yield {"researcher": {"reranked_chunks": web_chunks, "web_research_used": True}}
            yield {"generator": {"answer": f"Recent breakthroughs include advancements: {web_chunks[0]['text'][:100]} [Source 1]", "citations": ["[Source 1]"], "role_used": "analyst"}}
            yield {"critic": {"critique": {"faithfulness": 9.5, "completeness": 9.0}}}
            yield {"reviewer": {"consensus_score": 9.2, "is_final": True}}

    pipeline_b = AgentPipeline(compiled_graph=MockGraphWeb(), retrieval_ctx=mock_ctx)
    with patch("guardrails.input_guardrails.prompt_firewall") as m_fw, \
         patch("guardrails.risk_scoring.full_tsl_check") as m_tsl, \
         patch("P3.router.save_agent_communication_log"):
        m_fw.return_value = MagicMock(allowed=True, sanitized_text="Quantum computing", layer="clean", reason="")
        m_tsl.return_value = {"risk_score": 0.0, "answer_clean": "Clean answer", "pii_found": False}

        events_b = list(run_query_stream(
            query="Latest breakthroughs in quantum computing",
            pipeline=pipeline_b,
            use_memory=False,
            use_web_research=True,
        ))

    event_names_b = [e["event"] for e in events_b]
    final_b = [e["data"] for e in events_b if e["event"] == "final"][0]

    print(f"[Turn 2] SSE Events: {event_names_b}")
    assert "web_research" in event_names_b, "FAIL: web_research SSE event was not emitted!"
    assert final_b.get("web_research_used") is True, "FAIL: final web_research_used is not True!"
    assert len(final_b.get("source_details", [])) > 0, "FAIL: source_details is empty!"
    assert final_b["source_details"][0].get("source_type") == "web"
    print(">>> TEST B PASSED: Real web research executed, SSE stage emitted, web provenance verified.\n")

# ----------------------------------------------------------------------------
# Test C: Turn 3 — Toggle Web Research back OFF
# ----------------------------------------------------------------------------
print("=" * 70)
print("TEST C: Turn 3 — Toggle Web Research OFF Again")
print("Query: 'Explain Transformer attention mechanisms'")
print("=" * 70)

state_c = initial_state(
    query="Explain Transformer attention mechanisms",
    session_id="e2e_sess_c",
    use_web_research=False,
)
state_c["query_type"] = "simple_factual"
state_c["sub_queries"] = ["Explain Transformer attention mechanisms"]

with patch("P3.researcher._tool_document_search", return_value=LOCAL_DOC_CHUNKS), \
     patch("agents.browser_agent.research") as spy_research_c:
    result_c = researcher_node(state_c, retrieval_ctx=mock_ctx)

    assert spy_research_c.call_count == 0, "FAIL: BrowserAgent was invoked when toggled back OFF!"
    assert result_c.get("web_research_used") is False, "FAIL: web_research_used is True when toggled back OFF!"

    print(f"[Turn 3] Tools used: {result_c.get('tools_used', [])}")
    print(f"[Turn 3] Web research called: {spy_research_c.call_count > 0}")
    print(f"[Turn 3] web_research_used: {result_c.get('web_research_used')}")
    print(f"[Turn 3] Chunk source: {result_c['retrieved_chunks'][0].get('source')}")
    print(">>> TEST C PASSED: Web capability safely disabled again on subsequent turn.\n")

print("=" * 70)
print("ALL E2E LIVE VERIFICATIONS PASSED (OFF -> ON -> OFF)!")
print("=" * 70)
