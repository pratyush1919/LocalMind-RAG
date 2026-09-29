"""
guardrails/output_guardrails.py  —  P6-T3
Scan all generated outputs before returning to the user.

Concepts covered:
  - Output Validation        (every answer scanned before display)
  - PII Detection            (Microsoft Presidio — EMAIL, PHONE, SSN, etc.)
  - Data Leakage Prevention  (verbatim passage reproduction check)
  - Secure RAG               (every claim verified against retrieved context)
  - Hallucination Prevention (RAGAS faithfulness → LOW CONFIDENCE warning)
  - Secure Tool Calling      (tool call audit log in /logs/tool_audit.jsonl)

Output scanning pipeline (in order):
  1. PII scan + redaction  (presidio-analyzer + presidio-anonymizer)
  2. Data leakage check    (verbatim passages > 200 chars from source chunks)
  3. Secure RAG check      (unsupported claims tagged [UNVERIFIED])
  4. Hallucination check   (RAGAS faithfulness — prepend LOW CONFIDENCE if < 0.6)

All scanned answers are returned with:
  - answer_clean:       The safe, redacted, verified answer text
  - pii_found:          bool
  - leakage_detected:   bool
  - unverified_claims:  list of flagged sentences
  - faithfulness_score: float (0-1) from RAGAS or None if unavailable
  - low_confidence:     bool
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TOOL_AUDIT_LOG = Path("./logs/tool_audit.jsonl")
_SECURITY_LOG = Path("./logs/security_events.jsonl")
_LEAKAGE_THRESHOLD = 50   # characters
_FAITHFULNESS_THRESHOLD = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — PII Detection + Redaction
# ─────────────────────────────────────────────────────────────────────────────

def scan_and_redact_pii(text: str) -> Tuple[str, bool, List[str]]:
    """
    Detect and redact PII using Microsoft Presidio.

    Entity types detected: EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD,
                           IP_ADDRESS, PERSON, LOCATION, NRP (nationality/religion)

    Args:
        text: Answer text to scan.

    Returns:
        (redacted_text, pii_found, detected_entity_types)

    Concept: PII Detection
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        entities_to_detect = [
            "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
            "IP_ADDRESS", "PERSON", "LOCATION", "NRP",
        ]

        results = analyzer.analyze(
            text=text,
            entities=entities_to_detect,
            language="en",
        )

        if not results:
            return text, False, []

        detected_types = list({r.entity_type for r in results})
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        redacted_text = anonymized.text

        logger.info(f"[OutputGuard] PII detected and redacted: {detected_types}")
        return redacted_text, True, detected_types

    except ImportError:
        logger.warning("[OutputGuard] presidio not installed — using regex fallback for PII")
        return _regex_pii_redact(text)
    except Exception as e:
        logger.warning(
            f"[OutputGuard] Presidio scan failed — "
            f"using regex fallback: {e}"
        )
        return _regex_pii_redact(text)


def _regex_pii_redact(text: str) -> Tuple[str, bool, List[str]]:
    """Fallback regex-based PII redaction when Presidio is not installed."""
    pii_patterns = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
        "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }
    found_types = []
    redacted = text
    for entity_type, pattern in pii_patterns.items():
        if re.search(pattern, redacted):
            redacted = re.sub(pattern, f"<{entity_type}_REDACTED>", redacted)
            found_types.append(entity_type)

    return redacted, bool(found_types), found_types


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Data Leakage Prevention
# ─────────────────────────────────────────────────────────────────────────────

def check_data_leakage(
    answer: str,
    source_chunks: List[Dict],
    threshold: int = _LEAKAGE_THRESHOLD,
) -> Tuple[str, bool]:
    """
    Check if the answer reproduces verbatim passages from source documents.

    Logic: extract all substrings of length >= threshold from source chunks,
    check if any appear verbatim in the answer.
    If found, truncate the copied passage and add a [TRUNCATED] marker.

    Args:
        answer:        Generated answer text.
        source_chunks: Retrieved chunk dicts (with 'text' key).
        threshold:     Minimum character length to flag as leakage.

    Returns:
        (cleaned_answer, leakage_detected)

    Concept: Data Leakage Prevention
    """
    leakage_detected = False
    cleaned = answer

    for chunk in source_chunks:
        source_text = chunk.get("text", "")
        if len(source_text) < threshold:
            continue

        # Check for verbatim passages
        # Use sliding window over source text
        for start in range(0, len(source_text) - threshold, 50):
            excerpt = source_text[start: start + threshold]
            if len(excerpt) < threshold:
                continue
            # Normalize whitespace for comparison
            excerpt_norm = re.sub(r"\s+", " ", excerpt.strip())
            answer_norm = re.sub(r"\s+", " ", cleaned)

            if excerpt_norm in answer_norm:
                # Truncate the leaked passage
                cleaned = cleaned.replace(
                    excerpt.strip(),
                    excerpt.strip()[:100] + "... [TRUNCATED — potential data leakage]",
                )
                leakage_detected = True
                logger.warning(
                    f"[OutputGuard] Data leakage detected: "
                    f"{len(excerpt)} chars from {chunk.get('metadata', {}).get('filename', '?')}"
                )
                break  # one truncation per chunk

    return cleaned, leakage_detected


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Secure RAG (unsupported claim detection)
# ─────────────────────────────────────────────────────────────────────────────

def secure_rag_check(
    answer: str,
    source_chunks: List[Dict],
) -> Tuple[str, List[str]]:
    """
    Verify that every factual sentence in the answer is supported by retrieved context.
    Tag unsupported claims with [UNVERIFIED].

    Method: keyword overlap heuristic (fast, no LLM call).
    Sentences with < 20% keyword overlap with any source chunk are flagged.

    For production: replace with RAGAS answer_correctness or an LLM grounding check.

    Args:
        answer:        Generated answer text.
        source_chunks: Retrieved chunk dicts.

    Returns:
        (annotated_answer, unverified_claims_list)

    Concept: Secure RAG
    """
    # Extract all source text as one pool
    source_pool = " ".join(
        chunk.get("text", "") for chunk in source_chunks
    ).lower()

    if not source_pool.strip():
        return answer + "\n\n[NOTE: No source context available for verification]", []

    # Split answer into sentences
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    annotated = []
    unverified = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            annotated.append(sentence)
            continue

        # Extract meaningful keywords from sentence (skip stopwords + citation markers)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "it", "this",
            "that", "and", "or", "but", "in", "of", "to", "for", "with",
            "source", "according",
        }
        words = re.findall(r"\b[a-z]{4,}\b", sentence.lower())
        keywords = [w for w in words if w not in stopwords]

        if not keywords:
            annotated.append(sentence)
            continue

        # Count how many keywords appear in the source pool
        supported = sum(1 for kw in keywords if kw in source_pool)
        overlap = supported / len(keywords) if keywords else 1.0

        if overlap < 0.2:  # less than 20% keyword overlap → flag
            annotated.append(sentence + " [UNVERIFIED]")
            unverified.append(sentence[:100])
        else:
            annotated.append(sentence)

    return " ".join(annotated), unverified


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Hallucination check (RAGAS faithfulness)
# ─────────────────────────────────────────────────────────────────────────────

def check_faithfulness(
    query: str,
    answer: str,
    source_chunks: List[Dict],
) -> Tuple[float, bool]:
    """
    Compute RAGAS faithfulness score for the answer.

    faithfulness = proportion of answer claims supported by context.
    If score < _FAITHFULNESS_THRESHOLD, answer is flagged as low confidence.

    Args:
        query:         User query.
        answer:        Generated answer.
        source_chunks: Retrieved chunk dicts.

    Returns:
        (faithfulness_score, is_low_confidence)

    Concept: Hallucination Prevention
    """
    try:
        from ragas.metrics import faithfulness
        from ragas import evaluate
        from datasets import Dataset

        context_list = [chunk.get("text", "") for chunk in source_chunks[:5]]
        if not context_list:
            return 0.0, True

        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [context_list],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness])
        score = float(result["faithfulness"])
        return score, score < _FAITHFULNESS_THRESHOLD

    except Exception as e:
        logger.debug(f"[OutputGuard] RAGAS faithfulness check skipped: {e}")
        # Fallback: use keyword overlap as proxy
        source_pool = " ".join(c.get("text", "") for c in source_chunks).lower()
        words = re.findall(r"\b[a-z]{5,}\b", answer.lower())
        if not words:
            return 0.5, False
        supported = sum(1 for w in words if w in source_pool)
        score = supported / len(words)
        return round(score, 3), score < _FAITHFULNESS_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Secure tool call audit logger
# ─────────────────────────────────────────────────────────────────────────────

def log_tool_call(
    tool_name: str,
    inputs: Dict,
    outputs: Dict,
    latency_ms: float,
    success: bool,
    session_id: str = "",
    user_role: str = "",
) -> None:
    """
    Log every tool invocation with inputs, outputs, latency, and success status.
    Used for audit trails and Tool Call Monitoring (P8).

    Concept: Secure Tool Calling
    """
    _TOOL_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "inputs": {k: str(v)[:200] for k, v in inputs.items()},  # truncate for log size
        "output_preview": str(outputs)[:200],
        "latency_ms": round(latency_ms, 1),
        "success": success,
        "session_id": session_id,
        "user_role": user_role,
    }
    with open(_TOOL_AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main output scanner — runs all 4 steps
# ─────────────────────────────────────────────────────────────────────────────

class OutputScanResult:
    """Result of full output scanning pipeline."""
    def __init__(
        self,
        answer_clean: str,
        pii_found: bool,
        pii_types: List[str],
        leakage_detected: bool,
        unverified_claims: List[str],
        faithfulness_score: Optional[float],
        low_confidence: bool,
    ):
        self.answer_clean = answer_clean
        self.pii_found = pii_found
        self.pii_types = pii_types
        self.leakage_detected = leakage_detected
        self.unverified_claims = unverified_claims
        self.faithfulness_score = faithfulness_score
        self.low_confidence = low_confidence

    def to_dict(self) -> Dict:
        return {
            "pii_found": self.pii_found,
            "pii_types": self.pii_types,
            "leakage_detected": self.leakage_detected,
            "unverified_claims": self.unverified_claims,
            "faithfulness_score": self.faithfulness_score,
            "low_confidence": self.low_confidence,
        }


def scan_output(
    query: str,
    answer: str,
    source_chunks: List[Dict],
    run_faithfulness: bool = True,
) -> OutputScanResult:
    """
    Run the full output scanning pipeline on a generated answer.

    Steps:
      1. PII scan + redaction
      2. Data leakage check
      3. Secure RAG (unsupported claim tagging)
      4. Hallucination check (faithfulness)

    Args:
        query:             Original user query.
        answer:            Raw generated answer.
        source_chunks:     Retrieved chunk dicts used to generate answer.
        run_faithfulness:  Set False to skip slow RAGAS check in tests.

    Returns:
        OutputScanResult with cleaned answer and all scan metadata.

    Concepts: Output Validation, Secure RAG, Data Leakage Prevention,
              PII Detection, Hallucination Prevention
    """
    # Step 1: PII
    answer_clean, pii_found, pii_types = scan_and_redact_pii(answer)

    # Step 2: Data leakage
    answer_clean, leakage = check_data_leakage(answer_clean, source_chunks)

    # Step 3: Secure RAG — unsupported claims
    answer_clean, unverified = secure_rag_check(answer_clean, source_chunks)

    # Step 4: Hallucination check
    faithfulness_score = None
    low_confidence = False
    if run_faithfulness:
        faithfulness_score, low_confidence = check_faithfulness(
            query, answer_clean, source_chunks
        )
        if low_confidence:
            answer_clean = (
                f"⚠️ LOW CONFIDENCE (faithfulness={faithfulness_score:.2f}): "
                f"{answer_clean}"
            )

    logger.info(
        f"[OutputGuard] scan complete | pii={pii_found} | "
        f"leakage={leakage} | unverified={len(unverified)} | "
        f"faithfulness={faithfulness_score} | low_conf={low_confidence}"
    )

    return OutputScanResult(
        answer_clean=answer_clean,
        pii_found=pii_found,
        pii_types=pii_types,
        leakage_detected=leakage,
        unverified_claims=unverified,
        faithfulness_score=faithfulness_score,
        low_confidence=low_confidence,
    )
