"""
api/schemas/chat.py
Pydantic schemas for the Chat API endpoints.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request schema for POST /api/chat."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query text. Maximum length 1000 characters.",
        examples=["What is the difference between BERT and GPT?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID. If omitted, a unique ID is generated.",
        examples=["session_123"],
    )
    user_role: Literal["admin", "analyst", "viewer"] = Field(
        default="analyst",
        description="RBAC user role for access control and policy enforcement.",
    )
    use_memory: bool = Field(
        default=True,
        description="Whether to load and persist long-term memory for this turn.",
    )
    use_web_research: bool = Field(
        default=False,
        description="Whether to enable optional Groq-backed web research for this turn.",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or whitespace only.")
        return cleaned


class CritiqueSummary(BaseModel):
    """Quality critique evaluation metrics from the Critic agent."""
    faithfulness: int = Field(..., description="Score 0-10: factual grounding in retrieved sources")
    completeness: int = Field(..., description="Score 0-10: query aspect coverage")
    reasoning_quality: int = Field(..., description="Score 0-10: coherence and logical structure")
    issues: List[str] = Field(default_factory=list, description="Specific issues identified by Critic")
    summary: Optional[str] = Field(default=None, description="One-sentence critique assessment")


class TslSummary(BaseModel):
    """Trust and Safety Layer metrics and scan results."""
    risk_score: float = Field(..., description="Composite risk score (0.0 to 1.0)")
    pii_found: bool = Field(default=False, description="Whether PII was detected and redacted")
    leakage_detected: bool = Field(default=False, description="Whether sensitive data leakage was detected")
    low_confidence: bool = Field(default=False, description="Whether faithfulness confidence was low")
    compliance_passed: bool = Field(default=True, description="Whether regulatory compliance checks passed")


class ChatResponse(BaseModel):
    """Standard response model for POST /api/chat."""
    answer: str = Field(..., description="Final safe, generated answer")
    session_id: str = Field(..., description="Session identifier")
    run_id: str = Field(..., description="Unique query execution run identifier")
    user_role: str = Field(..., description="Active user role used during execution")
    query_type: Optional[str] = Field(None, description="Supervisor classification, e.g. simple_factual, multi_hop")
    role_used: Optional[str] = Field(None, description="Generator persona role used")
    strategy: Optional[str] = Field(None, description="Retrieval strategy selected")
    consensus_score: float = Field(0.0, description="Reviewer weighted consensus score (0.0 - 10.0)")
    is_final: bool = Field(False, description="Whether the answer met the finalization threshold")
    iterations: int = Field(0, description="Number of refinement iterations executed")
    graph_rag_used: bool = Field(False, description="Whether the Knowledge Graph was queried")
    web_research_used: bool = Field(False, description="Whether external web research was executed and yielded evidence")
    sources: List[str] = Field(default_factory=list, description="Source filenames used in the answer")
    source_details: List[dict] = Field(default_factory=list, description="Structured evidence provenance including URLs and source types")
    citations: List[str] = Field(default_factory=list, description="Citation identifiers extracted from the answer")
    critique: Optional[CritiqueSummary] = Field(None, description="Critic quality evaluation breakdown")
    tsl: Optional[TslSummary] = Field(None, description="Trust and Safety Layer metrics")
    guardrail_blocked: bool = Field(False, description="Whether the query was blocked by input or output guardrails")
    guardrail_reason: Optional[str] = Field(None, description="Reason if blocked by guardrails")
