"""
api/schemas/sessions.py
Pydantic schemas for Conversation Session Management.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class SessionSummary(BaseModel):
    """Metadata summary of a conversation session."""
    session_id: str = Field(..., description="Unique conversation identifier")
    title: str = Field(..., description="Short title derived from the initial query")
    created_at: str = Field(..., description="ISO timestamp of session creation")
    updated_at: str = Field(..., description="ISO timestamp of last activity")
    message_count: int = Field(..., description="Total user and assistant messages in session")


class SessionMessage(BaseModel):
    """A single user or assistant conversational turn."""
    role: Literal["user", "assistant"] = Field(..., description="Sender role")
    content: str = Field(..., description="User-visible message text")
    timestamp: str = Field(..., description="ISO timestamp when message was sent")
    citations: List[str] = Field(default_factory=list, description="Citations referenced by assistant")
    consensus_score: Optional[float] = Field(None, description="Reviewer consensus score for assistant turn")


class SessionDetail(BaseModel):
    """Full detail of a conversation including message history."""
    session_id: str = Field(..., description="Unique conversation identifier")
    title: str = Field(..., description="Session title")
    created_at: str = Field(..., description="ISO timestamp of session creation")
    updated_at: str = Field(..., description="ISO timestamp of last activity")
    messages: List[SessionMessage] = Field(default_factory=list, description="Ordered chronological messages")


class SessionListResponse(BaseModel):
    """Response model for GET /api/chat/sessions."""
    sessions: List[SessionSummary] = Field(default_factory=list, description="List of conversation summaries")
    total: int = Field(..., description="Total active sessions")


class SessionDeleteResponse(BaseModel):
    """Response model for DELETE /api/chat/sessions/{session_id}."""
    success: bool = Field(..., description="Whether the session was deleted")
    session_id: str = Field(..., description="Session ID that was removed")
    message: str = Field(..., description="Result message")
