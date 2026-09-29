"""
api/routes/sessions.py
Conversation Session Management routes for LocalMind-RAG.
"""

from fastapi import APIRouter, HTTPException, status

from api.schemas.sessions import (
    SessionDeleteResponse,
    SessionDetail,
    SessionListResponse,
)
from api.services.session_service import (
    delete_session,
    get_session_history,
    list_sessions,
)

router = APIRouter()


@router.get(
    "/chat/sessions",
    response_model=SessionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all conversation sessions",
    description="Retrieve a list of all active chat sessions with summary metadata.",
)
async def list_sessions_endpoint() -> SessionListResponse:
    """
    Handle GET /api/chat/sessions.
    Returns all persisted conversation sessions sorted by most recent activity.
    """
    return list_sessions()


@router.get(
    "/chat/sessions/{session_id}",
    response_model=SessionDetail,
    status_code=status.HTTP_200_OK,
    summary="Get conversation history for a session",
    description="Retrieve the chronological message history (user and assistant turns) for a specific session.",
)
async def get_session_history_endpoint(session_id: str) -> SessionDetail:
    """
    Handle GET /api/chat/sessions/{session_id}.
    Returns user-visible conversation history or 404 if not found.
    """
    history = get_session_history(session_id)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{session_id}' not found.",
        )
    return history


@router.delete(
    "/chat/sessions/{session_id}",
    response_model=SessionDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a conversation session",
    description="Delete a specific conversation session and its message history. Preserves long-term user memory.",
)
async def delete_session_endpoint(session_id: str) -> SessionDeleteResponse:
    """
    Handle DELETE /api/chat/sessions/{session_id}.
    Removes the session from the conversation store.
    """
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{session_id}' not found.",
        )
    return SessionDeleteResponse(
        success=True,
        session_id=session_id,
        message=f"Conversation session '{session_id}' was successfully deleted.",
    )
