"""
api/routes/chat.py
Chat routes for LocalMind-RAG.
Supports both synchronous POST /api/chat and streaming SSE POST /api/chat/stream.
"""

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from P3.agent_pipeline import AgentPipeline
from api.dependencies import get_pipeline
from api.schemas.chat import ChatRequest, ChatResponse
from api.services.chat_service import execute_chat, stream_chat

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute a synchronous chat query",
    description="Run a user query through the canonical multi-agent RAG pipeline, returning the grounded answer, citations, and quality metrics.",
)
async def chat_endpoint(
    request: ChatRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
) -> ChatResponse:
    """
    Handle POST /api/chat.
    Uses the canonical run_query execution path via the chat service.
    """
    return await execute_chat(request=request, pipeline=pipeline)


@router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Execute a streaming chat query with real-time progress",
    description="Stream agent lifecycle and retrieval progress events in real-time using Server-Sent Events (SSE).",
)
async def chat_stream_endpoint(
    request: ChatRequest,
    pipeline: AgentPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """
    Handle POST /api/chat/stream using Server-Sent Events (SSE).
    Streams safe progress events throughout the exact same single RAG execution lifecycle.
    """
    return StreamingResponse(
        stream_chat(request=request, pipeline=pipeline),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
