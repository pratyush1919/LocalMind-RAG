"""
api/services/chat_service.py
Service adapting HTTP Chat requests to the canonical LocalMind-RAG execution engine.
Supports both synchronous collection and real-time SSE progress streaming.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import HTTPException, status

from P3.agent_pipeline import AgentPipeline, run_query, run_query_stream
from api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CritiqueSummary,
    TslSummary,
)

logger = logging.getLogger(__name__)


def _map_raw_to_chat_response(raw_result: dict, request: ChatRequest) -> ChatResponse:
    """Map raw pipeline result dictionary into a validated ChatResponse model."""
    critique_data = raw_result.get("critique")
    critique_summary = None
    if critique_data and isinstance(critique_data, dict) and "faithfulness" in critique_data:
        critique_summary = CritiqueSummary(
            faithfulness=int(critique_data.get("faithfulness", 0)),
            completeness=int(critique_data.get("completeness", 0)),
            reasoning_quality=int(critique_data.get("reasoning_quality", 0)),
            issues=list(critique_data.get("issues", [])),
            summary=critique_data.get("summary"),
        )

    tsl_data = raw_result.get("tsl")
    tsl_summary = None
    if tsl_data and isinstance(tsl_data, dict):
        compliance_data = tsl_data.get("compliance") or {}
        compliance_passed = True
        if isinstance(compliance_data, dict):
            compliance_passed = bool(compliance_data.get("passed", True))

        tsl_summary = TslSummary(
            risk_score=float(tsl_data.get("risk_score", 0.0)),
            pii_found=bool(tsl_data.get("pii_found", False)),
            leakage_detected=bool(tsl_data.get("leakage_detected", False)),
            low_confidence=bool(tsl_data.get("low_confidence", False)),
            compliance_passed=compliance_passed,
        )

    return ChatResponse(
        answer=raw_result.get("answer", ""),
        session_id=raw_result.get("session_id", request.session_id or ""),
        run_id=raw_result.get("run_id", ""),
        user_role=raw_result.get("user_role", request.user_role),
        query_type=raw_result.get("query_type"),
        role_used=raw_result.get("role_used"),
        strategy=raw_result.get("strategy"),
        consensus_score=float(raw_result.get("consensus_score", 0.0)),
        is_final=bool(raw_result.get("is_final", False)),
        iterations=int(raw_result.get("iterations", 0)),
        graph_rag_used=bool(raw_result.get("graph_rag_used", False)),
        web_research_used=bool(raw_result.get("web_research_used", False)),
        sources=list(raw_result.get("sources", [])),
        source_details=list(raw_result.get("source_details", [])),
        citations=list(raw_result.get("citations", [])),
        critique=critique_summary,
        tsl=tsl_summary,
        guardrail_blocked=bool(raw_result.get("guardrail_blocked", False)),
        guardrail_reason=raw_result.get("guardrail_reason"),
    )


async def execute_chat(
    request: ChatRequest,
    pipeline: AgentPipeline,
) -> ChatResponse:
    """
    Execute a synchronous chat query using the canonical RAG execution pipeline.
    """
    logger.info(
        f"[ChatService] Executing synchronous query | session_id={request.session_id} | "
        f"role={request.user_role} | web_research={request.use_web_research} | query='{request.query[:60]}'"
    )

    try:
        raw_result = await asyncio.to_thread(
            run_query,
            query=request.query,
            pipeline=pipeline,
            session_id=request.session_id,
            user_role=request.user_role,
            use_memory=request.use_memory,
            use_web_research=request.use_web_research,
        )
    except Exception as e:
        logger.error(f"[ChatService] Canonical run_query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while executing the multi-agent RAG pipeline.",
        )

    response = _map_raw_to_chat_response(raw_result, request)
    try:
        from api.services.session_service import record_turn
        record_turn(
            session_id=response.session_id,
            query=request.query,
            answer=response.answer,
            citations=response.citations,
            consensus_score=response.consensus_score,
        )
    except Exception as e:
        logger.warning(f"[ChatService] Could not record turn to session store: {e}")

    return response


async def stream_chat(
    request: ChatRequest,
    pipeline: AgentPipeline,
) -> AsyncGenerator[str, None]:
    """
    Stream SSE events from the canonical run_query_stream generator.
    Runs the synchronous generator in a worker thread and yields real-time formatted SSE messages.
    """
    logger.info(
        f"[ChatService] Starting SSE stream | session_id={request.session_id} | "
        f"role={request.user_role} | query='{request.query[:60]}'"
    )

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    _DONE = object()

    def worker():
        try:
            for event in run_query_stream(
                query=request.query,
                pipeline=pipeline,
                session_id=request.session_id,
                user_role=request.user_role,
                use_memory=request.use_memory,
                use_web_research=request.use_web_research,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as e:
            logger.error(f"[ChatService] Stream worker exception: {e}", exc_info=True)
            error_event = {
                "event": "error",
                "data": {
                    "status": "error",
                    "message": "An internal error occurred while executing the multi-agent RAG pipeline.",
                },
            }
            loop.call_soon_threadsafe(queue.put_nowait, error_event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    loop.run_in_executor(None, worker)

    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break

            event_name = item.get("event", "message")
            event_data = item.get("data", {})

            if event_name == "final" and isinstance(event_data, dict):
                chat_resp = _map_raw_to_chat_response(event_data, request)
                event_data = chat_resp.model_dump()
                try:
                    from api.services.session_service import record_turn
                    record_turn(
                        session_id=chat_resp.session_id,
                        query=request.query,
                        answer=chat_resp.answer,
                        citations=chat_resp.citations,
                        consensus_score=chat_resp.consensus_score,
                    )
                except Exception as e:
                    logger.warning(f"[ChatService] Could not record turn to session store: {e}")

            yield f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"
    except asyncio.CancelledError:
        logger.info("[ChatService] Client disconnected from SSE stream.")
        raise
