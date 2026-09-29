"""
api/main.py
FastAPI application entry point for LocalMind-RAG.
Provides lifespan startup/shutdown, CORS configuration, and route registration.
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.documents import router as document_router
from api.routes.sessions import router as session_router
from api.routes.graph import router as graph_router

load_dotenv()
logger = logging.getLogger("localmind_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager.
    Initializes shared heavy resources (embedding model, vector store, and LangGraph AgentPipeline)
    once on startup and stores them on app.state to avoid per-request recreation.
    """
    logger.info("[Lifespan] Starting LocalMind-RAG API...")
    
    app.state.pipeline = None
    app.state.retrieval_ctx = None

    try:
        from app_context import EMBED_MODEL, LLM
        from P1.node_store import load_nodes
        from P1.vector_store import load_index
        from P3.agent_pipeline import build_agent_pipeline

        # Locate chunk nodes
        node_candidates = [
            Path("vectorstore/chunks_nodes.pkl"),
            Path("vectorstore/nodes.pkl"),
        ]
        node_path = next((p for p in node_candidates if p.exists()), None)

        if node_path:
            logger.info(f"[Lifespan] Loading persisted nodes from {node_path}...")
            payload = load_nodes(str(node_path))
            nodes = payload.get("nodes", [])

            logger.info(f"[Lifespan] Loading ChromaDB 'chunks' index...")
            vector_index = load_index(collection_name="chunks", embed_model=EMBED_MODEL)

            logger.info("[Lifespan] Compiling 7-agent LangGraph pipeline...")
            pipeline = build_agent_pipeline(
                nodes=nodes,
                vector_index=vector_index,
                embed_model=EMBED_MODEL,
                llm=LLM,
            )

            app.state.pipeline = pipeline
            app.state.retrieval_ctx = pipeline.retrieval_ctx
            logger.info("[Lifespan] Multi-Agent RAG Pipeline is ready.")
        else:
            logger.warning("[Lifespan] No persisted chunk nodes found. Pipeline running in uninitialized mode until documents are ingested.")

    except Exception as e:
        logger.error(f"[Lifespan] Failed to initialize RAG pipeline: {e}", exc_info=True)

    yield

    logger.info("[Lifespan] Shutting down LocalMind-RAG API...")


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title="LocalMind-RAG API",
        version="1.0.0",
        description="FastAPI service for the LocalMind-RAG multi-agent retrieval engine.",
        lifespan=lifespan,
    )

    # Configure CORS for Next.js frontend
    raw_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    allowed_origins: List[str] = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(document_router, prefix="/api", tags=["documents"])
    app.include_router(session_router, prefix="/api", tags=["sessions"])
    app.include_router(graph_router, prefix="/api", tags=["graph"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
