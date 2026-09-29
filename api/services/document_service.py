"""
api/services/document_service.py
Service adapting document upload, ingestion, and listing to the existing P1 engine.
"""

import asyncio
import logging
import os
import re
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, status

from P1.ingestion import ingest
from api.schemas.documents import (
    DocumentDeleteResponse,
    DocumentItem,
    DocumentListResponse,
    DocumentUploadResponse,
)

logger = logging.getLogger(__name__)

# Allowed document file extensions genuinely supported by P1 document_loader.py
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

# Maximum allowed file size in bytes (25 MB default, configurable via env)
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")) * 1024 * 1024

# Lock to synchronize concurrent ingestions and pipeline refreshes
_ingestion_lock = asyncio.Lock()


def refresh_pipeline(app: FastAPI):
    """
    Refresh the cached AgentPipeline on app.state after new documents are ingested.
    Reloads chunk nodes and vector index, constructing a fresh RetrievalContext and compiled LangGraph.
    """
    try:
        from app_context import EMBED_MODEL, LLM
        from P1.node_store import load_nodes
        from P1.vector_store import load_index
        from P3.agent_pipeline import build_agent_pipeline

        node_candidates = [
            Path("vectorstore/chunks_nodes.pkl"),
            Path("vectorstore/nodes.pkl"),
        ]
        node_path = next((p for p in node_candidates if p.exists()), None)

        if node_path:
            logger.info(f"[DocumentService] Reloading nodes from {node_path}...")
            payload = load_nodes(str(node_path))
            nodes = payload.get("nodes", [])

            logger.info("[DocumentService] Reloading ChromaDB 'chunks' index...")
            vector_index = load_index(collection_name="chunks", embed_model=EMBED_MODEL)

            logger.info("[DocumentService] Recompiling LangGraph AgentPipeline with updated evidence...")
            pipeline = build_agent_pipeline(
                nodes=nodes,
                vector_index=vector_index,
                embed_model=EMBED_MODEL,
                llm=LLM,
            )

            app.state.pipeline = pipeline
            app.state.retrieval_ctx = pipeline.retrieval_ctx
            logger.info(f"[DocumentService] Pipeline refreshed successfully ({len(nodes)} total nodes).")
        else:
            logger.warning("[DocumentService] No node store found during pipeline refresh.")
    except Exception as e:
        logger.error(f"[DocumentService] Pipeline refresh failed: {e}", exc_info=True)


def sanitize_filename(filename: str) -> str:
    """
    Strip path traversal and special characters, returning a safe ASCII basename.
    """
    base = Path(filename).name
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    return cleaned or "document.txt"


async def ingest_document(
    file: UploadFile,
    strategy: str,
    doc_category: Optional[str],
    build_graph: bool,
    app: FastAPI,
) -> DocumentUploadResponse:
    """
    Validate, stage, and ingest an uploaded document into the P1 RAG engine.
    Refreshes the FastAPI app.state.pipeline so subsequent chat requests see the new document.
    """
    # 1. Validate filename and extension
    original_name = file.filename or "unnamed_document"
    safe_name = sanitize_filename(original_name)
    ext = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported document format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Stage file to temporary location with size check
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"

    total_bytes = 0
    try:
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
                    )
                buffer.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded document is empty (0 bytes).",
            )

        # 3. Execute canonical P1 ingestion inside synchronized lock
        async with _ingestion_lock:
            from app_context import EMBED_MODEL

            logger.info(f"[DocumentService] Starting P1 ingestion for {safe_name} ({total_bytes} bytes)...")

            raw_result = await asyncio.to_thread(
                ingest,
                path=temp_path,
                strategy=strategy,
                doc_category=doc_category,
                embed_model=EMBED_MODEL,
                multi_vector_enabled=True,
                build_graph=build_graph,
            )

            if raw_result.get("status") == "error":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Ingestion failed: {raw_result.get('message', 'Unknown error')}",
                )

            # 4. Refresh application pipeline state
            refresh_pipeline(app)

            return DocumentUploadResponse(
                status="success",
                filename=safe_name,
                strategy=strategy,
                doc_category=doc_category or "general",
                chunks_created=int(raw_result.get("chunks_created", 0)),
                total_chunks_in_collection=int(raw_result.get("total_vectors_in_collection", 0)),
                graph_built=bool(raw_result.get("graph_built", False)),
                elapsed_seconds=float(raw_result.get("elapsed_seconds", 0.0)),
                message=f"Document '{safe_name}' successfully ingested and indexed into RAG pipeline.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DocumentService] Ingestion failed unexpectedly: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during document ingestion.",
        )
    finally:
        # Clean up temporary staging file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"[DocumentService] Failed to clean up staging file {temp_path}: {e}")


async def list_documents(app: FastAPI) -> DocumentListResponse:
    """
    List all indexed documents, grouping chunk nodes by document filename.
    """
    try:
        from P1.node_store import load_nodes
        from P1.vector_store import collection_count

        node_candidates = [
            Path("vectorstore/chunks_nodes.pkl"),
            Path("vectorstore/nodes.pkl"),
        ]
        node_path = next((p for p in node_candidates if p.exists()), None)

        if not node_path:
            return DocumentListResponse(total_documents=0, total_chunks=0, documents=[])

        payload = load_nodes(str(node_path))
        nodes = payload.get("nodes", [])

        # Group metadata by filename
        doc_map = defaultdict(lambda: {"chunk_count": 0, "file_type": "unknown", "doc_category": "general", "created_at": None})

        for node in nodes:
            meta = getattr(node, "metadata", {}) or {}
            fname = meta.get("filename", "unknown_document")
            doc_map[fname]["chunk_count"] += 1
            doc_map[fname]["file_type"] = meta.get("file_type", Path(fname).suffix.lstrip(".") or "txt")
            doc_map[fname]["doc_category"] = meta.get("doc_category", "general")
            if not doc_map[fname]["created_at"] and meta.get("created_at"):
                doc_map[fname]["created_at"] = meta.get("created_at")

        doc_items = [
            DocumentItem(
                filename=fname,
                file_type=info["file_type"],
                doc_category=info["doc_category"],
                chunk_count=info["chunk_count"],
                created_at=info["created_at"],
            )
            for fname, info in sorted(doc_map.items())
        ]

        total_vectors = collection_count("chunks")

        return DocumentListResponse(
            total_documents=len(doc_items),
            total_chunks=total_vectors if total_vectors > 0 else len(nodes),
            documents=doc_items,
        )

    except Exception as e:
        logger.error(f"[DocumentService] List documents error: {e}", exc_info=True)
        return DocumentListResponse(total_documents=0, total_chunks=0, documents=[])


async def delete_document(filename: str, app: FastAPI) -> DocumentDeleteResponse:
    """
    Remove an ingested document from the knowledge base:
    1. Removes all nodes with matching filename from vectorstore/chunks_nodes.pkl.
    2. Deletes vectors from ChromaDB 'chunks' and 'chunks_summary' collections.
    3. Rebuilds and persists the BM25 index over the remaining nodes.
    4. Refreshes the in-memory RAG pipeline on app.state.
    5. Cleans up any staging file in data/uploads/ or data/.
    """
    async with _ingestion_lock:
        try:
            from P1.node_store import load_nodes, save_nodes
            from P1.vector_store import get_chroma_client
            from P2.sparse_retriever import build_bm25_index

            node_candidates = [
                Path("vectorstore/chunks_nodes.pkl"),
                Path("vectorstore/nodes.pkl"),
            ]
            node_path = next((p for p in node_candidates if p.exists()), None)

            if not node_path:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{filename}' not found in knowledge base.",
                )

            payload = load_nodes(str(node_path))
            strategy = payload.get("strategy", "sentence")
            all_nodes = payload.get("nodes", [])

            remaining_nodes = []
            removed_count = 0
            for node in all_nodes:
                meta = getattr(node, "metadata", {}) or {}
                node_fname = meta.get("filename", "")
                if node_fname == filename or node_fname.endswith(f"_{filename}"):
                    removed_count += 1
                else:
                    remaining_nodes.append(node)

            if removed_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{filename}' not found in knowledge base.",
                )

            # 1. Save updated nodes.pkl
            save_nodes(remaining_nodes, strategy, path=str(node_path))

            # 2. Rebuild BM25 index
            if remaining_nodes:
                build_bm25_index(remaining_nodes, persist=True)
            else:
                bm25_path = Path("vectorstore/bm25")
                if bm25_path.exists():
                    import shutil
                    shutil.rmtree(bm25_path, ignore_errors=True)

            # 3. Delete from ChromaDB
            try:
                chroma_client = get_chroma_client()
                for coll_name in ["chunks", "chunks_summary"]:
                    try:
                        coll = chroma_client.get_collection(coll_name)
                        coll.delete(where={"filename": filename})
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[DocumentService] Chroma delete error: {e}")

            # 4. Refresh application pipeline state
            refresh_pipeline(app)

            # 5. Clean up staging file if present
            upload_candidate = Path("data/uploads") / filename
            if upload_candidate.exists():
                try:
                    upload_candidate.unlink()
                except Exception:
                    pass

            return DocumentDeleteResponse(
                success=True,
                filename=filename,
                chunks_removed=removed_count,
                message=f"Document '{filename}' ({removed_count} chunks) successfully removed from knowledge base.",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[DocumentService] Delete document error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete document: {str(e)}",
            )

