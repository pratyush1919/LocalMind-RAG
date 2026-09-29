"""
api/routes/documents.py
Document upload, ingestion, listing, and deletion routes for LocalMind-RAG.
"""

from typing import Literal, Optional
from fastapi import APIRouter, File, Form, Request, UploadFile, status

from api.schemas.documents import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from api.services.document_service import (
    delete_document,
    ingest_document,
    list_documents,
)

router = APIRouter()


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest a document into the RAG pipeline",
    description="Accepts a multipart file (.pdf, .docx, .doc, .txt, .md), chunks and embeds it, updating the vector index.",
)
async def upload_document_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Document file to ingest"),
    strategy: Literal["sentence", "parent_child", "semantic"] = Form("sentence"),
    doc_category: Optional[str] = Form(None),
    build_graph: bool = Form(False),
) -> DocumentUploadResponse:
    """
    Handle multipart/form-data document upload.
    Ingests file via the canonical P1 pipeline and refreshes the live AgentPipeline.
    """
    return await ingest_document(
        file=file,
        strategy=strategy,
        doc_category=doc_category,
        build_graph=build_graph,
        app=request.app,
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all indexed documents",
    description="Retrieve a summary of all unique documents currently indexed in the vector store.",
)
async def list_documents_endpoint(
    request: Request,
) -> DocumentListResponse:
    """
    Handle GET /api/documents.
    Returns list of indexed documents and their chunk counts.
    """
    return await list_documents(app=request.app)


@router.delete(
    "/documents/{filename:path}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete an indexed document",
    description="Deletes all indexed chunk vectors and metadata associated with the specified filename.",
)
async def delete_document_endpoint(
    filename: str,
    request: Request,
) -> DocumentDeleteResponse:
    """
    Handle DELETE /api/documents/{filename}.
    Removes document chunks from ChromaDB, BM25, and node store.
    """
    return await delete_document(filename=filename, app=request.app)
