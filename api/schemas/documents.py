"""
api/schemas/documents.py
Pydantic schemas for Document Ingestion, Listing, and Deletion endpoints.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    """Metadata summary of an ingested document."""
    filename: str = Field(..., description="Original filename of the document")
    file_type: str = Field(..., description="File extension / format, e.g. pdf, docx, txt")
    doc_category: str = Field(..., description="Category tag, e.g. technical, report, general")
    chunk_count: int = Field(..., description="Number of chunks created for this document")
    created_at: Optional[str] = Field(None, description="ISO timestamp of ingestion")


class DocumentListResponse(BaseModel):
    """Response model for GET /api/documents."""
    total_documents: int = Field(..., description="Total unique documents in vector store")
    total_chunks: int = Field(..., description="Total chunk vectors across all documents")
    documents: List[DocumentItem] = Field(default_factory=list, description="List of ingested documents")


class DocumentUploadResponse(BaseModel):
    """Response model for POST /api/documents/upload."""
    status: Literal["success", "error"] = Field(..., description="Ingestion outcome status")
    filename: str = Field(..., description="Sanitized filename of the ingested document")
    strategy: str = Field(..., description="Chunking strategy used, e.g. sentence, semantic")
    doc_category: str = Field(..., description="Document category assigned")
    chunks_created: int = Field(..., description="Number of new chunks created from this document")
    total_chunks_in_collection: int = Field(..., description="Total chunks in the vector collection after ingestion")
    graph_built: bool = Field(False, description="Whether Knowledge Graph triples were extracted")
    elapsed_seconds: float = Field(..., description="Processing time in seconds")
    message: Optional[str] = Field(None, description="Optional informational or error message")


class DocumentDeleteResponse(BaseModel):
    """Response model for DELETE /api/documents/{filename}."""
    success: bool = Field(..., description="Whether document was successfully deleted")
    filename: str = Field(..., description="Filename that was deleted")
    chunks_removed: int = Field(..., description="Number of chunk nodes removed")
    message: str = Field(..., description="Outcome message")
