"""
Loads PDF, TXT, and DOCX files and returns clean LlamaIndex Document objects
with rich metadata attached to each.

Key design decisions:
- PyMuPDF (fitz) for per-page PDF extraction (preserves page numbers in metadata)
- python-docx for DOCX (preserves paragraph structure)
- Regex post-processing to strip headers/footers/page numbers
- LlamaIndex SimpleDirectoryReader as the fallback for any other file type
- Every Document gets: filename, file_type, page_number, created_at, doc_category
"""

import os
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
from llama_index.core import Document

# Regex patterns to remove noise like page numbers, headers, footers, and horizontal rules.
patterns = [
    r"^\s*\d+\s*$",              # lone page numbers
    r"^page\s+\d+\s+of\s+\d+",  # "Page 1 of 10"
    r"^\s*[-–—]{3,}\s*$",        # horizontal rules
    r"^\s*©.*$",                  # copyright lines
]
regex = re.compile("|".join(patterns), re.IGNORECASE | re.MULTILINE)

def clean_text(text):
    """Remove noise from extracted text using regex patterns and collapses excess whitespace."""
    txt = regex.sub("", text)
    txt = re.sub(r"\n{3,}", "\n\n", txt)   # max 2consecutive newlines
    txt = re.sub(r" {2,}", " ", txt)       # collapse spaces
    return txt.strip()

def infer_doc_category(filename):
    """Assigns a category to each document based on its filename. Used as a simple heuristic to enrich metadata for better retrieval."""
    fname = Path(filename).name.lower()
    if any(k in fname for k in ["paper","research","arxiv","study"]):
        return "research_paper"
    
    elif any(k in fname for k in ["report", "annual", "quarterly", "financial"]):
        return "report"
    
    elif any(k in fname for k in ["manual", "spec", "technical", "api", "docs"]):
        return "technical"
    
    else:
        return "general"

def load_pdf(file_path, doc_category=None):
    """Extract text from a PDF file using PyMuPDF (fitz). Each page becomes a separate Document with page number in metadata.
    """
    import fitz  # PyMuPDF
    pdf = fitz.open(str(file_path))
    docs = []
    category = doc_category or infer_doc_category(file_path.name)
    for page_num, page in enumerate(pdf, start=1):
        raw_text = page.get_text("text")
        clean = clean_text(raw_text)
        if not clean:
            continue  # skip empty pages
 
        docs.append(Document(
            text=clean,
            metadata={
                "filename": file_path.name,
                "file_type": "pdf",
                "page_number": page_num,
                "total_pages": len(pdf),
                "source_path": str(file_path),
                "doc_category": category,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            # LlamaIndex uses doc_id for deduplication
            doc_id=f"{file_path.stem}_page_{page_num}",
        ))
 
    pdf.close()
    logger.info(f"[PDF] Loaded {len(docs)} pages from {file_path.name}")
    return docs

def load_docx(file_path, doc_category):
    """
    Extract text from a DOCX file using python-docx.
    Returns a single Document (DOCX has no concept of pages).
    """
    from docx import Document as DocxDocument
    docx = DocxDocument(str(file_path))
    paragraphs = [p.text for p in docx.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    clean = clean_text(full_text)
    category = doc_category or infer_doc_category(file_path.name)
 
    if not clean:
        logger.warning(f"[DOCX] Empty content in {file_path.name}")
        return []
 
    doc = Document(
        text=clean,
        metadata={
            "filename": file_path.name,
            "file_type": "docx",
            "page_number": 1,
            "source_path": str(file_path),
            "doc_category": category,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        doc_id=file_path.stem,
    )
    logger.info(f"[DOCX] Loaded {file_path.name} ({len(paragraphs)} paragraphs)")
    return [doc]

def load_txt(file_path, doc_category=None):
    """
    Load a plain text file as a single Document.
    """
    txt = file_path.read_text(encoding="utf-8", errors="ignore")
    clean = clean_text(txt)
    category = doc_category or infer_doc_category(file_path.name)
 
    if not clean:
        logger.warning(f"[TXT] Empty content in {file_path.name}")
        return []
 
    doc = Document(
        text=clean,
        metadata={
            "filename": file_path.name,
            "file_type": "txt",
            "page_number": 1,
            "source_path": str(file_path),
            "doc_category": category,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        doc_id=file_path.stem,
    )
    logger.info(f"[TXT] Loaded {file_path.name}")
    return [doc]

def load_document(file_path: Path, doc_category=None):
    """
    Main entry point to load a document. Determines file type and delegates to the appropriate loader function.
    If the file type is unsupported, it falls back to LlamaIndex's SimpleDirectoryReader which can handle a wide range of formats.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
 
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, doc_category)
    elif suffix in (".docx", ".doc"):
        return load_docx(path, doc_category)
    elif suffix == ".txt":
        return load_txt(path, doc_category)
    else:
        # Fallback to LlamaIndex's SimpleDirectoryReader for unsupported file types (e.g., HTML, EPUB, etc.)
        from llama_index.core import SimpleDirectoryReader

        logger.warning(f"Unsupported extension {suffix}, using SimpleDirectoryReader fallback")
        reader = SimpleDirectoryReader(input_files=[str(path)])
        return reader.load_data()

def load_directory(directory, doc_category=None,recursive=True):
    """
    Load all files in a directory and return a list of Documents Objects from all files.
    """
    dir = Path(directory)
    supported_exts = {".pdf", ".docx", ".doc", ".txt"}
    pattern = "**/*" if recursive else "*"
 
    all_docs = []
    for file_path in sorted(dir.glob(pattern)):
        if file_path.suffix.lower() in supported_exts and file_path.is_file():
            try:
                docs = load_document(file_path, doc_category)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
 
    logger.info(f"[Loader] Total documents loaded: {len(all_docs)}")
    return all_docs


# Quick Test : python document_loader.py ../data
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
 
    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path_to_file_or_directory>")
        sys.exit(1)
 
    target = sys.argv[1]
    if Path(target).is_dir():
        docs = load_directory(target)
    else:
        docs = load_document(target)
 
    print(f"\nLoaded {len(docs)} document(s)\n")
    for i, doc in enumerate(docs[:3]):
        print(f"--- Doc {i+1} ---")
        print(f"  Metadata : {doc.metadata}")
        print(f"  Preview  : {doc.text[:200].replace(chr(10), ' ')}...")
        print()
 