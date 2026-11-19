"""KYC document ingest routes.

Completely separate from the analytics knowledge route (src/routes/knowledge.py)
which uses vector-store / RAG. KYC Phase 1 uses full-document context via
Gemini's implicit caching — no chunking, no embeddings.

POST /kyc/ingest           — convert files to Markdown via docling, store locally,
                             return {doc_id, filename, char_count, markdown} per file
GET  /kyc/docs             — list stored docs (for BE to sync its KnowledgeDoc state)
DELETE /kyc/{doc_id}       — remove a doc by its doc_id
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.services.know_you_company.doc_loader import delete_doc, list_docs, save_doc
from src.utils.logging_utils import console_log

router = APIRouter(prefix="/kyc", tags=["KYC"])

_SUPPORTED = {".doc", ".docx", ".pdf", ".txt", ".md"}
_MODE = "know_your_company"


def _convert_to_markdown(filename: str, content: bytes) -> str:
    """Convert uploaded file bytes to Markdown.

    .md files are returned as-is (decoded UTF-8).
    All other supported types (.txt, .doc, .docx, .pdf) are processed via docling.
    """
    ext = Path(filename).suffix.lower()
    if ext not in _SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(_SUPPORTED))}",
        )

    if ext == ".md":
        return content.decode("utf-8", errors="replace")

    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="docling is not installed — run: pip install docling",
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        markdown = result.document.export_to_markdown()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not convert '{filename}': {exc}",
        ) from exc
    finally:
        os.unlink(tmp_path)

    return markdown


@router.post("/ingest")
async def ingest_documents(
    files: list[UploadFile] = File(...),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    ingested = []
    for upload in files:
        filename = upload.filename or "document.pdf"
        content = await upload.read()

        try:
            markdown = _convert_to_markdown(filename, content)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not process '{filename}': {exc}",
            ) from exc

        if not markdown.strip():
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}' produced no readable text after conversion",
            )

        doc_id = uuid.uuid4().hex
        save_doc(doc_id, filename, markdown)

        console_log(
            _MODE, "KYC_INGEST",
            f"doc_id={doc_id} filename={filename!r} chars={len(markdown)}",
        )
        ingested.append({
            "doc_id": doc_id,
            "filename": filename,
            "char_count": len(markdown),
            "markdown": markdown,
        })

    return {"success": True, "service_name": _MODE, "documents": ingested}


@router.get("/docs")
async def list_documents() -> dict:
    docs = list_docs()
    return {"service": _MODE, "documents": docs}


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    removed = delete_doc(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    console_log(_MODE, "KYC_DELETE", f"doc_id={doc_id}")
    return {"success": True, "removed": doc_id}
