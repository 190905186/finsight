from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.deps import require_auth

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile | None = None, user: dict = Depends(require_auth)) -> dict:
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
    return {"document_id": "doc-1", "filename": file.filename, "status": "ingested"}


@router.get("")
async def list_documents(user: dict = Depends(require_auth)) -> dict:
    return {"items": [{"id": "doc-1", "name": "Annual Report FY2025.pdf", "type": "annual_report"}]}


@router.get("/{document_id}")
async def get_document(document_id: str, user: dict = Depends(require_auth)) -> dict:
    return {"id": document_id, "status": "available", "metadata": {"document_type": "annual_report"}}


@router.delete("/{document_id}")
async def delete_document(document_id: str, user: dict = Depends(require_auth)) -> dict:
    return {"deleted": True, "document_id": document_id}


@router.post("/reindex")
async def reindex_documents(user: dict = Depends(require_auth)) -> dict:
    return {"reindexed": True, "documents": 12}
