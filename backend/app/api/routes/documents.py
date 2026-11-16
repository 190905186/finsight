from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select

from app.api.deps import require_auth
from app.services.memory_index import get_memory_index, load_docs_into_memory
from app.services.runtime import db_available

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents(user: dict = Depends(require_auth)) -> dict:
    if not db_available():
        index = get_memory_index()
        return {
            "items": index.documents,
            "indexed_filing_count": len(index.documents),
            "chunk_count": len(index.chunks),
            "index_version": index.index_version,
            "embedding_model": index.embedding_model,
            "mode": "memory",
        }

    from app.db.models import Document, IndexMeta
    from app.db.session import get_session

    with get_session() as session:
        docs = session.scalars(select(Document).order_by(Document.doc_name)).all()
        meta = session.scalar(select(IndexMeta).limit(1))
        return {
            "items": [
                {
                    "id": doc.id,
                    "name": doc.doc_name,
                    "company": doc.company,
                    "type": doc.doc_type,
                    "period": doc.period,
                    "page_count": doc.page_count,
                }
                for doc in docs
            ],
            "indexed_filing_count": len(docs),
            "chunk_count": meta.chunk_count if meta else 0,
            "index_version": meta.index_version if meta else None,
            "embedding_model": meta.embedding_model if meta else None,
            "mode": "postgres",
        }


@router.get("/companies")
async def list_companies(user: dict = Depends(require_auth)) -> dict:
    if not db_available():
        companies = sorted({doc.get("company") for doc in get_memory_index().documents if doc.get("company")})
        return {"items": companies}

    from app.db.models import Document
    from app.db.session import get_session

    with get_session() as session:
        companies = session.scalars(select(Document.company).distinct().order_by(Document.company)).all()
        return {"items": list(companies)}


@router.get("/{document_id}")
async def get_document(document_id: int, user: dict = Depends(require_auth)) -> dict:
    if not db_available():
        doc = next((item for item in get_memory_index().documents if item.get("id") == document_id), None)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {**doc, "chunk_count": sum(1 for chunk in get_memory_index().chunks if chunk.doc_name == doc["name"])}

    from app.db.models import Chunk, Document
    from app.db.session import get_session

    with get_session() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        chunk_count = session.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id))
        return {
            "id": doc.id,
            "name": doc.doc_name,
            "company": doc.company,
            "type": doc.doc_type,
            "period": doc.period,
            "page_count": doc.page_count,
            "chunk_count": chunk_count or 0,
            "content_hash": doc.content_hash,
            "metadata": doc.metadata_json,
        }


@router.post("/reindex")
async def reindex_documents(payload: dict | None = None, user: dict = Depends(require_auth)) -> dict:
    docs = (payload or {}).get("docs")
    if not db_available():
        names = docs or ["3M_2018_10K"]
        index = load_docs_into_memory(names)
        return {
            "reindexed": True,
            "mode": "memory",
            "documents": len(index.documents),
            "chunks": len(index.chunks),
        }

    from app.services.ingestion import run_ingestion

    summary = run_ingestion(only_doc_names=docs)
    return {
        "reindexed": True,
        "mode": "postgres",
        "documents": summary.get("document_count"),
        "chunks": summary.get("chunk_count"),
        "summary": {k: v for k, v in summary.items() if k != "results"},
    }
