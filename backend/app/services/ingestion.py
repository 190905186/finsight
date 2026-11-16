from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Chunk, Document, IndexMeta
from app.db.session import get_session, init_db
from app.services.chunking import PageText, chunk_pages
from app.services.embeddings_client import EmbeddingsClient

logger = logging.getLogger(__name__)


def _repo_financebench_root() -> Path:
    return Path(settings.financebench_root)


def load_document_metadata() -> dict[str, dict[str, Any]]:
    path = _repo_financebench_root() / "data" / "financebench_document_information.jsonl"
    meta: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return meta
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta[row["doc_name"]] = row
    return meta


def parse_pdf_pages(pdf_path: Path) -> list[PageText]:
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(PageText(page_index=index, text=text))
    return pages


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_uid(doc_name: str, page_start: int, page_end: int, content_hash: str) -> str:
    raw = f"{doc_name}:{page_start}:{page_end}:{content_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def upsert_document(
    session: Session,
    *,
    doc_name: str,
    meta: dict[str, Any],
    pdf_path: Path,
    page_count: int,
    content_hash: str,
) -> Document:
    existing = session.scalar(select(Document).where(Document.doc_name == doc_name))
    period = meta.get("doc_period")
    period_str = str(period) if period is not None else None
    if existing and existing.content_hash == content_hash:
        return existing
    if existing:
        session.query(Chunk).filter(Chunk.document_id == existing.id).delete()
        existing.company = meta.get("company") or existing.company
        existing.doc_type = str(meta.get("doc_type") or existing.doc_type)
        existing.period = period_str
        existing.content_hash = content_hash
        existing.page_count = page_count
        existing.source_path = str(pdf_path)
        existing.metadata_json = meta
        session.flush()
        return existing
    document = Document(
        doc_name=doc_name,
        company=meta.get("company") or doc_name.split("_")[0],
        doc_type=str(meta.get("doc_type") or "10k"),
        period=period_str,
        content_hash=content_hash,
        page_count=page_count,
        source_path=str(pdf_path),
        metadata_json=meta,
    )
    session.add(document)
    session.flush()
    return document


def ingest_pdf(
    session: Session,
    pdf_path: Path,
    meta: dict[str, Any],
    embedder: EmbeddingsClient,
    *,
    batch_size: int = 16,
) -> dict[str, Any]:
    doc_name = pdf_path.stem
    content_hash = file_sha256(pdf_path)
    pages = parse_pdf_pages(pdf_path)
    drafts = chunk_pages(pages)
    document = upsert_document(
        session,
        doc_name=doc_name,
        meta=meta or {"company": doc_name.split("_")[0], "doc_type": "10k"},
        pdf_path=pdf_path,
        page_count=len(pages),
        content_hash=content_hash,
    )

    # Skip re-embed if unchanged chunks already present.
    existing_count = session.query(Chunk).filter(Chunk.document_id == document.id).count()
    if existing_count and document.content_hash == content_hash:
        return {
            "doc_name": doc_name,
            "pages": len(pages),
            "chunks": existing_count,
            "status": "skipped_unchanged",
        }

    session.query(Chunk).filter(Chunk.document_id == document.id).delete()
    session.flush()

    created = 0
    for start in range(0, len(drafts), batch_size):
        batch = drafts[start : start + batch_size]
        vectors = embedder.embed([item.content for item in batch])
        for draft, vector in zip(batch, vectors):
            chunk = Chunk(
                document_id=document.id,
                chunk_uid=_chunk_uid(doc_name, draft.page_start, draft.page_end, draft.content_hash),
                company=document.company,
                doc_name=doc_name,
                doc_type=document.doc_type,
                period=document.period,
                section=draft.section,
                page_start=draft.page_start,
                page_end=draft.page_end,
                content=draft.content,
                content_hash=draft.content_hash,
                token_estimate=draft.token_estimate,
                embedding=vector,
            )
            session.add(chunk)
            created += 1
        session.flush()

    return {"doc_name": doc_name, "pages": len(pages), "chunks": created, "status": "indexed"}


def refresh_index_meta(session: Session) -> IndexMeta:
    chunk_count = session.query(Chunk).count()
    document_count = session.query(Document).count()
    meta = session.scalar(select(IndexMeta).limit(1))
    if meta is None:
        meta = IndexMeta(
            embedding_model=settings.embeddings_model,
            embedding_dim=settings.embeddings_dim,
            index_version=settings.index_version,
            chunk_count=chunk_count,
            document_count=document_count,
        )
        session.add(meta)
    else:
        if meta.embedding_model != settings.embeddings_model or meta.embedding_dim != settings.embeddings_dim:
            raise ValueError(
                f"Index embedding mismatch: index={meta.embedding_model}/{meta.embedding_dim}, "
                f"config={settings.embeddings_model}/{settings.embeddings_dim}"
            )
        meta.chunk_count = chunk_count
        meta.document_count = document_count
        meta.index_version = settings.index_version
    session.flush()
    return meta


def iter_pdf_paths(only_doc_names: Iterable[str] | None = None) -> list[Path]:
    pdf_dir = _repo_financebench_root() / "pdfs"
    names = set(only_doc_names or [])
    paths = sorted(pdf_dir.glob("*.pdf"))
    if names:
        paths = [path for path in paths if path.stem in names]
    return paths


def run_ingestion(only_doc_names: list[str] | None = None, batch_size: int = 16) -> dict[str, Any]:
    init_db(settings.embeddings_dim)
    metadata = load_document_metadata()
    results: list[dict[str, Any]] = []
    with EmbeddingsClient() as embedder, get_session() as session:
        for pdf_path in iter_pdf_paths(only_doc_names):
            meta = metadata.get(pdf_path.stem, {})
            logger.info("Ingesting %s", pdf_path.name)
            try:
                result = ingest_pdf(session, pdf_path, meta, embedder, batch_size=batch_size)
                results.append(result)
                session.commit()
            except Exception as exc:  # noqa: BLE001 - resume across documents
                session.rollback()
                logger.exception("Failed ingesting %s", pdf_path.name)
                results.append({"doc_name": pdf_path.stem, "status": "error", "error": str(exc)})
        index_meta = refresh_index_meta(session)
        summary = {
            "documents": len(results),
            "indexed": sum(1 for item in results if item.get("status") == "indexed"),
            "skipped": sum(1 for item in results if item.get("status") == "skipped_unchanged"),
            "errors": sum(1 for item in results if item.get("status") == "error"),
            "chunk_count": index_meta.chunk_count,
            "document_count": index_meta.document_count,
            "embedding_model": index_meta.embedding_model,
            "embedding_dim": index_meta.embedding_dim,
            "index_version": index_meta.index_version,
            "results": results,
        }
    return summary


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest FinanceBench PDFs into Postgres")
    parser.add_argument("--doc", action="append", dest="docs", help="Limit to doc_name (repeatable)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    summary = run_ingestion(only_doc_names=args.docs, batch_size=args.batch_size)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
