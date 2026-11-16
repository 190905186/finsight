from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.retrieval import HybridRetriever
from app.tools.decimal_metrics import MetricOp, calculate_metric


def search_filings(
    session: Session,
    query: str,
    *,
    company: str | None = None,
    doc_name: str | None = None,
    period: str | None = None,
    limit: int = 8,
    retriever: HybridRetriever | None = None,
) -> list[dict[str, Any]]:
    hybrid = retriever or HybridRetriever()
    hits = hybrid.retrieve(
        session,
        query,
        company=company,
        doc_name=doc_name,
        period=period,
        limit=limit,
    )
    return [hit.to_dict() for hit in hits]


def get_filing_metadata(session: Session, doc_name: str | None = None, company: str | None = None) -> list[dict[str, Any]]:
    stmt = select(Document)
    if doc_name:
        stmt = stmt.where(Document.doc_name == doc_name)
    if company:
        stmt = stmt.where(Document.company.ilike(company))
    docs = session.scalars(stmt.order_by(Document.doc_name).limit(100)).all()
    return [
        {
            "doc_name": doc.doc_name,
            "company": doc.company,
            "doc_type": doc.doc_type,
            "period": doc.period,
            "page_count": doc.page_count,
            "content_hash": doc.content_hash,
        }
        for doc in docs
    ]


REGISTERED_TOOLS = {
    "search_filings": search_filings,
    "get_filing_metadata": get_filing_metadata,
    "calculate_metric": calculate_metric,
}


def run_calculate_metric(op: MetricOp, **kwargs: Any) -> dict[str, Any]:
    return calculate_metric(op, **kwargs)
