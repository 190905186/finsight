from __future__ import annotations

from typing import Any, Mapping, Protocol

from app.rag.pipeline import RAGPipeline


class RAGAnswerer(Protocol):
    def answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> dict[str, Any]: ...


class RAGPromptResolver:
    """Resolve a natural-language prompt with retrieved evidence and citations."""

    def __init__(self, pipeline: RAGAnswerer | None = None) -> None:
        self.pipeline = pipeline if pipeline is not None else RAGPipeline()

    def resolve(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        company: str | None = None,
        doc_name: str | None = None,
    ) -> dict[str, Any]:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("prompt must not be empty")

        filters: dict[str, str] = {}
        if company is not None:
            normalized_company = company.strip()
            if not normalized_company:
                raise ValueError("company must not be empty")
            filters["company"] = normalized_company
        if doc_name is not None:
            normalized_doc_name = doc_name.strip()
            if not normalized_doc_name:
                raise ValueError("doc_name must not be empty")
            filters["doc_name"] = normalized_doc_name

        return self.pipeline.answer(
            normalized_prompt,
            top_k=top_k,
            filters=filters or None,
        )
