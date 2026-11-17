from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.config import settings
from app.rag.chunking import CHUNKING_PROFILES, ChunkingProfile, chunk_pdf
from app.rag.embeddings import EmbeddingClient, embed_texts
from app.rag.store import PgVectorStore


def load_document_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_name = row.get("doc_name")
            if not isinstance(doc_name, str) or not doc_name:
                raise ValueError(f"Missing doc_name in metadata line {line_number}")
            metadata[doc_name] = row
    return metadata


class PDFIngestor:
    def __init__(
        self,
        store: PgVectorStore,
        client: EmbeddingClient,
        *,
        embedding_model: str = settings.embeddings_model,
        dimensions: int = settings.embeddings_dimensions,
        batch_size: int = settings.rag_embedding_batch_size,
    ) -> None:
        self.store = store
        self.client = client
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.batch_size = batch_size

    def ingest_pdf(
        self,
        pdf_path: Path,
        profile: ChunkingProfile,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        chunks = chunk_pdf(pdf_path, profile, metadata)
        embeddings = embed_texts(
            self.client,
            [chunk.text for chunk in chunks],
            model=self.embedding_model,
            dimensions=self.dimensions,
            batch_size=self.batch_size,
        )
        self.store.replace_document(chunks, embeddings, self.embedding_model)
        return len(chunks)

    def ingest_directory(
        self,
        pdf_dir: Path,
        profile: ChunkingProfile,
        metadata_by_doc: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, int]:
        if not pdf_dir.is_dir():
            raise NotADirectoryError(f"PDF corpus directory does not exist: {pdf_dir}")
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            raise ValueError(f"No PDF documents found in: {pdf_dir}")

        results: dict[str, int] = {}
        for pdf_path in pdf_paths:
            document_metadata = (metadata_by_doc or {}).get(pdf_path.stem)
            results[pdf_path.name] = self.ingest_pdf(
                pdf_path, profile, document_metadata
            )
        return results


def load_configured_document_metadata() -> dict[str, dict[str, Any]]:
    return load_document_metadata(settings.financebench_metadata_path)


def configured_profiles(names: Sequence[str]) -> list[ChunkingProfile]:
    profile_map = {profile.name: profile for profile in CHUNKING_PROFILES}
    unknown = sorted(set(names) - profile_map.keys())
    if unknown:
        raise ValueError(f"Unknown chunking profile(s): {', '.join(unknown)}")
    return [profile_map[name] for name in names]
