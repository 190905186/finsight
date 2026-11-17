from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

_EMBEDDING_TOKENIZER_ID = "BAAI/bge-small-en-v1.5"
_MAX_EMBEDDING_TOKENS = 500


@dataclass(frozen=True)
class ChunkingProfile:
    name: str
    chunk_size_chars: int
    chunk_overlap_chars: int

    def __post_init__(self) -> None:
        if self.chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be positive")
        if (
            self.chunk_overlap_chars < 0
            or self.chunk_overlap_chars >= self.chunk_size_chars
        ):
            raise ValueError(
                "chunk_overlap_chars must be non-negative and smaller than chunk_size_chars"
            )


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_name: str
    source_path: str
    page_number: int
    chunk_index: int
    profile: str
    text: str
    metadata: dict[str, Any]


CHUNKING_PROFILES = (
    ChunkingProfile("small", chunk_size_chars=512, chunk_overlap_chars=30),
    ChunkingProfile("medium", chunk_size_chars=1024, chunk_overlap_chars=30),
    ChunkingProfile("large", chunk_size_chars=1536, chunk_overlap_chars=30),
)


@lru_cache(maxsize=1)
def _load_embedding_tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(_EMBEDDING_TOKENIZER_ID)


def _embedding_token_count(text: str) -> int:
    return len(_load_embedding_tokenizer().encode(text).ids)


def chunk_page(
    text: str,
    *,
    doc_name: str,
    source_path: str,
    source_hash: str,
    page_number: int,
    profile: ChunkingProfile,
    metadata: Mapping[str, Any] | None = None,
    token_count: Callable[[str], int] | None = None,
) -> list[TextChunk]:
    if page_number < 0:
        raise ValueError("page_number must be a zero-based PDF page index")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=profile.chunk_size_chars,
        chunk_overlap=profile.chunk_overlap_chars,
    )
    chunks: list[TextChunk] = []
    normalized_text = text.replace("\x00", "")
    page_chunks = splitter.split_text(normalized_text)
    if token_count is not None:
        token_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_MAX_EMBEDDING_TOKENS,
            chunk_overlap=30,
            length_function=token_count,
        )
        page_chunks = [
            safe_chunk
            for chunk in page_chunks
            for safe_chunk in token_splitter.split_text(chunk)
        ]
    for chunk_index, chunk_text in enumerate(page_chunks):
        if token_count is not None and token_count(chunk_text) > _MAX_EMBEDDING_TOKENS:
            raise ValueError(
                f"Chunk exceeds the {_MAX_EMBEDDING_TOKENS}-token embedding limit"
            )
        digest_input = (
            f"{source_hash}:{profile.name}:{page_number}:{chunk_index}:{chunk_text}"
        )
        chunk_id = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                doc_name=doc_name,
                source_path=source_path,
                page_number=page_number,
                chunk_index=chunk_index,
                profile=profile.name,
                text=chunk_text,
                metadata=dict(metadata or {}),
            )
        )
    return chunks


def chunk_pdf(
    pdf_path: Path,
    profile: ChunkingProfile,
    metadata: Mapping[str, Any] | None = None,
) -> list[TextChunk]:
    with pdf_path.open("rb") as pdf_file:
        source_hash = hashlib.file_digest(pdf_file, "sha256").hexdigest()

    chunks: list[TextChunk] = []
    with fitz.open(pdf_path) as document:
        if document.needs_pass and not document.authenticate(""):
            raise ValueError(f"PDF requires a password: {pdf_path}")
        for page_number, page in enumerate(document):
            chunks.extend(
                chunk_page(
                    page.get_text("text"),
                    doc_name=pdf_path.stem,
                    source_path=pdf_path.name,
                    source_hash=source_hash,
                    page_number=page_number,
                    profile=profile,
                    metadata=metadata,
                    token_count=_embedding_token_count,
                )
            )
    if not chunks:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")
    return chunks
