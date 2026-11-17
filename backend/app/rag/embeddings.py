from __future__ import annotations

from typing import Any, Protocol, Sequence


class EmbeddingClient(Protocol):
    def create_embeddings(
        self, input: str | Sequence[str], model: str | None = None
    ) -> dict[str, Any]: ...


def embed_texts(
    client: EmbeddingClient,
    texts: Sequence[str],
    *,
    model: str,
    dimensions: int,
    batch_size: int,
) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    if not texts:
        return []

    vectors: list[list[float]] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        response = client.create_embeddings(list(batch), model=model)
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise ValueError("Embeddings response count does not match request")

        indexed_flags = [
            isinstance(item, dict)
            and isinstance(item.get("index"), int)
            and not isinstance(item.get("index"), bool)
            for item in data
        ]
        if any(indexed_flags) and not all(indexed_flags):
            raise ValueError("Embeddings response contains incomplete item indices")
        if all(indexed_flags):
            data = sorted(data, key=lambda item: item["index"])
            if [item["index"] for item in data] != list(range(len(batch))):
                raise ValueError("Embeddings response contains invalid item indices")

        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Embeddings response item must be an object")
            embedding = item.get("embedding")
            if (
                not isinstance(embedding, list)
                or len(embedding) != dimensions
                or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in embedding
                )
            ):
                raise ValueError(
                    f"Embedding must be a numeric vector with {dimensions} dimensions"
                )
            vectors.append([float(value) for value in embedding])
    return vectors
