from __future__ import annotations

import asyncio
import importlib
import math
from typing import Protocol

from .models import Chunk, SearchResult


def validate_vectors(vectors: list[list[float]], dimensions: int | None = None) -> int | None:
    for vector in vectors:
        if not vector:
            raise ValueError("Embedding vectors must not be empty")
        if dimensions is None:
            dimensions = len(vector)
        if len(vector) != dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {dimensions}, got {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding vectors must contain only finite values")
    return dimensions


class VectorStore(Protocol):
    """add upserts chunk IDs; repeated identical indexing does not create duplicates."""

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    async def query(self, vector: list[float], limit: int = 5) -> list[SearchResult]: ...


class MemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}
        self._dimensions: int | None = None

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        dimensions = validate_vectors(vectors, self._dimensions)
        copies = [
            (chunk.model_copy(deep=True), list(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._dimensions = dimensions
        self._items.update((chunk.id, (chunk, vector)) for chunk, vector in copies)

    def snapshot(self) -> tuple[list[Chunk], list[list[float]]]:
        return (
            [c.model_copy(deep=True) for c, _ in self._items.values()],
            [list(v) for _, v in self._items.values()],
        )

    async def query(self, vector: list[float], limit: int = 5) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")
        validate_vectors([vector], self._dimensions)

        def cosine(other: list[float]) -> float:
            # hypot avoids overflow for large finite inputs.
            norm_a, norm_b = math.hypot(*vector), math.hypot(*other)
            if not norm_a or not norm_b:
                return 0.0
            return max(
                -1.0,
                min(
                    1.0,
                    sum((a / norm_a) * (b / norm_b) for a, b in zip(vector, other, strict=True)),
                ),
            )

        return sorted(
            (
                SearchResult(chunk=c.model_copy(deep=True), score=cosine(v))
                for c, v in self._items.values()
            ),
            key=lambda x: x.score,
            reverse=True,
        )[:limit]


class ChromaVectorStore:
    """Cosine-distance Chroma adapter. Synchronous database calls run in worker threads."""

    def __init__(self, collection_name: str = "brightdata_rag", path: str | None = None):
        try:
            chromadb = importlib.import_module("chromadb")
        except ImportError as exc:
            raise ImportError("Install brightdata-rag[chroma]") from exc
        client = chromadb.PersistentClient(path=path) if path else chromadb.Client()
        self.collection = client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        if (self.collection.metadata or {}).get("hnsw:space") != "cosine":
            raise ValueError("Existing Chroma collection must use cosine distance; use a new name")

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        validate_vectors(vectors)
        if not chunks:
            return
        # Chroma rejects duplicate IDs within one request. Last value wins, as in memory.
        unique = {c.id: (c, v) for c, v in zip(chunks, vectors, strict=True)}
        await asyncio.to_thread(
            self.collection.upsert,
            ids=list(unique),
            embeddings=[v for _, v in unique.values()],
            documents=[c.text for c, _ in unique.values()],
            metadatas=[{"chunk_json": c.model_dump_json()} for c, _ in unique.values()],
        )

    async def query(self, vector: list[float], limit: int = 5) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be positive")
        validate_vectors([vector])
        count = await asyncio.to_thread(self.collection.count)
        if not count:
            return []
        result = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[vector],
            n_results=min(limit, count),
            include=["metadatas", "distances"],
        )
        out: list[SearchResult] = []
        for meta, distance in zip(result["metadatas"][0], result["distances"][0], strict=True):
            chunk = Chunk.model_validate_json(meta["chunk_json"])
            out.append(SearchResult(chunk=chunk, score=max(-1.0, min(1.0, 1.0 - distance))))
        return out
