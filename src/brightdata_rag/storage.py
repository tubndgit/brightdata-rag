"""Versioned, validated JSON indexes for the CLI's in-memory backend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .embeddings import EmbeddingProvider, LocalEmbeddingProvider, OpenAIEmbeddingProvider
from .kb import KnowledgeBase
from .models import Chunk
from .vectorstores import MemoryVectorStore, validate_vectors


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["local", "openai"] = "local"
    dimensions: int = Field(default=256, ge=1)
    model: str = "text-embedding-3-small"
    local_algorithm: Literal["unicode-hash-v1"] = "unicode-hash-v1"

    def create(self) -> EmbeddingProvider:
        if self.provider == "openai":
            return OpenAIEmbeddingProvider(model=self.model)
        return LocalEmbeddingProvider(self.dimensions)


class SavedIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    embedding: EmbeddingSettings
    chunks: list[Chunk]
    vectors: list[list[float]]

    @model_validator(mode="after")
    def validate_index(self) -> SavedIndex:
        if len(self.chunks) != len(self.vectors):
            raise ValueError("Index chunk and vector counts do not match")
        if len({c.id for c in self.chunks}) != len(self.chunks):
            raise ValueError("Index contains duplicate chunk IDs")
        expected = self.embedding.dimensions if self.embedding.provider == "local" else None
        validate_vectors(self.vectors, expected)
        return self


def write_text_atomic(path: Path, text: str) -> None:
    """UTF-8, same-directory replacement so interrupted writes leave the old file intact."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def save_index(path: Path, store: MemoryVectorStore, embedding: EmbeddingSettings) -> None:
    chunks, vectors = store.snapshot()
    payload = SavedIndex(embedding=embedding, chunks=chunks, vectors=vectors)
    write_text_atomic(path, payload.model_dump_json(indent=2) + "\n")


async def restore_index(payload: SavedIndex) -> KnowledgeBase:
    store = MemoryVectorStore()
    await store.add(payload.chunks, payload.vectors)
    return KnowledgeBase(embeddings=payload.embedding.create(), store=store)


def read_index(path: Path) -> SavedIndex:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError("Legacy or invalid index: rebuild it with brightdata-rag index")
    return SavedIndex.model_validate(data)
