from __future__ import annotations

import hashlib
import importlib
import math
import os
import re
from typing import Any, Protocol

from .vectorstores import validate_vectors


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddingProvider:
    """Deterministic Unicode feature hashing; lexical similarity, not learned semantics."""

    def __init__(self, dimensions: int = 256):
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE):
                h = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
                vector[h % self.dimensions] += 1 if (h >> 8) % 2 else -1
            norm = math.hypot(*vector) or 1
            vectors.append([x / norm for x in vector])
        return vectors


class OpenAIEmbeddingProvider:
    """Batched OpenAI embeddings. Pass client for testing; caller owns injected clients."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        *,
        batch_size: int = 64,
        client: Any = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.batch_size = model, batch_size
        self._client, self._api_key = client, api_key
        self._owns_client = client is None

    async def __aenter__(self) -> OpenAIEmbeddingProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed blank text")
        if self._client is None:
            try:
                factory = importlib.import_module("openai").AsyncOpenAI
            except (ImportError, AttributeError) as exc:
                raise ImportError("Install brightdata-rag[openai]") from exc
            self._client = factory(api_key=self._api_key or os.getenv("OPENAI_API_KEY"))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = await self._client.embeddings.create(model=self.model, input=batch)
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise ValueError("Embedding response does not match input count/order")
            vectors.extend(item.embedding for item in items)
        validate_vectors(vectors)
        return vectors
