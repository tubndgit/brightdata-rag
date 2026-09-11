from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    url: str
    title: str = ""
    markdown: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(min_length=1)
    document_url: str
    text: str
    index: int = Field(ge=0)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: Chunk
    score: float = Field(allow_inf_nan=False)


class Citation(BaseModel):
    number: int = Field(ge=1)
    url: str
    chunk_ids: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    text: str
    sources: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
