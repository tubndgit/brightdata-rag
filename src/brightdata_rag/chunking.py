from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Split at ATX headings, never at heading-like text inside fenced code."""
    sections: list[tuple[str, str]] = []
    lines: list[str] = []
    heading = ""
    fence = ""
    for line in markdown.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(\x60{3,}|~{3,})", line)
        if marker:
            run = marker[1]
            if not fence:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not line.strip()[len(run) :]:
                fence = ""
        if not fence and re.match(r"^#{1,6}\s+", line):
            if lines:
                sections.append((heading, "".join(lines).strip()))
            heading = line.strip().lstrip("#").strip()
            lines = []
        lines.append(line)
    if lines:
        sections.append((heading, "".join(lines).strip()))
    return sections


def _recursive_cut(text: str, size: int, minimum: int, separators: tuple[str, ...]) -> int:
    """Try coarse boundaries first, falling back to words and then characters."""
    if not separators:
        return size
    separator, *rest = separators
    position = text.rfind(separator, 0, size + 1)
    cut = position + len(separator) if position >= 0 else -1
    if minimum <= cut <= size:
        return cut
    return _recursive_cut(text, size, minimum, tuple(rest))


def chunk_document(document: Document, chunk_size: int = 1000, overlap: int = 120) -> list[Chunk]:
    """Chunk by Markdown section, then recursively by paragraph/line/sentence/word.

    Sizes are characters, not tokens. Oversized code blocks may be split. Each iteration
    advances even with overlap close to chunk_size. Overlap does not cross headings.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    chunks: list[Chunk] = []
    for heading, section in _sections(document.markdown):
        start = 0
        while start < len(section):
            remaining = section[start:]
            cut = len(remaining)
            if cut > chunk_size:
                cut = _recursive_cut(
                    remaining,
                    chunk_size,
                    max(overlap + 1, chunk_size // 2),
                    ("\n\n", "\n", ". ", " "),
                )
            text = remaining[:cut].strip()
            if text:
                index = len(chunks)
                digest = hashlib.sha256(f"{document.url}\0{index}\0{text}".encode()).hexdigest()
                metadata = {**document.metadata, "title": document.title, "heading": heading}
                chunks.append(
                    Chunk(
                        id=digest,
                        document_url=document.url,
                        text=text,
                        index=index,
                        metadata=metadata,
                    )
                )
            if cut == len(remaining):
                break
            start += cut - overlap
    return chunks
