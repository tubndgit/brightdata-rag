"""Run with: uv run python examples/offline.py (no keys or network needed)."""

import asyncio

from brightdata_rag import Document, KnowledgeBase


async def main() -> None:
    kb = KnowledgeBase()
    await kb.index(
        [
            Document(
                url="https://example.com/auth", markdown="# Authentication\nUse an API token."
            ),
            Document(
                url="https://example.com/rate-limits", markdown="# Rate limits\nRetry HTTP 429."
            ),
        ]
    )
    for result in await kb.search("API token", limit=1):
        print(result.chunk.document_url, result.chunk.text, sep="\n")


if __name__ == "__main__":
    asyncio.run(main())
