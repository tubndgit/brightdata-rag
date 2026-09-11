import asyncio

from brightdata_rag import KnowledgeBase, crawl


async def main() -> None:
    documents = await crawl("https://docs.example.com", max_pages=25, max_depth=2)
    kb = KnowledgeBase()
    await kb.index(documents)
    for result in await kb.search("How does authentication work?"):
        print(result.score, result.chunk.document_url)


if __name__ == "__main__":
    asyncio.run(main())
