import pytest

from brightdata_rag import KnowledgeBase, chunk_document, crawl, scrape
from brightdata_rag.cleaning import clean_html
from brightdata_rag.models import Document


class FakeProvider:
    pages = {
        "https://example.com": (
            "<title>Home</title><nav>Noise</nav><main><h1>Hello</h1>"
            "<p>Useful text</p><a href='/a'>A</a>"
            "<a href='https://other.test/x'>X</a></main>"
        ),
        "https://example.com/a": (
            "<title>A</title><main><h2>Authentication</h2>"
            "<p>Use an API token.</p><a href='/deep'>D</a></main>"
        ),
        "https://example.com/deep": "<main>Too deep</main>",
    }

    async def fetch(self, url: str) -> str:
        return self.pages[url]


def test_clean_html_removes_noise_and_extracts_links():
    title, markdown, links = clean_html(
        FakeProvider.pages["https://example.com"], "https://example.com"
    )
    assert title == "Home"
    assert "# Hello" in markdown and "Noise" not in markdown
    assert "https://example.com/a" in links


@pytest.mark.asyncio
async def test_scrape():
    doc = await scrape("https://example.com", FakeProvider())
    assert doc.title == "Home" and "Useful text" in doc.markdown


@pytest.mark.asyncio
async def test_crawl_same_domain_and_depth():
    docs = await crawl("https://example.com", FakeProvider(), max_pages=10, max_depth=1)
    assert [d.url for d in docs] == ["https://example.com", "https://example.com/a"]


@pytest.mark.asyncio
async def test_crawl_patterns():
    docs = await crawl("https://example.com", FakeProvider(), include=["*/a"], max_depth=2)
    assert [d.url for d in docs] == ["https://example.com", "https://example.com/a"]


def test_chunking_is_stable_and_bounded():
    doc = Document(url="https://x", markdown="# Heading\n\n" + "word " * 100)
    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    assert len(chunks) > 1 and chunks[0].id == chunk_document(doc, 100, 10)[0].id
    assert all(len(c.text) <= 110 for c in chunks)


@pytest.mark.asyncio
async def test_semantic_search():
    kb = KnowledgeBase()
    await kb.index(
        [
            Document(url="https://fruit", markdown="Apples and oranges are fruit."),
            Document(url="https://code", markdown="Python software development."),
        ]
    )
    results = await kb.search("python programming", limit=1)
    assert results[0].chunk.document_url == "https://code"
