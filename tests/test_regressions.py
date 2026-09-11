import asyncio
import subprocess
import sys

import pytest

from brightdata_rag import Document, chunk_document, crawl
from brightdata_rag.cleaning import clean_html
from brightdata_rag.urls import normalize_url


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "example.com",
        "https://user:pass@example.com",
        "https://example.com:99999",
        "https://exa mple.com",
    ],
)
def test_invalid_urls(value):
    with pytest.raises(ValueError):
        normalize_url(value)


def test_url_normalization():
    assert normalize_url("HTTPS://EXAMPLE.COM:443/#part") == "https://example.com"
    assert normalize_url("https://example.com/a/?b=2&a=1#x") == "https://example.com/a/?b=2&a=1"
    assert normalize_url("http://[::1]:80/") == "http://[::1]"


def test_main_content_code_links_and_entities():
    source = (
        "<title>A &amp; B</title><nav><a href='/guide'>Guide</a></nav>"
        "<div>Boilerplate</div><main><h1>Title</h1><p>Hello <strong>world</strong>!</p>"
        "<pre><code>if x:\n    print('&amp;lt;')\n</code></pre>"
        "<a href='/api'>API</a><span hidden>Hidden</span></main>"
    )
    title, text, links = clean_html(source, "https://example.com")
    assert title == "A & B"
    assert "Boilerplate" not in text and "Hidden" not in text
    assert "**world**" in text and "    print('&lt;')" in text
    assert "[API](https://example.com/api)" in text
    assert links == ["https://example.com/guide", "https://example.com/api"]


def test_base_tag_and_duplicate_fragments():
    _, _, links = clean_html(
        "<base href='/docs/'><a href='a#one'>A</a><a href='a#two'>B</a>"
        "<a href='mailto:a@example.com'>Mail</a>",
        "https://example.com",
    )
    assert links == ["https://example.com/docs/a"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"concurrency": 0},
        {"concurrency": -1},
        {"max_depth": -1},
        {"max_pages": -1},
        {"on_error": "bad"},
    ],
)
async def test_invalid_crawl_settings(kwargs):
    with pytest.raises(ValueError):
        await crawl("https://example.com", **kwargs)


async def test_zero_budget_or_excluded_seed_needs_no_credentials(monkeypatch):
    monkeypatch.delenv("BRIGHTDATA_API_KEY", raising=False)
    assert await crawl("https://example.com", max_pages=0) == []
    assert await crawl("https://example.com", exclude=["*example.com"]) == []


async def test_crawl_budget_duplicates_concurrency_and_errors():
    class Pages:
        def __init__(self):
            self.calls = []
            self.active = 0
            self.peak = 0

        async def fetch(self, url):
            self.calls.append(url)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                if url.endswith("/bad"):
                    raise RuntimeError("failed")
                if url == "https://example.com":
                    return (
                        "<nav><a href='/bad'>Bad</a><a href='/good#1'>Good</a>"
                        "<a href='/good#2'>Again</a><a href='https://other.com'>Other</a>"
                        "<a href='/extra'>Extra</a></nav>"
                    )
                return "<main>Body</main>"
            finally:
                self.active -= 1

    pages = Pages()
    documents = await crawl(
        "https://EXAMPLE.com:443/",
        pages,
        max_pages=3,
        concurrency=2,
        on_error="skip",
    )
    assert [d.url for d in documents] == ["https://example.com", "https://example.com/good"]
    assert len(pages.calls) == 3 and pages.peak == 2 and pages.active == 0
    with pytest.raises(RuntimeError, match="failed"):
        await crawl("https://example.com", pages, concurrency=2)
    assert pages.active == 0


def test_high_overlap_terminates_in_subprocess():
    code = (
        "from brightdata_rag import Document, chunk_document; "
        "chunks=chunk_document(Document(url='https://x', markdown='a'*55+'\\n\\n'+'b'*160),"
        "100,90); assert chunks and all(len(c.text)<=100 for c in chunks)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_fenced_headings_and_content_coverage():
    fence = chr(96) * 3
    markdown = f"# Real\n\n{fence}python\n# comment\nprint(1)\n{fence}\n\n## Next\nEnd"
    chunks = chunk_document(Document(url="https://x", markdown=markdown), 200, 20)
    assert len(chunks) == 2
    assert "# comment" in chunks[0].text and chunks[0].metadata["heading"] == "Real"
    source = "".join(str(i % 10) for i in range(257))
    chunks = chunk_document(Document(url="https://x", markdown=source), 100, 0)
    assert "".join(c.text for c in chunks) == source


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(Document(url="https://x", markdown=""), size, overlap)
