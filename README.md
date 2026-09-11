# brightdata-rag

Turn public websites into searchable, AI-ready knowledge bases using [Bright Data](https://brightdata.grsm.io/p302843).

> **Community project:** Independently maintained; not an official [Bright Data](https://brightdata.grsm.io/p302843) product. [Bright Data](https://brightdata.grsm.io/p302843) is a trademark of its respective owner.

Python 3.11+. Async API, a command-line interface, and optional OpenAI / Chroma integrations.

## Install from this repository

```bash
uv sync --locked --extra dev
uv run brightdata-rag --help

# Optional integrations
uv sync --locked --extra dev --extra openai --extra chroma

# Or use pip in your own virtual environment
pip install -e .
pip install -e '.[openai,chroma]'
```

These instructions work from a checkout without requiring a published PyPI release. The base package includes HTTP, validation, CLI, and two lightweight HTML/Markdown parsing libraries. OpenAI and Chroma are imported only when used.

## Try it offline first

```bash
uv run python examples/offline.py
uv run brightdata-rag index --input examples/documents.jsonl --output index.json
uv run brightdata-rag search "How do I authenticate?" --index index.json
```

The default local embeddings use deterministic Unicode word hashing. They match shared words and are useful for offline demos and tests; they do **not** understand synonyms or provide learned semantic embeddings. Use the OpenAI embedding extra for semantic retrieval.

## Scrape and crawl

Export credentials in your shell. [.env.example](.env.example) lists the variables; the package does not automatically load it.

```bash
export BRIGHTDATA_API_KEY="your-api-key"
export BRIGHTDATA_ZONE="web_unlocker1"
```

```python
import asyncio
from brightdata_rag import KnowledgeBase, crawl, scrape


async def main():
    page = await scrape("https://example.com")
    print(page.title, page.markdown)

    docs = await crawl(
        "https://example.com",
        max_pages=20,
        max_depth=2,
        concurrency=5,
        include=["https://example.com/docs/*"],
        exclude=["*/login*", "*/logout*"],
    )
    kb = KnowledgeBase()
    await kb.index(docs)
    for result in await kb.search("authentication", limit=3):
        print(result.score, result.chunk.document_url, result.chunk.text)


asyncio.run(main())
```

Crawl behavior:

- Breadth-first, limited to the seed's exact normalized host and port. Subdomains are separate. URL fragments and default ports are deduplicated; query strings are preserved.
- The seed is depth 0. Include/exclude patterns are case-sensitive shell globs on full normalized URLs. The seed bypasses include patterns; exclude applies to the seed too.
- `max_pages` limits **attempted URLs**, including failures. Retries can create additional billable requests.
- Invalid limits fail immediately. `max_pages=0` performs no requests.
- Failures raise by default. Set `on_error="skip"` to log failures and return successful pages.
- Discovery includes navigation/sidebar links. Extraction prefers `main`, then `role="main"`, then `article`, then the body, and removes common boilerplate.
- Same-host checks apply to discovered URLs; redirects performed within Unlocker are opaque. The library does not enforce robots.txt or provide a network sandbox.

Only scrape content you are permitted to access and comply with site terms.

The HTTP adapter follows the [Unlocker API reference](https://docs.brightdata.com/api-reference/rest-api/unlocker/unlock-website). It accepts raw HTML and JSON envelopes, reuses connections, and retries transport errors, HTTP 429, and HTTP 5xx up to twice with bounded delays. Authentication errors are not retried. To customize retries or reuse one provider:

```python
from brightdata_rag import BrightDataProvider, scrape


async def fetch_pages():
    async with BrightDataProvider(max_retries=0, timeout=30) as provider:
        return await scrape("https://example.com", provider=provider)
```

Custom fetchers implement `async fetch(url: str) -> str`. Inject `httpx.AsyncClient` into `BrightDataProvider(client=...)` for mocked HTTP tests; injected clients remain owned by the caller.

## Chunking and retrieval

`chunk_document(document, chunk_size=1000, overlap=120)` splits at Markdown headings outside fenced code, then recursively at paragraph, line, sentence, word, and character boundaries. Sizes are **characters, not tokens**. Overlap stays within sections; oversized code blocks may be split. Each chunk retains its URL, title, heading, metadata, and a stable content-derived ID.

`Document`, `Chunk`, `SearchResult`, `Answer`, and `Citation` are Pydantic v2 models. `EmbeddingProvider` and `VectorStore` protocols support custom adapters.

`MemoryVectorStore` and `ChromaVectorStore` upsert by chunk ID. Indexing identical content twice does not duplicate it. Changed content creates new IDs: use a fresh store or collection for a full refresh so removed/changed passages do not remain in the index. Scores are cosine similarity (higher is better), not probabilities. Invalid dimensions and non-finite vectors are rejected.

## OpenAI embeddings and answers

Install the `openai` extra and export `OPENAI_API_KEY`. Choose one embedding model per store and use the same model for both indexing and querying.

```python
from brightdata_rag import KnowledgeBase, OpenAIEmbeddingProvider


async def answer_question(documents):
    async with OpenAIEmbeddingProvider(model="text-embedding-3-small") as embeddings:
        kb = KnowledgeBase(embeddings=embeddings)
        await kb.index(documents)
        answer = await kb.ask("How does authentication work?", model="gpt-4o-mini")
        print(answer.text)
        for citation in answer.citations:
            print(citation.number, citation.url, citation.chunk_ids)
```

OpenAI embeddings are batched, and clients are closed by their owning context. `ask()` uses the [Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create), with instructions separated from retrieved text. It limits excerpt content to 16,000 characters by default (`max_context_chars`), without counting question or URL metadata; this is not a token budget. An empty index returns an explicit no-sources answer without a generation request.

`answer.sources[N - 1]` is the URL for citation `[N]`. `answer.citations` contains the numbered references actually present in the answer. Unknown citation numbers raise an error. Source mapping is validated, but model-written claims still require review. Repeated CLI chat questions are independent; conversational history is not sent.

## Persistent Chroma

```python
from brightdata_rag import ChromaVectorStore, KnowledgeBase

store = ChromaVectorStore(path="./chroma-data", collection_name="docs_v1")
kb = KnowledgeBase(store=store)
```

Collections use cosine distance and no implicit embedding function. Database queries/writes run in worker threads to keep the async event loop responsive. The constructor performs synchronous initialization. A pre-existing collection with another distance metric is rejected; use a fresh name. Original chunk metadata is preserved.

## CLI

```bash
uv run brightdata-rag scrape https://example.com --output page.json
uv run brightdata-rag crawl https://example.com \
  --max-pages 20 --max-depth 2 --concurrency 5 \
  --include 'https://example.com/docs/*' --exclude '*/login*' \
  --output documents.jsonl

uv run brightdata-rag index --input page.json --output index.json
uv run brightdata-rag index --input documents.jsonl --output index.json \
  --embeddings openai --embedding-model text-embedding-3-small
uv run brightdata-rag search "authentication" --index index.json --limit 3
uv run brightdata-rag chat --index index.json --question "How does authentication work?"
uv run brightdata-rag chat --index index.json
```

`index` accepts one Document JSON object, an array, or JSONL. `crawl` supports repeated `--include` / `--exclude` and `--skip-errors`. `chat` accepts `--model` and `--limit`; type `exit`/`quit` or press Ctrl-D to finish. All commands accept `--help`.

Files are UTF-8 and written atomically. CLI indexes include a schema version and embedding configuration; querying reloads that provider. OpenAI indexes need the optional package and credentials for search. Legacy v0.1.0 indexes must be rebuilt with `index`. These JSON indexes are for small local collections; use Chroma for larger workloads.

## Development

```bash
uv sync --locked --extra dev
uv run --no-sync pytest --cov=brightdata_rag --cov-report=term-missing
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv build

# Enable the real, local Chroma integration test (no API key required)
uv sync --locked --extra dev --extra openai --extra chroma
uv run --no-sync pytest
```

Tests mock all external network services. A separate test exercises real local Chroma when installed. CI checks Python 3.11–3.14 for the core and Python 3.12 for optional integrations. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## [Bright Data](https://brightdata.grsm.io/p302843) affiliate disclosure

If you [sign up for Bright Data](https://brightdata.grsm.io/p302843) using this link, the project maintainer may receive a commission at no extra cost to you. This does not affect the technical recommendations or the open-source license.

Licensed under [MIT](LICENSE).
