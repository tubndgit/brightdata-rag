"""Adapter contracts and optional real SDK tests, without any external requests."""

import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest

from brightdata_rag import (
    ChromaVectorStore,
    Chunk,
    Document,
    KnowledgeBase,
    OpenAIEmbeddingProvider,
)


def test_import_and_local_search_without_optional_packages():
    script = """
import asyncio
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openai', 'chromadb'}:
            raise ImportError('Optional dependency deliberately unavailable')
sys.meta_path.insert(0, BlockOptional())
from brightdata_rag import KnowledgeBase, Document
async def run():
    kb = KnowledgeBase()
    await kb.index([Document(url='https://x', markdown='Python')])
    assert (await kb.search('Python'))[0].chunk.document_url == 'https://x'
asyncio.run(run())
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


async def test_chroma_contract_runs_io_off_event_loop(monkeypatch):
    main_thread = threading.get_ident()
    records = {}
    worker_threads = []

    class Collection:
        metadata = {"hnsw:space": "cosine"}

        def count(self):
            worker_threads.append(threading.get_ident())
            return len(records)

        def upsert(self, **kwargs):
            worker_threads.append(threading.get_ident())
            for id, metadata in zip(kwargs["ids"], kwargs["metadatas"], strict=True):
                records[id] = metadata

        def query(self, **kwargs):
            worker_threads.append(threading.get_ident())
            assert kwargs["n_results"] == 1
            return {"metadatas": [[next(iter(records.values()))]], "distances": [[0.25]]}

    collection = Collection()

    def get_collection(*args, **kwargs):
        assert kwargs["embedding_function"] is None
        assert kwargs["metadata"]["hnsw:space"] == "cosine"
        return collection

    fake = SimpleNamespace(Client=lambda: SimpleNamespace(get_or_create_collection=get_collection))
    monkeypatch.setitem(sys.modules, "chromadb", fake)
    store = ChromaVectorStore()
    item = Chunk(id="x", document_url="https://x", text="Python", index=0, metadata={"none": None})
    await store.add([item, item], [[1.0], [1.0]])
    result = (await store.query([1.0]))[0]
    assert result.chunk == item and result.score == pytest.approx(0.75)
    assert worker_threads and all(thread != main_thread for thread in worker_threads)
    with pytest.raises(ValueError):
        await store.add([item], [])
    with pytest.raises(ValueError):
        await store.query([1.0], 0)
    collection.metadata = {"hnsw:space": "l2"}
    with pytest.raises(ValueError, match="cosine"):
        ChromaVectorStore()


async def test_real_openai_sdk_against_mock_transport():
    sdk = pytest.importorskip("openai")
    requests = []

    def handler(request):
        requests.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1.0, 0.0]}
                        for i in range(len(body["input"]))
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        assert request.url.path.endswith("/responses")
        assert body["instructions"] and json.loads(body["input"])["sources"]
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body["model"],
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "Use a token [1].", "annotations": []}
                        ],
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with sdk.AsyncOpenAI(api_key="test-key", http_client=http) as client:
            embeddings = OpenAIEmbeddingProvider(client=client)
            kb = KnowledgeBase(embeddings=embeddings)
            await kb.index([Document(url="https://example.com", markdown="Use a token.")])
            answer = await kb.ask("How?", client=client)
            assert answer.citations[0].url == "https://example.com"
    assert requests == ["/v1/embeddings", "/v1/embeddings", "/v1/responses"]
