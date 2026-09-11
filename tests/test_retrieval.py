import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from brightdata_rag import (
    Chunk,
    Document,
    KnowledgeBase,
    LocalEmbeddingProvider,
    MemoryVectorStore,
    OpenAIEmbeddingProvider,
)


def chunk(id="one", url="https://one", text="Python"):
    return Chunk(id=id, document_url=url, text=text, index=0, metadata={"optional": None})


async def test_memory_upsert_copy_and_cosine():
    store = MemoryVectorStore()
    original = chunk()
    vector = [3.0, 4.0]
    await store.add([original], [vector])
    original.text = "changed"
    vector[0] = 0.0
    result = (await store.query([6.0, 8.0]))[0]
    assert result.score == pytest.approx(1)
    assert result.chunk.text == "Python"
    await store.add([chunk(text="updated")], [[1.0, 0.0]])
    result = await store.query([1.0, 0.0])
    assert len(result) == 1 and result[0].chunk.text == "updated"
    result[0].chunk.text = "caller mutation"
    assert (await store.query([1.0, 0.0]))[0].chunk.text == "updated"


@pytest.mark.parametrize("vector", [[], [1.0], [float("nan"), 0.0], [float("inf"), 0.0]])
async def test_bad_vectors_do_not_corrupt_store(vector):
    store = MemoryVectorStore()
    await store.add([chunk()], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        await store.add([chunk("two")], [vector])
    assert len(await store.query([1.0, 0.0])) == 1
    with pytest.raises(ValueError):
        await store.query(vector)


async def test_store_count_limits_and_zero_vectors():
    store = MemoryVectorStore()
    with pytest.raises(ValueError):
        await store.add([chunk()], [])
    with pytest.raises(ValueError):
        await store.query([1.0], 0)
    assert await store.query([0.0, 0.0]) == []
    await store.add([chunk()], [[0.0, 0.0]])
    assert (await store.query([0.0, 0.0]))[0].score == 0


async def test_unicode_local_vectors_and_validation():
    with pytest.raises(ValueError):
        LocalEmbeddingProvider(0)
    provider = LocalEmbeddingProvider()
    first, second = await provider.embed(["Tiếng Việt", "Tiếng Việt"])
    assert first == second and any(first)
    assert sum(v * v for v in first) == pytest.approx(1)


async def test_batched_openai_order_and_caller_ownership():
    async def create(**kwargs):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[float(i), 1.0])
                for i in reversed(range(len(kwargs["input"])))
            ]
        )

    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock(side_effect=create)), close=AsyncMock()
    )
    async with OpenAIEmbeddingProvider(client=client, batch_size=2) as provider:
        vectors = await provider.embed(["a", "b", "c"])
        assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
        assert await provider.embed([]) == []
    assert client.embeddings.create.await_count == 2
    client.close.assert_not_called()


async def test_embedding_count_mismatch():
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(data=[]),
            )
        )
    )
    with pytest.raises(ValueError, match="count/order"):
        await OpenAIEmbeddingProvider(client=client).embed(["one"])


async def test_kb_idempotence_and_blank_queries():
    kb = KnowledgeBase()
    docs = [Document(url="https://one", markdown="one")]
    await kb.index(docs)
    await kb.index(docs)
    assert len(await kb.search("one")) == 1
    with pytest.raises(ValueError):
        await kb.search(" ")
    with pytest.raises(ValueError):
        await kb.search("one", 0)


async def test_answer_citations_share_source_numbers_and_context_is_bounded():
    kb = KnowledgeBase()
    await kb.index(
        [
            Document(url="https://one", markdown="# One\nPython\n# Two\nPython"),
            Document(url="https://two", markdown="Python"),
        ]
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(output_text="Answer [1] and [2]."),
            )
        ),
        close=AsyncMock(),
    )
    answer = await kb.ask("Python", client=client)
    payload = json.loads(client.responses.create.call_args.kwargs["input"])
    assert [s["source"] for s in payload["sources"]] == [1, 2, 2]
    assert answer.sources == ["https://two", "https://one"]
    assert answer.citations[1].url == "https://one"
    assert len(answer.citations[1].chunk_ids) == 2
    assert "untrusted data" in client.responses.create.call_args.kwargs["instructions"]
    client.close.assert_not_called()
    client.responses.create.return_value.output_text = "Answer [1]."
    answer = await kb.ask("Python", client=client, max_context_chars=3)
    payload = json.loads(client.responses.create.call_args.kwargs["input"])
    assert sum(len(s["text"]) for s in payload["sources"]) <= 3
    assert len(answer.sources) == 1


async def test_answer_empty_index_no_generation_and_invalid_citations():
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock()))
    kb = KnowledgeBase()
    answer = await kb.ask("question", client=client)
    assert not answer.sources
    client.responses.create.assert_not_called()
    await kb.index([Document(url="https://one", markdown="Python")])
    client.responses.create.return_value = SimpleNamespace(output_text="Wrong [99]")
    with pytest.raises(ValueError, match="unknown source"):
        await kb.ask("Python", client=client)


async def test_owned_openai_clients_close_on_failure(monkeypatch):
    import sys

    generated = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("generation failed"))),
        close=AsyncMock(),
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda **_: generated))
    kb = KnowledgeBase()
    await kb.index([Document(url="https://one", markdown="Python")])
    with pytest.raises(RuntimeError, match="generation failed"):
        await kb.ask("Python")
    generated.close.assert_awaited_once()

    embedded = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("embedding failed"))),
        close=AsyncMock(),
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda **_: embedded))
    with pytest.raises(RuntimeError, match="embedding failed"):
        async with OpenAIEmbeddingProvider(api_key="test") as provider:
            await provider.embed(["Python"])
    embedded.close.assert_awaited_once()


async def test_missing_optional_packages_have_install_guidance(monkeypatch):
    import sys

    from brightdata_rag import ChromaVectorStore

    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(ImportError, match=r"brightdata-rag\[openai\]"):
        await OpenAIEmbeddingProvider(api_key="test").embed(["Python"])
    with pytest.raises(ImportError, match=r"brightdata-rag\[chroma\]"):
        ChromaVectorStore()
