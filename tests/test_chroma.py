import pytest

from brightdata_rag import ChromaVectorStore, Chunk, MemoryVectorStore


async def test_real_chroma_persistence_metadata_cosine_and_upsert(tmp_path, monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    pytest.importorskip("chromadb")
    store = ChromaVectorStore(path=str(tmp_path))
    memory = MemoryVectorStore()
    assert await store.query([1.0, 0.0]) == []
    await store.add([], [])
    chunks = [
        Chunk(id="a", document_url="https://a", text="A", index=0, metadata={"none": None}),
        Chunk(id="b", document_url="https://b", text="B", index=1, metadata={"lang": "vi"}),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    await store.add(chunks, vectors)
    await store.add(chunks, vectors)
    await memory.add(chunks, vectors)
    reopened = ChromaVectorStore(path=str(tmp_path))
    actual, expected = await reopened.query([1.0, 0.0]), await memory.query([1.0, 0.0])
    assert len(actual) == 2
    assert [r.chunk for r in actual] == [r.chunk for r in expected]
    assert [r.score for r in actual] == pytest.approx([r.score for r in expected])
