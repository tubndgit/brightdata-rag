import json

import pytest
from typer.testing import CliRunner

from brightdata_rag import Document, KnowledgeBase, MemoryVectorStore
from brightdata_rag.cli import app
from brightdata_rag.storage import (
    EmbeddingSettings,
    SavedIndex,
    read_index,
    restore_index,
    save_index,
)

runner = CliRunner()


@pytest.mark.parametrize("format", ["object", "array", "jsonl"])
def test_index_search_round_trip_all_document_formats(tmp_path, format):
    docs = [Document(url="https://example.com", markdown="Python tiếng Việt")]
    if format == "object":
        text = docs[0].model_dump_json(indent=2)
    elif format == "array":
        text = json.dumps([d.model_dump() for d in docs])
    else:
        text = "\n".join(d.model_dump_json() for d in docs) + "\n"
        text += Document(url="https://fruit", markdown="Apple banana").model_dump_json() + "\n"
    input_path, output_path = tmp_path / "docs.json", tmp_path / "index.json"
    input_path.write_text(text, encoding="utf-8")
    result = runner.invoke(app, ["index", "--input", str(input_path), "--output", str(output_path)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["search", "Python", "--index", str(output_path), "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["chunk"]["document_url"] == "https://example.com"
    assert read_index(output_path).version == 1


def test_bad_files_and_parameters_are_friendly(tmp_path):
    result = runner.invoke(app, ["search", "q", "--index", str(tmp_path / "missing.json")])
    assert result.exit_code == 1 and "Error:" in result.output
    bad = tmp_path / "bad.json"
    bad.write_text('{"chunks": [], "vectors": []}', encoding="utf-8")
    result = runner.invoke(app, ["search", "q", "--index", str(bad)])
    assert result.exit_code == 1 and "rebuild" in result.output
    result = runner.invoke(app, ["crawl", "https://example.com", "--concurrency", "0"])
    assert result.exit_code == 1 and "concurrency" in result.output
    bad.write_text("not json", encoding="utf-8")
    result = runner.invoke(app, ["index", "--input", str(bad)])
    assert result.exit_code == 1 and "line 1" in result.output


def test_cli_missing_key_and_help(monkeypatch):
    monkeypatch.delenv("BRIGHTDATA_API_KEY", raising=False)
    result = runner.invoke(app, ["scrape", "https://example.com"])
    assert result.exit_code == 1 and "BRIGHTDATA_API_KEY" in result.output
    for command in ["scrape", "crawl", "index", "search", "chat"]:
        assert runner.invoke(app, [command, "--help"]).exit_code == 0


def test_scrape_json_can_be_indexed(tmp_path, monkeypatch):
    async def fake(_):
        return Document(url="https://example.com", markdown="Hello")

    monkeypatch.setattr("brightdata_rag.cli.scrape_page", fake)
    path = tmp_path / "doc.json"
    assert runner.invoke(app, ["scrape", "https://example.com", "-o", str(path)]).exit_code == 0
    result = runner.invoke(
        app, ["index", "--input", str(path), "--output", str(tmp_path / "i.json")]
    )
    assert result.exit_code == 0, result.output


async def test_corrupt_vectors_and_dimension_mismatch(tmp_path):
    store = MemoryVectorStore()
    kb = KnowledgeBase(store=store)
    await kb.index([Document(url="https://x", markdown="Python")])
    path = tmp_path / "index.json"
    save_index(path, store, EmbeddingSettings())
    payload = read_index(path)
    restored = await restore_index(payload)
    assert (await restored.search("Python"))[0].chunk.document_url == "https://x"
    data = payload.model_dump()
    data["vectors"] = [[1.0]]
    with pytest.raises(ValueError):
        SavedIndex.model_validate(data)


def test_atomic_write_retains_old_file_on_replace_failure(tmp_path, monkeypatch):
    from brightdata_rag.storage import write_text_atomic

    path = tmp_path / "file.json"
    path.write_text("original", encoding="utf-8")

    def fail(*_):
        raise OSError("replace failed")

    monkeypatch.setattr("brightdata_rag.storage.os.replace", fail)
    with pytest.raises(OSError):
        write_text_atomic(path, "new")
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_chat_exit_and_empty_index_one_shot(tmp_path):
    path = tmp_path / "index.json"
    save_index(path, MemoryVectorStore(), EmbeddingSettings())
    result = runner.invoke(app, ["chat", "--index", str(path)], input="exit\n")
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["chat", "--index", str(path), "--question", "Anything?"])
    assert result.exit_code == 0 and "could not find" in result.output
