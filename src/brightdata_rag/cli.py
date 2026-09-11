from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import typer
from pydantic import ValidationError

from .core import crawl as crawl_pages
from .core import scrape as scrape_page
from .embeddings import OpenAIEmbeddingProvider
from .kb import KnowledgeBase
from .models import Document
from .providers import ScrapeError
from .storage import (
    EmbeddingSettings,
    read_index,
    restore_index,
    save_index,
    write_text_atomic,
)
from .vectorstores import MemoryVectorStore

app = typer.Typer(
    help="Turn websites into searchable, AI-ready knowledge bases.",
    no_args_is_help=True,
)


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except ValidationError as exc:
        typer.echo(
            f"Error: Invalid document or index ({exc.error_count()} validation errors).", err=True
        )
        raise typer.Exit(1) from None
    except (ValueError, ImportError, OSError, ScrapeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
    except httpx.HTTPError:
        typer.echo("Error: External service request failed.", err=True)
        raise typer.Exit(1) from None
    except Exception as exc:
        if type(exc).__module__.startswith("openai"):
            typer.echo(
                "Error: OpenAI request failed. Check credentials, quota, and model.", err=True
            )
            raise typer.Exit(1) from None
        raise


@app.command()
def scrape(url: str, output: Path | None = typer.Option(None, "--output", "-o")) -> None:
    """Scrape one URL to a Document JSON object."""
    with _errors():
        doc = asyncio.run(scrape_page(url))
        value = doc.model_dump_json(indent=2) + "\n"
        if output:
            write_text_atomic(output, value)
        else:
            typer.echo(value, nl=False)


@app.command()
def crawl(
    url: str,
    max_pages: int = 20,
    max_depth: int = 2,
    output: Path = Path("documents.jsonl"),
    concurrency: int = 5,
    include: list[str] | None = typer.Option(None, "--include"),
    exclude: list[str] | None = typer.Option(None, "--exclude"),
    skip_errors: bool = False,
) -> None:
    """Crawl same-host links to JSONL. Repeat --include/--exclude for multiple globs."""
    with _errors():
        docs = asyncio.run(
            crawl_pages(
                url,
                max_pages=max_pages,
                max_depth=max_depth,
                concurrency=concurrency,
                include=include,
                exclude=exclude,
                on_error="skip" if skip_errors else "raise",
            )
        )
        write_text_atomic(output, "".join(d.model_dump_json() + "\n" for d in docs))
        typer.echo(f"Saved {len(docs)} documents to {output}")


def _read(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        documents = []
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    documents.append(Document.model_validate_json(line))
                except ValidationError:
                    raise ValueError(f"Invalid document in {path} at line {number}") from None
        return documents
    if isinstance(parsed, list):
        return [Document.model_validate(item) for item in parsed]
    return [Document.model_validate(parsed)]


async def _close_embeddings(kb: KnowledgeBase) -> None:
    if isinstance(kb.embeddings, OpenAIEmbeddingProvider):
        await kb.embeddings.aclose()


@app.command()
def index(
    input: Path = Path("documents.jsonl"),
    output: Path = Path("index.json"),
    embeddings: str = "local",
    embedding_model: str = "text-embedding-3-small",
    dimensions: int = 256,
    chunk_size: int = 1000,
    overlap: int = 120,
) -> None:
    """Build a versioned index from Document JSON, JSON arrays, or JSONL."""
    with _errors():
        settings = EmbeddingSettings.model_validate(
            {
                "provider": embeddings,
                "model": embedding_model,
                "dimensions": dimensions,
            }
        )
        documents = _read(input)
        store = MemoryVectorStore()
        kb = KnowledgeBase(embeddings=settings.create(), store=store)

        async def run() -> int:
            try:
                return await kb.index(documents, chunk_size=chunk_size, overlap=overlap)
            finally:
                await _close_embeddings(kb)

        count = asyncio.run(run())
        save_index(output, store, settings)
        typer.echo(f"Indexed {count} chunks to {output}")


@app.command()
def search(query: str, index: Path = Path("index.json"), limit: int = 5) -> None:
    """Search a saved index; local embeddings work completely offline."""
    with _errors():
        payload = read_index(index)

        async def run() -> str:
            kb = await restore_index(payload)
            try:
                results = await kb.search(query, limit=limit)
                return json.dumps([r.model_dump() for r in results], ensure_ascii=False, indent=2)
            finally:
                await _close_embeddings(kb)

        typer.echo(asyncio.run(run()))


@app.command()
def chat(
    index: Path = Path("index.json"),
    question: str | None = None,
    model: str = "gpt-4o-mini",
    limit: int = 5,
) -> None:
    """Ask source-grounded questions. Use --question for a single answer."""
    with _errors():
        payload = read_index(index)
        # Reuse one event loop across questions and close clients before it exits.
        with asyncio.Runner() as runner:
            kb = runner.run(restore_index(payload))
            try:
                while True:
                    if question is None:
                        try:
                            prompt = typer.prompt("You (exit to quit)")
                        except (EOFError, KeyboardInterrupt, typer.Abort):
                            break
                        if prompt.strip().lower() in {"exit", "quit"}:
                            break
                        if not prompt.strip():
                            continue
                    else:
                        prompt = question
                    answer = runner.run(kb.ask(prompt, limit=limit, model=model))
                    typer.echo(answer.text)
                    for citation in answer.citations:
                        typer.echo(f"[{citation.number}] {citation.url}")
                    if question is not None:
                        break
            finally:
                runner.run(_close_embeddings(kb))


if __name__ == "__main__":
    app()
