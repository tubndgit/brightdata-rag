from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections import deque
from typing import Literal
from urllib.parse import urlsplit

from .cleaning import clean_html
from .models import Document
from .providers import BrightDataProvider, ScrapeProvider
from .urls import normalize_url

logger = logging.getLogger(__name__)


async def scrape(url: str, provider: ScrapeProvider | None = None) -> Document:
    url = normalize_url(url)
    owned = BrightDataProvider() if provider is None else None
    active = owned if owned is not None else provider
    assert active is not None
    try:
        source = await active.fetch(url)
        title, markdown, _ = clean_html(source, url)
        return Document(url=url, title=title, markdown=markdown)
    finally:
        if owned is not None:
            await owned.aclose()


def _matches(url: str, patterns: list[str] | None, default: bool) -> bool:
    return default if not patterns else any(fnmatch.fnmatchcase(url, p) for p in patterns)


async def crawl(
    url: str,
    provider: ScrapeProvider | None = None,
    *,
    max_pages: int = 20,
    max_depth: int = 2,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    concurrency: int = 5,
    on_error: Literal["raise", "skip"] = "raise",
) -> list[Document]:
    """Breadth-first crawl of the exact normalized host and port.

    max_pages bounds attempted URLs (including failures), excluding provider retries.
    Root is depth 0 and exempt from include; exclude applies to every URL.
    Only discovered URLs are checked: redirects inside Unlocker are opaque to this client.
    """
    if max_pages < 0 or max_depth < 0 or concurrency < 1:
        raise ValueError("Require max_pages >= 0, max_depth >= 0, and concurrency >= 1")
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    url = normalize_url(url)
    if max_pages == 0 or _matches(url, exclude, False):
        return []
    root_host = urlsplit(url).netloc
    queue: deque[tuple[str, int]] = deque([(url, 0)])
    scheduled = {url}
    documents: list[Document] = []
    owned = BrightDataProvider() if provider is None else None
    active = owned if owned is not None else provider
    assert active is not None

    async def one(target: str) -> tuple[Document, list[str]]:
        source = await active.fetch(target)
        title, markdown, links = clean_html(source, target)
        return Document(url=target, title=title, markdown=markdown), links

    try:
        while queue:
            batch = [queue.popleft() for _ in range(min(concurrency, len(queue)))]
            results = await asyncio.gather(*(one(t) for t, _ in batch), return_exceptions=True)
            for (_target, depth), result in zip(batch, results, strict=True):
                if isinstance(result, BaseException):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    if on_error == "raise":
                        raise result
                    logger.warning("Skipping failed page (%s)", type(result).__name__)
                    continue
                document, links = result
                documents.append(document)
                if depth >= max_depth:
                    continue
                for link in links:
                    if len(scheduled) >= max_pages:
                        break
                    if urlsplit(link).netloc != root_host or link in scheduled:
                        continue
                    if not _matches(link, include, True) or _matches(link, exclude, False):
                        continue
                    scheduled.add(link)
                    queue.append((link, depth + 1))
    finally:
        if owned is not None:
            await owned.aclose()
    return documents
