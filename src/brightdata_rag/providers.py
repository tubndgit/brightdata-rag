from __future__ import annotations

import asyncio
import json
import os
from typing import Protocol

import httpx

from .urls import normalize_url


class ScrapeProvider(Protocol):
    async def fetch(self, url: str) -> str: ...


class ScrapeError(RuntimeError):
    """A fetch failure with a safe message (no response bodies or credentials)."""


class BrightDataProvider:
    """Async Unlocker client with pooling, bounded retries, and injectable transport.

    Use as an async context manager when supplying it to multiple scrape/crawl calls.
    An injected httpx client remains owned by the caller.
    """

    def __init__(
        self,
        api_key: str | None = None,
        zone: str | None = None,
        timeout: float = 60,
        *,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ):
        self.api_key = api_key or os.getenv("BRIGHTDATA_API_KEY", "")
        self.zone = zone or os.getenv("BRIGHTDATA_ZONE", "web_unlocker1")
        if not self.api_key:
            raise ValueError("Set BRIGHTDATA_API_KEY or pass api_key")
        if timeout <= 0 or max_retries < 0 or retry_delay < 0:
            raise ValueError("Require timeout > 0 and non-negative retry settings")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> BrightDataProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> str:
        url = normalize_url(url)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    "https://api.brightdata.com/request",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"zone": self.zone, "url": url, "format": "raw"},
                    timeout=self.timeout,
                )
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise ScrapeError(
                        "Bright Data request failed after transport retries"
                    ) from None
                await asyncio.sleep(min(self.retry_delay * 2**attempt, 30))
                continue
            status = response.status_code
            if status == 429 or status >= 500:
                if attempt < self.max_retries:
                    delay = self.retry_delay * 2**attempt
                    try:
                        delay = float(response.headers.get("Retry-After", delay))
                    except ValueError:
                        pass
                    await asyncio.sleep(max(0, min(delay, 30)))
                    continue
            if not 200 <= status < 300:
                raise ScrapeError(f"Bright Data request failed (HTTP {status})")
            try:
                data = response.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                return response.text
            if isinstance(data, dict) and "status_code" in data:
                target_status = data["status_code"]
                if not isinstance(target_status, int) or not 200 <= target_status < 300:
                    raise ScrapeError("Target website returned an unsuccessful status")
                if not isinstance(data.get("body"), str):
                    raise ScrapeError("Bright Data response is missing a text body")
                return data["body"]
            if isinstance(data, str):
                return data
            return response.text
        raise AssertionError("Unreachable retry state")
