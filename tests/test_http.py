import asyncio
import json

import httpx
import pytest

from brightdata_rag import BrightDataProvider, ScrapeError


@pytest.mark.parametrize(
    "response, expected",
    [
        (httpx.Response(200, text="<main>Hello</main>"), "<main>Hello</main>"),
        (
            httpx.Response(200, json={"status_code": 200, "body": "<h1>Wrapped</h1>"}),
            "<h1>Wrapped</h1>",
        ),
        (httpx.Response(200, json="plain text"), "plain text"),
    ],
)
async def test_raw_and_enveloped_responses(response, expected):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "zone": "test-zone",
            "url": "https://example.com",
            "format": "raw",
        }
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with BrightDataProvider("test-key", "test-zone", client=client) as provider:
            assert await provider.fetch("https://EXAMPLE.com/#x") == expected
        assert not client.is_closed


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, text="SECRET"),
        httpx.Response(200, json={"status_code": 403, "body": "bad"}),
        httpx.Response(200, json={"status_code": 200, "body": None}),
    ],
)
async def test_errors_are_explicit_and_do_not_leak_bodies(response):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(ScrapeError) as exc:
            await BrightDataProvider("SECRET", client=client).fetch("https://example.com")
        assert "SECRET" not in str(exc.value)


async def test_retry_rate_limits_then_success():
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text="success")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrightDataProvider("key", client=client, retry_delay=0)
        assert await provider.fetch("https://example.com") == "success"
        assert calls == 2


async def test_transport_retry_exhaustion_and_cancellation():
    calls = 0

    def failing(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("secret", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        with pytest.raises(ScrapeError, match="transport retries"):
            await BrightDataProvider(
                "key",
                client=client,
                max_retries=1,
                retry_delay=0,
            ).fetch("https://example.com")
        assert calls == 2

    async def cancelled(_):
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancelled)) as client:
        with pytest.raises(asyncio.CancelledError):
            await BrightDataProvider("key", client=client).fetch("https://example.com")


async def test_owned_client_closed_on_scrape_error(monkeypatch):
    import brightdata_rag.core as core

    class Owned:
        closed = False

        async def fetch(self, _):
            raise RuntimeError("broken")

        async def aclose(self):
            self.closed = True

    owned = Owned()
    monkeypatch.setattr(core, "BrightDataProvider", lambda: owned)
    with pytest.raises(RuntimeError):
        await core.scrape("https://example.com")
    assert owned.closed
