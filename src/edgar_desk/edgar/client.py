"""A polite SEC EDGAR HTTP client.

The SEC publishes two hard requirements for automated access: send a descriptive
`User-Agent` with real contact information, and stay at or below 10 requests per second.
Violating either gets an IP blocked, so the rate limit is enforced here rather than left
to callers to remember.
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType

import httpx

from edgar_desk.settings import get_settings

DATA_HOST = 'https://data.sec.gov'
WWW_HOST = 'https://www.sec.gov'


class RateLimiter:
    """Simple async token-bucket, shared across all requests from one client."""

    def __init__(self, rate_per_second: float) -> None:
        self._min_interval = 1.0 / rate_per_second
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class EdgarClient:
    """Async client for the SEC's public endpoints.

    Deliberately under the published 10 req/s ceiling: the limit is per IP and shared
    with anything else on this machine talking to the SEC.
    """

    def __init__(self, rate_per_second: float = 6.0, timeout: float = 45.0) -> None:
        settings = get_settings()
        self._limiter = RateLimiter(rate_per_second)
        self._client = httpx.AsyncClient(
            headers={
                'User-Agent': settings.sec_user_agent,
                'Accept-Encoding': 'gzip, deflate',
            },
            timeout=timeout,
            follow_redirects=True,
        )

    async def __aenter__(self) -> EdgarClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, url: str, attempts: int = 4) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(attempts):
            await self._limiter.acquire()
            try:
                response = await self._client.get(url)
            except httpx.HTTPError as exc:
                last = exc
            else:
                # 403 is how the SEC signals rate-limit abuse, so back off rather than
                # treating it as a permanent failure.
                if response.status_code in (403, 429, 500, 502, 503, 504):
                    last = httpx.HTTPStatusError(
                        f'{response.status_code} from {url}',
                        request=response.request,
                        response=response,
                    )
                else:
                    response.raise_for_status()
                    return response
            await asyncio.sleep(2.0 * (2**attempt))
        assert last is not None
        raise last

    async def company_facts(self, cik: str) -> dict:
        """Every XBRL fact the company has ever reported."""
        url = f'{DATA_HOST}/api/xbrl/companyfacts/CIK{cik}.json'
        return (await self._get(url)).json()

    async def submissions(self, cik: str) -> dict:
        """Filing history, including accession numbers and primary document names."""
        url = f'{DATA_HOST}/submissions/CIK{cik}.json'
        return (await self._get(url)).json()

    async def filing_document(self, cik: str, accession: str, document: str) -> str:
        """Fetch one document from a filing.

        The archive path wants the CIK without zero padding and the accession without
        dashes, while the document name keeps its own form.
        """
        bare_cik = cik.lstrip('0')
        bare_accession = accession.replace('-', '')
        url = f'{WWW_HOST}/Archives/edgar/data/{bare_cik}/{bare_accession}/{document}'
        return (await self._get(url)).text
