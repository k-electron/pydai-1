"""MCP server tests.

Driven through a real MCP client over stdio rather than by calling the functions
directly, so the tool schemas, transport, and serialization are all exercised the way an
editor would exercise them.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from edgar_desk.observability import collector_reachable
from edgar_desk.settings import get_settings


def _postgres_up() -> bool:
    url = get_settings().database_url
    hostport = url.rsplit('@', 1)[-1].split('/')[0]
    host, _, port = hostport.partition(':')
    return collector_reachable(f'http://{host}:{port or 5432}')


pytestmark = pytest.mark.skipif(not _postgres_up(), reason='postgres not running')


@asynccontextmanager
async def mcp_session():
    """Start the server as a subprocess and connect a client over stdio.

    Deliberately not a pytest fixture: `stdio_client` uses anyio cancel scopes, and
    pytest-asyncio runs generator-fixture setup and teardown in different tasks, which
    trips "attempted to exit cancel scope in a different task". Entering and exiting
    inside the test body keeps both in one task.
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=['-m', 'edgar_desk.mcp_server'],
        env={**os.environ},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        yield client


async def test_server_advertises_its_tools() -> None:
    async with mcp_session() as session:
        tools = {t.name for t in (await session.list_tools()).tools}
    assert {
        'list_companies',
        'get_financials',
        'search_filings',
        'run_sql',
        'describe_schema',
        'research',
    } <= tools


async def test_tool_schemas_describe_their_arguments() -> None:
    """An MCP client picks tools from these schemas, so the descriptions matter."""
    async with mcp_session() as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    financials = tools['get_financials']
    assert financials.description
    properties = financials.inputSchema['properties']
    assert {'tickers', 'concepts', 'fiscal_years', 'period'} <= set(properties)


async def test_get_financials_returns_known_values() -> None:
    async with mcp_session() as session:
        result = await session.call_tool(
            'get_financials',
            {'tickers': ['NVDA'], 'concepts': ['Revenue'], 'fiscal_years': [2024]},
        )
    assert '60.922B' in result.content[0].text


async def test_uncovered_company_is_reported_not_raised() -> None:
    async with mcp_session() as session:
        result = await session.call_tool('get_financials', {'tickers': ['BA']})
    assert 'Not covered' in result.content[0].text


async def test_search_filings_returns_passages() -> None:
    async with mcp_session() as session:
        result = await session.call_tool(
            'search_filings',
            {'query': 'competition from other chip makers', 'tickers': ['NVDA'], 'limit': 3},
        )
    assert 'NVDA' in result.content[0].text


async def test_run_sql_rejects_writes() -> None:
    async with mcp_session() as session:
        result = await session.call_tool('run_sql', {'sql': 'DELETE FROM chunks'})
    assert 'rejected' in result.content[0].text.lower()


async def test_run_sql_returns_json_rows() -> None:
    async with mcp_session() as session:
        result = await session.call_tool(
            'run_sql',
            {
                'sql': 'SELECT ticker, fiscal_year, value FROM xbrl_facts '
                "WHERE ticker = 'AAPL' AND concept = 'Revenue' AND fiscal_period = 'FY' "
                'AND fiscal_year = 2024'
            },
        )
    rows = json.loads(result.content[0].text)
    assert rows[0]['ticker'] == 'AAPL'


async def test_list_companies_covers_the_universe() -> None:
    async with mcp_session() as session:
        text = (await session.call_tool('list_companies', {})).content[0].text
    assert 'NVDA' in text
    assert len(text.strip().splitlines()) == 20
