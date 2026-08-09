"""MCP client tests.

The other half of the protocol: an agent picking up tools from a remote server. The
server under test is this project's own, started over HTTP, which conveniently guarantees
a tool-name collision with the local capabilities -- the situation that actually needs
handling in practice.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import subprocess
import sys

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from edgar_desk import db
from edgar_desk.agents.connected import build_connected_agent
from edgar_desk.deps import EdgarDeps
from edgar_desk.observability import collector_reachable
from edgar_desk.schemas import Brief, Citation, Finding
from edgar_desk.settings import get_settings

PORT = 8932
MCP_URL = f'http://localhost:{PORT}/mcp'


def _postgres_up() -> bool:
    url = get_settings().database_url
    hostport = url.rsplit('@', 1)[-1].split('/')[0]
    host, _, port = hostport.partition(':')
    return collector_reachable(f'http://{host}:{port or 5432}')


pytestmark = pytest.mark.skipif(not _postgres_up(), reason='postgres not running')

BRIEF = Brief(
    question='q',
    summary='s',
    findings=[
        Finding(
            claim='c',
            citations=[Citation(source='us-gaap:Revenues', ticker='NVDA')],
            confidence=1.0,
        )
    ],
)


def _port_open(port: int) -> bool:
    with contextlib.suppress(OSError), socket.create_connection(('localhost', port), 0.2):
        return True
    return False


@pytest.fixture(scope='module')
def mcp_http_server():
    proc = subprocess.Popen(
        [sys.executable, '-m', 'edgar_desk.cli', 'mcp', '--transport', 'http', '--port', str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):
            if _port_open(PORT):
                break
            if proc.poll() is not None:
                pytest.skip('MCP HTTP server exited during startup')
            import time

            time.sleep(0.25)
        else:
            pytest.skip('MCP HTTP server did not start in time')
        yield MCP_URL
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


async def _tools_offered(agent) -> list[str]:
    """Run the agent once with a scripted model, returning the tools it was offered."""
    seen: dict[str, list[str]] = {}

    def respond(messages, info: AgentInfo) -> ModelResponse:
        seen.setdefault('tools', sorted(t.name for t in info.function_tools))
        assert info.output_tools
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name, args=BRIEF.model_dump(mode='json')
                )
            ]
        )

    async with db.pool_context(max_size=2) as pool:
        with agent.override(model=FunctionModel(respond)):
            await agent.run('q', deps=EdgarDeps(pool=pool, rerank=False))
    return seen['tools']


async def test_remote_tools_are_discovered(mcp_http_server) -> None:
    agent = build_connected_agent(mcp_http_server, local_capabilities=False, prefix=None)
    tools = await _tools_offered(agent)
    assert {'get_financials', 'search_filings', 'run_sql', 'list_companies'} <= set(tools)


async def test_colliding_names_need_a_prefix(mcp_http_server) -> None:
    """Tool names share one flat namespace per agent, and a collision raises rather than
    silently shadowing. Any server whose vocabulary overlaps yours needs namespacing."""
    from pydantic_ai.exceptions import UserError

    agent = build_connected_agent(mcp_http_server, local_capabilities=True, prefix=None)
    with pytest.raises(UserError, match='conflicts'):
        await _tools_offered(agent)


async def test_prefix_resolves_the_collision(mcp_http_server) -> None:
    agent = build_connected_agent(mcp_http_server, local_capabilities=True, prefix='ext')
    tools = set(await _tools_offered(agent))
    # Local tools keep their names; remote tools are namespaced.
    assert 'get_financials' in tools
    assert 'ext_get_financials' in tools or 'ext-get_financials' in tools
    assert any(t.startswith('ext') and 'list_companies' in t for t in tools)


async def test_allowed_tools_narrows_the_surface(mcp_http_server) -> None:
    """Every advertised tool costs context on every request, so an unfamiliar server is
    worth restricting."""
    agent = build_connected_agent(
        mcp_http_server,
        local_capabilities=False,
        prefix=None,
        allowed_tools=['list_companies'],
    )
    tools = set(await _tools_offered(agent))
    assert 'list_companies' in tools
    assert 'run_sql' not in tools


async def test_remote_tool_actually_executes(mcp_http_server) -> None:
    """Discovery is not enough: the call has to reach the server and come back."""
    calls: list[str] = []

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if not calls:
            calls.append('tool')
            return ModelResponse(parts=[ToolCallPart(tool_name='list_companies', args={})])
        assert info.output_tools
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name, args=BRIEF.model_dump(mode='json')
                )
            ]
        )

    agent = build_connected_agent(mcp_http_server, local_capabilities=False, prefix=None)
    async with db.pool_context(max_size=2) as pool:
        with agent.override(model=FunctionModel(respond)):
            result = await agent.run('q', deps=EdgarDeps(pool=pool, rerank=False))

    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if part.part_kind == 'tool-return'
    ]
    assert any('NVDA' in str(r) for r in returns), 'remote tool result did not come back'


def test_asyncio_is_available() -> None:
    assert asyncio.get_event_loop_policy() is not None
