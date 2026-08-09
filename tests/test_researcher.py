"""Researcher assembly tests.

The researcher is defined entirely by its `capabilities=[...]` list, so what matters is
that each capability actually reaches the model. These assert on the tool definitions the
model is offered, without running inference.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from edgar_desk import db
from edgar_desk.agents.researcher import build_researcher
from edgar_desk.deps import EdgarDeps
from edgar_desk.observability import collector_reachable
from edgar_desk.schemas import Brief, Citation, Finding
from edgar_desk.settings import get_settings


def _postgres_up() -> bool:
    url = get_settings().database_url
    hostport = url.rsplit('@', 1)[-1].split('/')[0]
    host, _, port = hostport.partition(':')
    return collector_reachable(f'http://{host}:{port or 5432}')


pytestmark = pytest.mark.skipif(not _postgres_up(), reason='postgres not running')

VALID = Brief(
    question='q',
    summary='A valid summary.',
    findings=[
        Finding(
            claim='NVIDIA revenue was $60.922B in FY2024.',
            citations=[Citation(source='us-gaap:Revenues', ticker='NVDA')],
            confidence=1.0,
        )
    ],
)


@pytest.fixture
async def captured(tmp_path):
    """Run the researcher once with a scripted model, capturing what it was offered."""
    seen: dict[str, object] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.setdefault('tools', sorted(t.name for t in info.function_tools))
        seen.setdefault('instructions', messages[0].instructions or '')
        assert info.output_tools
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name, args=VALID.model_dump(mode='json')
                )
            ]
        )

    agent = build_researcher(memory_dir=tmp_path / 'memory')
    async with db.pool_context(max_size=2) as pool:
        with agent.override(model=FunctionModel(respond)):
            await agent.run('q', deps=EdgarDeps(pool=pool, rerank=False))
    return seen


async def test_domain_capabilities_contribute_tools(captured) -> None:
    tools = set(captured['tools'])
    assert {'get_financials', 'run_sql', 'describe_schema'} <= tools
    assert {'search_filings', 'describe_coverage'} <= tools


async def test_harness_capabilities_contribute_tools(captured) -> None:
    """Planning, sub-agent delegation and memory each add model-facing tools."""
    tools = set(captured['tools'])
    assert any('plan' in t for t in tools), f'no planning tool in {sorted(tools)}'
    assert any('delegate' in t for t in tools), f'no delegation tool in {sorted(tools)}'
    assert any('memory' in t for t in tools), f'no memory tool in {sorted(tools)}'


async def test_instructions_compose_from_every_capability(captured) -> None:
    """Capability instructions concatenate, so each one's guidance must be present."""
    instructions = str(captured['instructions']).lower()
    assert 'get_financials' in instructions  # financials capability
    assert 'risk factors' in instructions  # narrative capability
    assert 'citation' in instructions  # custom citation enforcer
    assert 'covered companies' in instructions  # agent-level dynamic instructions


async def test_memory_directory_is_created(tmp_path) -> None:
    memory_dir = tmp_path / 'nested' / 'memory'
    build_researcher(memory_dir=memory_dir)
    assert memory_dir.is_dir()
