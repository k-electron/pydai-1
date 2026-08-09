"""Web layer and human-in-the-loop approval tests.

The approval flow is tested at the agent level with a scripted model, because what
matters is the pause-and-resume contract: an approval-gated tool must not run until a
result says it may, and must run once one does.
"""

from __future__ import annotations

import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

from edgar_desk import db
from edgar_desk.agents.chat import chat_agent
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

BRIEF = Brief(
    question='What was NVIDIA revenue in fiscal 2024?',
    summary='NVIDIA reported $60.922 billion of revenue in fiscal 2024.',
    findings=[
        Finding(
            claim='NVIDIA revenue was $60.922 billion in FY2024.',
            citations=[Citation(source='us-gaap:Revenues', ticker='NVDA', fiscal_period='FY2024')],
            confidence=1.0,
        )
    ],
)

UNCITED_BRIEF = Brief(
    question='q',
    summary='s',
    findings=[Finding(claim='An unsupported claim.', citations=[], confidence=0.5)],
)


def _publish_then_confirm(brief: Brief) -> FunctionModel:
    """Model that asks to publish, then acknowledges once the tool has run."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        already_returned = any(
            part.part_kind == 'tool-return' for message in messages for part in message.parts
        )
        if already_returned:
            return ModelResponse(parts=[TextPart(content='Done.')])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='publish_brief', args={'brief': brief.model_dump(mode='json')}
                )
            ]
        )

    return FunctionModel(respond)


@pytest.fixture
async def deps():
    async with db.pool_context(max_size=2) as pool:
        yield EdgarDeps(pool=pool, rerank=False)


async def test_approval_gated_tool_pauses_the_run(deps) -> None:
    """The run must stop and ask, not publish."""
    with chat_agent.override(model=_publish_then_confirm(BRIEF)):
        result = await chat_agent.run('publish it', deps=deps)

    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.approvals[0].tool_name == 'publish_brief'
    assert 'publish_brief' not in deps.tool_calls, 'tool ran before approval'


async def test_approving_resumes_and_publishes(deps) -> None:
    with chat_agent.override(model=_publish_then_confirm(BRIEF)):
        paused = await chat_agent.run('publish it', deps=deps)
        assert isinstance(paused.output, DeferredToolRequests)
        call = paused.output.approvals[0]

        resumed = await chat_agent.run(
            message_history=paused.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={call.tool_call_id: ToolApproved()}
            ),
            deps=deps,
        )

    assert 'publish_brief' in deps.tool_calls
    assert isinstance(resumed.output, str)

    async with deps.pool.acquire() as conn:
        row = await conn.fetchrow(
            'SELECT question, brief FROM review_queue ORDER BY id DESC LIMIT 1'
        )
    assert row is not None
    stored = json.loads(row['brief']) if isinstance(row['brief'], str) else row['brief']
    assert stored['findings'][0]['citations'][0]['ticker'] == 'NVDA'


async def test_denying_does_not_publish(deps) -> None:
    async with deps.pool.acquire() as conn:
        before = await conn.fetchval('SELECT count(*) FROM review_queue')

    with chat_agent.override(model=_publish_then_confirm(BRIEF)):
        paused = await chat_agent.run('publish it', deps=deps)
        call = paused.output.approvals[0]  # type: ignore[union-attr]
        await chat_agent.run(
            message_history=paused.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={call.tool_call_id: ToolDenied(message='No.')}
            ),
            deps=deps,
        )

    assert 'publish_brief' not in deps.tool_calls
    async with deps.pool.acquire() as conn:
        after = await conn.fetchval('SELECT count(*) FROM review_queue')
    assert after == before


async def test_uncited_brief_is_refused_even_after_approval(deps) -> None:
    """Approval is a human saying "go ahead", not a re-verification of the content, so
    the tool still checks its own invariants."""
    async with deps.pool.acquire() as conn:
        before = await conn.fetchval('SELECT count(*) FROM review_queue')

    with chat_agent.override(model=_publish_then_confirm(UNCITED_BRIEF)):
        paused = await chat_agent.run('publish it', deps=deps)
        call = paused.output.approvals[0]  # type: ignore[union-attr]
        await chat_agent.run(
            message_history=paused.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={call.tool_call_id: ToolApproved()}
            ),
            deps=deps,
        )

    async with deps.pool.acquire() as conn:
        after = await conn.fetchval('SELECT count(*) FROM review_queue')
    assert after == before, 'an uncited brief must not be published'


def test_api_routes_are_registered() -> None:
    from edgar_desk.web.app import app as fastapi_app

    paths = {route.path for route in fastapi_app.routes}  # type: ignore[attr-defined]
    assert {'/api/chat', '/api/health', '/api/companies', '/api/review'} <= paths
