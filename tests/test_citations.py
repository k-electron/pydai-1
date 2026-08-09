"""Tests for the custom citation-enforcing capability.

Instructions asking for citations are a request; this capability is the enforcement, so
it needs to actually reject bad output rather than merely describe the rule.
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from edgar_desk.capabilities.citations import RequireCitations
from edgar_desk.deps import EdgarDeps
from edgar_desk.schemas import Brief, Citation, Finding


def _brief(findings: list[Finding], summary: str = 'A summary of the answer.') -> Brief:
    return Brief(question='q', summary=summary, findings=findings)


GOOD = _brief(
    [
        Finding(
            claim='NVIDIA revenue was $60.922B in FY2024.',
            citations=[Citation(source='us-gaap:Revenues', ticker='NVDA', fiscal_period='FY2024')],
            confidence=1.0,
        )
    ]
)

UNCITED = _brief([Finding(claim='NVIDIA revenue roughly tripled.', citations=[], confidence=0.9)])

WRONG_COMPANY = _brief(
    [
        Finding(
            claim='Boeing revenue fell.',
            citations=[Citation(source='us-gaap:Revenues', ticker='BA')],
            confidence=0.8,
        )
    ]
)


def _scripted_agent(outputs: list[Brief], deps_type=EdgarDeps):
    """An agent whose model returns each brief in turn, so retries are observable."""
    calls = {'n': 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        index = min(calls['n'], len(outputs) - 1)
        calls['n'] += 1
        assert info.output_tools
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=outputs[index].model_dump(mode='json'),
                )
            ]
        )

    agent = Agent(
        FunctionModel(respond),
        deps_type=deps_type,
        output_type=Brief,
        capabilities=[RequireCitations()],
        name='scripted',
        retries=4,
    )
    return agent, calls


@pytest.fixture
def deps():
    return EdgarDeps(pool=None, rerank=False)  # type: ignore[arg-type]


async def test_cited_brief_passes_through(deps) -> None:
    agent, calls = _scripted_agent([GOOD])
    result = await agent.run('q', deps=deps)
    assert result.output.findings[0].citations
    assert calls['n'] == 1, 'a valid brief must not trigger a retry'


async def test_uncited_finding_triggers_a_retry(deps) -> None:
    """The model gets a second attempt with a specific correction, rather than the
    caller getting a confident but unsupported answer."""
    agent, calls = _scripted_agent([UNCITED, GOOD])
    result = await agent.run('q', deps=deps)
    assert calls['n'] == 2
    assert result.output.findings[0].citations


async def test_citation_to_uncovered_company_is_rejected(deps) -> None:
    agent, calls = _scripted_agent([WRONG_COMPANY, GOOD])
    result = await agent.run('q', deps=deps)
    assert calls['n'] == 2
    assert result.output.findings[0].citations[0].ticker == 'NVDA'


async def test_persistent_failure_degrades_instead_of_looping(deps) -> None:
    """After the retry budget, the brief comes back flagged. An endless retry loop is
    worse than a clearly-caveated answer."""
    agent, calls = _scripted_agent([UNCITED])
    result = await agent.run('q', deps=deps)
    assert calls['n'] == 3, 'two retries, then accept'
    assert any('could not be verified' in c for c in result.output.caveats)


async def test_empty_findings_are_rejected(deps) -> None:
    agent, calls = _scripted_agent([_brief([]), GOOD])
    await agent.run('q', deps=deps)
    assert calls['n'] == 2


async def test_capability_contributes_instructions() -> None:
    from pydantic_ai import capture_run_messages

    agent, _ = _scripted_agent([GOOD])
    with capture_run_messages() as messages:
        await agent.run('q', deps=EdgarDeps(pool=None, rerank=False))  # type: ignore[arg-type]
    assert 'citation' in (messages[0].instructions or '').lower()
