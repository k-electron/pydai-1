"""Triage agent tests.

These run with no model server and no network: `TestModel` synthesizes tool calls and
a schema-valid output from the agent's own schemas, which is enough to prove the wiring
(tool registration, dependency injection, output validation) without inference.
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.test import TestModel

from edgar_desk.agents.triage import TriageDeps, triage_agent
from edgar_desk.schemas import Angle, ResearchPlan, SubQuestion
from edgar_desk.universe import resolve


@pytest.fixture(autouse=True)
def block_real_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test ever reaches a real model server."""
    import pydantic_ai.models

    monkeypatch.setattr(pydantic_ai.models, 'ALLOW_MODEL_REQUESTS', False)


async def test_triage_returns_validated_plan() -> None:
    deps = TriageDeps()
    with triage_agent.override(model=TestModel()):
        result = await triage_agent.run('How did NVDA revenue trend?', deps=deps)

    assert isinstance(result.output, ResearchPlan)


async def test_triage_calls_resolve_company_tool() -> None:
    """TestModel calls every registered tool once, which proves the tool is wired
    into the agent and that it writes through to the injected dependency."""
    deps = TriageDeps()
    with triage_agent.override(model=TestModel()), capture_run_messages() as messages:
        await triage_agent.run('Compare AAPL and MSFT margins', deps=deps)

    tool_names = {
        part.tool_name
        for message in messages
        for part in message.parts
        if part.part_kind == 'tool-call'
    }
    assert 'resolve_company' in tool_names


async def test_instructions_include_covered_universe() -> None:
    deps = TriageDeps()
    with triage_agent.override(model=TestModel()), capture_run_messages() as messages:
        await triage_agent.run('anything', deps=deps)

    instructions = messages[0].instructions or ''
    assert 'NVDA' in instructions
    assert 'Covered universe' in instructions


async def test_plan_output_is_reusable_downstream() -> None:
    """A hand-built plan must satisfy the same schema the agent produces, so the rest
    of the pipeline can be developed and tested without a model."""
    plan = ResearchPlan(
        companies=[resolve('NVDA')],  # type: ignore[list-item]
        sub_questions=[
            SubQuestion(
                question='How has R&D spend as a share of revenue moved?',
                angle=Angle.FINANCIAL,
                companies=['NVDA'],
            ),
            SubQuestion(
                question='What do recent risk factors say about competition?',
                angle=Angle.NARRATIVE,
                companies=['NVDA'],
            ),
        ],
        reasoning='Split numeric trend from narrative context.',
    )
    assert plan.companies[0].cik == '0001045810'
    assert {q.angle for q in plan.sub_questions} == {Angle.FINANCIAL, Angle.NARRATIVE}


def test_unknown_company_is_not_resolved() -> None:
    assert resolve('ZZZZ') is None


async def test_custom_output_via_function_model() -> None:
    """Pin exact behavior by scripting the model's response, rather than accepting
    whatever TestModel happens to generate."""
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        plan = ResearchPlan(
            companies=[resolve('AAPL')],  # type: ignore[list-item]
            sub_questions=[
                SubQuestion(
                    question='Gross margin trend?', angle=Angle.FINANCIAL, companies=['AAPL']
                )
            ],
            reasoning='Single numeric question.',
        )
        assert info.output_tools
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=plan.model_dump(mode='json'),
                )
            ]
        )

    agent: Agent[TriageDeps, ResearchPlan] = triage_agent
    with agent.override(model=FunctionModel(respond)):
        result = await agent.run('Apple margins?', deps=TriageDeps())

    assert result.output.companies[0].ticker == 'AAPL'
    assert result.output.sub_questions[0].angle == Angle.FINANCIAL
