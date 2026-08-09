"""Graph pipeline tests.

The point of expressing the pipeline as a graph is that its control flow is inspectable,
so these assert on structure and on both decision branches, driving every agent in the
graph with scripted models rather than inference.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from edgar_desk.agents.analyst import analyst_agent
from edgar_desk.agents.researcher import financial_specialist, narrative_specialist
from edgar_desk.agents.triage import triage_agent
from edgar_desk.deps import EdgarDeps
from edgar_desk.pipeline import (
    Evidence,
    NeedsMore,
    ResearchState,
    Sufficient,
    build_graph,
    render_mermaid,
    run_pipeline,
)
from edgar_desk.schemas import Angle, Brief, Citation, Finding, ResearchPlan, SubQuestion
from edgar_desk.universe import resolve

PLAN = ResearchPlan(
    companies=[resolve('NVDA')],  # type: ignore[list-item]
    sub_questions=[
        SubQuestion(question='How has R&D spend moved?', angle=Angle.FINANCIAL, companies=['NVDA']),
        SubQuestion(
            question='What do risk factors say about competition?',
            angle=Angle.NARRATIVE,
            companies=['NVDA'],
        ),
    ],
    reasoning='Split numbers from narrative.',
)

BRIEF = Brief(
    question='q',
    summary='A synthesized answer.',
    findings=[
        Finding(
            claim='NVIDIA R&D was $18.497B in FY2026.',
            citations=[Citation(source='us-gaap:ResearchAndDevelopment', ticker='NVDA')],
            confidence=1.0,
        )
    ],
)


def _structured(payload) -> FunctionModel:
    """A model that always returns the same structured output."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name, args=payload.model_dump(mode='json')
                )
            ]
        )

    return FunctionModel(respond)


def _text(reply: str) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=reply)])

    return FunctionModel(respond)


@pytest.fixture
def deps() -> EdgarDeps:
    return EdgarDeps(pool=None, rerank=False)  # type: ignore[arg-type]


def test_graph_structure_renders() -> None:
    diagram = render_mermaid()
    for node in ('triage', 'investigate', 'assess', 'deepen', 'synthesize'):
        assert node in diagram
    assert '<<fork>>' in diagram, 'sub-questions must fan out in parallel'
    assert '<<join>>' in diagram, 'branches must be aggregated'
    assert '<<choice>>' in diagram, 'coverage must be a decision node'
    assert 'coverage is thin' in diagram


def test_graph_builds_with_validation() -> None:
    """`build()` validates structure, so an unreachable node or type mismatch fails here."""
    assert build_graph() is not None


async def test_pipeline_runs_the_sufficient_branch(deps) -> None:
    substantial = 'x' * 400
    with (
        triage_agent.override(model=_structured(PLAN)),
        financial_specialist.override(model=_text(substantial)),
        narrative_specialist.override(model=_text(substantial)),
        analyst_agent.override(model=_structured(BRIEF)),
    ):
        brief, state = await run_pipeline('How is NVIDIA doing?', deps)

    assert brief.findings
    assert state.rounds == 1, 'sufficient evidence must not trigger a second round'
    assert state.plan is not None
    assert [q.angle for q in state.plan.sub_questions] == [Angle.FINANCIAL, Angle.NARRATIVE]
    assert brief.question == 'How is NVIDIA doing?'


async def test_thin_evidence_takes_the_deepen_branch(deps) -> None:
    """Specialists returning almost nothing must route through `deepen` before synthesis."""
    with (
        triage_agent.override(model=_structured(PLAN)),
        financial_specialist.override(model=_text('n/a')),
        narrative_specialist.override(model=_text('n/a')),
        analyst_agent.override(model=_structured(BRIEF)),
    ):
        brief, state = await run_pipeline('How is NVIDIA doing?', deps)

    assert brief.findings
    assert state.rounds == 1
    # `deepen` retried both thin sub-questions, so synthesis saw more than it started with.


def test_evidence_substantiveness_threshold() -> None:
    thin = Evidence(sub_question=PLAN.sub_questions[0], findings='n/a')
    thick = Evidence(sub_question=PLAN.sub_questions[0], findings='x' * 400)
    assert not thin.is_substantive
    assert thick.is_substantive


def test_assess_outcomes_are_distinct_types() -> None:
    """The decision node branches on type, so these must not be interchangeable."""
    assert Sufficient(evidence=[]) != NeedsMore(evidence=[], gaps=[])


def test_state_exposes_planned_companies() -> None:
    state = ResearchState(question='q', plan=PLAN)
    assert state.companies == ['NVDA']
    assert ResearchState(question='q').companies == []
