"""Model bake-off: can a local model actually drive an agent?

Everything downstream assumes the model reliably (a) calls tools instead of guessing and
(b) emits schema-valid structured output. Local models vary enormously on both. This
suite measures it on the real triage agent before the rest of the system is built on top.

Run it with:  uv run edgar-desk bakeoff
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, MaxDuration, ToolCorrectness

from edgar_desk import observability
from edgar_desk.agents.triage import TriageDeps, triage_agent
from edgar_desk.models import ollama_model
from edgar_desk.schemas import Angle, ResearchPlan


@dataclass
class ExpectedPlan:
    """What a correct plan looks like, as case metadata."""

    tickers: set[str]
    angles: set[Angle]


@dataclass
class IdentifiesCompanies(Evaluator[str, ResearchPlan, ExpectedPlan]):
    """Did the plan scope exactly the companies the question is about?

    Scored as Jaccard overlap rather than pass/fail, so a model that finds two of three
    companies is distinguishable from one that finds none.
    """

    def evaluate(self, ctx: EvaluatorContext[str, ResearchPlan, ExpectedPlan]) -> float:
        if ctx.metadata is None or ctx.output is None:
            return 0.0
        got = {c.ticker.upper() for c in ctx.output.companies}
        want = ctx.metadata.tickers
        if not want and not got:
            return 1.0
        union = got | want
        return len(got & want) / len(union) if union else 0.0


@dataclass
class RoutesAngles(Evaluator[str, ResearchPlan, ExpectedPlan]):
    """Did the plan route sub-questions to the evidence source that can answer them?

    A model that tags a revenue question `narrative` will send it to vector search and
    get prose where a number was needed, so this is a real correctness signal.
    """

    def evaluate(self, ctx: EvaluatorContext[str, ResearchPlan, ExpectedPlan]) -> float:
        if ctx.metadata is None or ctx.output is None:
            return 0.0
        got: set[Angle] = set()
        for q in ctx.output.sub_questions:
            if q.angle is Angle.BOTH:
                got |= {Angle.FINANCIAL, Angle.NARRATIVE}
            else:
                got.add(q.angle)
        want = ctx.metadata.angles
        return len(got & want) / len(want) if want else 1.0


@dataclass
class ResolvedCiksAreReal(Evaluator[str, ResearchPlan, ExpectedPlan]):
    """Every in-universe company must carry the exact CIK the tool returned.

    This is the hallucination check, and it is strict in both directions: inventing a
    plausible 10-digit CIK fails, and so does dropping the one the tool handed back.
    Carrying tool output through to structured output is the single behavior the rest
    of the pipeline depends on most.
    """

    def evaluate(self, ctx: EvaluatorContext[str, ResearchPlan, ExpectedPlan]) -> bool:
        from edgar_desk.universe import BY_TICKER

        if ctx.output is None:
            return False
        for company in ctx.output.companies:
            known = BY_TICKER.get(company.ticker.upper())
            if known is None or company.cik != known.cik:
                return False
        return True


@dataclass
class PlanIsWellFormed(Evaluator[str, ResearchPlan, ExpectedPlan]):
    """Guard against degenerate plans that are technically schema-valid.

    An empty plan is the correct answer when nothing is in scope, so the sub-question
    floor only applies once at least one company resolved.
    """

    def evaluate(self, ctx: EvaluatorContext[str, ResearchPlan, ExpectedPlan]) -> bool:
        out = ctx.output
        if out is None or not out.reasoning.strip():
            return False
        if len(out.sub_questions) > 6:
            return False
        if not all(len(q.question.strip()) > 10 for q in out.sub_questions):
            return False
        return len(out.sub_questions) >= 1 if out.companies else True


# Every question below names at least one covered company, so a plan built without
# consulting the tool is a plan built on the model's memory. This lives per-case rather
# than on the dataset because the out-of-universe case has no company to resolve: both
# models correctly answer it from the injected universe list without calling anything,
# and asserting a tool call there would penalize the right behavior.
_MUST_RESOLVE = [ToolCorrectness(expected_tools=['resolve_company'], allow_extra=True)]

CASES: list[Case[str, ResearchPlan, ExpectedPlan]] = [
    Case(
        name='single_company_numeric',
        inputs='How has NVDA revenue trended over the last three fiscal years?',
        metadata=ExpectedPlan(tickers={'NVDA'}, angles={Angle.FINANCIAL}),
        evaluators=_MUST_RESOLVE,
    ),
    Case(
        name='single_company_narrative',
        inputs='What does Apple say about supply chain risk in its latest 10-K risk factors?',
        metadata=ExpectedPlan(tickers={'AAPL'}, angles={Angle.NARRATIVE}),
        evaluators=_MUST_RESOLVE,
    ),
    Case(
        name='mixed_angle',
        inputs=(
            'How has NVIDIA R&D spend as a share of revenue changed over five years, '
            'and what do their latest risk factors say about competition?'
        ),
        metadata=ExpectedPlan(tickers={'NVDA'}, angles={Angle.FINANCIAL, Angle.NARRATIVE}),
        evaluators=_MUST_RESOLVE,
    ),
    Case(
        name='multi_company_comparison',
        inputs='Compare gross margins for AAPL, MSFT and GOOGL in the most recent fiscal year.',
        metadata=ExpectedPlan(tickers={'AAPL', 'MSFT', 'GOOGL'}, angles={Angle.FINANCIAL}),
        evaluators=_MUST_RESOLVE,
    ),
    Case(
        name='name_not_ticker',
        inputs='Summarize what Tesla and Intel say about manufacturing capacity constraints.',
        metadata=ExpectedPlan(tickers={'TSLA', 'INTC'}, angles={Angle.NARRATIVE}),
        evaluators=_MUST_RESOLVE,
    ),
    Case(
        name='out_of_universe',
        inputs='How did Boeing and Ford revenue compare last year?',
        metadata=ExpectedPlan(tickers=set(), angles=set()),
    ),
]


def build_dataset() -> Dataset[str, ResearchPlan, ExpectedPlan]:
    return Dataset[str, ResearchPlan, ExpectedPlan](
        name='triage_bakeoff',
        cases=CASES,
        evaluators=[
            IdentifiesCompanies(),
            RoutesAngles(),
            ResolvedCiksAreReal(),
            PlanIsWellFormed(),
            MaxDuration(seconds=180),
        ],
    )


async def run_for_model(model_name: str, repeat: int = 1) -> None:
    """Evaluate the triage agent driven by one specific local model."""
    # Span-based evaluators such as `ToolCorrectness` read the OpenTelemetry span tree,
    # not the run result. That needs both a configured tracer provider and instrumented
    # agents; with only one of the two, the assertion silently reports false even when
    # the tool was called correctly.
    observability.configure('edgar-desk-bakeoff')
    Agent.instrument_all()

    dataset = build_dataset()
    model = ollama_model(model_name)

    async def triage(question: str) -> ResearchPlan:
        with triage_agent.override(model=model):
            result = await triage_agent.run(
                question,
                deps=TriageDeps(),
                # A model that cannot resolve anything tends to retry the tool forever.
                # Cap it so one bad case cannot dominate the suite's runtime.
                usage_limits=UsageLimits(request_limit=8, tool_calls_limit=12),
            )
        return result.output

    report = await dataset.evaluate(
        triage,
        name=f'triage::{model_name}',
        # Local inference is memory-bound; parallel runs thrash the model server.
        max_concurrency=1,
        repeat=repeat,
    )
    report.print(include_input=False, include_output=False)
