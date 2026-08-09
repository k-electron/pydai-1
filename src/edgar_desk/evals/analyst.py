"""The analyst eval suite.

Three kinds of check, deliberately layered:

1. **Exact** -- did the brief state the reported figure? Graded against XBRL, no judgment
   involved. This is the check worth trusting.
2. **Behavioral** -- did the agent actually call a retrieval tool, or produce a
   plausible-looking number from memory? Read from the OpenTelemetry span tree.
3. **Qualitative** -- is the narrative answer any good? An LLM judge, on a *different*
   local model, because a model grading its own output measures self-consistency.

Run it with:  uv run edgar-desk eval
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    LLMJudge,
    MaxDuration,
    ToolCorrectness,
)

from edgar_desk import observability
from edgar_desk.deps import EdgarDeps
from edgar_desk.evals.ground_truth import (
    ExpectedFact,
    brief_text,
    load_facts,
    mentions_ratio,
    mentions_value,
)
from edgar_desk.models import judge_model
from edgar_desk.schemas import Brief


@dataclass
class Expectation:
    """What a correct answer must contain, as case metadata."""

    facts: list[ExpectedFact] = field(default_factory=list)
    ratios: list[tuple[ExpectedFact, ExpectedFact]] = field(default_factory=list)
    """Pairs graded as numerator/denominator percentages, e.g. R&D over revenue."""

    forbidden_tickers: list[str] = field(default_factory=list)
    """Companies the answer must not claim data for."""


@dataclass
class StatesReportedFigures(Evaluator[str, Brief, Expectation]):
    """Fraction of the expected figures the brief actually states.

    The check the whole corpus choice is for: these targets come from the SEC's own XBRL
    data, so this is correctness rather than plausibility.
    """

    def evaluate(self, ctx: EvaluatorContext[str, Brief, Expectation]) -> float:
        if ctx.metadata is None or ctx.output is None or not ctx.metadata.facts:
            return 1.0
        text = brief_text(ctx.output)
        hits = sum(1 for fact in ctx.metadata.facts if mentions_value(text, fact.value))
        return hits / len(ctx.metadata.facts)


@dataclass
class ComputesRatiosCorrectly(Evaluator[str, Brief, Expectation]):
    """Whether derived percentages match the underlying reported figures."""

    def evaluate(self, ctx: EvaluatorContext[str, Brief, Expectation]) -> float:
        if ctx.metadata is None or ctx.output is None or not ctx.metadata.ratios:
            return 1.0
        text = brief_text(ctx.output)
        hits = sum(
            1
            for numerator, denominator in ctx.metadata.ratios
            if mentions_ratio(text, numerator.value, denominator.value)
        )
        return hits / len(ctx.metadata.ratios)


@dataclass
class EveryFindingIsCited(Evaluator[str, Brief, Expectation]):
    """Every finding present must be cited.

    Vacuously true for a brief with no findings, which is the correct answer when the
    question is about a company outside the covered universe. Requiring findings here
    would penalize a correct refusal; that a brief has findings when it should is a
    separate check.
    """

    def evaluate(self, ctx: EvaluatorContext[str, Brief, Expectation]) -> bool:
        out = ctx.output
        return out is not None and all(f.citations for f in out.findings)


@dataclass
class AnswersWhenDataExists(Evaluator[str, Brief, Expectation]):
    """A question the corpus can answer must actually get findings."""

    def evaluate(self, ctx: EvaluatorContext[str, Brief, Expectation]) -> bool:
        if ctx.output is None:
            return False
        if ctx.metadata is None or not ctx.metadata.facts:
            return True
        return bool(ctx.output.findings)


@dataclass
class NoUncoveredCompanies(Evaluator[str, Brief, Expectation]):
    """The agent must not answer for companies it has no data on."""

    def evaluate(self, ctx: EvaluatorContext[str, Brief, Expectation]) -> bool:
        if ctx.output is None:
            return False
        forbidden = {t.upper() for t in (ctx.metadata.forbidden_tickers if ctx.metadata else [])}
        if not forbidden:
            return True
        cited = {c.ticker.upper() for f in ctx.output.findings for c in f.citations}
        return not (cited & forbidden)


@dataclass
class RetrievedRatherThanRecalled(Evaluator[str, Brief, Expectation]):
    """Did a retrieval tool actually run?

    A model can produce a well-formed, correctly-shaped, entirely invented figure. This
    reads the span tree rather than the output, so it catches the case where the answer
    happens to be right but was never looked up -- which would be right by luck.
    """

    RETRIEVAL_TOOLS = ('get_financials', 'run_sql', 'search_filings')

    def evaluate(self, ctx: EvaluatorContext[str, Brief, Expectation]) -> bool:
        # A question about an uncovered company is correctly answered from the injected
        # universe list without touching the corpus, so retrieval is only required where
        # there is something to retrieve.
        if ctx.metadata is not None and not ctx.metadata.facts:
            return True
        # `SpanTree.any` walks the tree; the tool name lives on `gen_ai.tool.name`, with
        # the span name (`execute_tool <name>`) as a fallback.
        return ctx.span_tree.any(
            lambda span: (
                span.attributes.get('gen_ai.tool.name') in self.RETRIEVAL_TOOLS
                or span.name.removeprefix('execute_tool ') in self.RETRIEVAL_TOOLS
            )
        )


NARRATIVE_RUBRIC = """\
The answer should:
- respond to the question directly, without padding
- attribute statements about what a company said to a specific filing section
- avoid claims that go beyond the evidence described
- note when comparing companies whose fiscal years end at different times

Judge only these qualities. Do not verify the figures; they are checked separately.
"""


async def build_dataset(pool) -> Dataset[str, Brief, Expectation]:
    """Assemble cases, reading expected values from the corpus."""
    facts = await load_facts(
        pool,
        tickers=['NVDA', 'AAPL', 'MSFT', 'AMD', 'INTC'],
        concepts=['Revenue', 'ResearchAndDevelopment', 'NetIncome'],
        fiscal_years=[2024, 2025],
    )

    def fact(ticker: str, concept: str, year: int) -> ExpectedFact | None:
        return facts.get((ticker, concept, year))

    def require(*keys: tuple[str, str, int]) -> list[ExpectedFact]:
        return [f for f in (fact(*k) for k in keys) if f is not None]

    cases: list[Case[str, Brief, Expectation]] = [
        Case(
            name='single_figure',
            inputs="What was NVIDIA's revenue in fiscal year 2024?",
            metadata=Expectation(facts=require(('NVDA', 'Revenue', 2024))),
            evaluators=[ToolCorrectness(expected_tools=['get_financials'], allow_extra=True)],
        ),
        Case(
            name='two_figures_one_company',
            inputs="What were Apple's revenue and net income in fiscal 2024?",
            metadata=Expectation(
                facts=require(('AAPL', 'Revenue', 2024), ('AAPL', 'NetIncome', 2024))
            ),
        ),
        Case(
            name='cross_company_comparison',
            inputs='Compare revenue for NVDA, AMD and INTC in fiscal year 2025.',
            metadata=Expectation(
                facts=require(
                    ('NVDA', 'Revenue', 2025),
                    ('AMD', 'Revenue', 2025),
                    ('INTC', 'Revenue', 2025),
                )
            ),
        ),
        Case(
            name='computed_ratio',
            inputs=(
                'What was AMD R&D spend as a percentage of revenue in fiscal 2025? '
                'Show the underlying figures.'
            ),
            metadata=Expectation(
                facts=require(('AMD', 'ResearchAndDevelopment', 2025), ('AMD', 'Revenue', 2025)),
                ratios=[
                    (
                        facts[('AMD', 'ResearchAndDevelopment', 2025)],
                        facts[('AMD', 'Revenue', 2025)],
                    )
                ]
                if ('AMD', 'ResearchAndDevelopment', 2025) in facts
                and ('AMD', 'Revenue', 2025) in facts
                else [],
            ),
        ),
        Case(
            name='loss_making_company',
            inputs='What was Intel net income in fiscal 2024?',
            metadata=Expectation(facts=require(('INTC', 'NetIncome', 2024))),
        ),
        Case(
            name='narrative_question',
            inputs='What does NVIDIA say about competition in its most recent risk factors?',
            metadata=Expectation(),
            evaluators=[
                ToolCorrectness(expected_tools=['search_filings'], allow_extra=True),
                LLMJudge(rubric=NARRATIVE_RUBRIC, model=judge_model(), include_input=True),
            ],
        ),
        Case(
            name='mixed_numeric_and_narrative',
            inputs=(
                "How has Microsoft's R&D spend changed from fiscal 2024 to 2025, and what "
                'does the company say about competition?'
            ),
            metadata=Expectation(
                facts=require(
                    ('MSFT', 'ResearchAndDevelopment', 2024),
                    ('MSFT', 'ResearchAndDevelopment', 2025),
                )
            ),
            evaluators=[LLMJudge(rubric=NARRATIVE_RUBRIC, model=judge_model(), include_input=True)],
        ),
        Case(
            name='refuses_uncovered_company',
            inputs='What was Boeing revenue in fiscal 2024?',
            metadata=Expectation(forbidden_tickers=['BA']),
        ),
    ]

    return Dataset[str, Brief, Expectation](
        name='analyst',
        cases=cases,
        evaluators=[
            StatesReportedFigures(),
            ComputesRatiosCorrectly(),
            EveryFindingIsCited(),
            AnswersWhenDataExists(),
            NoUncoveredCompanies(),
            RetrievedRatherThanRecalled(),
            MaxDuration(seconds=300),
        ],
    )


async def run(deps: EdgarDeps, *, full: bool = False, repeat: int = 1) -> None:
    """Evaluate the analyst (or the full harness researcher) against ground truth."""
    # Span-based evaluators need both a tracer provider and instrumented agents.
    observability.configure('edgar-desk-eval')
    Agent.instrument_all()

    if full:
        from edgar_desk.agents.researcher import build_researcher

        agent = build_researcher()
    else:
        from edgar_desk.agents.analyst import analyst_agent

        agent = analyst_agent

    dataset = await build_dataset(deps.pool)

    async def answer(question: str) -> Brief:
        result = await agent.run(
            question,
            deps=deps,
            usage_limits=UsageLimits(request_limit=20, tool_calls_limit=30),
        )
        return result.output

    report = await dataset.evaluate(
        answer,
        name=f'analyst::{"researcher" if full else "analyst"}',
        # Local inference is memory-bound; parallel cases just queue at the model server.
        max_concurrency=1,
        repeat=repeat,
    )
    # `include_reasons` surfaces the LLM judge's justification, which is the only way to
    # tell a real quality problem from a rubric the judge is reading differently.
    report.print(include_input=False, include_output=False, include_reasons=True)
