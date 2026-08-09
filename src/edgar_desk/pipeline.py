"""The research pipeline as an explicit graph.

The researcher agent decides its own control flow; this expresses the same work as a
structure you can read, render, and reason about. Decomposition, fan-out, aggregation and
the decision to dig deeper become nodes and edges rather than emergent behavior, which is
the trade: less adaptive, far more predictable.

Shape:

    start -> triage -> (map over sub-questions) -> investigate -> join
          -> assess -> decision -> synthesize -> end
                              \\-> deepen -> synthesize

`graph.render()` produces a Mermaid diagram of exactly this, generated from the graph
rather than maintained by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_graph import GraphBuilder, StepContext, TypeExpression, reduce_list_append

from edgar_desk.agents.researcher import financial_specialist, narrative_specialist
from edgar_desk.agents.triage import TriageDeps, triage_agent
from edgar_desk.deps import EdgarDeps
from edgar_desk.schemas import Angle, Brief, ResearchPlan, SubQuestion

MAX_SUB_QUESTIONS = 6
MIN_EVIDENCE_CHARS = 200


@dataclass
class Evidence:
    """What one sub-question turned up."""

    sub_question: SubQuestion
    findings: str
    tools_used: list[str] = field(default_factory=list)

    @property
    def is_substantive(self) -> bool:
        return len(self.findings.strip()) >= MIN_EVIDENCE_CHARS


@dataclass
class Sufficient:
    """Enough evidence to write the brief."""

    evidence: list[Evidence]


@dataclass
class NeedsMore:
    """Coverage is thin; dig further before writing."""

    evidence: list[Evidence]
    gaps: list[SubQuestion]


@dataclass
class ResearchState:
    """Mutable state shared by every node in one run."""

    question: str
    plan: ResearchPlan | None = None
    rounds: int = 0
    tools_used: list[str] = field(default_factory=list)

    @property
    def companies(self) -> list[str]:
        return [c.ticker for c in self.plan.companies] if self.plan else []


def build_graph():
    """Assemble the pipeline.

    Built inside a function so the agents it references can be overridden in tests
    before the graph closes over them.
    """
    g = GraphBuilder(
        name='edgar-research',
        state_type=ResearchState,
        deps_type=EdgarDeps,
        input_type=str,
        output_type=Brief,
    )

    @g.step
    async def triage(ctx: StepContext[ResearchState, EdgarDeps, str]) -> list[SubQuestion]:
        """Decompose the question into independently answerable parts."""
        ctx.state.question = ctx.inputs
        result = await triage_agent.run(ctx.inputs, deps=TriageDeps(covered=ctx.deps.covered))
        ctx.state.plan = result.output
        return result.output.sub_questions[:MAX_SUB_QUESTIONS]

    @g.step
    async def investigate(
        ctx: StepContext[ResearchState, EdgarDeps, SubQuestion],
    ) -> Evidence:
        """Answer one sub-question with the specialist its angle calls for.

        This is the node `.map()` fans out to, so each sub-question runs concurrently
        against its own specialist.
        """
        question = ctx.inputs
        scope = ', '.join(question.companies or ctx.state.companies) or 'the covered universe'
        prompt = f'{question.question}\n\nCompanies in scope: {scope}'

        agents = (
            [financial_specialist, narrative_specialist]
            if question.angle is Angle.BOTH
            else [
                financial_specialist if question.angle is Angle.FINANCIAL else narrative_specialist
            ]
        )

        parts: list[str] = []
        for agent in agents:
            result = await agent.run(prompt, deps=ctx.deps)
            parts.append(str(result.output))

        return Evidence(sub_question=question, findings='\n\n'.join(parts))

    collect = g.join(reduce_list_append, initial_factory=list[Evidence])

    @g.step
    async def assess(
        ctx: StepContext[ResearchState, EdgarDeps, list[Evidence]],
    ) -> Sufficient | NeedsMore:
        """Decide whether the evidence supports a brief yet.

        Deliberately a plain function rather than a model call: "did every sub-question
        come back with something substantive" is a property of the data, and asking a
        model to judge it adds latency and a failure mode for no benefit.
        """
        evidence = ctx.inputs
        ctx.state.rounds += 1

        gaps = [e.sub_question for e in evidence if not e.is_substantive]
        # One extra round only. A second pass over the same empty corpus finds the same
        # nothing, and the caller would rather have a caveated answer than a long wait.
        if gaps and ctx.state.rounds < 2:
            return NeedsMore(evidence=evidence, gaps=gaps)
        return Sufficient(evidence=evidence)

    @g.step
    async def deepen(
        ctx: StepContext[ResearchState, EdgarDeps, NeedsMore],
    ) -> Sufficient:
        """Retry the thin sub-questions with a broader search."""
        filled = list(ctx.inputs.evidence)
        for gap in ctx.inputs.gaps:
            broadened = (
                f'{gap.question}\n\nEarlier search found little. Widen the search: drop '
                f'year filters, try related wording, and check coverage first.'
            )
            agent = financial_specialist if gap.angle is Angle.FINANCIAL else narrative_specialist
            result = await agent.run(broadened, deps=ctx.deps)
            filled.append(Evidence(sub_question=gap, findings=str(result.output)))
        return Sufficient(evidence=filled)

    @g.step
    async def synthesize(
        ctx: StepContext[ResearchState, EdgarDeps, Sufficient],
    ) -> Brief:
        """Turn gathered evidence into one cited brief."""
        from edgar_desk.agents.analyst import analyst_agent

        blocks = [
            f'### {e.sub_question.question}\n({e.sub_question.angle.value})\n\n{e.findings}'
            for e in ctx.inputs.evidence
        ]
        prompt = (
            f'Original question: {ctx.state.question}\n\n'
            'Evidence gathered by specialists:\n\n' + '\n\n'.join(blocks) + '\n\n'
            'Write the brief from this evidence. Verify any figure you are unsure of with '
            'your own tools before using it.'
        )
        result = await analyst_agent.run(prompt, deps=ctx.deps)
        brief = result.output
        brief.question = ctx.state.question
        return brief

    g.add(
        g.edge_from(g.start_node).to(triage),
        # One parallel branch per sub-question.
        g.edge_from(triage).map().to(investigate),
        g.edge_from(investigate).to(collect),
        g.edge_from(collect).to(assess),
        g.edge_from(assess).to(
            g.decision()
            .branch(
                g.match(TypeExpression[Sufficient]).label('evidence is sufficient').to(synthesize)
            )
            .branch(g.match(TypeExpression[NeedsMore]).label('coverage is thin').to(deepen))
        ),
        g.edge_from(deepen).to(synthesize),
        g.edge_from(synthesize).to(g.end_node),
    )

    return g.build()


async def run_pipeline(question: str, deps: EdgarDeps) -> tuple[Brief, ResearchState]:
    """Run the pipeline and return the brief alongside the state it accumulated."""
    graph = build_graph()
    state = ResearchState(question=question)
    brief = await graph.run(state=state, deps=deps, inputs=question)
    return brief, state


def render_mermaid() -> str:
    """Mermaid diagram of the pipeline, generated from the graph itself."""
    return build_graph().render(title='EDGAR research pipeline', direction='TB')
