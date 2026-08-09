"""The researcher: the analyst agent with a full harness around it.

Where `analyst_agent` is the minimal composition of two capabilities, this is what the
same agent looks like equipped for long, multi-company work: it plans, delegates,
remembers, keeps itself inside the context window, and refuses to answer without
evidence. Every one of those is a capability in a list, not a change to the agent loop.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.memory import FileStore, Memory
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.system_reminders import Reminder, SystemReminders
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

from edgar_desk.capabilities import financials_capability, narrative_capability
from edgar_desk.capabilities.citations import RequireCitations
from edgar_desk.deps import EdgarDeps
from edgar_desk.models import DEFAULT_MODEL_SETTINGS, primary_model
from edgar_desk.schemas import Brief

MEMORY_DIR = Path('data/memory')

INSTRUCTIONS = """\
You are a senior financial research analyst working only from SEC filings.

Work in this order:
1. Write a short plan when the question covers more than one company or more than one
   kind of evidence.
2. Gather evidence. Delegate per-company digging to your sub-agents when several
   companies are in scope; do the work yourself when only one is.
3. Write the brief.

Every finding needs a claim, at least one citation, and a confidence between 0 and 1.

Rules that matter:
- Never state a figure you did not retrieve with a tool.
- When you compute a ratio or growth rate, cite both inputs.
- Numeric coverage reaches back much further than filing text; check `describe_coverage`
  before saying a year is unavailable.
- Fiscal years are named for the calendar year they end in, and companies have different
  year ends, so note that when comparing across companies.
"""

# The specialists. Each is the same underlying loop with a different slice of the
# toolset, which keeps a sub-agent's context focused on one kind of evidence.
financial_specialist = Agent(
    primary_model(),
    deps_type=EdgarDeps,
    instructions=(
        'You answer one narrow question about reported financial figures. '
        'Retrieve the numbers, state them exactly, and name the concept and fiscal year '
        'for each. Do not speculate beyond what you retrieved.\n'
        'A relative window such as "the last five years" means the most recent years '
        'available, so check coverage and count back from the latest fiscal year rather '
        'than starting wherever the data happens to begin.'
    ),
    name='financial-specialist',
    model_settings=DEFAULT_MODEL_SETTINGS,
    capabilities=[financials_capability],
)

narrative_specialist = Agent(
    primary_model(),
    deps_type=EdgarDeps,
    instructions=(
        'You answer one narrow question about what companies say in their filings. '
        'Quote the filing where wording matters, and always name the company, fiscal '
        'year, and section. Do not speculate beyond the passages you retrieved.'
    ),
    name='narrative-specialist',
    model_settings=DEFAULT_MODEL_SETTINGS,
    capabilities=[narrative_capability],
)


def build_researcher(*, memory_dir: Path = MEMORY_DIR) -> Agent[EdgarDeps, Brief]:
    """Assemble the researcher.

    A function rather than a module-level agent because the memory store touches the
    filesystem, and tests need to point that somewhere disposable.
    """
    memory_dir.mkdir(parents=True, exist_ok=True)

    agent = Agent(
        primary_model(),
        deps_type=EdgarDeps,
        output_type=Brief,
        instructions=INSTRUCTIONS,
        name='researcher',
        retries=3,
        model_settings=DEFAULT_MODEL_SETTINGS,
        # Bounded fan-out: sub-agents share one resident model, and unbounded
        # concurrency just queues requests at the model server.
        max_concurrency=3,
        capabilities=[
            financials_capability,
            narrative_capability,
            Planning(),
            SubAgents(
                agents=[
                    SubAgent(
                        agent=financial_specialist,
                        name='financials',
                        description=(
                            'Ask for reported figures for one company: revenue, margins, '
                            'R&D, cash flow, balance sheet items.'
                        ),
                    ),
                    SubAgent(
                        agent=narrative_specialist,
                        name='narrative',
                        description=(
                            'Ask what one company says in its filings: risk factors, '
                            'strategy, competition, management discussion.'
                        ),
                    ),
                ],
                # Sub-agent token usage rolls up into the parent run, so one number
                # reflects the whole tree.
                forward_usage=True,
            ),
            Memory(store=FileStore(memory_dir), agent_name='edgar-desk'),
            # A single risk-factors section can run past 100k characters, and an
            # unbounded tool return stays in history and is re-sent on every request.
            # Head-and-tail truncation keeps the start of a passage and its conclusion,
            # which is where filings put the point.
            ToolOutputLimits(bands=[Band(over=12_000, action=Truncate(max_chars=12_000))]),
            SlidingWindowCompaction(max_tokens=90_000, keep_messages=6),
            SystemReminders(
                reminders=[
                    Reminder(
                        content=(
                            'Reminder: cite every finding, and never state a figure you '
                            'did not retrieve with a tool.'
                        ),
                        interval=4,
                    )
                ]
            ),
            RequireCitations(),
        ],
    )

    @agent.instructions
    def coverage(ctx: RunContext[EdgarDeps]) -> str:
        tickers = ', '.join(sorted(ctx.deps.covered_tickers))
        return f'Covered companies: {tickers}.\nToday is {date.today().isoformat()}.'

    return agent
