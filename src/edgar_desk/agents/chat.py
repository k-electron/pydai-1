"""The agent behind the web UI.

Differs from the analyst in two ways that matter for an interactive surface: it answers
in prose rather than a rigid `Brief` (a chat that can only reply with JSON is a poor
chat), and it can pause mid-run to ask for approval before publishing.
"""

from __future__ import annotations

from datetime import date

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import DeferredToolRequests

from edgar_desk.capabilities import financials_capability, narrative_capability
from edgar_desk.capabilities.publishing import publishing_capability
from edgar_desk.deps import EdgarDeps
from edgar_desk.models import DEFAULT_MODEL_SETTINGS, primary_model

INSTRUCTIONS = """\
You are EDGAR Desk, a research assistant that works only from SEC filings.

Answer conversationally, but never state a figure you did not retrieve with a tool. Name
the company, fiscal year, and source for every number and every quotation. When you
compute a ratio, show the inputs.

Numeric coverage reaches back further than filing text, so check `describe_coverage`
before saying a year is unavailable. Fiscal years are named for the calendar year they
end in, and companies have different year ends: say so when comparing across companies.

If the data cannot answer something, say so plainly instead of filling the gap.
"""

chat_agent = Agent(
    primary_model(),
    deps_type=EdgarDeps,
    # The union output is what lets the run pause: when the model calls an
    # approval-gated tool, the run ends with `DeferredToolRequests` instead of text, and
    # resumes once the caller supplies matching results.
    output_type=[str, DeferredToolRequests],
    instructions=INSTRUCTIONS,
    name='chat',
    retries=2,
    model_settings=DEFAULT_MODEL_SETTINGS,
    capabilities=[financials_capability, narrative_capability, publishing_capability],
)


@chat_agent.instructions
def coverage(ctx: RunContext[EdgarDeps]) -> str:
    tickers = ', '.join(sorted(ctx.deps.covered_tickers))
    return f'Covered companies: {tickers}.\nToday is {date.today().isoformat()}.'
