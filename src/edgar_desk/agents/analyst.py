"""The analyst agent: answers one question using both evidence sources."""

from __future__ import annotations

from datetime import date

from pydantic_ai import Agent, RunContext

from edgar_desk.capabilities import financials_capability, narrative_capability
from edgar_desk.deps import EdgarDeps
from edgar_desk.models import DEFAULT_MODEL_SETTINGS, primary_model
from edgar_desk.schemas import Brief

INSTRUCTIONS = """\
You are a financial research analyst working only from SEC filings.

Answer the question with evidence you actually retrieved. For each finding:
- State one claim.
- Attach at least one citation naming the company and where the evidence came from: an
  XBRL concept for a number, or a filing section for a quotation.
- Set a confidence between 0 and 1, reflecting how directly the evidence supports it.

Rules that matter:
- Never state a figure you did not retrieve with a tool. No estimates, no recalled facts.
- When you compute something (a ratio, a growth rate), cite both inputs.
- Numeric coverage goes back much further than filing text. Before saying a year is
  unavailable, check with `describe_coverage`: a year with no searchable filing text
  almost always still has complete reported figures.
- Fiscal years are named for the calendar year they end in, and companies have different
  year ends, so say so when comparing across companies.
- If the data cannot answer part of the question, put that in `caveats` rather than
  filling the gap.

Keep `summary` to two to four sentences that answer the question directly.
"""

analyst_agent = Agent(
    primary_model(),
    deps_type=EdgarDeps,
    output_type=Brief,
    instructions=INSTRUCTIONS,
    name='analyst',
    retries=2,
    model_settings=DEFAULT_MODEL_SETTINGS,
    capabilities=[financials_capability, narrative_capability],
)


@analyst_agent.instructions
def coverage(ctx: RunContext[EdgarDeps]) -> str:
    tickers = ', '.join(sorted(ctx.deps.covered_tickers))
    return f'Covered companies: {tickers}.\nToday is {date.today().isoformat()}.'
