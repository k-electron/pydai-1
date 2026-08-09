"""The triage agent: turns a free-form request into a typed `ResearchPlan`.

This is the smallest agent in the project and the first one built, so it doubles as
the reference for the three core mechanics: typed output, a tool, and dependency
injection through `RunContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext

from edgar_desk.models import DEFAULT_MODEL_SETTINGS, primary_model
from edgar_desk.schemas import CompanyRef, ResearchPlan
from edgar_desk.universe import SEED_COMPANIES, resolve

INSTRUCTIONS = """\
You are a research triage analyst for a system that answers questions about SEC filings.

Break the user's request into a small set of self-contained sub-questions, and identify
which companies are in scope.

Rules:
- Call `resolve_company` for every company you intend to include. Never guess a CIK.
- Copy the `cik` from the tool's result into your output verbatim. Do not leave it null
  for a company the tool resolved, and never write one the tool did not return.
- If a company is not in the covered universe, leave it out and note that in `reasoning`.
- Tag each sub-question with the evidence it needs:
  - `financial` for anything answerable from reported numbers (revenue, margin, R&D spend)
  - `narrative` for anything answerable from filing prose (risk factors, strategy, MD&A)
  - `both` when the question needs numbers explained by prose
- Produce at most 6 sub-questions. Fewer is better when the question is simple.
"""


@dataclass
class TriageDeps:
    """Dependencies for triage.

    A dataclass rather than loose kwargs so the agent stays `Agent[TriageDeps, ResearchPlan]`
    and a type checker catches a mismatched `ctx.deps` access at write time.
    """

    covered: tuple[CompanyRef, ...] = SEED_COMPANIES
    resolved: list[CompanyRef] = field(default_factory=list)
    """Records what the model actually looked up, so tests and evals can assert on it."""


triage_agent = Agent(
    primary_model(),
    deps_type=TriageDeps,
    output_type=ResearchPlan,
    instructions=INSTRUCTIONS,
    name='triage',
    retries=2,
    model_settings=DEFAULT_MODEL_SETTINGS,
)


@triage_agent.instructions
def covered_universe(ctx: RunContext[TriageDeps]) -> str:
    """Inject the covered universe so the model cannot invent a company we lack data for."""
    listing = ', '.join(f'{c.ticker} ({c.name})' for c in ctx.deps.covered)
    return f'Covered universe (only these companies have data): {listing}'


@triage_agent.tool
def resolve_company(ctx: RunContext[TriageDeps], symbol_or_name: str) -> CompanyRef | str:
    """Resolve a ticker or company name to its SEC identifiers.

    Args:
        symbol_or_name: A ticker like "NVDA" or a company name like "Nvidia".
    """
    match = resolve(symbol_or_name)
    if match is None:
        return f'{symbol_or_name!r} is not in the covered universe.'
    if match not in ctx.deps.resolved:
        ctx.deps.resolved.append(match)
    return match
