"""The financials capability: reported numbers, by structured lookup or SQL."""

from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability

from edgar_desk.deps import EdgarDeps
from edgar_desk.retrieval.facts import (
    CONCEPT_NAMES,
    SCHEMA_HINT,
    query_facts,
    run_readonly_sql,
)
from edgar_desk.retrieval.sql_guard import UnsafeQuery

INSTRUCTIONS = f"""\
You have access to reported financial figures from SEC filings.

Prefer `get_financials` for straightforward lookups: it takes tickers, concepts and years
and returns exact reported values. Reach for `run_sql` only when the question needs
something `get_financials` cannot express, such as a ratio computed across concepts, a
ranking, or a growth rate.

Never state a number you did not get from one of these tools. If a figure is missing,
say so rather than estimating it.

A relative window such as "the last five years" means the most recent years available.
Find the latest fiscal year first and count back from it, rather than starting from the
earliest year in the data.

Available concepts: {', '.join(CONCEPT_NAMES)}
"""

financials_capability: Capability[EdgarDeps] = Capability(
    id='financials',
    description='Look up reported financial figures from SEC XBRL data.',
    instructions=INSTRUCTIONS,
)


@financials_capability.tool
async def get_financials(
    ctx: RunContext[EdgarDeps],
    tickers: list[str],
    concepts: list[str] | None = None,
    fiscal_years: list[int] | None = None,
    period: str = 'FY',
) -> str:
    """Look up reported financial figures.

    Args:
        tickers: Ticker symbols, e.g. ["NVDA", "AMD"].
        concepts: Concepts to fetch, e.g. ["Revenue", "ResearchAndDevelopment"].
            Omit to return every available concept.
        fiscal_years: Fiscal years to include, e.g. [2023, 2024]. Omit for all years.
        period: "FY" for annual figures, "Q1".."Q4" for quarters, "ANY" for everything.
    """
    ctx.deps.record('get_financials')

    unknown = [t for t in (t.upper() for t in tickers) if t not in ctx.deps.covered_tickers]
    if unknown:
        return (
            f'Not covered: {", ".join(unknown)}. '
            f'Covered companies: {", ".join(sorted(ctx.deps.covered_tickers))}.'
        )

    records = await query_facts(
        ctx.deps.pool,
        tickers=tickers,
        concepts=concepts,
        fiscal_years=fiscal_years,
        period=period,
        limit=ctx.deps.max_rows,
    )
    if not records:
        return 'No matching figures. Try a different concept, year, or period.'
    return '\n'.join(r.render() for r in records)


@financials_capability.tool
async def run_sql(ctx: RunContext[EdgarDeps], sql: str) -> str:
    """Run a read-only SQL query for calculations `get_financials` cannot express.

    Args:
        sql: A single SELECT statement.
    """
    ctx.deps.record('run_sql')
    try:
        rows = await run_readonly_sql(ctx.deps.pool, sql, max_rows=ctx.deps.max_rows)
    except UnsafeQuery as exc:
        # Returned rather than raised: the model can fix a rejected query on the next
        # step, and a raise would end the run.
        return f'Query rejected: {exc}'
    except Exception as exc:  # noqa: BLE001
        return f'Query failed: {type(exc).__name__}: {exc}'

    if not rows:
        return 'Query returned no rows.'

    headers = list(rows[0])
    lines = [' | '.join(headers)]
    lines.extend(' | '.join(_format(row[h]) for h in headers) for row in rows)
    return '\n'.join(lines)


@financials_capability.tool_plain
def describe_schema() -> str:
    """Show the tables, columns, and concepts available to `run_sql`."""
    return SCHEMA_HINT


def _format(value: object) -> str:
    if isinstance(value, float):
        return f'{value:,.4g}'
    return str(value)
