"""The narrative capability: what companies say, in their own words."""

from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability

from edgar_desk.deps import EdgarDeps
from edgar_desk.retrieval.narrative import search_passages

SECTION_CHOICES = (
    'Risk Factors',
    'Business',
    "Management's Discussion",
    'Market Risk',
)

INSTRUCTIONS = f"""\
You can search the narrative sections of 10-K filings: risk factors, business
descriptions, and management's discussion.

Search with a full question or a descriptive phrase rather than keywords; the index is
semantic, so "pricing pressure from competitors" retrieves better than "pricing".

Quote filings rather than paraphrasing when the exact wording matters, and always name
the company, fiscal year, and section you took a statement from.

Sections you can filter on: {', '.join(SECTION_CHOICES)}
"""

narrative_capability: Capability[EdgarDeps] = Capability(
    id='narrative',
    description='Search what companies say in their 10-K filings.',
    instructions=INSTRUCTIONS,
)


@narrative_capability.tool
async def search_filings(
    ctx: RunContext[EdgarDeps],
    query: str,
    tickers: list[str] | None = None,
    fiscal_years: list[int] | None = None,
    sections: list[str] | None = None,
    limit: int = 6,
) -> str:
    """Search 10-K narrative text for passages relevant to a question.

    Args:
        query: A question or descriptive phrase. Full sentences work better than keywords.
        tickers: Restrict to these companies. Omit to search all covered companies.
        fiscal_years: Restrict to these fiscal years.
        sections: Restrict to sections, e.g. ["Risk Factors"].
        limit: How many passages to return, at most 10.
    """
    ctx.deps.record('search_filings')

    if tickers:
        unknown = [t for t in (t.upper() for t in tickers) if t not in ctx.deps.covered_tickers]
        if unknown:
            return f'Not covered: {", ".join(unknown)}.'

    passages = await search_passages(
        ctx.deps.pool,
        ctx.deps.embedder,
        query,
        tickers=tickers,
        fiscal_years=fiscal_years,
        sections=sections,
        limit=max(1, min(limit, 10)),
        rerank=ctx.deps.rerank,
    )
    if not passages:
        return 'No matching passages. Try rephrasing, or widen the company or year filter.'
    return '\n\n---\n\n'.join(p.render() for p in passages)


@narrative_capability.tool
async def describe_coverage(ctx: RunContext[EdgarDeps], ticker: str) -> str:
    """Report which years are available for a company, for numbers and for filing text.

    The two differ, so check this before concluding that a year is missing.

    Args:
        ticker: A ticker symbol, e.g. "NVDA".
    """
    ctx.deps.record('describe_coverage')
    symbol = ticker.upper()

    async with ctx.deps.pool.acquire() as conn:
        filings = await conn.fetch(
            """
            SELECT f.fiscal_year, f.form, f.period_end::text AS period_end, count(c.id) AS chunks
            FROM filings f
            LEFT JOIN chunks c ON c.accession = f.accession
            JOIN companies co ON co.cik = f.cik
            WHERE co.ticker = $1
            GROUP BY f.fiscal_year, f.form, f.period_end
            ORDER BY f.fiscal_year DESC
            """,
            symbol,
        )
        facts = await conn.fetchrow(
            """
            SELECT min(fiscal_year) AS first_year, max(fiscal_year) AS last_year,
                   count(DISTINCT concept) AS concepts
            FROM xbrl_facts
            WHERE ticker = $1 AND fiscal_period = 'FY'
            """,
            symbol,
        )

    lines: list[str] = []
    if facts and facts['first_year'] is not None:
        lines.append(
            f'Financial figures: annual data for FY{facts["first_year"]}-FY{facts["last_year"]} '
            f'across {facts["concepts"]} concepts. Use `get_financials` for any of these years.'
        )
    else:
        lines.append('Financial figures: none held.')

    if filings:
        years = ', '.join(f'FY{r["fiscal_year"]}' for r in filings)
        lines.append(
            f'Filing text: {len(filings)} 10-K(s) held ({years}). '
            'Narrative search only covers these years.'
        )
    else:
        lines.append('Filing text: none held.')

    lines.append(
        'These two ranges are independent: a year missing from filing text usually still '
        'has complete financial figures.'
    )
    return '\n'.join(lines)
