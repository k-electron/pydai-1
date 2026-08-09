"""Querying reported financials."""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from edgar_desk.edgar.facts import CONCEPTS
from edgar_desk.retrieval.sql_guard import validate_select

STATEMENT_TIMEOUT_MS = 8000

CONCEPT_NAMES: tuple[str, ...] = tuple(CONCEPTS)


@dataclass(frozen=True, slots=True)
class FactRecord:
    ticker: str
    concept: str
    fiscal_year: int
    fiscal_period: str
    unit: str
    value: float
    end_date: str
    form: str
    accession: str | None

    def render(self) -> str:
        """Compact one-line form.

        Rendered rather than handed over as JSON: the model reads these in bulk, and
        magnitudes are far easier to compare as 60.92B than as 60922000000.0.
        """
        return (
            f'{self.ticker} {self.concept} {self.fiscal_period}{self.fiscal_year} '
            f'= {_human(self.value)} {self.unit} (ended {self.end_date}, {self.form})'
        )


def _human(value: float) -> str:
    magnitude = abs(value)
    for threshold, suffix in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if magnitude >= threshold:
            return f'{value / threshold:,.3f}{suffix}'
    return f'{value:,.2f}'


async def query_facts(
    pool: asyncpg.Pool,
    *,
    tickers: list[str],
    concepts: list[str] | None = None,
    fiscal_years: list[int] | None = None,
    period: str = 'FY',
    limit: int = 200,
) -> list[FactRecord]:
    """Structured lookup of reported figures.

    The default `period='FY'` matters: it is what keeps a question about annual revenue
    from also returning the quarterly breakdowns that share the same filing.
    """
    conditions = ['ticker = ANY($1)']
    args: list[object] = [[t.upper() for t in tickers]]

    if concepts:
        args.append(concepts)
        conditions.append(f'concept = ANY(${len(args)})')
    if fiscal_years:
        args.append(fiscal_years)
        conditions.append(f'fiscal_year = ANY(${len(args)})')
    if period and period.upper() != 'ANY':
        args.append(period.upper())
        conditions.append(f'fiscal_period = ${len(args)}')

    args.append(limit)
    sql = f"""
        SELECT ticker, concept, fiscal_year, fiscal_period, unit, value,
               end_date::text AS end_date, form, accession
        FROM xbrl_facts
        WHERE {' AND '.join(conditions)}
        ORDER BY ticker, concept, fiscal_year DESC
        LIMIT ${len(args)}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [FactRecord(**dict(r)) for r in rows]


async def run_readonly_sql(
    pool: asyncpg.Pool,
    sql: str,
    *,
    max_rows: int = 200,
) -> list[dict]:
    """Execute a validated SELECT under read-only enforcement.

    Two independent protections, because guard code can have gaps: `validate_select`
    rejects the statement up front with a message the model can act on, and the
    transaction is `READ ONLY` with a statement timeout so anything that slips past the
    guard still cannot write or run away.
    """
    statement = validate_select(sql)

    async with pool.acquire() as conn, conn.transaction(readonly=True):
        await conn.execute(f'SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}')
        rows = await conn.fetch(f'SELECT * FROM ({statement}) AS model_query LIMIT {max_rows}')
    return [dict(r) for r in rows]


SCHEMA_HINT = f"""\
Tables available for read-only SQL:

xbrl_facts(ticker, concept, tag, unit, value, fiscal_year, fiscal_period, form,
           start_date, end_date, accession)
  One row per company per concept per period. `concept` is a canonical name; use it
  rather than `tag`. `fiscal_period` is one of FY, Q1-Q4, YTD, PIT.
  Filter on fiscal_period='FY' for annual figures, or annual and quarterly values
  will be mixed together.

filings(accession, cik, form, fiscal_year, filed_on, period_end, primary_doc)
companies(cik, ticker, name)
chunks(accession, cik, ticker, section, fiscal_year, chunk_index, text)

Available concepts: {', '.join(CONCEPT_NAMES)}

Fiscal years are named for the calendar year the year ends in, so NVIDIA's FY2024 ended
January 2024 and Apple's FY2024 ended September 2024. Comparing "the same year" across
companies with different year ends compares different periods.
"""
