"""EDGAR Desk as an MCP server.

Exposes the same corpus this project's agents use, so any MCP client -- Cursor, Claude
Desktop, another Pydantic AI agent -- can query SEC filings without reimplementing
ingestion. The tools mirror the capability layer, plus one that runs the whole analyst
agent and returns a cited brief.

Run over stdio (what editors expect):

    uv run edgar-desk mcp

Or over HTTP:

    uv run edgar-desk mcp --transport http --port 8931
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from edgar_desk.retrieval.facts import SCHEMA_HINT, query_facts, run_readonly_sql
from edgar_desk.retrieval.narrative import search_passages
from edgar_desk.retrieval.sql_guard import UnsafeQuery
from edgar_desk.universe import BY_TICKER, SEED_COMPANIES

server = FastMCP(
    'EDGAR Desk',
    instructions=(
        'Query SEC filings for 20 large-cap technology companies. '
        'Reported figures come from XBRL and are exact; narrative passages come from '
        '10-K sections and are retrieved semantically.'
    ),
)

_pool: Any = None
_pool_lock = asyncio.Lock()


async def _get_pool() -> Any:
    """One lazily-created pool for the server's lifetime.

    Created on first use rather than at import, so the module can be imported (and the
    tool list inspected) without a database running.
    """
    global _pool
    async with _pool_lock:
        if _pool is None:
            from edgar_desk import db

            _pool = await db.create_pool(min_size=1, max_size=5)
    return _pool


async def _get_embedder() -> Any:
    from edgar_desk.models import embedder

    return embedder()


@server.tool()
async def list_companies() -> str:
    """List the companies covered by this server, with their SEC identifiers."""
    return '\n'.join(f'{c.ticker}\t{c.name}\tCIK {c.cik}' for c in SEED_COMPANIES)


@server.tool()
async def get_financials(
    tickers: list[str],
    concepts: list[str] | None = None,
    fiscal_years: list[int] | None = None,
    period: str = 'FY',
) -> str:
    """Look up exact reported financial figures from SEC XBRL data.

    Args:
        tickers: Ticker symbols, e.g. ["NVDA", "AMD"].
        concepts: Concepts such as "Revenue" or "ResearchAndDevelopment". Omit for all.
        fiscal_years: Fiscal years to include. Omit for all available years.
        period: "FY" for annual, "Q1".."Q4" for quarters, "ANY" for everything.
    """
    unknown = [t for t in (t.upper() for t in tickers) if t not in BY_TICKER]
    if unknown:
        return f'Not covered: {", ".join(unknown)}. Call list_companies for the universe.'

    pool = await _get_pool()
    records = await query_facts(
        pool,
        tickers=tickers,
        concepts=concepts,
        fiscal_years=fiscal_years,
        period=period,
        limit=200,
    )
    if not records:
        return 'No matching figures.'
    return '\n'.join(r.render() for r in records)


@server.tool()
async def search_filings(
    query: str,
    tickers: list[str] | None = None,
    fiscal_years: list[int] | None = None,
    sections: list[str] | None = None,
    limit: int = 6,
) -> str:
    """Search 10-K narrative text semantically.

    Args:
        query: A question or descriptive phrase. Full sentences retrieve better than keywords.
        tickers: Restrict to these companies.
        fiscal_years: Restrict to these fiscal years.
        sections: Restrict to sections, e.g. ["Risk Factors"].
        limit: Passages to return, at most 10.
    """
    pool = await _get_pool()
    passages = await search_passages(
        pool,
        await _get_embedder(),
        query,
        tickers=tickers,
        fiscal_years=fiscal_years,
        sections=sections,
        limit=max(1, min(limit, 10)),
        # The cross-encoder would load a multi-gigabyte model into the editor's MCP
        # subprocess; vector order is good enough for interactive lookups.
        rerank=False,
    )
    if not passages:
        return 'No matching passages.'
    return '\n\n---\n\n'.join(p.render() for p in passages)


@server.tool()
async def run_sql(sql: str) -> str:
    """Run a read-only SELECT against the filings database.

    Args:
        sql: A single SELECT statement. Call describe_schema first for the tables.
    """
    pool = await _get_pool()
    try:
        rows = await run_readonly_sql(pool, sql, max_rows=200)
    except UnsafeQuery as exc:
        return f'Query rejected: {exc}'
    except Exception as exc:  # noqa: BLE001
        return f'Query failed: {type(exc).__name__}: {exc}'
    if not rows:
        return 'Query returned no rows.'
    return json.dumps(rows, indent=2, default=str)


@server.tool()
async def describe_schema() -> str:
    """Show the tables, columns, and concepts available to run_sql."""
    return SCHEMA_HINT


@server.tool()
async def research(question: str) -> str:
    """Answer a research question with the full analyst agent, returning a cited brief.

    Slower than the individual tools because it runs a local model over several steps.
    Use it for open questions; use get_financials or search_filings for direct lookups.

    Args:
        question: A question about the covered companies.
    """
    from edgar_desk.agents.analyst import analyst_agent
    from edgar_desk.deps import EdgarDeps

    pool = await _get_pool()
    deps = EdgarDeps(pool=pool, rerank=False)
    result = await analyst_agent.run(question, deps=deps)
    return result.output.model_dump_json(indent=2)


def run(transport: str = 'stdio', port: int = 8931) -> None:
    if transport == 'stdio':
        server.run()
    else:
        server.settings.port = port
        server.run(transport='streamable-http')


if __name__ == '__main__':
    run()
