"""Capability and tool tests.

Tools are exercised directly against the real corpus (skipped when Postgres is down),
and agent wiring is checked with `TestModel` so no inference is needed.
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.test import TestModel

from edgar_desk import db
from edgar_desk.capabilities import financials_capability, narrative_capability
from edgar_desk.deps import EdgarDeps
from edgar_desk.observability import collector_reachable
from edgar_desk.schemas import Brief
from edgar_desk.settings import get_settings


def _postgres_up() -> bool:
    url = get_settings().database_url
    hostport = url.rsplit('@', 1)[-1].split('/')[0]
    host, _, port = hostport.partition(':')
    return collector_reachable(f'http://{host}:{port or 5432}')


pytestmark = pytest.mark.skipif(not _postgres_up(), reason='postgres not running')


@pytest.fixture
async def deps():
    async with db.pool_context(max_size=2) as pool:
        # Reranking loads a multi-gigabyte cross-encoder; irrelevant to tool wiring.
        yield EdgarDeps(pool=pool, rerank=False)


@pytest.fixture
def probe_agent():
    """An agent carrying both capabilities, driven by a scripted model."""
    return Agent(
        TestModel(),
        deps_type=EdgarDeps,
        output_type=Brief,
        capabilities=[financials_capability, narrative_capability],
        name='probe',
    )


async def test_capabilities_contribute_their_tools(probe_agent, deps) -> None:
    with capture_run_messages() as messages:
        await probe_agent.run('anything', deps=deps)

    called = {
        part.tool_name
        for message in messages
        for part in message.parts
        if part.part_kind == 'tool-call'
    }
    assert {'get_financials', 'run_sql', 'describe_schema'} <= called
    assert {'search_filings', 'describe_coverage'} <= called


async def test_capability_instructions_reach_the_model(probe_agent, deps) -> None:
    with capture_run_messages() as messages:
        await probe_agent.run('anything', deps=deps)

    instructions = messages[0].instructions or ''
    assert 'get_financials' in instructions
    assert 'Risk Factors' in instructions


async def test_get_financials_returns_known_values(deps) -> None:
    """NVIDIA's reported FY2024 revenue is $60.922bn. If this drifts, ingestion broke."""
    from edgar_desk.capabilities.financials import get_financials

    ctx = _ctx(deps)
    out = await get_financials(ctx, tickers=['NVDA'], concepts=['Revenue'], fiscal_years=[2024])
    assert '60.922B' in out
    assert 'get_financials' in deps.tool_calls


async def test_get_financials_rejects_uncovered_company(deps) -> None:
    from edgar_desk.capabilities.financials import get_financials

    out = await get_financials(_ctx(deps), tickers=['BA'])
    assert 'Not covered' in out


async def test_annual_period_excludes_quarterly_rows(deps) -> None:
    """A 10-K carries quarterly breakdowns too; the FY filter must exclude them."""
    from edgar_desk.retrieval.facts import query_facts

    annual = await query_facts(
        deps.pool, tickers=['NVDA'], concepts=['Revenue'], period='FY', limit=50
    )
    years = [r.fiscal_year for r in annual]
    assert len(years) == len(set(years)), 'one annual revenue row per fiscal year'
    assert all(r.fiscal_period == 'FY' for r in annual)


async def test_run_sql_reports_rejection_instead_of_raising(deps) -> None:
    """A rejected query has to come back as text the model can act on: raising would
    end the run instead of giving it a chance to correct itself."""
    from edgar_desk.capabilities.financials import run_sql

    out = await run_sql(_ctx(deps), 'DELETE FROM chunks')
    assert out.startswith('Query rejected:')


async def test_run_sql_executes_a_valid_query(deps) -> None:
    from edgar_desk.capabilities.financials import run_sql

    out = await run_sql(
        _ctx(deps),
        'SELECT ticker, value FROM xbrl_facts '
        "WHERE ticker = 'AAPL' AND concept = 'Revenue' AND fiscal_period = 'FY' "
        'AND fiscal_year = 2024',
    )
    assert 'AAPL' in out
    assert 'rejected' not in out


async def test_search_filings_finds_relevant_passages(deps) -> None:
    from edgar_desk.capabilities.narrative import search_filings

    out = await search_filings(
        _ctx(deps), query='competition and pricing pressure', tickers=['NVDA'], limit=3
    )
    assert 'NVDA' in out
    assert 'Item' in out


async def test_describe_coverage_separates_facts_from_text(deps) -> None:
    """The agent previously concluded that missing filing text meant missing figures,
    and dropped three years of a five-year trend. Coverage has to say both."""
    from edgar_desk.capabilities.narrative import describe_coverage

    out = await describe_coverage(_ctx(deps), ticker='NVDA')
    assert 'Financial figures' in out
    assert 'Filing text' in out
    assert 'independent' in out


def _ctx(deps: EdgarDeps):
    """Minimal RunContext for calling a tool function directly."""
    from pydantic_ai import RunContext
    from pydantic_ai.usage import RunUsage

    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())
