"""Ingestion: SEC APIs in, queryable corpus out.

Two independent passes, because the two halves of the corpus fail differently. Facts
come from one JSON document per company and are cheap. Narrative text needs a multi-
megabyte HTML fetch per filing plus embedding, so it is slower and worth resuming.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date

import asyncpg

from edgar_desk.edgar import EdgarClient
from edgar_desk.edgar.chunking import chunk_text
from edgar_desk.edgar.documents import html_to_text, split_sections
from edgar_desk.edgar.facts import normalize_company_facts
from edgar_desk.models import embedder
from edgar_desk.schemas import CompanyRef

EMBED_BATCH = 32


@dataclass
class IngestStats:
    companies: int = 0
    facts: int = 0
    filings: int = 0
    chunks: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


async def ingest_facts(
    pool: asyncpg.Pool,
    client: EdgarClient,
    company: CompanyRef,
    min_fiscal_year: int = 2018,
) -> int:
    """Load one company's XBRL facts. Returns the number of rows written."""
    assert company.cik is not None
    payload = await client.company_facts(company.cik)
    facts = normalize_company_facts(
        payload, cik=company.cik, ticker=company.ticker, min_fiscal_year=min_fiscal_year
    )
    if not facts:
        return 0

    rows = [
        (
            f.cik,
            f.ticker,
            f.taxonomy,
            f.tag,
            f.concept,
            f.unit,
            f.value,
            f.fiscal_year,
            f.fiscal_period,
            f.form,
            f.start_date,
            f.end_date,
            f.accession,
            f.frame,
        )
        for f in facts
    ]

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO xbrl_facts (
                cik, ticker, taxonomy, tag, concept, unit, value,
                fiscal_year, fiscal_period, form, start_date, end_date, accession, frame
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (cik, concept, unit, start_date, end_date)
            DO UPDATE SET value = EXCLUDED.value,
                          accession = EXCLUDED.accession,
                          tag = EXCLUDED.tag,
                          fiscal_year = EXCLUDED.fiscal_year,
                          fiscal_period = EXCLUDED.fiscal_period,
                          form = EXCLUDED.form
            """,
            rows,
        )
    return len(rows)


async def _recent_10k_filings(
    client: EdgarClient, company: CompanyRef, limit: int
) -> list[dict[str, str]]:
    assert company.cik is not None
    submissions = await client.submissions(company.cik)
    recent = submissions['filings']['recent']
    out: list[dict[str, str]] = []
    for i, form in enumerate(recent['form']):
        if form != '10-K':
            continue
        out.append(
            {
                'accession': recent['accessionNumber'][i],
                'filed_on': recent['filingDate'][i],
                'period_end': recent['reportDate'][i],
                'document': recent['primaryDocument'][i],
            }
        )
        if len(out) >= limit:
            break
    return out


async def ingest_narrative(
    pool: asyncpg.Pool,
    client: EdgarClient,
    company: CompanyRef,
    *,
    filings_per_company: int = 2,
) -> tuple[int, int]:
    """Load recent 10-K narrative sections as embedded chunks.

    Returns (filings ingested, chunks written). Filings already present are skipped, so
    re-running after an interruption resumes rather than re-embedding.
    """
    assert company.cik is not None
    filings = await _recent_10k_filings(client, company, filings_per_company)
    embed = embedder()
    filings_done = 0
    chunks_written = 0

    for filing in filings:
        accession = filing['accession']
        async with pool.acquire() as conn:
            already = await conn.fetchval(
                'SELECT count(*) FROM chunks WHERE accession = $1', accession
            )
        if already:
            continue

        fiscal_year = int(filing['period_end'][:4])
        html = await client.filing_document(company.cik, accession, filing['document'])
        sections = split_sections(html_to_text(html))
        if not sections:
            continue

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO filings (accession, cik, form, fiscal_year, filed_on, period_end,
                                     primary_doc)
                VALUES ($1,$2,'10-K',$3,$4,$5,$6)
                ON CONFLICT (accession) DO NOTHING
                """,
                accession,
                company.cik,
                fiscal_year,
                # asyncpg binds by inferred parameter type rather than honoring an inline
                # cast, so DATE columns need real `date` objects, not ISO strings.
                date.fromisoformat(filing['filed_on']),
                date.fromisoformat(filing['period_end']),
                filing['document'],
            )

        pending: list[tuple[str, int, str]] = []
        for section in sections:
            for chunk in chunk_text(section.text):
                pending.append((section.title, chunk.index, chunk.text))

        for start in range(0, len(pending), EMBED_BATCH):
            batch = pending[start : start + EMBED_BATCH]
            result = await embed.embed([text for _, _, text in batch], input_type='document')
            rows = [
                (
                    accession,
                    company.cik,
                    company.ticker,
                    section_title,
                    fiscal_year,
                    chunk_index,
                    text,
                    max(1, len(text) // 4),
                    list(vector),
                )
                for (section_title, chunk_index, text), vector in zip(
                    batch, result.embeddings, strict=True
                )
            ]
            async with pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO chunks (accession, cik, ticker, section, fiscal_year,
                                        chunk_index, text, token_count, embedding)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    ON CONFLICT (accession, section, chunk_index) DO NOTHING
                    """,
                    rows,
                )
            chunks_written += len(rows)

        filings_done += 1

    return filings_done, chunks_written


async def ingest_all(
    pool: asyncpg.Pool,
    companies: tuple[CompanyRef, ...],
    *,
    facts: bool = True,
    narrative: bool = True,
    filings_per_company: int = 2,
    progress: object | None = None,
) -> IngestStats:
    stats = IngestStats()
    assert stats.errors is not None

    async with EdgarClient() as client:
        for company in companies:
            try:
                if facts:
                    stats.facts += await ingest_facts(pool, client, company)
                if narrative:
                    done, written = await ingest_narrative(
                        pool, client, company, filings_per_company=filings_per_company
                    )
                    stats.filings += done
                    stats.chunks += written
                stats.companies += 1
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f'{company.ticker}: {type(exc).__name__}: {exc}')
            if progress is not None and callable(progress):
                progress(company, stats)

    return stats


async def refresh_vector_index(pool: asyncpg.Pool) -> None:
    """Rebuild the HNSW index once rows exist.

    Building it on an empty table produced an index with no useful structure; a rebuild
    after ingestion is far faster than incremental insertion into an empty HNSW graph.
    """
    async with pool.acquire() as conn:
        await conn.execute('REINDEX INDEX chunks_embedding_idx')
        await conn.execute('ANALYZE chunks')


def run_ingest(**kwargs: object) -> IngestStats:
    from edgar_desk import db

    async def main() -> IngestStats:
        async with db.pool_context() as pool:
            await db.apply_schema(pool)
            await db.seed_companies(pool)
            return await ingest_all(pool, **kwargs)  # type: ignore[arg-type]

    return asyncio.run(main())
