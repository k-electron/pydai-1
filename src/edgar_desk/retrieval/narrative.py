"""Narrative retrieval: vector recall, then cross-encoder precision.

Two stages, because they are good at different things. The bi-encoder (bge-m3) embeds
query and document separately, which makes it fast enough to search every chunk but blunt
about relevance. The cross-encoder reads the query and passage *together*, which is far
more accurate and far too slow to run over the whole corpus. Recall broadly with the
first, then re-order the shortlist with the second.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import asyncpg

from edgar_desk.settings import get_settings

CANDIDATE_MULTIPLIER = 5
MAX_CANDIDATES = 60


@dataclass(frozen=True, slots=True)
class PassageRecord:
    ticker: str
    section: str
    fiscal_year: int
    accession: str
    text: str
    score: float

    def render(self, max_chars: int = 900) -> str:
        body = self.text if len(self.text) <= max_chars else f'{self.text[:max_chars]}...'
        return f'[{self.ticker} FY{self.fiscal_year} | {self.section} | {self.accession}]\n{body}'


@lru_cache(maxsize=1)
def _cross_encoder() -> Any:
    """Load the reranker once per process.

    Imported lazily: `sentence_transformers` pulls in torch, which costs seconds of
    import time and a couple of gigabytes, and most code paths never rerank.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model, device='mps')


async def search_passages(
    pool: asyncpg.Pool,
    embedder: Any,
    query: str,
    *,
    tickers: list[str] | None = None,
    fiscal_years: list[int] | None = None,
    sections: list[str] | None = None,
    limit: int = 6,
    rerank: bool = True,
) -> list[PassageRecord]:
    """Find the passages that best answer `query`."""
    embedded = await embedder.embed(query, input_type='query')
    vector = list(embedded.embeddings[0])

    conditions: list[str] = []
    args: list[object] = [vector]
    if tickers:
        args.append([t.upper() for t in tickers])
        conditions.append(f'ticker = ANY(${len(args)})')
    if fiscal_years:
        args.append(fiscal_years)
        conditions.append(f'fiscal_year = ANY(${len(args)})')
    if sections:
        args.append([f'%{s}%' for s in sections])
        conditions.append(f'section ILIKE ANY(${len(args)})')

    candidate_count = min(MAX_CANDIDATES, limit * CANDIDATE_MULTIPLIER) if rerank else limit
    args.append(candidate_count)

    where = f'WHERE {" AND ".join(conditions)}' if conditions else ''
    sql = f"""
        SELECT ticker, section, fiscal_year, accession, text,
               1 - (embedding <=> $1) AS score
        FROM chunks
        {where}
        ORDER BY embedding <=> $1
        LIMIT ${len(args)}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)

    candidates = [
        PassageRecord(
            ticker=r['ticker'],
            section=r['section'],
            fiscal_year=r['fiscal_year'],
            accession=r['accession'],
            text=r['text'],
            score=float(r['score']),
        )
        for r in rows
    ]
    if not candidates or not rerank:
        return candidates[:limit]
    return await _rerank(query, candidates, limit)


async def _rerank(query: str, candidates: list[PassageRecord], limit: int) -> list[PassageRecord]:
    """Re-order candidates with the cross-encoder, off the event loop."""

    def score() -> list[float]:
        model = _cross_encoder()
        pairs = [(query, c.text) for c in candidates]
        return [float(s) for s in model.predict(pairs)]

    try:
        scores = await asyncio.to_thread(score)
    except Exception:
        # Reranking is an improvement, not a requirement. If the model cannot load,
        # vector order is still a usable answer.
        return candidates[:limit]

    rescored = [
        PassageRecord(
            ticker=c.ticker,
            section=c.section,
            fiscal_year=c.fiscal_year,
            accession=c.accession,
            text=c.text,
            score=s,
        )
        for c, s in zip(candidates, scores, strict=True)
    ]
    rescored.sort(key=lambda p: p.score, reverse=True)
    return rescored[:limit]
