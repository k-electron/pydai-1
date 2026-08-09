"""Database access.

One asyncpg pool per process, registered with pgvector so `vector` columns round-trip
as Python lists instead of strings.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from edgar_desk.settings import get_settings

SCHEMA_PATH = Path(__file__).parent / 'schema.sql'


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def ensure_extension() -> None:
    """Create the `vector` extension before anything tries to use its type.

    Without this, a brand-new database cannot be initialized at all. `create_pool`
    registers the pgvector codec on every connection it opens, and registering requires
    the `vector` type to already exist -- but the extension is created by `schema.sql`,
    which needs a pool to run. Pool creation therefore fails with
    `ValueError: unknown type: public.vector` before the schema ever executes.

    Bootstrapping it here on a plain connection, with no codec registration, breaks the
    cycle. It stayed hidden during development because the extension had been created by
    hand while checking the container, so every later run found it already present.
    """
    conn = await asyncpg.connect(get_settings().database_url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await conn.close()


async def create_pool(min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    """Open a connection pool against the configured database."""
    await ensure_extension()
    return await asyncpg.create_pool(
        get_settings().database_url,
        min_size=min_size,
        max_size=max_size,
        init=_init_connection,
    )


@contextlib.asynccontextmanager
async def pool_context(**kwargs: int) -> AsyncIterator[asyncpg.Pool]:
    pool = await create_pool(**kwargs)
    try:
        yield pool
    finally:
        await pool.close()


async def apply_schema(pool: asyncpg.Pool) -> None:
    """Create every table and index if missing.

    The schema is written to be idempotent, so this doubles as the migration story for a
    project of this size: re-running it is always safe.
    """
    sql = SCHEMA_PATH.read_text()
    async with pool.acquire() as conn:
        # pgvector types must exist before `register_vector` can bind them, and the
        # extension is created inside this script, so a fresh database needs the codec
        # refreshed afterwards.
        await conn.execute(sql)
        await register_vector(conn)


async def seed_companies(pool: asyncpg.Pool) -> int:
    """Insert the covered universe. Returns the number of rows now present."""
    from edgar_desk.universe import SEED_COMPANIES

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO companies (cik, ticker, name)
            VALUES ($1, $2, $3)
            ON CONFLICT (cik) DO UPDATE SET ticker = EXCLUDED.ticker, name = EXCLUDED.name
            """,
            [(c.cik, c.ticker, c.name) for c in SEED_COMPANIES],
        )
        return await conn.fetchval('SELECT count(*) FROM companies')
