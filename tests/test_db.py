"""Database tests.

These need the Docker stack up, so they skip cleanly when it is not, keeping
`uv run pytest` useful on a laptop with nothing running.
"""

from __future__ import annotations

import pytest

from edgar_desk import db
from edgar_desk.observability import collector_reachable
from edgar_desk.settings import get_settings
from edgar_desk.universe import SEED_COMPANIES


def _postgres_up() -> bool:
    url = get_settings().database_url
    hostport = url.rsplit('@', 1)[-1].split('/')[0]
    host, _, port = hostport.partition(':')
    return collector_reachable(f'http://{host}:{port or 5432}')


pytestmark = pytest.mark.skipif(_postgres_up() is False, reason='postgres not running')


@pytest.fixture
async def pool():
    async with db.pool_context(max_size=2) as p:
        await db.apply_schema(p)
        yield p


async def test_schema_is_idempotent(pool) -> None:
    """Re-applying must be safe: it is the whole migration story for this project."""
    await db.apply_schema(pool)
    await db.apply_schema(pool)


async def test_setup_works_on_a_brand_new_database(monkeypatch) -> None:
    """The first run on an empty database must succeed.

    This is the case every other test misses, because they all run against a database
    that has been set up already. `create_pool` registers the pgvector codec on each
    connection, which needs the `vector` type to exist -- but the extension is created by
    `schema.sql`, which needs a pool. A fresh database failed with
    `unknown type: public.vector` before the schema ever ran.
    """
    import asyncpg

    from edgar_desk.settings import get_settings

    base_url = get_settings().database_url
    scratch = 'edgar_test_coldstart'
    admin = await asyncpg.connect(base_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {scratch}')
        await admin.execute(f'CREATE DATABASE {scratch}')
    finally:
        await admin.close()

    # Redirect via the environment rather than by patching `get_settings`: `db` imported
    # that name directly, so patching the attribute on `settings` leaves `db`'s bound
    # reference pointing at the original and the test silently hits the real database.
    cold_url = base_url.rsplit('/', 1)[0] + f'/{scratch}'
    monkeypatch.setenv('EDGAR_DATABASE_URL', cold_url)
    get_settings.cache_clear()
    assert get_settings().database_url == cold_url, 'test is not pointing at the cold database'

    try:
        async with db.pool_context(max_size=2) as cold_pool:
            await db.apply_schema(cold_pool)
            assert await db.seed_companies(cold_pool) == len(SEED_COMPANIES)
            async with cold_pool.acquire() as conn:
                vector = await conn.fetchval('SELECT $1::vector', [0.5] * 1024)
            assert len(vector.to_list()) == 1024
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
        admin = await asyncpg.connect(base_url)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS {scratch}')
        finally:
            await admin.close()


async def test_seed_companies_upserts(pool) -> None:
    first = await db.seed_companies(pool)
    second = await db.seed_companies(pool)
    assert first == second == len(SEED_COMPANIES)


async def test_vector_column_roundtrips(pool) -> None:
    """pgvector values must decode as numbers, not strings, or every retrieval path
    downstream silently degrades.

    Note the codec returns a `pgvector.Vector`, not a list: it needs `.to_list()` or
    `.to_numpy()` before anything treats it as a sequence.
    """
    async with pool.acquire() as conn:
        got = await conn.fetchval('SELECT $1::vector', [0.1] * 1024)
    values = got.to_list()
    assert len(values) == 1024
    assert abs(values[0] - 0.1) < 1e-6


async def test_review_queue_rejects_bad_status(pool) -> None:
    import asyncpg

    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.PostgresError):
            await conn.execute(
                "INSERT INTO review_queue (question, brief, status) VALUES ('q', '{}', 'bogus')"
            )
