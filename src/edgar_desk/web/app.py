"""FastAPI application: a streaming chat endpoint plus the review queue.

`VercelAIAdapter.dispatch_request` does the heavy lifting on `/api/chat`: it parses the
request the Vercel AI SDK sends, runs the agent, and streams Pydantic AI's events back as
the protocol's SSE chunks. The frontend needs no knowledge of Pydantic AI's message types.

Security note, because the adapter's own docs are emphatic about it: this endpoint is not
an authentication boundary. Both AG-UI and the Vercel AI protocol are built around the
client submitting the full conversation history on every request, so anything in
`message_history` -- including prior assistant turns, tool calls, tool results, and tool
*approvals* -- is under the caller's control. Run it behind your own authenticated route
and enforce authorization inside tool functions against the user in `deps`, never against
the client-supplied approval.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from edgar_desk import db, observability, runtime
from edgar_desk.agents.chat import chat_agent
from edgar_desk.deps import EdgarDeps
from edgar_desk.universe import SEED_COMPANIES


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    observability.configure('edgar-desk-web')
    pool = await db.create_pool(min_size=1, max_size=10)
    runtime.set_pool(pool)
    app.state.pool = pool
    try:
        yield
    finally:
        await pool.close()
        runtime.set_pool(None)


app = FastAPI(title='EDGAR Desk', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # The frontend dev server only. Widen this deliberately, not by habit.
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def _deps() -> EdgarDeps:
    # Reranking loads a multi-gigabyte cross-encoder and adds seconds per search, which
    # is the wrong trade for an interactive surface.
    return EdgarDeps(pool=app.state.pool, rerank=False)


@app.get('/', include_in_schema=False)
async def root() -> HTMLResponse:
    """Explain what this server is.

    Uvicorn prints its own address on startup, so this is the first place anyone opens --
    and it serves no pages, only `/api/*`. A bare 404 here reads as "the thing is broken"
    rather than "you want the other port".
    """
    return HTMLResponse(
        """<!doctype html>
<html><head><title>EDGAR Desk API</title>
<style>
 body{font-family:ui-sans-serif,-apple-system,system-ui,sans-serif;background:#0b0d10;
      color:#e6e9ee;max-width:40rem;margin:12vh auto;padding:0 1.5rem;line-height:1.65}
 code{background:#1b2027;padding:.15rem .4rem;border-radius:5px;font-size:.9em}
 pre{background:#14181d;border:1px solid #262d36;border-radius:10px;padding:1rem;
     overflow-x:auto}
 a{color:#5b9dff} h1{font-size:1.3rem;margin-bottom:.3rem}
 .muted{color:#8b95a3}
</style></head><body>
<h1>EDGAR Desk API</h1>
<p class="muted">This is the backend. It serves no pages &mdash; only <code>/api/*</code>.</p>
<p>The chat interface is a separate app on port 3000. Start it with:</p>
<pre>cd web &amp;&amp; npm run dev</pre>
<p>then open <a href="http://localhost:3000">http://localhost:3000</a>.</p>
<p class="muted">Endpoints here: <a href="/api/health">/api/health</a> &middot;
<a href="/api/companies">/api/companies</a> &middot; <code>POST /api/chat</code> &middot;
<a href="/api/review">/api/review</a> &middot; <a href="/docs">/docs</a></p>
</body></html>"""
    )


@app.get('/api/health')
async def health() -> dict[str, Any]:
    async with app.state.pool.acquire() as conn:
        facts = await conn.fetchval('SELECT count(*) FROM xbrl_facts')
        chunks = await conn.fetchval('SELECT count(*) FROM chunks')
    return {'status': 'ok', 'facts': facts, 'chunks': chunks, 'companies': len(SEED_COMPANIES)}


@app.get('/api/companies')
async def companies() -> list[dict[str, str | None]]:
    return [{'ticker': c.ticker, 'name': c.name, 'cik': c.cik} for c in SEED_COMPANIES]


@app.post('/api/chat')
async def chat(request: Request) -> Response:
    """Stream an agent run to a Vercel AI SDK client."""
    return await VercelAIAdapter.dispatch_request(
        request,
        agent=chat_agent,
        deps=_deps(),
        # Must match the `ai` package the frontend uses; the adapter defaults to 5 and
        # the chunk shapes differ between major versions.
        sdk_version=7,
    )


class ReviewDecision(BaseModel):
    status: str
    note: str | None = None


@app.get('/api/review')
async def list_review(status: str = 'approved', limit: int = 20) -> list[dict[str, Any]]:
    """Briefs that have been published, most recent first."""
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, question, brief, status, created_at, decided_at, decided_by, note
            FROM review_queue
            WHERE ($1 = 'all' OR status = $1)
            ORDER BY created_at DESC
            LIMIT $2
            """,
            status,
            limit,
        )
    return [
        {
            **dict(row),
            'brief': json.loads(row['brief']) if isinstance(row['brief'], str) else row['brief'],
            'created_at': row['created_at'].isoformat(),
            'decided_at': row['decided_at'].isoformat() if row['decided_at'] else None,
        }
        for row in rows
    ]


@app.post('/api/review/{review_id}')
async def decide(review_id: int, decision: ReviewDecision) -> dict[str, Any]:
    if decision.status not in ('approved', 'rejected'):
        return {'error': "status must be 'approved' or 'rejected'"}
    async with app.state.pool.acquire() as conn:
        updated = await conn.fetchval(
            """
            UPDATE review_queue
            SET status = $2, note = $3, decided_at = now(), decided_by = 'reviewer'
            WHERE id = $1
            RETURNING id
            """,
            review_id,
            decision.status,
            decision.note,
        )
    return {'id': updated, 'status': decision.status} if updated else {'error': 'not found'}
