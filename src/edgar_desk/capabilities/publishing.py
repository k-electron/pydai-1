"""Publishing, behind a human approval gate.

`requires_approval=True` makes the agent pause instead of calling the tool: the run comes
back with a `DeferredToolRequests` describing what it wants to do, and only resumes when
the caller supplies a matching `DeferredToolResults`.

Worth being precise about what this protects. Approval stops the *model* from acting
without a human saying yes. It is not an authorization boundary against the caller: on
the HTTP path, approvals arrive from the client alongside the message history, and a
client that can reach the endpoint can approve a call it invented. Authorization for
anything sensitive belongs inside the tool function, checked against the authenticated
user in `deps`.
"""

from __future__ import annotations

import json

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability

from edgar_desk.deps import EdgarDeps
from edgar_desk.schemas import Brief

INSTRUCTIONS = """\
When the user asks you to publish, save, or export a brief, call `publish_brief`.

Publishing needs a person's approval, so the call pauses for review rather than taking
effect immediately. Do not describe the brief as published until the tool returns.
"""

publishing_capability: Capability[EdgarDeps] = Capability(
    id='publishing',
    description='Save a finished brief to the review queue, subject to human approval.',
    instructions=INSTRUCTIONS,
)


@publishing_capability.tool(requires_approval=True)
async def publish_brief(ctx: RunContext[EdgarDeps], brief: Brief) -> str:
    """Save a finished brief to the review queue. Requires human approval.

    Args:
        brief: The completed brief, with every finding cited.
    """
    ctx.deps.record('publish_brief')

    uncited = [f.claim for f in brief.findings if not f.citations]
    if uncited:
        # Checked here rather than trusted from upstream: this function runs after
        # approval, and the approver is agreeing to publish, not re-verifying citations.
        return f'Refused: {len(uncited)} finding(s) have no citation. First: {uncited[0][:80]}'

    async with ctx.deps.pool.acquire() as conn:
        row_id = await conn.fetchval(
            """
            INSERT INTO review_queue (question, brief, status, decided_at, decided_by)
            VALUES ($1, $2, 'approved', now(), 'human-approval')
            RETURNING id
            """,
            brief.question,
            json.dumps(brief.model_dump(mode='json')),
        )
    return f'Published brief #{row_id} with {len(brief.findings)} findings.'
