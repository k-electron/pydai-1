"""Durable research runs on Temporal.

A full research run takes minutes and makes dozens of model and database calls. Without
durability, a crash, a restart, or an Ollama hiccup two minutes in loses everything.
`TemporalDurability` routes model requests and tool calls through Temporal activities, so
each completed step is recorded and a resumed run picks up where it stopped instead of
starting over.

Run the worker in one terminal:

    uv run edgar-desk worker

Then submit work from another:

    uv run edgar-desk durable "Compare NVDA and AMD R&D intensity"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

# Temporal re-imports the workflow's module inside a sandbox to enforce deterministic
# replay. Re-importing this dependency tree there is not merely slow -- with native
# extensions in it (torch by way of the reranker, pydantic-core, asyncpg) the worker
# segfaults the moment it picks up its first task. `imports_passed_through()` binds these
# modules to the already-imported originals instead of reloading them. Determinism is
# unaffected: the workflow body only awaits the agent, and the non-deterministic work
# already happens inside activities.
with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalDurability,
    )

    from edgar_desk.capabilities import financials_capability, narrative_capability
    from edgar_desk.deps import EdgarDeps
    from edgar_desk.models import DEFAULT_MODEL_SETTINGS, primary_model
    from edgar_desk.schemas import Brief
    from edgar_desk.settings import get_settings

TASK_QUEUE = 'edgar-desk'

INSTRUCTIONS = """\
You are a financial research analyst working only from SEC filings.

Every finding needs a claim, at least one citation, and a confidence between 0 and 1.
Never state a figure you did not retrieve with a tool. When you compute a ratio, cite
both inputs. Note differing fiscal year ends when comparing companies.
"""

# The durable agent is defined at module scope with a stable `name`, because Temporal
# identifies activities by that name. Building it per-call would register activities the
# worker does not recognise.
durable_agent = Agent(
    primary_model(),
    deps_type=EdgarDeps,
    output_type=Brief,
    instructions=INSTRUCTIONS,
    name='durable-analyst',
    retries=2,
    model_settings=DEFAULT_MODEL_SETTINGS,
    capabilities=[
        financials_capability,
        narrative_capability,
        TemporalDurability(),
    ],
)


@dataclass
class ResearchRequest:
    question: str


@workflow.defn
class ResearchWorkflow:
    """One research question, durably.

    The workflow body itself has to be deterministic: Temporal replays it after a
    failure, and anything non-deterministic in here would diverge on replay. The model
    calls and database queries are the non-deterministic parts, and the durability
    capability has already moved them into activities.
    """

    @workflow.run
    async def run(self, request: ResearchRequest) -> Brief:
        # Deps constructed with no pool: they are serialized to cross into activities,
        # and a connection pool cannot be. `EdgarDeps` resolves the worker's process-wide
        # pool on the activity side, where it is in scope.
        result = await durable_agent.run(request.question, deps=EdgarDeps(rerank=False))
        return result.output


def configure_worker_observability() -> None:
    """Set up tracing before the event loop exists.

    This must run *outside* `asyncio.run`. Temporal's core is a Rust extension whose
    pyo3 task locals may only be bound once per process, and configuring OpenTelemetry
    from inside the running loop binds them a second time -- which aborts the worker with
    a Rust panic ("must only be set once: TaskLocals") rather than a Python exception.
    """
    from edgar_desk import observability

    observability.configure('edgar-desk-worker')


async def run_worker() -> None:
    """Start a Temporal worker that can execute research workflows.

    Call `configure_worker_observability()` first if you want traces.
    """
    from edgar_desk import db, runtime

    settings = get_settings()

    # Registered process-wide rather than passed through the workflow, since activities
    # run in this process and can reach it, while the pool itself cannot be serialized.
    runtime.set_pool(await db.create_pool(min_size=1, max_size=10))

    client = await Client.connect(settings.temporal_address, plugins=[PydanticAIPlugin()])
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ResearchWorkflow],
        plugins=[AgentPlugin(durable_agent)],
    )
    # `await worker.run()` only -- entering the worker as an async context manager also
    # starts it, and doing both binds the Rust core's task locals twice, which aborts the
    # process with a panic rather than raising.
    await worker.run()


async def submit(question: str, *, workflow_id: str | None = None) -> Brief:
    """Submit a question and wait for the durable run to finish."""
    settings = get_settings()
    client = await Client.connect(settings.temporal_address, plugins=[PydanticAIPlugin()])
    return await client.execute_workflow(
        ResearchWorkflow.run,
        ResearchRequest(question=question),
        id=workflow_id or f'research-{abs(hash(question)) % 10**10}',
        task_queue=TASK_QUEUE,
        # Generous: a multi-company question on a local model can genuinely take minutes,
        # and a timeout here would look like a failure rather than slow inference.
        execution_timeout=timedelta(minutes=30),
    )
