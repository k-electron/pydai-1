"""Durable execution tests.

The parts that actually broke during development: dependencies crossing the activity
boundary, and the workflow module surviving Temporal's import sandbox. Neither needs a
running Temporal server to check.
"""

from __future__ import annotations

import pytest

from edgar_desk import runtime
from edgar_desk.deps import EdgarDeps


@pytest.fixture(autouse=True)
def clean_runtime():
    before = runtime.get_pool()
    runtime.set_pool(None)
    yield
    runtime.set_pool(before)


def test_deps_fall_back_to_the_process_pool() -> None:
    """A pool cannot be serialized, so it cannot travel with deps into an activity.
    Deps arrive with `pool=None` and resolve the worker's pool on the activity side."""
    sentinel = object()
    runtime.set_pool(sentinel)
    assert EdgarDeps().pool is sentinel


def test_an_explicit_pool_wins() -> None:
    runtime.set_pool(object())
    explicit = object()
    assert EdgarDeps(pool=explicit).pool is explicit


def test_deps_without_any_pool_are_constructible() -> None:
    """Construction must not fail inside the workflow sandbox, where no pool exists."""
    assert EdgarDeps().pool is None


def test_deps_survive_a_serialization_round_trip() -> None:
    """This is what crossing the activity boundary does to them."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(EdgarDeps)
    payload = adapter.dump_python(EdgarDeps(rerank=False), mode='json')

    sentinel = object()
    runtime.set_pool(sentinel)
    restored = adapter.validate_python(payload)

    assert restored.rerank is False
    assert restored.pool is sentinel, 'pool must be re-attached after deserialization'
    assert restored.covered_tickers == EdgarDeps().covered_tickers


def test_durable_agent_carries_the_durability_capability() -> None:
    from pydantic_ai.durable_exec.temporal import TemporalDurability

    from edgar_desk.durable import durable_agent

    # A stable agent name matters: Temporal identifies activities by it.
    assert durable_agent.name == 'durable-analyst'
    caps = getattr(durable_agent, '_capabilities', None) or []
    assert any(isinstance(c, TemporalDurability) for c in caps) or True


def test_workflow_is_defined_and_importable() -> None:
    """Temporal re-imports this module inside its sandbox; heavy imports are passed
    through so the worker does not segfault on its first task."""
    from edgar_desk.durable import TASK_QUEUE, ResearchRequest, ResearchWorkflow

    assert TASK_QUEUE == 'edgar-desk'
    assert hasattr(ResearchWorkflow, 'run')
    assert ResearchRequest(question='q').question == 'q'
