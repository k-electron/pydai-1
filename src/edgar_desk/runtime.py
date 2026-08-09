"""Process-wide resources that cannot be passed by value.

A connection pool cannot be serialized, so it cannot travel with an agent's dependencies
across a Temporal activity boundary. Activities run in the worker's own process (only
*workflow* code runs in the replay sandbox), so a process-global set once at worker
startup is visible to every tool that runs in an activity.
"""

from __future__ import annotations

from typing import Any

_pool: Any = None


def set_pool(pool: Any) -> None:
    """Register the process-wide connection pool. Called once by the worker."""
    global _pool
    _pool = pool


def get_pool() -> Any:
    """The process-wide pool, or None if nothing registered one."""
    return _pool
