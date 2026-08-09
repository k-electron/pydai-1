"""The dependency container every agent shares.

Pydantic AI's dependency injection is typed: an agent declares `deps_type=EdgarDeps` and
tools receive it through `RunContext[EdgarDeps]`, so a tool reaching for something the
container does not have is a type error rather than a runtime surprise.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from edgar_desk.models import embedder
from edgar_desk.schemas import CompanyRef
from edgar_desk.universe import SEED_COMPANIES


@dataclass
class EdgarDeps:
    """Everything a tool might need, resolved once per run.

    `pool` is optional and falls back to the process-wide pool. That matters for durable
    execution: dependencies are serialized to cross a Temporal activity boundary, and a
    connection pool cannot be. Under Temporal this arrives as `None` and is resolved
    on the activity side, where the worker's pool is in scope.
    """

    pool: Any = None
    covered: tuple[CompanyRef, ...] = SEED_COMPANIES

    max_rows: int = 200
    """Hard cap on rows any single tool returns, so one broad query cannot fill the
    context window with numbers the model will not read."""

    rerank: bool = True
    """Whether narrative search runs the cross-encoder. Disabled in tests, where loading
    a 2GB model to check SQL wiring is not a good trade."""

    tool_calls: list[str] = field(default_factory=list)
    """Names of tools invoked this run, for assertions in tests and evals."""

    _embedder: Any = None

    def __post_init__(self) -> None:
        # Runs on construction and again when pydantic rebuilds the dataclass on the
        # activity side, which is where the real pool gets attached under Temporal.
        if self.pool is None:
            from edgar_desk import runtime

            self.pool = runtime.get_pool()

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = embedder()
        return self._embedder

    def record(self, tool_name: str) -> None:
        self.tool_calls.append(tool_name)

    @property
    def covered_tickers(self) -> frozenset[str]:
        return frozenset(c.ticker for c in self.covered)


@contextlib.asynccontextmanager
async def build_deps(**overrides: Any) -> AsyncIterator[EdgarDeps]:
    """Open a pool and yield a ready dependency container."""
    from edgar_desk import db

    async with db.pool_context() as pool:
        yield EdgarDeps(pool=pool, **overrides)
