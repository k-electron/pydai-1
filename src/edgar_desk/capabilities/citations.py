"""A custom capability that refuses uncited work.

Instructions asking for citations are a request. This is enforcement: it inspects the
validated `Brief` and, when a finding cites nothing or cites a company the run never
looked up, raises `ModelRetry` so the model gets a specific correction and another
attempt rather than the caller getting a confident, unsupported answer.

This is the shape to copy when writing your own capability. `Capability` covers bundling
instructions, tools and toolsets; anything that needs a lifecycle hook -- as here --
subclasses `AbstractCapability`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, OutputContext

from edgar_desk.deps import EdgarDeps
from edgar_desk.schemas import Brief

INSTRUCTIONS = """\
Every finding must carry at least one citation, and every citation must name a company
you actually retrieved data for. A claim you cannot cite belongs in `caveats`, not in
`findings`.
"""


@dataclass(init=False)
class RequireCitations(AbstractCapability[EdgarDeps]):
    """Reject briefs whose findings are not backed by retrieved evidence."""

    min_confidence_for_uncited: float
    max_retries: int

    def __init__(self, *, max_retries: int = 2) -> None:
        super().__init__()
        self.min_confidence_for_uncited = 0.0
        self.max_retries = max_retries
        self._rejections = 0

    def get_instructions(self) -> str:
        # Static text, so no RunContext here. A capability needing per-request
        # instructions returns a callable rather than taking ctx as a parameter.
        return INSTRUCTIONS

    async def after_output_validate(
        self,
        ctx: RunContext[EdgarDeps],
        *,
        output_context: OutputContext,
        output: object,
    ) -> object:
        if not isinstance(output, Brief):
            return output

        problems = self._problems(ctx, output)
        if not problems:
            return output

        self._rejections += 1
        if self._rejections > self.max_retries:
            # Give up rather than loop: an endless retry is worse than a flagged answer.
            output.caveats.append(
                'Some findings could not be verified against retrieved evidence: '
                + '; '.join(problems)
            )
            return output

        raise ModelRetry(
            'This brief cannot be returned yet:\n'
            + '\n'.join(f'- {p}' for p in problems)
            + '\nFix each finding by citing evidence you retrieved, or move the claim '
            'into `caveats`.'
        )

    def _problems(self, ctx: RunContext[EdgarDeps], brief: Brief) -> list[str]:
        problems: list[str] = []
        covered = ctx.deps.covered_tickers

        if not brief.findings:
            problems.append('the brief contains no findings')

        for index, finding in enumerate(brief.findings, 1):
            label = f'finding {index} ("{finding.claim[:60]}...")'
            if not finding.citations:
                problems.append(f'{label} has no citations')
                continue
            for citation in finding.citations:
                ticker = citation.ticker.upper()
                if ticker not in covered:
                    problems.append(f'{label} cites {ticker}, which is not a covered company')
                if not citation.source.strip():
                    problems.append(f'{label} has a citation with an empty source')

        if not brief.summary.strip():
            problems.append('the summary is empty')

        return problems
