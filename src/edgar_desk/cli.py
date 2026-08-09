"""Command line entrypoint for EDGAR Desk."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from edgar_desk import observability
from edgar_desk.settings import get_settings

app = typer.Typer(help='EDGAR Desk: a local-only SEC filings research agent.', no_args_is_help=True)
console = Console()


def _print_brief(brief, question: str) -> None:
    console.print(f'\n[bold]{brief.question or question}[/bold]\n')
    console.print(brief.summary)
    console.print('\n[bold]Findings[/bold]')
    for finding in brief.findings:
        console.print(f'\n  [cyan]{finding.confidence:.2f}[/cyan]  {finding.claim}')
        for citation in finding.citations:
            period = f' {citation.fiscal_period}' if citation.fiscal_period else ''
            console.print(f'         [dim]{citation.ticker}{period} — {citation.source}[/dim]')
    if brief.caveats:
        console.print('\n[bold]Caveats[/bold]')
        for caveat in brief.caveats:
            console.print(f'  - {caveat}')
    console.print()


@app.command()
def bakeoff(
    models: list[str] = typer.Option(  # noqa: B008
        None, '--model', '-m', help='Ollama model to evaluate. Repeatable.'
    ),
    repeat: int = typer.Option(1, help='Runs per case; >1 measures run-to-run reliability.'),
    trace: bool = typer.Option(False, help='Export traces to the local OTel collector.'),
) -> None:
    """Compare local models on tool calling and structured output."""
    from edgar_desk.evals.bakeoff import run_for_model

    if trace:
        observability.configure('edgar-desk-bakeoff')

    settings = get_settings()
    chosen = models or [settings.primary_model, settings.judge_model]

    async def main() -> None:
        for name in chosen:
            console.rule(f'[bold]{name}')
            try:
                await run_for_model(name, repeat=repeat)
            except Exception as exc:  # noqa: BLE001
                console.print(f'[red]{name} failed: {type(exc).__name__}: {exc}')

    asyncio.run(main())


@app.command()
def ask(
    question: str = typer.Argument(..., help='A question about the covered companies.'),
    trace: bool = typer.Option(True, help='Export traces to the local OTel collector.'),
) -> None:
    """Run the triage agent against a question and print the plan."""
    from edgar_desk.agents.triage import TriageDeps, triage_agent

    if trace:
        observability.configure('edgar-desk-cli')

    async def main() -> None:
        deps = TriageDeps()
        result = await triage_agent.run(question, deps=deps)
        plan = result.output
        console.print(f'[dim]{plan.reasoning}[/dim]\n')
        console.print('[bold]Companies[/bold]')
        for c in plan.companies:
            console.print(f'  {c.ticker:<6} {c.name}  [dim]CIK {c.cik}[/dim]')
        console.print('\n[bold]Sub-questions[/bold]')
        for i, q in enumerate(plan.sub_questions, 1):
            console.print(f'  {i}. [cyan]{q.angle.value:<9}[/cyan] {q.question}')
        console.print(f'\n[dim]tokens: {result.usage}[/dim]')

    asyncio.run(main())


@app.command('db-init')
def db_init() -> None:
    """Apply the schema and seed the covered universe."""
    from edgar_desk import db

    async def main() -> None:
        async with db.pool_context() as pool:
            await db.apply_schema(pool)
            count = await db.seed_companies(pool)
            console.print(f'[green]schema applied[/green], {count} companies seeded')

    asyncio.run(main())


@app.command()
def analyze(
    question: str = typer.Argument(..., help='A question about the covered companies.'),
    full: bool = typer.Option(
        False,
        '--full',
        help='Use the full harness: planning, sub-agents, memory, compaction, citation '
        'enforcement. Slower, better on multi-company questions.',
    ),
    rerank: bool = typer.Option(True, help='Rerank passages with the local cross-encoder.'),
    trace: bool = typer.Option(True, help='Export traces to the local OTel collector.'),
) -> None:
    """Answer a question from filings and print the cited brief."""
    from edgar_desk.deps import build_deps

    if trace:
        observability.configure('edgar-desk-analyst')

    if full:
        from edgar_desk.agents.researcher import build_researcher

        agent = build_researcher()
    else:
        from edgar_desk.agents.analyst import analyst_agent

        agent = analyst_agent

    async def main() -> None:
        async with build_deps(rerank=rerank) as deps:
            with console.status('[bold]researching'):
                result = await agent.run(question, deps=deps)
            brief = result.output

        _print_brief(brief, question)
        console.print(f'[dim]tools: {", ".join(deps.tool_calls) or "none"}[/dim]')
        console.print(f'[dim]usage: {result.usage}[/dim]')

    asyncio.run(main())


@app.command()
def pipeline(
    question: str = typer.Argument(..., help='A question about the covered companies.'),
    rerank: bool = typer.Option(True, help='Rerank passages with the local cross-encoder.'),
    trace: bool = typer.Option(True, help='Export traces to the local OTel collector.'),
) -> None:
    """Answer a question through the explicit graph pipeline."""
    from edgar_desk.deps import build_deps
    from edgar_desk.pipeline import run_pipeline

    if trace:
        observability.configure('edgar-desk-pipeline')

    async def main() -> None:
        async with build_deps(rerank=rerank) as deps:
            with console.status('[bold]running pipeline'):
                brief, state = await run_pipeline(question, deps)

        plan = state.plan
        if plan:
            console.print('\n[bold]Plan[/bold]')
            for i, q in enumerate(plan.sub_questions, 1):
                console.print(f'  {i}. [cyan]{q.angle.value:<9}[/cyan] {q.question}')
        console.print(f'\n[dim]rounds: {state.rounds}[/dim]')
        _print_brief(brief, question)

    asyncio.run(main())


@app.command('pipeline-diagram')
def pipeline_diagram() -> None:
    """Print a Mermaid diagram of the pipeline, generated from the graph."""
    from edgar_desk.pipeline import render_mermaid

    print(render_mermaid())


@app.command()
def ingest(
    tickers: list[str] = typer.Option(  # noqa: B008
        None, '--ticker', '-t', help='Restrict to these tickers. Default: the whole universe.'
    ),
    facts: bool = typer.Option(True, help='Load XBRL numeric facts.'),
    narrative: bool = typer.Option(True, help='Load and embed 10-K narrative sections.'),
    filings: int = typer.Option(2, help='Recent 10-Ks per company.'),
) -> None:
    """Ingest SEC data for the covered universe."""
    from edgar_desk import db
    from edgar_desk.ingest import ingest_all, refresh_vector_index
    from edgar_desk.universe import BY_TICKER, SEED_COMPANIES

    chosen = (
        tuple(BY_TICKER[t.upper()] for t in tickers if t.upper() in BY_TICKER)
        if tickers
        else SEED_COMPANIES
    )
    if not chosen:
        console.print('[red]no known tickers selected[/red]')
        raise typer.Exit(1)

    async def main() -> None:
        async with db.pool_context() as pool:
            await db.apply_schema(pool)
            await db.seed_companies(pool)
            with console.status('[bold]ingesting') as status:

                def on_progress(company, stats) -> None:
                    status.update(
                        f'[bold]{company.ticker}[/bold]  '
                        f'{stats.companies + 1}/{len(chosen)}  '
                        f'facts={stats.facts} chunks={stats.chunks}'
                    )

                stats = await ingest_all(
                    pool,
                    chosen,
                    facts=facts,
                    narrative=narrative,
                    filings_per_company=filings,
                    progress=on_progress,
                )
            if narrative and stats.chunks:
                await refresh_vector_index(pool)

        console.print(
            f'[green]done[/green] companies={stats.companies} facts={stats.facts} '
            f'filings={stats.filings} chunks={stats.chunks}'
        )
        for err in stats.errors or []:
            console.print(f'  [red]{err}[/red]')

    asyncio.run(main())


@app.command()
def serve(
    host: str = typer.Option('127.0.0.1', help='Bind address.'),
    port: int = typer.Option(8000, help='Port.'),
    reload: bool = typer.Option(False, help='Reload on code changes.'),
) -> None:
    """Serve the streaming chat API for the web UI."""
    import uvicorn

    uvicorn.run('edgar_desk.web.app:app', host=host, port=port, reload=reload)


@app.command()
def worker(
    trace: bool = typer.Option(True, help='Export traces to the local OTel collector.'),
) -> None:
    """Run a Temporal worker that executes durable research workflows."""
    from edgar_desk.durable import configure_worker_observability, run_worker

    # Tracing must be configured before the event loop starts; see the function's
    # docstring for why doing it inside `asyncio.run` kills the worker.
    if trace:
        configure_worker_observability()

    console.print('[bold]worker starting[/bold] — submit work with `edgar-desk durable "..."`')
    asyncio.run(run_worker())


@app.command()
def durable(
    question: str = typer.Argument(..., help='A question about the covered companies.'),
) -> None:
    """Run a question as a durable Temporal workflow. Requires `edgar-desk worker`."""
    from edgar_desk.durable import submit

    observability.configure('edgar-desk-durable')

    async def main() -> None:
        with console.status('[bold]running durable workflow'):
            brief = await submit(question)
        _print_brief(brief, question)

    asyncio.run(main())


@app.command('eval')
def run_eval(
    full: bool = typer.Option(False, '--full', help='Evaluate the full harness researcher.'),
    repeat: int = typer.Option(1, help='Runs per case; >1 measures run-to-run reliability.'),
    rerank: bool = typer.Option(False, help='Rerank passages with the local cross-encoder.'),
) -> None:
    """Grade the analyst against figures reported to the SEC."""
    from edgar_desk.deps import build_deps
    from edgar_desk.evals.analyst import run

    async def main() -> None:
        async with build_deps(rerank=rerank) as deps:
            await run(deps, full=full, repeat=repeat)

    asyncio.run(main())


@app.command()
def mcp(
    transport: str = typer.Option('stdio', help='"stdio" for editors, or "http".'),
    port: int = typer.Option(8931, help='Port, when transport is http.'),
) -> None:
    """Serve the EDGAR corpus over the Model Context Protocol."""
    from edgar_desk.mcp_server import run

    run(transport=transport, port=port)


@app.command()
def doctor() -> None:
    """Check that the local environment is ready."""
    import httpx

    settings = get_settings()
    console.print('[bold]EDGAR Desk environment[/bold]\n')

    base = settings.ollama_base_url.removesuffix('/v1')
    try:
        tags = httpx.get(f'{base}/api/tags', timeout=5).json()
        installed = {m['name'] for m in tags.get('models', [])}
        console.print(f'[green]ok[/green]   ollama at {base} ({len(installed)} models)')
        for label, name in (
            ('primary', settings.primary_model),
            ('judge', settings.judge_model),
            ('embedding', settings.embedding_model),
        ):
            mark = '[green]ok[/green]  ' if name in installed else '[red]miss[/red]'
            console.print(f'{mark} {label}: {name}')
    except Exception as exc:  # noqa: BLE001
        console.print(f'[red]fail[/red] ollama at {base}: {exc}')

    async def check_db() -> str:
        from edgar_desk import db

        try:
            async with db.pool_context(max_size=1) as pool, pool.acquire() as conn:
                tables = await conn.fetchval(
                    "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
                )
                companies = await conn.fetchval('SELECT count(*) FROM companies') if tables else 0
            return f'[green]ok[/green]   postgres ({tables} tables, {companies} companies)'
        except Exception as exc:  # noqa: BLE001
            return f'[red]fail[/red] postgres: {type(exc).__name__}: {str(exc)[:60]}'

    console.print(asyncio.run(check_db()))

    exporting = observability.collector_reachable(settings.otlp_endpoint)
    mark = '[green]ok[/green]  ' if exporting else '[yellow]down[/yellow]'
    console.print(f'{mark} otel collector at {settings.otlp_endpoint}')

    host, _, port = settings.temporal_address.partition(':')
    reachable = observability.collector_reachable(f'http://{host}:{port}')
    mark = '[green]ok[/green]  ' if reachable else '[yellow]down[/yellow]'
    console.print(f'{mark} temporal at {settings.temporal_address}')


if __name__ == '__main__':
    app()
