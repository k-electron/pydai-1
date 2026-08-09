# EDGAR Desk

A research agent over SEC EDGAR filings, built to learn the Pydantic AI v2 ecosystem.
Runs entirely on local models: no cloud API keys anywhere.

It answers questions like:

> How has NVIDIA's R&D spend as a share of revenue changed over five years, and what do
> their latest risk factors say about competition?

That question needs two different kinds of evidence, which is the point. The numeric half
is answered from XBRL facts (exact, checkable), the narrative half from filing prose
(retrieved and reranked). Because XBRL gives verifiable ground truth, the eval suite can
grade answers against real numbers instead of relying only on an LLM judge.

## Requirements

- [Ollama](https://ollama.com) running locally
- [uv](https://docs.astral.sh/uv/)
- Docker (from Phase 1 on)

```bash
ollama pull qwen3.6:35b-a3b   # orchestrator and sub-agents
ollama pull gpt-oss:20b       # independent eval judge
ollama pull bge-m3:567m       # embeddings

uv sync
uv run edgar-desk doctor
```

## Getting started

```bash
docker compose up -d          # Postgres+pgvector, Temporal, Jaeger
uv run edgar-desk db-init     # apply schema, seed the 20 companies
uv run edgar-desk ingest      # ~4.5 min: 13k facts, 4.4k embedded passages
uv run edgar-desk doctor      # everything green?
```

Then ask it something:

```bash
uv run edgar-desk analyze "How has NVIDIA's R&D spend as a share of revenue changed \
  over five years, and what do their risk factors say about competition?"
```

## Commands

```bash
edgar-desk doctor                  # check models, database, tracing, Temporal
edgar-desk ingest [-t NVDA]        # load facts and filing text from the SEC
edgar-desk ask "..."               # triage only: decompose a question
edgar-desk analyze "..." [--full]  # answer with a cited brief; --full adds the harness
edgar-desk pipeline "..."          # answer through the explicit graph pipeline
edgar-desk pipeline-diagram        # Mermaid diagram, generated from the graph
edgar-desk bakeoff                 # compare local models on tool use and typed output
edgar-desk eval [--full]           # grade answers against figures reported to the SEC
edgar-desk mcp                     # serve the corpus over MCP (stdio, for editors)
edgar-desk worker                  # Temporal worker for durable runs
edgar-desk durable "..."           # run a question as a durable workflow
edgar-desk serve                   # streaming chat API for the web UI
```

## Web UI

```bash
uv run edgar-desk serve            # backend on :8000
cd web && npm install && npm run dev   # frontend on :3000
```

Chat streams token by token, tool calls appear as they run, and asking it to publish a
brief pauses for an explicit approval before anything is written.

## Interfaces

- **Local traces**: Jaeger at http://localhost:16686 (service `edgar-desk-*`)
- **Workflows**: Temporal UI at http://localhost:8233
- **MCP from Cursor**: already wired in [.cursor/mcp.json](.cursor/mcp.json)

## Testing

```bash
uv run pytest        # 114 tests; database-backed ones skip when Docker is down
uv run ruff check src tests
```

No test requires a model server: agents are driven by `TestModel` and `FunctionModel`.

## Configuration

Every setting has a local-first default; override via environment variables.

| Variable | Default |
| --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` |
| `EDGAR_PRIMARY_MODEL` | `qwen3.6:35b-a3b` |
| `EDGAR_JUDGE_MODEL` | `gpt-oss:20b` |
| `EDGAR_EMBEDDING_MODEL` | `bge-m3:567m` |
| `EDGAR_DATABASE_URL` | `postgresql://edgar:edgar@localhost:5433/edgar` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` |
| `EDGAR_SEC_USER_AGENT` | set this to your own contact info |

The SEC requires a descriptive `User-Agent` with real contact information on every
request and will reject traffic without one. Set `EDGAR_SEC_USER_AGENT` before ingesting.

## Layout

```
src/edgar_desk/
  settings.py       configuration
  models.py         model + embedder factories, shared model settings
  observability.py  local OpenTelemetry tracing
  schemas.py        the typed domain model everything agrees on
  universe.py       the 20 covered companies, with real CIKs
  deps.py           the dependency container agents share
  runtime.py        process-wide resources that cannot be serialized
  edgar/            SEC client, XBRL normalization, filing text extraction
  db/               schema and connection pool
  retrieval/        SQL over facts, vector search over prose, SQL guard
  capabilities/     financials, narrative, publishing, citation enforcement
  agents/           triage, analyst, researcher, chat, connected
  pipeline.py       the same work as an explicit pydantic-graph graph
  durable.py        Temporal workflow and worker
  mcp_server.py     the corpus, served over MCP
  web/              FastAPI app: streaming chat + review queue
  evals/            model bake-off and ground-truth eval suites
web/                Next.js frontend
```

## What this exercises

Core loop, dependency injection, typed output, streaming; the v2 capability system and
custom `AbstractCapability` hooks; harness capabilities (planning, sub-agents, memory,
compaction, tool output limits, system reminders); `pydantic-graph` with map/join/decision;
`pydantic-evals` with exact, span-based and LLM-judge evaluators; MCP as both client and
server; durable execution on Temporal; deferred tools for human approval; embeddings and
two-stage retrieval; OpenTelemetry throughout.

See [NOTES.md](NOTES.md) for what each phase established — including the measurements
that changed the design, and the mistakes worth not repeating.
