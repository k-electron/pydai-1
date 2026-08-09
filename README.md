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

---

## Before you start

**What you need installed**

- [Ollama](https://ollama.com), running
- [uv](https://docs.astral.sh/uv/)
- Docker, running

**What it will use.** The three models are about **37 GB** of download and roughly 40 GB
of disk. Inference is memory-bound: 32 GB of RAM is a realistic floor for the default
`qwen3.6:35b-a3b`, and the smaller `gpt-oss:20b` is a reasonable substitute on less (see
[Using different models](#using-different-models)). Ingestion pulls about 300 MB from the
SEC and writes roughly 500 MB to Postgres.

**Ports the stack claims**, all bound to `127.0.0.1` only. They are offset from the usual
defaults to avoid colliding with an existing Postgres or collector; if any are taken, edit
[docker-compose.yml](docker-compose.yml) and set the matching environment variable below.

| Port | Service |
| --- | --- |
| 5442 | Postgres + pgvector |
| 4317 / 4318 | OTLP trace ingest (Jaeger) |
| 16686 | Jaeger UI |
| 7233 | Temporal |
| 8233 | Temporal UI |
| 8000 | This app's API (`edgar-desk serve`) |
| 3000 | Web UI |

---

## Setup

### 1. Pull the models

```bash
ollama pull qwen3.6:35b-a3b   # 23 GB — orchestrator and sub-agents
ollama pull bge-m3:567m       # 1.2 GB — embeddings
ollama pull gpt-oss:20b       # 13 GB — independent eval judge (only needed for `eval`)
```

### 2. Install dependencies

```bash
uv sync                       # the app and its test tooling
uv sync --group rerank        # optional: adds the cross-encoder reranker (pulls torch, ~2.5 GB)
```

The reranker is a separate group because torch is large. Without it, narrative search
still works and simply returns vector-similarity order — results are a little worse, and
nothing errors, so it is easy not to notice. Add the group when you want the better
ranking. The reranker model itself (`BAAI/bge-reranker-v2-m3`, ~2 GB) downloads from
Hugging Face the first time it runs.

### 3. Tell the SEC who you are

The SEC requires a descriptive `User-Agent` with **real contact information** on every
request, and rate-limits or blocks traffic without one. Set this before ingesting:

```bash
export EDGAR_SEC_USER_AGENT="Your Name (you@example.com)"
```

There is no `.env` loading — configuration is read from the process environment, so export
it in your shell or prefix the command.

### 4. Start the infrastructure and load the corpus

```bash
docker compose up -d          # Postgres+pgvector, Temporal, Jaeger — ~30s to healthy
uv run edgar-desk db-init     # apply schema, seed the 20 covered companies
uv run edgar-desk ingest      # ~5 min: 13k XBRL facts, 4.4k embedded passages
uv run edgar-desk doctor      # everything should report ok
```

`doctor` is the checkpoint. It verifies each model is present, Postgres is reachable and
seeded, and tracing and Temporal are up. If anything is red, see
[Troubleshooting](#troubleshooting) before continuing.

### 5. Ask it something

```bash
uv run edgar-desk analyze "How has NVIDIA's R&D spend as a share of revenue changed \
  over five years, and what do their risk factors say about competition?"
```

Expect **30 seconds to two minutes** on a single-company question, and longer for
comparisons across several companies. It is running a 35B model locally; the first call
after a pause also pays to load the model into memory.

---

## Commands

```bash
edgar-desk doctor                  # check models, database, tracing, Temporal
edgar-desk db-init                 # apply the schema and seed the covered companies
edgar-desk ingest [-t NVDA]        # load facts and filing text from the SEC
edgar-desk ask "..."               # triage only: decompose a question, no retrieval
edgar-desk analyze "..." [--full]  # answer with a cited brief; --full adds the harness
edgar-desk pipeline "..."          # answer through the explicit graph pipeline
edgar-desk pipeline-diagram        # Mermaid diagram, generated from the graph itself
edgar-desk bakeoff                 # compare local models on tool use and typed output
edgar-desk eval [--full]           # grade answers against figures reported to the SEC
edgar-desk mcp                     # serve the corpus over MCP (stdio, for editors)
edgar-desk worker                  # Temporal worker for durable runs
edgar-desk durable "..."           # run a question as a durable workflow
edgar-desk serve                   # streaming chat API for the web UI
```

Every command takes `--help`.

## Web UI

Two processes, in separate terminals:

```bash
uv run edgar-desk serve                 # backend on :8000
cd web && npm install && npm run dev    # frontend on :3000
```

Then open http://localhost:3000. Chat streams token by token, tool calls appear as they
run, and asking it to publish a brief pauses for an explicit approval before anything is
written.

The frontend proxies `/api/*` to the backend, so the browser sees one origin. If you run
the backend on another port, point the proxy at it:

```bash
EDGAR_API_URL=http://127.0.0.1:8010 npm run dev
```

## Durable runs

Long research runs can be made crash-safe. Two terminals again:

```bash
uv run edgar-desk worker                       # leave running
uv run edgar-desk durable "Compare NVDA and AMD R&D intensity"
```

Progress is recorded in Temporal, so a restart resumes rather than starting over. Watch it
at http://localhost:8233.

## Where to look

- **Traces**: Jaeger at http://localhost:16686 — pick a service named `edgar-desk-*` to
  see a full agent run: model calls, tool calls, and their timings.
- **Workflows**: Temporal UI at http://localhost:8233
- **MCP from Cursor**: already wired up in [.cursor/mcp.json](.cursor/mcp.json); restart
  Cursor and the EDGAR tools appear.

## Testing

```bash
uv run pytest                 # 114 tests, ~10s
uv run ruff check src tests
```

No test needs a model server — agents are driven by `TestModel` and `FunctionModel`, so
the suite is fast and offline. Tests that need Postgres skip themselves when Docker is
down.

## Configuration

Every setting has a local-first default; override via environment variables.

| Variable | Default |
| --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` |
| `EDGAR_PRIMARY_MODEL` | `qwen3.6:35b-a3b` |
| `EDGAR_JUDGE_MODEL` | `gpt-oss:20b` |
| `EDGAR_EMBEDDING_MODEL` | `bge-m3:567m` |
| `EDGAR_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `EDGAR_DATABASE_URL` | `postgresql://edgar:edgar@localhost:5442/edgar` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` |
| `EDGAR_TEMPORAL_ADDRESS` | `localhost:7233` |
| `EDGAR_SEC_USER_AGENT` | *set this to your own contact info* |

### Using different models

Any Ollama model that supports tool calling will work:

```bash
export EDGAR_PRIMARY_MODEL=gpt-oss:20b
uv run edgar-desk bakeoff -m gpt-oss:20b     # measure it before trusting it
```

`bakeoff` grades a candidate on the two behaviors everything else depends on: calling
tools instead of guessing, and emitting schema-valid structured output. Both models above
score 100%. Worth running on anything new — local models vary a lot here, and
[NOTES.md](NOTES.md) documents a setting that silently broke one of them.

## Troubleshooting

**A port is already in use.** Edit the mapping in [docker-compose.yml](docker-compose.yml),
then set the matching variable above so the app follows. The Postgres port appears in both
places.

**`doctor` says a model is missing.** `ollama list` to see what you have; the names must
match exactly, tag included.

**Ingestion fails or returns 403.** Almost always `EDGAR_SEC_USER_AGENT`. The SEC wants
real contact details and blocks traffic without them. It is also rate-limited, so a
partial run is safe to re-run — ingestion is resumable and skips filings already loaded.

**Answers are slow.** Expected: this is local inference. `--no-rerank` on `analyze` skips
the cross-encoder, and `-t NVDA` limits ingestion to one company if you only want to try
things out.

**The web UI shows nothing.** Check the backend is up (`curl localhost:8000/api/health`)
and that the frontend proxy points at the right port.

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
