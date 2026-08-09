# Build notes

What each phase established, especially the things that were surprising or that changed
the design. Written for the next person (or the future me) who wonders why something
looks the way it does.

## Phase 0 — Foundation and model bake-off

### Local models can absolutely drive a typed agent

Both models scored **100% on assertions with perfect company-identification and
angle-routing scores** on the triage suite, once the eval and the model settings were
right. Ollama's self-hosted path enforces JSON schemas through llama.cpp's
grammar-constrained decoder, so structured output is valid at generation time rather
than validated after the fact.

Final numbers (6 cases, one run each):

| Model | Assertions | Avg duration | Avg model requests |
| --- | --- | --- | --- |
| `qwen3.6:35b-a3b` | 100% | 10.9s | 2.00 |
| `gpt-oss:20b` | 100% | 10.8s | 3.00 |

`qwen3.6` reaches the same answers in fewer round-trips, so it stays as the primary.
`gpt-oss:20b` is a genuinely independent second opinion, which is what makes it usable
as the eval judge later: a model grading its own output measures self-consistency, not
correctness.

### Temperature 0 was actively harmful

This one cost real time and is worth remembering. Greedy decoding is the obvious choice
for a structured-output classification task, so `temperature=0.0` went in early. It made
the agent hang.

Measured on one prompt naming only *uncovered* companies:

| Setting | Result |
| --- | --- |
| `temperature=0.0` | did not finish within 90s |
| `temperature=0.3` | 66.3s, 3,495 output tokens |
| provider default | 43.5s, 2,497 output tokens |

Easy prompts finished in 5-8s at every setting, so the failure only appeared on the
prompt the model found hardest to satisfy — exactly where it is least likely to be
noticed. The project now uses `temperature=0.3` in `DEFAULT_MODEL_SETTINGS`.

The mechanism is the familiar greedy-decoding repetition loop, made worse by a 262k
context window: there is a lot of room to keep generating before anything stops it.

### Capping `max_tokens` made things worse, not better

The instinct after seeing a 13,636-token runaway is to cap output. That backfires on a
reasoning model: the cap truncates the response mid-structure, the truncated output
fails validation, and the retry spends the same budget again. Worst-case latency went
*up*. Bounding the loop with `UsageLimits(request_limit=..., tool_calls_limit=...)`
is the control that actually works, because it limits the number of attempts rather than
crippling each one.

### Span-based evaluators need a tracer provider, and fail quietly without one

`ToolCorrectness` reported false on every single case while the tool was demonstrably
being called. Two separate things have to be true for span-based evaluators to work:

1. `logfire.configure(...)` must run before the evaluation, to install a tracer provider
2. agents must be instrumented (`Agent.instrument_all()` or the `Instrumentation` capability)

With only one of the two, the assertion silently reports false rather than raising —
so it reads as a model failure. An evaluator that touches `ctx.span_tree` directly does
raise a clear `SpanTreeRecordingError`, which is how this got diagnosed.

`observability.configure()` therefore always configures Logfire, and only makes the OTLP
*export* conditional on a collector actually listening, so nothing floods the console
with retries when Docker is not up.

### An eval that punishes correct behavior is an eval bug

The `out_of_universe` case asserted a `resolve_company` tool call. Both models correctly
answered it *without* calling the tool — they could see from the injected universe list
that Boeing and Ford were not covered, so there was nothing to resolve. The dataset-level
assertion was penalizing the efficient, correct path.

Fix: the tool-call assertion moved to a per-case evaluator on the five cases that name a
covered company, and `PlanIsWellFormed` now allows an empty plan when nothing is in
scope. Dataset-level evaluators apply to every case, so anything conditional belongs
either on the case or inside a custom evaluator.

## Phase 1 — Infrastructure

Four containers on 127.0.0.1 only, with ports offset from the defaults because this
machine already runs Postgres on 5432/5433 and other services on 6333, 8123 and 9000.

Three things worth knowing:

- **Jaeger v2 is built on the OpenTelemetry Collector**, so it ingests OTLP directly on
  4317/4318. That removes the separate collector container the plan assumed.
- **Jaeger has no `/v1/metrics` endpoint.** Logfire exports metrics alongside traces by
  default, which 404s on every flush. `metrics=False` in `logfire.configure()` fixes it.
- **Temporal's `auto-setup` binds its frontend to the container IP, not loopback**, so a
  healthcheck dialing `localhost:7233` fails inside the container while the service is
  perfectly healthy from outside. The check has to use the service name.

A full agent run now shows up in Jaeger as `invoke_agent triage` → `chat qwen3.6:35b-a3b`
→ two `execute_tool resolve_company` spans → a second model call.

### `CREATE TABLE IF NOT EXISTS` is not a migration system

Adding a `concept` column to `xbrl_facts` did nothing on a database that already had the
table, and the next ingest failed on the missing column. Anything added after a table's
first release needs an explicit `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, placed
*before* any index referencing it, since the whole file runs as a single batch.

## Phase 2 — Ingestion

20 companies, 13,122 facts and 4,434 embedded chunks in about 4.5 minutes.

### The SEC's `fy`/`fp` fields describe the filing, not the fact

This is the subtlety that decides whether the numbers are trustworthy, and getting it
wrong is silent. Three separate corrections were needed:

**One fact appears once per filing that restates it.** NVIDIA's FY2024 revenue
($60.922bn) appears in the FY2024, FY2025 and FY2026 10-Ks carrying `fy` 2024, 2025 and
2026. The economic identity is `(start_date, end_date)`, not the filing's fiscal year.
Keying on `fy` triple-counts it.

**Fiscal year must be derived from the period end date.** NVIDIA's FY2020 and FY2021
revenue, restated as comparatives under a tag the company only adopted in FY2022, all
arrived labelled 2022. Deriving the year from `end_date` fixed the whole series.

**Quarters are fiscal, not calendar.** A 10-K also contains quarterly breakdowns, and
every fact in it is labelled `FY`, so NVIDIA's $2.9bn Q4 sat next to its real $9.7bn
annual figure with the same label. Period is now classified by duration (≥300 days is a
year, 150-300 is a year-to-date cumulative, less is a quarter). Quarter *numbering* then
needs the company's fiscal year end month, inferred from its own annual periods: NVIDIA's
Q4 ends in January, which calendar arithmetic calls Q1.

**Synonymous tags double-count.** Filers report revenue as `Revenues` or
`RevenueFromContractWithCustomerExcludingAssessedTax`, and some report both for the same
period. Tags are collapsed to a canonical `concept` at ingestion, with an explicit
priority order, so the uniqueness constraint is on concept rather than tag.

The payoff is that FY2024 revenue now comes out exactly right across the universe:
Amazon $638.0bn, Apple $391.0bn, Alphabet $350.0bn, Microsoft $245.1bn, NVIDIA $60.9bn.
Those are gradeable eval targets, not approximations.

### Not every 10-K writes "Item 1A." above its risk factors

Intel's 10-K is a cross-reference layout: the item list points at page ranges ("Risk
Factors Pages 37 - 51") and the sections carry no item headings at all, so item-based
splitting returned nothing and Intel silently ingested zero chunks.

Intel marks sections with a *running page header* instead, repeating the title on each
page. The fallback finds titles that appear alone on a line many times over and takes the
span they bracket. Clustering matters: the same title also appears in the table of
contents, tens of thousands of characters away, and taking the first occurrence started
the section in the contents. Keeping only the evenly-spaced run drops that outlier.

### Two smaller API details

- **asyncpg binds by inferred parameter type** and ignores an inline `$1::date` cast, so
  DATE columns need real `date` objects rather than ISO strings.
- **`Embedder('ollama:...')` resolves its provider from `OLLAMA_BASE_URL`** in the
  environment and raises if unset, ignoring any configured base URL. Passing an explicit
  `OpenAIEmbeddingModel(..., provider=...)` keeps the configuration in one place.
- **pgvector returns a `Vector` object**, not a list; it needs `.to_list()` or
  `.to_numpy()` before anything treats it as a sequence.

## Phase 3 — Tools and capabilities

Two capabilities, `financials` and `narrative`, each bundling tools with the instructions
that explain when to use them. That bundling is the point: an agent gains "can analyze
financials" as one object in `capabilities=[...]` rather than as tools, instructions and
settings threaded separately through a constructor and kept in sync by hand.

### The SQL guard rewrote the queries it was validating

Model-written SQL is checked by scanning a copy with comments and string literals
stripped, so that `WHERE concept = 'delete this'` does not trip the keyword check. The
first version then *executed that stripped copy*, turning `WHERE ticker = 'AAPL'` into
`WHERE ticker = ''`. It returned zero rows, silently, for every query with a filter.

Nothing raised. A validator that returns a modified statement has to be explicit about
which version it returns, and this one now returns the original with only whitespace
trimmed. There are tests specifically for it.

Defence in depth otherwise: the guard rejects non-SELECT statements with a message the
model can act on, and the query still runs inside a `READ ONLY` transaction with a
statement timeout, so anything slipping past the guard cannot write.

### The agent generalized one tool's coverage to all data

Asked for a five-year R&D trend, the analyst returned two years and claimed the rest was
unavailable. It had called a tool listing *filings*, seen two 10-Ks, and concluded the
financial data was equally thin. In fact facts go back to FY2018; only narrative text is
limited to recent filings.

The tool was the problem, not the model: it reported one kind of coverage and was named
as though it reported all of it. Replacing it with `describe_coverage`, which states both
ranges and that they are independent, produced the full five-year series on the next run.
Worth remembering that a tool's return value is read as a statement about the world.

## Phase 4 — The harness

The researcher differs from the analyst only in its capability list: `Planning`,
`SubAgents` with two specialists, `Memory`, `ToolOutputLimits`, `SlidingWindowCompaction`,
`SystemReminders`, and a custom `RequireCitations`. No change to the agent loop.

### Writing a custom capability

`Capability` covers bundling instructions, tools and toolsets. Anything needing a
lifecycle hook has to subclass `AbstractCapability`. Two details cost a cycle each:

- **`get_instructions()` takes no `RunContext`.** It returns static text or a callable;
  a capability wanting per-request instructions returns the callable rather than
  accepting `ctx` as a parameter.
- **Hooks are awaited.** A synchronous `after_output_validate` fails with
  `TypeError: object Brief can't be used in 'await' expression`.

`after_output_validate` sees the semantic value (a `Brief`, not a dict) and can raise
`ModelRetry` to send the model back with a specific correction. `RequireCitations` uses
that to reject findings with no citations or citations to companies outside the covered
universe, and after two rejections gives up and appends a caveat instead — an endless
retry loop is worse than a clearly flagged answer.

### Harness constructor shapes

`ToolOutputLimits` takes `bands`, and a band is a threshold paired with an action:
`Band(over=12_000, action=Truncate(max_chars=12_000))`, not a bare `Truncate`.
`Reminder` takes `content` and `interval`, not `text` and `every`.

### The model picks the efficient path


Given a three-company comparison, the researcher used the direct tools and never called
`delegate_task` or the planning tools, answering in 4 requests. The capabilities are
offered, not imposed. Tests therefore assert that each capability *contributes its tools
and instructions*, which is the part under our control, rather than asserting the model
chooses to use them.

## Phase 5 — The graph pipeline

The same research work as an explicit `GraphBuilder` graph: decompose, fan out over
sub-questions with `.map()`, aggregate with `g.join(reduce_list_append)`, assess coverage
at a decision node, then synthesize. `graph.render()` emits the Mermaid diagram from the
graph itself, so the picture cannot drift from the code.

The trade against the agent version is worth being explicit about: the graph is less
adaptive and much more predictable. Control flow you can read, render, and test branch by
branch, versus control flow the model invents per run.

Two design choices worth keeping:

- **The coverage check is a plain function, not a model call.** "Did every sub-question
  come back with something substantive" is a property of the data. Asking a model adds
  latency and a failure mode for no benefit.
- **The deepen branch runs once.** A second pass over the same corpus finds the same
  nothing, and a caveated answer beats a long wait.

`build()` validates structure, so an unreachable node or a type mismatch between steps
fails at construction rather than mid-run. The builder API has no state persistence --
that is what the durable execution integration is for.

### Relative time windows need spelling out

Asked for "the last five years", the specialist returned FY2019-FY2023 rather than
FY2022-FY2026. The arithmetic was right and the window was wrong: it started where the
data began instead of counting back from the latest year. Both the capability and the
specialist now say so explicitly.

## Phase 6 — MCP in both directions

`edgar_desk.mcp_server` publishes the corpus over MCP so any client -- Cursor, Claude
Desktop, another agent -- can query it. `edgar_desk.agents.connected` does the reverse,
picking up tools from a remote server through the `MCP` capability.

### Tool names share one flat namespace, and collisions raise

Pointing the connected agent at this project's *own* MCP server surfaced it immediately:

```
UserError: MCPToolset 'localhost-mcp' defines a tool whose name conflicts with existing
tool from FunctionToolset 'financials': 'get_financials'.
```

A hard error rather than silent shadowing, which is the right call, and the message names
the fix. `PrefixTools(remote, 'ext')` namespaces the remote side while local tools keep
their names. Any server whose vocabulary overlaps yours needs this, so the helper
prefixes by default and takes `prefix=None` to opt out.

`allowed_tools=` is worth using on an unfamiliar server: every advertised tool costs
context on every request.

### `native=False` is the right default for local models

The `MCP` capability can hand a server to the model provider to execute server-side.
Local models have no native MCP support, and running the server in-process keeps its
credentials and traffic on this machine.

### Testing an MCP server means running it

The server tests drive a real client over stdio against the server as a subprocess, so
tool schemas, transport and serialization are all exercised the way an editor exercises
them. One wrinkle: `stdio_client` uses anyio cancel scopes, and pytest-asyncio runs
generator-fixture setup and teardown in *different tasks*, which fails with "attempted to
exit cancel scope in a different task". Every assertion passed and every teardown errored.
Entering the context inside the test body rather than in a fixture keeps both in one task.

## Phase 7 — Evals against ground truth, and durable runs

### The numbers are graded, not judged

This is the payoff from choosing EDGAR. Expected values are read from the corpus at eval
time -- not pasted into the file -- and the brief is checked for them directly.
`StatesReportedFigures` and `ComputesRatiosCorrectly` both scored **1.00 on every case**:
NVIDIA's FY2024 revenue, Apple's revenue and net income, a three-way comparison, AMD's
R&D-to-revenue ratio, and Intel's FY2024 *loss*. That is correctness, not plausibility,
and no judge model was involved.

Matching has to be forgiving about presentation and strict about value: "$60.922 billion",
"60,922 million", "60.922B" and "$60.9 billion" are all the same answer. One deliberate
trade, with a test naming it: bare unscaled magnitudes are accepted, because filings
tabulate in millions and rejecting them would fail correct answers. A figure wrong by
exactly 1000x could slip through, which is the rarer error than marking a passing agent
broken.

### Span-based evaluators catch right-by-luck

A model can emit a well-formed, correctly-shaped, entirely invented number.
`RetrievedRatherThanRecalled` reads the span tree rather than the output, so it
distinguishes an answer that was looked up from one that merely happens to be right.

The `SpanTree` API is `any`/`find`/`first`, not `flattened` -- and an evaluator that
raises shows up as an evaluator *failure* in the report rather than a failed assertion,
which is how the wrong method name got noticed.

### Eval bugs that punish correct behavior, again

`EveryFindingIsCited` originally required findings to exist, so the "what was Boeing
revenue" case failed for correctly returning none. Split into two evaluators: every
finding *present* must be cited (vacuously true when empty), and a question the corpus
*can* answer must produce findings. Same shape as the Phase 0 mistake -- the second time,
it was recognizable.

### Single runs are not results

Two cases failed their LLM judge on one run and passed with positive reasoning on the
next, at 95.5% and 100% overall. With a local model and a local judge there is real
run-to-run variance, so `repeat` exists for a reason, and `include_reasons=True` is what
makes a judge failure diagnosable rather than mysterious.

### Temporal broke in three different ways

Durable execution took the most debugging of any phase, and every failure was a crash
rather than an exception.

**Configuring tracing inside the event loop.** Temporal's core is a Rust extension whose
pyo3 task locals may only be bound once per process. Calling `logfire.configure()` from
inside `asyncio.run` binds them a second time and aborts the worker with
`must only be set once: TaskLocals`. Tracing has to be set up before the loop starts.

**Starting the worker twice.** `async with worker:` starts it, and so does
`await worker.run()`. Doing both is another double-bind and another Rust panic. One or
the other.

**The workflow sandbox re-importing native extensions.** Temporal re-imports the
workflow's module in a sandbox to enforce deterministic replay. With torch, pydantic-core
and asyncpg in the tree, that segfaulted the worker (exit 139) the instant it picked up a
task -- no traceback, no panic message, just a dead process 1.4 seconds in. Wrapping the
imports in `workflow.unsafe.imports_passed_through()` binds them to the already-imported
originals. Determinism is unaffected: the workflow body only awaits the agent, and all
the non-deterministic work already happens inside activities.

### Dependencies are serialized, so a pool cannot travel

Agent `deps` cross the activity boundary as JSON. A connection pool cannot. The pattern
that works: deps carry only serializable configuration and arrive with `pool=None`, and
`EdgarDeps.__post_init__` resolves the worker's process-wide pool on the activity side.
Activities run in the worker's own process -- only *workflow* code runs in the sandbox --
so a module-level global is visible there.

## Phase 8 — Streaming UI and human approval

`VercelAIAdapter.dispatch_request` is one line in a FastAPI route and does the whole job:
parse the SDK's request, run the agent, stream events back as protocol chunks. The
frontend never sees a Pydantic AI type.

### Two version numbers have to agree

`dispatch_request` defaults to `sdk_version=5`, and the installed `ai` package is 7. Basic
text streaming works either way, which is exactly why this is worth knowing: the mismatch
is invisible until you need a feature that only exists in the newer protocol. Tool
approval chunks require **v6 or later**, so with the default the run paused correctly on
the server and the client was simply never told. Setting `sdk_version=7` made
`tool-approval-request` appear.

(Compounding it: uvicorn without `--reload` was still serving the pre-edit module, so the
first fix appeared not to work. Worth restarting before concluding a change did nothing.)

### What approval actually protects

`requires_approval=True` makes the run end with `DeferredToolRequests` instead of calling
the tool; supplying `DeferredToolResults` resumes it. That guards against the *model*
acting without a human saying yes.

It is not an authorization boundary against the caller. Both AG-UI and the Vercel AI
protocol are built around the client submitting full history each request, so approvals
arrive from the client and a client that can reach the endpoint can approve a call it
invented. Authorization for anything sensitive belongs inside the tool function, checked
against the authenticated user in `deps`.

`publish_brief` therefore re-checks its own invariants after approval: a brief with an
uncited finding is refused even when approved, because the approver agreed to publish,
not to re-verify. There is a test for exactly that.
