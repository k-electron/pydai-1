-- EDGAR Desk schema.
--
-- Two storage shapes, because the agent answers two kinds of question:
--   xbrl_facts   exact reported numbers, queried with SQL  -> checkable ground truth
--   chunks       filing prose, queried by vector similarity -> narrative evidence

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS companies (
    cik         TEXT PRIMARY KEY,
    ticker      TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    ingested_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS filings (
    accession    TEXT PRIMARY KEY,
    cik          TEXT NOT NULL REFERENCES companies (cik) ON DELETE CASCADE,
    form         TEXT NOT NULL,
    fiscal_year  INTEGER NOT NULL,
    filed_on     DATE NOT NULL,
    period_end   DATE,
    primary_doc  TEXT
);

CREATE INDEX IF NOT EXISTS filings_cik_year_idx ON filings (cik, fiscal_year DESC);

-- One row per reported numeric fact. `value` stays DOUBLE PRECISION rather than NUMERIC
-- so it round-trips cleanly through the model layer; XBRL magnitudes do not need
-- exact decimal arithmetic for the comparisons this app makes.
CREATE TABLE IF NOT EXISTS xbrl_facts (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    cik           TEXT NOT NULL REFERENCES companies (cik) ON DELETE CASCADE,
    ticker        TEXT NOT NULL,
    taxonomy      TEXT NOT NULL,
    tag           TEXT NOT NULL,
    -- Canonical name grouping synonymous tags, so a question about "revenue" finds the
    -- number whether the filer tagged it `Revenues` or
    -- `RevenueFromContractWithCustomerExcludingAssessedTax`.
    concept       TEXT NOT NULL,
    unit          TEXT NOT NULL,
    value         DOUBLE PRECISION NOT NULL,
    fiscal_year   INTEGER NOT NULL,
    fiscal_period TEXT NOT NULL,
    form          TEXT NOT NULL,
    start_date    DATE,
    end_date      DATE NOT NULL,
    accession     TEXT,
    frame         TEXT
);

-- Identity is the economic period, not the filing.
--
-- The SEC's `fy`/`fp` fields label the filing a fact was reported in, so one fact
-- appears once per filing that restates it: NVIDIA's FY2024 revenue shows up in the
-- FY2024, FY2025 and FY2026 10-Ks with fy=2024, 2025 and 2026 respectively, all with
-- the same value. Keying on fy would triple-count it and make every ratio wrong.
-- (start_date, end_date) is what actually identifies the fact.
--
-- NULLS NOT DISTINCT matters: balance-sheet facts are instantaneous and carry no
-- start_date, and under the default NULL handling every one of them would be treated as
-- unique and insert a duplicate on each ingest.
DROP INDEX IF EXISTS xbrl_facts_unique_idx;
DROP INDEX IF EXISTS xbrl_facts_period_idx;

-- Additive migration.
--
-- `CREATE TABLE IF NOT EXISTS` is not a migration system: it silently does nothing when
-- the table exists, so a column added later never reaches a database created before it.
-- An explicit ALTER is idempotent and preserves existing rows. It has to run before any
-- index that references the column, since the whole file executes as one batch.
ALTER TABLE xbrl_facts ADD COLUMN IF NOT EXISTS concept TEXT;
UPDATE xbrl_facts SET concept = tag WHERE concept IS NULL;
ALTER TABLE xbrl_facts ALTER COLUMN concept SET NOT NULL;

-- Uniqueness is by concept, matching how ingestion collapses synonymous tags. Keying on
-- `tag` instead would let `Revenues` and `RevenueFromContractWithCustomerExcludingAssessedTax`
-- both persist for one period, and any aggregate over the concept would double-count.
CREATE UNIQUE INDEX IF NOT EXISTS xbrl_facts_concept_period_idx
    ON xbrl_facts (cik, concept, unit, start_date, end_date) NULLS NOT DISTINCT;

CREATE INDEX IF NOT EXISTS xbrl_facts_lookup_idx
    ON xbrl_facts (ticker, concept, fiscal_year DESC);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    accession    TEXT NOT NULL REFERENCES filings (accession) ON DELETE CASCADE,
    cik          TEXT NOT NULL REFERENCES companies (cik) ON DELETE CASCADE,
    ticker       TEXT NOT NULL,
    section      TEXT NOT NULL,
    fiscal_year  INTEGER NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    token_count  INTEGER,
    -- bge-m3 produces 1024-dimensional embeddings.
    embedding    vector(1024)
);

CREATE UNIQUE INDEX IF NOT EXISTS chunks_unique_idx ON chunks (accession, section, chunk_index);
CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (ticker, fiscal_year DESC);

-- HNSW over cosine distance. Built after ingestion in Phase 2; creating it on an empty
-- table is fine and cheap.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Briefs the agent produced, and whether a human signed off. Phase 8 gates export on this.
CREATE TABLE IF NOT EXISTS review_queue (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    question     TEXT NOT NULL,
    brief        JSONB NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at   TIMESTAMPTZ,
    decided_by   TEXT,
    note         TEXT,
    CONSTRAINT review_status_valid CHECK (status IN ('pending', 'approved', 'rejected'))
);

CREATE INDEX IF NOT EXISTS review_queue_status_idx ON review_queue (status, created_at DESC);
