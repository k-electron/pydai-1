"""Central configuration, read from the environment with local-first defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for every EDGAR Desk process."""

    ollama_base_url: str
    """OpenAI-compatible endpoint of the local Ollama server."""

    primary_model: str
    """Orchestrator and sub-agent model."""

    judge_model: str
    """Second model used to grade evals. Must differ from `primary_model`: a model
    grading its own output measures self-consistency, not correctness."""

    embedding_model: str
    reranker_model: str

    database_url: str
    otlp_endpoint: str
    temporal_address: str

    sec_user_agent: str
    """The SEC requires a descriptive User-Agent with contact info on every request,
    and rejects traffic without one."""

    @property
    def primary_model_id(self) -> str:
        return f'ollama:{self.primary_model}'

    @property
    def judge_model_id(self) -> str:
        return f'ollama:{self.judge_model}'

    @property
    def embedding_model_id(self) -> str:
        return f'ollama:{self.embedding_model}'


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        ollama_base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),
        primary_model=os.getenv('EDGAR_PRIMARY_MODEL', 'qwen3.6:35b-a3b'),
        judge_model=os.getenv('EDGAR_JUDGE_MODEL', 'gpt-oss:20b'),
        embedding_model=os.getenv('EDGAR_EMBEDDING_MODEL', 'bge-m3:567m'),
        reranker_model=os.getenv('EDGAR_RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3'),
        database_url=os.getenv(
            'EDGAR_DATABASE_URL', 'postgresql://edgar:edgar@localhost:5442/edgar'
        ),
        otlp_endpoint=os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318'),
        temporal_address=os.getenv('EDGAR_TEMPORAL_ADDRESS', 'localhost:7233'),
        sec_user_agent=os.getenv(
            'EDGAR_SEC_USER_AGENT', 'EDGAR Desk learning project (contact@example.com)'
        ),
    )
