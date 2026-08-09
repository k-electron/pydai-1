"""Model and embedder factories, all pointed at the local Ollama server."""

from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from edgar_desk.settings import get_settings

DEFAULT_MODEL_SETTINGS = ModelSettings(temperature=0.3)
"""Settings shared by every agent in the project.

Both choices here are load-bearing and came from measurement, not preference.

`temperature` is deliberately not 0. Greedy decoding looks like the obvious choice for a
structured-output task, but it makes this model spiral on prompts it cannot satisfy. On a
question naming only uncovered companies: temperature 0 ran past 90s without finishing,
0.3 finished in 66s, and the provider default in 44s. Easy prompts were unaffected
(5-8s at every setting), so the failure only shows up where it is hardest to notice.

There is deliberately no `max_tokens`. Capping output on a reasoning model truncates the
response mid-structure, which fails validation and triggers a retry that spends the same
budget again -- the cap made the worst case slower, not faster.
"""


@lru_cache(maxsize=1)
def _provider() -> OllamaProvider:
    return OllamaProvider(base_url=get_settings().ollama_base_url)


def ollama_model(model_name: str | None = None) -> OllamaModel:
    """Build a model backed by self-hosted Ollama.

    Self-hosted Ollama enforces `response_format` json_schema through llama.cpp's
    grammar-constrained decoder, so `NativeOutput` is schema-valid at generation time
    rather than validated after the fact.
    """
    settings = get_settings()
    return OllamaModel(model_name or settings.primary_model, provider=_provider())


def primary_model() -> OllamaModel:
    return ollama_model(get_settings().primary_model)


def judge_model() -> OllamaModel:
    return ollama_model(get_settings().judge_model)


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    """Embeddings from the same local Ollama server.

    Built from an explicit model object rather than the `'ollama:name'` string shorthand:
    the shorthand resolves its provider from `OLLAMA_BASE_URL` in the environment and
    fails if that is unset, ignoring the base URL configured here.
    """
    settings = get_settings()
    model = OpenAIEmbeddingModel(settings.embedding_model, provider=_provider())
    return Embedder(model, instrument=True)
