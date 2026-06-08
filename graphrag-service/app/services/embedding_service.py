from functools import lru_cache
import hashlib
import time
from typing import Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAI = None

from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.services.fallback_service import deterministic_fake_embedding, log_openai_fallback
from app.utils.vector_utils import l2_normalize

_openai_embedding_cache: dict[str, list[float]] = {}


@lru_cache
def load_embedding_model() -> SentenceTransformer:
    """Load and cache a local sentence-transformers model for legacy fallback paths."""

    settings = get_settings()
    if settings.embedding_model == settings.openai_embedding_model:
        raise ValueError(
            "EMBEDDING_MODEL points to the OpenAI embedding model. "
            "Use embed_text_openai() for OpenAI embeddings."
        )
    model = SentenceTransformer(settings.embedding_model)
    actual_dim = int(model.get_sentence_embedding_dimension() or 0)
    if actual_dim != settings.embedding_dim:
        raise ValueError(
            f"EMBEDDING_DIM={settings.embedding_dim} does not match "
            f"{settings.embedding_model} output dimension {actual_dim}."
        )
    return model


def embed_text(text: str) -> list[float]:
    """Embed one text string with the configured local embedding model."""

    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple text strings with a local model and return normalized float vectors."""

    model = load_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [validate_embedding_dim(vector.tolist()) for vector in embeddings]


def embed_text_openai(text: str, cache_key: str | None = None) -> list[float]:
    """Embed one text string using OpenAI text-embedding-3-small by default."""

    return embed_texts_openai([text], cache_keys=[cache_key] if cache_key else None)[0]


def embed_texts_openai(texts: list[str], cache_keys: list[str | None] | None = None) -> list[list[float]]:
    """Embed multiple texts with OpenAI, using a small in-memory cache when keys are provided."""

    settings = get_settings()
    if not settings.openai_api_key:
        if getattr(settings, "openai_fallback_on_error", True):
            log_openai_fallback("embedding skipped because OPENAI_API_KEY is missing, using fake embedding")
            return [deterministic_fake_embedding(text, settings.embedding_dim) for text in texts]
        raise ValueError("OPENAI_API_KEY is required to generate OpenAI embeddings.")
    if OpenAI is None:
        if getattr(settings, "openai_fallback_on_error", True):
            log_openai_fallback("embedding skipped because openai package is missing, using fake embedding")
            return [deterministic_fake_embedding(text, settings.embedding_dim) for text in texts]
        raise ValueError("openai package is required to generate OpenAI embeddings.")

    keys = cache_keys or [None for _ in texts]
    results: list[list[float] | None] = []
    missing_texts: list[str] = []
    missing_indexes: list[int] = []
    missing_cache_keys: list[str | None] = []

    for index, (text, key) in enumerate(zip(texts, keys, strict=True)):
        resolved_key = _build_embedding_cache_key(text, key)
        cached = _openai_embedding_cache.get(resolved_key)
        if cached is not None:
            results.append(cached)
            continue
        results.append(None)
        missing_texts.append(text)
        missing_indexes.append(index)
        missing_cache_keys.append(resolved_key)

    if missing_texts:
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            batch_size = max(1, int(getattr(settings, "openai_embed_batch_size", 30)))
            for start in range(0, len(missing_texts), batch_size):
                chunk_texts = missing_texts[start : start + batch_size]
                chunk_cache_keys = missing_cache_keys[start : start + batch_size]
                chunk_indexes = missing_indexes[start : start + batch_size]
                response = client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=chunk_texts,
                    dimensions=settings.embedding_dim
                )
                for cache_key_value, embedding_item, result_index in zip(
                    chunk_cache_keys,
                    response.data,
                    chunk_indexes,
                    strict=True,
                ):
                    vector = validate_embedding_dim(embedding_item.embedding)
                    _openai_embedding_cache[cache_key_value] = vector
                    results[result_index] = vector
                sleep_seconds = float(getattr(settings, "openai_request_sleep_seconds", 0.0))
                if sleep_seconds > 0 and start + batch_size < len(missing_texts):
                    time.sleep(sleep_seconds)
        except Exception as exc:
            if not getattr(settings, "openai_fallback_on_error", True):
                raise
            log_openai_fallback(f"embedding failed, using fake embedding: {exc}")
            for text, cache_key_value, result_index in zip(
                missing_texts,
                missing_cache_keys,
                missing_indexes,
                strict=True,
            ):
                vector = deterministic_fake_embedding(text, settings.embedding_dim)
                _openai_embedding_cache[cache_key_value] = vector
                results[result_index] = vector

    return [item for item in results if item is not None]


def get_news_embedding(news_id: int, text: str, provided_embedding: Sequence[float] | None = None) -> list[float]:
    """Return a provided news embedding or generate one with OpenAI without DB persistence."""

    if provided_embedding is not None:
        return validate_embedding_dim(provided_embedding)
    try:
        return embed_text_openai(text, cache_key=f"news:{news_id}")
    except Exception as exc:
        settings = get_settings()
        if not getattr(settings, "openai_fallback_on_error", True):
            raise
        log_openai_fallback(f"news embedding failed, using fake embedding: {exc}")
        return deterministic_fake_embedding(text, settings.embedding_dim)


def validate_embedding_dim(vector: Sequence[float]) -> list[float]:
    """Validate that a vector has the configured embedding dimension."""
    settings = get_settings()

    if len(vector) > settings.embedding_dim:
        vector = vector[:settings.embedding_dim]

    if len(vector) != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {settings.embedding_dim}, got {len(vector)}."
        )
    return [float(value) for value in vector]


def normalize_embedding(vector: Sequence[float]) -> list[float]:
    """Validate and L2-normalize an embedding vector."""

    return l2_normalize(validate_embedding_dim(vector))


def _build_embedding_cache_key(text: str, cache_key: str | None) -> str:
    """Build a stable in-memory cache key for repeated OpenAI embedding calls."""

    settings = get_settings()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    prefix = cache_key or "text"
    return f"{settings.openai_embedding_model}:{prefix}:{digest}"
