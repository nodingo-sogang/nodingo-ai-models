from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.utils.vector_utils import l2_normalize


@lru_cache
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the configured sentence-transformers embedding model."""

    settings = get_settings()
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
    """Embed multiple text strings and return normalized float vectors."""

    model = load_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [validate_embedding_dim(vector.tolist()) for vector in embeddings]


def validate_embedding_dim(vector: Sequence[float]) -> list[float]:
    """Validate that a vector has the configured embedding dimension."""

    settings = get_settings()
    if len(vector) != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {settings.embedding_dim}, got {len(vector)}."
        )
    return [float(value) for value in vector]


def normalize_embedding(vector: Sequence[float]) -> list[float]:
    """Validate and L2-normalize an embedding vector."""

    return l2_normalize(validate_embedding_dim(vector))
