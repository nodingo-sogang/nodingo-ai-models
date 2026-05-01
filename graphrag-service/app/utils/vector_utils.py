from typing import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculate cosine similarity and map invalid zero-vector cases to 0."""

    vec_a = np.asarray(a, dtype=float)
    vec_b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def weighted_average(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """Calculate a weighted vector average using numpy for stable math."""

    if not vectors:
        return []
    matrix = np.asarray(vectors, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    total = float(weight_array.sum())
    if total <= 0:
        return matrix.mean(axis=0).astype(float).tolist()
    return np.average(matrix, axis=0, weights=weight_array).astype(float).tolist()


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalized copy of the vector."""

    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(float).tolist()
    return (vec / norm).astype(float).tolist()


def zero_vector(dim: int) -> list[float]:
    """Create a zero vector with the configured embedding dimension."""

    return [0.0 for _ in range(dim)]


def clip_score(value: float) -> float:
    """Clip a score into the inclusive 0..1 range."""

    return float(max(0.0, min(1.0, value)))
