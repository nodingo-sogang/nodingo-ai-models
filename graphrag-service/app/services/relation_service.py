from app.schemas import NewsEmbeddingInput, NewsRelationResult
from app.services.embedding_service import validate_embedding_dim
from app.utils.vector_utils import clip_score, cosine_similarity


def calculate_keyword_relation_score(
    keyword_a,
    keyword_b,
    shared_news_count: int,
    avg_news_keyword_weight: float,
    embedding_similarity: float,
    recency_score: float,
) -> float:
    """Calculate a generic keyword relation score with 0..1 clipping."""

    cooccurrence_score = min(1.0, shared_news_count / 3.0)
    return clip_score(
        0.35 * cooccurrence_score
        + 0.35 * avg_news_keyword_weight
        + 0.20 * max(0.0, embedding_similarity)
        + 0.10 * recency_score
    )


def build_keyword_relations(news_keyword_results):
    """Delegate keyword relation generation to keyword_service."""

    from app.services.keyword_service import build_keyword_relations as _build_keyword_relations

    return _build_keyword_relations(news_keyword_results)


def build_news_relations(
    news_embeddings: list[NewsEmbeddingInput],
    top_k: int,
    min_score: float,
) -> list[NewsRelationResult]:
    """Build cosine-similarity based news relations, excluding self-relations."""

    for item in news_embeddings:
        validate_embedding_dim(item.embedding)

    best_pairs: dict[tuple[int, int], float] = {}
    for source in news_embeddings:
        scored = []
        for target in news_embeddings:
            if source.news_id == target.news_id:
                continue
            score = clip_score(max(0.0, cosine_similarity(source.embedding, target.embedding)))
            if score >= min_score:
                scored.append((target.news_id, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        for target_id, score in scored[: max(0, top_k)]:
            subject_id, related_id = sorted([source.news_id, target_id])
            key = (subject_id, related_id)
            best_pairs[key] = max(best_pairs.get(key, 0.0), score)

    return [
        NewsRelationResult(
            subject_news_id=subject_id,
            related_news_id=related_id,
            relation_score=score,
        )
        for (subject_id, related_id), score in sorted(best_pairs.items())
    ]
