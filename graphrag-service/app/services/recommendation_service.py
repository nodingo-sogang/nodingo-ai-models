from datetime import date

from app.schemas import CandidateKeywordInput, RecommendKeywordResult
from app.services.embedding_service import validate_embedding_dim
from app.utils.vector_utils import clip_score, cosine_similarity


def calculate_recommend_keyword_score(
    user_embedding: list[float],
    keyword_embedding: list[float],
    recent_importance: float,
    is_user_interest: bool,
) -> float:
    """Calculate recommended keyword score from user similarity and business signals."""

    cosine = max(0.0, cosine_similarity(user_embedding, keyword_embedding))
    interest_bonus = 1.0 if is_user_interest else 0.0
    return clip_score(0.55 * cosine + 0.25 * interest_bonus + 0.20 * clip_score(recent_importance))


def recommend_keywords(
    user_id: int,
    user_embedding: list[float],
    candidate_keywords: list[CandidateKeywordInput],
    target_date: date,
    top_k: int,
) -> list[RecommendKeywordResult]:
    """Rank candidate keywords and return top_k recommendation rows."""

    validate_embedding_dim(user_embedding)
    results: list[RecommendKeywordResult] = []
    for candidate in candidate_keywords:
        validate_embedding_dim(candidate.embedding)
        score = calculate_recommend_keyword_score(
            user_embedding,
            candidate.embedding,
            candidate.recent_importance,
            candidate.is_user_interest,
        )
        results.append(
            RecommendKeywordResult(
                user_id=user_id,
                keyword_id=candidate.keyword_id,
                target_date=target_date,
                score=score,
                summary=None,
            )
        )
    return sorted(results, key=lambda item: (-item.score, item.keyword_id))[: max(0, top_k)]
