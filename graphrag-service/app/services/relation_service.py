from app.schemas import NewsEmbeddingInput, NewsRelationResult
from app.services.embedding_service import validate_embedding_dim
from app.utils.vector_utils import clip_score, cosine_similarity

import numpy as np


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
    """Build cosine-similarity based news relations using ultra-fast NumPy matrix operations."""

    if not news_embeddings:
        return []

    for item in news_embeddings:
        validate_embedding_dim(item.embedding)

    # 1. ID와 임베딩을 분리하여 NumPy 배열로 변환
    news_ids = np.array([item.news_id for item in news_embeddings])
    embeddings = np.array([item.embedding for item in news_embeddings])

    # 2. 임베딩 정규화 (Normalization)
    # 정규화를 하면 코사인 유사도를 단순 행렬 내적(Dot Product)으로 초고속 계산 가능
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # 0 나누기 방지
    normalized_embeddings = embeddings / norms

    # 3. 🌟 핵심: N x N 유사도 행렬을 한 방에 계산 (이중 for문 탈출)
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # 4. 자기 자신과의 비교(대각선 값)는 0으로 처리
    np.fill_diagonal(similarity_matrix, 0.0)

    # 0 이하 클리핑 (음수 방지)
    similarity_matrix = np.clip(similarity_matrix, 0.0, None)

    best_pairs: dict[tuple[int, int], float] = {}

    # 5. 계산된 행렬에서 결과만 쏙쏙 뽑아내기
    for i in range(len(news_ids)):
        source_id = news_ids[i]

        # 현재 뉴스와 유사도가 min_score 이상인 타겟들의 인덱스 찾기
        valid_indices = np.where(similarity_matrix[i] >= min_score)[0]

        if len(valid_indices) == 0:
            continue

        valid_scores = similarity_matrix[i, valid_indices]

        # ID와 Score 묶어서 리스트 만들기
        scored = [(news_ids[idx], float(valid_scores[idx])) for idx in range(len(valid_indices))]

        # 점수는 내림차순, ID는 오름차순으로 정렬
        scored.sort(key=lambda pair: (-pair[1], pair[0]))

        # 상위 top_k개만 쌍으로 묶어서 저장
        for target_id, score in scored[:top_k]:
            subject_id, related_id = sorted([source_id, target_id])
            key = (subject_id, related_id)
            best_pairs[key] = max(best_pairs.get(key, 0.0), score)

    # 6. 최종 DTO 변환 및 반환
    return [
        NewsRelationResult(
            subject_news_id=subject_id,
            related_news_id=related_id,
            relation_score=score,
        )
        for (subject_id, related_id), score in sorted(best_pairs.items())
    ]
