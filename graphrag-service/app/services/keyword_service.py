import itertools
import re
from collections import defaultdict

from keybert import KeyBERT

from app.schemas import (
    AnalyzeNewsBatchRequest,
    AnalyzeNewsBatchResponse,
    ExistingKeywordInput,
    KeywordCandidate,
    KeywordRelationResult,
    KeywordResult,
    NewsAnalysisResult,
)
from app.services.embedding_service import embed_text, embed_texts, load_embedding_model, validate_embedding_dim
from app.utils.text_utils import clean_text, safe_join_title_body
from app.utils.vector_utils import clip_score, cosine_similarity


def analyze_news_batch(request: AnalyzeNewsBatchRequest) -> AnalyzeNewsBatchResponse:
    """Analyze news articles and return embeddings, keywords, and keyword relations."""

    news_texts = [safe_join_title_body(item.title, item.body) for item in request.news]
    news_embeddings = embed_texts(news_texts) if news_texts else []
    for existing in request.existing_keywords:
        validate_embedding_dim(existing.embedding)

    news_results: list[NewsAnalysisResult] = []
    for news, news_embedding in zip(request.news, news_embeddings, strict=True):
        candidates = extract_keywords_from_news(news.title, news.body, request.top_k_keywords)
        keywords = resolve_keywords(candidates, request.existing_keywords)
        weighted_keywords = []
        for keyword in keywords:
            weight = calculate_news_keyword_weight(
                news.title,
                news.body,
                keyword.normalized_word,
                keyword.extraction_score,
                news_embedding,
                keyword.embedding,
            )
            weighted_keywords.append(keyword.model_copy(update={"weight": weight}))
        news_results.append(
            NewsAnalysisResult(
                news_id=news.news_id,
                embedding=news_embedding,
                keywords=weighted_keywords,
            )
        )

    return AnalyzeNewsBatchResponse(
        news_results=news_results,
        keyword_relations=build_keyword_relations(news_results),
    )


def normalize_keyword(text: str) -> str:
    """Normalize a keyword for matching across aliases and backend records."""

    value = clean_text(text).lower()
    value = re.sub(r"[^\w가-힣\s.-]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_keywords_from_news(title: str, body: str, top_k: int) -> list[KeywordCandidate]:
    """Extract top keyword candidates from title and body using KeyBERT."""

    text = safe_join_title_body(title, body)
    if not text:
        return []

    model = KeyBERT(model=load_embedding_model())
    raw_keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=max(1, top_k),
        use_mmr=True,
        diversity=0.45,
    )
    candidates: list[KeywordCandidate] = []
    seen: set[str] = set()
    for word, score in raw_keywords:
        normalized_word = normalize_keyword(word)
        if not normalized_word or normalized_word in seen:
            continue
        seen.add(normalized_word)
        candidates.append(
            KeywordCandidate(
                word=clean_text(word),
                normalized_word=normalized_word,
                extraction_score=clip_score(float(score)),
            )
        )
    return candidates


def resolve_keywords(
    candidates: list[KeywordCandidate],
    existing_keywords: list[ExistingKeywordInput],
) -> list[KeywordResult]:
    """Resolve extracted keywords against existing keywords by normalized_word."""

    existing_by_normalized = {
        normalize_keyword(item.normalized_word): item for item in existing_keywords
    }
    results: list[KeywordResult] = []
    for candidate in candidates:
        existing = existing_by_normalized.get(candidate.normalized_word)
        embedding = existing.embedding if existing else embed_text(candidate.normalized_word)
        word = existing.word if existing else candidate.word
        normalized_word = existing.normalized_word if existing else candidate.normalized_word
        results.append(
            KeywordResult(
                keyword_id=existing.keyword_id if existing else None,
                word=word,
                normalized_word=normalized_word,
                embedding=embedding,
                weight=0.0,
                is_new=existing is None,
                aliases=build_keyword_aliases(word, normalized_word),
                extraction_score=candidate.extraction_score,
            )
        )
    return results


def calculate_news_keyword_weight(
    title: str,
    body: str,
    keyword: str,
    extraction_score: float,
    news_embedding: list[float],
    keyword_embedding: list[float],
) -> float:
    """Calculate keyword importance in a news article from extraction, text, and similarity."""

    text = safe_join_title_body(title, body).lower()
    normalized = normalize_keyword(keyword)
    occurrences = text.count(normalized.lower()) if normalized else 0
    occurrence_score = min(1.0, occurrences / 3.0)
    similarity_score = max(0.0, cosine_similarity(news_embedding, keyword_embedding))
    title_bonus = 0.1 if normalized and normalized in clean_text(title).lower() else 0.0
    return clip_score(0.55 * extraction_score + 0.25 * similarity_score + 0.20 * occurrence_score + title_bonus)


def build_keyword_aliases(word: str, normalized_word: str) -> list[str]:
    """Build simple aliases to help the backend persist KeywordAlias rows."""

    aliases = []
    for value in [word, normalized_word, normalize_keyword(word)]:
        if value and value not in aliases:
            aliases.append(value)
    return aliases


def calculate_keyword_relation_score(
    keyword_a: KeywordResult,
    keyword_b: KeywordResult,
    shared_news_count: int,
    avg_news_keyword_weight: float,
    embedding_similarity: float,
    recency_score: float,
) -> float:
    """Score keyword-to-keyword relation strength for graph edges."""

    cooccurrence_score = min(1.0, shared_news_count / 3.0)
    positive_similarity = max(0.0, embedding_similarity)
    return clip_score(
        0.35 * cooccurrence_score
        + 0.35 * avg_news_keyword_weight
        + 0.20 * positive_similarity
        + 0.10 * recency_score
    )


def build_keyword_relations(news_keyword_results: list[NewsAnalysisResult]) -> list[KeywordRelationResult]:
    """Build keyword relation candidates from keywords appearing in the same news."""

    pair_evidence: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"news_ids": set(), "weights": [], "left": None, "right": None}
    )

    for news_result in news_keyword_results:
        for left, right in itertools.combinations(news_result.keywords, 2):
            if left.normalized_word == right.normalized_word:
                continue
            source, target = _order_keyword_pair(left, right)
            key = (source.normalized_word, target.normalized_word)
            pair_evidence[key]["news_ids"].add(news_result.news_id)
            pair_evidence[key]["weights"].append((source.weight + target.weight) / 2.0)
            pair_evidence[key]["left"] = source
            pair_evidence[key]["right"] = target

    relations: list[KeywordRelationResult] = []
    for evidence in pair_evidence.values():
        left = evidence["left"]
        right = evidence["right"]
        news_ids = sorted(evidence["news_ids"])
        avg_weight = sum(evidence["weights"]) / len(evidence["weights"])
        relation_score = calculate_keyword_relation_score(
            left,
            right,
            len(news_ids),
            avg_weight,
            cosine_similarity(left.embedding, right.embedding),
            recency_score=1.0,
        )
        relations.append(
            KeywordRelationResult(
                source_keyword_id=left.keyword_id,
                target_keyword_id=right.keyword_id,
                source_normalized_word=left.normalized_word,
                target_normalized_word=right.normalized_word,
                relation_score=relation_score,
                evidence_news_ids=news_ids,
            )
        )
    return sorted(
        relations,
        key=lambda item: (-item.relation_score, item.source_normalized_word, item.target_normalized_word),
    )


def _order_keyword_pair(left: KeywordResult, right: KeywordResult) -> tuple[KeywordResult, KeywordResult]:
    """Order relation endpoints by id when possible, otherwise by normalized word."""

    if left.keyword_id is not None and right.keyword_id is not None:
        return (left, right) if left.keyword_id <= right.keyword_id else (right, left)
    return (left, right) if left.normalized_word <= right.normalized_word else (right, left)
