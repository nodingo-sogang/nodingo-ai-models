import itertools
import json
import re
from collections import defaultdict
from datetime import date, datetime, timezone

from keybert import KeyBERT
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAI = None

from app.config import get_settings
from app.schemas import (
    AnalyzeNewsBatchRequest,
    AnalyzeNewsBatchResponse,
    ExistingKeywordInput,
    KeywordCandidate,
    KeywordRelationResult,
    KeywordResult,
    NewsAnalysisResult,
)
from app.services.embedding_service import (
    embed_text_openai,
    get_news_embedding,
    load_embedding_model,
    validate_embedding_dim,
)
from app.services.relation_service import build_news_relations
from app.utils.text_utils import clean_text, safe_join_title_body
from app.utils.vector_utils import clip_score, cosine_similarity, zero_vector


def analyze_news_batch(request: AnalyzeNewsBatchRequest) -> AnalyzeNewsBatchResponse:
    """Analyze news articles and return embeddings, keywords, and keyword relations."""

    for existing in request.existing_keywords:
        validate_embedding_dim(existing.embedding)

    news_results: list[NewsAnalysisResult] = []
    for news in request.news:
        news_text = safe_join_title_body(news.title, news.body)
        news_embedding = get_news_embedding(news.news_id, news_text, news.embedding)
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
            weighted_keywords.append(
                keyword.model_copy(
                    update={
                        "weight": weight,
                        "evidence_text": keyword.evidence_text,
                    }
                )
            )
        news_results.append(
            NewsAnalysisResult(
                news_id=news.news_id,
                published_at=news.published_at,
                embedding=news_embedding,
                keywords=weighted_keywords,
            )
        )

    return AnalyzeNewsBatchResponse(
        news_results=news_results,
        keyword_relations=build_keyword_relations(news_results, target_date=request.target_date),
        news_relations=build_news_relations(
            news_results,
            request.top_k_news_relations,
            request.min_news_relation_score,
        ),
    )


def normalize_keyword(text: str) -> str:
    """Normalize a keyword for matching across aliases and backend records."""

    value = clean_text(text).lower()
    value = re.sub(r"[^\w가-힣\s.-]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_keywords_from_news(title: str, body: str, top_k: int) -> list[KeywordCandidate]:
    """Extract top keyword candidates from title and body using OpenAI with a fallback."""

    settings = get_settings()
    if settings.use_openai_keyword_extraction and settings.openai_api_key and OpenAI is not None:
        try:
            return extract_keywords_from_news_openai(title, body, top_k)
        except Exception:
            return extract_keywords_from_news_fallback(title, body, top_k)
    return extract_keywords_from_news_fallback(title, body, top_k)


def extract_keywords_from_news_openai(title: str, body: str, top_k: int) -> list[KeywordCandidate]:
    """Extract Korean news graph keywords with OpenAI and parse JSON output."""

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    text = safe_join_title_body(title, body)
    prompt = (
        "한국어 뉴스 제목과 200자 요약문에서 그래프 노드로 쓰기 좋은 짧은 핵심 키워드를 추출하세요. "
        "키워드는 사람/기업/산업/정책/시장 이슈 중심의 짧은 명사구여야 합니다. "
        "문장형 키워드나 너무 긴 구절은 피하세요. "
        "각 키워드는 word, normalized_word, weight, extraction_score, evidence_text 필드를 가져야 합니다. "
        "normalized_word는 중복 병합에 쓰이므로 소문자와 공백 정리를 적용하세요. "
        "weight와 extraction_score는 0과 1 사이 숫자입니다. "
        f"키워드는 최대 {max(5, min(top_k, 10))}개만 반환하세요. "
        "반드시 JSON object만 반환하세요.\n\n"
        "예시 키워드: 반도체, 기준금리, AI 투자, 삼성전자, 환율, 부동산 PF\n\n"
        f"title: {clean_text(title)}\n"
        f"body: {clean_text(body)}\n\n"
        '출력 형식: {"keywords":[{"word":"...", "normalized_word":"...", "weight":0.8, '
        '"extraction_score":0.8, "evidence_text":"..."}]}'
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": "You extract concise Korean news keywords for graph nodes and return strict JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=800,
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    raw_keywords = payload.get("keywords", [])
    candidates: list[KeywordCandidate] = []
    seen: set[str] = set()
    for item in raw_keywords:
        word = clean_text(str(item.get("word", "")))
        normalized_word = normalize_keyword(str(item.get("normalized_word") or word))
        if not word or not normalized_word or normalized_word in seen:
            continue
        seen.add(normalized_word)
        score = clip_score(float(item.get("extraction_score", item.get("weight", 0.5))))
        candidates.append(
            KeywordCandidate(
                word=word,
                normalized_word=normalized_word,
                weight=clip_score(float(item.get("weight", score))),
                extraction_score=score,
                evidence_text=clean_text(str(item.get("evidence_text", ""))) or None,
            )
        )
        if len(candidates) >= max(1, top_k):
            break
    return candidates


def extract_keywords_from_news_fallback(title: str, body: str, top_k: int) -> list[KeywordCandidate]:
    """Fallback keyword extraction using simple Korean/English token frequency."""

    text = safe_join_title_body(title, body)
    if not text:
        return []

    try:
        return extract_keywords_from_news_keybert(title, body, top_k)
    except Exception:
        return extract_keywords_from_news_frequency(title, body, top_k)


def extract_keywords_from_news_keybert(title: str, body: str, top_k: int) -> list[KeywordCandidate]:
    """Legacy KeyBERT fallback keyword extraction."""

    text = safe_join_title_body(title, body)

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
                weight=clip_score(float(score)),
            )
        )
    return candidates


def extract_keywords_from_news_frequency(title: str, body: str, top_k: int) -> list[KeywordCandidate]:
    """Lightweight frequency fallback for environments without OpenAI/local models."""

    text = safe_join_title_body(title, body)
    stopwords = {
        "기자",
        "뉴스",
        "관련",
        "이번",
        "통해",
        "대한",
        "있는",
        "없는",
        "것으로",
        "밝혔다",
        "했다",
        "한다",
        "그리고",
        "하지만",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.-]*|[가-힣]{2,}", text)
    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        normalized = normalize_keyword(token)
        if len(normalized) < 2 or normalized in stopwords:
            continue
        counts[normalized] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(1, top_k)]
    max_count = max((count for _, count in ranked), default=1)
    return [
        KeywordCandidate(
            word=word,
            normalized_word=word,
            extraction_score=clip_score(count / max_count),
            weight=clip_score(count / max_count),
            evidence_text=None,
        )
        for word, count in ranked
    ]


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
        embedding = existing.embedding if existing else embed_keyword(candidate.normalized_word)
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
                evidence_text=candidate.evidence_text,
            )
        )
    return results


def embed_keyword(normalized_word: str) -> list[float]:
    """Embed a keyword with OpenAI, falling back to a zero vector when unavailable."""

    try:
        return embed_text_openai(normalized_word, cache_key=f"keyword:{normalized_word}")
    except Exception:
        return zero_vector(get_settings().embedding_dim)


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


def build_keyword_relations(
    news_keyword_results: list[NewsAnalysisResult],
    target_date: date | None = None,
) -> list[KeywordRelationResult]:
    """Build keyword relation candidates from keywords appearing in the same news."""

    pair_evidence: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"news_ids": set(), "weights": [], "recency_scores": [], "left": None, "right": None}
    )

    for news_result in news_keyword_results:
        recency_score = calculate_recency_score(news_result.published_at, target_date)
        for left, right in itertools.combinations(news_result.keywords, 2):
            if left.normalized_word == right.normalized_word:
                continue
            source, target = _order_keyword_pair(left, right)
            key = (source.normalized_word, target.normalized_word)
            pair_evidence[key]["news_ids"].add(news_result.news_id)
            pair_evidence[key]["weights"].append((source.weight + target.weight) / 2.0)
            pair_evidence[key]["recency_scores"].append(recency_score)
            pair_evidence[key]["left"] = source
            pair_evidence[key]["right"] = target

    relations: list[KeywordRelationResult] = []
    for evidence in pair_evidence.values():
        left = evidence["left"]
        right = evidence["right"]
        news_ids = sorted(evidence["news_ids"])
        avg_weight = sum(evidence["weights"]) / len(evidence["weights"])
        avg_recency_score = sum(evidence["recency_scores"]) / len(evidence["recency_scores"])
        relation_score = calculate_keyword_relation_score(
            left,
            right,
            len(news_ids),
            avg_weight,
            cosine_similarity(left.embedding, right.embedding),
            recency_score=avg_recency_score,
        )
        relations.append(
            KeywordRelationResult(
                subject_keyword_id=left.keyword_id,
                related_keyword_id=right.keyword_id,
                subject_normalized_word=left.normalized_word,
                related_normalized_word=right.normalized_word,
                relation_score=relation_score,
                evidence_news_ids=news_ids,
            )
        )
    return sorted(
        relations,
        key=lambda item: (-item.relation_score, item.subject_normalized_word, item.related_normalized_word),
    )


def _order_keyword_pair(left: KeywordResult, right: KeywordResult) -> tuple[KeywordResult, KeywordResult]:
    """Order relation endpoints by id when possible, otherwise by normalized word."""

    if left.keyword_id is not None and right.keyword_id is not None:
        return (left, right) if left.keyword_id <= right.keyword_id else (right, left)
    return (left, right) if left.normalized_word <= right.normalized_word else (right, left)


def calculate_recency_score(published_at: datetime | None, target_date: date | None = None) -> float:
    """Calculate rule-based recency score from published_at and target_date.

    Missing or unparsable dates use 0.5 so relation scoring remains usable without
    pretending the article is maximally fresh.
    """

    if published_at is None:
        return 0.5
    if published_at.tzinfo is not None:
        published_date = published_at.astimezone(timezone.utc).date()
    else:
        published_date = published_at.date()
    base_date = target_date or datetime.now().date()
    days_diff = (base_date - published_date).days
    if days_diff <= 0:
        return 1.0
    if days_diff <= 1:
        return 0.9
    if days_diff <= 3:
        return 0.7
    if days_diff <= 7:
        return 0.5
    return 0.3
