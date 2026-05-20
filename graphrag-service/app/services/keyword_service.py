import itertools
import json
import re
import sys
import concurrent.futures
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


def generate_summary_with_llm(title: str, body: str) -> str:
    """OpenAI(gpt-4o-mini)를 이용해 뉴스 기사를 200자 이내로 요약합니다."""
    settings = get_settings()

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        prompt = f"다음 뉴스 기사의 제목과 본문을 읽고, 핵심 내용을 한국어로 200자 이내로 요약해줘.\n\n[제목]\n{title}\n\n[본문]\n{body}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 뉴스 기사 전문 요약 봇이야. 항상 객관적이고 간결하게 한국어로 요약해."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f">>>> [Python LLM Error] Summary generation failed: {e}")
        return body[:200] + "... (요약 실패)"


def process_single_news(news, request) -> NewsAnalysisResult:
    """뉴스 1개를 처리하는 함수 (병렬 스레드에서 실행됨)"""
    try:
        news_text = safe_join_title_body(news.title, news.body)
        news_embedding = get_news_embedding(news.news_id, news_text, news.embedding)

        # 요약은 잘 나오니까 건너뜁니다
        news_summary = generate_summary_with_llm(news.title, news.body)

        # 🚨 여기서 키워드 추출 시도
        candidates = extract_keywords_from_news(news.title, news.body, request.top_k_keywords)

        # 🔥 [DEBUG] 추출된 후보군이 진짜 들어오는지 확인
        print(f"DEBUG: NewsID {news.news_id} / 후보군 수: {len(candidates)}", file=sys.stderr, flush=True)

        keywords = resolve_keywords(candidates, request.existing_keywords)

        weighted_keywords = []
        for keyword in keywords:
            weight = calculate_news_keyword_weight(
                news.title, news.body, keyword.normalized_word,
                keyword.extraction_score, news_embedding, keyword.embedding
            )
            weighted_keywords.append(
                keyword.model_copy(update={"weight": weight, "evidence_text": keyword.evidence_text})
            )

        return NewsAnalysisResult(
            news_id=news.news_id,
            published_at=news.published_at,
            embedding=news_embedding,
            summary=news_summary,
            keywords=weighted_keywords,
        )

    except Exception as e:
        # 🔥 [강제 출력] 에러가 나면 여기서 무조건 터미널에 찍힙니다!
        import traceback
        print(f"\n🚨 [CRITICAL ERROR] NewsID {news.news_id} 처리 중 사망:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

        # 에러가 나도 배치가 멈추지 않게 빈 결과 반환
        return NewsAnalysisResult(
            news_id=news.news_id,
            published_at=news.published_at,
            embedding=[],
            summary="분석 실패",
            keywords=[]
        )

def analyze_news_batch(request: AnalyzeNewsBatchRequest) -> AnalyzeNewsBatchResponse:
    """Analyze news articles and return embeddings, keywords, and keyword relations."""

    for existing in request.existing_keywords:
        validate_embedding_dim(existing.embedding)

    # 구형 for문 삭제하고, ThreadPoolExecutor로 10개 사용
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        # request.news 리스트를 process_single_news에 매핑하여 병렬 실행 후 리스트로 묶음
        news_results = list(executor.map(lambda n: process_single_news(n, request), request.news))

    for res in news_results:
        for kw in res.keywords:
            print(f"DEBUG: Word={kw.word}, Macro={kw.macro}, Persona={kw.personas}")

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
    settings = get_settings()
    if settings.use_openai_keyword_extraction and settings.openai_api_key and OpenAI is not None:
        try:
            return extract_keywords_from_news_openai(title, body, top_k)
        except Exception as e:
            # 🔥 [수정] flush=True 로 버퍼링 무시하고 강제 출력!
            print(f"\n🚨 [OpenAI 메인 로직 에러] 원인: {e}\n", flush=True)
            import traceback
            traceback.print_exc()
            return extract_keywords_from_news_fallback(title, body, top_k)

    print("\n⚠️ [경고] 설정 문제로 메인 로직을 타지 못하고 Fallback으로 빠졌습니다!\n", flush=True)
    return extract_keywords_from_news_fallback(title, body, top_k)


def extract_keywords_from_news_openai(title: str, body: str, top_k: int) -> list[KeywordCandidate]:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    prompt = (
        f"제목: {title}\n"
        f"내용: {body}\n\n"
        f"위 뉴스에서 핵심 키워드 {top_k}개를 추출하세요."
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 뉴스 분석 전문가입니다. 반드시 아래 JSON 형식으로만 응답하세요. "
                        "다른 텍스트는 절대 포함하지 마세요.\n"
                        '{"keywords": [{"specific": "키워드", "personas": "ECONOMY", "macro": "중분류", "weight": 0.8, "evidence_text": "근거"}]}'
                    )
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_completion_tokens=800,
        )

        content = response.choices[0].message.content or ""
        print(f">>>> [DEBUG] AI 원본 응답: {repr(content)}", file=sys.stderr, flush=True)

        if not content.strip():
            print("🚨 [DEBUG] AI 응답이 빈 문자열!", file=sys.stderr, flush=True)
            return extract_keywords_from_news_fallback(title, body, top_k)

        # json_object 모드면 마크다운 없이 오지만, 혹시 몰라 방어
        clean_content = re.sub(r'^```json\s*|\s*```$', '', content.strip())
        payload = json.loads(clean_content)

        print(f">>>> [DEBUG] 파싱된 payload: {payload}", file=sys.stderr, flush=True)

        raw_keywords = payload.get("keywords") or payload.get("results") or []
        print(f">>>> [DEBUG] raw_keywords 개수: {len(raw_keywords)}", file=sys.stderr, flush=True)

    except json.JSONDecodeError as e:
        print(f"🚨 [JSON 파싱 에러] content: {repr(content)}, 원인: {e}", file=sys.stderr, flush=True)
        return extract_keywords_from_news_fallback(title, body, top_k)
    except Exception as e:
        print(f"🚨 [추출 에러] 원인: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return extract_keywords_from_news_fallback(title, body, top_k)

    candidates: list[KeywordCandidate] = []
    seen: set[str] = set()

    for item in raw_keywords:
        print(f">>>> [DEBUG] 처리 중 item: {item}", file=sys.stderr, flush=True)

        raw_word = item.get("specific") or item.get("word") or item.get("keyword") or ""
        word = clean_text(str(raw_word))

        if not word:
            print(f"🚨 [DEBUG] word 비어있음! item: {item}", file=sys.stderr, flush=True)
            continue

        norm = clean_text(str(item.get("normalized_specific") or item.get("normalized_word") or word))
        normalized_word = normalize_keyword(norm)

        if normalized_word in seen:
            continue
        seen.add(normalized_word)

        personas = str(item.get("personas") or item.get("persona") or "ECONOMY").upper()
        macro = clean_text(str(item.get("macro") or item.get("category") or "기타"))
        score = clip_score(float(item.get("weight") or item.get("extraction_score") or 0.5))

        candidates.append(
            KeywordCandidate(
                word=word,
                normalized_word=normalized_word,
                weight=score,
                extraction_score=score,
                evidence_text=clean_text(str(item.get("evidence_text", ""))) or None,
                personas=personas,
                macro=macro,
            )
        )

        if len(candidates) >= max(1, top_k):
            break

    print(f">>>> [DEBUG] 최종 candidates 개수: {len(candidates)}", file=sys.stderr, flush=True)
    return candidates


# [BUG FIX 3]
def classify_keywords_with_openai(words: list[str]) -> dict[str, dict]:
    """Classify a word list into personas/macro via OpenAI.

    Returns {word: {"personas": ..., "macro": ...}}.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    word_list_str = ", ".join(words)

    # 🔥 [수정 1] OpenAI로 넘기는 단어 목록 확인 (버퍼링 무시)
    print(f"\n🔥 [Fallback 분류 요청 단어들]: {word_list_str}", flush=True)

    prompt = (
        "아래 단어들을 각각 다음 6가지 대분류(personas) 중 하나로 분류하고, 중분류(macro)도 작성하세요: "
        "[POLITICS, ECONOMY, TECHNOLOGY, SOCIETY, CULTURE, INTERNATIONAL]. "
        "반드시 아래 형식의 JSON object만 반환하세요. 다른 텍스트는 절대 포함하지 마세요.\n\n"
        f"단어 목록: {word_list_str}\n\n"
        '출력 형식 예시: {{"classifications": [{{"word": "삼성전자", "personas": "ECONOMY", "macro": "반도체"}}]}}'
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You classify Korean words into categories and return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=800,
        )
        content = response.choices[0].message.content or "{}"

        # 🔥 [수정 2] AI가 뱉어낸 분류 결과 강제 출력!
        print(f"\n🔥 [Fallback AI 날것 응답]: {content}\n", flush=True)

        payload = json.loads(content)
        result: dict[str, dict] = {}
        for item in payload.get("classifications", []):
            word = item.get("word", "")
            if word:
                result[word] = {
                    "personas": str(item.get("personas", "ECONOMY")).upper(),
                    "macro": str(item.get("macro", "기타")),
                }
        return result

    except Exception as e:
        # 🔥 [수정 3] 여기서 터진 에러를 절대 묻히게 두지 않음!
        print(f"\n🚨 [classify_keywords_with_openai 내부 에러] 원인: {e}\n", flush=True)
        import traceback
        traceback.print_exc()
        raise e  # 에러를 숨기지 않고 바깥으로 던짐


def extract_keywords_from_news_fallback(title: str, body: str, top_k: int) -> list[KeywordCandidate]:
    """Fallback keyword extraction using simple Korean/English token frequency."""

    text = safe_join_title_body(title, body)
    if not text:
        return []

    try:
        candidates = extract_keywords_from_news_keybert(title, body, top_k)
    except Exception:
        candidates = extract_keywords_from_news_frequency(title, body, top_k)

    # [BUG FIX 3] Post-process fallback candidates: attach personas/macro via OpenAI
    settings = get_settings()
    if settings.openai_api_key and OpenAI is not None and candidates:
        try:
            words = [c.word for c in candidates]
            classifications = classify_keywords_with_openai(words)
            for candidate in candidates:
                cls = classifications.get(candidate.word, {})
                candidate.personas = cls.get("personas", "ECONOMY")
                candidate.macro = cls.get("macro", "기타")
            return candidates
        except Exception as e:
            # 🔥 [수정] 여기서 에러를 삼키고(pass) 있었습니다!! 무조건 출력하게 변경!
            print(f"\n🚨 [Fallback OpenAI 분류 실패] 원인: {e}\n", flush=True)
            import traceback
            traceback.print_exc()
            pass
    # [BUG FIX 3] Default when OpenAI unavailable or failed
    for candidate in candidates:
        if candidate.personas is None:
            candidate.personas = "ECONOMY"
        if candidate.macro is None:
            candidate.macro = "기타"
    return candidates


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
                # 🔥 새로 추가된 필드들 매핑!
                personas=candidate.personas,
                macro=candidate.macro
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
    # [BUG FIX 4] Guard against None/empty normalized_word before string comparison
    left_nw = left.normalized_word or ""
    right_nw = right.normalized_word or ""
    return (left, right) if left_nw <= right_nw else (right, left)


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
