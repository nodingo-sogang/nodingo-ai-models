from datetime import date
from typing import Any

from app.schemas import CandidateKeywordInput, RecommendKeywordResult
from app.services.embedding_service import validate_embedding_dim
from app.utils.vector_utils import clip_score, cosine_similarity

PERSONA_HINT_FIELDS = (
    "persona",
    "personas",
    "user_persona",
    "category",
    "main_category",
    "news_category",
    "topic",
    "domain",
)

PERSONA_ALIASES = {
    "POLITICS": "POLITICS",
    "정치": "POLITICS",
    "ECONOMY": "ECONOMY",
    "경제": "ECONOMY",
    "TECH": "TECH",
    "TECHNOLOGY": "TECH",
    "기술": "TECH",
    "IT": "TECH",
    "SOCIETY": "SOCIETY",
    "사회": "SOCIETY",
    "CULTURE": "CULTURE",
    "문화": "CULTURE",
    "SPORTS": "SPORTS",
    "스포츠": "SPORTS",
    "GLOBAL": "GLOBAL",
    "WORLD": "GLOBAL",
    "국제": "GLOBAL",
}

PERSONA_KEYWORD_TERMS = {
    "POLITICS": (
        "대선",
        "국회",
        "정부",
        "선거",
        "공약",
        "대통령",
        "정당",
        "국정",
        "여당",
        "야당",
        "의원",
        "장관",
        "총리",
        "정책",
        "개헌",
        "탄핵",
        "청와대",
        "국무회의",
        "지방선거",
    ),
    "ECONOMY": (
        "삼성전자",
        "금리",
        "환율",
        "증시",
        "주가",
        "코스피",
        "코스닥",
        "물가",
        "부동산",
        "아파트",
        "수출",
        "무역",
        "반도체",
        "기업",
        "은행",
        "투자",
        "경제",
        "금융",
    ),
    "TECH": (
        "HBM",
        "AI",
        "인공지능",
        "반도체",
        "클라우드",
        "로봇",
        "배터리",
        "전기차",
        "자율주행",
        "소프트웨어",
        "플랫폼",
        "데이터센터",
        "챗GPT",
        "GPU",
        "엔비디아",
        "기술",
    ),
    "SOCIETY": (
        "교육",
        "입시",
        "의대",
        "병원",
        "의료",
        "노동",
        "고용",
        "사건",
        "사고",
        "경찰",
        "검찰",
        "법원",
        "복지",
        "저출산",
        "인구",
        "사회",
    ),
    "CULTURE": (
        "영화",
        "드라마",
        "음악",
        "공연",
        "전시",
        "웹툰",
        "게임",
        "여행",
        "축제",
        "콘텐츠",
        "K팝",
        "아이돌",
        "문화",
    ),
    "SPORTS": (
        "야구",
        "축구",
        "농구",
        "배구",
        "골프",
        "올림픽",
        "월드컵",
        "대표팀",
        "선수",
        "감독",
        "리그",
        "경기",
        "스포츠",
    ),
    "GLOBAL": (
        "미국",
        "중국",
        "일본",
        "유럽",
        "러시아",
        "우크라이나",
        "이스라엘",
        "중동",
        "외교",
        "정상회담",
        "국제",
        "해외",
        "글로벌",
        "무역분쟁",
    ),
}


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
    raw_payload: dict[str, Any] | None = None,
) -> list[RecommendKeywordResult]:
    """Rank candidate keywords and return top_k recommendation rows."""

    validate_embedding_dim(user_embedding)
    scored_items: list[tuple[RecommendKeywordResult, float, str | None]] = []
    target_persona = _extract_request_persona(raw_payload)
    candidate_personas = _extract_candidate_personas(raw_payload)

    for candidate in candidate_keywords:
        validate_embedding_dim(candidate.embedding)
        base_score = calculate_recommend_keyword_score(
            user_embedding,
            candidate.embedding,
            candidate.recent_importance,
            candidate.is_user_interest,
        )
        candidate_persona = candidate_personas.get(candidate.keyword_id)
        if candidate_persona is None:
            candidate_persona = _infer_persona_from_keyword(candidate.word, candidate.normalized_word)
        final_score = _apply_persona_boost(base_score, target_persona, candidate_persona)
        scored_items.append(
            (
                RecommendKeywordResult(
                    user_id=user_id,
                    keyword_id=candidate.keyword_id,
                    target_date=target_date,
                    score=final_score,
                    summary=None,
                ),
                base_score,
                candidate_persona,
            )
        )

    limit = max(0, top_k)
    if limit == 0:
        return []

    ranked = sorted(scored_items, key=lambda item: (-item[0].score, item[0].keyword_id))
    if target_persona is None:
        return [item[0] for item in ranked[:limit]]

    persona_matches = [item for item in ranked if item[2] == target_persona]
    if len(persona_matches) >= limit:
        return [item[0] for item in persona_matches[:limit]]

    if not persona_matches:
        fallback_ranked = sorted(scored_items, key=lambda item: (-item[1], item[0].keyword_id))
        return [item[0] for item in fallback_ranked[:limit]]

    selected_ids = {item[0].keyword_id for item in persona_matches}
    fallback_ranked = sorted(
        (item for item in scored_items if item[0].keyword_id not in selected_ids),
        key=lambda item: (-item[1], item[0].keyword_id),
    )
    selected = persona_matches + fallback_ranked[: limit - len(persona_matches)]
    return [item[0] for item in selected]


def _extract_request_persona(raw_payload: dict[str, Any] | None) -> str | None:
    """Read persona hints from the raw request without changing the public schema."""

    if not raw_payload:
        return None
    for field in PERSONA_HINT_FIELDS:
        persona = _normalize_persona(raw_payload.get(field))
        if persona is not None:
            return persona
    return None


def _extract_candidate_personas(raw_payload: dict[str, Any] | None) -> dict[int, str]:
    """Map candidate keyword ids to persona hints already present in the request body."""

    if not raw_payload:
        return {}
    raw_candidates = raw_payload.get("candidate_keywords")
    if not isinstance(raw_candidates, list):
        return {}

    personas: dict[int, str] = {}
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        keyword_id = item.get("keyword_id")
        if not isinstance(keyword_id, int):
            continue
        for field in PERSONA_HINT_FIELDS:
            persona = _normalize_persona(item.get(field))
            if persona is not None:
                personas[keyword_id] = persona
                break
    return personas


def _normalize_persona(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            persona = _normalize_persona(item)
            if persona is not None:
                return persona
        return None

    text = str(value).strip().upper()
    if not text:
        return None
    return PERSONA_ALIASES.get(text)


def _infer_persona_from_keyword(word: str, normalized_word: str | None) -> str | None:
    text = f"{word} {normalized_word or ''}"
    upper_text = text.upper()
    for persona, terms in PERSONA_KEYWORD_TERMS.items():
        if any(term.upper() in upper_text for term in terms):
            return persona
    return None


def _apply_persona_boost(base_score: float, target_persona: str | None, candidate_persona: str | None) -> float:
    if target_persona is not None and candidate_persona == target_persona:
        return clip_score(base_score + 0.35)
    return base_score
