from __future__ import annotations

import hashlib
import math
import random
import re
from datetime import datetime, timezone
from typing import Iterable

from app.utils.text_utils import clean_text, truncate_text
from app.utils.vector_utils import l2_normalize


DOMAIN_KEYWORDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "POLITICS": ("정치", ("대선", "공약", "국회", "정부", "정책", "선거", "대통령", "여야", "법안")),
    "ECONOMY": (
        "경제",
        ("금리", "기준금리", "물가", "인플레이션", "환율", "부동산", "가계부채", "전세사기", "한국은행", "주식", "증권", "대출"),
    ),
    "TECHNOLOGY": ("기술", ("AI", "반도체", "HBM", "데이터", "플랫폼", "보안", "블록체인", "배터리", "전기차")),
    "SOCIETY": ("사회", ("의료", "교육", "노동", "복지", "저출생", "인구", "안전", "범죄", "법원")),
    "CULTURE": ("문화", ("영화", "드라마", "음악", "공연", "전시", "웹툰", "게임", "콘텐츠", "K팝")),
    "INTERNATIONAL": ("국제", ("미국", "중국", "일본", "유럽", "러시아", "우크라이나", "외교", "관세", "무역")),
}

MACRO_KEYWORDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "POLITICS",
        "선거",
        ("대선", "총선", "지방선거", "선거", "공약", "후보", "유권자", "투표", "경선"),
    ),
    (
        "POLITICS",
        "입법",
        ("국회", "법안", "상임위", "본회의", "의원", "여야", "정당", "개정안", "청문회"),
    ),
    (
        "POLITICS",
        "정부정책",
        ("정부", "정책", "대통령", "장관", "국정", "예산", "규제", "개혁", "행정"),
    ),
    (
        "ECONOMY",
        "금융",
        ("금리", "기준금리", "환율", "은행", "대출", "채권", "증권", "주식", "코스피", "코스닥"),
    ),
    (
        "ECONOMY",
        "물가",
        ("물가", "인플레이션", "소비자물가", "가격", "원자재", "유가", "생활비"),
    ),
    (
        "ECONOMY",
        "부동산",
        ("부동산", "아파트", "전세", "전세사기", "주택", "분양", "임대", "집값"),
    ),
    (
        "ECONOMY",
        "산업",
        ("기업", "실적", "수출", "무역", "투자", "제조", "공급망", "매출", "시장"),
    ),
    (
        "TECHNOLOGY",
        "AI",
        ("AI", "인공지능", "챗GPT", "생성형", "모델", "데이터센터", "GPU", "엔비디아"),
    ),
    (
        "TECHNOLOGY",
        "반도체",
        ("반도체", "HBM", "메모리", "파운드리", "삼성전자", "SK하이닉스", "칩", "양산"),
    ),
    (
        "TECHNOLOGY",
        "모빌리티",
        ("배터리", "전기차", "자율주행", "로봇", "자동차", "충전", "모빌리티"),
    ),
    (
        "TECHNOLOGY",
        "플랫폼",
        ("플랫폼", "클라우드", "데이터", "보안", "블록체인", "소프트웨어", "서비스"),
    ),
    (
        "SOCIETY",
        "의료",
        ("의료", "병원", "의대", "의사", "환자", "간호", "건강", "백신"),
    ),
    (
        "SOCIETY",
        "교육",
        ("교육", "입시", "학교", "대학", "수능", "교사", "학생", "학원"),
    ),
    (
        "SOCIETY",
        "노동복지",
        ("노동", "고용", "임금", "복지", "저출생", "인구", "연금", "일자리"),
    ),
    (
        "SOCIETY",
        "사건사고",
        ("안전", "범죄", "경찰", "검찰", "법원", "사고", "재난", "수사"),
    ),
    (
        "CULTURE",
        "콘텐츠",
        ("영화", "드라마", "음악", "웹툰", "게임", "콘텐츠", "OTT", "K팝", "아이돌"),
    ),
    (
        "CULTURE",
        "공연전시",
        ("공연", "전시", "축제", "미술", "박물관", "콘서트", "뮤지컬"),
    ),
    (
        "CULTURE",
        "여가",
        ("여행", "관광", "스포츠", "야구", "축구", "농구", "올림픽", "월드컵"),
    ),
    (
        "INTERNATIONAL",
        "미국",
        ("미국", "워싱턴", "연준", "트럼프", "바이든", "뉴욕", "나스닥"),
    ),
    (
        "INTERNATIONAL",
        "중국",
        ("중국", "베이징", "시진핑", "홍콩", "대만", "위안", "알리바바"),
    ),
    (
        "INTERNATIONAL",
        "외교안보",
        ("일본", "유럽", "러시아", "우크라이나", "중동", "외교", "관세", "무역분쟁", "정상회담"),
    ),
)

SUMMARY_TEMPLATES = {
    "금리": "금리는 돈을 빌릴 때 붙는 비용으로, 물가, 환율, 부동산, 가계부채와 강하게 연결됩니다.",
    "기준금리": "기준금리는 중앙은행이 정하는 대표 금리로, 시장 금리와 대출 이자에 영향을 줍니다.",
    "부동산": "부동산은 금리와 대출 부담의 영향을 크게 받으며, 주거 안정과 자산 시장의 핵심 이슈입니다.",
    "환율": "환율은 국가 간 돈의 교환 비율로, 금리 차이와 수출입 기업 실적에 영향을 줍니다.",
    "전세사기": "전세사기는 임대차 시장의 정보 비대칭과 보증금 보호 문제와 연결됩니다.",
    "AI": "AI는 산업 생산성, 반도체 수요, 데이터 정책과 연결되는 핵심 기술 이슈입니다.",
    "반도체": "반도체는 AI, 스마트폰, 자동차, 데이터센터 등 첨단 산업의 핵심 부품입니다.",
    "대선": "대선은 정책 방향과 경제, 사회 이슈의 우선순위를 결정하는 중요한 정치 이벤트입니다.",
    "공약": "공약은 후보나 정당이 제시하는 정책 약속으로, 유권자가 정책 방향을 비교하는 기준이 됩니다.",
    "국회": "국회는 법안과 예산을 심의하며, 정책이 실제 제도로 이어지는 핵심 기관입니다.",
}


def log_openai_fallback(message: str) -> None:
    print(f"[OPENAI_FALLBACK] {message}", flush=True)


def text_hash(*parts: object, length: int = 16) -> str:
    joined = "|".join(clean_text(str(part)) for part in parts if part is not None)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def normalize_keyword_text(text: str | None) -> str:
    value = clean_text(text).lower()
    value = re.sub(r"[^\w가-힣\s.-]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def deterministic_fake_embedding(text: str, dim: int) -> list[float]:
    """Generate a stable normalized pseudo embedding for the same input text."""

    seed = int(hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    normalized = l2_normalize(values)
    if not any(math.isfinite(value) for value in normalized):
        return [0.0 for _ in range(dim)]
    return normalized


def infer_persona_and_macro(text: str) -> tuple[str, str]:
    normalized = normalize_keyword_text(text).upper()
    for persona, macro, terms in MACRO_KEYWORDS:
        if any(term.upper() in normalized for term in terms):
            return persona, macro
    for persona, (macro, terms) in DOMAIN_KEYWORDS.items():
        if any(term.upper() in normalized for term in terms):
            return persona, macro
    if re.search(r"[A-Za-z]", normalized):
        return "TECHNOLOGY", "플랫폼"
    return "SOCIETY", "일반"


def template_summary_for_keyword(
    keyword: str,
    related_news_titles: Iterable[str] = (),
    related_keywords: Iterable[str] = (),
    category: str | None = None,
) -> str:
    word = clean_text(keyword) or "이 키워드"
    exact = SUMMARY_TEMPLATES.get(word)
    if exact:
        return exact

    titles = [clean_text(title) for title in related_news_titles if clean_text(title)]
    related = [clean_text(item) for item in related_keywords if clean_text(item)]
    if titles:
        summary = f"{word}은(는) '{titles[0]}' 등 관련 뉴스에서 반복적으로 등장한 주제입니다."
    else:
        summary = f"{word}은(는) 최근 뉴스 흐름 속에서 다른 개념과 연결해 이해할 필요가 있는 주제입니다."
    if related:
        summary += f" 함께 살펴볼 키워드는 {', '.join(related[:3])}입니다."
    if category:
        summary += f" {clean_text(category)} 관점에서 관련 흐름을 확인하면 이해가 쉽습니다."
    return summary


def template_news_summary(title: str, body: str, max_chars: int = 220) -> str:
    clean_title = clean_text(title) or "관련 뉴스"
    snippet = truncate_text(body, max_chars)
    if snippet:
        return f"{clean_title}: {snippet}"
    return f"{clean_title}의 핵심 내용을 바탕으로 관련 키워드와 뉴스 흐름을 이해할 수 있습니다."


def fallback_cache_metadata() -> dict[str, str]:
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "source": "fallback"}
