import json
import re

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAI = None

from app.config import get_settings
from app.schemas import GenerateQuizzesRequest, GenerateQuizzesResponse, QuizInfo, QuizRelatedNewsInput
from app.utils.text_utils import clean_text, truncate_text

MAX_OPTION_CHARS = 50
GENERIC_QUESTION_PATTERNS = (
    "핵심적으로 다룬 내용",
    "가장 적절한 것은",
    "근거로 삼아야 할 자료",
    "무엇과 관련이 있나요",
)
BANNED_QUIZ_PATTERNS = (
    "스포츠 경기 결과",
    "해외 축제 일정",
    "연예계 소식",
    "광고 문구",
    "개인 의견",
    "날씨 정보",
    "날씨 예보",
    "로그인",
    "회원가입",
    "메뉴",
)
QUIZ_STOPWORDS = {
    "뉴스",
    "기사",
    "관련",
    "이번",
    "내용",
    "요약",
    "근거",
    "설명",
    "확인",
    "있습니다",
    "했습니다",
    "합니다",
    "대한",
    "통해",
}
PLAUSIBLE_DISTRACTORS = {
    "금리": ("모든 대출 조건이 고정되기 때문", "물가 영향 없이 소비가 늘기 때문", "주택 수요가 자동으로 같아지기 때문"),
    "부동산": ("공급과 수요가 항상 같기 때문", "대출 조건과 무관하게 거래되기 때문", "세금만으로 가격이 결정되기 때문"),
    "AI": ("데이터와 연산 자원이 필요 없기 때문", "산업 생산성과 무관하게 쓰이기 때문", "반도체 수요와 연결되지 않기 때문"),
    "반도체": ("소비 심리만으로 생산량이 정해지기 때문", "공급망 변화와 무관하기 때문", "데이터 수요와 연결되지 않기 때문"),
    "정책": ("시장 반응과 무관하게 효과가 확정되기 때문", "이해관계 조정이 필요 없기 때문", "법과 예산 절차를 거치지 않기 때문"),
    "default": ("단일 요인만으로 이슈가 결정되기 때문", "관련 이해관계가 모두 같기 때문", "본문 근거와 다른 인과관계를 전제하기 때문"),
}


def generate_quizzes(request: GenerateQuizzesRequest) -> GenerateQuizzesResponse:
    """Generate stable news-grounded quizzes for the backend batch pipeline."""

    settings = get_settings()
    if settings.openai_api_key and OpenAI is not None:
        try:
            quizzes = generate_openai_quizzes(request)
            if quizzes:
                return GenerateQuizzesResponse(keyword_id=request.keyword_id, quizzes=quizzes)
        except Exception:
            pass

    return GenerateQuizzesResponse(
        keyword_id=request.keyword_id,
        quizzes=generate_fallback_quizzes(request),
    )


def generate_openai_quizzes(request: GenerateQuizzesRequest) -> list[QuizInfo]:
    """Generate quizzes with OpenAI and validate the JSON contract strictly."""

    settings = get_settings()
    if OpenAI is None:
        return []

    client = OpenAI(api_key=settings.openai_api_key)
    evidence = _format_news_evidence(request.related_news)
    target_count = _target_count(request.num_questions)
    prompt = (
        "다음 뉴스 근거만 사용해 한국어 4지선다 퀴즈를 생성하세요.\n"
        "퀴즈는 사용자가 키워드의 맥락, 관련 개념, 원인과 결과를 이해했는지 확인해야 합니다.\n"
        "질문은 구체적인 1문장이어야 하며, '핵심적으로 다룬 내용은?', '가장 적절한 것은?' 같은 범용 질문은 금지합니다.\n"
        "오답은 완전히 무관한 스포츠/축제/연예/광고 문구가 아니라, 같은 주제 안에서 헷갈릴 수 있는 선택지로 작성하세요.\n"
        "기사 제목이나 본문을 그대로 긴 선택지로 복사하지 마세요. 각 option은 가능하면 35자 이내, 최대 50자 이내로 작성하세요.\n"
        "날짜, 기자명, 출처명, 로그인/메뉴 문구만 묻는 문제는 만들지 마세요.\n"
        "반드시 JSON 객체만 반환하세요.\n\n"
        f"키워드: {request.word}\n"
        f"키워드 요약: {clean_text(request.summary or '')}\n"
        f"문항 수: {target_count}\n"
        "문제 유형: 핵심 내용 이해, 원인과 결과, 키워드 간 관계, 기사 근거, 개념 구분, 다음 이슈 흐름.\n"
        "JSON 형식:\n"
        '{"quizzes":[{"question":"...","options":["...","...","...","..."],'
        '"answer_index":0,"explanation":"...","source_news_ids":[1]}]}\n\n'
        f"뉴스 근거:\n{evidence}"
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 뉴스 기반 학습 앱의 퀴즈 출제자입니다. "
                    "퀴즈는 사용자가 뉴스 키워드의 맥락과 관련 개념을 이해했는지 확인해야 합니다. "
                    "반드시 제공된 뉴스 title/body/summary에 근거해서만 출제하세요. "
                    "문제는 구체적이고 자연스러운 한국어 문장이어야 합니다. "
                    "정답은 하나만 명확해야 합니다. "
                    "오답은 기사와 완전히 무관한 내용이 아니라 같은 주제 안에서 헷갈릴 수 있는 선택지여야 합니다. "
                    "너무 쉬운 오답을 만들지 말고, 기사 제목이나 본문을 그대로 긴 선택지로 복사하지 마세요. "
                    "광고 문구, 메뉴 텍스트, 로그인 문구, 기자명, 날짜, 출처명만 묻는 문제는 만들지 마세요. "
                    "각 선택지는 짧고 균형 있게 작성하세요. JSON object만 반환하세요."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=900,
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    raw_quizzes = payload.get("quizzes", [])
    valid_news_ids = {news.news_id for news in request.related_news}
    quizzes: list[QuizInfo] = []

    for index, item in enumerate(raw_quizzes):
        quiz = _coerce_quiz_info(item, valid_news_ids, request, index)
        if quiz is not None:
            quizzes.append(quiz)
        if len(quizzes) >= target_count:
            break

    if len(quizzes) < target_count:
        quizzes.extend(generate_fallback_quizzes(request, start_index=len(quizzes)))

    return quizzes[:target_count]


def generate_fallback_quizzes(
    request: GenerateQuizzesRequest,
    start_index: int = 0,
) -> list[QuizInfo]:
    """Create deterministic fallback quizzes when LLM output is unavailable."""

    target_count = _target_count(request.num_questions)
    source_news = request.related_news[0] if request.related_news else None
    source_ids = [news.news_id for news in request.related_news[:3]]
    source_title = clean_text(source_news.title) if source_news else f"{request.word} 관련 뉴스"
    summary = clean_text(request.summary or "")
    body = clean_text(source_news.body) if source_news else ""
    keyword = clean_text(request.word) or "이 키워드"
    key_terms = _extract_key_terms(keyword, summary, source_title, body)
    primary_term = key_terms[0] if key_terms else keyword
    secondary_term = key_terms[1] if len(key_terms) > 1 else keyword
    answer = _build_fallback_answer(keyword, primary_term, secondary_term, summary, body, source_title)
    distractors = _plausible_distractors(keyword, key_terms)

    templates = [
        {
            "question": f"기사에서 '{keyword}'와 '{primary_term}'를 함께 봐야 하는 이유는 무엇인가요?",
            "answer": answer,
            "explanation": f"제공된 뉴스는 '{keyword}'를 '{primary_term}' 맥락과 함께 이해해야 한다는 근거를 줍니다.",
        },
        {
            "question": f"'{keyword}' 이슈를 이해할 때 '{secondary_term}'가 중요한 이유는 무엇인가요?",
            "answer": f"{keyword} 흐름의 배경을 설명하기 때문",
            "explanation": f"'{source_title}'의 내용은 '{secondary_term}'를 통해 '{keyword}'의 맥락을 이해하게 합니다.",
        },
        {
            "question": f"'{keyword}' 관련 흐름에서 다음에 확인해야 할 쟁점은 무엇인가요?",
            "answer": f"{primary_term} 변화가 미치는 영향",
            "explanation": f"뉴스 근거상 '{primary_term}'의 변화는 '{keyword}' 이슈의 후속 흐름을 판단하는 단서입니다.",
        },
    ]

    quizzes = []
    for index in range(start_index, target_count):
        template = templates[index % len(templates)]
        answer_index = _deterministic_answer_index(keyword, index)
        options = _place_answer(template["answer"], distractors, answer_index)
        quizzes.append(
            QuizInfo(
                question=template["question"],
                options=options,
                answer_index=answer_index,
                explanation=template["explanation"],
                source_news_ids=source_ids,
            )
        )
    return quizzes


def _coerce_quiz_info(
    item: object,
    valid_news_ids: set[int],
    request: GenerateQuizzesRequest,
    fallback_index: int,
) -> QuizInfo | None:
    """Validate one LLM quiz object and return a backend-safe QuizInfo."""

    if not isinstance(item, dict):
        return None

    question = clean_text(str(item.get("question", "")))
    if _is_generic_question(question) or _contains_banned_pattern(question):
        return None

    raw_options = item.get("options", [])
    if not isinstance(raw_options, list):
        return None
    options = [_short_option(str(option)) for option in raw_options]
    options = [option for option in options if option]
    options = _dedupe_options(options)

    answer_index = item.get("answer_index")
    if not isinstance(answer_index, int):
        return None

    explanation = clean_text(str(item.get("explanation", "")))
    raw_source_ids = item.get("source_news_ids", [])
    if not isinstance(raw_source_ids, list):
        raw_source_ids = []
    source_ids = [
        int(news_id)
        for news_id in raw_source_ids
        if isinstance(news_id, int) and news_id in valid_news_ids
    ]

    if answer_index < 0 or answer_index > 3:
        return None

    answer_text = options[answer_index] if answer_index < len(options) else ""
    if not answer_text or _contains_banned_pattern(answer_text):
        return None

    options = _repair_options(options, answer_index, request, fallback_index)
    explanation = truncate_text(explanation, 120)

    if not question or len(options) != 4 or not explanation:
        return None

    return QuizInfo(
        question=question,
        options=options,
        answer_index=answer_index,
        explanation=explanation,
        source_news_ids=source_ids,
    )


def _extract_key_terms(keyword: str, summary: str, title: str, body: str) -> list[str]:
    text = " ".join([keyword, summary, title, truncate_text(body, 350)])
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.-]*|[가-힣]{2,}", text)
    counts: dict[str, int] = {}
    for token in tokens:
        word = clean_text(token)
        normalized = word.lower()
        if len(normalized) < 2 or normalized in QUIZ_STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [truncate_text(word, 14) for word, _ in ranked[:3]] or [keyword]


def _build_fallback_answer(keyword: str, primary_term: str, secondary_term: str, summary: str, body: str, title: str) -> str:
    text = f"{keyword} {primary_term} {secondary_term} {summary} {body} {title}".lower()
    if any(term in text for term in ("금리", "대출", "환율", "은행", "부동산")):
        return "금융 조건과 부담이 달라지기 때문"
    if any(term in text for term in ("ai", "반도체", "hbm", "데이터", "gpu")):
        return "기술 수요와 공급망이 연결되기 때문"
    if any(term in text for term in ("정부", "국회", "정책", "법안", "규제")):
        return "정책 결정이 생활과 시장에 영향을 주기 때문"
    return f"{primary_term}와 {secondary_term}의 맥락이 연결되기 때문"


def _plausible_distractors(keyword: str, key_terms: list[str]) -> list[str]:
    text = " ".join([keyword, *key_terms]).upper()
    for marker, options in PLAUSIBLE_DISTRACTORS.items():
        if marker.upper() in text:
            return list(options)
    return list(PLAUSIBLE_DISTRACTORS["default"])


def _deterministic_answer_index(keyword: str, index: int) -> int:
    return sum(ord(char) for char in f"{keyword}:{index}") % 4


def _place_answer(answer: str, distractors: list[str], answer_index: int) -> list[str]:
    clean_answer = _short_option(answer)
    clean_distractors = _dedupe_options([_short_option(option) for option in distractors])
    while len(clean_distractors) < 3:
        clean_distractors.append(_short_option(PLAUSIBLE_DISTRACTORS["default"][len(clean_distractors) % 3]))
    options = clean_distractors[:3]
    options.insert(answer_index, clean_answer)
    return options[:4]


def _short_option(option: str) -> str:
    value = clean_text(option)
    value = re.sub(r"^[0-9]+[.)]\s*", "", value)
    return truncate_text(value, MAX_OPTION_CHARS)


def _dedupe_options(options: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for option in options:
        normalized = option.lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(option)
    return result


def _repair_options(
    options: list[str],
    answer_index: int,
    request: GenerateQuizzesRequest,
    fallback_index: int,
) -> list[str]:
    if len(options) < 4:
        return []
    answer = options[answer_index]
    source_news = request.related_news[0] if request.related_news else None
    key_terms = _extract_key_terms(
        clean_text(request.word),
        clean_text(request.summary or ""),
        clean_text(source_news.title) if source_news else "",
        clean_text(source_news.body) if source_news else "",
    )
    replacements = _plausible_distractors(request.word, key_terms)
    repaired: list[str] = []
    for index, option in enumerate(options[:4]):
        if index == answer_index:
            repaired.append(_short_option(answer))
            continue
        candidate = _short_option(option)
        if _contains_banned_pattern(candidate) or candidate.lower() in {existing.lower() for existing in repaired}:
            candidate = _short_option(replacements[(fallback_index + index) % len(replacements)])
        repaired.append(candidate)
    repaired = _dedupe_options(repaired)
    if len(repaired) != 4:
        return _place_answer(answer, replacements, answer_index)
    return repaired


def _is_generic_question(question: str) -> bool:
    value = clean_text(question)
    return len(value) < 14 or any(pattern in value for pattern in GENERIC_QUESTION_PATTERNS)


def _contains_banned_pattern(text: str) -> bool:
    return any(pattern in clean_text(text) for pattern in BANNED_QUIZ_PATTERNS)


def _format_news_evidence(related_news: list[QuizRelatedNewsInput]) -> str:
    """Format compact evidence for the quiz prompt."""

    if not related_news:
        return "제공된 관련 뉴스가 없습니다."

    lines = []
    for news in related_news[:5]:
        title = truncate_text(news.title, 150)
        body = truncate_text(news.body, 850)
        url = news.url or ""
        lines.append(f"- news_id={news.news_id}, title={title}, url={url}, body={body}")
    return "\n".join(lines)


def _target_count(num_questions: int) -> int:
    """Clamp quiz count to keep the batch predictable."""

    return max(1, min(num_questions or 3, 5))
