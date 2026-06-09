import json
import re

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAI = None

from app.config import get_settings
from app.schemas import GenerateQuizzesRequest, GenerateQuizzesResponse, QuizInfo, QuizRelatedNewsInput
from app.services.cache_service import get_cache_value, set_cache_values
from app.services.fallback_service import log_openai_fallback, normalize_keyword_text, text_hash
from app.utils.text_utils import clean_text, truncate_text

QUIZ_CACHE_VERSION = "quality-v2"
MAX_OPTION_CHARS = 50
GENERIC_QUESTION_PATTERNS = (
    "핵심적으로 다룬 내용",
    "가장 적절한 것은",
    "근거로 삼아야 할 자료",
    "무엇과 관련",
    "관련이 있나요",
)
BANNED_QUIZ_PATTERNS = (
    "스포츠 경기 결과",
    "해외 축제 일정",
    "연예계 소식",
    "광고 문구",
    "날씨 예보",
    "로그인",
    "회원가입",
    "메뉴",
    "기자",
    "출처 없는",
    "기사와 무관한",
)
STOPWORDS = {
    "관련",
    "기사",
    "뉴스",
    "이번",
    "내용",
    "제공",
    "근거",
    "설명",
    "확인",
    "있습니다",
    "했습니다",
    "합니다",
    "대한",
    "통해",
    "위해",
}
PLAUSIBLE_DISTRACTORS = {
    "금리": ("물가가 즉시 사라지기 때문", "모든 대출 조건이 고정되기 때문", "주택 수요가 자동으로 동일해지기 때문"),
    "부동산": ("세금 구조만으로 가격이 결정되기 때문", "공급과 수요가 항상 같기 때문", "대출 조건과 무관하게 거래가 늘기 때문"),
    "반도체": ("소비자 물가만으로 수요가 결정되기 때문", "제조 공정과 공급망 영향이 없기 때문", "데이터 수요와 무관하게 생산되기 때문"),
    "AI": ("데이터와 연산 자원이 필요 없기 때문", "산업 생산성과 무관하게 쓰이기 때문", "반도체 수요와 연결되지 않기 때문"),
    "정책": ("시장 반응과 무관하게 효과가 확정되기 때문", "이해관계 조정이 필요 없기 때문", "법과 예산 절차를 거치지 않기 때문"),
    "default": ("단일 요인만으로 이슈가 결정되기 때문", "관련 이해관계가 모두 같기 때문", "본문 근거와 다른 인과관계를 전제하기 때문"),
}


def generate_quizzes(request: GenerateQuizzesRequest) -> GenerateQuizzesResponse:
    """Generate stable news-grounded quizzes for the backend batch pipeline."""

    settings = get_settings()
    cache_keys = _quiz_cache_keys(request)
    cache_path = getattr(settings, "quiz_cache_path", "test2/cache/quiz_cache.json")
    cache_enabled = bool(getattr(settings, "enable_quiz_cache", False))
    if cache_enabled:
        cached = get_cache_value(cache_path, cache_keys)
        cached_quizzes = _coerce_cached_quizzes(cached)
        if cached_quizzes:
            return GenerateQuizzesResponse(keyword_id=request.keyword_id, quizzes=cached_quizzes)

    if getattr(settings, "openai_cost_saver_mode", False):
        quizzes = generate_fallback_quizzes(request)
        if cache_enabled:
            _store_quiz_cache(cache_path, cache_keys, quizzes)
        return GenerateQuizzesResponse(keyword_id=request.keyword_id, quizzes=quizzes)

    if settings.openai_api_key and OpenAI is not None:
        try:
            quizzes = generate_openai_quizzes(request)
            if quizzes:
                if cache_enabled:
                    _store_quiz_cache(cache_path, cache_keys, quizzes)
                return GenerateQuizzesResponse(keyword_id=request.keyword_id, quizzes=quizzes)
        except Exception as exc:
            log_openai_fallback(f"quiz generation failed, using template quiz: {exc}")

    quizzes = generate_fallback_quizzes(request)
    if cache_enabled:
        _store_quiz_cache(cache_path, cache_keys, quizzes)
    return GenerateQuizzesResponse(
        keyword_id=request.keyword_id,
        quizzes=quizzes,
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
        "목표는 사용자가 키워드의 맥락, 관련 개념, 원인과 결과를 이해했는지 확인하는 것입니다.\n"
        "질문은 구체적인 1문장으로 작성하고, '핵심적으로 다룬 내용은?', '가장 적절한 것은?'처럼 범용적인 문장은 금지합니다.\n"
        "정답은 하나만 명확해야 하며, 오답은 같은 주제 안에서 헷갈릴 수 있지만 뉴스 근거와 다른 선택지로 만드세요.\n"
        "스포츠 경기 결과, 해외 축제 일정, 연예계 소식, 광고 문구, 날씨 예보, 로그인/회원가입/메뉴 문구 같은 쉬운 오답은 금지합니다.\n"
        "기사 제목이나 본문 문단을 그대로 긴 선택지로 복사하지 마세요. 각 option은 가능하면 35자 이내, 최대 50자 이내로 작성하세요.\n"
        "날짜, 기자명, 출처명만 묻는 문제는 만들지 마세요.\n"
        "반드시 JSON 객체만 반환하세요.\n\n"
        f"키워드: {request.word}\n"
        f"키워드 요약: {clean_text(request.summary or '')}\n"
        f"문항 수: {target_count}\n"
        "문제 유형 예시: 핵심 개념 이해, 원인과 결과, 키워드 간 관계, 기사 근거 해석, 다음 이슈 흐름.\n"
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
                    "너무 쉬운 오답, 광고 문구, 메뉴 텍스트, 로그인 문구, 기자명, 날짜, 출처명만 묻는 문제는 금지합니다. "
                    "기사 제목이나 본문을 그대로 긴 선택지로 복사하지 마세요. "
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


def _quiz_cache_keys(request: GenerateQuizzesRequest) -> list[str]:
    normalized = normalize_keyword_text(request.word)
    related_ids = ",".join(str(news.news_id) for news in request.related_news)
    body_digest = text_hash(request.summary or "", *(news.body for news in request.related_news))
    count = _target_count(request.num_questions)
    return [
        f"quiz:{QUIZ_CACHE_VERSION}:exact:{request.keyword_id}:{related_ids}:{body_digest}:{count}",
        f"quiz:{QUIZ_CACHE_VERSION}:keyword:{request.keyword_id}:{count}",
        f"quiz:{QUIZ_CACHE_VERSION}:word:{normalized}:{count}",
    ]


def _store_quiz_cache(path: str, keys: list[str], quizzes: list[QuizInfo]) -> None:
    if not quizzes:
        return
    payload = [quiz.model_dump() for quiz in quizzes]
    set_cache_values(path, {key: payload for key in keys})


def _coerce_cached_quizzes(cached: object) -> list[QuizInfo]:
    if not isinstance(cached, list):
        return []
    quizzes: list[QuizInfo] = []
    for item in cached:
        try:
            quizzes.append(QuizInfo.model_validate(item))
        except Exception:
            continue
    return quizzes


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
    evidence_answer = _build_evidence_answer(keyword, primary_term, secondary_term, summary, body, source_title)
    distractors = _build_plausible_distractors(keyword, key_terms)

    templates = [
        {
            "question": f"'{keyword}' 이슈에서 '{primary_term}'을 함께 봐야 하는 이유는 무엇인가요?",
            "answer": evidence_answer,
            "explanation": f"제공된 뉴스는 '{keyword}'를 '{primary_term}'와 연결해 이해해야 한다는 근거를 제시합니다.",
        },
        {
            "question": f"기사 맥락에서 '{keyword}'와 '{secondary_term}'의 관계로 가장 알맞은 설명은 무엇인가요?",
            "answer": f"{keyword} 흐름을 이해하는 핵심 맥락",
            "explanation": f"'{source_title}'의 내용은 '{keyword}'를 단독 사실보다 관련 맥락 속에서 해석하게 합니다.",
        },
        {
            "question": f"'{keyword}' 관련 흐름을 다음에 볼 때 확인할 쟁점은 무엇인가요?",
            "answer": f"{primary_term} 변화가 미치는 영향",
            "explanation": f"뉴스 근거상 '{primary_term}'의 변화는 '{keyword}' 이슈의 후속 흐름을 판단하는 단서입니다.",
        },
        {
            "question": f"'{keyword}'를 단순 사실이 아니라 개념으로 이해하려면 무엇을 구분해야 하나요?",
            "answer": f"{primary_term}와 {secondary_term}의 연결",
            "explanation": f"관련 뉴스는 '{primary_term}'와 '{secondary_term}'를 함께 제시해 개념 간 연결을 보여줍니다.",
        },
    ]

    quizzes = []
    for index in range(start_index, target_count):
        template = templates[index % len(templates)]
        answer_index = _deterministic_answer_index(keyword, index)
        options = _place_answer(
            _shorten_option(template["answer"]),
            [_shorten_option(option) for option in distractors],
            answer_index,
        )
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
    options = [_shorten_option(str(option)) for option in raw_options]
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
    if _contains_banned_pattern(answer_text) or len(answer_text) > MAX_OPTION_CHARS:
        return None

    options = _repair_options(options, answer_index, request, fallback_index)
    explanation = _shorten_explanation(explanation)

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
    """Extract compact domain terms for fallback quiz templates."""

    text = " ".join([keyword, summary, title, truncate_text(body, 350)])
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.-]*|[가-힣]{2,}", text)
    counts: dict[str, int] = {}
    for token in tokens:
        word = clean_text(token)
        normalized = normalize_keyword_text(word)
        if len(normalized) < 2 or normalized in STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    terms: list[str] = []
    for word, _ in ranked:
        if word not in terms:
            terms.append(_shorten_term(word))
        if len(terms) >= 4:
            break
    return terms or [keyword]


def _shorten_term(term: str) -> str:
    return truncate_text(clean_text(term), 14)


def _build_evidence_answer(keyword: str, primary_term: str, secondary_term: str, summary: str, body: str, title: str) -> str:
    lower_text = f"{summary} {body} {title}".lower()
    if any(term in lower_text for term in ("대출", "금리", "환율", "은행", "부동산")):
        return "금융 조건과 부담이 달라지기 때문"
    if any(term.lower() in lower_text for term in ("ai", "반도체", "hbm", "데이터", "gpu")):
        return "기술 수요와 공급망이 연결되기 때문"
    if any(term in lower_text for term in ("정부", "국회", "정책", "법안", "규제")):
        return "정책 결정이 시장과 생활에 영향을 주기 때문"
    if any(term in lower_text for term in ("의료", "교육", "복지", "노동", "안전")):
        return "사회 제도와 이용자 부담이 함께 변하기 때문"
    if any(term in lower_text for term in ("미국", "중국", "외교", "관세", "무역")):
        return "국제 관계가 국내 흐름에 영향을 주기 때문"
    return f"{primary_term}와 {secondary_term}의 맥락이 연결되기 때문"


def _build_plausible_distractors(keyword: str, key_terms: list[str]) -> list[str]:
    """Build plausible but incorrect distractors in the same broad topic."""

    text = " ".join([keyword, *key_terms]).upper()
    for marker, options in PLAUSIBLE_DISTRACTORS.items():
        if marker.upper() in text:
            return list(options)
    return list(PLAUSIBLE_DISTRACTORS["default"])


def _deterministic_answer_index(keyword: str, index: int) -> int:
    digest = text_hash(keyword, index, length=8)
    return int(digest, 16) % 4


def _place_answer(answer: str, distractors: list[str], answer_index: int) -> list[str]:
    clean_answer = _shorten_option(answer)
    clean_distractors = _dedupe_options([_shorten_option(option) for option in distractors if option])
    while len(clean_distractors) < 3:
        clean_distractors.append(_shorten_option(PLAUSIBLE_DISTRACTORS["default"][len(clean_distractors) % 3]))
    options = clean_distractors[:3]
    options.insert(answer_index, clean_answer)
    return options[:4]


def _shorten_option(option: str) -> str:
    """Keep options concise and avoid copied article-length text."""

    value = clean_text(option)
    value = re.sub(r"^[0-9]+[.)]\s*", "", value)
    for separator in ("。", ".", " - ", " — ", ": "):
        if len(value) > MAX_OPTION_CHARS and separator in value:
            value = value.split(separator)[0]
            break
    return truncate_text(value, MAX_OPTION_CHARS)


def _shorten_explanation(explanation: str) -> str:
    return truncate_text(clean_text(explanation), 120)


def _dedupe_options(options: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for option in options:
        normalized = normalize_keyword_text(option)
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
    """Replace banned/duplicate distractors while preserving the LLM answer."""

    if len(options) < 4:
        return []
    answer = options[answer_index]
    fallback_terms = _extract_key_terms(
        clean_text(request.word),
        clean_text(request.summary or ""),
        clean_text(request.related_news[0].title) if request.related_news else "",
        clean_text(request.related_news[0].body) if request.related_news else "",
    )
    replacements = _build_plausible_distractors(request.word, fallback_terms)
    repaired: list[str] = []
    replacement_index = fallback_index

    for index, option in enumerate(options[:4]):
        if index == answer_index:
            repaired.append(_shorten_option(answer))
            continue
        candidate = _shorten_option(option)
        if _contains_banned_pattern(candidate) or normalize_keyword_text(candidate) in {
            normalize_keyword_text(item) for item in repaired
        }:
            candidate = _shorten_option(replacements[replacement_index % len(replacements)])
            replacement_index += 1
        repaired.append(candidate)

    repaired = _dedupe_options(repaired)
    if len(repaired) != 4:
        return _place_answer(answer, replacements, answer_index)
    return repaired


def _is_generic_question(question: str) -> bool:
    if len(clean_text(question)) < 14:
        return True
    return any(pattern in question for pattern in GENERIC_QUESTION_PATTERNS)


def _contains_banned_pattern(text: str) -> bool:
    return any(pattern in clean_text(text) for pattern in BANNED_QUIZ_PATTERNS)


def _format_news_evidence(related_news: list[QuizRelatedNewsInput]) -> str:
    """Format compact evidence for the quiz prompt."""

    if not related_news:
        return "제공된 관련 뉴스가 없습니다."

    lines = []
    for news in related_news[:5]:
        title = truncate_text(news.title, 150)
        body = truncate_text(news.body, get_settings().max_openai_body_chars)
        url = news.url or ""
        lines.append(f"- news_id={news.news_id}, title={title}, url={url}, body={body}")
    return "\n".join(lines)


def _target_count(num_questions: int) -> int:
    """Clamp quiz count to keep the batch predictable."""

    return max(1, min(num_questions or 3, 5))
