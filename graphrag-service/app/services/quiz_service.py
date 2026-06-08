import json

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAI = None

from app.config import get_settings
from app.schemas import GenerateQuizzesRequest, GenerateQuizzesResponse, QuizInfo, QuizRelatedNewsInput
from app.services.cache_service import get_cache_value, set_cache_values
from app.services.fallback_service import log_openai_fallback, normalize_keyword_text, text_hash
from app.utils.text_utils import clean_text, truncate_text


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
        "다음 뉴스 근거만 사용해 한국어 4지선다 퀴즈를 생성하세요. "
        "기사에 없는 내용은 사용하지 말고, 정답은 반드시 하나만 명확해야 합니다. "
        "반드시 JSON 객체만 반환하세요.\n\n"
        f"키워드: {request.word}\n"
        f"요약: {clean_text(request.summary or '')}\n"
        f"문항 수: {target_count}\n"
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
                    "You generate Korean news quizzes as strict JSON. "
                    "Each quiz must have exactly four options and answer_index from 0 to 3."
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

    for item in raw_quizzes:
        quiz = _coerce_quiz_info(item, valid_news_ids)
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
        f"quiz:exact:{request.keyword_id}:{related_ids}:{body_digest}:{count}",
        f"quiz:keyword:{request.keyword_id}:{count}",
        f"quiz:word:{normalized}:{count}",
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
    source_ids = [source_news.news_id] if source_news else []
    source_title = clean_text(source_news.title) if source_news else f"{request.word} 관련 뉴스"
    summary = clean_text(request.summary or "")
    body = clean_text(source_news.body) if source_news else ""
    evidence = truncate_text(summary or body or source_title, 130)
    keyword = clean_text(request.word) or "이 키워드"

    templates = [
        (
            f"'{keyword}' 관련 기사에서 핵심적으로 다룬 내용으로 가장 적절한 것은?",
            [
                evidence,
                "기사와 무관한 스포츠 경기 결과",
                "제공된 근거에 없는 해외 축제 일정",
                "본문에서 확인되지 않는 연예계 소식",
            ],
            0,
            f"'{source_title}'의 근거 문장을 바탕으로 한 설명입니다.",
        ),
        (
            f"'{keyword}' 이슈를 이해할 때 근거로 삼아야 할 자료는 무엇인가요?",
            [
                "제공된 관련 뉴스의 제목과 본문",
                "출처가 없는 온라인 추측",
                "무작위 광고 문구",
                "기사에 없는 개인 의견",
            ],
            0,
            "퀴즈는 백엔드가 전달한 관련 뉴스 내용만 근거로 생성됩니다.",
        ),
        (
            f"'{keyword}'에 대한 설명 중 기사 근거에 가장 가까운 것은?",
            [
                evidence,
                "본문과 관계없는 기술 매뉴얼 설명",
                "근거 없이 확정적으로 단정한 전망",
                "제공된 뉴스와 연결되지 않는 일반 상식",
            ],
            0,
            f"제공된 요약 또는 '{source_title}' 본문에서 확인되는 내용입니다.",
        ),
    ]

    quizzes = []
    for index in range(start_index, target_count):
        question, options, answer_index, explanation = templates[index % len(templates)]
        quizzes.append(
            QuizInfo(
                question=question,
                options=[truncate_text(option, 120) for option in options],
                answer_index=answer_index,
                explanation=explanation,
                source_news_ids=source_ids,
            )
        )
    return quizzes


def _coerce_quiz_info(item: object, valid_news_ids: set[int]) -> QuizInfo | None:
    """Validate one LLM quiz object and return a backend-safe QuizInfo."""

    if not isinstance(item, dict):
        return None

    question = clean_text(str(item.get("question", "")))
    raw_options = item.get("options", [])
    if not isinstance(raw_options, list):
        return None
    options = [clean_text(str(option)) for option in raw_options]
    options = [option for option in options if option]

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

    if not question or len(options) != 4 or answer_index < 0 or answer_index > 3 or not explanation:
        return None

    return QuizInfo(
        question=question,
        options=options,
        answer_index=answer_index,
        explanation=explanation,
        source_news_ids=source_ids,
    )


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
