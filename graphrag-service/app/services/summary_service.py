try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAI = None

from app.config import get_settings
from app.schemas import SummaryKeywordInput, SummaryNewsInput, SummaryRelatedKeywordInput
from app.utils.text_utils import clean_text, truncate_text


def generate_recommend_keyword_summary(
    keyword: SummaryKeywordInput,
    related_news: list[SummaryNewsInput],
    related_keywords: list[SummaryRelatedKeywordInput],
) -> str:
    """Generate a grounded keyword summary with OpenAI when configured, otherwise fallback."""

    settings = get_settings()
    if settings.openai_api_key and settings.use_openai_summary:
        try:
            return generate_openai_summary(keyword, related_news, related_keywords)
        except Exception:
            return generate_fallback_summary(keyword, related_news, related_keywords)
    return generate_fallback_summary(keyword, related_news, related_keywords)


def generate_openai_summary(
    keyword: SummaryKeywordInput,
    related_news: list[SummaryNewsInput],
    related_keywords: list[SummaryRelatedKeywordInput],
) -> str:
    """Generate a 2-3 sentence Korean summary using only provided news evidence."""

    settings = get_settings()
    if OpenAI is None:
        return generate_fallback_summary(keyword, related_news, related_keywords)
    client = OpenAI(api_key=settings.openai_api_key)
    evidence = _format_news_evidence(related_news)
    related = ", ".join(item.word for item in related_keywords[:8]) or "없음"
    prompt = (
        "다음 기사 근거만 사용해 추천 키워드가 왜 중요한지 한국어 2~3문장으로 요약하세요. "
        "기사에 없는 내용은 추측하지 말고, 불확실한 내용은 가능성으로 단정하지 마세요.\n\n"
        f"키워드: {keyword.word}\n"
        f"관련 키워드: {related}\n"
        f"기사 근거:\n{evidence}"
    )
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": "You write concise, evidence-grounded Korean news briefings.",
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=350,
    )
    content = response.choices[0].message.content or ""
    return clean_text(content) or generate_fallback_summary(keyword, related_news, related_keywords)


def generate_fallback_summary(
    keyword: SummaryKeywordInput,
    related_news: list[SummaryNewsInput],
    related_keywords: list[SummaryRelatedKeywordInput],
) -> str:
    """Generate a deterministic 2-3 sentence fallback summary from provided news text."""

    keyword_word = clean_text(keyword.word)
    titles = [clean_text(news.title) for news in related_news if clean_text(news.title)]
    body_snippet = truncate_text(" ".join(news.body for news in related_news), 220)
    related_words = [item.word for item in related_keywords[:3]]

    if not related_news:
        return f"{keyword_word}은(는) 최근 추천 후보로 분류된 키워드입니다. 제공된 관련 기사가 없어 추가 근거 요약은 생성하지 않았습니다."

    first_title = titles[0] if titles else f"{keyword_word} 관련 기사"
    related_part = f" 함께 언급된 키워드는 {', '.join(related_words)}입니다." if related_words else ""
    return (
        f"{keyword_word}은(는) '{first_title}' 등 관련 기사에서 반복적으로 등장한 이슈입니다. "
        f"{body_snippet}{related_part}"
    )


def _format_news_evidence(related_news: list[SummaryNewsInput]) -> str:
    """Format news evidence compactly for the LLM prompt."""

    if not related_news:
        return "제공된 관련 뉴스가 없습니다."
    lines = []
    for news in related_news[:5]:
        title = truncate_text(news.title, 160)
        body = truncate_text(news.body, 700)
        lines.append(f"- news_id={news.news_id}, title={title}, body={body}")
    return "\n".join(lines)
