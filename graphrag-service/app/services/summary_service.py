try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenAI = None

from app.config import get_settings
from app.schemas import SummaryKeywordInput, SummaryNewsInput, SummaryRelatedKeywordInput
from app.services.cache_service import get_cache_value, set_cache_values
from app.services.fallback_service import (
    log_openai_fallback,
    normalize_keyword_text,
    template_summary_for_keyword,
    text_hash,
)
from app.utils.text_utils import clean_text, truncate_text


def generate_recommend_keyword_summary(
    keyword: SummaryKeywordInput,
    related_news: list[SummaryNewsInput],
    related_keywords: list[SummaryRelatedKeywordInput],
    persona: str | None = None,
    category: str | None = None,
) -> str:
    """Generate a grounded keyword summary with OpenAI when configured, otherwise fallback."""

    settings = get_settings()
    cache_keys = _summary_cache_keys(keyword, related_news, persona, category)
    cache_path = getattr(settings, "summary_cache_path", "test2/cache/summary_cache.json")
    cache_enabled = bool(getattr(settings, "enable_summary_cache", False))
    if cache_enabled:
        cached = get_cache_value(cache_path, cache_keys)
        if isinstance(cached, str) and cached.strip():
            return cached

    if getattr(settings, "openai_cost_saver_mode", False):
        summary = generate_fallback_summary(keyword, related_news, related_keywords, persona, category)
        if cache_enabled:
            _store_summary_cache(cache_path, cache_keys, summary)
        return summary

    if settings.openai_api_key and settings.use_openai_summary:
        try:
            summary = generate_openai_summary(keyword, related_news, related_keywords, persona, category)
            if cache_enabled:
                _store_summary_cache(cache_path, cache_keys, summary)
            return summary
        except Exception as exc:
            log_openai_fallback(f"summary failed, using template summary: {exc}")
            summary = generate_fallback_summary(keyword, related_news, related_keywords, persona, category)
            if cache_enabled:
                _store_summary_cache(cache_path, cache_keys, summary)
            return summary
    summary = generate_fallback_summary(keyword, related_news, related_keywords, persona, category)
    if cache_enabled:
        _store_summary_cache(cache_path, cache_keys, summary)
    return summary


def generate_openai_summary(
    keyword: SummaryKeywordInput,
    related_news: list[SummaryNewsInput],
    related_keywords: list[SummaryRelatedKeywordInput],
    persona: str | None = None,
    category: str | None = None,
) -> str:
    """Generate a 2-3 sentence Korean summary using only provided news evidence."""

    settings = get_settings()
    if OpenAI is None:
        return generate_fallback_summary(keyword, related_news, related_keywords, persona, category)
    client = OpenAI(api_key=settings.openai_api_key)
    evidence = _format_news_evidence(related_news)
    related = ", ".join(item.word for item in related_keywords[:8]) or "없음"
    perspective = _format_perspective(persona, category)
    prompt = (
        "다음 기사 근거만 사용해 추천 키워드가 왜 중요한지 한국어 2~3문장으로 요약하세요. "
        "기사에 없는 내용은 추측하지 말고, 불확실한 내용은 가능성으로 단정하지 마세요.\n\n"
        f"키워드: {keyword.word}\n"
        f"관련 키워드: {related}\n"
        f"{perspective}"
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
    return clean_text(content) or generate_fallback_summary(keyword, related_news, related_keywords, persona, category)


def generate_fallback_summary(
    keyword: SummaryKeywordInput,
    related_news: list[SummaryNewsInput],
    related_keywords: list[SummaryRelatedKeywordInput],
    persona: str | None = None,
    category: str | None = None,
) -> str:
    """Generate a deterministic 2-3 sentence fallback summary from provided news text."""

    keyword_word = clean_text(keyword.word)
    titles = [clean_text(news.title) for news in related_news if clean_text(news.title)]
    body_snippet = truncate_text(" ".join(news.body for news in related_news), 220)
    related_words = [item.word for item in related_keywords[:3]]

    if not related_news:
        return template_summary_for_keyword(keyword_word, related_keywords=related_words, category=category)

    first_title = titles[0] if titles else f"{keyword_word} 관련 기사"
    related_part = f" 함께 언급된 키워드는 {', '.join(related_words)}입니다." if related_words else ""
    category_part = f" {category} 관점에서 관련 흐름을 확인할 필요가 있습니다." if category else ""
    return (
        f"{keyword_word}은(는) '{first_title}' 등 관련 기사에서 반복적으로 등장한 이슈입니다. "
        f"{body_snippet}{related_part}{category_part}"
    )


def _summary_cache_keys(
    keyword: SummaryKeywordInput,
    related_news: list[SummaryNewsInput],
    persona: str | None,
    category: str | None,
) -> list[str]:
    normalized = normalize_keyword_text(keyword.word)
    related_ids = ",".join(str(news.news_id) for news in related_news)
    body_digest = text_hash(*(news.body for news in related_news))
    return [
        f"summary:exact:{keyword.keyword_id}:{related_ids}:{body_digest}:{persona or ''}:{category or ''}",
        f"summary:keyword_date:{keyword.keyword_id}:{body_digest}",
        f"summary:keyword:{keyword.keyword_id}",
        f"summary:word:{normalized}",
    ]


def _store_summary_cache(path: str, keys: list[str], summary: str) -> None:
    if not summary.strip():
        return
    set_cache_values(path, {key: summary for key in keys})


def _format_news_evidence(related_news: list[SummaryNewsInput]) -> str:
    """Format news evidence compactly for the LLM prompt."""

    if not related_news:
        return "제공된 관련 뉴스가 없습니다."
    lines = []
    for news in related_news[:5]:
        title = truncate_text(news.title, 160)
        body = truncate_text(news.body, get_settings().max_openai_body_chars)
        lines.append(f"- news_id={news.news_id}, title={title}, body={body}")
    return "\n".join(lines)


def _format_perspective(persona: str | None, category: str | None) -> str:
    """Format optional persona/category instructions without making them required."""

    lines = []
    if persona:
        lines.append(f"persona: {clean_text(persona)}")
    if category:
        lines.append(
            f"category: {clean_text(category)}. 해당 분야의 영향, 이해관계, 시장/정책 흐름을 우선해 요약하세요."
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
