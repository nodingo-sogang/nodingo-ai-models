import re


def clean_text(text: str | None) -> str:
    """Normalize whitespace and remove control-like empty content."""

    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str | None, max_chars: int) -> str:
    """Return text trimmed to max_chars without raising on empty input."""

    value = clean_text(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def safe_join_title_body(title: str | None, body: str | None) -> str:
    """Join title and body into the analysis text used by GraphRAG services."""

    parts = [clean_text(title), clean_text(body)]
    return "\n\n".join(part for part in parts if part)
