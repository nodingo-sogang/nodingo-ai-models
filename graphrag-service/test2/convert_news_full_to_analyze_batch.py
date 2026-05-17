import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


USER_ID = 1
TARGET_DATE = "2026-05-12"
MAX_BODY_CHARS = 300

INPUT_FILE = Path(__file__).resolve().parent / "news_full.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "analyze_batch_input.json"

TITLE_FIELDS = ["title", "headline", "name", "subject"]
BODY_FIELDS = ["body", "summary", "content", "description", "article", "text"]
DATE_FIELDS = ["published_at", "publishedAt", "pubDate", "published_date", "date", "created_at"]
ID_FIELDS = ["news_id", "id"]


def clean_text(value: Any) -> str:
    """Normalize text while preserving Korean characters."""

    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def truncate_text(value: str, max_chars: int = MAX_BODY_CHARS) -> str:
    """Trim long body text for lightweight analyze-batch tests."""

    value = clean_text(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def pick_first_text(item: dict[str, Any], fields: list[str]) -> str:
    """Pick the first non-empty string from candidate fields."""

    for field in fields:
        value = clean_text(item.get(field))
        if value:
            return value
    return ""


def pick_news_id(item: dict[str, Any], fallback_id: int) -> int:
    """Use news_id/id when available, otherwise assign a sequential id."""

    for field in ID_FIELDS:
        value = item.get(field)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return fallback_id


def normalize_published_at(value: Any) -> str:
    """Return an ISO-like datetime string when possible, otherwise keep the original string."""

    text = clean_text(value)
    if not text:
        return f"{TARGET_DATE}T00:00:00"

    normalized = text.replace("Z", "+00:00")
    known_formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y.%m.%d %H:%M",
        "%Y.%m.%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ]

    try:
        return datetime.fromisoformat(normalized).isoformat()
    except ValueError:
        pass

    for date_format in known_formats:
        try:
            return datetime.strptime(text, date_format).isoformat()
        except ValueError:
            continue
    return text


def extract_news_items(raw: Any) -> list[dict[str, Any]]:
    """Find the news list in common news_full.json shapes."""

    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if not isinstance(raw, dict):
        return []

    if isinstance(raw.get("data"), dict):
        data = raw["data"]
        for field in ["news_list", "news", "items", "articles", "results"]:
            value = data.get(field)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    for field in ["news_list", "news", "items", "articles", "results"]:
        value = raw.get(field)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def convert_item(item: dict[str, Any], fallback_id: int) -> dict[str, Any] | None:
    """Convert one source news item into /v1/news/analyze-batch news input."""

    title = pick_first_text(item, TITLE_FIELDS)
    body = truncate_text(pick_first_text(item, BODY_FIELDS))
    if not title and not body:
        return None

    published_at = normalize_published_at(pick_first_text(item, DATE_FIELDS))
    return {
        "news_id": pick_news_id(item, fallback_id),
        "title": title,
        "body": body,
        "published_at": published_at,
    }


def main() -> None:
    """Create test2/analyze_batch_input.json from test2/news_full.json."""

    raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    news_items = extract_news_items(raw)
    converted = []

    for index, item in enumerate(news_items, start=1):
        converted_item = convert_item(item, index)
        if converted_item is not None:
            converted.append(converted_item)

    payload = {
        "user_id": USER_ID,
        "target_date": TARGET_DATE,
        "news": converted,
    }

    OUTPUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    excluded_count = len(news_items) - len(converted)
    print(f"원본 뉴스 개수: {len(news_items)}")
    print(f"변환된 뉴스 개수: {len(converted)}")
    print(f"제외된 뉴스 개수: {excluded_count}")
    print("첫 번째 변환 결과 예시:")
    print(json.dumps(converted[0] if converted else None, ensure_ascii=False, indent=2))
    print(f"저장된 파일 경로: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
