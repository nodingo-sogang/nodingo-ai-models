import argparse
import json
import re
from pathlib import Path


def truncate_text(text: str | None, max_chars: int) -> str:
    """Trim long article content so local model tests stay manageable."""

    if not text:
        return ""
    value = " ".join(str(text).split())
    return value[:max_chars].rstrip()


def clean_article_content(text: str | None) -> str:
    """Remove common crawled-news boilerplate before making a model payload."""

    value = " ".join(str(text or "").split())
    start_markers = [
        "본문 글씨 키우기",
        "본문 글씨 줄이기",
        "기사스크랩하기",
        "AI 요약",
    ]
    for marker in start_markers:
        index = value.find(marker)
        if index >= 0:
            value = value[index + len(marker) :]
            break

    end_markers = [
        "다른기사 보기",
        "다른기사 추천",
        "기사제보",
        "반론신청",
        "광고문의",
        "개의 댓글",
        "내 댓글 모음",
    ]
    for marker in end_markers:
        index = value.find(marker)
        if index >= 0:
            value = value[:index]

    boilerplate_patterns = [
        r"회원로그인",
        r"댓글 내용입력 \d+ / \d+",
        r"본문 글씨 (줄이기|키우기)",
        r"SNS 기사보내기",
        r"Email Share Scrap Print",
        r"이 기사를 공유합니다",
        r"페이스북 카카오톡 네이버블로그 닫기",
        r"바로가기 복사하기",
    ]
    for pattern in boilerplate_patterns:
        value = re.sub(pattern, " ", value)
    return " ".join(value.split())


def convert_news_full_payload(input_path: Path, limit: int, max_body_chars: int) -> dict:
    """Convert news_full.json into /v1/news/analyze-batch request JSON."""

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    news_list = raw.get("data", {}).get("news_list", [])
    converted_news = []

    for item in news_list[:limit]:
        converted_news.append(
            {
                "news_id": int(item["news_id"]),
                "title": truncate_text(item.get("title"), 300),
                "body": truncate_text(clean_article_content(item.get("content")), max_body_chars),
                "published_at": item.get("published_at"),
            }
        )

    return {
        "news": converted_news,
        "existing_keywords": [],
        "top_k_keywords": 8,
    }


def main() -> None:
    """Write a converted test payload file for the GraphRAG analyze-batch API."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-body-chars", type=int, default=2500)
    args = parser.parse_args()

    payload = convert_news_full_payload(args.input, args.limit, args.max_body_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(payload['news'])} news items to {args.output}")


if __name__ == "__main__":
    main()
