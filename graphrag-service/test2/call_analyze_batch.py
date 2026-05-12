import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000/v1/news/analyze-batch"
TIMEOUT_SECONDS = 300

INPUT_FILE = Path(__file__).resolve().parent /"analyze_batch_input_small.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "analyze_batch_response.json"

REQUIRED_TOP_LEVEL_KEYS = {"news_results", "keyword_relations", "news_relations"}
FORBIDDEN_TOP_LEVEL_KEYS = {"newsResults", "keywordRelations", "newsRelations"}

REQUIRED_KEYWORD_RELATION_KEYS = {
    "subject_keyword_id",
    "related_keyword_id",
    "subject_normalized_word",
    "related_normalized_word",
    "relation_score",
    "evidence_news_ids",
}

FORBIDDEN_KEYWORD_RELATION_KEYS = {
    "source_keyword_id",
    "target_keyword_id",
    "source_normalized_word",
    "target_normalized_word",
    "relation_type",
    "sourceWord",
    "targetWord",
    "newsIds",
}


def load_input_payload() -> dict[str, Any]:
    """Load the converted analyze-batch request payload."""

    return json.loads(INPUT_FILE.read_text(encoding="utf-8"))


def post_json(payload: dict[str, Any]) -> tuple[int, Any]:
    """POST JSON to the running FastAPI server and return status plus parsed body."""

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            response_text = response.read().decode("utf-8")
            return response.status, parse_json_or_text(response_text)
    except HTTPError as error:
        error_text = error.read().decode("utf-8", errors="replace")
        return error.code, parse_json_or_text(error_text)
    except URLError as error:
        print(f"ERROR: 서버에 연결할 수 없습니다: {error}")
        print(f"API_URL: {API_URL}")
        sys.exit(1)


def parse_json_or_text(text: str) -> Any:
    """Parse a response as JSON when possible, otherwise keep text."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def save_response_json(response_json: Any) -> None:
    """Save successful JSON response without escaping Korean characters."""

    OUTPUT_FILE.write_text(
        json.dumps(response_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_json_preview(label: str, value: Any, max_chars: int = 2000) -> None:
    """Print a readable, bounded JSON preview."""

    print(label)
    if value is None:
        print("  없음")
        return
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n  ... 생략 ..."
    print(rendered)


def print_key_check(keys: set[str], required: set[str], forbidden: set[str]) -> None:
    """Print required/forbidden key validation results."""

    for key in sorted(required):
        if key in keys:
            print(f"OK: {key}")
        else:
            print(f"MISSING: {key}")

    for key in sorted(forbidden):
        if key in keys:
            print(f"주의: {key} 가 아직 있습니다.")
        else:
            print(f"OK 없음: {key}")


def validate_response_shape(response_json: dict[str, Any]) -> None:
    """Validate top-level and keyword_relations response keys."""

    print("\n[최상위 필드 검증]")
    top_level_keys = set(response_json.keys())
    print_key_check(top_level_keys, REQUIRED_TOP_LEVEL_KEYS, FORBIDDEN_TOP_LEVEL_KEYS)

    keyword_relations = response_json.get("keyword_relations", [])
    print("\n[keyword_relations 구조 검증]")
    if not isinstance(keyword_relations, list):
        print("MISSING: keyword_relations 가 list가 아닙니다.")
        return
    if not keyword_relations:
        print("주의: keyword_relations 가 비어 있어 relation 필드 검증은 샘플 없이 종료합니다.")
        return

    relation_keys = set(keyword_relations[0].keys())
    print_key_check(
        relation_keys,
        REQUIRED_KEYWORD_RELATION_KEYS,
        FORBIDDEN_KEYWORD_RELATION_KEYS,
    )


def main() -> None:
    """Call /v1/news/analyze-batch with test2/analyze_batch_input.json."""

    payload = load_input_payload()
    request_news_count = len(payload.get("news", []))

    print(f"입력 파일 경로: {INPUT_FILE}")
    print(f"요청 뉴스 개수: {request_news_count}")
    print(f"호출 대상 API: {API_URL}")
    print(f"timeout: {TIMEOUT_SECONDS}초")

    status_code, response_body = post_json(payload)
    print(f"\n응답 status code: {status_code}")

    if status_code != 200:
        print("\n[에러 응답]")
        print_json_preview("응답 내용:", response_body)
        sys.exit(1)

    if not isinstance(response_body, dict):
        print("\nERROR: 200 응답이지만 JSON object가 아닙니다.")
        print_json_preview("응답 내용:", response_body)
        sys.exit(1)

    save_response_json(response_body)

    news_results = response_body.get("news_results", [])
    keyword_relations = response_body.get("keyword_relations", [])
    news_relations = response_body.get("news_relations", [])

    print(f"응답 저장 경로: {OUTPUT_FILE}")
    print(f"최상위 필드 목록: {list(response_body.keys())}")
    print(f"news_results 개수: {len(news_results) if isinstance(news_results, list) else 'list 아님'}")
    print(
        "keyword_relations 개수: "
        f"{len(keyword_relations) if isinstance(keyword_relations, list) else 'list 아님'}"
    )
    print(f"news_relations 개수: {len(news_relations) if isinstance(news_relations, list) else 'list 아님'}")

    print_json_preview(
        "\n첫 번째 news_result 예시:",
        news_results[0] if isinstance(news_results, list) and news_results else None,
    )
    print_json_preview(
        "\n첫 번째 keyword_relation 예시:",
        keyword_relations[0] if isinstance(keyword_relations, list) and keyword_relations else None,
    )
    print_json_preview(
        "\n첫 번째 news_relation 예시:",
        news_relations[0] if isinstance(news_relations, list) and news_relations else None,
    )

    validate_response_shape(response_body)


if __name__ == "__main__":
    main()
