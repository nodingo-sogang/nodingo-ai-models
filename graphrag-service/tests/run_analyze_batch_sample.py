import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import AnalyzeNewsBatchRequest
from app.services.keyword_service import analyze_news_batch


def main() -> None:
    """Run the analyze-batch service with the converted sample payload."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tests/news_full_analyze_batch_sample.json")
    parser.add_argument("--output", default="tests/news_full_analyze_batch_result.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    request = AnalyzeNewsBatchRequest.model_validate_json(input_path.read_text(encoding="utf-8"))
    response = analyze_news_batch(request)
    output_path.write_text(
        json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"analyzed {len(response.news_results)} news items")
    print(f"keyword_relations={len(response.keyword_relations)}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
