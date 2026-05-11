import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import AnalyzeNewsBatchResponse


REQUIRED_RELATION_KEYS = {
    "subject_keyword_id",
    "related_keyword_id",
    "subject_normalized_word",
    "related_normalized_word",
    "relation_score",
    "evidence_news_ids",
}

FORBIDDEN_RELATION_KEYS = {
    "source_keyword_id",
    "target_keyword_id",
    "source_normalized_word",
    "target_normalized_word",
    "relation_type",
    "relationType",
    "sourceWord",
    "targetWord",
    "newsIds",
}


def main() -> None:
    """Verify /v1/news/analyze-batch keyword_relations response keys."""

    response = AnalyzeNewsBatchResponse(
        news_results=[],
        keyword_relations=[
            {
                "subject_keyword_id": None,
                "related_keyword_id": None,
                "subject_normalized_word": "삼성전자",
                "related_normalized_word": "hbm",
                "relation_score": 0.85,
                "evidence_news_ids": [1],
            }
        ],
        news_relations=[],
    )
    payload = response.model_dump(mode="json")

    assert "keyword_relations" in payload
    assert isinstance(payload["keyword_relations"], list)
    assert payload["keyword_relations"], "keyword_relations should contain a sample relation"

    relation = payload["keyword_relations"][0]
    missing = REQUIRED_RELATION_KEYS - set(relation)
    forbidden = FORBIDDEN_RELATION_KEYS & set(relation)

    assert not missing, f"missing keys: {sorted(missing)}"
    assert not forbidden, f"forbidden keys found: {sorted(forbidden)}"

    print("keyword_relations response schema ok")
    print(sorted(relation.keys()))


if __name__ == "__main__":
    main()
