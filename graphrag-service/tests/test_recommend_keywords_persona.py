from datetime import date
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import CandidateKeywordInput
from app.services.recommendation_service import recommend_keywords


DIM = 1536


def vector(first: float) -> list[float]:
    return [first] + [0.0] * (DIM - 1)


def candidate(
    keyword_id: int,
    word: str,
    embedding: list[float],
    recent_importance: float,
) -> CandidateKeywordInput:
    return CandidateKeywordInput(
        keyword_id=keyword_id,
        word=word,
        normalized_word=word,
        embedding=embedding,
        recent_importance=recent_importance,
        is_user_interest=False,
    )


class RecommendKeywordsPersonaTest(unittest.TestCase):
    def test_politics_candidates_are_ranked_before_higher_scoring_non_politics(self) -> None:
        candidates = [
            candidate(1, "삼성전자", vector(1.0), 1.0),
            candidate(2, "HBM", vector(1.0), 1.0),
            candidate(3, "금리", vector(1.0), 1.0),
            candidate(10, "대선", vector(0.0), 0.1),
            candidate(11, "국회", vector(0.0), 0.1),
            candidate(12, "정부", vector(0.0), 0.1),
            candidate(13, "선거", vector(0.0), 0.1),
            candidate(14, "공약", vector(0.0), 0.1),
        ]
        raw_payload = {
            "persona": "POLITICS",
            "candidate_keywords": [
                {"keyword_id": 1, "persona": "ECONOMY"},
                {"keyword_id": 2, "persona": "TECH"},
                {"keyword_id": 3, "persona": "ECONOMY"},
                {"keyword_id": 10, "persona": "POLITICS"},
                {"keyword_id": 11, "persona": "POLITICS"},
                {"keyword_id": 12, "persona": "POLITICS"},
                {"keyword_id": 13, "persona": "POLITICS"},
                {"keyword_id": 14, "persona": "POLITICS"},
            ],
        }

        results = recommend_keywords(1, vector(1.0), candidates, date(2026, 5, 26), 5, raw_payload)

        self.assertEqual([item.keyword_id for item in results], [10, 11, 12, 13, 14])

    def test_politics_candidates_fill_first_then_fallback_when_insufficient(self) -> None:
        candidates = [
            candidate(1, "삼성전자", vector(1.0), 1.0),
            candidate(2, "HBM", vector(1.0), 0.9),
            candidate(10, "대선", vector(0.0), 0.1),
            candidate(11, "국회", vector(0.0), 0.1),
        ]
        raw_payload = {
            "persona": "POLITICS",
            "candidate_keywords": [
                {"keyword_id": 1, "persona": "ECONOMY"},
                {"keyword_id": 2, "persona": "TECH"},
                {"keyword_id": 10, "persona": "POLITICS"},
                {"keyword_id": 11, "persona": "POLITICS"},
            ],
        }

        results = recommend_keywords(1, vector(1.0), candidates, date(2026, 5, 26), 4, raw_payload)

        self.assertEqual([item.keyword_id for item in results[:2]], [10, 11])
        self.assertEqual([item.keyword_id for item in results[2:]], [1, 2])

    def test_fallback_uses_existing_ranking_when_no_politics_candidates_exist(self) -> None:
        candidates = [
            candidate(1, "삼성전자", vector(1.0), 0.7),
            candidate(2, "HBM", vector(1.0), 1.0),
            candidate(3, "금리", vector(0.0), 1.0),
        ]
        raw_payload = {
            "persona": "POLITICS",
            "candidate_keywords": [
                {"keyword_id": 1, "persona": "ECONOMY"},
                {"keyword_id": 2, "persona": "TECH"},
                {"keyword_id": 3, "persona": "ECONOMY"},
            ],
        }

        results = recommend_keywords(1, vector(1.0), candidates, date(2026, 5, 26), 3, raw_payload)

        self.assertEqual([item.keyword_id for item in results], [2, 1, 3])

    def test_endpoint_response_shape_stays_unchanged_with_extra_persona_hints(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/v1/recommend-keywords",
            json={
                "user_id": 1,
                "user_embedding": vector(1.0),
                "persona": "POLITICS",
                "candidate_keywords": [
                    {
                        "keyword_id": 1,
                        "word": "삼성전자",
                        "normalized_word": "삼성전자",
                        "embedding": vector(1.0),
                        "recent_importance": 1.0,
                        "is_user_interest": False,
                        "persona": "ECONOMY",
                    },
                    {
                        "keyword_id": 10,
                        "word": "대선",
                        "normalized_word": "대선",
                        "embedding": vector(0.0),
                        "recent_importance": 0.1,
                        "is_user_interest": False,
                        "persona": "POLITICS",
                    },
                ],
                "target_date": "2026-05-26",
                "top_k": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload.keys()), {"recommend_keywords"})
        self.assertEqual(
            set(payload["recommend_keywords"][0].keys()),
            {"user_id", "keyword_id", "target_date", "score", "summary"},
        )
        self.assertEqual(payload["recommend_keywords"][0]["keyword_id"], 10)

    def test_all_supported_personas_can_be_inferred_from_keyword_text(self) -> None:
        cases = [
            ("ECONOMY", "금리", "대선"),
            ("TECH", "HBM", "금리"),
            ("SOCIETY", "교육", "HBM"),
            ("CULTURE", "영화", "교육"),
            ("SPORTS", "축구", "영화"),
            ("GLOBAL", "미국", "축구"),
        ]

        for persona, matching_word, other_word in cases:
            with self.subTest(persona=persona):
                candidates = [
                    candidate(1, other_word, vector(1.0), 1.0),
                    candidate(2, matching_word, vector(0.0), 0.1),
                ]
                raw_payload = {"persona": persona}

                results = recommend_keywords(
                    1,
                    vector(1.0),
                    candidates,
                    date(2026, 5, 26),
                    2,
                    raw_payload,
                )

                self.assertEqual(results[0].keyword_id, 2)


if __name__ == "__main__":
    unittest.main()
