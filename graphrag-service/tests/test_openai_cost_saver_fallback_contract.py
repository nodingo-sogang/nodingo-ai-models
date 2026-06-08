import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class CostSaverSettings:
    openai_api_key = None
    openai_model = "gpt-5-mini"
    openai_embedding_model = "text-embedding-3-small"
    embedding_model = "text-embedding-3-small"
    embedding_dim = 1536
    use_openai_summary = True
    use_openai_keyword_extraction = True
    openai_cost_saver_mode = True
    openai_fallback_on_error = True
    enable_analysis_cache = False
    enable_summary_cache = False
    enable_quiz_cache = False
    analyze_batch_chunk_size = 30
    openai_embed_batch_size = 30
    openai_request_sleep_seconds = 0.0
    max_openai_body_chars = 500


class OpenAiCostSaverFallbackContractTest(unittest.TestCase):
    def test_analyze_batch_returns_non_empty_fallback_contract_without_openai(self) -> None:
        client = TestClient(app)
        settings = CostSaverSettings()
        with patch("app.services.keyword_service.get_settings", return_value=settings), patch(
            "app.services.embedding_service.get_settings", return_value=settings
        ):
            response = client.post(
                "/v1/news/analyze-batch",
                json={
                    "news": [
                        {
                            "news_id": 101,
                            "title": "한국은행 기준금리 동결, 가계부채 부담 지속",
                            "body": "한국은행은 물가와 가계부채 흐름을 고려해 기준금리를 동결했습니다. 환율과 부동산 시장 영향도 함께 거론됩니다.",
                            "published_at": "2026-06-08T09:00:00",
                        }
                    ],
                    "existing_keywords": [],
                    "top_k_keywords": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["news_results"]), 1)
        result = payload["news_results"][0]
        self.assertEqual(result["news_id"], 101)
        self.assertEqual(len(result["embedding"]), 1536)
        self.assertTrue(result["summary"])
        self.assertGreaterEqual(len(result["keywords"]), 1)
        for keyword in result["keywords"]:
            self.assertTrue(keyword["word"])
            self.assertTrue(keyword["normalized_word"])
            self.assertEqual(len(keyword["embedding"]), 1536)
            self.assertTrue(keyword["personas"])
            self.assertTrue(keyword["macro"])

    def test_summary_and_quiz_use_loose_cache_or_fallback_without_openai(self) -> None:
        client = TestClient(app)
        settings = CostSaverSettings()
        with patch("app.services.summary_service.get_settings", return_value=settings):
            summary_response = client.post(
                "/v1/recommend-keywords/summarize",
                json={
                    "keyword": {"keyword_id": 5, "word": "금리"},
                    "related_news": [],
                    "related_keywords": [],
                    "target_date": "2026-06-08",
                },
            )
        with patch("app.services.quiz_service.get_settings", return_value=settings):
            quiz_response = client.post(
                "/v1/quizzes/generate",
                json={
                    "keyword_id": 5,
                    "word": "금리",
                    "summary": summary_response.json()["summary"],
                    "related_news": [],
                    "num_questions": 3,
                },
            )

        self.assertEqual(summary_response.status_code, 200)
        self.assertTrue(summary_response.json()["summary"])
        self.assertEqual(quiz_response.status_code, 200)
        quizzes = quiz_response.json()["quizzes"]
        self.assertEqual(len(quizzes), 3)
        for quiz in quizzes:
            self.assertTrue(quiz["question"])
            self.assertEqual(len(quiz["options"]), 4)
            self.assertIn(quiz["answer_index"], [0, 1, 2, 3])
            self.assertTrue(quiz["explanation"])


if __name__ == "__main__":
    unittest.main()
