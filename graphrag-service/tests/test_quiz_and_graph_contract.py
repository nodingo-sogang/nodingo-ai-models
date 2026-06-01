import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class NoOpenAiSettings:
    openai_api_key = None


class QuizAndGraphContractTest(unittest.TestCase):
    def test_generate_quizzes_endpoint_returns_backend_safe_shape(self) -> None:
        client = TestClient(app)
        with patch("app.services.quiz_service.get_settings", return_value=NoOpenAiSettings()):
            response = client.post(
                "/v1/quizzes/generate",
                json={
                    "keyword_id": 10,
                    "word": "HBM4",
                    "summary": "HBM4는 AI 반도체 수요 확대와 함께 주목받는 차세대 메모리입니다.",
                    "related_news": [
                        {
                            "news_id": 1,
                            "title": "삼성전자, HBM4 양산 가속화",
                            "body": "삼성전자가 HBM4 양산 일정을 앞당기며 AI 반도체 시장 대응을 강화하고 있습니다.",
                            "url": "https://example.com/news/1",
                        }
                    ],
                    "num_questions": 3,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["keyword_id"], 10)
        self.assertEqual(len(payload["quizzes"]), 3)
        for quiz in payload["quizzes"]:
            self.assertEqual(
                set(quiz.keys()),
                {"question", "options", "answer_index", "explanation", "source_news_ids"},
            )
            self.assertEqual(len(quiz["options"]), 4)
            self.assertIn(quiz["answer_index"], [0, 1, 2, 3])
            self.assertTrue(quiz["question"])
            self.assertTrue(quiz["explanation"])

    def test_graph_preview_nodes_include_unlock_metadata(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/v1/graph/preview",
            json={
                "recommend_keywords": [
                    {"keyword_id": i, "word": f"키워드{i}", "score": 1.0 - (i * 0.01)}
                    for i in range(1, 23)
                ],
                "keyword_relations": [
                    {"source_keyword_id": 1, "target_keyword_id": i, "relation_score": 0.8}
                    for i in range(2, 23)
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["nodes"]), 22)

        visible_count = sum(1 for node in payload["nodes"] if node["visibility"] == "VISIBLE")
        fog_count = sum(1 for node in payload["nodes"] if node["visibility"] == "FOG")
        self.assertEqual(visible_count, 20)
        self.assertEqual(fog_count, 2)

        for node in payload["nodes"]:
            self.assertIn("unlock_level", node)
            self.assertIn("visibility", node)


if __name__ == "__main__":
    unittest.main()
