import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class QuizFallbackSettings:
    openai_api_key = None


class QuizQualityContractTest(unittest.TestCase):
    def test_fallback_quizzes_are_specific_and_schema_safe_without_openai(self) -> None:
        client = TestClient(app)

        with patch("app.services.quiz_service.get_settings", return_value=QuizFallbackSettings()):
            response = client.post(
                "/v1/quizzes/generate",
                json={
                    "keyword_id": 7,
                    "word": "금리",
                    "summary": "금리 변화는 대출 이자 부담, 환율, 부동산 시장 흐름과 연결됩니다.",
                    "related_news": [
                        {
                            "news_id": 101,
                            "title": "기준금리 동결에 대출자 부담 지속",
                            "body": "한국은행의 기준금리 결정은 은행 대출 이자와 부동산 거래 심리에 영향을 줍니다.",
                        }
                    ],
                    "num_questions": 4,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["keyword_id"], 7)
        self.assertEqual(len(payload["quizzes"]), 4)

        banned_patterns = (
            "핵심적으로 다룬 내용",
            "가장 적절한 것은",
            "근거로 삼아야 할 자료",
            "스포츠 경기 결과",
            "해외 축제 일정",
            "연예계 소식",
            "광고 문구",
            "날씨 정보",
            "로그인",
            "회원가입",
            "메뉴",
        )
        answer_indexes = set()

        for quiz in payload["quizzes"]:
            self.assertEqual(
                set(quiz.keys()),
                {"question", "options", "answer_index", "explanation", "source_news_ids"},
            )
            self.assertTrue(quiz["question"])
            self.assertFalse(any(pattern in quiz["question"] for pattern in banned_patterns))
            self.assertEqual(len(quiz["options"]), 4)
            self.assertEqual(len(set(quiz["options"])), 4)
            self.assertIn(quiz["answer_index"], [0, 1, 2, 3])
            answer_indexes.add(quiz["answer_index"])
            self.assertTrue(quiz["explanation"])
            self.assertEqual(quiz["source_news_ids"], [101])
            for option in quiz["options"]:
                self.assertLessEqual(len(option), 50)
                self.assertFalse(any(pattern in option for pattern in banned_patterns))

        self.assertGreater(len(answer_indexes), 1)

    def test_fallback_quizzes_do_not_empty_out_without_related_news(self) -> None:
        client = TestClient(app)

        with patch("app.services.quiz_service.get_settings", return_value=QuizFallbackSettings()):
            response = client.post(
                "/v1/quizzes/generate",
                json={
                    "keyword_id": 8,
                    "word": "AI",
                    "summary": "AI는 데이터와 반도체 수요를 함께 이해해야 하는 기술 이슈입니다.",
                    "related_news": [],
                    "num_questions": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        quizzes = response.json()["quizzes"]
        self.assertEqual(len(quizzes), 2)
        for quiz in quizzes:
            self.assertEqual(len(quiz["options"]), 4)
            self.assertIn(quiz["answer_index"], [0, 1, 2, 3])
            self.assertTrue(quiz["question"])
            self.assertTrue(quiz["explanation"])
            self.assertEqual(quiz["source_news_ids"], [])


if __name__ == "__main__":
    unittest.main()
