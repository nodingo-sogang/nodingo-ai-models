import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class CostSaverSettings:
    openai_api_key = None
    openai_model = "gpt-4o-mini"
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
    analyze_batch_chunk_size = 3
    openai_embed_batch_size = 30
    openai_request_sleep_seconds = 0.0
    max_openai_body_chars = 1000
    analysis_cache_path = "test2/cache/news_analysis_cache.json"
    summary_cache_path = "test2/cache/summary_cache.json"
    quiz_cache_path = "test2/cache/quiz_cache.json"


class KeywordMacroDiversityTest(unittest.TestCase):
    def test_cost_saver_batch_preserves_macro_diversity_without_openai(self) -> None:
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
                            "news_id": 201,
                            "title": "국회 법안 논의와 정부 정책 조정",
                            "body": "국회는 새 법안을 논의하고 정부는 규제와 예산 정책을 조정했습니다.",
                            "published_at": "2026-06-08T09:00:00",
                        },
                        {
                            "news_id": 202,
                            "title": "기준금리와 환율 변동에 금융시장 주목",
                            "body": "은행 대출과 증권 시장은 기준금리, 환율, 물가 흐름을 함께 살피고 있습니다.",
                            "published_at": "2026-06-08T10:00:00",
                        },
                        {
                            "news_id": 203,
                            "title": "AI 반도체와 데이터센터 투자 확대",
                            "body": "AI 서비스 확산으로 GPU, HBM, 반도체, 데이터센터 투자가 늘고 있습니다.",
                            "published_at": "2026-06-08T11:00:00",
                        },
                        {
                            "news_id": 204,
                            "title": "의료와 교육 현안에 사회적 관심",
                            "body": "병원 의료 인력, 의대 정원, 학교 교육과 복지 정책이 주요 쟁점입니다.",
                            "published_at": "2026-06-08T12:00:00",
                        },
                        {
                            "news_id": 205,
                            "title": "영화 드라마 K팝 콘텐츠 수출 증가",
                            "body": "영화, 드라마, 음악, 웹툰, 게임 등 콘텐츠 산업이 해외 시장에서 성장했습니다.",
                            "published_at": "2026-06-08T13:00:00",
                        },
                        {
                            "news_id": 206,
                            "title": "미국 중국 관세와 외교 안보 이슈",
                            "body": "미국과 중국은 관세와 무역분쟁을 두고 외교, 정상회담, 안보 현안을 논의했습니다.",
                            "published_at": "2026-06-08T14:00:00",
                        },
                    ],
                    "existing_keywords": [],
                    "top_k_keywords": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("news_results", payload)
        self.assertIn("keyword_relations", payload)
        self.assertIn("news_relations", payload)

        macros = {
            keyword["macro"]
            for result in payload["news_results"]
            for keyword in result["keywords"]
            if keyword.get("macro")
        }
        self.assertGreaterEqual(len(macros), 4)
        self.assertLessEqual(len(macros), 6 * len(payload["news_results"]))

        for result in payload["news_results"]:
            self.assertGreaterEqual(len(result["keywords"]), 1)
            for keyword in result["keywords"]:
                self.assertTrue(keyword["personas"])
                self.assertTrue(keyword["macro"])


if __name__ == "__main__":
    unittest.main()
