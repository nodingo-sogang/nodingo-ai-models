import unittest

from app.schemas import KeywordResult, NewsAnalysisResult
from app.services.keyword_service import build_keyword_relations, extract_keywords_from_news_frequency


def keyword(word: str, vector: list[float], weight: float = 0.8) -> KeywordResult:
    return KeywordResult(
        keyword_id=None,
        word=word,
        normalized_word=word,
        embedding=vector,
        weight=weight,
        is_new=True,
        aliases=[word],
        extraction_score=weight,
        personas="SOCIETY",
        macro="사법",
    )


class KeywordRelationSimilarityTest(unittest.TestCase):
    def test_isolated_keywords_get_similarity_relations(self) -> None:
        news_results = [
            NewsAnalysisResult(
                news_id=1,
                embedding=[1.0, 0.0, 0.0],
                summary="감형 관련 기사",
                keywords=[keyword("감형", [1.0, 0.0, 0.0])],
            ),
            NewsAnalysisResult(
                news_id=2,
                embedding=[0.9, 0.1, 0.0],
                summary="양형 관련 기사",
                keywords=[keyword("양형", [0.9, 0.1, 0.0])],
            ),
            NewsAnalysisResult(
                news_id=3,
                embedding=[0.8, 0.2, 0.0],
                summary="판결 관련 기사",
                keywords=[keyword("판결", [0.8, 0.2, 0.0])],
            ),
            NewsAnalysisResult(
                news_id=4,
                embedding=[0.0, 1.0, 0.0],
                summary="무관한 기사",
                keywords=[keyword("반도체", [0.0, 1.0, 0.0], weight=0.4)],
            ),
        ]

        relations = build_keyword_relations(news_results)
        pairs = {
            (item.subject_normalized_word, item.related_normalized_word)
            for item in relations
        }

        self.assertIn(("감형", "양형"), pairs)
        self.assertIn(("감형", "판결"), pairs)

    def test_frequency_fallback_extracts_at_least_minimum_keywords_when_available(self) -> None:
        candidates = extract_keywords_from_news_frequency(
            "금리와 물가, 환율이 통화정책에 미치는 영향",
            "한국은행은 기준금리와 물가, 환율, 가계부채, 부동산 시장을 함께 점검했다.",
            5,
        )

        self.assertGreaterEqual(len(candidates), 5)


if __name__ == "__main__":
    unittest.main()
