# AI Model Update Notes

## 목적

백엔드 최신 계약에 맞춰 GraphRAG FastAPI 서버를 확장했다.

- `POST /v1/quizzes/generate` 추가
- `POST /v1/graph/preview`의 node 응답에 `unlock_level`, `visibility` 추가
- 백엔드 배치가 실패하지 않도록 퀴즈 응답을 항상 4지선다 형태로 방어

## 변경 파일

- `app/schemas.py`
  - `GraphNode.unlock_level`, `GraphNode.visibility` 추가
  - `GenerateQuizzesRequest`, `GenerateQuizzesResponse`, `QuizInfo`, `QuizRelatedNewsInput` 추가
- `app/main.py`
  - `/v1/quizzes/generate` endpoint 추가
- `app/services/quiz_service.py`
  - OpenAI 기반 뉴스 근거 퀴즈 생성
  - OpenAI 실패 또는 미설정 시 deterministic fallback 퀴즈 생성
  - `options` 4개, `answer_index` 0~3, `source_news_ids` 검증
- `app/services/graph_service.py`
  - 추천 키워드 점수와 관계 강도를 기반으로 node별 `unlock_level`, `visibility` 계산
- `tests/sample_payloads.py`
  - `QUIZ_GENERATE_PAYLOAD` 추가
- `tests/test_quiz_and_graph_contract.py`
  - 퀴즈 생성 응답 계약 테스트 추가
  - graph preview unlock metadata 테스트 추가

## 백엔드 계약

### POST /v1/quizzes/generate

Request:

```json
{
  "keyword_id": 10,
  "word": "HBM4",
  "summary": "키워드 요약",
  "related_news": [
    {
      "news_id": 1,
      "title": "기사 제목",
      "body": "기사 본문",
      "url": "https://example.com/news/1"
    }
  ],
  "num_questions": 3
}
```

Response:

```json
{
  "keyword_id": 10,
  "quizzes": [
    {
      "question": "문제",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer_index": 0,
      "explanation": "해설",
      "source_news_ids": [1]
    }
  ]
}
```

## Graph visibility 규칙

`/v1/graph/preview`는 node ranking을 계산해 다음 값을 내려준다.

- 상위 20개: `visibility = "VISIBLE"`, `unlock_level = 1`
- 21~30위: `visibility = "FOG"`, `unlock_level = 2`
- 31~40위: `visibility = "FOG"`, `unlock_level = 3`
- 41~50위: `visibility = "FOG"`, `unlock_level = 4`
- 51위 이후: `visibility = "HIDDEN"`, `unlock_level = 99`

Ranking score는 추천 점수, 연결 개수, 연결 weight 합, 최대 연결 weight를 함께 사용한다.

## 주의사항

- Python AI 서버는 DB에 저장하지 않는다. 생성된 퀴즈 저장은 Spring Boot 백엔드가 담당한다.
- 백엔드 `QuizGenerationService`가 `options[0]`부터 `options[3]`까지 바로 읽기 때문에 AI 응답은 반드시 선택지 4개를 보장해야 한다.
- OpenAI 응답이 JSON 형식이 아니거나 계약을 어기면 fallback 퀴즈를 반환한다.
- `source_news_ids`는 request의 `related_news.news_id`에 존재하는 값만 유지한다.
