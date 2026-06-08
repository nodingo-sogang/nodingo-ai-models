# OpenAI Cost Saver Fallback Plan

작성일: 2026-06-08

## 전제와 작업 경계

- 현재 작업 루트: `C:\Users\안가은\Desktop\빅종설`
- 실제 수정 대상: `nodingo-ai-models/graphrag-service`
- 참조만 할 프로젝트: `nodingo-frontend`, `nodingo-backend`
- 이 문서는 사전 조사와 수정 계획만 담는다. 코드 수정은 사용자 컨펌 후 진행한다.
- API endpoint, request schema, response schema, snake_case 필드명, 백엔드 DTO 계약은 변경하지 않는다.

## 1. 현재 프론트엔드가 호출하는 API 목록

프론트엔드 확인 파일:

- `nodingo-frontend/src/api/graph.ts`
- `nodingo-frontend/src/api/quiz.ts`
- `nodingo-frontend/src/api/user.ts`
- `nodingo-frontend/src/api/scrap.ts`
- `nodingo-frontend/src/pages/graph/GraphScreen.tsx`
- `nodingo-frontend/src/components/game/QuizModal.tsx`
- `nodingo-frontend/src/types/index.ts`

그래프/시연 화면 주요 호출:

| 프론트 API | HTTP | 용도 | 기대 응답 핵심 필드 |
|---|---:|---|---|
| `graphApi.getTabs()` | `GET /api/graphs/tabs` | 상단 추천 탭 | `data.tabs[].keyword_id`, `word`, `persona` |
| `graphApi.getGraphData(keywordId)` | `GET /api/graphs/nodes?keywordId={id}` | 그래프 노드/엣지 | `data.nodes[]`, `data.edges[]` |
| `graphApi.getNodeSummary(nodeId, page)` | `GET /api/graphs/nodes/{nodeId}/summaries?page={page}` | 노드 클릭 시 요약/뉴스 | `keyword_id`, `word`, `persona`, `summary`, `has_next`, `news[]` |
| `graphApi.exploreNode(keywordId)` | `POST /api/graphs/nodes/{keywordId}/explore` | 노드 탐험 기록 | `data=null` |
| `graphApi.scrapKeyword(keywordId)` | `POST /api/keywords/{keywordId}/scrap` | 키워드 스크랩 | `data=null` |
| `graphApi.unscrapKeyword(keywordId)` | `DELETE /api/keywords/{keywordId}/scrap` | 키워드 스크랩 해제 | `data=null` |
| `quizApi.getQuizzes(keywordId)` | `GET /api/graphs/nodes/{keywordId}/quizzes` | 노드 퀴즈 목록 | `data.quizzes[]` |
| `quizApi.submit(keywordId, quizId, body)` | `POST /api/graphs/nodes/{keywordId}/quizzes/{quizId}/submit` | 퀴즈 제출 | `correct`, `correct_answer_index`, `earned_xp`, `total_xp`, `level`, `level_up`, `new_badges`, `unlocked_nodes` |
| `userApi.getPersonas()` | `GET /api/users/keywords/personas` | 온보딩 페르소나 | `contents[]` |
| `userApi.getMacroKeywords(persona)` | `GET /api/users/keywords/macro?persona={persona}` | 온보딩 대분류/중분류 선택 | `contents[]` |
| `userApi.getSpecificKeywords(macroKeywordId)` | `GET /api/users/keywords/specific?macroKeywordId={id}` | 온보딩 세부 키워드 | `contents[]` |
| scrap API | `GET /api/users/scraps/keywords/nodes`, `GET /api/users/scraps/keywords/summaries` | 스크랩 보관함 | `content[]` |

프론트 기대 타입:

- `GraphNodeResponse`: `id`, `label`, `score`, `summary`, `persona`, optional `unlock_level`, `visibility`, `explored`, `scrapped`, `news_count`
- `GraphDataResponse`: `nodes`, `edges`
- `NodeSummaryResponse`: `keyword_id`, `word`, `persona`, `summary`, optional `has_next`, optional `news`
- `QuizResponse`: `quiz_id`, `question`, `options`, `source_outlet`, `source_date`, `source_url`, optional `solved`

프론트 빈 값 영향:

- `GraphScreen`은 노드 요약 API 응답의 `summary`가 비면 그래프 노드의 inline `summary`로 fallback한다.
- 둘 다 비면 상세 패널에 `요약 정보가 없습니다.`가 노출된다.
- 퀴즈 목록이 0개이면 퀴즈 버튼이 비활성화되고 `관련 뉴스가 더 모이면 이 노드의 퀴즈가 생성돼요.`가 노출된다.
- `/preview` 라우트는 `forceMock`로 mock 데이터를 사용하지만, 실제 `/graph` 시연은 백엔드 저장 데이터와 AI 모델 fallback 품질에 의존한다.

## 2. 현재 백엔드가 Python AI 모델 서버를 호출하는 API 목록

백엔드 확인 파일:

- `nodingo-backend/src/main/java/nodingo/core/ai/client/AiClient.java`
- `nodingo-backend/src/main/java/nodingo/core/ai/dto/**`
- `nodingo-backend/src/main/java/nodingo/core/batch/news/writer/NewsAiWriter.java`
- `nodingo-backend/src/main/java/nodingo/core/batch/recommend/processor/RecommendSummaryProcessor.java`
- `nodingo-backend/src/main/java/nodingo/core/quiz/service/command/QuizGenerationService.java`
- `nodingo-backend/src/main/java/nodingo/core/batch/edge/tasklet/NeighborKeywordQuizTasklet.java`
- `nodingo-backend/src/main/java/nodingo/core/graph/service/query/GraphQueryService.java`

Python AI 서버 호출 목록:

| 백엔드 Feign method | Python endpoint | 주요 호출 위치 | 용도 |
|---|---|---|---|
| `analyzeNewsBatch` | `POST /v1/news/analyze-batch` | `NewsAiWriter` | 뉴스 요약, 뉴스 임베딩, 키워드 추출, 키워드 관계 후보 |
| `buildNewsRelations` | `POST /v1/news/build-news-relations` | relation tasklet 계열 | 뉴스 임베딩 기반 뉴스 관계 |
| `initUserEmbedding` | `POST /v1/users/init-embedding` | 온보딩/유저 벡터 계열 | 관심 키워드 임베딩 평균 |
| `updateUserEmbedding` | `POST /v1/users/update-embedding` | 유저 활동 계열 | 클릭/스크랩 기반 유저 임베딩 갱신 |
| `recommendKeywords` | `POST /v1/recommend-keywords` | `KeywordRecommendService` 경유 | 후보 키워드 추천 랭킹 |
| `summarizeKeywords` | `POST /v1/recommend-keywords/summarize` | `RecommendSummaryProcessor`, `NeighborKeywordQuizTasklet`, `QuizGenerationService` | 추천/이웃 키워드 요약 |
| `getGraphPreview` | `POST /v1/graph/preview` | `GraphQueryService` | 노드/엣지 preview JSON 구성 |
| `generateQuizzes` | `POST /v1/quizzes/generate` | `QuizGenerationService` | 키워드별 퀴즈 생성 |
| `healthCheck` | `GET /health` | 헬스 체크 | 설정/상태 확인 |

백엔드 AI 서버 URL:

- `nodingo-backend/src/main/resources/application.yaml`
- `ai.server.url: https://nodingo-ai.ddns.net`

백엔드 직렬화 규격:

- `spring.jackson.property-naming-strategy: SNAKE_CASE`
- 프론트는 snake_case를 기대한다. 이 설정과 DTO 필드명/JsonProperty는 변경하면 안 된다.

## 3. Python AI 모델 endpoint/request/response schema

Python 확인 파일:

- `nodingo-ai-models/graphrag-service/app/main.py`
- `nodingo-ai-models/graphrag-service/app/schemas.py`

### `GET /health`

Response:

```json
{
  "status": "ok",
  "embedding_model": "text-embedding-3-small",
  "embedding_dim": 1536,
  "db_writes": false
}
```

### `POST /v1/news/analyze-batch`

Request:

- `user_id?: int`
- `target_date?: date`
- `news: [{ news_id, title, body, published_at?, embedding? }]`
- `existing_keywords: [{ keyword_id, word, normalized_word, embedding }]`
- `top_k_keywords: int`
- `top_k_news_relations: int`
- `min_news_relation_score: float`

Response:

- `news_results: [{ news_id, published_at?, embedding, summary, keywords }]`
- `news_results[].keywords[]`: `keyword_id`, `word`, `normalized_word`, `embedding`, `weight`, `is_new`, `aliases`, `extraction_score`, `evidence_text`, `personas`, `macro`
- `keyword_relations[]`: `subject_keyword_id`, `related_keyword_id`, `subject_normalized_word`, `related_normalized_word`, `relation_score`, `evidence_news_ids`
- `news_relations[]`: `subject_news_id`, `related_news_id`, `relation_score`

주의:

- 백엔드 `NewsBatch.Response`는 현재 `news_relations`를 DTO에 선언하지 않지만 Jackson은 알 수 없는 필드를 무시할 수 있다.
- `keyword_relations`의 normalized word 기반 매핑은 백엔드 `NewsAiWriter`가 사용하므로 절대 변경하지 않는다.

### `POST /v1/news/build-news-relations`

Request:

- `news: [{ news_id, embedding }]`
- `top_k`
- `min_score`

Response:

- `news_relations: [{ subject_news_id, related_news_id, relation_score }]`

### `POST /v1/users/init-embedding`

Request:

- `user_id`
- `interest_keywords: [{ keyword_id, word, embedding }]`

Response:

- `user_id`
- `embedding`

### `POST /v1/users/update-embedding`

Request:

- `user_id`
- `old_embedding`
- `activities: [{ type, news_id?, news_embedding?, keyword_id?, keyword_embedding?, weight }]`
- `decay`

Response:

- `user_id`
- `embedding`

### `POST /v1/recommend-keywords`

Request:

- `user_id`
- `user_embedding`
- `candidate_keywords: [{ keyword_id, word, normalized_word?, embedding, recent_importance, is_user_interest }]`
- `target_date`
- `top_k`

Response:

- `recommend_keywords: [{ user_id, keyword_id, target_date, score, summary }]`

### `POST /v1/recommend-keywords/summarize`

Request:

- `keyword: { keyword_id, word }`
- `related_news: [{ news_id, title, body }]`
- `related_keywords: [{ keyword_id, word }]`
- `target_date`
- `persona?`
- `category?`

Response:

- `keyword_id`
- `target_date`
- `summary`

### `POST /v1/graph/preview`

Request:

- `recommend_keywords: [{ keyword_id, word, score, summary?, persona? }]`
- `keyword_relations: [{ source_keyword_id, target_keyword_id, relation_score }]`

Response:

- `nodes: [{ id, label, score, summary?, persona?, unlock_level?, visibility }]`
- `edges: [{ source, target, weight }]`

### `POST /v1/quizzes/generate`

Request:

- `keyword_id`
- `word`
- `summary?`
- `related_news: [{ news_id, title, body, url? }]`
- `num_questions`

Response:

- `keyword_id`
- `quizzes: [{ question, options, answer_index, explanation, source_news_ids }]`

## 4. OpenAI API를 호출하는 모든 위치

Python 직접 호출 지점:

| 파일 | 함수 | OpenAI 호출 | 현재 실패 처리 |
|---|---|---|---|
| `app/services/embedding_service.py` | `embed_texts_openai` | `client.embeddings.create(...)` | 예외를 호출자에게 전파 |
| `app/services/keyword_service.py` | `generate_summary_with_llm` | `client.chat.completions.create(...)` | 예외 시 `body[:200] + "... (요약 실패)"` 반환 |
| `app/services/keyword_service.py` | `extract_keywords_from_news_openai` | `client.chat.completions.create(...)` | 내부 예외 시 keyword fallback 반환 |
| `app/services/keyword_service.py` | `classify_keywords_with_openai` | `client.chat.completions.create(...)` | 예외 출력 후 재발생 |
| `app/services/summary_service.py` | `generate_openai_summary` | `client.chat.completions.create(...)` | 상위 함수가 예외를 잡아 fallback summary 반환 |
| `app/services/quiz_service.py` | `generate_openai_quizzes` | `client.chat.completions.create(...)` | 상위 함수가 예외를 잡고 fallback quiz 반환 |

OpenAI client import 위치:

- `app/services/embedding_service.py`
- `app/services/keyword_service.py`
- `app/services/summary_service.py`
- `app/services/quiz_service.py`

OpenAI API key/model 설정:

- `app/config.py`
- `openai_api_key`
- `openai_model`
- `openai_embedding_model`
- `use_openai_summary`
- `use_openai_keyword_extraction`

## 5. API 비용이 발생하는 지점

비용 발생 지점:

- 뉴스 임베딩 생성: `get_news_embedding` -> `embed_text_openai` -> `embed_texts_openai`
- 신규 키워드 임베딩 생성: `resolve_keywords` -> `embed_keyword` -> `embed_text_openai`
- batch 뉴스 요약: `process_single_news` -> `generate_summary_with_llm`
- OpenAI 키워드 추출: `extract_keywords_from_news` -> `extract_keywords_from_news_openai`
- fallback 후보 키워드 후처리 분류: `extract_keywords_from_news_fallback` -> `classify_keywords_with_openai`
- 추천 키워드 요약: `generate_recommend_keyword_summary` -> `generate_openai_summary`
- 퀴즈 생성: `generate_quizzes` -> `generate_openai_quizzes`

현재 비용 위험도가 큰 지점:

- `/v1/news/analyze-batch`가 각 뉴스마다 embedding, summary, keyword extraction을 수행한다.
- fallback keyword extraction 후에도 `classify_keywords_with_openai`가 다시 OpenAI를 호출한다.
- `ThreadPoolExecutor(max_workers=30)`로 뉴스별 OpenAI 호출이 병렬 발생한다.
- `generate_summary_with_llm`은 `settings.openai_model`이 아니라 `gpt-4o-mini`를 하드코딩한다.
- OpenAI embedding cache는 process 메모리 dict만 사용하므로 서버 재시작/배치 재실행 시 재사용되지 않는다.

## 6. 현재 fallback/cache/retry 구현 여부

Python fallback 현황:

- embedding:
  - 메모리 캐시 `_openai_embedding_cache` 존재.
  - `embed_keyword`는 실패 시 zero vector 반환.
  - `get_news_embedding`은 실패 시 fallback 없이 예외 전파.
- keyword extraction:
  - OpenAI 추출 실패 시 KeyBERT/frequency fallback 존재.
  - 그러나 fallback 후보 분류에 다시 OpenAI를 호출하는 경로가 있다.
  - 모든 fallback이 실패하거나 process 전체 예외 발생 시 `keywords=[]`가 반환될 수 있다.
- batch summary:
  - `generate_summary_with_llm` 예외 시 body 앞 200자 기반 문자열 반환.
  - cost saver mode에서 summary 호출을 끄는 설정은 없다.
- recommend summary:
  - OpenAI 실패 시 deterministic fallback summary 반환.
  - cache는 없다.
- quiz:
  - OpenAI 실패 시 deterministic fallback quiz 반환.
  - cache는 없다.
- recommend keywords:
  - OpenAI 호출 없음.
  - embedding과 persona 기반 deterministic scoring.
- graph preview:
  - OpenAI 호출 없음.
  - 입력 기반 deterministic node/edge 구성.
- relation:
  - OpenAI 호출 없음.
  - keyword relation은 co-occurrence/embedding similarity 기반.
  - news relation은 embedding cosine similarity 기반.

Python retry 현황:

- OpenAI 호출에 명시적 retry/backoff/sleep 없음.

백엔드 fallback/retry 현황:

- Feign 429는 `AiClientErrorDecoder`에서 `AiRateLimitException`으로 변환.
- `newsStep`은 Spring Batch `retryLimit(3)`와 `skip(Exception.class)`가 있다.
- `RecommendSummaryProcessor`는 429/예외 시 `return null`이라 해당 추천 요약 저장이 누락될 수 있다.
- `QuizGenerationService`는 429/예외 시 로그만 남기고 퀴즈 저장 없이 종료한다.
- `GraphQueryService`의 `graph.preview` 실패는 예외를 다시 던진다. 다만 Python `graph.preview` 자체는 OpenAI를 쓰지 않는다.

## 7. 429 insufficient_quota 발생 시 API 전체가 죽는지 여부

현재 위험 판정:

- `/v1/news/analyze-batch`:
  - `get_news_embedding`에서 OpenAI embedding 429/insufficient_quota가 발생하면 `process_single_news`의 outer `except`로 들어간다.
  - API 전체가 반드시 500으로 죽지는 않을 수 있지만, 해당 news result가 `embedding=[]`, `summary="분석 실패"`, `keywords=[]`가 된다.
  - 이후 `build_news_relations(news_results, ...)`가 빈 embedding을 검증하다가 API 전체 500으로 이어질 가능성이 있다.
  - 백엔드 `NewsAiWriter`도 빈 keyword면 DB 저장/관계 생성이 불안정하다.
- `/v1/recommend-keywords/summarize`:
  - 상위 함수가 예외를 잡고 fallback summary를 반환하므로 200 유지 가능성이 높다.
  - 단, cache와 fallback 로그가 없다.
- `/v1/quizzes/generate`:
  - 상위 함수가 예외를 잡고 fallback quizzes를 반환하므로 200 유지 가능성이 높다.
  - 단, cache와 fallback 로그가 없다.
- `/v1/recommend-keywords`:
  - OpenAI 호출이 없어 quota 이슈와 직접 무관하다.
- `/v1/graph/preview`:
  - OpenAI 호출이 없어 quota 이슈와 직접 무관하다.
- `/v1/news/build-news-relations`, user embedding endpoints:
  - OpenAI 호출 없음. 입력 embedding 차원만 유효하면 안정적이다.

결론:

- 가장 위험한 API는 `/v1/news/analyze-batch`다.
- 현재는 OpenAI 429가 API 전체 500 또는 빈 분석 결과로 이어질 수 있으므로, response schema를 유지한 채 non-empty fallback을 보장해야 한다.

## 8. response schema를 유지하면서 비용을 줄일 수 있는 수정 계획

공통 원칙:

- endpoint 경로 변경 없음.
- request/response Pydantic 모델 필드 추가/삭제/이름 변경 없음.
- 백엔드 DTO와 프론트 snake_case 계약 변경 없음.
- fallback이 발생해도 기존 응답 모델 인스턴스를 반환한다.
- 빈 `summary`, 빈 `quizzes`, 빈 `keywords`, 잘못된 embedding dimension을 반환하지 않는다.

추가 설정 계획:

```env
OPENAI_COST_SAVER_MODE=true
OPENAI_FALLBACK_ON_ERROR=true
ENABLE_ANALYSIS_CACHE=true
ENABLE_SUMMARY_CACHE=true
ENABLE_QUIZ_CACHE=true
ANALYZE_BATCH_CHUNK_SIZE=30
OPENAI_EMBED_BATCH_SIZE=30
OPENAI_REQUEST_SLEEP_SECONDS=1.5
MAX_OPENAI_BODY_CHARS=1000
```

Cost saver 동작 계획:

- `OPENAI_COST_SAVER_MODE=true`:
  - batch 단계에서 OpenAI summary 호출 중지.
  - batch 단계에서 `title + clean_body[:MAX_OPENAI_BODY_CHARS]`만 사용.
  - keyword extraction은 rule-based fallback 우선 또는 OpenAI 호출 최소화.
  - fallback 후보 분류에서 OpenAI 재호출 금지.
  - embedding은 `text-embedding-3-small`, `embedding_dim=1536` 유지.
  - embedding batch size와 sleep 적용.
  - news_id/body_hash 기반 persistent analysis cache 우선 사용.
- `OPENAI_FALLBACK_ON_ERROR=true`:
  - OpenAI 관련 예외 발생 시 deterministic fallback으로 대체하고 200 응답 유지.
  - fallback 로그 prefix: `[OPENAI_FALLBACK]`

OpenAI 예외 대상:

- `RateLimitError`
- `insufficient_quota` 메시지 포함 예외
- `APITimeoutError`
- `APIConnectionError`
- `APIError`
- 기타 OpenAI client 예외

## 9. 시연 안정성을 위해 fallback이 필요한 API 목록

필수:

- `/v1/news/analyze-batch`
  - news embedding fallback
  - keyword extraction fallback
  - keyword embedding fallback
  - summary fallback
  - relation 계산이 빈/잘못된 embedding으로 죽지 않도록 보장
- `/v1/recommend-keywords/summarize`
  - summary cache
  - keyword별 template fallback
  - 빈 related_news에서도 non-empty summary 반환
- `/v1/quizzes/generate`
  - quiz cache
  - 최소 1개 이상, 기본 3개 fallback quiz 반환
  - 각 quiz는 `question`, 4개 `options`, `answer_index`, `explanation`, `source_news_ids` 유지
- `/v1/graph/preview`
  - OpenAI는 안 쓰지만 node summary가 빈 문자열이면 template summary 주입 계획 검토
- `/v1/recommend-keywords`
  - OpenAI는 안 쓰지만 candidate/user embedding 검증 실패 시 전체 API가 죽지 않도록 fallback scoring 가능 여부 검토

보조:

- `/v1/news/build-news-relations`
  - 입력 embedding 차원 불일치 시 fake/zero vector로 대체할지 여부는 컨펌 후 결정. 현재는 계약 위반 입력을 잡는 게 맞지만 시연 안정성만 보면 방어 가능하다.

## 10. 수정할 파일 목록

컨펌 후 실제 수정 후보:

- `nodingo-ai-models/graphrag-service/app/config.py`
  - cost saver/fallback/cache/chunk/sleep/body limit 설정 추가
- `nodingo-ai-models/graphrag-service/app/services/embedding_service.py`
  - deterministic fake embedding
  - OpenAI fallback wrapper
  - embedding batch size/sleep
  - persistent embedding or analysis cache 연동 검토
- `nodingo-ai-models/graphrag-service/app/services/keyword_service.py`
  - batch chunking
  - persistent analysis cache
  - cost saver mode에서 batch summary OpenAI 호출 금지
  - rule-based keyword extraction 우선/강화
  - fallback 후보 분류에서 OpenAI 재호출 방지
  - per-news 실패 시 non-empty result 보장
- `nodingo-ai-models/graphrag-service/app/services/summary_service.py`
  - summary cache
  - template fallback summary map
  - fallback 로그
  - evidence body truncation
- `nodingo-ai-models/graphrag-service/app/services/quiz_service.py`
  - quiz cache
  - fallback quiz schema 보강
  - fallback 로그
  - evidence body truncation
- `nodingo-ai-models/graphrag-service/app/services/recommendation_service.py`
  - 필요 시 invalid/missing embedding에 대한 deterministic fallback scoring 보강
- 신규 파일 후보:
  - `app/services/cache_service.py`
  - `app/services/fallback_service.py`
  - 또는 기존 서비스 내부에 최소 변경으로 구현
- 테스트:
  - `tests/test_quiz_and_graph_contract.py`
  - `tests/test_recommend_keywords_persona.py`
  - 신규 `tests/test_openai_fallback_contract.py`
  - 신규 `tests/test_analysis_cache_contract.py`

생성될 cache 파일 경로 후보:

- `test2/cache/news_analysis_cache.json`
- `test2/cache/summary_cache.json`
- `test2/cache/quiz_cache.json`

## 11. 절대 건드리면 안 되는 파일/규격

파일:

- `nodingo-frontend/**` 전체: 참조만, 수정 금지
- `nodingo-backend/**` 전체: 참조만, 수정 금지
- Python에서도 endpoint/schema 변경을 유발하는 수정 금지:
  - `app/main.py`의 route path/method/response_model
  - `app/schemas.py`의 기존 필드명, 타입, snake_case 구조

규격:

- API endpoint 변경 금지
- request schema 변경 금지
- response schema 변경 금지
- response field 추가/삭제/이름 변경 금지
- camelCase 변환 금지
- 기존 snake_case 구조 변경 금지
- 백엔드 DTO와 맞춰둔 필드명 변경 금지
- `/v1/news/analyze-batch`의 `keyword_relations` ERD 매핑 되돌리기 금지
- `subject_keyword_id`, `related_keyword_id`, `subject_normalized_word`, `related_normalized_word`, `relation_score`, `evidence_news_ids` 구조 유지
- Python 서버에 DB 조회 로직 추가 금지
- main 브랜치 직접 수정 금지

## 12. 컨펌 후 실제 수정할 단계별 작업 계획

1. 설정 추가
   - `app/config.py`에 env flag 추가.
   - 기본값은 기존 동작 보존을 위해 cost saver/fallback/cache를 명시적으로 켤 수 있게 설정한다.

2. 공통 fallback 유틸 설계
   - OpenAI 예외 감지 함수 추가.
   - `[OPENAI_FALLBACK] ...` 로그 출력.
   - text hash/body hash 유틸 추가.
   - JSON cache read/write를 atomic하게 처리.

3. deterministic fake embedding 구현
   - SHA-256 기반 seed로 1536차원 pseudo-random vector 생성.
   - 같은 text는 항상 같은 vector.
   - L2 normalize 적용.
   - `get_news_embedding`, `embed_keyword`, batch embedding 경로에 적용.

4. rule-based keyword extraction 강화
   - 경제/정치/기술/사회/국제/문화 기본 사전 적용.
   - title 등장 가중치, body 등장 횟수 기반 weight.
   - 아무 키워드도 없으면 title에서 의미 있는 토큰 최소 1개 생성.
   - `personas`, `macro`, `aliases`, `embedding` 모두 non-empty로 채운다.

5. `/v1/news/analyze-batch` 안정화
   - `ANALYZE_BATCH_CHUNK_SIZE` 단위 처리.
   - cache hit이면 OpenAI 재호출 없이 결과 사용.
   - chunk마다 cache 저장.
   - cost saver mode에서는 summary OpenAI 호출 금지.
   - OpenAI 실패 시 해당 news result를 non-empty fallback으로 반환.
   - relation 계산 전에 모든 embedding dimension 유효성 보장.

6. summary cache/fallback
   - fallback은 사용자가 미리 모든 키워드별 문장을 만들어두는 것이 아니라, 코드의 키워드 사전과 템플릿 생성기가 즉석에서 만든다.
   - 조회 순서: exact cache -> keyword_id loose cache -> normalized word loose cache -> cost saver mode가 아니면 OpenAI 시도 -> template fallback 생성.
   - exact key 후보: `keyword_id + related_news_ids + body_hash + persona/category`.
   - loose key 후보: `keyword_id`, `normalized_word`.
   - OpenAI 성공 summary와 template fallback summary 모두 cache에 저장한다.
   - keyword별 template map과 일반 template을 함께 적용한다.
   - related_news가 없어도 non-empty summary 반환.

7. quiz cache/fallback
   - fallback은 사용자가 직접 퀴즈를 준비하는 것이 아니라, 코드의 템플릿 생성기가 키워드/요약/관련 뉴스 제목을 바탕으로 즉석 생성한다.
   - 조회 순서: exact cache -> keyword_id loose cache -> normalized word loose cache -> cost saver mode가 아니면 OpenAI 시도 -> template fallback quiz 생성.
   - exact key 후보: `keyword_id + related_news_ids + body_hash + num_questions`.
   - loose key 후보: `keyword_id + num_questions`, `normalized_word + num_questions`.
   - OpenAI 성공 quiz와 template fallback quiz 모두 cache에 저장한다.
   - 기본 3개 quiz 생성.
   - 각 quiz의 `options` 4개, `answer_index` 0~3, `explanation` non-empty 보장.

7-1. cache hit 확률 보완
   - exact cache만 사용하면 관련 뉴스 순서나 body hash가 조금만 달라도 miss가 날 수 있다.
   - 따라서 시연 안정성 목적의 summary/quiz에는 loose cache를 함께 사용한다.
   - loose cache는 정확성은 조금 낮아질 수 있지만, 같은 키워드에서 반복 클릭 시 비용과 지연을 줄인다.
   - cache miss가 나더라도 fallback generator가 즉시 non-empty 결과를 만들기 때문에 화면이 비지 않는다.
   - 필요하면 시연 전 추천 키워드 목록에 대해 summary/quiz fallback을 미리 생성하는 prewarm 단계를 추가할 수 있다.

8. recommend/graph 안정성 점검
   - `/v1/recommend-keywords`는 OpenAI 호출이 없으므로 기존 deterministic scoring 유지.
   - missing/invalid candidate embedding에 대한 fallback 필요 여부만 최소 보강.
   - `/v1/graph/preview`는 summary가 비어 있으면 template summary를 채우는 방안을 검토하되 schema는 변경하지 않는다.

9. 테스트 추가/실행
   - OpenAI API key 없이도 analyze-batch, summarize, generate quiz가 200 계약 형태로 동작하는 테스트.
   - insufficient_quota/RateLimitError mock 시 fallback 반환 테스트.
   - keyword_relations response schema 유지 테스트.
   - quiz/graph contract 기존 테스트 유지.

10. 검증 기준
    - `news_results[].embedding` 길이 1536.
    - `news_results[].summary` non-empty.
    - `news_results[].keywords` 최소 1개 이상.
    - `keyword_relations` 기존 필드 유지.
    - summary response `summary` non-empty.
    - quiz response `quizzes` non-empty, 각 quiz option 4개.
    - 프론트가 기대하는 snake_case 응답 유지.

## 현재 결론

- OpenAI 비용/쿼터 문제의 핵심 위험은 `/v1/news/analyze-batch`다.
- 요약/퀴즈 endpoint에는 이미 기본 fallback이 있으나 cache와 명시적 fallback 로그가 없다.
- 추천/그래프/관계 계산은 대체로 OpenAI 없이 동작한다.
- 시연 안정성을 위해 Python AI 모델 서버에서 schema를 그대로 유지한 200 fallback 응답을 보장하는 방식으로 수정하는 것이 가장 안전하다.

## 컨펌 요청

위 계획대로 진행해도 되는지 확인이 필요합니다.

승인 후에는 `nodingo-ai-models/graphrag-service` 내부 코드만 수정하고, `nodingo-frontend`와 `nodingo-backend`는 계속 참조만 하겠습니다.
