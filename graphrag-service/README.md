# Lightweight GraphRAG Model Server

Spring Boot 백엔드와 연동하기 위한 Python FastAPI 기반 Lightweight GraphRAG 분석 서버입니다.

이 서버는 DB에 직접 저장하지 않고, 분석 결과 JSON만 반환합니다. News, Keyword, Relation, User, RecommendKeyword 저장은 Spring Boot 백엔드가 담당합니다.

## 설치

```bash
cd graphrag-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 환경 설정

`.env.example`을 참고해 `.env`를 만듭니다.

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
USE_OPENAI_SUMMARY=true
```

기본 임베딩 모델은 `BAAI/bge-m3`, 기본 차원은 `1024`입니다. 로컬 개발이 무거우면 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`와 `EMBEDDING_DIM=384`처럼 함께 변경하세요. News, Keyword, User embedding은 반드시 같은 차원을 사용해야 하며, 서버는 실제 모델 출력 차원과 `.env` 차원이 다르면 오류를 냅니다.

OpenAI는 추천 키워드 summary 생성에만 사용됩니다. `OPENAI_API_KEY`가 없거나 호출에 실패하면 기사 title/body 기반 fallback summary를 반환합니다. 임베딩은 항상 로컬 sentence-transformers 모델을 사용합니다.

## 실행

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

문서 UI:

```text
http://localhost:8000/docs
```

## 백엔드 연동 방식

Spring Boot 백엔드는 뉴스, 기존 키워드, 유저 활동, 추천 후보 데이터를 JSON으로 이 서버에 전달합니다. 이 서버는 embedding, keyword, relation, recommendation, summary 결과를 JSON으로 반환합니다. 반환된 JSON을 PostgreSQL/JPA 엔티티에 저장하는 책임은 Spring Boot 백엔드에 있습니다.

## Endpoint 예시

### Health

```bash
curl http://localhost:8000/health
```

### POST /v1/news/analyze-batch

```bash
curl -X POST http://localhost:8000/v1/news/analyze-batch ^
  -H "Content-Type: application/json" ^
  -d "{\"news\":[{\"news_id\":1,\"title\":\"삼성전자, HBM4 양산 가속화... 엔비디아 공급 임박\",\"body\":\"삼성전자가 차세대 고대역폭 메모리인 HBM4의 양산 일정을 앞당기며 엔비디아와의 파트너십을 강화하고 있습니다.\",\"published_at\":\"2026-04-28T09:00:00\"}],\"existing_keywords\":[],\"top_k_keywords\":8}"
```

### POST /v1/news/build-news-relations

```bash
curl -X POST http://localhost:8000/v1/news/build-news-relations ^
  -H "Content-Type: application/json" ^
  -d "{\"news\":[{\"news_id\":1,\"embedding\":[0.1,0.2]},{\"news_id\":2,\"embedding\":[0.2,0.3]}],\"top_k\":5,\"min_score\":0.55}"
```

실제 호출에서는 `.env`의 `EMBEDDING_DIM`과 같은 길이의 embedding을 전달해야 합니다.

### POST /v1/users/init-embedding

```bash
curl -X POST http://localhost:8000/v1/users/init-embedding ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":1,\"interest_keywords\":[{\"keyword_id\":10,\"word\":\"엔비디아\",\"embedding\":[0.1,0.2]}]}"
```

### POST /v1/users/update-embedding

```bash
curl -X POST http://localhost:8000/v1/users/update-embedding ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":1,\"old_embedding\":[0.1,0.2],\"activities\":[{\"type\":\"SCRAP\",\"news_id\":1,\"news_embedding\":[0.2,0.3],\"weight\":0.35}],\"decay\":0.7}"
```

### POST /v1/recommend-keywords

```bash
curl -X POST http://localhost:8000/v1/recommend-keywords ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":1,\"user_embedding\":[0.1,0.2],\"candidate_keywords\":[{\"keyword_id\":10,\"word\":\"엔비디아\",\"normalized_word\":\"엔비디아\",\"embedding\":[0.2,0.3],\"recent_importance\":0.8,\"is_user_interest\":true}],\"target_date\":\"2026-04-28\",\"top_k\":12}"
```

### POST /v1/recommend-keywords/summarize

```bash
curl -X POST http://localhost:8000/v1/recommend-keywords/summarize ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":1,\"keyword\":{\"keyword_id\":10,\"word\":\"HBM4\"},\"related_news\":[{\"news_id\":1,\"title\":\"삼성전자, HBM4 양산 가속화... 엔비디아 공급 임박\",\"body\":\"삼성전자가 차세대 고대역폭 메모리인 HBM4의 양산 일정을 앞당기며...\"}],\"related_keywords\":[{\"keyword_id\":11,\"word\":\"삼성전자\"}],\"target_date\":\"2026-04-28\"}"
```

### POST /v1/graph/preview

```bash
curl -X POST http://localhost:8000/v1/graph/preview ^
  -H "Content-Type: application/json" ^
  -d "{\"recommend_keywords\":[{\"keyword_id\":10,\"word\":\"HBM4\",\"score\":0.91,\"summary\":\"...\"}],\"keyword_relations\":[{\"source_keyword_id\":10,\"target_keyword_id\":11,\"relation_score\":0.86}]}"
```

## 구현 메모

- 모든 API 응답은 백엔드 저장이 쉽도록 snake_case key를 사용합니다.
- 자기 자신 relation은 생성하지 않습니다.
- NewsRelation은 작은 `news_id`를 `subject_news_id`로 정렬합니다.
- KeywordRelation은 양쪽 모두 `keyword_id`가 있으면 작은 id를 source로 정렬하고, 신규 키워드가 있으면 `normalized_word` 기준으로 반환합니다.
- 모든 점수는 0~1 범위로 clip됩니다.
- LLM summary 프롬프트에는 기사에 없는 내용을 추측하지 말라는 지시가 포함되어 있습니다.
