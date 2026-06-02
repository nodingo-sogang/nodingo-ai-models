# Keyword Relation Fix Notes

## 문제 원인

백엔드 테스트에서 일부 키워드가 relation 0개로 조회됐다.

확인된 핵심 원인은 AI 모델의 기존 `build_keyword_relations()`가 같은 뉴스 안에서 동시에 추출된 키워드 쌍만 관계로 만들었기 때문이다.

예를 들어 한 뉴스에서 `감형`만 추출되면 만들 수 있는 키워드 쌍이 없어서 relation이 0개가 된다.

```text
뉴스 A -> 감형
생성 가능한 relation 없음
```

반대로 한 뉴스에서 여러 키워드가 추출되면 조합이 생긴다.

```text
뉴스 B -> 대선, 후보, 여론조사
대선-후보
대선-여론조사
후보-여론조사
```

따라서 threshold 문제가 아니라 co-occurrence 기반 relation 생성 방식의 한계가 주요 원인이다.

## AI 모델 수정 내용

수정 파일:

- `app/services/keyword_service.py`
- `tests/test_keyword_relation_similarity.py`

### 1. 키워드 추출 보강

OpenAI 키워드 추출 결과가 부족할 때 fallback 후보를 병합하도록 수정했다.

- 기사당 최소 후보 목표: `MIN_KEYWORDS_PER_NEWS = 5`
- OpenAI 결과가 부족하면 frequency/KeyBERT fallback 후보를 normalized_word 기준으로 병합
- 너무 적은 키워드만 뽑혀 relation 후보가 없어지는 상황 완화

### 2. embedding similarity 기반 relation 추가

기존 co-occurrence relation은 유지하고, relation이 부족한 키워드에 대해 embedding similarity 기반 relation을 추가했다.

상수:

```py
MIN_RELATIONS_PER_KEYWORD = 3
MAX_RELATIONS_PER_KEYWORD = 8
MIN_SIMILARITY_RELATION_SCORE = 0.35
```

새 relation score:

```text
0.55 * embedding_similarity
+ 0.20 * average_keyword_weight
+ 0.15 * same_macro_bonus
+ 0.10 * same_persona_bonus
```

이제 같은 뉴스에 함께 등장하지 않은 키워드라도 embedding이 비슷하고 persona/macro가 가까우면 relation 후보가 생성된다.

### 3. 테스트 추가

`tests/test_keyword_relation_similarity.py`를 추가했다.

검증 내용:

- 각 뉴스에 키워드가 1개씩만 있어 co-occurrence가 없는 경우에도 유사 embedding 키워드끼리 relation 생성
- frequency fallback이 충분한 후보를 추출하는지 확인

## 백엔드 요청사항

AI 모델 수정만으로 relation 후보는 늘어나지만, 실제 저장/그래프 조회 품질을 확인하려면 백엔드 로그와 저장 정책 보완이 필요하다.

### 1. 배치 로그 추가

아래 값을 배치마다 로그로 남겨주세요.

```text
aiResponse.getKeywordRelations().size()
relationsToSave.size()
skippedNullNormalizedCount
skippedKeywordNotFoundCount
duplicateRelationCount
```

### 2. 중복 relation 저장 처리

`keyword_relations`에는 `(subject_keyword_id, related_keyword_id)` unique constraint가 있다.

같은 relation이 다음 배치에서 다시 생성될 수 있으므로 `saveAll()`만 하면 중복 키 예외가 날 수 있다.

권장:

- 이미 존재하는 relation이면 `updateRelation(score)` 수행
- 없으면 새로 저장
- 또는 bulk upsert 사용

### 3. GraphQueryService Unknown 필터링 확인

relation이 DB에 저장되어도 graph preview 생성 시 관련 키워드가 `Unknown`으로 들어가면 `filterUnknownNodes()`에서 제거될 수 있다.

요청:

- relation으로 붙은 keyword는 `recommendMap`이 아니라 실제 `Keyword` 엔티티에서 word/persona를 가져오기
- 추천 점수가 없는 related keyword에는 기본 score를 낮게 부여하기

예:

```text
related keyword 기본 score: 0.35 ~ 0.55
```

## 프론트엔드 요청사항

AI/백엔드 수정 후에도 특정 키워드는 일시적으로 edge가 없을 수 있다. UX 방어가 필요하다.

요청:

- edge 없는 노드도 단독 노드로 표시
- 노드 상세 바텀시트에는 관련 뉴스/요약을 계속 표시
- 관련 키워드가 없는 경우 빈 화면 대신 "아직 연결을 수집 중" 같은 안내 상태 제공
- `/graph` 연동 테스트 시 API fallback mock 여부를 LIVE/MOCK 배지로 확인

## 기대 효과

- `감형`처럼 단독 등장한 키워드도 유사 키워드와 최소 relation을 갖게 된다.
- 키워드별 relation 0개 비율이 줄어든다.
- 그래프 edge 밀도가 높아지고, preview UI의 안개/해금 구조가 실제 데이터에서도 더 자연스럽게 작동한다.
