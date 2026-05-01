ANALYZE_BATCH_PAYLOAD = {
    "news": [
        {
            "news_id": 1,
            "title": "삼성전자, HBM4 양산 가속화... 엔비디아 공급 임박",
            "body": "삼성전자가 차세대 고대역폭 메모리인 HBM4의 양산 일정을 앞당기며 엔비디아와의 파트너십을 강화하고 있습니다.",
            "published_at": "2026-04-28T09:00:00",
        }
    ],
    "existing_keywords": [
        {
            "keyword_id": 10,
            "word": "엔비디아",
            "normalized_word": "엔비디아",
            "embedding": [0.0] * 1024,
        }
    ],
    "top_k_keywords": 8,
}

BUILD_NEWS_RELATIONS_PAYLOAD = {
    "news": [
        {"news_id": 1, "embedding": [0.1] * 1024},
        {"news_id": 2, "embedding": [0.2] * 1024},
    ],
    "top_k": 5,
    "min_score": 0.55,
}

INIT_USER_EMBEDDING_PAYLOAD = {
    "user_id": 1,
    "interest_keywords": [
        {"keyword_id": 10, "word": "엔비디아", "embedding": [0.1] * 1024},
        {"keyword_id": 11, "word": "반도체", "embedding": [0.2] * 1024},
    ],
}

UPDATE_USER_EMBEDDING_PAYLOAD = {
    "user_id": 1,
    "old_embedding": [0.1] * 1024,
    "activities": [
        {
            "type": "SCRAP",
            "news_id": 1,
            "news_embedding": [0.2] * 1024,
            "weight": 0.35,
        },
        {
            "type": "CLICK",
            "keyword_id": 10,
            "keyword_embedding": [0.3] * 1024,
            "weight": 0.15,
        },
    ],
    "decay": 0.7,
}

RECOMMEND_KEYWORDS_PAYLOAD = {
    "user_id": 1,
    "user_embedding": [0.1] * 1024,
    "candidate_keywords": [
        {
            "keyword_id": 10,
            "word": "엔비디아",
            "normalized_word": "엔비디아",
            "embedding": [0.2] * 1024,
            "recent_importance": 0.8,
            "is_user_interest": True,
        }
    ],
    "target_date": "2026-04-28",
    "top_k": 12,
}

SUMMARIZE_PAYLOAD = {
    "user_id": 1,
    "keyword": {"keyword_id": 10, "word": "HBM4"},
    "related_news": [
        {
            "news_id": 1,
            "title": "삼성전자, HBM4 양산 가속화... 엔비디아 공급 임박",
            "body": "삼성전자가 차세대 고대역폭 메모리인 HBM4의 양산 일정을 앞당기며 엔비디아 공급 가능성이 언급되고 있습니다.",
        }
    ],
    "related_keywords": [
        {"keyword_id": 11, "word": "삼성전자"},
        {"keyword_id": 12, "word": "엔비디아"},
    ],
    "target_date": "2026-04-28",
}

GRAPH_PREVIEW_PAYLOAD = {
    "recommend_keywords": [
        {"keyword_id": 10, "word": "HBM4", "score": 0.91, "summary": "HBM4 briefing"}
    ],
    "keyword_relations": [
        {"source_keyword_id": 10, "target_keyword_id": 11, "relation_score": 0.86}
    ],
}
