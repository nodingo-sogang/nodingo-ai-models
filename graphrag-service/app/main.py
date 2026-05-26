from fastapi import FastAPI, Request

from app.config import get_settings
from app.schemas import (
    AnalyzeNewsBatchRequest,
    AnalyzeNewsBatchResponse,
    BuildNewsRelationsRequest,
    BuildNewsRelationsResponse,
    GraphPreviewRequest,
    GraphPreviewResponse,
    InitUserEmbeddingRequest,
    RecommendKeywordsRequest,
    RecommendKeywordsResponse,
    SummarizeRecommendKeywordRequest,
    SummarizeRecommendKeywordResponse,
    UpdateUserEmbeddingRequest,
    UserEmbeddingResponse,
)
from app.services.graph_service import build_graph_preview
from app.services.keyword_service import analyze_news_batch
from app.services.recommendation_service import recommend_keywords
from app.services.relation_service import build_news_relations
from app.services.summary_service import generate_recommend_keyword_summary
from app.services.user_embedding_service import initialize_user_embedding, update_user_embedding

app = FastAPI(
    title="Lightweight GraphRAG Model Server",
    version="0.1.0",
    description="FastAPI GraphRAG analysis server that returns JSON only and never writes to DB.",
)


@app.get("/health")
def health() -> dict:
    """Return service health and configured embedding metadata."""

    settings = get_settings()
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "db_writes": False,
    }


@app.post("/v1/news/analyze-batch", response_model=AnalyzeNewsBatchResponse)
def analyze_news_batch_endpoint(request: AnalyzeNewsBatchRequest) -> AnalyzeNewsBatchResponse:
    """Analyze a batch of news and return JSON for backend persistence."""

    return analyze_news_batch(request)


@app.post("/v1/news/build-news-relations", response_model=BuildNewsRelationsResponse)
def build_news_relations_endpoint(request: BuildNewsRelationsRequest) -> BuildNewsRelationsResponse:
    """Build news-to-news similarity relations from backend-provided embeddings."""

    return BuildNewsRelationsResponse(
        news_relations=build_news_relations(request.news, request.top_k, request.min_score)
    )


@app.post("/v1/users/init-embedding", response_model=UserEmbeddingResponse)
def init_user_embedding_endpoint(request: InitUserEmbeddingRequest) -> UserEmbeddingResponse:
    """Initialize user embedding from onboarding interest keywords."""

    embeddings = [item.embedding for item in request.interest_keywords]
    return UserEmbeddingResponse(
        user_id=request.user_id,
        embedding=initialize_user_embedding(embeddings),
    )


@app.post("/v1/users/update-embedding", response_model=UserEmbeddingResponse)
def update_user_embedding_endpoint(request: UpdateUserEmbeddingRequest) -> UserEmbeddingResponse:
    """Update user embedding from scrap and click activity signals."""

    return UserEmbeddingResponse(
        user_id=request.user_id,
        embedding=update_user_embedding(request.old_embedding, request.activities, request.decay),
    )


@app.post("/v1/recommend-keywords", response_model=RecommendKeywordsResponse)
async def recommend_keywords_endpoint(
    request: Request,
    payload: RecommendKeywordsRequest,
) -> RecommendKeywordsResponse:
    """Score and rank keyword recommendations without generating summaries."""

    raw_payload = await request.json()
    return RecommendKeywordsResponse(
        recommend_keywords=recommend_keywords(
            payload.user_id,
            payload.user_embedding,
            payload.candidate_keywords,
            payload.target_date,
            payload.top_k,
            raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
        )
    )


@app.post("/v1/recommend-keywords/summarize", response_model=SummarizeRecommendKeywordResponse)
def summarize_recommend_keyword_endpoint(
    request: SummarizeRecommendKeywordRequest,
) -> SummarizeRecommendKeywordResponse:
    """Generate a grounded summary for one recommended keyword."""

    summary = generate_recommend_keyword_summary(
        request.keyword,
        request.related_news,
        request.related_keywords,
        persona=request.persona,
        category=request.category,
    )
    return SummarizeRecommendKeywordResponse(
        user_id=request.user_id,
        keyword_id=request.keyword.keyword_id,
        target_date=request.target_date,
        summary=summary,
    )


@app.post("/v1/graph/preview", response_model=GraphPreviewResponse)
def graph_preview_endpoint(request: GraphPreviewRequest) -> GraphPreviewResponse:
    """Build graph preview JSON for frontend rendering."""

    return build_graph_preview(request)
