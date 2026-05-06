from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


Embedding = list[float]


class NewsInput(BaseModel):
    """News article input sent by the Spring Boot backend."""

    news_id: int
    title: str
    body: str
    published_at: datetime | None = None


class ExistingKeywordInput(BaseModel):
    """Existing keyword record used to resolve extracted keywords."""

    keyword_id: int
    word: str
    normalized_word: str
    embedding: Embedding


class AnalyzeNewsBatchRequest(BaseModel):
    """Request for batch news analysis."""

    news: list[NewsInput]
    existing_keywords: list[ExistingKeywordInput] = Field(default_factory=list)
    top_k_keywords: int = 8


class KeywordCandidate(BaseModel):
    """Candidate keyword extracted from a news article."""

    word: str
    normalized_word: str
    extraction_score: float


class KeywordResult(BaseModel):
    """Resolved keyword result returned for each news article."""

    keyword_id: int | None = None
    word: str
    normalized_word: str
    embedding: Embedding
    weight: float
    is_new: bool
    aliases: list[str]
    extraction_score: float = 0.0


class NewsAnalysisResult(BaseModel):
    """Per-news analysis output."""

    news_id: int
    embedding: Embedding
    keywords: list[KeywordResult]


class KeywordRelationResult(BaseModel):
    """Keyword relation candidate derived from co-occurrence and similarity."""

    source_keyword_id: int | None = None
    target_keyword_id: int | None = None
    source_normalized_word: str
    target_normalized_word: str
    relation_score: float
    evidence_news_ids: list[int]


class AnalyzeNewsBatchResponse(BaseModel):
    """Response for batch news analysis."""

    news_results: list[NewsAnalysisResult]
    keyword_relations: list[KeywordRelationResult]


class NewsEmbeddingInput(BaseModel):
    """News embedding input used to build news-to-news relations."""

    news_id: int
    embedding: Embedding


class BuildNewsRelationsRequest(BaseModel):
    """Request for news similarity relation generation."""

    news: list[NewsEmbeddingInput]
    top_k: int = 5
    min_score: float = 0.55


class NewsRelationResult(BaseModel):
    """News relation output sorted by smaller id as subject."""

    subject_news_id: int
    related_news_id: int
    relation_score: float


class BuildNewsRelationsResponse(BaseModel):
    """Response for news relation generation."""

    news_relations: list[NewsRelationResult]


class InterestKeywordInput(BaseModel):
    """Keyword embedding used to initialize a user embedding."""

    keyword_id: int
    word: str
    embedding: Embedding


class InitUserEmbeddingRequest(BaseModel):
    """Request to initialize user embedding from onboarding interests."""

    user_id: int
    interest_keywords: list[InterestKeywordInput] = Field(default_factory=list)


class UserEmbeddingResponse(BaseModel):
    """User embedding response."""

    user_id: int
    embedding: Embedding | None


class UserActivityInput(BaseModel):
    """User activity signal used to update user embedding."""

    type: Literal["SCRAP", "CLICK"]
    news_id: int | None = None
    news_embedding: Embedding | None = None
    keyword_id: int | None = None
    keyword_embedding: Embedding | None = None
    weight: float = 0.1


class UpdateUserEmbeddingRequest(BaseModel):
    """Request to update an existing user embedding from activity signals."""

    user_id: int
    old_embedding: Embedding
    activities: list[UserActivityInput] = Field(default_factory=list)
    decay: float = 0.7


class CandidateKeywordInput(BaseModel):
    """Candidate keyword used for recommendation scoring."""

    keyword_id: int
    word: str
    normalized_word: str | None = None
    embedding: Embedding
    recent_importance: float = 0.0
    is_user_interest: bool = False


class RecommendKeywordsRequest(BaseModel):
    """Request to score and rank keyword recommendations."""

    user_id: int
    user_embedding: Embedding
    candidate_keywords: list[CandidateKeywordInput]
    target_date: date
    top_k: int = 12


class RecommendKeywordResult(BaseModel):
    """Recommended keyword output for backend persistence."""

    user_id: int
    keyword_id: int
    target_date: date
    score: float
    summary: str | None = None


class RecommendKeywordsResponse(BaseModel):
    """Response for recommended keyword scoring."""

    recommend_keywords: list[RecommendKeywordResult]


class SummaryKeywordInput(BaseModel):
    """Keyword to summarize for a user-facing briefing."""

    keyword_id: int
    word: str


class SummaryNewsInput(BaseModel):
    """Related news evidence used for summary generation."""

    news_id: int
    title: str
    body: str


class SummaryRelatedKeywordInput(BaseModel):
    """Related keyword context used for summary generation."""

    keyword_id: int
    word: str


class SummarizeRecommendKeywordRequest(BaseModel):
    """Request to generate a grounded recommendation summary."""

    user_id: int
    keyword: SummaryKeywordInput
    related_news: list[SummaryNewsInput] = Field(default_factory=list)
    related_keywords: list[SummaryRelatedKeywordInput] = Field(default_factory=list)
    target_date: date


class SummarizeRecommendKeywordResponse(BaseModel):
    """Response containing a user-facing keyword briefing summary."""

    user_id: int
    keyword_id: int
    target_date: date
    summary: str


class GraphRecommendKeywordInput(BaseModel):
    """Recommended keyword node input for graph preview."""

    keyword_id: int
    word: str
    score: float
    summary: str | None = None
    persona: str | None = None


class GraphKeywordRelationInput(BaseModel):
    """Keyword relation edge input for graph preview."""

    source_keyword_id: int
    target_keyword_id: int
    relation_score: float


class GraphPreviewRequest(BaseModel):
    """Request to build frontend graph preview JSON."""

    recommend_keywords: list[GraphRecommendKeywordInput]
    keyword_relations: list[GraphKeywordRelationInput]


class GraphNode(BaseModel):
    """Frontend graph node."""

    id: int
    label: str
    score: float
    summary: str | None = None
    persona: str | None = None


class GraphEdge(BaseModel):
    """Frontend graph edge."""

    source: int
    target: int
    weight: float


class GraphPreviewResponse(BaseModel):
    """Frontend graph preview response."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
