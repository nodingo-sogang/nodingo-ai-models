from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    use_openai_summary: bool = True
    use_openai_keyword_extraction: bool = True
    openai_cost_saver_mode: bool = False
    openai_fallback_on_error: bool = True
    enable_analysis_cache: bool = True
    enable_summary_cache: bool = True
    enable_quiz_cache: bool = True
    analyze_batch_chunk_size: int = 30
    openai_embed_batch_size: int = 30
    openai_request_sleep_seconds: float = 0.0
    max_openai_body_chars: int = 1000
    analysis_cache_path: str = "test2/cache/news_analysis_cache.json"
    summary_cache_path: str = "test2/cache/summary_cache.json"
    quiz_cache_path: str = "test2/cache/quiz_cache.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
