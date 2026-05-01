from app.config import get_settings
from app.schemas import UserActivityInput
from app.services.embedding_service import normalize_embedding, validate_embedding_dim
from app.utils.vector_utils import weighted_average, zero_vector


def initialize_user_embedding(interest_keyword_embeddings: list[list[float]]) -> list[float]:
    """Initialize a user embedding from onboarding interest keyword embeddings."""

    settings = get_settings()
    if not interest_keyword_embeddings:
        return zero_vector(settings.embedding_dim)
    vectors = [validate_embedding_dim(vector) for vector in interest_keyword_embeddings]
    return normalize_embedding(weighted_average(vectors, [1.0 for _ in vectors]))


def update_user_embedding(
    old_embedding: list[float],
    activities: list[UserActivityInput],
    decay: float,
) -> list[float]:
    """Update user embedding from old embedding and weighted activity signals."""

    vectors = [validate_embedding_dim(old_embedding)]
    weights = [max(0.0, decay)]

    for activity in activities:
        activity_embedding = get_activity_embedding(activity)
        if not activity_embedding:
            continue
        signal_multiplier = 1.25 if activity.type == "SCRAP" else 0.75
        vectors.append(validate_embedding_dim(activity_embedding))
        weights.append(max(0.0, activity.weight) * signal_multiplier)

    return normalize_embedding(weighted_average(vectors, weights))


def get_activity_embedding(activity: UserActivityInput) -> list[float] | None:
    """Pick the embedding attached to a user activity."""

    if activity.type == "SCRAP":
        return activity.news_embedding
    if activity.type == "CLICK":
        return activity.keyword_embedding
    return None
