from app.schemas import GraphEdge, GraphNode, GraphPreviewRequest, GraphPreviewResponse
from app.utils.vector_utils import clip_score

def build_graph_preview(request: GraphPreviewRequest) -> GraphPreviewResponse:
    """Build frontend graph preview nodes and edges from recommendation rows."""

    node_ids = {item.keyword_id for item in request.recommend_keywords}
    node_meta = _rank_node_visibility(request)

    nodes = [
        GraphNode(
            id=item.keyword_id,
            label=item.word,
            score=clip_score(item.score),
            summary=item.summary,
            persona=item.persona,
            unlock_level=node_meta.get(item.keyword_id, (99, "HIDDEN"))[0],
            visibility=node_meta.get(item.keyword_id, (99, "HIDDEN"))[1],
        )
        for item in request.recommend_keywords
    ]

    edges = []
    seen_edges: set[tuple[int, int]] = set()

    for relation in request.keyword_relations:
        if relation.source_keyword_id == relation.target_keyword_id:
            continue
        if relation.source_keyword_id not in node_ids or relation.target_keyword_id not in node_ids:
            continue

        source, target = sorted([relation.source_keyword_id, relation.target_keyword_id])
        key = (source, target)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        edges.append(
            GraphEdge(
                source=source,
                target=target,
                weight=clip_score(relation.relation_score),
            )
        )

    return GraphPreviewResponse(nodes=nodes, edges=edges)


def _rank_node_visibility(request: GraphPreviewRequest) -> dict[int, tuple[int, str]]:
    """Assign unlock metadata that matches the gamified graph frontend."""

    degree: dict[int, int] = {}
    strength: dict[int, float] = {}
    max_weight: dict[int, float] = {}

    for relation in request.keyword_relations:
        source = relation.source_keyword_id
        target = relation.target_keyword_id
        weight = clip_score(relation.relation_score)
        if source == target:
            continue

        for node_id in (source, target):
            degree[node_id] = degree.get(node_id, 0) + 1
            strength[node_id] = strength.get(node_id, 0.0) + weight
            max_weight[node_id] = max(max_weight.get(node_id, 0.0), weight)

    ranked = sorted(
        request.recommend_keywords,
        key=lambda item: (
            -(
                clip_score(item.score) * 2.2
                + degree.get(item.keyword_id, 0) * 0.25
                + strength.get(item.keyword_id, 0.0) * 0.45
                + max_weight.get(item.keyword_id, 0.0) * 0.8
            ),
            item.keyword_id,
        ),
    )

    result: dict[int, tuple[int, str]] = {}
    for index, item in enumerate(ranked):
        if index < 20:
            result[item.keyword_id] = (1, "VISIBLE")
        elif index < 30:
            result[item.keyword_id] = (2, "FOG")
        elif index < 40:
            result[item.keyword_id] = (3, "FOG")
        elif index < 50:
            result[item.keyword_id] = (4, "FOG")
        else:
            result[item.keyword_id] = (99, "HIDDEN")

    return result
