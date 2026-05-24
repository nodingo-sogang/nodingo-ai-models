from app.schemas import GraphEdge, GraphNode, GraphPreviewRequest, GraphPreviewResponse
from app.utils.vector_utils import clip_score

def build_graph_preview(request: GraphPreviewRequest) -> GraphPreviewResponse:
    """Build frontend graph preview nodes and edges from recommendation rows."""

    node_ids = {item.keyword_id for item in request.recommend_keywords}

    nodes = [
        GraphNode(
            id=item.keyword_id,
            label=item.word,
            score=clip_score(item.score),
            summary=item.summary,
            persona=item.persona
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
