import argparse
import html
import json
from pathlib import Path


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clip a numeric value into a display-safe range."""

    return max(low, min(high, float(value)))


def build_graph_data(result: dict, max_nodes: int, max_edges: int) -> dict:
    """Build node and edge data from analyze-batch output."""

    node_map: dict[str, dict] = {}
    for news_result in result.get("news_results", []):
        news_id = news_result.get("news_id")
        for keyword in news_result.get("keywords", []):
            node_id = keyword.get("normalized_word") or keyword.get("word")
            if not node_id:
                continue
            node = node_map.setdefault(
                node_id,
                {
                    "id": node_id,
                    "label": keyword.get("word") or node_id,
                    "score": 0.0,
                    "count": 0,
                    "news_ids": set(),
                },
            )
            node["score"] = max(node["score"], clip(keyword.get("weight", 0.0)))
            node["count"] += 1
            node["news_ids"].add(news_id)

    ranked_nodes = sorted(
        node_map.values(),
        key=lambda item: (-item["score"], -item["count"], item["label"]),
    )[:max_nodes]
    visible_ids = {node["id"] for node in ranked_nodes}

    edges = []
    for relation in result.get("keyword_relations", []):
        source = relation.get("subject_normalized_word") or relation.get("source_normalized_word")
        target = relation.get("related_normalized_word") or relation.get("target_normalized_word")
        if not source or not target or source == target:
            continue
        if source not in visible_ids or target not in visible_ids:
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "score": clip(relation.get("relation_score", 0.0)),
                "evidence_news_ids": relation.get("evidence_news_ids", []),
            }
        )

    edges = sorted(edges, key=lambda item: -item["score"])[:max_edges]
    connected_ids = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
    nodes = [
        {
            "id": node["id"],
            "label": node["label"],
            "score": round(node["score"], 4),
            "count": node["count"],
            "news_ids": sorted(value for value in node["news_ids"] if value is not None),
        }
        for node in ranked_nodes
        if node["id"] in connected_ids
    ]
    return {"nodes": nodes, "edges": edges}


def build_html(graph: dict, title: str) -> str:
    """Render a standalone HTML document with an interactive SVG graph."""

    graph_json = json.dumps(graph, ensure_ascii=False)
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --ink: #17202a;
      --muted: #667085;
      --line: #d7dde8;
      --panel: #ffffff;
      --accent: #2563eb;
      --accent-2: #0f9f7a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
      overflow: hidden;
    }}
    header {{
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .stats {{
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    main {{
      height: calc(100vh - 64px);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
      background:
        linear-gradient(90deg, rgba(23, 32, 42, 0.04) 1px, transparent 1px),
        linear-gradient(rgba(23, 32, 42, 0.04) 1px, transparent 1px);
      background-size: 36px 36px;
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin: 0 0 18px;
    }}
    .details {{
      border-top: 1px solid var(--line);
      padding-top: 16px;
      font-size: 14px;
      line-height: 1.6;
    }}
    .details h2 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .details dl {{
      margin: 0;
    }}
    .details dt {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .details dd {{
      margin: 2px 0 0;
      word-break: break-word;
    }}
    .edge {{
      stroke: #9aa7bd;
      stroke-linecap: round;
    }}
    .node circle {{
      stroke: #ffffff;
      stroke-width: 2;
      cursor: grab;
    }}
    .node text {{
      font-size: 12px;
      fill: var(--ink);
      paint-order: stroke;
      stroke: rgba(246, 248, 251, 0.9);
      stroke-width: 4px;
      stroke-linejoin: round;
      pointer-events: none;
    }}
    .node.selected circle {{
      stroke: #111827;
      stroke-width: 3;
    }}
    @media (max-width: 820px) {{
      main {{
        grid-template-columns: 1fr;
        grid-template-rows: minmax(0, 1fr) 220px;
      }}
      aside {{
        border-left: 0;
        border-top: 1px solid var(--line);
      }}
      header {{
        align-items: flex-start;
        flex-direction: column;
        height: auto;
        min-height: 78px;
        gap: 8px;
        padding: 12px 16px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div class="stats">
      <span id="node-count"></span>
      <span id="edge-count"></span>
      <span>drag nodes to rearrange</span>
    </div>
  </header>
  <main>
    <svg id="graph" role="img" aria-label="keyword relation graph"></svg>
    <aside>
      <p class="hint">노드를 클릭하면 키워드 weight, 등장 뉴스, 연결 관계를 확인할 수 있습니다. 선이 두꺼울수록 relation_score가 높습니다.</p>
      <section class="details" id="details">
        <h2>선택된 키워드 없음</h2>
        <p class="hint">그래프의 원을 클릭하세요.</p>
      </section>
    </aside>
  </main>
  <script>
    const graph = {graph_json};
    const svg = document.getElementById("graph");
    const details = document.getElementById("details");
    document.getElementById("node-count").textContent = `${{graph.nodes.length}} nodes`;
    document.getElementById("edge-count").textContent = `${{graph.edges.length}} edges`;

    const width = () => svg.clientWidth || 900;
    const height = () => svg.clientHeight || 650;
    const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
    graph.edges = graph.edges.filter((edge) => nodeById.has(edge.source) && nodeById.has(edge.target));

    function initPositions() {{
      const cx = width() / 2;
      const cy = height() / 2;
      const radius = Math.min(width(), height()) * 0.34;
      graph.nodes.forEach((node, index) => {{
        const angle = (Math.PI * 2 * index) / Math.max(1, graph.nodes.length);
        node.x = cx + Math.cos(angle) * radius;
        node.y = cy + Math.sin(angle) * radius;
        node.vx = 0;
        node.vy = 0;
      }});
    }}

    function nodeRadius(node) {{
      return 12 + node.score * 20 + Math.min(8, node.count * 2);
    }}

    function color(node) {{
      const t = Math.max(0, Math.min(1, node.score));
      const r = Math.round(37 * (1 - t) + 15 * t);
      const g = Math.round(99 * (1 - t) + 159 * t);
      const b = Math.round(235 * (1 - t) + 122 * t);
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function tick() {{
      const cx = width() / 2;
      const cy = height() / 2;
      for (const node of graph.nodes) {{
        node.vx += (cx - node.x) * 0.0008;
        node.vy += (cy - node.y) * 0.0008;
      }}
      for (const edge of graph.edges) {{
        const a = nodeById.get(edge.source);
        const b = nodeById.get(edge.target);
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const target = 145 - edge.score * 55;
        const force = (dist - target) * 0.006;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }}
      for (let i = 0; i < graph.nodes.length; i++) {{
        for (let j = i + 1; j < graph.nodes.length; j++) {{
          const a = graph.nodes[i];
          const b = graph.nodes[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist2 = dx * dx + dy * dy || 1;
          const force = Math.min(1.8, 1200 / dist2);
          const dist = Math.sqrt(dist2);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          a.vx -= fx;
          a.vy -= fy;
          b.vx += fx;
          b.vy += fy;
        }}
      }}
      for (const node of graph.nodes) {{
        if (!node.dragging) {{
          node.vx *= 0.84;
          node.vy *= 0.84;
          node.x += node.vx;
          node.y += node.vy;
          const margin = nodeRadius(node) + 10;
          node.x = Math.max(margin, Math.min(width() - margin, node.x));
          node.y = Math.max(margin, Math.min(height() - margin, node.y));
        }}
      }}
    }}

    function render() {{
      svg.innerHTML = "";
      const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      svg.append(edgeGroup, nodeGroup);

      for (const edge of graph.edges) {{
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.classList.add("edge");
        line.dataset.source = edge.source;
        line.dataset.target = edge.target;
        line.setAttribute("stroke-width", String(1 + edge.score * 5));
        line.setAttribute("opacity", String(0.25 + edge.score * 0.6));
        edgeGroup.appendChild(line);
      }}

      for (const node of graph.nodes) {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.classList.add("node");
        group.dataset.id = node.id;
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("r", String(nodeRadius(node)));
        circle.setAttribute("fill", color(node));
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("dy", String(nodeRadius(node) + 17));
        text.textContent = node.label.length > 16 ? node.label.slice(0, 15) + "..." : node.label;
        group.append(circle, text);
        group.addEventListener("click", () => selectNode(node.id));
        group.addEventListener("pointerdown", (event) => startDrag(event, node));
        nodeGroup.appendChild(group);
      }}
    }}

    function updateSvg() {{
      for (const line of svg.querySelectorAll(".edge")) {{
        const a = nodeById.get(line.dataset.source);
        const b = nodeById.get(line.dataset.target);
        line.setAttribute("x1", a.x);
        line.setAttribute("y1", a.y);
        line.setAttribute("x2", b.x);
        line.setAttribute("y2", b.y);
      }}
      for (const group of svg.querySelectorAll(".node")) {{
        const node = nodeById.get(group.dataset.id);
        group.setAttribute("transform", `translate(${{node.x}},${{node.y}})`);
      }}
    }}

    function selectNode(id) {{
      // This is only for the standalone test HTML node info panel.
      // In production, the frontend should call a Spring API on keyword click.
      // Spring then calls Python /v1/recommend-keywords/summarize; this page does not call Python directly.
      const node = nodeById.get(id);
      for (const group of svg.querySelectorAll(".node")) {{
        group.classList.toggle("selected", group.dataset.id === id);
      }}
      const connected = graph.edges
        .filter((edge) => edge.source === id || edge.target === id)
        .sort((a, b) => b.score - a.score)
        .slice(0, 12)
        .map((edge) => {{
          const otherId = edge.source === id ? edge.target : edge.source;
          const other = nodeById.get(otherId);
          return `<dd>${{escapeHtml(other.label)}} <strong>${{edge.score.toFixed(3)}}</strong></dd>`;
        }})
        .join("");
      details.innerHTML = `
        <h2>${{escapeHtml(node.label)}}</h2>
        <dl>
          <dt>normalized_word</dt><dd>${{escapeHtml(node.id)}}</dd>
          <dt>weight</dt><dd>${{node.score.toFixed(3)}}</dd>
          <dt>news_ids</dt><dd>${{node.news_ids.join(", ") || "-"}}</dd>
          <dt>relations</dt>${{connected || "<dd>-</dd>"}}
        </dl>
      `;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function startDrag(event, node) {{
      node.dragging = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      const move = (moveEvent) => {{
        const rect = svg.getBoundingClientRect();
        node.x = moveEvent.clientX - rect.left;
        node.y = moveEvent.clientY - rect.top;
        node.vx = 0;
        node.vy = 0;
        updateSvg();
      }};
      const up = () => {{
        node.dragging = false;
        event.currentTarget.removeEventListener("pointermove", move);
        event.currentTarget.removeEventListener("pointerup", up);
      }};
      event.currentTarget.addEventListener("pointermove", move);
      event.currentTarget.addEventListener("pointerup", up);
    }}

    initPositions();
    render();
    let steps = 0;
    function animate() {{
      if (steps < 360) {{
        tick();
        steps++;
      }}
      updateSvg();
      requestAnimationFrame(animate);
    }}
    animate();
    window.addEventListener("resize", () => {{
      initPositions();
      steps = 0;
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    """Generate a standalone HTML keyword graph from analyze-batch output."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tests/news_full_analyze_batch_result.json")
    parser.add_argument("--output", default="tests/news_full_keyword_graph.html")
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--max-edges", type=int, default=80)
    parser.add_argument("--title", default="News Keyword Relation Graph")
    args = parser.parse_args()

    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    graph = build_graph_data(result, args.max_nodes, args.max_edges)
    Path(args.output).write_text(build_html(graph, args.title), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"nodes={len(graph['nodes'])} edges={len(graph['edges'])}")


if __name__ == "__main__":
    main()
