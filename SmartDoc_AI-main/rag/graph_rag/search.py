import networkx as nx
from thefuzz import process, fuzz

def search_graph(
    question,
    graph,
    node_to_texts,
    node_vector_store,
    hop=1,
    max_context=20
):
    results = node_vector_store.similarity_search_with_score(question, k=5)

    # Lấy seed node nhưng đảm bảo tồn tại trong graph
    seed_nodes = [
        str(doc.page_content)
        for doc, _ in results
        if str(doc.page_content) in graph
    ]

    # fallback nếu không có
    if not seed_nodes:
        seed_nodes = list(graph.nodes())[:3]

    visited_nodes = set()
    visited_edges = set()
    used_texts = set()
    context_parts = []

    for seed in seed_nodes:
        if seed not in graph:
            continue

        sub_nodes = nx.single_source_shortest_path_length(graph, seed, cutoff=hop)

        for node in sub_nodes:
            if node in visited_nodes:
                continue
            visited_nodes.add(node)

            # TEXT
            for text in node_to_texts.get(node, []):
                short_text = text[:300]

                if short_text not in used_texts:
                    used_texts.add(short_text)
                    context_parts.append(f"[TEXT] {node}: {short_text}")
                    break

            # GRAPH
            for neighbor in graph.neighbors(node):
                edge = tuple(sorted((node, neighbor)))  # tối ưu nhẹ

                if edge in visited_edges:
                    continue
                visited_edges.add(edge)

                rel = graph.get_edge_data(node, neighbor).get("relation", "")
                context_parts.append(f"[GRAPH] {node} [{rel}] {neighbor}")

    def score(x):
        x = x.lower()
        return sum(k.lower() in x for k in seed_nodes)

    context_parts = sorted(context_parts, key=score, reverse=True)

    return "\n".join(context_parts[:max_context])