import networkx as nx
from typing import List

def build_networkx_graph(all_graph_documents: List):
    # Khởi tạo đồ thị vô hướng (hoặc nx.DiGraph() nếu bạn muốn đồ thị có hướng)
    G = nx.Graph()

    for doc in all_graph_documents:
        # Thêm các nút (Nodes)
        for node in doc.nodes:
            # node.id thường là tên thực thể, node.type là nhãn (Concept, Process,...)
            G.add_node(node.id, label=node.type)

        # Thêm các cạnh (Relationships/Edges)
        for edge in doc.relationships:
            # edge.source và edge.target là các đối tượng node, ta lấy .id
            G.add_edge(
                edge.source.id, 
                edge.target.id, 
                relation=edge.type
            )

    return G