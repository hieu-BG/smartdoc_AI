from typing import List, Dict
from langchain_core.documents import Document

def map_nodes_to_source_texts(all_graph_documents: List) -> Dict[str, List[str]]:
    node_to_texts = {}

    for doc in all_graph_documents:
        # Lấy nội dung văn bản gốc của chunk
        source_text = getattr(doc.source, "page_content", "")
        
        if not source_text:
            continue

        for node in doc.nodes:
            # Chuẩn hóa ID của node (viết thường hoặc strip để tránh lệch tên)
            node_id = node.id.strip()

            if node_id not in node_to_texts:
                node_to_texts[node_id] = []

            # Tránh thêm trùng lặp văn bản trong cùng một node
            if source_text not in node_to_texts[node_id]:
                node_to_texts[node_id].append(source_text)

    return node_to_texts