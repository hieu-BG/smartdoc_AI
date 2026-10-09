from langchain_community.vectorstores import FAISS

def build_faiss_node_index(all_graph_documents, embeddings):
    from langchain_community.vectorstores import FAISS

    nodes_list = []
    for doc in all_graph_documents:
        for node in doc.nodes:
            nodes_list.append(node.id)

    unique_nodes = list(set(nodes_list))

    node_vector_store = FAISS.from_texts(
        texts=unique_nodes,
        embedding=embeddings
    )

    print(len(unique_nodes))

    return node_vector_store