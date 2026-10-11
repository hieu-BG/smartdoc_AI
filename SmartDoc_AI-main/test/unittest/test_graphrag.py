from unittest.mock import MagicMock, patch

import streamlit as st

from rag.graph_rag.graphrag import ask_pdf


@patch("rag.graph_rag.graphrag.get_groq_model")
@patch("rag.graph_rag.graphrag.search_graph")
@patch("rag.graph_rag.graphrag.build_faiss_node_index")
@patch("rag.graph_rag.graphrag.get_embeddings_model")
@patch("rag.graph_rag.graphrag.extract_graph_documents", return_value=[])
@patch("rag.graph_rag.graphrag.get_graph_transformer")
@patch("rag.graph_rag.graphrag.split_chunks_from_pages")
@patch("rag.graph_rag.graphrag.extract_content")
def test_ask_pdf_uses_chunk_text_when_graph_has_no_nodes(
    mock_extract_content,
    mock_split_chunks,
    _mock_transformer,
    _mock_extract_graph_documents,
    mock_get_embeddings,
    mock_build_index,
    mock_search_graph,
    mock_get_groq_model,
):
    st.session_state.chunk_size = 500
    st.session_state.chunk_overlap = 50
    mock_extract_content.return_value = [{"source": "document.pdf", "text": "source"}]
    mock_split_chunks.return_value = [
        {"source": "document.pdf", "text": "Graph extraction yielded no entities."}
    ]
    llm = MagicMock()
    llm.invoke.return_value.content = "Answer from source text."
    mock_get_groq_model.return_value = llm

    response = ask_pdf("document.pdf", "What is in the document?")

    assert response == "Answer from source text."
    prompt = llm.invoke.call_args.args[0]
    assert "[TEXT] document.pdf: Graph extraction yielded no entities." in prompt
    mock_get_embeddings.assert_not_called()
    mock_build_index.assert_not_called()
    mock_search_graph.assert_not_called()
