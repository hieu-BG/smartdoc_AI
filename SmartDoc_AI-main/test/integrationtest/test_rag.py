import pytest
from unittest.mock import patch, MagicMock
import streamlit as st

from rag.rag import ask_pdf


# =========================
# SETUP session_state
# =========================
@pytest.fixture(autouse=True)
def setup_session_state():
    st.session_state.chunk_size = 300
    st.session_state.chunk_overlap = 50
    st.session_state.temperature = 0.3
    st.session_state.top_p = 0.8
    st.session_state.repeat_penalty = 1.1


# =========================
# TEST CASE 1: Simple Factual
# =========================
@patch("rag.rag.FAISS")
@patch("rag.rag.OllamaLLM")
def test_tc01_simple_factual(mock_llm, mock_faiss):

    # Mock FAISS
    mock_doc = MagicMock()
    mock_doc.page_content = "Step 1: Install software"

    mock_db = MagicMock()
    mock_faiss.from_texts.return_value = mock_db
    mock_db.similarity_search.return_value = [mock_doc]

    # Mock LLM
    mock_llm_instance = MagicMock()
    mock_llm.return_value = mock_llm_instance
    mock_llm_instance.invoke.return_value = "Step 1: Install software"


    # Input
    file_path = "data/temp_rag_doc.pdf"
    question = "What is the installation procedure?"

    response = ask_pdf(file_path, question)

    # Assert
    assert isinstance(response, str)
    assert "install" in response.lower()

    mock_llm_instance.invoke.assert_called_once()


# =========================
# TEST CASE 2: Complex Reasoning
# =========================
@patch("rag.rag.FAISS")
@patch("rag.rag.OllamaLLM")
def test_tc02_complex_reasoning(mock_llm, mock_faiss):

    # Mock multiple chunks
    doc1 = MagicMock()
    doc1.page_content = "Finding 1: Model improves accuracy"

    doc2 = MagicMock()
    doc2.page_content = "Implication: Useful in real-world applications"

    mock_db = MagicMock()
    mock_db.similarity_search.return_value = [doc1, doc2]

    mock_faiss.from_texts.return_value = mock_db

    # Mock LLM
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Main findings show improvement and practical implications"
    mock_llm.return_value = mock_llm_instance

    file_path = "data/temp_rag_doc.pdf"
    question = "What are the main findings and their implications?"

    response = ask_pdf(file_path, question)

    # Assert
    assert isinstance(response, str)
    assert any(word in response.lower() for word in ["finding", "implication", "improve"])

    mock_llm_instance.invoke.assert_called_once()


# =========================
# TEST CASE 3: Out-of-context
# =========================
@patch("rag.rag.FAISS")
@patch("rag.rag.OllamaLLM")
def test_tc03_out_of_context(mock_llm, mock_faiss):

    # Mock irrelevant chunks
    mock_doc = MagicMock()
    mock_doc.page_content = "How to cook rice"

    mock_db = MagicMock()
    mock_db.similarity_search.return_value = [mock_doc]

    mock_faiss.from_texts.return_value = mock_db

    # Mock LLM trả fallback
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "I could not find the information in the document."
    mock_llm.return_value = mock_llm_instance

    file_path = "data/temp_rag_doc.pdf"
    question = "How to solve differential equations?"

    response = ask_pdf(file_path, question)

    # Assert
    assert isinstance(response, str)
    assert "could not find" in response.lower()

    mock_llm_instance.invoke.assert_called_once()