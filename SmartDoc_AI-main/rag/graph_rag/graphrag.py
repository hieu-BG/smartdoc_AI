from ..splitter import split_chunks_from_pages
from ..loader import extract_content
from ..prompt import build_prompt_graphrag
import streamlit as st
from .extractgraph import extract_graph_documents
from .vectorstore import build_faiss_node_index

from .mapping import map_nodes_to_source_texts
from .networkx import build_networkx_graph
from .search import search_graph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import OllamaLLM
from ..embeddings import get_embeddings_model
from ..groq_model import get_groq_model
import os


@st.cache_resource
def get_graph_transformer():
    return LLMGraphTransformer(
        llm=get_groq_model(),
        allowed_nodes=["Concept", "Process", "Method", "Example", "Component"],
        allowed_relationships=["INCLUDES", "DESCRIBES", "IS_EXAMPLE_OF", "APPLIES_TO"],
    )


def ask_pdf(file_path, question, progress=None):
    # ── Phase 1: Indexing ─────────────────────────────────────────────────────
    if progress:
        progress(10,"Đang đọc PDF ...")
    pages = extract_content(file_path)
    if progress:
        progress(30, "Đang chia nhỏ tài liệu...")
    
    chunks = split_chunks_from_pages(pages,
    chunk_size=st.session_state.chunk_size,
    chunk_overlap=st.session_state.chunk_overlap)
    # ── Phase 2: Extract Graph Documents ──────────────────────────────────────
    if progress:
        progress(60, "Đang trích xuất tri thức...")
    graph_docs = extract_graph_documents(chunks, get_graph_transformer())
    node_mapping = map_nodes_to_source_texts(graph_docs)
    networkx_graph = build_networkx_graph(graph_docs)
    # ── Phase 3: Search Graph ─────────────────────────────────────────────────────
    if progress:
        progress(80, "Đang tìm kiếm thông tin liên quan...")
    if networkx_graph.number_of_nodes():
        node_vector_store = build_faiss_node_index(graph_docs, get_embeddings_model())
        context = search_graph(
            question,
            networkx_graph,
            node_mapping,
            node_vector_store,
            hop=1,
            max_context=20,
        )
    else:
        context = "\n\n".join(
            f"[TEXT] {chunk['source']}: {chunk['text']}"
            for chunk in chunks[:10]
            if chunk.get("text")
        )
    if progress:
        progress(90, "AI đang tạo câu trả lời...")
    # ── Phase 4: Generation ─────────────────────────────────────────────────────
    prompt = build_prompt_graphrag(context, question)
    print("Prompt:", prompt)
    # llm_answer = OllamaLLM(
    # model="qwen2.5:7b",
    # temperature=st.session_state.temperature,
    # top_p=st.session_state.top_p,
    # repeat_penalty=st.session_state.repeat_penalty
    # )
    # print("param LLM:", st.session_state.temperature, st.session_state.top_p, st.session_state.repeat_penalty)
    response = get_groq_model().invoke(prompt).content
    if progress:
        progress(100, "Hoàn thành")

    return response
