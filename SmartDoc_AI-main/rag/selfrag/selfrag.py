import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ..embeddings import get_embeddings_model
from ..groq_model import get_groq_model

from ..splitter import split_chunks_from_pages
from ..loader import extract_content
from ..prompt import build_prompt

from .searchloop import run_corag_loop
from ..rag import get_or_create_vector_db


def _build_faiss(chunks: list[dict]) -> FAISS:
    docs = [
        Document(
            page_content=c["text"],
            metadata={"source": c["source"]},
        )
        for c in chunks
    ]
    return FAISS.from_documents(docs, get_embeddings_model())


def ask_pdf_selfrag(
    file_path: str,
    question: str,
    progress=None,
    max_hops: int = 3,
    k: int = 3,
) -> str:
    # ── Phase 1: Indexing ─────────────────────────────────────────────────────
    llm = get_groq_model()

    db = get_or_create_vector_db(file_path, get_embeddings_model(), progress=progress)

    # ── Phase 2: Chain-of-Retrieval loop ──────────────────────────────────────
    context_buffer = run_corag_loop(
        db=db,
        llm=llm,
        question=question,
        max_hops=max_hops,
        k=k,
        progress=progress,
    )

    # ── Phase 3: Generation ───────────────────────────────────────────────────
    if progress:
        progress(85, "AI đang tạo câu trả lời...")

    final_context = "\n\n".join(context_buffer)
    prompt = build_prompt(final_context, question)

    print(f"[CoRAG] Tổng context: {len(context_buffer)} đoạn | {len(prompt)} chars")

    response = llm.invoke(prompt).content

    if progress:
        progress(100, "Hoàn thành")

    return response