from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
import streamlit as st
from .embeddings import get_embeddings_model
from .splitter import split_chunks_from_pages
from .loader import extract_content
from .prompt import build_prompt
from langchain_core.documents import Document

import os
import hashlib
from langchain_community.vectorstores import FAISS

# Thư mục gốc lưu trữ các index local
INDEX_SAVE_PATH = "storage/vector_indices"

def get_or_create_vector_db(file_path, embeddings_model, progress=None):
    # 1. Tạo ID duy nhất dựa trên nội dung file (MD5 Hash)
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    file_id = hasher.hexdigest()
    
    # Đường dẫn thư mục lưu index của riêng file này
    save_dir = os.path.join(INDEX_SAVE_PATH, file_id)
    
    # 2. Kiểm tra xem Index đã tồn tại chưa
    if os.path.exists(save_dir):
        if progress:
            progress(50, "Đã tìm thấy bản lưu local, đang tải dữ liệu...")
        
        # Load index từ ổ cứng
        db = FAISS.load_local(
            save_dir, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        return db

    # 3. Nếu chưa có, tiến hành quy trình xử lý tài liệu từ đầu
    if progress:
        progress(10, "Tài liệu mới, đang bắt đầu đọc file...")
    
    # Đọc nội dung
    pages = extract_content(file_path)

    if progress:
        progress(30, "Đang chia nhỏ tài liệu...")
    
    # Chia nhỏ chunk
    chunks = split_chunks_from_pages(
        pages,
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap
    )

    # Chuyển thành LangChain Documents
    docs = [
        Document(page_content=chunk["text"], metadata={"source": chunk["source"]})
        for chunk in chunks
    ]

    if progress:
        progress(60, "Đang tạo embeddings và lưu vào bộ nhớ cục bộ...")
    
    # Tạo Vector DB từ docs
    db = FAISS.from_documents(docs, embeddings_model)
    
    # LƯU LOCAL để lần sau dùng luôn
    if not os.path.exists(INDEX_SAVE_PATH):
        os.makedirs(INDEX_SAVE_PATH)
    db.save_local(save_dir)

    return db
def ask_pdf(file_path, question, progress=None):

    # ── Phase 1: Indexing ─────────────────────────────────────────────────────
    embeddings_model = get_embeddings_model()
    db = get_or_create_vector_db(file_path, embeddings_model, progress=progress)
    # ── Phase 2: Retrieval ─────────────────────────────────────────────────────
    if progress:
        progress(75, "Đang tìm nội dung liên quan...")

    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([
    f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content}"
    for doc in docs
])
    print("Context:", context)
    # ── Phase 3: Generation ─────────────────────────────────────────────────────
    if progress:
        progress(90, "AI đang tạo câu trả lời...")

    prompt = build_prompt(context, question)

    num_ctx_val = getattr(st.session_state, "num_ctx", 4096)
    llm = OllamaLLM(
        model="qwen2.5-coder:7b",
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        repeat_penalty=st.session_state.repeat_penalty,
        num_ctx=num_ctx_val
    )
    print("param LLM:", st.session_state.temperature, st.session_state.top_p, st.session_state.repeat_penalty, "num_ctx:", num_ctx_val)
    response = llm.invoke(prompt)

    if progress:
        progress(100, "Hoàn thành")

    return response