import streamlit as st
from ui.theme import load_sidebar_style
from db.db import get_conversations,create_conversation,delete_conversation

def render_sidebar(stats):
    """
    Render sidebar SmartDoc AI
    stats = {
        "documents": int,
        "questions": int,
        "status": str
    }
    """

    load_sidebar_style()

    with st.sidebar:

        # =========================
        # HEADER
        # =========================
        st.markdown(
            '<div class="sidebar-subtitle">Document Intelligence</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ===== NEW CHAT BUTTON =====
        if st.button("➕ New Chat", use_container_width=True):
            new_id = create_conversation()
            st.session_state.conversation_id = new_id
            st.session_state.current_answer = None
            st.session_state.selected_answer = None
            st.rerun()
        # =========================
        # HƯỚNG DẪN SỬ DỤNG
        # =========================
        st.markdown(
            '<div class="section-title">HƯỚNG DẪN SỬ DỤNG</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="instruction">
                <div class="circle">1</div>
                Tải lên tài liệu PDF
            </div>

            <div class="instruction">
                <div class="circle">2</div>
                Hệ thống xử lý tài liệu
            </div>

            <div class="instruction">
                <div class="circle">3</div>
                Nhập câu hỏi về nội dung
            </div>

            <div class="instruction">
                <div class="circle">4</div>
                Nhận câu trả lời từ AI
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">CHUNK CONFIGURATION</div>',
            unsafe_allow_html=True
        )

        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 500

        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 50

        st.markdown("**Chunk Size**")
        st.session_state.chunk_size = st.slider("", 100, 2000, st.session_state.chunk_size, 50)

        st.markdown("**Chunk Overlap**")
        st.session_state.chunk_overlap = st.slider("", 0, 500, st.session_state.chunk_overlap, 10)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # =========================
        # MODEL CONFIG
        # =========================
        st.markdown(
            '<div class="section-title">MODEL CONFIGURATION</div>',
            unsafe_allow_html=True
        )

        MODEL_NAME = "qwen2.5-coder:7b"

        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
        if "top_p" not in st.session_state:
            st.session_state.top_p = 0.9
        if "repeat_penalty" not in st.session_state:
            st.session_state.repeat_penalty = 1.1
        if "num_ctx" not in st.session_state:
            st.session_state.num_ctx = 4096

        st.markdown(
            f"""
            <div class="model-card">
                <div style="font-size:14px;">Active Model</div>
                <div style="font-size:20px;font-weight:600;">
                    {MODEL_NAME}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Temperature**")
        st.session_state.temperature = st.slider("", 0.0, 1.5, st.session_state.temperature, 0.1)

        st.markdown("**Top P**")
        st.session_state.top_p = st.slider("", 0.0, 1.0, st.session_state.top_p, 0.05)

        st.markdown("**Repeat Penalty**")
        st.session_state.repeat_penalty = st.slider("", 0.5, 2.0, st.session_state.repeat_penalty, 0.1)

        st.markdown("**Context Window (num_ctx)**")
        st.session_state.num_ctx = st.select_slider("", options=[1024, 2048, 4096, 8192], value=st.session_state.num_ctx, key="slider_num_ctx")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">LỊCH SỬ TRÒ CHUYỆN</div>', unsafe_allow_html=True)

        conversations = get_conversations()

        container = st.container()

        with st.container(height=300):

            for cid, title in conversations:
                label = title if title else f"Chat {cid}"

                col1, col2 = st.columns([4, 1])

                with col1:
                    if st.button(label, key=f"select_{cid}"):
                        st.session_state.conversation_id = cid
                        st.session_state.current_answer = None
                        st.session_state.selected_answer = None
                        st.rerun()

                with col2:
                    if st.button("❌", key=f"delete_{cid}"):
                        delete_conversation(cid)
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)


