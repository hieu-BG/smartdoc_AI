import streamlit as st
from ui.theme import load_main_style
from rag.rag import ask_pdf
from rag.graph_rag.graphrag import ask_pdf as ask_pdf_graph
from db.db import save_message,get_qa_history,get_message_count,update_conversation_title
from db.session import init_session
from ui.ai_service import run_ai_pipeline
import json
MAX_FILE_SIZE = 10 * 1024 * 1024
def render_main():
    init_session()
    load_main_style()

    # ===== HEADER =====
    st.markdown("# SmartDoc AI")
    st.markdown("Phân tích tài liệu thông minh với trí tuệ nhân tạo")

    # ===== UPLOAD =====
    st.markdown(
        '<div class="section-title">Tải lên tài liệu</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "",
        type=["pdf","docx"]
    )
    if uploaded_file is not None:
        if uploaded_file.type != "application/pdf" and uploaded_file.type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            st.error("Only PDF and DOCX files are supported.")
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File quá lớn! Vui lòng tải file nhỏ hơn 10MB.")
        else:
            st.success(" File hợp lệ!")
        
    #  ===== QUESTION =====
    st.markdown(
        '<div class="section-title"> Đặt câu hỏi</div>',
        unsafe_allow_html=True
    )
    st.markdown("Chọn chế độ trả lời")

    mode = st.radio(
    "",
    ["RAG", "GraphRAG", "CoRAG", "SelfRAG"],
    horizontal=True
    )   
    
    with st.form(key="question_form"):
        col1, col2 = st.columns([12, 1])

        with col1:
            question = st.text_input(
                "",
                placeholder="Hỏi bất cứ điều gì về tài liệu của bạn..."
            )

        with col2:
            submitted = st.form_submit_button("➤")
    
    # ===== PROCESS =====
    if submitted:

        if uploaded_file is None:
            st.warning("Vui lòng tải tài liệu lên trước")

        elif question == "":
            st.warning("Vui lòng nhập câu hỏi")

        else:

            # lưu file tạm
            file_path = f"temp_{uploaded_file.name}"

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            # tạo progress UI
            progress_bar = st.progress(0)
            status = st.empty()

            def update_progress(percent, text):
                progress_bar.progress(percent)
                status.text(text)
            with st.spinner("AI đang phân tích tài liệu..."):
                try:
                    response = run_ai_pipeline(
                        file_path=file_path,
                        question=question,
                        mode=mode,
                        progress=update_progress
                    )
                    
                    progress_bar.progress(100)
                    status.empty()
                    save_message(st.session_state.conversation_id, "user", question)
                    save_message(st.session_state.conversation_id, "assistant", json.dumps(response, ensure_ascii=False))
                    count = get_message_count(st.session_state.conversation_id)

                    if count == 2: 
                        update_conversation_title(st.session_state.conversation_id, question[:50])

                    st.session_state.current_answer = response
                    st.session_state.selected_answer = None

                    st.rerun()
                    
                except ValueError as e:
                    progress_bar.empty()
                    status.empty()
                    st.warning(f" Lỗi: {str(e)}")
                except Exception as e:
                    progress_bar.empty()
                    status.empty()
                    st.error(f"Có lỗi xảy ra: {str(e)}")
    
    # ===== HIỂN THỊ ANSWER (LUÔN HIỂN THỊ) =====
    st.markdown("---")
    st.markdown("### Câu trả lời")

    if st.session_state.get("current_answer"):
        answer = st.session_state.current_answer
        st.markdown(answer)

    elif st.session_state.get("selected_answer"):
        answer = st.session_state.selected_answer
        st.markdown(answer)

    else:
        st.info("Chưa có câu trả lời")
    
    history = get_qa_history(st.session_state.conversation_id)
    if len(history) > 0:      
        st.markdown("### Lịch sử câu hỏi")

        for i, (q, a) in enumerate(history):
            if st.button(q, key=f"history_{i}"):

                try:
                    parsed = json.loads(a)
                    st.session_state.selected_answer = parsed
                except:
                    st.session_state.selected_answer = a

                st.session_state.current_answer = None
                st.rerun()