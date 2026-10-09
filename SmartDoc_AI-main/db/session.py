import streamlit as st
from db.db import create_conversation
def init_session():
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = create_conversation()

    if "selected_answer" not in st.session_state:
        st.session_state.selected_answer = None

    if "history" not in st.session_state:
        st.session_state.history = []