from ui.sidebar import render_sidebar 
from ui.main_content import render_main
import streamlit as st
from db.session import init_session
stats = {
    "documents": 0,
    "questions": 0,
    "status": "Chờ..."
}
st.set_page_config(layout="wide",
    initial_sidebar_state="expanded"
)   
init_session() 
render_sidebar(stats)
render_main()