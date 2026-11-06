import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq


@st.cache_resource
def get_groq_model():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is required for GraphRAG, CoRAG, and SelfRAG modes.")
    return ChatGroq(
        groq_api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=4096,
    )