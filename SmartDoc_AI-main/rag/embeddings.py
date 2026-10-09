import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )