from multiprocessing import Process, Queue
from rag.rag import ask_pdf
from rag.selfrag.selfrag import ask_pdf_selfrag
def run_ai_pipeline(file_path, question, mode, progress=None):
    rag_response = None
    selfrag_response = None
    if mode == "RAG":
        return ask_pdf(file_path, question,progress=progress)
    elif mode == "SelfRAG":
        return ask_pdf_selfrag(file_path, question, progress=progress)
