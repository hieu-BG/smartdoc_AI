from multiprocessing import Process, Queue
from rag.rag import ask_pdf
from rag.graph_rag.graphrag import ask_pdf as ask_pdf_graph
from rag.selfrag.selfrag import ask_pdf_selfrag
from rag.corag.corag import ask_pdf_corag
def run_ai_pipeline(file_path, question, mode, progress=None):
    rag_response = None
    graph_response = None
    corag_response = None
    selfrag_response = None
    if mode == "RAG":
        return ask_pdf(file_path, question,progress=progress)

    elif mode == "GraphRAG":
        return ask_pdf_graph(file_path, question, progress=progress)
        
    elif mode == "CoRAG":
        return ask_pdf_corag(file_path, question, progress=progress)
    elif mode == "SelfRAG":
        return ask_pdf_selfrag(file_path, question, progress=progress)
