import time
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer


def extract_graph_documents(
    chunks: List[Document], 
    transformer: LLMGraphTransformer, 
    limit: int = 10, 
    delay: int = 0
): 
    all_graph_documents = []
    
    processing_chunks = chunks[:limit]
    total = len(processing_chunks)
    
    print(f"Bắt đầu trích xuất tri thức từ {total} chunks...")

    for i, chunk in enumerate(processing_chunks):
        try:
            doc_input = [chunk] if isinstance(chunk, Document) else [Document(page_content=str(chunk))]
            
            graph_doc = transformer.convert_to_graph_documents(doc_input)
            all_graph_documents.extend(graph_doc)

            print(f" Đã xử lý xong chunk {i+1}/{total}")

            if i < total - 1:  # Không cần nghỉ ở chunk cuối cùng
                time.sleep(delay)
                
        except Exception as e:
            print(f" Lỗi tại chunk {i+1}: {str(e)}")
            continue # Tiếp tục xử lý các chunk tiếp theo
            
    print(f" Hoàn tất! Tổng cộng thu được {len(all_graph_documents)} graph documents.")
    return all_graph_documents

