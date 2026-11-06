
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def deduplicate(existing: list[str], new_docs: list[Document]) -> list[Document]:
    return [d for d in new_docs if d.page_content not in existing]
    
def retrieve(db: FAISS, query: str, k: int = 3) -> list[Document]:
    return db.similarity_search(query, k=k)