import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

load_dotenv()

print("[1/4] Dang khoi tao mo hinh Qwen2.5-Coder qua Ollama va HuggingFace Embedding...")
llm = OllamaLLM(model=os.getenv("LLM_MODEL", "qwen2.5-coder:7b"), base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), num_ctx=4096)
embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))

print("[2/4] Nap du lieu ngu lieu mau...")
docs = [
   Document(page_content="SmartDoc AI la he thong tro ly tai lieu thong minh chay hoan toan offline tren may cuc bo.", metadata={"source": "Huong_dan_SmartDoc.pdf", "page": 1}),
   Document(page_content="He thong su dung ky thuat RAG ket hop Ollama va FAISS de luu tru vector, dam bao an toan cho du lieu noi bo.", metadata={"source": "Tu_dien_thuat_ngu_CNTT.pdf", "page": 2})
]

print("[3/4] Tao Vector Store bang FAISS va lap chi muc...")
vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

question = "SmartDoc AI chay online hay offline va dung cong nghe gi de luu vector?"
print(f"\n=======================================================")
print(f"Cau hoi: {question}")
print(f"=======================================================")

print("[4/4] Dang tim kiem doan van ban lien quan va sinh cau tra loi qua Ollama...")
# 1. Truy xuat Top-2 doan tu FAISS
retrieved_docs = vectorstore.similarity_search(question, k=2)
context = "\n".join([doc.page_content for doc in retrieved_docs])

# 2. Xay dung prompt va goi truc tiep Ollama
prompt = f"""Dua vao ngu canh duoc cung cap duoi day, hay tra loi cau hoi ngan gon va chinh xac:
Ngu canh:
{context}

Cau hoi: {question}
Tra loi:"""

answer = llm.invoke(prompt)

print("\nKet qua SmartDoc AI tra loi:")
print(answer)

print("\nTrich dan nguon tai lieu (Citations):")
for doc in retrieved_docs:
    print(f"- Nguon: {doc.metadata['source']} | Trang: {doc.metadata['page']}")
print(f"=======================================================")
