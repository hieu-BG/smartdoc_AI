import pdfplumber
import docx
import os

def extract_content(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File không tồn tại")

    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".pdf":
        data = extract_pages_from_pdf(file_path)
    elif ext == ".docx":
        data = extract_paragraphs_from_docx(file_path)
    else:
        raise ValueError(f"Định dạng {ext} chưa được hỗ trợ")
    if not data:
        raise ValueError("Không extract được nội dung ")

    return data

def extract_pages_from_pdf(file_path):
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append({
                    "source": f"Page {i + 1}",
                    "text": text.strip()
                })
    return pages_text

def extract_paragraphs_from_docx(file_path):
    paragraphs_text = []
    doc = docx.Document(file_path)
    
    for i, para in enumerate(doc.paragraphs):
        if para.text and para.text.strip():
            paragraphs_text.append({
                "source": f"Paragraph {i + 1}",
                "text": para.text.strip()
            })
    return paragraphs_text