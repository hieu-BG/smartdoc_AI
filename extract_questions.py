import os
import pypdf
import re
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tài liệu chuẩn NISTIR 7298 nằm gọn trong thư mục dự án
pdf_path = os.path.join(BASE_DIR, "data", "NISTIR_7298.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Không tìm thấy file NISTIR 7298 PDF tại: {pdf_path}")

print(f"Đang đọc tài liệu chuẩn từ: {pdf_path}")
reader = pypdf.PdfReader(pdf_path)

candidates = [
    'Access Control', 'Attack Signature', 'Authentication', 'Availability',
    'Backdoor', 'Ciphertext', 'Confidentiality', 'Defense-in-Depth',
    'Denial of Service', 'Digital Signature', 'Encryption', 'Firewall',
    'Incident', 'Integrity', 'Malware', 'Non-repudiation',
    'Risk Management', 'Threat', 'Two-Person Control', 'Two-Person Integrity'
]

qa_list = []
for p_idx in range(5, 215):
    t = reader.pages[p_idx].extract_text()
    for term in candidates:
        if not any(q['term'] == term for q in qa_list):
            m = re.search(rf'({re.escape(term)}\s*(?:\([^)]*\))?\s*–\s*.*?(?=SOURCE:))', t, re.DOTALL)
            if m:
                raw_def = m.group(1).replace('\n', ' ')
                raw_def = re.sub(' +', ' ', raw_def).strip()
                qa_list.append({
                    'id': len(qa_list) + 1,
                    'term': term,
                    'page': p_idx + 1,
                    'question': f'Theo tài liệu NISTIR 7298, thuật ngữ "{term}" được định nghĩa như thế nào?',
                    'ground_truth': raw_def
                })

print(f"Total QA pairs extracted: {len(qa_list)}")
for q in qa_list:
    print(f"#{q['id']} [Page {q['page']}] {q['term']}: {q['ground_truth'][:70]}...")

output_path = os.path.join(BASE_DIR, "benchmark_20_questions.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(qa_list, f, ensure_ascii=False, indent=2)

print(f"Successfully saved 20 questions to {output_path}")
