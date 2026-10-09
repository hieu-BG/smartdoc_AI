
from langdetect import detect
def build_prompt(context, question):
    try:
        lang = detect(question)
    except:
        lang = 'vi'

    if lang == 'vi':
        return f"""
Bạn là trợ lý AI giúp trả lời câu hỏi dựa trên nội dung ngữ cảnh.
Trả lời ngắn gọn 3-4 câu bằng tiếng Việt.
Nếu không tìm thấy thông tin trong ngữ cảnh thì hãy nói:
"Tôi không tìm thấy thông tin trong tài liệu."

Ngữ cảnh:
{context}

Câu hỏi:
{question}
"""
    else:
        return f"""
You are an AI assistant that helps answer questions based on the provided context.
Answer briefly in 3-4 sentences in English.
If the information is not found in the context, please say:
"I could not find the information in the document."

Context:
{context}

Question:
{question}
"""
from langdetect import detect

def build_prompt_graphrag(context, question):
    try:
        lang = detect(question)
    except:
        lang = 'vi'

    if lang == 'vi':
        return f"""
Bạn là AI trả lời câu hỏi dựa trên dữ liệu truy xuất từ hệ thống GraphRAG.

Hướng dẫn:
- Ngữ cảnh gồm 2 loại:
  + [TEXT]&#58; thông tin mô tả
  + [GRAPH]&#58; quan hệ giữa các thực thể (dạng A --(RELATION)--> B)
- Hãy CHUYỂN các quan hệ GRAPH thành câu tự nhiên (KHÔNG giữ ký hiệu [], --, hoặc RELATION).
- Ưu tiên sử dụng thông tin từ [GRAPH] để suy luận (vì nó thể hiện quan hệ rõ ràng)
- Kết hợp với [TEXT] để bổ sung ý nghĩa
- Trả lời ngắn gọn 3-4 câu bằng tiếng Việt.

Nếu không tìm thấy thông tin thì trả lời:
"Tôi không tìm thấy thông tin trong tài liệu."

Ngữ cảnh:
{context}

Câu hỏi:
{question}
"""
    else:
        return f"""
You are an AI assistant using GraphRAG retrieved data.

Instructions:
- Context includes:
  + [TEXT]&#58; descriptive content
  + [GRAPH]&#58; relationships (A --(RELATION)--> B)
- Convert GRAPH relationships into natural language (DO NOT include [], --, or RELATION).
- Prioritize reasoning from GRAPH relationships
- Use TEXT to support explanation
- Answer in 3-4 sentences clearly

If not found:
"I could not find the information in the document."

Context:
{context}

Question:
{question}
"""
# CORAG
def sufficiency_prompt(question, context):
    try:
        lang = detect(question)
    except:
        lang = 'vi'

    if lang == 'vi':
        return f"""
Bạn là một trợ lý đánh giá chất lượng ngữ cảnh.
 
Câu hỏi gốc: {question}
 
Ngữ cảnh hiện tại đã thu thập:
{context}
 
Hãy đánh giá xem ngữ cảnh trên có đủ để trả lời câu hỏi không.
Trả lời ĐÚNG theo định dạng JSON sau, KHÔNG thêm gì khác:
{{
  "sufficient": true hoặc false,
  "reason": "lý do ngắn gọn",
  "refined_query": "câu truy vấn mới nếu cần tìm thêm, để trống nếu đủ rồi"
}}
"""
    else:
        return f"""
You are an AI assistant that evaluates whether the provided context is sufficient to answer a question.

Question:
{question}

Current context:
{context}

Task:
- Determine if the context is sufficient to answer the question.
- If NOT sufficient, suggest a better query to retrieve more relevant information.

Rules:
- Return ONLY valid JSON (no explanation, no extra text).
- Keep the reason concise.

Output format:
{{
  "sufficient": true or false,
  "reason": "brief explanation",
  "refined_query": "improved search query if needed, otherwise empty string"
}}
"""
def decomposition_prompt(question):
    try:
        lang = detect(question)
    except:
        lang = 'vi'

    if lang == 'vi':
        return f"""
Bạn là một trợ lý AI chuyên phân rã câu hỏi cho hệ thống CoRAG.

Câu hỏi gốc: {question}

Nhiệm vụ:
- Chia câu hỏi trên thành tối đa 3 câu hỏi con.
- Mỗi câu hỏi con phải rõ ràng, độc lập, phục vụ truy xuất thông tin.
- Các câu hỏi nên bổ sung cho nhau, không trùng lặp.
- Sắp xếp theo thứ tự logic để giải quyết câu hỏi chính.

Yêu cầu:
- Trả về ĐÚNG định dạng JSON.
- Không giải thích thêm.

Output:
{{
  "sub_questions": [
    "câu hỏi 1",
    "câu hỏi 2",
    "câu hỏi 3"
  ]
}}
"""
    else:
        return f"""
You are an AI assistant that decomposes a complex question for a CoRAG system.

Original question: {question}

Task:
- Break the question into up to 3 sub-questions.
- Each sub-question must be clear, independent, and useful for retrieval.
- Avoid redundancy.
- Ensure logical order to progressively answer the main question.

Rules:
- Return ONLY valid JSON.
- No explanation, no extra text.

Output:
{{
  "sub_questions": [
    "sub-question 1",
    "sub-question 2",
    "sub-question 3"
  ]
}}
"""
def answer_subquestion_prompt(sub_question, contexts):
    context_text = "\n\n".join(contexts)

    return f"""
Bạn là trợ lý AI trả lời câu hỏi dựa trên ngữ cảnh.

Câu hỏi:
{sub_question}

Ngữ cảnh:
{context_text}

Yêu cầu:
- Trả lời CHỈ dựa trên ngữ cảnh
- Ngắn gọn, rõ ràng
- Nếu không đủ thông tin → trả lời "Không đủ thông tin"

Câu trả lời:
"""
def combine_prompt(question, answers):
    combined = "\n\n".join(
        [f"Câu hỏi phụ: {a['sub_question']}\nTrả lời: {a['answer']}" for a in answers]
    )

    return f"""
Bạn là một trợ lý AI tổng hợp câu trả lời.

Câu hỏi chính:
{question}

Các câu trả lời từng phần:
{combined}

Nhiệm vụ:
- Tổng hợp thành một câu trả lời hoàn chỉnh, mạch lạc.
- Không lặp lại ý.
- Chỉ sử dụng thông tin từ các câu trả lời đã cho.
- Nếu có phần thiếu thông tin, hãy nói rõ.

Câu trả lời cuối:
"""