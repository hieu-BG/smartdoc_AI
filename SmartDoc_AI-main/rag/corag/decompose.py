from ..prompt import decomposition_prompt
import json
def decompose(question, llm):
    prompt = decomposition_prompt(question)

    try:
        response = llm.invoke(prompt)
        print("TYPE:", type(response))
        print("RAW RESPONSE:", response)

        content = response.content.strip()

        # parse JSON
        data = json.loads(content)
        print("RAW RESPONSE:", content)
        # đảm bảo key tồn tại
        sub_questions = data.get("sub_questions", [])

        # fallback nếu model trả thiếu
        if not isinstance(sub_questions, list) or len(sub_questions) == 0:
            return [question]

        return sub_questions[:5]

    except Exception as e:
        print("Decompose error:", e)
        return [question]