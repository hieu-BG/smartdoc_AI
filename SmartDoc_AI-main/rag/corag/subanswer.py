from ..prompt import answer_subquestion_prompt
def answer_subquestions(results, llm):
    """
    results: output từ retrieve_per_subquestion
    [
      {
        "sub_question": str,
        "contexts": [str]
      }
    ]
    """

    answers = []

    for item in results:
        sub_q = item["sub_question"]
        contexts = item["contexts"]

        try:
            prompt = answer_subquestion_prompt(sub_q, contexts)
            response = llm.invoke(prompt).content.strip()

        except Exception as e:
            print("Answer error:", e)
            response = "Lỗi khi gọi LLM"

        answers.append({
            "sub_question": sub_q,
            "answer": response,
        })

    return answers