def retrieve_per_subquestion(sub_questions, vectorstore, k=3):
    results = []
    seen = set()

    for sub_q in sub_questions:
        # 1. Retrieve trực tiếp
        docs = vectorstore.similarity_search(sub_q, k=k)

        contexts = []
        for d in docs:
            content = d.page_content.strip()

            if content not in seen:
                seen.add(content)
                contexts.append(content)

            if len(contexts) == k:
                break

        results.append({
            "sub_question": sub_q,
            "contexts": contexts
        })

    return results