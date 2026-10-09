from typing import List
from langchain_core.documents import Document

from .sufficiency import check_sufficiency
from .util import deduplicate ,retrieve


def run_corag_loop(
    db,
    llm,
    question: str,
    max_hops: int = 3,
    k: int = 3,
    progress=None
) -> List[str]:
    context_buffer: List[str] = []
    current_query = question
    hop = 0

    while hop < max_hops:
        hop += 1

        if progress:
            progress(50 + int(hop / max_hops * 30), f"Hop {hop}/{max_hops}...")
        preview = [c[:10] for c in context_buffer]
        print(f"[CoRAG] Context preview: {preview}")
        # 1. Retrieve
        retrieved = retrieve(db, current_query, k=k)

        # 2. Deduplicate
        new_docs = deduplicate(context_buffer, retrieved)

        if new_docs:
            context_buffer.extend(d.page_content for d in new_docs)

        context_so_far = "\n\n".join(context_buffer)

        print(f"[CoRAG] Hop {hop} | query: {current_query}")

        # 3. Critique
        verdict = check_sufficiency(llm, question, context_so_far)

        print(f"[CoRAG] sufficient={verdict['sufficient']}")

        if verdict["sufficient"]:
            break

        # 4. Correction (query refinement)
        refined = verdict.get("refined_query", "").strip()

        if not refined or refined == current_query:
            break

        current_query = refined

    return context_buffer