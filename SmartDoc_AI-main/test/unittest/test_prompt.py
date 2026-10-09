from rag.prompt import build_prompt, build_prompt_graphrag


# =========================
# TEST build_prompt (REAL)
# =========================

def test_build_prompt_real_vi():
    # dùng tiếng Việt thật để langdetect tự đoán
    question = "Xin chào bạn"
    context = "test context"

    result = build_prompt(context, question)

    # kiểm tra output có đúng format tiếng Việt
    assert "Bạn là trợ lý AI" in result
    assert context in result
    assert question in result


def test_build_prompt_real_en():
    # dùng tiếng Anh thật
    question = "What is artificial intelligence?"
    context = "test context"

    result = build_prompt(context, question)

    assert "You are an AI assistant" in result
    assert context in result
    assert question in result


# =========================
# TEST fallback (REAL)
# =========================

def test_build_prompt_fallback_real():
    # input rác để test fallback
    question = "###@@@"
    context = "test context"

    result = build_prompt(context, question)

    # fallback mặc định là VI
    assert "Bạn là trợ lý AI" in result


# =========================
# TEST GraphRAG (REAL)
# =========================

def test_graphrag_real_vi():
    question = "Nguyễn Huệ là ai"
    context = "[TEXT] thông tin + [GRAPH] A --(REL)--> B"

    result = build_prompt_graphrag(context, question)

    assert "[GRAPH]" in result
    assert "Ngữ cảnh" in result


def test_graphrag_real_en():
    question = "Who is Albert Einstein"
    context = "[TEXT] info + [GRAPH] A --(REL)--> B"

    result = build_prompt_graphrag(context, question)

    assert "GraphRAG" in result
    assert "Context:" in result