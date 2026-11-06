from rag.splitter import split_chunks_from_pages

# =========================
# Test 1: Kiểu dữ liệu trả về
# =========================
def test_returns_list():
    pages = [{"page": 1, "text": "Hello world " * 100}]
    result = split_chunks_from_pages(pages)

    # Hàm phải trả về list
    assert isinstance(result, list)


# =========================
# Test 2: Không được rỗng
# =========================
def test_not_empty():
    pages = [{"page": 1, "text": "Hello world " * 100}]
    result = split_chunks_from_pages(pages)

    # Phải có ít nhất 1 chunk
    assert len(result) > 0


# =========================
# Test 3: Cấu trúc từng chunk
# =========================
def test_structure():
    pages = [{"page": 2, "text": "Hello world " * 100}]
    result = split_chunks_from_pages(pages)

    for item in result:
        # Mỗi chunk phải có page và text
        assert "page" in item
        assert "text" in item

        # Kiểu dữ liệu phải đúng
        assert isinstance(item["page"], int)
        assert isinstance(item["text"], str)


# =========================
# Test 4: Giữ đúng page number
# =========================
def test_page_preserved():
    pages = [{"page": 5, "text": "Hello world " * 100}]
    result = split_chunks_from_pages(pages)

    for item in result:
        # Chunk nào cũng phải thuộc page 5
        assert item["page"] == 5


# =========================
# Test 5: Text không được rỗng
# =========================
def test_chunk_not_empty():
    pages = [{"page": 1, "text": "Hello world " * 100}]
    result = split_chunks_from_pages(pages)

    for item in result:
        # Không được có chunk rỗng hoặc chỉ toàn space
        assert item["text"].strip() != ""


# =========================
# Test 6: Kiểm tra chunking có hoạt động
# =========================
def test_chunking_works():
    pages = [{"page": 1, "text": "A" * 3000}]
    result = split_chunks_from_pages(pages)

    # Text dài phải bị tách thành nhiều chunk
    assert len(result) > 1