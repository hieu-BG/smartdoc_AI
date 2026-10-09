import pytest
from rag.loader import extract_pages_from_pdf

# Path tới file PDF bạn đã tạo
PDF_PATH = "data/temp_rag_doc.pdf"


def test_returns_list():
    result = extract_pages_from_pdf(PDF_PATH)
    assert isinstance(result, list)


def test_not_empty():
    result = extract_pages_from_pdf(PDF_PATH)
    assert len(result) > 0


def test_each_item_has_page_and_text():
    result = extract_pages_from_pdf(PDF_PATH)

    for item in result:
        assert "page" in item
        assert "text" in item
        assert isinstance(item["page"], int)
        assert isinstance(item["text"], str)


def test_page_number_starts_from_1():
    result = extract_pages_from_pdf(PDF_PATH)
    assert result[0]["page"] == 1


def test_text_not_empty():
    result = extract_pages_from_pdf(PDF_PATH)

    for item in result:
        assert item["text"].strip() != ""