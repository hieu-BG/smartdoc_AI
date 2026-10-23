import pdfplumber
import docx
import os
import warnings

# ── Cấu hình xử lý ảnh trong PDF ─────────────────────────────────────────────
ENABLE_PDF_OCR = True      # Đặt False để tắt hoàn toàn xử lý ảnh
OCR_LANGUAGES = ["vi", "en"]
OCR_RESOLUTION = 200       # DPI khi render ảnh/trang để OCR
MIN_IMAGE_SIZE = 50        # Bỏ qua ảnh nhỏ hơn 50pt (logo, icon, đường kẻ)
MIN_OCR_CHARS = 15         # Bỏ qua kết quả OCR quá ngắn (thường là nhiễu)

_ocr_reader = None
_ocr_unavailable = False


def _get_ocr_reader():
    """Tải EasyOCR một lần duy nhất (lần đầu sẽ tải model nên hơi lâu)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader


def ocr_image(pil_image):
    """Nhận một ảnh PIL, trả về chuỗi văn bản đọc được ('' nếu không có)."""
    global _ocr_unavailable
    if _ocr_unavailable:
        return ""
    try:
        import numpy as np
        reader = _get_ocr_reader()
        lines = reader.readtext(
            np.array(pil_image.convert("RGB")), detail=0, paragraph=True
        )
    except ImportError:
        _ocr_unavailable = True
        warnings.warn(
            "Chưa cài EasyOCR nên bỏ qua nội dung trong ảnh. "
            "Cài bằng: python -m pip install easyocr"
        )
        return ""
    return "\n".join(line.strip() for line in lines if line.strip())


def _ocr_full_page(page):
    image = page.to_image(resolution=OCR_RESOLUTION).original
    return ocr_image(image)


def _ocr_embedded_images(page):
    """OCR từng ảnh nhúng trong trang, trả về danh sách đoạn text."""
    x_min, y_min, x_max, y_max = page.bbox
    results = []
    for img in page.images:
        x0, x1 = max(img["x0"], x_min), min(img["x1"], x_max)
        top, bottom = max(img["top"], y_min), min(img["bottom"], y_max)
        if (x1 - x0) < MIN_IMAGE_SIZE or (bottom - top) < MIN_IMAGE_SIZE:
            continue
        crop = page.crop((x0, top, x1, bottom))
        text = ocr_image(crop.to_image(resolution=OCR_RESOLUTION).original)
        if len(text) >= MIN_OCR_CHARS:
            results.append(text)
    return results

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
            text = (page.extract_text() or "").strip()

            if ENABLE_PDF_OCR:
                if not text:
                    # Trang scan (không có chữ): OCR cả trang
                    text = _ocr_full_page(page)
                elif page.images:
                    # Trang có chữ và ảnh: OCR riêng từng ảnh rồi ghép vào
                    image_texts = _ocr_embedded_images(page)
                    if image_texts:
                        text += "\n\n[Nội dung trong hình ảnh]\n" + "\n".join(image_texts)

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