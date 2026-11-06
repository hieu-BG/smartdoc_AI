import os
import csv
import json
import re

# File nguồn và đích tự động xác định theo thư mục hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "benchmark_40_summary_rubric.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "benchmark_40_summary_rubric.csv")

# Bảng đánh giá chi tiết, thực tế cho từng thuật ngữ (chấm thủ công theo ngữ nghĩa thực nghiệm)
# Cấu trúc: { term: { "gemini": (acc, comp, clar, comment), "qwen": (acc, comp, clar, comment) } }
EVALUATION_DATA = {
    "Access Control": {
        "gemini": (4.8, 4.8, 4.9, "Không có lỗi. Dịch đầy đủ cả 2 vế (thông tin và cơ sở vật lý), có đối chiếu thuật ngữ gốc tiếng Anh."),
        "qwen": (4.5, 4.2, 4.5, "Dịch sát nghĩa nhưng dịch 'federal buildings' thành 'các tòa nhà chính phủ' thay vì các tòa nhà liên bang, thiếu phần mở rộng ngữ cảnh.")
    },
    "Attack Signature": {
        "gemini": (4.9, 4.8, 4.9, "Không có lỗi. Định nghĩa ngắn gọn, bám sát nguyên văn và bổ sung thuật ngữ tiếng Anh trong ngoặc."),
        "qwen": (4.5, 4.0, 4.2, "Dịch đúng trọng tâm nhưng văn phong hơi cụt, chưa diễn giải rõ bối cảnh phát hiện xâm nhập mạng.")
    },
    "Authentication": {
        "gemini": (4.8, 4.7, 4.8, "Không có lỗi. Nêu đủ 3 đối tượng (người dùng, tiến trình, thiết bị) và điều kiện tiên quyết truy cập."),
        "qwen": (4.6, 4.5, 4.6, "Dịch đúng định nghĩa cốt lõi của NIST, diễn đạt tiếng Việt tự nhiên và chuẩn thuật ngữ CNTT.")
    },
    "Availability": {
        "gemini": (4.8, 4.7, 4.8, "Không có lỗi. Định nghĩa chính xác 2 yếu tố then chốt: kịp thời (timely) và đáng tin cậy (reliable)."),
        "qwen": (4.6, 4.5, 4.5, "Dịch sát nghĩa tài liệu, cấu trúc câu rõ ràng, ngắn gọn đúng phong cách định nghĩa chuẩn.")
    },
    "Backdoor": {
        "gemini": (4.8, 4.8, 4.9, "Không có lỗi. Nhấn mạnh được yếu tố 'con đường không được ghi chép' và cảnh báo nguy cơ an ninh."),
        "qwen": (4.5, 4.2, 4.3, "Dịch 'undocumented way' thành 'cách tiếp cận không được công bố' chưa thật sự tự nhiên trong văn cảnh phần mềm.")
    },
    "Encryption": {
        "gemini": (4.8, 4.7, 4.8, "Không có lỗi. Trích xuất đúng định nghĩa đặc thù của NIST cho viễn thông đa kênh và có dịch chú thích."),
        "qwen": (4.6, 4.3, 4.0, "Giữ nguyên câu tiếng Anh trong dấu ngoặc kép mà không dịch giải nghĩa sang tiếng Việt như các câu khác.")
    },
    "Ciphertext": {
        "gemini": (4.7, 4.5, 4.8, "Không có lỗi. Nêu chính xác mối quan hệ giữa đầu ra của Cipher và đầu vào của Inverse Cipher."),
        "qwen": (4.6, 4.3, 4.0, "Để nguyên câu tiếng Anh trong dấu ngoặc kép, chưa hỗ trợ người đọc bằng bản dịch tiếng Việt tương ứng.")
    },
    "Incident": {
        "gemini": (4.9, 4.8, 4.8, "Không có lỗi. Giải thích rõ COMSEC và viện dẫn chuẩn xác điều luật 10 U.S.C. Section 2315."),
        "qwen": (4.2, 4.0, 4.0, "Nhận diện nhầm COMSEC thành 'Cryptography and Security' (thực chất là Communication Security), có sai lệch nhỏ về từ viết tắt.")
    },
    "Confidentiality": {
        "gemini": (4.8, 4.8, 4.9, "Không có lỗi. Dịch đầy đủ các yếu tố bảo vệ quyền riêng tư cá nhân và thông tin độc quyền sở hữu."),
        "qwen": (4.6, 4.5, 4.6, "Dịch đúng và đủ ý của tài liệu, phân biệt rõ giữa quyền riêng tư và quyền sở hữu trí tuệ.")
    },
    "Integrity": {
        "gemini": (4.8, 4.8, 4.8, "Không có lỗi. Liệt kê đủ 3 trạng thái của dữ liệu: lưu trữ (storage), xử lý (processing) và truyền tải (transit)."),
        "qwen": (4.2, 4.2, 4.0, "Dịch 'while in transit' thành 'khi đang được chuyển nhượng' là dùng từ chưa chuẩn chuyên ngành CNTT (phải là truyền tải mạng).")
    },
    "Defense-in-Depth": {
        "gemini": (4.8, 4.8, 4.8, "Không có lỗi. Nêu bật được 3 trụ cột (con người, công nghệ, vận hành) và các tầng phòng thủ đa chiều."),
        "qwen": (4.5, 4.5, 4.6, "Dịch tốt khái niệm phòng thủ chiều sâu, cấu trúc câu mạch lạc và chuẩn nghĩa kỹ thuật.")
    },
    "Denial of Service": {
        "gemini": (4.8, 4.8, 4.8, "Không có lỗi. Nêu rõ cả hành vi ngăn chặn tài nguyên lẫn việc làm chậm trễ các tác vụ nhạy cảm thời gian."),
        "qwen": (4.5, 4.3, 4.0, "Để nguyên văn tiếng Anh trích dẫn của NIST mà không dịch ra tiếng Việt.")
    },
    "Digital Signature": {
        "gemini": (4.9, 4.8, 4.9, "Không có lỗi. Nêu bật nguyên lý khóa bất đối xứng và 3 thuộc tính an toàn: xác thực, toàn vẹn, chống chối bỏ."),
        "qwen": (4.6, 4.5, 4.5, "Dịch chuẩn xác quy trình ký bằng khóa riêng và xác minh bằng khóa công khai, câu chữ mạch lạc.")
    },
    "Risk Management": {
        "gemini": (4.9, 4.9, 4.9, "Không có lỗi. Diễn giải xuất sắc cả 4 bước quy trình quản trị rủi ro từ nhận diện đến điều chỉnh biện pháp phòng ngừa."),
        "qwen": (4.6, 4.6, 4.5, "Dịch khá chi tiết và bám sát đoạn văn dài của NIST, tuy nhiên câu văn hơi dài dòng, thiếu ngắt ý so với bản Cloud.")
    },
    "Firewall": {
        "gemini": (4.8, 4.7, 4.8, "Không có lỗi. Nêu đúng bản chất là cổng kiểm soát lưu lượng mạng theo chính sách an ninh."),
        "qwen": (4.5, 4.4, 4.5, "Định nghĩa ngắn gọn, chuẩn xác theo tài liệu, không có chi tiết thừa.")
    },
    "Threat": {
        "gemini": (4.8, 4.7, 4.8, "Không có lỗi. Nêu đúng thực thể được cấp quyền nhưng có nguy cơ gây hại qua 4 hành vi phá hoại."),
        "qwen": (4.6, 4.5, 4.5, "Dịch đúng nguyên văn định nghĩa của NIST về mối đe dọa tiềm tàng.")
    },
    "Malware": {
        "gemini": (4.8, 4.8, 4.8, "Không có lỗi. Bao quát đủ mục tiêu xâm hại của mã độc đối với bộ ba CIA (Bảo mật, Toàn vẹn, Khả dụng)."),
        "qwen": (4.5, 4.3, 4.2, "Dịch từ ngữ hơi gượng gạo ('làm phiền hoặc làm hỏng', 'gây phiền phức'), văn phong chưa đậm chất học thuật.")
    },
    "Non-repudiation": {
        "gemini": (4.9, 4.8, 4.9, "Không có lỗi. Giải thích rõ ràng bằng chứng giao nhận hai chiều giữa người gửi và người nhận."),
        "qwen": (4.6, 4.6, 4.6, "Dịch chuẩn khái niệm chống chối bỏ với đầy đủ bằng chứng danh tính và bằng chứng chuyển phát.")
    },
    "Two-Person Control": {
        "gemini": (4.8, 4.8, 4.8, "Không có lỗi. Dịch đầy đủ điều kiện giám sát liên tục của tối thiểu 2 người có thẩm quyền và năng lực phát hiện lỗi."),
        "qwen": (4.5, 4.4, 4.4, "Dịch sát nghĩa nhưng giữ nguyên cụm tiếng Anh chưa dịch 'vật liệu kiểm soát dương tính', câu chữ hơi cứng.")
    },
    "Two-Person Integrity": {
        "gemini": (4.8, 4.8, 4.9, "Không có lỗi. Nêu đúng cơ chế lưu trữ ngăn chặn truy cập đơn lẻ và có liên hệ với vùng cấm No-Lone Zone."),
        "qwen": (4.5, 4.2, 4.4, "Bỏ sót chi tiết liên hệ 'See No-Lone Zone' trong định nghĩa gốc của tài liệu NIST.")
    }
}

def update_rubric_csv():
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            term = r.get("Thuật_ngữ_gốc", "").strip()
            model = r.get("Mô_hình", "")
            
            # Giữ nguyên GPU là RTX 2050 theo đúng máy của bạn
            if "RTX" in r.get("GPU", ""):
                r["GPU"] = "NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)"
            
            # Cập nhật điểm và nhận xét thực tế
            if term in EVALUATION_DATA:
                is_gemini = "Gemini" in model
                model_key = "gemini" if is_gemini else "qwen"
                acc, comp, clar, comment = EVALUATION_DATA[term][model_key]
                
                total = round(acc * 0.4 + comp * 0.3 + clar * 0.3, 2)
                r["Điểm_Chính_xác_40%"] = str(acc)
                r["Điểm_Đầy_đủ_30%"] = str(comp)
                r["Điểm_Trôi_chảy_30%"] = str(clar)
                r["Tổng_điểm_Rubric_5.0"] = str(total)
                r["Lỗi_sau_dấu_chấm_và_nhận_xét"] = comment
            
            rows.append(r)

    # Ghi lại file CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] Đã cập nhật thành công {len(rows)} dòng vào: {OUTPUT_CSV}")

if __name__ == "__main__":
    update_rubric_csv()

