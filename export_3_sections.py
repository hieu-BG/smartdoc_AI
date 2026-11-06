import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "benchmark_40_summary_rubric.csv")
OUTPUT_COMPARISON_CSV = os.path.join(BASE_DIR, "benchmark_so_sanh_3_phan.csv")

def export_comparison_csv():
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    gemini_dict = {r["Question_ID"]: r for r in rows if "Gemini" in r["Mô_hình"]}
    qwen4_dict = {r["Question_ID"]: r for r in rows if "Q4_K_M" in r["Mô_hình"]}
    qwen8_dict = {r["Question_ID"]: r for r in rows if "Q8_0" in r["Mô_hình"]}

    # Cấu trúc so sánh các mô hình:
    # PHẦN 1: THÔNG TIN CÂU HỎI & ĐÁP ÁN CHUẨN NISTIR 7298
    # PHẦN 2: KẾT QUẢ & ĐÁNH GIÁ GOOGLE GEMINI (CLOUD)
    # PHẦN 3: KẾT QUẢ & ĐÁNH GIÁ QWEN2.5-CODER:7B (LOCAL 4-BIT Q4_K_M)
    # PHẦN 4: KẾT QUẢ & ĐÁNH GIÁ QWEN2.5-CODER:7B (LOCAL 8-BIT Q8_0)
    headers = [
        # --- PHẦN 1: THÔNG TIN CÂU HỎI & TÀI LIỆU CHUẨN ---
        "[P1]_Mã_câu",
        "[P1]_Thuật_ngữ",
        "[P1]_Câu_hỏi_khảo_sát",
        "[P1]_Đáp_án_chuẩn_NISTIR_7298",

        # --- PHẦN 2: GOOGLE GEMINI FLASH LITE (CLOUD) ---
        "[P2_Gemini]_TTFT_giây",
        "[P2_Gemini]_Tốc_độ_tok_s",
        "[P2_Gemini]_Số_tokens",
        "[P2_Gemini]_Điểm_Chính_xác_40%",
        "[P2_Gemini]_Điểm_Đầy_đủ_30%",
        "[P2_Gemini]_Điểm_Trôi_chảy_30%",
        "[P2_Gemini]_Tổng_điểm_Rubric_5.0",
        "[P2_Gemini]_Nhận_xét_đánh_giá_lỗi",
        "[P2_Gemini]_Câu_trả_lời_chi_tiết",

        # --- PHẦN 3: QWEN 2.5 CODER 7B (LOCAL 4-BIT Q4_K_M) ---
        "[P3_Qwen4b]_TTFT_giây",
        "[P3_Qwen4b]_Tốc_độ_tok_s",
        "[P3_Qwen4b]_Số_tokens",
        "[P3_Qwen4b]_Điểm_Chính_xác_40%",
        "[P3_Qwen4b]_Điểm_Đầy_đủ_30%",
        "[P3_Qwen4b]_Điểm_Trôi_chảy_30%",
        "[P3_Qwen4b]_Tổng_điểm_Rubric_5.0",
        "[P3_Qwen4b]_Nhận_xét_đánh_giá_lỗi",
        "[P3_Qwen4b]_Câu_trả_lời_chi_tiết",
        "[P3_Qwen4b]_GPU_sử_dụng",

        # --- PHẦN 4: QWEN 2.5 CODER 7B (LOCAL 8-BIT Q8_0) ---
        "[P4_Qwen8b]_TTFT_giây",
        "[P4_Qwen8b]_Tốc_độ_tok_s",
        "[P4_Qwen8b]_Số_tokens",
        "[P4_Qwen8b]_Điểm_Chính_xác_40%",
        "[P4_Qwen8b]_Điểm_Đầy_đủ_30%",
        "[P4_Qwen8b]_Điểm_Trôi_chảy_30%",
        "[P4_Qwen8b]_Tổng_điểm_Rubric_5.0",
        "[P4_Qwen8b]_Nhận_xét_đánh_giá_lỗi",
        "[P4_Qwen8b]_Câu_trả_lời_chi_tiết",
        "[P4_Qwen8b]_GPU_sử_dụng"
    ]

    out_data = []
    for q_id_int in range(1, 21):
        q_id = str(q_id_int)
        g = gemini_dict.get(q_id, {})
        q4 = qwen4_dict.get(q_id, {})
        q8 = qwen8_dict.get(q_id, {})

        row_item = {
            # Phần 1
            "[P1]_Mã_câu": f"Q{int(q_id):02d}",
            "[P1]_Thuật_ngữ": g.get("Thuật_ngữ_gốc", q4.get("Thuật_ngữ_gốc", "")),
            "[P1]_Câu_hỏi_khảo_sát": g.get("Câu_hỏi", q4.get("Câu_hỏi", "")),
            "[P1]_Đáp_án_chuẩn_NISTIR_7298": g.get("Đáp_án_chuẩn_NISTIR", q4.get("Đáp_án_chuẩn_NISTIR", "")),

            # Phần 2: Gemini
            "[P2_Gemini]_TTFT_giây": g.get("TTFT_trung_bình_s", ""),
            "[P2_Gemini]_Tốc_độ_tok_s": g.get("Tốc_độ_trung_bình_tok_s", ""),
            "[P2_Gemini]_Số_tokens": g.get("Số_tokens_ước_tính", ""),
            "[P2_Gemini]_Điểm_Chính_xác_40%": g.get("Điểm_Chính_xác_40%", ""),
            "[P2_Gemini]_Điểm_Đầy_đủ_30%": g.get("Điểm_Đầy_đủ_30%", ""),
            "[P2_Gemini]_Điểm_Trôi_chảy_30%": g.get("Điểm_Trôi_chảy_30%", ""),
            "[P2_Gemini]_Tổng_điểm_Rubric_5.0": g.get("Tổng_điểm_Rubric_5.0", ""),
            "[P2_Gemini]_Nhận_xét_đánh_giá_lỗi": g.get("Lỗi_sau_dấu_chấm_và_nhận_xét", ""),
            "[P2_Gemini]_Câu_trả_lời_chi_tiết": g.get("Câu_trả_lời_đầy_đủ", ""),

            # Phần 3: Qwen 4-bit
            "[P3_Qwen4b]_TTFT_giây": q4.get("TTFT_trung_bình_s", ""),
            "[P3_Qwen4b]_Tốc_độ_tok_s": q4.get("Tốc_độ_trung_bình_tok_s", ""),
            "[P3_Qwen4b]_Số_tokens": q4.get("Số_tokens_ước_tính", ""),
            "[P3_Qwen4b]_Điểm_Chính_xác_40%": q4.get("Điểm_Chính_xác_40%", ""),
            "[P3_Qwen4b]_Điểm_Đầy_đủ_30%": q4.get("Điểm_Đầy_đủ_30%", ""),
            "[P3_Qwen4b]_Điểm_Trôi_chảy_30%": q4.get("Điểm_Trôi_chảy_30%", ""),
            "[P3_Qwen4b]_Tổng_điểm_Rubric_5.0": q4.get("Tổng_điểm_Rubric_5.0", ""),
            "[P3_Qwen4b]_Nhận_xét_đánh_giá_lỗi": q4.get("Lỗi_sau_dấu_chấm_và_nhận_xét", ""),
            "[P3_Qwen4b]_Câu_trả_lời_chi_tiết": q4.get("Câu_trả_lời_đầy_đủ", ""),
            "[P3_Qwen4b]_GPU_sử_dụng": q4.get("GPU", "NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)"),

            # Phần 4: Qwen 8-bit
            "[P4_Qwen8b]_TTFT_giây": q8.get("TTFT_trung_bình_s", ""),
            "[P4_Qwen8b]_Tốc_độ_tok_s": q8.get("Tốc_độ_trung_bình_tok_s", ""),
            "[P4_Qwen8b]_Số_tokens": q8.get("Số_tokens_ước_tính", ""),
            "[P4_Qwen8b]_Điểm_Chính_xác_40%": q8.get("Điểm_Chính_xác_40%", ""),
            "[P4_Qwen8b]_Điểm_Đầy_đủ_30%": q8.get("Điểm_Đầy_đủ_30%", ""),
            "[P4_Qwen8b]_Điểm_Trôi_chảy_30%": q8.get("Điểm_Trôi_chảy_30%", ""),
            "[P4_Qwen8b]_Tổng_điểm_Rubric_5.0": q8.get("Tổng_điểm_Rubric_5.0", ""),
            "[P4_Qwen8b]_Nhận_xét_đánh_giá_lỗi": q8.get("Lỗi_sau_dấu_chấm_và_nhận_xét", ""),
            "[P4_Qwen8b]_Câu_trả_lời_chi_tiết": q8.get("Câu_trả_lời_đầy_đủ", ""),
            "[P4_Qwen8b]_GPU_sử_dụng": q8.get("GPU", "NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)")
        }
        out_data.append(row_item)

    with open(OUTPUT_COMPARISON_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(out_data)

    print(f"[OK] Đã xuất thành công bảng so sánh các mô hình: {OUTPUT_COMPARISON_CSV}")

if __name__ == "__main__":
    export_comparison_csv()
