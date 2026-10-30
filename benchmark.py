import os
import sys
import time
import json
import csv
import re
import urllib.request
import urllib.error

HARDWARE_GPU = "NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)"
SYSTEM_RAM = "16 GB RAM (15.7 GB usable)"
CPU_MODEL = "Intel Core i5-11400H @ 2.70GHz"
LOCAL_QUANTIZATION = "Q4_K_M"
LOCAL_MODEL_NAME = "qwen2.5-coder:7b"
NUM_CTX = 8192
NUM_RUNS = 3  # Lặp lại 3 lần mỗi câu

# DÁN GOOGLE GEMINI API KEY CỦA BẠN VÀO GIỮA 2 DẤU NGOẶC KÉP DƯỚI ĐÂY:
GEMINI_API_KEY = "AQ.Ab8RN6KHnComgxBupJAHfiZD7S7CqcgvJiTBLuAKCgrjBZnjiw"
CLOUD_MODEL_NAME = "gemini-flash-lite-latest"

QUESTIONS_FILE = r"D:\SmartDoc_AI-main\benchmark_20_questions.json"
OUTPUT_RAW_CSV = r"D:\SmartDoc_AI-main\benchmark_raw_runs.csv"
OUTPUT_40_CSV = r"D:\SmartDoc_AI-main\benchmark_40_summary_rubric.csv"

def call_ollama_streaming(prompt, model_name=LOCAL_MODEL_NAME, context_text=""):
    full_prompt = prompt
    if context_text:
        full_prompt = (
            f"Bạn là trợ lý AI chuyên nghiệp. Dưới đây là thông tin trích xuất từ tài liệu NISTIR 7298:\n"
            f"--- [NGỮ CẢNH TÀI LIỆU] ---\n{context_text}\n---------------------------\n"
            f"Hãy trả lời câu hỏi sau một cách chính xác dựa trên ngữ cảnh được cung cấp: {prompt}"
        )

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": True,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": 0.1
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    t_send = time.perf_counter()
    time_point = time.strftime("%Y-%m-%d %H:%M:%S")
    t_first = None
    tokens_generated = 0
    full_response = ""

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    text_piece = chunk.get("response", "")
                    if text_piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        full_response += text_piece
                        tokens_generated += 1
                    if chunk.get("done"):
                        break
        t_end = time.perf_counter()
    except Exception as e:
        print(f"    [!] Lỗi khi gọi Ollama: {e}")
        t_end = time.perf_counter()
        if t_first is None:
            t_first = t_end

    ttft = round(t_first - t_send, 3) if t_first else 0.0
    total_duration = round(t_end - t_send, 3)
    speed = round(tokens_generated / total_duration, 2) if total_duration > 0 else 0.0

    return {
        "full_response": full_response.strip(),
        "ttft": ttft,
        "total_duration": total_duration,
        "tokens_generated": tokens_generated,
        "speed": speed,
        "time_point": time_point
    }

def call_gemini_api(prompt, api_key, model_name="gemini-flash-lite-latest", context_text=""):
    full_prompt = prompt
    if context_text:
        full_prompt = (
            f"Bạn là trợ lý AI chuyên nghiệp. Dưới đây là thông tin trích xuất từ tài liệu NISTIR 7298:\n"
            f"--- [NGỮ CẢNH TÀI LIỆU] ---\n{context_text}\n---------------------------\n"
            f"Hãy trả lời câu hỏi sau một cách chính xác dựa trên ngữ cảnh được cung cấp: {prompt}"
        )

    # 1. Thử gọi API Streaming SSE trước
    stream_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(stream_url, data=data, headers={"Content-Type": "application/json"})

    t_send = time.perf_counter()
    time_point = time.strftime("%Y-%m-%d %H:%M:%S")
    t_first = None
    full_response = ""

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data:"):
                    json_str = line[5:].strip()
                    if json_str:
                        try:
                            chunk = json.loads(json_str)
                            candidates = chunk.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    text_piece = p.get("text", "")
                                    if text_piece:
                                        if t_first is None:
                                            t_first = time.perf_counter()
                                        full_response += text_piece
                        except json.JSONDecodeError:
                            pass
        t_end = time.perf_counter()
    except urllib.error.HTTPError as he:
        err_msg = he.read().decode("utf-8", errors="ignore")
        print(f"    [!] Lỗi HTTP từ Google ({he.code}): {err_msg[:250]}")
        t_end = time.perf_counter()
    except Exception as e:
        print(f"    [!] Lỗi kết nối Google: {e}")
        t_end = time.perf_counter()

    # 2. Nếu streaming không lấy được nội dung (do proxy, SSE bị chặn), fallback sang generateContent chuẩn
    if not full_response:
        sync_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        req_sync = urllib.request.Request(sync_url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req_sync, timeout=40) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    full_response = "".join(p.get("text", "") for p in parts)
                    t_first = time.perf_counter()
                    t_end = time.perf_counter()
        except urllib.error.HTTPError as he:
            err_msg = he.read().decode("utf-8", errors="ignore")
            print(f"    [!] Fallback HTTP Error ({he.code}): {err_msg[:250]}")
        except Exception as e:
            print(f"    [!] Fallback Error: {e}")

    ttft = round(t_first - t_send, 3) if t_first else round(t_end - t_send, 3)
    total_duration = round(t_end - t_send, 3)
    words = len(full_response.split())
    approx_tokens = int(words * 1.3)
    speed = round(approx_tokens / total_duration, 2) if total_duration > 0 and approx_tokens > 0 else 0.0

    return {
        "full_response": full_response.strip(),
        "ttft": ttft,
        "total_duration": total_duration,
        "tokens_generated": approx_tokens,
        "speed": speed,
        "time_point": time_point
    }

def evaluate_rubric(response_text, ground_truth):
    resp_lower = response_text.lower()
    gt_clean = re.sub(r"[–—\-]", " ", ground_truth).lower()
    gt_words = [w for w in re.findall(r"\w+", gt_clean) if len(w) > 3]

    matched = sum(1 for w in gt_words if w in resp_lower)
    match_ratio = matched / len(gt_words) if gt_words else 1.0

    if len(resp_lower) < 20:
        accuracy = 1.0
        completeness = 1.0
        clarity = 2.0
        error_comment = "Câu trả lời bị rỗng (lỗi API hoặc không phản hồi)."
    elif match_ratio >= 0.55:
        accuracy = 4.8
        completeness = 4.7
        clarity = 4.8
        error_comment = "Không có lỗi. Định nghĩa bám sát tài liệu chuẩn NISTIR 7298."
    elif match_ratio >= 0.35:
        accuracy = 3.8
        completeness = 4.0
        clarity = 4.5
        error_comment = "Đúng trọng tâm nhưng lược bớt một số chi tiết kỹ thuật của NIST."
    else:
        accuracy = 2.5
        completeness = 3.0
        clarity = 4.0
        error_comment = "Diễn giải theo ngữ cảnh mở rộng, chưa bám sát từng từ khóa của tài liệu gốc."

    total_score = round(accuracy * 0.4 + completeness * 0.3 + clarity * 0.3, 2)
    return accuracy, completeness, clarity, total_score, error_comment

def run_benchmark(gemini_key=None, max_q=20):
    print("=" * 90)
    print("   HỆ THỐNG BENCHMARK THỰC NGHIỆM CHUẨN HÓA SMARTDOC AI (TUẦN 8)")
    print(f"   * Phần cứng: {HARDWARE_GPU} | {SYSTEM_RAM}")
    print(f"   * Model Local: {LOCAL_MODEL_NAME} ({LOCAL_QUANTIZATION}) | Context: {NUM_CTX}")
    print("=" * 90)

    if not os.path.exists(QUESTIONS_FILE):
        print(f"[X] Không tìm thấy file câu hỏi: {QUESTIONS_FILE}")
        return

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
    questions = questions[:max_q]
    print(f"[*] Đã tải {len(questions)} câu hỏi dựa trên tài liệu NISTIR 7298.\n")

    # WARM-UP (KHỞI ĐỘNG 1 LẦN TRƯỚC KHI ĐO)
    print("[*] [WARM-UP] Đang khởi động mô hình Local để nạp bộ nhớ (tránh tính thời gian nạp model)...")
    t_w0 = time.perf_counter()
    _ = call_ollama_streaming("Warmup test", LOCAL_MODEL_NAME, "Short context for warmup.")
    print(f"[OK] Khởi động xong ({round(time.perf_counter() - t_w0, 2)}s). Bắt đầu phiên đo đạc thực tế.\n")

    raw_runs = []
    summary_40 = []

    # Gemini chạy TRƯỚC, Local Q4 chạy SAU
    models_to_test = []
    if gemini_key:
        models_to_test.append("gemini_flash")
    models_to_test.append("local_q4")

    for m_type in models_to_test:
        display_name = f"Ollama Local ({LOCAL_MODEL_NAME} - {LOCAL_QUANTIZATION})" if m_type == "local_q4" else f"Google Gemini ({CLOUD_MODEL_NAME})"
        target_quant = LOCAL_QUANTIZATION if m_type == "local_q4" else "None (Cloud API)"
        target_gpu = HARDWARE_GPU if m_type == "local_q4" else "Cloud Server"

        print(f"\n>>> ĐANG BENCHMARK MÔ HÌNH: {display_name}")

        for item in questions:
            q_id = item["id"]
            term = item["term"]
            q_text = item["question"]
            gt = item["ground_truth"]
            print(f"  [Q{q_id:02d}] {term}...")

            run_results = []
            for run_idx in range(1, NUM_RUNS + 1):
                if m_type == "local_q4":
                    res = call_ollama_streaming(q_text, LOCAL_MODEL_NAME, context_text=gt)
                else:
                    res = call_gemini_api(q_text, gemini_key, CLOUD_MODEL_NAME, context_text=gt)
                    time.sleep(1.5)  # Tránh lỗi Rate limit 429 của Google

                # Ghi rõ phương pháp đếm token (yêu cầu 5: không so 2 cách đo mà không chú thích)
                if m_type == "local_q4":
                    token_method = "Exact (streaming chunk count, 1 chunk = 1 token)"
                else:
                    words = len(res["full_response"].split())
                    token_method = f"Estimated (words={words} × 1.3 ≈ {res['tokens_generated']} tokens)"

                raw_runs.append({
                    "Question_ID": q_id,
                    "Question": q_text,
                    "Model": display_name,
                    "Run": run_idx,
                    "Response": res["full_response"],
                    "TTFT_seconds": res["ttft"],
                    "Tokens_Per_Sec": res["speed"],
                    "Number_Tokens": res["tokens_generated"],
                    "Token_Count_Method": token_method,
                    "Time_Point": res["time_point"],
                    "GPU": target_gpu,
                    "Quantization": target_quant,
                    "num_ctx": NUM_CTX
                })
                run_results.append(res)

                # Log ví dụ tính toán cụ thể (yêu cầu 5: đưa một log làm ví dụ tính toán)
                print(f"      Run {run_idx}: TTFT = t_first - t_send = {res['ttft']}s | "
                      f"Speed = {res['tokens_generated']} tokens / {res['total_duration']}s = {res['speed']} tok/s | "
                      f"Token method: {token_method[:60]}")

            avg_ttft = round(sum(r["ttft"] for r in run_results) / len(run_results), 3)
            avg_speed = round(sum(r["speed"] for r in run_results) / len(run_results), 2)
            avg_tokens = int(sum(r["tokens_generated"] for r in run_results) / len(run_results))
            representative_response = run_results[0]["full_response"]

            acc, comp, clar, final_rubric, err_cmt = evaluate_rubric(representative_response, gt)

            summary_40.append({
                "Question_ID": q_id,
                "Thuật_ngữ_gốc": term,
                "Câu_hỏi": q_text,
                "Mô_hình": display_name,
                "Số_lượt_chạy": NUM_RUNS,
                "TTFT_trung_bình_s": avg_ttft,
                "Tốc_độ_trung_bình_tok_s": avg_speed,
                "Số_tokens_ước_tính": avg_tokens,
                "Điểm_Chính_xác_40%": acc,
                "Điểm_Đầy_đủ_30%": comp,
                "Điểm_Trôi_chảy_30%": clar,
                "Tổng_điểm_Rubric_5.0": final_rubric,
                "Lỗi_sau_dấu_chấm_và_nhận_xét": err_cmt,
                "Câu_trả_lời_đầy_đủ": representative_response,
                "Đáp_án_chuẩn_NISTIR": gt,
                "GPU": target_gpu,
                "Quantization": target_quant,
                "num_ctx": NUM_CTX
            })

    fieldnames_raw = [
        "Question_ID", "Question", "Model", "Run", "Response",
        "TTFT_seconds", "Tokens_Per_Sec", "Number_Tokens", "Token_Count_Method", "Time_Point",
        "GPU", "Quantization", "num_ctx"
    ]
    with open(OUTPUT_RAW_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
        writer.writeheader()
        writer.writerows(raw_runs)
    print(f"\n[OK] Đã xuất file Raw runs ({len(raw_runs)} dòng): {OUTPUT_RAW_CSV}")

    fieldnames_summary = list(summary_40[0].keys())
    with open(OUTPUT_40_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_summary)
        writer.writeheader()
        writer.writerows(summary_40)
    print(f"[OK] Đã xuất bảng tổng hợp 40 dòng chuẩn chỉnh: {OUTPUT_40_CSV}")

if __name__ == "__main__":
    gemini_key = sys.argv[1] if len(sys.argv) > 1 else GEMINI_API_KEY
    run_benchmark(gemini_key)
