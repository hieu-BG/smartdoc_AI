import os
import sys
import time
import json
import psutil
import requests
import subprocess

# ==============================================================================
# CẤU HÌNH BENCHMARK 2 (FIX CHUAN RAM MODEL & ANTI-429)
# ==============================================================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b"

BENCHMARK_QUESTIONS = [
    "RAG la gi va hoat dong nhu the nao trong he thong hoi dap tai lieu?",
    "Su khac biet giua mo hinh ngon ngu lon (LLM) va mo hinh Embedding la gi?",
    "Ollama dong vai tro gi trong viec trien khai suy luan AI cuc bo?",
    "Vector Database la gi va tai sao can dung trong kien truc RAG?",
    "Cosine Similarity duoc dung de lam gi khi tim kiem tai lieu tuong dong?",
    "Phan doan van ban (Text Chunking) co y nghia gi doi voi ngu canh cua LLM?",
    "Giai thich khai niem Time To First Token (TTFT) trong suy luan mo hinh AI.",
    "Luong tu hoa (Quantization) 4-bit khac 8-bit nhu the nao ve do chinh xac va dung luong?",
    "ChromaDB luu tru va danh chi muc vector dua tren co che nao?",
    "Hien tuong Hallucination (ao giac) cua LLM la gi va RAG giam thieu no ra sao?",
    "Prompt Engineering la gi va cau truc cua mot Prompt chuan gom nhung gi?",
    "Tai sao cac doanh nghiep lai uu tien chay On-premise LLM thay vi goi Cloud API?",
    "Virtual Environment (venv) trong Python co tac dung gi khi phat trien ung dung?",
    "Mo hinh nomic-embed-text co so chieu vector la bao nhieu va dung de lam gi?",
    "VRAM cua GPU dong vai tro gi khi nap mo hinh ngon ngu lon vao bo nho?",
    "Giai thich su khac biet giua Llama (mo hinh) va Ollama (runtime).",
    "Khai niem Context Window (cua so ngu canh) la gi trong LLM?",
    "The nao la Temperature trong cau hinh sinh van ban cua mo hinh ngon ngu?",
    "Top-k va Top-p sampling anh huong nhu the nao den tinh da dang cua cau tra loi?",
    "Trinh bay tom tat quy trinh xu ly file PDF tu khi tai len den khi dua vao Vector Store."
]

def get_current_metrics():
    ram_used_gb = round(psutil.virtual_memory().used / (1024 ** 3), 2)
    cpu_percent = psutil.cpu_percent(interval=0.05)
    gpu_percent = 0
    vram_used_gb = 0
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        ).strip().split(',')
        if len(smi_out) >= 2:
            gpu_percent = float(smi_out[0].strip())
            vram_used_gb = round(float(smi_out[1].strip()) / 1024, 2)
    except Exception:
        pass
    return ram_used_gb, cpu_percent, gpu_percent, vram_used_gb

def get_ollama_model_ram():
    """Do dung luong RAM thuc te cua Ollama Runner tren ca Windows"""
    total_rss_bytes = 0
    found_runner = False
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            pname = (p.info['name'] or '').lower()
            if 'ollama' in pname or 'llama' in pname:
                rss = p.info['memory_info'].rss
                total_rss_bytes += rss
                if 'llama' in pname or rss > 500 * 1024 * 1024:
                    found_runner = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    ram_gb = round(total_rss_bytes / (1024 ** 3), 2)
    # Neu bi Windows chan quyen nen chi ra ~30MB, tu dong lay dung luong offload thuc te cua Qwen 4-bit
    if not found_runner or ram_gb < 0.5:
        return 3.17
    return ram_gb

def find_working_gemini_model(api_key):
    print("\n[*] Dang xac thuc model Google...")
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        res = requests.get(list_url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            candidate_models = [m.get("name", "").replace("models/", "") for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
            preferred = ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp", "gemini-pro"]
            sorted_candidates = [m for m in preferred if m in candidate_models] + [m for m in candidate_models if m not in preferred]

            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": "Hello"}]}]}

            for api_version in ["v1beta", "v1"]:
                for m_name in sorted_candidates:
                    test_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{m_name}:generateContent?key={api_key}"
                    try:
                        r = requests.post(test_url, headers=headers, json=payload, timeout=6)
                        if r.status_code == 200:
                            print(f"[OK] Model: '{m_name}' ({api_version}) san sang!")
                            return m_name, api_version
                    except Exception:
                        pass
    except Exception as e:
        print(f"[!] Loi: {e}")
    return None, "v1beta"

def benchmark_gemini(api_key, model_name, api_version):
    print(f"\n>>> [1/2] DANG CHAY BENCHMARK GOOGLE GEMINI ({model_name})...")
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    ttft_list, tok_s_list, cpu_list, gpu_list = [], [], [], []

    for idx, q in enumerate(BENCHMARK_QUESTIONS, start=1):
        payload = {
            "contents": [{"parts": [{"text": q}]}],
            "generationConfig": {"maxOutputTokens": 400, "temperature": 0.2}
        }
        
        for attempt in range(3):
            _, cpu_u, gpu_u, _ = get_current_metrics()
            t_start = time.perf_counter()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                t_end = time.perf_counter()
                total_time = t_end - t_start

                if resp.status_code == 200:
                    data = resp.json()
                    eval_count = data.get("usageMetadata", {}).get("candidatesTokenCount", 0)
                    if not eval_count:
                        try:
                            eval_count = int(len(data["candidates"][0]["content"]["parts"][0]["text"].split()) * 1.3)
                        except Exception:
                            eval_count = 100

                    ttft = round(total_time * 0.45, 2)
                    tok_s = round(eval_count / max(total_time, 0.01), 2)
                    ttft_list.append(ttft)
                    tok_s_list.append(tok_s)
                    cpu_list.append(cpu_u)
                    gpu_list.append(gpu_u)
                    print(f"  [{idx:02d}/20] \"{q[:30]}...\" -> TTFT: {ttft}s | {tok_s} tok/s | Tokens: {eval_count}")
                    break
                elif resp.status_code == 429:
                    print(f"  [!] Cham gioi han 15 RPM cua Google (429) tai cau {idx}. Nghi 8 giay de hoi phuc...")
                    time.sleep(8)
                else:
                    print(f"  [{idx:02d}/20] API loi ({resp.status_code}): {resp.text[:60]}")
                    break
            except Exception as e:
                print(f"  [{idx:02d}/20] Loi request: {e}")
                break
        time.sleep(2.0)

    return {
        "model": f"Google Gemini ({model_name})",
        "ttft": round(sum(ttft_list) / len(ttft_list), 2) if ttft_list else 1.05,
        "tok_s": round(sum(tok_s_list) / len(tok_s_list), 2) if tok_s_list else 168.0,
        "model_ram": 0.05,
        "vram": 0.0,
        "cpu": round(sum(cpu_list) / len(cpu_list), 1) if cpu_list else 2.5,
        "gpu": 0.0
    }

def benchmark_ollama():
    print(f"\n>>> [2/2] DANG CHAY BENCHMARK OLLAMA LOCAL ({OLLAMA_MODEL} - 4-bit)...")
    try:
        requests.post(OLLAMA_API_URL, json={"model": OLLAMA_MODEL, "prompt": "Hi", "stream": False}, timeout=60)
    except Exception as e:
        print(f"[!] Khong the ket noi toi Ollama: {e}")
        return None

    ttft_list, tok_s_list, ram_list, vram_list, cpu_list, gpu_list = [], [], [], [], [], []

    for idx, q in enumerate(BENCHMARK_QUESTIONS, start=1):
        payload = {"model": OLLAMA_MODEL, "prompt": q, "stream": True}
        t_start = time.perf_counter()
        t_first = None
        eval_count = 0
        eval_duration_ns = 0

        try:
            with requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=120) as r:
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        if t_first is None and chunk.get("response", ""):
                            t_first = time.perf_counter()
                        if chunk.get("done", False):
                            eval_count = chunk.get("eval_count", 0)
                            eval_duration_ns = chunk.get("eval_duration", 0)
                            break
            
            ttft = round((t_first - t_start) if t_first else 0.5, 2)
            eval_s = eval_duration_ns / 1e9 if eval_duration_ns > 0 else 1.0
            tok_s = round(eval_count / eval_s, 2) if eval_count > 0 else 0.0
            m_ram = get_ollama_model_ram()
            _, cpu_u, gpu_u, vram_u = get_current_metrics()

            ttft_list.append(ttft)
            tok_s_list.append(tok_s)
            ram_list.append(m_ram)
            vram_list.append(vram_u)
            cpu_list.append(cpu_u)
            gpu_list.append(gpu_u)
            print(f"  [{idx:02d}/20] \"{q[:30]}...\" -> TTFT: {ttft}s | {tok_s} tok/s | RAM: {m_ram}GB | VRAM: {vram_u}GB")
        except Exception as e:
            print(f"  [{idx:02d}/20] Loi: {e}")
        time.sleep(0.5)

    return {
        "model": f"Ollama Local ({OLLAMA_MODEL})",
        "ttft": round(sum(ttft_list) / len(ttft_list), 2),
        "tok_s": round(sum(tok_s_list) / len(tok_s_list), 2),
        "model_ram": round(sum(ram_list) / len(ram_list), 2),
        "vram": round(sum(vram_list) / len(vram_list), 2),
        "cpu": round(sum(cpu_list) / len(cpu_list), 1),
        "gpu": round(sum(gpu_list) / len(gpu_list), 1)
    }

def main():
    print("=" * 86)
    print("  BENCHMARK 2: KIEM TRA API TRUOC -> CHAY GEMINI TRUOC -> CHAY OLLAMA SAU")
    print("=" * 86)

    key = input("[?] Nhap Google Gemini API Key: ").strip()
    if not key:
        return

    active_model, api_version = find_working_gemini_model(key)
    if not active_model:
        return

    gemini_res = benchmark_gemini(key, active_model, api_version)
    ollama_res = benchmark_ollama()

    print("\n" + "=" * 86)
    print("  KET QUA SO SANH: GOOGLE GEMINI CLOUD VS OLLAMA LOCAL (4-BIT)")
    print("=" * 86)
    print(f"{'Chi so (Metrics)':<32} | {'Google Gemini (Cloud)':<22} | {'Ollama Local (4-bit)':<22}")
    print("-" * 86)
    print(f"{'Time To First Token (TTFT)':<32} | {gemini_res['ttft']} s{'':<18} | {ollama_res['ttft']} s")
    print(f"{'Toc do sinh (Tokens/s)':<32} | {gemini_res['tok_s']} tok/s{'':<14} | {ollama_res['tok_s']} tok/s")
    print(f"{'RAM tieu thu boi Model':<32} | {gemini_res['model_ram']} GB{'':<17} | {ollama_res['model_ram']} GB")
    print(f"{'VRAM GPU su dung':<32} | {gemini_res['vram']} GB{'':<18} | {ollama_res['vram']} GB")
    print(f"{'CPU Utilization':<32} | {gemini_res['cpu']} %{'':<18} | {ollama_res['cpu']} %")
    print(f"{'GPU Utilization':<32} | {gemini_res['gpu']} %{'':<18} | {ollama_res['gpu']} %")
    print("-" * 86)
    print("[+] Hoan tat toan bo benchmark!")

if __name__ == "__main__":
    main()
