import os
import time
import json
import psutil
import requests
import subprocess

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Cấu hình đúng tên model đang có trên máy bạn
MODELS_TO_BENCHMARK = [
    "qwen2.5-coder:7b",
    "qwen2.5-coder:7b-instruct-q8_0"
]

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

def get_hardware_info():
    cpu_name = os.environ.get('PROCESSOR_IDENTIFIER', 'Standard CPU')
    ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    gpu_info = "Khong co GPU NVIDIA roi (Chay CPU thuan)"
    vram_total_gb = 0
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        ).strip().split(',')
        if len(smi_out) >= 2:
            gpu_info = smi_out[0].strip()
            vram_total_gb = round(float(smi_out[1].strip()) / 1024, 2)
    except Exception:
        pass

    return {
        "cpu": cpu_name,
        "ram_total_gb": ram_total_gb,
        "gpu": gpu_info,
        "vram_total_gb": vram_total_gb
    }

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

    return {
        "ram_used_gb": ram_used_gb,
        "cpu_percent": cpu_percent,
        "gpu_percent": gpu_percent,
        "vram_used_gb": vram_used_gb
    }

def benchmark_single_query(model_name: str, prompt: str):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.3}
    }

    start_time = time.time()
    first_token_time = None
    eval_count = 0
    eval_duration_ns = 0

    initial_metrics = get_current_metrics()
    peak_ram = initial_metrics["ram_used_gb"]
    peak_vram = initial_metrics["vram_used_gb"]
    peak_cpu = initial_metrics["cpu_percent"]
    peak_gpu = initial_metrics["gpu_percent"]

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=180)
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                
                if "error" in chunk:
                    return {"status": f"Ollama Error: {chunk['error']}"}

                if first_token_time is None and chunk.get("response"):
                    first_token_time = time.time() - start_time

                curr = get_current_metrics()
                peak_ram = max(peak_ram, curr["ram_used_gb"])
                peak_vram = max(peak_vram, curr["vram_used_gb"])
                peak_cpu = max(peak_cpu, curr["cpu_percent"])
                peak_gpu = max(peak_gpu, curr["gpu_percent"])

                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)

        total_latency = time.time() - start_time
        ttft = first_token_time if first_token_time is not None else total_latency

        if eval_duration_ns > 0:
            tokens_per_sec = round(eval_count / (eval_duration_ns / 1e9), 2)
        else:
            tokens_per_sec = round(eval_count / total_latency, 2) if total_latency > 0 else 0

        return {
            "prompt": prompt,
            "total_latency_s": round(total_latency, 2),
            "ttft_s": round(ttft, 2),
            "eval_count": eval_count,
            "tokens_per_sec": tokens_per_sec,
            "peak_ram_gb": peak_ram,
            "peak_vram_gb": peak_vram,
            "peak_cpu_percent": peak_cpu,
            "peak_gpu_percent": peak_gpu,
            "status": "Success"
        }

    except Exception as e:
        return {"status": f"Request Error: {str(e)}"}

def run_full_benchmark():
    hw = get_hardware_info()
    print("=" * 80)
    print("  HE THONG DO LUONG HIEN THUC OLLAMA (BENCHMARK SYSTEM) - SMARTDOC AI")
    print("=" * 80)
    print(f"[*] PHAN CUNG:")
    print(f"    - CPU: {hw['cpu']}")
    print(f"    - Tong RAM he thong: {hw['ram_total_gb']} GB")
    print(f"    - GPU: {hw['gpu']} | VRAM: {hw['vram_total_gb']} GB")
    print("=" * 80)

    summary_results = {}

    for model in MODELS_TO_BENCHMARK:
        print(f"\n>>> DANG TIEN HANH DO MODEL: {model}...")
        results = []
        
        for idx, q in enumerate(BENCHMARK_QUESTIONS, 1):
            print(f"    [{idx}/{len(BENCHMARK_QUESTIONS)}] Dang chay: \"{q[:35]}...\"", end="", flush=True)
            res = benchmark_single_query(model, q)
            if res.get("status") == "Success":
                results.append(res)
                print(f" -> TTFT: {res['ttft_s']}s | Toc do: {res['tokens_per_sec']} tok/s | RAM: {res['peak_ram_gb']}GB")
            else:
                print(f" -> {res.get('status')}")

        if results:
            avg_latency = round(sum(r['total_latency_s'] for r in results) / len(results), 2)
            avg_ttft = round(sum(r['ttft_s'] for r in results) / len(results), 2)
            avg_tokens_sec = round(sum(r['tokens_per_sec'] for r in results) / len(results), 2)
            avg_ram = round(sum(r['peak_ram_gb'] for r in results) / len(results), 2)
            avg_vram = round(sum(r['peak_vram_gb'] for r in results) / len(results), 2)
            avg_cpu = round(sum(r['peak_cpu_percent'] for r in results) / len(results), 1)
            avg_gpu = round(sum(r['peak_gpu_percent'] for r in results) / len(results), 1)

            summary_results[model] = {
                "avg_ttft": avg_ttft,
                "avg_tokens_sec": avg_tokens_sec,
                "avg_ram": avg_ram,
                "avg_vram": avg_vram,
                "avg_cpu": avg_cpu,
                "avg_gpu": avg_gpu,
                "details": results
            }

    if summary_results:
        print("\n" + "=" * 80)
        print("  BANG TONG HOP KET QUA BENCHMARK (COPY DAN THANG VAO BAO CAO - BANG 1.2.4)")
        print("=" * 80)
        print("\n### Bang 1.2.4: Bang so sanh hieu nang cac mo hinh tren Ollama\n")
        print("| Model | TTFT (s) | Toc do (Token/s) | RAM tieu thu (GB) | VRAM (GB) | Tai CPU (%) | Tai GPU (%) |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m, d in summary_results.items():
            print(f"| **{m}** | {d['avg_ttft']}s | {d['avg_tokens_sec']} tok/s | {d['avg_ram']} GB | {d['avg_vram']} GB | {d['avg_cpu']}% | {d['avg_gpu']}% |")

if __name__ == "__main__":
    run_full_benchmark()
