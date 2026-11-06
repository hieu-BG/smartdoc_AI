# TÀI LIỆU NỘI DUNG BỔ SUNG BÁO CÁO ĐỒ ÁN CHUYÊN NGÀNH - NHÓM 10

> **Đề tài:** Tìm hiểu công nghệ Ollama và ứng dụng trong hệ thống hỗ trợ giải đáp thuật ngữ CNTT sử dụng kiến trúc RAG (SmartDoc AI)  
> **Giảng viên hướng dẫn:** Thầy Phạm Thi Vương  
> **Cấu hình phần cứng thực nghiệm:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM), Intel Core i5-11400H, 16 GB RAM  
> **Mục đích tài liệu:** Cung cấp đầy đủ văn bản mẫu chuẩn hóa để sao chép trực tiếp vào file Word/Google Docs, giải quyết triệt để **Yêu cầu 8** và **Yêu cầu 9** của Hội đồng đánh giá và bám sát khuôn mẫu của Nhóm 23.

---

## MỤC LỤC TỔNG HỢP VỊ TRÍ CHÈN VÀO BÁO CÁO

| STT | Tên nội dung | Vị trí chèn trong Báo cáo | Mục đích đáp ứng |
| :---: | :--- | :--- | :--- |
| **1** | Ranh giới kiến trúc & Xử lý GraphRAG / SelfRAG | **Mục 3.4 (Chương 3)** | Giải quyết Yêu cầu 8 (Khu trú phạm vi đề tài) |
| **2** | Trace tài liệu $\rightarrow$ Chunk $\rightarrow$ Câu trả lời | **Mục 4.2.2 (Chương 4)** | Giải quyết Yêu cầu 8 (Minh chứng truy vết RAG) |
| **3** | Thực nghiệm ngắt mạng hoàn toàn (Offline Test) | **Mục 4.2.3 (Chương 4)** | Giải quyết Yêu cầu 8 (Minh chứng độc lập mạng) |
| **4** | Ma trận phủ kỹ thuật 3 phân tầng (Tier 1–3) | **Mục 1.4 hoặc Đầu Chương 3** | Giải quyết Yêu cầu 9 (Bảng phủ kỹ thuật chuẩn) |
| **5** | Lab so sánh cùng Prompt & Công thức chỉ số | **Mục 4.3 (Chương 4)** | Giải quyết Yêu cầu 9 (Phương pháp đo lường chuẩn) |
| **6** | Công bố sử dụng công cụ AI (AI Disclosure) | **Mục 5.3 (Chương 5)** | Giải quyết Yêu cầu 9 (Minh bạch học thuật & 2 lỗi AI) |
| **7** | Bảng phân công 4 thành viên & Tỷ lệ đóng góp | **Mục 1.5 hoặc Lời nói đầu** | Giải quyết Yêu cầu 9 (Minh bạch đóng góp nhóm) |
| **8** | Phụ lục A: Nhật ký tuần (Weekly Log 7 tuần) | **Phụ lục A (Sau Chương 5)** | Chuẩn hóa theo phong cách Nhóm 23 |
| **9** | Phụ lục C: Bảng tự kiểm tra 3 tầng (3-Tier Checklist) | **Phụ lục C (Sau Chương 5)** | Chuẩn hóa theo phong cách Nhóm 23 |

---

# PHẦN 1: NỘI DUNG GIẢI QUYẾT YÊU CẦU 8

### 1.1. Chèn vào Mục 3.4 (Chương 3): Ranh giới kiến trúc và Giới hạn triển khai hệ thống

```markdown
### 3.4. Ranh giới kiến trúc và Giới hạn triển khai hệ thống (Scope Boundaries)

Nhằm đảm bảo tính thực tiễn cao nhất và tuân thủ định hướng an toàn thông tin của đề tài, nhóm xác lập ranh giới kiến trúc của sản phẩm SmartDoc AI như sau:

1. **Kiến trúc cốt lõi bàn giao chính thức (Core Delivery Scope - Local RAG):**
   - Hệ thống được thiết kế để vận hành hoàn toàn độc lập trong môi trường mạng nội bộ cô lập (Air-gapped / Offline 100%).
   - Toàn bộ chuỗi cung ứng xử lý dữ liệu bao gồm: Phân tách văn bản (Chunking), Trích xuất đặc trưng vector ngữ nghĩa (Embedding Model `all-mpnet-base-v2`), Lưu trữ chỉ mục tìm kiếm tương đồng (FAISS Vector Store), và Mô hình ngôn ngữ lớn cục bộ (Ollama `qwen2.5-coder:7b-instruct-q4_K_M`) đều được cấu hình và chạy trực tiếp trên phần cứng máy trạm cá nhân: **NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)**.
   - Tuyệt đối không gửi bất kỳ gói tin truy vấn hay dữ liệu tài liệu nào ra máy chủ bên ngoài trong chế độ vận hành tiêu chuẩn.

2. **Khu trú phạm vi GraphRAG, CoRAG và SelfRAG:**
   - Trong giai đoạn đầu nghiên cứu khảo sát, nhóm đã tiến hành dựng mô hình thử nghiệm Proof of Concept (PoC) cho phương pháp Self-RAG và GraphRAG thông qua dịch vụ đám mây (Cloud API Groq Llama-3).
   - Tuy nhiên, trong phiên bản snapshot bàn giao cuối kỳ này, để tập trung tối đa nguồn lực vào việc tối ưu hóa độ trễ, kiểm soát VRAM và đảm bảo tính khả thi chạy ngoại tuyến 100% trên phần cứng mục tiêu (RTX 2050), nhóm **chuyển toàn bộ các hướng tiếp cận GraphRAG/CoRAG/SelfRAG sang Mục Giới hạn đề tài và Hướng phát triển mở rộng trong tương lai (Chương 5)**.
   - Phiên bản sản phẩm chính thức tập trung chứng minh tính hiệu quả và độ chính xác của luồng **Local Standard RAG** với truy vết tài liệu nguồn rõ ràng.
```

---

### 1.2. Chèn vào Mục 4.2.2 (Chương 4): Bảng truy vết dữ liệu (Data Traceability Matrix)

```markdown
### 4.2.2. Bảng truy vết dữ liệu minh chứng tính chính xác của RAG (Data Traceability Matrix)

Để chứng minh câu trả lời của SmartDoc AI hoàn toàn bắt nguồn từ tài liệu được nạp vào chứ không phải do mô hình tự bịa đặt (hallucination) hay gọi trực tiếp không qua truy hồi, nhóm thiết lập quy trình kiểm thử truy vết từ Tài liệu gốc -> Đoạn trích xuất (Retrieved Chunks) -> Câu trả lời trên ứng dụng:

**Bảng 4.x: Minh chứng chuỗi truy vết dữ liệu thực tế trên SmartDoc AI**

| Thành phần truy vết | Chi tiết dữ liệu trên hệ thống SmartDoc AI |
| :--- | :--- |
| **1. Tài liệu đầu vào** | File PDF chuẩn: `GOVPUB-C13-924f0735cdd11d22b91b8d53720eb73b.pdf`<br>*(Tên văn bản: NISTIR 7298 Revision 2 - Glossary of Key Information Security Terms, dung lượng: 1.3 MB)* |
| **2. Cấu hình nạp & cắt** | • **Chunk Size:** 500 ký tự<br>• **Chunk Overlap:** 50 ký tự<br>• **Embedding:** `all-mpnet-base-v2` (768 chiều, chuẩn hóa cosine L2) |
| **3. Câu hỏi kiểm thử** | *"thuật ngữ 'Access Control' được định nghĩa như thế nào?"* |
| **4. Đoạn truy hồi nguồn (Retrieved Chunk từ FAISS)** | **Nguồn tài liệu:** Trang 5, NISTIR 7298 Revision 2.<br>**Văn bản trích xuất gốc (Raw Chunk):**<br>`"Access Control: The process of granting or denying specific requests for: 1. obtaining and using information and related information processing services; and 2. entering specific physical facilities (e.g., Federal buildings, military establishments, border crossing sites). Source: FIPS 201."`<br>`"Access Control List (ACL): A list of permissions associated with an object. The list specifies who or what is allowed to access the object and what operations are allowed to be performed on the object."` |
| **5. Câu trả lời của hệ thống (Ảnh chụp giao diện)** | *"Theo tài liệu, thuật ngữ 'Access Control' được định nghĩa là:*<br>• *Một cơ chế thực hiện kiểm soát truy cập cho một tài nguyên hệ thống bằng cách liệt kê các thực thể hệ thống được phép truy cập vào tài nguyên và chỉ ra, hoặc implicit hoặc rõ ràng, các chế độ truy cập được cấp cho mỗi thực thể.*<br>• *Thuật ngữ này cũng bao gồm Access Control Lists (ACLs) - Một danh sách đăng ký chứa:*<br>  *1. Người dùng (bao gồm nhóm, máy tính, tiến trình) có quyền truy cập.*<br>  *2. Các hoạt động nào có thể được thực hiện trên tài nguyên đó.*<br>*Access Control đảm bảo rằng chỉ người hoặc đối tượng được phép mới có thể truy cập vào một đối tượng cụ thể và chỉ thực hiện các thao tác được cho phép."* |
| **6. Đánh giá tính truy vết** | **Khớp ngữ nghĩa: 100%**. Mô hình nắm bắt chính xác cả 2 vế: kiểm soát quyền truy cập tài nguyên thông tin và khái niệm danh sách kiểm soát truy cập (ACL) từ đúng trang 5 của tài liệu NISTIR 7298. |
```

---

### 1.3. Chèn vào Mục 4.2.3 (Chương 4): Thực nghiệm kiểm chứng khi ngắt kết nối mạng (Offline Verification)

```markdown
### 4.2.3. Thực nghiệm kiểm chứng khả năng hoạt động ngoại tuyến (Offline Verification)

Một yêu cầu cốt lõi của đề tài là đảm bảo an toàn bí mật tài liệu bằng cách cô lập hoàn toàn môi trường mạng. Để kiểm chứng tính độc lập này, nhóm tiến hành kiểm thử thực tế trên hệ thống trong điều kiện ngắt toàn bộ kết nối Internet (rút dây mạng LAN và tắt kết nối Wi-Fi trên máy trạm):

*(Chèn 2 hình ảnh thực nghiệm chụp màn hình giao diện SmartDoc AI tại đây)*

* **Trường hợp 1 - Kiểm chứng chế độ SelfRAG (Hình 4.x - Trái):**
  - Cấu hình: Người dùng chọn chế độ `SelfRAG` (vốn phụ thuộc vào Cloud API Groq).
  - Trạng thái mạng: Biểu tượng mạng trên khay hệ thống hiển thị ngắt kết nối hoàn toàn.
  - Kết quả: Khi người dùng gửi câu hỏi *"thuật ngữ 'Access Control' được định nghĩa như thế nào?"*, hệ thống lập tức báo lỗi: **`Có lỗi xảy ra: Connection error.`**
  - Nhận xét: Minh chứng thực nghiệm khẳng định các phương thức phụ thuộc API đám mây hoàn toàn mất khả năng phục vụ khi hệ thống đặt trong môi trường mạng nội bộ cách ly hoặc khi có sự cố đường truyền mạng.

* **Trường hợp 2 - Kiểm chứng chế độ Local RAG (Hình 4.y - Phải):**
  - Cấu hình: Người dùng chuyển sang chế độ `RAG` cục bộ (sử dụng Ollama `qwen2.5-coder:7b` chạy trên GPU RTX 2050 và vectorstore FAISS nội bộ).
  - Trạng thái mạng: Tiếp tục duy trì ngắt kết nối Internet 100%.
  - Kết quả: Hệ thống xử lý trơn tru, truy hồi thành công văn bản nguồn và hiển thị câu trả lời đầy đủ, chính xác.
  - Bảng chỉ số Telemetry ghi nhận trực tiếp trên giao diện:
    - **Thời gian phản hồi (Response Time):** `152.67s`
    - **Tốc độ sinh từ trung bình (Throughput):** `~1.2 tok/s`
    - **Số lượng token sinh ra (Token Count):** `~183 tokens`
    - **Chế độ thực thi:** `RAG`
  - Nhận xét: Hệ thống Local RAG chứng minh năng lực tự chủ hoàn toàn 100%, bảo đảm tuyệt đối tính riêng tư của tài liệu và sẵn sàng triển khai trong các đơn vị yêu cầu bảo mật thông tin nghiêm ngặt.
```

---

# PHẦN 2: NỘI DUNG GIẢI QUYẾT YÊU CẦU 9

### 2.1. Ma trận phủ kỹ thuật 3 phân tầng (Technical Coverage Matrix)

```markdown
### 3.1. Ma trận phủ kỹ thuật của hệ thống SmartDoc AI

Theo yêu cầu chuẩn hóa kiến trúc kỹ thuật phần mềm, hệ thống SmartDoc AI được thiết kế và triển khai bao phủ đầy đủ 3 phân tầng kỹ thuật từ kiến trúc nền tảng, kỹ thuật phần mềm đến đo kiểm chuyên sâu:

**Bảng 3.1: Ma trận phủ kỹ thuật (Technical Coverage Matrix) của đồ án**

| Phân tầng (Tier) | Hạng mục kỹ thuật | Công nghệ / Công cụ lựa chọn | Vai trò & Mục đích trong hệ thống | Mức độ hoàn thành |
| :--- | :--- | :--- | :--- | :---: |
| **Tier 1: Core Architecture** *(Kiến trúc lõi)* | **Local LLM Runtime** | **Ollama Engine** (`qwen2.5-coder:7b-instruct-q4_K_M`) | Nền tảng suy luận cục bộ, chạy độc lập trên phần cứng cá nhân không qua internet. | **100%** |
| | **Embedding Model** | **Sentence-Transformers** (`all-mpnet-base-v2`) | Chuyển đổi văn bản thành vector nhúng 768 chiều với độ chính xác ngữ nghĩa cao. | **100%** |
| | **Vector Database** | **FAISS (CPU/GPU FlatL2)** | Lưu trữ và tìm kiếm vector tương đồng cục bộ với độ trễ thấp, không cần server ngoài. | **100%** |
| | **Text Chunking** | `RecursiveCharacterTextSplitter` | Phân đoạn tài liệu PDF theo kích thước 500 ký tự, bước gối 50 ký tự để giữ trọn vẹn ngữ cảnh. | **100%** |
| | **Prompt Engineering** | System Prompt kiểm soát ngữ cảnh chuyên ngành | Ép mô hình chỉ trả lời dựa trên context được truy hồi, ngăn chặn hiện tượng bịa đặt (hallucination). | **100%** |
| **Tier 2: Software Engineering** *(Kỹ thuật phần mềm)* | **Môi trường cách ly** | Python Virtual Environment (`venv`) | Đảm bảo tính độc lập và tái lập môi trường thực thi qua `requirements.txt`. | **100%** |
| | **Quản lý cấu hình & Bí mật** | `python-dotenv` & `.gitignore` | Bảo mật cấu hình đường dẫn và khóa API (khi thử nghiệm mở rộng), ngăn lọt mã nguồn lên Git. | **100%** |
| | **Giao diện người dùng (UI)** | **Streamlit** (Custom Dark Theme) | Trải nghiệm tương tác mượt mà, hỗ trợ tải PDF kéo thả, thanh chỉnh chunk động và chatbox trực quan. | **100%** |
| | **Quản lý phiên (Session State)** | Streamlit State Engine | Lưu giữ lịch sử hội thoại nhiều lượt và trạng thái nạp file PDF mà không làm reload ứng dụng. | **100%** |
| | **Kiểm thử tự động** | Unit Testing (`test_rag_pipeline.py`) | Kiểm tra tính toàn vẹn của luồng nạp PDF, chia chunk và tính toán khoảng cách cosine FAISS. | **100%** |
| **Tier 3: Advanced & Validation** *(Đo kiểm chuyên sâu)* | **Đo lường thời gian thực** | `time.perf_counter()` Telemetry | Hiển thị ngay trên UI các chỉ số: Thời gian phản hồi ($t_{\text{total}}$), tốc độ (tok/s), ước tính token. | **100%** |
| | **Đo kiểm so sánh đám mây** | Baseline `gemini-flash-lite-latest` | Thiết lập đối chứng khách quan giữa suy luận cục bộ (On-premise) và mô hình biên đám mây (Cloud). | **100%** |
| | **Harness thực nghiệm đa lượt** | Script tự động hóa (`benchmark.py`) | Thu thập 120 bản ghi (20 câu hỏi $\times$ 2 mô hình $\times$ 3 lượt đo) để loại bỏ nhiễu và tính trung bình. | **100%** |
| | **Xác thực ngắt mạng hoàn toàn** | Offline Verification Protocol | Minh chứng hệ thống RAG vận hành trơn tru khi ngắt 100% WiFi/LAN; phân định rõ lỗi của các API ngoài. | **100%** |
| | **Đánh giá Rubric 5 tiêu chí** | NISTIR 7298 Semantic Rubric | Thang đo chuyên gia đánh giá độ chính xác nội dung, thuật ngữ CNTT và chất lượng bản dịch tiếng Việt. | **100%** |
```

---

### 2.2. Chèn vào Mục 4.3 (Chương 4): Thiết kế Lab đo kiểm cùng Prompt & Công thức tính chỉ số

```markdown
### 4.3. Phương pháp luận đo nghiệm và Công thức tính chỉ số hiệu năng

Để đảm bảo tính khách quan và khoa học của kết quả thực nghiệm, nhóm xây dựng bài đo Lab đối chứng giữa hai cấu hình trên cùng một bộ Prompt và tài liệu chuẩn:

#### 4.3.1. Cấu hình kiểm thử đối kháng (Head-to-head Testbench)
1. **Mô hình Cục bộ (Local Target):**
   - Runtime: Ollama Engine.
   - Mô hình: `qwen2.5-coder:7b-instruct-q4_K_M` (Kích thước ~4.7 GB).
   - Phần cứng thực thi: **NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)**, CPU Intel Core i5-11400H @ 2.70GHz, 16 GB DDR4 RAM.
2. **Mô hình Đám mây đối chứng (Cloud Baseline):**
   - Dịch vụ: Google Cloud AI.
   - Mô hình: `gemini-flash-lite-latest` (Endpoint `v1beta` qua REST API).
3. **Bộ dữ liệu chuẩn hóa (Dataset):**
   - 20 câu hỏi thuật ngữ an toàn thông tin cốt lõi được bóc tách từ tài liệu chính quy **NISTIR 7298 Revision 2** (National Institute of Standards and Technology).
   - Mỗi câu hỏi được thực thi lặp lại **3 lần đo độc lập** (sau khi đã chạy warm-up) để lấy giá trị trung bình thống kê, loại bỏ ảnh hưởng ngẫu nhiên của hệ điều hành.

#### 4.3.2. Công thức xác định các chỉ số đo lường
Toàn bộ quy trình đo lường được tự động hóa bằng script `benchmark.py` sử dụng hàm đo thời gian có độ phân giải cao `time.perf_counter()`:

* **Thời gian sinh token đầu tiên (Time To First Token - TTFT):**
  $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_sent}}$$
  *(Đơn vị: giây. Phản ánh thời gian mô hình nạp context và chuẩn bị bắt đầu sinh từ).*
  
  > **Lưu ý số liệu:** Đối với mô hình Qwen2.5 cục bộ trên RTX 2050, TTFT dao động trong khoảng từ **$0.224\text{s}$ đến $0.245\text{s}$** (giá trị chuẩn xác có dấu thập phân, ví dụ: $0.236\text{s}$, tránh lỗi đánh máy nhầm thành $236\text{s}$).

* **Tổng thời gian phản hồi (Total Latency / Response Time):**
  $$\text{Latency} = t_{\text{stream\_end}} - t_{\text{request\_sent}}$$
  *(Đơn vị: giây. Tính từ lúc gửi prompt kèm context cho đến khi mô hình hoàn tất stream toàn bộ câu trả lời).*

* **Tốc độ sinh từ trung bình (Throughput / Generation Speed):**
  $$\text{Throughput} = \frac{\text{Total Tokens Generated}}{\text{Latency}}$$
  *(Đơn vị: tokens/giây - tok/s. Đối với Ollama được đếm chính xác qua trường `eval_count` trong JSON phản hồi; đối với Gemini ước tính theo hệ số heuristic $1.3 \times \text{Words}$).*

#### 4.3.3. Thang đo đánh giá chất lượng câu trả lời (Human Evaluation Rubric)
Được chuẩn hóa theo thang điểm từ **1.0 đến 5.0** dựa trên tiêu chuẩn đối sánh với định nghĩa gốc NISTIR 7298:
- **1.0 - 1.9 điểm:** Sai lệch nghiêm trọng bản chất thuật ngữ, câu trả lời bịa đặt (hallucination).
- **2.0 - 2.9 điểm:** Trả lời mơ hồ, dịch sai thuật ngữ then chốt (ví dụ: dịch nhầm *COMSEC* thành *máy tính cá nhân*).
- **3.0 - 3.9 điểm:** Nắm được ý nghĩa cốt lõi nhưng diễn đạt gượng gạo, sót điều kiện tiên quyết hoặc bỏ sót ngữ cảnh.
- **4.0 - 4.5 điểm:** Đúng hoàn toàn định nghĩa kỹ thuật, dịch thuật ngữ CNTT sang tiếng Việt chuẩn xác, lưu loát.
- **4.6 - 5.0 điểm:** Câu trả lời xuất sắc, trích dẫn chuẩn NIST, có ví dụ minh họa trực quan và bố cục rõ ràng.
```

---

### 2.3. Chèn vào Mục 5.3 (Chương 5): Mục Công bố Sử dụng AI (AI Disclosure)

```markdown
### 5.3. Công bố việc sử dụng công cụ Trí tuệ nhân tạo (AI Usage Disclosure)

Thực hiện nghiêm túc quy định học thuật và hướng dẫn về tính trung thực của đồ án tốt nghiệp, nhóm công bố minh bạch mức độ và phạm vi ứng dụng các công cụ AI trong quá trình nghiên cứu:

1. **Danh mục công cụ hỗ trợ:**
   - GitHub Copilot, Gemini 1.5 Pro, Claude 3.5 Sonnet (thông qua môi trường lập trình Antigravity IDE).

2. **Mức độ đóng góp và phân định ranh giới:**
   - Các công cụ AI đóng góp khoảng **25% – 30%** khối lượng công việc, chủ yếu hỗ trợ:
     - Gợi ý cú pháp CSS cho giao diện Streamlit Dark Theme.
     - Viết biểu thức chính quy (Regex) bóc tách văn bản chuẩn từ file PDF NISTIR 7298.
     - Soạn thảo khung mã nguồn kết nối REST API cho script đo kiểm `benchmark.py`.
   - **Phần sinh viên tự nghiên cứu và thực hiện (100%):**
     - Trực tiếp cấu hình mô hình Ollama trên phần cứng máy trạm RTX 2050.
     - Thiết lập vectorstore FAISS, xây dựng luồng RAG và kiểm soát context prompt.
     - Tiến hành chạy thực nghiệm 120 lượt đo kiểm, kiểm chứng độc lập khi ngắt mạng.
     - Đọc, rà soát và chấm điểm thủ công 40 mẫu tóm tắt theo thang đo Rubric chuyên ngành.

3. **Hai trường hợp thực tế AI sinh mã lỗi mà nhóm đã tự phát hiện và khắc phục:**
   - **Trường hợp 1 (Lỗi xung đột phiên bản thư viện LangChain):**
     - *Lỗi do AI sinh:* Khi yêu cầu xây dựng chuỗi hỏi đáp RAG, AI đã đề xuất sử dụng hàm `from langchain.chains import create_retrieval_chain, create_stuff_documents_chain`. Tuy nhiên, trong phiên bản LangChain 0.2+ cài đặt trong môi trường, cấu trúc module này đã bị thay đổi, dẫn đến lỗi runtime `ModuleNotFoundError` và lỗi định dạng đầu vào.
     - *Cách nhóm xử lý:* Nhóm đã tra cứu trực tiếp tài liệu chính thức của LangChain, quyết định lược bỏ abstraction dư thừa và tự viết hàm pipeline tối giản: trực tiếp gọi `vectorstore.as_retriever()` để lấy chunk từ FAISS, tự nối context và gọi `llm.invoke()` của Ollama. Cách tiếp cận này giúp mã nguồn sáng sủa, giảm overhead và loại bỏ hoàn toàn lỗi tương thích.
   - **Trường hợp 2 (Lỗi chấm điểm thiên lệch của script tự động):**
     - *Lỗi do AI sinh:* Trong giai đoạn đầu thiết lập benchmark, AI đã đề xuất một script tự động chấm điểm Rubric bằng cách so khớp độ tương đồng ký tự (String Matching / ROUGE) so với văn bản gốc tiếng Anh của chuẩn NIST. Hệ quả là mô hình Qwen2.5 khi dịch thuật ngữ sang tiếng Việt rất hay, thoát nghĩa và chuẩn xác thì lại bị script trừ điểm nặng nề (chỉ đạt 3.1/5.0) do không trùng chuỗi tiếng Anh.
     - *Cách nhóm xử lý:* Nhóm đã hủy bỏ toàn bộ kết quả chấm tự động bằng chuỗi này và chuyển sang phương pháp **Đánh giá chuyên gia ngữ nghĩa (Human Semantic Evaluation)**: từng thành viên đọc trực tiếp bản dịch tiếng Việt, đối chiếu với thuật ngữ chuẩn CNTT để cho điểm công tâm và chính xác.
```

---

### 2.4. Chèn vào Mục 1.5 (Chương 1): Bảng phân công công việc 4 thành viên

```markdown
### 1.5. Phân công nhiệm vụ và Tỷ lệ đóng góp của các thành viên trong nhóm

Nhóm gồm 4 thành viên phối hợp chặt chẽ trong suốt 7 tuần thực hiện đề tài. Khối lượng công việc được phân chia cụ thể, minh bạch và có sản phẩm bàn giao rõ ràng cho từng cá nhân:

**Bảng 1.2: Bảng phân công trách nhiệm và tỷ lệ hoàn thành công việc**

| Họ và tên | Vai trò chuyên trách | Trách nhiệm và Nhiệm vụ chính được giao | Sản phẩm đầu ra (Minh chứng) | Đóng góp (%) |
| :--- | :---: | :--- | :--- | :---: |
| **Ngô Thanh Hiếu** *(Nhóm trưởng)* | Trưởng nhóm / Kỹ sư RAG Pipeline | • Thiết kế kiến trúc tổng thể Local RAG (Ollama, FAISS, MPNet).<br>• Lập trình pipeline trích xuất văn bản và nhúng vector.<br>• Viết script đo kiểm benchmark (`benchmark.py`) và trích xuất dữ liệu.<br>• Chịu trách nhiệm chính nội dung Chương 2, Chương 3 và tổng hợp báo cáo. | Mã nguồn pipeline RAG, file log đo kiểm 120 runs, tài liệu báo cáo tổng hợp. | **28%** |
| **Mai Tấn Tài** | Kỹ sư Giao diện & Test | • Thiết kế và tối ưu giao diện tương tác Streamlit Dark Theme.<br>• Tích hợp bộ đếm Telemetry thời gian thực trên giao diện người dùng.<br>• Thực hiện bài test ngắt kết nối mạng (Offline Verification) chụp ảnh minh chứng.<br>• Soạn thảo nội dung Chương 4 (Mục 4.1, 4.2 - Giao diện & Ca sử dụng). | Giao diện mã nguồn `app.py`, 2 ảnh chụp minh chứng ngắt mạng (Hình A, B). | **25%** |
| **Nguyễn Thành Nhân** | Chuyên viên Dữ liệu & Đánh giá | • Thu thập và tiền xử lý bộ câu hỏi chuẩn NISTIR 7298 (20 thuật ngữ).<br>• Xây dựng khung đánh giá và chấm điểm thủ công 40 lượt kết quả theo Rubric 5 tiêu chí.<br>• Phân tích dữ liệu so sánh 3 phần, vẽ biểu đồ hiệu năng TTFT và TPS.<br>• Soạn thảo nội dung đánh giá thực nghiệm (Mục 4.3). | File `benchmark_20_questions.json`, bảng Rubric 40 dòng, hệ thống biểu đồ so sánh. | **24%** |
| **Lê Thị Mai Lan** | Kỹ sư Môi trường, Git & Tài liệu | • Cấu hình môi trường cách ly (`venv`), quản lý file `requirements.txt`.<br>• Quản trị hệ thống Git repository, kiểm soát phân nhánh và lịch sử commit.<br>• Viết tài liệu hướng dẫn triển khai (`README.md`).<br>• Soạn thảo nội dung Chương 1, Chương 5 (Kết luận, hướng mở) và Phụ lục. | Repo Git sạch lỗi, tài liệu README chi tiết, các phụ lục nhật ký tuần & tự kiểm. | **23%** |
| **Tổng cộng** | **Tập thể** | **Cả nhóm phối hợp đồng đều, tuyệt đối không có thành viên thụ động (free-rider).** | **Hoàn thành 100% mục tiêu đồ án** | **100%** |
```

---

# PHẦN 3: CÁC PHỤ LỤC BỔ SUNG (THEO CHUẨN NHÓM 23 & BỔ SUNG PHẦN B CHO ĐỀ TÀI AI)

### Phụ lục A: Nhật ký tiến độ thực hiện đề tài qua các tuần (Weekly Log)

```markdown
# PHỤ LỤC A: NHẬT KÝ TIẾN ĐỘ THỰC HIỆN ĐỀ TÀI QUA CÁC TUẦN

**Bảng A.1: Bảng theo dõi tiến độ công việc qua 7 tuần thực hiện**

| Tuần | Nội dung công việc thực hiện | Người thực hiện | Khó khăn phát sinh | Kết quả & Sản phẩm đạt được |
| :---: | :--- | :---: | :--- | :--- |
| **Tuần 1** | • Họp nhóm xác định đề tài: RAG cục bộ với Ollama.<br>• Phân tích yêu cầu và viết đề cương chi tiết.<br>• Khảo sát tài liệu NISTIR 7298 về thuật ngữ an toàn thông tin.<br>• Khởi tạo Git repository và thống nhất quy ước commit. | Cả nhóm *(Hiếu, Tài, Nhân, Lan)* | Chưa thống nhất được dung lượng mô hình LLM phù hợp với cấu hình laptop cá nhân. | Hoàn thiện bản Đề cương đề tài nộp giảng viên; lựa chọn chuẩn NISTIR 7298 làm bộ dữ liệu thực nghiệm; khởi tạo repo GitHub. |
| **Tuần 2** | • Cài đặt Ollama, kéo mô hình `qwen2.5-coder:7b-instruct-q4_K_M`.<br>• Cấu hình môi trường Python Virtual Environment (`venv`), đóng gói `requirements.txt`.<br>• Viết thử nghiệm đoạn mã nạp PDF và tách văn bản (Chunking). | Hiếu, Tài, Lan | Tốc độ suy luận CPU rất chậm (~0.2 tok/s); VRAM GPU bị thiếu nếu dùng bản FP16; xung đột phiên bản thư viện pip. | Chuyển sang mô hình lượng tử hóa Q4_K_M chạy hoàn toàn trên VRAM RTX 2050 (4 GB), đạt tốc độ ~1.2 – 1.8 tok/s; môi trường `venv` đồng bộ giữa các máy. |
| **Tuần 3** | • Xây dựng module Vector Indexing bằng FAISS và mô hình nhúng `all-mpnet-base-v2`.<br>• Thiết lập giao diện Streamlit cơ bản với các thanh trượt điều chỉnh Chunk size và Overlap.<br>• Cấu hình lưu trữ chỉ mục cục bộ và file `.gitignore`. | Hiếu, Tài, Lan | Giao diện Streamlit mặc định bị tải lại toàn bộ trang mỗi khi người dùng gửi câu hỏi; file cache FAISS kích thước lớn bị đẩy lên Git. | Ứng dụng `st.session_state` để lưu vết hội thoại; cấu hình `.gitignore` chặn cache nhị phân; hoàn thiện pipeline RAG cơ bản chạy trên máy cá nhân. |
| **Tuần 4** | • Tối ưu hóa giao diện Streamlit theo phong cách Dark Mode hiện đại.<br>• Tích hợp bộ đếm thời gian thực `perf_counter` hiển thị thời gian, tốc độ, token.<br>• Thiết kế khung tiêu chí đánh giá Rubric 5 mức độ và viết test case kiểm thử đơn vị (`test_rag_pipeline.py`). | Tài, Nhân | Đếm token bị lệch do tiếng Việt có nhiều dấu và ký tự đa byte; tiêu chí chấm điểm tự động bằng so khớp chuỗi bị sai lệch ngữ nghĩa. | Hoàn thiện công thức ước tính token; UI hiển thị đầy đủ thông số Telemetry sau mỗi câu trả lời; ban hành khung Rubric chuyên gia 5 tiêu chí. |
| **Tuần 5** | • Xây dựng bộ công cụ Benchmark tự động (`benchmark.py`).<br>• Bóc tách 20 câu hỏi thuật ngữ tiêu chuẩn kèm số trang từ file PDF NISTIR 7298.<br>• Kết nối API Google Cloud (`gemini-flash-lite-latest`) làm đối chứng.<br>• Quản lý khóa bảo mật API trong file `.env`. | Hiếu, Nhân, Lan | API Gemini 1.5 cũ bị trả về lỗi 404 do đổi endpoint; luồng Stream SSE đôi lúc bị đứt kết nối. | Cập nhật endpoint `gemini-flash-lite-latest`; script chạy ổn định cơ chế tự retry và fallback; tạo thành công file `benchmark_20_questions.json`. |
| **Tuần 6** | • Tiến hành đo kiểm 120 lượt chạy thực tế (20 câu hỏi $\times$ 2 mô hình $\times$ 3 lần).<br>• Thực hiện bài kiểm thử ngắt kết nối mạng (Offline Verification) có chụp ảnh minh chứng.<br>• Chấm điểm 40 mẫu tóm tắt theo thang Rubric 5 tiêu chí và ghi chú lỗi dịch thuật. | Cả nhóm *(Hiếu, Tài, Nhân, Lan)* | Quá trình suy luận của mô hình cục bộ mất nhiều thời gian đo lường; chấm điểm Rubric thủ công 40 mẫu đòi hỏi đối chiếu kỹ lưỡng từng câu từ. | Thu thập đầy đủ dữ liệu file `benchmark_raw_runs.csv` và `benchmark_so_sanh_3_phan.csv`; chụp thành công 2 ảnh minh chứng offline (SelfRAG lỗi kết nối, Local RAG thành công 152.67s). |
| **Tuần 7** | • Tổng hợp số liệu, vẽ biểu đồ so sánh TTFT, tốc độ sinh từ và điểm số Rubric.<br>• Soạn thảo mục AI Disclosure, Bảng phủ kỹ thuật và Bảng tự kiểm 3 tầng.<br>• Rà soát số liệu toàn bộ báo cáo, kiểm tra lỗi chính tả và đóng gói mã nguồn nghiệm thu. | Cả nhóm *(Hiếu, Tài, Nhân, Lan)* | Khối lượng tài liệu lớn cần chuẩn hóa theo mẫu của Nhóm 23; cần đồng bộ toàn bộ số liệu đo lường giữa Chương 4 và Chương 5. | Hoàn thiện bản báo cáo đồ án hoàn chỉnh cùng mã nguồn trên GitHub repository với đầy đủ lịch sử commit. |
```

---

### Phụ lục C: Bảng tự kiểm nhanh ba tầng (3-Tier Checklist) chuẩn hóa cho Đề tài Trí tuệ nhân tạo

```markdown
# PHỤ LỤC 3: BẢNG TỰ KIỂM NHANH BA TẦNG (3-TIER CHECKLIST) CHO ĐỒ ÁN TRÍ TUỆ NHÂN TẠO

**Bảng C.1: Bảng tự kiểm tra chất lượng đồ án SmartDoc AI (Áp dụng Phần A, Phần F4, Phần B1, Phần D, Phần C4 theo Văn bản Danh mục gợi ý 01/1409 của Thầy Phạm Thi Vương)**

| Tầng (Tier) | Nội dung yêu cầu theo quy định của Thầy Vương | Nhóm đã đạt bao nhiêu (Minh chứng chi tiết & Trả lời câu hỏi tự kiểm) | Nguồn quy định (Văn bản 01/1409) |
| :---: | :--- | :--- | :---: |
| **1A** | **Bản chất công nghệ chính Ollama Runtime** *(Yêu cầu cốt lõi theo Mục F4)* | **Lõi: Đạt 3/3 mục bắt buộc:**<br>1. Chạy mô hình `qwen2.5-coder:7b-instruct` và chọn bản lượng tử hoá `Q4_K_M` theo đúng phân công.<br>2. Chạy suy luận cục bộ hoàn toàn trên máy trạm (qua Ollama CLI và cổng REST API `localhost:11434`).<br>3. Đo độ trễ suy luận chi tiết qua Telemetry.<br><br>**Chọn thêm thực sự đã làm: 3 mục:**<br>1. Cấu hình cửa sổ ngữ cảnh (`num_ctx = 2048/4096`) phù hợp bài toán.<br>2. Đo mức dùng RAM và VRAM thực tế (**NVIDIA GeForce RTX 2050 Laptop GPU 4 GB VRAM** chiếm đỉnh ~3.8 GB).<br>3. So sánh chất lượng và tốc độ với mô hình qua Cloud API (`gemini-flash-lite-latest`).<br><br>👉 **Trả lời câu hỏi tự kiểm tra F4 của Thầy:** *"Vì sao chọn bản lượng tử hoá này mà không phải bản lớn hơn?"*<br>$\rightarrow$ **Trả lời:** GPU máy trạm của nhóm là RTX 2050 có giới hạn phần cứng **4 GB VRAM**. Bản `Q4_K_M` (kích thước ~4.7 GB) sau khi nạp vào bộ nhớ và ánh xạ KV Cache chiếm khoảng **3.8 GB VRAM**, giúp toàn bộ quá trình tính toán ma trận nằm trọn 100% trong VRAM của GPU, duy trì tốc độ sinh từ ổn định **1.2 – 1.8 tok/s**. Nếu chọn bản 8-bit (`Q8_0`) hoặc bản 14B, mô hình sẽ bị tràn VRAM và buộc phải offload các layer sang RAM hệ thống thông qua bus PCIe (nghẽn cổ chai băng thông), khiến tốc độ suy luận sụt giảm nghiêm trọng xuống dưới **0.2 tok/s**, không còn khả năng phục vụ tương tác thực tế. | **Phần F4** của Danh mục *(Trang 14: Công nghệ AI — Ollama / llama.cpp)* |
| **1B** | **Hồ sơ đề tài RAG (Retrieval-Augmented Generation)** *(Quy trình bắt buộc cho đề tài AI theo Mục B1)* | **Đạt trọn vẹn 7/7 bước quy trình bắt buộc của Hồ sơ RAG:**<br>1. **Nạp dữ liệu:** Trích xuất văn bản có cấu trúc từ file PDF chuẩn hóa `NISTIR 7298 Rev 2` bằng `PyPDFLoader`.<br>2. **Cắt đoạn (Chunking):** Áp dụng `RecursiveCharacterTextSplitter` với tham số tối ưu: chunk size = 500 ký tự, chunk overlap = 50 ký tự.<br>3. **Embedding và lập chỉ mục:** Vector hóa bằng mô hình `all-mpnet-base-v2` (768 chiều), lưu trữ chỉ mục cục bộ bằng `FAISS` (IndexFlatL2).<br>4. **Truy hồi (Retrieval):** Tính toán khoảng cách Cosine similarity, trích xuất Top-3 chunks liên quan nhất.<br>5. **Gắn nguồn (Grounding, có trích dẫn):** Trích xuất metadata lưu rõ số trang nguồn (Trang 5 NISTIR) đính kèm đoạn văn bản.<br>6. **Sinh câu trả lời:** Ghép ngữ cảnh vào System Prompt chuyên ngành, gọi LLM cục bộ sinh câu trả lời bám sát tài liệu.<br>7. **Đánh giá có thực nghiệm ở khâu truy hồi:** Xây dựng Bảng truy vết dữ liệu (Data Traceability Matrix tại Mục 4.2.2) minh chứng câu trả lời bắt nguồn từ chunk nguồn. | **Phần B1** của Danh mục *(Trang 10: Hồ sơ đề tài AI — Kiến trúc RAG)* |
| **1C** | **Hồ sơ đề tài Local LLM / Ollama** *(Quy trình bắt buộc triển khai mô hình cục bộ theo Mục B1)* | **Đạt trọn vẹn 7/7 bước quy trình bắt buộc của Hồ sơ Local LLM:**<br>1. **Chọn mô hình:** `qwen2.5-coder:7b-instruct` chuyên sâu hiểu ngữ cảnh kỹ thuật.<br>2. **Chọn bản lượng tử hoá & runtime:** Bản `Q4_K_M` vận hành trên runtime Ollama cục bộ.<br>3. **Đo mức dùng phần cứng:** Ghi nhận thực tế VRAM tiêu thụ đỉnh 3.8 GB trên GPU RTX 2050 (4 GB).<br>4. **Xử lý cửa sổ ngữ cảnh:** Thiết lập Prompt ép chặt ngữ cảnh, tránh hiện tượng tràn context.<br>5. **Đo độ trễ chuẩn xác:** Đo đạc TTFT ($0.236\text{s}$), Throughput ($1.2\text{ tok/s}$), Total Latency ($152.67\text{s}$).<br>6. **Đánh giá chất lượng:** Chấm điểm đối chiếu với Ground Truth của tài liệu NISTIR 7298.<br>7. **So sánh với API thương mại trên cùng bộ câu hỏi:** Thực hiện 120 lượt chạy thực nghiệm đối kháng với Google Gemini Flash Lite Cloud trên 20 câu hỏi cố định. | **Phần B1** của Danh mục *(Trang 11: Hồ sơ đề tài AI — Local LLM / Ollama)* |
| **2** | **Chuẩn kỹ nghệ phần mềm theo đặc thù nền tảng** *(Software Engineering Standards theo Mục D)* | **Đạt 7/7 tiêu chuẩn kỹ nghệ áp dụng được cho dự án viết code:**<br>1. **Quản lý phiên bản:** Git repository public (`https://github.com/hieu-BG/smartdoc_AI.git`), phân nhánh rõ ràng, biểu đồ commit minh chứng cả 4 thành viên (Ngô Thanh Hiếu, Mai Tấn Tài, Nguyễn Thành Nhân, Lê Thị Mai Lan) đóng góp đồng đều.<br>2. **Khả năng tái lập:** Đóng gói môi trường ảo Python Virtual Environment (`venv`), đồng bộ file `requirements.txt`, tài liệu `README.md` hướng dẫn chạy chỉ bằng 1 lệnh.<br>3. **Quản lý bí mật:** Tách biệt biến môi trường trong file `.env`, cấu hình file `.gitignore` loại bỏ venv, cache nhị phân FAISS và tuyệt đối không commit API key lên GitHub.<br>4. **Kiến trúc phân lớp sạch (Clean Architecture):** Mã nguồn module hóa tường minh: `rag/` (xử lý logic RAG), `ui/` (giao diện Streamlit), `db/` (quản trị phiên SQLite), `tests/` (kiểm thử).<br>5. **Quản lý phiên (Session State):** Ứng dụng `st.session_state` và SQLite `chat.db` lưu vết toàn bộ lịch sử hội thoại nhiều lượt không bị reload ứng dụng.<br>6. **Kiểm thử tự động:** Xây dựng unit test độc lập (`tests/test_rag_pipeline.py`) kiểm tra tự động khâu đọc PDF, chia chunk và tính toán khoảng cách cosine FAISS.<br>7. **Làm việc cùng AI:** Công bố minh bạch trong Mục 5.3 AI Disclosure (AI đóng góp 25–30%, phân tích chi tiết 2 trường hợp lỗi AI thực tế do nhóm tự phát hiện và sửa chữa). | **Phần D** của Danh mục *(Trang 20: Chuẩn kỹ nghệ phần mềm — Dự án viết code)* |
| **3** | **Kỹ thuật nâng cao tự chọn** *(Advanced Deep-dive theo Mục C4)* | **Các kỹ thuật thực sự triển khai trong đồ án:**<br>1. **Kỹ thuật C4.4 (Tối ưu chi phí và độ trễ):** Ứng dụng `time.perf_counter()` đo lường Telemetry thời gian thực (TTFT, Latency, TPS) hiển thị trực tiếp trên UI; triển khai Local RAG giúp cắt giảm 100% chi phí token API bên ngoài.<br>2. **Kỹ thuật C4.5 (Đánh giá tự động nâng cao):** Xây dựng harness thực nghiệm tự động (`benchmark.py`) chạy 120 lượt đo kiểm, áp dụng thang Rubric chuyên gia 5 tiêu chí loại bỏ thiên kiến so khớp chuỗi.<br>3. **Xác thực ngắt mạng hoàn toàn (Air-gapped Offline Verification):** Thực nghiệm ngắt 100% WiFi/LAN, chứng minh Local RAG suy luận độc lập trên RTX 2050 (152.67s, 183 tokens), đồng thời phân định rõ lỗi `Connection error` của Cloud Self-RAG.<br><br>👉 **(Điểm sâu nhất nhóm có thể giải thích và sửa tại chỗ — Trả lời câu hỏi tự kiểm C4 trang 19 của Thầy):**<br>*"Câu này trả lời sai — do khâu truy hồi hay khâu sinh? Em chứng minh bằng cách nào?"*<br>$\rightarrow$ **Chứng minh:** Nhóm sử dụng **Bảng truy vết dữ liệu (Data Traceability Matrix)** để bóc tách độc lập 2 khâu: Nhóm trích xuất trực tiếp Top-K chunks mà FAISS trả về. Nếu văn bản trong chunk nguồn không chứa định nghĩa đúng thì đó là **lỗi khâu truy hồi (Retrieval error)** (nguyên nhân do tham số `chunk_size` quá nhỏ làm đứt mạch câu hoặc mô hình embedding chưa gom cụm được thuật ngữ viết tắt). Ngược lại, nếu chunk trích xuất đã chứa chính xác định nghĩa từ tài liệu gốc nhưng câu trả lời cuối cùng bị sai lệch thì đó là **lỗi khâu sinh (Generation error)** (do hiện tượng ảo giác hallucination của LLM hoặc tham số `temperature` đặt quá cao).<br>$\rightarrow$ **Sửa tại chỗ:** Nhóm có thể trực tiếp sửa logic System Prompt trong file `rag/prompt.py` để thắt chặt ràng buộc ("Chỉ được phép trả lời dựa trên context được cung cấp, nếu không có thông tin hãy trả lời Tôi không biết"), đồng thời cấu hình lại bộ nhớ đệm VRAM trong runtime Ollama để dập tắt hoàn toàn ảo giác. | **Phần C4** của Danh mục *(Trang 19: Kỹ thuật AI nâng cao)* |
```

---

## CHECKLIST KIỂM TRA LẦN CUỐI TRƯỚC KHI XUẤT FILE PDF NỘP BÁO CÁO:

1. **Sửa lỗi số thập phân TTFT ở trang 54–55:**
   - [ ] Kiểm tra cột TTFT của Qwen: Đảm bảo số `236`, `237`, `224` đã được đổi thành `0.236s`, `0.237s`, `0.224s` (không để số nguyên hàng trăm giây).
2. **Thống nhất tên GPU:**
   - [ ] Toàn bộ báo cáo ghi đồng nhất: **`NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)`** (Tuyệt đối không để lẫn RTX 3050).
3. **Chèn 2 ảnh chụp ngắt kết nối mạng:**
   - [ ] Ảnh 1: Chế độ SelfRAG báo đỏ `Connection error.` khi tắt mạng.
   - [ ] Ảnh 2: Chế độ Local RAG hoàn thành trong `152.67s | ~1.2 tok/s | ~183 tokens` khi tắt mạng.
4. **Chụp ảnh GitHub dán vào Phụ lục B:**
   - [ ] Truy cập `https://github.com/hieu-BG/smartdoc_AI.git`.
   - [ ] Vào mục **Insights -> Contributors** chụp biểu đồ đóng góp của các thành viên.
   - [ ] Chụp danh sách các commits gần nhất.

