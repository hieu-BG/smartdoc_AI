# 📄 SmartDoc AI

## 🚀 Giới thiệu dự án

Dự án xây dựng hệ thống hỏi đáp thông minh từ tài liệu PDF sử dụng kỹ thuật RAG, giúp người dùng tìm kiếm thông tin nhanh và chính xác. Hệ thống hỗ trợ tải tài liệu, chuyển đổi nội dung thành vector và lưu trữ bằng FAISS để truy xuất hiệu quả. Mô hình Qwen2.5:7b được tích hợp để sinh câu trả lời dựa trên ngữ cảnh. Ngoài ra, hệ thống cho phép điều chỉnh tham số và cấu hình chunk nhằm tối ưu hiệu suất và độ chính xác.

---

## ✨ Tính năng

**Tải lên tệp (File Upload):** Hỗ trợ tải PDF với giao diện kéo thả, kiểm tra kích thước và hiển thị thông báo kết quả.

**Hỏi đáp tài liệu (Question Answering):** Cho phép đặt câu hỏi tự nhiên, xử lý thời gian thực và hiển thị câu trả lời rõ ràng kèm trạng thái loading.

**Xử lý lỗi (Error Handling):** Phát hiện và thông báo lỗi định dạng, xử lý và kết nối mô hình một cách thân thiện với người dùng.

**Điều chỉnh tham số mô hình (Model Configuration):** Cho phép tùy chỉnh temperature, top-p, repeat penalty qua slider và áp dụng ngay lập tức.

**Cấu hình chia nhỏ tài liệu (Chunk Configuration):** Tùy chỉnh chunk size và overlap để tối ưu hiệu suất tìm kiếm và độ chính xác.

**Lưu trữ và quản lý lịch sử trò chuyện:** Lưu, hiển thị, chọn lại và xóa lịch sử, đồng thời bỏ qua các phiên rỗng.

---

## 🖼️ Ảnh minh họa

![App Screenshot](data/img.png)

---

## ⚙️ Hướng dẫn chạy dự án

### 🔧 Yêu cầu hệ thống

* Python 3.8+
* Ollama runtime
* pip package manager

---

### 📥 Cài đặt

```bash
git clone <repo-url>
cd SmartDoc_Ai
```

#### 👉 Kích hoạt môi trường ảo

**Linux / Mac:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

---

### 📦 Cài thư viện

```bash
pip install -r requirements.txt
```

---

### 🤖 Tải model

```bash
ollama pull qwen2.5-coder:7b
```

---

### ▶️ Chạy ứng dụng

```bash
streamlit run app.py
```

---

## 📌 Ghi chú

* Đảm bảo Ollama đang chạy trước khi khởi động ứng dụng
*  chế độ SelfRAG cần biến môi trường `GROQ_API_KEY`; chế độ RAG dùng model Ollama cục bộ
* Khi thay đổi chunk hoặc tham số, nên rebuild lại vector database để đảm bảo kết quả chính xác

---
