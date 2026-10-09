

| ỦY BAN NHÂN DÂN THÀNH PHỐ HỒ CHÍ  MINH TRƯỜNG ĐẠI HỌC SÀI GÒN  KHOA CÔNG NGHỆ THÔNG TIN  NGÔ THANH HIẾU  MAI TẤN TÀI  NGUYỄN THÀNH NHÂN LÊ THỊ MAI LAN XÂY DỰNG HỆ THỐNG HỎI ĐÁP TÀI LIỆU PDF SỬ DỤNG KIẾN TRÚC RAG VỚI THƯ VIỆN LANGCHAIN VÀ CƠ SỞ DỮ LIỆU VECTOR FAISS    GIẢNG VIÊN HƯỚNG DẪN:  PHẠM THI VƯƠNG Thành phố Hồ Chí Minh, năm 2026 |
| ----- |

                                                   **LỜI CẢM ƠN**  
Trước hết, chúng em xin bày tỏ lòng biết ơn chân thành nhất đến Ban Giám hiệu  Trường Đại học Sài Gòn cùng toàn thể quý thầy cô thuộc Khoa Công nghệ Thông tin  đã tận tình truyền đạt những kiến thức chuyên môn quý báu và tạo mọi điều kiện  thuận lợi cho em trong suốt quá trình học tập tại trường.   
Đặc biệt, chúng em xin gửi lời cảm ơn sâu sắc nhất đến giảng viên hướng dẫn là người đã dành nhiều thời gian quý báu để trực tiếp chỉ bảo, đưa ra những định hướng quan trọng và giúp đỡ em tận tình để có thể hoàn thành tốt nội  dung báo cáo của Đồ án chuyên ngành lần này.   
Bên cạnh đó, chúng em cũng xin gửi lời cảm ơn chân thành đến gia đình và bạn bè đã  luôn đồng hành, khích lệ và là chỗ dựa vững chắc hỗ trợ chúng em về mọi mặt trong suốt  thời gian nghiên cứu thực hiện đề tài. Dù đã nỗ lực hoàn thiện báo cáo một cách chỉn  chu nhất, song do hạn chế về mặt kiến thức và kinh nghiệm thực tế, bài làm chắc chắn  không tránh khỏi những thiếu sót ngoài ý muốn, vì vậy chúng em rất mong nhận được những  ý kiến đóng góp và nhận xét quý báu từ quý thầy cô để đồ án được hoàn thiện và đạt  kết quả tốt hơn trong giai đoạn tiếp theo.

 

**1\. Lý do chọn đề tài**

* Sự bùng nổ của mô hình ngôn ngữ lớn (LLM) đã thay đổi cách con người tương tác với thông tin, nhưng LLM thuần túy tồn tại hai giới hạn cố hữu: hallucination (sinh thông tin sai lệch, không có căn cứ) và giới hạn tri thức (chỉ biết những gì đã học đến thời điểm huấn luyện, không biết tài liệu riêng của người dùng). Kỹ thuật RAG ra đời chính là để giải quyết hai vấn đề này bằng cách "cắm" một nguồn tri thức bên ngoài (tài liệu thực tế) vào quy trình sinh câu trả lời.  
* Đề tài nhóm chọn để ứng dụng công nghệ là xây dựng SmartDoc AI — hệ thống hỏi đáp thông minh trên tài liệu PDF — một bài toán rất phù hợp để bộc lộ đầy đủ các thành phần cốt lõi của một pipeline RAG điển hình: nạp tài liệu → chia nhỏ (chunking) → mã hóa vector (embedding) → lưu trữ và truy xuất (vector store) → tăng cường ngữ cảnh cho prompt → sinh câu trả lời bằng LLM.  
* Về định hướng nghề nghiệp: RAG hiện là một trong những kỹ thuật nền tảng được nhắc đến nhiều nhất trong các tin tuyển dụng liên quan đến AI Engineer, LLM Engineer và Applied AI tại Việt Nam cũng như trên thế giới, khi phần lớn doanh nghiệp muốn triển khai chatbot/tra cứu nội bộ trên dữ liệu riêng mà không cần huấn luyện lại mô hình. Việc làm chủ RAG mang lại giá trị thực tế cho hồ sơ xin việc của các thành viên trong nhóm.

**2\. Mục tiêu báo cáo**  
Mục tiêu kiến thức (các mục BẮT BUỘC LÕI — điều kiện cần)

* Hiểu và trình bày được nguyên lý hoạt động của kiến trúc RAG: vì sao phải tách riêng bước "truy xuất" (retrieval) và bước "sinh" (generation), khác gì so với việc hỏi thẳng LLM hoặc fine-tuning LLM trên dữ liệu riêng.  
* Phân biệt rõ vai trò từng thành phần trong pipeline: bộ nạp tài liệu (document loader), bộ chia nhỏ văn bản (text splitter/chunker), mô hình embedding, cơ sở dữ liệu vector (vector store), và mô hình sinh câu trả lời (LLM).  
* Sử dụng thư viện FAISS để lưu trữ và tìm kiếm vector theo độ tương đồng ngữ nghĩa (cosine similarity/Euclidean distance), hiểu cơ chế tìm kiếm lân cận gần đúng (Approximate Nearest Neighbor).  
* Triển khai được mô hình LLM cục bộ (Qwen2.5 qua Ollama) để sinh câu trả lời dựa trên ngữ cảnh truy xuất được, quản lý phiên bản mã nguồn bằng Git, đóng gói bằng Docker, và có kiểm thử cơ bản cho từng module (unit test cho chunking/embedding).

Mục tiêu kỹ năng (các mục CHỌN THÊM / kỹ thuật nâng cao dự kiến khảo sát)

* Cấu hình động các tham số ảnh hưởng đến chất lượng câu trả lời: chunk size, chunk overlap, temperature, top-p, repeat penalty, và quan sát ảnh hưởng thực tế của từng tham số.  
* Cơ chế xử lý lỗi thân thiện: phát hiện file không hợp lệ, lỗi kết nối mô hình, lỗi định dạng, hiển thị trạng thái loading khi đang xử lý.  
* Lưu trữ và quản lý lịch sử hội thoại (session), cho phép chọn lại, xóa và tự động bỏ qua các phiên rỗng.  
* Cấu hình theo môi trường (development/production) để tách biệt cấu hình khi phát triển và triển khai.  
* Đây là kế hoạch học tập mang tính định hướng, không phải cam kết đạt một mức độ khó cố định; nhóm sẽ điều chỉnh khi hiểu bài toán sâu hơn.

**3\. Phạm vi nghiên cứu**  
Phạm vi nghiên cứu giới hạn trong thời gian thực hiện của môn học. Phần bắt buộc lõi tập trung vào việc xây dựng trọn vẹn một pipeline RAG hoàn chỉnh — từ tải tài liệu PDF, chia nhỏ, mã hóa vector, lưu trữ FAISS, truy xuất, đến sinh câu trả lời bằng mô hình Qwen2.5:7b chạy cục bộ qua Ollama — được minh chứng bằng mã nguồn chạy được trong sản phẩm cuối là ứng dụng SmartDoc AI với giao diện Streamlit.  
Phần chọn thêm dự kiến khảo sát gồm: cấu hình tham số mô hình và chunking qua giao diện, xử lý lỗi thân thiện, lưu trữ/quản lý lịch sử hội thoại, và cấu hình theo môi trường. Phạm vi ban đầu này có thể được điều chỉnh trong quá trình thực hiện; mọi thay đổi sẽ được giải thích và khai báo lại ở phần cuối báo cáo.

**4\. Kế hoạch và tiến độ thực hiện**

| Tuần | Nội dung chính | Mốc phải nộp |
| ----- | ----- | ----- |
| 1 | Lập nhóm, thống nhất chọn RAG làm công nghệ chính; cài môi trường (Python, Ollama, IDE); kéo model Qwen2.5, chạy thử một lệnh sinh văn bản đơn giản; dựng khung dự án và repo Git. | Đăng ký nhóm và công nghệ. Repo có commit thực chất. |
| 2 | Thiết kế pipeline tổng thể (sơ đồ luồng dữ liệu); học và làm phần BẮT BUỘC LÕI: document loader, text splitter, embedding; viết Chương 3 báo cáo. | Sơ đồ pipeline. Bản nháp Chương 3 kèm code minh họa chạy được. |
| 3 | Hoàn thiện phần bắt buộc lõi: tích hợp FAISS lưu trữ vector, cài đặt truy xuất top-k đoạn văn bản liên quan, kết nối với Qwen2.5 để sinh câu trả lời. | Demo hỏi–đáp trên một tài liệu PDF mẫu chạy được. |
| 4 | Làm các mục CHỌN THÊM: giao diện cấu hình tham số (temperature, top-p, chunk size/overlap), xử lý lỗi định dạng và kết nối. | Bản nháp lần 1 chức năng cấu hình tham số. |
| 5 | Tích hợp lưu trữ/quản lý lịch sử hội thoại; refactor pipeline nếu cần; đo lường chất lượng truy xuất và thời gian phản hồi theo các cấu hình khác nhau. | Ứng dụng chạy được luồng chính từ đầu đến cuối. Bảng số liệu đo lường. |
| 6 | Viết hands-on lab, chấm chéo lab giữa các nhóm. | Nộp lab. |
| 7 | Hoàn thiện báo cáo, slide, video demo. Niêm phong mã nguồn. | Nộp hồ sơ đầy đủ và commit hash trước buổi bảo vệ. |

# **PHẦN 1: TỔNG QUAN VÀ PHÂN TÍCH CÔNG NGHỆ**

Mục tiêu: cung cấp bức tranh toàn cảnh và tư duy phản biện về vị thế của RAG trong hệ sinh thái ứng dụng AI, làm nền tảng để nhóm lựa chọn kiến trúc phù hợp cho hệ thống SmartDoc AI ở Phần 2\.

## **Chương 1: Giới thiệu về Retrieval-Augmented Generation (RAG)**

### **1.1. Bối cảnh và lịch sử**

Khái niệm RAG được Lewis và cộng sự đề xuất chính thức trong nghiên cứu công bố năm 2020, kết hợp một mô-đun truy xuất dựa trên biểu diễn vector với một mô hình sinh dạng sequence-to-sequence. Trước đó, cách phổ biến để "dạy" mô hình ngôn ngữ kiến thức mới là fine-tuning — huấn luyện lại (một phần) trọng số mô hình trên dữ liệu mới, vốn tốn kém, cần dữ liệu huấn luyện lớn, và có nguy cơ làm mô hình "quên" kiến thức cũ (catastrophic forgetting).

Nỗi đau mà RAG ra đời để giải quyết là: làm sao để một LLM có sẵn (không cần huấn luyện lại) có thể trả lời chính xác các câu hỏi dựa trên tài liệu riêng, cập nhật liên tục, mà chi phí triển khai thấp. RAG giải quyết vấn đề này bằng ba trụ cột chính:

* Chunking & Embedding: tài liệu được chia thành các đoạn nhỏ (chunk) và mã hóa thành vector ngữ nghĩa, cho phép so sánh độ tương đồng về ý nghĩa thay vì chỉ khớp từ khóa.  
* Vector Store & Retrieval: các vector được lưu trong cơ sở dữ liệu chuyên dụng (FAISS, Chroma, Pinecone...), cho phép truy xuất nhanh top-k đoạn văn bản liên quan nhất đến câu hỏi.  
* Augmented Generation: các đoạn văn bản truy xuất được chèn vào prompt làm ngữ cảnh, giúp LLM sinh câu trả lời bám sát tài liệu nguồn thay vì "bịa" từ kiến thức nội tại.

### **1.2. Kiến trúc và nguyên lý hoạt động**

Về bản chất, RAG là một kỹ thuật/kiến trúc kết hợp nhiều thành phần lại với nhau, không phải một mô hình đơn lẻ hay một thư viện cụ thể — gọi RAG là "mô hình AI" là gọi sai loại theo phân loại kỹ thuật/mô hình/công cụ.

* Document Ingestion: tài liệu (PDF, DOCX, TXT...) được trích xuất văn bản thô, làm sạch (loại bỏ header/footer, ký tự lỗi).  
* Text Splitter (Chunking): văn bản dài được chia thành các đoạn nhỏ hơn theo kích thước (chunk size) và có thể chồng lấp một phần (overlap) để tránh cắt đứt ngữ cảnh giữa các đoạn.  
* Embedding Model: mỗi chunk được mã hóa thành một vector số thực nhiều chiều, sao cho các đoạn có ý nghĩa gần nhau thì vector cũng gần nhau trong không gian embedding.  
* Vector Store (FAISS): lưu trữ toàn bộ vector và cho phép tìm kiếm lân cận gần đúng — khi có câu hỏi, câu hỏi cũng được mã hóa thành vector rồi so sánh với các vector đã lưu để tìm ra top-k đoạn liên quan nhất.  
* Prompt Augmentation: các đoạn văn bản truy xuất được ghép cùng câu hỏi gốc thành một prompt hoàn chỉnh, thường có cấu trúc dạng "Dựa trên ngữ cảnh sau: \[context\]. Hãy trả lời câu hỏi: \[question\]".  
* LLM Generation: mô hình ngôn ngữ (Qwen2.5 qua Ollama) nhận prompt đã tăng cường và sinh câu trả lời cuối cùng.

### **1.3. So sánh và đánh giá**

Nhóm so sánh RAG với hai hướng tiếp cận khác thường được cân nhắc khi cần LLM "biết" về dữ liệu riêng: Fine-tuning và Long-context prompting (nhồi toàn bộ tài liệu vào cửa sổ ngữ cảnh).

| Tiêu chí | RAG | Fine-tuning | Long-context |
| ----- | ----- | ----- | ----- |
| Chi phí cập nhật dữ liệu mới | Thấp — chỉ cần thêm tài liệu vào vector store | Cao — cần huấn luyện lại | Thấp nhưng tốn token mỗi lần gọi |
| Độ minh bạch nguồn trả lời | Cao — có thể trích dẫn đoạn văn bản gốc | Thấp — kiến thức "ẩn" trong trọng số | Trung bình |
| Chi phí vận hành mỗi truy vấn | Thấp–trung bình (embedding \+ truy xuất) | Thấp sau khi đã huấn luyện | Cao (nhiều token đầu vào) |
| Phù hợp dữ liệu thay đổi liên tục | Rất phù hợp | Không phù hợp | Phù hợp nhưng giới hạn dung lượng |

Từ bảng so sánh, nhóm rút ra hai trường hợp KHÔNG NÊN dùng RAG:

* Câu hỏi cần suy luận sâu, tổng hợp xuyên suốt toàn bộ tài liệu dài mà không thể tách rời từng đoạn nhỏ (ví dụ: tóm tắt toàn bộ một cuốn sách 500 trang) — khi đó truy xuất top-k đoạn rời rạc có thể làm mất mạch logic tổng thể, long-context hoặc phương pháp tóm tắt phân cấp (hierarchical summarization) phù hợp hơn.  
* Ứng dụng cần độ trễ cực thấp và không có tài liệu ngoài để tham chiếu (ví dụ: chatbot trò chuyện phiếm) — khi đó việc thêm bước truy xuất chỉ làm tăng độ trễ không cần thiết.

Ngược lại, với bài toán hỏi đáp trên tài liệu PDF cụ thể do người dùng cung cấp như SmartDoc AI — nơi cần câu trả lời bám sát nguồn, có thể cập nhật tài liệu linh hoạt và không cần huấn luyện lại mô hình — RAG là lựa chọn phù hợp và là lý do nhóm quyết định chọn công nghệ này.

## **Chương 2: Vị thế công nghệ và phạm vi ứng dụng**

### **2.1. Loại sản phẩm/hệ thống thường dùng RAG**

RAG thường được dùng để xây dựng các hệ thống hỏi đáp và trợ lý ảo trên dữ liệu riêng (private data), nơi cần câu trả lời chính xác, có căn cứ và cập nhật liên tục. Ba nhóm ứng dụng tiêu biểu:

* Trợ lý tra cứu tài liệu nội bộ doanh nghiệp (như SmartDoc AI): tra cứu chính sách, hợp đồng, tài liệu kỹ thuật — nhân viên là người dùng chính.  
* Chatbot chăm sóc khách hàng dựa trên FAQ/tài liệu sản phẩm: trả lời câu hỏi khách hàng dựa trên tài liệu hướng dẫn, chính sách bảo hành — bộ phận CSKH và khách hàng cuối là người dùng chính.  
* Công cụ nghiên cứu/tổng hợp tài liệu học thuật: hỗ trợ sinh viên, nhà nghiên cứu tra cứu nhanh nội dung trong các bài báo khoa học, giáo trình.

### **2.2. Tình hình sử dụng và nhu cầu hiện tại**

RAG hiện là một trong những kỹ thuật được áp dụng phổ biến nhất khi doanh nghiệp muốn triển khai "chatbot nội bộ" hoặc "trợ lý AI" trên dữ liệu riêng mà không phải huấn luyện lại mô hình — cách tiếp cận này được ưa chuộng vì chi phí thấp hơn đáng kể so với fine-tuning một LLM lớn. Các vị trí tuyển dụng liên quan đến AI Engineer, LLM Engineer thường yêu cầu kết hợp RAG với các công nghệ mở rộng: vector database (FAISS, Milvus, Pinecone), framework điều phối (LangChain, LlamaIndex), và các nền tảng triển khai LLM cục bộ (Ollama, vLLM) hoặc qua API (OpenAI, Anthropic).

### **2.3. Xu hướng và giới hạn**

Xu hướng: RAG tiếp tục là kỹ thuật nền tảng trong các ứng dụng AI tạo sinh doanh nghiệp, với xu hướng gần đây là kết hợp thêm re-ranking (sắp xếp lại kết quả truy xuất bằng mô hình chuyên biệt), hybrid search (kết hợp tìm kiếm ngữ nghĩa và từ khóa), và agentic RAG (cho phép mô hình tự quyết định khi nào cần truy xuất thêm thông tin.  
Giới hạn: với các câu hỏi đòi hỏi suy luận đa bước phức tạp hoặc tổng hợp thông tin không có trong tài liệu cụ thể, RAG truyền thống (truy xuất một lần) có thể không đủ, các hướng nghiên cứu mới như multi-hop RAG hoặc agentic RAG đang được phát triển để khắc phục.  
Kết luận phạm vi NÊN và KHÔNG NÊN dùng: nhóm kết luận RAG NÊN được dùng cho các hệ thống hỏi đáp trên tài liệu riêng, cần độ chính xác và khả năng cập nhật cao — đúng như bài toán SmartDoc AI; và KHÔNG NÊN dùng cho các bài toán cần suy luận tổng hợp sâu xuyên suốt tài liệu dài hoặc không có nguồn tài liệu tham chiếu cụ thể.

# 

# **PHẦN 2: KIẾN THỨC BẢN CHẤT VÀ THỰC HÀNH**

Mục tiêu: chứng minh nhóm hiểu công nghệ qua chuỗi Khái niệm/cơ chế → Thực hành → Kết quả → Giải thích.

## **Chương 3: Thiết lập môi trường và kiến trúc dự án**

### **3.1. Công cụ cần thiết**

* IDE: VS Code hoặc PyCharm — hỗ trợ tốt Python và các thư viện AI.  
* Python 3.8+ — ngôn ngữ chính để xây dựng pipeline RAG.  
* Ollama runtime — nền tảng chạy LLM cục bộ, dùng để kéo và phục vụ mô hình Qwen2.5:7b.  
* Thư viện chính: FAISS (lưu trữ vector), một thư viện embedding (ví dụ sentence-transformers), PyPDF hoặc pdfplumber (trích xuất văn bản PDF), Streamlit (giao diện người dùng).  
* Git/GitHub — quản lý phiên bản mã nguồn theo nhánh, review chéo giữa các thành viên.  
* Docker (tùy chọn) — đóng gói ứng dụng để chạy lại được trên máy khác.

### **3.2. Cài đặt và Hello World**

Các bước cài đặt cơ bản (tóm tắt):

* Cài đặt Ollama, kéo mô hình bằng lệnh ollama pull qwen2.5:7b.  
* Tạo môi trường ảo Python, cài các thư viện qua pip install \-r requirements.txt.  
* Viết một đoạn "Hello World" đơn giản: gửi một câu hỏi trực tiếp đến Qwen2.5 qua Ollama (chưa có RAG) để xác nhận mô hình đã chạy được, ví dụ trả lời "Xin chào, tôi là trợ lý SmartDoc AI, tôi đã sẵn sàng."  
* Chạy thử ứng dụng bằng lệnh streamlit run app.py, mở trình duyệt để xác nhận giao diện khởi động thành công.

### **3.3. Kiến trúc**

Hệ thống SmartDoc AI được thiết kế theo mô hình kiến trúc pipeline tuần tự kết hợp phân chia module theo chức năng, đảm bảo mỗi thành phần có thể phát triển và kiểm thử độc lập. Cấu  
 trúc cụ thể gồm:

* Module Ingestion: tiếp nhận file PDF tải lên, trích xuất văn bản thô, kiểm tra định dạng và kích thước.  
* Module Chunking & Embedding: chia nhỏ văn bản theo chunk size/overlap cấu hình được, mã hóa từng chunk thành vector.  
* Module Vector Store: lưu trữ vector bằng FAISS, cung cấp hàm truy xuất top-k đoạn liên quan nhất theo câu hỏi.  
* Module Generation: ghép ngữ cảnh truy xuất được với câu hỏi thành prompt, gửi đến Qwen2.5 qua Ollama, nhận và trả về câu trả lời.  
* Module UI & Session: giao diện Streamlit cho phép tải file, đặt câu hỏi, điều chỉnh tham số, và quản lý lịch sử hội thoại.

Luồng dữ liệu tổng quát: Người dùng tải PDF → Ingestion trích xuất văn bản → Chunking chia nhỏ và Embedding mã hóa vector → lưu vào FAISS → người dùng đặt câu hỏi → câu hỏi được mã hóa và so khớp với FAISS để lấy top-k đoạn liên quan → Generation ghép ngữ cảnh và gọi Qwen2.5 → trả câu trả lời kèm hiển thị trạng thái loading trong lúc xử lý.  
Cách tổ chức này phù hợp với RAG vì tách biệt rõ giai đoạn chuẩn bị dữ liệu (offline: ingestion, chunking, embedding) và giai đoạn phục vụ truy vấn (online: retrieval, generation), giúp nhóm có thể tối ưu và kiểm thử riêng từng giai đoạn mà không ảnh hưởng lẫn nhau.

### **3.4. Thực hành khởi tạo**

Minh chứng môi trường hoạt động và các thành phần chính kết nối thành công gồm:  
Kiểm tra kết nối Ollama: gọi thử một câu hỏi đơn giản đến mô hình Qwen2.5 và xác nhận nhận được phản hồi.  
Kiểm tra pipeline chunking \+ embedding: tải một PDF mẫu, xác nhận số lượng chunk được tạo ra và kích thước vector embedding tương ứng.  
Kiểm tra truy xuất FAISS: đặt một câu hỏi có đáp án rõ ràng trong tài liệu mẫu, xác nhận hệ thống truy xuất đúng đoạn văn bản chứa đáp án.  
Kiểm tra luồng end-to-end: từ tải file đến nhận câu trả lời hoàn chỉnh trên giao diện Streamlit, xác nhận trạng thái loading hiển thị đúng trong lúc xử lý.  
