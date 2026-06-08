# ⚖️ Trợ Lý Pháp Luật Ma Tuý - RAG Chatbot & Evaluation Pipeline

Dự án nhóm xây dựng hệ thống **RAG Chatbot** tra cứu Văn bản Pháp luật về Ma tuý và tin tức liên quan trong năm 2024, tích hợp toàn bộ các module từ bài tập cá nhân và nâng cấp thêm các tính năng nâng cao.

---

## 🏗️ Kiến Trúc Hệ Thống (Integrated RAG Architecture)

Hệ thống kết hợp tìm kiếm Hybrid, xếp hạng lại (Reranking), cơ chế truy vấn giả định (HyDE), lưu giữ lịch sử chat (Conversation Memory) và tìm kiếm fallback không vector (PageIndex):

```
                       [Người dùng nhập câu hỏi]
                                  │
                       [Tái cấu trúc câu hỏi với]
                       [  Conversation Memory   ]
                                  │
                       [Sinh văn bản giả định  ]
                       [        (HyDE)          ]
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
      [Dense Retrieval]                    [Sparse Retrieval]
     (Cosine Similarity)                     (BM25Okapi)
     ChromaDB + MiniLM-L6                        src/
                │                                   │
                └─────────────────┬─────────────────┘
                                  ▼
                     [Reciprocal Rank Fusion]
                                  │
                       [Jina AI Cross-Encoder]
                       [      Reranker       ]
                                  │
                       [Kiểm tra Score Threshold]
                                  ▼
                        ┌─────────┴─────────┐
                        │ >= Threshold?     │
                     Có │                Không │
                        ▼                      ▼
                  [Sắp xếp Chunks]       [PageIndex Fallback]
                 (Avoid lost-in-middle)  (Vectorless Search)
                        │                      │
                        └─────────┬────────────┘
                                  ▼
                       [Prompt Injection & LLM]
                           (gpt-5.4-mini)
                                  │
                      [Câu trả lời kèm Citation]
                      [   và Highlight Source   ]
```

---

## 🚀 Các Tính Năng Nâng Cao Triển Khai (Bonus Features)

1. **Conversation Memory (Multi-turn Chat)**: Chatbot ghi nhớ lịch sử chat trong phiên làm việc. Sử dụng LLM để tự động viết lại câu hỏi mới (Reformulate Query) dựa trên ngữ cảnh hội thoại trước đó trước khi đưa vào pipeline tìm kiếm.
2. **HyDE (Hypothetical Document Embeddings)**: Sinh câu trả lời giả định ngắn từ câu hỏi gốc của user, sau đó dùng câu trả lời giả định này để vector search trong ChromaDB nhằm tìm đúng ngữ cảnh sâu sắc hơn.
3. **Trích Dẫn & Highlight Source**: Hiển thị chi tiết tên file nguồn, điểm số tương đồng (Relevance Score) của từng đoạn tài liệu trích dẫn và tự động tô sáng (highlight) các từ khóa quan trọng được đối chiếu.
4. **Trực Quan So Sánh Thuật Toán**: Tích hợp trang giả lập bão hòa điểm số (TF Saturation) để giải thích sự ưu việt của BM25 so với TF-IDF trong tìm kiếm văn bản pháp lý.

---

## 📂 Cấu Trúc Mã Nguồn Dự Án

* `group_project/app.py`: Giao diện chính chatbot tương tác viết bằng Streamlit.
* `group_project/evaluation/`:
  * `golden_dataset.json`: Tập dữ liệu 15 câu hỏi Q&A chuẩn (bao gồm expected answer và expected context).
  * `eval_pipeline.py`: Script chạy tự động đánh giá 4 metrics (Faithfulness, Answer Relevancy, Contextual Recall, Contextual Precision) sử dụng DeepEval và thực hiện so sánh A/B Testing.
  * `results.md`: Báo cáo kết quả đánh giá A/B Testing chi tiết và phân tích lỗi.
* `src/`: Thư mục chứa các module lõi được tái sử dụng trực tiếp từ bài cá nhân.

---

## 👥 Phân Công Công Việc Nhóm

| Thành viên | MSSV | Vai Trò | Nhiệm vụ chính | Trạng thái |
| :--- | :---: | :---: | :--- | :---: |
| **Thành viên 1** (Leader) | `123456` | **Integration Engineer** | Tích hợp các module `src/` bài cá nhân vào RAG pipeline, cấu hình HyDE và PageIndex Fallback. | `[x]` Hoàn thành |
| **Thành viên 2** | `234567` | **UI/UX Developer** | Xây dựng giao diện Streamlit Chatbot (`app.py`), implement Conversation Memory và highlight từ khóa. | `[x]` Hoàn thành |
| **Thành viên 3** | `345678` | **Evaluation Engineer** | Xây dựng 15+ cặp Golden Q&A, lập trình script chạy DeepEval metrics và viết báo cáo `results.md`. | `[x]` Hoàn thành |
| **Thành viên 4** | `456789` | **DevOps & QA** | Quản lý repo, cấu hình và deploy chatbot lên Hugging Face Spaces, QA test chất lượng câu trả lời. | `[x]` Hoàn thành |

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Cài đặt thư viện bổ sung
Đảm bảo bạn đã kích hoạt môi trường ảo và cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
# Hoặc cài đặt thêm deepeval và streamlit nếu chưa có
pip install deepeval streamlit
```

### 2. Thiết lập cấu hình biến môi trường
Tạo file `.env` ở thư mục root (hoặc copy từ `.env.example`) và điền đầy đủ API keys:
```env
OPENAI_API_KEY=sk-your-openai-api-key
JINA_API_KEY=jina-your-jina-api-key
PAGEINDEX_API_KEY=your-pageindex-api-key
```

### 3. Chạy ứng dụng Chatbot
Chạy giao diện Streamlit trên máy local:
```bash
streamlit run group_project/app.py
```
Ứng dụng sẽ tự động mở tại địa chỉ `http://localhost:8501`.

### 4. Chạy kiểm thử tự động Evaluation Pipeline
Để tự động chạy đánh giá 4 metrics của DeepEval trên 15 câu hỏi của Golden Dataset và xuất bảng so sánh A/B Testing:
```bash
python group_project/evaluation/eval_pipeline.py
```
Sau khi chạy xong, hãy kiểm tra báo cáo kết quả chi tiết được xuất tại: `group_project/evaluation/results.md`.
