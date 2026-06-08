# Báo Cáo Đánh Giá RAG Pipeline (DeepEval Evaluation Report)

Báo cáo kết quả đánh giá chất lượng câu trả lời RAG Chatbot và so sánh A/B Testing giữa hai cấu hình RAG khác nhau.

*Tài liệu này sẽ được tạo tự động với điểm số thực tế sau khi chạy script đánh giá:*
```bash
python group_project/evaluation/eval_pipeline.py
```

---

## 📊 1. Bảng Điểm So Sánh A/B Testing
So sánh giữa **Cấu hình A (Full RAG Pipeline)** có sử dụng Reranker và **Cấu hình B (Basic RAG Pipeline)** tắt Reranker.

| Metric | Cấu hình A (Full Pipeline với Reranker) | Cấu hình B (Basic Pipeline không Reranker) | Nhận xét sự chênh lệch |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | `Đang chờ đánh giá...` | `Đang chờ đánh giá...` | - |
| **Answer Relevancy** | `Đang chờ đánh giá...` | `Đang chờ đánh giá...` | - |
| **Contextual Recall** | `Đang chờ đánh giá...` | `Đang chờ đánh giá...` | - |
| **Contextual Precision** | `Đang chờ đánh giá...` | `Đang chờ đánh giá...` | - |

---

## 🔍 2. Phân Tích Worst Performers (Các trường hợp điểm thấp nhất)
Sau khi chạy đánh giá, 3 trường hợp câu hỏi có điểm số đánh giá kém nhất sẽ được liệt kê và phân tích nguyên nhân chi tiết tại đây kèm theo giải pháp cải tiến.
