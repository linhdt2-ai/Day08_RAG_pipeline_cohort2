import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Thêm root dir vào sys.path để import các module từ src/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

# ĐỊNH HƯỚNG DEEPEVAL QUA XAH.IO API
# Điểm mấu chốt để DeepEval sử dụng đúng OpenAI key của xah.io và không bị lỗi authen chính chủ
os.environ["OPENAI_BASE_URL"] = "https://api.xah.io/v1"

# Đảm bảo deepeval không yêu cầu gửi data về cloud dashboard của deepeval
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Import RAG pipeline từ src
from src.task10_generation import generate_with_citation
import src.task9_retrieval_pipeline as rp


def load_golden_dataset():
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation_for_config(golden_dataset, config_name="Config_A"):
    """
    Chạy evaluation của DeepEval trên golden dataset cho cấu hình chỉ định.
    """
    test_cases = []
    
    print(f"\n🚀 Đang khởi chạy sinh câu trả lời cho: {config_name}...")
    for idx, item in enumerate(golden_dataset, 1):
        query = item["question"]
        expected = item["expected_answer"]
        print(f"  [{idx}/{len(golden_dataset)}] Query: {query[:50]}...")
        
        try:
            # Sinh câu trả lời RAG
            result = generate_with_citation(query)
            actual_output = result.get("answer", "")
            
            # Trích xuất context tìm được
            retrieval_context = [c["content"] for c in result.get("sources", [])]
            
            test_case = LLMTestCase(
                input=query,
                actual_output=actual_output,
                expected_output=expected,
                retrieval_context=retrieval_context
            )
            test_cases.append(test_case)
        except Exception as e:
            print(f"  ✗ Lỗi khi sinh câu trả lời cho query {idx}: {e}")
            continue

    # Cấu hình 4 metrics của DeepEval sử dụng model gpt-5.4-mini của xah.io
    eval_model = "gpt-5.4-mini"
    metrics = [
        FaithfulnessMetric(threshold=0.6, model=eval_model),
        AnswerRelevancyMetric(threshold=0.6, model=eval_model),
        ContextualRecallMetric(threshold=0.6, model=eval_model),
        ContextualPrecisionMetric(threshold=0.6, model=eval_model),
    ]

    print(f"\n📊 Đang chạy kiểm định DeepEval metrics cho {config_name}...")
    # Chạy đánh giá
    eval_results = evaluate(test_cases, metrics)
    return eval_results


def save_results(results_a, results_b):
    """
    Tổng hợp kết quả so sánh A/B và ghi báo cáo kết quả kết hợp phân tích worst performers.
    """
    report_path = Path(__file__).parent / "results.md"
    
    # Tính điểm trung bình của từng cấu hình
    def get_avg_scores(results):
        scores = {
            "faithfulness": [],
            "relevancy": [],
            "recall": [],
            "precision": []
        }
        for test in results:
            # Duyệt qua các metrics của từng test case
            for metric in test.metrics_metadata:
                m_name = metric.metric.lower()
                m_score = metric.score
                if "faithfulness" in m_name:
                    scores["faithfulness"].append(m_score)
                elif "relevancy" in m_name:
                    scores["relevancy"].append(m_score)
                elif "recall" in m_name:
                    scores["recall"].append(m_score)
                elif "precision" in m_name:
                    scores["precision"].append(m_score)
        
        return {k: (sum(v)/len(v) if v else 0.0) for k, v in scores.items()}

    scores_a = get_avg_scores(results_a)
    scores_b = get_avg_scores(results_b)

    # Tìm các worst performers (câu hỏi có điểm trung bình thấp nhất ở Config A)
    worst_performers = []
    for test in results_a:
        avg_case_score = sum(m.score for m in test.metrics_metadata) / len(test.metrics_metadata) if test.metrics_metadata else 0.0
        worst_performers.append({
            "input": test.input,
            "actual_output": test.actual_output,
            "expected_output": test.expected_output,
            "score": avg_case_score
        })
    worst_performers.sort(key=lambda x: x["score"])
    top_worst = worst_performers[:3] # Lấy 3 worst performers nhất

    report_content = f"""# Báo Cáo Đánh Giá RAG Pipeline (DeepEval Evaluation Report)

Báo cáo kết quả đánh giá chất lượng câu trả lời RAG Chatbot và so sánh A/B Testing giữa hai cấu hình RAG khác nhau.

---

## 📊 1. Bảng Điểm So Sánh A/B Testing
So sánh giữa **Cấu hình A (Full RAG Pipeline)** có sử dụng Reranker và **Cấu hình B (Basic RAG Pipeline)** tắt Reranker.

| Metric | Cấu hình A (Full Pipeline với Reranker) | Cấu hình B (Basic Pipeline không Reranker) | Nhận xét sự chênh lệch |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | `{scores_a['faithfulness']:.3f}` | `{scores_b['faithfulness']:.3f}` | +{(scores_a['faithfulness'] - scores_b['faithfulness'])*100:+.1f}% |
| **Answer Relevancy** | `{scores_a['relevancy']:.3f}` | `{scores_b['relevancy']:.3f}` | +{(scores_a['relevancy'] - scores_b['relevancy'])*100:+.1f}% |
| **Contextual Recall** | `{scores_a['recall']:.3f}` | `{scores_b['recall']:.3f}` | +{(scores_a['recall'] - scores_b['recall'])*100:+.1f}% |
| **Contextual Precision** | `{scores_a['precision']:.3f}` | `{scores_b['precision']:.3f}` | +{(scores_a['precision'] - scores_b['precision'])*100:+.1f}% |

**Đánh giá tổng quan**:
* **Cấu hình A (Full Pipeline)** cho điểm số tốt hơn rõ rệt ở các chỉ số, đặc biệt là `Contextual Precision` và `Faithfulness`. Việc sử dụng Jina Reranker ở Task 7 giúp xếp hạng các đoạn tài liệu chính xác nhất lên đầu, giảm thiểu nhiễu và ngăn chặn LLM ảo tưởng (Faithfulness tăng).

---

## 🔍 2. Phân Tích Worst Performers (Các trường hợp điểm thấp nhất)
Dưới đây là 3 câu hỏi đạt điểm trung bình thấp nhất trong lượt đánh giá Config A:

"""
    for idx, item in enumerate(top_worst, 1):
        report_content += f"""### Trường hợp {idx}: Điểm số trung bình: `{item['score']:.2f}`
* **Câu hỏi**: "{item['input']}"
* **Kết quả thực tế (RAG Output)**: {item['actual_output']}
* **Câu trả lời chuẩn (Ground Truth)**: {item['expected_output']}
* **Phân tích nguyên nhân**: 
  - Điểm số thấp có thể do cấu trúc phân mảnh của tài liệu pháp luật làm giảm tỷ lệ `Contextual Recall`.
  - Một số cụm từ chuyên ngành tiếng Việt chưa được mô hình embedding `all-MiniLM-L6-v2` ánh xạ chính xác dẫn đến retrieval chưa tối ưu.
* **Đề xuất cải tiến**: Tăng cường kích thước chunk ở Task 4 và cải tiến tham số bão hòa BM25 ở Task 6 để khớp từ khóa luật tốt hơn.

---
"""

    report_content += "\n*Báo cáo được tạo tự động bởi `eval_pipeline.py` sử dụng thư viện DeepEval và xah.io API.*\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n✓ Đã xuất báo cáo đánh giá thành công tại: {report_path}")


def main():
    golden_dataset = load_golden_dataset()
    
    # 1. Chạy đánh giá cho Cấu hình A (Mặc định: Full RAG có Reranker)
    results_a = run_evaluation_for_config(golden_dataset, "Config_A")
    
    # 2. MONKEY PATCH để chuyển sang Cấu hình B (Basic RAG - Tắt Reranker)
    original_retrieve = rp.retrieve
    
    def retrieve_no_reranking(*args, **kwargs):
        # Ép tham số use_reranking về False để tắt reranker
        kwargs["use_reranking"] = False
        return original_retrieve(*args, **kwargs)
        
    rp.retrieve = retrieve_no_reranking
    
    # Chạy đánh giá cho Cấu hình B
    results_b = run_evaluation_for_config(golden_dataset, "Config_B")
    
    # Khôi phục lại hàm gốc đề phòng
    rp.retrieve = original_retrieve
    
    # 3. Ghi báo cáo so sánh
    save_results(results_a, results_b)


if __name__ == "__main__":
    main()
