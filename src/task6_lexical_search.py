"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# Corpus loading & BM25 index
# =============================================================================

def _load_corpus() -> list[dict]:
    """
    Load tất cả .md files từ data/standardized/ làm corpus.

    Dùng toàn bộ document (không phải chunks) vì BM25 hoạt động tốt
    với document dài hơn — term frequency có ngữ cảnh đủ rộng.
    """
    corpus = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        corpus.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "filepath": str(md_file),
            }
        })
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25Okapi index từ corpus.

    Tokenize đơn giản bằng lowercase + split — đủ cho tiếng Việt cơ bản.
    (Nếu cần chính xác hơn, có thể dùng underthesea word tokenizer)

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi object
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


# Module-level initialization: load corpus khi import
CORPUS: list[dict] = _load_corpus()
_bm25 = build_bm25_index(CORPUS) if CORPUS else None


# =============================================================================
# Lexical Search
# =============================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score (không bounded)
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not CORPUS or _bm25 is None:
        return []

    # Tokenize query giống corpus
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    # Sort descending và lấy top_k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0:  # Chỉ trả về docs có keyword match thực sự
            results.append({
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx]["metadata"]
            })

    # Đã sorted descending do argsort[::-1]
    return results


if __name__ == "__main__":
    # Test
    test_queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "nghệ sĩ bị bắt ma tuý",
        "hình phạt tù",
    ]
    print(f"Corpus size: {len(CORPUS)} documents")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = lexical_search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] [{r['metadata']['type']}] {r['content'][:80]}...")
