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
# Passage Extraction — Trích xuất đoạn ngắn từ document match
# =============================================================================

def _extract_best_passage(content: str, query_tokens: list, chunk_size: int = 500) -> str:
    """
    Tìm đoạn văn bản ngắn nhất (~chunk_size ký tự) chứa nhiều từ khóa nhất
    trong document, thay vì trả về toàn bộ nội dung file.

    BM25 tìm document match chính xác nhờ index toàn bộ (TF/IDF context đầy đủ),
    nhưng ta chỉ trả về đoạn passage liên quan nhất để tránh tốn token LLM.

    Args:
        content: Nội dung toàn bộ document
        query_tokens: Danh sách token từ khóa (đã lowercase)
        chunk_size: Số ký tự tối đa của passage trả về

    Returns:
        Đoạn văn bản ngắn ~chunk_size ký tự chứa nhiều từ khóa nhất.
    """
    if not content:
        return ""

    query_set = set(query_tokens)
    words = content.split()

    if not words:
        return content[:chunk_size]

    # Số từ xấp xỉ trong một window chunk_size ký tự (trung bình 5 ký tự/từ tiếng Việt)
    window_words = max(1, chunk_size // 5)
    step = max(1, window_words // 2)  # Sliding window, bước = 50% window

    best_start_word = 0
    best_overlap = -1

    for i in range(0, len(words), step):
        window = words[i:i + window_words]
        overlap = sum(1 for w in window if w.lower() in query_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_start_word = i

    # Tính vị trí ký tự tương ứng
    char_start = len(" ".join(words[:best_start_word]))
    if best_start_word > 0:
        char_start += 1  # bù khoảng trắng

    passage = content[char_start:char_start + chunk_size].strip()

    # Nếu passage quá ngắn (ít từ khóa đầu file), lấy từ đầu
    if len(passage) < chunk_size // 2:
        passage = content[:chunk_size].strip()

    return passage


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
            # Extract passage ngắn (~500 ký tự) chứa nhiều từ khóa nhất
            # thay vì trả nguyên file gốc có thể lên đến 432KB (~108k tokens)
            passage = _extract_best_passage(
                CORPUS[idx]["content"],
                tokenized_query,
                chunk_size=500
            )
            results.append({
                "content": passage,
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
