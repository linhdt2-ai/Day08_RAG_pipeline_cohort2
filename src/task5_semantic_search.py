"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from pathlib import Path

# Dùng cùng config với Task 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "drug_law_docs"

# Cache model và collection để tránh reload mỗi lần query
_model = None
_collection = None


def _get_model():
    """Lazy-load embedding model (cache sau lần đầu)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    """Lazy-load ChromaDB collection (cache sau lần đầu)."""
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (cosine).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score [0, 1]
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    model = _get_model()
    collection = _get_collection()

    # Embed query với cùng model đã dùng khi index
    query_embedding = model.encode(
        query,
        normalize_embeddings=True  # Phải nhất quán với Task 4
    ).tolist()

    # Query ChromaDB — giới hạn n_results không vượt quá số docs có sẵn
    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )

    output = []
    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for doc, dist, meta in zip(docs, distances, metadatas):
        # ChromaDB cosine metric: distance = 1 - cosine_similarity
        # → score = 1 - distance (range [0, 1], cao = giống nhau)
        score = round(1.0 - dist, 4)
        output.append({
            "content": doc,
            "score": score,
            "metadata": meta or {}
        })

    # Đảm bảo sorted descending theo score
    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    # Test
    test_queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "nghệ sĩ bị bắt vì sử dụng chất cấm",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = semantic_search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] [{r['metadata'].get('type', '?')}] {r['content'][:80]}...")
