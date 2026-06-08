"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) — đã có JINA_API_KEY
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


# =============================================================================
# Helper
# =============================================================================

def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity giữa 2 vectors."""
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


# =============================================================================
# Method 1: Cross-encoder (Jina API) — PRIMARY method (đã có JINA_API_KEY)
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Dùng Jina Reranker v2 Multilingual API (hỗ trợ tiếng Việt).
    Fallback sang RRF nếu không có JINA_API_KEY.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    if not JINA_API_KEY:
        # Fallback sang RRF nếu không có API key
        print("  ⚠ JINA_API_KEY not set, falling back to RRF")
        return rerank_rrf([candidates], top_k=top_k)

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k
            },
            timeout=30
        )
        response.raise_for_status()
        reranked = response.json()["results"]
        return [
            {**candidates[r["index"]], "score": r["relevance_score"]}
            for r in reranked
        ]
    except Exception as e:
        print(f"  ⚠ Jina API error: {e}, falling back to RRF")
        return rerank_rrf([candidates], top_k=top_k)


# =============================================================================
# Method 2: MMR (Maximal Marginal Relevance)
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = _cosine_sim(
                query_embedding,
                candidates[idx].get("embedding", [0] * len(query_embedding))
            )

            # Max similarity to already selected docs
            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sim = _cosine_sim(
                    candidates[idx].get("embedding", [0]),
                    candidates[sel_idx].get("embedding", [0])
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [
        {**candidates[i], "score": candidates[i].get("score", 0.0)}
        for i in selected_indices
    ]


# =============================================================================
# Method 3: RRF (Reciprocal Rank Fusion)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Cơ chế: Mỗi document được score dựa trên rank của nó trong từng
    ranked list. Document xuất hiện ở rank cao trong nhiều list sẽ có
    tổng RRF score cao hơn. k=60 là smoothing constant từ paper
    Cormack et al. 2009 — giảm ảnh hưởng của rank 1 so với rank 2.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}   # content → RRF score
    content_map: dict[str, dict] = {}   # content → full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Mặc định dùng cross_encoder (Jina API) vì đã có JINA_API_KEY.
    Tự động fallback sang RRF nếu Jina API unavailable.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    elif method == "mmr":
        # MMR cần query_embedding — gọi rerank_mmr trực tiếp với embedding
        raise NotImplementedError(
            "MMR cần query_embedding. Gọi rerank_mmr(query_embedding, candidates, top_k) trực tiếp."
        )
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test với dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
        {"content": "Python programming tutorial", "score": 0.3, "metadata": {}},
    ]

    print("=== Test RRF ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3, method="rrf")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content']}")

    print("\n=== Test Cross-Encoder (Jina) ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3, method="cross_encoder")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content']}")
