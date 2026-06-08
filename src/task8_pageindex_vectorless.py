"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Chỉ cần chạy một lần. Documents sẽ được lưu trên PageIndex cloud.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError(
            "PAGEINDEX_API_KEY not set. Register at https://pageindex.ai/\n"
            "Then add PAGEINDEX_API_KEY=... to .env"
        )

    # Thử import PageIndexClient (SDK mới) hoặc PageIndex (SDK cũ)
    PageIndexClientClass = None
    try:
        from pageindex import PageIndexClient
        PageIndexClientClass = PageIndexClient
    except ImportError:
        try:
            from pageindex import PageIndex
            PageIndexClientClass = PageIndex
        except ImportError as e:
            raise ImportError(f"Không thể import PageIndexClient hoặc PageIndex từ package 'pageindex': {e}")

    pi = PageIndexClientClass(api_key=PAGEINDEX_API_KEY)

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    print(f"Uploading {len(md_files)} documents to PageIndex...")

    for md_file in md_files:
        try:
            # Chọn phương thức upload tương thích với phiên bản
            if hasattr(pi, "submit_document"):
                res = pi.submit_document(file_path=str(md_file))
                print(f"  ✓ Uploaded: {md_file.name}, doc_id: {res.get('doc_id')}")
            elif hasattr(pi, "upload"):
                content = md_file.read_text(encoding="utf-8")
                pi.upload(
                    content=content,
                    metadata={
                        "filename": md_file.name,
                        "type": md_file.parent.name  # "legal" hoặc "news"
                    }
                )
                print(f"  ✓ Uploaded: {md_file.name}")
            else:
                print(f"  ✗ Client doesn't support upload or submit_document methods.")
        except Exception as e:
            print(f"  ✗ Failed to upload {md_file.name}: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    PageIndex dùng structural understanding (không cần embedding/vector store):
    - Phân tích cấu trúc document (heading, section, paragraph)
    - Tìm kiếm dựa trên ngữ nghĩa và cấu trúc

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError(
            "PAGEINDEX_API_KEY not configured. Add to .env file."
        )

    # Thử import PageIndexClient (SDK mới) hoặc PageIndex (SDK cũ)
    PageIndexClientClass = None
    try:
        from pageindex import PageIndexClient
        PageIndexClientClass = PageIndexClient
    except ImportError:
        try:
            from pageindex import PageIndex
            PageIndexClientClass = PageIndex
        except ImportError as e:
            raise ImportError(f"Không thể import PageIndexClient hoặc PageIndex từ package 'pageindex': {e}")

    import time
    pi = PageIndexClientClass(api_key=PAGEINDEX_API_KEY)

    # THỨ NHẤT: Nếu SDK cũ hỗ trợ phương thức 'query' đồng bộ
    if hasattr(pi, "query"):
        try:
            results = pi.query(query=query, top_k=top_k)
        except TypeError:
            # Fallback nếu positional arg
            results = pi.query(query, top_k)

        output = []
        for r in results:
            content = (
                getattr(r, 'text', None) or
                getattr(r, 'content', None) or
                getattr(r, 'chunk', None) or
                (r.get('text') if isinstance(r, dict) else None) or
                (r.get('content') if isinstance(r, dict) else None) or
                str(r)
            )
            score = float(getattr(r, 'score', 0.0) or (r.get('score') if isinstance(r, dict) else 0.0))
            metadata = getattr(r, 'metadata', {}) or (r.get('metadata') if isinstance(r, dict) else {})

            output.append({
                "content": content,
                "score": score,
                "metadata": metadata,
                "source": "pageindex"
            })
        return output

    # THỨ HAI: Nếu SDK mới hỗ trợ phương thức 'submit_query' & 'get_retrieval' bất đồng bộ
    elif hasattr(pi, "submit_query") and hasattr(pi, "get_retrieval"):
        try:
            doc_resp = pi.list_documents()
            documents = doc_resp.get("documents", [])
        except Exception as e:
            print(f"Error listing documents from PageIndex: {e}")
            return []

        if not documents:
            print("No documents found in PageIndex. Please run upload_documents first.")
            return []

        all_nodes = []

        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                continue
            try:
                job = pi.submit_query(doc_id=doc_id, query=query)
                retrieval_id = job.get("retrieval_id")
                if not retrieval_id:
                    continue

                max_retries = 30
                retries = 0
                while retries < max_retries:
                    res = pi.get_retrieval(retrieval_id)
                    status = res.get("status")
                    if status == "completed":
                        nodes = res.get("retrieved_nodes", [])
                        for node in nodes:
                            content = node.get("text") or node.get("content") or ""
                            score = float(node.get("score") or 1.0)
                            metadata = node.get("metadata") or {}
                            metadata["doc_id"] = doc_id
                            metadata["filename"] = doc.get("name", "")
                            
                            all_nodes.append({
                                "content": content,
                                "score": score,
                                "metadata": metadata,
                                "source": "pageindex"
                            })
                        break
                    elif status == "failed":
                        print(f"Retrieval job {retrieval_id} failed.")
                        break
                    time.sleep(1)
                    retries += 1
            except Exception as e:
                print(f"Error querying doc {doc_id}: {e}")
                continue

        all_nodes.sort(key=lambda x: x["score"], reverse=True)
        return all_nodes[:top_k]

    else:
        raise AttributeError("PageIndex client object does not support 'query' or 'submit_query' methods.")


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print(f"✓ PAGEINDEX_API_KEY found: {PAGEINDEX_API_KEY[:8]}...")

        print("\nUploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:100]}...")
