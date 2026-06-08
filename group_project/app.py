import streamlit as st
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Thêm root dir vào sys.path để import các module từ src/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables từ file .env ở root
load_dotenv(PROJECT_ROOT / ".env")

from src.task10_generation import generate_with_citation

# Cấu hình API endpoint và headers từ file .env cho phần Reformulate Query
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = "https://api.xah.io/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="RAG Trợ Lý Pháp Luật Ma Túy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling cho giao diện Glassmorphism và Bong bóng Chat đẹp mắt
st.markdown("""
<style>
    /* Tổng thể App */
    .stApp {
        background: radial-gradient(circle at top right, #1a1e29, #0d1117);
        color: #e6edf3;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Bong bóng chat user */
    .chat-user {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 15px 15px 0 15px;
        padding: 12px 18px;
        margin: 10px 0;
        width: fit-content;
        max-width: 80%;
        float: right;
        clear: both;
        color: #e6edf3;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    
    /* Bong bóng chat assistant */
    .chat-assistant {
        background: rgba(31, 111, 235, 0.1);
        border: 1px solid rgba(31, 111, 235, 0.4);
        border-radius: 15px 15px 15px 0;
        padding: 12px 18px;
        margin: 10px 0;
        width: fit-content;
        max-width: 80%;
        float: left;
        clear: both;
        color: #e6edf3;
        box-shadow: 0 4px 10px rgba(31, 111, 235, 0.05);
    }
    
    /* Nút bấm */
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #388bfd;
        box-shadow: 0 0 12px rgba(56, 139, 253, 0.4);
    }
    
    /* Highlight văn bản trích dẫn */
    mark {
        background-color: rgba(242, 193, 46, 0.3);
        color: #f2c12e;
        border-bottom: 1px dashed #f2c12e;
        padding: 0 4px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


def reformulate_query(chat_history: list, new_query: str) -> str:
    """
    Sử dụng LLM để viết lại câu hỏi chứa ngữ cảnh lịch sử chat (Conversation Memory).
    """
    if not chat_history:
        return new_query

    # Chuẩn bị lịch sử hội thoại dạng text
    history_text = ""
    for msg in chat_history[-4:]:  # Lấy tối đa 4 lượt chat gần nhất để tránh tràn context
        role_label = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role_label}: {msg['content']}\n"

    prompt = f"""Dựa trên lịch sử hội thoại sau đây và một câu hỏi mới, hãy viết lại câu hỏi mới này thành một câu hỏi duy nhất, độc lập và đầy đủ ý nghĩa (decontextualized query) để hệ thống RAG có thể tìm kiếm chính xác điều luật hoặc tin tức. 
Chỉ trả về duy nhất câu hỏi đã được viết lại bằng tiếng Việt, không kèm theo bất kỳ giải thích nào khác.

Lịch sử hội thoại:
{history_text}

Câu hỏi mới: {new_query}

Câu hỏi độc lập viết lại:"""

    payload = {
        "model": "gpt-5.4-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        res = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
        if res.status_code == 200:
            rewritten = res.json()["choices"][0]["message"]["content"].strip()
            # Loại bỏ ngoặc kép nếu LLM tự động thêm vào
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            return rewritten
    except Exception as e:
        pass
    return new_query


def highlight_keywords(text: str, query: str) -> str:
    """
    Tự động highlight các từ khóa bằng định dạng markdown bold **word**.
    """
    import re
    # Trích xuất các từ có nghĩa từ câu hỏi (bỏ từ ngắn)
    words = [w.strip() for w in re.split(r'[\s,.\-\?\!\"]+', query) if len(w) > 2]
    words = list(set(words))
    words.sort(key=len, reverse=True)
    
    highlighted = text
    for word in words[:5]: # Chỉ highlight tối đa 5 từ khóa chính
        # Dùng \b để khớp từ nguyên vẹn, tránh thay thế một phần của tag hoặc từ khác
        pattern = re.compile(r'\b(' + re.escape(word) + r')\b', re.IGNORECASE)
        highlighted = pattern.sub(r'**\1**', highlighted)
        
    return highlighted


def process_citations_and_sources(answer: str, sources: list):
    """
    Xử lý câu trả lời của LLM để:
    1. Chuyển các trích dẫn dạng [Nguồn] thành link neo HTML <a href="#source-X">.
    2. Trích xuất câu cụ thể trong tài liệu nguồn tương thích với trích dẫn để highlight.
    """
    import re
    import unicodedata
    
    def clean_text(t):
        # Chuẩn hóa văn bản tiếng Việt bỏ dấu và ký tự đặc biệt để so khớp tốt hơn
        t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8')
        return re.sub(r'[^a-z0-9]', '', t.lower())

    processed_answer = answer
    source_map = {} # map từ citation string gốc sang index của source (1-indexed)
    
    # 1. Tìm tất cả các trích dẫn trong ngoặc vuông (ví dụ [luat-phong-chong-ma-tuy-2021...])
    citations = re.findall(r'\[([^\]]+)\]', answer)
    
    # 2. So khớp từng citation với 5 sources
    for cit in citations:
        cit_clean = clean_text(cit)
        matched_idx = None
        
        for idx, src in enumerate(sources, 1):
            filename = src.get("metadata", {}).get("source", "")
            file_clean = clean_text(filename)
            
            # Khớp nếu citation chứa một phần tên file hoặc ngược lại, hoặc chỉ số document
            if cit_clean in file_clean or file_clean in cit_clean or f"document{idx}" in cit_clean:
                matched_idx = idx
                break
                
        if matched_idx is not None:
            source_map[cit] = matched_idx
            # Thay thế trích dẫn gốc bằng link neo HTML để người dùng click là cuộn xuống
            link_html = f'<a href="#source-{matched_idx}" style="color: #58a6ff; text-decoration: none; font-weight: bold; border-bottom: 1px dotted #58a6ff;">[{matched_idx}]</a>'
            processed_answer = processed_answer.replace(f"[{cit}]", link_html)
            
    # 3. Với mỗi source, tìm đúng vị trí câu được trích dẫn để highlight
    highlighted_sources = []
    for idx, src in enumerate(sources, 1):
        content = src.get("content", "")
        filename = src.get("metadata", {}).get("source", "Tài liệu")
        score = src.get("score", 0.0)
        
        # Tách content của document nguồn thành các câu
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
        
        # Tách các câu trong câu trả lời của LLM để tìm câu có chứa trích dẫn nguồn này
        llm_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
        relevant_llm_text = ""
        for s in llm_sentences:
            for cit, s_idx in source_map.items():
                if s_idx == idx and f"[{cit}]" in s:
                    relevant_llm_text += " " + s
                    
        best_sentence_idx = -1
        best_overlap_score = 0
        
        # So khớp từ khóa để tìm câu gốc tương thích nhất trong document nguồn
        if relevant_llm_text:
            llm_words = set(clean_text(relevant_llm_text))
            for s_idx, sent in enumerate(sentences):
                sent_words = set(clean_text(sent))
                overlap = len(llm_words.intersection(sent_words))
                if overlap > best_overlap_score:
                    best_overlap_score = overlap
                    best_sentence_idx = s_idx
                    
        # Định dạng nội dung nguồn: highlight câu đúng trích dẫn + hiển thị ngữ cảnh xung quanh
        formatted_content = ""
        if best_sentence_idx != -1 and best_overlap_score > 3:
            start = max(0, best_sentence_idx - 1)
            end = min(len(sentences), best_sentence_idx + 2)
            
            parts = []
            for s_idx in range(start, end):
                sent = sentences[s_idx]
                if s_idx == best_sentence_idx:
                    # Highlight câu được trích dẫn bằng chữ in đậm kèm emoji chỉ hướng
                    parts.append(f"👉 **{sent}**")
                else:
                    parts.append(sent)
            formatted_content = "... " + " ".join(parts) + " ..."
        else:
            # Fallback nếu không tìm thấy câu cụ thể: hiển thị 2 câu đầu của chunk nguồn
            formatted_content = " ".join(sentences[:2]) + " ..."
            
        highlighted_sources.append({
            "idx": idx,
            "filename": filename,
            "score": score,
            "content": formatted_content,
            "metadata": src.get("metadata", {})
        })
        
    return processed_answer, highlighted_sources


# Khởi tạo session state cho lịch sử chat nếu chưa tồn tại
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/scales.png", width=100)
    st.title("⚖️ RAG Drug Law")
    st.write("Dự án nhóm - RAG Chatbot hỗ trợ tra cứu văn bản pháp luật ma túy & tin tức 2024.")
    
    st.markdown("---")
    
    # Lựa chọn trang
    page = st.radio("Chọn chức năng:", ["💬 Chatbot Trợ Lý", "📊 So Sánh Thuật Toán Lexical"])
    
    st.markdown("---")
    
    if st.button("🧹 Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.rerun()

# --- PAGE 1: CHATBOT TRỢ LÝ ---
if page == "💬 Chatbot Trợ Lý":
    st.subheader("💬 Tra Cứu Luật Ma Túy & Tin Tức")
    st.caption("Chatbot sử dụng Hybrid Search (Semantic + BM25) + Reranker + PageIndex Fallback.")

    # Container hiển thị chat
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
                # Hiển thị tài liệu nguồn nếu có
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Xem nguồn trích dẫn"):
                        for src in msg["sources"]:
                            st.markdown(f'<div id="source-{src["idx"]}"></div>', unsafe_allow_html=True)
                            st.markdown(f"**Nguồn [{src['idx']}]: {src['filename']}** (Độ liên quan: `{src['score']:.3f}`)")
                            blockquote_content = "\n".join([f"> {line}" for line in src["content"].split("\n")])
                            st.markdown(blockquote_content)
        # Tạo khoảng cách trống ở cuối
        st.markdown('<div style="clear:both; height:20px;"></div>', unsafe_allow_html=True)

    # Input chat mới
    if prompt := st.chat_input("Hỏi về tội tàng trữ ma túy, án phạt, hoặc tin tức nghệ sĩ..."):
        # Hiển thị câu hỏi của user ngay lập tức
        st.markdown(f'<div class="chat-user">{prompt}</div>', unsafe_allow_html=True)
        
        # Reformulate query dựa trên lịch sử
        reformulated = reformulate_query(st.session_state.messages, prompt)
        if reformulated != prompt:
            st.caption(f"🔍 *Câu hỏi được tối ưu hóa ngữ cảnh:* \"{reformulated}\"")
        
        # Lưu câu hỏi của user vào session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Gọi RAG pipeline
        with st.spinner("Đang tìm kiếm tài liệu và suy luận..."):
            try:
                result = generate_with_citation(reformulated)
                raw_answer = result.get("answer", "Không nhận được phản hồi từ hệ thống.")
                raw_sources = result.get("sources", [])
                
                # Xử lý trích dẫn và tìm câu trỏ đến nguồn chính xác
                processed_answer, processed_sources = process_citations_and_sources(raw_answer, raw_sources)
            except Exception as e:
                processed_answer = f"Đã xảy ra lỗi khi xử lý: {e}"
                processed_sources = []
        
        # Hiển thị câu trả lời của trợ lý
        st.markdown(f'<div class="chat-assistant">{processed_answer}</div>', unsafe_allow_html=True)
        
        # Hiển thị tài liệu nguồn ngay lập tức
        if processed_sources:
            with st.expander("📚 Xem nguồn trích dẫn"):
                for src in processed_sources:
                    st.markdown(f'<div id="source-{src["idx"]}"></div>', unsafe_allow_html=True)
                    st.markdown(f"**Nguồn [{src['idx']}]: {src['filename']}** (Độ liên quan: `{src['score']:.3f}`)")
                    blockquote_content = "\n".join([f"> {line}" for line in src["content"].split("\n")])
                    st.markdown(blockquote_content)

        # Lưu câu trả lời vào session state kèm sources để hiển thị lại
        st.session_state.messages.append({
            "role": "assistant",
            "content": processed_answer,
            "sources": processed_sources,
            "query": prompt
        })
        
        st.rerun()

# --- PAGE 2: SO SÁNH THUẬT TOÁN LEXICAL ---
else:
    st.subheader("📊 So Sánh Thuật Toán Lexical Search: BM25 vs TF-IDF")
    
    st.markdown("""
    Trong các bài toán RAG về văn bản pháp luật (như Bộ luật Hình sự) hoặc tin tức dài, thuật toán tìm kiếm từ khóa đóng vai trò vô cùng quan trọng. 
    Dưới đây là so sánh chi tiết giữa hai thuật toán phổ biến: **TF-IDF** (Vector Space Model truyền thống) và **BM25** (Best Matching 25 - chuẩn công nghiệp hiện đại).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 📘 TF-IDF (Term Frequency - Inverse Document Frequency)")
        st.markdown("""
        * **Ý tưởng**: Tính điểm dựa trên tần suất xuất hiện của từ khóa trong document (TF) và độ hiếm của từ đó trên toàn bộ tập văn bản (IDF).
        * **Hạn chế chính (TF Saturation)**: Điểm số TF tăng tuyến tính không giới hạn với số lần xuất hiện của từ. Nếu từ khóa xuất hiện 100 lần, điểm số sẽ gấp nhiều lần xuất hiện 5 lần, gây sai lệch khi tài liệu lặp từ vô nghĩa.
        * **Hạn chế độ dài**: Không tự động tối ưu hóa điểm số cho các tài liệu quá dài hoặc quá ngắn.
        """)
        
    with col2:
        st.success("### 🚀 BM25 (Best Matching 25)")
        st.markdown("""
        * **Ý tưởng**: Cải tiến TF-IDF bằng cách đưa vào hai tham số kiểm soát quan trọng là **k1** (bão hòa tần suất từ khóa) và **b** (bù trừ độ dài tài liệu).
        * **Bão hòa tần suất (TF Saturation)**: Điểm số TF sẽ tiệm cận một giới hạn tối đa ($1 + k1$) thay vì tăng vô hạn. Dù từ khóa xuất hiện 10 lần hay 100 lần, điểm số nhận được gần như tương đương.
        * **Bù trừ độ dài (Length Normalization)**: Phạt các tài liệu quá dài chứa nhiều từ khóa loãng, ưu tiên các tài liệu ngắn có tính cô đọng cao (thông qua tham số $b$).
        """)

    st.markdown("---")
    st.subheader("🧮 Trình Giả Lập Bão Hòa Điểm Số (TF Saturation Simulator)")
    st.write("Hãy thử thay đổi tần suất xuất hiện của từ khóa để xem điểm số TF tăng như thế nào trong TF-IDF so với BM25.")
    
    tf = st.slider("Tần suất xuất hiện của từ khóa trong văn bản (TF):", 1, 20, 3)
    k1 = st.slider("Tham số bão hòa k1 (chỉ áp dụng cho BM25):", 1.0, 2.5, 1.5, step=0.1)
    
    # Tính điểm giả lập
    tfidf_score = float(tf) # Giả lập TF-IDF tăng tuyến tính
    bm25_score = (tf * (k1 + 1)) / (tf + k1) # Công thức phần TF bão hòa trong BM25
    
    # Vẽ biểu đồ so sánh đơn giản bằng markdown
    st.markdown(f"""
    * **Điểm TF-IDF giả lập (Tuyến tính)**: `{tfidf_score:.2f}`  
      `{"█" * int(tfidf_score)}`
    * **Điểm BM25 giả lập (Bão hòa với k1={k1})**: `{bm25_score:.2f}` (Giới hạn tối đa là {1+k1:.2f})  
      `{"█" * int(bm25_score * 3)}`
    
    > **Nhận xét**: Khi tần suất xuất hiện tăng lên, điểm số của **BM25 sẽ nhanh chóng bão hòa** và không tăng thêm nữa, giúp ngăn chặn việc các văn bản spam từ khóa chiếm top tìm kiếm. Đây là lý do vì sao ở **Task 6**, chúng ta sử dụng **BM25Okapi** thay thế cho các thuật toán so khớp cosine TF-IDF truyền thống.
    """)
