"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Namespace cho DOCX XML
DOCX_NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
}


def _extract_text_from_docx(filepath: str) -> str:
    """
    Extract text từ file DOCX bằng zipfile + xml (stdlib, không cần python-docx).

    DOCX là ZIP chứa XML. Text nằm trong word/document.xml,
    tag <w:t> chứa text, <w:p> là paragraph.
    """
    text_parts = []

    with zipfile.ZipFile(filepath, 'r') as z:
        with z.open('word/document.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()

    for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        # Thu thập text trong paragraph
        para_texts = []
        for run in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
            if run.text:
                para_texts.append(run.text)

        line = ''.join(para_texts).strip()
        if line:
            text_parts.append(line)

    return '\n\n'.join(text_parts)


def _docx_to_markdown(filepath: str) -> str:
    """
    Convert DOCX sang Markdown đơn giản.
    Heading detection dựa trên độ dài dòng và pattern tiếng Việt.
    """
    raw_text = _extract_text_from_docx(filepath)

    lines = raw_text.split('\n\n')
    md_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect heading patterns phổ biến trong văn bản pháp luật VN
        if re.match(r'^(Chương|Mục|Điều|CHƯƠNG|MỤC|ĐIỀU)\s+\w+', line):
            # Văn bản pháp luật: Chương, Mục, Điều → heading level 2
            md_lines.append(f'## {line}')
        elif len(line) < 80 and line.isupper():
            # Dòng ngắn toàn chữ hoa → heading level 1
            md_lines.append(f'# {line}')
        else:
            md_lines.append(line)

    return '\n\n'.join(md_lines)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                if filepath.suffix.lower() == ".docx":
                    # Dùng stdlib zipfile + xml thay vì MarkItDown (cần mammoth)
                    content = _docx_to_markdown(str(filepath))
                else:
                    # PDF — thử dùng MarkItDown
                    try:
                        from markitdown import MarkItDown
                        md = MarkItDown()
                        result = md.convert(str(filepath))
                        content = result.text_content
                    except Exception as e:
                        print(f"  ⚠ MarkItDown failed for {filepath.name}: {e}")
                        content = f"# {filepath.stem}\n\n[Conversion failed: {e}]"

                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(content, encoding="utf-8")
                char_count = len(content)
                print(f"  ✓ Saved: {output_path.name} ({char_count} chars)")
                converted += 1

            except Exception as e:
                print(f"  ✗ Error converting {filepath.name}: {e}")

    print(f"  Converted {converted} legal documents")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown.

    Lưu ý: JSON từ crawl4ai đã có field 'content_markdown' là markdown sẵn.
    Chỉ cần thêm header metadata rồi lưu ra file .md.
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            # Thêm metadata header
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            # content_markdown đã là markdown từ crawl4ai — dùng trực tiếp
            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path.name} ({len(content)} chars)")
            converted += 1

    print(f"  Converted {converted} news articles")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
