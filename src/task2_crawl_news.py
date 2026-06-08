"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam:
https://vtcnews.vn/ca-si-chi-dan-su-nghiep-truot-doc-2-lan-bi-nghi-lien-quan-ma-tuy-ar906502.html 
https://vtcnews.vn/ca-si-chi-dan-va-anh-trai-bi-de-nghi-truy-to-lien-quan-to-chuc-su-dung-ma-tuy-ar960946.html 
https://vtcnews.vn/truoc-khi-bi-bat-giam-vi-ma-tuy-chi-dan-thuong-khoe-cuoc-song-giau-sang-ar907306.html 
https://vtcnews.vn/ntk-cong-tri-tu-lao-dai-hang-dau-lang-thoi-trang-viet-toi-toi-pham-ma-tuy-ar955900.html 
https://vtcnews.vn/cong-tri-va-loat-sao-viet-tung-bi-bat-vi-ma-tuy-ar955936.html

    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo cần crawl (5 bài về nghệ sĩ liên quan ma tuý)
ARTICLE_URLS = [
    "https://vtcnews.vn/ca-si-chi-dan-su-nghiep-truot-doc-2-lan-bi-nghi-lien-quan-ma-tuy-ar906502.html",
    "https://vtcnews.vn/ca-si-chi-dan-va-anh-trai-bi-de-nghi-truy-to-lien-quan-to-chuc-su-dung-ma-tuy-ar960946.html",
    "https://vtcnews.vn/truoc-khi-bi-bat-giam-vi-ma-tuy-chi-dan-thuong-khoe-cuoc-song-giau-sang-ar907306.html",
    "https://vtcnews.vn/ntk-cong-tri-tu-lao-dai-hang-dau-lang-thoi-trang-viet-toi-toi-pham-ma-tuy-ar955900.html",
    "https://vtcnews.vn/cong-tri-va-loat-sao-viet-tung-bi-bat-vi-ma-tuy-ar955936.html",
]


def _html_to_markdown(element) -> str:
    """Chuyển đổi đơn giản HTML element sang dạng markdown text."""
    lines = []
    for tag in element.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue
        name = tag.name
        if name == "h1":
            lines.append(f"# {text}")
        elif name == "h2":
            lines.append(f"## {text}")
        elif name in ("h3", "h4"):
            lines.append(f"### {text}")
        elif name == "blockquote":
            lines.append(f"> {text}")
        elif name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def _crawl_vtcnews(url: str, soup: BeautifulSoup) -> tuple[str, str]:
    """Parse tiêu đề và nội dung từ trang VTCNews."""
    # Title
    title_tag = soup.find("h1") or soup.find("meta", property="og:title")
    if title_tag:
        title = title_tag.get("content", "") if title_tag.name == "meta" else title_tag.get_text(strip=True)
    else:
        title = "Unknown"

    # Content — VTCNews dùng class "content-detail" hoặc "detail-content"
    content_div = (
        soup.find("div", class_="content-detail")
        or soup.find("div", class_="detail-content")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|article|body", re.I))
    )

    if content_div:
        content_markdown = _html_to_markdown(content_div)
    else:
        # Fallback: lấy toàn bộ <p> trong body
        paragraphs = soup.find_all("p")
        content_markdown = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    return title, content_markdown


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Sử dụng requests + BeautifulSoup (không cần browser, không cần
    quyền hệ thống đặc biệt). Nếu muốn dùng Crawl4AI (cần playwright),
    hãy bỏ comment phần code mẫu trong README.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")

    # Dispatch theo domain
    if "vtcnews.vn" in url:
        title, content_markdown = _crawl_vtcnews(url, soup)
    else:
        # Generic fallback
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Unknown"
        paragraphs = soup.find_all("p")
        content_markdown = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    success_count = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)

            # Lưu file JSON
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ Saved: {filepath}")
            print(f"    Title   : {article['title']}")
            print(f"    Content : {len(article['content_markdown'])} chars")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Lỗi khi crawl {url}: {e}")

    print(f"\nHoàn thành: {success_count}/{len(ARTICLE_URLS)} bài báo đã được lưu vào {DATA_DIR}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
