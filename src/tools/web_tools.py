"""
Web tools for the ReAct agent ("trên mạng" / GitHub lookups).

These let the agent reach beyond the local lab knowledge base to fetch
datasheets, protocol references, STM32 HAL docs, etc. They degrade gracefully
when offline so the agent can still answer from local tools.
"""
import requests

from src.knowledge import loader

_TIMEOUT = 15
_HEADERS = {"User-Agent": "Mozilla/5.0 (EmbeddedLabAssistant)"}


def web_search(args: str) -> str:
    """Tìm kiếm trên Internet (DuckDuckGo). Tham số: câu truy vấn."""
    query = args.strip()
    if not query:
        return "Cần cung cấp từ khóa tìm kiếm."
    try:
        from ddgs import DDGS  # lazy import; optional dependency
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return ("Chưa cài thư viện tìm kiếm. Chạy: pip install ddgs "
                    "(hoặc dùng MCP server 'fetch').")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:  # network or rate limit
        return f"Lỗi tìm kiếm ({e}). Có thể đang offline."
    if not results:
        return f"Không có kết quả cho '{query}'."
    out = [f"Kết quả web cho '{query}':"]
    for r in results:
        title = r.get("title", "")
        href = r.get("href", "") or r.get("url", "")
        body = (r.get("body", "") or "")[:200]
        out.append(f"- {title} ({href})\n  {body}")
    return "\n".join(out)


def fetch_url(args: str) -> str:
    """Tải nội dung văn bản của một URL. Tham số: địa chỉ URL."""
    url = args.strip().strip("<>\"'")
    if not url.startswith(("http://", "https://")):
        return "URL không hợp lệ (phải bắt đầu bằng http:// hoặc https://)."
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        return f"Không tải được {url} ({e})."

    text = resp.text
    # Very light HTML -> text reduction to keep observations small.
    import re
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000] + ("..." if len(text) > 2000 else "")
def list_course(args: str = "") -> str:
    """
    Trả về thông tin tổng quan khóa học và các lab hiện có.
    Tham số: để trống.
    """
    data = loader.load()
    labs = loader.list_labs()

    course = data.get("course", "IT4210")
    kit = data.get("kit", "STM32F429I-DISC1")

    lines = [
        f"Khóa học: {course}",
        f"Kit chính: {kit}",
        "Danh sách lab:",
    ]

    for lab in labs:
        lines.append(f"- Lab {lab['id']}: {lab['title']}")

    return "\n".join(lines)