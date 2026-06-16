"""
Cầu nối giữa AGENT và GROUND TRUTH cho Retrieval Eval.

Agent ReAct không trả về "doc-id" — nó gọi các tool (`get_lab_preparation('2')`,
`search_lab_docs('oled')`...). Module này dịch mỗi lời gọi tool thành danh sách
doc-id mà tool đó thực sự truy hồi, theo đúng hệ ID mà SDG (`data/synthetic_gen.py`)
dùng cho `expected_retrieval_ids`. Nhờ đó RetrievalEvaluator so được:

    retrieved_ids (suy từ trace agent)  vs  expected_retrieval_ids (golden set)

Hệ doc-id (xem thêm data/synthetic_gen.py):
    course.overview | lab{N}.objective | lab{N}.preparation | lab{N}.exercises
    lab{N}.pin_mappings | lab{N}.section.{code} | lab{N}.nec_protocol
"""
from typing import List

from src.knowledge import loader


def _dedupe(ids: List[str]) -> List[str]:
    """Bỏ trùng nhưng giữ nguyên thứ tự (thứ tự = thứ hạng cho MRR)."""
    return list(dict.fromkeys(ids))


def _lab_section_ids(lab_id: str) -> List[str]:
    lab = loader.get_lab(lab_id)
    if not lab:
        return []
    return [f"lab{lab_id}.section.{s['code']}" for s in lab.get("sections", [])]


def _search_label_to_id(lab_id: str, label: str) -> str:
    """Nhãn 'section' do loader.search trả về -> doc-id tương ứng."""
    if label == "Mục đích":
        return f"lab{lab_id}.objective"
    if label == "Bài tập":
        return f"lab{lab_id}.exercises"
    if label == "Sơ đồ chân":
        return f"lab{lab_id}.pin_mappings"
    if label == "NEC protocol":
        return f"lab{lab_id}.nec_protocol"
    # còn lại có dạng "{code} {title}" -> section
    code = label.split()[0] if label else ""
    return f"lab{lab_id}.section.{code}"


def tool_call_to_doc_ids(tool: str, args: str) -> List[str]:
    """Dịch MỘT lời gọi tool thành các doc-id mà nó truy hồi (có thể rỗng)."""
    args = (args or "").strip()

    if tool in ("list_available_labs", "list_course"):
        return ["course.overview"]

    if tool == "get_lab_objective":
        return [f"lab{args}.objective"] if loader.get_lab(args) else []

    if tool == "get_lab_preparation":
        return [f"lab{args}.preparation"] if loader.get_lab(args) else []

    if tool == "get_lab_exercises":
        return [f"lab{args}.exercises"] if loader.get_lab(args) else []

    if tool == "get_lab_sections":
        # Trả TẤT CẢ section của lab -> truy hồi mọi section doc của lab đó.
        return _lab_section_ids(args)

    if tool == "get_lab_section":
        # Truy hồi CÓ LỌC: đúng 1 section theo mã. args = "<lab> <code>".
        parts = args.split(maxsplit=1)
        if len(parts) == 2 and loader.get_lab(parts[0]):
            return [f"lab{parts[0]}.section.{parts[1].strip()}"]
        return []

    if tool == "get_exercise_guide":
        # args: "{lab} [chủ đề]" -> các section (đã lọc) + exercises của lab.
        lab_id = args.split()[0] if args else ""
        ids = _lab_section_ids(lab_id)
        if loader.get_lab(lab_id):
            ids.append(f"lab{lab_id}.exercises")
        return _dedupe(ids)

    if tool == "lookup_pin_mapping":
        if loader.get_lab(args):
            return [f"lab{args}.pin_mappings"]
        # Tra theo tên linh kiện -> các lab chứa linh kiện đó.
        key = loader._strip_accents(args)
        ids = []
        for lab_id, lab in loader.load()["labs"].items():
            for comp, pins in lab.get("pin_mappings", {}).items():
                if key in loader._strip_accents(comp + " " + pins):
                    ids.append(f"lab{lab_id}.pin_mappings")
        return _dedupe(ids)

    if tool == "search_lab_docs":
        hits = loader.search(args, max_results=5)
        return _dedupe(_search_label_to_id(h["lab_id"], h["section"]) for h in hits)

    if tool in ("web_search", "fetch_url"):
        return []  # nguồn ngoài KB -> không tính vào retrieval nội bộ

    return []


def trace_to_retrieved_ids(trace: List[dict]) -> List[str]:
    """
    Gộp toàn bộ trace ReAct (list các {tool, args, ...}) thành danh sách doc-id
    đã truy hồi, theo thứ tự tool được gọi (giữ thứ hạng đầu tiên cho MRR).
    """
    ids: List[str] = []
    for step in trace:
        ids.extend(tool_call_to_doc_ids(step.get("tool", ""), step.get("args", "")))
    return _dedupe(ids)
