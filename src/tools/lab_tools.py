"""
Lab knowledge tools for the ReAct agent.

Each function takes a single string argument (as parsed from `Action: tool(args)`)
and returns a plain-text Observation. They cover the 3 student needs:
mục đích lab -> chuẩn bị lab -> hướng dẫn bài tập.
"""
from src.knowledge import loader


def list_available_labs(args: str = "") -> str:
    """Liệt kê các lab có trong knowledge base. Tham số: để trống."""
    data = loader.load()
    labs = loader.list_labs()
    lines = [f"- Lab {lab['id']}: {lab['title']}" for lab in labs]
    return (
        f"Khóa học: {data.get('course', 'IT4210')}\n"
        f"Kit chính: {data.get('kit', 'STM32F429I-DISC1')}\n"
        "Các lab có sẵn:\n"
        + "\n".join(lines)
    )


def get_lab_objective(args: str) -> str:
    """Trả về mục đích/mục tiêu của một bài lab. Tham số: số bài (1, 2 hoặc 3)."""
    lab = loader.get_lab(args)
    if not lab:
        return f"Không tìm thấy lab '{args}'. Các lab có sẵn: 1, 2, 3."
    lines = "\n".join(f"- {o}" for o in lab["objective"])
    return f"{lab['title']}\nMục đích:\n{lines}"


def get_lab_preparation(args: str) -> str:
    """Trả về phần chuẩn bị (phần cứng, phần mềm, tài liệu) của một lab. Tham số: số bài (1/2/3)."""
    lab = loader.get_lab(args)
    if not lab:
        return f"Không tìm thấy lab '{args}'. Các lab có sẵn: 1, 2, 3."
    prep = lab["preparation"]
    hw = "\n".join(f"  - {x}" for x in prep.get("hardware", []))
    sw = ", ".join(prep.get("software", []))
    docs = "\n".join(f"  - {x}" for x in prep.get("documents", []))
    return (
        f"{lab['title']}\n"
        f"Phần cứng:\n{hw}\n"
        f"Phần mềm: {sw}\n"
        f"Tài liệu:\n{docs}"
    )


def get_lab_sections(args: str) -> str:
    """Liệt kê các phần hướng dẫn chính của một lab. Tham số: số bài (1/2/3)."""
    lab = loader.get_lab(args)
    if not lab:
        return f"Không tìm thấy lab '{args}'. Các lab có sẵn: 1, 2, 3."
    sections = lab.get("sections", [])
    if not sections:
        return f"{lab['title']}\nChưa có phần hướng dẫn nào trong knowledge base."
    body = "\n".join(
        f"  [{section['code']}] {section['title']}: {section['guide']}"
        for section in sections
    )
    return f"{lab['title']}\nCác phần hướng dẫn:\n{body}"


def get_lab_section(args: str) -> str:
    """
    Lấy ĐÚNG MỘT phần hướng dẫn theo mã (tăng precision retrieval, tránh dump cả lab).
    Tham số: '<số bài> <mã section>', vd '2 3.7' hoặc '1 3.3'.
    """
    parts = args.strip().split(maxsplit=1)
    lab_id = parts[0] if parts else ""
    code = parts[1].strip() if len(parts) > 1 else ""

    lab = loader.get_lab(lab_id)
    if not lab:
        return f"Không tìm thấy lab '{lab_id}'. Các lab có sẵn: 1, 2, 3."
    for section in lab.get("sections", []):
        if section["code"] == code:
            return f"{lab['title']}\n[{section['code']}] {section['title']}: {section['guide']}"
    available = ", ".join(s["code"] for s in lab.get("sections", []))
    return f"Không thấy section '{code}' trong Lab {lab_id}. Các mã có: {available}."


def get_lab_exercises(args: str) -> str:
    """Liệt kê riêng các bài tập của một lab. Tham số: số bài (1/2/3)."""
    lab = loader.get_lab(args)
    if not lab:
        return f"Không tìm thấy lab '{args}'. Các lab có sẵn: 1, 2, 3."
    exercises = lab.get("exercises", [])
    if not exercises:
        return f"{lab['title']}\nChưa có bài tập nào trong knowledge base."
    body = "\n".join(f"  - {exercise}" for exercise in exercises)
    return f"{lab['title']}\nBài tập:\n{body}"


def get_exercise_guide(args: str) -> str:
    """
    Trả về hướng dẫn các phần và bài tập của một lab.
    Tham số: số bài (1/2/3), có thể kèm từ khóa chủ đề, vd '2 rfid' hoặc '1 led'.
    """
    parts = args.strip().split(maxsplit=1)
    lab_id = parts[0] if parts else ""
    topic = parts[1] if len(parts) > 1 else ""

    lab = loader.get_lab(lab_id)
    if not lab:
        return f"Không tìm thấy lab '{lab_id}'. Các lab có sẵn: 1, 2, 3."

    sections = lab.get("sections", [])
    if topic:
        t = loader._strip_accents(topic)
        sections = [
            s for s in sections
            if t in loader._strip_accents(s["title"] + " " + s["guide"])
        ] or lab.get("sections", [])

    sec_text = "\n".join(f"  [{s['code']}] {s['title']}: {s['guide']}" for s in sections)
    ex_text = "\n".join(f"  - {e}" for e in lab.get("exercises", []))
    return f"{lab['title']}\nCác phần hướng dẫn:\n{sec_text}\n\nBài tập:\n{ex_text}"


def search_lab_docs(args: str) -> str:
    """Tìm kiếm trong tài liệu 3 lab theo từ khóa (không phân biệt dấu). Tham số: từ khóa."""
    hits = loader.search(args, max_results=5)
    if not hits:
        return f"Không tìm thấy nội dung nào khớp với '{args}'."
    out = [f"Kết quả cho '{args}':"]
    for h in hits:
        out.append(f"- (Lab {h['lab_id']} | {h['section']}) {h['snippet']}")
    return "\n".join(out)


def lookup_pin_mapping(args: str) -> str:
    """
    Tra cứu sơ đồ chân/ghép nối của một linh kiện hoặc một lab.
    Tham số: số bài (1/2/3) hoặc tên linh kiện (vd 'rc522', 'led', 'hs0038').
    """
    arg = args.strip()
    lab = loader.get_lab(arg)
    if lab:
        pins = lab.get("pin_mappings", {})
        body = "\n".join(f"  - {k}: {v}" for k, v in pins.items())
        return f"Sơ đồ chân {lab['title']}:\n{body}"

    # Otherwise treat as a component keyword and search all labs' pin mappings.
    key = loader._strip_accents(arg)
    out = []
    for lab_id, lab in loader.load()["labs"].items():
        for comp, pins in lab.get("pin_mappings", {}).items():
            if key in loader._strip_accents(comp + " " + pins):
                out.append(f"- (Lab {lab_id}) {comp}: {pins}")
    if not out:
        return f"Không tìm thấy sơ đồ chân cho '{arg}'. Thử: 1/2/3, rc522, led, hs0038, ds1307..."
    return "Sơ đồ chân tìm được:\n" + "\n".join(out)
