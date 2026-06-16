"""
Knowledge base loader for the Embedded Systems (IT4210) lab assistant.

Reads data/embedded_labs.json (extracted from the 3 lab PDFs in docs/) and
provides simple lookup + diacritics-insensitive full-text search used by the
agent tools.
"""
import json
import os
import unicodedata
from typing import Any, Dict, List, Optional

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "embedded_labs.json",
)

_CACHE: Optional[Dict[str, Any]] = None


def _strip_accents(text: str) -> str:
    """Lowercase + remove Vietnamese diacritics so search is forgiving."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def load() -> Dict[str, Any]:
    """Load (and cache) the knowledge base JSON."""
    global _CACHE
    if _CACHE is None:
        with open(_DATA_PATH, encoding="utf-8") as f:
            _CACHE = json.load(f)
    return _CACHE


def get_lab(lab_id: Any) -> Optional[Dict[str, Any]]:
    """Return the lab dict for id 1/2/3, or None if not found."""
    labs = load()["labs"]
    return labs.get(str(lab_id).strip())


def list_labs() -> List[Dict[str, str]]:
    """Return a short list of all labs (id + title)."""
    return [{"id": k, "title": v["title"]} for k, v in load()["labs"].items()]


def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Diacritics-insensitive keyword search across all labs.
    Returns a list of {lab_id, title, section, snippet} matches.
    """
    q = _strip_accents(query)
    terms = [t for t in q.split() if t]
    results: List[Dict[str, str]] = []

    for lab_id, lab in load()["labs"].items():
        # Build searchable (section_label, text) chunks for this lab.
        chunks: List[tuple] = []
        for line in lab.get("objective", []):
            chunks.append(("Mục đích", line))
        for sec in lab.get("sections", []):
            chunks.append((f"{sec['code']} {sec['title']}", sec["guide"]))
        for ex in lab.get("exercises", []):
            chunks.append(("Bài tập", ex))
        for comp, pins in lab.get("pin_mappings", {}).items():
            chunks.append(("Sơ đồ chân", f"{comp}: {pins}"))
        # Chi tiết chuẩn NEC (timing, frame...) — index để câu hỏi về hồng ngoại
        # truy hồi được dữ liệu thật thay vì để agent tự suy đoán.
        for key, val in lab.get("nec_protocol", {}).items():
            chunks.append(("NEC protocol", f"{key}: {val}"))

        for label, text in chunks:
            haystack = _strip_accents(label + " " + text)
            score = sum(1 for t in terms if t in haystack)
            if score:
                results.append(
                    {
                        "lab_id": lab_id,
                        "title": lab["title"],
                        "section": label,
                        "snippet": text,
                        "_score": score,
                    }
                )

    results.sort(key=lambda r: r["_score"], reverse=True)
    for r in results:
        r.pop("_score", None)
    return results[:max_results]
