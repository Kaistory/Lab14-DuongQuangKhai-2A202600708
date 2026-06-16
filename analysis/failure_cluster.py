"""
Failure Clustering — phân cụm các case thất bại từ reports/benchmark_results.json.

Đây là CÔNG CỤ THẬT (thuần Python, không gọi LLM), dùng lại được cho cả khi chạy
benchmark thật lẫn bản mô phỏng. Nó nhóm các case "cần chú ý" theo:
  - tầng lỗi   : RETRIEVAL (truy hồi sai/thiếu) vs GENERATION (truy hồi đúng nhưng
                 trả lời kém) — phân biệt này quyết định nên sửa pipeline hay prompt.
  - loại case  : metadata.type (wrong_premise, reasoning, out_of_scope, ...)
  - lab        : để thấy lab nào yếu.

Ngưỡng: case có judge.final_score <= FAIL_THRESHOLD bị coi là "cần chú ý".

Chạy:
    python analysis/failure_cluster.py
    python analysis/failure_cluster.py reports/benchmark_results.json
"""
import json
import os
import sys
from collections import Counter, defaultdict

FAIL_THRESHOLD = 3.0


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify_layer(r):
    """Lỗi nằm ở tầng nào? Truy hồi đúng mà điểm thấp -> GENERATION; ngược lại RETRIEVAL."""
    retr = r.get("retrieval", {})
    if not retr.get("applicable", True):
        return "GENERATION"  # adversarial: không có gì để truy hồi, lỗi là ở cách trả lời
    return "GENERATION" if retr.get("hit_rate", 0) >= 1.0 else "RETRIEVAL"


def cluster(results):
    attention = [
        r for r in results
        if r.get("status") == "error"
        or r.get("judge", {}).get("final_score", 5) <= FAIL_THRESHOLD
    ]

    by_layer = Counter()
    by_type = Counter()
    by_lab = Counter()
    groups = defaultdict(list)

    for r in attention:
        if r.get("status") == "error":
            by_layer["ERROR"] += 1
            groups["ERROR (case lỗi runtime)"].append(r)
            continue
        layer = classify_layer(r)
        ctype = r.get("case_metadata", {}).get("type", "unknown")
        lab = r.get("case_metadata", {}).get("lab", "?")
        by_layer[layer] += 1
        by_type[ctype] += 1
        by_lab[f"Lab {lab}"] += 1
        groups[f"{layer} · {ctype}"].append(r)

    return attention, by_layer, by_type, by_lab, groups


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/benchmark_results.json"
    if not os.path.exists(path):
        print(f"❌ Không thấy {path}. Hãy chạy benchmark trước (main.py hoặc simulate_benchmark.py).")
        sys.exit(1)

    results = load_results(path)
    total = len(results)
    attention, by_layer, by_type, by_lab, groups = cluster(results)

    print(f"📊 Phân cụm lỗi từ {path}")
    print(f"   Tổng case: {total} | Cần chú ý (score <= {FAIL_THRESHOLD} hoặc error): {len(attention)}")
    print(f"\n   Theo TẦNG lỗi   : {dict(by_layer)}")
    print(f"   Theo LOẠI case  : {dict(by_type)}")
    print(f"   Theo LAB        : {dict(by_lab)}")

    print("\n   --- Các cụm lỗi (theo điểm Judge) ---")
    for name, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"\n   ▸ [{len(items)}] {name}")
        for r in items:
            j = r.get("judge", {})
            retr = r.get("retrieval", {})
            print(f"       - {r.get('id')}: score {j.get('final_score','?')} "
                  f"(hit={retr.get('hit_rate','-')}, mrr={retr.get('mrr','-')}, "
                  f"conflict={j.get('conflict','-')})")

    # ---- Điểm yếu RETRIEVAL (độc lập điểm Judge): nhiều câu trả lời ĐÚNG vẫn
    #      che giấu truy hồi kém (target chunk ngoài top-k / xếp hạng thấp). ----
    retr_weak = [
        r for r in results
        if r.get("retrieval", {}).get("applicable")
        and (r["retrieval"]["hit_rate"] < 1.0 or r["retrieval"]["mrr"] < 0.5)
    ]
    print(f"\n   --- Điểm yếu RETRIEVAL (hit@k miss hoặc MRR<0.5): {len(retr_weak)} case ---")
    for r in sorted(retr_weak, key=lambda x: x["retrieval"]["mrr"]):
        retr = r["retrieval"]
        print(f"       - {r.get('id')}: hit={retr['hit_rate']}, mrr={round(retr['mrr'],2)} "
              f"(judge score {r.get('judge', {}).get('final_score','?')})")

    # Ghi ra JSON để báo cáo / bước sau dùng.
    out = {
        "total": total,
        "attention_count": len(attention),
        "by_layer": dict(by_layer),
        "by_type": dict(by_type),
        "by_lab": dict(by_lab),
        "clusters": {name: [r.get("id") for r in items] for name, items in groups.items()},
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/failure_clusters.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n💾 Đã ghi reports/failure_clusters.json")


if __name__ == "__main__":
    main()
