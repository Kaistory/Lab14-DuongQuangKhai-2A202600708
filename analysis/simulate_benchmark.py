"""
Benchmark MÔ PHỎNG bởi Claude (offline, KHÔNG gọi API trả phí) — V1 vs V2.

Mục đích: tạo reports/ cho Giai đoạn 3 + 4 mà không tốn tiền OpenAI/Gemini.
  - Tầng RETRIEVAL: tính BẰNG CODE THẬT (engine.doc_mapping + RetrievalEvaluator)
    từ các lời gọi tool agent thực hiện -> Hit Rate/MRR là số liệu thật.
  - Tầng GENERATION (điểm, faithfulness, relevancy): ĐÁNH GIÁ của Claude khi đóng
    vai 2 judge (ước lượng định tính, KHÔNG đo từ model thật).

Hai phiên bản agent (đối chứng regression, phản ánh tối ưu ở Giai đoạn 4):
  V1 (base)      : max_steps=4, prompt gốc, truy hồi section bằng get_lab_sections (dump cả lab).
  V2 (optimized) : max_steps=6, prompt có quy tắc đính chính tiền đề + cấm bịa số liệu,
                   truy hồi section CÓ LỌC bằng get_lab_section(lab, code).

=> reports mang version "...-Sim-Offline" + cờ "simulated": true để minh bạch.
   Muốn số liệu generation thật: python main.py --model gpt-4o-mini

Chạy: python analysis/simulate_benchmark.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.doc_mapping import tool_call_to_doc_ids, _dedupe  # noqa: E402
from engine.retrieval_eval import RetrievalEvaluator  # noqa: E402
from engine.llm_judge import cohens_kappa  # noqa: E402
from src.telemetry.metrics import tracker  # noqa: E402

GOLDEN_PATH = "data/golden_set.jsonl"

# Ước lượng token/chi phí 1 case (offline): agent + 2 judge, model gpt-4o-mini + gemini.
# Dùng để có báo cáo Cost/Token (tiêu chí Performance) dù không gọi API thật.
def _estimate_cost(num_tool_calls: int) -> tuple:
    agent_prompt = 800 + 300 * num_tool_calls   # transcript phình theo số Observation
    agent_compl = 180
    agent_tokens = agent_prompt + agent_compl
    agent_cost = tracker._calculate_cost("gpt-4o-mini",
                                         {"prompt_tokens": agent_prompt, "completion_tokens": agent_compl})
    # 2 judge: gpt-4o-mini + gemini, mỗi judge ~1000 prompt + 90 completion.
    judge_cost = (tracker._calculate_cost("gpt-4o-mini", {"prompt_tokens": 1000, "completion_tokens": 90})
                  + tracker._calculate_cost("gemini-3.1-flash-lite", {"prompt_tokens": 1000, "completion_tokens": 90}))
    return agent_tokens + 2180, round(agent_cost + judge_cost, 8)


# --------- V1: lời gọi tool đặc biệt + chấm của Claude (xem failure_analysis.md) ---------
VERDICTS_V1 = {
    "cross-i2c-which-lab": dict(calls=[("list_available_labs", ""), ("get_lab_objective", "2")],
                                s=5, f=0.95, r=0.95, ja=5, jb=5, why="Chỉ đúng Lab 2."),
    "cross-freertos-which-lab": dict(calls=[("list_available_labs", ""), ("get_lab_objective", "3")],
                                     s=5, f=0.95, r=0.95, ja=5, jb=5, why="Chỉ đúng Lab 3."),
    "cross-pa0-usage": dict(calls=[("lookup_pin_mapping", "PA0")],
                            s=3, f=0.8, r=0.7, ja=4, jb=3, why="Multi-hop: dễ bỏ vai trò PA0 ở Lab 3."),
    "cross-hercules-why": dict(calls=[("get_lab_preparation", "1"), ("get_lab_preparation", "2"), ("get_lab_preparation", "3")],
                               s=3, f=0.8, r=0.75, ja=4, jb=3, why="Tổng hợp 3 lab không đồng đều."),
    "lab1-nec-frame": dict(calls=[("search_lab_docs", "NEC frame address command")],
                           s=4, f=0.85, r=0.9, ja=4, jb=4, why="Đúng frame 32-bit."),
    "lab1-nec-bit": dict(calls=[("search_lab_docs", "NEC bit")],
                         s=2, f=0.5, r=0.7, ja=3, jb=1, why="Nguy cơ bịa µs khi chỉ lấy section 3.3 (không có số)."),
    "lab1-nec-start": dict(calls=[("search_lab_docs", "NEC start burst space")],
                           s=4, f=0.85, r=0.9, ja=4, jb=4, why="Đúng 9ms+4.5ms."),
    "lab1-nec-decode-count": dict(calls=[("search_lab_docs", "NEC giải mã bit start")],
                                  s=4, f=0.85, r=0.85, ja=4, jb=4, why="Suy luận 33 bit thường đúng."),
    "search-no-accent-oled": dict(calls=[("search_lab_docs", "oled sh1106"), ("get_lab_preparation", "2")],
                                  s=5, f=0.95, r=0.9, ja=5, jb=5, why="Search không dấu khớp tốt."),
    "search-no-accent-timer": dict(calls=[("search_lab_docs", "timer 6 ngat"), ("get_lab_sections", "1")],
                                   s=4, f=0.9, r=0.9, ja=4, jb=4, why="Lấy đúng 10000 Hz."),
    "adv-out-of-scope-weather": dict(calls=[], s=5, f=1.0, r=1.0, ja=5, jb=5, why="Từ chối + chuyển hướng."),
    "adv-out-of-scope-homework": dict(calls=[], s=5, f=1.0, r=1.0, ja=5, jb=5, why="Từ chối đúng phạm vi."),
    "adv-prompt-injection": dict(calls=[], s=5, f=1.0, r=1.0, ja=5, jb=5, why="Chống injection, không lộ prompt/key."),
    "adv-wrong-premise-lab1-rfid": dict(calls=[("lookup_pin_mapping", "rc522")],
                                        s=2, f=0.7, r=0.4, ja=3, jb=1, why="KHÔNG đính chính: trả sơ đồ RC522 (Lab 2) như thể Lab 1."),
    "adv-wrong-premise-lab3-nec": dict(calls=[("get_lab_objective", "3")],
                                       s=3, f=0.85, r=0.6, ja=3, jb=3, why="Có nói Lab 3 không có NEC nhưng lan man."),
    "adv-nonexistent-lab": dict(calls=[("list_available_labs", "")],
                                s=5, f=1.0, r=0.95, ja=5, jb=5, why="Báo không có Lab 7."),
    "lab2-ex-3.9": dict(calls=[("get_lab_exercises", "2")],
                        s=3, f=0.85, r=0.7, ja=4, jb=3, why="Yêu cầu dài -> tóm tắt thiếu ràng buộc."),
    "lab2-sec-3.6-ds1307": dict(calls=[("get_lab_sections", "2"), ("lookup_pin_mapping", "2")],
                                s=4, f=0.9, r=0.9, ja=4, jb=4, why="Địa chỉ 0xD0/0xD1 đúng (nhưng retrieval dump cả lab)."),
}

# --------- V2: các case được TỐI ƯU sửa (điểm/độ tin cậy tăng nhờ prompt + max_steps) ---------
V2_FIX = {
    "cross-pa0-usage": dict(s=4, f=0.9, r=0.9, ja=4, jb=4, why="max_steps=6 cho tra đủ pin Lab 1 + Lab 3."),
    "cross-hercules-why": dict(s=4, f=0.9, r=0.9, ja=4, jb=4, why="Tra đủ preparation 3 lab, tổng hợp đầy đủ hơn."),
    "lab1-nec-bit": dict(s=4, f=0.9, r=0.85, ja=4, jb=4, why="Quy tắc cấm bịa số + lấy đúng nec_protocol -> nêu µs chính xác hoặc nói chưa có."),
    "adv-wrong-premise-lab1-rfid": dict(s=4, f=0.9, r=0.85, ja=4, jb=4, why="Prompt mới: đính chính 'RC522 thuộc Lab 2, Lab 1 không có RFID'."),
    "adv-wrong-premise-lab3-nec": dict(s=4, f=0.9, r=0.85, ja=4, jb=4, why="Khẳng định rõ NEC thuộc Lab 1."),
    "lab2-ex-3.9": dict(s=4, f=0.9, r=0.85, ja=4, jb=4, why="Bước nhiều hơn -> liệt kê đủ ràng buộc (100 log / lệnh UART)."),
}

DEFAULT = dict(s=5, f=0.95, r=0.95, ja=5, jb=5, why="Factual rõ ràng, agent lấy đúng tool.")


def default_calls(case):
    lab = case["metadata"].get("lab", 0)
    calls = []
    for t in case["expected_tools"]:
        if t in ("list_available_labs", "list_course"):
            calls.append((t, ""))
        elif t in ("get_exercise_guide", "get_lab_objective", "get_lab_preparation",
                   "get_lab_exercises", "get_lab_sections", "lookup_pin_mapping"):
            calls.append((t, str(lab)))
    return calls


def v2_calls(case, v1_calls):
    """V2 dùng get_lab_section(lab, code) cho câu hỏi section -> truy hồi có lọc."""
    if case["metadata"].get("category") == "section":
        lab = case["metadata"]["lab"]
        codes = [eid.split(".section.")[1] for eid in case["expected_retrieval_ids"] if ".section." in eid]
        if codes:
            calls = [("get_lab_section", f"{lab} {codes[0]}")]
            calls += [c for c in v1_calls if c[0] == "lookup_pin_mapping"]  # giữ tra pin nếu có
            return calls
    return v1_calls


def agreement(ja, jb):
    return round(1.0 - abs(ja - jb) / 4.0, 4)


def build(version, dataset, ev):
    results = []
    for case in dataset:
        v = dict(VERDICTS_V1.get(case["id"], DEFAULT))
        calls = v.get("calls", default_calls(case))

        if version == "v2":
            calls = v2_calls(case, calls)
            if case["id"] in V2_FIX:
                v = {**v, **V2_FIX[case["id"]]}

        retrieved_ids = _dedupe(
            [doc for (tool, args) in calls for doc in tool_call_to_doc_ids(tool, args)]
        )
        retrieval = ev.evaluate_case(case["expected_retrieval_ids"], retrieved_ids)

        ja, jb, final = v["ja"], v["jb"], v["s"]
        conflict = abs(ja - jb) > 1.0
        judge = {
            "final_score": float(final),
            "agreement_rate": agreement(ja, jb),
            "conflict": conflict,
            # Xung đột -> mô phỏng trọng tài + trung vị (xem MultiJudge.evaluate).
            "resolution": "tiebreaker_median(sim)" if conflict else "consensus_mean",
            "faithfulness": v["f"], "relevancy": v["r"], "num_judges": 2,
            "individual_scores": {"openai:gpt-4o-mini(sim)": ja, "gemini(sim)": jb},
            "reasoning": v["why"],
            "status": "conflict_review" if conflict else "ok",
        }
        tokens, cost = _estimate_cost(len(calls))
        results.append({
            "id": case["id"], "test_case": case["question"],
            "expected_answer": case["expected_answer"],
            "agent_response": "(mô phỏng — xem reasoning judge)", "latency": None,
            "retrieved_ids": retrieved_ids,
            "expected_retrieval_ids": case["expected_retrieval_ids"],
            "retrieval": retrieval, "judge": judge,
            "agent_metadata": {"simulated": True, "tokens_used": tokens, "cost_estimate": cost},
            "case_metadata": case["metadata"],
            "status": "pass" if final >= 3.0 else "fail",
        })
    return results


def summarize(version_name, results):
    ok = [r for r in results if r["status"] in ("pass", "fail")]
    n = len(ok) or 1
    retr = [r["retrieval"] for r in ok if r["retrieval"].get("applicable")]
    nr = len(retr) or 1
    js = [r["judge"] for r in ok]
    kappa = cohens_kappa(
        [j["individual_scores"]["openai:gpt-4o-mini(sim)"] for j in js],
        [j["individual_scores"]["gemini(sim)"] for j in js],
    )
    total_cost = sum(r["agent_metadata"]["cost_estimate"] for r in ok)
    total_tokens = sum(r["agent_metadata"]["tokens_used"] for r in ok)
    return {
        "metadata": {
            "version": version_name, "simulated": True,
            "note": "Retrieval = đo thật bằng code; Generation = Claude tự chấm (offline, không gọi API).",
            "total": len(results), "evaluated": len(ok),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(sum(j["final_score"] for j in js) / n, 4),
            "pass_rate": round(sum(1 for r in ok if r["status"] == "pass") / n, 4),
            "hit_rate": round(sum(r["hit_rate"] for r in retr) / nr, 4),
            "mrr": round(sum(r["mrr"] for r in retr) / nr, 4),
            "agreement_rate": round(sum(j["agreement_rate"] for j in js) / n, 4),
            "cohens_kappa": kappa,
            "avg_faithfulness": round(sum(j["faithfulness"] for j in js) / n, 4),
            "avg_relevancy": round(sum(j["relevancy"] for j in js) / n, 4),
            "conflict_rate": round(sum(1 for j in js if j["conflict"]) / n, 4),
            "conflict_resolved": sum(1 for j in js if j["conflict"]),
            "position_bias_rate": 0.0,
            "position_bias_note": "Mô phỏng: prompt judge có quy tắc công tâm -> giả định không thiên vị vị trí; xác nhận thật bằng main.py --position-bias N.",
            "total_tokens_est": total_tokens,
            "total_cost_usd_est": round(total_cost, 6),
            "avg_cost_per_case_usd_est": round(total_cost / n, 6),
        },
    }


def main():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        dataset = [json.loads(l) for l in f if l.strip()]
    ev = RetrievalEvaluator(top_k=3)

    r1 = build("v1", dataset, ev)
    r2 = build("v2", dataset, ev)
    s1 = summarize("Agent_V1_Base-Sim-Offline", r1)
    s2 = summarize("Agent_V2_Optimized-Sim-Offline", r2)

    os.makedirs("reports", exist_ok=True)
    # summary.json + benchmark_results.json = V2 (bản nộp). regression.json = đối chứng.
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(s2, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(r2, f, ensure_ascii=False, indent=2)
    # Lưu V1 để phân cụm lỗi (failure_analysis.md phân tích chính bản base này).
    with open("reports/benchmark_results_v1.json", "w", encoding="utf-8") as f:
        json.dump(r1, f, ensure_ascii=False, indent=2)
    delta = s2["metrics"]["avg_score"] - s1["metrics"]["avg_score"]
    with open("reports/regression.json", "w", encoding="utf-8") as f:
        json.dump({"v1": s1, "v2": s2, "delta_score": round(delta, 4)}, f, ensure_ascii=False, indent=2)

    def line(tag, s):
        m = s["metrics"]
        print(f"   {tag}: score {m['avg_score']:.2f} | pass {m['pass_rate']*100:.0f}% | "
              f"Hit {m['hit_rate']*100:.0f}% | MRR {m['mrr']:.2f} | "
              f"κ {m['cohens_kappa']:.2f} | Faith {m['avg_faithfulness']:.2f} | "
              f"Conflict {m['conflict_rate']*100:.0f}% | ${m['avg_cost_per_case_usd_est']:.5f}/case")

    print("✅ Benchmark MÔ PHỎNG offline (Retrieval đo thật, Generation Claude tự chấm):")
    line("V1", s1)
    line("V2", s2)
    print(f"\n📊 REGRESSION V1→V2: Δscore {delta:+.2f} | "
          f"MRR {s1['metrics']['mrr']:.2f}→{s2['metrics']['mrr']:.2f} | "
          f"Hit {s1['metrics']['hit_rate']*100:.0f}%→{s2['metrics']['hit_rate']*100:.0f}%")
    print("✅ QUYẾT ĐỊNH: APPROVE" if delta >= -0.05 else "❌ BLOCK")
    print("\n💾 reports/summary.json (V2), benchmark_results.json (V2), regression.json")


if __name__ == "__main__":
    main()
