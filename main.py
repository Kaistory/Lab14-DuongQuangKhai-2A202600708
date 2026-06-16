"""
AI Evaluation Factory — điểm chạy chính (Giai đoạn 2 + 3).

Luồng: nạp golden set -> chạy Agent V1 và V2 qua Eval Engine thật
(Retrieval Hit Rate/MRR + Multi-Judge Consensus) -> so sánh regression ->
ghi reports/ -> quyết định APPROVE / BLOCK.

Ví dụ chạy:
    python main.py                       # chạy đầy đủ V1 + V2 trên toàn golden set
    python main.py --limit 6             # chỉ 6 case đầu (test nhanh, ít token)
    python main.py --concurrency 1       # giảm song song nếu hay dính 429
    python main.py --model gpt-4o-mini   # ép model agent (rẻ + TPM cao hơn gpt-4o)
    python main.py --only v2             # chỉ chạy 1 phiên bản (tiết kiệm 1 nửa)
"""
import argparse
import asyncio
import json
import os
import sys
import time

# UTF-8 cho console Windows (in tiếng Việt + log).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv

from agent.main_agent import MainAgent
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import MultiJudge, build_default_judges, cohens_kappa

GOLDEN_PATH = "data/golden_set.jsonl"

# Cấu hình 2 phiên bản agent để so regression (V2 = bản "tối ưu": nhiều bước ReAct hơn).
VERSION_CONFIGS = {
    "v1": {"name": "Agent_V1_Base", "max_steps": 4},
    "v2": {"name": "Agent_V2_Optimized", "max_steps": 6},
}


def load_dataset(limit=None):
    if not os.path.exists(GOLDEN_PATH):
        print(f"❌ Thiếu {GOLDEN_PATH}. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if not dataset:
        print(f"❌ {GOLDEN_PATH} rỗng.")
        return None
    return dataset[:limit] if limit else dataset


def summarize(version_name: str, results: list) -> dict:
    """Tổng hợp chỉ số đầy đủ từ kết quả per-case."""
    ok = [r for r in results if r.get("status") in ("pass", "fail")]
    errors = [r for r in results if r.get("status") == "error"]
    n = len(ok) or 1  # tránh chia 0

    # Retrieval: chỉ tính trên case 'applicable' (có ground truth doc-id).
    retr = [r["retrieval"] for r in ok if r.get("retrieval", {}).get("applicable")]
    nr = len(retr) or 1

    judges = [r["judge"] for r in ok]
    total_cost = sum(r.get("agent_metadata", {}).get("cost_estimate", 0) for r in ok)
    total_tokens = sum(r.get("agent_metadata", {}).get("tokens_used", 0) for r in ok)

    # Cohen's Kappa giữa 2 judge nền (bỏ điểm của trọng tài) — đồng thuận loại trừ may rủi.
    base_labels = []
    for j in judges:
        for lbl in j.get("individual_scores", {}):
            if "(tiebreaker)" not in lbl and lbl not in base_labels:
                base_labels.append(lbl)
    kappa = 0.0
    if len(base_labels) >= 2:
        la, lb = base_labels[0], base_labels[1]
        pairs = [(j["individual_scores"].get(la), j["individual_scores"].get(lb)) for j in judges]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if pairs:
            kappa = cohens_kappa([x for x, _ in pairs], [y for _, y in pairs])

    return {
        "metadata": {
            "version": version_name,
            "total": len(results),
            "evaluated": len(ok),
            "errors": len(errors),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(sum(j["final_score"] for j in judges) / n, 4),
            "pass_rate": round(sum(1 for r in ok if r["status"] == "pass") / n, 4),
            "hit_rate": round(sum(r["hit_rate"] for r in retr) / nr, 4),
            "mrr": round(sum(r["mrr"] for r in retr) / nr, 4),
            "agreement_rate": round(sum(j["agreement_rate"] for j in judges) / n, 4),
            "cohens_kappa": kappa,
            "conflict_resolved": sum(1 for j in judges if j.get("conflict")),
            "avg_faithfulness": round(sum(j["faithfulness"] for j in judges) / n, 4),
            "avg_relevancy": round(sum(j["relevancy"] for j in judges) / n, 4),
            "conflict_rate": round(sum(1 for j in judges if j.get("conflict")) / n, 4),
            "avg_latency_s": round(sum(r.get("latency", 0) for r in ok) / n, 3),
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_case_usd": round(total_cost / n, 6),
            "total_tokens": total_tokens,
        },
    }


def print_summary(summary: dict):
    m, meta = summary["metrics"], summary["metadata"]
    print(f"\n  📌 {meta['version']}  ({meta['evaluated']}/{meta['total']} case, {meta['errors']} lỗi)")
    print(f"     Avg Score      : {m['avg_score']:.2f}/5   (pass rate {m['pass_rate']*100:.0f}%)")
    print(f"     Retrieval      : Hit Rate {m['hit_rate']*100:.0f}% | MRR {m['mrr']:.2f}")
    print(f"     Multi-Judge    : Agreement {m['agreement_rate']*100:.0f}% | Cohen's κ {m.get('cohens_kappa',0):.2f} | Conflict {m['conflict_rate']*100:.0f}%")
    print(f"     Generation     : Faithfulness {m['avg_faithfulness']:.2f} | Relevancy {m['avg_relevancy']:.2f}")
    print(f"     Hiệu năng/Chi phí: {m['avg_latency_s']:.2f}s/case | ${m['avg_cost_per_case_usd']:.5f}/case | {m['total_tokens']} tokens")


async def run_version(key: str, dataset: list, judge: MultiJudge, concurrency: int):
    cfg = VERSION_CONFIGS[key]
    print(f"\n🚀 Chạy {cfg['name']} (max_steps={cfg['max_steps']}) trên {len(dataset)} case...")

    agent = MainAgent(max_steps=cfg["max_steps"])
    retrieval = RetrievalEvaluator(top_k=3)

    done_mark = {"last": 0}

    def progress(done, total):
        if done == total or done - done_mark["last"] >= max(1, total // 10):
            done_mark["last"] = done
            print(f"   ... {done}/{total}")

    runner = BenchmarkRunner(agent, retrieval, judge,
                             max_concurrency=concurrency, on_progress=progress)
    results = await runner.run_all(dataset)
    summary = summarize(cfg["name"], results)
    print_summary(summary)
    return results, summary


async def main_async(args):
    load_dotenv()
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model  # ép model agent cho cả phiên chạy

    dataset = load_dataset(args.limit)
    if dataset is None:
        return

    # Multi-Judge: gpt-4o-mini (OpenAI) + gemini (Google) — dựng từ .env.
    judges = build_default_judges()
    judge = MultiJudge(judges)
    print(f"⚖️  Multi-Judge: {', '.join(j.label for j in judges)}")

    versions = ["v1", "v2"] if args.only == "both" else [args.only]
    summaries, all_results = {}, {}
    for key in versions:
        results, summary = await run_version(key, dataset, judge, args.concurrency)
        all_results[key] = results
        summaries[key] = summary

    # ---- Position Bias probe (tùy chọn): judge có thiên vị thứ tự A/B không? ----
    if args.position_bias > 0:
        primary = all_results.get("v2", all_results[versions[-1]])
        sample = [r for r in primary if r.get("agent_response")][:args.position_bias]
        print(f"\n🔍 Kiểm tra Position Bias trên {len(sample)} case...")
        biased = 0
        for r in sample:
            pb = await judge.check_position_bias(
                r["test_case"], r["agent_response"], r["expected_answer"], r["expected_answer"])
            biased += 1 if pb["position_biased"] else 0
            print(f"   - {r['id']}: r1={pb['round1']} r2={pb['round2']} -> "
                  f"{'BIASED' if pb['position_biased'] else 'ok'}")
        rate = biased / len(sample) if sample else 0
        print(f"   Position bias rate: {rate*100:.0f}% ({biased}/{len(sample)})")

    os.makedirs("reports", exist_ok=True)

    # summary.json = phiên bản mới nhất đã chạy (ưu tiên v2). benchmark_results = chi tiết.
    primary_key = "v2" if "v2" in summaries else versions[-1]
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries[primary_key], f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results[primary_key], f, ensure_ascii=False, indent=2)

    # ---- Regression Gate: chỉ khi có cả V1 và V2 ----
    if "v1" in summaries and "v2" in summaries:
        s1, s2 = summaries["v1"]["metrics"], summaries["v2"]["metrics"]
        delta = s2["avg_score"] - s1["avg_score"]
        cost_delta = s2["avg_cost_per_case_usd"] - s1["avg_cost_per_case_usd"]
        print("\n📊 --- REGRESSION (V1 → V2) ---")
        print(f"   Score : {s1['avg_score']:.2f} → {s2['avg_score']:.2f}  (Δ {delta:+.2f})")
        print(f"   HitRate: {s1['hit_rate']*100:.0f}% → {s2['hit_rate']*100:.0f}%")
        print(f"   Cost  : ${s1['avg_cost_per_case_usd']:.5f} → ${s2['avg_cost_per_case_usd']:.5f}/case (Δ ${cost_delta:+.5f})")

        with open("reports/regression.json", "w", encoding="utf-8") as f:
            json.dump({"v1": summaries["v1"], "v2": summaries["v2"],
                       "delta_score": round(delta, 4)}, f, ensure_ascii=False, indent=2)

        # Auto-gate: chấp nhận nếu chất lượng không giảm (cho dung sai nhỏ).
        if delta >= -0.05:
            print("\n✅ QUYẾT ĐỊNH: APPROVE — chất lượng không giảm.")
        else:
            print("\n❌ QUYẾT ĐỊNH: BLOCK RELEASE — chất lượng giảm so với V1.")

    print(f"\n💾 Đã ghi reports/summary.json, reports/benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(description="AI Evaluation Factory — Lab 14")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N case đầu (test nhanh)")
    parser.add_argument("--concurrency", type=int, default=2, help="Số request đồng thời (giảm nếu 429)")
    parser.add_argument("--model", default=None, help="Ép OPENAI_MODEL cho agent, vd gpt-4o-mini")
    parser.add_argument("--only", choices=["both", "v1", "v2"], default="both",
                        help="Chạy cả 2 phiên bản hay chỉ 1 (tiết kiệm token)")
    parser.add_argument("--position-bias", type=int, default=0, dest="position_bias",
                        help="Kiểm tra Position Bias trên N case (0 = bỏ qua)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
