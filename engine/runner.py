"""
Async Benchmark Runner.

Chạy toàn bộ golden set qua: Agent -> Retrieval Eval -> Multi-Judge, có kiểm soát
song song bằng Semaphore để KHÔNG vượt rate limit (TPM) của nhà cung cấp LLM.

Khác bản cũ (gather theo batch cứng): dùng Semaphore giới hạn số request ĐỒNG
THỜI trên toàn tiến trình -> ổn định hơn dưới trần TPM thấp, vẫn tận dụng async.
"""
import asyncio
import time
from typing import Dict, List, Optional

from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import MultiJudge


class BenchmarkRunner:
    def __init__(
        self,
        agent,
        retrieval_evaluator: RetrievalEvaluator,
        judge: MultiJudge,
        max_concurrency: int = 2,
        pass_threshold: float = 3.0,
        on_progress=None,
    ):
        self.agent = agent
        self.retrieval = retrieval_evaluator
        self.judge = judge
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.pass_threshold = pass_threshold
        self.on_progress = on_progress  # callback(done, total) tùy chọn

    async def run_single_test(self, test_case: Dict) -> Dict:
        question = test_case["question"]
        expected_answer = test_case.get("expected_answer", "")
        expected_ids = test_case.get("expected_retrieval_ids", [])

        start = time.perf_counter()
        # 1) Gọi Agent (thật) — có thể nhiều bước ReAct + nhiều lượt LLM.
        response = await self.agent.query(question)
        latency = time.perf_counter() - start

        answer = response.get("answer", "")
        contexts = response.get("contexts", [])
        retrieved_ids = response.get("retrieved_ids", [])

        # 2) Retrieval Eval (không tốn LLM) — Hit Rate / MRR theo ground truth.
        retrieval = self.retrieval.evaluate_case(expected_ids, retrieved_ids)

        # 3) Multi-Judge (LLM) — đồng thuận faithfulness / relevancy / score.
        judge_result = await self.judge.evaluate(
            question=question,
            answer=answer,
            ground_truth=expected_answer,
            contexts="\n".join(contexts),
        )

        status = "pass" if judge_result["final_score"] >= self.pass_threshold else "fail"

        return {
            "id": test_case.get("id"),
            "test_case": question,
            "expected_answer": expected_answer,
            "agent_response": answer,
            "latency": round(latency, 3),
            "retrieved_ids": retrieved_ids,
            "expected_retrieval_ids": expected_ids,
            "retrieval": retrieval,
            "judge": judge_result,
            "agent_metadata": response.get("metadata", {}),
            "case_metadata": test_case.get("metadata", {}),
            "status": status,
        }

    async def _guarded(self, case: Dict, counter: Dict, total: int) -> Dict:
        async with self.semaphore:
            try:
                result = await self.run_single_test(case)
            except Exception as e:  # 1 case lỗi không được làm sập cả benchmark
                result = {
                    "id": case.get("id"),
                    "test_case": case.get("question"),
                    "status": "error",
                    "error": str(e).splitlines()[0][:200],
                }
            counter["done"] += 1
            if self.on_progress:
                self.on_progress(counter["done"], total)
            return result

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        """Chạy toàn bộ dataset song song có giới hạn (Semaphore)."""
        total = len(dataset)
        counter = {"done": 0}
        tasks = [self._guarded(case, counter, total) for case in dataset]
        return await asyncio.gather(*tasks)
