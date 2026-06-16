"""
Agent thật cho hệ thống Eval Factory (Lab 14).

Thay cho agent giả lập cũ, file này bọc `ReActAgent` (trợ lý Lab Hệ nhúng IT4210
đã phát triển ở Day-3) lại sau interface mà BenchmarkRunner mong đợi:

    agent = MainAgent()
    resp = await agent.query("câu hỏi")     # -> {answer, contexts, metadata}

`ReActAgent.run()` là hàm đồng bộ (blocking), nên ta đẩy nó sang thread riêng
bằng `asyncio.to_thread` để Async Runner vẫn chạy song song nhiều case.
"""
import asyncio
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.core.provider_factory import create_provider
from src.agent.agent import ReActAgent
from src.tools import TOOLS
from src.telemetry.metrics import tracker
from engine.doc_mapping import trace_to_retrieved_ids

# Đọc .env (OPENAI_API_KEY, DEFAULT_PROVIDER, ...) một lần khi import module.
load_dotenv()


class MainAgent:
    """
    ReAct agent thật, đóng gói cho pipeline đánh giá.

    - `contexts`: các Observation thu được từ tool trong vòng ReAct — đây chính là
      "bằng chứng truy hồi" để các metric Faithfulness/Retrieval chấm điểm.
    - `metadata`: model, token, chi phí ước lượng và danh sách tool đã gọi (sources).
    """

    def __init__(self, provider: Optional[str] = None, max_steps: int = 6):
        self.provider_name = provider
        self.max_steps = max_steps
        # Khởi tạo sớm để báo lỗi cấu hình ngay (vd thiếu OPENAI_API_KEY) thay vì
        # đợi tới lúc chạy benchmark mới phát hiện.
        self.model_name = create_provider(provider).model_name
        self.name = f"ReActAgent-EmbeddedLab ({self.model_name})"

    async def query(self, question: str) -> Dict:
        """Chạy ReAct loop cho 1 câu hỏi và trả về kết quả chuẩn hóa."""
        return await asyncio.to_thread(self._run_sync, question)

    def _run_sync(self, question: str) -> Dict:
        # Mỗi câu hỏi dùng một provider RIÊNG để cô lập telemetry khi chạy song song
        # (tránh tranh chấp trên tracker toàn cục giữa các thread trong cùng batch).
        llm = create_provider(self.provider_name)

        # Bọc generate() để gom token usage của ĐÚNG câu hỏi này.
        usage_log: List[Dict] = []
        original_generate = llm.generate

        def tracked_generate(prompt, system_prompt=None):
            result = original_generate(prompt, system_prompt=system_prompt)
            usage_log.append(result.get("usage", {}) or {})
            return result

        llm.generate = tracked_generate  # type: ignore[method-assign]

        agent = ReActAgent(llm, TOOLS, max_steps=self.max_steps)

        trace: List[Dict] = []
        answer = agent.run(question, trace=trace)

        # contexts = các Observation tool đã trả về trong vòng suy luận.
        contexts = [
            f"[{step['tool']}({step['args']})] {step['observation']}"
            for step in trace
        ]

        tokens_used = sum(u.get("total_tokens", 0) for u in usage_log)
        cost_estimate = sum(
            tracker._calculate_cost(self.model_name, u) for u in usage_log
        )
        # Tool nào đã được gọi -> coi như "nguồn" trả lời (giữ thứ tự, bỏ trùng).
        sources = list(dict.fromkeys(step["tool"] for step in trace))
        # Doc-id đã truy hồi (suy từ trace) -> đầu vào cho Retrieval Eval (Hit Rate/MRR).
        retrieved_ids = trace_to_retrieved_ids(trace)

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": self.model_name,
                "tokens_used": tokens_used,
                "cost_estimate": round(cost_estimate, 8),
                "llm_calls": len(usage_log),
                "tool_calls": len(trace),
                "sources": sources,
                "retrieved_ids": retrieved_ids,
            },
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        resp = await agent.query("Lab 2 cần chuẩn bị những gì?")
        print(resp["answer"])
        print("---")
        print(resp["metadata"])

    asyncio.run(test())
