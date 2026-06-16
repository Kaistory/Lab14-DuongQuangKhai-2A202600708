"""
Retrieval Evaluator — đo chất lượng tầng TRUY HỒI (Retrieval) bằng Hit Rate & MRR.

Vì sao cần: trước khi chấm câu trả lời (Generation), phải biết agent có lấy đúng
tài liệu không. Hallucination thường bắt nguồn từ retrieval sai/thiếu. Toàn bộ
phần này KHÔNG gọi LLM -> nhanh, rẻ, tất định.

Đầu vào mỗi case:
  expected_ids   : ground truth doc-id (từ golden set, trường expected_retrieval_ids)
  retrieved_ids  : doc-id agent đã truy hồi (từ response['retrieved_ids'],
                   do engine.doc_mapping suy ra từ trace ReAct)
"""
from typing import Dict, List


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str],
                           top_k: int = None) -> float:
        """1.0 nếu có ÍT NHẤT 1 expected_id nằm trong top_k retrieved, ngược lại 0.0."""
        k = top_k or self.top_k
        top_retrieved = retrieved_ids[:k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """Mean Reciprocal Rank: 1/(vị trí đầu tiên 1-indexed của 1 expected_id), 0 nếu không thấy."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_case(self, expected_ids: List[str], retrieved_ids: List[str]) -> Dict:
        """
        Chấm retrieval cho MỘT case.

        Quy ước case adversarial / ngoài phạm vi (expected_ids rỗng): agent ĐÚNG khi
        KHÔNG truy hồi tài liệu nội bộ nào. Khi đó hit_rate/mrr = 1.0 nếu retrieved
        cũng rỗng, ngược lại 0.0 (đã truy hồi nhầm). `applicable=False` để batch loại
        các case này khỏi trung bình retrieval thuần (tránh làm nhiễu chỉ số).
        """
        if not expected_ids:
            clean = (len(retrieved_ids) == 0)
            return {
                "hit_rate": 1.0 if clean else 0.0,
                "mrr": 1.0 if clean else 0.0,
                "applicable": False,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            }
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            "applicable": True,
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
        }

    def aggregate(self, per_case: List[Dict]) -> Dict:
        """Tổng hợp chỉ số retrieval, chỉ tính trên các case 'applicable' (có ground truth)."""
        applicable = [c for c in per_case if c.get("applicable")]
        n = len(applicable)
        if n == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "evaluated": 0, "total": len(per_case)}
        return {
            "avg_hit_rate": sum(c["hit_rate"] for c in applicable) / n,
            "avg_mrr": sum(c["mrr"] for c in applicable) / n,
            "evaluated": n,
            "total": len(per_case),
        }
