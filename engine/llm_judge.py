"""
Multi-Judge Consensus Engine.

Vì sao Multi-Judge: chỉ tin MỘT model giám khảo (vd gpt-4o) là rủi ro — nó có thể
thiên vị hoặc sai một cách hệ thống. Ta dùng ÍT NHẤT 2 model KHÁC HỌ (OpenAI +
Google), mỗi model chấm độc lập, rồi tính:
  - final_score   : điểm đồng thuận (trung bình)
  - agreement_rate: mức đồng thuận giữa các judge (0..1)
  - conflict       : cờ khi các judge lệch > ngưỡng -> cần review

Tối ưu token (quan trọng vì trần TPM thấp): MỖI judge chỉ gọi LLM MỘT lần và trả
về GỘP cả 3 chỉ số (score tổng thể + faithfulness + relevancy) dưới dạng JSON,
thay vì gọi riêng từng tiêu chí.
"""
import asyncio
import json
import os
import re
from statistics import median
from typing import Any, Dict, List, Optional

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def cohens_kappa(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Cohen's Kappa giữa 2 judge trên thang điểm phân loại 1-5 (toàn dataset).

    κ = (po - pe) / (1 - pe), trong đó po = tỉ lệ đồng ý quan sát được, pe = tỉ lệ
    đồng ý KỲ VỌNG do ngẫu nhiên (từ phân phối điểm biên của mỗi judge). κ loại trừ
    phần đồng thuận may rủi -> chặt hơn "agreement thô". κ≥0.6 thường coi là tốt.
    """
    n = len(scores_a)
    if n == 0 or n != len(scores_b):
        return 0.0
    a = [int(round(x)) for x in scores_a]
    b = [int(round(x)) for x in scores_b]
    cats = set(a) | set(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)

_PROMPT_TEMPLATE = """Bạn là GIÁM KHẢO đánh giá câu trả lời của một trợ lý Lab môn Hệ nhúng (IT4210).
Hãy chấm KHÁCH QUAN dựa trên Đáp án chuẩn và Ngữ cảnh truy hồi.

[Câu hỏi]
{question}

[Đáp án chuẩn (ground truth)]
{ground_truth}

[Ngữ cảnh agent đã truy hồi]
{contexts}

[Câu trả lời của agent cần chấm]
{answer}

Chấm theo 3 tiêu chí và CHỈ trả về JSON đúng định dạng (không thêm chữ nào khác):
{{
  "score": <số nguyên 1-5: mức đúng/đầy đủ so với đáp án chuẩn (1=sai hẳn, 5=đúng và đủ)>,
  "faithfulness": <số thực 0-1: câu trả lời có bám ngữ cảnh/đáp án, KHÔNG bịa thông tin>,
  "relevancy": <số thực 0-1: câu trả lời có đúng trọng tâm câu hỏi, không lan man>,
  "reasoning": "<1-2 câu lý do ngắn gọn bằng tiếng Việt>"
}}
Lưu ý: với câu HỎI NGOÀI PHẠM VI hoặc cố tình gài bẫy, "đáp án chuẩn" mô tả hành vi
đúng (từ chối/đính chính) — hãy chấm theo việc agent có hành xử như vậy không."""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Bóc khối JSON đầu tiên trong output LLM (chịu được ```json fences, chữ thừa)."""
    if not text:
        return None
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    for candidate in (cleaned, text):
        m = _JSON_RE.search(candidate)
        if not m:
            continue
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


class JudgeModel:
    """Một giám khảo = một LLMProvider + prompt chấm điểm gộp."""

    def __init__(self, llm: LLMProvider, label: Optional[str] = None):
        self.llm = llm
        self.label = label or llm.model_name

    def _score_sync(self, question: str, answer: str, ground_truth: str, contexts: str) -> Optional[Dict]:
        prompt = _PROMPT_TEMPLATE.format(
            question=question, ground_truth=ground_truth,
            contexts=contexts or "(không có)", answer=answer,
        )
        try:
            result = self.llm.generate(prompt, system_prompt="Bạn là giám khảo đánh giá AI, công tâm và chặt chẽ.")
        except Exception as e:
            short = str(e).splitlines()[0][:160]
            logger.error(f"Judge '{self.label}' lỗi LLM: {short}", exc_info=False)
            return None

        parsed = _extract_json(result.get("content", ""))
        if parsed is None:
            logger.log_event("JUDGE_PARSE_FAIL", {"judge": self.label})
            return None

        return {
            "judge": self.label,
            "score": _clamp(parsed.get("score"), 1, 5, 1),
            "faithfulness": _clamp(parsed.get("faithfulness"), 0, 1, 0.0),
            "relevancy": _clamp(parsed.get("relevancy"), 0, 1, 0.0),
            "reasoning": str(parsed.get("reasoning", ""))[:500],
            "usage": result.get("usage", {}),
        }

    async def score(self, question: str, answer: str, ground_truth: str, contexts: str) -> Optional[Dict]:
        # provider.generate là blocking -> đẩy sang thread cho async runner.
        return await asyncio.to_thread(self._score_sync, question, answer, ground_truth, contexts)

    # ---- So sánh cặp (phục vụ kiểm tra Position Bias) ----
    _COMPARE_PROMPT = """So sánh 2 câu trả lời (FIRST và SECOND) cho cùng một câu hỏi.
Chọn câu nào ĐÚNG/ĐẦY ĐỦ hơn so với đáp án chuẩn. CHỈ trả JSON:
{{"winner": "first" | "second" | "tie", "reason": "<ngắn gọn>"}}

[Câu hỏi] {question}
[Đáp án chuẩn] {ground_truth}
[FIRST] {first}
[SECOND] {second}"""

    def _compare_sync(self, question: str, first: str, second: str, ground_truth: str) -> str:
        prompt = self._COMPARE_PROMPT.format(question=question, ground_truth=ground_truth,
                                             first=first, second=second)
        try:
            res = self.llm.generate(prompt, system_prompt="Bạn là giám khảo so sánh cặp, công tâm.")
            parsed = _extract_json(res.get("content", "")) or {}
            w = str(parsed.get("winner", "tie")).lower().strip()
            return w if w in ("first", "second", "tie") else "tie"
        except Exception:
            return "tie"

    async def compare(self, question: str, first: str, second: str, ground_truth: str = "") -> str:
        return await asyncio.to_thread(self._compare_sync, question, first, second, ground_truth)


class MultiJudge:
    """
    Điều phối nhiều JudgeModel chấm song song rồi tính đồng thuận.

    conflict_threshold: chênh lệch điểm (thang 1-5) vượt ngưỡng này -> đánh cờ conflict.
    """

    def __init__(self, judges: List[JudgeModel], conflict_threshold: float = 1.0,
                 tiebreaker: Optional[JudgeModel] = None):
        if not judges:
            raise ValueError("MultiJudge cần ít nhất 1 judge.")
        self.judges = judges
        self.conflict_threshold = conflict_threshold
        # Judge "trọng tài" gọi thêm KHI có xung đột để phá thế hòa (mặc định: judge đầu).
        self.tiebreaker = tiebreaker or (judges[0] if judges else None)

    @staticmethod
    def _agreement(scores: List[float]) -> float:
        """Đồng thuận chuẩn hóa: dựa trên khoảng cách lớn nhất giữa các điểm (thang 1-5)."""
        if len(scores) < 2:
            return 1.0
        spread = max(scores) - min(scores)
        return round(1.0 - spread / 4.0, 4)  # diff 0 -> 1.0 ; diff 4 (1 vs 5) -> 0.0

    async def evaluate(self, question: str, answer: str, ground_truth: str,
                       contexts: str = "") -> Dict[str, Any]:
        """Chấm 1 case bằng tất cả judge, trả về kết quả đồng thuận đầy đủ."""
        results = await asyncio.gather(
            *[j.score(question, answer, ground_truth, contexts) for j in self.judges]
        )
        valid = [r for r in results if r is not None]

        # Tất cả judge fail (vd rate limit) -> kết quả suy giảm, không làm sập runner.
        if not valid:
            return {
                "final_score": 0.0, "agreement_rate": 0.0, "conflict": True,
                "faithfulness": 0.0, "relevancy": 0.0, "num_judges": 0,
                "individual_scores": {}, "reasoning": "Mọi judge đều lỗi (có thể do rate limit).",
                "status": "all_failed",
            }

        scores = [r["score"] for r in valid]
        conflict = (max(scores) - min(scores)) > self.conflict_threshold

        # ---- Xử lý xung đột TỰ ĐỘNG: gọi trọng tài, lấy TRUNG VỊ 3 điểm ----
        resolution = "consensus_mean"
        if conflict and self.tiebreaker is not None and len(valid) >= 2:
            tb = await self.tiebreaker.score(question, answer, ground_truth, contexts)
            if tb is not None:
                scores.append(tb["score"])
                valid.append({**tb, "judge": f"{tb['judge']}(tiebreaker)"})
                final_score = round(median(scores), 4)   # trung vị bền với điểm lệch
                resolution = f"tiebreaker_median({self.tiebreaker.label})"
            else:
                final_score = round(sum(scores) / len(scores), 4)
        else:
            final_score = round(sum(scores) / len(scores), 4)

        return {
            "final_score": final_score,
            "agreement_rate": self._agreement([r["score"] for r in valid]),
            "conflict": conflict,
            "resolution": resolution,
            "faithfulness": round(sum(r["faithfulness"] for r in valid) / len(valid), 4),
            "relevancy": round(sum(r["relevancy"] for r in valid) / len(valid), 4),
            "num_judges": len(valid),
            "individual_scores": {r["judge"]: r["score"] for r in valid},
            "details": valid,
            "status": "conflict_review" if conflict else "ok",
            "reasoning": " | ".join(f"{r['judge']}: {r['reasoning']}" for r in valid),
        }

    async def check_position_bias(self, question: str, answer_a: str, answer_b: str,
                                  ground_truth: str = "", judge: Optional[JudgeModel] = None) -> Dict:
        """
        Kiểm tra thiên vị vị trí: hỏi judge 2 lần với THỨ TỰ đảo nhau.
        Judge công tâm phải chọn cùng MỘT câu trả lời thật (tức vị trí thắng đổi
        chỗ giữa 2 vòng). Nếu cả 2 vòng đều chọn cùng MỘT VỊ TRÍ (first/first hoặc
        second/second) -> có position bias.
        """
        j = judge or self.judges[0]
        r1 = await j.compare(question, answer_a, answer_b, ground_truth)  # [A, B]
        r2 = await j.compare(question, answer_b, answer_a, ground_truth)  # [B, A]
        biased = (r1 == r2) and r1 in ("first", "second")
        return {"judge": j.label, "round1": r1, "round2": r2, "position_biased": biased}


# ---------------------------------------------------------------------------
# Factory: dựng bộ judge mặc định (OpenAI gpt-4o-mini + Gemini) từ .env.
# ---------------------------------------------------------------------------
def build_default_judges(
    openai_model: str = "gpt-4o-mini",
    gemini_model: Optional[str] = None,
) -> List[JudgeModel]:
    """
    Tạo các judge từ key có trong .env. Bỏ qua provider nào thiếu key (không sập).
    Mặc định: gpt-4o-mini (rẻ, TPM cao) + gemini (khác họ -> consensus thực chất).
    """
    judges: List[JudgeModel] = []

    if os.getenv("OPENAI_API_KEY"):
        from src.core.openai_provider import OpenAIProvider
        judges.append(JudgeModel(
            OpenAIProvider(model_name=openai_model, api_key=os.getenv("OPENAI_API_KEY")),
            label=f"openai:{openai_model}",
        ))

    if os.getenv("GEMINI_API_KEY"):
        from src.core.gemini_provider import GeminiProvider
        gm = gemini_model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        judges.append(JudgeModel(
            GeminiProvider(model_name=gm, api_key=os.getenv("GEMINI_API_KEY")),
            label=f"gemini:{gm}",
        ))

    if not judges:
        raise RuntimeError(
            "Không tạo được judge nào: cần OPENAI_API_KEY và/hoặc GEMINI_API_KEY trong .env."
        )
    if len(judges) == 1:
        logger.error("⚠️ Chỉ có 1 judge — Multi-Judge cần ≥2 model để consensus có ý nghĩa.",
                     exc_info=False)
    return judges
