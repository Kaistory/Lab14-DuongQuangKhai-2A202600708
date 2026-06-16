# Methodology & Technical Depth

> **Ghi chú:** Đây là bài làm **CÁ NHÂN — thực hiện một mình** (Dương Quang Khải, 2A202600708).
> Tất cả module (SDG, Retrieval Eval, Multi-Judge, Regression, Failure Analysis) do một người làm.

Tài liệu này giải thích các khái niệm đánh giá dùng trong hệ thống — phục vụ tiêu chí
**Technical Depth** của rubric.

## 1. MRR (Mean Reciprocal Rank)
Đo **vị trí** của tài liệu đúng trong danh sách truy hồi, không chỉ "có tìm thấy hay không".

- Với một câu hỏi: `RR = 1 / rank` (rank = vị trí 1-indexed của tài liệu đúng đầu tiên). Không thấy → 0.
- MRR = trung bình RR trên toàn bộ câu hỏi.
- Ví dụ thật trong hệ: `lab2-sec-3.7-sh1106-api` — tool `get_lab_sections` trả 8 section, section đích ở hạng 7 → `RR = 1/7 ≈ 0.14`. **Hit Rate@3 = 0** dù tài liệu vẫn có trong kết quả.
- **Khác Hit Rate:** Hit Rate@k chỉ hỏi "đúng có nằm trong top-k?" (0/1); MRR phạt cả việc xếp hạng thấp → nhạy với chất lượng ranking. Code: [engine/retrieval_eval.py](../engine/retrieval_eval.py).

## 2. Cohen's Kappa (độ đồng thuận giữa 2 Judge)
"Agreement thô" (tỉ lệ 2 judge cho cùng điểm) bị thổi phồng bởi **đồng thuận may rủi**: nếu cả 2 judge hay cho điểm 5, chúng trùng nhau nhiều dù không thực sự "đồng thuận".

```
κ = (po − pe) / (1 − pe)
```
- `po` = tỉ lệ đồng ý quan sát được.
- `pe` = tỉ lệ đồng ý **kỳ vọng do ngẫu nhiên**, tính từ phân phối điểm biên của mỗi judge.
- κ = 1: đồng thuận tuyệt đối; κ = 0: chỉ bằng may rủi; κ < 0: tệ hơn ngẫu nhiên.
- Quy ước thường dùng: κ ≥ 0.6 *substantial*, ≥ 0.8 *almost perfect*.
- Kết quả hệ: V1 **κ = 0.73**, V2 **κ = 1.00**. Code: `cohens_kappa()` trong [engine/llm_judge.py](../engine/llm_judge.py).

## 3. Position Bias (thiên vị vị trí của LLM-Judge)
LLM khi so sánh cặp (A vs B) có xu hướng thiên vị câu **đứng trước/sau** thay vì xét nội dung.

- Cách kiểm: hỏi judge 2 lần với **thứ tự đảo nhau** — vòng 1 `[A, B]`, vòng 2 `[B, A]`.
- Judge công tâm: chọn cùng MỘT câu trả lời thật → vị trí thắng phải **đổi chỗ** giữa 2 vòng.
- Nếu cả 2 vòng cùng chọn một **vị trí** (first/first hoặc second/second) → **có position bias**.
- Code: `MultiJudge.check_position_bias()` + `JudgeModel.compare()`. Chạy: `python main.py --position-bias 5`.

## 4. Xử lý xung đột tự động (Conflict Resolution)
Khi 2 judge lệch > 1 điểm (`conflict = True`):
1. Gọi thêm **judge trọng tài** (`tiebreaker`) chấm độc lập.
2. Điểm cuối = **trung vị (median)** của 3 điểm — bền với điểm lệch (outlier) hơn trung bình.
3. Ghi `resolution = "tiebreaker_median(...)"` để truy vết.

Code: `MultiJudge.evaluate()` trong [engine/llm_judge.py](../engine/llm_judge.py).

## 5. Trade-off Chi phí ↔ Chất lượng
| Lựa chọn | Chất lượng | Chi phí / Rate limit |
|---|---|---|
| Agent `gpt-4o` | Cao nhất | Đắt (~$2.5/1M in), trần 30k TPM → **429 khi chạy 51×2** |
| Agent `gpt-4o-mini` | Đủ tốt cho lab này | Rẻ ~16×, TPM cao → chạy mượt |
| Multi-Judge gộp 1 call | — | Mỗi judge 1 lượt gọi (thay vì 1 lượt/tiêu chí) → tiết kiệm ~3× |
| Judge OpenAI **+** Gemini | Consensus thực chất | Tải chia 2 nhà cung cấp → ít đụng trần |

**Quyết định:** dùng `gpt-4o-mini` cho agent + judge gộp call + chia tải 2 provider → giảm >50% chi phí eval mà vẫn giữ chất lượng. Báo cáo cost/token: `reports/summary.json` (`total_cost_usd`, `total_tokens`).

> Đề xuất giảm thêm 30% chi phí: cache câu trả lời agent giữa V1/V2 cho các case không đổi; chỉ chạy Multi-Judge đầy đủ trên case có `conflict`, các case còn lại dùng 1 judge.
