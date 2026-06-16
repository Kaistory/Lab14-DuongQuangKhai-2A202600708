# Reflection cá nhân — Dương Quang Khải (2A202600708)

> **Hình thức:** bài làm **CÁ NHÂN — thực hiện một mình toàn bộ** (không chia nhóm). Một người đảm nhận tất cả các vai trò: Data/SDG, AI-Backend (Multi-Judge), DevOps/Analyst (Regression + Failure Analysis).
> ⚠️ Bản nháp dựng theo công việc thực tế trong repo; hãy chỉnh lại cho đúng trải nghiệm cá nhân trước khi nộp.

## 1. Phạm vi tôi đảm nhận (làm 1 mình)
Vì làm cá nhân nên tôi tự thực hiện **toàn bộ** pipeline thay vì chia theo nhóm:
- **Data/SDG:** Golden Dataset 51 case neo vào knowledge base + Ground Truth IDs.
- **AI-Backend:** tích hợp ReAct Agent thật, Retrieval Eval, Multi-Judge consensus.
- **DevOps/Analyst:** Regression Gate V1↔V2, Failure Clustering, 5 Whys.

## 2. Đóng góp cụ thể (toàn bộ là của cá nhân)
- `agent/main_agent.py` — adapter bọc ReActAgent giữ interface `async query()`.
- `engine/doc_mapping.py` — dịch lời gọi tool → doc-id để tính Hit Rate/MRR thật.
- `engine/retrieval_eval.py`, `engine/llm_judge.py` (Multi-Judge + Cohen's Kappa + tie-breaker + Position Bias), `engine/runner.py` (async + semaphore).
- `data/synthetic_gen.py` (51 case), `analysis/` (clustering, simulate, failure_analysis, methodology).

## 3. Điều tôi học được
- **Điểm cao che giấu retrieval kém:** nhiều câu judge chấm 5/5 nhưng MRR chỉ 0.14 (tool `get_lab_sections` dump cả lab). Phải đo retrieval riêng mới phát hiện.
- **Cohen's Kappa > agreement thô:** loại trừ đồng thuận may rủi; V1 κ=0.73, V2 κ=1.00.
- **Position Bias:** judge LLM có thể thiên vị thứ tự A/B — phải test đảo chỗ.
- **Trade-off chi phí/chất lượng:** gpt-4o trần 30k TPM gây 429; chuyển gpt-4o-mini + judge gộp call + chia tải 2 provider giảm >50% chi phí.

## 4. Khó khăn & cách vượt qua
- **Agent không trả doc-id** để so ground truth → viết `doc_mapping` tái dựng từ KB.
- **Rate limit 429 làm sai lệch kết quả** → Semaphore + gpt-4o-mini + retry/backoff.
- **Chi tiết NEC bị bịa** (timing chỉ ở field `nec_protocol` không được index) → index nó vào search + thêm quy tắc cấm bịa số liệu trong prompt.

## 5. Nếu làm lại, tôi sẽ
- Thêm **reranking** cho `search_lab_docs` (ưu tiên chunk chứa số liệu) để diệt hallucination từ gốc.
- Cache câu trả lời agent giữa V1/V2 và chỉ chạy full Multi-Judge trên case có conflict để giảm thêm chi phí.
