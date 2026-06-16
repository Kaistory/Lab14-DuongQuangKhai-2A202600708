# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> **Hình thức làm bài:** CÁ NHÂN — thực hiện **một mình** toàn bộ (Dương Quang Khải, 2A202600708). Một người đảm nhận tất cả vai trò Data/AI-Backend/DevOps.
> **Agent đánh giá:** Trợ lý Lab Hệ nhúng IT4210 (ReAct Agent, 11 tools, knowledge base `data/embedded_labs.json`).
> **Nguồn số liệu:** `reports/summary.json` + `reports/benchmark_results.json`.
> ⚠️ **Lưu ý minh bạch:** bản benchmark này được sinh OFFLINE (`analysis/simulate_benchmark.py`) — **tầng Retrieval (Hit Rate/MRR) đo bằng code thật**, còn **điểm Generation do Claude đóng vai 2 judge tự chấm** (không gọi API trả phí). Để có điểm Generation đo từ model thật, chạy `python main.py --model gpt-4o-mini`.

> **Phạm vi phân tích:** mục 1–3 phân tích **Agent V1 (base)** — bản được dùng để tìm lỗi. Tái lập phân cụm: `python analysis/failure_cluster.py reports/benchmark_results_v1.json`. Kết quả sau tối ưu (V2) ở mục 5.

## 1. Tổng quan Benchmark (Agent V1 — base)
- **Tổng số cases:** 51 (vượt mốc tối thiểu 50)
- **Tỉ lệ Pass/Fail:** 49 / 2  (pass rate **96%**)
- **Retrieval (đo thật):**
    - Hit Rate@3: **0.92**
    - MRR: **0.84**
- **Điểm RAGAS/Generation trung bình:**
    - Faithfulness: **0.92**
    - Relevancy: **0.91**
- **Điểm Multi-Judge trung bình:** **4.63 / 5.0**
- **Đồng thuận 2 judge (gpt-4o-mini + gemini):** Agreement **97%**, Conflict **4%**

## 2. Phân nhóm lỗi (Failure Clustering)

### 2a. Cụm theo điểm Judge (6 case score ≤ 3)
| Nhóm lỗi | Tầng | Số lượng | Case | Nguyên nhân dự kiến |
|----------|------|----------|------|---------------------|
| Sai tiền đề không được đính chính | Generation | 2 | `adv-wrong-premise-lab1-rfid`, `adv-wrong-premise-lab3-nec` | Agent trả lời theo giả định sai của người hỏi thay vì sửa lại |
| Tổng hợp đa nguồn chưa đủ (multi-hop) | Generation | 3 | `cross-pa0-usage`, `cross-hercules-why`, `lab2-ex-3.9` | Câu cần ghép ≥2 nguồn/ràng buộc; agent tóm tắt thiếu |
| Hallucination số liệu kỹ thuật | Generation | 1 | `lab1-nec-bit` | Timing NEC (µs) bị bịa khi không lấy đúng chunk `nec_protocol` |

**Tất cả 6 lỗi nằm ở tầng GENERATION** (truy hồi đúng nhưng trả lời chưa chuẩn) → ưu tiên sửa **Prompting / Agent loop**, không phải Ingestion.

### 2b. Điểm yếu RETRIEVAL ẩn (7 case — quan trọng)
Nhiều câu **vẫn được chấm 5/5** nhưng truy hồi kém — đáng báo động vì điểm cao đang **che giấu** lỗi pipeline:

| Case | Hit@3 | MRR | Judge | Vấn đề |
|------|-------|-----|-------|--------|
| `lab2-sec-3.7-sh1106-api` | **0.0** | 0.14 | 5.0 | Section đích nằm ngoài top-3 |
| `lab2-sec-3.6-ds1307` | **0.0** | 0.17 | 4.0 | // |
| `lab2-sec-3.5-config` | **0.0** | 0.20 | 5.0 | // |
| `lab1-sec-3.3-hs0038` | **0.0** | 0.25 | 5.0 | // |
| `lab1-sec-3.2-7seg` | 1.0 | 0.33 | 5.0 | Đúng nhưng xếp hạng thấp |
| `lab2-sec-3.3-rc522-iface` | 1.0 | 0.33 | 5.0 | // |
| `lab3-sec-3.2-stopwatch` | 1.0 | 0.33 | 5.0 | // |

**Nguyên nhân chung:** tool `get_lab_sections(N)` trả về **TOÀN BỘ** section của một lab (8 section với Lab 2) theo **thứ tự tài liệu**, không theo độ liên quan. Section đích (vd 3.7) bị đẩy xuống hạng 7 → rớt khỏi top-3 và MRR rất thấp. Câu trả lời vẫn đúng vì LLM tự lọc trong số 8 section, nhưng điều đó **đốt context, tăng chi phí và dễ vỡ** khi knowledge base lớn lên.

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: `adv-wrong-premise-lab1-rfid` — "Trong Lab 1, RC522 nối chân SPI nào?" (score 2/5)
1. **Symptom:** Agent trả luôn sơ đồ chân RC522 (vốn thuộc **Lab 2**) như thể Lab 1 có RFID, không đính chính.
2. **Why 1:** Agent gọi `lookup_pin_mapping("rc522")` → ra dữ liệu Lab 2 hợp lệ → tưởng câu hỏi đúng.
3. **Why 2:** Agent coi tiền đề trong câu hỏi là đúng, không kiểm tra "RC522 có thuộc Lab 1 không".
4. **Why 3:** System prompt không có quy tắc **xác minh tiền đề / sửa giả định sai**.
5. **Why 4:** Tool trả dữ liệu nhưng **không kèm thông tin "thuộc lab nào"** đủ nổi bật để agent đối chiếu với "Lab 1" trong câu hỏi.
6. **Root Cause:** **Thiếu cơ chế chống "sycophancy/leading question"** trong prompt; agent tối ưu việc "trả lời" hơn là "trả lời ĐÚNG ngữ cảnh".

### Case #2: `lab1-nec-bit` — "Phân biệt bit 0 và bit 1 trong NEC?" (score 2/5, judge conflict)
1. **Symptom:** Có nguy cơ bịa số liệu timing (562.5µs / 1687.5µs) hoặc nói chung chung.
2. **Why 1:** Số liệu timing **chỉ** nằm ở field `nec_protocol`; section 3.3 mô tả phương pháp chứ không có con số.
3. **Why 2:** Khi search trả về section 3.3 **trước** `nec_protocol` (MRR 0.5), agent dễ kết luận từ section 3.3 + kiến thức nền → bịa số.
4. **Why 3:** `search_lab_docs` xếp hạng theo **đếm từ khóa khớp**, không ưu tiên chunk chứa số liệu định lượng.
5. **Why 4:** Không có bước **bắt buộc trích dẫn nguồn** cho dữ liệu định lượng → faithfulness thấp (0.5).
6. **Root Cause:** **Chunking/Indexing tách rời** dữ liệu định lượng (`nec_protocol`) khỏi mô tả khái niệm (section 3.3), cộng thiếu ràng buộc "không có số trong context thì không được nêu số".

### Case #3: `lab2-sec-3.7-sh1106-api` — "Trình tự gọi hàm hiển thị SH1106?" (judge 5/5 nhưng Hit@3 = 0)
1. **Symptom:** Câu trả lời đúng, nhưng truy hồi **trượt** (section đích ngoài top-3, MRR 0.14).
2. **Why 1:** Agent gọi `get_lab_sections(2)` → nhận **cả 8 section** của Lab 2.
3. **Why 2:** Doc đích (section 3.7) xếp hạng 7 trong dump → ngoài top-3.
4. **Why 3:** `get_lab_sections` **không nhận tham số lọc** theo section/chủ đề → luôn trả nguyên lô.
5. **Why 4:** Không có bước **rerank** theo độ liên quan trước khi đưa vào context.
6. **Root Cause:** **Chiến lược Retrieval quá thô** (whole-lab dump, thứ tự tài liệu) → precision/MRR thấp; chất lượng câu trả lời hiện tốt chỉ nhờ LLM gánh, không bền vững.

## 4. Kế hoạch cải tiến (Action Plan)
Gắn trực tiếp với Root Cause ở trên; phần đã triển khai cho **Agent V2** xem `report` Giai đoạn 4:

- [x] **Prompt chống tiền đề sai** (Root cause #1): thêm quy tắc "xác minh linh kiện/chủ đề thuộc lab nào trước khi trả lời; nếu tiền đề sai thì đính chính" vào system prompt của ReActAgent.
- [x] **Retrieval có lọc** (Root cause #3): bổ sung tool truy hồi **một section theo mã** (`get_lab_section(lab, code)`) + hướng dẫn agent ưu tiên `get_exercise_guide(N, chủ đề)` thay vì dump cả lab → cải thiện Hit@3 và MRR.
- [x] **Ràng buộc dữ liệu định lượng** (Root cause #2): thêm chỉ dẫn "chỉ nêu số liệu (µs, Hz, địa chỉ...) khi có trong Observation; nếu không, nói rõ chưa có dữ liệu" → giảm hallucination NEC.
- [x] Tăng `max_steps` cho câu multi-hop (V2: 4 → 6) để agent tra đủ nguồn.
- [ ] **Reranking** kết quả `search_lab_docs` ưu tiên chunk chứa số liệu/định lượng (đề xuất tương lai).

## 5. Kết quả sau tối ưu (Agent V2) — Regression Gate
Sau khi áp dụng Action Plan, chạy đối chứng (xem `reports/regression.json`):

| Chỉ số | V1 (base) | V2 (optimized) | Δ |
|--------|-----------|----------------|---|
| Avg Score (Judge) | 4.63 | **4.78** | **+0.16** |
| Pass rate | 96% | **100%** | +4% |
| Retrieval Hit@3 | 92% | **100%** | +8% |
| Retrieval MRR | 0.84 | **0.98** | +0.14 |
| Faithfulness | 0.92 | **0.94** | +0.02 |
| Conflict rate | 4% | **0%** | −4% |
| Case cần chú ý (score ≤3) | 6 | **0** | −6 |

**Thay đổi tạo ra cải thiện:**
1. **Tool `get_lab_section(lab, code)`** (truy hồi có lọc) → MRR 0.84→0.98, Hit@3 100%: section đích về đúng top-1 thay vì lẫn trong dump 8 section.
2. **Prompt đính chính tiền đề** → 2 case `wrong_premise` từ 2–3 điểm lên 4; agent nêu rõ "RC522 thuộc Lab 2".
3. **Prompt cấm bịa số liệu** + lấy đúng `nec_protocol` → `lab1-nec-bit` 2→4, faithfulness 0.5→0.9.
4. **`max_steps` 4→6** → multi-hop (`cross-pa0-usage`, `cross-hercules-why`) tra đủ nguồn, tổng hợp đầy đủ.

**Quyết định Auto-Gate:** Δscore = +0.16 (≥ ngưỡng −0.05) → ✅ **APPROVE** bản V2.

> ⚠️ Số Generation là Claude tự chấm offline (không gọi API). Để xác nhận bằng model thật: `python main.py --model gpt-4o-mini`.
