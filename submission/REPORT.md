# Báo cáo Day 08 — <<2A202603124 - Bùi Lê Đình Văn>>

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26`) | `car=0.5`, các lớp còn lại mặc định `0.3` | Dẫn `t_y26` và `t_y26_c05`: YOLO26 bắt tốt các phương tiện phổ thông và người đi bộ. Quét ngưỡng cho thấy `car=0.5` đạt Recall 90% và Precision 69%, giúp lọc bớt box nhiễu; `person`, `motorbike`, `bicycle` không đạt Recall 90% ở ngưỡng ≤ 0.5 do nhiều đối tượng nhỏ/xa (< 20px) bị che khuất nên giữ mặc định `0.3`; `bus` (0 mẫu) và `truck` (1 mẫu) quá ít dữ liệu trong tập Gold nên giữ mặc định `0.3`. |
| three_wheeler | YOLOE (`yoloe`), prompt: `'tuk-tuk'`, `'cyclo'`, `'cargo tricycle'` | `0.15` | Dẫn `t_yoloe`: YOLO26 không có lớp xe ba gác/xích lô trong bộ nhãn gốc. Kết hợp YOLOE với text prompt tại ngưỡng `0.15` đạt Recall 100% và Precision 78% trên tập Gold (7/7 vật), bổ sung chính xác lớp `three_wheeler` bị thiếu. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): Chưa có lệnh chạy thực tế được lưu trong repo; cần bổ sung nguyên lệnh đã chạy ở Part 4.
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn **SAM 2.1** (tích hợp trong pipeline `t_sam` trên Colab T4). Lý do: không yêu cầu xin quyền truy cập và cấu hình token HuggingFace phức tạp như SAM 3, tốc độ suy luận nhanh (~1.93s/ảnh trên GPU T4), tiết kiệm VRAM/RAM hơn SAM 1 và chất lượng mask bám sát biên đối tượng tốt (`16/16` box có mask).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Lỗi nhận diện sai lớp (như xe ba gác/xích lô bị YOLO26 gắn nhầm thành `car` hoặc `motorbike`, xe máy chở hàng cồng kềnh bị YOLOE nhầm thành `three_wheeler`), lỗi gộp chung người lái với xe thành 1 box thay vì tách riêng 2 box (`person` + `motorbike`/`bicycle`) theo guideline, và các cụm xe/người quá nhỏ `< 20px` ở xa không được gom thành `ignore_region`. Cách sửa: dùng `merge_prelabels.py` (ưu tiên `three_wheeler` từ YOLOE hoặc NMS theo IoU) và chỉnh sửa thủ công (đổi nhãn, tách/vẽ lại box, gộp `ignore_region`) khi review trên CVAT.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - *Điểm cao mà sai:* Trên `img_032.jpg` (`t_y26_c05`), một số xe máy chở hàng lớn hoặc xe ba bánh bị gắn nhãn `car` với điểm tin cậy cao (`> 0.5`), hoặc người ngồi trên xe máy bị gộp chung vào box xe.
  - *Điểm thấp mà đúng:* Trên `img_032.jpg` (`t_yoloe`), các đối tượng `three_wheeler` (xích lô/xe ba gác) có điểm tin cậy thấp (`0.15–0.25`) nhưng là phát hiện đúng, cần hạ ngưỡng xuống `0.15` để không bị bỏ sót.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: Bộ ngưỡng (`car=0.5`, `three_wheeler=0.15`) được tìm bằng cách quét trực tiếp trên chính 4 ảnh `gold_manual.json`, nên các chỉ số Precision/Recall thu được mang tính lạc quan (vừa chọn ngưỡng vừa đánh giá trên cùng một tập dữ liệu nhỏ). Quy mô 4 ảnh khiến ngưỡng chỉ mang tính ước lượng thô, dễ bị khớp quá mức (overfit) vào bối cảnh của 4 ảnh này và chưa phản ánh đủ độ đa dạng cho các lớp hiếm như `bus` (0 mẫu) hay `truck` (1 mẫu).

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 452 | 452/504 = 89.7% | 504 box máy |
| edited | 14 | 14/504 = 2.8% | 504 box máy |
| deleted | 38 | 38/504 = 7.5% | 504 box máy |
| added | 87 | 87/553 = 15.7% | 553 đối tượng sau duyệt |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? **Lớp `three_wheeler` tệ nhất** (chỉ giữ lại `14/42` box máy = `33.3%`, phải xóa `22` box và sửa `6` box) do ngưỡng chạy YOLOE (`0.15`) quá thấp, khiến model nhận diện nhầm nhiều xe máy chở hàng cồng kềnh hoặc cụm xe máy đậu sát nhau thành xe ba gác (ví dụ `img_003.jpg`, `img_008.jpg`, `img_017.jpg`). Ngoài ra, lớp `person` (35 đối tượng thêm) và `car` (20 đối tượng thêm) bị bỏ sót nhiều nhất ở các cảnh đông đúc hoặc xe đậu nối đuôi bị che khuất một phần (như `img_018.jpg`, `img_021.jpg`, `img_024.jpg`).
- So với gold: precision / recall của **máy** đạt **Precision 76.7%** (56/73, KTC 95%: 66–85%) và **Recall 48.3%** (56/116, KTC 95%: 39–57%) (mẫu số: 116 đối tượng gold; nguồn: `qc/review_stats.md`).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **12/12 (100.0%, KTC 95%: 76–100%)**; loại nào lọt: **không có (0/12)** (sau khi rà soát lại cả 4 nhóm bẫy `wrong_class`: 3/3, `missing`: 3/3, `loose`: 3/3, `extra`: 3/3); vì sao (bạn đã bỏ bước nào của checklist)? Ở lần duyệt đầu tiên, do nhìn trực tiếp vào khung pre-label có sẵn trên ảnh đông đúc nên bị thiên kiến tự động hóa (automation bias), lướt qua 3 bẫy `loose` (`img_014.jpg`, `img_018.jpg`, `img_021.jpg` — box nằm đúng vị trí vật thể nhưng cạnh bị kéo rộng hơn biên thật) và 1 bẫy `missing` (`img_026.jpg` — người nhỏ ở xa trong vùng tối). Sau khi áp dụng đúng bước ẩn tạm toàn bộ nhãn (`H`) để quét ảnh gốc trước và soi từng cạnh box theo đúng tỷ lệ, toàn bộ 12/12 bẫy đã được phát hiện và xử lý.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Ở đây `accepted` cao (`89.7%`, 452/504 box) đi kèm bắt bẫy cuối cùng đạt `12/12 (100%)`. `Accepted` cao phản ánh chất lượng nền của YOLO26 trên các lớp phổ thông (`person`, `motorbike`), nhưng chênh lệch giữa lần duyệt đầu (8/12 bẫy — lọt cả 3 bẫy `loose`) và lần rà soát lại (12/12) cho thấy người duyệt dễ tin các khung có sẵn trên vật thể và bỏ qua việc kiểm tra độ ôm sát của box nếu không chủ động zoom, đối chiếu từng cạnh.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay **4.5 phút/ảnh** (18 phút/4 ảnh gold) vs duyệt **chưa có số liệu trong `time_log.csv`** (20 ảnh batch). Thiên lệch (bạn gán gold trước nên đã quen cảnh): Việc tự tay gán nhãn 4 ảnh gold trước giúp thuộc quy tắc tách người–xe và ngưỡng kích thước `< 20px` nên thao tác duyệt nhanh hơn, đồng thời cũng làm tăng xu hướng tin tưởng các box pre-label trông "có vẻ hợp lý" nếu không chủ động kiểm tra lại.
- Kết luận: **Có, đáng dùng làm bước khởi tạo (pre-label) cho các lớp phổ thông (`person`, `motorbike`, `car`, `bicycle`, `bus`)**, nhưng **cần hạn chế hoặc nâng ngưỡng cho lớp `three_wheeler`**. Dựa trên `qc/review_stats.md`, máy giúp giữ nguyên **89.7% (452/504)** box gợi ý; riêng lớp `three_wheeler` có **22/42** box bị xóa nên chỉ nên giữ lại khi điểm tin cậy thực sự rõ ràng.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quét mắt không nhãn trước khi duyệt:** Luôn bấm `H` để ẩn toàn bộ pre-label trong 5–10 giây đầu mở ảnh, tự đếm và xác định vị trí các đối tượng rõ ràng (`≥ 20px`) trước khi xem gợi ý của máy (kiểm tra bằng việc bắt đủ `3/3` bẫy `missing` và thêm `87` đối tượng máy bỏ sót).
  2. **Kiểm tra độ ôm sát 4 cạnh cho mọi box giữ lại:** Không duyệt hàng loạt; zoom vào nhóm đối tượng đông đúc và kéo sát cạnh box thừa (kiểm tra bằng việc sửa `14` box `edited` và bắt đủ `3/3` bẫy `loose`).
  3. **Đối chiếu chéo nhãn–vật thể và phân tách người/xe:** Rà soát riêng các lớp dễ nhầm (`three_wheeler`, `truck`, `bicycle`), tách riêng `person` khỏi phương tiện hai bánh (kiểm tra bằng việc xóa `38` box sai/thừa, xử lý đúng `10` trường hợp sai lớp và bắt đủ `3/3` bẫy `wrong_class`, `3/3` bẫy `extra`).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
