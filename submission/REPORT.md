# Báo cáo Day 08 — <<MSHV - Họ và tên>>

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
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

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
