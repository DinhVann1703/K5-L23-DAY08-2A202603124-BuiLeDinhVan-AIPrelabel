# Báo cáo Day 08 — 2A202603124 - Bùi Lê Đình Văn

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

- Thời gian (từ `time_log.csv`): gán tay **4.5 phút/ảnh** (18 phút / 4 ảnh gold) vs duyệt **2.3 phút/ảnh** (46 phút / 20 ảnh batch; các ảnh đông như `img_018`, `img_021`, `img_024` mất ~4 phút/ảnh). Thiên lệch (bạn gán gold trước nên đã quen cảnh): Việc tự tay gán nhãn 4 ảnh gold trước giúp thuộc lòng quy tắc tách người–xe và ngưỡng kích thước `< 20px` nên thao tác duyệt nhanh hơn (~2 lần), đồng thời cũng làm tăng xu hướng tin tưởng các box pre-label trông "có vẻ hợp lý" nếu không chủ động kiểm tra lại.
- Kết luận: **Có, đáng dùng làm bước khởi tạo (pre-label) cho các lớp phổ thông (`person`, `motorbike`, `car`, `bicycle`, `bus`)**, nhưng **cần hạn chế hoặc nâng ngưỡng cho lớp `three_wheeler`**. Dựa trên `qc/review_stats.md`, máy giúp giữ nguyên **89.7% (452/504)** box gợi ý; riêng lớp `three_wheeler` có **22/42** box bị xóa nên chỉ nên giữ lại khi điểm tin cậy thực sự rõ ràng.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quét mắt không nhãn trước khi duyệt:** Luôn bấm `H` để ẩn toàn bộ pre-label trong 5–10 giây đầu mở ảnh, tự đếm và xác định vị trí các đối tượng rõ ràng (`≥ 20px`) trước khi xem gợi ý của máy (kiểm tra bằng việc bắt đủ `3/3` bẫy `missing` và thêm `87` đối tượng máy bỏ sót).
  2. **Kiểm tra độ ôm sát 4 cạnh cho mọi box giữ lại:** Không duyệt hàng loạt; zoom vào nhóm đối tượng đông đúc và kéo sát cạnh box thừa (kiểm tra bằng việc sửa `14` box `edited` và bắt đủ `3/3` bẫy `loose`).
  3. **Đối chiếu chéo nhãn–vật thể và phân tách người/xe:** Rà soát riêng các lớp dễ nhầm (`three_wheeler`, `truck`, `bicycle`), tách riêng `person` khỏi phương tiện hai bánh (kiểm tra bằng việc xóa `38` box sai/thừa, xử lý đúng `10` trường hợp sai lớp và bắt đủ `3/3` bẫy `wrong_class`, `3/3` bẫy `extra`).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương, 1 âm) | Có dính người lái ngồi trong khoang lái; không có hàng hóa | Phủ trọn thân xe, mái bạt và bánh xe; bị lem nhẹ một mảng hồng nhỏ sang áo người ngồi phía sau qua khoảng trống khung xe | `submission/masks/tuktuk_img013.jpg` |
| `img_003.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 4 (3 dương, 1 âm) | Không gồm người ngồi trên xe; tách được phần thùng xe khỏi đống hàng hóa phía trên | Phủ trọn thùng xe và 2 bánh xe nhưng bỏ sót phần mái bạt đen phía trên do các thanh chống kim loại mảnh ngăn cách với thân thùng | `submission/masks/cyclo_img003.jpg` |
| `img_010.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 4 (3 dương, 1 âm) | Có dính trọn người đạp xích lô phía sau; không dính hành khách ngồi khoang trước | Bắt được người đạp, mái che đỏ và 2 bánh xe nhưng bị rỗng phần thùng xe ở giữa và thiếu bánh phải do khung sắt chia cắt vùng liên thông | `submission/masks/cargo_img010.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? Quy tắc phân tách **người lái/hành khách khỏi phương tiện** và **không tính hàng hóa cồng kềnh rời rạc** vào khung phương tiện giúp xác định mục tiêu chỉ cắt mask phần thân xe (`three_wheeler`) và đặt điểm click dương trực tiếp lên kết cấu thùng/khung xe. Tuy nhiên, guideline mục 4.2 chưa quy định rõ mức độ chấp nhận với các bộ phận nối bằng **thanh chống mảnh** (như mái bạt xe ba gác/xích lô dễ bị SAM tách rời thành thành phần không liên thông), cũng như cách xử lý khi người lái ngồi lọt hẳn vào trong khoang xe kín một phần như tuk-tuk.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: Không dùng Interactor SAM trực tiếp trong CVAT local do môi trường Docker CVAT chạy CPU local không gắn GPU Nuclio serverless function; chạy offline bằng `sam2_points.py` (`sam2.1_b.pt`) trên GPU T4 Colab, phản hồi khoảng 1.8 giây/ảnh và lưu ảnh kiểm tra trực quan (`viz`) cùng tọa độ click có thể tái lập.

## 6. Calibration (Part 6)

- Số quyết định: **4 quyết định** (`CAL-01` đến `CAL-04` trong `submission/qc/calibration_decisions.csv`, gồm 2 câu hỏi mở và 2 lỗi máy/bẫy); quan trọng nhất: **`CAL-03`** (YOLOE ở ngưỡng `conf=0.15` nhận nhầm xe máy chở sọt hàng cồng kềnh thành `three_wheeler` và YOLO26 gộp người lái vào phương tiện, gây ra `22` box `three_wheeler` dương tính giả phải xóa khi review) và **`CAL-04`** (bẫy `loose` trên `img_014.jpg` bị lọt ở lần duyệt đầu do nhìn đè pre-label có sẵn).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:
  1. **Mục 2.2 (`CAL-01`, ví dụ `img_018.jpg`):** Phân loại xe đạp điện dựa vào kết cấu bàn đạp/sàn để chân và biển số cơ giới (`bicycle` nếu giữ khung xe đạp có bàn đạp và không biển số; `motorbike` nếu dáng xe ga có sàn để chân hoặc gắn biển số).
  2. **Mục 2.4 & 4.2 (`CAL-02`, ví dụ `img_003.jpg`):** Box và mask `three_wheeler` ôm trọn thân xe cùng mái bạt gắn cố định trên khung; không kéo rộng theo hàng hóa rời nhô cao tách biệt và luôn tách riêng `person`.
  3. **Mục 3.1 (`CAL-03`, ví dụ `img_008.jpg`):** Xác định lớp phương tiện chở hàng theo số bánh thực tế (`2` bánh gán `motorbike`), xóa nhãn `three_wheeler` giả của YOLOE và tách độc lập box `person` cho người lái.
  4. **Mục 5.2 (`CAL-04`, ví dụ `img_014.jpg`):** Bắt buộc ẩn nhãn (`H`) để quét ảnh gốc trước khi bật pre-label và kiểm tra độ ôm sát 4 cạnh box ($\text{IoU} \ge 0.9$).
- Nguyên nhân gốc: **Cả 3 yếu tố kết hợp** — (1) *guideline v1.0 còn câu hỏi mở* về xe đạp điện và phạm vi mái bạt/hàng hóa trên xe ba gác; (2) *lỗi lặp từ máy* (YOLOE ngưỡng thấp `0.15` bắt nhầm xe máy chở hàng thành `three_wheeler`); và (3) *thiên kiến tự động hóa của người duyệt* khiến bỏ qua khâu soi độ khít cạnh box ở lần duyệt đầu.

## 7. QC và quyết định (Part 6)

- Mẫu: **5 ảnh** (`0` ảnh `can_review` + `5` ảnh ngẫu nhiên: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`, seed 42; tổng **105 đối tượng**); người kiểm: **Bùi Lê Đình Văn** (**hạn chế**: người kiểm QC cũng chính là người thực hiện duyệt lô nên có thể tồn tại điểm mù chủ quan giống lúc review).
- Lỗi theo **đối tượng**: **0/105 = 0.0%** · theo **ảnh**: **0/5 = 0.0%** (ngưỡng Release $\le 2\%$).
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: **Không ghi nhận lỗi nào (0/105 đối tượng, 0/5 ảnh)** trong mẫu kiểm tra. Các ca từng gặp ở vòng duyệt đầu như `img_003.jpg` (`three_wheeler` chở hàng), `img_014.jpg` (bẫy `loose`) và `img_031.jpg` (bẫy `wrong_class`) đã được xử lý trước khi xuất bản release.
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): Với 0 ảnh lỗi trên mẫu nhỏ `5` ảnh, cận trên khoảng tin cậy 95% (Wilson) của tỉ lệ ảnh lỗi thực tế lên tới **43%**. Do đó, mẫu 5 ảnh chỉ đủ phát hiện lỗi hệ thống rõ rệt chứ chưa thể chứng minh về mặt thống kê rằng tỉ lệ lỗi toàn lô thực sự dưới ngưỡng 2%.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): Các script (`validate_labels.py`, `review_stats.py`, `qc_report.py`, `check_submission.py`) kiểm tra tự động cấu trúc JSON COCO, diện tích tối thiểu, box trùng lặp, tỉ lệ `accepted/edited/deleted/added` và độ phủ bắt bẫy (`12/12`). Tuy nhiên, script **không thể tự phát hiện** nếu người duyệt và máy cùng bỏ sót một đối tượng thật không có nhãn, cũng không tự đánh giá độ ôm sát biên pixel thật của từng vật thể nếu không có kiểm tra trực quan.

**Quyết định: RELEASE**, ngày **2026-11-17**. Lý do (đi theo Hold → Rework → Release): Guideline v1.1 đã chốt xong các câu hỏi mở và dữ liệu hợp lệ (không Hold); tỉ lệ ảnh lỗi mẫu đạt `0/5 = 0.0%` ($\le 2\%$), không còn lỗi lặp hệ thống và bắt bẫy đạt `12/12 = 100.0%` ($\ge 80\%$, không Rework), nên đủ điều kiện xuất bản `submission/release/` kèm `qc_report_auto.md` và `REPORT.md`.
Nếu Rework / Hold, việc cần làm tiếp: **Không áp dụng (đã Release).**

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Sử dụng **YOLO26m**, **YOLOE** (open-vocabulary) và **SAM 2.1 (`sam2.1_b.pt`)** trên Google Colab GPU T4 để sinh pre-label phát hiện vật thể (`Part 2–4`) và cắt mask phân đoạn (`Part 5`); kết quả mô hình được kiểm tra trực quan trên CVAT và đo định lượng bằng `review_stats.py` trên tập `gold_manual.json` (4 ảnh) cùng 12 ảnh bẫy ẩn. Ngoài ra, dùng trợ lý AI hỗ trợ rà soát cú pháp lệnh PowerShell/Python và cấu trúc Markdown; các số liệu được đối chiếu với output thực tế của script trong repo.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? **Không** — 4 ảnh gold được gán thủ công hoàn toàn từ đầu trên CVAT trước khi chạy mô hình pre-label ở Part 2–3.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Máy cá nhân chạy **Windows 11** (CPU local, không có GPU rời; chạy inference YOLO26/YOLOE/SAM 2.1 trên **Google Colab GPU T4**); gán nhãn và review bằng **CVAT Docker local (`v2.81.0` / `localhost:8080`)**. Khi khởi động lại Docker CVAT từng gặp lỗi kết nối tạm thời (`Bad Gateway`) trên cổng `8080` do `cvat_server` khởi động chậm hơn proxy; khắc phục bằng cách đợi dịch vụ ổn định rồi tải lại trang.
- Dùng cache `data/cache/` thay vì tự chạy? **Không** — tự chạy pipeline trên Google Colab T4 và xuất kết quả về repo.
- Lỗi / bug script đã gặp (nếu có): Không gặp lỗi logic script của bài lab; trên Windows PowerShell, cần dùng `Get-Content -Encoding UTF8` để hiển thị đúng UTF-8 tiếng Việt.
- Một điều bạn làm khác lần sau: Ngay từ lượt duyệt đầu tiên trên CVAT, luôn bấm `H` để ẩn toàn bộ pre-label trong 10 giây đầu mở ảnh và zoom kiểm tra độ khít 4 cạnh từng box thay vì chỉ rà nhanh box bất thường, để bắt đủ 100% bẫy `loose` ngay vòng đầu mà không cần rà soát bổ sung.
