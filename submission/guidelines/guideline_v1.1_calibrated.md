# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603124 - Bùi Lê Đình Văn

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện: gán `bicycle` hay `motorbike`? | Gán `bicycle` nếu giữ khung xe đạp truyền thống có bàn đạp và không gắn biển số; gán `motorbike` nếu có sàn để chân kiểu xe tay ga hoặc gắn biển số cơ giới. | `img_018.jpg` | `CAL-01` |
| 2 | Hàng hóa cồng kềnh và mái bạt tự chế trên xe ba gác/xích lô có tính vào box `three_wheeler` không? | Tính phần mái bạt gắn cố định vào khung xe; không kéo box bao trùm hàng hóa rời nhô cao tách biệt và luôn tách riêng box `person` cho người ngồi/lái. | `img_003.jpg` | `CAL-02` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_008.jpg` | Xe máy chở hàng cồng kềnh và người lái ngồi trên xe hai bánh | YOLOE (`conf=0.15`) gắn nhầm xe máy chở hàng thành `three_wheeler`; YOLO26 đôi khi gộp người lái vào xe | Xóa nhãn `three_wheeler` giả, giữ đúng 1 box `motorbike` theo số bánh thực tế và 1 box `person` độc lập cho người lái | Bổ sung vào **Mục 3.1 (Phân biệt phương tiện & tách người–xe)** (`CAL-03`) |
| `img_014.jpg` | Khung pre-label bao đúng vật thể nhưng các cạnh bị kéo lỏng rộng hơn biên thật (bẫy `loose`) | Chèn nhiễu biên box từ bước `inject_traps` / sai lệch hồi quy biên của detector | Ẩn toàn bộ nhãn (`H`) để quan sát biên thật trước, sau đó zoom kéo sát 4 cạnh box đạt độ khít $\text{IoU} \ge 0.9$ | Bổ sung vào **Mục 5.2 (Checklist kiểm tra độ khít cạnh box)** (`CAL-04`) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt 2 câu hỏi mở về phân loại xe đạp điện và phạm vi mái bạt/hàng hóa trên `three_wheeler`; bổ sung quy tắc chống nhận nhầm xe máy chở hàng và checklist siết biên box chống lỗi `loose`. | Khắc phục lỗi lặp của YOLOE (22 box `three_wheeler` dương tính giả) và lỗi lọt bẫy `loose` ở lần duyệt đầu (`CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`). |
