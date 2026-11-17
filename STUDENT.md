# Thông tin học viên

- Họ và tên: Bùi Lê Đình Văn
- MSHV: 2A202603124
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: DinhVann1703
- Repo bài nộp (URL): https://github.com/DinhVann1703/K5-L23-DAY08-2A202603124-BuiLeDinhVan-AIPrelabel
- Commit chốt bài (hash): 5b85257
- Máy: Windows 11 (CPU local) + Google Colab (GPU NVIDIA Tesla T4, CUDA)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 452 / 14 / 38 (mẫu số: 504 box máy); added: 87 (mẫu số: 553 đối tượng sau duyệt)
- So với gold: precision / recall của **máy**: Precision 76.7% (56/73) / Recall 48.3% (56/116 đối tượng gold)
- Bắt bẫy: 12/12 = 100.0%
- Thời gian (từ `time_log.csv`): gán tay 4.5 phút/ảnh (18 phút / 4 ảnh gold) vs duyệt 2.3 phút/ảnh (46 phút / 20 ảnh batch)
- Lỗi QC: 0.0% theo ảnh (0/5), 0.0% theo đối tượng (0/105)
- **Quyết định:** RELEASE — lỗi QC trong mẫu đạt 0/5 ảnh và 0/105 đối tượng; bắt bẫy đạt 12/12.
