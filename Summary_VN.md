# Text-to-Music: So sánh Transformer và Diffusion

Đây là đồ án môn học **Generative AI**. Mục tiêu là tự động tạo ra nhạc nền cho game dựa trên một **câu lệnh (prompt) có cấu trúc** gồm 6 thuộc tính. Dự án so sánh hiệu quả của hai phương pháp AI khác nhau được huấn luyện trên cùng một bộ dữ liệu.

## Ý tưởng cốt lõi

- **Đầu vào (Input)**: Một prompt chứa 6 thuộc tính cụ thể: `tâm trạng (mood)`, `thể loại (genre)`, `hoàn cảnh (scene)`, `nhịp độ (tempo)`, `nhạc cụ (instrument)`, và `năng lượng (energy)`.
- **Đầu ra (Output)**: Một file nhạc **MIDI** (`.mid`).

## Các phương pháp được so sánh

1. **Music Transformer** (Thư mục `GenAI_Transformer/`):
   - Sử dụng phương pháp sinh tự hồi quy (autoregressive) với các token REMI để tạo ra nốt nhạc nối tiếp nhau (step-by-step).
2. **Piano-roll Diffusion** (Thư mục `GenAI_Diffusion/`):
   - Sử dụng mô hình khử nhiễu (DDPM/DDIM) kết hợp với Classifier-Free Guidance (CFG) để tạo ra bản nhạc từ tín hiệu nhiễu ngẫu nhiên.

## Cấu trúc dự án

- `data/`: Bộ dữ liệu dùng chung chứa khoảng 14,000 file MIDI (được tổng hợp từ ComMU, MidiCaps, MAESTRO, v.v.).
- `GenAI_Transformer/`: Mã nguồn cho mô hình Music Transformer.
- `GenAI_Diffusion/`: Mã nguồn cho mô hình Piano-roll Diffusion.
- `compare/`: Các script và tiêu chí dùng để đánh giá và so sánh hai mô hình (kết quả xuất ra file CSV).
- `docs/`: Tài liệu chi tiết bao gồm hướng dẫn huấn luyện, cấu trúc dữ liệu, và báo cáo nghiên cứu.
- `scripts/`: Các đoạn script tiện ích dùng để xử lý dữ liệu.

## Bắt đầu nhanh (Quick Start)

### Yêu cầu hệ thống
- Python 3.10+
- Khuyến nghị dùng Card đồ hoạ (GPU) NVIDIA
- Thư viện cần thiết: `torch`, `pretty_midi`

### Huấn luyện (Training)
```powershell
# Train mô hình Transformer
cd GenAI_Transformer
python train.py --epochs 10 --batch_size 8 --max_seq_len 1024 --no_early_stop

# Train mô hình Diffusion
cd GenAI_Diffusion
python train.py --epochs 10 --batch_size 4
```

### Sinh nhạc (Generating Music)
```powershell
# Sinh nhạc với Transformer (mất khoảng ~30 giây)
cd GenAI_Transformer
python generate.py --mood happy --genre fantasy --scene village --tempo fast --instrument piano --duration_sec 30

# Sinh nhạc với Diffusion (mất khoảng ~10 giây)
cd GenAI_Diffusion
python generate.py --checkpoint checkpoints/best_model.pt --epoch 10 --duration_sec 12 --evaluate
```

### Chạy so sánh (Running Comparisons)
```powershell
# So sánh kết quả tại các epoch 1, 5, và 10
python -m compare.run_comparison_eval --epochs 1 5 10
python -m compare.compare_results
```
*Kết quả sẽ được lưu thành file CSV trong thư mục `compare/results/`.*

### Tích hợp API (Tùy chọn)
Dự án có đi kèm một API sử dụng FastAPI dành cho mô hình Transformer.
```powershell
cd GenAI_Transformer
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*Truy cập tài liệu API tại địa chỉ: `http://localhost:8000/docs`.*

---
*Để xem hướng dẫn chi tiết hơn về cách huấn luyện, yêu cầu phần cứng, và cách chuẩn bị dữ liệu, vui lòng tham khảo các file markdown trong thư mục `docs/`.*
