# Hướng Dẫn Huấn Luyện (Training) Mô Hình Text-To-Music bằng GPU

Chào mừng bạn đến với hướng dẫn chi tiết cách huấn luyện mô hình Transformer sinh nhạc. File này sẽ giải thích các thông số quan trọng, cấu hình phần cứng cần thiết (GPU) và ước lượng thời gian chạy để bạn có thể lên kế hoạch train một mô hình xịn cho Game của mình.

---

## 1. Tại Sao Phải Cần GPU?
Mô hình Music Transformer của chúng ta có khoảng **7.7 triệu tham số (Parameters)**. 
- Khi bạn chạy trên **CPU** (như lúc nãy), mỗi nốt nhạc (token) sinh ra phải duyệt qua hàng triệu phép tính phức tạp (Attention), làm tốc độ cực kỳ chậm (chỉ có thể chạy nháp với data siêu nhỏ).
- Khi bạn chạy trên **GPU (Card Đồ Họa)** (ví dụ: NVIDIA RTX 3060, RTX 4070, hoặc Google Colab T4/A100), hàng ngàn lõi CUDA sẽ tính toán song song, giúp tốc độ học **nhanh hơn CPU từ 20 đến 50 lần**.

---

## 2. Các Thông Số (Parameters) Quan Trọng Khi Train

Khi gõ lệnh `python train.py`, bạn có thể can thiệp vào các thông số sau để điều khiển quá trình học:

| Thông số | Giải thích | Khuyến nghị cho Full Dataset (3000 bài) |
| :--- | :--- | :--- |
| `--data_dir` | Nơi chứa các file MIDI đã lọc. | `data/processed` |
| `--epochs` | Số vòng lặp qua **toàn bộ** tập dữ liệu. Học ít quá thì mô hình chưa hiểu gì, học nhiều quá thì bị "học vẹt" (Overfitting). | `30` đến `50` vòng |
| `--batch_size` | Số bài nhạc đưa vào học cùng một lúc. Batch_size càng to thì học càng nhanh, nhưng sẽ **tốn nhiều VRAM của Card màn hình**. | `8` hoặc `16` (Cần VRAM 6GB - 8GB) |
| `--max_seq_len` | Chiều dài (số nốt nhạc) tối đa cho mỗi bài. Để `2048` thì bài nhạc sẽ rất dài và chi tiết, nhưng tốn nhiều RAM. | `1024` hoặc `2048` |
| `--lr` | Tốc độ học (Learning Rate). Tốc độ mà mô hình điều chỉnh lại sai sót. | `0.0001` (Mặc định chuẩn) |

---

## 3. Lệnh Training Mẫu (Cho GPU)

Đầu tiên, hãy chắc chắn rằng máy bạn (hoặc Colab) đã cài đặt PyTorch phiên bản hỗ trợ CUDA. Kiểm tra bằng cách gõ trong Python: `import torch; print(torch.cuda.is_available())` (Ra `True` là thành công).

**Lệnh Train Chuẩn (Full Dataset 3000 bài):**
```bash
python train.py --epochs 50 --batch_size 16 --max_seq_len 2048 --data_dir data/processed
```
*(Nếu báo lỗi "Out of Memory - Hết VRAM", hãy hạ `--batch_size` xuống `8` hoặc `4`)*

**Lệnh Train Cấp Tốc (Test xem GPU chạy mượt chưa):**
```bash
python train.py --epochs 3 --batch_size 4 --max_seq_len 512 --max_files 500
```

---

## 4. Ước Lượng Thời Gian Huấn Luyện (Training Time)

Với mô hình **7.7 triệu tham số** và bộ dữ liệu **~3.000 file MIDI** (mỗi file bị cắt ở mức `max_seq_len=2048`), thời gian ước lượng dựa trên các Card Đồ Họa phổ biến hiện nay như sau:

> ⚙️ **Thông số tính toán cơ sở:**
> - Số lượng bài (Samples): ~3000 bài.
> - Nếu `batch_size = 16`, chúng ta sẽ có khoảng **187 steps (bước)** mỗi Epoch.

### ⏱️ Trên Google Colab (Free GPU T4 16GB) / Laptop RTX 3060:
- Thời gian cho 1 Epoch: Khoảng **45 giây đến 1 phút**.
- Tổng thời gian cho 50 Epochs: **~45 phút đến 1 tiếng**.
- *Đánh giá:* Rất khả thi để chạy xong ngay trong 1 buổi chiều!

### ⏱️ Trên GPU Mạnh (RTX 4090 / Colab Pro A100):
- Thời gian cho 1 Epoch: Khoảng **15 - 20 giây**.
- Tổng thời gian cho 50 Epochs: **Chưa tới 15 phút**.
- *Đánh giá:* Nhanh như chớp!

### 🐌 Nếu cố chấp chạy bằng CPU (Intel i5/i7 - Ryzen 5/7):
- Thời gian cho 1 Epoch: Thường mất từ **20 phút đến 40 phút**.
- Tổng thời gian cho 50 Epochs: **15 - 30 TIẾNG**.
- *Đánh giá:* Rất dễ làm máy quá nhiệt và không hiệu quả.

---

## 5. Lời Khuyên Quý Báu Cho Dự Án Game BGM

1. **Giai đoạn đầu (Tiết kiệm thời gian):** Đừng cắm đầu train 50 Epochs ngay. Hãy train khoảng `10 epochs`, sau đó dừng lại, lấy mô hình sinh ra từ thư mục `checkpoints/best_model.pt` đi tạo nhạc thử xem nó đã ra được cấu trúc nhạc chưa.
2. **Sử dụng Google Colab:** Nếu máy tính của bạn không có Card rời NVIDIA (ví dụ xài Macbook hoặc Laptop văn phòng), hãy nén toàn bộ project này lại (nhớ dùng file `.gitignore` để xóa bớt các file rác), up lên Google Drive, và mở bằng Google Colab. Bật T4 GPU lên và chạy hoàn toàn miễn phí.
3. **Hiện tượng Overfitting:** Nếu bạn thấy giá trị `Val Loss` bắt đầu tăng lên dù `Train Loss` tiếp tục giảm, điều đó có nghĩa là AI đang học vẹt. Quá trình huấn luyện nên dừng lại ở điểm `Val Loss` thấp nhất (Hệ thống đã tự động lưu lại `best_model.pt` cho bạn rồi).
