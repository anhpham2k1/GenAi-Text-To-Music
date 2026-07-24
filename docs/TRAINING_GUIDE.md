# Hướng dẫn Training — Transformer & Diffusion

Tài liệu train **hai project** trên **cùng dataset** `D:\Master\Ky3\data`.

- **GenAI_Transformer**: Music Transformer (REMI tokens + **English text → MiniLM**)  
- **GenAI_Diffusion**: Conditional piano-roll diffusion (UNet + CFG + **same text encoder**)  

**Conditioning:** free-text English prompt (not 6-ID embeddings).  
Shared: `compare/text_conditioner.py`, captions from `data/labels/captions.json` (or build on the fly from labels).  

---

## 0. Chuẩn bị chung

```powershell
cd D:\Master\Ky3\GenAI_Transformer
pip install -r requirements.txt

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Dataset phải có:

- `data/processed/` (~14k MIDI)
- `data/labels/labels.json`
- `data/labels/captions.json` (English — optional, auto from labels)
- `compare/split.json` (train/val cố định)

```powershell
cd D:\Master\Ky3
python scripts/build_captions.py
python -m compare.make_split --midi_dir data/processed
```

> Checkpoint **6-ID cũ không load** được cho text pipeline — train lại từ đầu.

---

## 1. Music Transformer (`GenAI_Transformer/`)

### Lệnh chuẩn

```powershell
cd D:\Master\Ky3\GenAI_Transformer

python train.py `
  --epochs 30 `
  --batch_size 8 `
  --max_seq_len 1024 `
  --no_early_stop `
  --save_epochs "1,5,10,20,30"
```

### Tham số chính

| Tham số | Ý nghĩa |
|---------|---------|
| `--epochs` | Số vòng qua hết data |
| `--batch_size` | Số bài / batch (tốn VRAM) |
| `--max_seq_len` | Độ dài token REMI (1024 hoặc 2048) |
| `--lr` | Learning rate (mặc định 1e-4) |
| `--no_early_stop` | Chạy đủ epoch (cần cho so 1/5/10) |
| `--save_epochs` | Checkpoint các mốc so sánh |
| `--max_files` | Giới hạn file (smoke test) |

### Output

- `GenAI_Transformer/checkpoints/best_model.pt`
- `GenAI_Transformer/checkpoints/checkpoint_epoch_{N}.pt`
- `compare/results/transformer_training.csv`

### Generate

```powershell
python generate.py `
  --prompt "Happy fantasy village music, fast tempo, piano, medium energy" `
  --duration_sec 30 `
  --checkpoint checkpoints/best_model.pt
```

### Smoke test (nhanh)

```powershell
python train.py --epochs 2 --batch_size 4 --max_seq_len 512 --max_files 500 --no_early_stop
```

---

## 2. Diffusion (`GenAI_Diffusion/`)

### Lệnh chuẩn

```powershell
cd D:\Master\Ky3\GenAI_Diffusion

python train.py `
  --epochs 30 `
  --batch_size 4
```

Config mặc định (`config/config.yaml`):

- Piano-roll `88 × 256` @ 24 Hz (~10.6s)
- Cosine schedule, CFG (`cond_drop=0.1`), EMA
- Save epoch: 1, 5, 10, 20, 30

### Tham số CLI

| Tham số | Ý nghĩa |
|---------|---------|
| `--epochs` | Số epoch |
| `--batch_size` | Batch (roll 2D tốn VRAM hơn token) |
| `--lr` | Mặc định ~1.5e-4 |
| `--max_files` | Smoke test |

### Output

- `GenAI_Diffusion/checkpoints/best_model.pt`
- `checkpoint_epoch_{N}.pt`
- `compare/results/diffusion_training.csv`

### Generate

```powershell
python generate.py `
  --checkpoint checkpoints/best_model.pt `
  --epoch 30 `
  --duration_sec 12 `
  --guidance_scale 3.5 `
  --sample_steps 80 `
  --evaluate
```

| Flag | Ý nghĩa |
|------|---------|
| `--duration_sec` | Độ dài ≈ `n_frames / fs` |
| `--guidance_scale` | CFG (2–5; default 3.5) |
| `--sample_steps` | Bước DDIM |

### Smoke test

```powershell
python train.py --epochs 2 --batch_size 2 --max_files 300
```

---

## 3. Thông số gợi ý theo card đồ họa

> **Lưu ý:** RTX **5060 / 5060 Ti** — thông số VRAM có thể khác theo bản chính thức; bảng dưới theo class phổ biến (**8GB** entry / **16GB** Ti).  
> Nếu OOM: giảm `batch_size` → giảm `max_seq_len` / `n_frames` → tăng `gradient_accumulation_steps` trong config.

### Tóm tắt VRAM

| Card | VRAM (điển hình) | Vai trò |
|------|------------------|---------|
| **RTX 4060** | **8 GB** | Laptop/desktop phổ biến — đủ train cả 2 model |
| **RTX 5060** | **~8 GB** | Tương đương class 4060, CUDA/Tensor mới hơn → hơi nhanh hơn |
| **RTX 5060 Ti** | **~16 GB** | Batch lớn hơn, seq/roll dài hơn |

---

### 3.1. Music Transformer

Giả sử data ~14k, `d_model=256`, `layers=6`, AdamW + AMP.

| Card | batch_size | max_seq_len | grad_accum (config) | Effective batch | Epochs gợi ý | Ước lượng 1 epoch* |
|------|------------|-------------|---------------------|-----------------|--------------|---------------------|
| **4060 8GB** | **8** | **1024** | 4 | ~32 | 30 | ~2–5 phút |
| **4060 8GB** (an toàn) | 4 | 1024 | 4 | ~16 | 30 | chậm hơn chút |
| **5060 ~8GB** | **8–12** | **1024** | 2–4 | ~24–48 | 30 | nhanh hơn 4060 ~10–30% |
| **5060 Ti ~16GB** | **16–24** | **1024** | 2 | ~32–48 | 30–50 | ~1–3 phút |
| **5060 Ti ~16GB** (dài) | 8–12 | **2048** | 2–4 | ~16–48 | 30 | tốn VRAM hơn |

\*Thời gian phụ thuộc CPU load data, ổ SSD, driver; chỉ mang tính định hướng.

**Lệnh copy-paste:**

```powershell
# --- RTX 4060 (8GB) ---
python train.py --epochs 30 --batch_size 8 --max_seq_len 1024 --no_early_stop --save_epochs "1,5,10,20,30"

# --- RTX 5060 (~8GB) ---
python train.py --epochs 30 --batch_size 10 --max_seq_len 1024 --no_early_stop --save_epochs "1,5,10,20,30"

# --- RTX 5060 Ti (~16GB) ---
python train.py --epochs 40 --batch_size 16 --max_seq_len 1024 --no_early_stop --save_epochs "1,5,10,20,30"
# hoặc seq dài:
# python train.py --epochs 30 --batch_size 8 --max_seq_len 2048 --no_early_stop
```

---

### 3.2. Diffusion (piano-roll)

Config mặc định: `base_channels=64`, roll `88×256`, AMP.

| Card | batch_size | n_frames (config) | grad_accum | Epochs gợi ý | Ước lượng 1 epoch* |
|------|------------|-------------------|------------|--------------|---------------------|
| **4060 8GB** | **2–4** | 256 | 4 | 30 | ~3–8 phút |
| **4060 8GB** (OOM) | **2** | 256 hoặc 128 | 4–8 | 30 | ổn định hơn |
| **5060 ~8GB** | **4** | 256 | 4 | 30 | nhanh hơn 4060 nhẹ |
| **5060 Ti ~16GB** | **8–12** | 256 | 2 | 30–50 | ~1–4 phút |
| **5060 Ti ~16GB** (dài) | 4–6 | **384–512** (~16–21s) | 2–4 | 30 | train lại shape lớn hơn |

Chỉnh `n_frames` / `batch_size` trong `GenAI_Diffusion/config/config.yaml` nếu cần.

**Lệnh copy-paste:**

```powershell
# --- RTX 4060 (8GB) ---
python train.py --epochs 30 --batch_size 4
# nếu OOM:
python train.py --epochs 30 --batch_size 2

# --- RTX 5060 (~8GB) ---
python train.py --epochs 30 --batch_size 4

# --- RTX 5060 Ti (~16GB) ---
python train.py --epochs 40 --batch_size 8
```

**Generate guidance (mọi card):**

| guidance_scale | Ý nghĩa |
|----------------|---------|
| 1.0 | Gần unconditional |
| **3.0–3.5** | Mặc định cân bằng |
| 5.0 | Bám prompt mạnh, dễ cứng |

---

### 3.3. So sánh công bằng trên cùng card

1. Cùng `compare/split.json`  
2. Cùng `eval_prompts.json` (20 prompt)  
3. Checkpoint epoch **1, 5, 10** (và 20/30 nếu có)  
4. Cùng script:

```powershell
cd D:\Master\Ky3
python -m compare.run_comparison_eval --epochs 1 5 10
python -m compare.compare_results
```

**Không so** `val_loss` CE (Transformer) với MSE (Diffusion) — chỉ so metric MIDI + time/VRAM trong CSV.

---

## 4. Bảng “nên dùng gì” theo máy

| Máy | Transformer | Diffusion | Ghi chú |
|-----|-------------|-----------|---------|
| **4060 8GB** | bs=8, seq=1024, 30 ep | bs=2–4, 30 ep | An toàn, đủ đồ án |
| **5060 8GB** | bs=8–12, seq=1024 | bs=4 | Giống 4060, nhanh hơn |
| **5060 Ti 16GB** | bs=16, seq=1024 (hoặc 2048) | bs=8+, 40–50 ep | Đẩy chất lượng / batch lớn |

---

## 5. Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|-----|------------|
| **CUDA out of memory** | Giảm `batch_size` → `max_seq_len` / `n_frames` |
| Không thấy GPU | Cài PyTorch đúng CUDA; `nvidia-smi` |
| Data path not found | Chạy từ đúng thư mục project; data ở `Ky3/data` |
| Checkpoint architecture mismatch | Train lại Diffusion sau khi đổi `base_channels` / `cond_dim` |
| Bài Transformer quá ngắn | Tăng `--duration_sec` / `--max_length` |

---

## 6. Checklist trước khi nộp / bảo vệ

- [ ] Train đủ epoch đã chọn (khuyến nghị ≥ 10–30)  
- [ ] Có `checkpoint_epoch_1/5/10` cả hai model  
- [ ] CSV trong `compare/results/`  
- [ ] Nghe thử vài MIDI cùng prompt  
- [ ] Bảng so sánh params / time / quality (xem `comparison_table.csv`)  

---

## 7. Pipeline nhanh — chạy local trên macOS (`.venv`)

Từ build data đến test sinh nhạc, dùng `.venv` có sẵn ở gốc repo. Train thật nên
chạy trên GPU thuê (Vast — xem `train_watchdog.sh` để tự resume khi crash); trên
Mac (CPU/MPS) chỉ nên chạy smoke test trước khi đẩy lên GPU.

### 7.1. Setup

```bash
cd /Users/tranbadat/Documents/study-projects/GenAi-Text-To-Music
source .venv/bin/activate
pip install -r GenAI_Transformer/requirements.txt
python -c "import torch; print('mps:', torch.backends.mps.is_available(), '| cuda:', torch.cuda.is_available())"
```

### 7.2. Build data

Chỉ cần chạy lại khi sửa `data/labels/labels.json` hoặc `compare/caption.py`:

```bash
python scripts/build_captions.py   # → data/labels/captions.json (5 caption/file, xem compare/caption.py)
python -m compare.make_split        # chỉ khi compare/split.json chưa có / muốn tạo lại
```

### 7.3. Smoke test (bắt buộc trước khi train thật)

```bash
cd GenAI_Transformer
python train.py --epochs 2 --batch_size 4 --max_seq_len 512 --max_files 500 --no_early_stop --device mps

cd ../GenAI_Diffusion
python train.py --epochs 2 --batch_size 2 --max_files 300
```

### 7.4. Train thật

```bash
cd GenAI_Transformer
python train.py --epochs 30 --batch_size 8 --max_seq_len 1024 --no_early_stop --save_epochs "1,5,10,20,30"
# Vast crash → bash train_watchdog.sh để tự resume

cd ../GenAI_Diffusion
python train.py --epochs 30 --batch_size 4
```

### 7.5. Test sinh nhạc bằng prompt tự do

```bash
cd GenAI_Transformer
python generate.py --prompt "Happy fantasy village music, fast tempo, piano, medium energy" \
  --duration_sec 30 --checkpoint checkpoints/best_model.pt

cd ../GenAI_Diffusion
python generate.py --checkpoint checkpoints/best_model.pt --epoch 30 \
  --duration_sec 12 --guidance_scale 3.5 --sample_steps 80 --evaluate
```

### 7.6. Đánh giá + so sánh

```bash
cd /Users/tranbadat/Documents/study-projects/GenAi-Text-To-Music
python -m compare.run_comparison_eval --epochs 1 5 10
python -m compare.compare_results
# → compare/results/comparison_table.csv
```

### 7.7. (Tuỳ chọn) Web demo

```bash
cd GenAI_Transformer
uvicorn api.main:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs — POST /generate {"prompt": "..."}
```

> **Checkpoint cũ (kiểu 6-attribute, trước bản MiniLM) không tương thích —**
> **bước 7.4 bắt buộc train lại từ đầu, không load lại được checkpoint cũ.**

---

*Cập nhật theo cấu trúc repo: data ngoài `GenAI_Transformer/`, docs trong `docs/`.*
