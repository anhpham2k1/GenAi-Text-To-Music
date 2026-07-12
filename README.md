# Text-to-Music — So sánh Transformer vs Diffusion

Đồ án **Generative AI**: sinh nhạc nền game từ **prompt có cấu trúc** (6 thuộc tính), so sánh hai phương pháp trên **cùng dataset**.

| Method | Thư mục | Ý tưởng (slide IT5410) |
|--------|---------|------------------------|
| **Music Transformer** | `GenAI_Transformer/` | Week4 + Week6 — autoregressive token REMI |
| **Piano-roll Diffusion** | `GenAI_Diffusion/` | Week8 — DDPM/DDIM + CFG |
| **Data dùng chung** | `data/` | raw / processed / labels |
| **So sánh + CSV** | `compare/` | split, eval prompts, metrics |
| **Tài liệu** | `docs/` | HDSD, train, research, slide map |

**Input chung:** `mood, genre, scene, tempo, instrument, energy` (không BERT).  
**Output chung:** file **MIDI** (`.mid`).

---

## Cấu trúc thư mục

```
Ky3/
├── README.md                 ← file này
├── docs/                     ← toàn bộ tài liệu
│   ├── TRAINING_GUIDE.md     ← hướng dẫn train 2 model + GPU
│   ├── DATA.md
│   ├── COMPARE.md
│   └── ...
├── data/                     ← dataset (KHÔNG nằm trong GenAI_Transformer/)
│   ├── raw/                  # MAESTRO, MidiCaps, ComMU, ...
│   ├── processed/            # ~14k MIDI train
│   └── labels/labels.json
├── GenAI_Transformer/                    # Transformer
├── GenAI_Diffusion/          # Diffusion
├── compare/                  # eval + CSV results
├── scripts/                  # merge_commu, tiện ích
└── slide/                    # slide môn học (PDF)
```

---

## Cài đặt

```powershell
cd D:\Master\Ky3\GenAI_Transformer
pip install -r requirements.txt
# PyTorch CUDA: cài đúng bản GPU máy bạn (cu118/cu121/...)
```

Cần: Python 3.10+, NVIDIA GPU (khuyến nghị), `pretty_midi`, `torch`.

---

## Train nhanh

Chi tiết + batch size theo card: **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**

```powershell
# Transformer
cd D:\Master\Ky3\GenAI_Transformer
python train.py --epochs 10 --batch_size 8 --max_seq_len 1024 --no_early_stop

# Diffusion
cd D:\Master\Ky3\GenAI_Diffusion
python train.py --epochs 10 --batch_size 4
```

---

## Vì sao đổi config để train lại (Transformer)

Lần train đầu (50 epoch, V100) kết thúc với:

```
Epoch  50/50 │ Train Loss: 3.8750 │ Val Loss: 3.9905 │ LR: 1.00e-06 │ PPL: 54.1
```

**Chẩn đoán: under-fit, KHÔNG phải over-fit.**

| Dấu hiệu | Số liệu | Ý nghĩa |
|----------|---------|---------|
| Khoảng cách train/val | chỉ **0.115** (3.875 vs 3.990) | Nếu over-fit, val loss phải **tăng** khi train loss giảm. Ở đây val loss giảm đều tới epoch cuối, epoch 50 vẫn là "new best" |
| Loss cuối | PPL **54.1**, còn cao | Hai loss bám sát nhau và đều cao → model **chưa đủ sức học**, không phải học thuộc lòng |
| LR cuối | chạm đáy **1.0e-06** | 5 epoch cuối chỉ cải thiện val loss **0.0002** → model đứng yên vì **hết learning rate**, không phải vì "chưa train xong" |

→ Tăng epoch suông **không giải quyết được gì**: LR đã về 0, model không học thêm nữa.

Lưu ý: `num_epochs` **định hình luôn lịch LR** (warmup + cosine tính trên tổng số step). Nên tăng `num_epochs` không phải là "train thêm", mà là **giãn lịch cosine ra**, giữ LR ở mức cao lâu hơn — đó mới là chỗ có tác dụng.

### Thay đổi

| Config | Cũ | Mới | Lý do |
|--------|-----|-----|-------|
| `d_model` | 256 | **512** | Tăng capacity — đòn bẩy mạnh nhất, đánh trúng nguyên nhân under-fit |
| `num_layers` | 6 | **8** | 〃 |
| `d_ff` | 1024 | **2048** | 〃 |
| `num_heads` | 8 | **8** (giữ) | `d_model` phải chia hết cho `num_heads`; 512/8 = 64 dim/head là chuẩn. Tăng head mà không tăng `d_model` chỉ làm mỗi head *hẹp* đi |
| `learning_rate` | 1.0e-4 | **3.0e-4** | Quan trọng ngang việc tăng model. LR cũ quá thấp, chạm đáy rồi đứng yên |
| `num_epochs` | 50 | **100** | Giãn lịch cosine, giữ LR cao lâu hơn |
| `batch_size` | 16 | **32** | V100 32GB dư sức (bản cũ 16 là để né card 6GB) |
| `gradient_accumulation_steps` | 4 | **2** | 32×2 = **effective batch 64, giữ nguyên** như cũ |
| `early_stopping_patience` | 10 | **10** (giữ) | Lưới an toàn: nếu over-fit thật, tự dừng, không tốn giờ GPU oan |

Model: **7.65M → ~28M params**.

Không dùng `--batch_size 64 --grad_accum 1` ngay từ đầu vì attention tốn bộ nhớ theo **bình phương** seq length (2048), mà model vừa to gấp đôi. Nếu chạy trơn và VRAM còn dư nhiều thì đổi sang 64+1 để nhanh hơn.

### Chạy lại

Phải train **từ đầu**, KHÔNG `--resume`: checkpoint cũ có kiến trúc khác (`d_model` 256), load sẽ lỗi shape mismatch.

```bash
cd GenAI_Transformer
python3 train.py --max_files 500 --epochs 2 --num_workers 8   # smoke test, bắt OOM sớm
python3 train.py --num_workers 8                              # train thật
```

Nếu `CUDA out of memory` → hạ `--batch_size 16 --grad_accum 4`.

> **Bẫy đã gặp:** thiếu `pretty_midi` thì **mọi file MIDI fail tokenize âm thầm** (`dataset.py` chỉ warn rồi bỏ qua), model học chuỗi rỗng và cho val loss ~1.0 / PPL ~2.7 — **đẹp giả tạo**. Loss thật phải bắt đầu quanh **6.1**. Luôn smoke test trước khi đốt giờ GPU.

---

## Generate

```powershell
# Transformer (~30s)
cd D:\Master\Ky3\GenAI_Transformer
python generate.py --mood happy --genre fantasy --scene village --tempo fast --instrument piano --duration_sec 30

# Diffusion (~10s mặc định config)
cd D:\Master\Ky3\GenAI_Diffusion
python generate.py --checkpoint checkpoints/best_model.pt --epoch 10 --duration_sec 12 --evaluate
```

## So sánh epoch 1 / 5 / 10

```powershell
cd D:\Master\Ky3
python -m compare.run_comparison_eval --epochs 1 5 10
python -m compare.compare_results
# → compare/results/comparison_table.csv
```

## API (tùy chọn)

```powershell
cd D:\Master\Ky3\GenAI_Transformer
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

---

## Dataset (tóm tắt)

| Nguồn | Trong train (processed) |
|-------|-------------------------|
| ComMU | ~11 144 |
| MidiCaps (một phần) | ~2 679 |
| MAESTRO (một phần) | ~360 |
| **Tổng** | **~14 232** (train 12 809 / val 1 423) |

Chi tiết: [docs/DATA.md](docs/DATA.md)

---

## Tài liệu trong `docs/`

| File | Nội dung |
|------|----------|
| [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Train 2 model, thông số **4060 / 5060 / 5060 Ti** |
| [DATA.md](docs/DATA.md) | Cấu trúc data, merge |
| [COMPARE.md](docs/COMPARE.md) | Protocol so sánh + CSV |
| [IO_CONTRACT.md](docs/IO_CONTRACT.md) | Input/output chung |
| [SLIDE_ALIGNMENT.md](docs/SLIDE_ALIGNMENT.md) | Map slide Week8 |
| [RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md) | Báo cáo nghiên cứu |
| [HDSD_DATASET.md](docs/HDSD_DATASET.md) | HDSD dataset cũ |

---

## License

Đồ án nghiên cứu — Generative AI / IT5410.
