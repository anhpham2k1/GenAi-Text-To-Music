# Text-to-Music — So sánh Transformer vs Diffusion

Đồ án **Generative AI**: sinh nhạc nền game từ **prompt có cấu trúc** (6 thuộc tính), so sánh hai phương pháp trên **cùng dataset**.

| Method | Thư mục | Ý tưởng (slide IT5410) |
|--------|---------|------------------------|
| **Music Transformer** | `GenAI_Transformer/` | Week4 + Week6 — autoregressive token REMI |
| **Piano-roll Diffusion** | `GenAI_Diffusion/` | Week8 — DDPM/DDIM + CFG |
| **Data dùng chung** | `data/` | raw / processed / labels |
| **So sánh + CSV** | `compare/` | split, eval prompts, metrics |
| **Tài liệu** | `docs/` | HDSD, train, research, slide map |

**Input chung:** **một câu English** (free-text) → MiniLM freeze + projection.  
Ví dụ: `"Happy fantasy village music, fast tempo, piano, medium energy"`  
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

## Generate

```powershell
# (Sau khi train lại với text conditioning)

# Transformer
cd D:\Master\Ky3\GenAI_Transformer
python generate.py --prompt "Happy fantasy village music, fast tempo, piano, medium energy" --duration_sec 30

# Diffusion
cd D:\Master\Ky3\GenAI_Diffusion
python generate.py --checkpoint checkpoints/best_model.pt --epoch 10 `
  --prompt "Tense horror dungeon theme, slow organ, high energy" --duration_sec 12 --evaluate
```

Build captions (optional, từ labels):

```powershell
cd D:\Master\Ky3
python scripts/build_captions.py
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
