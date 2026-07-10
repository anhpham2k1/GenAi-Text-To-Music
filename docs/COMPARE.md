# So sánh Transformer vs Diffusion (cùng dataset)

## Cấu trúc

```
Ky3/
├── GenAI_Transformer/                 # Music Transformer (REMI tokens)
├── GenAI_Diffusion/       # Piano-roll Diffusion (cùng data/labels)
└── compare/
    ├── split.json           # train/val cố định (tạo 1 lần)
    ├── eval_prompts.json    # 20 prompt chuẩn
    ├── results/             # ★ tất cả CSV log
    │   ├── model_summary.csv
    │   ├── transformer_training.csv
    │   ├── diffusion_training.csv
    │   ├── transformer_quality_epoch_XXX.csv
    │   ├── diffusion_quality_epoch_XXX.csv
    │   ├── quality_summary.csv
    │   └── comparison_table.csv
    └── outputs/
        ├── transformer/epoch_{1,5,10}/
        └── diffusion/epoch_{1,5,10}/
```

## CSV nào dùng để báo cáo?

| File | Nội dung |
|------|----------|
| `model_summary.csv` | #params, GPU, batch size, #samples |
| `*_training.csv` | mỗi epoch: train/val loss, time, VRAM |
| `*_quality_epoch_XXX.csv` | từng file MIDI: density, silence, instrument_match, … |
| `quality_summary.csv` | trung bình metric theo method+epoch |
| `comparison_table.csv` | bảng gộp final cho slide/báo cáo |

**Lưu ý:** Không so trực tiếp `val_loss` CE (Transformer) với MSE (Diffusion). So **quality metrics** + **time/VRAM/params**.

## Pipeline chạy

### 0. Tạo split chung (1 lần)

```powershell
cd D:\Master\Ky3
python -m compare.make_split
```

### 1. Train Transformer (log CSV tự động)

```powershell
cd D:\Master\Ky3\GenAI_Transformer
python train.py --epochs 10 --batch_size 8 --max_seq_len 1024
# CSV: ../compare/results/transformer_training.csv
# CKPT: checkpoints/checkpoint_epoch_1.pt, _5, _10 + best_model.pt
```

### 2. Train Diffusion (cùng data)

```powershell
cd D:\Master\Ky3\GenAI_Diffusion
python train.py --epochs 10 --batch_size 8
# CSV: ../compare/results/diffusion_training.csv
```

Smoke test nhanh:

```powershell
python train.py --epochs 2 --max_files 200 --batch_size 4
```

### 3. Generate + evaluate epoch 1/5/10

```powershell
cd D:\Master\Ky3
python -m compare.run_comparison_eval --epochs 1 5 10
```

Hoặc từng bên:

```powershell
cd GenAI_Transformer
python generate_eval.py --checkpoint checkpoints/checkpoint_epoch_5.pt --epoch 5 --evaluate

cd ..\GenAI_Diffusion
python generate.py --checkpoint checkpoints/checkpoint_epoch_5.pt --epoch 5 --evaluate
```

### 4. Gộp bảng so sánh

```powershell
cd D:\Master\Ky3
python -m compare.compare_results
```

## Metric chất lượng (cột chính)

- `note_density` — nốt/giây  
- `silence_ratio` — % im lặng  
- `polyphony_mean` — độ dày hợp âm  
- `pitch_range`, `pitch_mean`  
- `instrument_match` — khớp program với prompt  
- `has_bass_track` — có track bass không  
- `pitch_class_js_vs_ref` — khoảng cách histogram so train  
- `infer_time_per_sample_sec`  
- `epoch_time_sec`, `peak_vram_mb`, `params_m`  

## Công bằng so sánh

1. Cùng `data/processed` + `labels.json`  
2. Cùng `compare/split.json`  
3. Cùng `eval_prompts.json`  
4. Checkpoint đúng epoch 1, 5, 10  
5. Cùng script `compare.evaluate`  
