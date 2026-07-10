# GenAI_Diffusion — Conditional Piano-roll Diffusion (improved)

So sánh với `GenAI_Transformer` (Music Transformer), **cùng dataset**.

- Data: `../data/processed` (shared `Ky3/data`)
- Labels: `../data/labels/labels.json`
- Split: `../compare/split.json`
- CSV: `../compare/results/`

## Pipeline

1. MIDI → piano-roll `(1, 88, 256)` @ 24 Hz (~10.6s), `[-1, 1]`
2. DDPM + **cosine schedule**, train noise MSE + **CFG** (`cond_drop=0.1`)
3. UNet + FiLM condition + **mid self-attention** + **EMA**
4. DDIM sample (`guidance_scale≈3.5`, 80 steps) → decode cleanup → MIDI

Chi tiết cải tiến: [`IMPROVEMENTS.md`](IMPROVEMENTS.md)

## Chạy

```powershell
cd D:\Master\Ky3
python -m compare.make_split

cd GenAI_Diffusion
# Quality: 30 epochs (10 chỉ đủ so sánh)
python train.py --epochs 30 --batch_size 4

python generate.py --checkpoint checkpoints/best_model.pt --epoch 30 `
  --guidance_scale 3.5 --sample_steps 80 --evaluate
```

OOM → `--batch_size 2` hoặc giảm `n_frames` / `base_channels` trong `config/config.yaml`.

Xem thêm: `../compare/README.md`
