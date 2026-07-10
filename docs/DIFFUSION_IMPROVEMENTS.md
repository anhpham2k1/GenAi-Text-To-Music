# Cải tiến Diffusion để gen nhạc chuẩn / ngon hơn

## Vấn đề baseline cũ

| Vấn đề | Hậu quả |
|--------|---------|
| Roll ngắn 128 @ 16Hz (~8s) | Nhạc cụt, nhịp thô |
| Crop luôn từ đầu file | Model chỉ học intro |
| Linear schedule, no CFG | Condition yếu, “noise piano-roll” |
| No EMA | Sample không ổn định |
| UNet nhỏ, không attention | Không bắt được phrase dài |
| Decode threshold cứng | Nốt rời / lủng lỗ / rè |

## Đã implement trong code

1. **Cosine beta schedule** — noise schedule mượt hơn linear  
2. **Classifier-Free Guidance (CFG)** — train `cond_drop_prob=0.1`, infer `guidance_scale≈3.5`  
3. **EMA 0.999** — checkpoint/sample dùng trọng số EMA  
4. **Mid self-attention** trong UNet — quan hệ pitch–time dài hơn  
5. **Model lớn hơn** (`base_channels=64`, `cond_dim=256`)  
6. **Piano-roll dày hơn** (`n_frames=256`, `fs=24` ≈ 10.6s)  
7. **Random crop + pitch shift ±2** khi train  
8. **Decode cleanup** — bỏ speckles, min duration, threshold thích nghi, program theo prompt  

## Cách train / gen (khuyến nghị)

```powershell
cd D:\Master\Ky3\GenAI_Diffusion

# Chất lượng: 30 epoch (10 epoch chỉ đủ so sánh, chưa “ngon”)
python train.py --epochs 30 --batch_size 4

# Nếu OOM: giảm batch hoặc frames trong config
python train.py --epochs 30 --batch_size 2

# Generate — chỉnh guidance nếu cần
python generate.py --checkpoint checkpoints/best_model.pt --epoch 30 `
  --guidance_scale 3.5 --sample_steps 80 --evaluate
```

**Guidance scale:**
- `2.0` — tự nhiên hơn, bám prompt vừa  
- `3.5` — mặc định cân bằng  
- `5.0+` — bám prompt mạnh, dễ “cứng” / artifact  

## Hướng cải tiếp (chưa làm — nếu còn thời gian)

| Ưu tiên | Ý tưởng | Ghi chú |
|--------|---------|--------|
| Cao | Train **50–100 epoch** + early stop theo val MSE | Data ~3k vẫn cần nhiều pass |
| Cao | **Lọc piano-only** khi so “happy piano” | Roll gộp multi-track → bass lẫn melody |
| Trung | Onset + frame 2-channel (như onsets-and-frames) | Rõ nốt hơn velocity-only |
| Trung | **Discrete diffusion / token diffusion** cùng REMI | So fairer với Transformer |
| Trung | LR cosine decay | Ổn định late training |
| Thấp | DiT / latent diffusion trên VAE roll | Nặng hơn, paper-level |
| Thấp | Multi-instrument channels | 16 program groups |

## Kỳ vọng thực tế

- Diffusion piano-roll **khó** bằng Music Transformer về “giai điệu có truyện” nếu chỉ 10 epoch.  
- Với CFG + EMA + 30 epoch, kỳ vọng: **có nốt rõ, đúng program, density hợp lý**, phrase ngắn ổn.  
- Muốn BGM “nghe được lâu”: ưu tiên **data sạch (piano/game)** + **nhiều epoch** hơn là chỉ tăng params.

## So sánh với Transformer sau cải tiến

Vẫn dùng `compare/` CSV.  
**Re-train diffusion** với config mới trước khi so epoch 1/5/10 (checkpoint cũ **không** load được architecture mới nếu đổi `base_channels` / `cond_dim` — generate đã `strict=False` nhưng nên train lại).
