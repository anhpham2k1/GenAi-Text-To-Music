# Hợp đồng Input / Output chung (Transformer ≡ Diffusion)

Sau khi **bỏ BERT**, hai model dùng **cùng input schema** và **cùng loại output chính**.

## INPUT (identical)

| Field | Type | Ví dụ |
|-------|------|--------|
| mood | string → id | happy, sad, … |
| genre | string → id | fantasy, rpg, … |
| scene | string → id | village, dungeon, … |
| tempo | string → id | slow, fast, … |
| instrument | string → id | piano, strings, … |
| energy | string → id | calm, high, … |

- Map ID: `compare/prompt_schema.py`
- Prompt eval cố định: `compare/eval_prompts.json`
- **Không** free-text, **không** BERT

## OUTPUT (identical primary)

| Artifact | Transformer | Diffusion |
|----------|-------------|-----------|
| **MIDI `.mid`** | ✅ REMI decode | ✅ piano-roll decode |
| WAV `.wav` | Optional (FluidSynth) | Optional (cùng renderer nếu cần) |
| Quality metrics | `compare.evaluate` | `compare.evaluate` |

So sánh công bằng = **cùng 20 prompt structured** → **cùng folder MIDI** → **cùng CSV metrics**.

## CLI

```powershell
# Transformer
cd GenAI_Transformer
python generate.py --mood happy --genre fantasy --scene village --tempo fast --instrument piano --energy medium

# Diffusion
cd GenAI_Diffusion
python generate.py --checkpoint checkpoints/best_model.pt --epoch 10
# (đọc eval_prompts.json — cùng 6 fields)
```
