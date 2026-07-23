# Hợp đồng Input / Output chung (Transformer ≡ Diffusion)

Hai model dùng **cùng input schema** (English free-text) và **cùng loại output chính**.

## INPUT (identical)

| Field | Type | Ví dụ |
|-------|------|--------|
| **prompt** / **text** | string (English) | `"Happy fantasy village music, fast tempo, piano, medium energy"` |

- Encoder: **frozen MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) + trainable projection  
- Module dùng chung: `compare/text_conditioner.py` (`TextPromptEncoder`)  
- Caption train từ labels: `compare/caption.py` → `labels_to_english_caption`  
- Prompt eval: `compare/eval_prompts.json` (field `text` + structured metadata cho metrics)  
- **Không** còn 6 ID embedding là cổng chính của model

### Structured fields (phụ)

6 attribute (`mood, genre, scene, tempo, instrument, energy`) vẫn có trong:

- `labels.json` → build caption English lúc train  
- eval meta → `instrument_match` / program MIDI  

Không đi vào model dưới dạng `nn.Embedding` riêng.

## OUTPUT (identical primary)

| Artifact | Transformer | Diffusion |
|----------|-------------|-----------|
| **MIDI `.mid`** | ✅ REMI decode | ✅ piano-roll decode |
| WAV `.wav` | Optional (FluidSynth) | Optional |
| Quality metrics | `compare.evaluate` | `compare.evaluate` |

So sánh công bằng = **cùng 20 câu English** → **cùng folder MIDI** → **cùng CSV metrics**.

## CLI

```powershell
# Transformer
cd GenAI_Transformer
python generate.py --prompt "Happy fantasy village music, fast tempo, piano, medium energy"

# Diffusion
cd GenAI_Diffusion
python generate.py --checkpoint checkpoints/best_model.pt --epoch 10 `
  --prompt "Tense horror dungeon theme, slow organ, high energy"
```

## Lưu ý checkpoint

Checkpoint **structured 6-ID cũ không tương thích**. Cần **train lại** cả 2 model với text conditioning.
