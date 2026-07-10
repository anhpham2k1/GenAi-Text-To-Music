# Text-to-Music: Sinh Nhạc Nền Cho Trò Chơi Bằng Generative AI

Hệ thống AI sinh nhạc nền cho game từ mô tả văn bản, sử dụng **Music Transformer** với **Relative Position Encoding** và **Cross-Attention Text Conditioning**.

## ✨ Features

- **Text-to-Music Generation**: Sinh nhạc MIDI từ mô tả game prompt (mood, genre, scene, tempo, instrument, energy)
- **Music Transformer**: Mô hình tự xây dựng (~7.7M parameters) với RoPE và Cross-Attention
- **REMI Tokenization**: Biểu diễn MIDI hiệu quả (~336 tokens vocabulary)
- **Multiple Sampling**: Temperature, Top-k, Top-p (Nucleus) sampling
- **MIDI → WAV**: Render audio bằng FluidSynth + SoundFont
- **Web Demo**: FastAPI backend + Premium dark-theme frontend
- **Auto-labeling**: Tự động gán labels cho MIDI files

## 🏗️ Architecture

```
User Prompt → Text Encoder → Cross-Attention
                                    ↓
          [BOS] → Token Embedding → Decoder (×6) → Linear → Softmax → Next Token
                   (+ Rotary Position Encoding - RoPE)
                                    ↓
                              MIDI Tokens → MIDI File → FluidSynth → WAV
```

## 📁 Project Structure

```
text-to-music/
├── config/config.yaml           # Hyperparameters
├── src/
│   ├── data/
│   │   ├── tokenizer.py         # REMI MIDI tokenizer
│   │   ├── dataset.py           # PyTorch Dataset + DataLoader
│   │   └── preprocessing.py     # Data filtering & labeling
│   ├── model/
│   │   ├── embedding.py         # Token + Relative Position embeddings
│   │   ├── attention.py         # Self-Attention (RoPE, GQA) + Cross-Attention
│   │   ├── layers.py            # FFN + DecoderBlock
│   │   ├── prompt_encoder.py    # Attribute-based prompt encoder
│   │   └── transformer.py       # Full Music Transformer model
│   ├── training/
│   │   └── trainer.py           # Training loop (AdamW, Cosine LR, etc.)
│   ├── inference/
│   │   ├── sampling.py          # Sampling strategies
│   │   ├── generator.py         # Autoregressive generation
│   │   └── renderer.py          # MIDI → WAV (FluidSynth)
│   └── utils/
│       └── visualization.py     # Piano roll & training curves
├── api/
│   ├── main.py                  # FastAPI backend
│   └── schemas.py               # Request/Response models
├── frontend/
│   ├── index.html               # Web UI
│   ├── style.css                # Premium dark theme
│   └── app.js                   # Frontend logic
├── train.py                     # Training entry point
├── generate.py                  # Generation entry point
├── requirements.txt             # Dependencies
└── Dockerfile                   # Container
```

## 🎯 Prompt System (Mô tả nhạc)

Hệ thống sinh nhạc nhận vào một bộ **Prompt có cấu trúc** (Structured Prompt) gồm 6 thuộc tính. Điều này giúp AI hiểu chính xác ngữ cảnh của trò chơi:

| Thuộc tính | Ý nghĩa | Các giá trị hỗ trợ (Tùy chọn) |
|---|---|---|
| **🎭 Mood** | Tâm trạng | `happy`, `sad`, `tense`, `peaceful`, `epic`, `mysterious`, `dark`, `heroic`, `nostalgic`, `playful` |
| **🕹️ Genre** | Thể loại game | `fantasy`, `sci-fi`, `horror`, `adventure`, `rpg`, `puzzle`, `platformer`, `simulation`, `fighting`, `racing` |
| **🏔️ Scene** | Bối cảnh | `forest`, `dungeon`, `village`, `castle`, `ocean`, `space`, `mountain`, `desert`, `city`, `battlefield` |
| **⏱️ Tempo** | Tốc độ | `very_slow`, `slow`, `moderate`, `fast`, `very_fast` |
| **🎻 Instrument**| Nhạc cụ chính | `piano`, `strings`, `brass`, `flute`, `guitar`, `organ`, `synth`, `full_orchestra` |
| **⚡ Energy** | Độ kịch tính | `calm`, `low`, `medium`, `high`, `intense` |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Tạo thư mục
mkdir -p data/raw data/processed data/labels

# Xem và tải dataset (hỗ trợ nhiều nguồn)
python -m src.data.download_dataset

# Hoặc thủ công: bỏ thêm MIDI vào data/raw/ (subfolders như game_midi/, lakh/ rất tốt)
# Sau đó chạy:
python -c "
from src.data.preprocessing import filter_midi_files, generate_labels
filter_midi_files('data/raw', 'data/processed')
generate_labels('data/processed', 'data/labels/labels.json')
"

# Khuyến nghị COMBO (theo yêu cầu của bạn):
# VGMusic (game-specific) + MidiCaps (có text caption sẵn) 
# + ComMU (có label có cấu trúc) + Tegridy/GigaMIDI (đa dạng + volume)
#
# - Tải vào data/raw/ (dùng subfolder)
# - Chạy python -m src.data.download_dataset hoặc filter + labels
# - Giữ MAESTRO nếu cần.
# Xem: python -m src.data.preprocessing
```

### 3. Train Model

```bash
python train.py --data_dir data/processed --epochs 50 --batch_size 16

# Hoặc với custom settings:
python train.py --lr 0.0001 --max_seq_len 1024 --max_files 1000
```

### 4. Generate Music

```bash
python generate.py \
    --mood happy --genre fantasy --scene village \
    --tempo fast --instrument piano --energy medium \
    --checkpoint checkpoints/best_model.pt \
    --piano_roll

# Or resume training
python train.py --resume checkpoints/best_model.pt --epochs 100
```

### 5. Run Web Demo

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Open: http://localhost:8000
```

## 🐳 Docker

```bash
docker build -t text-to-music .
docker run -p 8000:8000 text-to-music
```

## ⚙️ Configuration

Tất cả hyperparameters trong `config/config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| d_model | 256 | Model dimension |
| num_heads | 8 | Attention heads |
| num_layers | 6 | Decoder blocks |
| d_ff | 1024 | FFN hidden dim |
| max_seq_len | 2048 | Max sequence length |
| learning_rate | 1e-4 | Initial LR |
| batch_size | 16 | Batch size per GPU |
| temperature | 0.85 | Sampling temperature |
| top_p | 0.9 | Nucleus sampling |

## 📊 Model Summary

- **Architecture**: LLaMA-style Conditional Music Transformer (Decoder-only)
  - **Normalization**: RMSNorm (Root Mean Square Normalization)
  - **Feed-Forward**: SwiGLU (Swish-Gated Linear Unit)
- **Parameters**: ~7.7M
- **Attention**: FlashAttention (SDPA) + Rotary Position Encoding (RoPE)
- **Advanced Optimizations**: GQA (Grouped Query Attention), QK-Norm, Weight Tying
- **Conditioning**: Cross-Attention on **structured attributes only** (mood/genre/scene/tempo/instrument/energy) — **no BERT**; same input schema as Diffusion
- **Tokenizer**: REMI (~336 vocab)
- **GPU Requirement**: Tối ưu hóa cực tốt. Có thể train dễ dàng trên RTX 4060 (8GB VRAM) hoặc Colab T4.

## 📚 References

1. Huang et al. (2018). *Music Transformer: Generating Music with Long-Term Structure*
2. Vaswani et al. (2017). *Attention Is All You Need*
3. Huang & Yang (2020). *Pop Music Transformer* (REMI representation)
4. Holtzman et al. (2020). *The Curious Case of Neural Text Degeneration* (Nucleus sampling)

## 📄 License

Đồ án nghiên cứu — Generative AI.
