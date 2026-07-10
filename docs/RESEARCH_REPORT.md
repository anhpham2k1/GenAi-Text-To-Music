# Text-to-Music: Sinh Nhạc Nền Cho Trò Chơi Bằng Generative AI

## Báo Cáo Nghiên Cứu Hoàn Chỉnh

> **Lĩnh vực:** Generative AI — Music Generation  
> **Chủ đề:** Xây dựng hệ thống AI sinh nhạc nền cho game từ mô tả văn bản  
> **Framework:** PyTorch  
> **Ngày:** Tháng 7/2026

---

# Mục Lục

1. [Phân tích bài toán](#1-phân-tích-bài-toán)
2. [Khảo sát các hướng nghiên cứu](#2-khảo-sát-các-hướng-nghiên-cứu)
3. [Thiết kế kiến trúc hệ thống](#3-thiết-kế-kiến-trúc-hệ-thống)
4. [Thiết kế mô hình](#4-thiết-kế-mô-hình)
5. [Dataset](#5-dataset)
6. [Tiền xử lý dữ liệu](#6-tiền-xử-lý-dữ-liệu)
7. [Thiết kế Prompt](#7-thiết-kế-prompt)
8. [Thiết kế quá trình Training](#8-thiết-kế-quá-trình-training)
9. [Sinh nhạc (Inference)](#9-sinh-nhạc-inference)
10. [Chuyển MIDI sang WAV](#10-chuyển-midi-sang-wav)
11. [Đánh giá mô hình](#11-đánh-giá-mô-hình)
12. [Thiết kế Demo](#12-thiết-kế-demo)
13. [Công nghệ sử dụng](#13-công-nghệ-sử-dụng)
14. [Lộ trình thực hiện](#14-lộ-trình-thực-hiện)
15. [Hướng phát triển](#15-hướng-phát-triển)

---

# 1. Phân Tích Bài Toán

## 1.1. Đây là bài toán gì?

Bài toán **Text-to-Music** thuộc nhóm **Conditional Sequence Generation** — một dạng bài toán trong đó mô hình AI nhận đầu vào là một chuỗi điều kiện (condition) dưới dạng văn bản tự nhiên và sinh ra đầu ra là một chuỗi sự kiện âm nhạc (music sequence).

Cụ thể hơn, đây là bài toán **Cross-modal Generation** (sinh dữ liệu liên phương thức):

```
┌─────────────┐          ┌──────────────┐          ┌──────────────┐
│  Text Mode  │  ─────►  │   AI Model   │  ─────►  │  Music Mode  │
│ (Language)  │          │ (Generator)  │          │   (Audio)    │
└─────────────┘          └──────────────┘          └──────────────┘
```

**Đặc trưng của bài toán:**

| Khía cạnh | Mô tả |
|---|---|
| **Input** | Mô tả văn bản (mood, genre, tempo, instrument, scene) |
| **Output** | Chuỗi sự kiện MIDI → File WAV |
| **Bản chất** | Sequence-to-Sequence với cross-modal conditioning |
| **Không gian đầu ra** | Rời rạc (MIDI tokens) hoặc liên tục (audio waveform) |
| **Tính sáng tạo** | Mô hình phải "sáng tác" — không phải truy xuất hay sao chép |
| **Ràng buộc** | Phải tuân thủ cấu trúc âm nhạc (harmony, rhythm, melody) |

## 1.2. Text-to-Music khác Music Generation như thế nào?

| Tiêu chí | Music Generation (Unconditional) | Text-to-Music (Conditional) |
|---|---|---|
| **Đầu vào** | Không có hoặc seed ngẫu nhiên | Mô tả văn bản chi tiết |
| **Kiểm soát** | Rất ít — mô hình tự quyết định | Người dùng kiểm soát mood, genre, tempo, instrument |
| **Kiến trúc** | Decoder-only (GPT-like) | Encoder-Decoder hoặc Cross-attention conditioning |
| **Ứng dụng** | Sinh nhạc ngẫu nhiên, sáng tác tự do | Sinh nhạc theo yêu cầu cụ thể cho game, phim, quảng cáo |
| **Đánh giá** | Chất lượng âm nhạc tổng thể | Chất lượng + Mức độ phù hợp với mô tả |
| **Độ khó** | Trung bình | Cao hơn — cần alignment giữa text và music |

> [!IMPORTANT]
> Điểm khác biệt cốt lõi: Text-to-Music yêu cầu mô hình **hiểu ngữ nghĩa** của văn bản (ví dụ: "happy" → major key, fast tempo; "dark dungeon" → minor key, slow, low pitch) và **ánh xạ** ngữ nghĩa đó sang không gian âm nhạc.

## 1.3. Vì sao phù hợp với Generative AI?

**Lý do 1: Bản chất sáng tạo**

Âm nhạc là nghệ thuật sáng tạo — không có một đáp án duy nhất đúng cho mỗi mô tả. Với prompt "happy fantasy village music", có vô số bản nhạc hợp lệ có thể sinh ra. Đây chính là thế mạnh của Generative AI: mô hình học phân phối xác suất của dữ liệu và **lấy mẫu** (sampling) từ phân phối đó để tạo ra đầu ra đa dạng.

**Lý do 2: Cấu trúc chuỗi**

Âm nhạc có cấu trúc tuần tự (sequential): note theo note, beat theo beat, measure theo measure. Điều này hoàn toàn phù hợp với các mô hình sinh chuỗi như **Transformer**, **RNN**, hay **Diffusion Models** trên chuỗi.

**Lý do 3: Dữ liệu phong phú**

Có hàng triệu file MIDI công khai, cung cấp đủ dữ liệu để huấn luyện mô hình sinh nhạc.

**Lý do 4: Nhu cầu thực tế trong game development**

- Indie game developers cần nhạc nền nhưng không có ngân sách thuê nhạc sĩ
- AAA studios cần prototype nhạc nhanh trong giai đoạn thiết kế
- Adaptive music systems cần sinh nhạc real-time theo trạng thái game

**Các nghiên cứu tiền đề:**

| Công trình | Năm | Đóng góp |
|---|---|---|
| Music Transformer (Huang et al.) | 2018 | Relative attention cho music generation |
| MuseNet (OpenAI) | 2019 | Transformer lớn sinh nhạc đa nhạc cụ |
| Jukebox (Dhariwal et al.) | 2020 | VQ-VAE + Transformer sinh raw audio |
| MusicLM (Agostinelli et al.) | 2023 | Text-to-Music với hierarchical generation |
| MusicGen (Copet et al., Meta) | 2023 | Single-stage Transformer cho text-to-music |
| Stable Audio (Stability AI) | 2023 | Latent Diffusion cho audio generation |

---

# 2. Khảo Sát Các Hướng Nghiên Cứu

## 2.1. Phân tích chi tiết từng hướng

### A. Transformer (Vanilla)

**Mô tả:** Kiến trúc Transformer gốc (Vaswani et al., 2017) với self-attention mechanism. Áp dụng cho music generation bằng cách coi MIDI events như một chuỗi tokens.

**Ưu điểm:**
- Kiến trúc đơn giản, dễ hiểu và triển khai
- Bắt được long-range dependencies tốt hơn RNN
- Có thể train song song (parallelizable)
- Thư viện hỗ trợ phong phú (PyTorch nn.Transformer)

**Nhược điểm:**
- Absolute positional encoding không tối ưu cho âm nhạc (âm nhạc có tính lặp lại theo chu kỳ)
- Complexity O(n²) với chiều dài chuỗi — giới hạn khả năng sinh nhạc dài
- Chưa có cơ chế đặc biệt cho cấu trúc âm nhạc

---

### B. Music Transformer (Huang et al., 2018)

**Mô tả:** Cải tiến Transformer với **Relative Position Encoding** — thay vì mã hóa vị trí tuyệt đối, mã hóa khoảng cách tương đối giữa các tokens. Điều này phù hợp tự nhiên với âm nhạc vì các pattern âm nhạc (motif, chord progression) được định nghĩa bởi khoảng cách tương đối giữa các nốt.

**Ưu điểm:**
- Relative attention phù hợp với bản chất lặp lại của âm nhạc
- Sinh nhạc có cấu trúc tốt hơn vanilla Transformer
- Đã được chứng minh hiệu quả trên MAESTRO dataset
- Có thể tự xây dựng với effort vừa phải

**Nhược điểm:**
- Vẫn có complexity O(n²)
- Paper gốc chỉ hỗ trợ unconditional generation (cần thêm conditioning module)
- Memory-intensive hơn vanilla Transformer do relative attention matrix

**Tài liệu tham khảo:** *Music Transformer: Generating Music with Long-Term Structure* (Huang et al., 2018)

---

### C. MuseNet (OpenAI, 2019)

**Mô tả:** Sử dụng kiến trúc GPT-2 (Sparse Transformer) để sinh nhạc đa nhạc cụ. MuseNet có thể sinh nhạc theo phong cách của các nhà soạn nhạc khác nhau.

**Ưu điểm:**
- Chất lượng sinh nhạc rất cao
- Hỗ trợ đa nhạc cụ
- Sparse attention giảm complexity

**Nhược điểm:**
- Mô hình rất lớn (~72 layers), cần GPU mạnh
- OpenAI không công bố toàn bộ chi tiết kiến trúc
- Khó tự xây dựng do quy mô
- Không có text conditioning trong thiết kế gốc

---

### D. MusicLM (Google, Agostinelli et al., 2023)

**Mô tả:** Hierarchical generative model kết hợp MuLan (Music-Language joint embedding), w2v-BERT, và SoundStream. Sinh nhạc từ text descriptions ở mức audio.

**Ưu điểm:**
- Chất lượng audio cực cao (24kHz)
- Text conditioning mạnh mẽ nhờ MuLan embedding
- Hierarchical generation: semantic → acoustic tokens

**Nhược điểm:**
- Cực kỳ phức tạp — nhiều thành phần phụ thuộc lẫn nhau
- Cần pre-trained MuLan model (yêu cầu dataset cực lớn)
- Yêu cầu GPU cluster (TPU pods trong paper gốc)
- Không thực tế cho đồ án tự xây dựng

---

### E. AudioLM (Google, Borsos et al., 2023)

**Mô tả:** Language modeling approach cho audio generation. Sử dụng hierarchical tokenization: semantic tokens (từ w2v-BERT) và acoustic tokens (từ SoundStream).

**Ưu điểm:**
- Sinh audio tự nhiên, chất lượng cao
- Framework tổng quát cho mọi loại audio

**Nhược điểm:**
- Không có text conditioning trong thiết kế gốc
- Cần pre-trained w2v-BERT và SoundStream
- Yêu cầu tài nguyên lớn
- Quá phức tạp cho đồ án

---

### F. AudioLDM (Liu et al., 2023)

**Mô tả:** Áp dụng **Latent Diffusion Model** (LDM) cho audio generation. Sử dụng VAE để nén mel-spectrogram vào latent space, sau đó dùng diffusion process trong latent space với CLAP text conditioning.

**Ưu điểm:**
- Chất lượng audio tốt
- Text conditioning hiệu quả qua CLAP
- Diffusion trong latent space → tiết kiệm hơn diffusion trên raw audio

**Nhược điểm:**
- Cần pre-trained CLAP model
- Cần pre-trained VAE
- Diffusion sampling chậm (nhiều bước)
- Phức tạp để tự xây dựng từ đầu

---

### G. Stable Audio (Stability AI, 2023)

**Mô tả:** Latent Diffusion Model với timing conditioning — cho phép kiểm soát thời lượng và cấu trúc thời gian của audio.

**Ưu điểm:**
- Kiểm soát timing tốt
- Chất lượng cao
- Kiến trúc rõ ràng

**Nhược điểm:**
- Proprietary — không công bố đầy đủ chi tiết
- Cần dataset lớn với metadata thời gian
- Yêu cầu GPU A100
- Không thực tế cho đồ án tự xây dựng

---

### H. Diffusion Models (cho MIDI)

**Mô tả:** Áp dụng Denoising Diffusion Probabilistic Models (DDPM) cho MIDI generation. Thay vì diffusion trên audio, thực hiện diffusion trên **piano roll representation** hoặc **latent MIDI features**.

**Ưu điểm:**
- Chất lượng sinh cao — diffusion models nổi tiếng về chất lượng
- Conditioning dễ dàng qua classifier-free guidance
- Có thể kiểm soát diversity qua guidance scale

**Nhược điểm:**
- Sampling chậm (cần 50-1000 bước denoising)
- Piano roll representation chiếm nhiều bộ nhớ
- Chưa có nhiều research cho MIDI diffusion (chủ yếu cho audio)
- Cần thiết kế noise schedule phù hợp cho discrete data

---

### I. VAE (Variational Autoencoder)

**Mô tả:** Encoder nén MIDI vào latent space, decoder sinh MIDI từ latent vector. Conditioning bằng cách concat text embedding với latent vector.

**Ưu điểm:**
- Kiến trúc đơn giản
- Latent space có tính chất tốt (smooth, interpolable)
- Train nhanh

**Nhược điểm:**
- Chất lượng sinh thường kém hơn Transformer và Diffusion
- Posterior collapse — latent code bị bỏ qua
- Khó sinh chuỗi dài có cấu trúc
- Output thường bị "mờ" (blurry) — mất chi tiết

---

### J. GAN (Generative Adversarial Network)

**Mô tả:** Generator sinh MIDI, discriminator phân biệt MIDI thật/giả. Conditioning qua text embedding inject vào generator.

**Ưu điểm:**
- Sinh nhanh (single forward pass)
- Output sắc nét

**Nhược điểm:**
- Training rất không ổn định (mode collapse, training instability)
- Khó áp dụng cho sequential/discrete data như MIDI
- Không có log-likelihood → khó đánh giá
- GAN cho music generation là hướng ít được nghiên cứu, thiếu baseline

---

## 2.2. Bảng so sánh tổng hợp

| Tiêu chí | Transformer | Music Transformer | MuseNet | MusicLM | AudioLDM | Diffusion (MIDI) | VAE | GAN |
|---|---|---|---|---|---|---|---|---|
| **Chất lượng** | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ |
| **Tốc độ train** | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| **GPU yêu cầu** | 1× RTX 3090 | 1× RTX 3090 | 4-8× V100 | TPU pod | 2-4× A100 | 1-2× RTX 3090 | 1× RTX 3060 | 1× RTX 3090 |
| **Độ khó triển khai** | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ |
| **Khả năng tự xây** | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★☆☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★★☆☆ |
| **Text conditioning** | Dễ thêm | Dễ thêm | Không native | Native | Native | Dễ thêm | Dễ thêm | Khó |
| **Sinh nhạc dài** | Trung bình | Tốt | Rất tốt | Rất tốt | Tốt | Trung bình | Kém | Kém |

## 2.3. Đề xuất mô hình cho đồ án

> [!TIP]
> **Đề xuất: Music Transformer + Cross-Attention Text Conditioning**

**Lý do lựa chọn:**

1. **Cân bằng chất lượng và khả năng tự xây dựng:** Music Transformer có chất lượng sinh nhạc tốt (★★★★☆) và hoàn toàn có thể tự xây dựng từ đầu với PyTorch.

2. **Relative Position Encoding phù hợp tự nhiên với âm nhạc:** Các pattern âm nhạc (motif, chord) được định nghĩa bởi khoảng cách tương đối — không phải vị trí tuyệt đối.

3. **Text conditioning dễ tích hợp:** Thêm Cross-Attention layer giữa text embedding và music decoder — pattern đã được chứng minh hiệu quả trong Stable Diffusion (text-to-image).

4. **GPU hợp lý:** Có thể train trên 1× RTX 3090 (24GB VRAM) — phù hợp với đồ án.

5. **Sinh MIDI (không phải raw audio):** Giảm complexity đáng kể so với sinh waveform trực tiếp. MIDI có thể dễ dàng chuyển sang WAV bằng FluidSynth.

6. **Có nền tảng research vững chắc:** Paper Music Transformer (Huang et al., 2018) được trích dẫn rộng rãi, có nhiều implementation tham khảo.

**Kiến trúc đề xuất tổng quát:**

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────────┐
│ Text Encoder │────►│ Cross-Attn    │────►│ Music Transformer    │
│ (Frozen/     │     │ Conditioning  │     │ (Relative Attention  │
│  Trainable)  │     │               │     │  + Autoregressive)   │
└──────────────┘     └───────────────┘     └──────────┬───────────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │  MIDI Tokens  │
                                              └───────┬───────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │   FluidSynth  │
                                              │   → WAV       │
                                              └───────────────┘
```

---

# 3. Thiết Kế Kiến Trúc Hệ Thống

## 3.1. Pipeline tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM PIPELINE                              │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │             │    │              │    │                        │ │
│  │ User Prompt │───►│ Text Encoder │───►│ Prompt Embedding       │ │
│  │             │    │              │    │ (d_model = 256)        │ │
│  └─────────────┘    └──────────────┘    └───────────┬────────────┘ │
│                                                     │              │
│                                                     ▼              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   MUSIC TRANSFORMER                          │  │
│  │                                                              │  │
│  │  ┌─────────────────┐    ┌──────────────────────────────────┐│  │
│  │  │ MIDI Token      │    │  Transformer Decoder Block ×N   ││  │
│  │  │ Embedding       │───►│                                  ││  │
│  │  │                 │    │  ┌────────────────────────────┐  ││  │
│  │  └─────────────────┘    │  │ Masked Self-Attention      │  ││  │
│  │                         │  │ (Relative Position)        │  ││  │
│  │                         │  └─────────────┬──────────────┘  ││  │
│  │                         │                │                  ││  │
│  │                         │  ┌─────────────▼──────────────┐  ││  │
│  │                         │  │ Cross-Attention            │  ││  │
│  │                         │  │ (Text Conditioning)        │  ││  │
│  │                         │  └─────────────┬──────────────┘  ││  │
│  │                         │                │                  ││  │
│  │                         │  ┌─────────────▼──────────────┐  ││  │
│  │                         │  │ Feed-Forward Network       │  ││  │
│  │                         │  └─────────────┬──────────────┘  ││  │
│  │                         │                │                  ││  │
│  │                         └────────────────┼──────────────────┘│  │
│  │                                          │                   │  │
│  │                                          ▼                   │  │
│  │                              ┌───────────────────┐           │  │
│  │                              │ Linear + Softmax  │           │  │
│  │                              └─────────┬─────────┘           │  │
│  └────────────────────────────────────────┼─────────────────────┘  │
│                                           │                        │
│                                           ▼                        │
│                               ┌───────────────────┐               │
│                               │   MIDI Tokens     │               │
│                               │   (Predicted)     │               │
│                               └─────────┬─────────┘               │
│                                         │                          │
│                                         ▼                          │
│                               ┌───────────────────┐               │
│                               │ Token-to-MIDI     │               │
│                               │ Decoder           │               │
│                               └─────────┬─────────┘               │
│                                         │                          │
│                                         ▼                          │
│                               ┌───────────────────┐               │
│                               │ MIDI File (.mid)  │               │
│                               └─────────┬─────────┘               │
│                                         │                          │
│                                         ▼                          │
│                               ┌───────────────────┐               │
│                               │ FluidSynth +      │               │
│                               │ SoundFont         │               │
│                               └─────────┬─────────┘               │
│                                         │                          │
│                                         ▼                          │
│                               ┌───────────────────┐               │
│                               │ WAV File (.wav)   │               │
│                               └───────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.2. Giải thích chức năng từng thành phần

### 1. User Prompt (Đầu vào)
**Chức năng:** Nhận mô tả văn bản từ người dùng về loại nhạc mong muốn.

```
Input ví dụ:
{
  "mood": "happy",
  "genre": "fantasy",
  "scene": "forest village",
  "tempo": "fast",
  "instruments": ["piano", "strings"],
  "duration": 30
}
```

Prompt được format thành câu tự nhiên:
```
"Happy fantasy music for a forest village scene, fast tempo, 
 played with piano and strings, 30 seconds"
```

---

### 2. Text Encoder
**Chức năng:** Chuyển đổi mô tả văn bản thành vector embedding mang ngữ nghĩa.

Có 2 lựa chọn:

**Phương án A — Trainable Embedding (Đề xuất cho đồ án):**
- Xây dựng vocabulary riêng cho các thuộc tính âm nhạc
- Mỗi attribute (mood, genre, tempo, instrument) có embedding riêng
- Concat hoặc sum các embeddings
- **Ưu điểm:** Nhẹ, nhanh, dễ kiểm soát
- **Nhược điểm:** Không hiểu ngữ nghĩa tự do — chỉ hiểu các attribute được định nghĩa sẵn

**Phương án B — Pre-trained Text Encoder:**
- Sử dụng BERT-tiny hoặc DistilBERT (frozen hoặc fine-tuned)
- Lấy [CLS] token hoặc mean pooling làm sentence embedding
- **Ưu điểm:** Hiểu ngữ nghĩa tự do
- **Nhược điểm:** Nặng hơn, thêm dependency

> **Đề xuất:** Phương án A cho baseline, Phương án B cho version nâng cao.

---

### 3. Music Transformer (Core Model)
**Chức năng:** Bộ sinh nhạc chính — nhận text embedding và sinh chuỗi MIDI tokens theo kiểu autoregressive (token by token).

Chi tiết kiến trúc → Xem [Mục 4](#4-thiết-kế-mô-hình).

---

### 4. Token-to-MIDI Decoder
**Chức năng:** Chuyển đổi chuỗi tokens dự đoán thành file MIDI.

```python
# Ví dụ: Tokens → MIDI Events
tokens = [SET_TEMPO_120, NOTE_ON_60, SET_VEL_80, TIME_SHIFT_100, NOTE_OFF_60, ...]

# Chuyển thành MIDI messages
midi = pretty_midi.PrettyMIDI(initial_tempo=120)
piano = pretty_midi.Instrument(program=0)  # Piano
piano.notes.append(pretty_midi.Note(
    velocity=80, pitch=60, start=0.0, end=0.5
))
midi.instruments.append(piano)
midi.write('output.mid')
```

---

### 5. Audio Renderer (FluidSynth)
**Chức năng:** Render file MIDI thành audio WAV sử dụng SoundFont (bộ mẫu âm thanh của nhạc cụ).

```
MIDI File ──► FluidSynth + SoundFont (.sf2) ──► WAV File
```

---

# 4. Thiết Kế Mô Hình

## 4.1. Tổng quan kiến trúc

Mô hình sử dụng kiến trúc **Conditional Transformer Decoder** với **Relative Position Encoding** và **Cross-Attention** cho text conditioning.

```
┌─────────────────────────────────────────────────────────────────┐
│              CONDITIONAL MUSIC TRANSFORMER                       │
│                                                                  │
│  Input: [<BOS>, tok_1, tok_2, ..., tok_{t-1}]                  │
│  Condition: text_embedding (from Text Encoder)                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  TOKEN EMBEDDING LAYER                                   │    │
│  │  ┌────────────┐  ┌─────────────────────┐                │    │
│  │  │ Token      │  │ Relative Positional │                │    │
│  │  │ Embedding  │ +│ Encoding (RPE)      │  = X₀          │    │
│  │  │ (V × d)   │  │ (learned)           │                │    │
│  │  └────────────┘  └─────────────────────┘                │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DECODER BLOCK ×N (N=6)                                   │   │
│  │                                                           │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │ Layer Norm                                           │ │   │
│  │  └───────────────────────┬─────────────────────────────┘ │   │
│  │                          │                                │   │
│  │  ┌───────────────────────▼─────────────────────────────┐ │   │
│  │  │ Masked Multi-Head Self-Attention                     │ │   │
│  │  │ with Relative Position Bias                          │ │   │
│  │  │                                                      │ │   │
│  │  │ heads = 8, d_k = d_v = d_model/heads = 32          │ │   │
│  │  │                                                      │ │   │
│  │  │ Attn(Q,K,V) = softmax((QK^T + S_rel) / √d_k) · V  │ │   │
│  │  │                                                      │ │   │
│  │  │ S_rel[i,j] = Q_i · R_{clip(j-i)}                   │ │   │
│  │  │ R: learned relative position embeddings              │ │   │
│  │  └───────────────────────┬─────────────────────────────┘ │   │
│  │                          │ + Residual                     │   │
│  │                          ▼                                │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │ Layer Norm                                           │ │   │
│  │  └───────────────────────┬─────────────────────────────┘ │   │
│  │                          │                                │   │
│  │  ┌───────────────────────▼─────────────────────────────┐ │   │
│  │  │ Cross-Attention (Text Conditioning)                  │ │   │
│  │  │                                                      │ │   │
│  │  │ Q = from music decoder                               │ │   │
│  │  │ K, V = from text encoder output                      │ │   │
│  │  │                                                      │ │   │
│  │  │ CrossAttn(Q,K,V) = softmax(QK^T / √d_k) · V        │ │   │
│  │  └───────────────────────┬─────────────────────────────┘ │   │
│  │                          │ + Residual                     │   │
│  │                          ▼                                │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │ Layer Norm                                           │ │   │
│  │  └───────────────────────┬─────────────────────────────┘ │   │
│  │                          │                                │   │
│  │  ┌───────────────────────▼─────────────────────────────┐ │   │
│  │  │ Feed-Forward Network                                 │ │   │
│  │  │                                                      │ │   │
│  │  │ FFN(x) = GELU(xW₁ + b₁)W₂ + b₂                    │ │   │
│  │  │ d_ff = 4 × d_model = 1024                           │ │   │
│  │  └───────────────────────┬─────────────────────────────┘ │   │
│  │                          │ + Residual                     │   │
│  │                          ▼                                │   │
│  └──────────────────────────┼────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OUTPUT LAYER                                             │   │
│  │                                                           │   │
│  │  Layer Norm → Linear(d_model, vocab_size) → Softmax      │   │
│  │                                                           │   │
│  │  Output: P(next_token | previous_tokens, text_condition)  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 4.2. Chi tiết từng thành phần

### A. Token Embedding

**Chức năng:** Chuyển đổi MIDI token ID (số nguyên) thành dense vector.

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # Scale embedding by sqrt(d_model) theo paper gốc
        return self.embedding(x) * math.sqrt(self.d_model)
```

**Hyperparameters:**
- `vocab_size` ≈ 388 (tổng số MIDI tokens — xem Mục 6)
- `d_model` = 256

---

### B. Relative Positional Encoding (RPE)

**Chức năng:** Mã hóa thông tin vị trí tương đối giữa các tokens — thay vì biết "token này ở vị trí 42", mô hình biết "token này cách token kia 5 bước".

**Lý do sử dụng RPE thay vì Absolute PE:**

Trong âm nhạc, một motif (ví dụ: C-E-G) có ý nghĩa giống nhau bất kể nó xuất hiện ở measure 1 hay measure 8. RPE cho phép mô hình nhận ra điều này.

```python
class RelativePositionBias(nn.Module):
    """
    Relative position bias cho self-attention.
    Thêm learned bias dựa trên khoảng cách tương đối giữa query và key.
    """
    def __init__(self, max_relative_position, num_heads):
        super().__init__()
        # Clip relative distance vào [-max_rel, +max_rel]
        self.max_relative_position = max_relative_position
        # Learned embedding cho mỗi relative distance
        # Tổng: 2 * max_rel + 1 vị trí (âm, 0, dương)
        self.relative_position_bias = nn.Embedding(
            2 * max_relative_position + 1, num_heads
        )
    
    def forward(self, seq_len):
        # Tạo ma trận khoảng cách tương đối [seq_len, seq_len]
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Clip vào phạm vi [-max_rel, +max_rel]
        relative_positions = relative_positions.clamp(
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift để index từ 0
        relative_positions += self.max_relative_position
        
        # Lookup bias: [seq_len, seq_len, num_heads]
        bias = self.relative_position_bias(relative_positions)
        
        # Reshape: [num_heads, seq_len, seq_len]
        return bias.permute(2, 0, 1)
```

**Hyperparameters:**
- `max_relative_position` = 128 (đủ cho các pattern âm nhạc phổ biến)
- `num_heads` = 8

---

### C. Masked Multi-Head Self-Attention với Relative Position Bias

**Chức năng:** Cho phép mỗi token "nhìn" vào các tokens trước đó (causal mask) và học mối quan hệ giữa chúng, với bias dựa trên khoảng cách tương đối.

**Công thức:**

```
Attention(Q, K, V) = softmax( (Q·K^T + S_rel) / √d_k ) · V

Trong đó:
- Q = X·W_Q  (Query)
- K = X·W_K  (Key)  
- V = X·W_V  (Value)
- S_rel = Relative Position Bias matrix
- d_k = d_model / num_heads = 32
- Causal mask: vị trí j > i bị mask = -∞
```

```python
class MultiHeadSelfAttentionWithRPE(nn.Module):
    def __init__(self, d_model, num_heads, max_relative_position, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.rpe = RelativePositionBias(max_relative_position, num_heads)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, D = x.shape
        
        # Linear projections → [B, T, num_heads, d_k] → [B, num_heads, T, d_k]
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores: [B, num_heads, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Thêm relative position bias
        rel_bias = self.rpe(T)  # [num_heads, T, T]
        scores = scores + rel_bias.unsqueeze(0)  # broadcast over batch
        
        # Causal mask (chặn nhìn vào tương lai)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        output = torch.matmul(attn_weights, V)  # [B, num_heads, T, d_k]
        
        # Concat heads
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.W_o(output)
```

---

### D. Cross-Attention (Text Conditioning)

**Chức năng:** "Chú ý" vào text embedding — cho phép music decoder biết mô tả văn bản yêu cầu gì.

**Cơ chế:**
```
Q = from music decoder (current layer output)
K = from text encoder output
V = from text encoder output

CrossAttn(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Đây chính xác là cơ chế mà Stable Diffusion dùng để conditioning text vào image generation. Ở đây, ta áp dụng cho music generation.

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, d_cond, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_cond, d_model)  # project từ cond dim
        self.W_v = nn.Linear(d_cond, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cond):
        """
        x: [B, T_music, d_model] - music decoder features
        cond: [B, T_text, d_cond] - text encoder output
        """
        B, T_m, D = x.shape
        T_c = cond.shape[1]
        
        Q = self.W_q(x).view(B, T_m, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(cond).view(B, T_c, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(cond).view(B, T_c, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, T_m, D)
        
        return self.W_o(output)
```

---

### E. Feed-Forward Network (FFN)

**Chức năng:** Biến đổi phi tuyến tính — tăng khả năng biểu diễn của mô hình. Mỗi token được xử lý độc lập.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU > ReLU cho Transformer
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

**Hyperparameters:**
- `d_ff` = 1024 (= 4 × d_model)
- Activation: GELU (smooth hơn ReLU, chuẩn trong Transformer hiện đại)

---

### F. Decoder Block

**Chức năng:** Kết hợp tất cả sub-layers với residual connections và layer normalization.

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, d_cond, 
                 max_relative_position, dropout=0.1):
        super().__init__()
        # Sub-layers
        self.self_attn = MultiHeadSelfAttentionWithRPE(
            d_model, num_heads, max_relative_position, dropout
        )
        self.cross_attn = CrossAttention(
            d_model, d_cond, num_heads, dropout
        )
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Layer Norms (Pre-LN variant — ổn định hơn Post-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cond, mask=None):
        # Self-Attention + Residual
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        
        # Cross-Attention + Residual
        x = x + self.dropout(self.cross_attn(self.norm2(x), cond))
        
        # FFN + Residual
        x = x + self.dropout(self.ffn(self.norm3(x)))
        
        return x
```

---

### G. Output Layer (Linear + Softmax)

**Chức năng:** Dự đoán xác suất cho token tiếp theo trong vocabulary.

```python
# Trong forward pass của model chính
logits = self.output_projection(self.final_norm(decoder_output))
# logits shape: [B, T, vocab_size]

# Khi training:
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

# Khi inference:
probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
next_token = torch.multinomial(probs, 1)
```

## 4.3. Bảng tóm tắt Hyperparameters

| Parameter | Giá trị | Lý do |
|---|---|---|
| `d_model` | 256 | Đủ capacity, phù hợp 1 GPU |
| `num_heads` | 8 | d_k = 32, chuẩn cho d_model=256 |
| `num_layers` | 6 | Cân bằng capacity và tốc độ train |
| `d_ff` | 1024 | 4 × d_model (chuẩn) |
| `max_seq_len` | 2048 | Đủ cho ~30s nhạc |
| `vocab_size` | ~336 | Xem Mục 6 |
| `d_cond` | 256 | Text conditioning dimension |
| `max_relative_position` | 128 | Đủ cho pattern detection |
| `dropout` | 0.1 | Regularization chuẩn |
| **Tổng parameters** | **~8.2M** | Tối ưu hóa cho RTX 4060 |

## 4.4. Nâng Cấp Kiến Trúc Theo Chuẩn LLM Hiện Đại (LLaMA-style)

Để tối đa hóa tốc độ và chất lượng sinh nhạc (đặc biệt phù hợp chạy trên các GPU cá nhân như RTX 4060), mô hình đã được áp dụng 3 kỹ thuật tiên tiến nhất từ các LLM (như LLaMA 3, Mistral):

1. **RMSNorm (Root Mean Square Normalization):**
   Thay thế `LayerNorm` truyền thống bằng `RMSNorm`. Thay vì tính cả trung bình (mean) và phương sai (variance), `RMSNorm` chỉ tính mean square. Điều này giúp giảm 10-20% thời gian tính toán normalization mà không làm giảm chất lượng mô hình.

2. **SwiGLU (Swish-Gated Linear Unit):**
   Thay thế mạng Feed-Forward (Linear → GELU → Linear) bằng cấu trúc Gating (SwiGLU) giống LLaMA.
   - Công thức: `SwiGLU(x) = (xW₁ ⊗ Swish(xW₁))W₂`
   - Hiệu quả: Khả năng học các biểu diễn âm nhạc phức tạp tăng lên đáng kể.

3. **FlashAttention (Scaled Dot Product Attention):**
   Sử dụng hàm `F.scaled_dot_product_attention` (SDPA) của PyTorch thay cho các phép toán ma trận thủ công.
   - Cơ chế: Không tạo ra ma trận attention trung gian kích thước `(N, N)` trên VRAM.
   - Hiệu quả: Giải quyết bài toán "Out Of Memory" (OOM) cho các chuỗi nhạc dài, giảm một nửa VRAM yêu cầu và tăng tốc độ forward pass cực mạnh.

---

# 5. Dataset

## 5.1. Đề xuất các dataset

### A. MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization)

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | Google Magenta |
| **Dung lượng** | ~200 giờ, ~1,300 file MIDI |
| **Định dạng** | MIDI + WAV (aligned) |
| **Nội dung** | Piano classical music (performances từ International Piano-e-Competition) |
| **Ưu điểm** | Chất lượng cực cao; MIDI aligned với audio; tempo/velocity chính xác |
| **Nhược điểm** | Chỉ có piano; chỉ classical music; không có game music |
| **Link** | https://magenta.tensorflow.org/datasets/maestro |

**Đánh giá cho đồ án:** ★★★★☆ — Tuyệt vời để pretrain model trên piano music, nhưng cần bổ sung thêm dữ liệu game music.

---

### B. Lakh MIDI Dataset

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | Colin Raffel (PhD thesis) |
| **Dung lượng** | ~176,000 file MIDI |
| **Định dạng** | MIDI |
| **Nội dung** | Đa thể loại: pop, rock, jazz, classical, game, film |
| **Ưu điểm** | Rất lớn; đa dạng genre và instruments; nhiều multi-track |
| **Nhược điểm** | Chất lượng không đồng đều (user-created MIDI); thiếu metadata chuẩn hóa |
| **Link** | https://colinraffel.com/projects/lmd/ |

**Đánh giá cho đồ án:** ★★★★★ — Dataset chính cho training. Lớn, đa dạng, có nhiều nhạc cụ.

---

### C. GiantMIDI-Piano

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | Qiuqiang Kong et al. |
| **Dung lượng** | ~10,800 file MIDI (từ ~2,700 nhà soạn nhạc) |
| **Định dạng** | MIDI (transcribed từ audio bằng AI) |
| **Nội dung** | Piano music — classical và phổ thông |
| **Ưu điểm** | Lớn; velocity/timing chính xác (AI-transcribed) |
| **Nhược điểm** | Chỉ piano; có thể có lỗi từ quá trình transcription |
| **Link** | https://github.com/bytedance/GiantMIDI-Piano |

**Đánh giá cho đồ án:** ★★★☆☆ — Bổ sung tốt cho piano data, nhưng hạn chế về instrument diversity.

---

### D. Video Game Music MIDI Dataset

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | VGMusic.com, NinSheetMusic, các nguồn tổng hợp |
| **Dung lượng** | ~30,000+ file MIDI |
| **Định dạng** | MIDI |
| **Nội dung** | Nhạc nền game: NES, SNES, PS, RPG, platformer, v.v. |
| **Ưu điểm** | Đúng domain (game music); có metadata game/console; loop-friendly |
| **Nhược điểm** | Chất lượng không đồng đều; bản quyền không rõ ràng cho research; cần crawl và clean |
| **Link** | https://www.vgmusic.com/ |

**Đánh giá cho đồ án:** ★★★★★ — Cực kỳ quan trọng — đây là domain-specific data cho bài toán game music.

---

### E. ADL Piano MIDI Dataset

| Thuộc tính | Chi tiết |
|---|---|
| **Nguồn** | Lucas Maia |
| **Dung lượng** | ~11,000 file MIDI |
| **Định dạng** | MIDI |
| **Nội dung** | Piano music phân loại theo thể loại |
| **Ưu điểm** | Có nhãn thể loại; clean |
| **Nhược điểm** | Chỉ piano |

---

## 5.2. Chiến lược sử dụng Dataset

```
┌─────────────────────────────────────────────────────┐
│               DATASET STRATEGY                       │
│                                                      │
│  PHASE 1: Pre-training (General Music Knowledge)     │
│  ├── Lakh MIDI Dataset (176K files)                  │
│  └── MAESTRO (1.3K files, high quality)              │
│                                                      │
│  PHASE 2: Fine-tuning (Game Music Domain)            │
│  ├── Video Game Music MIDI Dataset (30K+ files)      │
│  └── Curated game soundtrack MIDIs                   │
│                                                      │
│  PHASE 3: Prompt-Music Pairing                       │
│  └── Annotate subset with text descriptions          │
│      (mood, genre, tempo, instrument labels)         │
│      → ~5,000-10,000 paired samples                  │
└─────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Vấn đề quan trọng:** Hầu hết MIDI datasets không có text descriptions đi kèm. Cần tạo text labels bằng:
> 1. **Rule-based annotation:** Phân tích MIDI để tự động gán tempo (slow/medium/fast), key (major/minor → happy/sad), instruments, note density
> 2. **Manual annotation:** Nhờ người nghe và gán labels mood/scene cho subset
> 3. **Semi-automatic:** Kết hợp cả hai

---

# 6. Tiền Xử Lý Dữ Liệu

## 6.1. Đọc MIDI

File MIDI chứa các **MIDI messages** (events) như:
- `note_on(pitch, velocity, time)` — bắt đầu nốt
- `note_off(pitch, time)` — kết thúc nốt
- `set_tempo(bpm)` — thay đổi tempo
- `program_change(instrument)` — thay đổi nhạc cụ

```python
import pretty_midi

def read_midi(filepath):
    """Đọc file MIDI và trích xuất thông tin."""
    midi = pretty_midi.PrettyMIDI(filepath)
    
    info = {
        'tempo': midi.estimate_tempo(),
        'total_time': midi.get_end_time(),
        'instruments': [],
        'notes': []
    }
    
    for instrument in midi.instruments:
        info['instruments'].append({
            'name': instrument.name,
            'program': instrument.program,
            'is_drum': instrument.is_drum,
            'num_notes': len(instrument.notes)
        })
        
        for note in instrument.notes:
            info['notes'].append({
                'instrument': instrument.program,
                'pitch': note.pitch,         # 0-127 (MIDI pitch)
                'velocity': note.velocity,    # 0-127 (volume)
                'start': note.start,          # seconds
                'end': note.end,              # seconds
                'duration': note.end - note.start
            })
    
    return info
```

## 6.2. Event Representation — REMI (REvamped MIDI-derived)

Sử dụng representation **REMI** (Huang & Yang, 2020) — một phương pháp tokenize MIDI đã được chứng minh hiệu quả cho music generation.

**Ý tưởng:** Biểu diễn MIDI như một chuỗi các **events**, mỗi event là một token. Các event được sắp xếp theo thời gian.

### Các loại tokens:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIDI TOKEN VOCABULARY                         │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ SPECIAL TOKENS (4)                                        │  │
│  │ <PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ NOTE_ON (128 tokens)                                      │  │
│  │ NOTE_ON_0 ... NOTE_ON_127                                 │  │
│  │ → Pitch: C-1 (0) đến G9 (127)                            │  │
│  │ → Thực tế dùng: 21-108 (piano range) → 88 tokens         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ NOTE_OFF (128 tokens)                                     │  │
│  │ NOTE_OFF_0 ... NOTE_OFF_127                               │  │
│  │ → Đánh dấu kết thúc nốt                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ VELOCITY (32 tokens)                                      │  │
│  │ VEL_0 ... VEL_31                                          │  │
│  │ → Quantize 0-127 → 32 bins (mỗi bin = 4 giá trị)        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ TIME_SHIFT (100 tokens)                                   │  │
│  │ TIME_SHIFT_1 ... TIME_SHIFT_100                           │  │
│  │ → Mỗi unit = 10ms, max = 1000ms = 1 giây                │  │
│  │ → Khoảng thời gian > 1s: dùng nhiều TIME_SHIFT liên tiếp│  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ SET_TEMPO (8 tokens)                                      │  │
│  │ TEMPO_SLOW ... TEMPO_VERY_FAST                            │  │
│  │ → Quantize BPM: <70, 70-90, 90-110, 110-130,            │  │
│  │   130-150, 150-170, 170-200, >200                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ INSTRUMENT (16 tokens)                                    │  │
│  │ INST_PIANO, INST_STRINGS, INST_BRASS, ...                │  │
│  │ → Gom 128 MIDI programs → 16 nhóm chính                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  TỔNG VOCABULARY SIZE ≈ 388 tokens                              │
│  (4 + 128 + 128 + 32 + 100 + 8 + 16 = 416,                    │
│   giảm còn ~388 nếu giới hạn pitch range)                      │
└─────────────────────────────────────────────────────────────────┘
```

## 6.3. Ví dụ minh họa tokenization

**Input MIDI:** Đoạn nhạc piano đơn giản — C major chord (C4-E4-G4)

```
Note: C4 (pitch=60), velocity=80, start=0.0s, end=0.5s
Note: E4 (pitch=64), velocity=75, start=0.0s, end=0.5s
Note: G4 (pitch=67), velocity=78, start=0.0s, end=0.5s
Note: C5 (pitch=72), velocity=85, start=0.5s, end=1.0s
```

**Tokenized sequence:**

```
<BOS>
INST_PIANO          ← Chọn instrument: Piano
TEMPO_MEDIUM        ← Set tempo: 120 BPM

# Chord C major (3 nốt đồng thời)
VEL_20              ← Velocity bin 20 (≈80/4)
NOTE_ON_60          ← C4 on
VEL_19              ← Velocity bin 19 (≈75/4)
NOTE_ON_64          ← E4 on
VEL_20              ← Velocity bin 20 (≈78/4)
NOTE_ON_67          ← G4 on

# Thời gian trôi 500ms
TIME_SHIFT_50       ← 50 × 10ms = 500ms

# Tắt chord
NOTE_OFF_60         ← C4 off
NOTE_OFF_64         ← E4 off
NOTE_OFF_67         ← G4 off

# Nốt C5
VEL_21              ← Velocity bin 21 (≈85/4)
NOTE_ON_72          ← C5 on
TIME_SHIFT_50       ← 500ms
NOTE_OFF_72         ← C5 off

<EOS>
```

**Chuỗi token IDs:**
```
[1, 300, 310, 164, 64, 163, 68, 164, 71, 232, 192, 196, 199, 165, 76, 232, 204, 2]
```

## 6.4. Code tiền xử lý hoàn chỉnh

```python
import pretty_midi
import numpy as np
from collections import OrderedDict

class MidiTokenizer:
    """
    Tokenizer cho MIDI files sử dụng REMI-like representation.
    """
    
    def __init__(self, 
                 pitch_range=(21, 108),      # Piano range
                 velocity_bins=32,
                 time_shift_bins=100,         # 10ms per bin
                 tempo_bins=8,
                 num_instrument_groups=16):
        
        self.pitch_range = pitch_range
        self.velocity_bins = velocity_bins
        self.time_shift_bins = time_shift_bins
        
        # Build vocabulary
        self.vocab = OrderedDict()
        self._build_vocab()
    
    def _build_vocab(self):
        idx = 0
        
        # Special tokens
        for token in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
            self.vocab[token] = idx
            idx += 1
        
        # Note On events (pitch_min to pitch_max)
        for pitch in range(self.pitch_range[0], self.pitch_range[1] + 1):
            self.vocab[f'NOTE_ON_{pitch}'] = idx
            idx += 1
        
        # Note Off events
        for pitch in range(self.pitch_range[0], self.pitch_range[1] + 1):
            self.vocab[f'NOTE_OFF_{pitch}'] = idx
            idx += 1
        
        # Velocity bins
        for v in range(self.velocity_bins):
            self.vocab[f'VEL_{v}'] = idx
            idx += 1
        
        # Time shift (10ms increments, 1-100 → 10ms-1000ms)
        for t in range(1, self.time_shift_bins + 1):
            self.vocab[f'TIME_SHIFT_{t}'] = idx
            idx += 1
        
        # Tempo bins
        tempo_labels = ['VERY_SLOW', 'SLOW', 'MODERATE_SLOW', 'MODERATE',
                       'MODERATE_FAST', 'FAST', 'VERY_FAST', 'EXTREME']
        for label in tempo_labels:
            self.vocab[f'TEMPO_{label}'] = idx
            idx += 1
        
        # Instrument groups
        inst_labels = ['PIANO', 'CHROMATIC_PERC', 'ORGAN', 'GUITAR',
                      'BASS', 'STRINGS', 'ENSEMBLE', 'BRASS',
                      'REED', 'PIPE', 'SYNTH_LEAD', 'SYNTH_PAD',
                      'SYNTH_FX', 'ETHNIC', 'PERCUSSIVE', 'SFX']
        for label in inst_labels:
            self.vocab[f'INST_{label}'] = idx
            idx += 1
        
        self.vocab_size = len(self.vocab)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def midi_to_tokens(self, midi_path, max_length=2048):
        """Chuyển file MIDI thành chuỗi token IDs."""
        midi = pretty_midi.PrettyMIDI(midi_path)
        tokens = [self.vocab['<BOS>']]
        
        # Set tempo
        tempo = midi.estimate_tempo()
        tempo_token = self._quantize_tempo(tempo)
        tokens.append(self.vocab[tempo_token])
        
        # Collect all events, sort by time
        events = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            
            inst_group = self._get_instrument_group(instrument.program)
            
            for note in instrument.notes:
                events.append({
                    'time': note.start,
                    'type': 'note_on',
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'instrument': inst_group
                })
                events.append({
                    'time': note.end,
                    'type': 'note_off',
                    'pitch': note.pitch,
                    'instrument': inst_group
                })
        
        # Sort by time
        events.sort(key=lambda x: (x['time'], x['type'] == 'note_on'))
        
        current_time = 0.0
        current_instrument = None
        
        for event in events:
            # Time shift
            dt = event['time'] - current_time
            if dt > 0:
                time_tokens = self._encode_time_shift(dt)
                tokens.extend(time_tokens)
                current_time = event['time']
            
            # Instrument change
            if event.get('instrument') != current_instrument:
                current_instrument = event['instrument']
                tokens.append(self.vocab[f'INST_{current_instrument}'])
            
            # Note event
            if event['type'] == 'note_on':
                vel_bin = min(event['velocity'] // 4, self.velocity_bins - 1)
                tokens.append(self.vocab[f'VEL_{vel_bin}'])
                
                pitch = np.clip(event['pitch'], *self.pitch_range)
                tokens.append(self.vocab[f'NOTE_ON_{pitch}'])
            
            elif event['type'] == 'note_off':
                pitch = np.clip(event['pitch'], *self.pitch_range)
                tokens.append(self.vocab[f'NOTE_OFF_{pitch}'])
            
            if len(tokens) >= max_length - 1:
                break
        
        tokens.append(self.vocab['<EOS>'])
        
        # Padding
        if len(tokens) < max_length:
            tokens.extend([self.vocab['<PAD>']] * (max_length - len(tokens)))
        
        return tokens[:max_length]
    
    def _quantize_tempo(self, bpm):
        """Quantize BPM vào bins."""
        boundaries = [70, 90, 110, 130, 150, 170, 200]
        labels = ['VERY_SLOW', 'SLOW', 'MODERATE_SLOW', 'MODERATE',
                 'MODERATE_FAST', 'FAST', 'VERY_FAST', 'EXTREME']
        for i, bound in enumerate(boundaries):
            if bpm < bound:
                return f'TEMPO_{labels[i]}'
        return f'TEMPO_{labels[-1]}'
    
    def _encode_time_shift(self, dt_seconds):
        """Encode khoảng thời gian thành chuỗi TIME_SHIFT tokens."""
        dt_units = int(round(dt_seconds * 100))  # 10ms units
        tokens = []
        while dt_units > 0:
            shift = min(dt_units, self.time_shift_bins)
            tokens.append(self.vocab[f'TIME_SHIFT_{shift}'])
            dt_units -= shift
        return tokens
    
    def _get_instrument_group(self, program):
        """Map MIDI program number (0-127) → instrument group."""
        groups = ['PIANO', 'CHROMATIC_PERC', 'ORGAN', 'GUITAR',
                 'BASS', 'STRINGS', 'ENSEMBLE', 'BRASS',
                 'REED', 'PIPE', 'SYNTH_LEAD', 'SYNTH_PAD',
                 'SYNTH_FX', 'ETHNIC', 'PERCUSSIVE', 'SFX']
        return groups[program // 8]
```

---

# 7. Thiết Kế Prompt

## 7.1. Prompt Schema cho Game Music

Thiết kế prompt có cấu trúc với các **attributes** phù hợp cho game music:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAME MUSIC PROMPT SCHEMA                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ MOOD (Tâm trạng)                                        │    │
│  │ happy | sad | tense | peaceful | epic | mysterious |     │    │
│  │ dark | heroic | nostalgic | playful                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ GENRE (Thể loại game)                                    │    │
│  │ fantasy | sci-fi | horror | adventure | rpg | puzzle |   │    │
│  │ platformer | simulation | fighting | racing              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ SCENE (Bối cảnh)                                         │    │
│  │ forest | dungeon | village | castle | ocean | space |    │    │
│  │ mountain | desert | city | battlefield                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ TEMPO (Tốc độ)                                           │    │
│  │ very_slow | slow | moderate | fast | very_fast           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ INSTRUMENT (Nhạc cụ chính)                               │    │
│  │ piano | strings | brass | flute | guitar | organ |       │    │
│  │ synth | full_orchestra                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ENERGY (Năng lượng)                                      │    │
│  │ calm | low | medium | high | intense                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ DURATION (Thời lượng mong muốn)                          │    │
│  │ 10s | 15s | 30s | 60s                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 7.2. Ví dụ Prompt

**Ví dụ 1: Nhạc làng quê Fantasy**
```json
{
  "mood": "happy",
  "genre": "fantasy",
  "scene": "village",
  "tempo": "moderate",
  "instrument": "piano",
  "energy": "medium",
  "duration": 30
}
```
→ Text: *"Happy fantasy music for a village scene, moderate tempo, piano, medium energy, 30 seconds"*

**Ví dụ 2: Nhạc trận chiến Epic**
```json
{
  "mood": "epic",
  "genre": "rpg",
  "scene": "battlefield",
  "tempo": "fast",
  "instrument": "full_orchestra",
  "energy": "intense",
  "duration": 60
}
```
→ Text: *"Epic RPG battle music, fast tempo, full orchestra, intense energy, 60 seconds"*

**Ví dụ 3: Nhạc hầm ngục kinh dị**
```json
{
  "mood": "dark",
  "genre": "horror",
  "scene": "dungeon",
  "tempo": "slow",
  "instrument": "strings",
  "energy": "low",
  "duration": 30
}
```
→ Text: *"Dark horror dungeon music, slow tempo, strings, low energy, 30 seconds"*

## 7.3. Chuyển Prompt thành Vector (Prompt Embedding)

### Phương án A: Attribute Embedding (Đề xuất cho đồ án)

Mỗi attribute có một embedding table riêng. Prompt embedding là **tổng hoặc concat** các attribute embeddings.

```python
class PromptEncoder(nn.Module):
    """
    Chuyển structured prompt thành embedding vector.
    Mỗi attribute có embedding riêng → concat → project → prompt embedding.
    """
    
    def __init__(self, d_model=256):
        super().__init__()
        
        # Embedding tables cho mỗi attribute
        self.mood_embedding = nn.Embedding(10, 64)        # 10 moods
        self.genre_embedding = nn.Embedding(10, 64)       # 10 genres
        self.scene_embedding = nn.Embedding(10, 64)       # 10 scenes
        self.tempo_embedding = nn.Embedding(5, 32)        # 5 tempo levels
        self.instrument_embedding = nn.Embedding(8, 32)   # 8 instruments
        self.energy_embedding = nn.Embedding(5, 32)       # 5 energy levels
        
        # Project concatenated embeddings → d_model
        concat_dim = 64 + 64 + 64 + 32 + 32 + 32  # = 288
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, mood, genre, scene, tempo, instrument, energy):
        """
        Mỗi input là tensor chứa index: [batch_size]
        Output: [batch_size, 1, d_model] (1 token = prompt embedding)
        """
        emb = torch.cat([
            self.mood_embedding(mood),
            self.genre_embedding(genre),
            self.scene_embedding(scene),
            self.tempo_embedding(tempo),
            self.instrument_embedding(instrument),
            self.energy_embedding(energy)
        ], dim=-1)  # [B, 288]
        
        prompt_emb = self.projection(emb)  # [B, d_model]
        
        # Thêm sequence dimension cho cross-attention
        return prompt_emb.unsqueeze(1)  # [B, 1, d_model]
```

### Phương án B: Text-based Embedding (Nâng cao)

Sử dụng pre-trained language model (BERT-tiny) để encode free-form text descriptions.

```python
from transformers import AutoTokenizer, AutoModel

class TextPromptEncoder(nn.Module):
    def __init__(self, d_model=256, pretrained='prajjwal1/bert-tiny'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.bert = AutoModel.from_pretrained(pretrained)
        
        # Freeze BERT weights (optional)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Project BERT hidden dim → d_model
        bert_dim = self.bert.config.hidden_size  # 128 for bert-tiny
        self.projection = nn.Linear(bert_dim, d_model)
    
    def forward(self, text_list):
        """
        text_list: list of strings, ví dụ:
          ["Happy fantasy village music, moderate tempo, piano"]
        Output: [batch_size, seq_len, d_model]
        """
        encoded = self.tokenizer(
            text_list, padding=True, truncation=True, 
            max_length=64, return_tensors='pt'
        )
        
        with torch.no_grad():
            bert_output = self.bert(**encoded)
        
        # Sử dụng tất cả token embeddings (không chỉ [CLS])
        hidden_states = bert_output.last_hidden_state  # [B, T, bert_dim]
        
        return self.projection(hidden_states)  # [B, T, d_model]
```

### So sánh hai phương án

| Tiêu chí | Attribute Embedding | Text-based (BERT) |
|---|---|---|
| **Dung lượng** | ~200KB | ~17MB (BERT-tiny) |
| **Tốc độ** | Rất nhanh | Chậm hơn |
| **Tính linh hoạt** | Chỉ hiểu attributes định nghĩa sẵn | Hiểu free-form text |
| **Chất lượng conditioning** | Tốt cho structured prompts | Tốt hơn cho open-ended descriptions |
| **Dễ triển khai** | ★★★★★ | ★★★☆☆ |
| **Phù hợp đồ án** | ★★★★★ | ★★★★☆ |

---

# 8. Thiết Kế Quá Trình Training

## 8.1. Tổng quan Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                              │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ MIDI Dataset │───►│ Tokenizer    │───►│ Token Sequences  │   │
│  │ + Labels     │    │ (REMI)       │    │ + Prompt IDs     │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                   │              │
│                                                   ▼              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ DataLoader (shuffle, batch, pad)                          │   │
│  └────────────────────────────┬─────────────────────────────┘   │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    TRAINING LOOP                          │   │
│  │                                                           │   │
│  │  for epoch in range(num_epochs):                          │   │
│  │      for batch in dataloader:                             │   │
│  │          prompt, tokens = batch                           │   │
│  │                                                           │   │
│  │          # Teacher Forcing:                               │   │
│  │          input  = tokens[:, :-1]   # [BOS, t1, ..., tN-1]│   │
│  │          target = tokens[:, 1:]    # [t1, t2, ..., EOS]  │   │
│  │                                                           │   │
│  │          # Forward pass                                   │   │
│  │          logits = model(input, prompt)                    │   │
│  │          loss = cross_entropy(logits, target)             │   │
│  │                                                           │   │
│  │          # Backward pass                                  │   │
│  │          optimizer.zero_grad()                            │   │
│  │          loss.backward()                                  │   │
│  │          clip_grad_norm_(model.parameters(), max_norm=1)  │   │
│  │          optimizer.step()                                 │   │
│  │          scheduler.step()                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 8.2. Chi tiết từng thành phần

### A. Loss Function: Cross-Entropy Loss

```python
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore <PAD> token
```

**Lý do:** Bài toán next-token prediction là bài toán phân loại đa lớp (vocab_size classes). Cross-entropy loss là lựa chọn chuẩn và tối ưu cho bài toán này.

**Biến thể có thể xem xét:**
- **Label Smoothing (ε=0.1):** Giảm over-confidence, tăng generalization
  ```python
  loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
  ```
- **Focal Loss:** Tập trung vào các tokens khó predict (ít phổ biến)

---

### B. Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,            # Initial learning rate
    betas=(0.9, 0.98),  # Momentum parameters
    eps=1e-9,           # Numerical stability
    weight_decay=0.01   # L2 regularization
)
```

**Lý do chọn AdamW:**
- **Adam** là optimizer chuẩn cho Transformer (được đề xuất trong paper gốc "Attention Is All You Need")
- **AdamW** cải tiến: decoupled weight decay → regularization hiệu quả hơn Adam + L2
- `betas=(0.9, 0.98)` — chuẩn cho Transformer, β₂ cao hơn mặc định (0.999) giúp ổn định hơn với sparse gradients
- `weight_decay=0.01` — ngăn overfitting mà không ảnh hưởng bias/norm layers

---

### C. Learning Rate Scheduler: Cosine Annealing with Warm-up

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_training_steps,
    pct_start=0.05,      # 5% warm-up
    anneal_strategy='cos' # Cosine annealing
)
```

**Hoặc custom warm-up + cosine decay:**

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warm-up
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

**Lý do:**
- **Warm-up (5%):** Transformer rất nhạy cảm với learning rate ban đầu. Tăng dần LR giúp ổn định gradient ở những bước đầu tiên
- **Cosine annealing:** Giảm LR mượt mà → model converge tốt hơn step decay

```
Learning Rate Schedule:

LR ▲
   │    ╱╲
   │   ╱  ╲
   │  ╱    ╲
   │ ╱      ╲
   │╱        ╲              ← Cosine decay
   │          ╲
   │  warm-up  ╲
   │            ╲
   │             ╲────────
   └──────────────────────► Steps
   0    5%              100%
```

---

### D. Batch Size & Gradient Accumulation

```python
# Effective batch size = batch_size × gradient_accumulation_steps
batch_size = 16                    # Per GPU
gradient_accumulation_steps = 4    # Accumulate 4 mini-batches
# Effective batch size = 64
```

**Lý do:**
- `batch_size=16` phù hợp VRAM 24GB (RTX 3090) với seq_len=2048
- `gradient_accumulation=4` cho effective batch size 64 — đủ lớn để gradient ổn định
- Nếu VRAM không đủ: giảm `batch_size` xuống 8, tăng `gradient_accumulation` lên 8

---

### E. Epochs & Training Duration

```python
num_epochs = 50          # Tổng số epochs
eval_every = 1           # Evaluate mỗi epoch
save_every = 5           # Save checkpoint mỗi 5 epochs
early_stopping_patience = 10  # Stop nếu val_loss không giảm sau 10 epochs
```

**Ước tính thời gian training:**
- Dataset: ~50,000 MIDI files × ~1,000 tokens/file = 50M tokens
- Batch size effective: 64
- Steps per epoch: 50M / (64 × 2048) ≈ 380 steps
- Time per step: ~0.5s (RTX 3090)
- Time per epoch: ~3 phút
- Total: ~50 epochs × 3 phút ≈ 2.5 giờ

> [!NOTE]
> Thời gian thực tế có thể dài hơn tùy data loading, evaluation, và checkpointing. Ước tính 5-10 giờ cho full training pipeline.

---

### F. Teacher Forcing

**Chức năng:** Trong training, sử dụng ground-truth token ở vị trí t-1 làm input để predict token ở vị trí t (thay vì dùng prediction của model).

```python
# Teacher Forcing (100% trong training)
input_tokens  = tokens[:, :-1]   # [<BOS>, t₁, t₂, ..., t_{N-1}]
target_tokens = tokens[:, 1:]    # [t₁, t₂, t₃, ..., <EOS>]

logits = model(input_tokens, prompt_embedding)  # [B, N-1, vocab_size]
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    target_tokens.reshape(-1),
    ignore_index=0  # <PAD>
)
```

**Lý do:** Teacher forcing giúp training ổn định và nhanh converge. Nếu dùng prediction của model (scheduled sampling), lỗi sẽ tích lũy (exposure bias), làm training chậm hơn.

**Exposure Bias:** Khi inference, model dùng prediction của chính nó → có thể drift. Giảm thiểu bằng:
- **Scheduled Sampling** (tùy chọn): Giảm dần tỷ lệ teacher forcing qua các epoch
- **Nucleus Sampling** khi inference (xem Mục 9)

---

### G. Regularization

```python
# Dropout
dropout = 0.1  # Applied trong attention, FFN, và embedding

# Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Weight Decay (đã có trong AdamW)
weight_decay = 0.01

# Label Smoothing
label_smoothing = 0.1
```

## 8.3. Training Code tổng hợp

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, train_loader, val_loader, config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=0.1
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader)):
            prompt = batch['prompt'].to(config.device)
            tokens = batch['tokens'].to(config.device)
            
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            
            # Forward
            logits = model(input_tokens, prompt)
            loss = loss_fn(
                logits.reshape(-1, config.vocab_size),
                target_tokens.reshape(-1)
            )
            
            # Gradient Accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss = evaluate(model, val_loader, loss_fn, config)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered!")
                break
```

## 8.4. Tối Ưu Hóa Hiệu Năng (Hardware Optimizations)

Để mô hình có thể huấn luyện hiệu quả và tránh lỗi bộ nhớ (OOM) trên các GPU phổ thông như RTX 4060 (8GB VRAM), hệ thống đã được tích hợp các kỹ thuật sau:

1. **Automatic Mixed Precision (AMP):**
   - Sử dụng `torch.cuda.amp.autocast` kết hợp `GradScaler` trong file `trainer.py`.
   - Các phép tính ma trận lớn của Transformer được thực thi ở chuẩn `float16`, giảm **50% lượng VRAM tiêu thụ** và tăng tốc phần cứng thông qua Tensor Cores của NVIDIA.
   - `GradScaler` đóng vai trò phóng to gradients để tránh bị underflow (mất độ chuẩn xác do số quá nhỏ) khi backprop.

2. **Parallel DataLoader Workers:**
   - Chỉnh `num_workers=4` và `pin_memory=True` trong cấu hình `create_dataloaders`.
   - CPU sẽ giải mã và parse các file MIDI song song rồi đẩy sẵn vào RAM, giúp GPU lúc nào cũng có sẵn batch dữ liệu để train thay vì phải chờ đợi (starvation).

---

# 9. Sinh Nhạc (Inference)

## 9.1. Autoregressive Generation

Mô hình sinh nhạc theo kiểu **autoregressive**: sinh từng token một, mỗi token mới dựa trên tất cả tokens đã sinh trước đó và text condition.

```
Step 1: P(t₁ | <BOS>, condition)           → sinh t₁
Step 2: P(t₂ | <BOS>, t₁, condition)       → sinh t₂
Step 3: P(t₃ | <BOS>, t₁, t₂, condition)   → sinh t₃
...
Step N: P(tₙ | <BOS>, t₁, ..., tₙ₋₁, cond) → sinh tₙ
```

## 9.2. Các thuật toán Sampling

### A. Temperature Sampling

**Ý tưởng:** Điều chỉnh "độ sắc" của phân phối xác suất bằng tham số temperature τ.

```
P(token_i) = softmax(logit_i / τ)
```

| Temperature | Hiệu ứng |
|---|---|
| τ → 0 | Greedy — luôn chọn token có xác suất cao nhất. Nhạc lặp lại, monotone |
| τ = 1.0 | Phân phối gốc — cân bằng giữa quality và diversity |
| τ > 1.0 | Phân phối phẳng hơn — nhạc đa dạng nhưng có thể incoherent |

```python
def temperature_sampling(logits, temperature=0.8):
    """
    logits: [vocab_size] - raw logits từ model
    temperature: float - điều chỉnh diversity
    """
    probs = F.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

**Đề xuất:** τ = 0.8 cho game music — đủ diverse nhưng vẫn coherent.

---

### B. Top-k Sampling (Fan et al., 2018)

**Ý tưởng:** Chỉ giữ lại k tokens có xác suất cao nhất, đặt xác suất còn lại = 0, rồi re-normalize.

```python
def top_k_sampling(logits, k=50, temperature=0.8):
    """
    Chỉ sample từ top-k tokens có xác suất cao nhất.
    """
    logits = logits / temperature
    
    # Tìm top-k
    top_k_values, top_k_indices = torch.topk(logits, k)
    
    # Đặt tất cả giá trị khác = -inf
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(0, top_k_indices, top_k_values)
    
    probs = F.softmax(logits_filtered, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

**Ưu điểm:**
- Loại bỏ các tokens không hợp lý (xác suất rất thấp)
- Đơn giản, dễ hiểu

**Nhược điểm:**
- k cố định — không thích ứng với "độ tin cậy" của model. Khi model rất tự tin (1 token chiếm 90% probability), k=50 vẫn giữ 49 token không cần thiết.

---

### C. Top-p (Nucleus) Sampling (Holtzman et al., 2020)

**Ý tưởng:** Chọn tập tokens nhỏ nhất sao cho tổng xác suất ≥ p. Tự động điều chỉnh số lượng tokens dựa trên phân phối.

```python
def top_p_sampling(logits, p=0.9, temperature=0.8):
    """
    Nucleus sampling: chỉ sample từ tập tokens nhỏ nhất
    có tổng xác suất >= p.
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    # Sort theo xác suất giảm dần
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Tính cumulative probability
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Tìm cutoff: giữ tokens cho đến khi cumsum >= p
    # Shift cumsum sang phải 1 vị trí để giữ token đầu tiên vượt p
    sorted_indices_to_remove = cumulative_probs - sorted_probs > p
    
    # Đặt xác suất = 0 cho tokens bị loại
    sorted_probs[sorted_indices_to_remove] = 0.0
    
    # Re-normalize
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    # Sample
    sampled_index = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices[sampled_index]
    return next_token
```

**Ưu điểm:**
- Tự động điều chỉnh số tokens — khi model tự tin thì chọn ít, khi không chắc chắn thì chọn nhiều
- Kết quả tự nhiên hơn top-k

**Nhược điểm:**
- Phức tạp hơn top-k một chút
- p cần tune — p=0.9 thường là mặc định tốt

---

### D. Beam Search

**Ý tưởng:** Giữ beam_width chuỗi tốt nhất tại mỗi bước, mở rộng tất cả, giữ lại beam_width tốt nhất.

```python
def beam_search(model, prompt_emb, max_length=2048, beam_width=5):
    """
    Beam search cho music generation.
    """
    device = prompt_emb.device
    
    # Khởi tạo beam: [(token_sequence, cumulative_log_prob)]
    beams = [([BOS_TOKEN], 0.0)]
    
    for step in range(max_length):
        all_candidates = []
        
        for seq, score in beams:
            if seq[-1] == EOS_TOKEN:
                all_candidates.append((seq, score))
                continue
            
            input_tensor = torch.tensor([seq], device=device)
            logits = model(input_tensor, prompt_emb)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            
            # Mở rộng top-k candidates
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            
            for log_prob, idx in zip(top_log_probs, top_indices):
                new_seq = seq + [idx.item()]
                new_score = score + log_prob.item()
                all_candidates.append((new_seq, new_score))
        
        # Giữ beam_width tốt nhất
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = beams[:beam_width]
        
        # Check nếu tất cả beams đã kết thúc
        if all(seq[-1] == EOS_TOKEN for seq, _ in beams):
            break
    
    # Trả về sequence tốt nhất
    best_seq, best_score = beams[0]
    return best_seq
```

**Ưu điểm:**
- Tìm được sequence có tổng log-probability cao nhất
- Kết quả ổn định (deterministic)

**Nhược điểm:**
- **Chậm** — cần beam_width forward passes mỗi bước
- Nhạc sinh ra thiếu đa dạng, thường "safe" và boring
- Không phù hợp cho creative generation

---

## 9.3. So sánh các thuật toán sampling

| Tiêu chí | Temperature | Top-k | Top-p (Nucleus) | Beam Search |
|---|---|---|---|---|
| **Diversity** | Tùy τ | Trung bình | Cao | Thấp |
| **Quality** | Tùy τ | Tốt | Rất tốt | Tốt nhất (theo metric) |
| **Tốc độ** | ★★★★★ | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| **Sáng tạo** | Tùy τ | Trung bình | Cao | Thấp |
| **Phù hợp music** | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ |

> [!TIP]
> **Đề xuất cho đồ án:** Kết hợp **Top-p (p=0.9) + Temperature (τ=0.85)** — cho kết quả tốt nhất cho music generation: đủ diverse để tạo nhạc thú vị nhưng vẫn coherent.

## 9.4. Full Inference Pipeline

```python
@torch.no_grad()
def generate_music(model, prompt, max_length=2048, 
                   temperature=0.85, top_p=0.9):
    """
    Sinh nhạc từ prompt.
    
    Args:
        model: Trained Music Transformer
        prompt: dict với mood, genre, scene, tempo, instrument, energy
        max_length: Số tokens tối đa
        temperature: Diversity control
        top_p: Nucleus sampling threshold
    
    Returns:
        List[int] - MIDI token sequence
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    prompt_emb = model.encode_prompt(prompt)  # [1, 1, d_model]
    
    # Start with <BOS>
    generated = [BOS_TOKEN]
    
    for _ in range(max_length - 1):
        input_tensor = torch.tensor([generated], device=device)
        logits = model(input_tensor, prompt_emb)
        next_logits = logits[0, -1, :]  # [vocab_size]
        
        # Top-p + Temperature sampling
        next_token = top_p_sampling(next_logits, p=top_p, 
                                     temperature=temperature)
        
        generated.append(next_token.item())
        
        # Stop nếu sinh <EOS>
        if next_token.item() == EOS_TOKEN:
            break
    
    return generated
```

---

# 10. Chuyển MIDI Sang WAV

## 10.1. Tổng quan pipeline

```
┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│ MIDI Tokens  │────►│  MIDI File   │────►│  WAV File   │
│ (from model) │     │  (.mid)      │     │  (.wav)     │
└──────────────┘     └──────────────┘     └─────────────┘
      │                    │                     │
      │                    │                     │
  Token-to-MIDI        FluidSynth +          Final output
  Decoder              SoundFont (.sf2)      (playable audio)
```

## 10.2. Bước 1: Tokens → MIDI File

```python
import pretty_midi

def tokens_to_midi(tokens, tokenizer, default_tempo=120):
    """
    Chuyển chuỗi token IDs thành file MIDI.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)
    
    # Track thời gian hiện tại
    current_time = 0.0
    current_velocity = 80
    current_instrument_program = 0  # Piano mặc định
    
    # Tạo instruments
    instruments = {}
    
    # Stack theo dõi note_on chưa có note_off
    active_notes = {}  # pitch → (start_time, velocity, instrument)
    
    for token_id in tokens:
        token_str = tokenizer.idx_to_token.get(token_id, '<UNK>')
        
        if token_str.startswith('TIME_SHIFT_'):
            shift = int(token_str.split('_')[-1])
            current_time += shift * 0.01  # 10ms per unit
        
        elif token_str.startswith('VEL_'):
            vel_bin = int(token_str.split('_')[-1])
            current_velocity = min(vel_bin * 4 + 2, 127)
        
        elif token_str.startswith('INST_'):
            inst_name = token_str.replace('INST_', '')
            inst_map = {
                'PIANO': 0, 'STRINGS': 48, 'BRASS': 56,
                'FLUTE': 73, 'GUITAR': 24, 'ORGAN': 16,
                'SYNTH_LEAD': 80, 'ENSEMBLE': 48
            }
            current_instrument_program = inst_map.get(inst_name, 0)
        
        elif token_str.startswith('NOTE_ON_'):
            pitch = int(token_str.split('_')[-1])
            active_notes[pitch] = (
                current_time, current_velocity, 
                current_instrument_program
            )
        
        elif token_str.startswith('NOTE_OFF_'):
            pitch = int(token_str.split('_')[-1])
            if pitch in active_notes:
                start, vel, prog = active_notes.pop(pitch)
                
                # Tạo instrument nếu chưa có
                if prog not in instruments:
                    inst = pretty_midi.Instrument(program=prog)
                    instruments[prog] = inst
                
                # Thêm note
                note = pretty_midi.Note(
                    velocity=vel,
                    pitch=pitch,
                    start=start,
                    end=current_time
                )
                instruments[prog].notes.append(note)
        
        elif token_str.startswith('TEMPO_'):
            tempo_map = {
                'VERY_SLOW': 60, 'SLOW': 80, 'MODERATE_SLOW': 100,
                'MODERATE': 120, 'MODERATE_FAST': 140, 'FAST': 160,
                'VERY_FAST': 180, 'EXTREME': 200
            }
            label = token_str.replace('TEMPO_', '')
            if label in tempo_map:
                # PrettyMIDI doesn't easily change tempo mid-song
                pass  # Tempo set at initialization
    
    # Close any remaining active notes
    for pitch, (start, vel, prog) in active_notes.items():
        if prog not in instruments:
            instruments[prog] = pretty_midi.Instrument(program=prog)
        instruments[prog].notes.append(
            pretty_midi.Note(vel, pitch, start, current_time)
        )
    
    # Add all instruments to MIDI
    for inst in instruments.values():
        midi.instruments.append(inst)
    
    return midi
```

## 10.3. Bước 2: MIDI → WAV (FluidSynth)

### FluidSynth là gì?

**FluidSynth** là một software synthesizer (bộ tổng hợp âm thanh phần mềm) mã nguồn mở. Nó đọc file MIDI và render thành audio sử dụng **SoundFont** — một file chứa các mẫu âm thanh (samples) của nhạc cụ thật.

### SoundFont (.sf2) là gì?

SoundFont chứa recordings thực tế của từng nhạc cụ ở nhiều nốt và cường độ khác nhau. Khi FluidSynth gặp `NOTE_ON(pitch=60, velocity=80, program=0)`, nó tra SoundFont để tìm sample piano tương ứng với nốt C4, velocity 80, rồi phát sample đó.

**SoundFont phổ biến:**

| SoundFont | Dung lượng | Chất lượng | Đặc điểm |
|---|---|---|---|
| FluidR3_GM.sf2 | ~148MB | Tốt | General MIDI đầy đủ, miễn phí |
| TimGM6mb.sf2 | ~6MB | Trung bình | Nhẹ, phù hợp dev/test |
| SGM-V2.01.sf2 | ~256MB | Rất tốt | Nhiều instrument, chất lượng cao |
| MuseScore_General.sf3 | ~36MB | Tốt | Compact, đủ dùng |

### Code render MIDI → WAV

```python
import subprocess

def midi_to_wav_fluidsynth(midi_path, wav_path, 
                            soundfont_path='FluidR3_GM.sf2',
                            sample_rate=44100):
    """
    Render MIDI sang WAV sử dụng FluidSynth command-line.
    
    Yêu cầu: FluidSynth được cài đặt và có trong PATH.
    """
    cmd = [
        'fluidsynth',
        '-ni',                    # Non-interactive mode
        soundfont_path,           # SoundFont file
        midi_path,                # Input MIDI
        '-F', wav_path,           # Output WAV
        '-r', str(sample_rate),   # Sample rate
        '-g', '1.0'              # Gain
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth error: {result.stderr}")
    
    return wav_path


# Hoặc dùng midi2audio (wrapper Python cho FluidSynth)
from midi2audio import FluidSynth

def midi_to_wav_python(midi_path, wav_path, 
                        soundfont_path='FluidR3_GM.sf2'):
    """
    Render MIDI sang WAV sử dụng midi2audio library.
    """
    fs = FluidSynth(soundfont_path)
    fs.midi_to_audio(midi_path, wav_path)
    return wav_path
```

### Pipeline hoàn chỉnh

```python
def generate_and_render(model, prompt, output_dir='output/',
                        soundfont='FluidR3_GM.sf2'):
    """
    Pipeline hoàn chỉnh: Prompt → MIDI → WAV
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate MIDI tokens
    tokens = generate_music(model, prompt, max_length=2048,
                           temperature=0.85, top_p=0.9)
    
    # 2. Tokens → MIDI file
    midi = tokens_to_midi(tokens, tokenizer)
    midi_path = os.path.join(output_dir, 'background_music.mid')
    midi.write(midi_path)
    
    # 3. MIDI → WAV
    wav_path = os.path.join(output_dir, 'background_music.wav')
    midi_to_wav_fluidsynth(midi_path, wav_path, soundfont)
    
    print(f"Generated MIDI: {midi_path}")
    print(f"Generated WAV: {wav_path}")
    print(f"Duration: {midi.get_end_time():.1f}s")
    
    return midi_path, wav_path
```

---

# 11. Đánh Giá Mô Hình

## 11.1. Đánh giá Định lượng (Quantitative Metrics)

### A. Training Loss & Validation Loss

```
Loss ▲
     │╲
     │ ╲   Train Loss
     │  ╲─────────────────────────
     │   ╲
     │    ╲  Val Loss
     │     ╲──────────── ← Best checkpoint
     │      ╲___________
     │
     └──────────────────────────► Epoch
```

- **Mục tiêu:** Val Loss giảm và ổn định
- **Overfitting signal:** Train Loss giảm nhưng Val Loss tăng

---

### B. Perplexity (PPL)

```
PPL = exp(CrossEntropyLoss)
```

**Ý nghĩa:** Trung bình, mô hình "phân vân" giữa bao nhiêu tokens ở mỗi bước. PPL thấp = model tự tin và chính xác hơn.

| PPL | Đánh giá |
|---|---|
| < 10 | Rất tốt — model dự đoán chính xác |
| 10-50 | Tốt — model học được cấu trúc âm nhạc |
| 50-100 | Trung bình — cần cải thiện |
| > 100 | Kém — model chưa học được pattern |

```python
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            prompt = batch['prompt'].to(device)
            
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            
            logits = model(input_tokens, prompt)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_tokens.reshape(-1),
                ignore_index=0,
                reduction='sum'
            )
            
            non_pad = (target_tokens != 0).sum()
            total_loss += loss.item()
            total_tokens += non_pad.item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity
```

---

### C. Pitch Distribution Analysis

**Mục đích:** Kiểm tra xem phân phối nốt nhạc sinh ra có tương tự phân phối thực tế không.

```python
import numpy as np
from scipy.stats import wasserstein_distance

def pitch_distribution_distance(generated_tokens, real_tokens, tokenizer):
    """
    Tính khoảng cách Wasserstein giữa pitch distributions.
    """
    def extract_pitches(tokens):
        pitches = []
        for t in tokens:
            token_str = tokenizer.idx_to_token.get(t, '')
            if token_str.startswith('NOTE_ON_'):
                pitch = int(token_str.split('_')[-1])
                pitches.append(pitch)
        return pitches
    
    gen_pitches = extract_pitches(generated_tokens)
    real_pitches = extract_pitches(real_tokens)
    
    # Histogram (128 bins for MIDI pitch range)
    gen_hist, _ = np.histogram(gen_pitches, bins=128, range=(0, 127), 
                                density=True)
    real_hist, _ = np.histogram(real_pitches, bins=128, range=(0, 127), 
                                 density=True)
    
    # Wasserstein distance (Earth Mover's Distance)
    distance = wasserstein_distance(gen_hist, real_hist)
    return distance
```

**Khoảng cách thấp** → phân phối pitch giống nhau → nhạc sinh ra "nghe" hợp lý.

---

### D. Rhythm Consistency

**Mục đích:** Đo mức độ đều đặn của nhịp trong nhạc sinh ra.

```python
def rhythm_consistency(generated_tokens, tokenizer):
    """
    Đo độ nhất quán của inter-onset intervals (IOIs).
    IOI = khoảng thời gian giữa 2 note_on liên tiếp.
    """
    onsets = []
    current_time = 0.0
    
    for t in generated_tokens:
        token_str = tokenizer.idx_to_token.get(t, '')
        if token_str.startswith('TIME_SHIFT_'):
            shift = int(token_str.split('_')[-1])
            current_time += shift * 0.01
        elif token_str.startswith('NOTE_ON_'):
            onsets.append(current_time)
    
    if len(onsets) < 2:
        return 0.0
    
    # Tính IOIs
    iois = np.diff(onsets)
    
    # Coefficient of variation (CV) = std/mean
    # CV thấp → rhythm đều đặn
    if np.mean(iois) > 0:
        cv = np.std(iois) / np.mean(iois)
    else:
        cv = float('inf')
    
    # Normalize: 1 - cv (clipped to [0, 1])
    consistency = max(0, 1 - cv)
    return consistency
```

---

### E. Note Density

**Mục đích:** Đếm số nốt trên mỗi giây — kiểm tra xem nhạc có quá thưa hoặc quá dày không.

```python
def note_density(generated_tokens, tokenizer):
    """Tính số nốt trên giây."""
    num_notes = sum(1 for t in generated_tokens 
                    if tokenizer.idx_to_token.get(t, '').startswith('NOTE_ON_'))
    
    # Tổng thời gian
    total_time = 0.0
    for t in generated_tokens:
        token_str = tokenizer.idx_to_token.get(t, '')
        if token_str.startswith('TIME_SHIFT_'):
            total_time += int(token_str.split('_')[-1]) * 0.01
    
    if total_time > 0:
        return num_notes / total_time
    return 0.0
```

| Note Density | Đánh giá |
|---|---|
| < 1.0 notes/s | Rất thưa — ambient, drone |
| 1.0 - 3.0 | Bình thường — background music |
| 3.0 - 8.0 | Dày — action music |
| > 8.0 | Rất dày — có thể quá chaotic |

---

## 11.2. Đánh giá Định tính (Qualitative Evaluation)

### A. Tiêu chí đánh giá chủ quan

| Tiêu chí | Mô tả | Thang điểm |
|---|---|---|
| **Tính tự nhiên (Naturalness)** | Nhạc nghe có tự nhiên, mượt mà không? | 1-5 (1=rất giả, 5=rất tự nhiên) |
| **Phù hợp với game (Relevance)** | Nhạc có phù hợp với mô tả (mood, scene)? | 1-5 |
| **Độ mượt (Smoothness)** | Các nốt chuyển tiếp có mượt không? | 1-5 |
| **Không lặp (Non-repetitiveness)** | Nhạc có đa dạng, không lặp lại quá nhiều? | 1-5 |
| **Cảm xúc (Emotional Fit)** | Nhạc có truyền tải đúng cảm xúc mong muốn? | 1-5 |
| **Tổng thể (Overall Quality)** | Đánh giá tổng thể chất lượng nhạc | 1-5 |

---

### B. Khảo sát người dùng (User Study)

**Thiết kế khảo sát:**

1. **Participants:** 20-30 người (game developers, musicians, gamers)
2. **Stimuli:** 10 cặp (prompt, generated music)
3. **Phương pháp:** Mean Opinion Score (MOS)

**Quy trình:**
```
1. Hiển thị text prompt cho người nghe
2. Phát nhạc sinh ra
3. Người nghe đánh giá 6 tiêu chí (thang 1-5)
4. So sánh với nhạc gốc (A/B test) — tùy chọn
```

**Kết quả mong đợi:**

| Tiêu chí | Mục tiêu MOS |
|---|---|
| Tính tự nhiên | ≥ 3.5 |
| Phù hợp game | ≥ 3.0 |
| Tổng thể | ≥ 3.5 |

---

## 11.3. Bảng tổng hợp các metrics

```
┌─────────────────────────────────────────────────────────────┐
│              EVALUATION METRICS DASHBOARD                    │
│                                                              │
│  QUANTITATIVE                  │  QUALITATIVE                │
│  ─────────────                 │  ───────────                │
│  ☐ Cross-Entropy Loss          │  ☐ Naturalness (MOS)        │
│  ☐ Perplexity (PPL)           │  ☐ Game Relevance (MOS)     │
│  ☐ Pitch Distribution (EMD)   │  ☐ Smoothness (MOS)         │
│  ☐ Rhythm Consistency (CV)    │  ☐ Non-Repetitiveness (MOS) │
│  ☐ Note Density (notes/s)     │  ☐ Emotional Fit (MOS)      │
│  ☐ Self-Similarity (SSM)      │  ☐ Overall Quality (MOS)    │
│                                │  ☐ A/B Test vs Real Music   │
└─────────────────────────────────────────────────────────────┘
```

---

# 12. Thiết Kế Demo

## 12.1. Kiến trúc Demo Application

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEMO APPLICATION                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    FRONTEND (Web UI)                        │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  ♪ Text-to-Music Generator                           │  │  │
│  │  │                                                      │  │  │
│  │  │  Mood:       [▼ Happy        ]                       │  │  │
│  │  │  Genre:      [▼ Fantasy      ]                       │  │  │
│  │  │  Scene:      [▼ Forest       ]                       │  │  │
│  │  │  Tempo:      [▼ Fast         ]                       │  │  │
│  │  │  Instrument: [▼ Piano        ]                       │  │  │
│  │  │  Energy:     [▼ Medium       ]                       │  │  │
│  │  │  Duration:   [    30    ] seconds                     │  │  │
│  │  │                                                      │  │  │
│  │  │         [ 🎵 Generate Music ]                        │  │  │
│  │  │                                                      │  │  │
│  │  │  ┌────────────────────────────────────────────────┐  │  │  │
│  │  │  │  🎼 Generated Music                            │  │  │  │
│  │  │  │                                                │  │  │  │
│  │  │  │  ▶ ──────●───────────────── 0:15 / 0:30      │  │  │  │
│  │  │  │                                                │  │  │  │
│  │  │  │  📊 Piano Roll Visualization                   │  │  │  │
│  │  │  │  ┌────────────────────────────────────────┐    │  │  │  │
│  │  │  │  │ ■■    ■■■  ■   ■■■■   ■■  ■■■        │    │  │  │  │
│  │  │  │  │  ■■  ■   ■■ ■■    ■■ ■  ■■   ■■      │    │  │  │  │
│  │  │  │  │   ■■■     ■   ■■    ■     ■■    ■■    │    │  │  │  │
│  │  │  │  └────────────────────────────────────────┘    │  │  │  │
│  │  │  │                                                │  │  │  │
│  │  │  │  [ ⬇ Download MIDI ]  [ ⬇ Download WAV ]     │  │  │  │
│  │  │  └────────────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                             │ HTTP API                           │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    BACKEND (FastAPI)                        │  │
│  │                                                            │  │
│  │  POST /generate                                            │  │
│  │  ├── Input: {mood, genre, scene, tempo, instrument, ...}  │  │
│  │  ├── Process: Prompt → Model → MIDI → WAV                │  │
│  │  └── Output: {midi_url, wav_url, duration, metadata}      │  │
│  │                                                            │  │
│  │  GET /download/{file_id}                                   │  │
│  │  └── Download generated MIDI or WAV file                  │  │
│  │                                                            │  │
│  │  GET /health                                               │  │
│  │  └── Health check endpoint                                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    MODEL SERVICE                            │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐│  │
│  │  │ Music        │  │ Token-to-    │  │ FluidSynth       ││  │
│  │  │ Transformer  │──│ MIDI Decoder │──│ WAV Renderer     ││  │
│  │  │ (PyTorch)    │  │              │  │                  ││  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘│  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 12.2. Backend API (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import os

app = FastAPI(title="Text-to-Music Generator")

class MusicRequest(BaseModel):
    mood: str = "happy"
    genre: str = "fantasy"
    scene: str = "village"
    tempo: str = "moderate"
    instrument: str = "piano"
    energy: str = "medium"
    duration: int = 30
    temperature: float = 0.85
    top_p: float = 0.9

class MusicResponse(BaseModel):
    request_id: str
    midi_url: str
    wav_url: str
    duration: float
    num_notes: int

# Load model on startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = MusicTransformer.load('best_model.pt')
    model.eval()

@app.post("/generate", response_model=MusicResponse)
async def generate_music_endpoint(request: MusicRequest):
    request_id = str(uuid.uuid4())[:8]
    output_dir = f"outputs/{request_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    prompt = {
        'mood': request.mood,
        'genre': request.genre,
        'scene': request.scene,
        'tempo': request.tempo,
        'instrument': request.instrument,
        'energy': request.energy,
    }
    
    midi_path, wav_path = generate_and_render(
        model, prompt, output_dir,
        temperature=request.temperature,
        top_p=request.top_p
    )
    
    return MusicResponse(
        request_id=request_id,
        midi_url=f"/download/{request_id}/midi",
        wav_url=f"/download/{request_id}/wav",
        duration=pretty_midi.PrettyMIDI(midi_path).get_end_time(),
        num_notes=count_notes(midi_path)
    )

@app.get("/download/{request_id}/{format}")
async def download(request_id: str, format: str):
    ext = 'mid' if format == 'midi' else 'wav'
    path = f"outputs/{request_id}/background_music.{ext}"
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)
```

---

# 13. Công Nghệ Sử Dụng

## 13.1. Tổng hợp Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  CORE / DEEP LEARNING                                     │   │
│  │  ├── Python 3.10+                                         │   │
│  │  ├── PyTorch 2.0+ (model, training, inference)           │   │
│  │  ├── NumPy (numerical computing)                          │   │
│  │  └── SciPy (evaluation metrics)                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MUSIC PROCESSING                                         │   │
│  │  ├── pretty_midi (MIDI read/write, analysis)             │   │
│  │  ├── mido (low-level MIDI I/O)                           │   │
│  │  ├── music21 (music theory analysis - tùy chọn)          │   │
│  │  ├── FluidSynth (MIDI → WAV rendering)                   │   │
│  │  ├── midi2audio (Python wrapper cho FluidSynth)          │   │
│  │  └── SoundFont (.sf2) — FluidR3_GM hoặc SGM-V2          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DATA PROCESSING                                          │   │
│  │  ├── pandas (dataset management)                          │   │
│  │  ├── tqdm (progress bars)                                 │   │
│  │  └── h5py hoặc lmdb (efficient data storage - tùy chọn) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  VISUALIZATION & MONITORING                               │   │
│  │  ├── TensorBoard (training monitoring)                    │   │
│  │  ├── matplotlib (plots, piano roll)                       │   │
│  │  └── Weights & Biases (wandb) — tùy chọn                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DEPLOYMENT                                               │   │
│  │  ├── FastAPI (REST API backend)                           │   │
│  │  ├── Uvicorn (ASGI server)                                │   │
│  │  ├── Docker (containerization)                            │   │
│  │  ├── Gradio (quick demo UI — alternative)                │   │
│  │  └── HTML/CSS/JS (custom frontend)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OPTIONAL / ADVANCED                                      │   │
│  │  ├── Transformers (HuggingFace — cho text encoder)       │   │
│  │  ├── librosa (audio analysis)                             │   │
│  │  ├── soundfile (WAV I/O)                                  │   │
│  │  └── pytest (testing)                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  HARDWARE                                                 │   │
│  │  ├── GPU: NVIDIA RTX 3090 (24GB) hoặc tương đương       │   │
│  │  ├── RAM: 32GB+                                           │   │
│  │  ├── Storage: 100GB+ (dataset + checkpoints)             │   │
│  │  └── CUDA 11.8+ / cuDNN 8.6+                             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 13.2. Cấu trúc thư mục dự án

```
text-to-music/
├── config/
│   └── config.yaml              # Hyperparameters
├── data/
│   ├── raw/                     # Raw MIDI files
│   ├── processed/               # Tokenized data
│   └── labels/                  # Text labels/annotations
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py       # Music Transformer
│   │   ├── attention.py         # Self-Attention + Cross-Attention
│   │   ├── embedding.py         # Token + Position embeddings
│   │   ├── prompt_encoder.py    # Text/Attribute encoder
│   │   └── layers.py            # FFN, LayerNorm, etc.
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py         # MIDI Tokenizer (REMI)
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── preprocessing.py     # Data cleaning
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop
│   │   ├── scheduler.py         # LR scheduler
│   │   └── evaluator.py         # Metrics
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generator.py         # Music generation
│   │   ├── sampling.py          # Sampling strategies
│   │   └── renderer.py          # MIDI → WAV
│   └── utils/
│       ├── __init__.py
│       ├── midi_utils.py
│       └── visualization.py     # Piano roll plotting
├── api/
│   ├── main.py                  # FastAPI app
│   └── schemas.py               # Request/Response models
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_evaluation.ipynb
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_generation.py
├── soundfonts/
│   └── FluidR3_GM.sf2
├── checkpoints/
│   └── best_model.pt
├── outputs/
│   ├── background_music.mid
│   └── background_music.wav
├── Dockerfile
├── requirements.txt
├── train.py                     # Training script
├── generate.py                  # Inference script
└── README.md
```

## 13.3. requirements.txt

```
# Core
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Music Processing
pretty_midi>=0.2.10
mido>=1.3.0
midi2audio>=0.1.1

# Data
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0

# Visualization
tensorboard>=2.13.0
matplotlib>=3.7.0

# API
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6

# Optional
# transformers>=4.30.0    # Nếu dùng BERT text encoder
# librosa>=0.10.0         # Audio analysis
# wandb>=0.15.0           # Experiment tracking
```

---

# 14. Lộ Trình Thực Hiện

## 14.1. Kế hoạch 12 tuần

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECT TIMELINE (12 WEEKS)                   │
│                                                                  │
│  PHASE 1: RESEARCH & SETUP (Tuần 1-2)                          │
│  ═══════════════════════════════════════                         │
│                                                                  │
│  Tuần 1: Khảo sát tài liệu                                     │
│  ├── Đọc papers: Music Transformer, MusicLM, MusicGen          │
│  ├── Khảo sát các implementation mã nguồn mở                   │
│  ├── Xác định kiến trúc mô hình                                 │
│  ├── Viết phần giới thiệu và phân tích bài toán (báo cáo)      │
│  └── Deliverable: Research survey document                      │
│                                                                  │
│  Tuần 2: Thu thập và khám phá dataset                           │
│  ├── Download Lakh MIDI, MAESTRO, Game MIDI datasets            │
│  ├── Phân tích phân phối dữ liệu (pitch, tempo, duration)      │
│  ├── Lọc và clean dữ liệu                                       │
│  ├── Thiết kế MIDI tokenizer (REMI)                              │
│  └── Deliverable: Clean dataset + EDA notebook                  │
│                                                                  │
│  PHASE 2: DATA PIPELINE (Tuần 3-4)                              │
│  ═════════════════════════════════════                           │
│                                                                  │
│  Tuần 3: Tiền xử lý dữ liệu                                    │
│  ├── Implement MidiTokenizer hoàn chỉnh                         │
│  ├── Tokenize toàn bộ dataset                                    │
│  ├── Tạo text labels (rule-based + manual)                      │
│  ├── Xây dựng PyTorch Dataset và DataLoader                     │
│  └── Deliverable: Data pipeline code + processed data           │
│                                                                  │
│  Tuần 4: Thiết kế Prompt và Text Encoder                        │
│  ├── Implement PromptEncoder (Attribute Embedding)              │
│  ├── Thiết kế prompt schema cho game music                       │
│  ├── Tạo prompt-music pairs cho training                        │
│  ├── Test pipeline: prompt → embedding → verify                 │
│  └── Deliverable: Prompt encoder + paired dataset               │
│                                                                  │
│  PHASE 3: MODEL DEVELOPMENT (Tuần 5-7)                          │
│  ═════════════════════════════════════════                       │
│                                                                  │
│  Tuần 5: Xây dựng Music Transformer (Core)                      │
│  ├── Implement Relative Position Encoding                       │
│  ├── Implement Multi-Head Self-Attention with RPE               │
│  ├── Implement Cross-Attention                                   │
│  ├── Implement Feed-Forward Network                              │
│  ├── Assemble Decoder Block                                      │
│  ├── Unit tests cho mỗi component                                │
│  └── Deliverable: Model code + unit tests passed                │
│                                                                  │
│  Tuần 6: Training (Phase 1 — Pre-training)                      │
│  ├── Setup training loop + optimizer + scheduler                │
│  ├── Pre-train trên Lakh MIDI (general music)                   │
│  ├── Monitor loss, perplexity trên TensorBoard                  │
│  ├── Debug training issues (NaN, divergence)                    │
│  └── Deliverable: Pre-trained model checkpoint                  │
│                                                                  │
│  Tuần 7: Training (Phase 2 — Fine-tuning)                       │
│  ├── Fine-tune trên Game MIDI dataset                            │
│  ├── Train với text conditioning (prompt-music pairs)           │
│  ├── Tune hyperparameters (lr, batch_size, dropout)             │
│  ├── Early stopping + checkpoint selection                      │
│  └── Deliverable: Fine-tuned model + training curves            │
│                                                                  │
│  PHASE 4: INFERENCE & EVALUATION (Tuần 8-9)                    │
│  ════════════════════════════════════════════                    │
│                                                                  │
│  Tuần 8: Sinh nhạc và đánh giá                                  │
│  ├── Implement sampling algorithms (top-p, top-k, temperature)  │
│  ├── Implement token-to-MIDI decoder                            │
│  ├── Setup FluidSynth + SoundFont                               │
│  ├── Generate mẫu nhạc cho mỗi prompt type                     │
│  ├── Tính quantitative metrics                                   │
│  └── Deliverable: Generated samples + metrics report            │
│                                                                  │
│  Tuần 9: Cải thiện và đánh giá chuyên sâu                      │
│  ├── Tune sampling parameters (temperature, top_p)              │
│  ├── So sánh các sampling strategies                             │
│  ├── Qualitative evaluation (user listening test)               │
│  ├── Phân tích lỗi thường gặp                                   │
│  └── Deliverable: Evaluation report + improved samples          │
│                                                                  │
│  PHASE 5: DEMO & DOCUMENTATION (Tuần 10-12)                    │
│  ═══════════════════════════════════════════                     │
│                                                                  │
│  Tuần 10: Xây dựng Demo Application                             │
│  ├── Implement FastAPI backend                                   │
│  ├── Build web frontend (HTML/CSS/JS)                           │
│  ├── Tích hợp model serving                                      │
│  ├── Test end-to-end pipeline                                    │
│  └── Deliverable: Working demo application                      │
│                                                                  │
│  Tuần 11: Docker + Polish                                       │
│  ├── Dockerize application                                       │
│  ├── Optimize inference speed                                    │
│  ├── UI polish (piano roll visualization)                        │
│  ├── Error handling + edge cases                                 │
│  └── Deliverable: Dockerized demo + documentation               │
│                                                                  │
│  Tuần 12: Báo cáo cuối cùng                                     │
│  ├── Viết báo cáo nghiên cứu hoàn chỉnh                        │
│  ├── Chuẩn bị slide thuyết trình                                 │
│  ├── Record demo video                                           │
│  ├── Tổng hợp code + documentation trên GitHub                  │
│  └── Deliverable: Final report + presentation + demo video      │
└─────────────────────────────────────────────────────────────────┘
```

## 14.2. Gantt Chart (ASCII)

```
Tuần          1    2    3    4    5    6    7    8    9   10   11   12
             ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Research     ████│    │    │    │    │    │    │    │    │    │    │
Dataset      │    ████│    │    │    │    │    │    │    │    │    │
Preprocess   │    │    ████│    │    │    │    │    │    │    │    │
Prompt       │    │    │    ████│    │    │    │    │    │    │    │
Model Build  │    │    │    │    ████│    │    │    │    │    │    │
Pre-training │    │    │    │    │    ████│    │    │    │    │    │
Fine-tuning  │    │    │    │    │    │    ████│    │    │    │    │
Inference    │    │    │    │    │    │    │    ████│    │    │    │
Evaluation   │    │    │    │    │    │    │    │    ████│    │    │
Demo Build   │    │    │    │    │    │    │    │    │    ████│    │
Docker       │    │    │    │    │    │    │    │    │    │    ████│
Report       │    │    │    │    │    │    │    │    │    │    │    ████
```

## 14.3. Milestones

| Milestone | Tuần | Deliverable |
|---|---|---|
| **M1: Research Complete** | 1 | Research survey, kiến trúc xác định |
| **M2: Data Ready** | 3 | Tokenized dataset, DataLoader hoạt động |
| **M3: Model Training** | 6 | Pre-trained model, loss curve giảm |
| **M4: Music Generation** | 8 | Sinh nhạc thành công từ prompt |
| **M5: Demo Ready** | 10 | Web demo hoạt động end-to-end |
| **M6: Final Submission** | 12 | Báo cáo, code, demo video |

---

# 15. Hướng Phát Triển

## 15.1. Các hướng mở rộng ngắn hạn

### A. Text-to-Music bằng Diffusion Model

Thay thế hoặc bổ sung Music Transformer bằng **Latent Diffusion Model trên piano roll**:

```
Text Prompt → Text Encoder → Cross-Attention
                                    ↓
            Noise ──► U-Net (Denoising) ──► Clean Piano Roll ──► MIDI
                        ↑
                   Noise Scheduler
```

**Lợi ích:**
- Chất lượng sinh cao hơn (diffusion models excel at generation quality)
- Classifier-free guidance cho controlling output
- Không bị exposure bias (không autoregressive)

**Thách thức:**
- Cần thiết kế representation phù hợp cho piano roll diffusion
- Sampling chậm hơn Transformer

---

### B. Điều khiển nhiều nhạc cụ (Multi-Track Generation)

Mở rộng từ single-instrument sang **multi-track orchestration**:

```
Prompt: "Epic battle music, full orchestra"
                    ↓
        ┌───────────────────────┐
        │  Multi-Track Model    │
        └───────────┬───────────┘
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    ┌───────┐ ┌──────────┐ ┌────────┐
    │Melody │ │ Harmony  │ │  Bass  │
    │Track  │ │  Track   │ │ Track  │
    └───────┘ └──────────┘ └────────┘
        ↓           ↓           ↓
    Piano       Strings      Cello
```

**Approaches:**
- **Sequential:** Sinh melody → harmony → bass tuần tự (mỗi track conditioned trên tracks trước)
- **Parallel:** Sinh tất cả tracks đồng thời với multi-head attention across tracks
- **Hierarchical:** High-level structure → individual track details

---

### C. Adaptive Music — Đồng bộ nhạc theo trạng thái game

**Ý tưởng:** Nhạc thay đổi real-time dựa trên gameplay events.

```
┌──────────────────────────────────────────────────────────────┐
│                   ADAPTIVE MUSIC SYSTEM                       │
│                                                               │
│  Game State:                    Music Response:               │
│  ─────────────                  ────────────────              │
│  Exploring village      →      Calm piano melody              │
│  Enemy approaching      →      Tempo tăng, thêm strings      │
│  Boss battle starts     →      Full orchestra, fast tempo     │
│  Victory                →      Major key, triumphant brass    │
│  Game over              →      Slow, minor key, sad strings   │
│                                                               │
│  ┌─────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ Game    │───►│ State        │───►│ Music             │   │
│  │ Engine  │    │ Classifier   │    │ Transition Engine │   │
│  └─────────┘    └──────────────┘    └───────────────────┘   │
│                                            │                  │
│                                            ▼                  │
│                                    ┌───────────────┐         │
│                                    │ Cross-fade    │         │
│                                    │ Controller    │         │
│                                    └───────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

**Kỹ thuật:**
- **Horizontal re-sequencing:** Chuyển đổi giữa các đoạn nhạc pre-generated
- **Vertical remixing:** Thêm/bớt layers (tracks) real-time
- **Real-time generation:** Sinh nhạc mới mỗi vài giây dựa trên game state

---

### D. Sinh nhạc theo thời gian thực (Real-time Generation)

**Thách thức:** Transformer autoregressive quá chậm cho real-time.

**Giải pháp tiềm năng:**
- **Speculative decoding:** Dùng model nhỏ draft + model lớn verify
- **Knowledge distillation:** Train model nhỏ từ model lớn
- **Non-autoregressive models:** Sinh toàn bộ sequence trong 1 forward pass
- **Streaming generation:** Sinh từng đoạn ngắn (2-4 giây), cross-fade

---

### E. Hệ thống AI tạo nội dung game hoàn chỉnh

**Tầm nhìn dài hạn:** Kết hợp Text-to-Music với các hệ thống AI khác để tạo trải nghiệm game hoàn chỉnh.

```
┌─────────────────────────────────────────────────────────────────┐
│              AI GAME CONTENT GENERATION PLATFORM                 │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Text-to-    │  │ Text-to-     │  │ Text-to-              │  │
│  │ Music       │  │ Environment  │  │ Character             │  │
│  │ (This       │  │ (Procedural  │  │ (3D Generation,       │  │
│  │  project)   │  │  world gen)  │  │  Animation)           │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬────────────┘  │
│         │                │                      │               │
│         └────────────────┼──────────────────────┘               │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  Game Engine          │                          │
│              │  Integration Layer    │                          │
│              │  (Unity / Unreal)     │                          │
│              └───────────────────────┘                          │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  Complete AI-         │                          │
│              │  Generated Game       │                          │
│              │  Experience           │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## 15.2. Bảng tóm tắt hướng phát triển

| Hướng | Độ khó | Tác động | Thời gian ước tính |
|---|---|---|---|
| Diffusion Model cho MIDI | ★★★★☆ | Cải thiện chất lượng | 4-6 tuần |
| Multi-track generation | ★★★☆☆ | Nhạc phong phú hơn | 3-4 tuần |
| Adaptive Music | ★★★★☆ | Game integration | 6-8 tuần |
| Real-time generation | ★★★★★ | Production-ready | 8-12 tuần |
| Full AI game platform | ★★★★★ | Paradigm shift | 6-12 tháng |

---

# Tài Liệu Tham Khảo

1. **Vaswani, A., et al.** (2017). *Attention Is All You Need.* NeurIPS 2017. — Kiến trúc Transformer gốc.

2. **Huang, C.Z.A., et al.** (2018). *Music Transformer: Generating Music with Long-Term Structure.* ICLR 2019. — Relative attention cho music generation.

3. **Payne, C.** (2019). *MuseNet.* OpenAI Blog. — Transformer lớn cho sinh nhạc đa nhạc cụ.

4. **Dhariwal, P., et al.** (2020). *Jukebox: A Generative Model for Music.* arXiv:2005.00341. — VQ-VAE + Transformer cho raw audio.

5. **Agostinelli, A., et al.** (2023). *MusicLM: Generating Music From Text.* arXiv:2301.11325. — Text-to-Music hierarchical.

6. **Copet, J., et al.** (2023). *Simple and Controllable Music Generation.* arXiv:2306.05284 (MusicGen, Meta). — Single-stage text-to-music.

7. **Huang, Y.S., & Yang, Y.H.** (2020). *Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions.* ACM MM 2020. — REMI representation.

8. **Ho, J., et al.** (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. — Foundation cho diffusion models.

9. **Rombach, R., et al.** (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. — Latent Diffusion (Stable Diffusion).

10. **Holtzman, A., et al.** (2020). *The Curious Case of Neural Text Degeneration.* ICLR 2020. — Nucleus (top-p) sampling.

11. **Hawthorne, C., et al.** (2019). *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset.* ICLR 2019. — MAESTRO dataset.

12. **Raffel, C.** (2016). *Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching.* PhD Thesis, Columbia University. — Lakh MIDI Dataset.

13. **Evans, Z., et al.** (2024). *Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion.* arXiv:2402.04825. — Timing-conditioned diffusion.

14. **Liu, H., et al.** (2023). *AudioLDM: Text-to-Audio Generation with Latent Diffusion Models.* ICML 2023. — Latent Diffusion cho audio.

15. **Fan, A., Lewis, M., & Dauphin, Y.** (2018). *Hierarchical Neural Story Generation.* ACL 2018. — Top-k sampling.

---

> [!NOTE]
> **Ghi chú cuối cùng:** Báo cáo này thiết kế một hệ thống Text-to-Music hoàn chỉnh có thể tự xây dựng trong khuôn khổ đồ án nghiên cứu. Kiến trúc **Conditional Music Transformer** với **Relative Position Encoding** và **Cross-Attention Text Conditioning** được chọn vì cân bằng tốt nhất giữa chất lượng, độ khó triển khai, và tài nguyên GPU yêu cầu. Toàn bộ mô hình ước tính ~8M parameters, có thể train trên 1× RTX 3090 trong vài giờ.
