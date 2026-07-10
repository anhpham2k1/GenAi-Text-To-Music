# Diffusion Text-to-Music ↔ Kiến thức slide IT5410

Tài liệu này **bám slide môn Nền tảng AI tạo sinh**, chủ yếu **Week8–Diffusion**, bổ sung Week5–VAE, Week6–AR, Week4–Transformer, Week2–Learning Theory.  
Mục tiêu: giải thích **vì sao** kiến trúc/cải tiến hiện tại đúng lý thuyết môn, và **cải gì tiếp theo** vẫn nằm trong slide (không nhảy lung tung SOTA ngoài giáo trình).

---

## 1. Bài toán trên slide → bài toán đồ án

| Slide | Nội dung | Trong đồ án |
|-------|----------|-------------|
| Week8 p10 | Diffusion = **forward** (thêm noise) + **reverse** (denoising sinh dữ liệu) | Piano-roll `x0` → `q(x_t\|x0)` → mạng học reverse |
| Week8 p16 | Reverse dùng **trainable network** (U-Net / denoising AE) | `ConditionalUNet` |
| Week8 p21 | Parameterize mean qua **noise-prediction network** ε_θ | Loss MSE: `ε_θ(x_t,t,c) ≈ noise` |
| Week8 Part2 | **Conditional generation** (text/class → sample) | Prompt 6 attribute → condition `c` |
| Week6 | Chain rule AR `p(x)=∏p(x_t\|x_<t)` | **Đối thủ so sánh**: Music Transformer (không phải diffusion) |
| Week5 | VAE: latent + ELBO | Slide p26: diffusion ≈ hierarchical VAE (encoder cố định) — chỉ **liên hệ lý thuyết** |
| Week4 | Attention / U-Net blocks | Mid **self-attention** trong U-Net (slide p24: U-Net + self-attn) |
| Week2 | Loss thực nghiệm, generalization | Train/val MSE, early/best ckpt, so epoch 1/5/10 |

**Không** làm audio waveform diffusion kiểu Imagen/audio — slide Part3 là text-to-**image**; ta map tương tự:  
**text/prompt condition → “ảnh” piano-roll 2D (pitch × time) → decode MIDI**.

---

## 2. Phần (1) DDPM trên slide — đã map vào code

### 2.1. Forward process (slide p11–14)

\[
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1},\,\beta_t I)
\]

- Code: `GaussianDiffusion.q_sample` dùng \(\bar\alpha_t\), \(\sqrt{1-\bar\alpha_t}\).  
- \(x_0\): piano-roll chuẩn hóa **[-1, 1]** (cùng tinh thần data continuous trên slide).

### 2.2. Reverse / noise prediction (slide p16–21)

Ho et al.: học **ε_θ** thay vì mean tường minh.

- Code: `p_losses` → `MSE(ε_θ(x_t, t, c), ε)`.  
- Đúng “parameterizing the denoising model” trên slide p21.

### 2.3. Objective weighting (slide p22)

Slide: full variational weight phức tạp; **Ho et al. đặt weight = 1** (simple MSE) → sample quality tốt hơn likelihood thuần.

- Code hiện tại: **simple MSE** (đúng lựa chọn slide, không ELBO đầy đủ từng term).  
- *Cải tùy chọn (vẫn slide)*: SNR / perception-prioritized weighting (Choi et al., slide p22 “for more advanced…”).

### 2.4. Noise schedule (slide p25)

Slide: thường **linear** β; cũng nhắc parameterization SNR / học schedule (Kingma).

- Code: **cosine schedule** (cải tiến phổ biến, cùng họ “diffusion parameters”) + vẫn hỗ trợ `linear`.  
- Nói với thầy: *schedule ∈ {linear, cosine} là hyperparameter của forward process* (slide p25, p27 “devil in the details”).

### 2.5. Architecture (slide p24, p27)

Slide: **U-Net + ResNet + self-attention**, time embedding.

- Code: U-Net 2D + ResBlock + **sinusoidal time emb** + **mid self-attention** + FiLM(time, cond).  
- Time embedding: cùng họ positional/time representation trên slide.

### 2.6. Liên hệ VAE (slide p26)

Slide: diffusion = hierarchical VAE đặc biệt:

| VAE (Week5) | Diffusion (Week8) | Đồ án |
|-------------|-------------------|--------|
| Encoder học q(z\|x) | Encoder **fixed** (forward noise) | `q_sample` cố định |
| Latent dim nhỏ | Latent **cùng dim data** | noise cùng shape piano-roll |
| Decoder riêng | **Shared** denoiser mọi t | 1 UNet mọi timestep |
| ELBO | Reweight variational bound | MSE noise (reweight đơn giản) |

→ Khi bảo vệ: “Chọn diffusion theo Week8, **không** train VAE Week5; nhưng hiểu diffusion như VAE phân cấp có encoder cố định.”

---

## 3. Phần (2) Conditional generation — **trọng tâm gen “đúng prompt”**

Slide p31–49 nêu **3 cách control**:

```
Explicit Conditioning | Classifier Guidance | Classifier-Free Guidance
```

### 3.1. Explicit conditioning (slide p33–34) — **đang dùng**

- Ghép điều kiện **c** vào mạng, train:

\[
\mathcal{L} = \mathbb{E}\,\|\varepsilon - \varepsilon_\theta(x_t, t, c)\|^2
\]

- Code: `PromptEncoder` → vector `c` → FiLM trong ResBlock.  
- Dataset: MIDI + labels (mood/genre/…) ≈ “image–text pairs” trên slide (LAION) nhưng **structured attributes** (nhẹ, đúng game prompt).

### 3.2. Classifier guidance (slide p36–42) — **không làm (có lý do slide)**

Slide nêu vấn đề:

- Cần classifier trên **noisy** `x_t`  
- Phải train/finetune classifier  
- Khó với free-form / nhiều attribute  

→ Đồ án **bỏ classifier guidance**, chuyển CFG (slide khuyến nghị cho text control).

### 3.3. Classifier-Free Guidance (slide p44–48) — **đã implement, then chốt**

Slide:

1. Train conditional **và** unconditional bằng **conditioning dropout**  
2. Khi sample:

\[
\hat\varepsilon = \varepsilon_\theta(x_t,t,\emptyset) + s\cdot\bigl(\varepsilon_\theta(x_t,t,c)-\varepsilon_\theta(x_t,t,\emptyset)\bigr)
\]

- Code: `cond_drop_prob=0.1`, `predict_noise(..., guidance_scale=s)`.  
- `s` (guidance scale): slide p38 — tăng s → bám condition mạnh hơn (trade-off đa dạng / artifact).

**Đây là cải tiến “chuẩn slide” quan trọng nhất để nhạc bám mood/instrument.**

---

## 4. Phần (3) Text-to-image trên slide → map Text-to-Music

| Slide (image) | Map music đồ án |
|---------------|-----------------|
| Pixel / latent image x0 | Piano-roll 2D (pitch × time) |
| Text encoder (CLIP, T5, …) | PromptEncoder attributes (+ optional BERT sau) |
| CFG scale | `guidance_scale` |
| Cascaded / super-res (Imagen) | *Chưa*: có thể “coarse roll → fine roll” sau |
| Latent Diffusion (p70) | *Chưa*: VAE nén roll rồi diffuse latent (Week5+8) |
| CLIP guidance | *Không ưu tiên* — slide đã có CFG thay thế |

Imagen (p71–75): **text encoder mạnh** giúp deep language understanding.  
→ Cải **đúng slide** tiếp theo: encode caption (MidiCaps) bằng text encoder (BERT-tiny đã có phác thảo ở Transformer project), vẫn CFG.

---

## 5. Checklist: slide → code hiện tại

| Kiến thức slide | Code | Trạng thái |
|-----------------|------|------------|
| Forward Markov noise | `q_sample` | ✅ |
| Reverse denoising U-Net | `ConditionalUNet` | ✅ |
| ε-prediction + simple MSE (Ho) | `p_losses` | ✅ |
| Noise schedule β | cosine / linear | ✅ |
| Time embedding | `SinusoidalPosEmb` | ✅ |
| Self-attention in U-Net | `SelfAttention2d` mid | ✅ |
| Explicit conditioning | PromptEncoder + FiLM | ✅ |
| CFG dropout + guided sampling | `cond_drop_prob`, `guidance_scale` | ✅ |
| Classifier guidance | — | ❌ cố ý (slide: phức tạp) |
| Latent diffusion | — | ⬜ optional |
| Cascaded diffusion | — | ⬜ optional |
| CLIP / strong text encoder | — | ⬜ optional (Part3) |
| Full ELBO từng KL term | — | ❌ dùng simple objective (slide p22) |

---

## 6. Cải tiến **nên làm tiếp** (chỉ những gì slide ủng hộ)

### Mức A — đúng slide, ROI cao cho đồ án

1. **Chỉnh guidance scale s** (slide p38, p48–49)  
   - Quét s ∈ {1, 2, 3.5, 5} trên cùng prompt, log CSV.  
   - Báo cáo: s↑ → instrument_match↑, diversity có thể↓.

2. **Nhiều step sample** (slide sampling)  
   - DDIM 50 → 80–100: reverse xấp xỉ tốt hơn (trade-off time).

3. **Explicit cond + caption** (slide Part3 text encoder)  
   - Train thêm với `caption` trong labels (MidiCaps).  
   - Vẫn ε-prediction + CFG.

4. **Train đủ epoch + val** (Week2 generalization)  
   - 30–50 epoch, theo dõi val MSE; tránh underfit (loss chưa hội tụ).

### Mức B — slide có, nặng hơn

5. **Latent Diffusion** (slide p70 + Week5 VAE)  
   - VAE nén piano-roll → diffuse latent → decode.  
   - Nói được: “kết hợp Week5 + Week8 như LDM”.

6. **Objective weighting theo t** (slide p22 advanced)  
   - Perception-prioritized / SNR weight.

7. **Cascaded** (Imagen style)  
   - Model 64-frame → upsample 256-frame.

### Mức C — **không** cần để “đúng môn”

- DiT thay U-Net, consistency models, flow matching… (ngoài slide chính).  
- Có thể nhắc “hướng phát triển” 1 dòng.

---

## 7. So sánh Transformer (Week4+6) vs Diffusion (Week8) — câu trả lời thầy

| | **Music Transformer** | **Diffusion piano-roll** |
|--|----------------------|---------------------------|
| Slide | Week4 + Week6 AR | Week8 DDPM + conditional |
| Phân rã | \(p(x)=\prod p(x_t\|x_{<t})\) (token) | \(p(x_0)\) qua chuỗi denoising |
| Condition | Cross-attn / prompt emb | Explicit cond + **CFG** |
| Loss | CE / NLL token | MSE noise (reweighted VLB) |
| Sample | Autoregressive token | Parallel roll + DDIM steps |
| Ưu (slide) | Likelihood tốt, sequence dài | Chất lượng sample, control đa dạng (CFG) |

Cùng dataset + cùng metric CSV = so **hai họ generative models** trong giáo trình.

---

## 8. Câu trả lời bảo vệ (30 giây)

> “Em implement **Denoising Diffusion** theo Week8: forward thêm Gaussian noise, reverse bằng **U-Net dự đoán noise ε_θ** (Ho et al.), objective MSE đơn giản như slide.  
> Điều kiện game prompt theo **explicit conditioning**; để bám prompt mạnh em dùng **Classifier-Free Guidance** (conditioning dropout + scale s) đúng Part (2).  
> Piano-roll là x0 dạng ‘ảnh’ 2D để map text-to-image pipeline trên slide sang text-to-music.  
> Song song em có **Music Transformer autoregressive** (Week4+6) để so sánh cùng data.”

---

## 9. Hyperparameters “nói được bằng slide”

| Tham số | Slide | Gợi ý đồ án |
|---------|-------|-------------|
| T (timesteps) | Forward T steps | 1000 |
| β schedule | p25 | cosine (hoặc linear baseline) |
| Objective | p21–22 ε-pred, weight≈1 | MSE |
| Architecture | p24 U-Net + attn | base_ch 64 + mid attn |
| Cond dropout | p44 CFG train | 0.1 |
| Guidance s | p38, p47–49 | 2–5 (default 3.5) |
| Sample steps | reverse / DDIM | 50–100 |

File config: `config/config.yaml` — các key trên khớp bảng này.
