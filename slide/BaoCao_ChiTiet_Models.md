# Báo cáo chi tiết: Kiến trúc và Phương pháp tối ưu của 2 Mô hình Sinh Âm Nhạc (GenAI)

Dựa trên mã nguồn của dự án, dưới đây là phân tích chi tiết từng lớp và các kỹ thuật tối ưu hóa được sử dụng cho hai mô hình: **GenAI_Diffusion** và **GenAI_Transformer**.

---

## 1. Mô hình GenAI_Diffusion (Mô hình khuếch tán dựa trên U-Net)
Mô hình này sinh ra các tệp âm nhạc dưới dạng biểu diễn piano-roll (ảnh 2D) thông qua quá trình khử nhiễu (denoising).

### 1.1. Kiến trúc các lớp (Layer Architecture)
Cốt lõi là mạng **Conditional 2D U-Net** nhận input đầu vào có dạng `(Batch, 1, n_pitches, n_frames)`.

*   **Lớp đầu vào (Input Layer):**
    *   Sử dụng một lớp tích chập `in_conv` (Conv2d: kernel 3x3, padding 1) để mở rộng 1 kênh âm thanh đầu vào thành `base_channels` (mặc định 64 kênh).
*   **Mạng nhúng điều kiện (Conditioning & Prompt Encoder):**
    *   **Time Embedding:** Sử dụng hàm `SinusoidalPosEmb` để mã hóa bước thời gian khuếch tán, sau đó đi qua mạng MLP (Linear -> SiLU -> Linear) để tạo ra vector nhúng thời gian.
    *   **Prompt Embedding:** Các tham số định dạng nhạc (mood, genre, scene, tempo, instrument, energy) được mã hóa qua lớp `PromptEncoder` độc lập.
    *   Sự kết hợp của Time và Prompt được dùng làm điều kiện cho mạng qua cơ chế **FiLM (Feature-wise Linear Modulation)**, thay đổi quy mô (scale) và độ dời (shift) của các feature map.
*   **Phần mã hóa (Encoder / Downsampling):**
    *   Bao gồm nhiều khối (block) được lặp lại dựa trên mảng `channel_mults` (ví dụ x1, x2, x4 số kênh).
    *   Mỗi khối chứa 2 lớp `ResBlock`. Một `ResBlock` bao gồm:
        *   `GroupNorm` -> Hàm kích hoạt `SiLU` -> `Conv2d` (3x3).
        *   Cộng FiLM (điều chỉnh tỷ lệ và dịch chuyển bằng thông tin Time/Cond).
        *   `Dropout` (giảm overfitting) -> `SiLU` -> `Conv2d` (3x3) -> Cộng **Skip Connection** (kết nối tắt).
    *   Lớp `Downsample`: Nằm ở cuối mỗi khối (trừ khối cuối cùng), dùng `Conv2d` (kernel 4x4, stride 2) để giảm đi một nửa không gian 2D, trích xuất đặc trưng bậc cao.
*   **Phần trung tâm (Bottleneck / Mid):**
    *   Chứa 1 lớp `ResBlock` đầu tiên -> Lớp `SelfAttention2d` -> 1 lớp `ResBlock` thứ hai.
    *   `SelfAttention2d`: Là lớp Attention nhẹ áp dụng trên không gian đã được làm phẳng. Rất quan trọng để mô hình nắm bắt được sự tương quan nhịp điệu (rhythm) tầm xa, sử dụng 4 attention heads.
*   **Phần giải mã (Decoder / Upsampling):**
    *   Mở rộng lại kích thước không gian. Nhận đầu vào là các feature map và ghép nối (concatenate) với các đặc trưng được lưu từ phần Encoder (để giữ chi tiết ảnh).
    *   Mỗi khối cũng gồm 2 lớp `ResBlock`.
    *   Lớp `Upsample`: Phóng to kích thước x2 bằng nội suy `nearest` (nearest neighbor), sau đó dùng `Conv2d` (3x3) để tinh chỉnh.
*   **Lớp đầu ra (Output Layer):**
    *   Gồm `GroupNorm` -> `SiLU` -> `Conv2d` để đưa số kênh trở về 1 kênh duy nhất.
    *   **Khởi tạo bằng 0 (Zero-init):** Trọng số của lớp cuối cùng được khởi tạo là 0 nhằm giúp mô hình sinh nhiễu ổn định ở những bước học đầu tiên.

### 1.2. Các phương pháp tối ưu hóa (Optimization Methods)
*   **Thuật toán tối ưu:** `AdamW` (`torch.optim.AdamW`) với learning rate `2e-4` và weight decay `0.01`.
*   **Hàm mất mát:** `MSE Loss` (sai số toàn phương trung bình) đo lường sự chênh lệch giữa nhiễu thực tế được thêm vào và nhiễu do mô hình dự đoán.
*   **Gradient Accumulation & Clipping:** Tích lũy Gradient qua nhiều bước để giả lập Batch Size lớn. Kết hợp Gradient Clipping (`max_grad_norm=1.0`) để ngăn chặn bùng nổ gradient.
*   **Exponential Moving Average (EMA):** Một bản sao của trọng số mô hình được cập nhật dần dần bằng EMA với hệ số suy giảm `0.999`. Bản sao EMA này được dùng khi đánh giá và sinh nhạc giúp tăng cường độ mượt và chất lượng âm thanh đáng kể.
*   **Mixed Precision Training (AMP):** Có hỗ trợ `torch.cuda.amp.GradScaler`, nhưng mặc định thiết lập chạy ở `fp32` và tắt một số thuật toán của cuDNN (`allow_tf32=False`) để giải quyết tình trạng lỗi tràn bộ nhớ / xung đột kernel trên một số hệ thống GPU.

---

## 2. Mô hình GenAI_Transformer (Mô hình ngôn ngữ dạng Decoder-only)
Khác với Diffusion, mô hình này dự đoán chuỗi các token MIDI theo nguyên lý tự hồi quy (giống GPT).

### 2.1. Kiến trúc các lớp (Layer Architecture)
*   **Lớp nhúng (Embeddings & Prompting):**
    *   `TokenEmbedding`: Biến đổi mỗi token MIDI thành một vector dense có kích thước `d_model` (mặc định 256).
    *   `PromptEncoder`: Nén các thông tin chỉ định (mood, genre, instrument...) thành một vector duy nhất `cond` có cùng chiều `d_model` làm ngữ cảnh sinh. Hoàn toàn không sử dụng BERT hay NLP tự do.
*   **Các Khối Giải Mã (Decoder Blocks):**
    Bao gồm nhiều lớp (mặc định 6 lớp), xếp chồng lên nhau. Mỗi lớp có:
    *   **Self-Attention:** Chú ý các token trước đó trong chuỗi để đoán token tiếp theo. Điểm đặc biệt:
        *   Sử dụng Causal Mask để đảm bảo token hiện tại không "nhìn thấy" tương lai.
        *   Dùng **Group Query Attention (GQA)** thông qua tham số `num_kv_heads` giúp tăng tốc tính toán và tiết kiệm VRAM so với Multi-Head Attention thông thường.
        *   Sử dụng cơ chế chuẩn hóa QK (`use_qk_norm`) để ổn định điểm số attention.
    *   **Cross-Attention:** Áp dụng Attention trực tiếp lên vector `cond` (Prompt) giúp mô hình liên tục bám sát yêu cầu từ người dùng trong suốt quá trình sinh chuỗi.
    *   **Feed-Forward Network (FFN):** Mạng MLP mở rộng kích thước lên gấp 4 lần (`d_ff` = d_model * 4) rồi thu nhỏ lại, giúp mô hình học các biểu diễn phi tuyến tính phức tạp.
    *   **Chuẩn hóa:** Khác với Transformer truyền thống dùng LayerNorm, mô hình này áp dụng **RMSNorm** (Root Mean Square Normalization) tính toán nhẹ hơn và hội tụ nhanh hơn.
*   **Lớp đầu ra (Output Projection):**
    *   Sau khối cuối cùng, dữ liệu đi qua `RMSNorm` tổng quát.
    *   Một lớp Tuyến tính (`Linear`) dự đoán xác suất cho toàn bộ bộ từ vựng (`vocab_size`).
    *   **Weight Tying:** Trọng số của lớp Output này được trỏ trực tiếp đến trọng số của `TokenEmbedding` lúc đầu. Cải tiến này giúp tiết kiệm lượng lớn tham số và cải thiện tốc độ hội tụ của mô hình ngôn ngữ.

### 2.2. Các phương pháp tối ưu hóa (Optimization Methods)
*   **Thuật toán tối ưu:** `AdamW` (`lr=1e-4`, `weight_decay=0.01`), với các hệ số đặc thù cho Transformer như `betas=(0.9, 0.98)` và `eps=1e-9`.
*   **Scheduler Warmup + Cosine Decay:** Sử dụng bộ lập lịch học suất `WarmupCosineScheduler`. Học suất sẽ tăng tuyến tính trong 5% số bước đầu (warm-up), sau đó giảm dần theo hình sin (cosine) về `1e-6`. Điều này giúp Transformer không bị sụp đổ trong các epoch đầu và tinh chỉnh kỹ ở cuối.
*   **Label Smoothing:** Khi tính toán hàm `CrossEntropyLoss`, áp dụng độ làm mịn nhãn (`label_smoothing=0.1`). Tránh việc mô hình quá tự tin vào một token duy nhất, giảm hiện tượng overfitting.
*   **Cơ chế Snapshot và Rollback an toàn (Safe Training):** 
    Đây là kỹ thuật xuất sắc trong file `trainer.py`: Ở cuối mỗi bước tối ưu, Trainer tự lưu một bản sao an toàn (`_snapshot_good_weights`). Nếu phát hiện Loss nhảy lên vô cực (Inf/NaN) hoặc hàm Gradients bị vỡ, nó sẽ tự động hủy bước đó và **Rollback** trọng số về bản sao an toàn trước đó. Rất hữu ích trên các cụm GPU chia sẻ dễ bị sập đột ngột.
*   **Early Stopping:** Dừng học sớm nếu Validation Loss không được cải thiện sau `10` epochs, tránh lãng phí thời gian và giảm overfitting.
*   **Kiểm soát Math SDPA:** Vô hiệu hóa `flash_sdp` và `mem_efficient_sdp` tích hợp sẵn của PyTorch, chỉ giữ lại `math_sdp` để tránh các lỗi truy cập bộ nhớ bất hợp pháp trên các GPU phổ thông như RTX 3060.
