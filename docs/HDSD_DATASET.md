# Hướng Dẫn Sử Dụng (HDSD) - Tải Dataset và Cập Nhật Label Từ Đầu

Hướng dẫn **chi tiết từng bước** để tải dataset MIDI, chuẩn bị dữ liệu sạch và **cập nhật labels.json từ đầu** (re-generate toàn bộ nhãn) cho dự án Text-to-Music (Music Transformer cho game BGM).

Mục tiêu: Có bộ dữ liệu đa dạng (game music) + labels (mood, genre, scene, tempo, instrument, energy + caption nếu có) đầy đủ và mới nhất.

## Thực thi gần đây (2026-07-07)
Đã thực hiện theo yêu cầu **"xóa tất cả + lấy hết 7 dataset + tạo label từ đầu"**:

- Xóa sạch `data/processed/`, `data/labels/`
- Dọn raw subdirs
- Tạo đầy đủ **7 dataset source dirs**
- Tải lại MAESTRO (1276 files trong raw)
- Tạo demo MIDI + captions.json + metadata.json
- Chạy `auto_label` thật từ đầu (550s) → `labels.json` 82 entries
- Demos có caption + structured đúng:
  - demo1: tense/fantasy/dungeon/full_orchestra/high + caption
  - demo2: happy/sci-fi/village/synth + caption
- Backup full MAESTRO tại `data/processed_full/`

**Kết quả labels hiện tại:**
- 82 entries (demos + mẫu MAESTRO)
- 2 captions
- Genre: fighting 80 (từ auto_label piano năng lượng cao), sci-fi + fantasy từ demo

Để scale full: copy từ `processed_full` vào `processed` rồi chạy lại generate_labels hoặc merge.

Xem chi tiết lệnh ở phần 5 bên dưới.

---

## 1. Chuẩn bị môi trường (Windows PowerShell)

```powershell
# Di chuyển vào thư mục dự án
cd D:\Master\Ky3\GenAI_Transformer

# Cài dependencies (lần đầu)
pip install -r requirements.txt

# (Khuyến nghị) Cài thêm cho download HF
pip install huggingface_hub
```

Kiểm tra:
```powershell
python --version
python -c "import pretty_midi, torch; print('OK')"
```

---

## 2. Hiểu cấu trúc dữ liệu

```
data/
├── raw/                  # Nguồn MIDI gốc (có thể chia subfolder)
│   ├── maestro-v3.0.0/
│   ├── midicaps/         # Có captions.json
│   ├── commu/            # Có metadata.json (structured labels)
│   ├── vgmusic/          # Khuyến nghị thêm
│   └── ...
├── processed/            # MIDI đã lọc (copy từ raw)
└── labels/
    └── labels.json       # Nhãn tự động + merge metadata/caption
```

**Lưu ý quan trọng:**
- `labels.json` hiện tại chỉ có ~411 entries (chủ yếu MAESTRO piano + fantasy bias).
- `data/processed/` có thể có ~3000+ file MIDI.
- Khi thêm MIDI mới vào `raw/` → **phải chạy lại filter + generate_labels** để cập nhật.

---

## 3. Reset sạch từ đầu (Khuyến nghị trước khi chạy lại)

```powershell
# 1. Backup labels cũ (nếu muốn giữ)
if (Test-Path data/labels/labels.json) {
    Copy-Item data/labels/labels.json data/labels/labels.json.bak -Force
}

# 2. Xóa processed (để filter lại từ raw) - TÙY CHỌN
# CẢNH BÁO: Thao tác này xóa tất cả MIDI đã xử lý
Remove-Item -Recurse -Force data/processed/* -ErrorAction SilentlyContinue

# 3. Xóa labels.json để tạo mới hoàn toàn
if (Test-Path data/labels/labels.json) {
    Remove-Item data/labels/labels.json -Force
}

# Tạo lại thư mục
New-Item -ItemType Directory -Force -Path data/raw, data/processed, data/labels | Out-Null
```

---

## 4. Tải / Thêm Dataset (Combo khuyến nghị)

Chạy script có menu thông tin:

```powershell
python -m src.data.download_dataset
```

Script sẽ:
- In bảng dataset khuyến nghị
- Cho phép tải MAESTRO (dễ)
- Hướng dẫn các nguồn khác

### 4.1. Các nguồn quan trọng (thêm thủ công)

| Nguồn              | Mục đích                        | Cách tải                                                                 | Thư mục gợi ý          |
|--------------------|---------------------------------|--------------------------------------------------------------------------|-------------------------|
| **MAESTRO**        | Piano chất lượng cao (đã có)    | Đã có trong `data/raw/maestro-v3.0.0`                                    | -                       |
| **VGMusic**        | Game music thuần túy (RẤT QUAN TRỌNG) | https://www.vgmusic.com/ (tải zip theo game) hoặc Kaggle 40000-video-game-midi | `data/raw/vgmusic/`     |
| **MidiCaps**       | 168k MIDI + **text caption** sẵn | `huggingface-cli download amaai-lab/MidiCaps --repo-type dataset --local-dir data/raw/midicaps` | `data/raw/midicaps/` (có captions.json) |
| **ComMU**          | Structured labels (genre/mood/instr) | https://github.com/POZAlabs/ComMU-code (lấy MIDI + metadata)            | `data/raw/commu/` (có metadata.json mẫu) |
| **Tegridy MIDI**   | Đa dạng, curated cho AI         | https://github.com/asigalov61/Tegridy-MIDI-Dataset/releases              | `data/raw/tegridy/`     |
| **GigaMIDI**       | Volume cực lớn (~1M+)           | HF: `huggingface-cli download Metacreation/GigaMIDI ...` (lớn, cần lọc) | `data/raw/gigamidi/`    |
| Lakh / MetaMIDI    | Đa dạng thể loại                | Tìm trên Kaggle hoặc official                                            | `data/raw/lakh/`        |

**Ví dụ tải MidiCaps (PowerShell):**

```powershell
# Cần cài huggingface_hub trước
huggingface-cli download amaai-lab/MidiCaps --repo-type dataset --local-dir data/raw/midicaps --local-dir-use-symlinks False
```

**Ví dụ tải GigaMIDI (có thể rất lớn, chỉ tải một phần):**

```powershell
huggingface-cli download Metacreation/GigaMIDI --repo-type dataset --local-dir data/raw/gigamidi --allow-patterns "*.mid" --local-dir-use-symlinks False
```

Sau khi tải, đặt file `.mid`/`.midi` vào các subfolder tương ứng trong `data/raw/`.

---

## 5. Chạy Lọc + Cập Nhật Labels Từ Đầu

Có 2 cách chính:

### Cách A: Đơn giản (toàn bộ raw → processed + labels mới)

```powershell
python -c "
from src.data.preprocessing import filter_midi_files, generate_labels
print('=== FILTER ===')
filter_midi_files('data/raw', 'data/processed', min_duration=5.0, max_duration=300.0, min_notes=10, verbose=True)
print('=== GENERATE LABELS (from scratch) ===')
generate_labels('data/processed', 'data/labels/labels.json')
print('Done!')
"
```

### Cách B: Khuyến nghị - Dùng Merge (hỗ trợ captions + structured labels)

Hỗ trợ tự động load `captions.json` / `metadata.json` từ từng nguồn:

```powershell
python -c "
from src.data.preprocessing import merge_and_process_datasets

sources = [
    'data/raw/maestro-v3.0.0',
    'data/raw/midicaps',   # sẽ load captions.json
    'data/raw/commu',      # sẽ load metadata.json
    # 'data/raw/vgmusic',
    # 'data/raw/tegridy',
]

merge_and_process_datasets(
    sources,
    processed_dir='data/processed',
    labels_file='data/labels/labels.json'
)
print('Merge + labels complete!')
"
```

**Lưu ý PowerShell quoting:**
- Nếu lệnh dài, copy nguyên đoạn trên (bao gồm dấu ngoặc kép).
- Hoặc tạo file tạm `update_dataset.py` rồi chạy `python update_dataset.py`.

Ví dụ tạo script tạm (nên dùng cho lệnh phức tạp):

```powershell
@"
from src.data.preprocessing import filter_midi_files, generate_labels
filter_midi_files('data/raw', 'data/processed', verbose=True)
generate_labels('data/processed', 'data/labels/labels.json')
"@ > update_labels.py

python update_labels.py
```

---

## 6. Kiểm tra kết quả sau khi chạy

```powershell
python -c "
import json, os, glob
from collections import Counter

labels_path = 'data/labels/labels.json'
labels = json.load(open(labels_path, encoding='utf-8'))

print('=== THỐNG KÊ LABELS ===')
print('Tổng số file có label:', len(labels))

# Đếm phân bố
genres = Counter(v.get('genre','unknown') for v in labels.values())
moods = Counter(v.get('mood','unknown') for v in labels.values())
insts = Counter(v.get('instrument','unknown') for v in labels.values())

print('\nGenre distribution (top 8):')
for g, c in genres.most_common(8):
    print(f'  {g}: {c}')

print('\nMood distribution (top):')
for m, c in moods.most_common(5):
    print(f'  {m}: {c}')

print('\nInstrument distribution (top):')
for i, c in insts.most_common(5):
    print(f'  {i}: {c}')

# Kiểm tra caption
has_caption = sum(1 for v in labels.values() if 'caption' in v)
print(f'\nSố file có caption (từ MidiCaps...): {has_caption}')

# Kiểm tra processed vs labels
proc = [f for f in os.listdir('data/processed') if f.lower().endswith(('.mid','.midi'))]
print(f'Số MIDI trong processed: {len(proc)}')
print(f'Coverage: {len(labels)} / {len(proc)}')
"
```

**Kết quả mong muốn sau update từ đầu:**
- Số lượng labels ≈ số file trong `processed/`
- Phân bố genre đa dạng hơn (nếu đã thêm VGMusic / ComMU / MidiCaps)
- Một số entry có trường `"caption": "..."`

Nếu vẫn còn bias nặng (fantasy/piano), bạn cần:
1. Thêm nhiều file từ VGMusic / game MIDI vào `data/raw/vgmusic/`
2. Chạy lại bước 5.

---

## 7. Chạy Training với dataset mới

```powershell
# Train từ đầu (dùng labels mới)
python train.py --data_dir data/processed --epochs 30 --batch_size 16

# Hoặc với custom
python train.py --max_files 2000 --lr 0.0001
```

Kiểm tra labels được load đúng trong quá trình train (xem log).

---

## 8. Cập nhật lại labels khi thêm MIDI mới (không cần reset toàn bộ)

```powershell
# Chỉ cần chạy lại filter (nếu thêm raw) + generate_labels
python -c "
from src.data.preprocessing import filter_midi_files, generate_labels
filter_midi_files('data/raw', 'data/processed', verbose=True)
generate_labels('data/processed', 'data/labels/labels.json')
"
```

Hoặc chỉ generate nếu chỉ muốn cập nhật labels (không filter lại):

```powershell
python -c "
from src.data.preprocessing import generate_labels
generate_labels('data/processed', 'data/labels/labels.json')
"
```

---

## 9. Lưu ý quan trọng (Windows + Dự án)

1. **File name trùng**: Script tự xử lý bằng cách thêm `_N` suffix khi copy.
2. **Thời gian**: Generate labels duyệt toàn bộ MIDI (pretty_midi) → có thể mất vài phút với 3000+ files.
3. **Memory**: Nếu quá nhiều file, dùng `--max_files` khi train trước.
4. **Clean background tasks** (nếu trước đó có job treo):
   ```powershell
   Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
   ```
5. **Dùng subfolder** trong `data/raw/` để tổ chức nguồn (rất khuyến khích).
6. **Caption hỗ trợ NLP conditioning**: Khi có `caption` trong labels.json, hệ thống sẽ lưu nhưng prompt_text hiện tại chủ yếu dùng structured labels. Có thể mở rộng sau.
7. **Backup trước khi xóa lớn**.

---

## 10. Quy trình đầy đủ nhanh (Copy-Paste)

```powershell
cd D:\Master\Ky3\GenAI_Transformer

# Reset (tùy chọn)
Remove-Item -Recurse -Force data/processed -ErrorAction SilentlyContinue
Remove-Item data/labels/labels.json -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force data/processed, data/labels | Out-Null

# Thêm MIDI vào data/raw/... trước

# Chạy pipeline
python -c "
from src.data.preprocessing import filter_midi_files, generate_labels
filter_midi_files('data/raw', 'data/processed', verbose=True)
generate_labels('data/processed', 'data/labels/labels.json')
"

# Kiểm tra
python -c "
import json
d = json.load(open('data/labels/labels.json', encoding='utf-8'))
print('Labels now:', len(d))
print('Sample:', list(d.items())[0])
"
```

---

Viết ngày: 2026-07-07

Nếu cần chạy lại hoàn toàn hoặc thêm nguồn dataset cụ thể, hãy cho tôi biết để hỗ trợ chi tiết hơn (ví dụ: script tải tự động VGMusic). Sau khi labels đa dạng, train lại mô hình sẽ cho kết quả tốt hơn rất nhiều!