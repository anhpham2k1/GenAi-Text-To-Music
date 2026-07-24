# Dataset dùng chung (Transformer + Diffusion)

`data/` là kho data **ngoài** hai project. **Không còn commit vào git**
(`data/raw/`, `data/processed/`, `data/labels/*.json`, `compare/split.json`
đều gitignore) — build lại bằng 1 lệnh:

```bash
python scripts/download_dataset.py
```

Script tự tải MAESTRO + ComMU (nguồn public, không cần tài khoản), lọc/gán
nhãn, rồi tự chạy `compare.make_split` + `scripts/build_captions.py`. Chạy
lại an toàn (idempotent) — không tải/copy trùng nếu đã có sẵn. Xem
`python scripts/download_dataset.py --help` cho các tuỳ chọn (`--sources`,
`--force-remerge`).

```
data/
├── raw/              # MIDI gốc theo nguồn (gitignore)
├── processed/        # MIDI đã lọc, dùng để train (gitignore)
├── labels/           # labels.json + captions.json (gitignore)
└── README.md         # file này (có commit)
```

## raw/ — trạng thái

| Thư mục | Vai trò | Ghi chú |
|---------|---------|---------|
| `maestro-v3.0.0/` | Piano chất lượng cao | Dataset **cũ** |
| `midicaps/` | MIDI + caption (lớn) | Dataset **cũ** |
| `midicaps_sample/` | Sample midicaps | Dataset **cũ** |
| `commu/` | ComMU structured (genre/mood/inst…) | Dataset **mới** (~11k MIDI + `commu_meta.csv`) |
| `vgmidi/` | Clone repo VGMIDI | Dataset **mới** — cần tải thêm file MIDI nếu repo trống |
| `vgmusic/` | Game MIDI | Chưa có file — bỏ zip vào đây |
| `tegridy/`, `gigamidi/`, `lakh/` | Tuỳ chọn | Slot sẵn |

## Train đang dùng

Sau `python scripts/download_dataset.py` (MAESTRO + ComMU, tự động):

- `processed/` + `labels/labels.json`: **~14641** file
  - MAESTRO (lọc chất lượng qua `filter_midi_files`): **~409**
  - ComMU (structured map từ `commu_meta.csv`): **11144**
  - Legacy đã merge trước đó (MidiCaps một phần…): phần còn lại
- Split: `compare/split.json` — tự chia train/val (~90/10), seed cố định nên
  reproducible
- `data/.dataset_build_state.json` — ghi nhớ nguồn nào đã merge, để chạy lại
  script không bị copy trùng (dùng `--force-remerge` nếu cố tình muốn làm lại)

## Thêm nguồn mới / merge lại thủ công

`scripts/download_dataset.py` gọi đúng các hàm dưới đây — chỉ cần tự chạy
tay khi muốn kiểm soát chi tiết hơn:

```bash
cd GenAI_Transformer
python -c "
from src.data.preprocessing import filter_midi_files, generate_labels
filter_midi_files('../data/raw/<nguồn>', '../data/processed', verbose=True)
generate_labels('../data/processed', '../data/labels/labels.json')
"

cd ..
python -m compare.make_split --midi_dir data/processed --split_path compare/split.json
python scripts/build_captions.py
```

Merge có metadata (giống ComMU — genre/inst/bpm → mood/genre/scene/tempo):

```bash
cd GenAI_Transformer
python -c "
from src.data.preprocessing import merge_and_process_datasets
merge_and_process_datasets(
    ['../data/raw/midicaps_sample', '../data/raw/vgmusic', '../data/raw/vgmidi'],
    processed_dir='../data/processed',
    labels_file='../data/labels/labels.json',
)
"
```

## VGMIDI — tải file nhạc

Repo `raw/vgmidi` thường chỉ là code. File MIDI/emotion thường ở:

- https://github.com/lucasnfe/vgmidi  
- Releases / Google Drive trong README repo  

Giải nén `.mid` vào `data/raw/vgmidi/` (có thể để subfolder).

## VGMusic

Tải từ https://www.vgmusic.com/ hoặc Kaggle video-game-midi → bỏ vào `data/raw/vgmusic/`.
