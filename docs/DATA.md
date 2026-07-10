# Dataset dùng chung (Transformer + Diffusion)

Thư mục **`D:\Master\Ky3\data`** là kho data **ngoài** hai project.

```
Ky3/
├── data/                 ← BẠN ĐANG Ở ĐÂY (source of truth)
│   ├── raw/              # MIDI gốc theo nguồn
│   ├── processed/        # MIDI đã lọc (train)
│   ├── labels/           # labels.json (6 attribute)
│   └── README.md
├── GenAI_Transformer/data  → junction trỏ về ../data
├── GenAI_Diffusion/      # config trỏ ../data
└── ComMU-code/           # code repo ComMU (meta đã copy vào raw/commu)
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

## Train đang dùng (cập nhật sau merge ComMU)

- `processed/` + `labels/labels.json`: **~14232** file  
  - Legacy (MAESTRO/MidiCaps sample…): **3088**  
  - ComMU (structured map từ `commu_meta.csv`): **11144**  
- Split: `compare/split.json` — train **12809** / val **1423**  
- Script merge lại: `python scripts/merge_commu.py`

## Thêm MIDI mới rồi cập nhật train set

```powershell
cd D:\Master\Ky3\GenAI_Transformer

python -c "
from src.data.preprocessing import filter_midi_files, generate_labels
filter_midi_files('../data/raw', '../data/processed', verbose=True)
generate_labels('../data/processed', '../data/labels/labels.json')
"

cd D:\Master\Ky3
python -m compare.make_split --midi_dir data/processed --split_path compare/split.json
```

Hoặc merge có metadata ComMU:

```powershell
cd D:\Master\Ky3\GenAI_Transformer
python -c "
from src.data.preprocessing import merge_and_process_datasets
merge_and_process_datasets(
    [
        '../data/raw/maestro-v3.0.0',
        '../data/raw/midicaps_sample',  # hoặc midicaps (rất lớn)
        '../data/raw/commu',
        '../data/raw/vgmusic',
        '../data/raw/vgmidi',
    ],
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
