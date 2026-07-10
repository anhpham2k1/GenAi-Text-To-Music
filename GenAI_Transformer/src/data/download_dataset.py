import os
import sys
import urllib.request
import zipfile

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Ensure the root project directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.preprocessing import filter_midi_files, generate_labels, print_dataset_info

def download_and_extract(url, extract_to, name="dataset"):
    print(f"[Download] Downloading {name} from {url}...")
    zip_path = f"temp_{name.replace(' ', '_')}.zip"
    
    def report(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading... {percent}%")
            sys.stdout.flush()
        
    urllib.request.urlretrieve(url, zip_path, reporthook=report)
    print("\n[Download] Download complete. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    os.remove(zip_path)
    print("[Download] Extraction complete.")

def download_from_hf(repo_id, extract_to, name="dataset", allow_patterns=None):
    """Download MIDI dataset from Hugging Face."""
    if not HF_AVAILABLE:
        print(f"[HF] huggingface_hub not installed. Run: pip install huggingface_hub")
        print(f"   Then manually download from https://huggingface.co/datasets/{repo_id}")
        return False
    print(f"[HF] Downloading {name} from {repo_id} ...")
    os.makedirs(extract_to, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=extract_to,
            allow_patterns=allow_patterns or ["*.mid", "*.midi", "**/*.mid", "**/*.midi"],
            local_dir_use_symlinks=False
        )
        print(f"[HF] Downloaded to {path}")
        return True
    except Exception as e:
        print(f"[HF] Error: {e}")
        print("   Try manual download or install 'datasets' and 'huggingface_hub'")
        return False

if __name__ == "__main__":
    print_dataset_info()
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    labels_file = "data/labels/labels.json"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    
    print("\n=== UPDATE DATASET: Recommended Combo ===")
    print("VGMusic (game) + MidiCaps (captions) + ComMU (structured labels) + Tegridy/GigaMIDI (volume)")
    print("1. MAESTRO (current piano)")
    print("2. VGMusic (game-specific - manual or Kaggle scrape)")
    print("3. MidiCaps (text captions - HF)")
    print("4. ComMU (structured labels - GitHub)")
    print("5. Tegridy MIDI (curated multi-instr - GitHub)")
    print("6. GigaMIDI (massive - HF)")
    print("7. All recommended (manual steps)")
    
    choice = input("\nChoose (1-7, or 'a' for all info): ").strip().lower()
    
    if choice == "1":
        maestro_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
        download_and_extract(maestro_url, raw_dir, "MAESTRO")
    elif choice == "3":
        print("[MidiCaps] Use HF: huggingface-cli download amaai-lab/MidiCaps --repo-type dataset --local-dir data/raw/midicaps")
        if HF_AVAILABLE:
            download_from_hf("amaai-lab/MidiCaps", os.path.join(raw_dir, "midicaps"), "MidiCaps")
    elif choice == "4":
        print("[ComMU] Clone or download from: https://github.com/POZAlabs/ComMU-code")
        print("   Dataset MIDI usually in the repo or linked in paper.")
    elif choice == "5":
        print("[Tegridy] Download zips from: https://github.com/asigalov61/Tegridy-MIDI-Dataset/releases or raw files")
    elif choice == "6":
        print("[GigaMIDI] Use HF (large, may need login):")
        if HF_AVAILABLE:
            download_from_hf("Metacreation/GigaMIDI", os.path.join(raw_dir, "gigamidi"), "GigaMIDI")
    else:
        print("[INFO] For VGMusic: Visit https://www.vgmusic.com/ and download game folders.")
        print("       Or Kaggle: https://www.kaggle.com/datasets/hansespinosa2/40000-video-game-midi-files")
        print("       Place in data/raw/vgmusic/")
        print("       For others use HF/GitHub as above. Then run filter + labels.")
    
    print("\n[Preprocessing] Filtering MIDI in data/raw/ (including subfolders)...")
    filter_midi_files(raw_dir, processed_dir, verbose=True)
    
    print("\n[Labels] Generating labels (improved for diverse sources)...")
    generate_labels(processed_dir, labels_file)
    
    print("\n✅ Dataset combo ready! Use the recommended mix for better game music variety.")
    print("Tip: After adding VGMusic/MidiCaps etc., re-run filter and labels.")
