# Text-to-Music: Transformer vs Diffusion

This is a **Generative AI** academic project. Its goal is to generate video game background music based on a **structured prompt** containing 6 attributes. The project compares two different AI approaches trained on the same dataset.

## Core Concepts

- **Input**: A prompt with 6 specific attributes: `mood`, `genre`, `scene`, `tempo`, `instrument`, and `energy`.
- **Output**: A generated **MIDI** file (`.mid`).

## Approaches Compared

1. **Music Transformer** (`GenAI_Transformer/` directory):
   - Uses an autoregressive approach with REMI tokens to generate music step-by-step.
2. **Piano-roll Diffusion** (`GenAI_Diffusion/` directory):
   - Uses Denoising Diffusion Probabilistic Models (DDPM/DDIM) combined with Classifier-Free Guidance (CFG) to generate music from noise.

## Project Structure

- `data/`: The shared dataset containing approximately 14,000 MIDI files (merged from ComMU, MidiCaps, MAESTRO, etc.).
- `GenAI_Transformer/`: Source code for the Music Transformer model.
- `GenAI_Diffusion/`: Source code for the Piano-roll Diffusion model.
- `compare/`: Scripts and metrics used to evaluate and compare the two models (outputs results to CSV files).
- `docs/`: Comprehensive documentation including training guides, data structures, and research reports.
- `scripts/`: Utility scripts for data processing.

## Quick Start

### Requirements
- Python 3.10+
- NVIDIA GPU (Recommended)
- Packages: `torch`, `pretty_midi`

### Training
```powershell
# Train Transformer model
cd GenAI_Transformer
python train.py --epochs 10 --batch_size 8 --max_seq_len 1024 --no_early_stop

# Train Diffusion model
cd GenAI_Diffusion
python train.py --epochs 10 --batch_size 4
```

### Generating Music
```powershell
# Generate with Transformer (~30 seconds)
cd GenAI_Transformer
python generate.py --mood happy --genre fantasy --scene village --tempo fast --instrument piano --duration_sec 30

# Generate with Diffusion (~10 seconds)
cd GenAI_Diffusion
python generate.py --checkpoint checkpoints/best_model.pt --epoch 10 --duration_sec 12 --evaluate
```

### Running Comparisons
```powershell
# Compare results across epochs 1, 5, and 10
python -m compare.run_comparison_eval --epochs 1 5 10
python -m compare.compare_results
```
*Results will be saved as a CSV file in the `compare/results/` folder.*

### API (Optional)
The project includes a FastAPI setup for the Transformer model.
```powershell
cd GenAI_Transformer
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*Access the API documentation at `http://localhost:8000/docs`.*

---
*For more detailed instructions on training, hardware requirements, and dataset preparation, please refer to the markdown files inside the `docs/` folder.*
