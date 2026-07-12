"""
FastAPI backend for Text-to-Music Generator.

Endpoints:
    POST /generate     — Generate music from text prompt
    GET  /download/{id}/{format} — Download generated MIDI/WAV
    GET  /health       — Health check

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
import yaml

import torch

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import MusicRequest, MusicResponse, HealthResponse
from src.data.tokenizer import MidiTokenizer
from src.model.transformer import MusicTransformer
from src.inference.generator import MusicGenerator
from src.inference.model_loader import load_model
from src.inference.renderer import MidiRenderer


# Globals
generator = None
renderer = None
model_device = "cpu"
model_info = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan (replaces deprecated @app.on_event("startup"))."""
    global generator, renderer, model_device, model_info

    # Web config is separate from the training config on purpose: config.yaml
    # changes between training runs, and the web app must not break when it does.
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_cfg_path = os.environ.get(
        "WEB_CONFIG", os.path.join(base, "api", "config.web.yaml")
    )
    config = {}
    if os.path.exists(web_cfg_path):
        with open(web_cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        print(f"[API] Web config: {web_cfg_path}")

    # Tokenizer vocab must match how the checkpoint was trained; it is not a
    # web-tunable setting, so it still comes from the training config.
    train_cfg_path = os.path.join(base, "config", "config.yaml")
    train_cfg = {}
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f) or {}

    model_cfg = train_cfg.get("model", {})
    tok_cfg = train_cfg.get("tokenizer", {})
    audio_cfg = config.get("audio", {})

    # Tokenizer
    tokenizer = MidiTokenizer(
        pitch_range=tuple(tok_cfg.get("pitch_range", [21, 108])),
        velocity_bins=tok_cfg.get("velocity_bins", 32),
        time_shift_bins=tok_cfg.get("time_shift_bins", 100),
    )

    # Model — architecture comes from the checkpoint itself, so an edited
    # config.yaml can never desync from the trained weights.
    checkpoint_path = os.environ.get("CHECKPOINT") or config.get(
        "checkpoint", "checkpoints/best_model.pt"
    )
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(base, checkpoint_path)

    if os.path.exists(checkpoint_path):
        print(f"[API] Loading model from {checkpoint_path}")
        model, model_info = load_model(checkpoint_path)
        print(
            f"[API] Arch from checkpoint: d_model={model_info['d_model']}, "
            f"layers={model_info['num_layers']}, d_ff={model_info['d_ff']}, "
            f"{model_info['params_m']}M params (epoch {model_info['epoch']})"
        )
        if model_info["vocab_size"] != tokenizer.vocab_size:
            print(
                f"[API WARNING] Vocab mismatch: checkpoint={model_info['vocab_size']} "
                f"vs tokenizer={tokenizer.vocab_size}"
            )
    else:
        print(f"[API] No checkpoint at {checkpoint_path} — using random weights (demo only)")
        model = MusicTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=model_cfg.get("d_model", 256),
            num_heads=model_cfg.get("num_heads", 8),
            num_layers=model_cfg.get("num_layers", 6),
            d_ff=model_cfg.get("d_ff", 1024),
            max_seq_len=model_cfg.get("max_seq_len", 2048),
            dropout=0.0,
            prompt_config=config.get("prompt", {}),
            num_kv_heads=model_cfg.get("num_kv_heads", 4),
            use_qk_norm=model_cfg.get("use_qk_norm", True),
            weight_tying=model_cfg.get("weight_tying", True),
        )
        model_info = {"trained": False}

    model_info["trained"] = os.path.exists(checkpoint_path)

    device = "auto"
    generator = MusicGenerator(model, tokenizer, device=device)
    model_device = str(generator.device)

    # Renderer
    sf_path = audio_cfg.get("soundfont", "soundfonts/FluidR3_GM.sf2")
    renderer = MidiRenderer(
        soundfont_path=sf_path,
        sample_rate=audio_cfg.get("sample_rate", 44100),
    )

    os.makedirs("outputs", exist_ok=True)
    print(f"[API] Ready! Device: {model_device}")

    yield

    # Shutdown cleanup
    print("[API] Shutting down...")
    generator = None
    renderer = None


# ============================================================
# App setup (with modern lifespan)
# ============================================================

app = FastAPI(
    title="Text-to-Music Generator",
    description="Sinh nhạc nền cho game từ mô tả văn bản",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Endpoints
# ============================================================

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the web demo."""
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index, media_type="text/html")
    return JSONResponse({"message": "Text-to-Music API", "docs": "/docs"})


@app.get("/model_info", include_in_schema=False)
async def get_model_info():
    """Architecture + training provenance of the loaded checkpoint (shown in the UI)."""
    return JSONResponse({**model_info, "device": model_device})


@app.post("/generate", response_model=MusicResponse)
async def generate_music(request: MusicRequest):
    """Generate music from structured attributes only (same input as Diffusion)."""
    if generator is None:
        raise HTTPException(503, "Model not loaded")

    # Shared schema with Diffusion (no BERT / free-text)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from compare.prompt_schema import format_prompt_display, labels_to_ids, normalize_structured

    request_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join("outputs", request_id)
    os.makedirs(output_dir, exist_ok=True)

    labels = normalize_structured(
        mood=request.mood,
        genre=request.genre,
        scene=request.scene,
        tempo=request.tempo,
        instrument=request.instrument,
        energy=request.energy,
    )
    ids = labels_to_ids(labels)
    prompt_display = format_prompt_display(labels)

    # Duration → token budget (~12 tokens/sec REMI, clamp to model max_seq)
    max_seq = 2048
    if getattr(generator, "model", None) is not None:
        max_seq = getattr(generator.model, "max_seq_len", 2048) or 2048
    if request.max_length is not None:
        max_length = int(request.max_length)
    else:
        max_length = int(max(128, min(max_seq, request.duration_sec * 12)))
    max_length = min(max_length, max_seq)

    # Generate MIDI
    midi_path = os.path.join(output_dir, "background_music.mid")
    generator.generate_midi(
        output_path=midi_path,
        max_length=max_length,
        temperature=request.temperature,
        top_p=request.top_p,
        **ids,
    )

    # Render WAV. The renderer degrades to MIDI-only if every backend fails,
    # so tell the client rather than handing it a URL that 404s.
    wav_path = os.path.join(output_dir, "background_music.wav")
    try:
        renderer.render(midi_path, wav_path)
    except Exception as e:
        print(f"[API WARNING] WAV render failed: {e}")
    wav_available = os.path.exists(wav_path) and os.path.getsize(wav_path) > 0

    # Get info
    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)
        duration = midi.get_end_time()
        num_notes = sum(len(inst.notes) for inst in midi.instruments)
    except Exception:
        duration = 0.0
        num_notes = 0

    return MusicResponse(
        request_id=request_id,
        midi_url=f"/download/{request_id}/midi",
        wav_url=f"/download/{request_id}/wav" if wav_available else None,
        wav_available=wav_available,
        duration=round(duration, 1),
        num_notes=num_notes,
        prompt_text=prompt_display,
    )


@app.get("/download/{request_id}/{format}")
async def download(request_id: str, format: str):
    """Download generated MIDI or WAV file."""
    if format == "midi":
        ext = "mid"
        media_type = "audio/midi"
    elif format == "wav":
        ext = "wav"
        media_type = "audio/wav"
    else:
        raise HTTPException(400, f"Invalid format: {format}. Use 'midi' or 'wav'.")

    path = os.path.join("outputs", request_id, f"background_music.{ext}")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found. Generate music first.")

    return FileResponse(
        path,
        media_type=media_type,
        filename=f"background_music.{ext}",
    )


@app.get("/notes/{request_id}", include_in_schema=False)
async def notes(request_id: str):
    """Note list of a generated MIDI, for the piano-roll canvas in the UI."""
    path = os.path.join("outputs", request_id, "background_music.mid")
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")

    import pretty_midi
    midi = pretty_midi.PrettyMIDI(path)
    # Cast explicitly: pretty_midi hands back numpy scalars, which json.dumps rejects.
    out = [
        {
            "pitch": int(n.pitch),
            "start": round(float(n.start), 3),
            "end": round(float(n.end), 3),
            "velocity": int(n.velocity),
            "program": int(inst.program),
        }
        for inst in midi.instruments
        for n in inst.notes
    ]
    out.sort(key=lambda n: n["start"])
    return JSONResponse({"notes": out, "duration": round(float(midi.get_end_time()), 2)})


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        model_loaded=generator is not None,
        device=model_device,
    )
