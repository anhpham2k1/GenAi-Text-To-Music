"""API schemas — structured prompt only (same as Diffusion)."""

from pydantic import BaseModel, Field
from typing import Optional


class MusicRequest(BaseModel):
    """Same 6 attributes as GenAI_Diffusion / compare.eval_prompts.json."""

    mood: Optional[str] = Field(default="peaceful")
    genre: Optional[str] = Field(default="fantasy")
    scene: Optional[str] = Field(default="village")
    tempo: Optional[str] = Field(default="moderate")
    instrument: Optional[str] = Field(default="piano")
    energy: Optional[str] = Field(default="medium")

    temperature: float = Field(default=0.85, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    max_length: Optional[int] = Field(default=None, ge=100, le=4096, description="Max tokens; if null use duration_sec")
    duration_sec: float = Field(default=30.0, ge=5.0, le=180.0, description="Target music length in seconds")


class MusicResponse(BaseModel):
    request_id: str
    midi_url: str
    wav_url: Optional[str] = None   # null when no synth backend is available
    wav_available: bool = False
    duration: float
    num_notes: int
    prompt_text: str  # display summary of structured attrs


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
