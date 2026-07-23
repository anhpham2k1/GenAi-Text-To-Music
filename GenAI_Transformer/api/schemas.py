"""API schemas — English free-text prompt (primary)."""

from pydantic import BaseModel, Field
from typing import Optional


class MusicRequest(BaseModel):
    """
    Primary input: English text prompt (ChatGPT-style).

    Optional structured fields are only used to build a caption
    when `prompt` is empty.
    """

    prompt: Optional[str] = Field(
        default=None,
        description="English text prompt, e.g. 'Happy fantasy village music, fast piano'",
    )
    # Optional helpers if prompt omitted
    mood: Optional[str] = Field(default=None)
    genre: Optional[str] = Field(default=None)
    scene: Optional[str] = Field(default=None)
    tempo: Optional[str] = Field(default=None)
    instrument: Optional[str] = Field(default=None)
    energy: Optional[str] = Field(default=None)

    temperature: float = Field(default=0.85, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    max_length: Optional[int] = Field(default=None, ge=100, le=4096)
    duration_sec: float = Field(default=30.0, ge=5.0, le=180.0)


class MusicResponse(BaseModel):
    request_id: str
    midi_url: str
    wav_url: Optional[str] = None   # null when no synth backend is available
    wav_available: bool = False
    duration: float
    num_notes: int
    prompt_text: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
