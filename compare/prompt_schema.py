"""
Shared structured prompt schema for BOTH Transformer and Diffusion.

Same INPUT (6 attributes) → same evaluation prompts / API contract.
Output for both methods: MIDI file (.mid). Optional WAV is render-only.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

# String label → integer ID (must match training labels)
MOOD_MAP = {
    "happy": 0, "sad": 1, "tense": 2, "peaceful": 3, "epic": 4,
    "mysterious": 5, "dark": 6, "heroic": 7, "nostalgic": 8, "playful": 9,
}
GENRE_MAP = {
    "fantasy": 0, "sci-fi": 1, "horror": 2, "adventure": 3, "rpg": 4,
    "puzzle": 5, "platformer": 6, "simulation": 7, "fighting": 8, "racing": 9,
}
SCENE_MAP = {
    "forest": 0, "dungeon": 1, "village": 2, "castle": 3, "ocean": 4,
    "space": 5, "mountain": 6, "desert": 7, "city": 8, "battlefield": 9,
}
TEMPO_MAP = {
    "very_slow": 0, "slow": 1, "moderate": 2, "fast": 3, "very_fast": 4,
}
INSTRUMENT_MAP = {
    "piano": 0, "strings": 1, "brass": 2, "flute": 3,
    "guitar": 4, "organ": 5, "synth": 6, "full_orchestra": 7,
}
ENERGY_MAP = {
    "calm": 0, "low": 1, "medium": 2, "high": 3, "intense": 4,
}

DEFAULTS = {
    "mood": "peaceful",
    "genre": "fantasy",
    "scene": "village",
    "tempo": "moderate",
    "instrument": "piano",
    "energy": "medium",
}

MAPS = {
    "mood": MOOD_MAP,
    "genre": GENRE_MAP,
    "scene": SCENE_MAP,
    "tempo": TEMPO_MAP,
    "instrument": INSTRUMENT_MAP,
    "energy": ENERGY_MAP,
}


def labels_to_ids(labels: Dict[str, str]) -> Dict[str, int]:
    """Convert string labels → int IDs for model conditioning."""
    out = {}
    for key, mapping in MAPS.items():
        raw = (labels.get(key) or DEFAULTS[key]).lower().strip()
        # default index = DEFAULTS mapped
        default_id = mapping[DEFAULTS[key]]
        out[key] = mapping.get(raw, default_id)
    return out


def format_prompt_display(labels: Dict[str, str]) -> str:
    """Human-readable summary (logging / UI only — NOT model input)."""
    mood = labels.get("mood") or DEFAULTS["mood"]
    genre = labels.get("genre") or DEFAULTS["genre"]
    scene = labels.get("scene") or DEFAULTS["scene"]
    tempo = labels.get("tempo") or DEFAULTS["tempo"]
    instrument = labels.get("instrument") or DEFAULTS["instrument"]
    energy = labels.get("energy") or DEFAULTS["energy"]
    return (
        f"{mood} | {genre} | {scene} | {tempo} | {instrument} | {energy}"
    )


def normalize_structured(
    mood: Optional[str] = None,
    genre: Optional[str] = None,
    scene: Optional[str] = None,
    tempo: Optional[str] = None,
    instrument: Optional[str] = None,
    energy: Optional[str] = None,
    **_extra,
) -> Dict[str, str]:
    """Fill defaults for any missing attribute."""
    return {
        "mood": (mood or DEFAULTS["mood"]).lower(),
        "genre": (genre or DEFAULTS["genre"]).lower(),
        "scene": (scene or DEFAULTS["scene"]).lower(),
        "tempo": (tempo or DEFAULTS["tempo"]).lower(),
        "instrument": (instrument or DEFAULTS["instrument"]).lower(),
        "energy": (energy or DEFAULTS["energy"]).lower(),
    }
