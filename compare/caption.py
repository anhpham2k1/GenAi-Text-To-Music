"""
English caption builder for text-conditioned music models.

Train captions are built from structured labels (labels.json).
Inference uses free-form English sentences (same style recommended).
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

_A_AN = re.compile(r"\ba(\s+[aeiouAEIOU])")

DEFAULTS = {
    "mood": "peaceful",
    "genre": "fantasy",
    "scene": "village",
    "tempo": "moderate",
    "instrument": "piano",
    "energy": "medium",
}

# Bare adjective/noun synonyms per attribute value. TEMPLATES below supply
# the connector words ("tempo", "energy") themselves, so entries here must
# NOT bake those in (e.g. tempo values read "fast", not "fast tempo").
SYNONYMS: Dict[str, Dict[str, List[str]]] = {
    "mood": {
        "happy": ["happy", "cheerful", "joyful", "upbeat", "bright"],
        "peaceful": ["peaceful", "calm", "serene", "tranquil", "relaxing"],
        "playful": ["playful", "whimsical", "fun", "lighthearted", "quirky"],
        "sad": ["sad", "melancholic", "mournful", "somber", "wistful"],
        "tense": ["tense", "anxious", "suspenseful", "edgy", "nerve-wracking"],
    },
    "genre": {
        "adventure": ["adventure", "adventurous", "exploration"],
        "epic": ["epic", "heroic", "grand", "cinematic"],
        "fantasy": ["fantasy", "magical", "mythical", "fairy-tale"],
        "fighting": ["fighting", "combat", "battle", "action"],
        "platformer": ["platformer", "arcade", "retro platforming"],
        "rpg": ["RPG", "role-playing game", "adventure RPG"],
        "sci-fi": ["sci-fi", "science fiction", "futuristic", "space-age"],
    },
    "scene": {
        "battlefield": ["battlefield", "war zone", "battleground"],
        "castle": ["castle", "fortress", "stronghold"],
        "dungeon": ["dungeon", "underground lair", "crypt"],
        "forest": ["forest", "woodland", "woods"],
        "mountain": ["mountain", "mountain range", "highlands"],
        "space": ["space", "outer space", "cosmos"],
        "village": ["village", "small town", "hamlet"],
    },
    "tempo": {
        "fast": ["fast", "quick", "brisk"],
        "moderate": ["moderate", "medium-paced", "steady"],
        "slow": ["slow", "relaxed", "leisurely"],
        "very_fast": ["very fast", "rapid", "frantic", "high-speed"],
        "very_slow": ["very slow", "extremely slow", "sluggish"],
    },
    "instrument": {
        "brass": ["brass", "brass section", "horns"],
        "flute": ["flute", "woodwinds"],
        "full_orchestra": ["full orchestra", "orchestral ensemble", "symphony orchestra"],
        "guitar": ["guitar", "acoustic guitar", "electric guitar"],
        "organ": ["organ", "pipe organ", "church organ"],
        "piano": ["piano", "solo piano", "grand piano"],
        "strings": ["strings", "string section", "string ensemble"],
        "synth": ["synth", "synthesizer", "electronic synth"],
    },
    "energy": {
        "calm": ["calm", "gentle", "soft", "mellow"],
        "high": ["high", "energetic", "powerful", "driving"],
        "intense": ["intense", "explosive", "fierce", "overwhelming"],
        "low": ["low", "subdued", "quiet", "understated"],
        "medium": ["medium", "moderate", "balanced"],
    },
}

# Each template must reference all 6 placeholders exactly once.
TEMPLATES = [
    "{mood} {genre} {scene} music, {tempo} tempo, {instrument}, {energy} energy",
    "a {energy} energy, {tempo} tempo {mood} track set in a {genre} {scene}, featuring {instrument}",
    "{mood} background music for a {scene} in a {genre} game, {tempo} tempo, {energy} energy, played on {instrument}",
    "a {tempo} tempo {genre} soundtrack for a {scene}, {mood} mood, {energy} energy, using {instrument}",
    "{instrument}-driven {mood} music, {tempo} tempo, {energy} energy, set in a {genre} {scene}",
    "a {mood}, {energy} energy theme for {scene} exploration in a {genre} setting, {tempo} tempo, {instrument} lead",
]


def _normalize(labels: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> Dict[str, str]:
    lab = dict(DEFAULTS)
    if labels:
        for k, v in labels.items():
            if k in lab and v is not None and str(v).strip():
                lab[k] = str(v).strip().lower()
    for k, v in kwargs.items():
        if k in lab and v is not None and str(v).strip():
            lab[k] = str(v).strip().lower()
    return lab


def labels_to_english_caption(labels: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """
    Convert 6 structured attributes → one English sentence for MiniLM.

    Example:
      happy fantasy village music, fast tempo, piano, medium energy

    Fixed single template — used where a stable/reproducible caption is
    wanted (eval prompts, API caption fallback). For diverse training
    captions see labels_to_english_captions.
    """
    lab = _normalize(labels, kwargs)
    return (
        f"{lab['mood']} {lab['genre']} {lab['scene']} music, "
        f"{lab['tempo']} tempo, {lab['instrument']}, {lab['energy']} energy"
    )


def labels_to_english_captions(
    labels: Optional[Dict[str, Any]] = None,
    n: int = 5,
    seed: Optional[int] = None,
    **kwargs,
) -> List[str]:
    """
    Convert 6 structured attributes → up to `n` distinct English sentences,
    varying sentence structure (TEMPLATES) and per-attribute wording
    (SYNONYMS). Used to diversify training captions so the MiniLM-
    conditioned model isn't only ever exposed to one rigid phrasing per
    attribute combo.
    """
    lab = _normalize(labels, kwargs)
    rng = random.Random(seed)

    seen = set()
    captions: List[str] = []
    attempts = 0
    max_attempts = n * 10
    while len(captions) < n and attempts < max_attempts:
        attempts += 1
        template = rng.choice(TEMPLATES)
        slot_values = {
            field: rng.choice(SYNONYMS.get(field, {}).get(value, [value.replace("_", " ")]))
            for field, value in lab.items()
        }
        caption = _A_AN.sub(r"an\1", template.format(**slot_values))
        if caption not in seen:
            seen.add(caption)
            captions.append(caption)

    if not captions:
        captions = [labels_to_english_caption(lab)]
    return captions


def entry_to_prompt_text(entry: Dict[str, Any]) -> str:
    """
    Resolve English prompt text from an eval / API entry.

    Priority:
      1. entry['text'] or entry['prompt'] if non-empty
      2. build from structured mood/genre/... fields
    """
    for key in ("text", "prompt", "prompt_text"):
        raw = entry.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return labels_to_english_caption(entry)


def structured_fields_from_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    """Keep structured fields for metrics / MIDI program (optional)."""
    out = {}
    for k in DEFAULTS:
        if entry.get(k) is not None and str(entry.get(k)).strip():
            out[k] = str(entry[k]).strip().lower()
        else:
            out[k] = DEFAULTS[k]
    return out
