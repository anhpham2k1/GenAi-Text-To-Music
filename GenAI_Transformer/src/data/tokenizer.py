"""
MIDI Tokenizer — REMI-like representation.

Chuyển đổi file MIDI thành chuỗi token IDs (và ngược lại).
Sử dụng REMI (REvamped MIDI-derived) representation (Huang & Yang, 2020).

Token vocabulary (~388 tokens):
- Special: <PAD>, <BOS>, <EOS>, <UNK>
- NOTE_ON_21 ... NOTE_ON_108 (88 nốt piano range)
- NOTE_OFF_21 ... NOTE_OFF_108
- VEL_0 ... VEL_31 (32 velocity bins)
- TIME_SHIFT_1 ... TIME_SHIFT_100 (10ms per unit, max 1s)
- TEMPO_0 ... TEMPO_7 (8 tempo bins)
- INST_0 ... INST_15 (16 instrument groups)
"""

import numpy as np
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

try:
    import pretty_midi
except ImportError:
    pretty_midi = None
    print("[WARNING] pretty_midi not installed. Install with: pip install pretty_midi")


# ============================================================
# Constants
# ============================================================

# Special tokens
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

# Tempo labels & boundaries (BPM)
TEMPO_LABELS = [
    "VERY_SLOW", "SLOW", "MODERATE_SLOW", "MODERATE",
    "MODERATE_FAST", "FAST", "VERY_FAST", "EXTREME"
]
TEMPO_BOUNDARIES = [70, 90, 110, 130, 150, 170, 200]

# General MIDI instrument groups (program // 8)
INSTRUMENT_GROUPS = [
    "PIANO", "CHROMATIC_PERC", "ORGAN", "GUITAR",
    "BASS", "STRINGS", "ENSEMBLE", "BRASS",
    "REED", "PIPE", "SYNTH_LEAD", "SYNTH_PAD",
    "SYNTH_FX", "ETHNIC", "PERCUSSIVE", "SFX"
]


class MidiTokenizer:
    """
    REMI-like MIDI tokenizer.

    Chuyển đổi giữa MIDI files và chuỗi token IDs.
    Hỗ trợ:
    - NOTE_ON / NOTE_OFF events
    - Velocity quantization (32 bins)
    - Time shift (10ms resolution, max 1s per token)
    - Tempo classification (8 bins)
    - Instrument groups (16 groups)
    """

    def __init__(
        self,
        pitch_range: Tuple[int, int] = (21, 108),
        velocity_bins: int = 32,
        time_shift_bins: int = 100,
        tempo_bins: int = 8,
        num_instrument_groups: int = 16,
    ):
        self.pitch_range = pitch_range
        self.pitch_min, self.pitch_max = pitch_range
        self.num_pitches = self.pitch_max - self.pitch_min + 1
        self.velocity_bins = velocity_bins
        self.time_shift_bins = time_shift_bins
        self.tempo_bins = tempo_bins
        self.num_instrument_groups = num_instrument_groups

        # Build vocabulary
        self.token_to_idx: OrderedDict[str, int] = OrderedDict()
        self.idx_to_token: Dict[int, str] = {}
        self._build_vocab()

        # Bounded cache for tokenized results to avoid unlimited memory usage
        self._encode_cache: Dict[str, List[int]] = {}
        self._max_cache_size = 500  # reasonable default for MIDI datasets

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    def _build_vocab(self):
        """Build token vocabulary."""
        idx = 0

        # Special tokens
        for token in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            self.token_to_idx[token] = idx
            idx += 1

        # NOTE_ON events
        for pitch in range(self.pitch_min, self.pitch_max + 1):
            self.token_to_idx[f"NOTE_ON_{pitch}"] = idx
            idx += 1

        # NOTE_OFF events
        for pitch in range(self.pitch_min, self.pitch_max + 1):
            self.token_to_idx[f"NOTE_OFF_{pitch}"] = idx
            idx += 1

        # Velocity bins
        for v in range(self.velocity_bins):
            self.token_to_idx[f"VEL_{v}"] = idx
            idx += 1

        # Time shift (1-100, each = 10ms)
        for t in range(1, self.time_shift_bins + 1):
            self.token_to_idx[f"TIME_SHIFT_{t}"] = idx
            idx += 1

        # Tempo bins
        for i, label in enumerate(TEMPO_LABELS[: self.tempo_bins]):
            self.token_to_idx[f"TEMPO_{label}"] = idx
            idx += 1

        # Instrument groups
        for i, label in enumerate(INSTRUMENT_GROUPS[: self.num_instrument_groups]):
            self.token_to_idx[f"INST_{label}"] = idx
            idx += 1

        # Reverse mapping
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)

    # --------------------------------------------------------
    # Special token IDs
    # --------------------------------------------------------

    @property
    def pad_id(self) -> int:
        return self.token_to_idx[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_idx[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_idx[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_idx[UNK_TOKEN]

    # --------------------------------------------------------
    # Encode: MIDI → Tokens
    # --------------------------------------------------------

    def encode(self, midi_path: str, max_length: int = 2048) -> List[int]:
        """
        Chuyển file MIDI thành chuỗi token IDs.

        Optimized with simple cache for repeated calls on same file.
        """
        cache_key = f"{midi_path}:{max_length}"
        if cache_key in self._encode_cache:
            return self._encode_cache[cache_key][:]

        if pretty_midi is None:
            raise ImportError("pretty_midi is required. pip install pretty_midi")

        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            raise ValueError(f"Cannot read MIDI file {midi_path}: {e}")

        tokens: List[int] = [self.bos_id]

        # Tempo token
        tempo = midi.estimate_tempo()
        tempo_token = self._quantize_tempo(tempo)
        tokens.append(self.token_to_idx.get(tempo_token, self.unk_id))

        # Collect all note events from all instruments, sorted by time
        events = self._collect_events(midi)

        # Convert events to tokens
        current_time = 0.0
        current_instrument = None

        for event in events:
            # --- Time shift ---
            dt = event["time"] - current_time
            if dt > 0.005:  # Ignore very small time differences (<5ms)
                time_tokens = self._encode_time_shift(dt)
                tokens.extend(time_tokens)
                current_time = event["time"]

            # --- Instrument change ---
            if event.get("instrument") != current_instrument:
                current_instrument = event["instrument"]
                inst_token = f"INST_{current_instrument}"
                if inst_token in self.token_to_idx:
                    tokens.append(self.token_to_idx[inst_token])

            # --- Note events ---
            if event["type"] == "note_on":
                # Velocity
                vel_bin = min(event["velocity"] // 4, self.velocity_bins - 1)
                tokens.append(self.token_to_idx[f"VEL_{vel_bin}"])

                # Pitch (clamped to range)
                pitch = int(np.clip(event["pitch"], self.pitch_min, self.pitch_max))
                tokens.append(self.token_to_idx[f"NOTE_ON_{pitch}"])

            elif event["type"] == "note_off":
                pitch = int(np.clip(event["pitch"], self.pitch_min, self.pitch_max))
                tokens.append(self.token_to_idx[f"NOTE_OFF_{pitch}"])

            # Check max length (reserve 1 for EOS)
            if len(tokens) >= max_length - 1:
                break

        tokens.append(self.eos_id)

        # Pad to max_length
        if len(tokens) < max_length:
            tokens.extend([self.pad_id] * (max_length - len(tokens)))

        tokens = tokens[:max_length]

        # Bounded cache (evict oldest if full)
        if len(self._encode_cache) >= self._max_cache_size:
            # Simple FIFO eviction
            self._encode_cache.pop(next(iter(self._encode_cache)))
        self._encode_cache[cache_key] = tokens[:]
        return tokens

    def _collect_events(self, midi) -> List[Dict]:
        """Thu thập tất cả note events từ MIDI, sort theo time."""
        events = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            inst_group = INSTRUMENT_GROUPS[
                min(instrument.program // 8, len(INSTRUMENT_GROUPS) - 1)
            ]

            for note in instrument.notes:
                # Lọc nốt ngoài phạm vi
                if note.pitch < self.pitch_min or note.pitch > self.pitch_max:
                    continue

                events.append(
                    {
                        "time": note.start,
                        "type": "note_on",
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "instrument": inst_group,
                    }
                )
                events.append(
                    {
                        "time": note.end,
                        "type": "note_off",
                        "pitch": note.pitch,
                        "instrument": inst_group,
                    }
                )

        # Sort by time, note_off before note_on at same time
        events.sort(key=lambda x: (x["time"], 0 if x["type"] == "note_off" else 1))
        return events

    def _quantize_tempo(self, bpm: float) -> str:
        """Quantize BPM thành tempo bin label."""
        for i, boundary in enumerate(TEMPO_BOUNDARIES):
            if bpm < boundary:
                return f"TEMPO_{TEMPO_LABELS[i]}"
        return f"TEMPO_{TEMPO_LABELS[-1]}"

    def _encode_time_shift(self, dt_seconds: float) -> List[int]:
        """Encode khoảng thời gian thành TIME_SHIFT tokens (10ms per unit)."""
        dt_units = int(round(dt_seconds * 100))  # Convert to 10ms units
        tokens = []
        while dt_units > 0:
            shift = min(dt_units, self.time_shift_bins)
            token_str = f"TIME_SHIFT_{shift}"
            if token_str in self.token_to_idx:
                tokens.append(self.token_to_idx[token_str])
            dt_units -= shift
        return tokens

    # --------------------------------------------------------
    # Decode: Tokens → MIDI
    # --------------------------------------------------------

    def decode(self, token_ids: List[int], default_tempo: float = 120.0):
        """
        Chuyển chuỗi token IDs thành PrettyMIDI object.

        Args:
            token_ids: Chuỗi token IDs
            default_tempo: Tempo mặc định (BPM)

        Returns:
            pretty_midi.PrettyMIDI object
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi is required. pip install pretty_midi")

        midi = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)

        current_time = 0.0
        current_velocity = 80
        current_program = 0  # Piano

        instruments: Dict[int, pretty_midi.Instrument] = {}
        active_notes: Dict[int, Tuple[float, int, int]] = {}
        # active_notes[pitch] = (start_time, velocity, program)

        for token_id in token_ids:
            token_str = self.idx_to_token.get(token_id, UNK_TOKEN)

            if token_str in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
                if token_str == EOS_TOKEN:
                    break
                continue

            if token_str.startswith("TIME_SHIFT_"):
                try:
                    shift = int(token_str.split("_")[-1])
                    current_time += shift * 0.01  # 10ms per unit
                except ValueError:
                    continue

            elif token_str.startswith("VEL_"):
                try:
                    vel_bin = int(token_str.split("_")[-1])
                    current_velocity = min(vel_bin * 4 + 2, 127)
                except ValueError:
                    continue

            elif token_str.startswith("INST_"):
                inst_name = token_str.replace("INST_", "")
                program_map = {
                    "PIANO": 0,
                    "CHROMATIC_PERC": 8,
                    "ORGAN": 16,
                    "GUITAR": 24,
                    "BASS": 32,
                    "STRINGS": 48,
                    "ENSEMBLE": 48,
                    "BRASS": 56,
                    "REED": 64,
                    "PIPE": 73,
                    "SYNTH_LEAD": 80,
                    "SYNTH_PAD": 88,
                    "SYNTH_FX": 96,
                    "ETHNIC": 104,
                    "PERCUSSIVE": 112,
                    "SFX": 120,
                }
                current_program = program_map.get(inst_name, 0)

            elif token_str.startswith("NOTE_ON_"):
                try:
                    pitch = int(token_str.split("_")[-1])
                    active_notes[pitch] = (
                        current_time,
                        current_velocity,
                        current_program,
                    )
                except ValueError:
                    continue

            elif token_str.startswith("NOTE_OFF_"):
                try:
                    pitch = int(token_str.split("_")[-1])
                    if pitch in active_notes:
                        start, vel, prog = active_notes.pop(pitch)
                        duration = current_time - start
                        if duration < 0.01:
                            duration = 0.1  # Minimum duration 100ms

                        if prog not in instruments:
                            instruments[prog] = pretty_midi.Instrument(program=prog)

                        note = pretty_midi.Note(
                            velocity=vel,
                            pitch=pitch,
                            start=start,
                            end=start + duration,
                        )
                        instruments[prog].notes.append(note)
                except ValueError:
                    continue

            elif token_str.startswith("TEMPO_"):
                tempo_map = {
                    "VERY_SLOW": 60,
                    "SLOW": 80,
                    "MODERATE_SLOW": 100,
                    "MODERATE": 120,
                    "MODERATE_FAST": 140,
                    "FAST": 160,
                    "VERY_FAST": 180,
                    "EXTREME": 200,
                }
                label = token_str.replace("TEMPO_", "")
                # Note: PrettyMIDI sets tempo at init; mid-song changes not trivially supported
                pass

        # Close remaining active notes
        for pitch, (start, vel, prog) in active_notes.items():
            if prog not in instruments:
                instruments[prog] = pretty_midi.Instrument(program=prog)
            instruments[prog].notes.append(
                pretty_midi.Note(
                    velocity=vel,
                    pitch=pitch,
                    start=start,
                    end=max(current_time, start + 0.1),
                )
            )

        # Add instruments to MIDI
        for inst in instruments.values():
            if len(inst.notes) > 0:
                midi.instruments.append(inst)

        # Fallback: nếu không có note nào, thêm silence
        if len(midi.instruments) == 0:
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(
                pretty_midi.Note(velocity=0, pitch=60, start=0.0, end=1.0)
            )
            midi.instruments.append(inst)

        return midi

    # --------------------------------------------------------
    # Auto-labeling (Rule-based)
    # --------------------------------------------------------

    def auto_label(self, midi_path: str) -> Dict:
        """
        Tự động gán labels cho MIDI file dựa trên phân tích nội dung.
        Cải tiến: hỗ trợ đa dạng nhạc cụ, thể loại game music tốt hơn.
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi is required.")

        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            return self._default_labels()

        # --- Tempo ---
        tempo = midi.estimate_tempo()
        if tempo < 70:
            tempo_label = "very_slow"
        elif tempo < 90:
            tempo_label = "slow"
        elif tempo < 115:
            tempo_label = "moderate"
        elif tempo < 140:
            tempo_label = "fast"
        else:
            tempo_label = "very_fast"

        # --- Instrument (cải tiến: hỗ trợ nhiều nhóm hơn) ---
        programs = set()
        has_drums = any(inst.is_drum for inst in midi.instruments)
        for inst in midi.instruments:
            if not inst.is_drum:
                programs.add(inst.program)

        instrument_label = "piano"
        if any(48 <= p < 56 for p in programs) or any(40 <= p < 48 for p in programs):
            instrument_label = "strings"
        elif any(56 <= p < 64 for p in programs):
            instrument_label = "brass"
        elif any(24 <= p < 32 for p in programs) or any(32 <= p < 40 for p in programs):
            instrument_label = "guitar"
        elif any(73 <= p < 80 for p in programs) or any(64 <= p < 72 for p in programs):
            instrument_label = "woodwind"  # flute, reed etc.
        elif any(0 <= p < 8 for p in programs):
            instrument_label = "piano"
        elif any(80 <= p < 104 for p in programs):
            instrument_label = "synth"
        elif len(programs) > 3:
            instrument_label = "full_orchestra"
        if has_drums and len(programs) > 2:
            instrument_label = "full_orchestra"

        # --- Energy (note density + polyphony) ---
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        duration = midi.get_end_time()
        if duration > 0:
            density = total_notes / duration
        else:
            density = 0
        polyphony = max((len([n for n in inst.notes if n.start <= t < n.end]) for inst in midi.instruments for t in [n.start for n in inst.notes]), default=0)

        if density < 1.2 or polyphony < 2:
            energy_label = "calm"
        elif density < 2.8 or polyphony < 4:
            energy_label = "low"
        elif density < 5.5 or polyphony < 7:
            energy_label = "medium"
        elif density < 9.0 or polyphony < 10:
            energy_label = "high"
        else:
            energy_label = "intense"

        # --- Mood (cải tiến: kết hợp key + tempo + energy) ---
        all_pitches = []
        for inst in midi.instruments:
            for note in inst.notes:
                all_pitches.append(note.pitch % 12)

        mood_label = "peaceful"
        if len(all_pitches) > 0:
            pitch_counts = np.bincount(all_pitches, minlength=12)
            major_score = pitch_counts[0] + pitch_counts[4] + pitch_counts[7]
            minor_score = pitch_counts[0] + pitch_counts[3] + pitch_counts[7]

            if major_score > minor_score * 1.15:
                if tempo > 115 and energy_label in ["high", "intense"]:
                    mood_label = "happy"
                elif energy_label in ["calm", "low"]:
                    mood_label = "peaceful"
                else:
                    mood_label = "playful"
            else:
                if tempo > 115 or energy_label in ["high", "intense"]:
                    mood_label = "tense"
                else:
                    mood_label = "sad"
        else:
            mood_label = "peaceful"

        # --- Genre & Scene (cải tiến dựa trên instrument + energy + tempo) ---
        genre_label = "fantasy"
        scene_label = "village"

        if instrument_label in ["brass", "full_orchestra"] and energy_label in ["high", "intense"]:
            genre_label = "epic"
            scene_label = "battlefield"
        elif instrument_label == "synth" or "synth" in str(programs):
            genre_label = "sci-fi"
            scene_label = "space"
        elif instrument_label in ["strings", "woodwind"] and energy_label in ["calm", "low"]:
            genre_label = "adventure"
            scene_label = "forest"
        elif energy_label == "intense" or tempo_label == "very_fast":
            genre_label = "fighting"
            scene_label = "dungeon"
        elif "piano" in instrument_label and energy_label in ["calm", "low"]:
            genre_label = "rpg"
            scene_label = "castle"
        elif instrument_label == "guitar":
            genre_label = "platformer"
            scene_label = "mountain"

        return {
            "mood": mood_label,
            "genre": genre_label,
            "scene": scene_label,
            "tempo": tempo_label,
            "instrument": instrument_label,
            "energy": energy_label,
        }

    def _default_labels(self) -> Dict:
        return {
            "mood": "peaceful",
            "genre": "fantasy",
            "scene": "village",
            "tempo": "moderate",
            "instrument": "piano",
            "energy": "medium",
        }

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def tokens_to_str(self, token_ids: List[int]) -> List[str]:
        """Chuyển token IDs thành token strings."""
        return [self.idx_to_token.get(t, UNK_TOKEN) for t in token_ids]

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"MidiTokenizer(vocab_size={self.vocab_size}, "
            f"pitch_range={self.pitch_range}, "
            f"velocity_bins={self.velocity_bins}, "
            f"time_shift_bins={self.time_shift_bins})"
        )
