"""
MIDI Renderer — Chuyển MIDI sang WAV.

Pipeline: MIDI tokens → MIDI file → FluidSynth + SoundFont → WAV

FluidSynth: Software synthesizer mã nguồn mở
SoundFont (.sf2): File chứa samples âm thanh nhạc cụ thật
"""

import os
import subprocess
from typing import Optional


class MidiRenderer:
    """
    Render MIDI files → WAV audio sử dụng FluidSynth.

    Usage:
        renderer = MidiRenderer(soundfont_path="FluidR3_GM.sf2")
        renderer.render("input.mid", "output.wav")
    """

    def __init__(
        self,
        soundfont_path: str = "soundfonts/FluidR3_GM.sf2",
        sample_rate: int = 44100,
    ):
        self.soundfont_path = soundfont_path
        self.sample_rate = sample_rate

    def render(
        self,
        midi_path: str,
        wav_path: str,
        gain: float = 1.0,
    ) -> str:
        """
        Render MIDI → WAV sử dụng FluidSynth CLI.

        Args:
            midi_path: Đường dẫn file MIDI input
            wav_path: Đường dẫn file WAV output
            gain: Volume gain (0.0-10.0)

        Returns:
            str — đường dẫn file WAV
        """
        os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)

        # Kiểm tra FluidSynth
        if not self._check_fluidsynth():
            print("[WARNING] FluidSynth not found. Trying midi2audio fallback...")
            return self._render_midi2audio(midi_path, wav_path)

        # Kiểm tra SoundFont
        if not os.path.exists(self.soundfont_path):
            print(f"[WARNING] SoundFont not found: {self.soundfont_path}")
            print("  Download FluidR3_GM.sf2 and place in soundfonts/ directory")
            print("  URL: https://member.keymusician.com/Member/FluidR3_GM/FluidR3_GM.htm")
            return self._render_midi2audio(midi_path, wav_path)

        cmd = [
            "fluidsynth",
            "-ni",                          # Non-interactive
            self.soundfont_path,            # SoundFont
            midi_path,                      # Input MIDI
            "-F", wav_path,                 # Output WAV
            "-r", str(self.sample_rate),    # Sample rate
            "-g", str(gain),               # Gain
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                print(f"[WARNING] FluidSynth error: {result.stderr}")
                return self._render_midi2audio(midi_path, wav_path)

            print(f"[Renderer] WAV saved: {wav_path}")
            return wav_path

        except subprocess.TimeoutExpired:
            print("[WARNING] FluidSynth timeout")
            return self._render_midi2audio(midi_path, wav_path)
        except FileNotFoundError:
            print("[WARNING] FluidSynth not in PATH")
            return self._render_midi2audio(midi_path, wav_path)

    def _render_midi2audio(self, midi_path: str, wav_path: str) -> str:
        """Fallback: render sử dụng midi2audio library."""
        try:
            from midi2audio import FluidSynth as FS

            fs = FS(self.soundfont_path)
            fs.midi_to_audio(midi_path, wav_path)
            print(f"[Renderer] WAV saved (midi2audio): {wav_path}")
            return wav_path
        except ImportError:
            print("[ERROR] midi2audio not installed. pip install midi2audio")
            print("[INFO] MIDI file was saved but WAV rendering requires FluidSynth.")
            print("  Install FluidSynth: https://www.fluidsynth.org/")
            return midi_path
        except Exception as e:
            print(f"[ERROR] midi2audio failed: {e}")
            return midi_path

    def _check_fluidsynth(self) -> bool:
        """Check if FluidSynth is installed and accessible."""
        try:
            result = subprocess.run(
                ["fluidsynth", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def render_full_pipeline(
        self,
        tokens: list,
        tokenizer,
        wav_path: str,
        midi_path: Optional[str] = None,
        default_tempo: float = 120.0,
    ) -> tuple:
        """
        Full pipeline: tokens → MIDI → WAV

        Args:
            tokens: MIDI token IDs
            tokenizer: MidiTokenizer instance
            wav_path: Output WAV path
            midi_path: Output MIDI path (auto-generated if None)
            default_tempo: Default tempo BPM

        Returns:
            (midi_path, wav_path)
        """
        if midi_path is None:
            midi_path = wav_path.replace(".wav", ".mid")

        # Tokens → MIDI
        midi = tokenizer.decode(tokens, default_tempo=default_tempo)
        os.makedirs(os.path.dirname(midi_path) or ".", exist_ok=True)
        midi.write(midi_path)
        print(f"[Renderer] MIDI saved: {midi_path}")

        # MIDI → WAV
        self.render(midi_path, wav_path)

        return midi_path, wav_path
