"""Tiny numpy piano-ish synth so the demo can play audio with no FluidSynth/soundfont."""
import io
import wave

import numpy as np
import pretty_midi

SR = 22050


def synthesize(pm: pretty_midi.PrettyMIDI, sr: int = SR) -> np.ndarray:
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return np.zeros(sr, dtype=np.float32)
    end = max(n.end for n in notes) + 1.5
    out = np.zeros(int(end * sr) + 1, dtype=np.float32)
    harmonics = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)]
    for n in notes:
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        dur = min(n.end - n.start, 4.0) + 0.4  # short release tail
        t = np.arange(int(dur * sr)) / sr
        env = np.exp(-t * (1.5 + f0 / 600))          # piano-like decay
        rel = np.clip((dur - t) / 0.4, 0, 1)         # release fade
        att = np.clip(t / 0.005, 0, 1)               # click-free attack
        wave_ = sum(a * np.sin(2 * np.pi * f0 * h * t) for h, a in harmonics if f0 * h < sr / 2)
        s = (n.velocity / 127) * env * rel * att * wave_
        i = int(n.start * sr)
        out[i : i + len(s)] += s.astype(np.float32)[: len(out) - i]
    peak = np.abs(out).max()
    return out / peak * 0.9 if peak > 0 else out


def to_wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()
