"""Build the six blind survey clips: equal length, similar loudness, neutral names.

    python -m scripts.make_survey_clips      (run from the repo root)

Needs: data/processed/{test.npy,meta.json}, data/raw/maestro/, models/lstm.npz,
models/markov.pkl (order 2: run `python -m src.markov --order 2` first).

Outputs: outputs/survey_clips/clip_A.wav ... clip_F.wav   (give THESE to participants)
         docs/survey_clips_manifest.md + docs/survey_answer_key.json  (answer key, keep private)
"""
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pretty_midi

from src.audio import SR, synthesize, to_wav_bytes
from src.generate import generate_lstm, generate_markov, get_seed
from src.tokens import tokens_to_midi

ROOT = Path(__file__).resolve().parent.parent
SECONDS = 15
TARGET_RMS = 0.08
TMP = ROOT / "outputs" / "midi" / "_survey_tmp.mid"


def cut(pm, start, length):
    """Notes overlapping [start, start+length], shifted so the clip starts at 0."""
    out = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for src in pm.instruments:
        if src.is_drum:
            continue
        for n in src.notes:
            if n.end <= start or n.start >= start + length:
                continue
            s, e = max(n.start, start) - start, min(n.end, start + length) - start
            if e > s:
                inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, s, e))
    out.instruments.append(inst)
    return out


def render(pm):
    """Synthesize, force exactly SECONDS long, fade out, match loudness."""
    a = synthesize(pm).astype(np.float64)
    n = SECONDS * SR
    a = a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))
    fade = int(0.75 * SR)
    a[-fade:] *= np.linspace(1, 0, fade)
    gain = min(TARGET_RMS / max(np.sqrt((a ** 2).mean()), 1e-9), 0.95 / max(np.abs(a).max(), 1e-9))
    return (a * gain).astype(np.float32)


def generated_clip(kind, seed_idx):
    rng = np.random.default_rng(100 + seed_idx)
    seed = get_seed(32, rng)
    for n_tokens in (1500, 3000, 6000):
        if kind == "lstm":
            toks = generate_lstm(ROOT / "models" / "lstm.pt", seed, n_tokens, 1.0, 20, rng)
        else:
            toks = generate_markov(ROOT / "models" / "markov.pkl", seed, n_tokens, 1.0, rng)
        TMP.parent.mkdir(parents=True, exist_ok=True)
        pm = cut(tokens_to_midi(toks, TMP), 0, SECONDS)
        notes = pm.instruments[0].notes
        if notes and max(n.end for n in notes) >= SECONDS - 3:
            return pm
    return pm


def real_clip(idx, test_files):
    rng = random.Random(500 + idx)
    name = rng.choice(test_files)
    path = next((ROOT / "data" / "raw" / "maestro").rglob(name))
    pm = pretty_midi.PrettyMIDI(str(path))
    dur = pm.get_end_time()
    start = rng.uniform(20, max(21, dur - 20)) if dur > 40 else 0.0
    return cut(pm, start, SECONDS), f"{name} @ {start:.0f}s"


def main():
    meta = json.loads((ROOT / "data" / "processed" / "meta.json").read_text())
    test_files = meta["splits"]["test"]["files"]
    try:
        with open(ROOT / "models" / "markov.pkl", "rb") as f:
            order = pickle.load(f).order
        if order != 2:
            raise SystemExit(f"models/markov.pkl is order {order}; run `python -m src.markov --order 2` first")
    except FileNotFoundError:
        raise SystemExit("models/markov.pkl missing; run `python -m src.markov --order 2` first")

    clips = []  # (source, detail, pm)
    for i in (1, 2):
        clips.append(("Markov (order 2)", f"sample {i}, temp 1.0", generated_clip("markov", i)))
    for i in (1, 2):
        clips.append(("LSTM", f"sample {i}, temp 1.0, top-k 20", generated_clip("lstm", i)))
    for i in (1, 2):
        pm, detail = real_clip(i, test_files)
        clips.append(("Human (MAESTRO test piece)", detail, pm))

    order_ = list(range(6))
    random.Random(7).shuffle(order_)
    out_dir = ROOT / "outputs" / "survey_clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.wav"):
        old.unlink()

    key, lines = {}, ["| Clip | Source | Details |", "|---|---|---|"]
    for label, idx in zip("ABCDEF", order_):
        source, detail, pm = clips[idx]
        (out_dir / f"clip_{label}.wav").write_bytes(to_wav_bytes(render(pm)))
        key[label] = source
        lines.append(f"| Clip {label} | {source} | {detail} |")
        print(f"clip_{label}.wav  <-  {source}")
    TMP.unlink(missing_ok=True)

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "survey_answer_key.json").write_text(json.dumps(key, indent=2))
    (ROOT / "docs" / "survey_clips_manifest.md").write_text(
        "# Survey clip manifest (ANSWER KEY - do not show participants)\n\n"
        f"All clips are {SECONDS} s, mono, loudness-matched, with neutral names. "
        "Present Clip A to Clip F in that order (already shuffled).\n\n" + "\n".join(lines) + "\n")
    print("\nWrote docs/survey_clips_manifest.md and docs/survey_answer_key.json")


if __name__ == "__main__":
    main()
